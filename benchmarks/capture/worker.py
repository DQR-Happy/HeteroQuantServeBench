"""One independent process per adapter/case. Run through the remote proxy."""

import argparse
import json
import math
import os
import resource
import struct
import subprocess
import threading
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from benchmarks.capture.adapters import ADAPTERS
from benchmarks.capture.schedule import latency_sample, payload, plan, producer_for

# index, actual call, scheduled time, call duration, result code.
RECEIPT = struct.Struct("<QQQQB")
RESULTS = {"accepted": 0, "submitted_unknown": 1, "queue_full": 2,
           "budget_exceeded": 3, "io_failed": 4, "closed": 5,
           "adapter_error": 6, "discarded": 7}


def wait_until(deadline, mode="hybrid_spin"):
    remaining = deadline - time.monotonic_ns()
    if mode == "cooperative":
        if remaining > 0:
            time.sleep(remaining / 1e9)
        return
    if remaining > 100_000:
        time.sleep((remaining - 50_000) / 1e9)
    while time.monotonic_ns() < deadline:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.config.read_text())
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(case, indent=2))
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2:
        raise RuntimeError("benchmark requires >=2 allowed CPUs for declared placement")
    collector_cpu, producer_cpus = cpus[1], [cpus[0]] + cpus[2:]
    resource.setrlimit(resource.RLIMIT_AS, (case.get("address_space_MiB", 1024) * 1024**2,) * 2)
    started = time.monotonic_ns()
    adapter = ADAPTERS[case["adapter"]](directory, case, args.upstream)
    opened = time.monotonic_ns()
    # Existing helper threads get one collector core. No host-wide setting changes.
    for task in Path("/proc/self/task").iterdir():
        if int(task.name) != os.getpid():
            os.sched_setaffinity(int(task.name), {collector_cpu})
    offsets = plan(case)
    n = len(offsets)
    samples = (bytearray(latency_sample(i, case["seed"], case.get("sample_every", 64)) for i in range(n))
               if case.get("measurement_pass", "latency") == "latency" else None)
    producers = case.get("producers", 1)
    warmups = case.get("warmup_events", 100)
    # Warmup uses a different run identity and is excluded from all output counts.
    for i in range(warmups):
        adapter.offer({"run_id": "warmup", "producer_id": 0,
                       "source_seq": i, "source_timestamp": 0,
                       "body": payload(i, case)})
    # Give asynchronous warmup a chance to drain. It remains a separate tagged cohort.
    time.sleep(case.get("warmup_s", 0.2))
    go = threading.Event()
    counts = [Counter() for _ in range(producers)]
    errors = []
    receipts = [None] * producers
    logical = [0] * producers
    base = [0]
    stop_attempts = [0]
    competitor = None
    if case.get("competitor"):
        spec = case["competitor"]
        cpu = producer_cpus[0] if spec["placement"] == "producer_core" else producer_cpus[-1]
        competitor = subprocess.Popen(["taskset", "-c", str(cpu),
            str(args.upstream.parent / "build/capture-competitor"), str(spec["MiB"]), str(spec["duty"])],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    usage_start = resource.getrusage(resource.RUSAGE_SELF)

    def produce(p):
        try:
            os.sched_setaffinity(0, {producer_cpus[p % len(producer_cpus)]})
            data = bytearray()
            counts[p]["sampled"] = 0

            @lru_cache(maxsize=128)
            def body_for(key):
                return payload(key, case)

            period = math.lcm(case.get("cardinality", 64), 256 if case.get("entropy") == "high" else 1)
            go.wait()
            wait_until(base[0], case.get("pacer", "hybrid_spin"))
            indices = [i for i in range(n) if producer_for(i, case) == p] if case.get("skew") else range(p, n, producers)
            for pos, i in enumerate(indices):
                if offsets[i] is None:
                    scheduled = time.monotonic_ns()
                else:
                    scheduled = base[0] + offsets[i]
                    wait_until(scheduled, case.get("pacer", "hybrid_spin"))
                if time.monotonic_ns() > stop_attempts[0]:
                    counts[p]["scheduled_not_attempted"] += len(indices) - pos
                    break
                # Mixed/giant/nested cases use the actual index to preserve shapes.
                body = (payload(i, case) if case.get("mix") or case.get("giant_every")
                        else body_for(i % period))
                event = {"run_id": case["run_id"], "producer_id": p,
                         "source_seq": i, "source_timestamp": scheduled,
                         "body": json.loads(body) if case.get("structured") else body}
                sampled = samples is not None and samples[i]
                begin = time.monotonic_ns() if sampled else 0
                status = adapter.offer(event)
                end = time.monotonic_ns() if sampled else 0
                code = RESULTS[status]
                counts[p][status] += 1
                counts[p]["attempted"] += 1
                logical[p] += 64 + len(body.encode())
                # Retain every identity/result, only sample hot-path latency fields.
                if sampled:
                    counts[p]["sampled"] += 1
                data.extend(RECEIPT.pack(i, begin if sampled else 0, scheduled,
                                         end - begin if sampled else 0, code))
            receipts[p] = data
        except BaseException as exc:  # noqa: BLE001 - thread exceptions must reach the controller
            errors.append({"producer": p, "type": type(exc).__name__, "error": str(exc)})

    threads = [threading.Thread(target=produce, args=(p,), name=f"producer-{p}") for p in range(producers)]
    for thread in threads:
        thread.start()
    base[0] = case.get("start_at_ns", time.monotonic_ns() + 50_000_000)
    if base[0] < time.monotonic_ns():
        raise RuntimeError("HARNESS_INVALID: shared start deadline missed")
    horizon_ns = int(case["duration_s"] * 1e9)
    stop_attempts[0] = base[0] + horizon_ns + int(case.get("dispatch_drain_s", 1) * 1e9)
    (directory / "measuring.json").write_text(json.dumps({"t0": base[0], "H_ns": horizon_ns}))
    go.set()
    if n == 0:
        time.sleep(case["duration_s"])
    for thread in threads:
        thread.join()
    attempts_end = time.monotonic_ns()
    if case.get("arrival") == "closed":
        horizon_ns = max(1, attempts_end - base[0])
    if case["adapter"] == "hqsb":
        problems = list(errors)
        unattempted = n - sum(counter.get("attempted", 0) for counter in counts)
        if unattempted:
            problems.append({"source": "benchmark_schedule", "reason": "scheduled_not_attempted", "count": unattempted})
        adapter.source_problems = problems
    closed = adapter.close()
    ended = closed["close_end_ns"]
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    if competitor is not None:
        competitor.terminate()
        try:
            competitor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            competitor.kill()
            competitor.wait()
    if case["adapter"] == "hqsb":
        report = closed["writer_report"]
        if report["counts"]["written"] <= 100_000 and sum(a["bytes"] for a in report["artifacts"]) <= 512 * 1024**2:
            from hqsb.observation.storage import publish

            closed["publication_start_ns"] = time.monotonic_ns()
            published = publish(directory, expected_run_id=case["run_id"])
            closed.update(published_ns=time.monotonic_ns(), publication=published["publication"])
        else:
            closed["publication_unavailable"] = "CORE exceeds current public reader limits"
    for p, data in enumerate(receipts):
        if data is not None:
            (directory / f"receipts-{p}.bin").write_bytes(data)
    report = {"adapter": case["adapter"], "track": "PYTHON_STRUCTURED" if case.get("structured") else "PYTHON", "case": case,
              "scheduled": n, "per_producer": counts, "errors": errors,
              "logical_attempted_bytes": sum(logical), "t0_ns": base[0],
              "H_ns": horizon_ns, "attempts_end_ns": attempts_end,
              "endpoint_end_ns": ended, "startup_ns": opened - started,
              "terminal": closed, "cpu_seconds": (usage_end.ru_utime + usage_end.ru_stime
                                                   - usage_start.ru_utime - usage_start.ru_stime),
              "maxrss_KiB": usage_end.ru_maxrss,
              "minor_faults": usage_end.ru_minflt - usage_start.ru_minflt,
              "major_faults": usage_end.ru_majflt - usage_start.ru_majflt,
              "voluntary_switches": usage_end.ru_nvcsw - usage_start.ru_nvcsw,
              "involuntary_switches": usage_end.ru_nivcsw - usage_start.ru_nivcsw,
              "placement": {"collector_cpu": collector_cpu, "producer_cpus": producer_cpus},
              "acknowledgement": adapter.acknowledgement,
              "measurement": f"CLOCK_MONOTONIC; latency pass hash sample 1/{case.get('sample_every',64)}, throughput pass no call clocks; Python dispatch/source CPU included; no per-event completion timestamp"}
    (directory / "worker.json").write_text(json.dumps(report, indent=2))
    if errors:
        raise RuntimeError("producer errors retained in worker report")


if __name__ == "__main__":
    main()
