"""Native matched-contract capture experiments. All execution stays on Linux CPU."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import signal
import struct
import subprocess
import time
import uuid

from benchmarks.capture.fair.decode import events
from benchmarks.capture.oracle import verify
from benchmarks.capture.fair.lttng_control import (
    bounded_command as bounded_command, cli, descendants, start as start_lttng, stop as stop_lttng,
)
from benchmarks.capture.run import sample
from benchmarks.capture.schedule import payload, plan

EXTENSIONS = {"null": "null", "nanolog": "nanolog.bin", "hindsight": "hindsight.blocks",
              "quill": "quill.log", "perfetto": "perfetto.pftrace", "lttng": "unused",
              "dftracer": "dftrace", "hqsb": "hqsb.unused", "otel": "otel.jsonl"}
LIMIT = 1024 * 1024**2


def input_files(root, case):
    offsets = plan(case)
    if len(offsets) > 4_000_000:
        raise ValueError("resource_limited: 4M source bound")
    period = math.lcm(1000 if case.get("mix") else 1,
                      case.get("giant_every", 1), case.get("cardinality", 64),
                      256 if case.get("entropy") == "high" else 1)
    key = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:20]
    directory = root / "inputs" / key
    if not directory.exists():
        directory.mkdir(parents=True)
        (directory / "offsets.bin").write_bytes(b"".join(struct.pack("<Q", t) for t in offsets))
        with (directory / "bodies.bin").open("wb") as stream:
            stream.write(struct.pack("<I", period))
            for i in range(period):
                body = payload(i, case).encode()
                stream.write(struct.pack("<I", len(body)))
                stream.write(body)
        (directory / "case.json").write_text(json.dumps(case, indent=2))
    return directory


def execute(work, output, case, adapter, profile, repeat, measurement_pass, label):
    directory = output / label
    directory.mkdir()
    case = dict(case, seed=case.get("seed", 100 + repeat), repeat=repeat,
                adapter=adapter, measurement_pass=measurement_pass,
                run_id="run_" + uuid.uuid4().hex, warmup_events=20,
                resource_profile="native CPU4; aggregate AS ceiling 1GiB; matched D2; no Python pacing")
    (directory / "config.json").write_text(json.dumps(case, indent=2))
    source = input_files(output, {k: v for k, v in case.items() if k not in {
        "run_id", "repeat", "adapter", "measurement_pass", "resource_profile", "warmup_events"}})
    env = dict(os.environ, MALLOC_ARENA_MAX="2", HQSB_FAIR_PROFILE=str(profile), HQSB_FAIR_ADAPTER=adapter,
               PYTHONPATH=str(work / "source") + ":/tmp/hqsb-capture-bench-20261017/venv/lib/python3.12/site-packages")
    old = Path("/tmp/hqsb-capture-bench-20261017")
    libraries = {"null": "null", "nanolog": env.get("HQSB_NANOLOG_LIBRARY", str(old / "build/libcapture_nanolog.so")),
                 "hindsight": str(work / "build/hindsight.so"), "quill": str(work / f"build/quill{profile}.so"),
                 "perfetto": str(work / "build/perfetto.so"), "lttng": str(Path(env["HQSB_LTTNG_PREFIX"]) / "lttng216.so") if env.get("HQSB_LTTNG_PREFIX") else str(work / "build/lttng.so"),
                 "dftracer": str(work / "build/dftracer.so"),
                 "hqsb": str(work / "build/python_bridge.so"), "otel": str(work / "build/python_bridge.so")}
    cpus = [192, 193, 194, 195]
    daemon = process = None
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    timeline = []
    try:
        if adapter == "lttng":
            daemon = start_lttng(directory, env, cpus[1], profile)
        cmd = ["taskset", "-c", ",".join(map(str, cpus)), "prlimit",
               "--as=" + str(384 * 1024**2 if adapter == "lttng" else LIMIT), "--stack=2097152", "--",
               str(work / "build/capture-fair"), libraries[adapter], str(directory / EXTENSIONS[adapter]),
               str(directory), case["run_id"], str(source / "offsets.bin"), str(source / "bodies.bin"),
               str(case.get("producers", 1)), str(int(case["duration_s"] * 1e9)),
               str(1 if adapter == "null" else 16 if measurement_pass == "latency" else 0),
               str(case["seed"]), str(int(case.get("skew", False))), str(case["warmup_events"])]
        with (directory / "native.log").open("w") as log:
            process = subprocess.Popen(cmd, cwd=directory, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + case["duration_s"] + 60
            while process.poll() is None:
                # Include known sessiond descendants; don't charge unrelated host processes.
                ids = {process.pid}
                if daemon:
                    ids.update(descendants({daemon.pid} | set(getattr(daemon, "hqsb_detached_pids", []))))
                timeline.append({"monotonic_ns": time.monotonic_ns(), "processes": {str(pid): sample(pid) for pid in ids}})
                if time.monotonic() > deadline:
                    raise TimeoutError("SUT exceeded bounded close deadline")
                time.sleep(0.02)
            if process.wait():
                raise RuntimeError("native worker failed; inspect native.log")
        raw = json.loads((directory / "native.json").read_text())
        if adapter == "lttng":
            cli(["stop"], env, directory, "stop", cpus[1])
            cli(["destroy", "hqsb"], env, directory, "destroy", cpus[1])
            for path in (directory / "ctf").rglob("*"):
                if path.is_file():
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
            raw["endpoint_end_ns"] = time.monotonic_ns()
            raw["encoded_output_files"] = [str(p.relative_to(directory)) for p in (directory / "ctf").rglob("*") if p.is_file()]
        elif adapter == "dftracer":
            raw["encoded_output_files"] = [p.name for p in directory.glob("dftrace*") if p.is_file()]
        elif adapter == "hqsb":
            raw["encoded_output_files"] = [str(p.relative_to(directory)) for p in (directory / "observations").glob("*")]
        else:
            raw["encoded_output_files"] = [] if adapter == "null" else [EXTENSIONS[adapter]]
        raw.update(adapter=adapter, case=case, track="NATIVE_SOURCE", acknowledgement="explicit" if adapter == "hqsb" else "unavailable_void_api",
                   errors=["adapter_error"] if raw.pop("adapter_errors") else [],
                   terminal={"endpoint": "discard_control" if adapter == "null" else "D2-close",
                             "close_start_ns": raw["close_start_ns"], "close_end_ns": raw["endpoint_end_ns"]})
        if adapter in {"hqsb", "otel"}:
            raw["terminal"].update(json.loads((directory / "python-terminal.json").read_text()))
        (directory / "worker.json").write_text(json.dumps(raw, indent=2))
    finally:
        if daemon:
            stop_lttng(daemon)
        for child in (process,):
            if child and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
        (directory / "process-tree-timeline.json").write_text(json.dumps(timeline))
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        (directory / "lifecycle-resource.json").write_text(json.dumps({
            "children_cpu_s": usage_after.ru_utime + usage_after.ru_stime - usage_before.ru_utime - usage_before.ru_stime,
            "scope": "reaped capture lifecycle children including setup/stop CLI; daemon descendant accounting depends on daemon reaping; decoder excluded",
            "aggregate_virtual_AS_ceiling_MiB": 1024, "cgroup_resident_limit": "unavailable; read-only cgroup"}, indent=2))
    os.environ["HQSB_TRACE_PROCESSOR"] = str(work / "build/trace_processor_shell")
    metrics = verify(directory, old / "upstream", decode_events=events)
    metrics["profile"] = profile
    metrics["pacer"] = "native_absolute_monotonic"
    metrics["lifecycle_children_cpu_s"] = usage_after.ru_utime + usage_after.ru_stime - usage_before.ru_utime - usage_before.ru_stime
    metrics["sampled_peak_tree_PSS_KiB"] = max((sum(p.get("smaps_rollup:Pss", 0) for p in x["processes"].values()) for x in timeline), default=None)
    if adapter == "null":
        receipt = struct.Struct("<QQQQB")
        lags = [b - s for path in directory.glob("receipts-*.bin")
                for _, b, s, _, _ in receipt.iter_unpack(path.read_bytes())]
        metrics["source_late_over_1ms_fraction"] = sum(t > 1_000_000 for t in lags) / len(lags) if lags else 0
        metrics["source_gate_pass"] = metrics["attempted"] == metrics["scheduled"] and metrics["source_late_over_1ms_fraction"] <= .01
    # Derived pretty-text is regenerable; preserve original encoded outputs/receipts.
    for name in ("decoded.txt", "decoded.csv", "decoded.jsonl"):
        if (directory / name).exists():
            (directory / name).unlink()
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--adapters", default="null,nanolog,hindsight,quill,perfetto,lttng")
    parser.add_argument("--profiles", default="1")
    parser.add_argument("--profile-map", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--passes", default="throughput")
    args = parser.parse_args()
    work, output = args.work.resolve(), args.output.resolve()
    profile_map = json.loads(args.profile_map.read_text()) if args.profile_map else {}
    output.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {196})
    resolved, exclusions = [], []
    for case in json.loads(args.cases.read_text()):
        for repeat in range(args.repeats):
            block = []
            for adapter in args.adapters.split(","):
                if (adapter == "nanolog" and not os.environ.get("HQSB_NANOLOG_LIBRARY")
                        and (case.get("mix") or case.get("giant_bytes", 0) >= 1024**2)):
                    exclusions.append(dict(case_id=case["id"], adapter=adapter, reason="original 1MiB staging; no record splitting")); continue
                for profile in profile_map.get(adapter, list(map(int, args.profiles.split(",")))):
                    for measurement_pass in args.passes.split(","):
                        block.append(dict(case=case, adapter=adapter, profile=profile, repeat=repeat, measurement_pass=measurement_pass))
            random.Random(20261017 + repeat).shuffle(block)
            resolved.extend(block)
    (output / "resolved.json").write_text(json.dumps(resolved, indent=2))
    (output / "exclusions.json").write_text(json.dumps(exclusions, indent=2))
    rows = []
    with Path("/tmp/hqsb-capture-bench-20261017/measurement.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index, item in enumerate(resolved):
            label = f"{index:04d}-{item['case']['id']}-{item['adapter']}-p{item['profile']}-r{item['repeat']}"
            print(index + 1, len(resolved), label, flush=True)
            if shutil.disk_usage(output).free < 2 * 1024**3:
                raise RuntimeError("storage floor; no evidence deletion")
            try:
                row = execute(work, output, label=label, **item)
            except Exception as exc:  # Boundary preserves failed trials rather than deleting them.
                row = dict(case_id=item["case"]["id"], adapter=item["adapter"], profile=item["profile"],
                           status="failed", error=f"{type(exc).__name__}: {exc}")
                path = output / label; path.mkdir(exist_ok=True)
                (path / "failure.json").write_text(json.dumps(row, indent=2))
            rows.append(row)
            (output / "all-runs.json").write_text(json.dumps(rows, indent=2))
    print(dict(Counter(r["status"] for r in rows)), flush=True)


if __name__ == "__main__":
    main()
