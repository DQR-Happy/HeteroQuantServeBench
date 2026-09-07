"""Campaign controller: isolated processes, bounded storage, raw results retained."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from benchmarks.capture.cases import BASE, applicability, cases
from benchmarks.capture.oracle import verify

ACTIVE = set()


def terminate(signum, frame):
    for pid in list(ACTIVE):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    raise KeyboardInterrupt("campaign interrupted; completed raw runs retained")


def sample(pid):
    result = {"monotonic_ns": time.monotonic_ns()}
    try:
        for name in ("status", "smaps_rollup", "io"):
            for line in Path(f"/proc/{pid}/{name}").read_text().splitlines():
                key, _, rest = line.partition(":")
                if key in ("VmRSS", "VmHWM", "Rss", "Pss", "Private_Dirty", "Shared_Clean",
                           "read_bytes", "write_bytes", "syscr", "syscw"):
                    result[name + ":" + key] = int(rest.split()[0])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return result


def write_summary(root):
    records = []
    for path in sorted(root.glob("*/metrics.json")):
        records.append(json.loads(path.read_text()))
    for path in sorted(root.glob("*/failure.json")):
        records.append(json.loads(path.read_text()))
    (root / "all-runs.json").write_text(json.dumps(records, indent=2))
    columns = ["case_id", "adapter", "track", "endpoint", "status", "scheduled", "attempted",
               "valid_unique", "retained_ratio", "cohort_completion_rate", "logical_bandwidth_Bps",
               "cpu_ns_per_valid_event", "process_maxrss_KiB", "encoded_bytes", "close_ms",
               "semantic_corruption", "duplicate_records"]
    with (root / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return records


def campaign(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows = ([dict(BASE, **row) for row in json.loads(args.case_file.read_text())]
            if args.case_file else cases(args.tier))
    for row in rows:
        if not isinstance(row.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", row["id"]):
            raise ValueError("case ID must be a safe filename component")
        if type(row["rate"]) is not int or row["rate"] <= 0 or row["duration_s"] <= 0:
            raise ValueError("case requires positive integral rate and duration")
    if args.cases:
        selected = set(args.cases.split(","))
        rows = [row for row in rows if row["id"] in selected]
        if {row["id"] for row in rows} != selected:
            raise ValueError("unknown selected case")
    repeats = args.repeats or {"smoke": 1, "screening": 3, "formal": 10, "soak": 1}[args.tier]
    adapters = args.adapters.split(",")
    passes = args.passes.split(",")
    if not set(passes) <= {"latency", "throughput"}:
        raise ValueError("unknown measurement pass")
    exclusions = []
    resolved = []
    rng = random.Random(20261017)
    for case in rows:
        for repeat in range(repeats):
            order = [(adapter, pacer) for adapter in adapters
                     for pacer in case.get("pacer_variants", [case.get("pacer", "cooperative")])]
            rng.shuffle(order)
            for adapter, pacer in order:
                reason = applicability(adapter, case)
                if reason:
                    exclusions.append({"case_id": case["id"], "adapter": adapter,
                                       "status": reason[0], "reason": reason[1]})
                    continue
                for measurement_pass in passes:
                    c = dict(case, adapter=adapter, pacer=pacer, seed=100 + repeat,
                             run_id="run_" + uuid.uuid4().hex, repeat=repeat,
                             measurement_pass=measurement_pass,
                             resource_profile="CPU4-affinity-v1; collector=1 core; producers share remaining 3; observer separate",
                             tier=args.tier)
                    resolved.append(c)
    (root / "resolved-case-list.json").write_text(json.dumps(resolved, indent=2))
    (root / "exclusions.json").write_text(json.dumps(exclusions, indent=2))
    fingerprints = {}
    source = Path(__file__).resolve().parents[2]
    for path in sorted((source / "benchmarks/capture").rglob("*")):
        if path.is_file() and "__pycache__" not in str(path):
            fingerprints[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (source / "hqsb/observation").glob("*.py"):
        fingerprints[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "source-manifest.json").write_text(json.dumps(fingerprints, indent=2))
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 5:
        raise RuntimeError("CPU4 profile requires 4 SUT + 1 observer allowed CPUs")
    sut_cpus = allowed[:4]
    os.sched_setaffinity(0, {allowed[4]})
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONHASHSEED="0")
    # Conda's Python may preload an older C++ ABI than the system compiler.
    # Freeze and apply the same runtime to every worker, never replace Conda files.
    libstdcxx = subprocess.check_output(["g++", "-print-file-name=libstdc++.so.6"], text=True).strip()
    if not Path(libstdcxx).is_file():
        raise RuntimeError("compiler C++ runtime not found")
    env["LD_PRELOAD"] = libstdcxx
    (root / "runtime.json").write_text(json.dumps({"python": sys.executable, "libstdcxx": libstdcxx,
        "allowed_cpus": allowed, "sut_cpus": sut_cpus, "observer_cpu": allowed[4],
        "note": "stock buffers retained; CPU affinity equal, exact collector memory quota not claimed"}, indent=2))
    for index, case in enumerate(resolved):
        if shutil.disk_usage(root).free < 3 * 1024**3:
            raise RuntimeError("disk free below 3GiB floor; campaign stopped, not silently deleting evidence")
        run_name = f"{index:04d}-{case['id']}-{case['adapter']}-r{case['repeat']}"
        directory = root / run_name
        config = root / (run_name + ".json")
        config.write_text(json.dumps(case))
        print(f"[{index+1}/{len(resolved)}] {run_name}", flush=True)
        # NanoLog may create its default file before hf_open: unique CWD per run.
        cwd = root / (run_name + "-cwd")
        cwd.mkdir()
        with (root / (run_name + ".log")).open("wb") as log:
            cmd = ["taskset", "-c", ",".join(map(str, sut_cpus)), sys.executable,
                   "-m", "benchmarks.capture.worker", "--config", str(config),
                   "--output", str(directory), "--upstream", str(args.upstream.resolve())]
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            ACTIVE.add(proc.pid)
            samples = []
            deadline = time.monotonic() + case["duration_s"] + case.get("warmup_s", 0.2) + 90
            killed = False
            while proc.poll() is None:
                samples.append(sample(proc.pid))
                if time.monotonic() > deadline:
                    os.killpg(proc.pid, signal.SIGKILL)
                    killed = True
                    break
                time.sleep(0.05)
            code = proc.wait()
            ACTIVE.discard(proc.pid)
        for path in Path("/dev/shm").glob(f"hqsbbench_{proc.pid}__*"):
            path.unlink()  # Only this benchmark worker's unique service prefix.
        directory.mkdir(exist_ok=True)
        (directory / "process-timeline.json").write_text(json.dumps(samples))
        if code:
            failure = {"case_id": case["id"], "adapter": case["adapter"],
                       "status": "timeout" if killed else "worker_failed", "exit_code": code}
            (directory / "failure.json").write_text(json.dumps(failure, indent=2))
        else:
            try:
                verify(directory, args.upstream.resolve())
            except Exception as exc:  # noqa: BLE001 - preserve oracle failures as benchmark evidence
                (directory / "failure.json").write_text(json.dumps({"case_id": case["id"],
                    "adapter": case["adapter"], "status": "oracle_failed", "error": f"{type(exc).__name__}: {exc}"}))
        write_summary(root)
    write_summary(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("smoke", "screening", "formal", "soak"), default="smoke")
    parser.add_argument("--adapters", default="null,hqsb,nanolog,otel")
    parser.add_argument("--cases")
    parser.add_argument("--case-file", type=Path)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--passes", default="throughput,latency")
    parser.add_argument("--wait-for-lock", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    with (args.upstream.resolve().parent / "measurement.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_for_lock else fcntl.LOCK_NB))
        campaign(args)


if __name__ == "__main__":
    main()
