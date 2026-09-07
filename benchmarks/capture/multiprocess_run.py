"""W11 independent captures sharing a fixed CPU set; not shared-collector IPC."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from benchmarks.capture.oracle import verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    root, upstream = args.output.resolve(), args.upstream.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpus[4]})
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONHASHSEED="0")
    env["LD_PRELOAD"] = subprocess.check_output(["g++", "-print-file-name=libstdc++.so.6"], text=True).strip()
    rows = []
    with (upstream.parent / "measurement.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for count in (1, 2, 4):
            for repeat in range(args.repeats):
                # Rotate order to reduce fixed temporal bias.
                adapters = ["hqsb", "nanolog", "otel", "hindsight"]
                adapters = adapters[repeat:] + adapters[:repeat]
                for adapter in adapters:
                    group = root / f"p{count}-{adapter}-r{repeat}"
                    group.mkdir()
                    start = time.monotonic_ns() + 5_000_000_000
                    children, metrics = [], []
                    try:
                        for index in range(count):
                            out = group / f"worker-{index}"
                            config = group / f"worker-{index}.json"
                            case = {"adapter": adapter, "id": f"independent-{count}", "workload": "W11",
                                    "duration_s": 10, "rate": 1000, "body_bytes": 64, "producers": 1,
                                    "seed": 100 + repeat, "repeat": repeat, "measurement_pass": "latency",
                                    "start_at_ns": start, "run_id": "run_" + uuid.uuid4().hex}
                            config.write_text(json.dumps(case))
                            cwd = group / f"cwd-{index}"
                            cwd.mkdir()
                            log = (group / f"worker-{index}.log").open("wb")
                            cmd = ["taskset", "-c", ",".join(map(str, cpus[:4])), sys.executable,
                                   "-m", "benchmarks.capture.worker", "--config", str(config),
                                   "--output", str(out), "--upstream", str(upstream)]
                            process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log,
                                                       stderr=subprocess.STDOUT, start_new_session=True)
                            log.close()
                            children.append((process, out))
                        for process, out in children:
                            code = process.wait(timeout=60)
                            if code:
                                raise RuntimeError(f"worker {process.pid} exit {code}")
                            metrics.append(verify(out, upstream))
                        finish = max(json.loads((out / "worker.json").read_text())["endpoint_end_ns"] for _, out in children)
                        seconds = max(10, (finish - start) / 1e9)
                        valid = sum(m["valid_unique"] for m in metrics)
                        row = {"processes": count, "adapter": adapter, "repeat": repeat,
                               "input_events": 10_000 * count, "valid_unique": valid,
                               "group_elapsed_s": seconds, "group_goodput": valid / seconds,
                               "combined_cpu_seconds": sum(m["cpu_seconds"] for m in metrics),
                               "status": "complete" if valid == 10_000 * count else "lossy",
                               "scope": "independent capture files, one collector core shared across processes; not one shared IPC collector"}
                    except Exception as exc:  # noqa: BLE001 - isolate each failed process group
                        row = {"processes": count, "adapter": adapter, "repeat": repeat,
                               "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                    finally:
                        for process, out in children:
                            if process.poll() is None:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                            for path in Path("/dev/shm").glob(f"hqsbbench_{process.pid}__*"):
                                path.unlink()
                    rows.append(row)
                    (group / "group-metrics.json").write_text(json.dumps(row, indent=2))
                    (root / "all-groups.json").write_text(json.dumps(rows, indent=2))
                    print(count, adapter, repeat, row["status"], flush=True)


if __name__ == "__main__":
    main()
