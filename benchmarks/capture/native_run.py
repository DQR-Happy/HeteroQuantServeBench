"""Native NanoLog and empty-adapter diagnostics, separate from Python rankings."""

import argparse
import fcntl
import json
import os
import random
import shutil
import subprocess
import uuid
from pathlib import Path

from benchmarks.capture.oracle import verify
from benchmarks.capture.run import write_summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    root, upstream = args.output.resolve(), args.upstream.resolve()
    root.mkdir(parents=True, exist_ok=False)
    lock = (upstream.parent / "measurement.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    (root / "storage-policy.json").write_text(json.dumps({"raw": "original NanoLog binary and all receipts retained",
        "derived": "decoded.txt removed only after independent verification; regenerable with pinned decompressor"}, indent=2))
    cases = [(rate, size, producers, arrival)
             for rate, size, producers, arrival in (
                 (100_000, 64, 1, "periodic"), (1_000_000, 64, 1, "periodic"),
                 (1_000_000, 64, 4, "periodic"), (100_000, 4096, 1, "periodic"),
                 (100_000, 64, 1, "closed"), (100_000, 64, 4, "closed"))]
    rng = random.Random(20261017)
    affinities = sorted(os.sched_getaffinity(0))[:4]
    resolved = []
    for rate, size, producers, arrival in cases:
        for repeat in range(args.repeats):
            adapters = ["null", "nanolog"]
            rng.shuffle(adapters)
            for adapter in adapters:
                if shutil.disk_usage(root).free < 4 * 1024**3:
                    raise RuntimeError("disk free below 4GiB safety floor; refusing new native run")
                seconds = 2
                n = 1_000_000 if arrival == "closed" else rate * seconds
                run_id = "run_" + uuid.uuid4().hex
                directory = root / f"{len(resolved):04d}-{adapter}-{rate}-{size}-{producers}-{arrival}"
                directory.mkdir()
                command = ["taskset", "-c", ",".join(map(str, affinities)), str(upstream.parent / "build/capture-native"),
                           adapter, str(directory), run_id, str(rate), str(seconds), str(size), str(producers),
                           str(100 + repeat), arrival, str(n)]
                with (directory / "native.log").open("wb") as log:
                    proc = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, timeout=90, check=False)
                resolved.append({"argv": command, "repeat": repeat, "returncode": proc.returncode})
                if proc.returncode:
                    (directory / "failure.json").write_text(json.dumps({"adapter": adapter,
                        "case_id": str(directory.name), "status": "native_failed", "exit_code": proc.returncode}))
                else:
                    verify(directory, upstream)
                    # Pretty text is a reproducible derivative, not the raw log.
                    if (directory / "decoded.txt").exists():
                        (directory / "decoded.txt").unlink()
                print(directory.name, proc.returncode, flush=True)
                write_summary(root)
    (root / "resolved-case-list.json").write_text(json.dumps(resolved, indent=2))


if __name__ == "__main__":
    main()
