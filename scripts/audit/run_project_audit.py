#!/usr/bin/env python3
"""Run reproducible project checks on the execution host, preserving logs.

Invoke from the Mac with ./scripts/remote_run.sh python3
scripts/audit/run_project_audit.py. This is a software regression audit, not a
stage experiment or permission to change historical acceptance verdicts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import shutil

ROOT = Path(__file__).resolve().parents[2]


def source_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    return sorted(
        {
            path
            for path in result.stdout.decode().split("\0")
            if path
            and (root / path).is_file()
            and not path.startswith(("docs/manual/generated/", "docs/audit/"))
        }
    )


def run_check(
    name: str,
    command: list[str],
    *,
    root: Path,
    output: Path,
    env: dict[str, str],
    timeout: int,
) -> dict:
    log = output / f"{name}.log"
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            returncode = 124
    record = {
        "name": name,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "seconds": round(time.monotonic() - start, 3),
        "log": log.name,
        "passed": returncode == 0,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def audit(root: Path, output: Path, *, timeout: int, hardware: bool) -> list[dict]:
    env = dict(os.environ)
    env.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    if not hardware:
        env["CUDA_VISIBLE_DEVICES"] = ""
    # Never inherit a PYTHONPATH pointing at the original source snapshot.
    env["PYTHONPATH"] = str(root)
    checks = [
        (
            "pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "hardware"
                if hardware
                else "not hardware and not e2e and not performance",
                "-q",
                f"--junitxml={output / 'pytest.xml'}",
            ],
        ),
        ("dependencies", [sys.executable, "scripts/audit/import_dependency_gate.py"]),
    ]
    return [
        run_check(name, command, root=root, output=output, env=env, timeout=timeout)
        for name, command in checks
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="new evidence directory (must not exist)")
    parser.add_argument("--timeout", type=int, default=900, help="seconds per check")
    parser.add_argument(
        "--clean-source",
        action="store_true",
        help="copy public source without ignored experiments/caches",
    )
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="run only hardware-marked tests on the target",
    )
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (ROOT / (args.output or f"reports/project_audit/{stamp}")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = source_paths(ROOT)
    manifest = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths
    }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if args.clean_source:
        with tempfile.TemporaryDirectory(prefix="hqsb-clean-audit-") as tmp:
            clean = Path(tmp) / "source"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--no-hardlinks",
                    "--no-checkout",
                    str(ROOT),
                    str(clean),
                ],
                check=True,
            )
            for path in paths:
                target = clean / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / path, target)
            checks = audit(clean, output, timeout=args.timeout, hardware=args.hardware)
    else:
        checks = audit(ROOT, output, timeout=args.timeout, hardware=args.hardware)
    summary = {
        "schema": "hqsb.project_audit/v1",
        "started_at_utc": stamp,
        "git_commit": git,
        "git_dirty": bool(dirty),
        "source_files": len(manifest),
        "python": sys.version,
        "clean_source": args.clean_source,
        "hardware": args.hardware,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "stage_claim_allowed": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
