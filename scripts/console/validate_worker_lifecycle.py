"""Verify only our sudo IPC workers, without loading a model or using a GPU.

Run through scripts/remote_run.sh. This script never signals the Console server
or unrelated processes and does not read or record its access token.
"""

import argparse
import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hqsb.console.privileged_worker import PrivilegedActor


def process_snapshot(pid):
    try:
        fields = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"Name", "State", "Uid", "PPid"}:
                fields[key] = value.strip()
        raw_stat = Path(f"/proc/{pid}/stat").read_text()
        fields["start_ticks"] = raw_stat.rsplit(")", 1)[1].split()[19]
        return {"pid": pid, **fields}
    except (OSError, ValueError, IndexError):
        return None


def wait_gone(before, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        current = process_snapshot(before["pid"])
        if current is None or current.get("start_ticks") != before.get("start_ticks"):
            return {"exited": True, "reaped": True}
        if current.get("State", "").startswith("Z"):
            return {"exited": True, "reaped": False, "state": "zombie_no_execution"}
        time.sleep(0.1)
    return {"exited": False, "last_snapshot": process_snapshot(before["pid"])}


def actor(data_dir):
    return PrivilegedActor(
        {
            "id": "lifecycle-probe",
            "name": "Lifecycle handshake only",
            "provider": "pytorch",
            "model_path": "~/models/hqsb/Qwen3-1.7B",
        },
        data_dir=data_dir,
    )


def helper(data_dir):
    worker = actor(data_dir)
    worker._start()
    print(
        json.dumps(
            {
                "worker_pid": worker.process.pid,
                "sudo_pid": worker.process.child.pid,
                "helper_pid": os.getpid(),
                "helper_uid": os.geteuid(),
            }
        ),
        flush=True,
    )
    # Default SIGTERM handling intentionally skips graceful Python cleanup.
    while True:
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(".console"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/console/v02/worker-lifecycle.json")
    )
    parser.add_argument("--helper", action="store_true")
    args = parser.parse_args()
    if args.helper:
        helper(args.data_dir)
        return
    evidence = {
        "started_at": time.time(),
        "verdict": "FAIL",
        "checks": [],
        "model_loaded": False,
        "gpu_work_requested": False,
        "api_uid": os.geteuid(),
    }
    worker = child = None

    def check(name, condition, details=None):
        evidence["checks"].append(
            {"name": name, "passed": bool(condition), "details": details}
        )
        print(json.dumps({"check": name, "passed": bool(condition)}), flush=True)
        if not condition:
            raise AssertionError(name)

    try:
        check("control_process_not_root", os.geteuid() != 0)
        worker = actor(args.data_dir)
        worker._start()
        sudo = worker.process.child
        snapshot = process_snapshot(worker.process.pid)
        evidence["normal_worker"] = snapshot
        check(
            "device_worker_root_uid",
            snapshot is not None and snapshot["Uid"].split()[0] == "0",
        )
        worker.close()
        result = wait_gone(snapshot)
        evidence["normal_close"] = {**result, "sudo_returncode": sudo.returncode}
        check("normal_close_exits_worker", result["exited"])
        check(
            "normal_close_without_interpreter_abort",
            sudo.returncode == 0,
            sudo.returncode,
        )
        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--helper",
                "--data-dir",
                str(args.data_dir.resolve()),
            ],
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
        )
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            check("helper_handshake_ready", bool(selector.select(timeout=40)))
            line = child.stdout.readline(8192)
        identity = json.loads(line)
        evidence["helper"] = identity
        check(
            "helper_is_owned_process",
            identity["helper_pid"] == child.pid
            and identity["helper_uid"] == os.geteuid(),
        )
        snapshot = process_snapshot(identity["worker_pid"])
        evidence["disconnected_worker"] = snapshot
        check(
            "helper_device_worker_root_uid",
            snapshot is not None and snapshot["Uid"].split()[0] == "0",
        )
        child.terminate()
        child.wait(timeout=5)
        result = wait_gone(snapshot)
        evidence["control_disconnect"] = {
            **result,
            "helper_returncode": child.returncode,
        }
        check("control_loss_reclaims_root_worker", result["exited"])
        evidence["verdict"] = "PASS"
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            if child.stdout is not None:
                child.stdout.close()
        if worker is not None:
            try:
                worker.close()
            except Exception as exc:
                evidence["cleanup_error"] = str(exc)
                evidence["verdict"] = "FAIL"
        evidence["finished_at"] = time.time()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "verdict": evidence["verdict"],
                    "checks": len(evidence["checks"]),
                    "output": str(args.output),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
