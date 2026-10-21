"""Replay one frozen native workload under Nsight, including warmup and cleanup."""
import contextlib
import json
from pathlib import Path
import signal
import sys
import threading
import time
import uuid

from hqsb.observation.storage import atomic_json


def main():
    from hqsb.backends.native_engine import NativeEngineProvider
    from hqsb.serving.native_benchmark import execute
    spec = json.loads(Path(sys.argv[1]).read_text())
    root = Path(spec["directory"])
    attempt = root / "attempts" / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    cancel = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: cancel.set())

    def watch():
        while not cancel.wait(.1):
            if (root / "cancel").exists():
                cancel.set()
    threading.Thread(target=watch, daemon=True).start()
    config = {**spec["config"], "engine_log_root": str(attempt / "engine"),
              "_native_profile_control": True}
    if config.get("_eretrace"):
        config["_eretrace"]["directory"] = str(root)
    provider = NativeEngineProvider(config)
    report = {"attempt_id": attempt.name, "measurement_role": "diagnostic",
              "run_id": spec["run_id"], "source_run_id": spec["plan"]["source_run_id"]}
    try:
        atomic_json(root / "progress.json", {"stage": "loading", "attempt_id": attempt.name})
        provider.load(engine_options=spec["options"])

        @contextlib.contextmanager
        def window():
            import httpx
            body = {"activities": ["CUDA_PROFILER"]} if config["provider"] == "sglang" else None
            with httpx.Client(timeout=60, trust_env=False) as client:
                client.post(provider.base_url + "/start_profile", json=body).raise_for_status()
                report["window_start_unix_s"] = time.time()
                atomic_json(root / "progress.json", {"stage": "capturing", "attempt_id": attempt.name})
                try:
                    yield
                finally:
                    client.post(provider.base_url + "/stop_profile").raise_for_status()
                    report["window_end_unix_s"] = time.time()

        workload = {**spec["workload"], "expected_epoch": "diagnostic", "remaining_ms": spec["plan"]["timeout_s"] * 1000}
        result = execute(provider, workload, cancel, capture_window=window)
        status = "cancelled" if cancel.is_set() else "completed" if result.get("finish_reason") == "stop" and not result.get("request_error") else "failed"
        report.update(result=result, engine=provider.receipt, status=status)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        if provider.receipt:
            report["engine"] = provider.receipt
        try:
            provider.close()
            report["cleanup"] = "succeeded"
        except Exception as exc:
            report.update(status="failed", cleanup="unconfirmed", error=f"NATIVE_ENGINE_CLEANUP:{exc}")
        atomic_json(attempt / "result.json", report)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
