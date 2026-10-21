"""Replay saved PyTorch inference in an owned memory-probe child process."""
import ctypes
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import uuid

from hqsb.core.optimization import fingerprint
from hqsb.observation.storage import atomic_json


def main():
    from hqsb.backends.interactive import InteractivePyTorch
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
    config = {**spec["config"], "attention": spec["options"]["attention"]}
    provider = InteractivePyTorch(config)
    report = {"attempt_id": attempt.name, "run_id": spec["run_id"]}
    try:
        atomic_json(root / "progress.json", {"stage": "loading", "attempt_id": attempt.name})
        load = provider.load(**spec["options"]["load_options"])
        actual = {"engine": "pytorch", "model_manifest_sha256": provider.source_identity["hash"],
                  "kernel_plan_sha256": fingerprint(provider.kernel_session.plan) if getattr(provider, "kernel_session", None) else None,
                  "requested": spec["options"], "versions": provider.source_environment["packages"],
                  "hardware_inventory": {"device_name": load["device_name"], "machine": provider.source_environment["machine"]},
                  "environment_freeze_sha256": None}
        report["engine"] = actual
        # The hook was loaded by the child's loader; only this producer opens
        # the collection window after InteractivePyTorch's explicit warmup.
        hook = ctypes.CDLL(os.environ["NEUTRINO_HOOK_DRIVER"])
        for name in ("hqsb_eretrace_start", "hqsb_eretrace_stop"):
            getattr(hook, name).argtypes = []
            getattr(hook, name).restype = ctypes.c_int
        import torch
        torch.cuda.synchronize()
        if hook.hqsb_eretrace_start():
            raise RuntimeError("DMAT_WINDOW_START")
        report["window_start_unix_s"] = time.time()
        atomic_json(root / "progress.json", {"stage": "capturing", "attempt_id": attempt.name})
        result = None
        try:
            request = {**spec["workload"]["request"], "remaining_ms": spec["plan"]["timeout_s"] * 1000}
            for event in provider.generate(request, cancel):
                if event["kind"] == "result":
                    result = event
        finally:
            torch.cuda.synchronize()
            if hook.hqsb_eretrace_stop():
                raise RuntimeError("DMAT_WINDOW_STOP")
            report["window_end_unix_s"] = time.time()
        if cancel.is_set():
            report["status"] = "cancelled"
        elif not result or result.get("request_error") or result.get("finish_reason") in {"timed_out", "cancelled"}:
            raise RuntimeError("DMAT_GENERATION_FAILED:" + str(result))
        else:
            report.update(status="completed", result={"metrics": {"native_benchmark": {
                "requests": [{"text": result["text"], "status": "completed"}],
                "workload_sha256": fingerprint(spec["workload"]),
            }}}, generated_token_ids=result["metrics"].get("generated_token_ids"))
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        cancel.set()
        try:
            provider.close()
            report["cleanup"] = "succeeded"
        except Exception as exc:
            report.update(status="failed", cleanup="unconfirmed", error=str(exc))
        atomic_json(attempt / "result.json", report)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
