"""Transactional RTN jobs for an already-loaded, serially owned model.

The background thread is only a progress bridge: no second model is loaded and
the caller must serialize this job with inference/unload in the same worker.
"""

from __future__ import annotations

import queue
import shutil
import threading
import time
from pathlib import Path


def quantize_loaded_model(model, request, cancel, source_identity):
    from hqsb.quant.model_weight_only import (
        QuantizationCancelled,
        artifact_disk_usage,
        save_model_quant_artifact,
    )

    root = Path(request["artifact_dir"])
    if not root.is_absolute():
        raise ValueError("artifact_dir must be an absolute server-selected path")
    bits = request["bits"]
    group_size = request.get("group_size", 128 if bits == 4 else None)
    if bits not in {4, 8} or (bits == 8 and group_size is not None):
        raise ValueError(
            "Supported formats are RTN-W4 per-group and RTN-W8 per-channel"
        )
    if bits == 4 and group_size not in (32, 64, 128):
        raise ValueError("Console RTN-W4 requires group_size=32, 64 or 128")
    if root.exists():
        raise FileExistsError("Quantization will not overwrite an existing artifact")
    partial = root.with_name(root.name + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    updates = queue.Queue(maxsize=1)
    done, stop = threading.Event(), threading.Event()
    outcome = {}
    deadline = time.monotonic() + request.get("remaining_ms", 600000) / 1000

    def cancelled():
        return cancel.is_set() or stop.is_set() or time.monotonic() >= deadline

    def progress(value):
        # Retain the most recent update instead of making disk/GPU work wait for UI.
        try:
            updates.put_nowait(value)
        except queue.Full:
            try:
                updates.get_nowait()
            except queue.Empty:
                pass
            try:
                updates.put_nowait(value)
            except queue.Full:
                pass

    def work():
        try:
            manifest = save_model_quant_artifact(
                model,
                partial,
                bits=bits,
                group_size=group_size,
                source_model_hash=source_identity["hash"],
                source_revision=source_identity["revision"],
                progress=progress,
                cancel_check=cancelled,
            )
            if cancelled():
                raise QuantizationCancelled(
                    "Quantization cancelled before directory commit"
                )
            # The public artifact path only becomes visible after the manifest is complete.
            size = artifact_disk_usage(partial)
            partial.rename(root)
            outcome["manifest"] = manifest
            outcome["bytes"] = size
        except Exception as exc:
            outcome["error"] = exc
        finally:
            try:
                if partial.exists():
                    shutil.rmtree(partial)
            except OSError as exc:
                outcome["error"] = RuntimeError(
                    f"Failed to remove incomplete artifact: {exc}"
                )
            finally:
                done.set()

    thread = threading.Thread(target=work, name="hqsb-rtn-job", daemon=True)
    try:
        thread.start()
        yield {"kind": "progress", "stage": "starting", "completed_tensors": 0}
        last_progress_at = time.monotonic()
        while not done.is_set() or not updates.empty():
            try:
                value = updates.get(timeout=0.2)
            except queue.Empty:
                continue
            now = time.monotonic()
            if now - last_progress_at < 0.1 and not done.is_set():
                continue
            last_progress_at = now
            yield {"kind": "progress", **value}
        error = outcome.get("error")
        if error is not None and not isinstance(error, QuantizationCancelled):
            raise error
        status = "cancelled" if error is not None else "completed"
        manifest = outcome.get("manifest")
        quantization = {
            "status": status,
            "method": "rtn",
            "bits": bits,
            "group_size": group_size,
            "artifact_id": manifest["artifact_id"] if manifest else None,
            "source_identity_scope": source_identity["scope"],
            "execution_path": "storage_only",
            "quality": "not_evaluated",
            "native_deployment_available": False,
            "resident_model_modified": False,
            "bytes": outcome.get("bytes"),
            "coverage": manifest["coverage"] if manifest else None,
            "weight_error": manifest["offline"] if manifest else None,
            "limitations": [
                "This artifact does not include excluded FP16 weights and is not a standalone model.",
                "Weight reconstruction error does not establish model output quality.",
                "Creating packed storage does not change the resident FP16 inference path.",
            ],
        }
        yield {
            "kind": "result",
            "finish_reason": (
                "timed_out"
                if error is not None and time.monotonic() >= deadline
                else "cancelled"
                if error is not None
                else "stop"
            ),
            "artifact_id": quantization["artifact_id"],
            "manifest": manifest,
            "metrics": {"quantization": quantization},
        }
    finally:
        stop.set()
        if thread.ident is not None:
            # Do not let unload/inference race a cancelled quantization chunk.
            thread.join()
        elif partial.exists():
            shutil.rmtree(partial)
