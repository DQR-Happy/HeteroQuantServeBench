"""Persistent reference selection over sealed runs; no model or device ownership."""

import re
import threading
import time

from hqsb.console.captures import safe_directory
from hqsb.core.optimization import fingerprint
from hqsb.experiments.artifacts import load_bundle, read_json
from hqsb.experiments.quality_contract import (
    compare_scores,
    context,
    is_floating_reference,
)
from hqsb.observation.storage import atomic_json


class QualityBaselines:
    def __init__(self, directory, store):
        self.directory, self.store = directory, store
        self.root = directory / "quality-baselines"
        self.lock = threading.RLock()

    def _read(self, identity):
        if not re.fullmatch(r"run_[0-9a-f]{32}", identity):
            raise ValueError("BASELINE_INVALID_RUN_ID")
        run = self.store.get(identity)
        if run["state"] != "completed" or run["cleanup"] != "succeeded":
            raise ValueError("BASELINE_COMPLETED_CLEAN_RUN_REQUIRED")
        bundle = load_bundle(safe_directory(self.directory / "captures", identity))
        signature = context(bundle)
        signature["device_name"] = (run["config"].get("actual_runtime") or {}).get(
            "device_name"
        )
        return run, bundle, signature

    def _path(self, key):
        if not re.fullmatch(r"[0-9a-f]{64}", key) or self.root.is_symlink():
            raise ValueError("BASELINE_INDEX_PATH_INVALID")
        path = self.root / (key + ".json")
        if path.is_symlink():
            raise ValueError("BASELINE_INDEX_PATH_INVALID")
        return path

    def _write(self, run, bundle, signature, previous=None, *, selection):
        key = fingerprint(signature)
        entry = {
            "run_id": run["id"],
            "manifest_sha256": bundle["manifest_sha256"],
            "registered_at": time.time(),
            "selection": selection,
        }
        history = (previous or {}).get("history", [])
        if not history or history[-1]["run_id"] != run["id"]:
            if len(history) >= 256:
                raise ValueError("BASELINE_HISTORY_LIMIT")
            history = [*history, entry]
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        atomic_json(
            self._path(key),
            {
                "schema_version": "quality-baseline-v1",
                "context": signature,
                "context_sha256": key,
                "primary": entry,
                "history": history,
            },
        )

    def register(self, identity):
        with self.lock:
            run, bundle, signature = self._read(identity)
            if not is_floating_reference(bundle):
                raise ValueError(
                    "BASELINE_UNQUANTIZED_WEIGHTS_DEFAULT_KV_NATIVE_KERNEL_REQUIRED"
                )
            compare_scores(bundle, bundle)
            path = self._path(fingerprint(signature))
            previous = read_json(path) if path.exists() else None
            self._write(run, bundle, signature, previous, selection="explicit")
            return self.for_run(identity)

    def for_run(self, identity):
        with self.lock:
            run, candidate, signature = self._read(identity)
            key = fingerprint(signature)
            path = self._path(key)
            eligible = is_floating_reference(candidate)
            if not path.exists():
                # Earliest compatible valid reference, never the lowest observed
                # PPL. This also backfills historical runs outside list pagination.
                for reference_id in self.store.quality_reference_ids(
                    signature["model_identity"]["hash"], signature["source_digest"]
                ):
                    try:
                        reference_run, reference, reference_context = self._read(
                            reference_id
                        )
                        if reference_context != signature or not is_floating_reference(
                            reference
                        ):
                            continue
                        compare_scores(reference, candidate)
                    except (ValueError, OSError, KeyError):
                        continue
                    self._write(
                        reference_run,
                        reference,
                        signature,
                        selection="earliest_verified_floating_run",
                    )
                    break
            common = {
                "context_sha256": key,
                "reference_eligible": eligible,
                "candidate_run_id": identity,
                "scope": "quality_only_not_performance",
            }
            if not path.exists():
                return {
                    **common,
                    "status": "missing",
                    "reason": "NO_COMPATIBLE_FLOATING_BASELINE",
                    "instruction": "Load unquantized floating weights with default KV and native kernels; run the identical quality protocol once.",
                }
            try:
                document = read_json(path)
                if (
                    document["context_sha256"] != key
                    or document["context"] != signature
                ):
                    raise ValueError("BASELINE_INDEX_CONTEXT_MISMATCH")
                selected = document["primary"]
                reference_run, reference, reference_context = self._read(
                    selected["run_id"]
                )
                if (
                    reference_context != signature
                    or selected["manifest_sha256"] != reference["manifest_sha256"]
                    or not is_floating_reference(reference)
                ):
                    raise ValueError("BASELINE_REGISTERED_EVIDENCE_CHANGED")
                quality = compare_scores(reference, candidate)
            except (ValueError, OSError, KeyError) as exc:
                return {
                    **common,
                    "status": "unavailable",
                    "reason": str(exc),
                    "baseline_run_id": document.get("primary", {}).get("run_id")
                    if "document" in locals()
                    else None,
                }
            q, p = reference["quality"], reference["receipt"]["precision_receipt"]
            return {
                **common,
                "status": "baseline" if identity == reference_run["id"] else "compared",
                "baseline": {
                    **selected,
                    "compute_dtype": p["compute_dtype"],
                    "model": reference_run["config"].get("deployment", {}).get("model"),
                    "created_at": reference_run["created_at"],
                    "perplexity": q.get("perplexity"),
                    "mean_nll": q["mean_nll"],
                    "scored_tokens": q.get("scored_tokens"),
                    "task_accuracy": q.get("task_accuracy"),
                },
                "quality": quality,
            }
