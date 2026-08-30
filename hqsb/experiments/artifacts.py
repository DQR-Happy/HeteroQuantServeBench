"""Bounded experimental artifact publication and verified reads.

Worker files are pending until the supervising process acknowledges cleanup.
An immutable manifest binds receipt, quality rows and framework evidence.
"""

import json
import os
from pathlib import Path

from hqsb.observation.storage import atomic_json, sha256

MAX_BYTES = 256 * 1024 * 1024


def seal(directory, *, run_id, status="complete", error=None):
    directory = Path(directory)
    target = directory / "experiment-manifest.json"
    if target.exists():
        return read_manifest(directory)
    files = []
    for path in sorted(directory.iterdir()):
        if path.name in {"execution-receipt.json", "quality-result.json", "quality-pending.json", "trace.json",
                         "kernel-plan.json", "kernel-bindings.json", "kernel-summary.json"} or path.name.startswith("quality-logits-"):
            if path.is_symlink() or not path.is_file():
                raise ValueError("EXPERIMENT_ARTIFACT_PATH_INVALID")
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    if sum(item["bytes"] for item in files) > MAX_BYTES:
        status, error = "partial", "EXPERIMENT_ARTIFACT_BUDGET"
    if not any(item["path"] == "execution-receipt.json" for item in files):
        status, error = "partial", error or "EXECUTION_RECEIPT_MISSING"
    pending = directory / "quality-pending.json"
    if pending.exists() and read_json(pending).get("status") != "complete":
        status, error = "partial", error or "QUALITY_INCOMPLETE"
    manifest = {"schema_version": "1.0.0", "run_id": run_id, "status": status,
                "publication": "supervisor_acknowledged", "error": error, "artifacts": files}
    atomic_json(target, manifest)
    return manifest


def read_json(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("EXPERIMENT_JSON_INVALID")
    return json.loads(path.read_text())


def read_manifest(directory):
    directory = Path(directory)
    manifest = read_json(directory / "experiment-manifest.json")
    if manifest.get("schema_version") != "1.0.0" or manifest.get("run_id") != directory.name or manifest.get("publication") != "supervisor_acknowledged":
        raise ValueError("EXPERIMENT_MANIFEST_INVALID")
    if not isinstance(manifest.get("artifacts"), list) or len(manifest["artifacts"]) > 1024:
        raise ValueError("EXPERIMENT_MANIFEST_INVALID")
    total = 0
    names = set()
    for item in manifest["artifacts"]:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"} or not isinstance(item["path"], str) or type(item["bytes"]) is not int or not 0 <= item["bytes"] <= MAX_BYTES or not isinstance(item["sha256"], str):
            raise ValueError("EXPERIMENT_ARTIFACT_METADATA_INVALID")
        name = item["path"]
        if name != Path(name).name or name in names:
            raise ValueError("EXPERIMENT_ARTIFACT_PATH_INVALID")
        names.add(name)
        path = directory / name
        total += item["bytes"]
        if total > MAX_BYTES or path.is_symlink() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("EXPERIMENT_ARTIFACT_INTEGRITY")
    return manifest


def load_bundle(directory):
    directory = Path(directory)
    manifest = read_manifest(directory)
    names = {item["path"] for item in manifest["artifacts"]}
    bundle = {"manifest": manifest, "manifest_sha256": sha256(directory / "experiment-manifest.json"),
            "receipt": read_json(directory / "execution-receipt.json") if "execution-receipt.json" in names else None,
            "quality": read_json(directory / "quality-result.json") if "quality-result.json" in names else None}
    if bundle["receipt"] and bundle["receipt"].get("run_id") != manifest["run_id"]:
        raise ValueError("EXPERIMENT_RECEIPT_ID_MISMATCH")
    if bundle["quality"]:
        if bundle["quality"].get("run_id") != manifest["run_id"]:
            raise ValueError("QUALITY_RUN_ID_MISMATCH")
        for position in bundle["quality"]["positions"]:
            item = next((item for item in manifest["artifacts"] if item["path"] == position["path"]), None)
            if not item or any(item[key] != position[key] for key in ("bytes", "sha256")):
                raise ValueError("QUALITY_POSITION_INTEGRITY")
    return bundle
