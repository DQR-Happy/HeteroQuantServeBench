"""Deployment-scoped file identity checks; CUDA compatibility stays in the worker.

Only administrator-owned paths enter this module. No torch import, device probe,
binary loading, compilation or user-provided library path belongs in an API read.
"""
import hashlib
import json
from pathlib import Path

from hqsb.core.kernel_profiles import catalogue


def native_inventory(config, *, source_root=None):
    if not config.kernel_manifest:
        return {"status": "missing", "reason": "KERNEL_NATIVE_MANIFEST_REQUIRED"}
    root = source_root or Path(__file__).resolve().parents[2]
    try:
        path = Path(config.kernel_manifest).expanduser().resolve()
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("KERNEL_MANIFEST_SIZE_LIMIT")
        raw = path.read_bytes()
        document = json.loads(raw)
        if document.get("schema_version") != "l2-native-v1" or document.get("abi") != 1:
            raise ValueError("KERNEL_ABI_MANIFEST_MISMATCH")
        library = Path(document["library"])
        if not library.is_absolute() or library.stat().st_size > 256 * 1024**2:
            raise ValueError("KERNEL_ARTIFACT_PATH_OR_SIZE")
        if hashlib.sha256(library.read_bytes()).hexdigest() != document["library_sha256"]:
            raise ValueError("KERNEL_ARTIFACT_HASH_MISMATCH")
        if not document["sources"]:
            raise ValueError("KERNEL_SOURCE_IDENTITY_MISSING")
        for name, expected in document["sources"].items():
            source = (root / name).resolve()
            if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError("KERNEL_SOURCE_MISMATCH")
        return {"status": "files_verified", "reason": None,
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "library_sha256": document["library_sha256"],
                "runtime_validation": "required_on_worker_load"}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        # Paths are administrator-owned and are deliberately not exposed to browsers.
        return {"status": "invalid", "reason": "KERNEL_NATIVE_ARTIFACT_INVALID"}


def deployment_catalogue(config):
    result = catalogue()
    native = native_inventory(config)
    result.update(deployment_id=config.id, native_artifact=native)
    for item in result["profiles"]:
        reason = None
        if config.provider != "pytorch":
            reason = "EXTERNAL_PROVIDER_MANAGES_KERNELS"
        elif item["requires_native_artifact"] and native["status"] != "files_verified":
            reason = native["reason"]
        item["availability"] = {"loadable": reason is None, "reason": reason,
                                "scope": "configuration_and_file_checks_not_runtime_certification"}
    return result
