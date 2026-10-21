"""Deployment-scoped file identity checks; CUDA compatibility stays in the worker.

Only administrator-owned paths enter this module. No torch import, device probe,
binary loading, compilation or user-provided library path belongs in an API read.
"""
import hashlib
import json
from pathlib import Path

from hqsb.core.kernel_profiles import catalogue


def model_kernel_compatibility(config):
    """Configuration/file check only; the worker still validates source/shape."""
    if config.precision != "float16":
        return "KERNEL_PRECISION_UNSUPPORTED"
    if config.model != "Qwen/Qwen3-1.7B":
        return "KERNEL_MODEL_SHAPE_UNSUPPORTED"
    import os

    path = Path(os.path.expandvars(config.model_path)).expanduser() / "config.json"
    if path.exists():
        try:
            if path.stat().st_size > 1024**2:
                return "KERNEL_MODEL_CONFIG_INVALID"
            model = json.loads(path.read_text())
            fields = ("hidden_size", "intermediate_size", "num_attention_heads",
                      "num_key_value_heads", "num_hidden_layers")
            if model.get("model_type") != "qwen3" or tuple(model.get(k) for k in fields) != (2048, 6144, 16, 8, 28):
                return "KERNEL_MODEL_SHAPE_UNSUPPORTED"
        except (OSError, ValueError, AttributeError):
            return "KERNEL_MODEL_CONFIG_INVALID"
    return None


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
    if config.provider in {"vllm", "sglang"}:
        from hqsb.core.native_engine import catalogue as native_catalogue
        native = native_catalogue(config.provider)
        return {"deployment_id": config.id, "profiles": [], "native_engine": config.provider,
                "native_controls": [row for row in native["items"] if 2 in row["layers"]],
                "entrypoint": "/engines?deployment=" + config.id + "&layer=2",
                "native_artifact": {"status": "upstream_environment", "reason": None}}
    result = catalogue()
    native = native_inventory(config)
    result.update(deployment_id=config.id, native_artifact=native)
    for item in result["profiles"]:
        reason = None
        if config.provider != "pytorch":
            reason = "EXTERNAL_PROVIDER_MANAGES_KERNELS"
        elif item["operators"] and model_kernel_compatibility(config):
            reason = model_kernel_compatibility(config)
        elif item["requires_native_artifact"] and native["status"] != "files_verified":
            reason = native["reason"]
        item["availability"] = {"loadable": reason is None, "reason": reason,
                                "scope": "configuration_and_file_checks_not_runtime_certification"}
    return result
