"""One auxiliary model owned by the resident serial worker, independent request KV."""

import gc
import json
import math
import os
from pathlib import Path
import time
from hqsb.core.optimization import fingerprint
from hqsb.observation.storage import sha256


def planned_parameter_bytes(root, element_size):
    """Inspect headers only; budget the requested dtype rather than file dtype."""
    from safetensors import safe_open

    source_config = json.loads((root / "config.json").read_text())
    if source_config.get("quantization_config"):
        raise ValueError("L7_DRAFT_REQUIRES_UNQUANTIZED_CHECKPOINT")
    files = list(root.glob("*.safetensors"))
    if not files:
        raise ValueError("L7_DRAFT_REQUIRES_SAFETENSORS_CHECKPOINT")
    elements = 0
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                elements += math.prod(handle.get_slice(name).get_shape())
    # Duplicate tied tensors in the checkpoint conservatively overestimate.
    return elements * element_size


def tokenizer_hash(tokenizer):
    value = json.loads(tokenizer.backend_tokenizer.to_str())
    value.pop("padding", None)
    value.pop("truncation", None)
    return fingerprint(value)


def summary(provider):
    resource = getattr(provider, "decoding_draft", None)
    return (
        resource["receipt"]
        if resource
        else {"status": "not_loaded", "deployment_id": None, "parameter_bytes": 0}
    )


def release(provider):
    import torch

    before = summary(provider)
    if getattr(provider, "decoding_draft", None):
        torch.cuda.synchronize()
        provider.decoding_draft = None
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "released": before,
        "current": summary(provider),
        "memory": provider.memory(),
    }


def acquire(provider, config, total, check):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from hqsb.models.manifest import verify_or_raise
    from hqsb.models.model_admission import effective_host_available
    from hqsb.backends.observation import host_memory
    from hqsb.backends.decoding_engine import cache_for

    root = Path(os.path.expandvars(config["model_path"])).expanduser()
    manifest = Path(os.path.expandvars(config["manifest"])).expanduser()
    key = fingerprint(
        [
            str(root),
            sha256(manifest),
            provider.config["precision"],
            tokenizer_hash(provider.tokenizer),
        ]
    )
    old = getattr(provider, "decoding_draft", None)
    if old:
        if old["key"] != key:
            raise ValueError("L7_UNLOAD_RESIDENT_DRAFT_BEFORE_SELECTING_ANOTHER")
        if total > old["context_limit"]:
            raise ValueError("L7_DRAFT_CONTEXT_LIMIT")
        return old["model"], False
    if total > config["context_limit"]:
        raise ValueError("L7_DRAFT_DEPLOYMENT_CONTEXT_LIMIT")
    check()
    tok = AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    if tokenizer_hash(tok) != tokenizer_hash(provider.tokenizer):
        raise ValueError(
            "L7_DRAFT_TOKENIZER_MISMATCH_USE_IDENTICAL_TOKEN_ID_MAPPING_AND_ENCODING"
        )
    verify_or_raise(
        str(root),
        str(manifest),
        strict_extra=True,
        allow_extra=tuple(config.get("manifest_allow_extra", [])),
    )
    weights = sum(p.stat().st_size for p in root.glob("*.safetensors"))
    dtype = next(provider.model.parameters()).dtype
    planned_weights = planned_parameter_bytes(root, dtype.itemsize)
    free, _ = torch.cuda.mem_get_info()
    available = effective_host_available(
        host_memory().get("host_available_bytes"), Path("/sys/fs/cgroup")
    )
    if (
        weights > 6 * 1024**3
        or free < planned_weights + 2 * 1024**3
        or available is None
        or available < planned_weights + max(weights, planned_weights) + 1024**3
    ):
        raise ValueError("L7_DRAFT_WEIGHT_RAM_OR_GPU_BUDGET_EXCEEDED")
    model = None
    started = time.monotonic()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            device_map="cpu",
            attn_implementation=config["attention"],
        ).eval()
        cache_for(model, total)
        if model.config.vocab_size != provider.model.config.vocab_size:
            raise ValueError("L7_DRAFT_VOCABULARY_SIZE_MISMATCH")
        check()
        model.to("cuda:0")
        torch.cuda.synchronize()
        check()
        receipt = {
            "status": "loaded",
            "deployment_id": config["id"],
            "model": config["model"],
            "model_manifest_sha256": sha256(manifest),
            "tokenizer_sha256": tokenizer_hash(tok),
            "parameter_bytes": sum(
                p.numel() * p.element_size() for p in model.parameters()
            ),
            "planned_parameter_bytes": planned_weights,
            "load_seconds": time.monotonic() - started,
            "precision": provider.config["precision"],
            "same_checkpoint_as_target": sha256(manifest)
            == provider.source_identity["hash"],
            "ownership": "resident_target_worker_until_explicit_draft_or_target_unload",
        }
        provider.decoding_draft = {
            "key": key,
            "model": model,
            "receipt": receipt,
            "context_limit": config["context_limit"],
        }
        return model, True
    except BaseException:
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        raise
