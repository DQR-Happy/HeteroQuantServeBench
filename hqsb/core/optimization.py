"""Optimization plans and capabilities. No runtime imports or device probing.

The catalogue describes implementations, never infers hardware support from an
installed package. Executors own application and supply the effective receipt.
"""

import hashlib
import json

VERSION = "1.0.0"
LAYERS = {
    1: "hardware_system", 2: "kernel", 3: "compiler_runtime", 4: "quantization",
    5: "model_structure", 6: "memory_kv", 7: "decoding", 8: "scheduling",
    9: "distributed", 10: "service_application",
}
IMPLEMENTED = {
    "cleanup_policy": {"layers": [6], "scope": "request", "values": ["empty_cache", "preserve_allocator"]},
    "attention_backend": {"layers": [2, 3], "scope": "deployment", "values": ["eager", "sdpa"]},
    "output_budget": {"layers": [7, 10], "scope": "request", "minimum": 1, "maximum": 512},
}
PENDING = {
    "power_frequency": [1], "placement": [1, 9], "custom_kernel": [2],
    "compile_graph": [3], "weight_quantization": [4], "activation_quantization": [4],
    "kv_quantization": [4, 6], "model_compression": [5], "paged_kv": [6],
    "prefix_cache": [6, 10], "offload_prefetch": [6], "speculative_decode": [7],
    "continuous_batching": [8], "chunked_prefill": [8], "parallel_plan": [9],
    "cache_routing": [10], "autoscaling": [10],
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def catalogue(provider="pytorch"):
    items = []
    for name, definition in IMPLEMENTED.items():
        items.append({"id": name, **definition,
                      "implementation_status": "implemented_unverified",
                      "runtime_capability": "requires_executor_validation" if provider == "pytorch" else "unsupported_provider",
                      "proof_required": True})
    for name, layers in PENDING.items():
        items.append({"id": name, "layers": layers, "implementation_status": "not_integrated",
                      "runtime_capability": "unavailable", "proof_required": True})
    return {"schema_version": VERSION, "layers": LAYERS, "items": items}


def resolve_plan(raw, defaults, *, provider="pytorch", deployment_attention=None):
    raw = {} if raw is None else raw
    if not isinstance(raw, dict) or set(raw) - {"schema_version", "changes"}:
        raise ValueError("OPTIMIZATION_PLAN_INVALID")
    if raw.get("schema_version", VERSION) != VERSION:
        raise ValueError("OPTIMIZATION_VERSION_UNSUPPORTED")
    changes = raw.get("changes", {})
    if not isinstance(changes, dict) or len(changes) > len(IMPLEMENTED):
        raise ValueError("OPTIMIZATION_CHANGES_INVALID")
    if changes and provider != "pytorch":
        raise ValueError("OPTIMIZATION_PROVIDER_UNSUPPORTED")
    effective = dict(defaults)
    for name, value in changes.items():
        if name not in IMPLEMENTED:
            raise ValueError("OPTIMIZATION_NOT_INTEGRATED:" + str(name))
        definition = IMPLEMENTED[name]
        if "values" in definition and value not in definition["values"]:
            raise ValueError("OPTIMIZATION_VALUE_INVALID:" + name)
        if name == "output_budget" and (type(value) is not int or not 1 <= value <= 512):
            raise ValueError("OPTIMIZATION_VALUE_INVALID:output_budget")
        if name == "attention_backend" and deployment_attention is not None and value != deployment_attention:
            raise ValueError("OPTIMIZATION_REQUIRES_RELOAD:attention_backend")
        effective[name] = value
    return {"schema_version": VERSION, "requested": changes, "resolved": effective,
            "plan_sha256": fingerprint({"schema_version": VERSION, "changes": changes}),
            "fallback": "none", "status": "resolved_not_applied"}


def validate_quality_plan(raw):
    """Bound both device work and full-vocabulary CPU output; no silent truncation."""
    if not isinstance(raw, dict) or set(raw) - {"samples", "max_tokens", "generation_tokens", "max_bytes", "top_k"}:
        raise ValueError("QUALITY_PLAN_INVALID")
    samples = raw.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 4:
        raise ValueError("QUALITY_SAMPLE_LIMIT")
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) - {"id", "text", "expected", "slice"}:
            raise ValueError("QUALITY_SAMPLE_INVALID")
        if not isinstance(sample.get("text"), str) or not 1 <= len(sample["text"]) <= 4096:
            raise ValueError("QUALITY_TEXT_INVALID")
        for name in ("id", "expected", "slice"):
            if name in sample and (not isinstance(sample[name], str) or len(sample[name]) > 512):
                raise ValueError("QUALITY_SAMPLE_FIELD_INVALID:" + name)
    ids = [sample.get("id", str(i)) for i, sample in enumerate(samples)]
    if len(set(ids)) != len(ids):
        raise ValueError("QUALITY_SAMPLE_ID_DUPLICATE")
    result = {"samples": samples, "max_tokens": 16, "generation_tokens": 4,
              "max_bytes": 64 * 1024 * 1024, "top_k": 5, **raw}
    for field, low, high in (("max_tokens", 2, 64), ("generation_tokens", 1, 32),
                             ("max_bytes", 1024, 128 * 1024 * 1024), ("top_k", 1, 20)):
        if type(result[field]) is not int or not low <= result[field] <= high:
            raise ValueError("QUALITY_BOUND_INVALID:" + field)
    return result
