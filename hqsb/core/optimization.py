"""Optimization plans and capabilities. No runtime imports or device probing.

The catalogue describes implementations, never infers hardware support from an
installed package. Executors own application and supply the effective receipt.
"""

import hashlib
import json

from hqsb.core.kernel_profiles import PROFILES

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
    "kernel_profile": {"layers": [2], "scope": "deployment", "values": list(PROFILES)},
}
PENDING = {
    "application_policy": [10], "system_profile": [1], "power_frequency": [1], "placement": [1, 9], "custom_kernel": [2],
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
    if provider in {"vllm", "sglang"}:
        from hqsb.core.native_engine import catalogue as native_catalogue
        return {"schema_version": VERSION, "layers": LAYERS, **native_catalogue(provider)}
    items = []
    for name, definition in IMPLEMENTED.items():
        items.append({"id": name, **definition,
                      "implementation_status": "implemented_unverified",
                      "runtime_capability": "requires_executor_validation" if provider == "pytorch" else "unsupported_provider",
                      "proof_required": True})
    for name, layers in PENDING.items():
        session = name in {"continuous_batching", "chunked_prefill"}
        precision = name in {"weight_quantization", "activation_quantization", "kv_quantization"}
        compiler = name == "compile_graph"
        decoding = name == "speculative_decode"
        parallel = name == "parallel_plan"
        system = name == "system_profile"
        application = name == "application_policy"
        items.append({"id": name, "layers": layers,
                      "implementation_status": "session_api_only" if parallel or system or application else "request_api_only" if compiler or decoding else "session_api_only" if session else "load_api_only" if precision else "not_integrated",
                      "runtime_capability": "requires_system_hardware_and_permission_validation" if system and provider == "pytorch" else "requires_parallel_hardware_and_model_validation" if parallel and provider == "pytorch" else "requires_decoding_validation" if decoding and provider == "pytorch" else "native_float_runtime_validation_required" if compiler and provider == "pytorch" else "requires_scheduling_capabilities" if session and provider == "pytorch" else "requires_precision_capabilities" if precision and provider == "pytorch" else "unavailable",
                      **({"api":"/system/jobs","catalogue":"/system/catalog","entrypoint":"/system","scope":"exclusive_paired_system_session"} if system else {}),
                      **({"api":"/application/jobs","catalogue":"/application/catalog","entrypoint":"/application","scope":"exclusive_service_workload_session","runtime_capability":"requires_ready_targets_or_managed_replica_admission" if provider == "pytorch" else "unsupported_provider"} if application else {}),
                      **({"api":"/parallel/jobs","catalogue":"/parallel/catalog","entrypoint":"/parallel","scope":"exclusive_parallel_session"} if parallel else {}),
                      **({"api":"/requests","catalogue":"/decoding/catalog","entrypoint":"/decoding","scope":"request_decoding_plan"} if decoding else {}),
                      **({"api": "/requests", "catalogue": "/runtime/catalog", "entrypoint": "/model-architecture", "scope": "request_runtime_plan"} if compiler else {}),
                      **({"api": "/deployments/{id}/load", "catalogue": "/precision/catalog", "scope": "deployment_reload"} if precision else {}),
                      **({"entrypoint": "/scheduling", "api": "/scheduling-benchmarks",
                          "scope": "exclusive_multi_request_session"} if session else {}),
                      "proof_required": True})
    return {"schema_version": VERSION, "layers": LAYERS, "items": items}


def resolve_plan(raw, defaults, *, provider="pytorch", deployment_attention=None, deployment_kernel_profile=None):
    raw = {} if raw is None else raw
    if not isinstance(raw, dict) or set(raw) - {"schema_version", "changes"}:
        raise ValueError("OPTIMIZATION_PLAN_INVALID")
    if raw.get("schema_version", VERSION) != VERSION:
        raise ValueError("OPTIMIZATION_VERSION_UNSUPPORTED")
    changes = raw.get("changes", {})
    if not isinstance(changes, dict) or len(changes) > len(IMPLEMENTED):
        raise ValueError("OPTIMIZATION_CHANGES_INVALID")
    if changes and provider in {"vllm", "sglang"}:
        if set(changes) - {"output_budget"}:
            raise ValueError("NATIVE_ENGINE_OPTIONS_REQUIRE_DEPLOYMENT_RELOAD")
    elif changes and provider != "pytorch":
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
        if name == "kernel_profile" and deployment_kernel_profile is not None and value != deployment_kernel_profile:
            raise ValueError("OPTIMIZATION_REQUIRES_RELOAD:kernel_profile")
        effective[name] = value
    return {"schema_version": VERSION, "requested": changes, "resolved": effective,
            "plan_sha256": fingerprint({"schema_version": VERSION, "changes": changes}),
            "fallback": "none", "status": "resolved_not_applied"}


def validate_quality_plan(raw):
    """Bound both device work and full-vocabulary CPU output; no silent truncation."""
    if isinstance(raw, dict) and "benchmark_id" in raw:
        from hqsb.core.corpus_benchmark import resolve_plan

        return resolve_plan(raw)
    if not isinstance(raw, dict) or set(raw) - {"samples", "max_tokens", "generation_tokens", "max_bytes", "top_k", "forward_mode"}:
        raise ValueError("QUALITY_PLAN_INVALID")
    if raw.get("forward_mode", "incremental_decode") not in ("incremental_decode", "full_prefill"):
        raise ValueError("QUALITY_FORWARD_MODE_INVALID")
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
