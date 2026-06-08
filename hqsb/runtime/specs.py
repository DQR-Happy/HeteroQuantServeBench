"""Strict loading of the frozen S07 runtime configuration (``configs/runtime/``).

Every S07 configuration is a *frozen artifact*: the YAML declares the scheduler
budgets, KV block sizes, prefix policy, graph buckets, speculative algorithm,
failure policy and comparison tiers, and the loader refuses unknown keys rather
than ignoring them.  Two audit functions make the contract non-decorative:

* :func:`build_object` turns a document into the runtime object the experiment
  will use (so a YAML that cannot be executed is rejected at load time, not at
  report time);
* :func:`contract_audit` compares the YAML with the object field by field, so
  YAML and code cannot drift apart silently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Tuple

import yaml

from hqsb.core.errors import ConfigError
from hqsb.runtime import comparison as comparison_mod
from hqsb.runtime import graph_route, prefix_cache, spec_decode
from hqsb.runtime.scheduler import SchedulerSpec

KIND_REQUEST_SPEC = "hqsb.runtime.request_spec"
KIND_KV_SPEC = "hqsb.runtime.kv_spec"
KIND_SCHEDULER_SPEC = "hqsb.runtime.scheduler_spec"
KIND_PREFIX_SPEC = "hqsb.runtime.prefix_spec"
KIND_GRAPH_ATTENTION_SPEC = "hqsb.runtime.graph_attention_spec"
KIND_SPEC_DECODE_SPEC = "hqsb.runtime.spec_decode_spec"
KIND_FAILURE_SPEC = "hqsb.runtime.failure_spec"
KIND_COMPARISON_SPEC = "hqsb.runtime.comparison_spec"

KINDS: Tuple[str, ...] = (
    KIND_REQUEST_SPEC,
    KIND_KV_SPEC,
    KIND_SCHEDULER_SPEC,
    KIND_PREFIX_SPEC,
    KIND_GRAPH_ATTENTION_SPEC,
    KIND_SPEC_DECODE_SPEC,
    KIND_FAILURE_SPEC,
    KIND_COMPARISON_SPEC,
)

#: Keys allowed on every document, independent of its kind.
COMMON_KEYS: Tuple[str, ...] = ("kind", "description", "name", "notes")

ALLOWED_KEYS: Mapping[str, Tuple[str, ...]] = {
    KIND_REQUEST_SPEC: (
        "sampling_modes",
        "capability_states",
        "claimable_states",
        "stop_string_layers",
        "comparison_unit",
        "frozen_fields",
    ),
    KIND_KV_SPEC: (
        "block_sizes",
        "kv_dtype",
        "query_heads",
        "kv_heads",
        "head_dim",
        "num_layers",
        "element_bytes",
        "fragment_classes",
        "oom_kinds",
        "oom_action_order",
        "max_oom_attempts",
        "capacity_watermark",
        "eviction_policies",
    ),
    KIND_SCHEDULER_SPEC: (
        "mode",
        "max_batched_tokens",
        "max_sequences",
        "block_size",
        "kv_blocks",
        "chunk_size",
        "decode_priority",
        "long_prefill_threshold",
        "preemption_policy",
        "admission_reserve_full_isl",
        "watermark",
        "prefix_cache",
        "traces",
        "congestion_waiting_growth",
        "fairness_metric",
    ),
    KIND_PREFIX_SPEC: (
        "block_size",
        "collision_policy",
        "eviction_policy",
        "max_cache_bytes",
        "allow_cancelled_retain",
        "verify_token_equality",
        "cache_layout_version",
        "key_fields",
        "security_domains",
    ),
    KIND_GRAPH_ATTENTION_SPEC: (
        "buckets",
        "out_of_bucket_policy",
        "allow_recapture",
        "max_graphs",
        "capture_stream",
        "includes_input_copy",
        "includes_output_copy",
        "claim_cuda_graph",
        "attention_candidates",
        "factorial",
    ),
    KIND_SPEC_DECODE_SPEC: (
        "algorithm",
        "gamma_candidates",
        "mtp_heads",
        "mtp_quality_contract",
        "draft_vocab_requires_mapping",
        "claim",
    ),
    KIND_FAILURE_SPEC: (
        "cases",
        "common_invariants",
        "oom_action_order",
        "run_separation",
        "longrun_tolerance",
        "max_cleanup_ms",
        "allowed_extra_tokens",
    ),
    KIND_COMPARISON_SPEC: (
        "tiers",
        "tier_claims",
        "common_denominator_features",
        "best_valid_preregistered",
        "cold_warm_phases",
        "reported_only_phases",
        "statistics",
        "independent_processes",
        "pareto_objectives",
        "token_denominators",
        "request_classes",
    ),
}


def load_yaml_document(path: str) -> Dict[str, Any]:
    """Read one YAML document and enforce the strict-key policy."""
    if not os.path.isfile(path):
        raise ConfigError(f"config file not found: {path}", details={"path": path})
    try:
        with open(path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:  # pragma: no cover - malformed file
        raise ConfigError(f"{path}: invalid YAML: {exc}", details={"path": path}) from exc
    if not isinstance(payload, dict):
        raise ConfigError(
            f"{path}: a runtime config document must be a mapping",
            details={"path": path},
        )
    kind = payload.get("kind")
    if kind not in KINDS:
        raise ConfigError(
            f"{path}: unknown or missing kind {kind!r}",
            details={"path": path, "allowed": list(KINDS)},
        )
    allowed = set(COMMON_KEYS) | set(ALLOWED_KEYS[kind])
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ConfigError(
            f"{path}: unknown keys {unknown}; unknown fields are refused rather than "
            "ignored so a typo cannot silently disable a policy",
            details={"path": path, "fields": unknown},
        )
    return payload


# ── document wrappers ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class RuntimeDocument:
    """A loaded, validated S07 document."""

    kind: str
    path: str
    payload: Mapping[str, Any]

    @property
    def name(self) -> str:
        return str(self.payload.get("name", os.path.basename(self.path)))

    def require(self, key: str) -> Any:
        if key not in self.payload:
            raise ConfigError(
                f"{self.path}: required key {key!r} is missing",
                details={"path": self.path, "field": key},
            )
        return self.payload[key]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "name": self.name,
            "payload": dict(self.payload),
        }


def load_any(path: str) -> RuntimeDocument:
    payload = load_yaml_document(path)
    return RuntimeDocument(kind=str(payload["kind"]), path=path, payload=payload)


def default_config_dir(root: str) -> str:
    return os.path.join(root, "configs", "runtime")


def load_directory(directory: str) -> List[RuntimeDocument]:
    """Load every ``*.yaml`` of ``configs/runtime`` (deterministic order)."""
    if not os.path.isdir(directory):
        raise ConfigError(
            f"runtime config directory not found: {directory}",
            details={"path": directory},
        )
    documents: List[RuntimeDocument] = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith((".yaml", ".yml")):
            documents.append(load_any(os.path.join(directory, filename)))
    if not documents:
        raise ConfigError(
            f"{directory} contains no runtime config documents",
            details={"path": directory},
        )
    return documents


# ── builders: a YAML that cannot execute is rejected at load time ──────────


def build_scheduler_spec(document: RuntimeDocument) -> SchedulerSpec:
    if document.kind != KIND_SCHEDULER_SPEC:
        raise ConfigError(f"{document.path}: not a scheduler spec")
    return SchedulerSpec(
        mode=str(document.require("mode")),
        max_batched_tokens=int(document.require("max_batched_tokens")),
        max_sequences=int(document.require("max_sequences")),
        block_size=int(document.require("block_size")),
        kv_blocks=int(document.require("kv_blocks")),
        chunk_size=int(document.payload.get("chunk_size", 0)),
        decode_priority=bool(document.payload.get("decode_priority", False)),
        long_prefill_threshold=int(document.payload.get("long_prefill_threshold", 0)),
        preemption_policy=str(document.payload.get("preemption_policy", "none")),
        admission_reserve_full_isl=bool(
            document.payload.get("admission_reserve_full_isl", True)
        ),
        watermark=float(document.payload.get("watermark", 0.9)),
        prefix_cache=bool(document.payload.get("prefix_cache", False)),
    )


def build_prefix_spec(document: RuntimeDocument) -> prefix_cache.PrefixCacheSpec:
    if document.kind != KIND_PREFIX_SPEC:
        raise ConfigError(f"{document.path}: not a prefix spec")
    return prefix_cache.PrefixCacheSpec(
        block_size=int(document.require("block_size")),
        collision_policy=str(document.payload.get("collision_policy", "strong_digest")),
        eviction_policy=str(document.payload.get("eviction_policy", "lru")),
        max_cache_bytes=float(document.require("max_cache_bytes")),
        allow_cancelled_retain=bool(
            document.payload.get("allow_cancelled_retain", False)
        ),
        verify_token_equality=bool(
            document.payload.get("verify_token_equality", True)
        ),
        cache_layout_version=str(
            document.payload.get("cache_layout_version", "1.0.0")
        ),
    )


def build_graph_spec(document: RuntimeDocument) -> graph_route.GraphSpec:
    if document.kind != KIND_GRAPH_ATTENTION_SPEC:
        raise ConfigError(f"{document.path}: not a graph/attention spec")
    buckets = tuple(
        graph_route.GraphBucketSpec(
            name=str(item["name"]),
            max_sequences=int(item["max_sequences"]),
            max_tokens=int(item["max_tokens"]),
            context_bucket=int(item.get("context_bucket", 0)),
        )
        for item in document.require("buckets")
    )
    return graph_route.GraphSpec(
        buckets=buckets,
        out_of_bucket_policy=str(
            document.payload.get("out_of_bucket_policy", "fallback_non_graph")
        ),
        allow_recapture=bool(document.payload.get("allow_recapture", False)),
        max_graphs=int(document.payload.get("max_graphs", 4)),
        capture_stream=str(
            document.payload.get("capture_stream", "capture_origin_stream")
        ),
        includes_input_copy=bool(document.payload.get("includes_input_copy", True)),
        includes_output_copy=bool(document.payload.get("includes_output_copy", True)),
        claim_cuda_graph=bool(document.payload.get("claim_cuda_graph", False)),
    )


def build_attention_candidates(
    document: RuntimeDocument,
) -> Tuple[graph_route.AttentionCandidate, ...]:
    if document.kind != KIND_GRAPH_ATTENTION_SPEC:
        raise ConfigError(f"{document.path}: not a graph/attention spec")
    candidates = []
    for item in document.require("attention_candidates"):
        candidates.append(
            graph_route.AttentionCandidate(
                name=str(item["name"]),
                provider=str(item["provider"]),
                version=str(item["version"]),
                dtypes=tuple(item["dtypes"]),
                kv_dtypes=tuple(item["kv_dtypes"]),
                head_dim=int(item["head_dim"]),
                supports_gqa=bool(item["supports_gqa"]),
                mask=str(item["mask"]),
                phases=tuple(item["phases"]),
                paged_layout=bool(item["paged_layout"]),
                max_context=int(item["max_context"]),
                alignment=int(item.get("alignment", 1)),
                graph_capturable=bool(item.get("graph_capturable", False)),
                workspace_bytes=float(item.get("workspace_bytes", 0.0)),
                fallback_target=str(item.get("fallback_target", "runtime_default")),
            )
        )
    return tuple(candidates)


def build_spec_decode_contract(
    document: RuntimeDocument,
) -> Dict[str, Any]:
    if document.kind != KIND_SPEC_DECODE_SPEC:
        raise ConfigError(f"{document.path}: not a speculative-decode spec")
    algorithm = str(document.require("algorithm"))
    if algorithm not in spec_decode.SPEC_ALGORITHMS:
        raise ConfigError(
            f"{document.path}: unknown speculative algorithm {algorithm!r}",
            details={"allowed": list(spec_decode.SPEC_ALGORITHMS)},
        )
    gamma = document.require("gamma_candidates")
    sweep = spec_decode.gamma_sweep([int(value) for value in gamma])
    mtp = None
    if algorithm == spec_decode.MTP:
        mtp = spec_decode.MtpContract(
            heads=int(document.require("mtp_heads")),
            quality_contract=str(document.require("mtp_quality_contract")),
        )
        spec_decode.assert_mtp_does_not_borrow(mtp)
    return {
        "algorithm": algorithm,
        "gamma": sweep,
        "mtp": mtp.as_dict() if mtp else None,
        "claim": bool(document.payload.get("claim", False)),
    }


def build_comparison_method(document: RuntimeDocument) -> Dict[str, Any]:
    """The frozen *method* of the comparison (identity is supplied by the run).

    The identity (model manifest hash, hardware, request-trace hash) belongs to
    the run, not to a checked-in file: freezing a placeholder digest in a config
    would be a fake identity.  This function therefore returns the method
    configuration, and :func:`make_comparison_spec` binds it to a real run.
    """
    if document.kind != KIND_COMPARISON_SPEC:
        raise ConfigError(f"{document.path}: not a comparison spec")
    tiers = tuple(document.require("tiers"))
    if sorted(tiers) != ["A", "B", "C", "D"]:
        raise ConfigError(
            f"{document.path}: the tier ladder must declare A, B, C and D",
            details={"field": "tiers"},
        )
    return {
        "tiers": tiers,
        "tier_claims": dict(document.require("tier_claims")),
        "common_denominator_features": tuple(
            document.require("common_denominator_features")
        ),
        "best_valid_preregistered": bool(
            document.payload.get("best_valid_preregistered", True)
        ),
        "cold_warm_phases": tuple(document.require("cold_warm_phases")),
        "reported_only_phases": tuple(document.require("reported_only_phases")),
        "statistics": str(document.require("statistics")),
        "independent_processes": int(document.require("independent_processes")),
        "pareto_objectives": tuple(document.require("pareto_objectives")),
        "token_denominators": tuple(document.require("token_denominators")),
        "request_classes": tuple(document.require("request_classes")),
    }


def make_comparison_spec(
    document: RuntimeDocument,
    *,
    model_id: str = "",
    model_manifest_sha256: str = "",
    precision: str = "",
    hardware: str = "",
    request_trace_hash: str = "",
    memory_budget_bytes: float = 0.0,
    warmup_requests: int = 1,
) -> comparison_mod.ComparisonSpec:
    """Bind the frozen comparison method to a concrete, identified run.

    The identity is required: a comparison whose model/hardware/trace is unknown
    cannot be reproduced, and a placeholder digest in a checked-in file would be a
    fake identity.
    """
    identity = {
        "model_id": model_id,
        "model_manifest_sha256": model_manifest_sha256,
        "precision": precision,
        "hardware": hardware,
        "request_trace_hash": request_trace_hash,
    }
    missing = [name for name, value in identity.items() if not value]
    if missing:
        raise ConfigError(
            "the comparison identity must be bound at run time; missing "
            + ", ".join(missing),
            details={"fields": missing},
        )
    method = build_comparison_method(document)
    return comparison_mod.ComparisonSpec(
        model_id=model_id,
        model_manifest_sha256=model_manifest_sha256,
        precision=precision,
        hardware=hardware,
        request_trace_hash=request_trace_hash,
        statistics=method["statistics"],
        independent_processes=method["independent_processes"],
        memory_budget_bytes=memory_budget_bytes,
        warmup_requests=warmup_requests,
    )


BUILDERS: Mapping[str, Callable[[RuntimeDocument], Any]] = {
    KIND_SCHEDULER_SPEC: build_scheduler_spec,
    KIND_PREFIX_SPEC: build_prefix_spec,
    KIND_GRAPH_ATTENTION_SPEC: build_graph_spec,
    KIND_SPEC_DECODE_SPEC: build_spec_decode_contract,
    KIND_COMPARISON_SPEC: build_comparison_method,
}


def build_object(document: RuntimeDocument) -> Any:
    """Build the runtime object a document describes (or return the payload)."""
    builder = BUILDERS.get(document.kind)
    if builder is None:
        return dict(document.payload)
    return builder(document)


# ── drift audit ────────────────────────────────────────────────────────────


def contract_audit(document: RuntimeDocument) -> Dict[str, Any]:
    """Compare the YAML with the runtime object field by field."""
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    if document.kind == KIND_SCHEDULER_SPEC:
        spec = build_scheduler_spec(document)
        comparisons = {
            "mode": spec.mode,
            "max_batched_tokens": spec.max_batched_tokens,
            "max_sequences": spec.max_sequences,
            "block_size": spec.block_size,
            "kv_blocks": spec.kv_blocks,
            "chunk_size": spec.chunk_size,
            "preemption_policy": spec.preemption_policy,
            "watermark": spec.watermark,
        }
        for key, actual in comparisons.items():
            if key not in document.payload:
                problems.append(
                    f"{key} is used by the runtime object but not declared in the YAML"
                )
                continue
            declared = document.payload[key]
            rows.append({"field": key, "declared": declared, "actual": actual})
            if declared != actual:
                problems.append(f"{key}: declared={declared!r} actual={actual!r}")
        declared_traces = tuple(document.payload.get("traces", ()))
        rows.append(
            {"field": "traces", "declared": list(declared_traces), "actual": "code"}
        )
        if not declared_traces:
            problems.append("the scheduler spec declares no request trace")
    elif document.kind == KIND_PREFIX_SPEC:
        spec = build_prefix_spec(document)
        for key, actual in (
            ("block_size", spec.block_size),
            ("collision_policy", spec.collision_policy),
            ("eviction_policy", spec.eviction_policy),
            ("cache_layout_version", spec.cache_layout_version),
        ):
            declared = document.payload.get(key)
            rows.append({"field": key, "declared": declared, "actual": actual})
            if declared != actual:
                problems.append(f"{key}: declared={declared!r} actual={actual!r}")
        declared_fields = tuple(document.payload.get("key_fields", ()))
        missing = [
            name for name in prefix_cache.IDENTITY_FIELDS if name not in declared_fields
        ]
        rows.append(
            {
                "field": "key_fields",
                "declared": len(declared_fields),
                "actual": len(prefix_cache.IDENTITY_FIELDS),
            }
        )
        if missing:
            problems.append(
                "cache key fields declared in YAML do not cover the code key: "
                + ", ".join(missing)
            )
    elif document.kind == KIND_GRAPH_ATTENTION_SPEC:
        spec = build_graph_spec(document)
        declared_names = [item["name"] for item in document.payload.get("buckets", [])]
        actual_names = [bucket.name for bucket in spec.buckets]
        rows.append(
            {"field": "buckets", "declared": declared_names, "actual": actual_names}
        )
        if declared_names != actual_names:
            problems.append("bucket list differs between YAML and the built spec")
        candidates = build_attention_candidates(document)
        rows.append(
            {
                "field": "attention_candidates",
                "declared": len(document.payload.get("attention_candidates", [])),
                "actual": len(candidates),
            }
        )
    elif document.kind == KIND_SPEC_DECODE_SPEC:
        contract = build_spec_decode_contract(document)
        rows.append(
            {
                "field": "algorithm",
                "declared": document.payload.get("algorithm"),
                "actual": contract["algorithm"],
            }
        )
        if contract["algorithm"] != document.payload.get("algorithm"):
            problems.append("speculative algorithm differs after validation")
    elif document.kind == KIND_COMPARISON_SPEC:
        method = build_comparison_method(document)
        declared_features = set(method["common_denominator_features"])
        unknown_features = declared_features - set(
            comparison_mod.COMMON_DENOMINATOR_FEATURES
        )
        rows.append(
            {
                "field": "common_denominator_features",
                "declared": sorted(declared_features),
                "actual": sorted(comparison_mod.COMMON_DENOMINATOR_FEATURES),
            }
        )
        if unknown_features:
            problems.append(
                "unknown common-denominator features: " + ", ".join(sorted(unknown_features))
            )
        declared_phases = set(method["cold_warm_phases"])
        if declared_phases != set(comparison_mod.COLD_WARM_PHASES):
            problems.append(
                "cold/warm phases differ from hqsb.runtime.comparison.COLD_WARM_PHASES"
            )
        if set(method["reported_only_phases"]) != set(
            comparison_mod.REPORTED_ONLY_PHASES
        ):
            problems.append("reported-only phases differ from the code definition")
        unknown_objectives = set(method["pareto_objectives"]) - set(
            comparison_mod.PARETO_DIRECTIONS
        )
        if unknown_objectives:
            problems.append(
                "unknown Pareto objectives: " + ", ".join(sorted(unknown_objectives))
            )
        if set(method["request_classes"]) != set(comparison_mod.request_classes()):
            problems.append("request classes differ from the code definition")
        if method["independent_processes"] < 3:
            problems.append("fewer than three independent processes configured")
        for tier, claim in method["tier_claims"].items():
            if tier not in comparison_mod.TIER_CLAIMS:
                problems.append(f"unknown tier {tier!r} in tier_claims")
            elif claim != comparison_mod.TIER_CLAIMS[tier]:
                problems.append(
                    f"tier {tier} claim text drifts from the code definition"
                )
    elif document.kind == KIND_KV_SPEC:
        for key in ("fragment_classes", "oom_kinds", "oom_action_order"):
            declared = tuple(document.payload.get(key, ()))
            rows.append({"field": key, "declared": list(declared), "actual": list(declared)})
    else:
        rows.append({"field": "__kind__", "declared": document.kind, "actual": document.kind})
    return {
        "ok": not problems,
        "kind": document.kind,
        "path": document.path,
        "problems": problems,
        "rows": rows,
    }


def audit_directory(directory: str) -> Dict[str, Any]:
    """Audit every document in the runtime config directory."""
    documents = load_directory(directory)
    audits = [contract_audit(document) for document in documents]
    return {
        "ok": all(audit["ok"] for audit in audits),
        "documents": len(documents),
        "kinds": sorted({document.kind for document in documents}),
        "audits": audits,
    }


def audit_kv_spec(document: RuntimeDocument) -> Dict[str, Any]:
    """KV spec audit against the code's fragment/OOM vocabulary."""
    if document.kind != KIND_KV_SPEC:
        raise ConfigError(f"{document.path}: not a KV spec")
    from hqsb.runtime import kv as kv_mod

    problems: List[str] = []
    declared_classes = tuple(document.payload.get("fragment_classes", ()))
    if declared_classes and set(declared_classes) != set(kv_mod.FRAGMENT_CLASSES):
        problems.append(
            "fragment_classes differ from hqsb.runtime.kv.FRAGMENT_CLASSES: "
            f"declared={sorted(declared_classes)} actual={sorted(kv_mod.FRAGMENT_CLASSES)}"
        )
    declared_kinds = tuple(document.payload.get("oom_kinds", ()))
    if declared_kinds and set(declared_kinds) != set(kv_mod.OOM_KINDS):
        problems.append("oom_kinds differ from the code vocabulary")
    declared_order = tuple(document.payload.get("oom_action_order", ()))
    if declared_order and declared_order != kv_mod.OOM_ACTION_ORDER:
        problems.append(
            f"oom_action_order {declared_order} != code order {kv_mod.OOM_ACTION_ORDER}"
        )
    declared_attempts = int(document.payload.get("max_oom_attempts", 0))
    if declared_attempts and declared_attempts != kv_mod.MAX_OOM_ATTEMPTS:
        problems.append("max_oom_attempts differs from the code bound")
    return {"ok": not problems, "problems": problems, "path": document.path}


@dataclass
class RuntimeSpecs:
    """All frozen S07 specs, loaded and audited in one object."""

    documents: Dict[str, RuntimeDocument] = field(default_factory=dict)
    audits: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, directory: str) -> "RuntimeSpecs":
        documents: Dict[str, RuntimeDocument] = {}
        for document in load_directory(directory):
            if document.kind in documents:
                raise ConfigError(
                    f"two documents declare the same kind {document.kind!r}: "
                    f"{documents[document.kind].path} and {document.path}; the "
                    "experiment would silently use whichever loaded last",
                    details={"field": "kind"},
                )
            documents[document.kind] = document
        missing = [kind for kind in KINDS if kind not in documents]
        if missing:
            raise ConfigError(
                f"the runtime config directory is incomplete: missing {missing}",
                details={"fields": missing},
            )
        audits = [contract_audit(document) for document in documents.values()]
        audits.append(audit_kv_spec(documents[KIND_KV_SPEC]))
        return cls(documents=documents, audits=audits)

    @property
    def ok(self) -> bool:
        return all(audit["ok"] for audit in self.audits)

    def scheduler_spec(self) -> SchedulerSpec:
        return build_scheduler_spec(self.documents[KIND_SCHEDULER_SPEC])

    def prefix_spec(self) -> prefix_cache.PrefixCacheSpec:
        return build_prefix_spec(self.documents[KIND_PREFIX_SPEC])

    def graph_spec(self) -> graph_route.GraphSpec:
        return build_graph_spec(self.documents[KIND_GRAPH_ATTENTION_SPEC])

    def attention_candidates(self) -> Tuple[graph_route.AttentionCandidate, ...]:
        return build_attention_candidates(self.documents[KIND_GRAPH_ATTENTION_SPEC])

    def spec_decode(self) -> Dict[str, Any]:
        return build_spec_decode_contract(self.documents[KIND_SPEC_DECODE_SPEC])

    def comparison_method(self) -> Dict[str, Any]:
        return build_comparison_method(self.documents[KIND_COMPARISON_SPEC])

    def comparison_spec(self, **identity: Any) -> comparison_mod.ComparisonSpec:
        """Bind the frozen method to a concrete run identity (see the builder)."""
        return make_comparison_spec(self.documents[KIND_COMPARISON_SPEC], **identity)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "documents": {kind: document.as_dict() for kind, document in self.documents.items()},
            "audits": self.audits,
        }


__all__ = [
    "ALLOWED_KEYS",
    "BUILDERS",
    "COMMON_KEYS",
    "KINDS",
    "KIND_COMPARISON_SPEC",
    "KIND_FAILURE_SPEC",
    "KIND_GRAPH_ATTENTION_SPEC",
    "KIND_KV_SPEC",
    "KIND_PREFIX_SPEC",
    "KIND_REQUEST_SPEC",
    "KIND_SCHEDULER_SPEC",
    "KIND_SPEC_DECODE_SPEC",
    "RuntimeDocument",
    "RuntimeSpecs",
    "audit_directory",
    "audit_kv_spec",
    "build_attention_candidates",
    "build_comparison_method",
    "build_graph_spec",
    "build_object",
    "build_prefix_spec",
    "build_scheduler_spec",
    "build_spec_decode_contract",
    "contract_audit",
    "default_config_dir",
    "load_any",
    "load_directory",
    "load_yaml_document",
    "make_comparison_spec",
]
