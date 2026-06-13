"""Strict loading and contract auditing of the frozen S08 configuration.

Every S08 configuration is a *frozen artifact*: the protocol subset, the error
catalog, the SLO template, the capacity protocol, the arrival processes, the
workload suite, the admission policy, the routing spec, the cache-routing spec,
the stream lifecycle, the fault matrix and the telemetry spec.  The loader
refuses unknown keys rather than ignoring them, and two functions keep the
YAML honest:

* :func:`build_object` turns a document into the object the experiment will
  actually use — a YAML that cannot be executed fails at load time, not at
  report time;
* :func:`audit_documents` compares the documents with the code field by field,
  so configuration and behaviour cannot drift apart silently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import yaml

from hqsb.core.errors import ConfigError
from hqsb.serving import arrival as arrival_mod
from hqsb.serving import cache_routing
from hqsb.serving import circuit as circuit_mod
from hqsb.serving import faults as faults_mod
from hqsb.serving import protocol as protocol_mod
from hqsb.serving import slo as slo_mod
from hqsb.serving.admission import PressureSpec
from hqsb.serving.router import ScoreSpec

KIND_PROTOCOL_PROFILE = "hqsb.serving.protocol_profile"
KIND_ERROR_CATALOG = "hqsb.serving.error_catalog"
KIND_SLO_SPEC = "hqsb.serving.slo_spec"
KIND_CAPACITY_SPEC = "hqsb.serving.capacity_spec"
KIND_ARRIVAL_SPEC = "hqsb.serving.arrival_spec"
KIND_WORKLOAD_SUITE = "hqsb.serving.workload_suite"
KIND_ADMISSION_SPEC = "hqsb.serving.admission_spec"
KIND_ROUTING_SPEC = "hqsb.serving.routing_spec"
KIND_CACHE_ROUTING_SPEC = "hqsb.serving.cache_routing_spec"
KIND_STREAM_LIFECYCLE_SPEC = "hqsb.serving.stream_lifecycle_spec"
KIND_FAULT_SPEC = "hqsb.serving.fault_spec"
KIND_TELEMETRY_SPEC = "hqsb.serving.telemetry_spec"

KINDS: Tuple[str, ...] = (
    KIND_PROTOCOL_PROFILE,
    KIND_ERROR_CATALOG,
    KIND_SLO_SPEC,
    KIND_CAPACITY_SPEC,
    KIND_ARRIVAL_SPEC,
    KIND_WORKLOAD_SUITE,
    KIND_ADMISSION_SPEC,
    KIND_ROUTING_SPEC,
    KIND_CACHE_ROUTING_SPEC,
    KIND_STREAM_LIFECYCLE_SPEC,
    KIND_FAULT_SPEC,
    KIND_TELEMETRY_SPEC,
)

#: Keys allowed on every document, independent of its kind.
COMMON_KEYS: Tuple[str, ...] = ("kind", "description", "name", "notes")

ALLOWED_KEYS: Mapping[str, Tuple[str, ...]] = {
    KIND_PROTOCOL_PROFILE: (
        "profile_version",
        "schema_version",
        "upstream_reference",
        "endpoints",
        "stream_modes",
        "completion_request_fields",
        "chat_request_fields",
        "message_roles",
        "response_fields",
        "choice_fields",
        "finish_reasons",
        "sse",
        "limits",
        "policies",
        "trace_headers",
        "extension_namespace",
        "unsupported_features",
        "error_envelope_fields",
    ),
    KIND_ERROR_CATALOG: (
        "schema_version",
        "envelope_fields",
        "stages",
        "retryable_values",
        "entries",
    ),
    KIND_SLO_SPEC: (
        "schema_version",
        "preregistration_status",
        "preregistration_note",
        "frozen_by",
        "frozen_at",
        "measurement_window_sec",
        "warmup_sec",
        "ramp_sec",
        "drain_sec",
        "percentile_method",
        "tail_min_samples",
        "min_samples_per_class",
        "goodput_uses_lower_confidence_bound",
        "lcb_method",
        "lcb_resamples",
        "allow_degraded_backend",
        "allow_precision_downgrade",
        "max_error_ratio",
        "classes",
        "classification",
        "slos_do_not_allow",
    ),
    KIND_CAPACITY_SPEC: (
        "schema_version",
        "preregistration_status",
        "stream_mode",
        "stream_and_non_stream_separate",
        "primary_load_model",
        "secondary_load_model",
        "secondary_may_not_replace_primary",
        "offered_qps_grid",
        "per_point",
        "sweep_plan",
        "order_policy",
        "run_isolated_processes",
        "per_point_reset_required",
        "client_retry",
        "client_retry_reason",
        "steady_state_criteria",
        "saturation_may_be_caused_by",
        "must_identify_with_trace",
        "tail_rules",
        "recovery",
        "loadgen_validity",
    ),
    KIND_ARRIVAL_SPEC: (
        "schema_version",
        "preregistration_status",
        "seeds",
        "mean_rate_source",
        "distributions",
        "comparability",
        "fidelity_gate",
        "load_bands",
        "overload_band_requires_bounded_policy",
        "recovery_metrics",
        "arrival_payload_separation",
        "arrival_trace_fields",
    ),
    KIND_WORKLOAD_SUITE: (
        "schema_version",
        "preregistration_status",
        "payload_trace_separate_from_arrival_trace",
        "request_classes",
        "tenants",
        "priority_levels",
        "cost_model",
        "fairness_guardrails",
        "holtrace_constructs",
        "isolated_baseline_required_for_slowdown",
        "homogeneous_control_required",
        "prediction_error_records",
        "forbidden_online_inputs",
        "overload_waveforms",
        "slow_client_behaviors",
    ),
    KIND_ADMISSION_SPEC: (
        "schema_version",
        "preregistration_status",
        "pressure_states",
        "transitions",
        "min_dwell_windows",
        "window_ms",
        "signals",
        "hard_caps",
        "admission_cost_model",
        "reject_policy",
        "retry_policy",
        "recovery",
        "safety_invariants",
    ),
    KIND_ROUTING_SPEC: (
        "schema_version",
        "preregistration_status",
        "registry",
        "hard_filter_order",
        "score_features",
        "score_sense",
        "tie_break",
        "missing_telemetry_never_zero",
        "low_confidence_telemetry_policy",
        "slo_feasibility",
        "fallback_ladder",
        "reduce_precision_because_backend_is_busy",
        "quarantine_triggers",
        "quarantine_is_not_a_transient_open",
        "no_feasible_policy",
        "route_record_fields",
        "route_vs_actual_conservation_required",
    ),
    KIND_CACHE_ROUTING_SPEC: (
        "schema_version",
        "preregistration_status",
        "identity_fields",
        "matcher",
        "policies",
        "joint_score",
        "telemetry",
        "net_value",
        "skew",
        "correctness_oracle",
        "correctness_assertions",
        "load_locality_matrix",
        "zero_locality_falsification_required",
    ),
    KIND_STREAM_LIFECYCLE_SPEC: (
        "schema_version",
        "preregistration_status",
        "layers",
        "silent_token_drop",
        "overflow_action_must_be_observable",
        "write",
        "cancel",
        "drain",
        "resource_invariants",
        "longrun",
        "client_behaviors_reference",
    ),
    KIND_FAULT_SPEC: (
        "schema_version",
        "preregistration_status",
        "faults",
        "circuit_breaker",
        "recovery_metrics",
        "process_restart_is_not_service_recovery",
        "commit_phase_vocabulary",
        "attempt_lineage_fields",
        "blast_radius_groups",
        "injector_requirements",
    ),
    KIND_TELEMETRY_SPEC: (
        "schema_version",
        "preregistration_status",
        "identifiers",
        "trace_context",
        "clock_domains",
        "cross_process_subtraction_without_offset",
        "gpu_time_from_host_enqueue_duration",
        "spans",
        "metrics",
        "histogram_rules",
        "log_fields",
        "redaction",
        "cardinality",
        "sampling_modes",
        "tail_and_error_never_fully_sampled_out",
        "overhead_gate",
        "root_cause_classes",
        "root_cause_requires_counterfactual",
    ),
}


@dataclass(frozen=True)
class ServingDocument:
    """A loaded, validated S08 document."""

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
        raise ConfigError(f"{path}: a serving config document must be a mapping")
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


# ── object construction ────────────────────────────────────────────────────


def build_object(kind: str, payload: Mapping[str, Any]) -> Any:
    """Turn a document into the object the run will use."""
    if kind == KIND_PROTOCOL_PROFILE:
        return protocol_mod.ProtocolProfile.from_document(payload)
    if kind == KIND_ERROR_CATALOG:
        return protocol_mod.ErrorCatalog.from_document(payload)
    if kind == KIND_SLO_SPEC:
        return slo_mod.SLOSpec.from_document(payload)
    if kind == KIND_ARRIVAL_SPEC:
        return arrival_mod.ArrivalSpec.from_document(payload)
    if kind == KIND_ADMISSION_SPEC:
        return PressureSpec.from_document(payload)
    if kind == KIND_ROUTING_SPEC:
        return ScoreSpec.from_document(payload)
    if kind == KIND_FAULT_SPEC:
        return {
            "faults": faults_mod.FaultSpec.from_document(payload),
            "circuit": circuit_mod.CircuitSpec.from_document(payload),
        }
    if kind == KIND_CACHE_ROUTING_SPEC:
        return cache_routing.JointWeights.from_document(payload)
    return dict(payload)


@dataclass(frozen=True)
class ServingSpecs:
    """All loaded documents plus their audits."""

    documents: Mapping[str, ServingDocument]
    audits: Tuple[Mapping[str, Any], ...]

    @property
    def ok(self) -> bool:
        return all(bool(audit["ok"]) for audit in self.audits)

    @classmethod
    def load(cls, directory: str) -> "ServingSpecs":
        documents: Dict[str, ServingDocument] = {}
        for name in sorted(os.listdir(directory)):
            if not name.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(directory, name)
            payload = load_yaml_document(path)
            kind = str(payload["kind"])
            if kind in documents:
                raise ConfigError(
                    f"{path}: duplicate document for kind {kind!r}; two sources for one "
                    "policy would make the run ambiguous"
                )
            # construction is part of loading: an unusable document fails here
            build_object(kind, payload)
            documents[kind] = ServingDocument(kind=kind, path=path, payload=payload)
        missing = [kind for kind in KINDS if kind not in documents]
        if missing:
            raise ConfigError(
                f"{directory}: missing serving config document(s) for {missing}",
                details={"missing": missing},
            )
        return cls(documents=documents, audits=tuple(audit_documents(documents.values())))

    def kind(self, kind: str) -> ServingDocument:
        if kind not in self.documents:
            raise ConfigError(f"document {kind!r} was not loaded")
        return self.documents[kind]


# ── audits ─────────────────────────────────────────────────────────────────


def _audit_protocol(payload: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    profile = protocol_mod.ProtocolProfile.from_document(payload)
    fields = set(profile.completion_request_fields) | set(profile.chat_request_fields)
    if not profile.sse.terminal_marker:
        problems.append("the SSE terminal marker must be declared")
    if profile.policies.get("unknown_field_policy") != "reject_400":
        problems.append("unknown fields must be refused, not dropped")
    for name, status in list(profile.completion_request_fields.items()) + list(
        profile.chat_request_fields.items()
    ):
        if status not in ("required", "optional", "unsupported"):
            problems.append(f"field {name!r} has unknown status {status!r}")
    del fields
    return problems


def _audit_error_catalog(payload: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    catalog = protocol_mod.ErrorCatalog.from_document(payload)
    audit = catalog.validate()
    problems.extend(audit["problems"])
    for code in ("queue_full", "service_overloaded", "backend_unavailable", "client_cancelled"):
        if code not in catalog.entries:
            problems.append(f"the error catalog is missing the required code {code!r}")
    return problems


def _audit_slo(payload: Mapping[str, Any], suite: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    spec = slo_mod.SLOSpec.from_document(payload)
    declared_classes = {str(item["name"]) for item in suite.get("request_classes", [])} | {
        str(item.get("class", "")) for item in suite.get("request_classes", [])
    }
    declared_classes.discard("")
    for name in spec.classes:
        if declared_classes and name not in declared_classes:
            problems.append(
                f"SLO class {name!r} is not used by any workload class; a threshold for a "
                "class that never runs would never be exercised"
            )
    if not spec.classification.get("rejected_enters_goodput_denominator"):
        problems.append("rejected requests must stay in the offered denominator")
    if spec.allow_degraded_backend:
        problems.append(
            "silent backend degradation is enabled in the SLO template; it must be an "
            "explicit, recorded decision"
        )
    return problems


def _audit_capacity(payload: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    if payload.get("client_retry"):
        problems.append("client retries must be disabled in a capacity run")
    points = [float(item) for item in payload["offered_qps_grid"]]
    if points != sorted(points) or len(points) < 4:
        problems.append("the offered-load grid must be sorted and have at least four points")
    if int(payload["per_point"]["repeats"]) < 3:
        problems.append("each load point needs at least three independent runs")
    if not payload["per_point"].get("independent_processes_required"):
        problems.append("load points must come from independent processes")
    if not payload["run_isolated_processes"] or not payload["per_point_reset_required"]:
        problems.append("state reset between load points is required")
    if not payload["recovery"]["criteria"]:
        problems.append("the recovery criteria list must not be empty")
    return problems


def _audit_arrival(payload: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    spec = arrival_mod.ArrivalSpec.from_document(payload)
    for distribution in ("constant", "poisson", "on_off_burst", "batch_burst"):
        if distribution not in spec.distributions:
            problems.append(f"distribution {distribution!r} is missing")
    if len(spec.seeds) < 2:
        problems.append("at least two seeds are required: one seed cannot support a conclusion")
    if spec.comparability.get("same_payload_sequence_or_paired_permutation") is not True:
        problems.append("the payload sequence must be shared or paired across distributions")
    if not spec.comparability.get("same_warmup_and_drain"):
        problems.append("warmup/drain must be identical across distributions")
    return problems


def _audit_workload_suite(payload: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    if not payload.get("payload_trace_separate_from_arrival_trace"):
        problems.append("payload and arrival traces must be separated so they can be recombined")
    tenants = [dict(item) for item in payload["tenants"]]
    if len(tenants) < 2:
        problems.append("at least two tenants are needed for a fairness comparison")
    for tenant in tenants:
        if float(tenant.get("entitlement", 0)) <= 0:
            problems.append(f"tenant {tenant.get('name')!r} has a non-positive entitlement")
    cost = dict(payload["cost_model"])
    for key in ("alpha_uncached_input_token", "beta_committed_output_token", "version"):
        if key not in cost:
            problems.append(f"the cost model is missing {key!r}")
    if not payload.get("isolated_baseline_required_for_slowdown"):
        problems.append("slowdown requires a matched isolated baseline")
    if not payload.get("homogeneous_control_required"):
        problems.append("a homogeneous control is required to separate policy overhead")
    return problems


def _audit_admission(payload: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    spec = PressureSpec.from_document(payload)
    caps = dict(payload["hard_caps"])
    for key in ("max_queued_requests", "max_queued_tokens", "max_inflight_requests"):
        if int(caps.get(key, 0)) <= 0:
            problems.append(f"hard cap {key!r} must be positive")
    if not caps.get("unbounded_queue_kill_guard_required"):
        problems.append("the unbounded-queue counter-example needs a kill guard")
    retry = dict(payload["retry_policy"])
    for key in ("max_attempts_per_request", "max_elapsed_budget_ms", "jitter_ratio"):
        if key not in retry:
            problems.append(f"the retry policy is missing {key!r}")
    if not retry.get("no_transparent_retry_after_stream_commit"):
        problems.append("a stream that already committed bytes must not be retried transparently")
    if spec.min_dwell_windows < 2:
        problems.append("hysteresis needs a minimum dwell of at least two windows")
    return problems


def _audit_routing(payload: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    order = [str(item) for item in payload["hard_filter_order"]]
    if order[-1] != "deadline_slo_envelope":
        problems.append("the deadline/SLO filter must run after the hard capability filters")
    for name in ("model_alias_resolve", "precision_quality_tenant_policy", "circuit_health"):
        if name not in order:
            problems.append(f"hard filter {name!r} is missing from the declared order")
    features = [dict(item) for item in payload["score_features"]]
    if not features:
        problems.append("at least one score feature is required")
    for item in features:
        for key in ("name", "unit", "normalization", "weight", "missing_policy"):
            if key not in item:
                problems.append(f"score feature {item.get('name')!r} is missing {key!r}")
        if item.get("missing_policy") in ("zero", "treat_as_zero"):
            problems.append(f"missing telemetry for {item.get('name')!r} must not be zero")
    if not payload.get("missing_telemetry_never_zero"):
        problems.append("missing telemetry must never be read as zero")
    if payload.get("reduce_precision_because_backend_is_busy") != "forbidden":
        problems.append("degrading precision because a Backend is busy must be forbidden")
    return problems


def _audit_cache_routing(payload: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    identity = [str(item) for item in payload["identity_fields"]]
    for field_name in ("model_id", "weight_revision", "precision", "tokenizer_id", "cache_epoch"):
        if field_name not in identity:
            problems.append(f"cache identity field {field_name!r} is missing")
    policies = [str(item) for item in payload["policies"]]
    for policy in cache_routing.CACHE_POLICIES:
        if policy not in policies:
            problems.append(f"cache policy {policy!r} is missing from the comparison set")
    if payload["matcher"].get("text_hash_is_not_identity") is not True:
        problems.append("a text hash must not be used as the cache identity")
    if payload["matcher"].get("backend_second_check_required") is not True:
        problems.append("the Backend must re-validate the block identity")
    if payload["joint_score"].get("per_workload_retuning_forbidden") is not True:
        problems.append("joint weights may not be retuned per workload")
    if not payload["zero_locality_falsification_required"]:
        problems.append("a zero-locality falsification run is required")
    return problems


def _audit_stream_lifecycle(payload: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    layers = [dict(item) for item in payload["layers"]]
    if len(layers) < 5:
        problems.append("all five buffer layers must be declared with their caps")
    for layer in layers:
        for key in ("name", "owner", "cap_bytes", "on_cap"):
            if key not in layer:
                problems.append(f"layer {layer.get('name')!r} is missing {key!r}")
    if payload.get("silent_token_drop") != "forbidden":
        problems.append("silently dropping tokens inside a stream must be forbidden")
    if not payload["cancel"].get("linearization_point_order"):
        problems.append("the cancel linearization point must be defined")
    if not payload["drain"].get("drain_deadline_ms"):
        problems.append("drain needs a deadline")
    if not payload["drain"].get("close_is_idempotent"):
        problems.append("close must be idempotent")
    return problems


def _audit_fault_spec(payload: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    spec = faults_mod.FaultSpec.from_document(payload)
    for required in (
        "backend_process_crash",
        "backend_hang_no_progress",
        "runtime_oom",
        "model_load_failure",
        "identity_mismatch_response",
        "cache_unavailable",
        "mid_request_network_interruption",
    ):
        if required not in spec.entries:
            problems.append(f"the fault matrix is missing {required!r}")
    breaker = dict(payload["circuit_breaker"])
    excluded = set(breaker.get("excluded_failure_classes", []))
    for name in ("client_4xx", "user_cancel", "admission_reject"):
        if name not in excluded:
            problems.append(f"{name!r} must not be counted as a Backend failure")
    if not breaker.get("quarantine_classes"):
        problems.append("correctness failures need a quarantine class")
    for metric in ("MTTD", "isolation_time", "SLO_recovery"):
        if metric not in payload["recovery_metrics"]:
            problems.append(f"recovery metric {metric!r} is missing")
    return problems


def _audit_telemetry(payload: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    from hqsb.serving import observability as obs

    for kind in obs.SPAN_KINDS:
        if kind not in payload["spans"]:
            problems.append(f"span kind {kind!r} is missing from the telemetry spec")
    declared = set(payload["metrics"]["counters"]) | set(payload["metrics"]["histograms"]) | set(
        payload["metrics"]["gauges"]
    )
    for name in obs.METRIC_KINDS:
        if name not in declared:
            problems.append(f"metric {name!r} is missing from the telemetry spec")
    if not payload["histogram_rules"].get("buckets_must_cover_slo_and_timeout"):
        problems.append("histogram buckets must cover the SLO and the timeout")
    if payload["histogram_rules"].get("aggregate_summary_quantiles") != "forbidden":
        problems.append("aggregate summary quantiles must be forbidden")
    if payload["redaction"].get("prompt_text") != "forbidden":
        problems.append("prompt text must never enter telemetry")
    if not payload.get("tail_and_error_never_fully_sampled_out"):
        problems.append("tail/error traces must not be fully sampled out")
    if not payload.get("root_cause_requires_counterfactual"):
        problems.append("a root-cause claim needs a counterfactual")
    return problems


def audit_documents(documents: Sequence[ServingDocument]) -> List[Dict[str, Any]]:
    """Compare every document with the code that consumes it."""
    by_kind = {document.kind: document.payload for document in documents}
    reports: List[Dict[str, Any]] = []
    checks: Tuple[Tuple[str, Callable[[], List[str]]], ...] = (
        (KIND_PROTOCOL_PROFILE, lambda: _audit_protocol(by_kind[KIND_PROTOCOL_PROFILE])),
        (KIND_ERROR_CATALOG, lambda: _audit_error_catalog(by_kind[KIND_ERROR_CATALOG])),
        (
            KIND_SLO_SPEC,
            lambda: _audit_slo(
                by_kind[KIND_SLO_SPEC], by_kind.get(KIND_WORKLOAD_SUITE, {})
            ),
        ),
        (KIND_CAPACITY_SPEC, lambda: _audit_capacity(by_kind[KIND_CAPACITY_SPEC])),
        (KIND_ARRIVAL_SPEC, lambda: _audit_arrival(by_kind[KIND_ARRIVAL_SPEC])),
        (KIND_WORKLOAD_SUITE, lambda: _audit_workload_suite(by_kind[KIND_WORKLOAD_SUITE])),
        (KIND_ADMISSION_SPEC, lambda: _audit_admission(by_kind[KIND_ADMISSION_SPEC])),
        (KIND_ROUTING_SPEC, lambda: _audit_routing(by_kind[KIND_ROUTING_SPEC])),
        (
            KIND_CACHE_ROUTING_SPEC,
            lambda: _audit_cache_routing(by_kind[KIND_CACHE_ROUTING_SPEC]),
        ),
        (
            KIND_STREAM_LIFECYCLE_SPEC,
            lambda: _audit_stream_lifecycle(by_kind[KIND_STREAM_LIFECYCLE_SPEC]),
        ),
        (KIND_FAULT_SPEC, lambda: _audit_fault_spec(by_kind[KIND_FAULT_SPEC])),
        (KIND_TELEMETRY_SPEC, lambda: _audit_telemetry(by_kind[KIND_TELEMETRY_SPEC])),
    )
    for kind, check in checks:
        document = next(item for item in documents if item.kind == kind)
        problems = check()
        reports.append(
            {
                "kind": kind,
                "path": document.path,
                "ok": not problems,
                "problems": problems,
            }
        )
    return reports


__all__ = [
    "ALLOWED_KEYS",
    "COMMON_KEYS",
    "KINDS",
    "KIND_ADMISSION_SPEC",
    "KIND_ARRIVAL_SPEC",
    "KIND_CACHE_ROUTING_SPEC",
    "KIND_CAPACITY_SPEC",
    "KIND_ERROR_CATALOG",
    "KIND_FAULT_SPEC",
    "KIND_PROTOCOL_PROFILE",
    "KIND_ROUTING_SPEC",
    "KIND_SLO_SPEC",
    "KIND_STREAM_LIFECYCLE_SPEC",
    "KIND_TELEMETRY_SPEC",
    "KIND_WORKLOAD_SUITE",
    "ServingDocument",
    "ServingSpecs",
    "audit_documents",
    "build_object",
    "load_yaml_document",
]
