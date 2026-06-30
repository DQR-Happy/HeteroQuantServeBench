"""Four-layer benchmark semantics, estimands and token accounting (E12-03 §5–§8).

The four layers are *not* four ways of saying "speed":

``operator``
    one kernel under a frozen OperatorSpec (shape/dtype/layout/epilogue).
``model_core``
    the model without tokenizer/HTTP/queue: prefill, per-token decode, KV.
``service``
    request-level behaviour under open-loop/closed-loop/offline load.
``distributed``
    multi-device execution with an explicit scaling kind.

Sharing one name (``latency_ms``) across layers is the single most common way a
cross-hardware table lies, so each layer carries its own boundary, estimand
names and mandatory fields, and :func:`validate_layer_metric` rejects a metric
whose boundary does not belong to the layer.

Token accounting follows §8.1: the primary denominators are
``accepted_output_tokens`` / ``served_output_tokens``, never "tokens" in
general.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

LAYER_OPERATOR = "operator"
LAYER_MODEL_CORE = "model_core"
LAYER_SERVICE = "service"
LAYER_DISTRIBUTED = "distributed"

LAYERS: Tuple[str, ...] = (LAYER_OPERATOR, LAYER_MODEL_CORE, LAYER_SERVICE, LAYER_DISTRIBUTED)

#: Scenarios per layer (E12-03 §8).  Mixing scenarios inside one estimate is a
#: FAIL condition, so the vocabulary is closed.
SCENARIOS: Mapping[str, Tuple[str, ...]] = {
    LAYER_OPERATOR: ("fixed_shape", "shape_sweep", "boundary_shape", "prefill_shape", "decode_shape", "collective_micro"),
    LAYER_MODEL_CORE: ("prefill", "decode", "prefill_decode_e2e", "fixed_output", "long_context", "decode_heavy"),
    LAYER_SERVICE: ("single_request", "low_load", "target_zone", "pre_saturation", "overload", "closed_loop", "offline_saturation"),
    LAYER_DISTRIBUTED: ("collective_micro", "strong_scaling", "weak_scaling", "capacity_scaling", "multi_node"),
}

#: Load modes: open-loop is the formal main line, closed-loop/offline are
#: separate scenarios and must never be averaged into one throughput.
LOAD_MODES: Tuple[str, ...] = ("open_loop", "closed_loop", "offline", "single_stream")

#: Allowed normalizations (E12-01 §11 / E12-03 §35): each answers a defined
#: question and keeps the raw values.
ALLOWED_NORMALIZATIONS: Tuple[str, ...] = (
    "per_request",
    "per_accepted_token",
    "per_input_token",
    "per_device",
    "per_device_second",
    "utilization_vs_sustainable_peak",
    "goodput_at_same_slo",
    "cost_per_token_same_scenario",
    "joule_per_token_same_boundary",
    "unit_conversion",
)

#: Forbidden normalizations (never allowed, independent of the contract).
FORBIDDEN_NORMALIZATIONS: Tuple[str, ...] = (
    "vendor_peak_flops_correction",
    "linear_batch_scaling_fill",
    "tdp_as_energy",
    "parameter_ratio_extrapolation",
    "mixed_tokenizer_ranking",
    "quality_mixed_score",
    "zero_fill_for_unsupported",
    "micro_ratio_to_service_throughput",
)

#: Estimands per layer.  ``boundary`` names the measurement edge, ``unit`` the
#: raw unit; ``formula`` is the frozen definition text.
ESTIMANDS: Mapping[str, Tuple[Mapping[str, str], ...]] = {
    LAYER_OPERATOR: (
        {"name": "operator_latency", "unit": "ns", "boundary": "kernel_launch_to_sync", "formula": "completion_event - launch_event"},
        {"name": "effective_bandwidth", "unit": "byte/s", "boundary": "kernel_launch_to_sync", "formula": "logical_bytes / latency"},
        {"name": "effective_ops_rate", "unit": "op/s", "boundary": "kernel_launch_to_sync", "formula": "declared_useful_ops / latency"},
        {"name": "operator_error", "unit": "rel", "boundary": "tensor_compare", "formula": "max_abs_err / reference_scale"},
    ),
    LAYER_MODEL_CORE: (
        {"name": "core_ttft", "unit": "ns", "boundary": "model_core_first_token", "formula": "first_token_time - core_admission_time"},
        {"name": "core_tpot", "unit": "ns", "boundary": "model_core_decode_interval", "formula": "median(ITL_i) after first token"},
        {"name": "core_e2e", "unit": "ns", "boundary": "model_core_completion", "formula": "complete_time - core_admission_time"},
        {"name": "core_token_rate", "unit": "token/s", "boundary": "model_core_completion", "formula": "accepted_output_tokens / core_seconds"},
        {"name": "core_peak_memory", "unit": "byte", "boundary": "device_allocator", "formula": "max(reserved_bytes over run)"},
    ),
    LAYER_SERVICE: (
        {"name": "client_ttft", "unit": "ns", "boundary": "request_admission_to_first_visible_token", "formula": "first_token_visible_time - request_admission_time"},
        {"name": "client_tpot", "unit": "ns", "boundary": "visible_token_interval", "formula": "statistic(ITL_i)"},
        {"name": "client_e2e", "unit": "ns", "boundary": "request_admission_to_completion", "formula": "request_complete_time - request_admission_time"},
        {"name": "request_throughput", "unit": "request/s", "boundary": "measurement_window", "formula": "completed_requests / measurement_seconds"},
        {"name": "token_goodput", "unit": "token/s", "boundary": "measurement_window", "formula": "accepted_tokens_from_compliant_requests / measurement_seconds"},
        {"name": "queue_delay", "unit": "ns", "boundary": "admission_to_execution_start", "formula": "execution_start_time - request_admission_time"},
        {"name": "error_rate", "unit": "rel", "boundary": "measurement_window", "formula": "(failed + rejected + timed_out) / admitted"},
    ),
    LAYER_DISTRIBUTED: (
        {"name": "strong_scaling_speedup", "unit": "ratio", "boundary": "same_global_workload", "formula": "T_1 / T_p"},
        {"name": "parallel_efficiency", "unit": "ratio", "boundary": "same_global_workload", "formula": "T_1 / (p * T_p)"},
        {"name": "device_seconds", "unit": "device*s", "boundary": "wall_clock_all_ranks", "formula": "p * wall_seconds"},
        {"name": "collective_latency", "unit": "ns", "boundary": "collective_enqueue_to_completion", "formula": "completion - enqueue"},
        {"name": "collective_bandwidth", "unit": "byte/s", "boundary": "collective_enqueue_to_completion", "formula": "message_bytes / latency"},
        {"name": "global_goodput", "unit": "token/s", "boundary": "cluster_window", "formula": "accepted_tokens / wall_seconds"},
    ),
}


def estimand_names(layer: str) -> Tuple[str, ...]:
    if layer not in ESTIMANDS:
        raise ConfigError(f"unknown layer {layer!r}")
    return tuple(row["name"] for row in ESTIMANDS[layer])


def estimand(layer: str, name: str) -> Mapping[str, str]:
    for row in ESTIMANDS.get(layer, ()):
        if row["name"] == name:
            return row
    raise ConfigError(f"unknown estimand {name!r} for layer {layer!r}")


# ── measurement boundaries ────────────────────────────────────────────────


@dataclass(frozen=True)
class MeasurementBoundary:
    """Start/end events plus what the boundary does and does not include."""

    layer: str
    boundary_id: str
    start_event: str
    end_event: str
    includes_tokenizer: bool
    includes_queue: bool
    includes_http: bool
    includes_host_device_copy: bool
    includes_load_or_compile: bool
    includes_sync: bool
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "boundary_id": self.boundary_id,
            "start_event": self.start_event,
            "end_event": self.end_event,
            "includes_tokenizer": self.includes_tokenizer,
            "includes_queue": self.includes_queue,
            "includes_http": self.includes_http,
            "includes_host_device_copy": self.includes_host_device_copy,
            "includes_load_or_compile": self.includes_load_or_compile,
            "includes_sync": self.includes_sync,
            "notes": self.notes,
        }


MEASUREMENT_BOUNDARIES: Mapping[str, MeasurementBoundary] = {
    LAYER_OPERATOR: MeasurementBoundary(
        layer=LAYER_OPERATOR,
        boundary_id="operator_kernel_only",
        start_event="kernel_launch_enqueued",
        end_event="completion_event_after_sync",
        includes_tokenizer=False,
        includes_queue=False,
        includes_http=False,
        includes_host_device_copy=False,
        includes_load_or_compile=False,
        includes_sync=True,
        notes="microkernel boundary; a micro speedup may never be extrapolated to model throughput",
    ),
    LAYER_MODEL_CORE: MeasurementBoundary(
        layer=LAYER_MODEL_CORE,
        boundary_id="model_core_forward",
        start_event="core_admission_token_ids_ready",
        end_event="last_token_produced",
        includes_tokenizer=False,
        includes_queue=False,
        includes_http=False,
        includes_host_device_copy=True,
        includes_load_or_compile=False,
        includes_sync=True,
        notes="no tokenizer/HTTP/queue; load/compile recorded as cold-start stages instead",
    ),
    LAYER_SERVICE: MeasurementBoundary(
        layer=LAYER_SERVICE,
        boundary_id="service_client_visible",
        start_event="request_admission_at_client",
        end_event="request_complete_at_client",
        includes_tokenizer=True,
        includes_queue=True,
        includes_http=True,
        includes_host_device_copy=True,
        includes_load_or_compile=False,
        includes_sync=True,
        notes="client-visible boundary; server-side boundary requires different field names",
    ),
    LAYER_DISTRIBUTED: MeasurementBoundary(
        layer=LAYER_DISTRIBUTED,
        boundary_id="cluster_wall_clock",
        start_event="cluster_run_start",
        end_event="all_ranks_complete",
        includes_tokenizer=True,
        includes_queue=True,
        includes_http=True,
        includes_host_device_copy=True,
        includes_load_or_compile=False,
        includes_sync=True,
        notes="per-rank traces must be aligned; wall time alone cannot explain the result",
    ),
}


def boundary_for(layer: str) -> MeasurementBoundary:
    try:
        return MEASUREMENT_BOUNDARIES[layer]
    except KeyError as exc:
        raise ConfigError(f"unknown layer {layer!r}") from exc


def validate_layer_metric(layer: str, metric_name: str) -> List[str]:
    """A metric belongs to exactly one layer: boundary mixing is rejected."""
    problems: List[str] = []
    owners = [name for name, rows in ESTIMANDS.items() if any(row["name"] == metric_name for row in rows)]
    if not owners:
        problems.append(f"unknown metric {metric_name!r}: it has no estimand definition")
    elif layer not in owners:
        problems.append(
            f"metric {metric_name!r} belongs to layer(s) {owners}, not {layer!r} "
            f"(same-named metrics from different boundaries are not one estimand)"
        )
    return problems


#: Mandatory result fields per layer (E12-03 §12 projection).
LAYER_REQUIRED_FIELDS: Mapping[str, Tuple[str, ...]] = {
    LAYER_OPERATOR: (
        "shape", "dtype", "layout", "operator_spec_id", "reference_id", "tolerance_policy_id",
        "actual_backend", "latency_ns", "logical_bytes", "useful_ops", "quality_status",
    ),
    LAYER_MODEL_CORE: (
        "workload_spec_id", "input_token_ids_hash", "batch", "isl", "requested_osl", "accepted_osl",
        "phase", "kv_state_status", "actual_backend", "core_ttft_ns", "core_tpot_ns", "core_e2e_ns",
        "accepted_output_tokens", "quality_status",
    ),
    LAYER_SERVICE: (
        "request_trace_id", "load_mode", "arrival_process", "slo_id", "admission_policy", "scheduler",
        "streaming", "timeout_policy", "retry_policy", "goodput", "error_rate", "token_accounting_closed",
    ),
    LAYER_DISTRIBUTED: (
        "scaling_kind", "device_count", "rank_mapping", "parallel_plan_id", "topology_id",
        "collective_bytes", "global_workload", "local_workload", "device_seconds", "baseline_run_id",
        "fault_status",
    ),
}

# ── token and request accounting (§8.1, §8.3) ─────────────────────────────

TOKEN_ACCOUNTING_FIELDS: Tuple[str, ...] = (
    "prompt_tokens",
    "requested_output_tokens",
    "generated_tokens_before_stop",
    "accepted_output_tokens",
    "served_output_tokens",
    "rejected_tokens",
    "speculative_draft_tokens",
)

REQUEST_ACCOUNTING_FIELDS: Tuple[str, ...] = (
    "admitted_requests",
    "completed_requests",
    "failed_requests",
    "rejected_requests",
    "timed_out_requests",
    "cancelled_requests",
    "retried_requests",
    "backlog_requests",
)

#: The default denominator for primary metrics (§8.1).
PRIMARY_TOKEN_DENOMINATOR = "accepted_output_tokens"


@dataclass
class TokenAccounting:
    """One run's token/request ledger; the only legal source of denominators."""

    run_id: str
    prompt_tokens: int = 0
    requested_output_tokens: int = 0
    generated_tokens_before_stop: int = 0
    accepted_output_tokens: int = 0
    served_output_tokens: int = 0
    rejected_tokens: int = 0
    speculative_draft_tokens: int = 0
    admitted_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    timed_out_requests: int = 0
    cancelled_requests: int = 0
    retried_requests: int = 0
    backlog_requests: int = 0

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.run_id:
            problems.append("token accounting needs a run_id")
        for name in TOKEN_ACCOUNTING_FIELDS + REQUEST_ACCOUNTING_FIELDS:
            value = getattr(self, name)
            if value < 0:
                problems.append(f"{name} must not be negative (got {value})")
        if self.accepted_output_tokens > self.generated_tokens_before_stop:
            problems.append(
                "accepted_output_tokens must not exceed generated_tokens_before_stop"
            )
        if self.generated_tokens_before_stop > self.requested_output_tokens + self.speculative_draft_tokens:
            problems.append(
                "generated tokens exceed requested output tokens plus draft tokens: "
                "the runner counter and the external re-count disagree"
            )
        if self.admitted_requests < self.completed_requests + self.failed_requests + self.timed_out_requests + self.cancelled_requests:
            problems.append("admitted requests must cover completed + failed + timed out + cancelled")
        return problems

    def closed(self, *, tolerance: int = 0) -> bool:
        """Whether request/token accounting closes (E12-06 §22)."""
        terminal = (
            self.completed_requests
            + self.failed_requests
            + self.timed_out_requests
            + self.cancelled_requests
            + self.backlog_requests
        )
        return abs(terminal - self.admitted_requests) <= tolerance

    def as_dict(self) -> Dict[str, Any]:
        payload = {"run_id": self.run_id}
        for name in TOKEN_ACCOUNTING_FIELDS + REQUEST_ACCOUNTING_FIELDS:
            payload[name] = getattr(self, name)
        payload["closed"] = self.closed()
        return payload


def recompute_token_accounting(
    *,
    run_id: str,
    prompt_token_count: int,
    returned_token_ids: Sequence[Sequence[int]],
    requested_output_tokens: int,
    stop_reason_by_request: Optional[Sequence[str]] = None,
    draft_tokens: Sequence[int] = (),
) -> TokenAccounting:
    """Re-derive accounting from returned token ids (E12-03 step 17).

    The runner's own counters must agree with this external re-count; a
    mismatch is a validation failure, not a rounding detail.
    """
    generated = sum(len(list(ids)) for ids in returned_token_ids)
    accepted = generated
    earliest = [
        index
        for index, reason in enumerate(stop_reason_by_request or [])
        if reason in ("eos", "stop_sequence")
    ]
    if earliest:
        earliest_index = min(earliest)
        accepted = sum(len(list(ids)) for ids in list(returned_token_ids)[: earliest_index + 1])
    return TokenAccounting(
        run_id=run_id,
        prompt_tokens=int(prompt_token_count),
        requested_output_tokens=int(requested_output_tokens),
        generated_tokens_before_stop=int(generated),
        accepted_output_tokens=int(accepted),
        served_output_tokens=int(accepted),
        speculative_draft_tokens=int(sum(draft_tokens)),
        admitted_requests=len(list(returned_token_ids)),
        completed_requests=len(list(returned_token_ids)),
    )


# ── SLO qualification and goodput (§8.3) ──────────────────────────────────


@dataclass
class SloSpec:
    """Frozen SLO; goodput is defined *relative to this object*."""

    slo_id: str
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    e2e_ms: Optional[float] = None
    error_rate_max: Optional[float] = None
    quality_required: bool = True
    quantile: str = "p95"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.slo_id:
            problems.append("slo_id is required")
        if not any(value is not None for value in (self.ttft_ms, self.tpot_ms, self.e2e_ms)):
            problems.append("an SLO must constrain at least one latency metric")
        if self.quantile not in ("p50", "p90", "p95", "p99", "mean"):
            problems.append(f"unknown quantile {self.quantile!r}")
        for name in ("ttft_ms", "tpot_ms", "e2e_ms"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                problems.append(f"{name} must be positive")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "slo_id": self.slo_id,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "e2e_ms": self.e2e_ms,
            "error_rate_max": self.error_rate_max,
            "quality_required": self.quality_required,
            "quantile": self.quantile,
        }


def qualify_request(
    row: Mapping[str, Any],
    slo: SloSpec,
    *,
    quality_ok: bool,
) -> Dict[str, Any]:
    """Judge one request against the SLO; never modifies the raw row."""
    problems = slo.validate()
    if problems:
        raise ConfigError("invalid SLO: " + "; ".join(problems))
    failures: List[str] = []
    if row.get("status") not in (None, "ok", "OK", "completed"):
        failures.append("REQUEST_STATUS_NOT_OK")
    if quality_ok is False and slo.quality_required:
        failures.append("QUALITY_NOT_SATISFIED")
    for field_name, limit in (("ttft_ms", slo.ttft_ms), ("tpot_ms", slo.tpot_ms), ("e2e_ms", slo.e2e_ms)):
        if limit is None:
            continue
        value = row.get(field_name)
        if value is None:
            failures.append(f"{field_name.upper()}_MISSING")
            continue
        if float(value) > float(limit):
            failures.append(f"{field_name.upper()}_EXCEEDED")
    if slo.error_rate_max is not None:
        error_rate = row.get("error_rate")
        if error_rate is None:
            failures.append("ERROR_RATE_MISSING")
        elif float(error_rate) > slo.error_rate_max:
            failures.append("ERROR_RATE_EXCEEDED")
    return {
        "request_id": row.get("request_id", ""),
        "compliant": not failures,
        "failures": failures,
        "slo_id": slo.slo_id,
    }


def goodput(
    rows: Sequence[Mapping[str, Any]],
    slo: SloSpec,
    *,
    measurement_seconds: float,
    quality_ok_by_request: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Any]:
    """SLO-qualified goodput: over-SLO requests never count (E12-03 step 26)."""
    if measurement_seconds <= 0:
        raise ConfigError("measurement_seconds must be positive for goodput")
    verdicts = [
        qualify_request(row, slo, quality_ok=(quality_ok_by_request or {}).get(str(row.get("request_id", "")), True))
        for row in rows
    ]
    compliant = [verdict for verdict in verdicts if verdict["compliant"]]
    accepted = sum(int(row.get("accepted_output_tokens", 0)) for row, verdict in zip(rows, verdicts) if verdict["compliant"])
    return {
        "slo_id": slo.slo_id,
        "requests": len(rows),
        "compliant_requests": len(compliant),
        "goodput": len(compliant) / measurement_seconds,
        "token_goodput": accepted / measurement_seconds,
        "compliant_tokens": accepted,
        "measurement_seconds": measurement_seconds,
        "failure_histogram": _histogram(failures for verdict in verdicts for failures in [verdict["failures"]] for failures in failures),
        "non_compliant_requests": len(verdicts) - len(compliant),
    }


def _histogram(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


# ── cross-layer conversion template (E12-03 §14) ──────────────────────────


@dataclass
class CrossLayerConversion:
    """operator effect → model-core → service → distributed, with residuals."""

    chain_id: str
    candidate_id: str
    workload_spec_id: str
    operator_effect: Optional[float] = None
    op_time_share: Optional[float] = None
    dispatch_hit_rate: Optional[float] = None
    predicted_model_core_effect: Optional[float] = None
    observed_model_core_effect: Optional[float] = None
    model_core_overheads: Mapping[str, float] = field(default_factory=dict)
    observed_service_effect: Optional[float] = None
    single_device_effect: Optional[float] = None
    observed_distributed_effect: Optional[float] = None
    residual_status: str = "UNEXPLAINED"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.chain_id or not self.candidate_id:
            problems.append("cross-layer conversion needs chain_id and candidate_id")
        if self.dispatch_hit_rate is not None and not 0.0 <= self.dispatch_hit_rate <= 1.0:
            problems.append("dispatch_hit_rate must be inside [0, 1]")
        if self.op_time_share is not None and not 0.0 <= self.op_time_share <= 1.0:
            problems.append("op_time_share must be inside [0, 1]")
        if self.residual_status not in ("EXPLAINED", "PARTIALLY_EXPLAINED", "UNEXPLAINED", "NOT_APPLICABLE"):
            problems.append(f"unknown residual status {self.residual_status!r}")
        if self.residual_status == "UNEXPLAINED" and self.predicted_model_core_effect is None:
            problems.append("an unexplained conversion must at least carry a prediction or a stated assumption")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "candidate_id": self.candidate_id,
            "workload_spec_id": self.workload_spec_id,
            "operator_effect": self.operator_effect,
            "op_time_share": self.op_time_share,
            "dispatch_hit_rate": self.dispatch_hit_rate,
            "predicted_model_core_effect": self.predicted_model_core_effect,
            "observed_model_core_effect": self.observed_model_core_effect,
            "model_core_overheads": dict(sorted(self.model_core_overheads.items())),
            "observed_service_effect": self.observed_service_effect,
            "single_device_effect": self.single_device_effect,
            "observed_distributed_effect": self.observed_distributed_effect,
            "residual_status": self.residual_status,
        }


def amdahl_upper_bound(fraction: float, local_speedup: float) -> float:
    """``1 / ((1 - f) + f / s)`` with fail-closed input validation."""
    if not 0.0 <= fraction <= 1.0:
        raise ConfigError(f"fraction must be inside [0, 1], got {fraction}")
    if local_speedup <= 0:
        raise ConfigError(f"local speedup must be positive, got {local_speedup}")
    return 1.0 / ((1.0 - fraction) + fraction / local_speedup)


def layer_of_scenario(scenario: str) -> str:
    for layer, scenarios in SCENARIOS.items():
        if scenario in scenarios:
            return layer
    raise ConfigError(f"unknown scenario {scenario!r}")


def validate_scenario(layer: str, scenario: str) -> List[str]:
    if layer not in SCENARIOS:
        return [f"unknown layer {layer!r}"]
    if scenario not in SCENARIOS[layer]:
        return [f"scenario {scenario!r} does not belong to layer {layer!r}"]
    return []


def layer_required_fields(layer: str) -> Tuple[str, ...]:
    return LAYER_REQUIRED_FIELDS.get(layer, ())
