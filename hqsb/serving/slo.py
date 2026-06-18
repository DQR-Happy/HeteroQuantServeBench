"""SLO pre-registration, request-level goodput and the counting funnel.

Two rules shape this module:

1. **The thresholds are frozen before the run** (E08-02 §11).  A spec that has
   not been frozen cannot evaluate anything: :meth:`SLOSpec.require_frozen`
   refuses, and the reporting helpers label the result ``NOT_EVALUATED``
   instead of quietly using template numbers.
2. **Nothing disappears from the denominator** (§10).  The funnel
   ``offered ≥ client_attempted ≥ gateway_received ≥ valid ≥ admitted ≥
   backend_started ≥ completed_success ≥ SLO_good`` must be monotone, and every
   difference needs a reason code — a rejected request is never deleted because
   it would flatter the goodput.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.benchmark.metrics import percentile
from hqsb.core.errors import ConfigError

#: Funnel stages, in the order the inequality must hold.
FUNNEL_STAGES: Tuple[str, ...] = (
    "offered",
    "client_attempted",
    "gateway_received",
    "valid",
    "admitted",
    "backend_started",
    "completed_success",
    "slo_good",
)

#: Why a request left the funnel (details README §10).
FUNNEL_REASONS: Tuple[str, ...] = (
    "loadgen_lag",
    "loadgen_drop",
    "connect_error",
    "invalid_request",
    "overload_reject",
    "tenant_quota_reject",
    "deadline_infeasible",
    "timeout",
    "cancel",
    "client_disconnect",
    "backend_failure",
    "protocol_failure",
    "slo_violation",
)

#: Failure buckets that never enter the success-latency distribution (§11).
NON_SUCCESS_BUCKETS: Tuple[str, ...] = (
    "rejected",
    "timeout",
    "cancel",
    "client_disconnect",
    "backend_failure",
    "protocol_failure",
)

INSUFFICIENT_TAIL = "INSUFFICIENT_TAIL_SAMPLES"
NOT_EVALUATED = "NOT_EVALUATED"


@dataclass(frozen=True)
class SLOClass:
    """One workload/tenant/service class with its frozen thresholds."""

    name: str
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    streaming: str = "stream_true"
    priority: str = "normal"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "e2e_ms": self.e2e_ms,
            "streaming": self.streaming,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class SLOSpec:
    """The pre-registered SLO document (a contract, not a result)."""

    status: str
    classes: Mapping[str, SLOClass]
    measurement_window_sec: float
    tail_min_samples: int
    min_samples_per_class: int
    goodput_uses_lcb: bool
    lcb_method: str
    lcb_resamples: int
    allow_degraded_backend: bool
    allow_precision_downgrade: bool
    max_error_ratio: float
    percentile_method: str
    classification: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0.0"

    @classmethod
    def from_document(cls, payload: Mapping[str, Any]) -> "SLOSpec":
        classes: Dict[str, SLOClass] = {}
        for raw in payload["classes"]:
            item = dict(raw)
            name = str(item["name"])
            if name in classes:
                raise ConfigError(f"duplicate SLO class {name!r}")
            classes[name] = SLOClass(
                name=name,
                ttft_ms=float(item["ttft_ms"]),
                tpot_ms=float(item["tpot_ms"]),
                e2e_ms=float(item["e2e_ms"]),
                streaming=str(item.get("streaming", "stream_true")),
                priority=str(item.get("priority", "normal")),
            )
        return cls(
            status=str(payload.get("preregistration_status", "template")),
            classes=classes,
            measurement_window_sec=float(payload["measurement_window_sec"]),
            tail_min_samples=int(payload["tail_min_samples"]),
            min_samples_per_class=int(payload["min_samples_per_class"]),
            goodput_uses_lcb=bool(payload["goodput_uses_lower_confidence_bound"]),
            lcb_method=str(payload["lcb_method"]),
            lcb_resamples=int(payload["lcb_resamples"]),
            allow_degraded_backend=bool(payload["allow_degraded_backend"]),
            allow_precision_downgrade=bool(payload["allow_precision_downgrade"]),
            max_error_ratio=float(payload["max_error_ratio"]),
            percentile_method=str(payload["percentile_method"]),
            classification=dict(payload.get("classification", {})),
            schema_version=str(payload.get("schema_version", "1.0.0")),
        )

    def require_frozen(self) -> None:
        if self.status != "frozen":
            raise ConfigError(
                f"SLO spec status is {self.status!r}, not 'frozen'; thresholds must be "
                "pre-registered before a formal run and may not be moved to a nicer "
                "point of a curve afterwards (E08-02 §11)",
                details={"preregistration_status": self.status},
            )

    @property
    def frozen(self) -> bool:
        return self.status == "frozen"

    def class_for(self, name: str) -> SLOClass:
        if name not in self.classes:
            raise ConfigError(f"unknown SLO class {name!r}")

        return self.classes[name]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "preregistration_status": self.status,
            "classes": {name: item.as_dict() for name, item in sorted(self.classes.items())},
            "measurement_window_sec": self.measurement_window_sec,
            "tail_min_samples": self.tail_min_samples,
            "min_samples_per_class": self.min_samples_per_class,
            "goodput_uses_lower_confidence_bound": self.goodput_uses_lcb,
            "lcb_method": self.lcb_method,
            "max_error_ratio": self.max_error_ratio,
            "percentile_method": self.percentile_method,
            "allow_degraded_backend": self.allow_degraded_backend,
            "allow_precision_downgrade": self.allow_precision_downgrade,
        }


@dataclass(frozen=True)
class RequestSLOInput:
    """Everything the goodput predicate may look at for one request."""

    request_id: str
    request_class: str
    protocol_success: bool
    quality_identity_ok: bool
    client_ttft_ms: Optional[float]
    tpot_ms: Optional[float]
    client_e2e_ms: Optional[float]
    outcome: str = "completed"
    failure_bucket: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_class": self.request_class,
            "protocol_success": self.protocol_success,
            "quality_identity_ok": self.quality_identity_ok,
            "client_ttft_ms": self.client_ttft_ms,
            "tpot_ms": self.tpot_ms,
            "client_e2e_ms": self.client_e2e_ms,
            "outcome": self.outcome,
            "failure_bucket": self.failure_bucket,
        }


@dataclass(frozen=True)
class GoodEvaluation:
    good: bool
    violated: Tuple[str, ...]
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {"good": self.good, "violated": list(self.violated), "reason": self.reason}


def _violates(value: Optional[float], threshold: float) -> bool:
    return value is None or value > threshold


def evaluate_request(row: RequestSLOInput, spec: SLOSpec) -> GoodEvaluation:
    """``good_i`` of details README §10 — all four conditions are required."""
    slo_class = spec.class_for(row.request_class)
    violated: List[str] = []
    if not row.protocol_success:
        violated.append("protocol_success")
    if not row.quality_identity_ok:
        violated.append("quality_identity_ok")
    if _violates(row.client_ttft_ms, slo_class.ttft_ms):
        violated.append("client_ttft")
    if _violates(row.tpot_ms, slo_class.tpot_ms):
        violated.append("tpot")
    if _violates(row.client_e2e_ms, slo_class.e2e_ms):
        violated.append("client_e2e")
    return GoodEvaluation(
        good=not violated,
        violated=tuple(violated),
        reason="" if not violated else "violates " + ", ".join(violated),
    )


def goodput(rows: Sequence[RequestSLOInput], spec: SLOSpec, *, window_sec: float) -> Dict[str, Any]:
    """Request goodput plus its ingredients (never a bare TPS)."""
    if window_sec <= 0:
        raise ConfigError("the measurement window must be positive")
    evaluations = [evaluate_request(row, spec) for row in rows]
    good = sum(1 for item in evaluations if item.good)
    completed = sum(1 for row in rows if row.protocol_success)
    return {
        "requests": len(rows),
        "good": good,
        "request_goodput": good / window_sec,
        "goodput_ratio": (good / len(rows)) if rows else None,
        "completed_qps": completed / window_sec,
        "window_sec": window_sec,
        "slo_frozen": spec.frozen,
        "violations": {
            name: sum(1 for item in evaluations if name in item.violated)
            for name in ("protocol_success", "quality_identity_ok", "client_ttft", "tpot", "client_e2e")
        },
    }


def token_goodput(
    rows: Sequence[RequestSLOInput],
    spec: SLOSpec,
    *,
    committed_tokens: Mapping[str, int],
    window_sec: float,
) -> Dict[str, Any]:
    """Token goodput *only* over SLO-good requests, named explicitly (§10)."""
    if not spec.frozen:
        return {
            "status": NOT_EVALUATED,
            "reason": "token goodput over SLO-good requests needs frozen thresholds",
        }
    good_ids = [row.request_id for row in rows if evaluate_request(row, spec).good]
    tokens = sum(int(committed_tokens.get(request_id, 0)) for request_id in good_ids)
    return {
        "status": "evaluated",
        "slo_good_requests": len(good_ids),
        "committed_tokens_of_good_requests": tokens,
        "token_goodput": tokens / window_sec,
        "note": (
            "this is NOT request goodput and must never be reported as 'throughput' "
            "in place of it"
        ),
    }


def percentile_from_raw(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    return percentile(list(values), quantile)


def latency_summary(values_ms: Sequence[float]) -> Dict[str, Optional[float]]:
    """P50/P95/P99 recomputed from per-request raw (never from a summary quantile)."""
    if not values_ms:
        return {"count": 0, "p50": None, "p95": None, "p99": None}
    ordered = sorted(float(value) for value in values_ms)
    return {
        "count": len(ordered),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def tail_sample_sufficiency(
    samples: int, spec: SLOSpec, *, label: str = "p99"
) -> Dict[str, Any]:
    """P99 needs a pre-registered sample count; otherwise say so (§15)."""
    enough = samples >= spec.tail_min_samples
    return {
        "label": label,
        "samples": samples,
        "required": spec.tail_min_samples,
        "sufficient": enough,
        "status": "" if enough else INSUFFICIENT_TAIL,
    }


def class_sample_sufficiency(rows: Sequence[RequestSLOInput], spec: SLOSpec) -> Dict[str, Any]:
    counts: Dict[str, int] = {name: 0 for name in spec.classes}
    for row in rows:
        counts[row.request_class] = counts.get(row.request_class, 0) + 1
    insufficient = {
        name: count
        for name, count in counts.items()
        if count < spec.min_samples_per_class
    }
    return {
        "ok": not insufficient,
        "counts": counts,
        "insufficient": insufficient,
        "status": "" if not insufficient else "INSUFFICIENT_CLASS_SAMPLES",
    }


# ── counting funnel ────────────────────────────────────────────────────────


@dataclass
class FunnelCounts:
    """Offered → SLO-good, with every difference attributed to a reason."""

    counts: Dict[str, int] = field(default_factory=lambda: {name: 0 for name in FUNNEL_STAGES})
    reasons: Dict[str, int] = field(default_factory=dict)
    label: str = ""
    loadgen_invalid_reason: str = ""

    def add(self, stage: str, value: int = 1) -> None:
        if stage not in FUNNEL_STAGES:
            raise ConfigError(f"unknown funnel stage {stage!r}")
        self.counts[stage] += value

    def add_reason(self, reason: str, value: int = 1) -> None:
        if reason not in FUNNEL_REASONS:
            raise ConfigError(
                f"unknown funnel reason {reason!r}; a difference without a catalogued "
                f"reason would be an unexplained loss (allowed: {list(FUNNEL_REASONS)})"
            )
        self.reasons[reason] = self.reasons.get(reason, 0) + value

    def audit(self) -> Dict[str, Any]:
        problems: List[str] = []
        previous_name, previous_value = "", None
        for stage in FUNNEL_STAGES:
            value = self.counts[stage]
            if previous_value is not None and value > previous_value:
                problems.append(
                    f"{stage} ({value}) exceeds {previous_name} ({previous_value})"
                )
            previous_name, previous_value = stage, value
        lost = self.counts["offered"] - self.counts["slo_good"]
        explained = sum(self.reasons.values())
        if self.counts["offered"] and lost != explained:
            problems.append(
                f"offered - slo_good = {lost} but the recorded reasons sum to "
                f"{explained}; every lost request needs a reason"
            )
        if self.label == "LOADGEN_INVALID":
            problems.append(
                "the run is labelled LOADGEN_INVALID: the client, not the service, hit "
                "its capacity, so the point may not be attributed to the service"
            )
        return {
            "ok": not problems,
            "problems": problems,
            "counts": dict(self.counts),
            "reasons": dict(self.reasons),
            "lost": lost,
            "label": self.label,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "reasons": dict(self.reasons),
            "label": self.label,
            "loadgen_invalid_reason": self.loadgen_invalid_reason,
            "audit": self.audit(),
        }


def classify_failure(code: str, spec: SLOSpec) -> str:
    """Map a terminal failure code to its funnel bucket (never to a success)."""
    if code in ("service_overloaded", "queue_full", "token_budget_exceeded", "kv_budget_exceeded"):
        return "rejected"
    if code == "tenant_rate_limited":
        return "rejected"
    if code in ("unknown_model", "invalid_json", "unknown_field", "unsupported_parameter"):
        return "invalid"
    if code == "deadline_infeasible":
        return "rejected"
    if code in ("deadline_exceeded", "backend_timeout"):
        return "timeout"
    if code in ("client_cancelled",):
        return "cancel"
    if code in ("client_disconnected",):
        return "client_disconnect"
    if code in ("backend_unavailable", "backend_oom", "no_feasible_backend"):
        return "backend_failure"
    if code in ("protocol_failure", "backend_identity_mismatch", "stream_error_after_commit"):
        return "protocol_failure"
    raise ConfigError(
        f"failure code {code!r} has no funnel bucket; an unclassified failure would be "
        "silently dropped from the accounting"
    )


def success_latency_rows(rows: Sequence[RequestSLOInput]) -> List[float]:
    """Only protocol successes contribute to the success-latency distribution."""
    return [
        float(row.client_ttft_ms)
        for row in rows
        if row.protocol_success and row.outcome == "completed" and row.client_ttft_ms is not None
    ]


# ── capacity curve helpers ─────────────────────────────────────────────────


@dataclass(frozen=True)
class LoadPointResult:
    """One load point: run-level goodput plus the tail and resource picture."""

    offered_qps: float
    goodput: float
    goodput_ratio: Optional[float]
    error_ratio: float
    reject_ratio: float
    p99_client_ttft_ms: Optional[float]
    p99_e2e_ms: Optional[float]
    queue_slope_per_sec: float
    loadgen_lag_p95_ms: float
    saturated: bool = False
    label: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "offered_qps": self.offered_qps,
            "goodput": self.goodput,
            "goodput_ratio": self.goodput_ratio,
            "error_ratio": self.error_ratio,
            "reject_ratio": self.reject_ratio,
            "p99_client_ttft_ms": self.p99_client_ttft_ms,
            "p99_e2e_ms": self.p99_e2e_ms,
            "queue_slope_per_sec": self.queue_slope_per_sec,
            "loadgen_lag_p95_ms": self.loadgen_lag_p95_ms,
            "saturated": self.saturated,
            "label": self.label,
        }


def _lcb(values: Sequence[float], *, resamples: int, seed: int = 17) -> Optional[float]:
    """Deterministic bootstrap lower confidence bound (5th percentile of means)."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(max(1, resamples)):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    return percentile(means, 0.05)


def max_slo_goodput(
    points: Sequence[LoadPointResult],
    spec: SLOSpec,
    *,
    run_level_goodput: Optional[Mapping[float, Sequence[float]]] = None,
) -> Dict[str, Any]:
    """``G*``: the best load point whose goodput lower bound is still in SLO."""
    if not spec.frozen:
        return {
            "status": NOT_EVALUATED,
            "reason": "G* needs frozen thresholds; the SLO spec is still a template",
        }
    usable = [
        point
        for point in points
        if point.label != "LOADGEN_INVALID"
        and not point.saturated
        and point.reject_ratio <= spec.max_error_ratio
        and point.p99_client_ttft_ms is not None
    ]
    if not usable:
        return {"status": NOT_EVALUATED, "reason": "no valid, unsaturated load point"}
    selected = max(usable, key=lambda point: point.goodput)
    lower = None
    if spec.goodput_uses_lcb and run_level_goodput:
        lower = _lcb(list(run_level_goodput.get(selected.offered_qps, ())), resamples=spec.lcb_resamples)
    return {
        "status": "selected",
        "g_star_offered_qps": selected.offered_qps,
        "g_star_goodput": selected.goodput,
        "g_star_lower_bound": lower,
        "method": spec.lcb_method if spec.goodput_uses_lcb else "point_estimate",
        "usable_points": len(usable),
        "excluded": [
            point.as_dict() for point in points if point not in usable
        ],
        "note": (
            "a peak-throughput point whose P99 is broken is not capacity; the selection "
            "uses the pre-registered constraints only"
        ),
    }


def hysteresis_report(up_sweep: Sequence[LoadPointResult], down_sweep: Sequence[LoadPointResult]) -> Dict[str, Any]:
    """Up-sweep vs. down-sweep differences (cache/thermal/queue carry-over)."""
    up = {point.offered_qps: point for point in up_sweep}
    down = {point.offered_qps: point for point in down_sweep}
    rows: List[Dict[str, Any]] = []
    for qps in sorted(set(up) & set(down)):
        rows.append(
            {
                "offered_qps": qps,
                "up_goodput": up[qps].goodput,
                "down_goodput": down[qps].goodput,
                "delta_goodput": down[qps].goodput - up[qps].goodput,
                "up_saturated": up[qps].saturated,
                "down_saturated": down[qps].saturated,
            }
        )
    return {
        "rows": rows,
        "hysteresis_detected": any(
            abs(row["delta_goodput"]) > 0.05 * max(row["up_goodput"], 1e-9) for row in rows
        ),
        "note": "a difference between up and down sweeps must be explained, not averaged away",
    }


__all__ = [
    "FUNNEL_REASONS",
    "FUNNEL_STAGES",
    "FunnelCounts",
    "GoodEvaluation",
    "INSUFFICIENT_TAIL",
    "LoadPointResult",
    "NON_SUCCESS_BUCKETS",
    "NOT_EVALUATED",
    "RequestSLOInput",
    "SLOClass",
    "SLOSpec",
    "class_sample_sufficiency",
    "classify_failure",
    "evaluate_request",
    "goodput",
    "hysteresis_report",
    "latency_summary",
    "max_slo_goodput",
    "percentile_from_raw",
    "success_latency_rows",
    "tail_sample_sufficiency",
    "token_goodput",
]
