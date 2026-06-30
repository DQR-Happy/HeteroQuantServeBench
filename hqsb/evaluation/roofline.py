"""E12-05: hierarchical Roofline / Amdahl models, prediction errors and residuals.

The experiment's contract is "prediction → measurement → residual → validation",
not "draw a roof and drop points on it".  The module therefore:

* keeps four *separate* roofs (theoretical vendor, theoretically derived,
  sustainable micro, achieved workload) and refuses to make a primary prediction
  from a theoretical peak;
* keeps ``logical`` / ``expected`` / ``measured`` traffic apart — mixing them is
  how a cache-friendly kernel gets classified as bandwidth-bound;
* requires calibration and validation cells to be disjoint and consumes the
  validation split exactly once;
* stores every unresolved residual instead of writing a story, and labels a
  bottleneck only with an evidence vector plus a confidence.

Nothing here runs a benchmark or fits anything to real data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.evaluation.identity import stable_id
from hqsb.evaluation.layers import amdahl_upper_bound
from hqsb.evaluation.records import TABLE_SCHEMAS

EXPERIMENT_ID = "E12-05"
TITLE = "分层 Roofline、Amdahl、容量/互联模型与预测误差归因"
CLAIM_BOUNDARY = (
    "本实验通过证明差异可以被可检验模型解释到声明程度，"
    "不证明 Roofline 能完全预测复杂 LLM serving；predicted 与 measured 永不混列。"
)

PEAK_EVIDENCE_LEVELS: Tuple[str, ...] = (
    "THEORETICAL_VENDOR",
    "THEORETICAL_DERIVED",
    "SUSTAINABLE_MICRO",
    "ACHIEVED_WORKLOAD",
)

#: A primary prediction needs a measured roof; theoretical values only explain
#: potential.
ROOF_PRIORITY: Mapping[str, int] = {name: index for index, name in enumerate(PEAK_EVIDENCE_LEVELS)}
PREDICTION_ROOF_LEVELS: Tuple[str, ...] = ("SUSTAINABLE_MICRO", "ACHIEVED_WORKLOAD")

MEMORY_LEVELS: Tuple[str, ...] = ("L1", "L2", "HBM", "DDR", "PCIE", "NVLINK", "XGMI", "HCCS", "NETWORK")

DEVICE_LOCAL_LEVELS: Tuple[str, ...] = ("L1", "L2", "HBM")

BOTTLENECK_CLASSES: Tuple[str, ...] = (
    "COMPUTE_THROUGHPUT",
    "HBM_OR_DDR_BANDWIDTH",
    "CACHE_OR_LOCAL_MEMORY",
    "LATENCY_DEPENDENCY",
    "LAUNCH_OR_HOST_RUNTIME",
    "LAYOUT_CONVERSION_OR_COPY",
    "LOW_OCCUPANCY_OR_TAIL_WASTE",
    "REGISTER_SPILL_OR_RESOURCE",
    "KV_CAPACITY_OR_MEMORY_PRESSURE",
    "INTERCONNECT_OR_COLLECTIVE",
    "LOAD_IMBALANCE_OR_STRAGGLER",
    "QUEUE_OR_SCHEDULER",
    "POWER_THERMAL_CLOCK",
    "MEASUREMENT_OR_MODEL_MISSPECIFICATION",
    "MIXED_OR_UNRESOLVED",
)

RESIDUAL_CLASSES: Tuple[str, ...] = (
    "kernel",
    "layout",
    "runtime",
    "launch",
    "interconnect",
    "capacity",
    "queue",
    "power",
    "measurement_or_model_misspecification",
)

RESIDUAL_STATUSES: Tuple[str, ...] = ("EXPLAINED", "PARTIALLY_EXPLAINED", "UNEXPLAINED", "NOT_APPLICABLE")


def is_device_local(level: str) -> bool:
    if level not in MEMORY_LEVELS:
        raise ConfigError(f"unknown memory level {level!r}")
    return level in DEVICE_LOCAL_LEVELS


def assert_roof_usable_for_prediction(level: str) -> None:
    if level not in PEAK_EVIDENCE_LEVELS:
        raise ConfigError(f"unknown peak evidence level {level!r}")
    if level not in PREDICTION_ROOF_LEVELS:
        raise ConfigError(
            f"roof level {level!r} may explain hardware potential but must not carry a primary "
            "prediction: use a sustainable micro or achieved-workload roof"
        )


# ── operations and traffic taxonomies ─────────────────────────────────────


@dataclass(frozen=True)
class OperationDefinition:
    semantic_op: str
    formula: str
    dtype_note: str


OPERATION_DEFINITIONS: Tuple[OperationDefinition, ...] = (
    OperationDefinition("gemm", "2 * M * N * K", "dtype and accumulation determine the instruction path"),
    OperationDefinition("reduction", "N reads, 1 write", "accumulation dtype must be declared"),
    OperationDefinition("elementwise", "N reads, N writes", "vector width and dtype matter"),
    OperationDefinition("attention", "2 * B * H * S^2 * D * 2 (fwd, causal halves)", "module counts as half when causal"),
    OperationDefinition("quant_dequant", "N * (pack + unpack) + N scales", "packing layout is part of the definition"),
    OperationDefinition("collective", "message_bytes moved across ranks", "algorithm (ring/tree) changes the byte count"),
)

OPS_UNIT_NOTE = (
    "'TOPS' is not comparable across operation semantics: a reduction TOPS and a GEMM TOPS describe "
    "different work"
)


def useful_operations(
    semantic_op: str,
    *,
    dims: Mapping[str, float],
    dtype: str = "fp16",
) -> Dict[str, Any]:
    """Declared useful operations for one semantic op (definition-tagged)."""
    known = {row.semantic_op for row in OPERATION_DEFINITIONS}
    if semantic_op not in known:
        raise ConfigError(f"unknown semantic op {semantic_op!r}; declare its useful-operations formula first")
    required: Mapping[str, Tuple[str, ...]] = {
        "gemm": ("M", "N", "K"),
        "reduction": ("N",),
        "elementwise": ("N",),
        "attention": ("B", "H", "S", "D"),
        "quant_dequant": ("N",),
        "collective": ("message_bytes",),
    }
    missing = [name for name in required[semantic_op] if name not in dims]
    if missing:
        raise ConfigError(f"{semantic_op} needs dimensions {missing}")
    if semantic_op == "gemm":
        ops = 2.0 * dims["M"] * dims["N"] * dims["K"]
    elif semantic_op == "reduction":
        ops = float(dims["N"])
    elif semantic_op == "elementwise":
        ops = float(dims["N"])
    elif semantic_op == "attention":
        ops = 2.0 * dims["B"] * dims["H"] * dims["S"] * dims["S"] * dims["D"]
    elif semantic_op == "quant_dequant":
        ops = 2.0 * float(dims["N"])
    else:
        ops = float(dims["message_bytes"])
    return {
        "semantic_op": semantic_op,
        "dtype": dtype,
        "useful_operations": ops,
        "definition": f"{semantic_op}: {next(row.formula for row in OPERATION_DEFINITIONS if row.semantic_op == semantic_op)}",
        "dtype_note": next(row.dtype_note for row in OPERATION_DEFINITIONS if row.semantic_op == semantic_op),
    }


def logical_bytes(semantic_op: str, *, dims: Mapping[str, float], dtype_bytes: float = 2.0) -> Dict[str, Any]:
    """Algorithm-minimum traffic (what the maths needs, not what the kernel does)."""
    if semantic_op == "gemm":
        traffic = dtype_bytes * (dims["M"] * dims["K"] + dims["K"] * dims["N"] + dims["M"] * dims["N"])
    elif semantic_op in ("reduction", "elementwise"):
        traffic = dtype_bytes * dims["N"] * 2.0
    elif semantic_op == "attention":
        traffic = dtype_bytes * dims["B"] * dims["H"] * dims["S"] * dims["D"] * 4.0
    elif semantic_op == "quant_dequant":
        traffic = dtype_bytes * dims["N"]
    else:
        traffic = float(dims["message_bytes"])
    return {"traffic_kind": "logical", "bytes": float(traffic), "definition": "algorithm minimum"}


def expected_bytes(
    semantic_op: str,
    *,
    dims: Mapping[str, float],
    dtype_bytes: float = 2.0,
    layout_conversions: int = 0,
    dequant_passes: int = 0,
    intermediate_copies: int = 0,
) -> Dict[str, Any]:
    """Compiler/layout-expected traffic (adds the conversions the implementation needs)."""
    base = logical_bytes(semantic_op, dims=dims, dtype_bytes=dtype_bytes)["bytes"]
    fixed = dtype_bytes * dims.get("N", dims.get("M", 1.0)) * (
        layout_conversions + dequant_passes + intermediate_copies
    )
    return {
        "traffic_kind": "expected",
        "bytes": base + fixed,
        "conversions": layout_conversions + dequant_passes + intermediate_copies,
        "definition": "logical traffic plus declared layout/cast/pack/copy passes",
    }


def measured_bytes(
    *,
    counter_value: float,
    counter_source: str,
    counter_semantics: str,
    level: str,
) -> Dict[str, Any]:
    """Profiler traffic; the counter's source and level are part of the datum."""
    if level not in MEMORY_LEVELS:
        raise ConfigError(f"unknown memory level {level!r}")
    if not counter_source or not counter_semantics:
        raise ConfigError(
            "measured traffic must record the counter source and semantics: vendor counters are "
            "not interchangeable"
        )
    return {
        "traffic_kind": "measured",
        "bytes": float(counter_value),
        "counter_source": counter_source,
        "counter_semantics": counter_semantics,
        "level": level,
        "definition": "profiler counter value at the declared level",
    }


def assert_bytes_not_mixed(*traffic_rows: Mapping[str, Any]) -> None:
    kinds = {str(row.get("traffic_kind", "")) for row in traffic_rows}
    if len(kinds - {"logical", "expected", "measured"}) > 0:
        raise ConfigError(f"unknown traffic kind(s) {sorted(kinds)}")
    if len(kinds) > 1 and any(row.get("mixed_into_one_field") for row in traffic_rows):
        raise ConfigError(
            "logical/expected/measured traffic must stay in separate fields; a single combined "
            "'bytes' value hides whether cache reuse was real"
        )


def arithmetic_intensity(ops: float, bytes_: float, *, flavor: str) -> Dict[str, Any]:
    if flavor not in ("logical", "measured"):
        raise ConfigError(f"arithmetic intensity flavor must be logical or measured, got {flavor!r}")
    if bytes_ <= 0:
        raise ConfigError("traffic must be positive to compute arithmetic intensity")
    return {"flavor": flavor, "arithmetic_intensity": float(ops) / float(bytes_)}


def roofline_bound(
    *, arithmetic_intensity_value: float, compute_roof: "Roof", bandwidth_roof: "Roof"
) -> Dict[str, Any]:
    """``min(P_compute, AI x BW)`` with both roofs required to be comparable."""
    if compute_roof.kind != "compute" or bandwidth_roof.kind != "bandwidth":
        raise ConfigError("roofline_bound needs one compute roof and one bandwidth roof")
    if compute_roof.unit != "op/s" or bandwidth_roof.unit != "byte/s":
        raise ConfigError(
            f"roof units must be op/s and byte/s (got {compute_roof.unit!r}/{bandwidth_roof.unit!r})"
        )
    compute = compute_roof.value
    memory = arithmetic_intensity_value * bandwidth_roof.value
    return {
        "bound": min(compute, memory),
        "bound_kind": "compute" if compute <= memory else "bandwidth",
        "compute_roof_id": compute_roof.roof_id,
        "bandwidth_roof_id": bandwidth_roof.roof_id,
    }


def predicted_latency(*, useful_ops: float, bound_ops_per_s: float) -> float:
    if bound_ops_per_s <= 0:
        raise ConfigError("the bound must be positive")
    return float(useful_ops) / float(bound_ops_per_s)


# ── roofs ─────────────────────────────────────────────────────────────────


@dataclass
class Roof:
    roof_id: str
    kind: str
    dtype: str
    value: float
    unit: str
    evidence_level: str
    level: str = "HBM"
    interval_low: Optional[float] = None
    interval_high: Optional[float] = None
    source_id: str = ""
    measured_at: str = ""
    conditions: Mapping[str, Any] = field(default_factory=dict)
    assumptions: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.kind not in ("compute", "bandwidth", "launch", "interconnect"):
            problems.append(f"unknown roof kind {self.kind!r}")
        if self.evidence_level not in PEAK_EVIDENCE_LEVELS:
            problems.append(f"unknown peak evidence level {self.evidence_level!r}")
        if self.value <= 0:
            problems.append("a roof must be positive")
        if self.kind == "bandwidth" and self.level not in MEMORY_LEVELS:
            problems.append(f"unknown memory level {self.level!r}")
        if self.evidence_level == "THEORETICAL_VENDOR" and not self.assumptions:
            problems.append("a vendor peak needs its assumptions (dtype, sparsity, frequency)")
        if self.evidence_level == "SUSTAINABLE_MICRO" and not self.conditions:
            problems.append("a sustainable roof needs its power/clock/thermal conditions")
        if self.evidence_level == "SUSTAINABLE_MICRO" and self.interval_low is None:
            problems.append("a sustainable roof needs an interval, not a single number")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "roof_id": self.roof_id,
            "kind": self.kind,
            "dtype": self.dtype,
            "level": self.level,
            "value": self.value,
            "unit": self.unit,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "evidence_level": self.evidence_level,
            "source_id": self.source_id,
            "measured_at": self.measured_at,
            "conditions": dict(sorted(self.conditions.items())),
            "assumptions": list(self.assumptions),
        }


def theoretical_roof(
    *, kind: str, dtype: str, value: float, unit: str, source_id: str, assumptions: Sequence[str], level: str = "HBM"
) -> Roof:
    if not assumptions:
        raise ConfigError("a theoretical roof without assumptions is a marketing number")
    roof_id = stable_id("roof", {"kind": kind, "dtype": dtype, "value": value, "source": source_id})
    return Roof(
        roof_id=roof_id,
        kind=kind,
        dtype=dtype,
        value=value,
        unit=unit,
        evidence_level="THEORETICAL_VENDOR",
        level=level,
        source_id=source_id,
        assumptions=tuple(assumptions),
    )


def sustainable_roof(
    *,
    kind: str,
    dtype: str,
    value: float,
    unit: str,
    interval_low: float,
    interval_high: float,
    conditions: Mapping[str, Any],
    measured_at: str = "",
    level: str = "HBM",
) -> Roof:
    roof_id = stable_id("roof", {"kind": kind, "dtype": dtype, "value": value, "at": measured_at})
    return Roof(
        roof_id=roof_id,
        kind=kind,
        dtype=dtype,
        value=value,
        unit=unit,
        evidence_level="SUSTAINABLE_MICRO",
        level=level,
        interval_low=interval_low,
        interval_high=interval_high,
        measured_at=measured_at,
        conditions=dict(conditions),
    )


# ── roofline points and capacity ──────────────────────────────────────────


@dataclass
class RooflinePoint:
    point_id: str
    cell_id: str
    ai_logical: float
    ai_measured: Optional[float]
    compute_roof_id: str
    bandwidth_roof_ids: Tuple[str, ...]
    bound_latency_ns: Optional[float]
    measured_latency_ns: Optional[float]
    utilization_theoretical: Optional[float]
    utilization_sustainable: Optional[float]
    actual_backend: str
    quality_status: str = "not_run"
    comparability_status: str = ""
    stability_status: str = "unknown"
    evidence_refs: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.ai_logical <= 0:
            problems.append("arithmetic intensity must be positive")
        if self.ai_measured is not None and self.ai_measured <= 0:
            problems.append("measured arithmetic intensity must be positive when present")
        if not self.actual_backend:
            problems.append("a roofline point needs its actual backend")
        if self.utilization_theoretical is not None and self.utilization_theoretical < 0:
            problems.append("utilization cannot be negative")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "cell_id": self.cell_id,
            "ai_logical": self.ai_logical,
            "ai_measured": self.ai_measured,
            "compute_roof_id": self.compute_roof_id,
            "bandwidth_roof_ids": list(self.bandwidth_roof_ids),
            "bound_latency_ns": self.bound_latency_ns,
            "measured_latency_ns": self.measured_latency_ns,
            "utilization_theoretical": self.utilization_theoretical,
            "utilization_sustainable": self.utilization_sustainable,
            "actual_backend": self.actual_backend,
            "quality_status": self.quality_status,
            "comparability_status": self.comparability_status,
            "stability_status": self.stability_status,
            "evidence_refs": list(self.evidence_refs),
        }


def place_point(
    *,
    cell_id: str,
    ai_logical: float,
    ai_measured: Optional[float],
    compute_roof: Roof,
    bandwidth_roof: Roof,
    measured_latency_ns: Optional[float],
    useful_ops: float,
    actual_backend: str,
    quality_status: str,
    comparability_status: str = "",
    stability_status: str = "unknown",
    theoretical_compute_roof: Optional[Roof] = None,
    evidence_refs: Sequence[str] = (),
) -> RooflinePoint:
    """Place one workload point with both utilizations (theoretical/sustainable)."""
    bound = roofline_bound(
        arithmetic_intensity_value=ai_logical, compute_roof=compute_roof, bandwidth_roof=bandwidth_roof
    )
    bound_latency = predicted_latency(useful_ops=useful_ops, bound_ops_per_s=bound["bound"])
    utilization_sustainable = (
        bound_latency / measured_latency_ns if measured_latency_ns else None
    )
    utilization_theoretical = None
    if theoretical_compute_roof is not None and measured_latency_ns:
        theoretical_bound = roofline_bound(
            arithmetic_intensity_value=ai_logical,
            compute_roof=theoretical_compute_roof,
            bandwidth_roof=bandwidth_roof,
        )
        utilization_theoretical = (
            predicted_latency(useful_ops=useful_ops, bound_ops_per_s=theoretical_bound["bound"])
            / measured_latency_ns
        )
    point = RooflinePoint(
        point_id=stable_id("point", {"cell": cell_id, "ai": ai_logical}),
        cell_id=cell_id,
        ai_logical=ai_logical,
        ai_measured=ai_measured,
        compute_roof_id=compute_roof.roof_id,
        bandwidth_roof_ids=(bandwidth_roof.roof_id,),
        bound_latency_ns=bound_latency,
        measured_latency_ns=measured_latency_ns,
        utilization_theoretical=utilization_theoretical,
        utilization_sustainable=utilization_sustainable,
        actual_backend=actual_backend,
        quality_status=quality_status,
        comparability_status=comparability_status,
        stability_status=stability_status,
        evidence_refs=tuple(evidence_refs),
    )
    problems = point.validate()
    if problems:
        raise ConfigError("invalid roofline point: " + "; ".join(problems))
    return point


def point_visible_in_main_figure(point: RooflinePoint) -> bool:
    """Only quality/comparability/stability-passing points enter the main figure."""
    return (
        point.quality_status == "pass"
        and point.comparability_status in ("", "COMPARABLE")
        and point.stability_status in ("stable", "conditional", "unknown")
    )


@dataclass
class CapacityModel:
    capacity_id: str
    candidate_id: str
    weights_bytes: int
    kv_bytes: int
    workspace_bytes: int
    allocator_overhead_bytes: int
    comm_bytes: int
    margin_bytes: int

    def predicted_peak_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.kv_bytes
            + self.workspace_bytes
            + self.allocator_overhead_bytes
            + self.comm_bytes
            + self.margin_bytes
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "capacity_id": self.capacity_id,
            "candidate_id": self.candidate_id,
            "weights_bytes": self.weights_bytes,
            "kv_bytes": self.kv_bytes,
            "workspace_bytes": self.workspace_bytes,
            "allocator_overhead_bytes": self.allocator_overhead_bytes,
            "comm_bytes": self.comm_bytes,
            "margin_bytes": self.margin_bytes,
            "predicted_peak_bytes": self.predicted_peak_bytes(),
        }


def predict_capacity(
    *,
    candidate_id: str,
    weights_bytes: int,
    kv_bytes: int,
    workspace_bytes: int = 0,
    allocator_overhead_bytes: int = 0,
    comm_bytes: int = 0,
    margin_bytes: int = 0,
) -> CapacityModel:
    if weights_bytes <= 0:
        raise ConfigError("a capacity model needs the weight footprint")
    return CapacityModel(
        capacity_id=stable_id("cap", {"candidate": candidate_id, "weights": weights_bytes, "kv": kv_bytes}),
        candidate_id=candidate_id,
        weights_bytes=weights_bytes,
        kv_bytes=kv_bytes,
        workspace_bytes=workspace_bytes,
        allocator_overhead_bytes=allocator_overhead_bytes,
        comm_bytes=comm_bytes,
        margin_bytes=margin_bytes,
    )


def capacity_error(model: CapacityModel, *, measured_peak_bytes: Optional[int]) -> Dict[str, Any]:
    if measured_peak_bytes is None:
        return {
            "capacity_id": model.capacity_id,
            "predicted_peak_bytes": model.predicted_peak_bytes(),
            "measured_peak_bytes": None,
            "error_bytes": None,
            "underestimated": None,
            "missing_reason": "MEASUREMENT_UNAVAILABLE",
        }
    error = measured_peak_bytes - model.predicted_peak_bytes()
    return {
        "capacity_id": model.capacity_id,
        "predicted_peak_bytes": model.predicted_peak_bytes(),
        "measured_peak_bytes": measured_peak_bytes,
        "error_bytes": error,
        "underestimated": error > 0,
        "advice": "increase the S13 safety margin" if error > 0 else "",
    }


# ── phase shares, Amdahl, service, distributed ────────────────────────────


def phase_shares(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    """Component shares of one phase; shares must sum to ~1 within the phase."""
    buckets: Dict[Tuple[str, str], float] = {}
    for row in rows:
        phase = str(row.get("phase", ""))
        component = str(row.get("component", ""))
        buckets[(phase, component)] = buckets.get((phase, component), 0.0) + float(row.get("time_ns", 0.0))
    totals: Dict[str, float] = {}
    for (phase, _), value in buckets.items():
        totals[phase] = totals.get(phase, 0.0) + value
    out: List[Dict[str, Any]] = []
    for (phase, component), value in sorted(buckets.items()):
        total = totals[phase]
        out.append(
            {
                "phase": phase,
                "component": component,
                "share": value / total if total else 0.0,
                "stability_status": str(next((row.get("stability_status", "unknown") for row in rows if row.get("phase") == phase), "unknown")),
            }
        )
    for phase, total in sorted(totals.items()):
        share_sum = sum(row["share"] for row in out if row["phase"] == phase)
        if abs(share_sum - 1.0) > 1e-6:
            raise ConfigError(f"phase {phase!r} shares sum to {share_sum:.4f}, not 1")
    return tuple(out)


def prefill_op_mix(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    return tuple(row for row in phase_shares(rows) if row["phase"] == "prefill")


def decode_op_mix(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    return tuple(row for row in phase_shares(rows) if row["phase"] == "decode")


def amdahl_prediction(*, fraction: float, local_speedup: float) -> Dict[str, Any]:
    return {
        "fraction": fraction,
        "local_speedup": local_speedup,
        "upper_bound": amdahl_upper_bound(fraction, local_speedup),
        "note": "a local upper bound only: queueing, batching, KV and overlap are not in it",
    }


def dispatch_adjusted_effect(
    op_effect: float, *, op_time_share: float, dispatch_hit_rate: float, graph_break_rate: float = 0.0
) -> Dict[str, Any]:
    """Project an operator effect through the *effective* coverage of the path."""
    for name, value in (
        ("op_time_share", op_time_share),
        ("dispatch_hit_rate", dispatch_hit_rate),
        ("graph_break_rate", graph_break_rate),
    ):
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"{name} must be inside [0, 1], got {value}")
    effective_share = op_time_share * dispatch_hit_rate * (1.0 - graph_break_rate)
    predicted = amdahl_upper_bound(effective_share, op_effect)
    return {
        "effective_share": effective_share,
        "predicted_effect": predicted,
        "coverage_note": "an unmapped/failed path contributes nothing to the projection",
    }


def service_capacity_prediction(
    *,
    service_time_s: Optional[float],
    batch_efficiency: Optional[float],
    kv_capacity_tokens: Optional[int],
    queue_model: str,
    assumptions: Sequence[str],
) -> Dict[str, Any]:
    missing = [
        name
        for name, value in (
            ("service_time_s", service_time_s),
            ("batch_efficiency", batch_efficiency),
            ("kv_capacity_tokens", kv_capacity_tokens),
        )
        if value in (None, 0)
    ]
    if missing:
        return {
            "status": "MEASUREMENT_UNAVAILABLE",
            "missing": missing,
            "reason": "the service capacity model needs measured service time, batch efficiency and KV capacity",
            "queue_model": queue_model,
        }
    if not assumptions:
        raise ConfigError("a queueing model without its assumptions is not auditable")
    return {
        "status": "OK",
        "predicted_goodput": float(batch_efficiency) / float(service_time_s),
        "predicted_arrival_max": float(batch_efficiency) / float(service_time_s),
        "queue_model": queue_model,
        "assumptions": list(assumptions),
        "kv_capacity_tokens": kv_capacity_tokens,
    }


def distributed_critical_path(segments: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    kinds = ("compute", "collective", "wait", "idle", "overlap")
    totals = {kind: 0.0 for kind in kinds}
    for segment in segments:
        kind = str(segment.get("kind", ""))
        if kind not in kinds:
            raise ConfigError(f"unknown critical-path segment kind {kind!r}")
        totals[kind] += float(segment.get("duration_ns", 0.0))
    accounted = totals["compute"] + totals["collective"] + totals["wait"] + totals["idle"] - totals["overlap"]
    return {
        "totals": totals,
        "accounted_ns": accounted,
        "note": "overlap is subtracted once; device count and parallel plan must accompany this row",
    }


# ── predictions and residuals ─────────────────────────────────────────────


@dataclass
class PredictionRecord:
    prediction_id: str
    model_version: str
    target_cell_id: str
    split: str
    operations_definition_id: str
    traffic_definition_id: str
    compute_roof_id: str
    bandwidth_roof_ids: Tuple[str, ...]
    predicted_value: Optional[float]
    unit: str
    interval_low: Optional[float] = None
    interval_high: Optional[float] = None
    measured_result_id: str = ""
    measured_value: Optional[float] = None
    measured_interval: Tuple[float, float] = ()
    capacity_model_id: str = ""
    interconnect_model_id: str = ""
    bottleneck_class: str = ""
    evidence_refs: Tuple[str, ...] = ()
    confidence: str = "low"
    residual_status: str = "UNEXPLAINED"
    followup_ablation_ids: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.split not in ("calibration", "validation"):
            problems.append(f"a prediction must be on the calibration or validation split, got {self.split!r}")
        if not self.model_version:
            problems.append("a prediction must cite its model version")
        if self.predicted_value is None:
            problems.append("a prediction record needs a predicted value (keep it separate from measured)")
        if self.residual_status not in RESIDUAL_STATUSES:
            problems.append(f"unknown residual status {self.residual_status!r}")
        if self.bottleneck_class and self.bottleneck_class not in BOTTLENECK_CLASSES:
            problems.append(f"unknown bottleneck class {self.bottleneck_class!r}")
        if self.residual_status == "EXPLAINED" and not self.evidence_refs:
            problems.append("an explained residual needs evidence")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "model_version": self.model_version,
            "target_cell_id": self.target_cell_id,
            "split": self.split,
            "operations_definition_id": self.operations_definition_id,
            "traffic_definition_id": self.traffic_definition_id,
            "compute_roof_id": self.compute_roof_id,
            "bandwidth_roof_ids": list(self.bandwidth_roof_ids),
            "capacity_model_id": self.capacity_model_id,
            "interconnect_model_id": self.interconnect_model_id,
            "predicted_value": self.predicted_value,
            "unit": self.unit,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "measured_result_id": self.measured_result_id,
            "measured_value": self.measured_value,
            "measured_interval": list(self.measured_interval),
            "bottleneck_class": self.bottleneck_class,
            "evidence_refs": list(self.evidence_refs),
            "confidence": self.confidence,
            "residual_status": self.residual_status,
            "followup_ablation_ids": list(self.followup_ablation_ids),
        }


def prediction_error(*, predicted: float, measured: float) -> Dict[str, Any]:
    if measured == 0:
        raise ConfigError("relative/log error is undefined for a zero measurement")
    if predicted <= 0 or measured <= 0:
        raise ConfigError("error metrics need strictly positive values (use log scale deliberately)")
    return {
        "absolute_error": predicted - measured,
        "relative_error": (predicted - measured) / measured,
        "log_error": math.log(predicted / measured),
    }


@dataclass
class CalibrationValidationSplit:
    calibration: Tuple[str, ...]
    validation: Tuple[str, ...]
    _consumed: bool = field(default=False, repr=False)

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.calibration or not self.validation:
            problems.append("both calibration and validation cells are required")
        overlap = sorted(set(self.calibration) & set(self.validation))
        if overlap:
            problems.append(f"calibration and validation overlap on {overlap}: that is leakage")
        return problems

    def consume_validation(self) -> Tuple[str, ...]:
        if self._consumed:
            raise ConfigError(
                "the validation split was already consumed: re-using it to tune the model turns "
                "validation into calibration"
            )
        self._consumed = True
        return self.validation


def assert_no_leakage(split: CalibrationValidationSplit, fitted_cell_ids: Sequence[str]) -> None:
    leaked = sorted(set(fitted_cell_ids) & set(split.validation))
    if leaked:
        raise ConfigError(f"validation cells used for fitting: {leaked}")


def classify_bottleneck(
    evidence: Mapping[str, Any], *, confidence: str = "low"
) -> Dict[str, Any]:
    """Classify a bottleneck from an *evidence vector*, never one utilization.

    Each candidate class gets a support/against/absent judgement; the verdict is
    ``MIXED_OR_UNRESOLVED`` whenever no class has supporting evidence, which is
    the honest answer for "we only looked at one counter".
    """
    if confidence not in ("low", "medium", "high"):
        raise ConfigError(f"unknown confidence {confidence!r}")
    for name in BOTTLENECK_CLASSES:
        if name not in evidence:
            continue
        judgement = str(evidence[name])
        if judgement not in ("support", "against", "absent"):
            raise ConfigError(
                f"evidence for {name!r} must be support/against/absent, got {judgement!r}"
            )
    supporting = sorted(
        name for name in BOTTLENECK_CLASSES if str(evidence.get(name, "absent")) == "support"
    )
    against = sorted(name for name in BOTTLENECK_CLASSES if str(evidence.get(name, "absent")) == "against")
    if not supporting:
        return {
            "bottleneck_class": "MIXED_OR_UNRESOLVED",
            "supporting": [],
            "against": against,
            "confidence": "low",
            "reason": "no class has supporting evidence: a single utilization percentage is not a diagnosis",
        }
    if len(supporting) > 1:
        return {
            "bottleneck_class": "MIXED_OR_UNRESOLVED" if confidence == "low" else supporting[0],
            "supporting": supporting,
            "against": against,
            "confidence": confidence,
            "reason": "several classes are supported; keep them listed rather than picking one",
        }
    return {
        "bottleneck_class": supporting[0],
        "supporting": supporting,
        "against": against,
        "confidence": confidence,
        "reason": "single supported class",
    }


class ResidualLedger:
    def __init__(self) -> None:
        self._rows: List[Dict[str, Any]] = []

    def record(
        self,
        *,
        prediction_id: str,
        residual_class: str,
        evidence_refs: Sequence[str],
        confidence: str,
        resolved: bool,
        followup_ablation_id: str = "",
    ) -> Dict[str, Any]:
        if residual_class not in RESIDUAL_CLASSES:
            raise ConfigError(f"unknown residual class {residual_class!r}")
        if confidence not in ("low", "medium", "high"):
            raise ConfigError(f"unknown confidence {confidence!r}")
        if resolved and not evidence_refs:
            raise ConfigError("an explained residual needs evidence (no storytelling)")
        row = {
            "residual_id": stable_id(
                "res", {"prediction": prediction_id, "class": residual_class, "resolved": resolved}
            ),
            "prediction_id": prediction_id,
            "class": residual_class,
            "evidence_refs": list(evidence_refs),
            "confidence": confidence,
            "resolved": resolved,
            "followup_ablation_id": followup_ablation_id,
        }
        self._rows.append(row)
        return row

    def rows(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(self._rows)

    def unresolved(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(row for row in self._rows if not row["resolved"])


def classify_residual(
    *, residual_class: str, evidence_refs: Sequence[str], supports_class: bool
) -> Dict[str, Any]:
    """Residual classification never upgrades "no evidence" to a refutation."""
    if residual_class not in RESIDUAL_CLASSES:
        raise ConfigError(f"unknown residual class {residual_class!r}")
    if not evidence_refs:
        return {"status": "INCONCLUSIVE", "reason": "no evidence: the residual stays unexplained"}
    return {
        "status": "EXPLAINED" if supports_class else "PARTIALLY_EXPLAINED",
        "reason": "evidence supports the class" if supports_class else "evidence does not isolate the class",
    }


def ablation_plan(*, cell_id: str, changed_factor: str, expected_shift: str) -> Dict[str, Any]:
    return {
        "ablation_id": stable_id("abl", {"cell": cell_id, "factor": changed_factor}),
        "cell_id": cell_id,
        "changed_factor": changed_factor,
        "expected_shift": expected_shift,
        "single_factor": True,
        "alternative_explanations": (),
    }


def evaluate_ablation(
    *,
    ablation: Mapping[str, Any],
    changed_factors: Sequence[str],
    observed_shift: str,
    alternative_explanations: Sequence[str],
) -> Dict[str, Any]:
    if len(changed_factors) != 1:
        raise ConfigError(
            f"an ablation must change exactly one factor, got {list(changed_factors)}: with two changes "
            "the observed shift cannot be attributed"
        )
    if changed_factors[0] != ablation.get("changed_factor"):
        raise ConfigError("the changed factor does not match the planned ablation")
    if not alternative_explanations:
        raise ConfigError("every ablation must list the alternative explanations it did not exclude")
    matches_expectation = observed_shift == ablation.get("expected_shift")
    return {
        "ablation_id": ablation.get("ablation_id", ""),
        "changed_factor": changed_factors[0],
        "expected_shift": ablation.get("expected_shift", ""),
        "observed_shift": observed_shift,
        "matches_expectation": matches_expectation,
        "alternative_explanations": list(alternative_explanations),
        "status": "SUPPORTS_MODEL" if matches_expectation else "CONTRADICTS_MODEL",
    }


def propagate_uncertainty(
    inputs: Mapping[str, Tuple[float, float]], *, model_fn: Any
) -> Dict[str, Any]:
    """Interval propagation: evaluate the model at the interval corners."""
    if not inputs:
        raise ConfigError("uncertainty propagation needs at least one interval input")
    for name, (low, high) in inputs.items():
        if low > high:
            raise ConfigError(f"interval for {name!r} is inverted")
    lows = {name: interval[0] for name, interval in inputs.items()}
    highs = {name: interval[1] for name, interval in inputs.items()}
    return {
        "interval_low": float(model_fn(lows)),
        "interval_high": float(model_fn(highs)),
        "inputs": {name: list(interval) for name, interval in sorted(inputs.items())},
        "method": "corner_propagation",
    }


def figure_contract() -> Dict[str, Any]:
    """What a Roofline figure must carry for the experiment to pass."""
    return {
        "required_metadata": (
            "precision_path",
            "memory_level",
            "roof_source",
            "roof_evidence_level",
            "theoretical_vs_sustainable",
            "error_bars",
            "result_ids",
            "prediction_error",
        ),
        "forbidden": ("points failing quality/comparability/stability in the main figure",),
    }


def validate_figure_metadata(meta: Mapping[str, Any]) -> Tuple[str, ...]:
    required = figure_contract()["required_metadata"]
    problems = [f"figure metadata is missing {name!r}" for name in required if meta.get(name) in (None, "", (), [])]
    if meta.get("roof_evidence_level") == "THEORETICAL_VENDOR" and meta.get("theoretical_vs_sustainable") == "sustainable":
        problems.append("a vendor peak may not be labelled as the sustainable roof")
    return tuple(problems)


def transferability_check(
    errors_by_group: Mapping[str, float], *, leave_out: str = "shape"
) -> Dict[str, Any]:
    if leave_out not in ("shape", "workload", "device"):
        raise ConfigError(f"unknown leave-out axis {leave_out!r}")
    if not errors_by_group:
        raise ConfigError("transferability needs per-group errors")
    worst = max(errors_by_group.items(), key=lambda item: abs(item[1]))
    return {
        "leave_out": leave_out,
        "groups": dict(sorted(errors_by_group.items())),
        "worst_group": worst[0],
        "worst_error": worst[1],
        "scope_limit": f"the model is only claimed inside the tested {leave_out} domain",
    }


def table_schemas() -> Mapping[str, Tuple[str, ...]]:
    return {name: TABLE_SCHEMAS[name] for name in sorted(TABLE_SCHEMAS) if name.startswith("e12_05.")}


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only callability check; never an experiment."""
    compute = sustainable_roof(
        kind="compute",
        dtype="fp16",
        value=1.0e14,
        unit="op/s",
        interval_low=0.9e14,
        interval_high=1.05e14,
        conditions={"power_cap_w": 250, "clock_mhz": 1800},
    )
    bandwidth = sustainable_roof(
        kind="bandwidth",
        dtype="fp16",
        value=1.0e12,
        unit="byte/s",
        interval_low=0.95e12,
        interval_high=1.02e12,
        conditions={"level": "HBM"},
    )
    theoretical = theoretical_roof(
        kind="compute",
        dtype="fp16",
        value=2.0e14,
        unit="op/s",
        source_id="vendor-datasheet",
        assumptions=("dense", "boost clock", "no sparsity"),
    )
    ops = useful_operations("gemm", dims={"M": 512, "N": 512, "K": 512})
    point = place_point(
        cell_id="cell-1",
        ai_logical=50.0,
        ai_measured=40.0,
        compute_roof=compute,
        bandwidth_roof=bandwidth,
        measured_latency_ns=5.0e5,
        useful_ops=ops["useful_operations"],
        actual_backend="cuda",
        quality_status="pass",
        comparability_status="COMPARABLE",
        stability_status="stable",
        theoretical_compute_roof=theoretical,
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "theoretical_roof_rejected_for_prediction": _expect_config_error(
            lambda: assert_roof_usable_for_prediction("THEORETICAL_VENDOR")
        ),
        "bound_kind": "bandwidth" if point.ai_logical * bandwidth.value < compute.value else "compute",
        "point_has_two_utilizations": point.utilization_sustainable is not None
        and point.utilization_theoretical is not None,
        "visible_in_main_figure": point_visible_in_main_figure(point),
        "phase_share_sum_guard": _expect_config_error(
            lambda: phase_shares(
                [
                    {"phase": "prefill", "component": "gemm", "time_ns": 1.0},
                    {"phase": "prefill", "component": "attn", "time_ns": 1.0, "stability_status": "unknown"},
                ]
            )
        )
        is False,
        "ablation_needs_single_factor": _expect_config_error(
            lambda: evaluate_ablation(
                ablation=ablation_plan(cell_id="c", changed_factor="tile", expected_shift="+5%"),
                changed_factors=["tile", "layout"],
                observed_shift="+5%",
                alternative_explanations=["clock drift"],
            )
        ),
    }


def _expect_config_error(callable_: Any) -> bool:
    try:
        callable_()
    except ConfigError:
        return True
    return False


PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结建模问题和验证集", ("roofline:CalibrationValidationSplit", "roofline:PredictionRecord")),
    (2, "绑定稳定 benchmark evidence", ("repeatability:stability_verdicts", "roofline:RooflinePoint")),
    (3, "盘点硬件理论资源", ("roofline:theoretical_roof", "roofline:Roof.assumptions")),
    (4, "定义 operations taxonomy", ("roofline:useful_operations", "roofline:OPERATION_DEFINITIONS")),
    (5, "定义 bytes taxonomy", ("roofline:logical_bytes", "roofline:expected_bytes", "roofline:measured_bytes")),
    (6, "验证 shape/layout/precision path", ("roofline:assert_bytes_not_mixed", "capability:CapabilityEvidence")),
    (7, "测可持续 compute roof", ("roofline:sustainable_roof", "roofline:Roof.conditions")),
    (8, "测分层内存 roof", ("roofline:Roof", "roofline:MEMORY_LEVELS")),
    (9, "测 launch/runtime floor", ("roofline:Roof", "roofline:predicted_latency")),
    (10, "测 interconnect roof", ("roofline:Roof", "roofline:distributed_critical_path")),
    (11, "建立容量模型", ("roofline:predict_capacity", "roofline:capacity_error")),
    (12, "构建 operator arithmetic intensity", ("roofline:arithmetic_intensity", "roofline:assert_bytes_not_mixed")),
    (13, "生成 theoretical roofline", ("roofline:theoretical_roof", "roofline:assert_roof_usable_for_prediction")),
    (14, "生成 sustainable roofline", ("roofline:sustainable_roof", "roofline:roofline_bound")),
    (15, "投放 workload points", ("roofline:place_point", "roofline:point_visible_in_main_figure")),
    (16, "计算 operator predictions", ("roofline:predicted_latency", "roofline:PredictionRecord")),
    (17, "分析小 kernel/依赖限制", ("roofline:classify_bottleneck", "roofline:BOTTLENECK_CLASSES")),
    (18, "分析 layout/copy/conversion", ("roofline:expected_bytes", "roofline:measured_bytes")),
    (19, "分析 prefill op mix", ("roofline:prefill_op_mix", "roofline:phase_shares")),
    (20, "分析 decode op mix", ("roofline:decode_op_mix", "roofline:phase_shares")),
    (21, "构建 phase Amdahl 上界", ("roofline:amdahl_prediction", "layers:amdahl_upper_bound")),
    (22, "纳入 dispatch/fallback", ("roofline:dispatch_adjusted_effect", "capability:CapabilityEvidence.fallback_only")),
    (23, "构建 service 容量模型", ("roofline:service_capacity_prediction", "layers:SloSpec")),
    (24, "构建 distributed critical path", ("roofline:distributed_critical_path", "layers:ESTIMANDS")),
    (25, "计算理论与可持续 utilization", ("roofline:RooflinePoint", "roofline:place_point")),
    (26, "在 validation cells 上预测", ("roofline:CalibrationValidationSplit.consume_validation", "roofline:prediction_error")),
    (27, "构建残差 ledger", ("roofline:ResidualLedger", "roofline:classify_residual")),
    (28, "选择高残差 case", ("roofline:ResidualLedger.unresolved", "roofline:RESIDUAL_CLASSES")),
    (29, "执行 kernel/layout ablation", ("roofline:ablation_plan", "roofline:evaluate_ablation")),
    (30, "执行 size/regime sweep", ("roofline:roofline_bound", "roofline:RooflinePoint")),
    (31, "执行 power/clock 敏感性", ("roofline:Roof", "roofline:sustainable_roof")),
    (32, "执行互联/并行消融", ("roofline:distributed_critical_path", "roofline:evaluate_ablation")),
    (33, "传播测量不确定性", ("roofline:propagate_uncertainty", "repeatability:bootstrap_ci")),
    (34, "评估模型可迁移性", ("roofline:transferability_check", "roofline:CalibrationValidationSplit")),
    (35, "独立 confirmation profile", ("roofline:assert_no_leakage", "roofline:PredictionRecord")),
    (36, "形成 HW/SW co-design verdict", ("roofline:classify_bottleneck", "campaign:AcceptanceDecision")),
)
