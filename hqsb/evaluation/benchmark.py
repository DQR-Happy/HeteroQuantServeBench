"""E12-03: the four-layer unified replay — raw observations → normalized results.

This module owns the *data contract* of the main S12 experiment:

* ``Observation`` is the raw event (append-only, uniquely identified, carrying
  the requested **and** actual backend);
* ``NormalizedResult`` is a derived estimator with its source ids, basis,
  denominator, interval method and statuses — never a hand-copied number;
* ``validate_cross_constraints`` implements the eleven cross-cutting rules of
  ``details/S12/README.md`` §23, so an illegal row fails at write time rather
  than inside a chart;
* cold start, warmup, thermal state and run health are recorded as *separate*
  stages: picking the best of cold and steady, or starting while throttled, are
  FAIL conditions, not optimisations.

Nothing here runs a benchmark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.evaluation.identity import stable_id
from hqsb.evaluation.layers import (
    ALLOWED_NORMALIZATIONS,
    FORBIDDEN_NORMALIZATIONS,
    LAYERS,
    LAYER_REQUIRED_FIELDS,
    SloSpec,
    TokenAccounting,
    CrossLayerConversion,
    goodput,
    validate_layer_metric,
    validate_scenario,
)
from hqsb.evaluation.records import (
    COMPARABILITY_STATES,
    MISSINGNESS_CODES,
    is_missing_status,
)

EXPERIMENT_ID = "E12-03"
TITLE = "Operator、Model-core、Service、Distributed 四层统一重放"
CLAIM_BOUNDARY = (
    "本实验通过只证明统一协议下的数据被正确生产，"
    "不证明结果已跨日稳定，也不单独给出硬件推荐。"
)

OBSERVATION_STATUS_OK = "OK"

#: Minimum independent repetitions for a formal S12 cell (handbook §5.4).
MIN_REPETITIONS = 3

#: The eleven cross-cutting validator constraints (§23), as data.
VALIDATOR_CONSTRAINTS: Tuple[Mapping[str, str], ...] = (
    {
        "constraint_id": "C1_status_not_ok_has_no_numeric",
        "description": "status != OK 时数值字段不得伪装成有效零",
        "scope": "observation",
    },
    {
        "constraint_id": "C2_comparable_requires_contract",
        "description": "COMPARABLE 必须引用通过的 comparison contract",
        "scope": "normalized",
    },
    {
        "constraint_id": "C3_benchmarked_requires_run_and_quality",
        "description": "BENCHMARKED capability 必须引用正式 run 和 quality status",
        "scope": "capability",
    },
    {
        "constraint_id": "C4_energy_requires_window_and_meter",
        "description": "energy 必须有同 run 时间窗口和 meter boundary",
        "scope": "energy",
    },
    {
        "constraint_id": "C5_cost_requires_validity_and_inputs",
        "description": "cost 必须有有效期、地区、币种、价格来源和利用率",
        "scope": "cost",
    },
    {
        "constraint_id": "C6_derived_metric_requires_sources_formula_unit",
        "description": "derived metric 必须列出 source IDs、公式版本和单位",
        "scope": "normalized",
    },
    {
        "constraint_id": "C7_actual_backend_required",
        "description": "actual_backend 缺失或 fallback 不明时结果不能进入正式矩阵",
        "scope": "observation",
    },
    {
        "constraint_id": "C8_quality_fail_not_in_pareto",
        "description": "quality fail 的 candidate 不能进入 Pareto",
        "scope": "normalized",
    },
    {
        "constraint_id": "C9_distributed_requires_topology",
        "description": "distributed 结果必须有 device count/topology/parallel plan",
        "scope": "observation",
    },
    {
        "constraint_id": "C10_interval_requires_method_unit_confidence",
        "description": "interval 必须带 method、replication unit 和 confidence level",
        "scope": "normalized",
    },
    {
        "constraint_id": "C11_point_and_recommendation_need_ids",
        "description": "plot/table point 必须有 normalized result ID；recommendation 必须有 profile/constraint/frontier IDs",
        "scope": "downstream",
    },
)

#: Health hints recorded per run (labels are owned by E12-04; these are strings
#: so the two experiments stay decoupled).
HEALTH_FLAG_HINTS: Tuple[str, ...] = (
    "HEALTHY_INCLUDED",
    "HEALTHY_BUT_HIGH_VARIANCE",
    "THERMAL_THROTTLE",
    "POWER_LIMIT_THROTTLE",
    "CLOCK_DEVIATION",
    "CPU_CONTENTION",
    "MEMORY_PRESSURE",
    "SWAP_ACTIVITY",
    "IO_CONTENTION",
    "NEIGHBOR_PROCESS",
    "ECC_RAS_OR_LINK_EVENT",
    "OOM_OR_RECOVERY",
    "SERVICE_BACKLOG_CONTAMINATION",
    "LOADGEN_BOTTLENECK",
    "TELEMETRY_GAP",
    "UNKNOWN_ANOMALY",
)

COLD_START_STAGES: Tuple[str, ...] = (
    "artifact_download",
    "model_load",
    "compile_or_jit",
    "autotune",
    "graph_capture",
    "warmup",
    "ready",
)


# ── raw observations ──────────────────────────────────────────────────────


@dataclass
class Observation:
    """One raw benchmark event (``BenchmarkObservation``, README §22.1)."""

    observation_id: str
    run_id: str
    sample_id: str
    timestamp_monotonic_ns: int
    candidate_id: str
    comparison_id: str
    layer: str
    scenario: str
    workload_spec_id: str
    requested_backend: str = ""
    actual_backend: str = ""
    fallback_reason: str = ""
    input_tokens: int = 0
    requested_output_tokens: int = 0
    accepted_output_tokens: int = 0
    latency_component: str = ""
    latency_ns: Optional[int] = None
    status: str = OBSERVATION_STATUS_OK
    error_code: str = ""
    memory_boundary: str = ""
    memory_bytes: Optional[int] = None
    quality_gate_id: str = ""
    quality_status: str = "not_run"
    source_artifact_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.observation_id:
            self.observation_id = stable_id(
                "obs",
                {
                    "run": self.run_id,
                    "sample": self.sample_id,
                    "candidate": self.candidate_id,
                    "layer": self.layer,
                    "metric": self.latency_component,
                },
            )

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("run_id", "sample_id", "candidate_id", "comparison_id", "workload_spec_id"):
            if not getattr(self, name):
                problems.append(f"observation is missing {name!r}")
        if self.layer not in LAYERS:
            problems.append(f"unknown layer {self.layer!r}")
        else:
            problems.extend(validate_scenario(self.layer, self.scenario))
            if self.latency_component:
                problems.extend(validate_layer_metric(self.layer, self.latency_component))
        if self.status != OBSERVATION_STATUS_OK:
            if self.latency_ns is not None or self.memory_bytes is not None:
                problems.append(
                    f"status {self.status!r} must not carry numbers: a failed run is an evidence "
                    "state, not a zero"
                )
            if self.status not in MISSINGNESS_CODES and not self.error_code:
                problems.append("a non-OK observation needs an error code or a missingness state")
        if not self.actual_backend:
            problems.append(
                "actual_backend is required: a requested backend without execution evidence "
                "cannot enter the matrix (C7)"
            )
        if (
            self.requested_backend
            and self.actual_backend
            and self.requested_backend != self.actual_backend
            and not self.fallback_reason
        ):
            problems.append("requested != actual requires a fallback reason (no silent fallback)")
        if self.layer == "distributed" and not self.scenario:
            problems.append("a distributed observation needs an explicit scaling kind")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "timestamp_monotonic_ns": self.timestamp_monotonic_ns,
            "candidate_id": self.candidate_id,
            "comparison_id": self.comparison_id,
            "layer": self.layer,
            "scenario": self.scenario,
            "workload_spec_id": self.workload_spec_id,
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "fallback_reason": self.fallback_reason,
            "input_tokens": self.input_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "accepted_output_tokens": self.accepted_output_tokens,
            "latency_component": self.latency_component,
            "latency_ns": self.latency_ns,
            "status": self.status,
            "error_code": self.error_code,
            "memory_boundary": self.memory_boundary,
            "memory_bytes": self.memory_bytes,
            "quality_gate_id": self.quality_gate_id,
            "quality_status": self.quality_status,
            "source_artifact_refs": list(self.source_artifact_refs),
        }


class ObservationStore:
    """Append-only store: rewriting raw evidence is refused, not warned about."""

    def __init__(self) -> None:
        self._rows: List[Observation] = []
        self._by_id: Dict[str, Observation] = {}

    def append(self, observation: Observation) -> str:
        problems = observation.validate()
        if problems:
            raise ConfigError("invalid observation: " + "; ".join(problems))
        existing = self._by_id.get(observation.observation_id)
        if existing is not None:
            if existing.as_dict() != observation.as_dict():
                raise ConfigError(
                    f"observation {observation.observation_id!r} already exists with different "
                    "content: raw evidence is append-only"
                )
            return observation.observation_id
        self._rows.append(observation)
        self._by_id[observation.observation_id] = observation
        return observation.observation_id

    def rows(self) -> Tuple[Observation, ...]:
        return tuple(self._rows)

    def by_run(self, run_id: str) -> Tuple[Observation, ...]:
        return tuple(row for row in self._rows if row.run_id == run_id)

    def __len__(self) -> int:
        return len(self._rows)


# ── normalized results ────────────────────────────────────────────────────

ESTIMATORS: Tuple[str, ...] = ("median", "mean", "p95", "p99")
INTERVAL_METHODS: Tuple[str, ...] = ("run_level_bootstrap", "t_interval", "none")


@dataclass
class NormalizedResult:
    """One derived metric (``NormalizedResult``, README §22.2)."""

    normalized_result_id: str
    source_observation_ids: Tuple[str, ...]
    transform_activity_id: str
    metric_name: str
    value: Optional[float]
    unit: str
    direction: str
    estimator: str
    interval_low: Optional[float] = None
    interval_high: Optional[float] = None
    interval_method: str = ""
    confidence_level: float = 0.0
    normalization_basis: str = "none"
    denominator_value: Optional[float] = None
    replication_unit: str = ""
    comparability_status: str = ""
    quality_status: str = "not_run"
    stability_status: str = "unknown"
    evidence_level: str = "L0"
    missing_reason: str = ""
    limitations: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.source_observation_ids:
            problems.append("a normalized result must list its source observation ids (C6)")
        if not self.transform_activity_id:
            problems.append("a normalized result must reference its transform activity (C6)")
        if self.normalization_basis not in ALLOWED_NORMALIZATIONS:
            problems.append(
                f"unknown/forbidden normalization basis {self.normalization_basis!r} (C6)"
            )
        if self.direction not in ("higher_is_better", "lower_is_better", "neutral"):
            problems.append(f"unknown metric direction {self.direction!r}")
        if self.estimator and self.estimator not in ESTIMATORS:
            problems.append(f"unknown estimator {self.estimator!r}")
        if is_missing_status(self.missing_reason) or self.missing_reason.startswith("NOT_"):
            if self.value is not None:
                problems.append(
                    f"missing state {self.missing_reason!r} must not carry a value (missing is not zero)"
                )
        elif self.value is None and not self.missing_reason:
            problems.append("a value is required unless a missing reason is given")
        if self.interval_low is not None or self.interval_high is not None:
            if self.interval_method not in INTERVAL_METHODS:
                problems.append(
                    f"an interval needs a declared method (got {self.interval_method!r}) (C10)"
                )
            if not self.replication_unit:
                problems.append("an interval must name its replication unit (C10)")
            if not (0.0 < self.confidence_level < 1.0):
                problems.append("an interval must carry a confidence level inside (0, 1) (C10)")
            if self.interval_low is None or self.interval_high is None:
                problems.append("both interval bounds are required when an interval is reported")
        if self.comparability_status and self.comparability_status not in COMPARABILITY_STATES:
            problems.append(f"unknown comparability status {self.comparability_status!r}")
        if self.quality_status.upper().startswith("QUALITY") and self.value is not None:
            problems.append("a quality-failed result may not be published as a value (C8)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "normalized_result_id": self.normalized_result_id,
            "source_observation_ids": list(self.source_observation_ids),
            "transform_activity_id": self.transform_activity_id,
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "direction": self.direction,
            "estimator": self.estimator,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "confidence_level": self.confidence_level,
            "normalization_basis": self.normalization_basis,
            "denominator_value": self.denominator_value,
            "replication_unit": self.replication_unit,
            "comparability_status": self.comparability_status,
            "quality_status": self.quality_status,
            "stability_status": self.stability_status,
            "evidence_level": self.evidence_level,
            "missing_reason": self.missing_reason,
            "limitations": list(self.limitations),
        }


def normalize(
    observations: Sequence[Observation],
    *,
    metric_name: str,
    estimator: str,
    basis: str,
    quality_status: str,
    comparability_status: str,
    evidence_level: str = "L2",
    denominator: Optional[float] = None,
    replication_unit: str = "run",
    confidence_level: float = 0.0,
    transform_activity_id: str = "",
) -> NormalizedResult:
    """Aggregate raw samples into one normalized result (no forbidden basis)."""
    if basis in FORBIDDEN_NORMALIZATIONS:
        raise ConfigError(
            f"normalization basis {basis!r} is forbidden by the S12 contract: it would turn a "
            "different question into a comparable-looking number"
        )
    if basis not in ALLOWED_NORMALIZATIONS:
        raise ConfigError(f"unknown normalization basis {basis!r}")
    if estimator not in ESTIMATORS:
        raise ConfigError(f"unknown estimator {estimator!r}")
    values = [
        float(row.latency_ns if row.latency_ns is not None else 0.0)
        for row in observations
        if row.status == OBSERVATION_STATUS_OK and row.latency_ns is not None
    ]
    usable = [
        row
        for row in observations
        if row.status == OBSERVATION_STATUS_OK and row.latency_ns is not None
    ]
    missing_reason = "" if usable else "NOT_RUN_PREREQUISITE"
    value = _estimate(values, estimator) if values else None
    if basis != "none" and denominator in (None, 0) and value is not None:
        raise ConfigError(
            f"basis {basis!r} needs a denominator: a per-* metric without one cannot be audited"
        )
    if denominator and value is not None:
        value = value / denominator
    return NormalizedResult(
        normalized_result_id=stable_id(
            "norm",
            {
                "metric": metric_name,
                "basis": basis,
                "estimator": estimator,
                "sources": sorted(row.observation_id for row in usable),
            },
        ),
        source_observation_ids=tuple(sorted(row.observation_id for row in observations)),
        transform_activity_id=transform_activity_id or stable_id("act", {"metric": metric_name, "basis": basis}),
        metric_name=metric_name,
        value=value,
        unit="ns",
        direction="lower_is_better",
        estimator=estimator,
        interval_low=None,
        interval_high=None,
        confidence_level=confidence_level,
        normalization_basis=basis,
        denominator_value=denominator,
        replication_unit=replication_unit,
        comparability_status=comparability_status,
        quality_status=quality_status,
        evidence_level=evidence_level,
        missing_reason=missing_reason,
        limitations=("within_campaign_preliminary: cross-day stability is E12-04's job",),
    )


def _estimate(values: Sequence[float], estimator: str) -> float:
    ordered = sorted(values)
    if estimator == "mean":
        return sum(ordered) / len(ordered)
    if estimator == "median":
        return quantile(ordered, 0.5)
    if estimator == "p95":
        return quantile(ordered, 0.95)
    if estimator == "p99":
        return quantile(ordered, 0.99)
    raise ConfigError(f"unknown estimator {estimator!r}")


def quantile(ordered_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (the deterministic R-7 definition).

    The definition is pinned here because an unpinned quantile is not
    reproducible: the median of two samples must be their midpoint, and the
    p95 of a small sample must not silently collapse to the maximum.
    """
    if not ordered_values:
        raise ConfigError("quantile needs at least one value")
    if not 0.0 < q <= 1.0:
        raise ConfigError(f"quantile must be inside (0, 1], got {q}")
    ordered = sorted(float(value) for value in ordered_values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


# ── cross-cutting validation ──────────────────────────────────────────────


def _violation(constraint_id: str, detail: str) -> Dict[str, Any]:
    return {"check_id": constraint_id, "status": "FAIL", "detail": detail}


def validate_cross_constraints(row: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    """Apply every §23 constraint that the row's shape can be checked against."""
    findings: List[Dict[str, Any]] = []
    status = str(row.get("status", ""))
    if status and status != OBSERVATION_STATUS_OK:
        for column in ("value", "raw_value", "normalized_value", "latency_ns", "memory_bytes"):
            if isinstance(row.get(column), (int, float)):
                findings.append(
                    _violation(
                        "C1_status_not_ok_has_no_numeric",
                        f"status {status!r} carries numeric {column!r}",
                    )
                )
    if row.get("comparability_status") == "COMPARABLE" and not row.get("contract_sha256"):
        findings.append(
            _violation("C2_comparable_requires_contract", "COMPARABLE without a contract reference")
        )
    if row.get("capability_level") == "BENCHMARKED" and not (
        row.get("run_id") and row.get("quality_status") == "pass"
    ):
        findings.append(
            _violation(
                "C3_benchmarked_requires_run_and_quality",
                "BENCHMARKED without a formal run and a passing quality status",
            )
        )
    if row.get("metric_family") == "energy" or row.get("energy_measurement_id"):
        if not (row.get("window_id") and row.get("meter_boundary")):
            findings.append(
                _violation(
                    "C4_energy_requires_window_and_meter",
                    "energy value without a same-run window and meter boundary",
                )
            )
    if row.get("metric_family") == "cost" or row.get("cost_result_id"):
        required = ("effective_date", "region", "currency", "price_source_ids", "utilization")
        missing = [name for name in required if row.get(name) in (None, "", [], {})]
        if missing:
            findings.append(
                _violation(
                    "C5_cost_requires_validity_and_inputs",
                    f"cost value missing {missing}",
                )
            )
    if row.get("derived") in (True, "yes"):
        required = ("source_observation_ids", "formula_version", "unit")
        missing = [name for name in required if row.get(name) in (None, "", [], {})]
        if missing:
            findings.append(
                _violation(
                    "C6_derived_metric_requires_sources_formula_unit",
                    f"derived metric missing {missing}",
                )
            )
    if not row.get("actual_backend"):
        findings.append(_violation("C7_actual_backend_required", "actual backend is unknown"))
    if (
        str(row.get("quality_status", "")).upper().startswith("QUALITY")
        and row.get("in_pareto") is True
    ):
        findings.append(
            _violation("C8_quality_fail_not_in_pareto", "quality-failed row entered the Pareto set")
        )
    if row.get("layer") == "distributed":
        missing = [
            name
            for name in ("device_count", "topology_id", "parallel_plan_id")
            if row.get(name) in (None, "", [], {})
        ]
        if missing:
            findings.append(
                _violation("C9_distributed_requires_topology", f"distributed row missing {missing}")
            )
    if row.get("interval_low") is not None or row.get("interval_high") is not None:
        missing = [
            name
            for name in ("interval_method", "replication_unit", "confidence_level")
            if row.get(name) in (None, "", [], {})
        ]
        if missing:
            findings.append(
                _violation("C10_interval_requires_method_unit_confidence", f"interval missing {missing}")
            )
    if row.get("point_id") and not row.get("normalized_result_ids"):
        findings.append(
            _violation("C11_point_and_recommendation_need_ids", "plot point without normalized result IDs")
        )
    if row.get("recommendation_id"):
        missing = [
            name
            for name in ("profile_id", "constraint_ids", "frontier_id")
            if row.get(name) in (None, "", [], {})
        ]
        if missing:
            findings.append(
                _violation(
                    "C11_point_and_recommendation_need_ids",
                    f"recommendation missing {missing}",
                )
            )
    return tuple(findings)


def validate_matrix_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate check; any violation makes the matrix unusable."""
    findings: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        findings.extend(
            dict(finding, row_index=index) for finding in validate_cross_constraints(row)
        )
    return {
        "rows": len(rows),
        "checks": [dict(row) for row in VALIDATOR_CONSTRAINTS],
        "violations": findings,
        "ok": not findings and bool(rows),
    }


def validate_layer_row(layer: str, row: Mapping[str, Any]) -> Tuple[str, ...]:
    """A layer row must carry that layer's mandatory fields (§12)."""
    if layer not in LAYERS:
        raise ConfigError(f"unknown layer {layer!r}")
    missing = [name for name in LAYER_REQUIRED_FIELDS[layer] if row.get(name) in (None, "")]
    metric = str(row.get("metric_name", ""))
    if metric:
        missing.extend(validate_layer_metric(layer, metric))
    return tuple(f"{layer}: missing {name}" for name in missing)


# ── run planning ──────────────────────────────────────────────────────────


@dataclass
class RunPlan:
    """One planned run of a cell."""

    cell_id: str
    comparison_group_id: str
    candidate_id: str
    layer: str
    scenario: str
    workload_spec_id: str
    measurement_contract_id: str
    repetition: int
    process_index: int
    run_family_id: str
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = next_run_id(self.cell_id, self.repetition, self.process_index)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "comparison_group_id": self.comparison_group_id,
            "candidate_id": self.candidate_id,
            "layer": self.layer,
            "scenario": self.scenario,
            "workload_spec_id": self.workload_spec_id,
            "measurement_contract_id": self.measurement_contract_id,
            "repetition": self.repetition,
            "process_index": self.process_index,
            "run_family_id": self.run_family_id,
            "run_id": self.run_id,
        }


def next_run_id(cell_id: str, repetition: int, process_index: int) -> str:
    return stable_id("run", {"cell": cell_id, "rep": repetition, "proc": process_index})


def plan_cells(
    *,
    comparison_group_id: str,
    candidate_id: str,
    layer: str,
    scenario: str,
    workload_spec_id: str,
    measurement_contract_id: str,
    planned_repetitions: int = MIN_REPETITIONS,
    processes: int = 1,
) -> Tuple[RunPlan, ...]:
    """Build the run matrix for one cell; fewer than three runs is refused."""
    if planned_repetitions < MIN_REPETITIONS:
        raise ConfigError(
            f"planned_repetitions={planned_repetitions} < {MIN_REPETITIONS}: the handbook requires "
            "at least three independent runs before a performance conclusion"
        )
    if layer not in LAYERS:
        raise ConfigError(f"unknown layer {layer!r}")
    cell_id = stable_id(
        "cell",
        {
            "group": comparison_group_id,
            "candidate": candidate_id,
            "layer": layer,
            "scenario": scenario,
            "workload": workload_spec_id,
        },
    )
    family = stable_id("fam", {"cell": cell_id, "contract": measurement_contract_id})
    return tuple(
        RunPlan(
            cell_id=cell_id,
            comparison_group_id=comparison_group_id,
            candidate_id=candidate_id,
            layer=layer,
            scenario=scenario,
            workload_spec_id=workload_spec_id,
            measurement_contract_id=measurement_contract_id,
            repetition=repetition,
            process_index=process,
            run_family_id=family,
        )
        for repetition in range(1, planned_repetitions + 1)
        for process in range(processes)
    )


# ── cold start / warmup / thermal / health ────────────────────────────────


class ColdStartLedger:
    """Cold-start stages are measured separately and never merged into steady."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._stages: Dict[str, Dict[str, Any]] = {}

    def record_stage(self, stage: str, *, started_ns: int, ended_ns: int, backend: str = "") -> None:
        if stage not in COLD_START_STAGES:
            raise ConfigError(f"unknown cold-start stage {stage!r}")
        if ended_ns < started_ns:
            raise ConfigError(f"stage {stage!r} ends before it starts")
        if stage in self._stages:
            raise ConfigError(f"stage {stage!r} already recorded: cold start is not overwritten")
        self._stages[stage] = {
            "stage": stage,
            "started_ns": started_ns,
            "ended_ns": ended_ns,
            "duration_ns": ended_ns - started_ns,
            "backend": backend,
        }

    def as_rows(self) -> List[Dict[str, Any]]:
        return [self._stages[name] for name in COLD_START_STAGES if name in self._stages]

    def total_ns(self) -> int:
        return sum(row["duration_ns"] for row in self._stages.values())

    def missing_stages(self) -> Tuple[str, ...]:
        return tuple(name for name in COLD_START_STAGES if name not in self._stages)


def steady_state_excludes_cold_start(ledger: ColdStartLedger, steady_start_ns: int) -> Dict[str, Any]:
    """The steady window must begin after the last cold-start stage."""
    stages = ledger.as_rows()
    last_end = max((row["ended_ns"] for row in stages), default=0)
    return {
        "steady_start_ns": steady_start_ns,
        "last_cold_stage_end_ns": last_end,
        "excludes_cold_start": steady_start_ns >= last_end,
        "missing_stages": list(ledger.missing_stages()),
    }


@dataclass
class WarmupResult:
    converged: bool
    rounds: int
    backend: str = ""
    reason: str = ""


class WarmupStateMachine:
    """Deterministic warmup: it converges or reports a failure, never extends silently."""

    def __init__(self, *, rounds: int = 5, tolerance_rel: float = 0.05) -> None:
        if rounds < 1:
            raise ConfigError("warmup needs at least one round")
        if not 0 < tolerance_rel < 1:
            raise ConfigError("warmup tolerance must be inside (0, 1)")
        self.rounds = rounds
        self.tolerance_rel = tolerance_rel
        self._history: List[float] = []

    def observe(self, latency_ns: float, *, backend: str) -> Optional[WarmupResult]:
        if latency_ns <= 0:
            raise ConfigError("warmup latency must be positive")
        self._history.append(float(latency_ns))
        if len(self._history) < self.rounds:
            return None
        window = self._history[-self.rounds :]
        spread = (max(window) - min(window)) / (sum(window) / len(window))
        if spread <= self.tolerance_rel:
            return WarmupResult(converged=True, rounds=len(self._history), backend=backend)
        if len(self._history) >= 2 * self.rounds:
            return WarmupResult(
                converged=False,
                rounds=len(self._history),
                backend=backend,
                reason=f"warmup did not converge within {2 * self.rounds} rounds (spread={spread:.3f})",
            )
        return None


def thermal_gate(
    snapshot: Mapping[str, Any],
    *,
    temperature_max_c: float = 85.0,
    clock_min_ratio: float = 0.9,
    reference_clock_mhz: float = 0.0,
) -> Dict[str, Any]:
    """Start-of-run state gate: wait or mark, never start silently."""
    temperature = snapshot.get("temperature_c")
    clock = snapshot.get("clock_mhz")
    load = snapshot.get("system_load")
    reasons: List[str] = []
    if temperature is None:
        reasons.append("temperature unavailable: the run cannot be shown to start in range")
    elif float(temperature) > temperature_max_c:
        reasons.append(f"temperature {temperature}C above the gate")
    if reference_clock_mhz and clock is not None:
        if float(clock) < clock_min_ratio * reference_clock_mhz:
            reasons.append(f"clock {clock}MHz below {clock_min_ratio:.2f} of the reference")
    if load is not None and float(load) > 1.0:
        reasons.append(f"background load {load} above the gate")
    return {
        "action": "PROCEED" if not reasons else "WAIT_OR_MARK",
        "reasons": reasons,
        "snapshot": dict(sorted(snapshot.items())),
    }


def run_health_check(
    run_id: str,
    telemetry: Mapping[str, Any],
    *,
    oom_events: int = 0,
    neighbor_processes: Sequence[str] = (),
    backlog_requests: int = 0,
) -> Dict[str, Any]:
    """Per-run health labels: mark the run, never delete it here."""
    labels: List[str] = []
    if telemetry.get("throttle_flags"):
        labels.append("THERMAL_THROTTLE")
    if telemetry.get("power_limited"):
        labels.append("POWER_LIMIT_THROTTLE")
    if telemetry.get("swap_used_bytes"):
        labels.append("SWAP_ACTIVITY")
    if telemetry.get("ram_used_ratio") and float(telemetry["ram_used_ratio"]) > 0.9:
        labels.append("MEMORY_PRESSURE")
    if oom_events:
        labels.append("OOM_OR_RECOVERY")
    if neighbor_processes:
        labels.append("NEIGHBOR_PROCESS")
    if backlog_requests:
        labels.append("SERVICE_BACKLOG_CONTAMINATION")
    if telemetry.get("telemetry_gap_s"):
        labels.append("TELEMETRY_GAP")
    unknown = [label for label in labels if label not in HEALTH_FLAG_HINTS]
    if unknown:
        raise ConfigError(f"unknown health labels {unknown}")
    healthy = not labels
    return {
        "run_id": run_id,
        "labels": labels + (["HEALTHY_INCLUDED"] if healthy else []),
        "healthy": healthy,
        "unknown_is_not_healthy": "UNKNOWN_ANOMALY" not in labels,
        "status": "health_checked",
    }


# ── run estimates and matrices ────────────────────────────────────────────


def run_estimates(
    observations: Sequence[Observation],
    *,
    estimator: str = "median",
    inclusion_status_by_run: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, Any], ...]:
    """Run-level estimates first; cross-day aggregation is E12-04's job."""
    buckets: Dict[Tuple[str, str], List[float]] = {}
    for row in observations:
        if row.status != OBSERVATION_STATUS_OK or row.latency_ns is None:
            continue
        metric = row.latency_component or "latency"
        buckets.setdefault((row.run_id, metric), []).append(float(row.latency_ns))
    rows: List[Dict[str, Any]] = []
    for (run_id, metric), values in sorted(buckets.items()):
        ordered = sorted(values)
        rows.append(
            {
                "run_id": run_id,
                "cell_id": "",
                "metric_name": metric,
                "estimator": estimator,
                "value": _estimate(ordered, estimator),
                "unit": "ns",
                "sample_count": len(ordered),
                "inclusion_status": (inclusion_status_by_run or {}).get(run_id, "included"),
                "interval_low": ordered[0],
                "interval_high": ordered[-1],
            }
        )
    return tuple(rows)


def comparison_matrix_rows(
    estimates: Sequence[Mapping[str, Any]],
    *,
    cell_meta: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    """The normalized comparison matrix: raw value plus its allowed normalization."""
    rows: List[Dict[str, Any]] = []
    for estimate in estimates:
        cell_id = str(estimate.get("cell_id", ""))
        meta = cell_meta.get(cell_id, {})
        denominator = meta.get("denominator_value")
        value = estimate.get("value")
        rows.append(
            {
                "cell_id": cell_id,
                "comparison_group_id": meta.get("comparison_group_id", ""),
                "candidate_id": meta.get("candidate_id", ""),
                "layer": meta.get("layer", ""),
                "metric_name": estimate.get("metric_name", ""),
                "raw_value": value,
                "raw_unit": estimate.get("unit", ""),
                "normalized_value": (value / denominator) if (value is not None and denominator) else value,
                "normalized_unit": f"{estimate.get('unit', '')}/{meta.get('denominator_unit', 'unit')}" if denominator else estimate.get("unit", ""),
                "denominator": denominator,
                "status": "VALID" if value is not None else "MISSING",
            }
        )
    return tuple(rows)


def cross_layer_conversion_row(
    *,
    chain_id: str,
    candidate_id: str,
    workload_spec_id: str,
    operator_effect: Optional[float],
    op_time_share: Optional[float],
    dispatch_hit_rate: Optional[float],
    observed_model_core_effect: Optional[float],
    observed_service_effect: Optional[float] = None,
    single_device_effect: Optional[float] = None,
    observed_distributed_effect: Optional[float] = None,
    model_core_overheads: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """operator → model-core → service → distributed with an explicit residual."""
    predicted = None
    if operator_effect is not None and op_time_share is not None and dispatch_hit_rate is not None:
        predicted = 1.0 + (float(operator_effect) - 1.0) * float(op_time_share) * float(dispatch_hit_rate)
    conversion = CrossLayerConversion(
        chain_id=chain_id,
        candidate_id=candidate_id,
        workload_spec_id=workload_spec_id,
        operator_effect=operator_effect,
        op_time_share=op_time_share,
        dispatch_hit_rate=dispatch_hit_rate,
        predicted_model_core_effect=predicted,
        observed_model_core_effect=observed_model_core_effect,
        model_core_overheads=dict(model_core_overheads or {}),
        observed_service_effect=observed_service_effect,
        single_device_effect=single_device_effect,
        observed_distributed_effect=observed_distributed_effect,
        residual_status="NOT_APPLICABLE" if observed_model_core_effect is None else "UNEXPLAINED",
    )
    problems = conversion.validate()
    if problems:
        raise ConfigError("invalid cross-layer conversion: " + "; ".join(problems))
    return conversion.as_dict()


def service_goodput_rows(
    requests: Sequence[Mapping[str, Any]],
    slo: SloSpec,
    *,
    measurement_seconds: float,
    quality_ok_by_request: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Any]:
    """SLO-qualified goodput; over-SLO requests never count (§8.3)."""
    return goodput(
        requests,
        slo,
        measurement_seconds=measurement_seconds,
        quality_ok_by_request=quality_ok_by_request,
    )


def overload_recovery_check(
    requests: Sequence[Mapping[str, Any]], *, measurement_end_ns: int, drain_deadline_ns: int
) -> Dict[str, Any]:
    """Overload bookkeeping: backlog must be measured and drained, not hidden."""
    states = ("failed", "rejected", "timed_out", "cancelled", "retried", "backlog")
    counts = {state: 0 for state in states}
    for row in requests:
        state = str(row.get("status", ""))
        if state in counts:
            counts[state] += 1
    backlog_after_measurement = [
        row for row in requests if str(row.get("status", "")) == "backlog"
    ]
    drained = not backlog_after_measurement or measurement_end_ns >= drain_deadline_ns
    return {
        "counts": counts,
        "backlog_present": bool(backlog_after_measurement),
        "drain_window_observed": drained,
        "killing_the_service_at_window_end_would_hide_debt": not drained,
        "status": "ok" if drained else "BACKLOG_CONTAMINATION",
    }


def token_accounting_closes(accounting: TokenAccounting) -> Dict[str, Any]:
    problems = accounting.validate()
    return {
        "closed": accounting.closed(),
        "problems": problems,
        "ok": not problems and accounting.closed(),
    }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only callability check; never an experiment."""
    observation = Observation(
        observation_id="",
        run_id="run-1",
        sample_id="s1",
        timestamp_monotonic_ns=1,
        candidate_id="cand_a",
        comparison_id="cmp",
        layer="model_core",
        scenario="decode",
        workload_spec_id="ws",
        requested_backend="cuda",
        actual_backend="cuda",
        latency_component="core_tpot",
        latency_ns=1_000_000,
    )
    store = ObservationStore()
    store.append(observation)
    normalized = normalize(
        [observation],
        metric_name="core_tpot",
        estimator="median",
        basis="per_accepted_token",
        denominator=100,
        quality_status="pass",
        comparability_status="COMPARABLE",
    )
    forbidden = False
    try:
        normalize(
            [observation],
            metric_name="core_tpot",
            estimator="median",
            basis="tdp_as_energy",
            quality_status="pass",
            comparability_status="COMPARABLE",
        )
    except ConfigError:
        forbidden = True
    clean = {
        "status": "OK",
        "layer": "model_core",
        "actual_backend": "cuda",
        "comparability_status": "COMPARABLE",
        "contract_sha256": "a" * 64,
        "quality_status": "pass",
    }
    dirty = dict(clean, status="RUN_FAILED", value=0.0)
    return {
        "status": "smoke",
        "claim_allowed": False,
        "stored_observations": len(store),
        "normalized_has_sources": bool(normalized.source_observation_ids),
        "forbidden_basis_rejected": forbidden,
        "failed_row_with_number_rejected": bool(validate_cross_constraints(dirty)),
        "clean_row_ok": validate_cross_constraints(clean) == (),
        "fewer_than_three_runs_rejected": _expect_config_error(
            lambda: plan_cells(
                comparison_group_id="g",
                candidate_id="c",
                layer="model_core",
                scenario="decode",
                workload_spec_id="ws",
                measurement_contract_id="mc",
                planned_repetitions=2,
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
    (1, "读取冻结 suite manifest", ("comparability:suite_manifest_valid", "comparability:ComparisonGroup")),
    (2, "创建 campaign/run family", ("benchmark:RunPlan", "benchmark:next_run_id")),
    (3, "验证平台身份与独占范围", ("platform:PlatformIdentity", "platform:identity_conflicts")),
    (4, "验证 ModelArtifact/QuantArtifact", ("contracts:SemanticIdentity", "campaign:UpstreamEvidence")),
    (5, "验证输入和 reference", ("layers:MEASUREMENT_BOUNDARIES", "benchmark:Observation")),
    (6, "记录 cold-start 分段时间", ("benchmark:ColdStartLedger", "benchmark:COLD_START_STAGES")),
    (7, "执行预热状态机", ("benchmark:WarmupStateMachine", "benchmark:WarmupResult")),
    (8, "执行 thermal/cooldown gate", ("benchmark:thermal_gate", "records:MISSINGNESS_CODES")),
    (9, "运行 operator correctness", ("benchmark:Observation", "layers:LAYER_REQUIRED_FIELDS")),
    (10, "运行 operator 无 profiler timing", ("benchmark:ObservationStore", "benchmark:quantile")),
    (11, "记录 operator actual dispatch", ("benchmark:Observation.validate", "telemetry:DISPATCH_ROW_FIELDS")),
    (12, "运行 operator profile 子集", ("layers:MeasurementBoundary", "telemetry:project_c7")),
    (13, "计算 operator logical metrics", ("layers:ESTIMANDS", "benchmark:normalize")),
    (14, "运行 model-core correctness", ("layers:TokenAccounting", "benchmark:token_accounting_closes")),
    (15, "运行 model-core 冷/暖分离", ("benchmark:steady_state_excludes_cold_start", "benchmark:ColdStartLedger")),
    (16, "运行六类 model-core workload", ("layers:SCENARIOS", "benchmark:run_estimates")),
    (17, "验证 model-core token accounting", ("layers:recompute_token_accounting", "layers:TOKEN_ACCOUNTING_FIELDS")),
    (18, "采集 model-core memory", ("telemetry:S12ResultFields", "records:TABLE_SCHEMAS")),
    (19, "执行 model-core profile 子集", ("telemetry:project_c7", "benchmark:run_health_check")),
    (20, "启动服务并验证 readiness", ("layers:MEASUREMENT_BOUNDARIES", "benchmark:Observation")),
    (21, "校验 load generator", ("benchmark:overload_recovery_check", "layers:SloSpec")),
    (22, "运行单请求服务基线", ("layers:qualify_request", "benchmark:Observation")),
    (23, "运行 open-loop request-rate sweep", ("layers:LOAD_MODES", "benchmark:plan_cells")),
    (24, "运行 closed-loop/offline 对照", ("layers:SCENARIOS", "benchmark:Observation")),
    (25, "采集 service 内部状态", ("telemetry:S12ResultFields", "records:TABLE_SCHEMAS")),
    (26, "计算 SLO-qualified goodput", ("benchmark:service_goodput_rows", "layers:goodput")),
    (27, "验证过载和恢复完整性", ("benchmark:overload_recovery_check", "benchmark:HEALTH_FLAG_HINTS")),
    (28, "运行 collective correctness/micro", ("layers:ESTIMANDS", "benchmark:Observation")),
    (29, "运行 distributed model correctness", ("benchmark:validate_layer_row", "layers:LAYER_REQUIRED_FIELDS")),
    (30, "运行 scaling/capacity suite", ("layers:SCENARIOS", "benchmark:plan_cells")),
    (31, "采集 distributed trace", ("telemetry:project_c7", "telemetry:span_chain_check")),
    (32, "执行每 run 健康检查", ("benchmark:run_health_check", "benchmark:HEALTH_FLAG_HINTS")),
    (33, "运行 schema/contract validator", ("benchmark:validate_matrix_rows", "benchmark:VALIDATOR_CONSTRAINTS")),
    (34, "生成 run-level estimates", ("benchmark:run_estimates", "benchmark:quantile")),
    (35, "生成 normalized matrix", ("benchmark:comparison_matrix_rows", "benchmark:NormalizedResult")),
    (36, "形成跨层转化报告", ("benchmark:cross_layer_conversion_row", "layers:CrossLayerConversion")),
)
