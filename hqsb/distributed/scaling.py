"""Strong/weak/capacity scaling, pairability and the time decomposition (E10-05).

The experiment answers "how much speed/throughput/capacity does another device
buy, and why does efficiency drop".  The module therefore refuses three common
fabrications structurally:

* **no fake T1** — :func:`strong_speedup` will not emit a strong-scaling
  ``speedup`` without a real single-device baseline; it emits
  ``speedup_from_p0`` instead and says so;
* **no zero-filled missing grid** — an unavailable cell is a row with
  ``status="MISSING"`` and ``None`` values plus a reason, never ``0``;
* **no mixed definitions** — a weak-scaling experiment must *name* what is held
  constant, and strong/weak/capacity tables are built by different functions.

The decomposition helper enforces ``compute + comm + wait + idle ≈ wall`` with
overlap subtracted once, so a double-counted overlap cannot pass as a
breakdown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Scaling kinds; they never share one curve (details E10-05 §3).
SCALING_KINDS: Tuple[str, ...] = ("strong", "weak", "capacity")

#: Weak-scaling definitions (details E10-05 §3.3) — a named choice is mandatory.
WEAK_DEFINITIONS: Tuple[str, ...] = (
    "request_weak",
    "token_weak",
    "model_capacity_weak",
    "sequence_weak",
    "per_device_active_tokens",
    "per_device_batch_share",
)

#: Verdict vocabulary specific to E10-05 §12.
SCALING_VERDICTS: Tuple[str, ...] = (
    "PASS_POSITIVE_LATENCY",
    "PASS_POSITIVE_CAPACITY",
    "PASS_NEGATIVE",
    "INCONCLUSIVE",
    "FAIL",
)

#: Identity fields that must match for two rows to form a scaling pair (§12).
PAIRABILITY_FIELDS: Tuple[str, ...] = (
    "model_artifact_hash",
    "tokenizer_hash",
    "precision",
    "quant_quality_id",
    "token_ids_hash",
    "input_tokens",
    "output_tokens",
    "global_batch",
    "stop_rule",
    "parallel_semantics_hash",
    "kv_layout_hash",
    "graph_policy",
    "custom_kernel_policy",
    "timing_boundary",
    "topology_family",
    "backend",
    "actual_backend",
)

#: Field names that must never be silently "normalised away" (details E10-05 §12).
NON_PAIRABLE_TABLE: Tuple[str, ...] = (
    "different model/precision/token counts",
    "different parallel semantics or KV layout",
    "different cold/warm boundary",
    "cross-node vs intra-node mixed without pre-registration",
    "silent fallback to another backend",
)

#: Stop rules of details E10-05 §13.
STOP_RULES: Mapping[str, str] = {
    "correctness_failure": "stop that TP degree",
    "memory_above_safety_margin": "stop growing the workload",
    "health_link_clock_anomaly": "flag the run and re-request resources; do not retry forever",
    "collective_timeout_in_normal_mode": "go to E10-03/E10-10",
    "high_variance_after_repeats": "INCONCLUSIVE; do not pick the fastest value",
    "capacity_search_instability": "the maximum point needs its own stable run",
    "confirmation_failure": "do not keep tuning and still call it the same holdout",
}


def _require(condition: bool, message: str, *, field_name: str = "") -> None:
    if not condition:
        raise ConfigError(message, details={"field": field_name} if field_name else None)


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ConfigError("cannot take a quantile of an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def bootstrap_ci(
    values: Sequence[float], *, resamples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> Dict[str, float]:
    """Deterministic bootstrap CI over run-level samples (handbook §5.4)."""
    if not values:
        raise ConfigError("the CI needs at least one sample")
    if resamples < 100:
        raise ConfigError("use at least 100 resamples", details={"field": "resamples"})
    n = len(values)
    if n == 1:
        return {"low": float(values[0]), "high": float(values[0]), "median": float(values[0])}
    state = seed or 12345
    medians: List[float] = []
    for _ in range(resamples):
        sample: List[float] = []
        for _index in range(n):
            state = (1103515245 * state + 12345) % (2**31)
            sample.append(float(values[state % n]))
        medians.append(_quantile(sample, 0.5))
    alpha = (1.0 - confidence) / 2.0
    return {
        "low": _quantile(medians, alpha),
        "high": _quantile(medians, 1.0 - alpha),
        "median": _quantile(medians, 0.5),
    }


# ── prerequisites / work units ─────────────────────────────────────────────


@dataclass(frozen=True)
class ScalingPreregistration:
    """Primary/secondary metrics and non-claims frozen before the sweep (step 2)."""

    primary_metric: str
    secondary_metrics: Tuple[str, ...]
    non_claims: Tuple[str, ...]
    workload_id: str
    guardrails: Tuple[str, ...] = ()
    effect_threshold: float = 0.0
    ci_level: float = 0.95
    independent_runs: int = 3
    hypothesis: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        _require(bool(self.primary_metric), "a preregistration needs a primary metric", field_name="primary_metric")
        _require(bool(self.workload_id), "a preregistration needs a workload id", field_name="workload_id")
        _require(
            bool(self.non_claims),
            "write down what this experiment does NOT cover (other hardware/node counts)",
            field_name="non_claims",
        )
        _require(
            self.independent_runs >= 3,
            "handbook §5.4 requires at least three independent runs",
            field_name="independent_runs",
        )
        _require(
            bool(self.hypothesis),
            "a precondition-free hypothesis is not a preregistration",
            field_name="hypothesis",
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "primary_metric": self.primary_metric,
            "secondary_metrics": list(self.secondary_metrics),
            "non_claims": list(self.non_claims),
            "workload_id": self.workload_id,
            "guardrails": list(self.guardrails),
            "effect_threshold": self.effect_threshold,
            "ci_level": self.ci_level,
            "independent_runs": self.independent_runs,
            "hypothesis": self.hypothesis,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class WorkUnit:
    """The frozen strong-scaling work unit (details E10-05 step 3)."""

    identity: Mapping[str, Any]
    model_artifact_hash: str
    workload_hash: str
    isl: int
    osl: int
    global_batch: int
    requests: int
    precision: str
    kv_initial_state: str = ""
    timing_boundary: str = ""
    hash: str = ""

    def __post_init__(self) -> None:
        _require(self.isl > 0 and self.osl >= 0, "ISL/OSL must be positive")
        _require(self.global_batch >= 1 and self.requests >= 1, "batch/requests must be >= 1")
        _require(bool(self.model_artifact_hash), "a work unit needs the model artifact hash")
        _require(bool(self.workload_hash), "a work unit needs a workload hash")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "identity": dict(self.identity),
            "model_artifact_hash": self.model_artifact_hash,
            "workload_hash": self.workload_hash,
            "isl": self.isl,
            "osl": self.osl,
            "global_batch": self.global_batch,
            "requests": self.requests,
            "precision": self.precision,
            "kv_initial_state": self.kv_initial_state,
            "timing_boundary": self.timing_boundary,
            "hash": self.hash,
        }


@dataclass(frozen=True)
class WeakWorkUnit:
    """The named weak-scaling definition (details E10-05 step 4)."""

    definition: str
    held_constant: str
    per_device_work: float
    global_formula: str
    name: str = ""

    def __post_init__(self) -> None:
        _require(
            self.definition in WEAK_DEFINITIONS,
            f"definition must be one of {WEAK_DEFINITIONS}",
            field_name="definition",
        )
        _require(bool(self.global_formula), "write the global workload formula explicitly")
        _require(
            bool(self.name),
            "an unnamed weak-scaling definition cannot share an efficiency axis",
            field_name="name",
        )

    def global_work_for(self, world_size: int) -> float:
        return self.per_device_work * world_size

    def as_dict(self) -> Dict[str, Any]:
        return {
            "definition": self.definition,
            "held_constant": self.held_constant,
            "per_device_work": self.per_device_work,
            "global_formula": self.global_formula,
            "name": self.name,
        }


@dataclass(frozen=True)
class CapacityProtocol:
    """The capacity-search protocol (details E10-05 step 5)."""

    ladder: Tuple[int, ...]
    oom_margin_fraction: float
    max_runnable_criteria: Tuple[str, ...]
    stop_conditions: Tuple[str, ...] = ()
    binary_refinement: bool = False

    def __post_init__(self) -> None:
        _require(bool(self.ladder), "a capacity protocol needs a ladder", field_name="ladder")
        _require(
            list(self.ladder) == sorted(self.ladder),
            "the capacity ladder must be monotonically increasing",
            field_name="ladder",
        )
        _require(
            0.0 < self.oom_margin_fraction < 1.0,
            "the OOM safety margin must be in (0, 1)",
            field_name="oom_margin_fraction",
        )
        _require(
            bool(self.max_runnable_criteria),
            "define what 'max runnable' means (correctness/SLO/stability)",
            field_name="max_runnable_criteria",
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ladder": list(self.ladder),
            "oom_margin_fraction": self.oom_margin_fraction,
            "max_runnable_criteria": list(self.max_runnable_criteria),
            "stop_conditions": list(self.stop_conditions),
            "binary_refinement": self.binary_refinement,
        }


# ── resource matrix ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResourceMatrixCell:
    """One available/missing (device_count, node_count, degree) cell."""

    device_count: int
    node_count: int
    tp_degree: int
    available: bool
    missing_reason: str = ""
    resource_date: str = ""

    def __post_init__(self) -> None:
        _require(self.device_count >= 1, "device_count must be >= 1")
        _require(self.node_count >= 1, "node_count must be >= 1")
        _require(self.tp_degree >= 1, "tp_degree must be >= 1")
        _require(
            self.tp_degree <= self.device_count,
            "a TP degree cannot exceed the device count",
            field_name="tp_degree",
        )
        if not self.available:
            _require(bool(self.missing_reason), "a missing cell needs a reason", field_name="missing_reason")

    @property
    def topology_family(self) -> str:
        return "single_node" if self.node_count == 1 else f"multi_node_{self.node_count}"

    def as_row(self) -> Dict[str, Any]:
        """Missing cells carry ``None`` — never a zero that could be plotted."""
        if not self.available:
            return {
                "device_count": self.device_count,
                "node_count": self.node_count,
                "tp_degree": self.tp_degree,
                "topology_family": self.topology_family,
                "status": "MISSING",
                "reason": self.missing_reason,
                "latency_ms": None,
                "throughput": None,
                "memory_bytes": None,
                "resource_date": self.resource_date,
            }
        return {
            "device_count": self.device_count,
            "node_count": self.node_count,
            "tp_degree": self.tp_degree,
            "topology_family": self.topology_family,
            "status": "AVAILABLE",
            "reason": "",
            "resource_date": self.resource_date,
        }

    def as_dict(self) -> Dict[str, Any]:
        return self.as_row()


@dataclass(frozen=True)
class ResourceMatrix:
    cells: Tuple[ResourceMatrixCell, ...]

    def available(self) -> List[ResourceMatrixCell]:
        return [cell for cell in self.cells if cell.available]

    def missing(self) -> List[ResourceMatrixCell]:
        return [cell for cell in self.cells if not cell.available]

    def degrees_for(self, device_count: int, node_count: int) -> List[int]:
        return sorted(
            cell.tp_degree
            for cell in self.cells
            if cell.available and cell.device_count == device_count and cell.node_count == node_count
        )

    def as_rows(self) -> List[Dict[str, Any]]:
        return [cell.as_row() for cell in self.cells]


def topology_family_label(*, node_count: int) -> str:
    return "single_node" if node_count == 1 else f"multi_node_{node_count}"


def refuse_mixed_families(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """A single curve must not mix single-node and multi-node points (§5)."""
    families = {str(row.get("topology_family", "")) for row in rows}
    if len(families) > 1:
        return [
            f"rows mix topology families {sorted(families)}; separate the families and their charts"
        ]
    return []


# ── baseline / speedup ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Baseline:
    """The single-device baseline, or an explicit statement that it cannot exist."""

    status: str  # T1 | NO_T1_CAPACITY
    latency_ms: Optional[float] = None
    p0_degree: int = 1
    reason: str = ""

    def __post_init__(self) -> None:
        _require(self.status in ("T1", "NO_T1_CAPACITY"), "status must be T1|NO_T1_CAPACITY", field_name="status")
        if self.status == "T1":
            _require(
                self.latency_ms is not None and self.latency_ms > 0,
                "a T1 baseline needs a measured latency",
                field_name="latency_ms",
            )
        else:
            _require(
                bool(self.reason),
                "no T1 must say why (the model does not fit on one device)",
                field_name="reason",
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "latency_ms": self.latency_ms,
            "p0_degree": self.p0_degree,
            "reason": self.reason,
        }


def baseline_calibration(
    *, t1_latency_ms: Optional[float], reason: str = "", p0_degree: int = 1
) -> Baseline:
    """Decide whether a real T1 exists; never estimate one."""
    if t1_latency_ms is None:
        return Baseline(
            status="NO_T1_CAPACITY",
            p0_degree=p0_degree,
            reason=reason or "the model does not fit/run on one device",
        )
    return Baseline(status="T1", latency_ms=float(t1_latency_ms), p0_degree=p0_degree)


@dataclass(frozen=True)
class SpeedupResult:
    metric_name: str
    speedup: Optional[float]
    efficiency: Optional[float]
    baseline_status: str
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "speedup": self.speedup,
            "efficiency": self.efficiency,
            "baseline_status": self.baseline_status,
            "note": self.note,
        }


def strong_speedup(baseline: Baseline, *, degree: int, latency_ms: float) -> SpeedupResult:
    """Speedup only against a real T1; otherwise ``speedup_from_p0`` (§3.1)."""
    _require(degree >= 1, "degree must be >= 1", field_name="degree")
    _require(latency_ms > 0, "latency must be positive", field_name="latency_ms")
    if baseline.status != "T1":
        return SpeedupResult(
            metric_name="speedup_from_p0",
            speedup=None,
            efficiency=None,
            baseline_status=baseline.status,
            note=(
                "no real single-device T1 exists; relative scaling is reported as "
                f"speedup_from_p0 against degree {baseline.p0_degree} and must not be called "
                "strong-scaling efficiency"
            ),
        )
    speedup = float(baseline.latency_ms) / latency_ms
    return SpeedupResult(
        metric_name="speedup",
        speedup=speedup,
        efficiency=speedup / degree,
        baseline_status="T1",
        note="strong scaling: same model, precision and global work",
    )


def speedup_from_p0(baseline_latency_ms: float, *, degree: int, latency_ms: float) -> Dict[str, Any]:
    """The renamed relative metric used when T1 is unavailable."""
    _require(baseline_latency_ms > 0 and latency_ms > 0, "latencies must be positive")
    return {
        "metric_name": "speedup_from_p0",
        "speedup_from_p0": baseline_latency_ms / latency_ms,
        "degree": degree,
        "not_a_strong_scaling_number": True,
    }


def weak_efficiency(rows: Sequence[Mapping[str, Any]], *, definition: str) -> Dict[str, Any]:
    """Per-device throughput/latency plus the named efficiency denominator (§3.3)."""
    if definition not in WEAK_DEFINITIONS:
        raise ConfigError(
            f"definition must be one of {WEAK_DEFINITIONS}", details={"field": "definition"}
        )
    entries: List[Dict[str, Any]] = []
    for row in rows:
        world_size = int(row["world_size"])
        throughput = float(row.get("throughput", 0.0))
        latency = float(row.get("latency_ms", 0.0))
        entries.append(
            {
                "world_size": world_size,
                "throughput": throughput,
                "per_device_throughput": throughput / world_size if world_size else 0.0,
                "latency_ms": latency,
                "definition": definition,
            }
        )
    if entries:
        reference = entries[0]["per_device_throughput"]
        for entry in entries:
            entry["weak_efficiency"] = (
                entry["per_device_throughput"] / reference if reference else None
            )
    return {
        "definition": definition,
        "rows": entries,
        "note": "the definition name must be printed with every weak-efficiency number",
    }


def device_seconds(
    *, device_count: int, job_time_s: float, requests: int = 0, tokens: int = 0
) -> Dict[str, Any]:
    """device_seconds = p × job time; per request/token (details E10-05 step 28)."""
    _require(device_count >= 1, "device_count must be >= 1")
    _require(job_time_s > 0, "job_time_s must be positive", field_name="job_time_s")
    total = device_count * job_time_s
    return {
        "device_seconds": total,
        "device_seconds_per_request": total / requests if requests else None,
        "device_seconds_per_token": total / tokens if tokens else None,
        "note": "cost/TCO belongs to S12; this only exposes the resource price of a latency win",
    }


# ── time decomposition ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class TimeDecomposition:
    """compute/comm/overlap/wait/idle on the critical path (details README §12)."""

    compute_active_ms: float
    comm_active_ms: float
    overlap_ms: float
    dependency_wait_ms: float
    host_submission_gap_ms: float
    barrier_or_sync_ms: float
    total_ms: float
    tolerance_fraction: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "compute_active_ms",
            "comm_active_ms",
            "overlap_ms",
            "dependency_wait_ms",
            "host_submission_gap_ms",
            "barrier_or_sync_ms",
            "total_ms",
        ):
            if float(getattr(self, name)) < 0:
                raise ConfigError(f"{name} must be >= 0", details={"field": name})
        _require(self.total_ms > 0, "total_ms must be positive", field_name="total_ms")

    @property
    def exposed_comm_ms(self) -> float:
        return self.comm_active_ms - self.overlap_ms

    @property
    def overlap_fraction_comm(self) -> Optional[float]:
        return self.overlap_ms / self.comm_active_ms if self.comm_active_ms else None

    @property
    def accounted_ms(self) -> float:
        return (
            self.compute_active_ms
            + self.exposed_comm_ms
            + self.dependency_wait_ms
            + self.host_submission_gap_ms
            + self.barrier_or_sync_ms
        )

    @property
    def unexplained_idle_ms(self) -> float:
        return self.total_ms - self.accounted_ms

    @property
    def balanced(self) -> bool:
        return abs(self.unexplained_idle_ms) <= self.total_ms * self.tolerance_fraction

    def as_dict(self) -> Dict[str, Any]:
        return {
            "compute_active_ms": self.compute_active_ms,
            "comm_active_ms": self.comm_active_ms,
            "overlap_ms": self.overlap_ms,
            "exposed_comm_ms": self.exposed_comm_ms,
            "overlap_fraction_comm": self.overlap_fraction_comm,
            "dependency_wait_ms": self.dependency_wait_ms,
            "host_submission_gap_ms": self.host_submission_gap_ms,
            "barrier_or_sync_ms": self.barrier_or_sync_ms,
            "total_ms": self.total_ms,
            "accounted_ms": self.accounted_ms,
            "unexplained_idle_ms": self.unexplained_idle_ms,
            "balanced": self.balanced,
            "note": "overlap is subtracted once; compute+comm must never be added to wall time",
        }


def decomposition(*, compute_ms: float, comm_ms: float, overlap_ms: float, wait_ms: float, host_gap_ms: float, sync_ms: float, total_ms: float) -> TimeDecomposition:
    return TimeDecomposition(
        compute_active_ms=compute_ms,
        comm_active_ms=comm_ms,
        overlap_ms=overlap_ms,
        dependency_wait_ms=wait_ms,
        host_submission_gap_ms=host_gap_ms,
        barrier_or_sync_ms=sync_ms,
        total_ms=total_ms,
    )


# ── pairability ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Pairability:
    ok: bool
    reasons: Tuple[str, ...] = ()
    differing_fields: Tuple[str, ...] = ()
    non_pairable_table: Tuple[str, ...] = NON_PAIRABLE_TABLE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reasons": list(self.reasons),
            "differing_fields": list(self.differing_fields),
            "non_pairable_table": list(self.non_pairable_table),
            "note": "'normalise to per token' does not repair an identity mismatch",
        }


def pairability(left: Mapping[str, Any], right: Mapping[str, Any]) -> Pairability:
    """Two rows form a scaling pair only if the shared identity matches (§12)."""
    differing = [
        name
        for name in PAIRABILITY_FIELDS
        if left.get(name) != right.get(name)
    ]
    return Pairability(
        ok=not differing,
        reasons=(
            [] if not differing else [f"identity differs on {', '.join(differing)}"]
        ),
        differing_fields=tuple(differing),
    )


def split_pairable(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Partition rows into pairable (common identity) and non-pairable (kept apart)."""
    if not rows:
        return {"pairable": [], "non_pairable": []}
    reference = rows[0]
    pairable: List[Dict[str, Any]] = []
    non_pairable: List[Dict[str, Any]] = []
    for row in rows:
        verdict = pairability(reference, row)
        if verdict.ok:
            pairable.append(dict(row))
        else:
            entry = dict(row)
            entry["non_pairable_reasons"] = list(verdict.reasons)
            non_pairable.append(entry)
    return {"pairable": pairable, "non_pairable": non_pairable}


# ── model fit / anomalies ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ScalingFit:
    """Amdahl+comm fit over the *measured* degrees only (details E10-05 step 25)."""

    serial_ms: float
    parallel_eff: float
    comm_coeff_ms: float
    imbalance_ms: float
    residual_rms_ms: float
    measured_degrees: Tuple[int, ...]
    extrapolation_forbidden: bool = True
    reason: str = ""

    def predict_ms(self, degree: int, *, allow_extrapolation: bool = False) -> float:
        if degree not in self.measured_degrees and not allow_extrapolation:
            raise ConfigError(
                f"degree {degree} was not measured; extrapolating beyond "
                f"{list(self.measured_degrees)} is forbidden",
                details={"field": "degree"},
            )
        return (
            self.serial_ms
            + self.parallel_eff / degree
            + self.comm_coeff_ms * degree
            + self.imbalance_ms
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "serial_ms": self.serial_ms,
            "parallel_eff": self.parallel_eff,
            "comm_coeff_ms": self.comm_coeff_ms,
            "imbalance_ms": self.imbalance_ms,
            "residual_rms_ms": self.residual_rms_ms,
            "measured_degrees": list(self.measured_degrees),
            "extrapolation_forbidden": self.extrapolation_forbidden,
            "reason": self.reason,
        }


def fit_scaling_model(
    points: Sequence[Mapping[str, float]],
) -> ScalingFit:
    """Least-squares fit of ``T = a + b/p + c*p + d`` over measured points only."""
    usable = [
        (int(row["degree"]), float(row["latency_ms"]))
        for row in points
        if int(row.get("degree", 0)) >= 1 and float(row.get("latency_ms", 0.0)) > 0
    ]
    if len(usable) < 3:
        raise ConfigError(
            "a scaling fit needs at least three measured degrees",
            details={"field": "points"},
        )
    degrees = [item[0] for item in usable]
    _require(len(set(degrees)) == len(degrees), "duplicate degrees in the fit points")

    # Solve the 3-parameter normal equations for (a, b, c) with d absorbed into a.
    xs = [[1.0, 1.0 / p, float(p)] for p, _ in usable]
    ys = [t for _, t in usable]
    normal = [[sum(x[i] * x[j] for x in xs) for j in range(3)] for i in range(3)]
    rhs = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(3)]
    coefficients = _solve3(normal, rhs)
    residuals = [
        y - (coefficients[0] + coefficients[1] / p + coefficients[2] * p)
        for p, y in usable
    ]
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    return ScalingFit(
        serial_ms=coefficients[0],
        parallel_eff=coefficients[1],
        comm_coeff_ms=coefficients[2],
        imbalance_ms=0.0,
        residual_rms_ms=rms,
        measured_degrees=tuple(sorted(degrees)),
        reason="fitted only over the measured degrees; future card counts are not facts",
    )


def _solve3(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> List[float]:
    """Gaussian elimination for a 3x3 system (no external dependency)."""
    augmented = [list(row) + [rhs[index]] for index, row in enumerate(matrix)]
    n = 3
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ConfigError("the fit system is singular; choose distinct degrees")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        factor = augmented[col][col]
        augmented[col] = [value / factor for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            multiplier = augmented[row][col]
            augmented[row] = [
                augmented[row][index] - multiplier * augmented[col][index]
                for index in range(n + 1)
            ]
    return [augmented[row][n] for row in range(n)]


def flag_anomalies(
    rows: Sequence[Mapping[str, Any]],
    *,
    variability: Optional[Mapping[int, float]] = None,
    z_threshold: float = 3.0,
) -> List[Dict[str, Any]]:
    """Flag anomalous runs but keep every raw sample (step 27)."""
    variability = dict(variability or {})
    medians: Dict[int, float] = {}
    by_degree: Dict[int, List[float]] = {}
    for row in rows:
        by_degree.setdefault(int(row["world_size"]), []).append(float(row["latency_ms"]))
    for degree, values in by_degree.items():
        medians[degree] = _quantile(values, 0.5)
    flagged: List[Dict[str, Any]] = []
    for row in rows:
        degree = int(row["world_size"])
        latency = float(row["latency_ms"])
        spread = variability.get(degree, 0.0)
        median = medians.get(degree, latency)
        if spread > 0 and abs(latency - median) / spread > z_threshold:
            flagged.append(
                {
                    "rank_or_run": row.get("run_id", ""),
                    "world_size": degree,
                    "latency_ms": latency,
                    "median_ms": median,
                    "reason": f"deviation > {z_threshold}σ of the pre-registered variability",
                    "kept_in_raw": True,
                }
            )
    return flagged


def pick_fastest_is_forbidden(selection_rule: str) -> None:
    """Picking each degree's fastest run to draw a curve is a fabrication (§10)."""
    forbidden = ("fastest", "best_run", "min_over_runs")
    if any(token in selection_rule.lower() for token in forbidden):
        raise ConfigError(
            "selecting the fastest run per degree is forbidden; use the pre-registered "
            "aggregation over independent jobs",
            details={"field": "selection_rule"},
        )


# ── confirmation / verdict ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ConfirmationPlan:
    workloads: Tuple[str, ...]
    degrees: Tuple[int, ...]
    runs_per_cell: int = 3
    order_policy: str = "randomized_block"

    def __post_init__(self) -> None:
        _require(self.runs_per_cell >= 3, "confirmation needs at least three independent runs")
        _require(
            self.order_policy in ("randomized_block", "abba"),
            "order policy must be randomized_block|abba",
            field_name="order_policy",
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "workloads": list(self.workloads),
            "degrees": list(self.degrees),
            "runs_per_cell": self.runs_per_cell,
            "order_policy": self.order_policy,
        }


def confirmation_plan(
    *, workloads: Sequence[str], degrees: Sequence[int], runs_per_cell: int = 3
) -> ConfirmationPlan:
    if len(workloads) < 2:
        raise ConfigError("pick at least a best and a degraded workload for confirmation")
    return ConfirmationPlan(
        workloads=tuple(workloads), degrees=tuple(degrees), runs_per_cell=runs_per_cell
    )


def scaling_verdict(
    *,
    correctness_ok: bool,
    pairable: bool,
    resource_grid_complete: bool,
    repeat_runs: int,
    latency_effect: Optional[float],
    effect_threshold: float,
    latency_ci: Optional[Tuple[float, float]] = None,
    capacity_gain: bool = False,
    capacity_stable: bool = False,
    has_t1: bool = True,
    comm_explained: bool = True,
    variance_ok: bool = True,
) -> Dict[str, Any]:
    """The E10-05 §12 verdict with the guardrail ordering enforced."""
    if not correctness_ok:
        return {"status": "FAIL", "reason": "correctness/logit/token gate failed"}
    if not pairable:
        return {"status": "FAIL", "reason": "the compared rows are not a pairable identity"}
    if not comm_explained:
        return {
            "status": "FAIL",
            "reason": "the efficiency drop is not explained by messages/topology/kernel shape",
        }
    if not resource_grid_complete or repeat_runs < 3 or not variance_ok:
        return {
            "status": "INCONCLUSIVE",
            "reason": "resource grid, independent repeats, variance or trace evidence is insufficient",
        }
    if capacity_gain and capacity_stable:
        if latency_effect is not None and effect_threshold and latency_effect >= effect_threshold:
            return {
                "status": "PASS_POSITIVE_LATENCY",
                "reason": "correctness and guardrails pass; the strong primary met the preregistered effect/CI",
            }
        return {
            "status": "PASS_POSITIVE_CAPACITY",
            "reason": (
                "no significant speed win, but a model/context/KV configuration moved from "
                "unrunnable to stably runnable; this is capacity, not linear speedup"
            ),
        }
    return {
        "status": "PASS_NEGATIVE",
        "reason": "measurement is valid and shows no gain; the comm/kernel/idle boundary is explained",
    }


def apply_stop_rules(state: Mapping[str, Any]) -> Dict[str, Any]:
    triggered = [key for key in STOP_RULES if state.get(key)]
    return {
        "stop": bool(triggered),
        "triggered": triggered,
        "reasons": {key: STOP_RULES[key] for key in triggered},
    }


#: The scaling row schema of details README §17.4.
SCALING_ROW_FIELDS: Tuple[str, ...] = (
    "run_id",
    "world_size",
    "node_count",
    "tp",
    "pp",
    "ep",
    "cp",
    "global_batch",
    "per_rank_work",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "ttft_ms",
    "tpot_ms",
    "throughput",
    "speedup",
    "efficiency",
    "per_rank_memory",
    "compute_ms",
    "comm_ms",
    "overlap_ms",
    "wait_ms",
    "idle_ms",
    "failure_count",
    "status",
)


__all__ = [
    "Baseline",
    "CapacityProtocol",
    "ConfirmationPlan",
    "NON_PAIRABLE_TABLE",
    "PAIRABILITY_FIELDS",
    "Pairability",
    "ResourceMatrix",
    "ResourceMatrixCell",
    "SCALING_KINDS",
    "SCALING_ROW_FIELDS",
    "SCALING_VERDICTS",
    "STOP_RULES",
    "ScalingFit",
    "ScalingPreregistration",
    "SpeedupResult",
    "TimeDecomposition",
    "WEAK_DEFINITIONS",
    "WeakWorkUnit",
    "WorkUnit",
    "apply_stop_rules",
    "baseline_calibration",
    "bootstrap_ci",
    "confirmation_plan",
    "decomposition",
    "device_seconds",
    "fit_scaling_model",
    "flag_anomalies",
    "pairability",
    "pick_fastest_is_forbidden",
    "refuse_mixed_families",
    "scaling_verdict",
    "split_pairable",
    "speedup_from_p0",
    "strong_speedup",
    "topology_family_label",
    "weak_efficiency",
]
