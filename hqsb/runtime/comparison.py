"""Fair reference/edge/cloud comparison at the model-core boundary (E07-10).

The experiment's core difficulty is not measuring, it is **comparability**: the
same number can support a strong causal claim or a weak deployment portrait
depending on what was held constant.  This module encodes that ladder:

* :class:`Tier` A–D with the claim each tier permits (E07-10 §2);
* :func:`common_denominator_rows` / :func:`best_valid_rows` — two separate tables,
  never merged into one ranking;
* :func:`recompute_metrics` — every rate is recomputed from raw per-request and
  per-iteration records; a runtime's self-reported TPS is never copied;
* :func:`token_denominator_audit` — cached/speculative/recompute/cancelled
  positions cannot inflate throughput (details README §13.5);
* :func:`cold_warm_report` — load/convert/compile/capture/first-request/steady/
  close phases are separated and the offline build cost is reported, not hidden;
* :func:`pareto_front` — per hardware × workload, with NA rows kept;
* :func:`s08_interface_surface` — only stable contract fields cross the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.runtime.metrics import (
    PairedEffect,
    TokenConservationReport,
    TokenLedger,
    distribution_summary,
    paired_effect,
    percentile,
    token_accounting_from_requests,
)

# ── comparability tiers (E07-10 §2) ────────────────────────────────────────

TIER_A = "A"
TIER_B = "B"
TIER_C = "C"
TIER_D = "D"

TIER_CLAIMS: Mapping[str, str] = {
    TIER_A: "strong runtime causal comparison (same hardware, model, precision)",
    TIER_B: "comparable end-to-end runtime, but actual kernel/route must be recorded",
    TIER_C: "deployment portrait only; no runtime attribution across hardware",
    TIER_D: "capability case study only; no ranking across model format/precision",
}

TIER_RULES: Mapping[str, Tuple[str, ...]] = {
    TIER_A: ("hardware", "model", "precision"),
    TIER_B: ("hardware", "model"),
    TIER_C: ("model", "precision"),
    TIER_D: (),
}


@dataclass(frozen=True)
class ComparisonSpec:
    """Frozen comparison configuration (E07-10 step 1)."""

    model_id: str
    model_manifest_sha256: str
    precision: str
    hardware: str
    request_trace_hash: str
    statistics: str = "median_p50_p95_ci95"
    independent_processes: int = 3
    memory_budget_bytes: float = 0.0
    warmup_requests: int = 1
    quality_gate: str = "s05_quality_gate"
    token_denominators: Tuple[str, ...] = (
        "logical_input",
        "useful_committed",
        "model_computed",
    )

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_manifest_sha256",
            "precision",
            "hardware",
            "request_trace_hash",
        ):
            if not getattr(self, name):
                raise ConfigError(
                    f"comparison spec requires {name!r}: an unbound comparison "
                    "cannot be reproduced",
                    details={"field": name},
                )
        if self.independent_processes < 3:
            raise ConfigError(
                "at least three independent runtime processes are required "
                "(details README §13.3)",
                details={"field": "independent_processes"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_manifest_sha256": self.model_manifest_sha256,
            "precision": self.precision,
            "hardware": self.hardware,
            "request_trace_hash": self.request_trace_hash,
            "statistics": self.statistics,
            "independent_processes": self.independent_processes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "warmup_requests": self.warmup_requests,
            "quality_gate": self.quality_gate,
            "token_denominators": list(self.token_denominators),
        }


@dataclass(frozen=True)
class BackendIdentity:
    """Identity/precision facts a comparison row must carry."""

    backend_id: str
    role: str
    version: str
    commit: str
    model_id: str
    precision: str
    hardware: str
    actual_kernel_route: str = ""
    observability: str = "internal"

    def tier(self, spec: ComparisonSpec) -> str:
        """Derive the tier automatically — no hand-assigned tiers."""
        if (
            self.hardware == spec.hardware
            and self.model_id == spec.model_id
            and self.precision == spec.precision
        ):
            return TIER_A
        if self.hardware == spec.hardware and self.model_id == spec.model_id:
            return TIER_B
        if self.model_id == spec.model_id and self.precision == spec.precision:
            return TIER_C
        return TIER_D

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "role": self.role,
            "version": self.version,
            "commit": self.commit,
            "model_id": self.model_id,
            "precision": self.precision,
            "hardware": self.hardware,
            "actual_kernel_route": self.actual_kernel_route,
            "observability": self.observability,
        }


# ── rows and recomputation ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ComparisonRow:
    """One backend × workload row of the comparison table."""

    backend: BackendIdentity
    workload: str
    tier: str
    mode: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    comparable: bool = True
    na_reason: str = ""
    observability_limitation: str = ""
    quality_gate_passed: bool = True
    excluded_features: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.tier not in TIER_RULES:
            raise ConfigError(f"unknown tier {self.tier!r}", details={"field": "tier"})
        if self.mode not in ("common_denominator", "best_valid"):
            raise ConfigError(
                f"unknown comparison mode {self.mode!r}; the two tables must never be "
                "merged into one ranking",
                details={"field": "mode"},
            )
        if not self.comparable and not self.na_reason:
            raise ConfigError(
                "an incomparable row must carry a reason; 'NA' without a reason is "
                "indistinguishable from a missing measurement",
                details={"field": "na_reason"},
            )
        if self.comparable and not self.quality_gate_passed:
            raise ConfigError(
                "a row that fails its quality gate may not enter a comparison table",
                details={"backend": self.backend.backend_id},
            )

    @property
    def claim_strength(self) -> str:
        return TIER_CLAIMS[self.tier]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.backend_id,
            "role": self.backend.role,
            "workload": self.workload,
            "tier": self.tier,
            "claim_strength": self.claim_strength,
            "mode": self.mode,
            "metrics": dict(self.metrics),
            "comparable": self.comparable,
            "na_reason": self.na_reason,
            "observability_limitation": self.observability_limitation,
            "quality_gate_passed": self.quality_gate_passed,
            "excluded_features": list(self.excluded_features),
        }


#: Feature flags whose availability decides common-denominator comparability.
COMMON_DENOMINATOR_FEATURES: Tuple[str, ...] = (
    "streaming",
    "cancel",
    "continuous_batching",
    "chunked_prefill",
    "prefix_cache",
    "cuda_graph",
    "speculative",
    "custom_ops",
)


def common_denominator_rows(
    rows: Sequence[ComparisonRow],
    features: Mapping[str, Mapping[str, bool]],
) -> List[ComparisonRow]:
    """Keep only the features *every* backend supports, and record the exclusions."""
    if not features:
        raise ConfigError("common-denominator tables need the capability matrix")
    backends = sorted({row.backend.backend_id for row in rows})
    shared = [
        feature
        for feature in COMMON_DENOMINATOR_FEATURES
        if all(features.get(backend, {}).get(feature, False) for backend in backends)
    ]
    missing_rows = [backend for backend in backends if backend not in features]
    if missing_rows:
        raise ConfigError(
            "the capability matrix is missing backends "
            f"{missing_rows}; a common-denominator table cannot be built from an "
            "incomplete matrix",
            details={"field": "features"},
        )
    common_rows: List[ComparisonRow] = []
    for row in rows:
        if not row.comparable:
            common_rows.append(row)
            continue
        enabled = [
            feature
            for feature in COMMON_DENOMINATOR_FEATURES
            if features.get(row.backend.backend_id, {}).get(feature, False)
        ]
        excluded = tuple(
            feature for feature in enabled if feature not in shared
        )
        common_rows.append(
            ComparisonRow(
                backend=row.backend,
                workload=row.workload,
                tier=row.tier,
                mode="common_denominator",
                metrics=dict(row.metrics),
                comparable=row.comparable,
                na_reason=row.na_reason,
                observability_limitation=row.observability_limitation,
                quality_gate_passed=row.quality_gate_passed,
                excluded_features=excluded,
            )
        )
    return common_rows


def best_valid_rows(
    rows: Sequence[ComparisonRow], *, preregistered: bool
) -> List[ComparisonRow]:
    """Each backend at its pre-registered, quality-approved best configuration."""
    if not preregistered:
        raise ConfigError(
            "the best-valid table must be pre-registered; choosing each backend's "
            "best configuration after seeing the numbers is a ranking artifact "
            "(E07-10 §6)",
            details={"field": "preregistered"},
        )
    return [
        ComparisonRow(
            backend=row.backend,
            workload=row.workload,
            tier=row.tier,
            mode="best_valid",
            metrics=dict(row.metrics),
            comparable=row.comparable,
            na_reason=row.na_reason,
            observability_limitation=row.observability_limitation,
            quality_gate_passed=row.quality_gate_passed,
            excluded_features=row.excluded_features,
        )
        for row in rows
    ]


def recompute_metrics(
    *, requests: Sequence[Mapping[str, Any]], iterations: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Recompute every rate from raw records (never trust a self-reported TPS).

    The makespan is taken from the iteration ledger, the token counts from the
    per-request raw records, and the conservation audit runs before any rate is
    returned — so a runtime that reports a higher TPS than its own ledger
    supports produces a *different* number here, not a bigger one.
    """
    if not requests:
        raise ConfigError("recomputing metrics needs per-request raw records")
    ledger: TokenLedger = token_accounting_from_requests(requests)
    audit: TokenConservationReport = ledger.audit()
    if not audit.ok:
        raise ConfigError(
            "cannot recompute metrics from an inconsistent ledger: "
            + "; ".join(audit.problems),
            details={"fields": list(audit.problems)},
        )
    makespan_ms = 0.0
    model_time_ms = 0.0
    if iterations:
        makespan_ms = sum(
            float(entry.get("model_runner_ms", 0.0))
            + float(entry.get("scheduler_cpu_ms", 0.0))
            + float(entry.get("sample_ms", 0.0))
            for entry in iterations
        )
        model_time_ms = sum(float(entry.get("model_runner_ms", 0.0)) for entry in iterations)
    ttfts = [float(item["ttft_ms"]) for item in requests if "ttft_ms" in item]
    tpots = [float(item["tpot_ms"]) for item in requests if "tpot_ms" in item]
    result: Dict[str, Any] = {
        "requests": len(requests),
        "iterations": len(iterations),
        "makespan_ms": makespan_ms,
        "model_time_ms": model_time_ms,
        "token_ledger": ledger.as_dict(),
        "conservation": audit.as_dict(),
        "ttft_ms": distribution_summary(ttfts).as_dict() if ttfts else {},
        "tpot_ms": distribution_summary(tpots).as_dict() if tpots else {},
    }
    if makespan_ms > 0:
        result["output_tps"] = ledger.output_tps(makespan_ms / 1000.0)
        result["logical_token_tps"] = ledger.logical_token_tps(makespan_ms / 1000.0)
    if model_time_ms > 0:
        result["compute_position_tps"] = ledger.compute_position_tps(model_time_ms / 1000.0)
    return result


def token_denominator_audit(
    *,
    raw: Mapping[str, Any],
    reported: Mapping[str, float],
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Compare the runtime's self-reported rates against the recomputed ones.

    A rate the ledger cannot reproduce is a *problem*, not a rounding detail:
    the unified recomputation is the published number (E07-10 §16).
    """
    if tolerance < 0:
        raise ConfigError("the audit tolerance must be non-negative")
    problems: List[str] = []
    mappings = {
        "output_tps": "output_tps",
        "logical_token_tps": "logical_token_tps",
        "compute_position_tps": "compute_position_tps",
    }
    rows: List[Dict[str, Any]] = []
    for reported_key, recomputed_key in mappings.items():
        if reported_key not in reported:
            continue
        if recomputed_key not in raw:
            problems.append(
                f"{reported_key}: no recomputed value (a rate the ledger cannot "
                "reproduce must not be published)"
            )
            continue
        delta = float(reported[reported_key]) - float(raw[recomputed_key])
        if abs(delta) > tolerance * max(1.0, abs(float(raw[recomputed_key]))):
            problems.append(
                f"{reported_key}: reported {reported[reported_key]:.6g} vs "
                f"recomputed {raw[recomputed_key]:.6g} (delta {delta:.6g}); the "
                "runtime's own number is not reproducible from the raw ledger"
            )
        rows.append(
            {
                "metric": reported_key,
                "reported": float(reported[reported_key]),
                "recomputed": float(raw[recomputed_key]),
                "delta": delta,
            }
        )
    ledger = raw.get("token_ledger", {})
    if not ledger:
        problems.append(
            "the recomputation carries no token ledger; throughput without the "
            "three denominators cannot be audited (details README §13.5)"
        )
    return {
        "ok": not problems,
        "problems": problems,
        "rows": rows,
        "denominators": {
            key: ledger.get(key, 0)
            for key in (
                "logical_input_tokens",
                "useful_committed_tokens",
                "model_computed_positions",
            )
        },
    }


# ── cold / warm phases ─────────────────────────────────────────────────────

COLD_WARM_PHASES: Tuple[str, ...] = (
    "install_build",
    "model_load",
    "engine_conversion",
    "compile",
    "graph_capture",
    "first_request",
    "cache_hit",
    "steady",
    "close",
)

#: Phases excluded from steady-state timing but always reported.
REPORTED_ONLY_PHASES: Tuple[str, ...] = ("install_build",)


@dataclass(frozen=True)
class PhaseRecord:
    """One cold/warm phase of one backend."""

    phase: str
    backend_id: str
    wall_ms: float
    included_in_steady: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.phase not in COLD_WARM_PHASES:
            raise ConfigError(
                f"unknown phase {self.phase!r}",
                details={"allowed": list(COLD_WARM_PHASES)},
            )
        if self.phase in REPORTED_ONLY_PHASES and self.included_in_steady:
            raise ConfigError(
                f"{self.phase} may not be folded into steady-state timing; it must be "
                "reported separately (E07-10 §7)",
                details={"field": "included_in_steady"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "backend_id": self.backend_id,
            "wall_ms": self.wall_ms,
            "included_in_steady": self.included_in_steady,
            "note": self.note,
        }


def cold_warm_report(records: Sequence[PhaseRecord]) -> Dict[str, Any]:
    """Separate cold costs from steady state and require every phase."""
    by_backend: Dict[str, Dict[str, PhaseRecord]] = {}
    for record in records:
        by_backend.setdefault(record.backend_id, {})[record.phase] = record
    missing: Dict[str, List[str]] = {}
    for backend_id, phases in sorted(by_backend.items()):
        absent = [phase for phase in COLD_WARM_PHASES if phase not in phases]
        if absent:
            missing[backend_id] = absent
    return {
        "ok": not missing,
        "missing_phases": missing,
        "rows": [record.as_dict() for record in records],
        "reported_only": list(REPORTED_ONLY_PHASES),
    }


# ── statistics and Pareto ──────────────────────────────────────────────────


@dataclass(frozen=True)
class RunSamples:
    """Per-run values of one metric for one backend/workload/class."""

    backend_id: str
    workload: str
    request_class: str
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ConfigError("run samples must not be empty")

    def as_dict(self) -> Dict[str, Any]:
        summary = distribution_summary(self.values)
        return {
            "backend_id": self.backend_id,
            "workload": self.workload,
            "request_class": self.request_class,
            "runs": len(self.values),
            "summary": summary.as_dict(),
            "values": list(self.values),
        }


def compare_runs(
    left: RunSamples, right: RunSamples, *, metric_tolerance: float = 0.0
) -> Dict[str, Any]:
    """Paired comparison; a winner is only named when the CI excludes zero."""
    if (left.workload, left.request_class) != (right.workload, right.request_class):
        raise ConfigError(
            "paired comparison requires the same workload and request class; "
            "mixing classes is exactly the request-variance error §13.6 warns about",
            details={"field": "class"},
        )
    effect: PairedEffect = paired_effect(left.values, right.values)
    return {
        "left": left.backend_id,
        "right": right.backend_id,
        "workload": left.workload,
        "request_class": left.request_class,
        "effect": effect.as_dict(),
        "winner": "" if effect.crosses_zero else (left.backend_id if effect.mean_difference < 0 else right.backend_id),
        "tolerance": metric_tolerance,
    }


@dataclass(frozen=True)
class ParetoPoint:
    """One multi-objective point with its direction-corrected values."""

    backend_id: str
    workload: str
    hardware: str
    objectives: Mapping[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "workload": self.workload,
            "hardware": self.hardware,
            "objectives": dict(self.objectives),
        }


#: Objectives with their optimisation directions (smaller/larger is better).
PARETO_DIRECTIONS: Mapping[str, str] = {
    "memory_bytes": "min",
    "ttft_ms": "min",
    "tpot_ms": "min",
    "throughput": "max",
    "energy_j_per_token": "min",
}


def pareto_front(points: Sequence[ParetoPoint]) -> Dict[str, Any]:
    """Per hardware × workload front; cross-hardware mixing is refused."""
    groups: Dict[Tuple[str, str], List[ParetoPoint]] = {}
    for point in points:
        groups.setdefault((point.hardware, point.workload), []).append(point)
    fronts: List[Dict[str, Any]] = []
    for (hardware, workload), group in sorted(groups.items()):
        dominated: List[str] = []
        front: List[str] = []
        for candidate in group:
            if any(
                _dominates(other, candidate) for other in group if other is not candidate
            ):
                dominated.append(candidate.backend_id)
            else:
                front.append(candidate.backend_id)
        fronts.append(
            {
                "hardware": hardware,
                "workload": workload,
                "front": front,
                "dominated": dominated,
                "points": [point.as_dict() for point in group],
            }
        )
    return {
        "fronts": fronts,
        "note": (
            "fronts are computed per hardware × workload; a single global ranking "
            "would hide the trade-offs E07-10 is meant to expose"
        ),
    }


def _dominates(left: ParetoPoint, right: ParetoPoint) -> bool:
    better_or_equal = True
    strictly_better = False
    for objective, direction in PARETO_DIRECTIONS.items():
        if objective not in left.objectives or objective not in right.objectives:
            continue
        left_value = float(left.objectives[objective])
        right_value = float(right.objectives[objective])
        if direction == "min":
            if left_value > right_value:
                better_or_equal = False
            elif left_value < right_value:
                strictly_better = True
        else:
            if left_value < right_value:
                better_or_equal = False
            elif left_value > right_value:
                strictly_better = True
    return better_or_equal and strictly_better


def limitations_report(rows: Sequence[ComparisonRow]) -> Dict[str, Any]:
    """Collect the explicit limitations a reader must see next to the table."""
    na_rows = [row.as_dict() for row in rows if not row.comparable]
    observability = [
        {
            "backend": row.backend.backend_id,
            "limitation": row.observability_limitation,
        }
        for row in rows
        if row.observability_limitation
    ]
    tiers = sorted({row.tier for row in rows})
    return {
        "na_rows": na_rows,
        "observability_limitations": observability,
        "tiers_present": tiers,
        "claim_boundaries": {tier: TIER_CLAIMS[tier] for tier in tiers},
        "note": (
            "rows that could not be compared are kept with their reason; dropping "
            "them would turn a capability gap into a performance win"
        ),
    }


def s08_interface_surface() -> Dict[str, Any]:
    """The stable surface S08 may consume (no private runtime SDK objects)."""
    return {
        "backend_operations": [
            "load",
            "warmup",
            "generate",
            "stream",
            "cancel",
            "metrics",
            "close",
        ],
        "request_fields": ["request_id", "input_token_ids", "sampling", "stop"],
        "metrics": [
            "queue_delay_ms",
            "runtime_ttft_ms",
            "tpot_ms",
            "e2e_core_ms",
            "itl_ms",
            "output_tokens",
        ],
        "ledgers": ["requests", "iterations", "kv", "prefix", "graph"],
        "forbidden": [
            "private runtime objects",
            "runtime-internal identifiers",
            "self-reported aggregate TPS without the raw ledger",
        ],
        "note": (
            "S08 depends on this surface only; a private version difference stays "
            "inside the S07 adapter (details README §19)"
        ),
    }


def request_classes() -> Tuple[str, ...]:
    """Request classes a report must break results down by (E07-04 §6)."""
    return ("short", "long_prefill", "decode_heavy", "mixed", "cancelled", "failed")


__all__ = [
    "BackendIdentity",
    "COLD_WARM_PHASES",
    "COMMON_DENOMINATOR_FEATURES",
    "ComparisonRow",
    "ComparisonSpec",
    "PARETO_DIRECTIONS",
    "ParetoPoint",
    "PhaseRecord",
    "REPORTED_ONLY_PHASES",
    "RunSamples",
    "TIER_A",
    "TIER_B",
    "TIER_C",
    "TIER_D",
    "TIER_CLAIMS",
    "TIER_RULES",
    "best_valid_rows",
    "cold_warm_report",
    "common_denominator_rows",
    "compare_runs",
    "limitations_report",
    "pareto_front",
    "recompute_metrics",
    "request_classes",
    "s08_interface_surface",
    "percentile",
    "token_denominator_audit",
]
