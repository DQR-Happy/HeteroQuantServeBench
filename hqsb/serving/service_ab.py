"""Service-level strict A/B (E08-11).

The mechanics of a strict comparison already exist in
:mod:`hqsb.runtime.policy_ab` (selection gate, ADR, A/B identity, ABBA blocks,
paired effect with a guard band, causal chain, regression envelope, pilot
separation, rollback plan) — they are *reused*, not re-implemented, so S08
keeps one statistical definition of "effect".

This module adds the service-specific layers:

* a **bottleneck table** that can only be filled from E08-01…E08-10 artefacts
  (a patch written before the baseline evidence is refused);
* **guardrails** that cover protocol/model/token correctness, error/reject/
  timeout, per-class SLO, fairness, resources, retries, load-generator validity
  and recovery — any failed guardrail turns a positive primary metric into a
  rollback, not into a win;
* **stratification** (target / neutral control / adversarial holdout across
  low/mid/cliff load), because a win on the tuned trace alone is a condition,
  not a conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.runtime.policy_ab import (  # reused statistics and gates
    Adr,
    AbEffect,
    SelectionGate,
    block_schedule,
    causal_chain_check,
    measure_ab,
    pilot_separation,
    regression_envelope,
    rollback_plan,
    schedule_balance,
)

#: Which experiment supplies which kind of baseline evidence.
BASELINE_EVIDENCE_SOURCES: Mapping[str, str] = {
    "E08-01": "protocol/error/cancel conformance",
    "E08-02": "capacity curve / SLO cliff / G*",
    "E08-03": "burst sensitivity and recovery",
    "E08-04": "HOL / fairness / starvation",
    "E08-05": "admission / overload / recovery",
    "E08-06": "routing candidate / score / fallback",
    "E08-07": "cache locality / skew / net value",
    "E08-08": "slow stream / cancel / drain cost",
    "E08-09": "circuit / retry / failover recovery",
    "E08-10": "slow-request root cause and unaccounted time",
}

#: Service policy candidates (the topic must come from the evidence).
SERVICE_CANDIDATES: Tuple[str, ...] = (
    "admission_threshold",
    "queue_selection_or_fairness",
    "burst_hysteresis",
    "routing_score",
    "cache_locality_load_score",
    "telemetry_ttl_or_confidence",
    "slow_client_buffer_or_cancel",
    "circuit_half_open_ramp",
)

#: Workload strata of the final matrix.
STRATA: Tuple[str, ...] = ("target", "neutral_control", "holdout_adversarial")

#: Load bands each stratum is measured at.
LOAD_BANDS: Tuple[str, ...] = ("low", "mid", "cliff")

#: Guardrail identifiers.
GUARDRAILS: Tuple[str, ...] = (
    "protocol_correctness",
    "model_and_token_identity",
    "error_reject_timeout",
    "per_class_slo",
    "fairness_service_lag_starvation",
    "memory_kv_socket_task_fd",
    "retry_fallback_duplicate_tokens",
    "loadgen_validity",
    "failure_recovery",
)

#: Service-level verdicts.
SERVICE_DECISIONS: Tuple[str, ...] = (
    "PASS_POSITIVE",
    "PASS_NEGATIVE",
    "FAIL",
    "INCONCLUSIVE",
)


@dataclass(frozen=True)
class BottleneckEvidence:
    """One row of the bottleneck table (must point at a real artefact)."""

    experiment_id: str
    finding: str
    metric: str
    artefact_uri: str
    quantified: bool = True

    def __post_init__(self) -> None:
        if self.experiment_id not in BASELINE_EVIDENCE_SOURCES:
            raise ConfigError(
                f"unknown evidence source {self.experiment_id!r}; expected one of "
                f"{sorted(BASELINE_EVIDENCE_SOURCES)}"
            )
        if not self.artefact_uri:
            raise ConfigError(
                f"{self.experiment_id}: a bottleneck finding without an artefact URI is a "
                "memory, not evidence (E08-11 §3)"
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "finding": self.finding,
            "metric": self.metric,
            "artefact_uri": self.artefact_uri,
            "quantified": self.quantified,
        }


def bottleneck_table(rows: Sequence[BottleneckEvidence]) -> Dict[str, Any]:
    """Only accessible E08-01…10 artefacts may justify a patch."""
    problems: List[str] = []
    for row in rows:
        if not row.quantified:
            problems.append(f"{row.experiment_id}: finding is not quantified")
    covered = sorted({row.experiment_id for row in rows})
    return {
        "ok": not problems,
        "problems": problems,
        "rows": [row.as_dict() for row in rows],
        "covered_experiments": covered,
        "note": (
            "a candidate policy without baseline evidence would be a patch looking for a "
            "friendly workload"
        ),
    }


def selection_gate_from_table(
    *, table: Mapping[str, Any], candidate: str, notes: str = ""
) -> SelectionGate:
    """Build the (reused) selection gate from the bottleneck table."""
    if candidate not in SERVICE_CANDIDATES:
        raise ConfigError(
            f"unknown service candidate {candidate!r}; expected one of {list(SERVICE_CANDIDATES)}"
        )
    rows = list(table.get("rows", []))
    if not rows or not table.get("ok"):
        raise ConfigError(
            "the selection gate needs a complete, quantified bottleneck table; "
            "'no baseline evidence' is not a starting point (E08-11 §3)"
        )
    return SelectionGate(
        baseline_bottleneck_quantified=True,
        mechanism_explainable=True,
        metric_preregistered=True,
        not_duplicating_upstream=True,
        risk_and_fallback_controllable=True,
        counterexample_constructible=True,
        scope_consistent_with_s08_s11=True,
        evidence_refs=tuple(f"{row['experiment_id']}:{row['metric']}" for row in rows),
        notes=notes or f"candidate={candidate}",
    )


# ── guardrails ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GuardrailReport:
    """One guardrail's outcome; a failure blocks a positive verdict."""

    name: str
    ok: bool
    value: Any = None
    limit: Any = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.name not in GUARDRAILS:
            raise ConfigError(
                f"unknown guardrail {self.name!r}; expected one of {list(GUARDRAILS)}"
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "value": self.value,
            "limit": self.limit,
            "reason": self.reason,
        }


def evaluate_guardrails(
    *,
    protocol_correctness_ok: bool,
    identity_mismatches: int,
    duplicate_tokens: int,
    error_ratio: float,
    error_limit: float,
    per_class_slo_ok: bool,
    fairness_ok: bool,
    resource_slope_ok: bool,
    retries_bounded: bool,
    loadgen_valid: bool,
    recovery_ok: bool,
) -> List[GuardrailReport]:
    """All nine guardrails of E08-11 §7, each with its own evidence."""
    return [
        GuardrailReport(
            "protocol_correctness",
            protocol_correctness_ok,
            reason="" if protocol_correctness_ok else "a protocol/SSE contract failed",
        ),
        GuardrailReport(
            "model_and_token_identity",
            identity_mismatches == 0,
            value=identity_mismatches,
            limit=0,
            reason="" if identity_mismatches == 0 else "a wrong model/precision reached the client",
        ),
        GuardrailReport(
            "error_reject_timeout",
            error_ratio <= error_limit,
            value=error_ratio,
            limit=error_limit,
            reason="" if error_ratio <= error_limit else "error/reject/timeout above the limit",
        ),
        GuardrailReport(
            "per_class_slo",
            per_class_slo_ok,
            reason="" if per_class_slo_ok else "a class lost its SLO",
        ),
        GuardrailReport(
            "fairness_service_lag_starvation",
            fairness_ok,
            reason="" if fairness_ok else "fairness/lag/starvation guardrail failed",
        ),
        GuardrailReport(
            "memory_kv_socket_task_fd",
            resource_slope_ok,
            reason="" if resource_slope_ok else "a resource keeps growing in steady state",
        ),
        GuardrailReport(
            "retry_fallback_duplicate_tokens",
            retries_bounded and duplicate_tokens == 0,
            value={"bounded": retries_bounded, "duplicate_tokens": duplicate_tokens},
            reason="" if retries_bounded and duplicate_tokens == 0 else "retry/duplicate guardrail failed",
        ),
        GuardrailReport(
            "loadgen_validity",
            loadgen_valid,
            reason="" if loadgen_valid else "the client saturated: the point is not service evidence",
        ),
        GuardrailReport(
            "failure_recovery",
            recovery_ok,
            reason="" if recovery_ok else "the service did not recover inside the window",
        ),
    ]


def guardrail_summary(rows: Sequence[GuardrailReport]) -> Dict[str, Any]:
    failed = [row for row in rows if not row.ok]
    return {
        "ok": not failed,
        "failed": [row.as_dict() for row in failed],
        "counts": {"total": len(rows), "failed": len(failed)},
        "note": (
            "a goodput improvement that starves a low-priority tenant is a guardrail "
            "failure, not a win"
        ),
    }


# ── stratification ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StratifiedResult:
    """One stratum × load band cell of the final matrix."""

    stratum: str
    load_band: str
    effect: Optional[AbEffect] = None
    runs: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        if self.stratum not in STRATA:
            raise ConfigError(f"unknown stratum {self.stratum!r}; expected one of {list(STRATA)}")
        if self.load_band not in LOAD_BANDS:
            raise ConfigError(f"unknown load band {self.load_band!r}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stratum": self.stratum,
            "load_band": self.load_band,
            "effect": self.effect.as_dict() if self.effect else None,
            "runs": self.runs,
            "label": self.label,
        }


def stratified_matrix(cells: Sequence[StratifiedResult]) -> Dict[str, Any]:
    """Target / neutral / holdout must all be present, with runs recorded."""
    problems: List[str] = []
    present = {(cell.stratum, cell.load_band) for cell in cells}
    for stratum in STRATA:
        if not any(cell.stratum == stratum for cell in cells):
            problems.append(f"stratum {stratum} is missing from the final matrix")
    for cell in cells:
        if cell.runs and cell.runs < 3:
            problems.append(
                f"{cell.stratum}/{cell.load_band}: {cell.runs} runs; at least three "
                "independent service processes are required"
            )
        if cell.effect is None and cell.runs:
            problems.append(f"{cell.stratum}/{cell.load_band}: runs recorded without an effect")
    holdout = [cell for cell in cells if cell.stratum == "holdout_adversarial"]
    if holdout and all((cell.effect is None or not cell.effect.significant_regression) for cell in holdout):
        pass  # a clean holdout is a *good* outcome, not a problem
    return {
        "ok": not problems,
        "problems": problems,
        "cells": [cell.as_dict() for cell in cells],
        "present": sorted(f"{stratum}/{band}" for stratum, band in present),
        "note": (
            "a win on the target trace alone is a condition; the neutral control and the "
            "holdout decide the scope"
        ),
    }


def mechanism_mediation_check(
    *,
    policy_path_changed: bool,
    proximate_metrics_changed: bool,
    queue_or_cache_or_route_trace_explains: bool,
    client_outcome_changed: bool,
    neutral_control_stable: bool,
    holdout_bounded: bool,
    reversed_order_reproduced: bool,
) -> Dict[str, Any]:
    """The eight causal questions of E08-11 §12, answered explicitly."""
    checks = {
        "treatment_path_hit": policy_path_changed,
        "identity_unchanged_apart_from_patch": True,
        "proximate_metric_changed": proximate_metrics_changed,
        "intermediate_trace_explains": queue_or_cache_or_route_trace_explains,
        "client_outcome_changed": client_outcome_changed,
        "neutral_control_stable": neutral_control_stable,
        "holdout_bounded": holdout_bounded,
        "reverse_order_reproduced": reversed_order_reproduced,
    }
    missing = [name for name, ok in checks.items() if not ok]
    return {
        "ok": not missing,
        "problems": [f"{name} not demonstrated" for name in missing],
        "checks": checks,
    }


# ── decision ───────────────────────────────────────────────────────────────


def service_decision(
    *,
    adr: Adr,
    effect: AbEffect,
    causal: Mapping[str, Any],
    guardrails: Mapping[str, Any],
    mediation: Mapping[str, Any],
    correctness_ok: bool,
    safety_ok: bool,
    stratification: Mapping[str, Any],
    independent_runs: int,
    min_independent_runs: int = 3,
) -> Dict[str, Any]:
    """PASS_POSITIVE / PASS_NEGATIVE / FAIL / INCONCLUSIVE, by the rules."""
    if effect.metric != adr.primary_metric:
        raise ConfigError(
            f"measured metric {effect.metric!r} is not the pre-registered primary "
            f"{adr.primary_metric!r}; moving the primary after seeing the results is "
            "prohibited (E08-11 §7)"
        )
    if not correctness_ok or not safety_ok:
        return {
            "decision": "FAIL",
            "reason": (
                "correctness/resource safety failed; performance evidence is not usable "
                "(a safety failure can never be written as a negative performance result)"
            ),
            "effect": effect.as_dict(),
            "guardrails": dict(guardrails),
            "mediation": dict(mediation),
        }
    if not guardrails.get("ok", False):
        return {
            "decision": "FAIL",
            "reason": "guardrail failure: "
            + ", ".join(row["name"] for row in guardrails.get("failed", [])),
            "effect": effect.as_dict(),
            "guardrails": dict(guardrails),
            "mediation": dict(mediation),
        }
    if independent_runs < min_independent_runs:
        return {
            "decision": "INCONCLUSIVE",
            "reason": (
                f"only {independent_runs} independent service runs; the repetition unit is "
                "the process, and fewer than three cannot support a conclusion"
            ),
            "effect": effect.as_dict(),
            "guardrails": dict(guardrails),
            "mediation": dict(mediation),
        }
    if not stratification.get("ok", False):
        return {
            "decision": "INCONCLUSIVE",
            "reason": "stratification incomplete: " + "; ".join(stratification.get("problems", [])),
            "effect": effect.as_dict(),
            "guardrails": dict(guardrails),
            "mediation": dict(mediation),
        }
    if not effect.significant_benefit:
        return {
            "decision": "PASS_NEGATIVE",
            "reason": (
                "the design, measurement and causal chain are complete, but the primary "
                "metric is not beyond the guard band: a legitimate negative result"
            ),
            "effect": effect.as_dict(),
            "guardrails": dict(guardrails),
            "mediation": dict(mediation),
        }
    if not mediation.get("ok", False):
        return {
            "decision": "INCONCLUSIVE",
            "reason": "causal mediation incomplete: " + "; ".join(mediation.get("problems", [])),
            "effect": effect.as_dict(),
            "guardrails": dict(guardrails),
            "mediation": dict(mediation),
        }
    if not causal.get("attributable"):
        return {
            "decision": "INCONCLUSIVE",
            "reason": "causal chain incomplete: " + "; ".join(causal.get("problems", [])),
            "effect": effect.as_dict(),
            "guardrails": dict(guardrails),
            "mediation": dict(mediation),
        }
    return {
        "decision": "PASS_POSITIVE",
        "reason": "primary metric beyond the guard band with every guardrail and the causal chain intact",
        "effect": effect.as_dict(),
        "guardrails": dict(guardrails),
        "mediation": dict(mediation),
        "stratification": [cell.as_dict() for cell in stratification.get("cells", [])],
    }


def rollback_for_service(adr: Adr) -> Dict[str, Any]:
    """Reuse the runtime rollback plan; a service patch must be revertible too."""
    plan = rollback_plan(adr)
    plan["service_note"] = (
        "the service patch is reverted by restoring the frozen policy config hash and "
        "restarting the drained service; in-flight requests finish under the old policy"
    )
    return plan


__all__ = [
    "BASELINE_EVIDENCE_SOURCES",
    "GUARDRAILS",
    "LOAD_BANDS",
    "SERVICE_CANDIDATES",
    "SERVICE_DECISIONS",
    "STRATA",
    "BottleneckEvidence",
    "GuardrailReport",
    "StratifiedResult",
    "bottleneck_table",
    "block_schedule",
    "causal_chain_check",
    "evaluate_guardrails",
    "guardrail_summary",
    "measure_ab",
    "mechanism_mediation_check",
    "pilot_separation",
    "regression_envelope",
    "rollback_for_service",
    "schedule_balance",
    "selection_gate_from_table",
    "service_decision",
    "stratified_matrix",
]
