"""Strict single-variable A/B of one runtime policy change (E07-07).

E07-07 is the experiment where the project earns the right to say "I changed the
runtime and this happened".  The module enforces the parts that are easy to get
wrong:

* :func:`selection_gate` — the change must come from the E07-02…E07-06 evidence
  (quantified bottleneck, explainable mechanism, pre-registered metric, no
  duplicate of an upstream feature, controllable risk, a constructible
  counterexample) *before* the patch is written.  "Write the patch first, then
  look for a friendly workload" is refused;
* :class:`AbIdentity` + :func:`identity_equal` — A and B differ only in the patch
  or the target policy;
* :func:`block_schedule` — ABBA / randomised blocks over independent processes;
* :func:`causal_chain_check` — end-to-end change plus proximate change plus phase
  change plus the ablation; an end-to-end difference without a proximate change
  is *not* attributable;
* :func:`decide` — the verdict follows the pre-registered primary metric and
  guardrails; a failed primary may not be replaced after the fact.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.runtime.metrics import PairedEffect, paired_effect

# ── ADR and selection gate (E07-07 §2–§4) ──────────────────────────────────

CANDIDATE_POLICIES: Tuple[str, ...] = (
    "kv_block_size_or_allocation",
    "token_budget_or_chunk",
    "prefix_eviction_or_key_lookup",
    "graph_bucket_selection",
    "attention_route_threshold",
    "watermark_or_admission",
)


@dataclass(frozen=True)
class SelectionGate:
    """The evidence a candidate change must satisfy before implementation."""

    baseline_bottleneck_quantified: bool
    mechanism_explainable: bool
    metric_preregistered: bool
    not_duplicating_upstream: bool
    risk_and_fallback_controllable: bool
    counterexample_constructible: bool
    scope_consistent_with_s08_s11: bool
    evidence_refs: Tuple[str, ...] = ()
    notes: str = ""

    @property
    def ok(self) -> bool:
        return all(
            (
                self.baseline_bottleneck_quantified,
                self.mechanism_explainable,
                self.metric_preregistered,
                self.not_duplicating_upstream,
                self.risk_and_fallback_controllable,
                self.counterexample_constructible,
                self.scope_consistent_with_s08_s11,
            )
        )

    def missing(self) -> List[str]:
        names = (
            "baseline_bottleneck_quantified",
            "mechanism_explainable",
            "metric_preregistered",
            "not_duplicating_upstream",
            "risk_and_fallback_controllable",
            "counterexample_constructible",
            "scope_consistent_with_s08_s11",
        )
        return [name for name in names if not getattr(self, name)]

    def require_ok(self) -> None:
        if not self.ok:
            raise ConfigError(
                "the change is not eligible: missing " + ", ".join(self.missing()) +
                ". A patch without pre-registered evidence produces correlation, not "
                "causality (E07-07 §2)",
                details={"fields": self.missing()},
            )
        if not self.evidence_refs:
            raise ConfigError(
                "the selection gate must cite the E07-02…E07-06 artefacts that "
                "justify the change",
                details={"field": "evidence_refs"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "missing": self.missing(),
            "evidence_refs": list(self.evidence_refs),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Adr:
    """The architectural decision record of one runtime change (E07-07 §4)."""

    decision_id: str
    candidate: str
    problem: str
    baseline_evidence: str
    hypothesis: str
    mechanism: str
    change_scope: str
    expected_benefit: str
    expected_regressions: Tuple[str, ...]
    invariants: Tuple[str, ...]
    metrics: Tuple[str, ...]
    workloads: Tuple[str, ...]
    rollback: str
    stop_rule: str
    non_goals: Tuple[str, ...]
    quality_and_safety: str = ""

    def __post_init__(self) -> None:
        if self.candidate not in CANDIDATE_POLICIES:
            raise ConfigError(
                f"unknown candidate policy {self.candidate!r}",
                details={"allowed": list(CANDIDATE_POLICIES)},
            )
        for name in (
            "decision_id",
            "problem",
            "baseline_evidence",
            "hypothesis",
            "mechanism",
            "change_scope",
            "rollback",
            "stop_rule",
        ):
            if not getattr(self, name):
                raise ConfigError(
                    f"the ADR needs {name!r}; without it the change cannot be audited",
                    details={"field": name},
                )
        if not self.metrics:
            raise ConfigError("the ADR must pre-register the metrics")
        if not self.workloads:
            raise ConfigError("the ADR must pre-register the workloads")
        if not self.invariants:
            raise ConfigError("the ADR must state the invariants the change preserves")
        if not self.rollback:
            raise ConfigError("the ADR must state a one-command rollback")

    @property
    def primary_metric(self) -> str:
        return self.metrics[0]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "candidate": self.candidate,
            "problem": self.problem,
            "baseline_evidence": self.baseline_evidence,
            "hypothesis": self.hypothesis,
            "mechanism": self.mechanism,
            "change_scope": self.change_scope,
            "expected_benefit": self.expected_benefit,
            "expected_regressions": list(self.expected_regressions),
            "invariants": list(self.invariants),
            "metrics": list(self.metrics),
            "workloads": list(self.workloads),
            "rollback": self.rollback,
            "stop_rule": self.stop_rule,
            "non_goals": list(self.non_goals),
            "quality_and_safety": self.quality_and_safety,
            "primary_metric": self.primary_metric,
        }


# ── A/B identity and schedule ──────────────────────────────────────────────


@dataclass(frozen=True)
class AbIdentity:
    """Everything that must be identical between arm A and arm B."""

    model_id: str
    tokenizer_id: str
    precision: str
    runtime_base_commit: str
    hardware: str
    request_trace_hash: str
    scheduler_config_hash: str
    kv_graph_attention_config_hash: str
    warmup_policy: str
    measurement_policy: str
    seed: int
    patch_hash: str
    build_hash: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "precision": self.precision,
            "runtime_base_commit": self.runtime_base_commit,
            "hardware": self.hardware,
            "request_trace_hash": self.request_trace_hash,
            "scheduler_config_hash": self.scheduler_config_hash,
            "kv_graph_attention_config_hash": self.kv_graph_attention_config_hash,
            "warmup_policy": self.warmup_policy,
            "measurement_policy": self.measurement_policy,
            "seed": self.seed,
            "patch_hash": self.patch_hash,
            "build_hash": self.build_hash,
        }


def identity_equal(left: AbIdentity, right: AbIdentity) -> Dict[str, Any]:
    """Compare two arm identities; the patch/build are the *only* allowed diff."""
    allowed = ("patch_hash", "build_hash")
    differing = [
        name
        for name in left.as_dict()
        if getattr(left, name) != getattr(right, name) and name not in allowed
    ]
    return {
        "ok": not differing,
        "differing_fields": differing,
        "allowed_differences": list(allowed),
        "patch_differs": left.patch_hash != right.patch_hash,
        "build_differs": left.build_hash != right.build_hash,
    }


def require_identity_equal(left: AbIdentity, right: AbIdentity) -> None:
    report = identity_equal(left, right)
    if not report["ok"]:
        raise ConfigError(
            "A and B differ in fields other than the patch/build: "
            + ", ".join(report["differing_fields"]),
            details={"fields": report["differing_fields"]},
        )
    if not report["patch_differs"] and not report["build_differs"]:
        raise ConfigError(
            "A and B are identical artifacts: there is no variable to test",
            details={"field": "patch_hash"},
        )


def block_schedule(
    *, blocks: int, scheme: str = "ABBA", seed: Optional[int] = None
) -> List[str]:
    """Independent-process run order; ABBA by default, randomised on request."""
    if blocks < 3:
        raise ConfigError(
            "at least three blocks/processes are required (E07-07 §6)",
            details={"field": "blocks"},
        )
    if scheme == "ABBA":
        order = ["A", "B", "B", "A"]
    elif scheme == "RANDOM_BLOCKS":
        if seed is None:
            raise ConfigError("a randomised block design needs a seed to be replayable")
        rng = random.Random(seed)
        order = []
        for index in range(blocks):
            pair = ["A", "B"]
            rng.shuffle(pair)
            order.extend(pair)
    else:
        raise ConfigError(
            f"unknown schedule scheme {scheme!r}; S07 uses ABBA or randomised blocks",
            details={"field": "scheme"},
        )
    # Ensure exactly `blocks` runs per arm (or the closest ABBA multiple).
    per_arm = max(len(order) // 2, 1)
    while per_arm < blocks:
        order = order + order
        per_arm *= 2
    return order


def schedule_balance(order: Sequence[str]) -> Dict[str, Any]:
    """A schedule must be balanced and interleaved, not AAAA BBBB."""
    if not order:
        raise ConfigError("the schedule is empty")
    arm_a = order.count("A")
    arm_b = order.count("B")
    runs = max(
        len(segment)
        for segment in _segments(order)
    )
    return {
        "ok": arm_a == arm_b and runs <= 2,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "longest_run": runs,
        "order": list(order),
        "note": (
            "a long consecutive run of one arm confounds the comparison with "
            "thermal/host drift"
        ),
    }


def _segments(order: Sequence[str]) -> List[List[str]]:
    segments: List[List[str]] = []
    for arm in order:
        if segments and segments[-1][0] == arm:
            segments[-1].append(arm)
        else:
            segments.append([arm])
    return segments


# ── effect, causal chain, regression envelope ──────────────────────────────


@dataclass(frozen=True)
class AbEffect:
    """Paired effect of the primary metric with its guard band.

    Convention (fixed so the sign cannot be argued after the fact): the paired
    effect is ``mean(arm_a) - mean(arm_b)`` where **A is the baseline and B is the
    candidate**.  ``improvement`` is then that difference expressed in the metric's
    optimisation direction: for ``better="lower"`` an improvement means the
    candidate is smaller (positive difference), for ``better="higher"`` it means
    the candidate is larger (negative difference).
    """

    metric: str
    effect: PairedEffect
    guard_band: float
    better: str

    def __post_init__(self) -> None:
        if self.guard_band < 0:
            raise ConfigError("the guard band must be non-negative")
        if self.better not in ("lower", "higher"):
            raise ConfigError("'better' must state the optimisation direction")

    @property
    def significant_benefit(self) -> bool:
        """The whole improvement interval is beyond the guard band."""
        if self.better == "lower":
            return self.effect.ci_low > self.guard_band
        return self.effect.ci_high < -self.guard_band

    @property
    def significant_regression(self) -> bool:
        """The whole interval shows the candidate getting worse."""
        if self.better == "lower":
            return self.effect.ci_high < -self.guard_band
        return self.effect.ci_low > self.guard_band

    @property
    def ci_low(self) -> float:
        return self.effect.ci_low

    @property
    def ci_high(self) -> float:
        return self.effect.ci_high

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "better": self.better,
            "guard_band": self.guard_band,
            "effect": self.effect.as_dict(),
            "significant_benefit": self.significant_benefit,
            "significant_regression": self.significant_regression,
        }


def measure_ab(
    *,
    metric: str,
    arm_a_values: Sequence[float],
    arm_b_values: Sequence[float],
    guard_band: float,
    better: str,
) -> AbEffect:
    return AbEffect(
        metric=metric,
        effect=paired_effect(arm_a_values, arm_b_values),
        guard_band=guard_band,
        better=better,
    )


def causal_chain_check(
    *,
    patch_changed_mechanism: bool,
    proximate_metric_changed: bool,
    phase_metric_changed: bool,
    end_to_end_changed: bool,
    ablation_isolates_mechanism: bool,
) -> Dict[str, Any]:
    """Four conditions must hold together for a causal claim (E07-07 §2/§18)."""
    problems: List[str] = []
    if not patch_changed_mechanism:
        problems.append("the patch did not change the predicted internal mechanism")
    if end_to_end_changed and not proximate_metric_changed:
        problems.append(
            "the end-to-end metric changed while the proximate metric did not: the "
            "difference is not attributable to this patch"
        )
    if proximate_metric_changed and not end_to_end_changed:
        problems.append(
            "the proximate metric improved but the end-to-end metric did not; this "
            "is an Amdahl limit (a valid PASS_NEGATIVE, not a benefit)"
        )
    if not ablation_isolates_mechanism:
        problems.append(
            "the ablation does not isolate the mechanism; another change could "
            "explain the result"
        )
    if not phase_metric_changed:
        problems.append("no phase-level metric moved; the causal chain has a gap")
    return {
        "attributable": not problems,
        "problems": problems,
        "conditions": {
            "patch_changed_mechanism": patch_changed_mechanism,
            "proximate_metric_changed": proximate_metric_changed,
            "phase_metric_changed": phase_metric_changed,
            "end_to_end_changed": end_to_end_changed,
            "ablation_isolates_mechanism": ablation_isolates_mechanism,
        },
    }


@dataclass(frozen=True)
class RegressionRow:
    """One workload slice of the regression envelope."""

    workload: str
    request_class: str
    effect: PairedEffect
    guardrail: float

    @property
    def regression(self) -> bool:
        """The whole interval exceeds the guardrail (a real regression)."""
        return self.effect.ci_low > self.guardrail

    def as_dict(self) -> Dict[str, Any]:
        return {
            "workload": self.workload,
            "request_class": self.request_class,
            "guardrail": self.guardrail,
            "regression": self.regression,
            "effect": self.effect.as_dict(),
        }


def regression_envelope(rows: Sequence[RegressionRow]) -> Dict[str, Any]:
    """Which slices get worse, by how much, and what the fallback is."""
    if not rows:
        raise ConfigError("a regression envelope needs at least one slice")
    return {
        "rows": [row.as_dict() for row in rows],
        "regressions": [
            f"{row.workload}/{row.request_class}" for row in rows if row.regression
        ],
        "note": (
            "a candidate that is faster on average but regresses a class beyond its "
            "guardrail must be rejected or gated by a capability check"
        ),
    }


def ablation_matrix(mechanisms: Sequence[str]) -> List[Dict[str, Any]]:
    """One ablation row per sub-mechanism (E07-07 step 16)."""
    if not mechanisms:
        raise ConfigError("the ablation matrix needs at least one mechanism")
    return [
        {
            "mechanism": mechanism,
            "disabled": True,
            "expected": "the proximate metric returns to the baseline level",
        }
        for mechanism in mechanisms
    ]


# ── decision ───────────────────────────────────────────────────────────────

DECISIONS: Tuple[str, ...] = (
    "MERGE",
    "MERGE_WITH_GATE",
    "ROLLBACK",
    "PASS_NEGATIVE",
    "INCONCLUSIVE",
)


def decide(
    *,
    adr: Adr,
    effect: AbEffect,
    causal: Mapping[str, Any],
    regressions: Mapping[str, Any],
    correctness_ok: bool,
    safety_ok: bool,
) -> Dict[str, Any]:
    """Verdict from the pre-registered primary metric and guardrails only."""
    if effect.metric != adr.primary_metric:
        raise ConfigError(
            f"the measured metric {effect.metric!r} is not the pre-registered "
            f"primary {adr.primary_metric!r}; changing the primary after seeing "
            "results is prohibited (E07-07 §10 step 20)",
            details={"field": "metric"},
        )
    if not correctness_ok or not safety_ok:
        return {
            "decision": "ROLLBACK",
            "reason": (
                "correctness/safety failed; performance evidence is not usable "
                "(performance claims stop at the first correctness failure)"
            ),
            "effect": effect.as_dict(),
            "causal": dict(causal),
            "regressions": dict(regressions),
        }
    if regressions.get("regressions"):
        return {
            "decision": "ROLLBACK",
            "reason": "guardrail regression in " + ", ".join(regressions["regressions"]),
            "effect": effect.as_dict(),
            "causal": dict(causal),
            "regressions": dict(regressions),
        }
    if effect.significant_benefit and causal.get("attributable"):
        return {
            "decision": "MERGE",
            "reason": "",
            "effect": effect.as_dict(),
            "causal": dict(causal),
            "regressions": dict(regressions),
        }
    if causal.get("attributable") and not effect.significant_benefit:
        return {
            "decision": "PASS_NEGATIVE",
            "reason": (
                "the mechanism changed and was explained, but the end-to-end effect "
                "is not beyond the guard band: a legitimate negative result, not a "
                "benefit"
            ),
            "effect": effect.as_dict(),
            "causal": dict(causal),
            "regressions": dict(regressions),
        }
    return {
        "decision": "INCONCLUSIVE",
        "reason": "the causal chain is incomplete: " + "; ".join(causal.get("problems", [])),
        "effect": effect.as_dict(),
        "causal": dict(causal),
        "regressions": dict(regressions),
    }


def pilot_separation(pilot_runs: Sequence[str], final_runs: Sequence[str]) -> Dict[str, Any]:
    """Pilot data must not enter the final matrix (E07-07 §10 step 9)."""
    overlap = sorted(set(pilot_runs) & set(final_runs))
    return {
        "ok": not overlap,
        "overlapping_runs": overlap,
        "pilot_runs": list(pilot_runs),
        "final_runs": list(final_runs),
    }


def rollback_plan(adr: Adr) -> Dict[str, Any]:
    """The change must be revertible in one command with the reference path intact."""
    return {
        "decision_id": adr.decision_id,
        "rollback": adr.rollback,
        "feature_flag_required": True,
        "reference_path_preserved": True,
        "note": (
            "a change without a feature flag and a reference path cannot be merged "
            "even when it is faster (E07-07 §13)"
        ),
    }


__all__ = [
    "AbEffect",
    "AbIdentity",
    "Adr",
    "CANDIDATE_POLICIES",
    "DECISIONS",
    "RegressionRow",
    "SelectionGate",
    "ablation_matrix",
    "block_schedule",
    "causal_chain_check",
    "decide",
    "identity_equal",
    "measure_ab",
    "pilot_separation",
    "regression_envelope",
    "require_identity_equal",
    "rollback_plan",
    "schedule_balance",
]
