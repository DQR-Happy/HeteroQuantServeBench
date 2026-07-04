"""E13-10: correctness/performance canary, automatic gates, promotion and rollback.

Implements ``details/S13/E13-10_*.md`` as data:

* the canary state machine of §2 with machine-driven transitions and audited
  manual overrides (an override never counts as an automatic pass);
* the traffic assignment units of §5 including session stickiness — a KV/prefix
  cache cannot be split per token;
* the gate hierarchy of §4: ``G0``–``G4`` are hard gates that stop a candidate
  immediately, so a later performance win can never compensate a correctness or
  supply-chain failure;
* the statistics of §6: a practical regression ``δ``, intervals, a sequential
  boundary and an explicit ``INCONCLUSIVE`` outcome instead of peeking at a
  p-value every minute;
* the pre-registered risk budget (traffic fraction, requests/tokens, duration,
  error-budget spend) and the exposure stages;
* control/candidate comparability (case mix, placement, cache, autoscaling) and
  the guardrail that cold/warm states may not be compared;
* rollback closure: traffic stopped, drain, active identity restored, residual
  candidate objects cleaned up.

Nothing here deploys a candidate or routes traffic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.infra import identity as idn
from hqsb.infra import records as rec

EXPERIMENT_ID = "E13-10"
TITLE = "Correctness/Performance Canary、自动 Gate、渐进发布与回滚"
CLAIM_BOUNDARY = (
    "本实验通过证明已测退化类别可由当前 canary policy 安全治理；不证明未知退化、极低频长尾或区域级"
    "发布风险被消除。"
)

SCHEMA_VERSION = "1.0.0"

#: State machine (§2) — states/terminal statuses come from ``records``.
STATES: Tuple[str, ...] = rec.CANARY_STATES
DECISIONS: Tuple[str, ...] = rec.CANARY_DECISIONS
GATES: Tuple[str, ...] = rec.CANARY_GATES
HARD_GATES: Tuple[str, ...] = rec.CANARY_HARD_GATES
ASSIGNMENT_UNITS: Tuple[str, ...] = rec.ASSIGNMENT_UNITS

#: Promotion path (stage names in order).
PROMOTION_PATH: Tuple[str, ...] = (
    "CANDIDATE_REGISTERED",
    "SUPPLY_CHAIN_PASSED",
    "OFFLINE_CORRECTNESS_QUALITY_PASSED",
    "DEPLOYED_NOT_READY",
    "WARMED_READY_NO_TRAFFIC",
    "SHADOW_OR_SMOKE",
    "CANARY_EXPOSURE_STAGE_1",
    "CANARY_EXPOSURE_STAGE_2",
    "CANARY_EXPOSURE_STAGE_3",
    "PROMOTED",
)

#: Failure path (any stage may enter it).
FAILURE_PATH: Tuple[str, ...] = (
    "STOPPED_FAILED",
    "ROLLBACK_INITIATED",
    "CONTROL_RESTORED",
    "POSTCHECK",
)

CANARY_TRANSITIONS: Tuple[Tuple[str, str], ...] = tuple(
    (PROMOTION_PATH[index], PROMOTION_PATH[index + 1]) for index in range(len(PROMOTION_PATH) - 1)
) + tuple(
    (PROMOTION_PATH[index], "STOPPED_FAILED")
    for index in range(1, len(PROMOTION_PATH) - 1)
) + (
    ("CANDIDATE_REGISTERED", "EXPIRED"),
    ("SHADOW_OR_SMOKE", "INCONCLUSIVE_HOLD"),
    ("CANARY_EXPOSURE_STAGE_1", "INCONCLUSIVE_HOLD"),
    ("CANARY_EXPOSURE_STAGE_2", "INCONCLUSIVE_HOLD"),
    ("INCONCLUSIVE_HOLD", "STOPPED_FAILED"),
    ("INCONCLUSIVE_HOLD", "CANARY_EXPOSURE_STAGE_1"),
    ("STOPPED_FAILED", "ROLLBACK_INITIATED"),
    ("ROLLBACK_INITIATED", "CONTROL_RESTORED"),
    ("CONTROL_RESTORED", "POSTCHECK"),
)

CANARY_STATE_MACHINE = rec.StateMachine(
    "canary_release", STATES, CANARY_TRANSITIONS, terminal_states=("PROMOTED", "POSTCHECK", "EXPIRED")
)

#: Candidate kinds of step 7 (bad candidates have ground truth; good ones must pass).
CANDIDATE_KINDS: Tuple[str, ...] = (
    "GOOD_EQUIVALENT",
    "GOOD_IMPROVED",
    "CORRECTNESS_BAD",
    "PERFORMANCE_BAD",
    "RESOURCE_BAD",
    "FALLBACK_BAD",
)

#: Metrics with their comparison direction (§10 step 5).
METRIC_DIRECTIONS: Mapping[str, str] = {
    "quality_pass_rate": "higher_is_better",
    "correctness_error_rate": "lower_is_better",
    "error_rate": "lower_is_better",
    "reject_rate": "lower_is_better",
    "timeout_rate": "lower_is_better",
    "partial_stream_rate": "lower_is_better",
    "retry_rate": "lower_is_better",
    "ttft_seconds": "lower_is_better",
    "tpot_seconds": "lower_is_better",
    "e2e_seconds": "lower_is_better",
    "slo_goodput": "higher_is_better",
    "memory_bytes": "lower_is_better",
    "kv_bytes": "lower_is_better",
    "oom_events": "lower_is_better",
    "leak_rate_bytes_per_request": "lower_is_better",
    "power_watts": "lower_is_better",
    "energy_joules_per_token": "lower_is_better",
    "replica_minutes": "lower_is_better",
}

#: Hard-gate failure consequences (§6).
HARD_GATE_ACTION = "STOP_IMMEDIATELY"

#: Overrides must be audited with an expiry (step 37).
OVERRIDE_ACTIONS: Tuple[str, ...] = ("HOLD", "ABORT", "ROLLBACK", "FORCE_PROMOTE")
FORBIDDEN_OVERRIDE_ACTIONS: Tuple[str, ...] = ("FORCE_PROMOTE",)


# ── policy, assignment and gates ─────────────────────────────────────────


@dataclass
class CanaryPolicy:
    """Steps 2–6: the frozen state machine, gates, statistics and risk budget."""

    policy_id: str
    stages: Tuple[float, ...] = (0.01, 0.05, 0.25, 1.0)
    min_samples_per_stage: int = 0
    min_exposure_s: float = 0.0
    practical_delta: Mapping[str, float] = field(default_factory=dict)
    alpha: float = 0.0
    beta: float = 0.0
    sequential_boundary: str = ""
    multiple_metric_policy: str = ""
    inconclusive_action: str = "HOLD"
    max_traffic_fraction: float = 0.0
    max_error_budget_spend: float = 0.0
    abort_on_hard_gate: bool = True
    assignment_unit: str = ""
    sticky: bool = False
    warmup_separate_from_evaluation: bool = True

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("policy_id", "sequential_boundary", "multiple_metric_policy", "assignment_unit"):
            if not getattr(self, name):
                problems.append(f"canary policy requires {name!r}")
        if self.assignment_unit not in ASSIGNMENT_UNITS:
            problems.append(f"unknown assignment unit {self.assignment_unit!r}")
        if not self.stages or any(not 0 < fraction <= 1 for fraction in self.stages):
            problems.append("exposure stages must be fractions inside (0, 1]")
        elif list(self.stages) != sorted(self.stages):
            problems.append("exposure stages must increase monotonically")
        if self.min_samples_per_stage <= 0:
            problems.append("a minimum sample size per stage is required (otherwise the gate is noise)")
        if self.min_exposure_s <= 0:
            problems.append("a minimum exposure duration is required")
        if not self.practical_delta:
            problems.append("the practical regression δ must be pre-registered per metric")
        unknown = sorted(set(self.practical_delta) - set(METRIC_DIRECTIONS))
        if unknown:
            problems.append(f"practical δ declared for unknown metrics: {unknown}")
        if not 0 < self.alpha < 0.5 or not 0 < self.beta < 0.5:
            problems.append("alpha/beta must be inside (0, 0.5) (both error directions matter)")
        if not 0 < self.max_traffic_fraction <= 1:
            problems.append("the maximum traffic fraction must be inside (0, 1]")
        if self.max_error_budget_spend <= 0:
            problems.append("a maximum error-budget spend must be pre-registered")
        if not self.abort_on_hard_gate:
            problems.append("a hard gate must abort the candidate")
        if not self.sticky:
            problems.append(
                "LLM KV/prefix/session state requires sticky assignment; per-token randomisation would "
                "break sessions and pollute the comparison"
            )
        if not self.warmup_separate_from_evaluation:
            problems.append("canary warmup/cold start must be excluded from the steady evaluation window")
        if self.inconclusive_action not in ("HOLD", "STOP"):
            problems.append("an inconclusive result must HOLD or STOP, never promote")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "stages": list(self.stages),
            "min_samples_per_stage": self.min_samples_per_stage,
            "min_exposure_s": self.min_exposure_s,
            "practical_delta": dict(sorted(self.practical_delta.items())),
            "alpha": self.alpha,
            "beta": self.beta,
            "sequential_boundary": self.sequential_boundary,
            "assignment_unit": self.assignment_unit,
            "sticky": self.sticky,
            "max_traffic_fraction": self.max_traffic_fraction,
            "max_error_budget_spend": self.max_error_budget_spend,
        }


@dataclass
class CandidateIdentity:
    """Step 1: control/candidate releases and their declared change scope."""

    candidate_id: str
    release_id: str
    kind: str = ""
    declared_changes: Tuple[str, ...] = ()
    ground_truth: str = ""
    offline_correctness_status: str = ""
    offline_quality_status: str = ""
    deployment_readiness: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.candidate_id or not self.release_id:
            problems.append("a candidate needs an id and a release id")
        if self.kind not in CANDIDATE_KINDS:
            problems.append(f"unknown candidate kind {self.kind!r}")
        if not self.declared_changes:
            problems.append("the declared change scope is required (one variable per canary)")
        if self.kind != "GOOD_EQUIVALENT" and self.kind != "GOOD_IMPROVED" and not self.ground_truth:
            problems.append("a bad candidate must carry its ground truth (what was injected)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "release_id": self.release_id,
            "kind": self.kind,
            "declared_changes": list(self.declared_changes),
            "ground_truth": self.ground_truth,
        }


@dataclass
class AssignmentPolicy:
    """Step 4: how requests are assigned, including their stickiness key."""

    policy_id: str
    unit: str
    hash_key: str = ""
    seed: str = ""
    eligibility_rule: str = ""
    retry_attribution: str = ""
    tenant_stratified: bool = False
    workload_bucket_stratified: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("policy_id", "hash_key", "seed", "eligibility_rule", "retry_attribution"):
            if not getattr(self, name):
                problems.append(f"assignment policy requires {name!r}")
        if self.unit not in ASSIGNMENT_UNITS:
            problems.append(f"unknown assignment unit {self.unit!r}")
        if self.unit == "SESSION_STICKY" and "session" not in self.hash_key:
            problems.append("a session-sticky policy must hash on the session identifier, not the request id")
        return problems

    def assign(self, *, unit_key: str, traffic_fraction: float) -> str:
        """Deterministic bucketing so the assignment can be replayed and audited."""
        if not 0 <= traffic_fraction <= 1:
            raise ConfigError("traffic fraction must be inside [0, 1]")
        digest = idn.sha256_text(f"{self.seed}|{unit_key}")
        bucket = int(digest.split(":", 1)[1][:8], 16) / 0xFFFFFFFF
        return "candidate" if bucket < traffic_fraction else "control"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "unit": self.unit,
            "hash_key": self.hash_key,
            "seed": self.seed,
            "eligibility_rule": self.eligibility_rule,
            "retry_attribution": self.retry_attribution,
            "tenant_stratified": self.tenant_stratified,
            "workload_bucket_stratified": self.workload_bucket_stratified,
        }


@dataclass
class GateResult:
    """One gate evaluation at one stage."""

    canary_run_id: str
    stage: str
    gate_id: str
    status: str = ""
    hard_gate: bool = False
    evidence_count: int = 0
    reason: str = ""
    value: Optional[float] = None
    threshold: Optional[float] = None

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.gate_id not in GATES:
            problems.append(f"unknown gate {self.gate_id!r}")
        if self.status not in ("PASS", "FAIL", "INCONCLUSIVE", "NOT_RUN"):
            problems.append(f"unknown gate status {self.status!r}")
        if self.hard_gate != (self.gate_id in HARD_GATES):
            problems.append(
                f"{self.gate_id}: hard_gate flag disagrees with the pre-registered gate hierarchy"
            )
        if self.status == "FAIL" and not self.reason:
            problems.append(f"{self.gate_id}: a failing gate must carry a reason")
        if self.status in ("PASS", "FAIL") and self.evidence_count <= 0:
            problems.append(f"{self.gate_id}: a gate verdict needs evidence")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "canary_run_id": self.canary_run_id,
            "stage": self.stage,
            "gate_id": self.gate_id,
            "status": self.status,
            "hard_gate": self.hard_gate,
            "evidence_count": self.evidence_count,
            "reason": self.reason,
        }


def evaluate_gates(*, results: Sequence[GateResult], policy: CanaryPolicy) -> Dict[str, Any]:
    """§4: a later gate can never compensate an earlier failure."""
    problems: List[str] = []
    for result in results:
        problems.extend(result.validate())
    failed = [result for result in results if result.status == "FAIL"]
    hard_failed = [result for result in failed if result.gate_id in HARD_GATES]
    inconclusive = [result for result in results if result.status == "INCONCLUSIVE"]
    missing = [gate for gate in GATES if gate not in {result.gate_id for result in results}]
    if hard_failed and not policy.abort_on_hard_gate:
        problems.append("a hard gate failed while abort_on_hard_gate is disabled")
    return {
        "gates": len(results),
        "failed": [result.gate_id for result in failed],
        "hard_failed": [result.gate_id for result in hard_failed],
        "inconclusive": [result.gate_id for result in inconclusive],
        "not_evaluated": missing,
        "decision": (
            "STOP"
            if hard_failed
            else "HOLD"
            if (failed or inconclusive or missing)
            else "CONTINUE"
        ),
        "problems": problems,
        "ok": not problems,
    }


# ── comparability and statistics ─────────────────────────────────────────


def case_mix_balance(
    *, rows: Sequence[Mapping[str, Any]], dimensions: Sequence[str] = ("isl_bucket", "osl_bucket", "tenant")
) -> Dict[str, Any]:
    """Step 16: control and canary must see the same workload mix."""
    problems: List[str] = []
    observations: List[Dict[str, Any]] = []
    for dimension in dimensions:
        counts: Dict[str, Dict[str, int]] = {"control": {}, "candidate": {}}
        for row in rows:
            arm = str(row.get("arm", ""))
            if arm not in counts:
                continue
            bucket = str(row.get(dimension, "unknown"))
            counts[arm][bucket] = counts[arm].get(bucket, 0) + 1
        total = {arm: sum(values.values()) for arm, values in counts.items()}
        for bucket in sorted(set(counts["control"]) | set(counts["candidate"])):
            control_share = counts["control"].get(bucket, 0) / total["control"] if total["control"] else 0.0
            candidate_share = (
                counts["candidate"].get(bucket, 0) / total["candidate"] if total["candidate"] else 0.0
            )
            delta = abs(control_share - candidate_share)
            observations.append(
                {
                    "canary_run_id": rows[0].get("canary_run_id", "") if rows else "",
                    "stage": rows[0].get("stage", "") if rows else "",
                    "metric": f"case_mix.{dimension}",
                    "arm": bucket,
                    "value": delta,
                    "unit": "ratio",
                    "sample_count": total["control"] + total["candidate"],
                    "interval_low": 0.0,
                    "interval_high": delta,
                }
            )
            if delta > 0.1:
                problems.append(
                    f"case mix for {dimension}={bucket} differs by {delta:.3f} between arms: the comparison "
                    "would mix a release effect with a workload effect"
                )
    return {"observations": observations, "problems": problems, "ok": not problems}


def paired_interval(
    *, control: Sequence[float], candidate: Sequence[float], direction: str, alpha: float = 0.05
) -> Dict[str, Any]:
    """A distribution-free paired comparison (median of per-pair differences).

    Deliberately simple and transparent: the report cites the estimator and the
    interval instead of a bare "faster/slower" claim.
    """
    if direction not in ("higher_is_better", "lower_is_better"):
        raise ConfigError(f"unknown metric direction {direction!r}")
    if not control or not candidate:
        return {"estimator": "median_paired_difference", "estimate": None, "interval": [None, None],
                "n": 0, "comparable": False}
    size = min(len(control), len(candidate))
    differences = [candidate[index] - control[index] for index in range(size)]
    ordered = sorted(differences)
    median = ordered[size // 2] if size % 2 else (ordered[size // 2 - 1] + ordered[size // 2]) / 2
    # Bootstrap-free interval: order statistics at alpha/2 (documented as conservative).
    low_index = max(int(math.floor(alpha / 2 * size)) - 1, 0)
    high_index = min(int(math.ceil((1 - alpha / 2) * size)) - 1, size - 1)
    interval = [ordered[low_index], ordered[high_index]]
    crosses_zero = interval[0] <= 0 <= interval[1]
    regression = (median > 0) if direction == "lower_is_better" else (median < 0)
    return {
        "estimator": "median_paired_difference",
        "direction": direction,
        "estimate": median,
        "interval": interval,
        "n": size,
        "crosses_zero": crosses_zero,
        "regression": regression and not crosses_zero,
        "comparable": True,
    }


def sequential_decision(
    *,
    stage: str,
    metric: str,
    comparison: Mapping[str, Any],
    practical_delta: float,
    information_fraction: float,
    policy: CanaryPolicy,
    error_budget_spent: float,
) -> Dict[str, Any]:
    """Steps 6/22/23: decide with the frozen rule, never by watching a dashboard."""
    problems = policy.validate()
    if metric not in METRIC_DIRECTIONS:
        raise ConfigError(f"unknown canary metric {metric!r}")
    if metric not in policy.practical_delta:
        problems.append(f"{metric}: no pre-registered practical δ")
    if information_fraction < 1.0 and comparison.get("regression"):
        decision, reason = "HOLD", "regression observed before the stage's information fraction is reached"
    elif comparison.get("regression") and comparison.get("estimate") is not None:
        exceeds = abs(float(comparison["estimate"])) > practical_delta
        decision = "ROLLBACK" if exceeds and not comparison.get("crosses_zero") else "HOLD"
        reason = (
            f"{metric}: paired difference {comparison['estimate']} vs practical δ {practical_delta}"
        )
    elif information_fraction >= 1.0 and not comparison.get("crosses_zero"):
        decision, reason = "CONTINUE", f"{metric}: within δ at full information"
    else:
        decision, reason = "CONTINUE", "insufficient information yet"
    if error_budget_spent > policy.max_error_budget_spend:
        decision, reason = "STOP", (
            f"error budget spent {error_budget_spent} exceeds the pre-registered cap "
            f"{policy.max_error_budget_spend}"
        )
    return {
        "stage": stage,
        "metric": metric,
        "information_fraction": information_fraction,
        "decision": decision,
        "reason": reason,
        "problems": problems,
        "ok": not problems,
    }


def exposure_budget(
    *, stages: Sequence[Mapping[str, Any]], policy: CanaryPolicy
) -> Dict[str, Any]:
    """Step 3/23: the risk budget is enforced per stage, not discovered afterwards."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    for stage in stages:
        fraction = float(stage.get("traffic_fraction", 0.0))
        spent = float(stage.get("error_budget_spent", 0.0))
        eligible = int(stage.get("eligible_count", 0))
        rows.append(
            {
                "stage": stage.get("stage", ""),
                "traffic_fraction": fraction,
                "eligible_count": eligible,
                "error_budget_spent": spent,
                "information_fraction": min(
                    eligible / policy.min_samples_per_stage if policy.min_samples_per_stage else 0.0, 1.0
                ),
            }
        )
        if fraction > policy.max_traffic_fraction:
            problems.append(f"{stage.get('stage')}: traffic fraction {fraction} exceeds the budget")
        if spent > policy.max_error_budget_spend:
            problems.append(f"{stage.get('stage')}: error budget spend {spent} exceeds the budget")
    return {"rows": rows, "problems": problems, "ok": not problems}


@dataclass
class CanaryDecisionRow:
    """§12 ``CanaryDecision``."""

    decision_id: str
    canary_run_id: str
    timestamp: float = 0.0
    stage: str = ""
    control_release_id: str = ""
    candidate_release_id: str = ""
    assignment_policy: str = ""
    traffic_fraction: float = 0.0
    eligible_count: int = 0
    case_mix_balance: str = ""
    gate_results: Tuple[str, ...] = ()
    information_fraction: float = 0.0
    sequential_boundary: str = ""
    decision: str = ""
    reason_codes: Tuple[str, ...] = ()
    error_budget_spent: float = 0.0
    rollback_id: str = ""
    final_active_release: str = ""
    manual_override_id: str = ""
    evidence_refs: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.decision not in DECISIONS:
            problems.append(f"unknown canary decision {self.decision!r}")
        if not self.stage:
            problems.append("a decision must name the stage")
        if not self.reason_codes:
            problems.append("a decision must carry reason codes")
        if self.decision == "PROMOTE" and self.information_fraction < 1.0:
            problems.append("promotion before the information fraction is reached")
        if self.decision == "PROMOTE" and self.manual_override_id:
            problems.append(
                "a manual override must not masquerade as an automatic promotion"
            )
        if self.decision == "ROLLBACK" and not self.rollback_id:
            problems.append("a rollback decision must reference its rollback id")
        if not self.control_release_id or not self.candidate_release_id:
            problems.append("a decision must name control and candidate releases")
        if not self.evidence_refs:
            problems.append("a decision must reference the evidence it used")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "canary_run_id": self.canary_run_id,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "control_release_id": self.control_release_id,
            "candidate_release_id": self.candidate_release_id,
            "traffic_fraction": self.traffic_fraction,
            "information_fraction": self.information_fraction,
            "decision": self.decision,
            "reason_codes": list(self.reason_codes),
        }


def decide_stage(
    *,
    gates: Mapping[str, Any],
    balance: Mapping[str, Any],
    sequential: Mapping[str, Any],
    policy: CanaryPolicy,
    row: CanaryDecisionRow,
) -> CanaryDecisionRow:
    """Combine gate/stage statistics into one machine decision."""
    # The caller has already validated the policy/row; this function only maps the
    # stage evidence onto the frozen decision rule.
    if gates.get("hard_failed"):
        row.decision = "STOP"
        row.reason_codes = ("HARD_GATE_FAILED", *tuple(gates.get("hard_failed", ())))
    elif not balance.get("ok"):
        row.decision = "STOP"
        row.reason_codes = ("CASE_MIX_NOT_COMPARABLE",)
    elif sequential.get("decision") == "ROLLBACK":
        row.decision = "ROLLBACK"
        row.reason_codes = ("PRACTICAL_REGRESSION",)
    elif sequential.get("decision") == "STOP":
        row.decision = "STOP"
        row.reason_codes = ("ERROR_BUDGET_EXCEEDED",)
    elif gates.get("not_evaluated"):
        row.decision = "INCONCLUSIVE"
        row.reason_codes = ("GATES_NOT_EVALUATED",)
    elif row.information_fraction < 1.0:
        row.decision = "CONTINUE"
        row.reason_codes = ("INFORMATION_INCOMPLETE",)
    else:
        row.decision = "PROMOTE"
        row.reason_codes = ("ALL_GATES_PASSED", "INFORMATION_SUFFICIENT")
    row.case_mix_balance = "OK" if balance.get("ok") else "MISMATCH"
    return row


def rollback_closure(
    *, rollback_id: str, canary_run_id: str, decision_ts: float, traffic_stopped_ts: float,
    drain_done_ts: float, baseline_restored_ts: float, request_integrity_ok: bool,
    control_quality_verified: bool, resource_state_consistent: bool,
    residual_objects: Sequence[str], max_rollback_s: float,
) -> Dict[str, Any]:
    """Steps 30–33: a rollback is closed only when traffic, identity, quality and state are back."""
    problems: List[str] = []
    for name, value in (("decision_ts", decision_ts), ("traffic_stopped_ts", traffic_stopped_ts),
                        ("baseline_restored_ts", baseline_restored_ts)):
        if not value:
            problems.append(f"the rollback timeline must record {name}")
    if not request_integrity_ok:
        problems.append("inflight/new request semantics were not verified across the rollback")
    if not control_quality_verified:
        problems.append("control quality must be re-verified after the rollback")
    if not resource_state_consistent:
        problems.append("candidate state/resources must be cleaned up and verified")
    if residual_objects:
        problems.append(f"candidate residuals remain: {sorted(residual_objects)}")
    if baseline_restored_ts and decision_ts and (baseline_restored_ts - decision_ts) > max_rollback_s:
        problems.append(
            f"the rollback took {baseline_restored_ts - decision_ts:.1f}s > the pre-registered limit "
            f"{max_rollback_s}s (MTTR is defined by user steady state, not by the control plane)"
        )
    return {
        "rollback_id": rollback_id,
        "canary_run_id": canary_run_id,
        "decision_to_traffic_stopped_s": (traffic_stopped_ts - decision_ts) if traffic_stopped_ts else None,
        "decision_to_baseline_restored_s": (baseline_restored_ts - decision_ts) if baseline_restored_ts else None,
        "drain_done_ts": drain_done_ts,
        "problems": problems,
        "ok": not problems,
    }


def false_rate_accounting(
    *, outcomes: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Step 34: bad candidates must be caught; good candidates must not be killed by noise."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    for outcome in outcomes:
        kind = str(outcome.get("kind", ""))
        if kind not in CANDIDATE_KINDS:
            raise ConfigError(f"unknown candidate kind {kind!r}")
        bad = kind.endswith("_BAD")
        stopped = outcome.get("decision") in ("STOP", "ROLLBACK")
        promoted = outcome.get("decision") == "PROMOTE"
        rows.append(
            {
                "canary_run_id": outcome.get("canary_run_id", ""),
                "bad_kind": kind if bad else "",
                "detected": stopped if bad else promoted,
                "detection_stage": outcome.get("stage", ""),
                "false_negative": bool(bad and not stopped),
                "exposure_at_detection": outcome.get("exposure_at_detection", ""),
            }
        )
        if bad and not stopped:
            problems.append(f"{kind}: a bad candidate was not stopped (false negative)")
        if not bad and stopped:
            problems.append(f"{kind}: a good candidate was stopped (systematic false positive)")
    bad_rows = [row for row in rows if row["bad_kind"]]
    good_rows = [row for row in rows if not row["bad_kind"]]
    return {
        "rows": rows,
        "bad_candidates": len(bad_rows),
        "good_candidates": len(good_rows),
        "false_negative": [row["bad_kind"] for row in bad_rows if row["false_negative"]],
        "false_positive": [row["canary_run_id"] for row in good_rows if not row["detected"]],
        "problems": problems,
        "ok": not problems,
    }


def sensitivity_replay(
    *, rows: Sequence[Mapping[str, Any]], registered_deltas: Sequence[float], confirmatory: bool
) -> Dict[str, Any]:
    """Step 35: a threshold replay is sensitivity analysis, not confirmation."""
    problems: List[str] = []
    unregistered: List[Dict[str, Any]] = []
    for row in rows:
        delta = float(row.get("delta", 0.0))
        if delta not in {float(value) for value in registered_deltas}:
            unregistered.append({"row": row.get("row_id", ""), "delta": delta})
    if unregistered:
        problems.append(f"{len(unregistered)} replay rows used unregistered δ values")
    if confirmatory:
        problems.append(
            "replaying raw data with different thresholds is sensitivity analysis; it may not be "
            "presented as a confirmatory result"
        )
    return {"rows": list(rows), "unregistered": unregistered, "problems": problems, "ok": not problems}


@dataclass
class OverrideAudit:
    """Step 37: an override is recorded with actor, reason, expiry — and never counts as a pass."""

    override_id: str
    actor: str
    action: str
    reason: str = ""
    expires_at: str = ""
    canary_run_id: str = ""
    counts_as_automatic: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("override_id", "actor", "reason", "canary_run_id"):
            if not getattr(self, name):
                problems.append(f"override audit requires {name!r}")
        if self.action not in OVERRIDE_ACTIONS:
            problems.append(f"unknown override action {self.action!r}")
        if self.action in FORBIDDEN_OVERRIDE_ACTIONS:
            problems.append(
                f"action {self.action} is forbidden for a candidate: it would bypass the quality/perf gates "
                "(the override may only hold, abort or roll back)"
            )
        if not self.expires_at:
            problems.append("an override must expire")
        if self.counts_as_automatic:
            problems.append("a manual override never counts as an automatic gate result")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "override_id": self.override_id,
            "actor": self.actor,
            "action": self.action,
            "reason": self.reason,
            "expires_at": self.expires_at,
            "counts_as_automatic": self.counts_as_automatic,
        }


def release_governance_verdict(
    *,
    policy: CanaryPolicy,
    gates: Mapping[str, Any],
    balance: Mapping[str, Any],
    budget: Mapping[str, Any],
    rollback: Mapping[str, Any],
    false_rates: Mapping[str, Any],
    overrides: Sequence[OverrideAudit],
    holdout: Mapping[str, Any],
) -> Dict[str, Any]:
    """Step 38: detectable degradation domain, false rates, rollback and applicability."""
    problems: List[str] = []
    for name, axis in (
        ("gates", gates),
        ("case_mix_balance", balance),
        ("exposure_budget", budget),
        ("rollback_closure", rollback),
        ("false_rates", false_rates),
        ("holdout", holdout),
    ):
        if not axis.get("ok"):
            problems.append(f"{name}: " + "; ".join(axis.get("problems", []) or ["(no detail)"]))
    for override in overrides:
        problems.extend(override.validate())
    return {
        "problems": problems,
        "verdict": "PASSABLE_AT_CODE_LEVEL" if not problems else "BLOCKED",
        "automatic_only": all(not override.counts_as_automatic for override in overrides),
        "forbidden_claims": [
            "automatic safe release without bad/good candidate pairs",
            "quality compensation by performance",
            "promotion with insufficient information",
        ],
        "note": "false-negative/false-positive rates and exposure-to-harm come from the executed campaign",
    }


def holdout_canary(
    *, cases: Sequence[Mapping[str, Any]], policy_frozen: bool, ground_truth_visible: bool
) -> Dict[str, Any]:
    """Step 36: the frozen policy meets new variants without their ground truth."""
    problems: List[str] = []
    if not policy_frozen:
        problems.append("the holdout canary requires a frozen policy (no tuning during holdout)")
    if ground_truth_visible:
        problems.append("the gate must not receive the candidate ground truth during the holdout run")
    for case in cases:
        if not case.get("expected_decision"):
            problems.append(f"holdout case {case.get('case_id', '?')} has no expected decision")
    return {"cases": len(cases), "problems": problems, "ok": not problems}


# ── protocol steps and smoke self-check ──────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 release change taxonomy", ("canary:CandidateIdentity", "identity:validate_release_change_scope")),
    (2, "冻结 canary 状态机", ("canary:CANARY_STATE_MACHINE", "canary:PROMOTION_PATH")),
    (3, "冻结业务风险预算", ("canary:CanaryPolicy", "canary:exposure_budget")),
    (4, "冻结 traffic assignment", ("canary:AssignmentPolicy", "canary:ASSIGNMENT_UNITS")),
    (5, "冻结 metrics 和方向", ("canary:METRIC_DIRECTIONS", "records:CANARY_GATES")),
    (6, "冻结统计/序贯规则", ("canary:sequential_decision", "canary:paired_interval")),
    (7, "冻结 bad/good candidate 构造", ("canary:CANDIDATE_KINDS", "canary:CandidateIdentity")),
    (8, "验证 control 稳态", ("faults:SteadyState", "observability:SLIDefinition")),
    (9, "执行 G0 供应链/兼容 gate", ("supply_chain:release_gate", "canary:GateResult")),
    (10, "执行 G1 离线 correctness", ("canary:GateResult", "deployment:QualityProbe")),
    (11, "执行 G2 质量 gate", ("canary:GateResult", "deployment:QualityProbe")),
    (12, "部署 candidate 不接流量", ("artifacts:ActivationGeneration", "deployment:readiness_verdict")),
    (13, "执行 shadow/synthetic smoke", ("canary:GateResult", "deployment:FirstRequestRecord")),
    (14, "启动 Stage 1 小流量", ("canary:AssignmentPolicy.assign", "canary:exposure_budget")),
    (15, "验证 assignment 正确性", ("canary:AssignmentPolicy.assign",)),
    (16, "验证 control/canary 可比性", ("canary:case_mix_balance",)),
    (17, "运行 correctness/stream gate", ("canary:GateResult", "lifecycle:request_integrity_report")),
    (18, "运行 error/availability gate", ("canary:GateResult", "serving:slo.SLOSpec")),
    (19, "运行 latency/goodput gate", ("canary:paired_interval", "canary:GateResult")),
    (20, "运行 resource/capacity gate", ("canary:GateResult", "capacity:memory_ledger")),
    (21, "运行 thermal/power/cost 监测", ("canary:GateResult", "autoscaling:cost_accounting")),
    (22, "执行序贯 decision", ("canary:sequential_decision", "canary:CanaryDecisionRow")),
    (23, "推进 exposure stages", ("canary:exposure_budget", "canary:decide_stage")),
    (24, "协调 autoscaling", ("autoscaling:coordination_checks",)),
    (25, "协调 cache/placement", ("canary:case_mix_balance", "scheduling:PlacementEvidence")),
    (26, "运行 correctness-bad canary", ("canary:evaluate_gates", "canary:false_rate_accounting")),
    (27, "运行 performance-bad canary", ("canary:paired_interval", "canary:false_rate_accounting")),
    (28, "运行 resource-bad canary", ("canary:false_rate_accounting", "capacity:memory_ledger")),
    (29, "运行 good/equivalent canary", ("canary:false_rate_accounting",)),
    (30, "触发自动 rollback", ("canary:rollback_closure", "artifacts:rollback_timeline")),
    (31, "验证 rollback 请求语义", ("canary:rollback_closure", "lifecycle:request_integrity_report")),
    (32, "验证 control 恢复", ("canary:rollback_closure", "artifacts:rollback_timeline")),
    (33, "检查 candidate 残留", ("canary:rollback_closure", "artifacts:gc_safety_check")),
    (34, "计算 gate 质量", ("canary:false_rate_accounting",)),
    (35, "做 threshold/sample 敏感性", ("canary:sensitivity_replay",)),
    (36, "独立 holdout canary", ("canary:holdout_canary",)),
    (37, "验证人工 override/审计", ("canary:OverrideAudit",)),
    (38, "形成 release governance verdict", ("canary:release_governance_verdict",)),
)


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the canary contracts (smoke, not an experiment)."""
    checks: Dict[str, Any] = {}
    policy = CanaryPolicy(
        policy_id="canary-1", min_samples_per_stage=200, min_exposure_s=600.0,
        practical_delta={"ttft_seconds": 0.05, "slo_goodput": 0.05}, alpha=0.05, beta=0.2,
        sequential_boundary="O'Brien-Fleming-like, 3 looks",
        multiple_metric_policy="hard gates first, then primary latency/goodput",
        max_traffic_fraction=0.25, max_error_budget_spend=0.2, assignment_unit="SESSION_STICKY",
        sticky=True,
    )
    checks["policy_valid"] = policy.validate() == []

    bad_policy = CanaryPolicy(policy_id="p2", stages=(0.5, 0.1), practical_delta={}, assignment_unit="PER_TOKEN")
    checks["invalid_policy_rejected"] = (
        len(bad_policy.validate()) >= 3
        and "unknown assignment unit 'PER_TOKEN'" in " ".join(bad_policy.validate())
    )

    machine = CANARY_STATE_MACHINE
    checks["state_machine_valid"] = machine.validate() == []
    checks["promotion_path_legal"] = machine.walk(list(PROMOTION_PATH))["ok"] is True
    checks["illegal_skip_detected"] = (
        machine.walk(["CANDIDATE_REGISTERED", "CANARY_EXPOSURE_STAGE_1"])["ok"] is False
    )

    assignment = AssignmentPolicy(
        policy_id="a1", unit="SESSION_STICKY", hash_key="session_id", seed="s-2026",
        eligibility_rule="authenticated tenants only", retry_attribution="first attempt arm",
    )
    checks["assignment_valid"] = assignment.validate() == []
    arm_a = assignment.assign(unit_key="session-1", traffic_fraction=0.1)
    arm_b = assignment.assign(unit_key="session-1", traffic_fraction=0.1)
    checks["assignment_deterministic"] = arm_a == arm_b

    gates = evaluate_gates(
        results=[
            GateResult(canary_run_id="run-1", stage="stage1", gate_id="G1_OFFLINE_CORRECTNESS",
                       status="FAIL", hard_gate=True, evidence_count=5, reason="token mismatch"),
            GateResult(canary_run_id="run-1", stage="stage1", gate_id="G5_LATENCY_SLO_GOODPUT",
                       status="PASS", hard_gate=False, evidence_count=100),
        ],
        policy=policy,
    )
    checks["hard_gate_stops"] = gates["decision"] == "STOP"

    comparison = paired_interval(control=[1.0] * 20, candidate=[1.2] * 20, direction="lower_is_better")
    checks["paired_regression_detected"] = comparison["regression"] is True

    sealed = sequential_decision(
        stage="stage1", metric="ttft_seconds", comparison=comparison, practical_delta=0.05,
        information_fraction=1.0, policy=policy, error_budget_spent=0.01,
    )
    checks["regression_rolls_back"] = sealed["decision"] == "ROLLBACK"

    early = sequential_decision(
        stage="stage1", metric="ttft_seconds", comparison=comparison, practical_delta=0.05,
        information_fraction=0.3, policy=policy, error_budget_spent=0.01,
    )
    checks["premature_promotion_blocked"] = early["decision"] == "HOLD"

    override = OverrideAudit(
        override_id="o1", actor="sre", action="FORCE_PROMOTE", reason="demo", expires_at="",
        canary_run_id="run-1", counts_as_automatic=True,
    )
    checks["override_abuse_rejected"] = len(override.validate()) >= 3

    rollback = rollback_closure(
        rollback_id="rb-1", canary_run_id="run-1", decision_ts=100.0, traffic_stopped_ts=105.0,
        drain_done_ts=140.0, baseline_restored_ts=400.0, request_integrity_ok=True,
        control_quality_verified=True, resource_state_consistent=True, residual_objects=(),
        max_rollback_s=600.0,
    )
    checks["rollback_closure_ok"] = rollback["ok"] is True
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "checks": checks,
        "note": "接口自检；未部署 candidate，未路由任何流量",
    }
