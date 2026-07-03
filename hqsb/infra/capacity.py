"""E13-06: S12 capacity model reconciliation, admission, token/KV budget and OOM margin.

Implements ``details/S13/E13-06_*.md`` as data:

* the memory ledger of §2 (weights + engine/graph + KV + workspace + allocator
  reserved + scheduler cache + collective buffers + system reserve), per rank;
* the per-token KV estimate of §3 **with block rounding, prefix sharing and
  metadata** so the S12 formula is calibrated rather than trusted;
* ``token_work`` as the admission signal (request count alone is a documented
  failure mode for mixed-length workloads);
* the six admission policies of step 3 and their comparison;
* :class:`AdmissionDecision` with policy/reason/budget and the atomic
  reserve/commit check that catches oversell races;
* prediction residuals, false admit/reject estimation with an explicit
  counterfactual method, safety-margin holdout validation and the S12 feedback
  artifact (the original prediction is never deleted).

Nothing here allocates memory or runs a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.infra import records as rec

EXPERIMENT_ID = "E13-06"
TITLE = "S12 容量模型、Admission、Token/KV Budget 与 OOM Margin 验证"
CLAIM_BOUNDARY = (
    "本实验通过证明容量与准入在声明 workload 内安全；不证明 autoscaler 能及时增加容量，也不证明"
    "节点/网络/存储故障恢复（由 E13-07/E13-09 验证）。"
)

SCHEMA_VERSION = "1.0.0"

#: Memory components come from ``records`` (one source for ledger + report).
MEMORY_COMPONENTS: Tuple[str, ...] = rec.RESOURCE_MEMORY_COMPONENTS

#: Admission policies compared in step 3/20 (request count is only a baseline).
ADMISSION_POLICIES: Tuple[str, ...] = (
    "REQUEST_COUNT",
    "ACCELERATOR_UTIL",
    "QUEUE_TOKEN_WORK",
    "MEMORY_KV_PREDICTED",
    "SLO_RISK_COMBINED",
    "REQUEST_COUNT_PLUS_PROCESSING_POLICY",
)

#: Decision reason codes (stable, citable in the report).
REASON_CODES: Tuple[str, ...] = (
    "ADMIT_WITHIN_BUDGET",
    "QUEUE_OVER_TOKEN_BUDGET",
    "REJECT_TOKEN_BUDGET_EXCEEDED",
    "REJECT_REQUEST_TOO_LARGE",
    "REJECT_MEMORY_PREDICTION_OVER_MARGIN",
    "REJECT_SLO_RISK",
    "REJECT_INVALID_INPUT",
    "REJECT_QUOTA",
    "ROUTE_TO_OTHER_CAPACITY",
    "DEGRADE_DECLARED",
)

#: Sweep plan (§9 steps 15–19): safe → boundary → overload with an abort guard.
SWEEP_STAGES: Tuple[str, ...] = ("SAFE", "BOUNDARY", "OVERLOAD", "RECOVERY")

#: Feature axes that change the budget (§9 steps 22–25).
BUDGET_FEATURES: Tuple[str, ...] = (
    "prefix_cache",
    "speculative_decoding",
    "chunked_prefill",
    "kv_quantization",
    "distributed_shards",
    "model_switch_dual_residency",
)

#: Fault cases (steps 26/27/30/31).
FAULT_CASES: Tuple[str, ...] = (
    "CANCEL_AT_QUEUED",
    "CANCEL_AT_PREFILL",
    "CANCEL_AT_DECODE",
    "CANCEL_AT_STREAM",
    "REQUEST_FAILURE",
    "BACKEND_ERROR",
    "ALLOCATOR_FRAGMENTATION_HISTORY",
    "OVER_BUDGET_NEGATIVE",
    "CONTROLLED_OOM",
)

#: Maximum tolerated false-admit rate (a *policy* input, frozen at preregistration).
DEFAULT_FALSE_ADMIT_GATE = 0.0  #: any protected OOM is a failure


# ── memory ledger and KV model ────────────────────────────────────────────


@dataclass
class MemoryComponent:
    """One line of the memory ledger, per rank, with its evidence source."""

    component: str
    bytes_value: int = 0
    rank_id: str = "0"
    source: str = ""
    state: str = ""
    measurement_kind: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.component not in MEMORY_COMPONENTS:
            problems.append(f"unknown memory component {self.component!r}")
        if self.state and self.state not in rec.MISSINGNESS_CODES:
            problems.append(f"unknown state {self.state!r}")
        if not self.state and self.bytes_value < 0:
            problems.append(f"{self.component}: negative bytes")
        if not self.source:
            problems.append(
                f"{self.component}: the evidence source (allocator/vendor API/KV manager) must be recorded"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": "",
            "rank_id": self.rank_id,
            "component": self.component,
            "bytes": self.bytes_value,
            "source": self.source,
            "state": self.state,
        }


def memory_ledger(
    components: Sequence[MemoryComponent], *, usable_memory_bytes: int, per_rank: bool = True
) -> Dict[str, Any]:
    """§2/§13: total memory is *not* enough — the largest rank is the constraint."""
    problems: List[str] = []
    by_rank: Dict[str, int] = {}
    missing_components: List[str] = []
    for component in components:
        problems.extend(component.validate())
        by_rank[component.rank_id] = by_rank.get(component.rank_id, 0) + component.bytes_value
    covered = {component.component for component in components}
    missing_components = sorted(set(MEMORY_COMPONENTS) - covered)
    if missing_components:
        problems.append(
            "ledger does not account for every component: " + ", ".join(missing_components)
        )
    max_rank = max(by_rank.values()) if by_rank else 0
    if per_rank and max_rank > usable_memory_bytes:
        problems.append(
            f"the largest rank uses {max_rank} bytes > usable {usable_memory_bytes}: "
            "a total-memory argument would hide the per-rank bottleneck"
        )
    return {
        "rows": [component.as_dict() for component in components],
        "by_rank": dict(sorted(by_rank.items())),
        "max_rank_bytes": max_rank,
        "usable_memory_bytes": usable_memory_bytes,
        "missing_components": missing_components,
        "problems": problems,
        "ok": not problems,
    }


@dataclass
class KVModel:
    """Per-token KV estimate plus the effects the formula alone misses."""

    layers: int = 0
    kv_heads: int = 0
    head_dim: int = 0
    bytes_per_element: int = 2
    block_tokens: int = 16
    block_bytes_overhead: int = 0
    scale_metadata_bytes_per_token: int = 0
    alignment_bytes_per_token: int = 0

    def bytes_per_token(self) -> int:
        base = self.layers * 2 * self.kv_heads * self.head_dim * self.bytes_per_element
        return base + self.scale_metadata_bytes_per_token + self.alignment_bytes_per_token

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("layers", "kv_heads", "head_dim", "bytes_per_element", "block_tokens"):
            if getattr(self, name) <= 0:
                problems.append(f"KV model requires a positive {name!r}")
        if self.block_bytes_overhead <= 0:
            problems.append(
                "block rounding/allocator overhead must be a measured input (0 would mean a perfect allocator)"
            )
        return problems

    def incremental_bytes(self, token_budget: int, *, prefix_shared_tokens: int = 0) -> int:
        """§3: ``incremental_KV`` with block rounding.

        Prefix-shared tokens are only discounted when the caller *knows* the prefix
        is in cache; a miss may not reuse the hit budget.
        """
        if token_budget < 0:
            raise ConfigError("token budget must not be negative")
        effective = max(token_budget - max(prefix_shared_tokens, 0), 0)
        blocks = -(-effective // self.block_tokens)  # ceil
        padded = blocks * self.block_tokens
        return padded * (self.bytes_per_token() + self.block_bytes_overhead)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layers": self.layers,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "bytes_per_element": self.bytes_per_element,
            "block_tokens": self.block_tokens,
            "bytes_per_token": self.bytes_per_token(),
        }


def token_work(*, queued_prompt_tokens: int, queued_output_budget: int, active_prefill: int,
               active_decode_sequences: int) -> Dict[str, Any]:
    """§3 ``token_work``: the signal that request count cannot represent."""
    if min(queued_prompt_tokens, queued_output_budget, active_prefill, active_decode_sequences) < 0:
        raise ConfigError("token work components must not be negative")
    total = queued_prompt_tokens + queued_output_budget + active_prefill + active_decode_sequences
    return {
        "queued_prompt_tokens": queued_prompt_tokens,
        "queued_output_budget": queued_output_budget,
        "active_prefill_tokens": active_prefill,
        "active_decode_sequences": active_decode_sequences,
        "token_work": total,
        "note": "the same request count can mean very different token work (mixed-length workloads)",
    }


# ── admission ────────────────────────────────────────────────────────────


@dataclass
class AdmissionPolicy:
    """One admission policy with its frozen parameters."""

    policy_id: str
    kind: str
    safety_margin_bytes: int = 0
    token_budget: int = 0
    queue_token_limit: int = 0
    max_queue_age_s: float = 0.0
    slo_risk_threshold: float = 0.0
    version: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.kind not in ADMISSION_POLICIES:
            problems.append(f"unknown admission policy kind {self.kind!r}")
        if not self.version:
            problems.append("a policy version is required (decisions must be attributable)")
        if self.kind in ("MEMORY_KV_PREDICTED", "SLO_RISK_COMBINED") and self.safety_margin_bytes <= 0:
            problems.append(
                f"{self.kind} requires a pre-registered safety margin (it is not a magic constant)"
            )
        if self.kind == "QUEUE_TOKEN_WORK" and self.queue_token_limit <= 0:
            problems.append("a queue-token policy needs a queue/token limit")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "kind": self.kind,
            "safety_margin_bytes": self.safety_margin_bytes,
            "token_budget": self.token_budget,
            "queue_token_limit": self.queue_token_limit,
            "max_queue_age_s": self.max_queue_age_s,
            "slo_risk_threshold": self.slo_risk_threshold,
            "version": self.version,
        }


@dataclass
class AdmissionDecision:
    """§11 ``AdmissionDecision`` (also the input of the §23 V07 validator)."""

    decision_id: str
    request_id: str
    timestamp: str = ""
    policy_version: str = ""
    workload_bucket: str = ""
    prompt_tokens: int = 0
    requested_output_tokens: int = 0
    actual_output_tokens: int = 0
    predicted_incremental_kv_bytes: int = 0
    predicted_workspace_bytes: int = 0
    predicted_peak_memory_bytes: int = 0
    current_allocated_bytes: int = 0
    current_reserved_bytes: int = 0
    free_bytes: int = 0
    usable_memory_bytes: int = 0
    safety_margin_bytes: int = 0
    queue_requests: int = 0
    queue_tokens: int = 0
    oldest_age_s: float = 0.0
    predicted_slo_risk: float = 0.0
    quota_state: str = ""
    decision: str = ""
    reason_code: str = ""
    reservation_id: str = ""
    actual_peak_bytes: int = 0
    outcome: str = ""
    oom: bool = False
    resource_released: bool = False
    model_switch_dual_residency: bool = False
    release_id: str = ""
    model_artifact_id: str = ""

    def as_decision_row(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "policy_version": self.policy_version,
            "prompt_tokens": self.prompt_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "predicted_incremental_kv_bytes": self.predicted_incremental_kv_bytes,
            "usable_memory_bytes": self.usable_memory_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "decision": self.decision,
            "reason_code": self.reason_code,
        }

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.decision not in rec.ADMISSION_DECISIONS:
            problems.append(f"unknown admission decision {self.decision!r}")
        if self.reason_code not in REASON_CODES:
            problems.append(f"unknown admission reason code {self.reason_code!r}")
        if not self.policy_version:
            problems.append("an admission decision must name the policy version it used")
        if not self.release_id or not self.model_artifact_id:
            problems.append("an admission decision must be bound to the release/model it admitted against")
        if self.decision == "ADMIT_NOW" and not self.reservation_id:
            problems.append("an admitted request must hold a reservation id")
        if self.decision == "DEGRADE_EXPLICITLY" and self.reason_code != "DEGRADE_DECLARED":
            problems.append("explicit degradation must be declarable (never a silent quality drop)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "release_id": self.release_id,
            "model_artifact_id": self.model_artifact_id,
            "policy_version": self.policy_version,
            "workload_bucket": self.workload_bucket,
            "prompt_tokens": self.prompt_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "actual_output_tokens": self.actual_output_tokens,
            "predicted_incremental_kv_bytes": self.predicted_incremental_kv_bytes,
            "predicted_workspace_bytes": self.predicted_workspace_bytes,
            "predicted_peak_memory_bytes": self.predicted_peak_memory_bytes,
            "current_allocated_bytes": self.current_allocated_bytes,
            "current_reserved_bytes": self.current_reserved_bytes,
            "free_bytes": self.free_bytes,
            "usable_memory_bytes": self.usable_memory_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "queue_requests": self.queue_requests,
            "queue_tokens": self.queue_tokens,
            "oldest_age_s": self.oldest_age_s,
            "predicted_slo_risk": self.predicted_slo_risk,
            "quota_state": self.quota_state,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "reservation_id": self.reservation_id,
            "actual_peak_bytes": self.actual_peak_bytes,
            "outcome": self.outcome,
            "oom": self.oom,
            "resource_released": self.resource_released,
            "model_switch_dual_residency": self.model_switch_dual_residency,
        }


def admit(
    *,
    policy: AdmissionPolicy,
    kv_model: KVModel,
    decision: AdmissionDecision,
    prefix_shared_tokens: int = 0,
    queue_token_budget: Optional[int] = None,
) -> AdmissionDecision:
    """Decide one request and *explain* it (§4).

    The function fills ``predicted_*``/``decision``/``reason_code`` so an operator
    can reproduce the reasoning; it never mutates memory state itself.
    """
    problems = policy.validate() + kv_model.validate()
    if problems:
        raise ConfigError("invalid admission policy/KV model: " + "; ".join(problems))
    if decision.prompt_tokens <= 0 and decision.requested_output_tokens <= 0:
        decision.decision = "REJECT_INVALID_OR_TOO_LARGE"
        decision.reason_code = "REJECT_INVALID_INPUT"
        return decision

    token_budget = decision.prompt_tokens + decision.requested_output_tokens
    predicted_kv = kv_model.incremental_bytes(token_budget, prefix_shared_tokens=prefix_shared_tokens)
    predicted_peak = predicted_kv + decision.predicted_workspace_bytes
    decision.predicted_incremental_kv_bytes = predicted_kv
    decision.predicted_peak_memory_bytes = predicted_peak

    if policy.token_budget and token_budget > policy.token_budget:
        decision.decision = "REJECT_INVALID_OR_TOO_LARGE"
        decision.reason_code = "REJECT_REQUEST_TOO_LARGE"
        return decision
    if policy.kind == "QUEUE_TOKEN_WORK":
        limit = queue_token_budget if queue_token_budget is not None else policy.queue_token_limit
        if decision.queue_tokens + token_budget > limit:
            decision.decision = "REJECT_RETRYABLE_OVERLOAD"
            decision.reason_code = "REJECT_TOKEN_BUDGET_EXCEEDED"
            return decision
    if policy.kind in ("MEMORY_KV_PREDICTED", "SLO_RISK_COMBINED"):
        usable = decision.usable_memory_bytes - policy.safety_margin_bytes
        if predicted_peak > max(usable, 0):
            decision.decision = "REJECT_RETRYABLE_OVERLOAD"
            decision.reason_code = "REJECT_MEMORY_PREDICTION_OVER_MARGIN"
            return decision
        if policy.kind == "SLO_RISK_COMBINED" and decision.predicted_slo_risk > policy.slo_risk_threshold:
            decision.decision = "QUEUE_BOUNDED"
            decision.reason_code = "REJECT_SLO_RISK"
            return decision
    if decision.quota_state == "EXHAUSTED":
        decision.decision = "REJECT_RETRYABLE_OVERLOAD"
        decision.reason_code = "REJECT_QUOTA"
        return decision
    decision.decision = "ADMIT_NOW"
    decision.reason_code = "ADMIT_WITHIN_BUDGET"
    decision.reservation_id = decision.reservation_id or f"res-{decision.request_id}"
    return decision


def reservation_atomicity(trials: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 21/§12 invariant 2: reserve/commit must be atomic (no oversell race)."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    for trial in trials:
        capacity = int(trial.get("capacity_bytes", 0))
        reserved = int(trial.get("reserved_bytes", 0))
        committed = int(trial.get("committed_bytes", 0))
        rows.append(
            {
                "trial_id": trial.get("trial_id", ""),
                "concurrent_requests": int(trial.get("concurrent_requests", 0)),
                "capacity_bytes": capacity,
                "reserved_bytes": reserved,
                "committed_bytes": committed,
                "oversell": committed > capacity,
            }
        )
        if committed > capacity:
            problems.append(
                f"trial {trial.get('trial_id')}: committed {committed} > capacity {capacity} (oversell)"
            )
    return {"rows": rows, "trials": len(rows), "problems": problems, "ok": not problems}


def cancel_release(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Steps 26/27: cancel/error must return budget, KV blocks and queue slots."""
    problems: List[str] = []
    for row in rows:
        stage = str(row.get("stage", ""))
        if stage not in BUDGET_FEATURES and stage not in ("queued", "prefill", "decode", "stream", "backend_error"):
            problems.append(f"unknown cancel/error stage {stage!r}")
        if not row.get("budget_released", False):
            problems.append(f"{stage}: budget was not released")
        if not row.get("kv_released", False):
            problems.append(f"{stage}: KV blocks were not released")
        if not row.get("slot_released", False):
            problems.append(f"{stage}: queue/execution slot was not released")
        if row.get("duplicate_reservation"):
            problems.append(f"{stage}: a retry reserved the budget twice")
    return {"rows": list(rows), "ok": not problems, "problems": problems}


# ── residuals, false decisions and margin validation ─────────────────────


def prediction_residual(
    *, rows: Sequence[Mapping[str, Any]], s12_capacity_model_id: str, tolerance_fraction: float = 0.05
) -> Dict[str, Any]:
    """Steps 32/36: residual by component with an explanation, never a bigger constant."""
    if not s12_capacity_model_id:
        raise ConfigError(
            "the frozen S12 capacity model id is required: without it the residual cannot be fed back"
        )
    residual_rows: List[Dict[str, Any]] = []
    unexplained: List[Dict[str, Any]] = []
    for row in rows:
        try:
            predicted = rec.numeric_value(row.get("predicted_bytes"))
            actual = rec.numeric_value(row.get("actual_bytes"))
        except ConfigError as exc:
            raise ConfigError(f"residual row must carry numbers or a state, not both: {exc}") from exc
        residual = actual - predicted
        fraction = (residual / predicted) if predicted else float("inf")
        entry = {
            "run_id": row.get("run_id", ""),
            "workload_bucket": row.get("workload_bucket", ""),
            "rank_id": row.get("rank_id", ""),
            "component": row.get("component", ""),
            "predicted_bytes": predicted,
            "actual_bytes": actual,
            "residual": residual,
            "residual_fraction": fraction,
            "explanation": row.get("explanation", ""),
        }
        residual_rows.append(entry)
        if abs(fraction) > tolerance_fraction and not entry["explanation"]:
            unexplained.append(entry)
    return {
        "rows": residual_rows,
        "s12_capacity_model_id": s12_capacity_model_id,
        "unexplained": unexplained,
        "ok": not unexplained,
        "reason": (
            ""
            if not unexplained
            else "residuals exceed the tolerance without an explanation: enlarge the margin only after "
            "attributing the component (rounding/workspace/allocator/graph/comm/hidden allocation)"
        ),
    }


def false_decisions(
    *,
    decisions: Sequence[AdmissionDecision],
    counterfactual_method: str = "",
    counterfactual_rows: Sequence[Mapping[str, Any]] = (),
    false_admit_gate: float = DEFAULT_FALSE_ADMIT_GATE,
) -> Dict[str, Any]:
    """Step 33: false admit comes from real violations; false reject needs a counterfactual."""
    problems: List[str] = []
    false_admits = [
        decision
        for decision in decisions
        if decision.decision == "ADMIT_NOW" and (decision.oom or decision.outcome == "RESOURCE_INVARIANT_VIOLATION")
    ]
    counterfactual_rejects = [row for row in counterfactual_rows if row.get("safe_under_counterfactual")]
    eligible = [decision for decision in decisions if decision.decision == "ADMIT_NOW"]
    false_admit_rate = (len(false_admits) / len(eligible)) if eligible else 0.0
    if false_admits and false_admit_rate > false_admit_gate:
        problems.append(
            f"false admit rate {false_admit_rate:.4f} exceeds the pre-registered gate {false_admit_gate}"
        )
    if counterfactual_rejects and not counterfactual_method:
        problems.append(
            "false rejects are claimed without a counterfactual method (shadow policy / replay / "
            "independent controlled run)"
        )
    rejected = [decision for decision in decisions if decision.decision.startswith("REJECT")]
    false_reject_rate = (len(counterfactual_rejects) / len(rejected)) if rejected else 0.0
    return {
        "accepted": len(eligible),
        "rejected": len(rejected),
        "false_admit": len(false_admits),
        "false_admit_rate": false_admit_rate,
        "false_reject": len(counterfactual_rejects),
        "false_reject_rate": false_reject_rate,
        "counterfactual_method": counterfactual_method,
        "problems": problems,
        "ok": not problems,
    }


def margin_holdout_validation(
    *, holdout_rows: Sequence[Mapping[str, Any]], selected_margin_bytes: int, tuned_on_same_data: bool
) -> Dict[str, Any]:
    """Step 34: the margin must be validated on data that did not tune it."""
    problems: List[str] = []
    if tuned_on_same_data:
        problems.append(
            "the margin was tuned and validated on the same workload/day: that is not a validation"
        )
    ooms = [row for row in holdout_rows if row.get("oom")]
    margin_hits = [
        row for row in holdout_rows if int(row.get("predicted_peak_bytes", 0)) + selected_margin_bytes
        > int(row.get("usable_memory_bytes", 0))
    ]
    if ooms:
        problems.append(f"{len(ooms)} holdout rows hit the protected OOM path inside the safety region")
    return {
        "holdout_rows": len(holdout_rows),
        "selected_margin_bytes": selected_margin_bytes,
        "oom_rows": len(ooms),
        "rows_near_margin": len(margin_hits),
        "problems": problems,
        "ok": not problems,
    }


def s12_feedback(
    *, s12_capacity_model_id: str, residuals: Mapping[str, Any], update: Mapping[str, Any]
) -> Dict[str, Any]:
    """Step 36: keep the original prediction and produce a *separate* update."""
    problems: List[str] = []
    if not s12_capacity_model_id:
        problems.append("the original S12 capacity model id must be preserved")
    if not residuals.get("rows"):
        problems.append("no residuals: a model update without residuals is a fabricated calibration")
    if not update.get("model_version") or update.get("model_version") == s12_capacity_model_id:
        problems.append(
            "the calibration update must be a new model version (the original prediction may not be "
            "edited in place)"
        )
    if update.get("deleted_original_prediction"):
        problems.append("the original prediction must not be deleted")
    return {
        "s12_capacity_model_id": s12_capacity_model_id,
        "new_model_version": update.get("model_version", ""),
        "policy_binding": update.get("policy_version", ""),
        "problems": problems,
        "ok": not problems,
    }


def autoscaling_capacity_interface(
    *, policy: AdmissionPolicy, safe_capacity_per_replica: int, cold_start_headroom_s: float,
    metric_max_age_s: float, reject_policy: str
) -> Dict[str, Any]:
    """Step 37: the interface E13-07 consumes (capacity + thresholds + metric age)."""
    problems = policy.validate()
    if safe_capacity_per_replica <= 0:
        problems.append("safe per-replica capacity must be positive")
    if cold_start_headroom_s <= 0:
        problems.append("cold-start headroom must be declared (reactive scaling cannot cover it)")
    if metric_max_age_s <= 0:
        problems.append("a metric max-age is required (stale metrics may not drive scaling)")
    if not reject_policy:
        problems.append("the reject policy must be exported so autoscaling knows the overload response")
    return {
        "policy_version": policy.version,
        "policy_kind": policy.kind,
        "safe_capacity_per_replica": safe_capacity_per_replica,
        "queue_token_limit": policy.queue_token_limit,
        "cold_start_headroom_s": cold_start_headroom_s,
        "metric_max_age_s": metric_max_age_s,
        "reject_policy": reject_policy,
        "problems": problems,
        "ok": not problems,
    }


# ── sweep and fault cases ────────────────────────────────────────────────


@dataclass
class SweepPlan:
    """Step 4: the safe→boundary→overload stepping with its abort guard."""

    plan_id: str
    stages: Tuple[str, ...] = SWEEP_STAGES
    batch_values: Tuple[int, ...] = ()
    isl_values: Tuple[int, ...] = ()
    osl_values: Tuple[int, ...] = ()
    concurrency_values: Tuple[int, ...] = ()
    rate_values: Tuple[float, ...] = ()
    abort_on_oom: bool = True
    abort_on_slo_violation: bool = True
    max_device_memory_fraction: float = 0.0
    recovery_step: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.plan_id:
            problems.append("sweep plan needs an id")
        if tuple(self.stages) != SWEEP_STAGES:
            problems.append(f"the sweep must cover {list(SWEEP_STAGES)} in order")
        for name in ("batch_values", "isl_values", "osl_values", "concurrency_values"):
            if not getattr(self, name):
                problems.append(f"sweep plan must declare {name!r} (no ad-hoc sweeping)")
        if not self.abort_on_oom:
            problems.append("an OOM abort guard is mandatory")
        if not self.abort_on_slo_violation:
            problems.append("an SLO-violation abort guard is mandatory")
        if not 0 < self.max_device_memory_fraction <= 1:
            problems.append("a maximum device memory fraction (safety ceiling) must be pre-registered")
        if not self.recovery_step:
            problems.append("the recovery step after the boundary must be declared (drain the window)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "stages": list(self.stages),
            "batch_values": list(self.batch_values),
            "isl_values": list(self.isl_values),
            "osl_values": list(self.osl_values),
            "concurrency_values": list(self.concurrency_values),
            "rate_values": list(self.rate_values),
            "abort_on_oom": self.abort_on_oom,
            "abort_on_slo_violation": self.abort_on_slo_violation,
            "max_device_memory_fraction": self.max_device_memory_fraction,
            "recovery_step": self.recovery_step,
        }


def run_capacity_fault_cases(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Steps 26/27/30/31: cancel/error/over-budget/OOM must be bounded and recoverable."""
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        kind = str(case.get("kind", ""))
        if kind not in FAULT_CASES:
            raise ConfigError(f"unknown capacity fault case {kind!r}")
        served = bool(case.get("served_traffic", False))
        recovered = bool(case.get("recovered", False))
        isolated = bool(case.get("blast_radius_isolated", True))
        rows.append(
            {
                "case_id": str(case.get("case_id", f"cap-fault-{index:03d}")),
                "kind": kind,
                "expected": str(case.get("expected", "BOUNDED_AND_RECOVERABLE")),
                "observed": str(case.get("observed", "")),
                "released": bool(case.get("released", False)),
                "blast_radius_isolated": isolated,
                "ok": recovered and not served and isolated,
            }
        )
    failures = [row for row in rows if not row["ok"]]
    return {
        "rows": rows,
        "cases": len(rows),
        "failures": failures,
        "ok": not failures,
        "reason": (
            ""
            if not failures
            else "capacity fault cases that were unbounded, served traffic or affected other tenants: "
            + ", ".join(row["kind"] for row in failures)
        ),
    }


def capacity_verdict(
    *,
    ledger: Mapping[str, Any],
    policy_comparison: Mapping[str, Any],
    residuals: Mapping[str, Any],
    false_rates: Mapping[str, Any],
    margin: Mapping[str, Any],
    faults: Mapping[str, Any],
    feedback: Mapping[str, Any],
) -> Dict[str, Any]:
    """Step 38: the safety domain and limits the stage may claim."""
    problems: List[str] = []
    for name, axis in (
        ("memory_ledger", ledger),
        ("policy_comparison", policy_comparison),
        ("residuals", residuals),
        ("false_decisions", false_rates),
        ("margin_holdout", margin),
        ("faults", faults),
        ("s12_feedback", feedback),
    ):
        if not axis.get("ok"):
            problems.append(f"{name}: " + "; ".join(axis.get("problems", []) or ["(no detail)"]))
    return {
        "problems": problems,
        "verdict": "PASSABLE_AT_CODE_LEVEL" if not problems else "BLOCKED",
        "note": "the safe region, boundary behaviour and OOM margin come from the executed experiment",
    }


# ── protocol steps and smoke self-check ──────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结容量模型版本", ("capacity:capacity_model_binding", "capacity:prediction_residual")),
    (2, "冻结可用内存边界", ("capacity:memory_ledger", "capacity:MEMORY_COMPONENTS")),
    (3, "冻结 admission policies", ("capacity:AdmissionPolicy", "capacity:ADMISSION_POLICIES")),
    (4, "冻结安全扫参计划", ("capacity:SweepPlan", "capacity:SWEEP_STAGES")),
    (5, "冻结 workload 与 quality/SLO", ("capacity:token_work", "serving:slo.SLOSpec")),
    (6, "建立内存观测账本", ("capacity:MemoryComponent",)),
    (7, "测空 Pod/runtime baseline", ("capacity:memory_ledger",)),
    (8, "测模型 loaded baseline", ("capacity:memory_ledger", "artifacts:load_and_measure")),
    (9, "测 warmup 增量", ("capacity:memory_ledger", "artifacts:WarmupResult")),
    (10, "校准单请求 KV 增量", ("capacity:KVModel", "capacity:KVModel.incremental_bytes")),
    (11, "校准 batch/workspace 增量", ("capacity:MemoryComponent",)),
    (12, "校准 prefill/decode 相位", ("capacity:MemoryComponent", "capacity:KVModel")),
    (13, "验证多 rank 内存模型", ("capacity:memory_ledger",)),
    (14, "运行低负载 policy 基线", ("capacity:admit",)),
    (15, "运行 homogeneous 安全扫参", ("capacity:SweepPlan", "capacity:admit")),
    (16, "运行 mixed-length 扫参", ("capacity:token_work", "capacity:admit")),
    (17, "运行 long-context 边界", ("capacity:KVModel.incremental_bytes",)),
    (18, "运行 decode-heavy 边界", ("capacity:token_work",)),
    (19, "运行 open-loop 过载", ("capacity:admit", "autoscaling:StaleMetricPolicy")),
    (20, "比较 admission policies", ("capacity:admission_policy_comparison",)),
    (21, "验证 decision linearization", ("capacity:reservation_atomicity",)),
    (22, "验证 prefix cache 影响", ("capacity:KVModel.incremental_bytes",)),
    (23, "验证 speculative decoding 影响", ("capacity:BUDGET_FEATURES",)),
    (24, "验证 chunked prefill/batching 影响", ("capacity:BUDGET_FEATURES",)),
    (25, "验证 KV quant/precision 影响", ("capacity:KVModel",)),
    (26, "注入取消/超时", ("capacity:cancel_release",)),
    (27, "注入请求失败/backend 错误", ("capacity:cancel_release",)),
    (28, "构造 allocator fragmentation 历史", ("capacity:fragmentation_history",)),
    (29, "验证模型切换/双驻留", ("capacity:dual_residency_budget",)),
    (30, "执行 over-budget 负例", ("capacity:run_capacity_fault_cases",)),
    (31, "执行受控 OOM 故障", ("capacity:run_capacity_fault_cases",)),
    (32, "计算 prediction residual", ("capacity:prediction_residual",)),
    (33, "估计 false admit/reject", ("capacity:false_decisions",)),
    (34, "选择/验证 safety margin", ("capacity:margin_holdout_validation",)),
    (35, "验证预算恢复/长期稳定性", ("capacity:cancel_release", "capacity:reservation_atomicity")),
    (36, "回流 S12 容量模型", ("capacity:s12_feedback",)),
    (37, "生成 S13-07 capacity 接口", ("capacity:autoscaling_capacity_interface",)),
    (38, "形成 capacity/admission verdict", ("capacity:capacity_verdict",)),
)


def capacity_model_binding(
    *, s12_capacity_model_id: str, s12_prediction_ids: Sequence[str], policy_version: str
) -> Dict[str, Any]:
    """Step 1: freeze the S12 model and predictions *before* looking at S13 results."""
    problems: List[str] = []
    if not s12_capacity_model_id:
        problems.append("the S12 capacity model id must be frozen")
    if not s12_prediction_ids:
        problems.append(
            "the original S12 predictions must be listed: they may not be re-derived after the S13 run"
        )
    if not policy_version:
        problems.append("the S13 admission policy version must be bound to a model version")
    return {
        "s12_capacity_model_id": s12_capacity_model_id,
        "s12_prediction_ids": list(s12_prediction_ids),
        "policy_version": policy_version,
        "problems": problems,
        "ok": not problems,
    }


def admission_policy_comparison(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 20: policies compared on the same trace with the same tuning budget."""
    problems: List[str] = []
    seen: List[str] = []
    for row in rows:
        policy_id = str(row.get("policy_id", ""))
        seen.append(policy_id)
        if not row.get("same_trace", True):
            problems.append(f"{policy_id}: compared on a different trace")
        if row.get("tuning_budget_minutes") is not None and int(row["tuning_budget_minutes"]) <= 0:
            problems.append(f"{policy_id}: no tuning budget recorded (an unfair baseline is not a baseline)")
    if len(set(seen)) != len(seen):
        problems.append("duplicate policy ids in the comparison")
    return {"rows": list(rows), "policies": sorted(set(seen)), "problems": problems, "ok": not problems}


def fragmentation_history(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 28: fragmentation is a measured residual, not an excuse for a bigger constant."""
    problems: List[str] = []
    for row in rows:
        if not row.get("cycles"):
            problems.append("a fragmentation episode must record the interleaving cycles")
        if row.get("recovered") is False:
            problems.append("fragmentation did not recover within the declared window")
        if row.get("residual_unexplained"):
            problems.append("fragmentation residual is unexplained")
    return {"rows": list(rows), "problems": problems, "ok": not problems}


def dual_residency_budget(
    *, active_bytes: int, candidate_bytes: int, usable_memory_bytes: int, safety_margin_bytes: int
) -> Dict[str, Any]:
    """Step 29: a model switch peak must fit the budget (or the switch rolls, not OOMs)."""
    total = active_bytes + candidate_bytes
    allowed = usable_memory_bytes - safety_margin_bytes
    fits = total <= allowed
    return {
        "active_bytes": active_bytes,
        "candidate_bytes": candidate_bytes,
        "total_bytes": total,
        "allowed_bytes": allowed,
        "fits": fits,
        "admission_during_switch": "MEMORY_KV_PREDICTED",
        "note": (
            "if both versions cannot be resident, use pod-level rolling instead of raising the OOM margin"
            if not fits
            else ""
        ),
    }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the capacity/admission contracts (smoke, not an experiment)."""
    checks: Dict[str, Any] = {}
    kv = KVModel(layers=28, kv_heads=4, head_dim=128, bytes_per_element=2, block_tokens=16,
                 block_bytes_overhead=64)
    checks["kv_positive"] = kv.bytes_per_token() > 0 and kv.incremental_bytes(100) > 0
    checks["prefix_share_reduces_budget"] = kv.incremental_bytes(100, prefix_shared_tokens=64) < kv.incremental_bytes(100)

    try:
        KVModel(layers=0, block_bytes_overhead=0).validate()
        validation = KVModel(layers=0, block_bytes_overhead=0).validate()
    except ConfigError:  # pragma: no cover - validation never raises
        validation = ["error"]
    checks["invalid_kv_model_flagged"] = len(validation) >= 2

    policy = AdmissionPolicy(policy_id="p1", kind="MEMORY_KV_PREDICTED", safety_margin_bytes=1 << 30,
                             token_budget=100_000, version="v1")
    checks["policy_valid"] = policy.validate() == []

    decision = AdmissionDecision(
        decision_id="d1", request_id="r1", policy_version="v1", prompt_tokens=1000,
        requested_output_tokens=500, usable_memory_bytes=24 << 30, safety_margin_bytes=1 << 30,
        release_id="rel1", model_artifact_id="m1",
    )
    admit(policy=policy, kv_model=kv, decision=decision)
    checks["admit_within_budget"] = decision.decision == "ADMIT_NOW" and decision.reservation_id != ""

    huge = AdmissionDecision(
        decision_id="d2", request_id="r2", policy_version="v1", prompt_tokens=10,
        requested_output_tokens=10, usable_memory_bytes=1 << 20, safety_margin_bytes=1 << 19,
        release_id="rel1", model_artifact_id="m1",
    )
    admit(policy=policy, kv_model=kv, decision=huge)
    checks["over_budget_rejected_before_execution"] = huge.decision.startswith("REJECT")

    oversell = reservation_atomicity(
        [{"trial_id": "t1", "capacity_bytes": 100, "reserved_bytes": 90, "committed_bytes": 120}]
    )
    checks["oversell_detected"] = oversell["ok"] is False

    release = cancel_release(
        [{"stage": "decode", "budget_released": True, "kv_released": True, "slot_released": True}]
    )
    checks["cancel_releases_budget"] = release["ok"] is True

    bad_release = cancel_release([{"stage": "decode", "budget_released": False, "kv_released": False,
                                   "slot_released": True}])
    checks["cancel_leak_detected"] = bad_release["ok"] is False
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "checks": checks,
        "note": "接口自检；未分配显存、未运行任何模型",
    }