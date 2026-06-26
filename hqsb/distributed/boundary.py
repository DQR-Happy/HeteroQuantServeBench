"""PP/CP/SP selection gate, minimal loops, bubble and memory boundaries (E10-07).

E10-07 is P1: it only becomes a hard gate when the project claims pipeline,
context or sequence parallelism.  The module therefore starts with an
*activation decision* (``NOT_RUN_NOT_CLAIMED`` is a legitimate delivered state)
and refuses two anti-patterns:

* "``SP=True``" without a named algorithm — :func:`validate_candidates`
  deletes candidates whose mechanism is undefined;
* shallow demos of all three — :func:`select_primary` returns exactly one
  primary (or a documented ``PASS_NEGATIVE`` selection when none qualifies).

The PP bubble approximation and the per-stage objective are kept as *models*
with the plan; the measured timeline is what decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.distributed.parallel_plan import (
    QwenArchitectureCensus,
    kv_per_token_bytes,
)

#: Parallelism candidates (details E10-07 §2).
CANDIDATE_KINDS: Tuple[str, ...] = ("pp", "cp", "sp")

#: Named CP/SP mechanisms; an unnamed algorithm is not a candidate.
CP_SP_ALGORITHMS: Tuple[str, ...] = (
    "ulysses_all_to_all",
    "ring_attention",
    "megatron_sequence_split",
    "chunked_sequence_shard",
)

#: The rubric rows of details E10-07 §4.
RUBRIC_FACTORS: Tuple[str, ...] = (
    "weights_do_not_fit",
    "long_kv_attention_memory",
    "high_concurrency_fills_pipeline",
    "low_concurrency_interactive",
    "inter_layer_imbalance_risk",
    "alltoall_or_ring_topology_dependency",
    "current_runtime_support",
)

#: Rubric weights: `high`=3, `medium`=2, `low`=1.
RUBRIC_SCORES: Mapping[str, int] = {"high": 3, "medium": 2, "low": 1, "risk": 0}

#: Per-factor expectation per candidate (details E10-07 §4 table).
RUBRIC_TABLE: Mapping[str, Mapping[str, str]] = {
    "weights_do_not_fit": {"pp": "high", "cp": "low", "sp": "low"},
    "long_kv_attention_memory": {"pp": "medium", "cp": "high", "sp": "high"},
    "high_concurrency_fills_pipeline": {"pp": "high", "cp": "medium", "sp": "medium"},
    "low_concurrency_interactive": {"pp": "low", "cp": "medium", "sp": "medium"},
    "inter_layer_imbalance_risk": {"pp": "risk", "cp": "low", "sp": "low"},
    "alltoall_or_ring_topology_dependency": {"pp": "low", "cp": "high", "sp": "high"},
    "current_runtime_support": {"pp": "medium", "cp": "medium", "sp": "medium"},
}

#: Adoption criteria of details E10-07 §12.
ADOPT_REJECT_CRITERIA: Tuple[str, ...] = (
    "correctness and quality gates",
    "capacity problem solved or primary metric improved",
    "no unacceptable TTFT/TPOT/P99 regression",
    "max-rank memory inside the safety bound",
    "communication ledger closed",
    "fault/cleanup semantics viable",
    "complexity justified by the target workload frequency",
)

#: When a rejected choice should be re-evaluated (details E10-07 §12 end).
RE_EVALUATION_TRIGGERS: Tuple[str, ...] = (
    "model layer count changes",
    "context length distribution changes",
    "interconnect/topology changes",
    "runtime version changes",
    "target concurrency profile changes",
)

#: Health probe chain used after a boundary experiment changes the plan.
PROBE_CHAIN: Tuple[str, ...] = (
    "device",
    "p2p_network",
    "small_collective",
    "tp_block",
    "qwen_tiny",
    "representative_short_request",
)

#: Unsupported/failure cases the boundary experiment must cover (step 26).
UNSUPPORTED_CASES: Tuple[str, ...] = (
    "microbatch_not_divisible",
    "sequence_not_divisible",
    "stage_oom",
    "rank_mismatch",
    "timeout",
)


def _require(condition: bool, message: str, *, field_name: str = "") -> None:
    if not condition:
        raise ConfigError(message, details={"field": field_name} if field_name else None)


# ── activation ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ActivationDecision:
    """Whether E10-07 is in scope at all (details E10-07 step 1)."""

    activated: bool
    reason: str
    claim_scope: str = ""
    status: str = "NOT_RUN_NOT_CLAIMED"

    def __post_init__(self) -> None:
        if not self.activated:
            _require(
                bool(self.reason),
                "a non-activated P1 must record why (no claim, TP satisfies the goal)",
                field_name="reason",
            )
            if self.status != "NOT_RUN_NOT_CLAIMED":
                raise ConfigError(
                    "an inactivated P1 is NOT_RUN_NOT_CLAIMED; it is never PASS/FAIL",
                    details={"field": "status"},
                )
        else:
            _require(bool(self.claim_scope), "an activated experiment needs a claim scope", field_name="claim_scope")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "activated": self.activated,
            "reason": self.reason,
            "claim_scope": self.claim_scope,
            "status": self.status,
        }


def activation_decision(
    *, claiming_parallelism: bool, tp_satisfies_goal: bool, reason: str = ""
) -> ActivationDecision:
    """Activate only when the project claims PP/CP/SP or TP cannot meet the goal."""
    if claiming_parallelism:
        return ActivationDecision(activated=True, reason="", claim_scope="pp/cp/sp")
    if tp_satisfies_goal:
        return ActivationDecision(
            activated=False,
            reason=reason or "TP already satisfies the capacity/latency goal; no PP/CP/SP claim",
        )
    return ActivationDecision(
        activated=True,
        reason="",
        claim_scope=reason or "TP does not meet the capacity/long-context goal",
    )


# ── candidates ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CandidateDefinition:
    """One concrete candidate; the mechanism must be named (step 3)."""

    kind: str
    algorithm_name: str
    framework_impl: str
    inference_semantics: str
    communication_pattern: str
    supported: bool = True

    def __post_init__(self) -> None:
        _require(self.kind in CANDIDATE_KINDS, "unknown candidate kind", field_name="kind")
        _require(
            bool(self.algorithm_name.strip()),
            "a candidate without a named algorithm is deleted, not scored (no bare 'SP=True')",
            field_name="algorithm_name",
        )
        if self.kind in ("cp", "sp"):
            _require(
                self.algorithm_name in CP_SP_ALGORITHMS,
                f"CP/SP algorithm must be one of {CP_SP_ALGORITHMS}",
                field_name="algorithm_name",
            )
        _require(bool(self.inference_semantics), "state the inference semantics", field_name="inference_semantics")
        _require(bool(self.communication_pattern), "state the communication pattern", field_name="communication_pattern")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "algorithm_name": self.algorithm_name,
            "framework_impl": self.framework_impl,
            "inference_semantics": self.inference_semantics,
            "communication_pattern": self.communication_pattern,
            "supported": self.supported,
        }


def validate_candidates(candidates: Sequence[CandidateDefinition]) -> Dict[str, Any]:
    """Reject undefined mechanisms and duplicates (step 3)."""
    problems: List[str] = []
    seen: set = set()
    for candidate in candidates:
        key = (candidate.kind, candidate.algorithm_name)
        if key in seen:
            problems.append(f"duplicate candidate {key}")
        seen.add(key)
    if not candidates:
        problems.append("no candidate survived the definition gate")
    return {"ok": not problems, "problems": problems, "candidates": len(candidates)}


@dataclass(frozen=True)
class RubricScore:
    """Scores per candidate with the hard capability gate separated (step 6)."""

    kind: str
    factor_scores: Mapping[str, int]
    hard_gate_ok: bool
    total: int = 0

    def __post_init__(self) -> None:
        _require(self.kind in CANDIDATE_KINDS, "unknown candidate kind", field_name="kind")
        unknown = sorted(set(self.factor_scores) - set(RUBRIC_FACTORS))
        if unknown:
            raise ConfigError(f"unknown rubric factors {unknown}", details={"field": "factor_scores"})

    def with_total(self) -> "RubricScore":
        return RubricScore(
            kind=self.kind,
            factor_scores=self.factor_scores,
            hard_gate_ok=self.hard_gate_ok,
            total=sum(self.factor_scores.values()) if self.hard_gate_ok else 0,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "factor_scores": dict(sorted(self.factor_scores.items())),
            "hard_gate_ok": self.hard_gate_ok,
            "total": self.total,
        }


def score_candidates(
    *,
    capability: Mapping[str, bool],
    override_factors: Optional[Mapping[str, Mapping[str, int]]] = None,
) -> List[RubricScore]:
    """Score every candidate from the rubric table, then the hard gate decides."""
    override_factors = dict(override_factors or {})
    scores: List[RubricScore] = []
    for kind in CANDIDATE_KINDS:
        factors = {
            factor: RUBRIC_SCORES[RUBRIC_TABLE[factor][kind]] for factor in RUBRIC_FACTORS
        }
        for factor, values in override_factors.items():
            if kind in values:
                factors[factor] = int(values[kind])
        scores.append(
            RubricScore(
                kind=kind,
                factor_scores=factors,
                hard_gate_ok=bool(capability.get(kind, False)),
            ).with_total()
        )
    return scores


def select_primary(scores: Sequence[RubricScore]) -> Dict[str, Any]:
    """Pick exactly one primary, or a documented negative selection (step 6)."""
    eligible = [score for score in scores if score.hard_gate_ok]
    if not eligible:
        return {
            "selected": None,
            "status": "PASS_NEGATIVE",
            "reason": (
                "no candidate passed the capability/correctness hard gate; do not force one "
                "(details E10-07 §10: 未定义具体 SP 算法 / 同时浅做三种 都被禁止)"
            ),
            "scores": [score.as_dict() for score in scores],
        }
    winner = max(eligible, key=lambda score: score.total)
    ties = [score for score in eligible if score.total == winner.total and score.kind != winner.kind]
    return {
        "selected": winner.kind,
        "status": "SELECTED",
        "reason": "highest rubric total among candidates that passed the hard gate",
        "tie_break_required": [score.kind for score in ties],
        "scores": [score.as_dict() for score in scores],
    }


# ── capability / analytic model ────────────────────────────────────────────


def capability_gate(
    *,
    runtime_support: bool,
    head_gqa_ok: bool,
    kv_ok: bool,
    collective_ok: bool,
    dynamic_request_ok: bool,
    hardware_ok: bool,
    reasons: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Capability/correctness is a hard gate; performance comes later (step 4)."""
    checks = {
        "runtime_support": runtime_support,
        "head_gqa_ok": head_gqa_ok,
        "kv_ok": kv_ok,
        "collective_ok": collective_ok,
        "dynamic_request_ok": dynamic_request_ok,
        "hardware_ok": hardware_ok,
    }
    reasons = dict(reasons or {})
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "ok": not failed,
        "failed": failed,
        "reasons": {name: reasons.get(name, "not verified") for name in failed},
        "per_candidate": checks,
    }


@dataclass(frozen=True)
class AnalyticalCosts:
    """Per-rank weights/KV/activation, P2P bytes, steps and ideal bubble (step 5)."""

    kind: str
    degree: int
    per_rank_weight_bytes: int
    per_rank_kv_bytes: int
    per_rank_activation_bytes: int
    p2p_bytes_per_step: int
    steps_per_token: int
    ideal_bubble_fraction: Optional[float]
    predicted_compute_ms: float = 0.0
    assumptions: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require(self.kind in CANDIDATE_KINDS, "unknown candidate kind", field_name="kind")
        _require(self.degree >= 1, "degree must be >= 1", field_name="degree")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "degree": self.degree,
            "per_rank_weight_bytes": self.per_rank_weight_bytes,
            "per_rank_kv_bytes": self.per_rank_kv_bytes,
            "per_rank_activation_bytes": self.per_rank_activation_bytes,
            "p2p_bytes_per_step": self.p2p_bytes_per_step,
            "steps_per_token": self.steps_per_token,
            "ideal_bubble_fraction": self.ideal_bubble_fraction,
            "predicted_compute_ms": self.predicted_compute_ms,
            "assumptions": list(self.assumptions),
        }


def pipeline_bubble_ideal(*, stages: int, microbatches: int) -> Dict[str, float]:
    """``utilization ≈ m/(m+s-1)`` (details E10-07 §3.1) — a baseline, not a result."""
    _require(stages >= 1 and microbatches >= 1, "stages/microbatches must be >= 1")
    denominator = microbatches + stages - 1
    return {
        "stages": stages,
        "microbatches": microbatches,
        "ideal_utilization": microbatches / denominator,
        "ideal_bubble_fraction": (stages - 1) / denominator,
    }


def analytical_costs(
    census: QwenArchitectureCensus,
    *,
    kind: str,
    degree: int,
    stages: int = 1,
    microbatches: int = 1,
    tokens: int = 0,
    dtype_bytes: int = 2,
    kv_dtype_bytes: int = 2,
) -> AnalyticalCosts:
    """Derive the per-rank costs with the assumptions written down (step 5)."""
    _require(kind in CANDIDATE_KINDS, "unknown candidate kind", field_name="kind")
    _require(degree >= 1, "degree must be >= 1", field_name="degree")
    total_weight = census.total_weight_elements() * dtype_bytes
    if kind == "pp":
        _require(stages >= 1, "PP needs stages", field_name="stages")
        layers_per_stage = max(1, census.num_layers // stages)
        weight = total_weight * layers_per_stage / census.num_layers
        activation = tokens * census.hidden_size * dtype_bytes * 2
        p2p = activation
        steps = 1
        bubble = pipeline_bubble_ideal(stages=stages, microbatches=microbatches)
        kv_local, kv_per_token = kv_per_token_bytes(census, degree=1, kv_dtype_bytes=kv_dtype_bytes)
        kv = kv_per_token * tokens
        assumptions = (
            "equal layer split per stage (stage balance is measured, not assumed)",
            "activation send size = batch×hidden×2 send+recv",
            f"layers per stage ≈ {layers_per_stage}",
        )
        return AnalyticalCosts(
            kind="pp",
            degree=stages,
            per_rank_weight_bytes=int(weight),
            per_rank_kv_bytes=kv,
            per_rank_activation_bytes=activation,
            p2p_bytes_per_step=p2p,
            steps_per_token=steps,
            ideal_bubble_fraction=bubble["ideal_bubble_fraction"],
            assumptions=assumptions,
        )
    local_kv_heads, kv_per_token = kv_per_token_bytes(
        census, degree=degree, kv_dtype_bytes=kv_dtype_bytes
    )
    if kind == "cp":
        activation = tokens * census.hidden_size * dtype_bytes
        p2p = 2 * activation  # ring/all-to-all exchange both directions
        steps = 2
        assumptions = (
            "context is split evenly; causal mask must be handled per shard",
            "KV heads are sharded or ring-exchanged per the named algorithm",
        )
    else:  # sp (sequence parallel: activation only)
        activation = tokens * census.hidden_size * dtype_bytes
        p2p = 2 * activation
        steps = 2
        assumptions = (
            "sequence activation sharded; attention needs an AllToAll/ring per layer",
            "no KV sharding is assumed unless the named algorithm says so",
        )
    return AnalyticalCosts(
        kind=kind,
        degree=degree,
        per_rank_weight_bytes=int(total_weight / degree),
        per_rank_kv_bytes=kv_per_token * tokens,
        per_rank_activation_bytes=activation,
        p2p_bytes_per_step=p2p,
        steps_per_token=steps,
        ideal_bubble_fraction=None,
        assumptions=assumptions,
    )


# ── plans / partitions ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelinePlan:
    """PP stage/layer/microbatch plan (details E10-07 step 8)."""

    stages: int
    layers_per_stage: Tuple[int, ...]
    stage_rank: Mapping[int, int]
    microbatches: int
    activation_send_bytes: int = 0
    kv_owner: str = "producing stage"
    group_placement_hash: str = ""
    artifact_hash: str = ""

    def __post_init__(self) -> None:
        _require(self.stages >= 1, "stages must be >= 1", field_name="stages")
        _require(
            len(self.layers_per_stage) == self.stages,
            "layers_per_stage must have one entry per stage",
            field_name="layers_per_stage",
        )
        _require(self.microbatches >= 1, "microbatches must be >= 1", field_name="microbatches")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stages": self.stages,
            "layers_per_stage": list(self.layers_per_stage),
            "stage_rank": {str(k): v for k, v in sorted(self.stage_rank.items())},
            "microbatches": self.microbatches,
            "activation_send_bytes": self.activation_send_bytes,
            "kv_owner": self.kv_owner,
            "group_placement_hash": self.group_placement_hash,
            "artifact_hash": self.artifact_hash,
        }


def verify_partition(
    *, partitions: Mapping[int, Tuple[int, int]], total: int, name: str
) -> Dict[str, Any]:
    """Coverage/no-overlap conservation for any partition (step 9)."""
    if total <= 0:
        raise ConfigError("total must be positive", details={"field": "total"})
    ranges = sorted(partitions.values())
    problems: List[str] = []
    cursor = 0
    for start, end in ranges:
        if start < cursor:
            problems.append(f"{name}: range [{start},{end}) overlaps the previous one")
        if start > cursor:
            problems.append(f"{name}: gap [{cursor},{start})")
        cursor = max(cursor, end)
    if cursor != total:
        problems.append(f"{name}: coverage ends at {cursor}, expected {total}")
    return {"ok": not problems, "problems": problems, "partitions": {str(k): list(v) for k, v in sorted(partitions.items())}}


@dataclass(frozen=True)
class StageBalanceRow:
    stage: int
    rank: int
    layers: int
    compute_ms: float
    weight_bytes: int
    kv_bytes: int
    activation_send_bytes: int
    workspace_peak_bytes: int
    exposed_comm_ms: float = 0.0
    runtime_gap_ms: float = 0.0

    @property
    def stage_latency_ms(self) -> float:
        return self.compute_ms + self.exposed_comm_ms + self.runtime_gap_ms

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "rank": self.rank,
            "layers": self.layers,
            "compute_ms": self.compute_ms,
            "weight_bytes": self.weight_bytes,
            "kv_bytes": self.kv_bytes,
            "activation_send_bytes": self.activation_send_bytes,
            "workspace_peak_bytes": self.workspace_peak_bytes,
            "exposed_comm_ms": self.exposed_comm_ms,
            "runtime_gap_ms": self.runtime_gap_ms,
            "stage_latency_ms": self.stage_latency_ms,
        }


def stage_balance(rows: Sequence[StageBalanceRow]) -> Dict[str, Any]:
    """Slowest stage / max memory decide; the mean hides the hotspot (step 20)."""
    if not rows:
        raise ConfigError("stage_balance needs rows")
    slowest = max(rows, key=lambda row: row.stage_latency_ms)
    heaviest = max(rows, key=lambda row: row.weight_bytes + row.kv_bytes + row.workspace_peak_bytes)
    total = sum(row.stage_latency_ms for row in rows)
    return {
        "slowest_stage": slowest.stage,
        "slowest_rank": slowest.rank,
        "slowest_latency_ms": slowest.stage_latency_ms,
        "heaviest_stage": heaviest.stage,
        "max_stage_memory_bytes": heaviest.weight_bytes + heaviest.kv_bytes + heaviest.workspace_peak_bytes,
        "total_stage_latency_ms": total,
        "imbalance_fraction": (slowest.stage_latency_ms - total / len(rows)) / (total / len(rows)),
        "note": "throughput is bounded by max_s(stage_latency), capacity by max_s(stage_memory)",
        "rows": [row.as_dict() for row in rows],
    }


def bubble_metrics(*, measured_idle_ms: float, total_ms: float, stages: int, microbatches: int) -> Dict[str, Any]:
    """Measured bubble vs the ideal approximation, with the difference explained (step 21)."""
    if total_ms <= 0:
        raise ConfigError("total_ms must be positive", details={"field": "total_ms"})
    ideal = pipeline_bubble_ideal(stages=stages, microbatches=microbatches)
    measured_fraction = measured_idle_ms / total_ms
    return {
        "measured_bubble_fraction": measured_fraction,
        "ideal_bubble_fraction": ideal["ideal_bubble_fraction"],
        "difference_fraction": measured_fraction - ideal["ideal_bubble_fraction"],
        "explained_by": [
            "fill/drain",
            "stage imbalance",
            "communication",
            "host submission gaps",
        ],
        "note": "an ideal-bubble number without a timeline is not evidence",
    }


@dataclass(frozen=True)
class ContextPlan:
    """CP/SP shard plan (details E10-07 step 8)."""

    kind: str
    algorithm_name: str
    degree: int
    sequence_shards: Tuple[Tuple[int, int], ...]
    head_groups: Tuple[Tuple[int, int], ...] = ()
    kv_sharded: bool = False
    mask_correctness_note: str = ""
    group_placement_hash: str = ""

    def __post_init__(self) -> None:
        _require(self.kind in ("cp", "sp"), "ContextPlan is for cp/sp", field_name="kind")
        _require(
            self.algorithm_name in CP_SP_ALGORITHMS,
            f"algorithm must be one of {CP_SP_ALGORITHMS}",
            field_name="algorithm_name",
        )
        _require(
            len(self.sequence_shards) == self.degree,
            "one sequence shard per rank",
            field_name="sequence_shards",
        )
        _require(bool(self.mask_correctness_note), "causal mask handling must be stated", field_name="mask_correctness_note")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "algorithm_name": self.algorithm_name,
            "degree": self.degree,
            "sequence_shards": [list(item) for item in self.sequence_shards],
            "head_groups": [list(item) for item in self.head_groups],
            "kv_sharded": self.kv_sharded,
            "mask_correctness_note": self.mask_correctness_note,
            "group_placement_hash": self.group_placement_hash,
        }


def cp_sp_metrics(
    *,
    plan: ContextPlan,
    sequence_length: int,
    hidden: int,
    dtype_bytes: int,
    kv_heads_local: int,
    head_dim: int,
    layers: int,
) -> Dict[str, Any]:
    """Per-rank context/head/KV, AllToAll/ring steps and bytes (step 22)."""
    if sequence_length <= 0:
        raise ConfigError("sequence_length must be positive", details={"field": "sequence_length"})
    per_rank_tokens = sequence_length // plan.degree + (1 if sequence_length % plan.degree else 0)
    local_activation = per_rank_tokens * hidden * dtype_bytes
    if plan.algorithm_name == "ulysses_all_to_all":
        steps = 2
        alltoall_bytes = 2 * local_activation
        ring_bytes = 0
    else:
        steps = plan.degree - 1
        ring_bytes = local_activation * (plan.degree - 1)
        alltoall_bytes = 0
    local_kv = 0
    if plan.kv_sharded:
        local_kv = layers * sequence_length * kv_heads_local * head_dim * 2 * dtype_bytes
    return {
        "per_rank_tokens": per_rank_tokens,
        "local_activation_bytes": local_activation,
        "alltoall_bytes": alltoall_bytes,
        "ring_bytes": ring_bytes,
        "steps": steps,
        "local_kv_bytes": local_kv,
        "note": "communication bytes/steps come from the named algorithm, not from a generic label",
    }


def stage_partition_objective(rows: Sequence[StageBalanceRow]) -> Dict[str, Any]:
    """The auditable PP objective of §11 (not "equal layers per stage")."""
    balance = stage_balance(rows)
    return {
        "throughput_lower_bound_ms": balance["slowest_latency_ms"],
        "capacity_upper_bound_bytes": balance["max_stage_memory_bytes"],
        "cut_communication_bytes": sum(row.activation_send_bytes for row in rows),
        "slowest_stage": balance["slowest_stage"],
        "objective_note": (
            "weigh max stage latency, max stage memory, cut communication and re-partition "
            "complexity for the target workload; do not divide layers equally and call it done"
        ),
    }


# ── sweeps / comparison / verdict ──────────────────────────────────────────


def microbatch_sweep(*, stages: int, values: Sequence[int]) -> List[Dict[str, Any]]:
    if not values:
        raise ConfigError("microbatch sweep needs values")
    return [
        {
            "microbatches": value,
            "ideal_bubble_fraction": pipeline_bubble_ideal(stages=stages, microbatches=value)[
                "ideal_bubble_fraction"
            ],
            "workload_changed": False,
        }
        for value in values
    ]


def sequence_sweep(*, degree: int, lengths: Sequence[int]) -> List[Dict[str, Any]]:
    return [
        {
            "sequence_length": length,
            "degree": degree,
            "per_rank_tokens": length // degree + (1 if length % degree else 0),
            "divisible": length % degree == 0,
        }
        for length in lengths
    ]


def adversarial_cases(*, stages: int, degree: int) -> List[Dict[str, Any]]:
    """Un-tuned sequence/request/imbalance/tail cases (step 28)."""
    return [
        {"case": "uneven_microbatch", "detail": f"microbatches=1 with {stages} stages"},
        {"case": "uneven_sequence", "detail": f"sequence not divisible by {degree}"},
        {"case": "stage_imbalance", "detail": "one stage holds the attention/LM-head hotspot"},
        {"case": "tail_request", "detail": "a single short request cannot fill the pipeline"},
        {"case": "short_context", "detail": "context below the CP/SP benefit threshold"},
    ]


def tp_comparison_rows(
    *, kind: str, tp: Mapping[str, float], candidate: Mapping[str, float]
) -> List[Dict[str, Any]]:
    """Compare against the TP baseline on latency/throughput/memory/device-seconds (§27)."""
    rows: List[Dict[str, Any]] = []
    for metric in sorted(set(tp) | set(candidate)):
        left = tp.get(metric)
        right = candidate.get(metric)
        rows.append(
            {
                "metric": metric,
                "tp_baseline": left,
                "candidate": right,
                "kind": kind,
                "delta": (right - left) if (left is not None and right is not None) else None,
            }
        )
    return rows


def adopt_reject_verdict(
    *,
    kind: str,
    checklist: Mapping[str, bool],
    primary_improved: bool,
    capacity_problem_solved: bool,
    regressions: Sequence[str] = (),
    reasons: Sequence[str] = (),
    re_evaluation_triggers: Sequence[str] = (),
) -> Dict[str, Any]:
    """The §12 adoption standard; a negative decision keeps its evidence."""
    missing = [name for name in ADOPT_REJECT_CRITERIA if not checklist.get(name, False)]
    if missing:
        return {
            "decision": "REJECT",
            "kind": kind,
            "reason": f"adoption criteria not satisfied: {missing}",
            "reasons": list(reasons),
            "re_evaluate_when": list(re_evaluation_triggers or RE_EVALUATION_TRIGGERS),
        }
    if regressions:
        return {
            "decision": "REJECT",
            "kind": kind,
            "reason": f"unacceptable regressions: {list(regressions)}",
            "reasons": list(reasons),
            "re_evaluate_when": list(re_evaluation_triggers or RE_EVALUATION_TRIGGERS),
        }
    if not (primary_improved or capacity_problem_solved):
        return {
            "decision": "REJECT",
            "kind": kind,
            "reason": (
                "neither the primary metric improved nor a capacity problem was solved; "
                "'it runs' is not an adoption reason"
            ),
            "reasons": list(reasons),
            "re_evaluate_when": list(re_evaluation_triggers or RE_EVALUATION_TRIGGERS),
        }
    return {
        "decision": "ADOPT",
        "kind": kind,
        "reason": (
            "correctness/quality gates pass, the capacity problem is solved or the primary "
            "metric improved, no unacceptable regression"
        ),
        "reasons": list(reasons),
        "re_evaluate_when": list(re_evaluation_triggers or RE_EVALUATION_TRIGGERS),
    }


def health_probe_plan(kind: str) -> Dict[str, Any]:
    """The layered probe chain used after a plan change (step 26/27)."""
    if kind not in CANDIDATE_KINDS:
        raise ConfigError(f"unknown candidate kind {kind!r}", details={"field": "kind"})
    return {
        "kind": kind,
        "chain": list(PROBE_CHAIN),
        "unsupported_cases": list(UNSUPPORTED_CASES),
        "note": "locate the layer that is still broken instead of declaring a binary success",
    }


def boundary_verdict(
    *,
    activated: bool,
    primary: Optional[str],
    correctness_ok: bool,
    ledger_closed: bool,
    decision: str,
    confirmation_ok: bool,
) -> Dict[str, Any]:
    """The E10-07 stage verdict (only meaningful when the P1 is activated)."""
    if not activated:
        return {
            "status": "NOT_RUN_NOT_CLAIMED",
            "reason": "PP/CP/SP is not claimed; the experiment is not a gate",
            "allowed_claim": "none",
        }
    if not correctness_ok:
        return {"status": "FAIL", "reason": "partition/tensor/logits/token correctness failed"}
    if not ledger_closed:
        return {"status": "FAIL", "reason": "the communication ledger is not closed"}
    if not confirmation_ok:
        return {"status": "INCONCLUSIVE", "reason": "the independent confirmation did not reproduce"}
    if decision == "ADOPT":
        return {
            "status": "PASS",
            "reason": f"adopted {primary} with correctness, closed ledger and confirmation",
            "allowed_claim": f"{primary} for the tested model/workload/topology/runtime",
        }
    return {
        "status": "PASS_NEGATIVE",
        "reason": f"{primary} was rejected with evidence; the negative result is the deliverable",
        "allowed_claim": f"{primary} not adopted for the tested scope",
    }


__all__ = [
    "ADOPT_REJECT_CRITERIA",
    "CANDIDATE_KINDS",
    "CP_SP_ALGORITHMS",
    "PROBE_CHAIN",
    "RE_EVALUATION_TRIGGERS",
    "RUBRIC_FACTORS",
    "RUBRIC_SCORES",
    "RUBRIC_TABLE",
    "UNSUPPORTED_CASES",
    "ActivationDecision",
    "AnalyticalCosts",
    "CandidateDefinition",
    "ContextPlan",
    "PipelinePlan",
    "RubricScore",
    "StageBalanceRow",
    "activation_decision",
    "adopt_reject_verdict",
    "adversarial_cases",
    "analytical_costs",
    "boundary_verdict",
    "bubble_metrics",
    "capability_gate",
    "cp_sp_metrics",
    "health_probe_plan",
    "microbatch_sweep",
    "pipeline_bubble_ideal",
    "score_candidates",
    "select_primary",
    "sequence_sweep",
    "stage_balance",
    "stage_partition_objective",
    "tp_comparison_rows",
    "validate_candidates",
    "verify_partition",
]
