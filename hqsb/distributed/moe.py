"""MoE expert parallelism: routing, dispatch/combine oracles, skew, placement.

E10-08 is P0 but *graded*: the deliverable states its claim level (L0–L4) and
only L4 supports a full "MoE/EP implemented" claim.  This module supplies:

* :class:`RouteArtifact` with conservation auditing and a hash, so a route set
  is a frozen artifact rather than a per-configuration re-randomisation;
* :func:`dispatch_oracle` / :func:`combine_oracle` — the CPU permutation and
  inverse maps, including zero-token experts, tails and duplicated top-k;
* :func:`count_matrix` with checked arithmetic for AllToAllV offsets;
* :func:`imbalance_metrics` (max/mean, CV, Gini, entropy, hot persistence,
  communication cut, critical-rank load) — never just the mean;
* placement baseline/treatment with a holdout check and an explicit fallback
  when the router distribution drifts.

Nothing here claims model quality: that requires a real MoE artifact (L4).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Completion levels of details E10-08 §2.
CLAIM_LEVELS: Tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4")

#: What each level is allowed to claim.
CLAIM_LEVEL_TABLE: Mapping[str, str] = {
    "L0": "routing-distribution analysis only",
    "L1": "dispatch/combine semantics implemented (CPU oracle)",
    "L2": "MoE collective micro (real AllToAll(V) + skew/placement)",
    "L3": "EP runtime layer (router → expert → combine)",
    "L4": "full MoE/EP model quality and end-to-end performance",
}

#: Skew profiles that must be covered (details E10-08 §5).
SKEW_PROFILES: Tuple[str, ...] = (
    "uniform",
    "mild_zipf",
    "severe_zipf",
    "single_hot_expert",
    "hot_pair_same_rank",
    "hot_pair_different_rank",
    "per_source_correlated",
    "prefill_large_token",
    "decode_small_batch",
    "time_varying_hot_expert",
)

#: Overflow policies for expert capacity (details E10-08 §4.2).
OVERFLOW_POLICIES: Tuple[str, ...] = ("dynamic", "padding", "drop", "reroute")

#: The routing statistics every route artifact must report (details E10-08 §11).
ROUTING_STATS_FIELDS: Tuple[str, ...] = (
    "n_tokens",
    "n_assignments",
    "tokens_per_expert_before_capacity",
    "tokens_per_expert_after_capacity",
    "tokens_per_rank",
    "send_count_matrix",
    "dropped_assignments",
    "rerouted_assignments",
    "duplicated_assignments",
    "padding_slots",
    "gate_weight_sum_per_token",
)

#: Expert placement strategies.
PLACEMENT_KINDS: Tuple[str, ...] = ("round_robin", "topology_aware", "framework_default")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str, *, field_name: str = "") -> None:
    if not condition:
        raise ConfigError(message, details={"field": field_name} if field_name else None)


# ── claim level ────────────────────────────────────────────────────────────


def claim_gate(
    *,
    level: str,
    has_real_moe_artifact: bool,
    runtime_closed: bool,
    model_quality_rows: bool,
) -> Dict[str, Any]:
    """What a given completion level may and may not claim (details E10-08 step 1)."""
    if level not in CLAIM_LEVELS:
        raise ConfigError(f"level must be one of {CLAIM_LEVELS}", details={"field": "level"})
    max_allowed = level
    blockers: List[str] = []
    if level in ("L3", "L4") and not runtime_closed:
        max_allowed = "L2"
        blockers.append("the runtime loop (router → expert → combine) is not closed")
    if level == "L4":
        if not has_real_moe_artifact:
            max_allowed = "L3" if runtime_closed else "L2"
            blockers.append("no real MoE ModelArtifact: model quality cannot be claimed")
        if not model_quality_rows:
            max_allowed = "L3" if runtime_closed else "L2"
            blockers.append("no logits/top-k/KL/greedy token evidence")
    return {
        "requested_level": level,
        "allowed_level": max_allowed,
        "allowed_claim": CLAIM_LEVEL_TABLE[max_allowed],
        "blockers": blockers,
        "must_not_claim": (
            "complete MoE/EP model quality or end-to-end EP gains"
            if max_allowed != "L4"
            else ""
        ),
        "note": "dense-Qwen synthetic traces never impersonate a real MoE model",
    }


# ── operator spec / route artifact ─────────────────────────────────────────


@dataclass(frozen=True)
class MoeOperatorSpec:
    """Frozen MoE operator semantics (details E10-08 step 3)."""

    hidden_size: int
    num_experts: int
    top_k: int
    gate_weight_normalization: str = "softmax_over_topk"
    capacity_factor: float = 0.0
    overflow_policy: str = "dynamic"
    dtype: str = "fp16"
    output_order: str = "original_token_order"
    error_semantics: str = "reject_on_count_mismatch"
    expert_compute: str = "grouped_gemm"

    def __post_init__(self) -> None:
        _require(self.hidden_size > 0, "hidden_size must be positive", field_name="hidden_size")
        _require(self.num_experts >= 1, "num_experts must be >= 1", field_name="num_experts")
        _require(1 <= self.top_k <= self.num_experts, "top_k must be within the expert count", field_name="top_k")
        _require(
            self.overflow_policy in OVERFLOW_POLICIES,
            f"overflow_policy must be one of {OVERFLOW_POLICIES}",
            field_name="overflow_policy",
        )
        _require(
            self.gate_weight_normalization in ("softmax_over_topk", "renormalized_topk", "none"),
            "unknown gate weight normalisation",
            field_name="gate_weight_normalization",
        )

    def capacity_per_expert(self, tokens: int) -> Optional[int]:
        if not self.capacity_factor:
            return None
        import math

        return int(math.ceil(tokens * self.top_k / self.num_experts * self.capacity_factor))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "gate_weight_normalization": self.gate_weight_normalization,
            "capacity_factor": self.capacity_factor,
            "overflow_policy": self.overflow_policy,
            "dtype": self.dtype,
            "output_order": self.output_order,
            "error_semantics": self.error_semantics,
            "expert_compute": self.expert_compute,
        }


@dataclass(frozen=True)
class RouteAssignment:
    """One ``(token, expert, slot)`` routing decision."""

    token_index: int
    source_rank: int
    expert_id: int
    topk_slot: int
    weight: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "token_index": self.token_index,
            "source_rank": self.source_rank,
            "expert_id": self.expert_id,
            "topk_slot": self.topk_slot,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class RouteArtifact:
    """A frozen routing set with its distribution parameters and hash (step 4)."""

    artifact_id: str
    token_count: int
    top_k: int
    num_experts: int
    assignments: Tuple[RouteAssignment, ...]
    profile: str
    seed: int
    distribution_params: Mapping[str, Any] = field(default_factory=dict)
    model_ref: str = ""
    layer_ref: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        _require(self.profile in SKEW_PROFILES, "unknown skew profile", field_name="profile")
        expected = self.token_count * self.top_k
        _require(
            len(self.assignments) == expected,
            f"assignments {len(self.assignments)} != tokens×top_k = {expected}; dropped "
            "assignments must be counted explicitly, not silently missing",
            field_name="assignments",
        )
        for item in self.assignments:
            _require(
                0 <= item.expert_id < self.num_experts,
                f"expert id {item.expert_id} outside [0, {self.num_experts})",
                field_name="expert_id",
            )

    def send_count_matrix(self, *, expert_to_rank: Mapping[int, int], world_size: int) -> List[List[int]]:
        """``C[src][dst]``: assignments per source rank → destination rank."""
        matrix = [[0] * world_size for _ in range(world_size)]
        for item in self.assignments:
            destination = expert_to_rank[item.expert_id]
            matrix[item.source_rank][destination] += 1
        return matrix

    def tokens_per_expert(self) -> List[int]:
        counts = [0] * self.num_experts
        for item in self.assignments:
            counts[item.expert_id] += 1
        return counts

    def gate_weight_sum_per_token(self) -> List[float]:
        sums = [0.0] * self.token_count
        for item in self.assignments:
            sums[item.token_index] += item.weight
        return sums

    def stats(self, *, expert_to_rank: Mapping[int, int], world_size: int) -> Dict[str, Any]:
        matrix = self.send_count_matrix(expert_to_rank=expert_to_rank, world_size=world_size)
        per_rank = [sum(row) for row in matrix]
        return {
            "n_tokens": self.token_count,
            "n_assignments": len(self.assignments),
            "tokens_per_expert_before_capacity": self.tokens_per_expert(),
            "tokens_per_expert_after_capacity": self.tokens_per_expert(),
            "tokens_per_rank": per_rank,
            "send_count_matrix": matrix,
            "dropped_assignments": 0,
            "rerouted_assignments": 0,
            "duplicated_assignments": 0,
            "padding_slots": 0,
            "gate_weight_sum_per_token": self.gate_weight_sum_per_token(),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "artifact_id": self.artifact_id,
                "token_count": self.token_count,
                "top_k": self.top_k,
                "num_experts": self.num_experts,
                "profile": self.profile,
                "seed": self.seed,
                "distribution_params": dict(self.distribution_params),
                "assignments": [item.as_dict() for item in self.assignments],
            },
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )

    @property
    def sha256(self) -> str:
        return _sha256_text(self.canonical_json())


def generate_route_artifact(
    *,
    artifact_id: str,
    token_count: int,
    world_size: int,
    experts_per_rank: int,
    top_k: int,
    profile: str,
    seed: int,
    distribution_params: Optional[Mapping[str, Any]] = None,
) -> RouteArtifact:
    """Deterministic route generation for every skew profile (fixed seed)."""
    if profile not in SKEW_PROFILES:
        raise ConfigError(f"unknown profile {profile!r}", details={"field": "profile"})
    _require(token_count > 0, "token_count must be > 0", field_name="token_count")
    num_experts = world_size * experts_per_rank
    _require(1 <= top_k <= num_experts, "top_k must be within the expert count", field_name="top_k")
    rng = random.Random(seed)
    params = dict(distribution_params or {})
    # -1 means "no explicit hot expert": the generator then picks one from its
    # own shuffled order, so distinct seeds produce distinct hot experts.
    hot = int(params.get("hot_expert", -1))
    hot_pair = (int(params.get("hot_pair", (0, min(1, num_experts - 1)))[0]),
                int(params.get("hot_pair", (0, min(1, num_experts - 1)))[1]))
    zipf_alpha = float(params.get("zipf_alpha", 1.1))
    correlation = float(params.get("source_correlation", 0.6))
    assignments: List[RouteAssignment] = []
    for token in range(token_count):
        source = token % world_size
        weights = _expert_weight_vector(
            profile=profile,
            rng=rng,
            num_experts=num_experts,
            source=source,
            hot=hot,
            hot_pair=hot_pair,
            zipf_alpha=zipf_alpha,
            correlation=correlation,
            time_index=token,
            time_phase=int(params.get("time_phase", max(1, token_count // 4))),
        )
        chosen = _top_k_indices(weights, top_k)
        normaliser = sum(weights[index] for index in chosen) or 1.0
        for slot, expert in enumerate(chosen):
            assignments.append(
                RouteAssignment(
                    token_index=token,
                    source_rank=source,
                    expert_id=int(expert),
                    topk_slot=slot,
                    weight=float(weights[expert] / normaliser),
                )
            )
    return RouteArtifact(
        artifact_id=artifact_id,
        token_count=token_count,
        top_k=top_k,
        num_experts=num_experts,
        assignments=tuple(assignments),
        profile=profile,
        seed=seed,
        distribution_params=params,
    )


def _expert_weight_vector(
    *,
    profile: str,
    rng: random.Random,
    num_experts: int,
    source: int,
    hot: int,
    hot_pair: Tuple[int, int],
    zipf_alpha: float,
    correlation: float,
    time_index: int,
    time_phase: int,
) -> List[float]:
    if profile == "uniform":
        return [rng.random() + 1e-9 for _ in range(num_experts)]
    if profile in ("mild_zipf", "severe_zipf", "single_hot_expert"):
        # The Zipf weights are assigned over a *shuffled* expert order: tying
        # them to the expert index would make every zipf profile degenerate to
        # "everything goes to expert 0" (a silent skew-profile bug).
        order = list(range(num_experts))
        rng.shuffle(order)
        weights = [0.0] * num_experts
        if profile == "single_hot_expert":
            for expert in range(num_experts):
                weights[expert] = 0.01
            weights[(hot if hot >= 0 else order[0]) % num_experts] = 1.0
            return weights
        alpha = 1.05 if profile == "mild_zipf" else zipf_alpha + 1.2
        for position, expert in enumerate(order):
            weights[expert] = (1.0 / ((position + 1) ** alpha)) * rng.uniform(0.8, 1.2)
        return weights
    if profile in ("hot_pair_same_rank", "hot_pair_different_rank"):
        weights = [rng.random() * 0.05 for _ in range(num_experts)]
        weights[hot_pair[0]] = 1.0
        weights[hot_pair[1]] = 0.9
        return weights
    if profile == "per_source_correlated":
        weights = [rng.random() for _ in range(num_experts)]
        favoured = source % num_experts
        weights[favoured] += correlation * num_experts
        return weights
    if profile == "prefill_large_token":
        return [1.0 + rng.random() for _ in range(num_experts)]
    if profile == "decode_small_batch":
        weights = [rng.random() * 0.5 for _ in range(num_experts)]
        weights[(hot if hot >= 0 else 0) % num_experts] = 2.0
        return weights
    if profile == "time_varying_hot_expert":
        weights = [rng.random() * 0.2 for _ in range(num_experts)]
        epoch = time_index // max(1, time_phase)
        weights[((hot if hot >= 0 else 0) + epoch) % num_experts] = 1.5
        return weights
    raise ConfigError(f"profile {profile!r} has no generator", details={"field": "profile"})


def _top_k_indices(weights: Sequence[float], top_k: int) -> List[int]:
    indexed = sorted(range(len(weights)), key=lambda index: (-weights[index], index))
    return indexed[:top_k]


# ── dispatch / combine oracles ─────────────────────────────────────────────


@dataclass
class DispatchResult:
    """Packed activation plus the permutation/inverse maps (details E10-08 §4.1)."""

    token_order: Tuple[int, ...]  # packed position → original token index
    expert_order: Tuple[int, ...]  # packed position → expert id
    weights: Tuple[float, ...]
    source_order: Tuple[int, ...]
    slot_order: Tuple[int, ...]
    send_counts: Tuple[int, ...]
    offsets: Tuple[int, ...]
    dropped: Tuple[int, ...] = ()
    empty_experts: Tuple[int, ...] = ()

    @property
    def packed_count(self) -> int:
        return len(self.token_order)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "packed_count": self.packed_count,
            "token_order": list(self.token_order),
            "expert_order": list(self.expert_order),
            "weights": list(self.weights),
            "send_counts": list(self.send_counts),
            "offsets": list(self.offsets),
            "dropped": list(self.dropped),
            "empty_experts": list(self.empty_experts),
        }


def compute_offsets(counts: Sequence[int], *, alignment: int = 1) -> List[int]:
    """Checked exclusive prefix sum with optional alignment (never overflow)."""
    if alignment < 1:
        raise ConfigError("alignment must be >= 1", details={"field": "alignment"})
    offsets: List[int] = []
    cursor = 0
    for count in counts:
        if count < 0:
            raise ConfigError("negative count in an AllToAllV matrix", details={"field": "counts"})
        offsets.append(cursor)
        cursor += count
        if alignment > 1 and cursor % alignment:
            cursor += alignment - (cursor % alignment)
    return offsets


def dispatch_oracle(route: RouteArtifact, *, alignment: int = 1) -> DispatchResult:
    """Group assignments by expert, keeping token/slot/weight provenance (step 5)."""
    per_expert: Dict[int, List[RouteAssignment]] = {}
    for item in route.assignments:
        per_expert.setdefault(item.expert_id, []).append(item)
    token_order: List[int] = []
    expert_order: List[int] = []
    weights: List[float] = []
    source_order: List[int] = []
    slot_order: List[int] = []
    send_counts: List[int] = []
    for expert in range(route.num_experts):
        items = sorted(
            per_expert.get(expert, []),
            key=lambda item: (item.token_index, item.topk_slot),
        )
        send_counts.append(len(items))
        token_order.extend(item.token_index for item in items)
        expert_order.extend([expert] * len(items))
        weights.extend(item.weight for item in items)
        source_order.extend(item.source_rank for item in items)
        slot_order.extend(item.topk_slot for item in items)
    empty = tuple(
        expert for expert, count in enumerate(send_counts) if count == 0
    )
    return DispatchResult(
        token_order=tuple(token_order),
        expert_order=tuple(expert_order),
        weights=tuple(weights),
        source_order=tuple(source_order),
        slot_order=tuple(slot_order),
        send_counts=tuple(send_counts),
        offsets=tuple(compute_offsets(send_counts, alignment=alignment)),
        empty_experts=empty,
    )


def combine_oracle(
    dispatch: DispatchResult,
    *,
    expert_outputs: Mapping[int, Any],
    hidden_size: int,
) -> Dict[str, Any]:
    """Inverse map + gate weights restore the original token order (step 6).

    ``expert_outputs`` maps packed position → hidden vector (a stub or a real
    expert).  The return value is ``{token_index: [summed hidden]}``.
    """
    if hidden_size <= 0:
        raise ConfigError("hidden_size must be positive", details={"field": "hidden_size"})
    token_count = max(dispatch.token_order) + 1 if dispatch.token_order else 0
    accumulated: Dict[int, List[float]] = {index: [0.0] * hidden_size for index in range(token_count)}
    seen_positions = 0
    for position in range(dispatch.packed_count):
        token = dispatch.token_order[position]
        expert = dispatch.expert_order[position]
        vector = expert_outputs.get(position)
        if vector is None:
            raise ConfigError(
                f"missing expert output for packed position {position} (expert {expert})",
                details={"field": "expert_outputs"},
            )
        if len(vector) != hidden_size:
            raise ConfigError(
                f"expert output at position {position} has {len(vector)} values, expected "
                f"{hidden_size}",
                details={"field": "expert_outputs"},
            )
        weight = dispatch.weights[position]
        target = accumulated[token]
        for index in range(hidden_size):
            target[index] += weight * float(vector[index])
        seen_positions += 1
    return {
        "tokens": accumulated,
        "positions": seen_positions,
        "token_conservation": seen_positions == dispatch.packed_count,
    }


def deterministic_stub_expert(*, position: int, expert_id: int, hidden_size: int) -> List[float]:
    """A deterministic, invertible stand-in expert used to isolate permutation bugs."""
    base = (expert_id * 1_000_003 + position * 7_919) % 65_521
    return [float((base + index) % 1021) / 1021.0 for index in range(hidden_size)]


def expert_mlp_plan(
    *, operator: MoeOperatorSpec, tokens_per_expert: Sequence[int]
) -> Dict[str, Any]:
    """The L3/L4 grouped-GEMM plan: per-expert shapes, weights and required evidence."""
    if len(tokens_per_expert) == 0:
        raise ConfigError("tokens_per_expert must not be empty")
    shapes: List[Dict[str, Any]] = []
    for expert, tokens in enumerate(tokens_per_expert):
        shapes.append(
            {
                "expert_id": expert,
                "tokens": tokens,
                "empty": tokens == 0,
                "gemm_m": tokens,
                "gemm_k": operator.hidden_size,
                "compute": operator.expert_compute,
            }
        )
    return {
        "expert_compute": operator.expert_compute,
        "dtype": operator.dtype,
        "per_expert": shapes,
        "required_evidence": [
            "actual kernel/precision visible",
            "single-process MoE layer output compared with the oracle",
            "per-expert time and efficiency recorded separately from communication waits",
        ],
        "note": (
            "an empty expert still costs a launch; grouped GEMM efficiency at M=1..few tokens "
            "is a measured property, not an assumption"
        ),
    }


def HAND_CASES() -> Tuple[Dict[str, Any], ...]:
    """Small hand-checkable cases (2 ranks/experts, k=1/2, empty/duplicate) (step 7)."""
    return (
        {"case": "k1_two_experts", "tokens": 2, "top_k": 1, "expected": "identity permutation"},
        {"case": "k2_duplicate_expert", "tokens": 1, "top_k": 2, "expected": "two slots same expert"},
        {"case": "zero_token_expert", "tokens": 1, "top_k": 1, "expected": "empty expert counted"},
        {"case": "tail_token", "tokens": 3, "top_k": 2, "expected": "last token slots complete"},
        {"case": "severe_skew", "tokens": 8, "top_k": 1, "expected": "one expert holds all tokens"},
    )


@dataclass(frozen=True)
class CountMatrix:
    """AllToAllV send/recv counts with conservation auditing (details E10-08 §4.5)."""

    send_counts: Tuple[Tuple[int, ...], ...]
    world_size: int
    tokens: int
    top_k: int
    dropped: int = 0
    duplicated: int = 0
    padding_slots: int = 0

    def __post_init__(self) -> None:
        _require(
            len(self.send_counts) == self.world_size
            and all(len(row) == self.world_size for row in self.send_counts),
            "the count matrix must be world_size × world_size",
            field_name="send_counts",
        )
        for row in self.send_counts:
            for value in row:
                _require(value >= 0, "negative counts are not a matrix", field_name="send_counts")

    def send_tokens(self) -> List[int]:
        return [sum(row) for row in self.send_counts]

    def recv_tokens(self) -> List[int]:
        return [
            sum(self.send_counts[src][dst] for src in range(self.world_size))
            for dst in range(self.world_size)
        ]

    def audit(self) -> Dict[str, Any]:
        expected = self.tokens * self.top_k - self.dropped
        send_total = sum(self.send_tokens())
        recv_total = sum(self.recv_tokens())
        problems: List[str] = []
        if send_total != expected:
            problems.append(f"global send {send_total} != tokens×top_k - dropped {expected}")
        if recv_total != expected:
            problems.append(f"global recv {recv_total} != {expected}")
        if send_total != recv_total:
            problems.append(f"send {send_total} != recv {recv_total} (conservation violated)")
        return {
            "ok": not problems,
            "problems": problems,
            "send_tokens": self.send_tokens(),
            "recv_tokens": self.recv_tokens(),
            "expected_assignments": expected,
            "padding_slots": self.padding_slots,
            "note": "fixed AllToAll padding slots are counted as bytes and memory, not hidden",
        }

    def offsets(self, *, alignment: int = 1) -> List[List[int]]:
        return [compute_offsets(row, alignment=alignment) for row in self.send_counts]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "world_size": self.world_size,
            "tokens": self.tokens,
            "top_k": self.top_k,
            "send_counts": [list(row) for row in self.send_counts],
            "dropped": self.dropped,
            "duplicated": self.duplicated,
            "padding_slots": self.padding_slots,
            "audit": self.audit(),
        }


def count_matrix(route: RouteArtifact, *, expert_to_rank: Mapping[int, int], world_size: int) -> CountMatrix:
    matrix = route.send_count_matrix(expert_to_rank=expert_to_rank, world_size=world_size)
    return CountMatrix(
        send_counts=tuple(tuple(row) for row in matrix),
        world_size=world_size,
        tokens=route.token_count,
        top_k=route.top_k,
    )


@dataclass(frozen=True)
class DispatchSpec:
    """The AllToAll(V) dispatch specification (details E10-08 step 10)."""

    op: str
    algorithm: str
    layout: str
    stream: str
    collective_seq: int = 0
    counts_validated: bool = False

    def __post_init__(self) -> None:
        _require(
            self.op in ("all_to_all", "all_to_all_v"),
            "op must be all_to_all|all_to_all_v",
            field_name="op",
        )
        _require(bool(self.algorithm), "record the actual algorithm", field_name="algorithm")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "algorithm": self.algorithm,
            "layout": self.layout,
            "stream": self.stream,
            "collective_seq": self.collective_seq,
            "counts_validated": self.counts_validated,
        }


def pack_plan(
    *, send_counts: Sequence[int], hidden_size: int, dtype_bytes: int, alignment: int = 8
) -> Dict[str, Any]:
    """Device pack/unpack plan with guard regions and byte accounting (step 9)."""
    total = sum(send_counts)
    payload = total * hidden_size * dtype_bytes
    metadata = len(send_counts) * 8
    padding_slots = 0
    if alignment > 1:
        for count in send_counts:
            if count % alignment:
                padding_slots += alignment - (count % alignment)
    return {
        "packed_tokens": total,
        "payload_bytes": payload,
        "metadata_bytes": metadata,
        "padding_slots": padding_slots,
        "padding_bytes": padding_slots * hidden_size * dtype_bytes,
        "guard_required": True,
        "actual_format": f"row-major {dtype_bytes}-byte elements, alignment={alignment}",
    }


# ── skew / imbalance ───────────────────────────────────────────────────────


def gini(counts: Sequence[int]) -> float:
    values = sorted(float(value) for value in counts)
    n = len(values)
    if n == 0 or sum(values) == 0:
        return 0.0
    cumulative = 0.0
    for index, value in enumerate(values):
        cumulative += (2 * (index + 1) - n - 1) * value
    return cumulative / (n * sum(values))


def entropy(counts: Sequence[int]) -> float:
    import math

    total = sum(counts)
    if total == 0:
        return 0.0
    value = 0.0
    for count in counts:
        if count:
            probability = count / total
            value -= probability * math.log(probability)
    return value


def imbalance_metrics(
    *,
    tokens_per_expert: Sequence[int],
    tokens_per_rank: Sequence[int],
    expert_unit_costs: Optional[Sequence[float]] = None,
    previous_hot_experts: Sequence[int] = (),
    top_k: int = 1,
    hidden_size: int = 1,
    dtype_bytes: int = 2,
) -> Dict[str, Any]:
    """The full imbalance report of details E10-08 §11 (never just the mean)."""
    if not tokens_per_expert:
        raise ConfigError("tokens_per_expert must not be empty")
    mean_expert = sum(tokens_per_expert) / len(tokens_per_expert)
    max_expert = max(tokens_per_expert)
    mean_rank = sum(tokens_per_rank) / len(tokens_per_rank) if tokens_per_rank else 0.0
    max_rank = max(tokens_per_rank) if tokens_per_rank else 0
    costs = list(expert_unit_costs) if expert_unit_costs else [1.0] * len(tokens_per_expert)
    if len(costs) != len(tokens_per_expert):
        raise ConfigError("expert_unit_costs must match the expert count")
    critical_loads = [count * cost for count, cost in zip(tokens_per_expert, costs)]
    hot_now = sorted(
        range(len(tokens_per_expert)),
        key=lambda index: -tokens_per_expert[index],
    )[: max(1, min(2, len(tokens_per_expert)))]
    persistence = (
        len(set(hot_now) & set(previous_hot_experts)) / len(hot_now)
        if previous_hot_experts
        else 0.0
    )
    return {
        "max_per_mean_expert": max_expert / mean_expert if mean_expert else None,
        "max_per_mean_rank": max_rank / mean_rank if mean_rank else None,
        "cv": _cv(tokens_per_expert),
        "gini": gini(tokens_per_expert),
        "entropy": entropy(tokens_per_expert),
        "hot_experts": hot_now,
        "hot_expert_persistence": persistence,
        "communication_cut_bytes": sum(
            count for rank_index, count in enumerate(tokens_per_rank) if rank_index
        )
        * hidden_size
        * dtype_bytes,
        "critical_rank_load": max(critical_loads) if critical_loads else None,
        "empty_experts": sum(1 for count in tokens_per_expert if count == 0),
        "note": (
            "token counts alone are not load: critical-rank load multiplies by the measured "
            "per-expert unit cost"
        ),
    }


def _cv(values: Sequence[float]) -> Optional[float]:
    import math

    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    if mean == 0:
        return None
    variance = sum((value - mean) ** 2 for value in values) / n
    return math.sqrt(variance) / mean


def padding_waste(
    *, route: RouteArtifact, expert_to_rank: Mapping[int, int], world_size: int, top_k: int
) -> Dict[str, Any]:
    """Fixed AllToAll padding cost vs AllToAllV for the same route (step 20)."""
    matrix = route.send_count_matrix(expert_to_rank=expert_to_rank, world_size=world_size)
    max_row = max(sum(row) for row in matrix) if matrix else 0
    fixed_slots = max_row * world_size
    actual_slots = sum(sum(row) for row in matrix)
    return {
        "fixed_slots": fixed_slots,
        "actual_slots": actual_slots,
        "padding_slots": fixed_slots - actual_slots,
        "padding_fraction": (fixed_slots - actual_slots) / fixed_slots if fixed_slots else 0.0,
        "top_k": top_k,
        "note": "padding is charged as bytes and memory; it is never free",
    }


def capacity_policy_rows(
    *,
    operator: MoeOperatorSpec,
    tokens: int,
    assignments: Sequence[int],
) -> List[Dict[str, Any]]:
    """Capacity-factor/overflow comparison; a drop changes quality semantics (step 21)."""
    capacity = operator.capacity_per_expert(tokens)
    rows: List[Dict[str, Any]] = []
    for policy in OVERFLOW_POLICIES:
        if policy == "dynamic":
            rows.append(
                {
                    "policy": policy,
                    "capacity_per_expert": None,
                    "dropped": 0,
                    "padding_slots": 0,
                    "quality_semantics_changed": False,
                }
            )
            continue
        dropped = 0
        padded = 0
        if capacity is not None:
            for count in assignments:
                if count > capacity:
                    dropped += count - capacity
                    padded += capacity - count if policy == "padding" else 0
        rows.append(
            {
                "policy": policy,
                "capacity_per_expert": capacity,
                "dropped": dropped,
                "padding_slots": max(0, padded),
                "quality_semantics_changed": policy in ("drop", "reroute"),
                "note": (
                    "a drop/reroute changes the per-token gate weight sum, so quality is no "
                    "longer comparable to the no-drop baseline"
                ),
            }
        )
    return rows


# ── placement ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExpertPlacement:
    """An expert→rank mapping with its strategy and memory note."""

    name: str
    kind: str
    expert_to_rank: Mapping[int, int]
    memory_bytes_per_rank: Mapping[int, int] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        _require(self.kind in PLACEMENT_KINDS, "unknown placement kind", field_name="kind")
        if not self.expert_to_rank:
            raise ConfigError("a placement needs at least one expert", details={"field": "expert_to_rank"})

    def rank_of(self, expert: int) -> int:
        if expert not in self.expert_to_rank:
            raise ConfigError(f"expert {expert} has no placement", details={"field": "expert"})
        return self.expert_to_rank[expert]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "expert_to_rank": {str(k): v for k, v in sorted(self.expert_to_rank.items())},
            "memory_bytes_per_rank": {str(k): v for k, v in sorted(self.memory_bytes_per_rank.items())},
            "notes": self.notes,
        }


def round_robin_placement(*, num_experts: int, world_size: int) -> ExpertPlacement:
    """The baseline placement (step 22)."""
    return ExpertPlacement(
        name="baseline_round_robin",
        kind="round_robin",
        expert_to_rank={expert: expert % world_size for expert in range(num_experts)},
        notes="baseline: experts cycle across ranks without load knowledge",
    )


def topology_aware_placement(
    *,
    route: RouteArtifact,
    world_size: int,
    num_experts: int,
    fast_edges: Sequence[Tuple[int, int]] = (),
    max_memory_bytes_per_rank: int = 0,
    expert_bytes: int = 0,
) -> ExpertPlacement:
    """Place hot experts by measured traffic/compute, respecting memory limits (step 23).

    The heuristic here is deliberately simple and *auditable*: heavy experts are
    spread so that the largest per-rank token load is minimised, then pinned to
    the ranks that share fast edges with the sources that send to them.
    """
    counts = route.tokens_per_expert()
    order = sorted(range(num_experts), key=lambda expert: -counts[expert])
    loads = [0] * world_size
    placement: Dict[int, int] = {}
    for expert in order:
        candidates = list(range(world_size))
        if fast_edges:
            favoured = [dst for _src, dst in fast_edges]
            candidates = sorted(
                set(candidates),
                key=lambda rank: (0 if rank in favoured else 1, loads[rank]),
            )
        else:
            candidates = sorted(candidates, key=lambda rank: loads[rank])
        target = candidates[0]
        placement[expert] = target
        loads[target] += counts[expert]
    memory = {rank: 0 for rank in range(world_size)}
    for expert, rank in placement.items():
        memory[rank] += expert_bytes
    if max_memory_bytes_per_rank and any(
        value > max_memory_bytes_per_rank for value in memory.values()
    ):
        raise ConfigError(
            "the topology-aware placement exceeds the per-rank memory limit; replication/"
            "migration would add weight transfer, so it is not an allowed mitigation here",
            details={"field": "max_memory_bytes_per_rank"},
        )
    return ExpertPlacement(
        name="treatment_topology_aware",
        kind="topology_aware",
        expert_to_rank=placement,
        memory_bytes_per_rank=memory,
        notes=(
            "hot experts spread by measured token load and pinned near their sources; only the "
            "expert→rank mapping changes"
        ),
    )


def placement_ab_rows(
    *,
    route: RouteArtifact,
    baseline: ExpertPlacement,
    treatment: ExpertPlacement,
    world_size: int,
) -> Dict[str, Any]:
    """Baseline vs treatment per-edge traffic and max-rank load (step 24)."""
    rows: List[Dict[str, Any]] = []
    for name, placement in (("baseline", baseline), ("treatment", treatment)):
        matrix = route.send_count_matrix(
            expert_to_rank=placement.expert_to_rank, world_size=world_size
        )
        per_rank = [sum(row) for row in matrix]
        rows.append(
            {
                "arm": name,
                "placement": placement.name,
                "max_rank_tokens": max(per_rank) if per_rank else 0,
                "mean_rank_tokens": (sum(per_rank) / len(per_rank)) if per_rank else 0.0,
                "cross_rank_assignments": sum(
                    matrix[src][dst]
                    for src in range(world_size)
                    for dst in range(world_size)
                    if src != dst
                ),
                "send_count_matrix": matrix,
            }
        )
    return {
        "rows": rows,
        "note": (
            "a communication-cut improvement with a worse max-rank compute is not an end-to-end "
            "win; judge on the critical path"
        ),
    }


def holdout_check(
    *,
    tuning_route: RouteArtifact,
    holdout_route: RouteArtifact,
    policy: ExpertPlacement,
    divergence_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Does the placement overfit the tuning distribution? (step 25)"""
    tuning_hist = _normalised_histogram(tuning_route.tokens_per_expert())
    holdout_hist = _normalised_histogram(holdout_route.tokens_per_expert())
    if len(tuning_hist) != len(holdout_hist):
        raise ConfigError("the tuning and holdout routes must have the same expert count")
    divergence = 0.5 * sum(
        abs(left - right) for left, right in zip(tuning_hist, holdout_hist)
    )
    return {
        "policy": policy.name,
        "tuning_tokens_per_expert": list(tuning_route.tokens_per_expert()),
        "holdout_tokens_per_expert": list(holdout_route.tokens_per_expert()),
        "divergence": divergence,
        "threshold": divergence_threshold,
        "overfits": divergence > divergence_threshold,
        "fallback_recommended": divergence > divergence_threshold,
        "fallback_policy": (
            "fall back to the robust round-robin baseline and alert on drift"
            if divergence > divergence_threshold
            else ""
        ),
        "note": "placement trained and tested on the same route is not evidence",
    }


def _normalised_histogram(counts: Sequence[int]) -> List[float]:
    total = sum(counts) or 1
    return [count / total for count in counts]


# ── phase cases / verdict ──────────────────────────────────────────────────


def phase_cases(*, prefill_tokens: int, decode_tokens: int, hidden_size: int) -> List[Dict[str, Any]]:
    """Prefill (large token) and decode (small batch) are analysed separately (step 27)."""
    return [
        {
            "phase": "prefill",
            "tokens": prefill_tokens,
            "payload_bytes": prefill_tokens * hidden_size * 2,
            "expectation": "bandwidth regime; padding waste is amortised",
        },
        {
            "phase": "decode",
            "tokens": decode_tokens,
            "payload_bytes": decode_tokens * hidden_size * 2,
            "expectation": "latency regime; AllToAll α and hot-expert spread dominate",
        },
    ]


def l4_gate(*, level: str, has_real_moe_artifact: bool, model_rows: bool) -> Dict[str, Any]:
    """The real-model gate; without it no model quality/perf claim exists (step 28)."""
    if level != "L4":
        return {
            "ok": False,
            "reason": f"level {level} does not reach the real-model gate",
            "allowed_claim": CLAIM_LEVEL_TABLE[level],
        }
    blocking: List[str] = []
    if not has_real_moe_artifact:
        blocking.append("no real MoE ModelArtifact")
    if not model_rows:
        blocking.append("no selected layer/logits/top-k/KL/greedy token evidence")
    return {
        "ok": not blocking,
        "blocking": blocking,
        "reason": "" if not blocking else "the L4 claim is not supported by evidence",
        "allowed_claim": CLAIM_LEVEL_TABLE["L4"] if not blocking else CLAIM_LEVEL_TABLE["L3"],
    }


def moe_verdict(
    *,
    highest_level: str,
    conservation_ok: bool,
    l2_or_l3_closed: bool,
    skew_covered: bool,
    placement_conclusion: Optional[str],
    holdout_ok: bool,
) -> Dict[str, Any]:
    """The graded E10-08 verdict: level + skew + placement + gaps (step 30)."""
    if not conservation_ok:
        return {"status": "FAIL", "reason": "token conservation or count matrix is violated"}
    if not l2_or_l3_closed:
        return {
            "status": "FAIL",
            "reason": "neither the L2 micro loop nor the L3 runtime loop is closed; the P0 gate needs one",
        }
    if not skew_covered:
        return {"status": "FAIL", "reason": "the skew matrix does not cover uniform → severe skew"}
    if not holdout_ok:
        return {
            "status": "INCONCLUSIVE",
            "reason": "the placement overfits the tuning route; the holdout regressed",
        }
    if placement_conclusion is None:
        return {
            "status": "INCONCLUSIVE",
            "reason": "no placement/skew mitigation conclusion (positive or negative) is recorded",
        }
    return {
        "status": "PASS",
        "reason": f"highest level {highest_level}: conservation, skew coverage and holdout are complete",
        "allowed_claim": CLAIM_LEVEL_TABLE.get(highest_level, ""),
        "remaining_gap": (
            "full MoE/EP model quality requires L4"
            if highest_level != "L4"
            else ""
        ),
    }


__all__ = [
    "CLAIM_LEVELS",
    "CLAIM_LEVEL_TABLE",
    "CountMatrix",
    "DispatchResult",
    "DispatchSpec",
    "ExpertPlacement",
    "HAND_CASES",
    "MoeOperatorSpec",
    "OVERFLOW_POLICIES",
    "PLACEMENT_KINDS",
    "ROUTING_STATS_FIELDS",
    "RouteArtifact",
    "RouteAssignment",
    "SKEW_PROFILES",
    "capacity_policy_rows",
    "claim_gate",
    "combine_oracle",
    "compute_offsets",
    "count_matrix",
    "deterministic_stub_expert",
    "dispatch_oracle",
    "entropy",
    "generate_route_artifact",
    "gini",
    "holdout_check",
    "imbalance_metrics",
    "l4_gate",
    "moe_verdict",
    "pack_plan",
    "padding_waste",
    "phase_cases",
    "placement_ab_rows",
    "round_robin_placement",
    "topology_aware_placement",
]
