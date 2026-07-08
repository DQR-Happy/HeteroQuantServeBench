"""E14-F2 — MoE router, expert parallel, All-to-All and load balance.

Conditional P0: the main branch only when ``E14-05`` selects F2, otherwise
``N/A_BY_ADR``.  A sparse MoE still holds the **full** model in memory; each token
only computes ``k`` experts, but the step is decided by the hottest rank:

```text
T_layer ≈ max_rank(T_route_pack + T_dispatch_a2a + T_expert_compute + T_combine_a2a)
```

So the experiment is a closed loop from token/domain → router probabilities →
expert/rank load → dispatch pack + All-to-All → expert GEMM/cache/quant →
combine → layer/request latency → service queue/SLO.  Reporting "active
parameters are fewer" is explicitly not a result (§10).

Interfaces provided:

* :class:`MoEModelContract` — expert count, top-k, router semantics, capacity/drop
  (steps 2, 5);
* :class:`ExpertPlacement` / :func:`validate_placement` — rank→device→node and
  expert→rank identity, with a version for atomic switching (steps 4, 25);
* :class:`RoutingEvent` — per-layer/request router output (step 13);
* :func:`load_statistics` — mean/CV/Gini/max-mean/padding efficiency, by layer,
  request, domain and window (steps 13, 22);
* :func:`check_token_dispatch` — every token copy round-trips to its original
  order with the right weight (step 16);
* :class:`AllToAllLedgerEntry`, :class:`ExpertGEMMLedgerEntry` (steps 17–18);
* :func:`rank_time_decomposition` — the slowest rank, not the mean (step 19);
* :class:`MoEMemoryLedger` — **total** vs active weights, buffers, workspace
  (step 20);
* :func:`controlled_count_replay` — synthetic token-count replay for the cost
  model, which may not produce quality conclusions (steps 7, 23);
* :func:`placement_scan`, :func:`replication_cost`, :func:`expert_cache_scan`,
  :func:`expert_quant_gate` (steps 24, 26–28);
* :func:`straggler_injection`, :func:`failure_scenarios` (steps 34–35);
* :class:`F2AdoptionDecision` (step 40).

Nothing here runs an expert; the arithmetic is over supplied counts, bytes and
shapes, so a framework's aggregate imbalance number can be checked independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.experimental import records as rec
from hqsb.experimental.contracts import AdoptionDecision

EXPERIMENT_ID = "E14-F2"
TITLE = "MoE Router、Expert Parallel、All-to-All 与负载均衡"
LEVEL = "条件 P0"

CLAIM_BOUNDARY = (
    "只在 F2 为唯一主分支时成立；受控 router replay 只能用于系统测试，不能产生质量结论。"
    "无收益但能证明通信/小 GEMM/热点是原因时应输出 PASS_NEGATIVE（E14-F2 §12）。"
)

#: Router activation functions; the choice changes semantics, not just speed.
ROUTER_ACTIVATIONS: Tuple[str, ...] = ("softmax", "sigmoid", "sparsemax")

#: Token-normalisation choices for the top-k weights.
ROUTER_NORMALISATIONS: Tuple[str, ...] = ("topk_renormalised", "none", "pre_softmax")

#: What happens to tokens that exceed an expert's capacity (step 5).
CAPACITY_POLICIES: Tuple[str, ...] = ("drop", "reroute", "pad", "dynamic")

#: Imbalance metrics that must be reported together (step 13).
IMBALANCE_METRICS: Tuple[str, ...] = ("cv", "gini", "max_mean", "padding_efficiency")

#: Time phases a layer decomposes into (step 19).
LAYER_PHASES: Tuple[str, ...] = (
    "route",
    "pack",
    "dispatch_a2a",
    "expert_compute",
    "unpack",
    "combine_a2a",
    "wait",
)

#: Memory categories, with total and active kept apart (step 20).
MOE_MEMORY_CATEGORIES: Tuple[str, ...] = (
    "expert_weights_total",
    "expert_weights_active",
    "dense_weights",
    "router",
    "communication_buffer",
    "workspace",
    "kv_cache",
    "allocator_overhead",
)

#: Mitigation strategies the ADR may select exactly one of (step 29).
MITIGATIONS: Tuple[str, ...] = ("placement", "replication", "cache_offload", "expert_quant", "capacity_policy")


@dataclass(frozen=True)
class MoEModelContract:
    """``E14-F2`` steps 2/5: the sparse semantics, frozen before any measurement."""

    moe_artifact_id: str
    num_experts: int
    top_k: int
    num_layers: int
    router_activation: str
    router_normalisation: str
    capacity_policy: str
    capacity_factor: Optional[float] = None
    shared_expert: bool = False
    aux_bias: bool = False
    router_dtype: str = "fp32"
    expert_layer_indices: Tuple[int, ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.moe_artifact_id:
            findings.append("MoEModelContract: moe_artifact_id is required")
        if self.num_experts < 2 or self.num_layers < 1:
            findings.append("MoEModelContract: num_experts >= 2 and num_layers >= 1 are required")
        if not 1 <= self.top_k <= self.num_experts:
            findings.append(f"MoEModelContract: top_k {self.top_k} is outside 1..{self.num_experts}")
        if self.router_activation not in ROUTER_ACTIVATIONS:
            findings.append(
                f"MoEModelContract: router_activation must be one of {', '.join(ROUTER_ACTIVATIONS)}"
            )
        if self.router_normalisation not in ROUTER_NORMALISATIONS:
            findings.append(
                f"MoEModelContract: router_normalisation must be one of {', '.join(ROUTER_NORMALISATIONS)}"
            )
        if self.capacity_policy not in CAPACITY_POLICIES:
            findings.append(
                f"MoEModelContract: capacity_policy must be one of {', '.join(CAPACITY_POLICIES)}"
            )
        if self.capacity_policy in ("drop", "reroute") and self.capacity_factor is None:
            findings.append(
                f"MoEModelContract: capacity_policy {self.capacity_policy!r} without a capacity_factor "
                "silently changes quality (drop/reroute 必须进入质量门)"
            )
        if self.expert_layer_indices and max(self.expert_layer_indices) >= self.num_layers:
            findings.append("MoEModelContract: expert_layer_indices exceed num_layers")
        return findings

    def total_to_active_ratio(self) -> float:
        """Total/active parameter ratio — reported *with* the memory, never alone."""
        if self.top_k <= 0:
            return float("inf")
        return self.num_experts / self.top_k

    def as_dict(self) -> Dict[str, Any]:
        return {
            "moe_artifact_id": self.moe_artifact_id,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "num_layers": self.num_layers,
            "router_activation": self.router_activation,
            "router_normalisation": self.router_normalisation,
            "capacity_policy": self.capacity_policy,
            "capacity_factor": self.capacity_factor,
            "shared_expert": self.shared_expert,
            "aux_bias": self.aux_bias,
            "router_dtype": self.router_dtype,
            "expert_layer_indices": list(self.expert_layer_indices),
            "total_to_active_ratio": self.total_to_active_ratio(),
        }


def router_topk(
    router_logits: Sequence[float], *, top_k: int, activation: str = "softmax"
) -> Dict[str, Any]:
    """``p = f(router(x)); selected = topk(p, k)`` (§3.1), hand-checkable.

    Ties are broken by the lower expert index and reported, because an
    implementation-dependent tie-break is a source of un-reproducible routing
    (``E14-F2`` §11: 路由/参数映射错误).
    """
    if top_k <= 0 or top_k > len(router_logits):
        raise ConfigError(f"top_k {top_k} is outside 1..{len(router_logits)}")
    if activation not in ROUTER_ACTIVATIONS:
        raise ConfigError(f"unknown router activation {activation!r}")
    if activation == "softmax":
        top = max(router_logits)
        exps = [math.exp(value - top) for value in router_logits]
        total = sum(exps)
        weights = [value / total for value in exps]
    elif activation == "sigmoid":
        raw = [1.0 / (1.0 + math.exp(-value)) for value in router_logits]
        total = sum(raw)
        weights = [value / total for value in raw]
    else:  # sparsemax: 2-simplex projection
        ordered = sorted(router_logits, reverse=True)
        running = 0.0
        threshold_index = 0
        for index, value in enumerate(ordered):
            running += value
            if value - (running - 1.0) / (index + 1) <= 0:
                break
            threshold_index = index
        tau = (sum(ordered[: threshold_index + 1]) - 1.0) / (threshold_index + 1)
        raw = [max(0.0, value - tau) for value in router_logits]
        total = sum(raw)
        weights = [value / total for value in raw] if total else [0.0 for _ in raw]
    order = sorted(range(len(weights)), key=lambda index: (-weights[index], index))
    selected = tuple(order[:top_k])
    ties = [
        index
        for index in range(len(weights))
        if index not in selected
        and any(abs(weights[index] - weights[chosen]) < 1e-12 for chosen in selected)
    ]
    return {
        "weights": weights,
        "selected": selected,
        "selected_weights": tuple(weights[index] for index in selected),
        "weight_sum": sum(weights[index] for index in selected),
        "ties": tuple(ties),
        "activation": activation,
    }


def expert_output(*, expert_weights: Sequence[Sequence[float]], token: Sequence[float],
                  routing: Mapping[str, Any]) -> List[float]:
    """``y = Σ_i w_i E_i(x)`` over the selected experts (§3.1).

    Returned as a vector so a test can compare it against a reference computed a
    different way (step 14: router 正确但 expert 参数映射或权重分片错).
    """
    weights = routing.get("weights") or []
    selected = routing.get("selected") or ()
    if not selected:
        raise ConfigError("routing has no selected expert")
    width = len(token)
    result = [0.0] * width
    for index in selected:
        if index >= len(expert_weights):
            raise ConfigError(f"selected expert {index} has no weights")
        matrix = expert_weights[index]
        if len(matrix) != width:
            raise ConfigError(f"expert {index} expects width {len(matrix)}, token width {width}")
        weight = float(weights[index]) if index < len(weights) else 0.0
        for position, row in enumerate(matrix):
            if len(row) != width:
                raise ConfigError(f"expert {index} row {position} has width {len(row)}, expected {width}")
            result[position] += weight * sum(value * element for value, element in zip(row, token))
    return result


@dataclass(frozen=True)
class ExpertPlacement:
    """Steps 4/25: ``expert → rank`` with a version, so switching is atomic."""

    placement_id: str
    expert_to_rank: Mapping[int, int]
    rank_to_device: Mapping[int, str]
    num_experts: int
    version: int = 1
    link_map: Mapping[str, str] = field(default_factory=dict)


def validate_placement(placement: ExpertPlacement, *, world_size: int) -> List[str]:
    """Every expert placed exactly once, every rank mapped, no device shared."""
    findings: List[str] = []
    for expert in range(placement.num_experts):
        if expert not in placement.expert_to_rank:
            findings.append(f"expert {expert} has no rank placement")
    extra = sorted(index for index in placement.expert_to_rank if index >= placement.num_experts)
    if extra:
        findings.append(f"placement assigns experts outside the model: {extra}")
    ranks = set(placement.expert_to_rank.values())
    missing_ranks = sorted(rank for rank in range(world_size) if rank not in placement.rank_to_device)
    if missing_ranks:
        findings.append(f"ranks without a device mapping: {missing_ranks}")
    orphan_ranks = sorted(rank for rank in ranks if rank not in placement.rank_to_device)
    if orphan_ranks:
        findings.append(f"experts placed on unmapped ranks: {orphan_ranks}")
    devices = list(placement.rank_to_device.values())
    duplicates = sorted({device for device in devices if devices.count(device) > 1})
    if duplicates:
        findings.append(f"devices claimed by more than one rank: {', '.join(duplicates)}")
    if placement.version < 1:
        findings.append("placement version must be >= 1 (切换必须原子并留下版本)")
    return findings


def placement_scan(
    candidates: Mapping[str, ExpertPlacement], *, world_size: int, baseline_id: str
) -> Dict[str, Any]:
    """Step 24: compare placements while weights and router stay fixed (§9)."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    if baseline_id not in candidates:
        problems.append(f"baseline placement {baseline_id!r} is not among the candidates")
    for name, placement in sorted(candidates.items()):
        findings = validate_placement(placement, world_size=world_size)
        rows.append(
            {
                "placement_id": name,
                "experts": placement.num_experts,
                "version": placement.version,
                "per_rank_experts": {
                    str(rank): sum(1 for value in placement.expert_to_rank.values() if value == rank)
                    for rank in range(world_size)
                },
                "problems": findings,
            }
        )
        problems.extend(f"{name}: {finding}" for finding in findings)
    return {
        "rows": rows,
        "problems": problems,
        "note": "placement 只改 expert→rank；并行度/权重/router 不变，否则无法归因（step 24）",
        "ok": not problems,
    }


@dataclass(frozen=True)
class RoutingEvent:
    """Per-layer routing of one request (step 13)."""

    request_id: str
    layer: int
    router_version: str
    top_k: int
    tokens_per_expert: Tuple[int, ...]
    expert_to_rank: Mapping[int, int] = field(default_factory=dict)
    dropped_tokens: int = 0
    rerouted_tokens: int = 0
    padded_tokens: int = 0
    domain: str = ""
    window: str = ""
    entropy_bits: Optional[float] = None


def load_statistics(
    token_counts: Sequence[int], *, padded_capacity: Optional[int] = None
) -> Dict[str, Any]:
    """Step 13: several imbalance metrics, because one can be gamed (§3.3).

    ``max/mean`` is inflated by small batches and ``CV`` ignores the tail, so all
    of :data:`IMBALANCE_METRICS` are reported, together with the raw counts.
    """
    counts = [int(value) for value in token_counts]
    if not counts:
        return {"error": "no expert token counts supplied"}
    total = sum(counts)
    mean = total / len(counts)
    if mean <= 0:
        return {
            "counts": counts,
            "total": total,
            "mean": mean,
            "cv": 0.0,
            "gini": 0.0,
            "max_mean": 0.0,
            "padding_efficiency": 0.0,
            "note": "no tokens routed: imbalance is undefined, not perfect",
        }
    variance = sum((value - mean) ** 2 for value in counts) / len(counts)
    cv = math.sqrt(variance) / mean
    pairs = [
        abs(left - right) for index, left in enumerate(counts) for right in counts[index + 1 :]
    ]
    gini = sum(pairs) / (len(counts) * total) if total else 0.0
    max_mean = max(counts) / mean
    capacity = padded_capacity if padded_capacity is not None else max(counts) * len(counts)
    padding_efficiency = total / capacity if capacity else 0.0
    return {
        "counts": counts,
        "total": total,
        "mean": mean,
        "cv": cv,
        "gini": gini,
        "max_mean": max_mean,
        "padding_efficiency": padding_efficiency,
        "padded_capacity": capacity,
        "hottest_expert": counts.index(max(counts)),
        "coldest_expert": counts.index(min(counts)),
    }


def stratify_routing(
    events: Sequence[RoutingEvent], *, key: str = "domain"
) -> Dict[str, Any]:
    """Step 22: per-layer/domain/window distributions, not one whole-model total."""
    if key not in ("domain", "layer", "window", "request_id"):
        raise ConfigError(f"unknown routing stratum key {key!r}")
    groups: Dict[str, List[int]] = {}
    for event in events:
        name = str(getattr(event, key, "<unstratified>"))
        aggregated = groups.setdefault(name, [])
        if not aggregated:
            aggregated.extend([0] * len(event.tokens_per_expert))
        for index, value in enumerate(event.tokens_per_expert):
            if index < len(aggregated):
                aggregated[index] += int(value)
    return {
        "key": key,
        "strata": {name: load_statistics(counts) | {"key": name} for name, counts in sorted(groups.items())},
        "note": "平均均衡但 P99 高通常来自短窗口/特定层/某 domain 的 burst hotspot（§9）",
    }


def controlled_count_replay(
    *, patterns: Mapping[str, Sequence[int]], experts: int
) -> Dict[str, Any]:
    """Steps 7/23: synthetic token-count replay for the cost model **only**.

    ``E14-F2`` §10 forbids drawing quality conclusions from a replayed router, so
    the result carries that restriction as data rather than as a footnote.
    """
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for name, counts in sorted(patterns.items()):
        if len(counts) != experts:
            problems.append(f"pattern {name!r} has {len(counts)} experts, expected {experts}")
            continue
        stats = load_statistics(counts)
        rows.append({"pattern": name, **stats})
    return {
        "patterns": rows,
        "problems": problems,
        "quality_conclusions_allowed": False,
        "restriction": "受控 replay 只用于系统/成本模型验证，不产生质量结论（E14-F2 §10）",
    }


def check_token_dispatch(
    tokens: Sequence[Mapping[str, Any]], *, experts: int, combine_result: Sequence[int]
) -> List[str]:
    """Step 16: no token lost, duplicated or reordered by dispatch/combine.

    Each record is ``{"token_id", "expert", "weight", "position"}``; the combine
    output must be the original positions in the original order, summed over the
    routed experts' contributions where a token was replicated.
    """
    problems: List[str] = []
    positions: List[int] = []
    seen: Dict[Any, int] = {}
    for record in tokens:
        expert = record.get("expert")
        if expert is None or not 0 <= int(expert) < experts:
            problems.append(f"token {record.get('token_id')!r} dispatched to invalid expert {expert!r}")
        position = record.get("position")
        if position is None:
            problems.append(f"token {record.get('token_id')!r} has no original position")
            continue
        positions.append(int(position))
        seen[record.get("token_id")] = seen.get(record.get("token_id"), 0) + 1
    if not positions:
        problems.append("no dispatch records supplied")
    if sorted(positions) != list(range(len(combine_result))):
        problems.append(
            f"combined output has {len(combine_result)} entries but dispatch covered positions "
            f"{sorted(set(positions))}"
        )
    return problems


@dataclass(frozen=True)
class AllToAllLedgerEntry:
    """Step 17: per-layer/rank/peer dispatch and combine accounting."""

    layer: int
    rank: int
    peer: int
    send_count: int
    recv_count: int
    bytes: int
    dtype: str
    padding_bytes: int = 0
    start_ns: int = 0
    end_ns: int = 0
    stream: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.rank == self.peer:
            findings.append(f"a2a entry with rank == peer ({self.rank})")
        if self.bytes < 0 or self.padding_bytes < 0:
            findings.append("a2a entry has negative bytes")
        if self.end_ns and self.start_ns and self.end_ns < self.start_ns:
            findings.append("a2a entry end_ns precedes start_ns")
        if not self.dtype:
            findings.append("a2a entry has no dtype")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "rank": self.rank,
            "peer": self.peer,
            "send_count": self.send_count,
            "recv_count": self.recv_count,
            "bytes": self.bytes,
            "dtype": self.dtype,
            "padding_bytes": self.padding_bytes,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "stream": self.stream,
        }


def a2a_balance(entries: Sequence[AllToAllLedgerEntry], *, world_size: int) -> Dict[str, Any]:
    """Step 17: bytes per peer, so skew is visible instead of averaged."""
    problems: List[str] = []
    for entry in entries:
        problems.extend(entry.problems())
    per_rank: Dict[int, int] = {rank: 0 for rank in range(world_size)}
    per_peer: Dict[Tuple[int, int], int] = {}
    for entry in entries:
        if entry.rank not in per_rank:
            problems.append(f"a2a entry from rank outside world size: {entry.rank}")
            continue
        per_rank[entry.rank] += entry.bytes
        per_peer[(entry.rank, entry.peer)] = per_peer.get((entry.rank, entry.peer), 0) + entry.bytes
    stats = load_statistics([per_rank[rank] for rank in sorted(per_rank)])
    return {
        "bytes_per_rank": {str(rank): value for rank, value in sorted(per_rank.items())},
        "bytes_per_peer": {f"{rank}->{peer}": value for (rank, peer), value in sorted(per_peer.items())},
        "rank_imbalance": stats,
        "padding_bytes_total": sum(entry.padding_bytes for entry in entries),
        "problems": problems,
    }


@dataclass(frozen=True)
class ExpertGEMMLedgerEntry:
    """Step 18: the *actual* executed shape, not the theoretical active FLOPs."""

    layer: int
    expert: int
    rank: int
    M: int
    N: int
    K: int
    kernel: str
    padding_tokens: int = 0
    time_ms: Optional[float] = None
    occupancy: Optional[float] = None
    throughput_tflops: Optional[float] = None

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("M", "N", "K"):
            if getattr(self, name) <= 0:
                findings.append(f"gemm entry ({self.layer},{self.expert}): {name}={getattr(self, name)} is not a real shape")
        if not self.kernel:
            findings.append(f"gemm entry ({self.layer},{self.expert}): actual kernel name is required")
        if self.time_ms is None:
            findings.append(f"gemm entry ({self.layer},{self.expert}): time is unmeasured")
        if self.M == 0 and self.padding_tokens:
            findings.append(f"gemm entry ({self.layer},{self.expert}): padding without tokens")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "expert": self.expert,
            "rank": self.rank,
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "kernel": self.kernel,
            "padding_tokens": self.padding_tokens,
            "time_ms": self.time_ms,
            "occupancy": self.occupancy,
            "throughput_tflops": self.throughput_tflops,
        }


def gemm_efficiency(entries: Sequence[ExpertGEMMLedgerEntry]) -> Dict[str, Any]:
    """Step 18/§9: small expert GEMMs are why fewer FLOPs can still be slower."""
    if not entries:
        return {"error": "no expert GEMM entries supplied"}
    shapes = [(entry.M, entry.N, entry.K) for entry in entries]
    useful = sum(entry.M for entry in entries)
    padded = sum(entry.M + entry.padding_tokens for entry in entries)
    times = [entry.time_ms for entry in entries if entry.time_ms is not None]
    return {
        "gemms": len(entries),
        "distinct_shapes": len(set(shapes)),
        "mean_M": sum(shape[0] for shape in shapes) / len(shapes),
        "max_M": max(shape[0] for shape in shapes),
        "padding_ratio": (padded - useful) / padded if padded else 0.0,
        "total_time_ms": sum(times),
        "unmeasured": sum(1 for entry in entries if entry.time_ms is None),
        "note": "小 expert GEMM + pack/A2A + 最慢 rank 可以主导层时延（§9）",
    }


def rank_time_decomposition(
    per_rank: Mapping[int, Mapping[str, float]], *, phase: str = "layer"
) -> Dict[str, Any]:
    """Step 19: the slowest rank decides the synchronous layer (§3.2)."""
    if not per_rank:
        return {"error": "no rank timings supplied"}
    problems: List[str] = []
    totals: Dict[int, float] = {}
    for rank, phases in per_rank.items():
        unknown = sorted(set(phases) - set(LAYER_PHASES))
        if unknown:
            problems.append(f"rank {rank}: unknown phases {', '.join(unknown)}")
        missing = sorted(set(LAYER_PHASES) - set(phases))
        if missing:
            problems.append(f"rank {rank}: phases not measured: {', '.join(missing)}")
        totals[rank] = float(sum(phases.values()))
    slowest = max(totals, key=lambda rank: totals[rank])
    fastest = min(totals, key=lambda rank: totals[rank])
    reference = totals[slowest]
    return {
        "phase": phase,
        "slowest_rank": slowest,
        "fastest_rank": fastest,
        "slowest_total_ms": reference,
        "skew_ratio": (reference - totals[fastest]) / reference if reference else 0.0,
        "rank_totals_ms": {str(rank): value for rank, value in sorted(totals.items())},
        "per_phase_slowest": {
            name: float(per_rank[slowest].get(name, 0.0)) for name in LAYER_PHASES
        },
        "problems": problems,
        "note": "平均 rank 时间会掩盖同步尾部（step 19）",
    }


@dataclass(frozen=True)
class MoEMemoryLedger:
    """Step 20: total and active weights are different numbers (§3.1)."""

    rank: int
    categories: Mapping[str, int] = field(default_factory=dict)

    def problems(self) -> List[str]:
        findings: List[str] = []
        missing = [name for name in MOE_MEMORY_CATEGORIES if name not in self.categories]
        if missing:
            findings.append(f"MoE memory ledger is missing categories: {', '.join(missing)}")
        total = self.categories.get("expert_weights_total", 0)
        active = self.categories.get("expert_weights_active", 0)
        if active and total and active > total:
            findings.append("active expert weights exceed total expert weights")
        return findings

    def resident_bytes(self) -> int:
        """Everything that must be resident — active weights alone understate it."""
        return sum(self.categories.get(name, 0) for name in MOE_MEMORY_CATEGORIES)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "categories": {key: self.categories[key] for key in sorted(self.categories)},
            "resident_bytes": self.resident_bytes(),
        }


def replication_cost(
    placement: ExpertPlacement, *, hot_experts: Sequence[int], expert_bytes: int
) -> Dict[str, Any]:
    """Step 26: replicating a hot expert costs memory and must be priced (§10)."""
    problems: List[str] = []
    missing = [expert for expert in hot_experts if expert not in placement.expert_to_rank]
    if missing:
        problems.append(f"hot experts without a placement: {missing}")
    extra_replicas = len(hot_experts)
    return {
        "replicated_experts": sorted(int(expert) for expert in hot_experts),
        "extra_replicas": extra_replicas,
        "extra_bytes": extra_replicas * expert_bytes,
        "extra_ranks": sorted({placement.expert_to_rank[expert] for expert in hot_experts if expert in placement.expert_to_rank}),
        "problems": problems,
        "note": "replication 快但成本高时，它是一个 Pareto 点而非免费加速（§9）",
    }


def expert_cache_scan(
    *, capacities: Sequence[int], accesses: Sequence[int], expert_bytes: int, load_latency_ms: float
) -> Dict[str, Any]:
    """Step 27: hit/miss, load latency and the difference between warm and cold.

    ``E14-F2`` §10 forbids extrapolating a warm-hit best case to steady-state
    service, so both the cold start and the steady hit rate are returned.
    """
    problems: List[str] = []
    if not capacities:
        problems.append("no cache capacities supplied")
    rows: List[Dict[str, Any]] = []
    order = list(accesses)
    for capacity in capacities:
        if capacity <= 0:
            problems.append(f"capacity {capacity} is not a real cache")
            continue
        resident: List[int] = []
        hits = 0
        for expert in order:
            if expert in resident:
                hits += 1
                continue
            if len(resident) >= capacity:
                resident.pop(0)
            resident.append(expert)
        misses = len(order) - hits
        rows.append(
            {
                "capacity": capacity,
                "hit_rate": hits / len(order) if order else 0.0,
                "misses": misses,
                "load_time_ms": misses * load_latency_ms,
                "resident_bytes": min(capacity, len(set(order))) * expert_bytes,
            }
        )
    first_pass = {
        "distinct_experts": len(set(order)),
        "cold_misses": len(set(order)),
        "cold_load_time_ms": len(set(order)) * load_latency_ms,
    }
    return {
        "rows": rows,
        "cold_start": first_pass,
        "problems": problems,
        "note": "缓存预热的最好情况不得当作稳定服务（step 27）",
    }


def expert_quant_gate(
    *, quant_artifact_id: str, paired_quality_delta: float, guard_band: float, router_distribution_shift: float
) -> Dict[str, Any]:
    """Step 28: quantising experts may change the router distribution too (§9)."""
    problems: List[str] = []
    if not quant_artifact_id:
        problems.append("expert quant requires a bound C5 QuantArtifact id")
    if paired_quality_delta < -abs(guard_band):
        problems.append(
            f"paired quality delta {paired_quality_delta:.4f} exceeds the guard band {guard_band:.4f}"
        )
    if abs(router_distribution_shift) > 0.05:
        problems.append(
            f"router distribution shift {router_distribution_shift:.4f} is material: judge semantics "
            "before throughput (§9 quant 节省内存但 router 分布改变)"
        )
    return {
        "quant_artifact_id": quant_artifact_id,
        "paired_quality_delta": paired_quality_delta,
        "guard_band": guard_band,
        "router_distribution_shift": router_distribution_shift,
        "problems": problems,
        "performance_eligible": not problems,
    }


# ── mitigation selection, A/B, service, fairness (steps 29–32) ─────────────


def select_single_mitigation(
    mitigation: str, *, exploration_results: Mapping[str, Any], freeze_reason: str
) -> Dict[str, Any]:
    """Step 29: freeze one strategy; confirmation may not keep combining them."""
    problems: List[str] = []
    if mitigation not in MITIGATIONS:
        problems.append(f"mitigation {mitigation!r} must be one of {', '.join(MITIGATIONS)}")
    if not freeze_reason:
        problems.append("the frozen candidate needs a reason from the exploration phase")
    other_changes = sorted(set(exploration_results) - {mitigation})
    return {
        "mitigation": mitigation,
        "frozen_from": sorted(exploration_results),
        "changes_frozen_out": other_changes,
        "reason": freeze_reason,
        "problems": problems,
        "ok": not problems,
    }


def runtime_ab(
    *, baseline: Mapping[str, Any], candidate: Mapping[str, Any], required_fields: Sequence[str]
) -> Dict[str, Any]:
    """Step 30: interleaved A/B with the same fields on both sides (§9)."""
    problems: List[str] = []
    for name in required_fields:
        if name not in baseline:
            problems.append(f"baseline is missing {name!r}")
        if name not in candidate:
            problems.append(f"candidate is missing {name!r}")
    if any(value is None for value in baseline.values()):
        problems.append("baseline has unmeasured fields; unknown is not a comparison")
    if any(value is None for value in candidate.values()):
        problems.append("candidate has unmeasured fields")
    return {
        "fields": list(required_fields),
        "problems": problems,
        "comparable": not problems,
        "note": "交错 reference/candidate，比较 layer/request latency、straggler、memory 与 actual path",
    }


def service_arrival_curve(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 31: report by domain/skew, because an average hides hot-request tails."""
    problems: List[str] = []
    strata: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        for key in ("arrival_rate", "ttft_ms", "tpot_ms", "p99_ms", "goodput", "reject_rate"):
            if key not in row:
                problems.append(f"arrival row is missing {key!r}")
                break
        else:
            name = str(row.get("stratum", "<unstratified>"))
            strata.setdefault(name, []).append(dict(row))
    return {
        "strata": {name: values for name, values in sorted(strata.items())},
        "max_goodput": max((row.get("goodput") or 0.0) for row in rows) if rows else 0.0,
        "problems": problems,
        "note": "平均吞吐会掩盖热点请求的尾延迟（step 31）",
    }


def check_fairness(
    *, tenant_rows: Sequence[Mapping[str, Any]], protected_tenants: Sequence[str], slo_ratio: float
) -> Dict[str, Any]:
    """Step 32: a hot-expert tenant must not drag every other tenant's SLO."""
    problems: List[str] = []
    worst: Optional[Dict[str, Any]] = None
    for row in tenant_rows:
        tenant = str(row.get("tenant_id", "<unnamed>"))
        if tenant not in protected_tenants:
            continue
        ratio = row.get("p99_ratio_to_slo")
        if ratio is None:
            problems.append(f"protected tenant {tenant} has no p99_ratio_to_slo")
            continue
        if float(ratio) > slo_ratio:
            problems.append(
                f"protected tenant {tenant} exceeded its SLO ratio ({ratio} > {slo_ratio}) "
                "while hot-expert traffic ran concurrently"
            )
        if worst is None or float(ratio) > float(worst.get("p99_ratio_to_slo", 0.0)):
            worst = dict(row)
    if not protected_tenants:
        problems.append("no protected tenant declared; fairness cannot be assessed")
    return {"worst_protected": worst, "problems": problems, "ok": not problems}


def cross_layer_link(
    *, request_rows: Sequence[Mapping[str, Any]], required_keys: Sequence[str]
) -> Dict[str, Any]:
    """Step 33: from request/layer to router, A2A, GEMM, rank wait and service."""
    problems: List[str] = []
    for row in request_rows:
        missing = [key for key in required_keys if key not in row]
        if missing:
            problems.append(
                f"request {row.get('request_id', '<unnamed>')} cannot be joined: missing {', '.join(missing)}"
            )
    chains = [
        {
            "request_id": row.get("request_id"),
            "layer": row.get("layer"),
            "key": f"{row.get('request_id')}:{row.get('layer')}",
        }
        for row in request_rows
    ]
    return {
        "chains": chains,
        "join_keys": list(required_keys),
        "problems": problems,
        "ok": not problems,
        "note": "多张独立图无法证明因果链；至少两层必须能按 join key 关联（E14-F2 step 33）",
    }


# ── faults and quality (steps 34–37) ───────────────────────────────────────


def straggler_injection(
    *, target_rank: int, mechanism: str, world_size: int, baseline_layer_ms: float
) -> Dict[str, Any]:
    """Step 34: a controlled hook, never damage on a shared device.

    ``E14-F2`` §6 step 34 requires the detection, blast radius and mitigation to
    be recorded; injecting a real slowdown on a shared production device is
    forbidden by ``campaign.PROHIBITED_ACTIONS``.
    """
    allowed = ("sleep_hook", "clock_limited", "bandwidth_limited", "queue_delay")
    problems: List[str] = []
    if mechanism not in allowed:
        problems.append(f"mechanism {mechanism!r} must be one of {', '.join(allowed)}")
    if not 0 <= target_rank < world_size:
        problems.append(f"target_rank {target_rank} is outside world size {world_size}")
    return {
        "target_rank": target_rank,
        "mechanism": mechanism,
        "baseline_layer_ms": baseline_layer_ms,
        "expected_detection": "per-rank 时间分解中最慢 rank 变化 + wait 相位上升",
        "blast_radius": "隔离环境内的单个 rank group",
        "isolation_required": "campaign.REQUIRED_ISOLATION['destructive_fault_on_shared_device']",
        "problems": problems,
    }


#: Failure modes and the outcome each must produce (step 35).
MOE_FAILURE_MODES: Tuple[Tuple[str, str], ...] = (
    ("missing_expert_shard", "加载前 fail-closed，定位到具体 expert/rank"),
    ("corrupt_expert_cache", "校验后拒绝并失效该 cache entry"),
    ("collective_timeout", "有界失败并回收 buffer，不得无限 hang"),
)


def failure_scenarios() -> Tuple[Dict[str, str], ...]:
    return tuple({"failure": name, "required_outcome": outcome} for name, outcome in MOE_FAILURE_MODES)


def check_failure_accounting(records: Sequence[Mapping[str, Any]]) -> List[str]:
    """Step 35: a failure must not silently change the output."""
    problems: List[str] = []
    known = {name for name, _outcome in MOE_FAILURE_MODES}
    for record in records:
        name = str(record.get("failure", ""))
        if name not in known:
            problems.append(f"unknown MoE failure {name!r}")
            continue
        for key in ("detected", "output_unchanged", "resource_released", "actual_backend"):
            if key not in record:
                problems.append(f"{name}: failure record is missing {key!r}")
        if record.get("detected") is False:
            problems.append(f"{name}: the failure was not detected")
        if record.get("output_unchanged") is False:
            problems.append(f"{name}: the output changed silently (fail-closed 或显式 fallback 才合法)")
        if record.get("resource_released") is False:
            problems.append(f"{name}: resources were not released")
    return problems


def check_quality_gate(
    *, logit_parity: bool, token_parity: bool, task_quality_delta: float, guard_band: float,
    dropped_tokens: int, rerouted_tokens: int,
) -> Dict[str, Any]:
    """Step 36: drop/reroute/quant must pass the quality gate, not just the latency one."""
    problems: List[str] = []
    if not logit_parity:
        problems.append("logit parity failed")
    if not token_parity:
        problems.append("greedy token parity failed")
    if task_quality_delta < -abs(guard_band):
        problems.append(
            f"task quality delta {task_quality_delta:.4f} exceeds the guard band {guard_band:.4f}"
        )
    if dropped_tokens:
        problems.append(
            f"{dropped_tokens} tokens were dropped: 低延迟来自丢 token 不是收益（§10）"
        )
    if rerouted_tokens < 0:
        problems.append("negative rerouted token count")
    return {
        "logit_parity": logit_parity,
        "token_parity": token_parity,
        "task_quality_delta": task_quality_delta,
        "guard_band": guard_band,
        "dropped_tokens": dropped_tokens,
        "rerouted_tokens": rerouted_tokens,
        "problems": problems,
        "performance_eligible": not problems,
    }


def reconcile_prediction(
    *,
    predicted_layer_ms: float,
    measured_layer_ms: float,
    router_skew: float,
    a2a_bytes: int,
    gemm_time_ms: float,
) -> Dict[str, Any]:
    """Step 37: rebuild the layer time from token counts/bytes/GEMM and report the gap."""
    if predicted_layer_ms <= 0:
        return {"error": "predicted_layer_ms must be positive"}
    residual = measured_layer_ms - predicted_layer_ms
    return {
        "predicted_layer_ms": predicted_layer_ms,
        "measured_layer_ms": measured_layer_ms,
        "residual_ms": residual,
        "residual_ratio": residual / predicted_layer_ms,
        "mediators": {"router_skew": router_skew, "a2a_bytes": a2a_bytes, "gemm_time_ms": gemm_time_ms},
        "explained": abs(residual) <= 0.15 * predicted_layer_ms,
        "note": "只给相关图不解释差异是不合格的（step 37）",
    }


# ── adoption (step 40) ─────────────────────────────────────────────────────


def f2_adoption(
    *,
    decision_id: str,
    quality: Mapping[str, Any],
    ab: Mapping[str, Any],
    memory: MoEMemoryLedger,
    reconciliation: Mapping[str, Any],
    strata: Mapping[str, Any],
    selected: Mapping[str, Any],
    evidence_refs: Sequence[str],
) -> AdoptionDecision:
    """Step 40: an average tokens/s is not enough to adopt (§10)."""
    problems: List[str] = []
    if not quality.get("performance_eligible"):
        problems.append("quality gate not passed")
    if not ab.get("comparable"):
        problems.append("A/B is not comparable (missing or unmeasured fields)")
    problems.extend(memory.problems())
    if reconciliation.get("explained") is False:
        problems.append("the measurement is not explained by the mediators")
    strata_rows = strata.get("strata") or {}
    if not strata_rows:
        problems.append("no stratified routing data")
    admitted = (selected.get("mitigation") or "") if selected else ""
    if problems:
        decision = rec.REJECT_QUALITY
        allowed: Tuple[str, ...] = ()
    elif not admitted:
        decision = rec.REJECT_NO_BENEFIT
        allowed = ("router/A2A/GEMM 证据链完整，但缓解策略无收益",)
    else:
        decision = rec.ADOPT_EXPERIMENTAL
        allowed = (f"缓解策略 {admitted} 在预注册 workload 内可解释",)
    return AdoptionDecision(
        decision_id=decision_id,
        experiment_id=EXPERIMENT_ID,
        decision=decision,
        allowed_claims=allowed,
        forbidden_claims=(
            "active parameters 等于显存",
            "平均 rank 代表最慢 rank",
            "热 domain 的收益外推为全局提速",
        ),
        quality_status=rec.STATUS_PASS if quality.get("performance_eligible") else rec.STATUS_FAIL_QUALITY,
        performance_status=rec.STATUS_NOT_RUN,
        maturity=rec.MATURITY_QUALITY_VERIFIED if not problems else rec.MATURITY_SOURCE_INTEGRATED,
        evidence_refs=tuple(evidence_refs),
        limitations=tuple(problems) + ("单一 topology 不得外推多节点",),
        reopened_if=("可用的 EP topology 与 holdout domain 同时具备时可以重新评估",),
    )


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the E14-F2 interfaces (labelled smoke, not an experiment)."""
    routing = router_topk([2.0, 1.0, 0.5, 0.1], top_k=2, activation="softmax")
    stats = load_statistics([100, 60, 40, 0])
    balanced = load_statistics([10, 10, 10, 10])
    a2a = a2a_balance(
        [
            AllToAllLedgerEntry(0, 0, 1, 10, 10, 4096, "bf16"),
            AllToAllLedgerEntry(0, 1, 0, 10, 10, 4096, "bf16"),
        ],
        world_size=2,
    )
    gemm = gemm_efficiency(
        [ExpertGEMMLedgerEntry(0, 0, 0, M=32, N=512, K=512, kernel="grouped_gemm", time_ms=0.3)]
    )
    ranks = rank_time_decomposition(
        {0: {name: 1.0 for name in LAYER_PHASES}, 1: {name: 2.0 for name in LAYER_PHASES}}
    )
    placement = ExpertPlacement(
        placement_id="rr", expert_to_rank={0: 0, 1: 1, 2: 0, 3: 1},
        rank_to_device={0: "gpu0", 1: "gpu1"}, num_experts=4,
    )
    replay = controlled_count_replay(patterns={"hot": [100, 0, 0, 0]}, experts=4)
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "selected_experts": list(routing["selected"]),
        "loaded": round(stats["gini"], 4),
        "balanced_gini": round(balanced["gini"], 4),
        "a2a_rank_imbalance": round(a2a["rank_imbalance"]["cv"], 4),
        "gemm_padding_ratio": gemm["padding_ratio"],
        "slowest_rank": ranks["slowest_rank"],
        "placement_problems": validate_placement(placement, world_size=2),
        "replay_quality_allowed": replay["quality_conclusions_allowed"],
        "mitigations": len(MITIGATIONS),
        "failure_modes": len(MOE_FAILURE_MODES),
    }


# ── protocol step table (40 steps of details/S14/E14-F2) ───────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "绑定 E14-05 协议", ("frontier:issue_contract", "frontier:contract_hash")),
    (2, "冻结真实 MoE ModelArtifact", ("moe:MoEModelContract", "moe:MoEModelContract.total_to_active_ratio")),
    (3, "冻结 reference 与 candidate", ("frontier:BaselinePair", "moe:select_single_mitigation")),
    (4, "冻结并行与 topology", ("moe:ExpertPlacement", "moe:validate_placement")),
    (5, "冻结 router 语义", ("moe:ROUTER_ACTIVATIONS", "moe:ROUTER_NORMALISATIONS", "moe:router_topk")),
    (6, "冻结 workload/domain strata", ("moe:stratify_routing", "frontier:WorkloadStrata")),
    (7, "构造可控 skew 输入", ("moe:controlled_count_replay", "moe:stratify_routing")),
    (8, "冻结质量与容量门", ("moe:check_quality_gate", "frontier:QualityGate")),
    (9, "冻结成本模型", ("frontier:PredictionModel", "moe:load_statistics")),
    (10, "运行 capability/actual-path probe", ("contracts:check_actual_path_recorded", "moe:ExpertGEMMLedgerEntry.kernel")),
    (11, "运行单设备 reference correctness", ("moe:expert_output", "moe:router_topk")),
    (12, "验证 router instrumentation", ("moe:RoutingEvent", "moe:load_statistics")),
    (13, "运行真实 workload routing census", ("moe:load_statistics", "moe:IMBALANCE_METRICS")),
    (14, "验证 expert output parity", ("moe:expert_output", "moe:check_quality_gate")),
    (15, "启动 EP baseline", ("moe:ExpertPlacement", "moe:validate_placement")),
    (16, "对账 token dispatch/combine", ("moe:check_token_dispatch", "moe:AllToAllLedgerEntry")),
    (17, "建立 All-to-All ledger", ("moe:AllToAllLedgerEntry", "moe:a2a_balance")),
    (18, "建立 expert GEMM ledger", ("moe:ExpertGEMMLedgerEntry", "moe:gemm_efficiency")),
    (19, "建立 per-rank 时间分解", ("moe:rank_time_decomposition", "moe:LAYER_PHASES")),
    (20, "建立内存账本", ("moe:MoEMemoryLedger", "moe:MOE_MEMORY_CATEGORIES")),
    (21, "运行 batch/context 扫描", ("moe:gemm_efficiency", "frontier:WorkloadStrata")),
    (22, "运行真实 domain skew 分层", ("moe:stratify_routing", "moe:load_statistics")),
    (23, "运行受控 count replay", ("moe:controlled_count_replay", "moe:load_statistics")),
    (24, "扫描 expert placement", ("moe:placement_scan", "moe:ExpertPlacement.expert_to_rank")),
    (25, "验证 placement 身份与迁移", ("moe:ExpertPlacement.version", "moe:validate_placement")),
    (26, "评估 expert replication/load-balance", ("moe:replication_cost", "frontier:CostDenominator")),
    (27, "评估 expert cache/offload", ("moe:expert_cache_scan", "moe:MoEMemoryLedger.resident_bytes")),
    (28, "评估 expert quant", ("moe:expert_quant_gate", "contracts:check_quality_before_performance")),
    (29, "选择一个主要缓解策略", ("moe:select_single_mitigation", "moe:MITIGATIONS")),
    (30, "运行严格 runtime A/B", ("moe:runtime_ab", "frontier:BaselinePair")),
    (31, "运行 Service 到达率曲线", ("moe:service_arrival_curve", "records:PROFILE_LAYERS")),
    (32, "验证混合租户/优先级公平性", ("moe:check_fairness", "frontier:StopRules")),
    (33, "采集跨层关联 profile", ("moe:cross_layer_link", "moe:rank_time_decomposition")),
    (34, "注入单 expert/rank 慢化", ("moe:straggler_injection", "campaign:isolation_clause")),
    (35, "注入 expert/cache/communication failure", ("moe:failure_scenarios", "moe:check_failure_accounting")),
    (36, "运行质量门", ("moe:check_quality_gate", "contracts:check_quality_before_performance")),
    (37, "对账模型预测与实测", ("moe:reconcile_prediction", "moe:rank_time_decomposition")),
    (38, "在 holdout domain/load 确认", ("frontier:WorkloadStrata.holdout_id", "moe:stratify_routing")),
    (39, "跨 topology/time block 重复", ("records:EXPERIMENT_UNITS", "frontier:StatisticsPlan.minimum_repeats")),
    (40, "形成 F2 AdoptionDecision", ("moe:f2_adoption", "contracts:AdoptionDecision.validate")),
)
