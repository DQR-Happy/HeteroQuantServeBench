"""E14-02 — distributed training semantics, state, communication and checkpoint.

Core judgement: pick **one** strategy (DDP/FSDP/ZeRO/Megatron) by capability/ADR;
a real small-model run must be reproducible, its global-batch update must equal a
single-device reference within a preregistered tolerance, the actual world
size/collective path must be provable (no silent single-device fallback), memory
must be explained by a ledger, and the checkpoint must resume a *continuous*
trajectory.

Provided here (the forty steps' interface surface):

* :class:`StrategyChoice` — the ADR pick, capability-gated (step 1);
* :class:`GlobalBatchSpec`, :func:`effective_token_count` — the global batch as
  an *identity of samples*, not a number (steps 5, 15);
* :func:`reconstruct_global_loss` — rebuild the global loss from raw
  numerator/denominator instead of averaging averaged values (steps 4, 18);
* :func:`tolerance_gate`, :func:`reconcile_tensors` — abs/rel/cosine/ULP with the
  first-divergent layer named (steps 19–20);
* :class:`CollectiveLedgerEntry` — the collective account (step 25);
* :class:`MemoryLedger`, :func:`attribute_memory_gap` — the memory account and
  the theoretical-`1/W` reconciliation (steps 26–27);
* :class:`CheckpointInventory`, :func:`verify_inventory` — completeness and
  fail-closed checks (steps 29–30, 36–37);
* :func:`compare_resume_continuity` — step-by-step trajectory continuity after a
  stop→new-process resume (steps 33–34);
* :func:`reshard_capability` — reshard only when the format supports it (step 35);
* :class:`TrainingStepRecord`, :class:`DistributedTrainingVerdict` (steps 23, 40).

Nothing here launches a process group or computes a gradient; a single-device
environment is ``BLOCKED_CAPABILITY``, never a mock PASS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.experimental import records as rec
from hqsb.experimental.contracts import CheckpointArtifact
from hqsb.experimental.identity import (
    SCHEMA_PREFIX,
    RankIdentity,
    canonical_digest,
    topology_digest,
)

EXPERIMENT_ID = "E14-02"
TITLE = "分布式训练语义、状态、通信与 Checkpoint Smoke"
LEVEL = "P0"

CLAIM_BOUNDARY = (
    "只证明小模型、给定 topology 的训练能力；不证明模型质量更好，也不证明所选框架优于其他框架"
    "（E14-02 §13）。"
)

#: The four strategies; exactly one is primary (``E14-02`` step 1).
STRATEGIES: Tuple[str, ...] = ("ddp", "fsdp", "zero", "megatron")

#: Sharding stages FSDP/ZeRO must declare.
SHARD_STAGES: Tuple[str, ...] = ("none", "1", "2", "3")

#: A "mixed precision" label must expand into a real per-component contract
#: (§28 error: 一个 BF16 标签掩盖实际混合精度).
PRECISION_COMPONENTS: Tuple[str, ...] = (
    "parameter",
    "activation",
    "accumulation",
    "gradient_reduction",
    "master_weight",
    "optimizer",
    "loss",
)

#: Memory categories the ledger must separate (§8.2).
MEMORY_CATEGORIES: Tuple[str, ...] = (
    "parameters",
    "gradients",
    "optimizer_states",
    "master_weights",
    "activations",
    "communication_buckets",
    "workspace",
    "fragmentation",
)

#: Wall-clock phases the step time must decompose into (``E14-02`` step 24).
TIMING_PHASES: Tuple[str, ...] = (
    "data",
    "forward",
    "backward",
    "collective",
    "optimizer",
    "zero_grad",
    "logging",
    "checkpoint",
)

#: Quantities compared by the resume-continuity check (``E14-02`` step 34).
RESUME_QUANTITIES: Tuple[str, ...] = (
    "batch_ids",
    "loss",
    "grad_norm",
    "parameter_digest",
    "optimizer_state_digest",
    "lr",
    "scaler",
    "rng_output",
    "data_cursor",
)


@dataclass(frozen=True)
class StrategyChoice:
    """The ADR pick (``E14-02`` step 1), capability-gated."""

    strategy: str
    world_size: int
    shard_stage: str = "none"
    checkpoint_format: str = ""
    rationale: str = ""
    requested_implementation: str = ""
    actual_implementation: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.strategy not in STRATEGIES:
            findings.append(f"strategy must be one of {', '.join(STRATEGIES)}")
        if self.world_size < 1:
            findings.append("world_size must be >= 1 (launcher 参数不等于 actual world size)")
        if self.shard_stage not in SHARD_STAGES:
            findings.append(f"shard_stage must be one of {', '.join(SHARD_STAGES)}")
        if self.strategy in ("fsdp", "zero") and self.shard_stage == "none":
            findings.append(f"{self.strategy} must declare a shard stage")
        if not self.rationale:
            findings.append("a strategy without a rationale is a default, not an ADR")
        if not self.requested_implementation:
            findings.append("requested_implementation is required")
        if self.requested_implementation and self.requested_implementation != self.actual_implementation:
            findings.append(
                f"declared {self.requested_implementation!r} but actual is "
                f"{self.actual_implementation!r} (隐式单卡回退 = FAIL)"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "world_size": self.world_size,
            "shard_stage": self.shard_stage,
            "checkpoint_format": self.checkpoint_format,
            "rationale": self.rationale,
            "requested_implementation": self.requested_implementation,
            "actual_implementation": self.actual_implementation,
        }


def strategy_verdict(choice: StrategyChoice, *, distributed_available: bool) -> Dict[str, Any]:
    """Step 10: a missing multi-device capability is ``BLOCKED_CAPABILITY``.

    ``E14-02`` §5/§11: 若只有单设备，分布式执行保持 ``BLOCKED_CAPABILITY``，不能
    mock 成通过.
    """
    if not distributed_available:
        return {
            "status": rec.STATUS_BLOCKED_CAPABILITY,
            "capability": "distributed_training",
            "state": rec.CAP_DEVICE_UNAVAILABLE,
            "reason_code": "DEVICE_UNAVAILABLE",
            "requested_implementation": choice.strategy,
            "actual_implementation": "single-device",
            "reason": "fewer devices than the declared world size; mock execution is not a PASS",
        }
    problems = choice.problems()
    return {
        "status": rec.STATUS_NOT_RUN if not problems else rec.STATUS_INVALID_IDENTITY,
        "capability": "distributed_training",
        "state": rec.CAP_AVAILABLE,
        "requested_implementation": choice.requested_implementation,
        "actual_implementation": choice.actual_implementation,
        "problems": problems,
    }


@dataclass(frozen=True)
class GlobalBatchSpec:
    """``E14-02`` step 5 — the global batch as a set of samples."""

    microbatch: int
    accumulation: int
    world_size: int
    sample_ids: Tuple[str, ...] = ()
    token_counts: Tuple[int, ...] = ()
    drop_last: bool = False

    @property
    def global_batch(self) -> int:
        return self.microbatch * self.accumulation * self.world_size

    def effective_token_count(self) -> int:
        """Token-mean denominator (§8.1): sum of per-sample valid tokens."""
        return sum(self.token_counts)

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("microbatch", "accumulation", "world_size"):
            if getattr(self, name) <= 0:
                findings.append(f"{name} must be positive")
        if self.sample_ids and len(self.sample_ids) != self.global_batch:
            findings.append(
                f"global batch {self.global_batch} but {len(self.sample_ids)} sample ids "
                "(重复/遗漏改变等效 global batch)"
            )
        if self.sample_ids and len(set(self.sample_ids)) != len(self.sample_ids):
            findings.append("sample_ids contain duplicates (数据分片失败)")
        if self.token_counts and len(self.token_counts) != len(self.sample_ids):
            findings.append("token_counts and sample_ids have different lengths")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "microbatch": self.microbatch,
            "accumulation": self.accumulation,
            "world_size": self.world_size,
            "global_batch": self.global_batch,
            "effective_token_count": self.effective_token_count(),
            "sample_ids": list(self.sample_ids),
            "token_counts": list(self.token_counts),
            "drop_last": self.drop_last,
        }


@dataclass(frozen=True)
class LossReduction:
    """The frozen loss definition (``E14-02`` step 4)."""

    reduction: str  # token_mean | sample_mean | sum
    ignore_index: int = -100
    label_shift: int = 0
    regularization: str = ""

    def problems(self) -> List[str]:
        if self.reduction not in ("token_mean", "sample_mean", "sum"):
            return [f"unknown reduction {self.reduction!r}"]
        return []


def reconstruct_global_loss(
    rank_numerators: Sequence[float],
    rank_denominators: Sequence[int],
    reduction: LossReduction,
) -> Dict[str, Any]:
    """Rebuild the global loss from raw sums (``E14-02`` step 18).

    Averaging each rank's already-averaged display loss double-averages when
    ranks hold different valid-token counts, so the numerators and denominators
    are summed and reduced exactly once.
    """
    if len(rank_numerators) != len(rank_denominators):
        raise ConfigError("rank_numerators and rank_denominators must have the same length")
    if not rank_numerators:
        raise ConfigError("at least one rank is required")
    total_num = float(sum(rank_numerators))
    total_den = int(sum(rank_denominators))
    if reduction.reduction == "sum":
        return {"loss": total_num, "denominator": 1, "method": "sum", "ranks": len(rank_numerators)}
    if total_den <= 0:
        return {"loss": None, "denominator": total_den, "error": "denominator must be positive"}
    return {
        "loss": total_num / total_den,
        "numerator": total_num,
        "denominator": total_den,
        "method": reduction.reduction,
        "ranks": len(rank_numerators),
    }


def naive_rank_mean(rank_losses: Sequence[float]) -> Dict[str, Any]:
    """The *wrong* reduction, kept so the protocol's warning is executable.

    ``E14-02`` §10: 优先检查 logging reduction，而非立即判数学错误 — the gap
    between this and :func:`reconstruct_global_loss` is exactly that artefact.
    """
    if not rank_losses:
        raise ConfigError("at least one rank loss is required")
    return {
        "loss": sum(rank_losses) / len(rank_losses),
        "method": "mean_of_rank_means",
        "caveat": "valid only when every rank holds the same number of valid tokens",
    }


def tolerance_gate(
    reference: float,
    candidate: float,
    *,
    abs_tol: float,
    rel_tol: float,
    field_name: str = "value",
) -> Dict[str, Any]:
    """One comparison against a *preregistered* tolerance."""
    diff = candidate - reference
    within = abs(diff) <= abs_tol + rel_tol * abs(reference)
    if reference:
        rel_diff = diff / reference
    else:
        rel_diff = 0.0 if diff == 0 else float("inf")
    return {
        "field": field_name,
        "reference": reference,
        "candidate": candidate,
        "abs_diff": diff,
        "rel_diff": rel_diff,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "within": within,
    }


def reconcile_tensors(
    pairs: Sequence[Mapping[str, Any]],
    *,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
    cosine_floor: float = 0.999999,
    ulp_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Layer-by-layer reconciliation (``E14-02`` steps 19–20).

    A scalar gradient norm can match while a middle layer is wrong, so every
    supplied tensor is compared and the first divergent one is named.
    """
    rows: List[Dict[str, Any]] = []
    first_divergence: Optional[str] = None
    for pair in pairs:
        name = str(pair.get("tensor", pair.get("name", "")))
        reference = pair.get("reference")
        candidate = pair.get("candidate")
        if reference is None or candidate is None:
            rows.append({"tensor": name, "status": "missing", "within": False})
            if first_divergence is None:
                first_divergence = name
            continue
        abs_diff = float(candidate) - float(reference)
        rel_diff = abs(abs_diff / reference) if reference else (0.0 if abs_diff == 0 else float("inf"))
        cosine = float(pair.get("cosine", 1.0))
        ulp = pair.get("ulp")
        within = abs(abs_diff) <= abs_tol + rel_tol * abs(reference) and cosine >= cosine_floor
        if ulp_budget is not None and ulp is not None:
            within = within and int(ulp) <= ulp_budget
        rows.append(
            {
                "tensor": name,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
                "cosine": cosine,
                "ulp": ulp,
                "within": within,
            }
        )
        if not within and first_divergence is None:
            first_divergence = name
    return {
        "tensors": rows,
        "all_within": all(row["within"] for row in rows),
        "first_divergence": first_divergence,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "cosine_floor": cosine_floor,
        "ulp_budget": ulp_budget,
    }


@dataclass(frozen=True)
class CollectiveLedgerEntry:
    """One collective call (``E14-02`` step 25)."""

    collective_type: str
    shape: Tuple[int, ...]
    dtype: str
    bytes: int
    calls: int
    stream: str = ""
    start_ns: int = 0
    end_ns: int = 0
    overlapped: bool = False

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("collective_type", "dtype"):
            if not getattr(self, name):
                findings.append(f"collective entry is missing {name!r}")
        if self.bytes <= 0 and self.calls:
            findings.append("a non-empty collective must account for nonzero bytes")
        if self.end_ns and self.start_ns and self.end_ns < self.start_ns:
            findings.append("end_ns precedes start_ns")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "collective_type": self.collective_type,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "bytes": self.bytes,
            "calls": self.calls,
            "stream": self.stream,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "overlapped": self.overlapped,
        }


@dataclass(frozen=True)
class MemoryLedger:
    """Per-rank memory account (``E14-02`` step 26)."""

    rank: int
    phase: str
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    peak_bytes: int = 0
    host_bytes: int = 0
    pinned_bytes: int = 0
    categories: Mapping[str, int] = field(default_factory=dict)

    def problems(self) -> List[str]:
        findings: List[str] = []
        missing = [cat for cat in MEMORY_CATEGORIES if cat not in self.categories]
        if missing:
            findings.append(f"memory ledger is missing categories: {', '.join(missing)}")
        if self.peak_bytes < self.allocated_bytes:
            findings.append("peak_bytes < allocated_bytes is inconsistent")
        if self.reserved_bytes and self.allocated_bytes > self.reserved_bytes:
            findings.append("allocated_bytes > reserved_bytes is inconsistent")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "phase": self.phase,
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "peak_bytes": self.peak_bytes,
            "host_bytes": self.host_bytes,
            "pinned_bytes": self.pinned_bytes,
            "categories": {key: self.categories[key] for key in sorted(self.categories)},
        }


def attribute_memory_gap(
    ledger: MemoryLedger, *, theoretical_per_rank: int, world_size: int
) -> Dict[str, Any]:
    """Theoretical `1/W` sharding vs measured peak (``E14-02`` step 27).

    ``E14-02`` §10: 参数状态节省但峰值不降 is expected when activation,
    all-gather, bucket or fragmentation dominate — the point is that the gap is
    *explained*, not that it is zero.
    """
    if world_size <= 0:
        return {"error": "world_size must be positive"}
    expected = theoretical_per_rank // world_size
    gap = ledger.peak_bytes - expected
    return {
        "theoretical_per_rank": expected,
        "measured_peak": ledger.peak_bytes,
        "gap_bytes": gap,
        "gap_ratio": gap / expected if expected else float("inf"),
        "unexplained_if_nonzero": gap > 0,
        "categories": {key: ledger.categories.get(key) for key in MEMORY_CATEGORIES},
        "note": "activation/bucket/gather/fragmentation 使实测 > 理论 1/W；必须解释而非宣称",
    }


# ── precision, sharding, straggler (steps 8, 15, 21, 28) ───────────────────


def validate_precision_contract(contract: Mapping[str, str]) -> List[str]:
    """Step 8: expand ``mixed precision`` into a per-component dtype contract."""
    findings: List[str] = []
    for component in PRECISION_COMPONENTS:
        value = contract.get(component)
        if not value:
            findings.append(f"precision contract is missing component {component!r}")
        elif value not in ("fp32", "tf32", "bf16", "fp16", "fp8", "int8", "int4"):
            findings.append(f"precision component {component!r} has unknown dtype {value!r}")
    if contract.get("gradient_reduction") and contract.get("accumulation") == "fp16":
        findings.append("fp16 accumulation with fp16 gradient reduction is a known instability, not a default")
    return findings


def check_data_sharding(
    per_rank_sample_ids: Mapping[int, Sequence[str]], *, world_size: int, expected_total: Optional[int] = None
) -> Dict[str, Any]:
    """Step 15: the union must be complete and the intersection empty."""
    problems: List[str] = []
    if len(per_rank_sample_ids) != world_size:
        problems.append(f"{len(per_rank_sample_ids)} ranks reported for world_size {world_size}")
    all_ids: List[str] = []
    for rank, ids in sorted(per_rank_sample_ids.items()):
        all_ids.extend(ids)
        if len(set(ids)) != len(ids):
            problems.append(f"rank {rank} repeats a sample inside one step")
    union = set(all_ids)
    if len(union) != len(all_ids):
        problems.append("sample sets overlap across ranks (重复样本改变等效 global batch)")
    if expected_total is not None and len(union) != expected_total:
        problems.append(f"union has {len(union)} samples, expected {expected_total}")
    return {
        "union_size": len(union),
        "total_assignments": len(all_ids),
        "disjoint": len(union) == len(all_ids),
        "problems": problems,
        "ok": not problems,
    }


def accumulation_sync_plan(*, accumulation: int, sync_every: int) -> Dict[str, Any]:
    """Step 21: when the gradient sync happens (and that the last microbatch does)."""
    problems: List[str] = []
    if accumulation <= 0:
        problems.append("accumulation must be positive")
    if sync_every != accumulation:
        problems.append(
            f"sync_every={sync_every} != accumulation={accumulation}; either a microbatch syncs too "
            "often or the last step omits its sync"
        )
    return {
        "accumulation": accumulation,
        "sync_every": sync_every,
        "loss_scaling": "per_microbatch_then_reduced",
        "final_microbatch_syncs": sync_every == accumulation,
        "problems": problems,
        "ok": not problems,
    }


def detect_straggler(
    phase_times_ms: Mapping[int, Mapping[str, float]], *, tolerance_ratio: float = 0.05
) -> Dict[str, Any]:
    """Step 28: the slowest rank decides the synchronous step, not the mean."""
    if not phase_times_ms:
        return {"error": "no rank timings supplied"}
    totals = {rank: float(sum(phases.values())) for rank, phases in phase_times_ms.items()}
    slowest = max(totals, key=lambda rank: totals[rank])
    fastest = min(totals, key=lambda rank: totals[rank])
    reference = totals[slowest]
    skew = (reference - totals[fastest]) / reference if reference else 0.0
    per_phase: Dict[str, Dict[str, float]] = {}
    for phase in TIMING_PHASES:
        values = [float(phases.get(phase, 0.0)) for phases in phase_times_ms.values()]
        if values:
            per_phase[phase] = {"max": max(values), "min": min(values)}
    return {
        "slowest_rank": slowest,
        "fastest_rank": fastest,
        "slowest_total_ms": reference,
        "skew_ratio": skew,
        "exceeds_tolerance": skew > tolerance_ratio,
        "per_phase_spread": per_phase,
        "rank_totals_ms": {str(rank): value for rank, value in sorted(totals.items())},
    }


# ── checkpoint completeness and resume (steps 29–37) ───────────────────────


@dataclass
class CheckpointInventory:
    """Checkpoint completeness plus the fail-closed fixtures (steps 29–37)."""

    artifact: CheckpointArtifact
    expected_shards: int = 1
    present_shards: Tuple[str, ...] = ()
    missing_shards: Tuple[str, ...] = ()
    completion_marker: str = ""
    topology: Tuple[RankIdentity, ...] = ()
    save_mode: str = "synchronous"
    pause_ms: Optional[float] = None

    def topology_digest(self) -> str:
        if not self.topology:
            return ""
        return topology_digest(self.topology)

    def verify(self) -> Dict[str, Any]:
        """Completeness: every required state component, all shards, a real root."""
        problems: List[str] = self.artifact.validate()
        absent = self.artifact.missing_state()
        if absent:
            problems.append("checkpoint is missing: " + ", ".join(absent))
        if len(self.present_shards) != self.expected_shards:
            problems.append(
                f"{len(self.present_shards)} of {self.expected_shards} shards present"
                + (f" (missing: {', '.join(self.missing_shards)})" if self.missing_shards else "")
            )
        if not self.completion_marker:
            problems.append(
                "no completion marker: 目录存在或 _SUCCESS 文件单独不构成完整性证明（step 30）"
            )
        if self.topology and self.topology_digest() != self.artifact.parallel_topology_digest:
            problems.append("checkpoint topology digest does not match the recorded parallel topology")
        if not self.artifact.atomic:
            problems.append(
                "checkpoint is not marked atomic: 部分写入会让消费者看到半个状态（step 37）"
            )
        return {
            "checkpoint_id": self.artifact.checkpoint_id,
            "shards_present": len(self.present_shards),
            "shards_expected": self.expected_shards,
            "aggregate_root": self.artifact.aggregate_root() if self.artifact.inventory else "",
            "complete": not problems,
            "problems": problems,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "artifact": self.artifact.as_dict(),
            "expected_shards": self.expected_shards,
            "present_shards": list(self.present_shards),
            "missing_shards": list(self.missing_shards),
            "completion_marker": self.completion_marker,
            "topology_digest": self.topology_digest(),
            "save_mode": self.save_mode,
            "pause_ms": self.pause_ms,
        }


def compare_resume_continuity(
    control: Mapping[str, Any], resumed: Mapping[str, Any], *, tolerances: Mapping[str, float]
) -> Dict[str, Any]:
    """Step 34: step-by-step trajectory continuity, not a final-loss comparison.

    ``E14-02`` §11: 只比较恢复后的最终 loss is not an acceptance criterion, so
    every quantity of :data:`RESUME_QUANTITIES` is compared at the first resumed
    step and the missing ones are reported as unverified rather than skipped.
    """
    diffs: List[Dict[str, Any]] = []
    unverified: List[str] = []
    for quantity in RESUME_QUANTITIES:
        if quantity not in control or quantity not in resumed:
            unverified.append(quantity)
            continue
        left, right = control[quantity], resumed[quantity]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            tolerance = float(tolerances.get(quantity, 0.0))
            diff = float(right) - float(left)
            within = abs(diff) <= tolerance
            diffs.append({"quantity": quantity, "control": left, "resumed": right, "abs_diff": diff,
                          "tolerance": tolerance, "within": within})
        else:
            diffs.append({"quantity": quantity, "control": left, "resumed": right,
                          "within": left == right, "tolerance": 0.0})
    return {
        "quantities": len(RESUME_QUANTITIES),
        "diffs": diffs,
        "unverified": unverified,
        "continuous": not unverified and all(item["within"] for item in diffs),
    }


def reshard_capability(
    *, checkpoint_format: str, target_world_size: int, supported_targets: Sequence[int]
) -> Dict[str, Any]:
    """Step 35: reshard only when the *format* states it supports the target.

    ``E14-02`` §8.3: 不能把框架概念上的 reshard 能力外推到具体版本/optimizer 格式.
    """
    supported = int(target_world_size) in {int(item) for item in supported_targets}
    return {
        "checkpoint_format": checkpoint_format,
        "target_world_size": target_world_size,
        "supported_targets": sorted(int(item) for item in supported_targets),
        "reshard_supported": supported,
        "state": rec.CAP_AVAILABLE if supported else rec.CAP_UNAVAILABLE_CAPABILITY,
        "reason_code": "" if supported else "SHAPE_UNSUPPORTED",
        "note": "" if supported else "记录 capability 限制，不把不支持的 reshard 当作通过",
    }


def inject_checkpoint_fault(
    inventory: CheckpointInventory, *, fault: str, target: str
) -> Dict[str, Any]:
    """Steps 36–37: missing shard / corrupt metadata / rank failure must fail closed.

    The fixtures are *descriptions*: this function does not touch a real
    checkpoint (``campaign.REQUIRED_ISOLATION`` requires a copied artefact and an
    isolated environment for the actual injection).
    """
    known = {
        "missing_shard": "删除一个测试分片",
        "corrupt_metadata": "篡改 metadata/索引字段",
        "wrong_world_size": "以错误 world size 加载",
        "rank_failure": "可控 hook 终止一个 rank",
    }
    if fault not in known:
        raise ConfigError(f"unknown checkpoint fault {fault!r}; known: {', '.join(sorted(known))}")
    expected = (
        rec.STATUS_FAIL_RECOVERY if fault == "rank_failure" else rec.STATUS_FAIL_CORRECTNESS
    )
    return {
        "fault": fault,
        "target": target,
        "description": known[fault],
        "expected_detection": "load 前 fail-closed，并定位到具体对象",
        "forbidden_outcome": "以随机值/零值补齐、跳过、无限 hang、部分 rank 写出成功 checkpoint",
        "expected_status_on_detection": expected,
        "isolation_required": "E14-02 step 36/37：复制制品 + 隔离环境 + 可控 hook",
        "current_inventory_complete": inventory.verify()["complete"],
    }


# ── records and verdict (steps 23, 40) ─────────────────────────────────────


@dataclass
class TrainingStepRecord:
    """``E14-02`` §8 ``TrainingStepRecord`` (schema ``…e14-02.step.v1``)."""

    training_run_id: str
    step: int
    rank: int
    world_size: int
    sample_ids: Tuple[str, ...] = ()
    effective_token_count: int = 0
    loss_sum: Optional[float] = None
    loss_denominator: Optional[int] = None
    grad_norm: Optional[float] = None
    lr: Optional[float] = None
    scaler: Optional[float] = None
    timing_ms: Mapping[str, float] = field(default_factory=dict)
    collective_bytes: int = 0
    memory_bytes: Mapping[str, int] = field(default_factory=dict)
    parameter_digest_after: str = ""
    checkpoint_id: Optional[str] = None
    status: str = rec.STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e14-02.step.v1"

    def validate(self) -> List[str]:
        findings: List[str] = []
        if not self.training_run_id:
            findings.append("TrainingStepRecord: training_run_id is required")
        if self.step < 0 or self.rank < 0 or self.world_size < 1:
            findings.append("TrainingStepRecord: step/rank must be >= 0 and world_size >= 1")
        if self.rank >= self.world_size:
            findings.append("TrainingStepRecord: rank must be < world_size")
        if self.sample_ids and len(self.sample_ids) != len(set(self.sample_ids)):
            findings.append("TrainingStepRecord: duplicate sample_ids in one step")
        if self.loss_denominator is None and self.loss_sum is not None:
            findings.append(
                "TrainingStepRecord: loss_sum without loss_denominator cannot be reduced globally"
            )
        unknown = sorted(set(self.timing_ms) - set(TIMING_PHASES))
        if unknown:
            findings.append(f"TrainingStepRecord: unknown timing phases {', '.join(unknown)}")
        missing = sorted(set(TIMING_PHASES) - set(self.timing_ms))
        if self.timing_ms and missing:
            findings.append(f"TrainingStepRecord: timing is missing phases {', '.join(missing)}")
        if self.status in rec.CONCLUSION_STATUSES:
            findings.append("TrainingStepRecord: a single step carries no conclusion status")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "training_run_id": self.training_run_id,
            "step": self.step,
            "rank": self.rank,
            "world_size": self.world_size,
            "sample_ids": list(self.sample_ids),
            "effective_token_count": self.effective_token_count,
            "loss_sum": self.loss_sum,
            "loss_denominator": self.loss_denominator,
            "grad_norm": self.grad_norm,
            "lr": self.lr,
            "scaler": self.scaler,
            "timing_ms": {key: self.timing_ms[key] for key in sorted(self.timing_ms)},
            "collective_bytes": self.collective_bytes,
            "memory_bytes": {key: self.memory_bytes[key] for key in sorted(self.memory_bytes)},
            "parameter_digest_after": self.parameter_digest_after,
            "checkpoint_id": self.checkpoint_id,
            "status": self.status,
        }


@dataclass
class DistributedTrainingVerdict:
    """Step 40 — separate rulings, so one axis cannot cover another."""

    verdict_id: str
    decision: str = rec.STATUS_NOT_RUN
    math_equivalence: Mapping[str, Any] = field(default_factory=dict)
    actual_path: Mapping[str, Any] = field(default_factory=dict)
    memory: Mapping[str, Any] = field(default_factory=dict)
    communication: Mapping[str, Any] = field(default_factory=dict)
    checkpoint: Mapping[str, Any] = field(default_factory=dict)
    resume: Mapping[str, Any] = field(default_factory=dict)
    faults: Tuple[Mapping[str, Any], ...] = ()
    limitations: Tuple[str, ...] = ()

    schema_version = f"{SCHEMA_PREFIX}.e14-02.verdict.v1"

    def validate(self) -> List[str]:
        findings: List[str] = []
        if self.decision not in rec.ALL_STATUSES:
            findings.append(f"DistributedTrainingVerdict: unknown decision {self.decision!r}")
        if self.decision == rec.STATUS_PASS:
            required = {
                "math_equivalence": self.math_equivalence.get("all_within") is True,
                "actual_path": bool(self.actual_path.get("actual_implementation")),
                "checkpoint": self.checkpoint.get("complete") is True,
                "resume": self.resume.get("continuous") is True,
            }
            for name, ok in required.items():
                if not ok:
                    findings.append(
                        f"DistributedTrainingVerdict: PASS requires {name}, which is not satisfied "
                        "（loss 下降不能覆盖 checkpoint 或回退失败）"
                    )
            if not self.faults:
                findings.append("DistributedTrainingVerdict: PASS requires at least one fault case")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "verdict_id": self.verdict_id,
            "decision": self.decision,
            "math_equivalence": dict(self.math_equivalence),
            "actual_path": dict(self.actual_path),
            "memory": dict(self.memory),
            "communication": dict(self.communication),
            "checkpoint": dict(self.checkpoint),
            "resume": dict(self.resume),
            "faults": [dict(item) for item in self.faults],
            "limitations": list(self.limitations),
            "digest": canonical_digest(
                {
                    "verdict_id": self.verdict_id,
                    "decision": self.decision,
                    "checkpoint": dict(self.checkpoint),
                    "resume": dict(self.resume),
                }
            ),
        }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the E14-02 interfaces (labelled smoke, not an experiment)."""
    spec = GlobalBatchSpec(
        microbatch=1, accumulation=2, world_size=2,
        sample_ids=("s0", "s1", "s2", "s3"), token_counts=(10, 10, 10, 10),
    )
    loss = reconstruct_global_loss([1.0, 3.0], [10, 10], LossReduction(reduction="token_mean"))
    block = strategy_verdict(
        StrategyChoice(strategy="ddp", world_size=2, rationale="smoke", requested_implementation="ddp",
                       actual_implementation="ddp"),
        distributed_available=False,
    )
    ledger = MemoryLedger(rank=0, phase="step_end", categories={cat: 0 for cat in MEMORY_CATEGORIES})
    sharding = check_data_sharding({0: ("a", "b"), 1: ("c", "d")}, world_size=2, expected_total=4)
    reshard = reshard_capability(checkpoint_format="dcp", target_world_size=4, supported_targets=(2,))
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "global_batch": spec.global_batch,
        "batch_problems": spec.problems(),
        "global_loss": loss["loss"],
        "naive_rank_mean": naive_rank_mean([0.1, 0.3])["loss"],
        "single_device_status": block["status"],
        "memory_problems": ledger.problems(),
        "sharding_ok": sharding["ok"],
        "reshard_unsupported_state": reshard["state"],
        "resume_quantities": len(RESUME_QUANTITIES),
    }


# ── result accessors ───────────────────────────────────────────────────────
#
# Several protocol steps observe a *single field* of a result dict (the first
# divergent layer, the slowest rank, whether the last microbatch syncs).  Naming
# those as functions keeps the step→interface table pointing at real symbols, so
# the mapping cannot rot into a string that resolves to nothing.

def reconciliation_all_within(result: Mapping[str, Any]) -> bool:
    """Step 19: did every reconciled tensor stay inside tolerance?"""
    return bool(result.get("all_within"))


def reconciliation_first_divergence(result: Mapping[str, Any]) -> Optional[str]:
    """Step 20: the first layer that left tolerance (needed to locate a bad update)."""
    return result.get("first_divergence")


def straggler_slowest_rank(result: Mapping[str, Any]) -> Any:
    """Step 28: the rank that decides the synchronous step time."""
    return result.get("slowest_rank")


def accumulation_final_sync(result: Mapping[str, Any]) -> bool:
    """Step 21: the last microbatch must not omit its gradient sync."""
    return bool(result.get("final_microbatch_syncs"))


# ── protocol step table (40 steps of details/S14/E14-02) ───────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结研究范围和主策略", ("training:StrategyChoice", "training:STRATEGIES",
                                 "training:StrategyChoice.problems")),
    (2, "冻结 base ModelArtifact", ("contracts:TrainingRunArtifact.base_model_artifact_id",
                                    "identity:artifact_ref")),
    (3, "冻结数据制品", ("training:GlobalBatchSpec.sample_ids", "identity:file_inventory")),
    (4, "冻结 objective 与 reduction", ("training:LossReduction", "training:reconstruct_global_loss")),
    (5, "冻结 global batch 等价关系", ("training:GlobalBatchSpec", "training:GlobalBatchSpec.global_batch")),
    (6, "冻结 optimizer/scheduler/scaler", ("contracts:CheckpointArtifact.optimizer_identity",
                                            "training:TrainingStepRecord.scaler")),
    (7, "冻结 seed bundle", ("identity:SeedBundle", "identity:SEED_ROLES", "identity:SeedBundle.derive")),
    (8, "冻结 precision contract", ("training:PRECISION_COMPONENTS", "training:validate_precision_contract")),
    (9, "冻结 topology 和设备映射", ("identity:RankIdentity", "identity:topology_digest")),
    (10, "运行 capability probe", ("training:strategy_verdict", "records:CAP_DEVICE_UNAVAILABLE")),
    (11, "构建未并行单卡 reference", ("training:reconstruct_global_loss", "training:tolerance_gate")),
    (12, "验证 reference 的可重复性", ("training:LossReduction", "training:tolerance_gate")),
    (13, "启动分布式进程组", ("identity:RankIdentity.process_group", "training:StrategyChoice.world_size")),
    (14, "执行 rank/device 一致性握手", ("identity:assert_rank_identities", "identity:RankIdentity.device_id")),
    (15, "验证数据分片", ("training:check_data_sharding", "training:GlobalBatchSpec.problems")),
    (16, "验证初始参数一致性", ("training:reconcile_tensors", "training:tolerance_gate")),
    (17, "运行分布式高精度一步", ("training:reconstruct_global_loss", "training:TrainingStepRecord")),
    (18, "对账 loss", ("training:reconstruct_global_loss", "training:naive_rank_mean")),
    (19, "对账 gradients", ("training:reconcile_tensors", "training:reconciliation_all_within")),
    (20, "对账参数更新", ("training:reconciliation_first_divergence", "training:tolerance_gate")),
    (21, "验证 accumulation no-sync 语义", ("training:accumulation_sync_plan",
                                             "training:accumulation_final_sync")),
    (22, "运行目标 mixed precision", ("training:validate_precision_contract", "records:REASON_CODES")),
    (23, "运行短程 loss 轨迹", ("training:TrainingStepRecord", "training:TrainingStepRecord.validate")),
    (24, "测每阶段时间", ("training:TIMING_PHASES", "training:TrainingStepRecord.timing_ms")),
    (25, "建立 collective 账本", ("training:CollectiveLedgerEntry", "records:TABLE_SCHEMAS")),
    (26, "建立每 rank 显存账本", ("training:MemoryLedger", "training:MEMORY_CATEGORIES")),
    (27, "归因理论与实测显存差", ("training:attribute_memory_gap", "training:MemoryLedger.categories")),
    (28, "检测 straggler 和 rank skew", ("training:detect_straggler", "training:straggler_slowest_rank")),
    (29, "保存完整 checkpoint", ("contracts:CheckpointArtifact", "training:CheckpointInventory")),
    (30, "验证 checkpoint inventory/hash", ("training:CheckpointInventory.verify",
                                             "identity:content_address_aggregate")),
    (31, "执行同 topology load-only", ("training:CheckpointInventory.topology_digest",
                                        "contracts:CheckpointArtifact.missing_state")),
    (32, "执行 uninterrupted control", ("training:compare_resume_continuity", "training:RESUME_QUANTITIES")),
    (33, "执行 stop→resume", ("training:compare_resume_continuity", "training:TrainingStepRecord.step")),
    (34, "比较恢复连续性", ("training:compare_resume_continuity", "training:RESUME_QUANTITIES")),
    (35, "测试 reshard 能力", ("training:reshard_capability", "records:CAP_UNAVAILABLE_CAPABILITY")),
    (36, "注入缺 shard/损坏 metadata", ("training:inject_checkpoint_fault", "training:CheckpointInventory.verify",
                                        "campaign:isolation_clause")),
    (37, "注入 rank failure", ("training:inject_checkpoint_fault", "records:STATUS_FAIL_RECOVERY")),
    (38, "测 checkpoint save/load 开销", ("training:CheckpointInventory.save_mode",
                                           "training:CheckpointInventory.pause_ms")),
    (39, "跨独立运行重复", ("training:DistributedTrainingVerdict.limitations", "records:EXPERIMENT_UNITS")),
    (40, "形成 DistributedTrainingVerdict", ("training:DistributedTrainingVerdict",
                                              "training:DistributedTrainingVerdict.validate")),
)
