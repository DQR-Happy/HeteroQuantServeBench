"""Communication–computation overlap: DAG, schedule, interval algebra (E10-06).

The single most abused claim in this area is "``async_op=True`` therefore we
overlap".  This module makes that claim structurally hard:

* overlap is computed from **interval sets** (union ∩ union), never from summed
  per-kernel durations, and the same interval is never counted twice;
* when the clock uncertainty is the same order as the candidate difference the
  verdict degrades to ``INCONCLUSIVE`` (details E10-06 §12);
* a schedule is legal only if independent compute exists — a fake overlap built
  by reordering dependent work is refused at the DAG level;
* the blocking / async-no-overlap / overlap arms may differ in exactly one
  place, and the identity hash proves it.

The causal verdict matrix of §11 is data here, so a report can be audited
against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: The three schedule arms (details E10-06 step 4).
SCHEDULE_KINDS: Tuple[str, ...] = ("blocking_baseline", "async_no_overlap", "overlap")

#: Chunk candidates that are always considered (details E10-06 step 5).
CHUNK_STRATEGIES: Tuple[str, ...] = (
    "whole_message",
    "few_large_chunks",
    "natural_module_boundaries",
    "smaller_candidates",
)

#: The causal verdict matrix of details E10-06 §11.
CAUSAL_MATRIX: Tuple[Mapping[str, str], ...] = (
    {
        "observation": "API returns early, timeline shows no intersection",
        "interpretation": "asynchronous submission only",
        "verdict": "must not be called overlap",
    },
    {
        "observation": "timeline intersects but exposed comm does not drop",
        "interpretation": "the overlap is off the critical path or the comm got slower",
        "verdict": "no end-to-end causality",
    },
    {
        "observation": "exposed comm drops but wait rises elsewhere",
        "interpretation": "dependency shifted",
        "verdict": "net gain is zero",
    },
    {
        "observation": "overlap rises while both compute and comm slow down",
        "interpretation": "resource contention",
        "verdict": "decide on net E2E",
    },
    {
        "observation": "smaller chunks start comm earlier but add events/α",
        "interpretation": "granularity trade-off",
        "verdict": "pick the knee, not the maximum ratio",
    },
    {
        "observation": "B is correct and E2E improves with the near cause in the same direction",
        "interpretation": "causal candidate",
        "verdict": "enter confirmation",
    },
    {
        "observation": "tuning improves, holdout regresses",
        "interpretation": "shape overfit",
        "verdict": "the policy must not be published",
    },
)

#: Counterfactual checks that must accompany a positive claim (§11).
COUNTERFACTUALS: Tuple[str, ...] = (
    "replacing the independent compute with a no-op must remove the claimed gain",
    "replacing communication with a same-duration dependency-free placeholder only validates "
    "the scheduler and is reported separately as a test double",
)

#: Guardrails a positive overlap verdict must respect (details E10-06 §5).
OVERLAP_GUARDRAILS: Tuple[str, ...] = (
    "tensor/logit/token/KV correctness",
    "no race/deadlock/timeout",
    "collective seq/order preserved",
    "P99 regression within the preregistered bound",
    "no memory/workspace/event/stream leak",
    "compute/comm singles do not collapse from contention",
    "non-target workloads do not regress",
)


def _require(condition: bool, message: str, *, field_name: str = "") -> None:
    if not condition:
        raise ConfigError(message, details={"field": field_name} if field_name else None)


# ── interval algebra ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Interval:
    """A half-open time interval in one clock domain."""

    start_ns: int
    end_ns: int
    kind: str = ""
    label: str = ""
    rank: int = -1
    stream: str = ""

    def __post_init__(self) -> None:
        if self.end_ns < self.start_ns:
            raise ConfigError(
                f"negative duration interval {self.label!r}", details={"field": "end_ns"}
            )

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    def as_dict(self) -> Dict[str, Any]:
        return {
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.duration_ns,
            "kind": self.kind,
            "label": self.label,
            "rank": self.rank,
            "stream": self.stream,
        }


def merge_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    """Union of intervals; overlapping kernels are merged, never summed twice."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: (item.start_ns, item.end_ns))
    merged: List[Interval] = [ordered[0]]
    for item in ordered[1:]:
        last = merged[-1]
        if item.start_ns <= last.end_ns:
            merged[-1] = Interval(
                start_ns=last.start_ns,
                end_ns=max(last.end_ns, item.end_ns),
                kind=last.kind,
                label=last.label,
                rank=last.rank,
                stream=last.stream,
            )
        else:
            merged.append(item)
    return merged


def union_measure_ns(intervals: Sequence[Interval]) -> int:
    return sum(item.duration_ns for item in merge_intervals(intervals))


def intersection_measure_ns(left: Sequence[Interval], right: Sequence[Interval]) -> int:
    """Measure of ``union(left) ∩ union(right)`` (details E10-06 §12)."""
    left_merged = merge_intervals(left)
    right_merged = merge_intervals(right)
    total = 0
    i = j = 0
    while i < len(left_merged) and j < len(right_merged):
        a, b = left_merged[i], right_merged[j]
        start = max(a.start_ns, b.start_ns)
        end = min(a.end_ns, b.end_ns)
        if end > start:
            total += end - start
        if a.end_ns < b.end_ns:
            i += 1
        else:
            j += 1
    return total


def wall_covered_ns(left: Sequence[Interval], right: Sequence[Interval]) -> int:
    return union_measure_ns(list(left) + list(right))


@dataclass(frozen=True)
class OverlapMetrics:
    """Interval-based overlap metrics (details README §12)."""

    compute_active_ms: float
    comm_active_ms: float
    overlap_ms: float
    exposed_comm_ms: float
    overlap_fraction_comm: Optional[float]
    net_gain_ms: Optional[float]
    clock_uncertainty_ms: float = 0.0
    wall_covered_ms: float = 0.0

    def __post_init__(self) -> None:
        for name in ("compute_active_ms", "comm_active_ms", "overlap_ms", "exposed_comm_ms"):
            if float(getattr(self, name)) < 0:
                raise ConfigError(f"{name} must be >= 0", details={"field": name})
        if self.overlap_ms > min(self.compute_active_ms, self.comm_active_ms) + 1e-6:
            raise ConfigError(
                "overlap cannot exceed min(compute_active, comm_active)",
                details={"field": "overlap_ms"},
            )

    @property
    def conclusive(self) -> bool:
        """A difference on the order of the clock uncertainty is not a result."""
        if self.clock_uncertainty_ms <= 0:
            return True
        return abs(self.exposed_comm_ms) > self.clock_uncertainty_ms and (
            self.net_gain_ms is None or abs(self.net_gain_ms) > self.clock_uncertainty_ms
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "compute_active_ms": self.compute_active_ms,
            "comm_active_ms": self.comm_active_ms,
            "overlap_ms": self.overlap_ms,
            "overlap_fraction_comm": self.overlap_fraction_comm,
            "exposed_comm_ms": self.exposed_comm_ms,
            "net_gain_ms": self.net_gain_ms,
            "wall_covered_ms": self.wall_covered_ms,
            "clock_uncertainty_ms": self.clock_uncertainty_ms,
            "conclusive": self.conclusive,
            "note": (
                "publish overlap_ms, overlap_fraction_comm, exposed_comm_ms and net_gain_ms "
                "together; a bare percentage exaggerates small transfers"
            ),
        }


def overlap_metrics_from_intervals(
    compute: Sequence[Interval],
    comm: Sequence[Interval],
    *,
    no_overlap_total_ms: Optional[float] = None,
    overlap_total_ms: Optional[float] = None,
    clock_uncertainty_ms: float = 0.0,
) -> OverlapMetrics:
    """Compute the metrics from raw intervals in one (calibrated) clock domain."""
    compute_ns = union_measure_ns(compute)
    comm_ns = union_measure_ns(comm)
    overlap_ns = intersection_measure_ns(compute, comm)
    exposed_ns = comm_ns - overlap_ns
    fraction = overlap_ns / comm_ns if comm_ns else None
    net_gain = None
    if no_overlap_total_ms is not None and overlap_total_ms is not None:
        net_gain = no_overlap_total_ms - overlap_total_ms
    return OverlapMetrics(
        compute_active_ms=compute_ns / 1e6,
        comm_active_ms=comm_ns / 1e6,
        overlap_ms=overlap_ns / 1e6,
        exposed_comm_ms=exposed_ns / 1e6,
        overlap_fraction_comm=fraction,
        net_gain_ms=net_gain,
        clock_uncertainty_ms=clock_uncertainty_ms,
        wall_covered_ms=wall_covered_ns(compute, comm) / 1e6,
    )


# ── dependency DAG ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DagNode:
    """One tensor role with its producer/consumer and the collective that carries it."""

    tensor_role: str
    producer: str
    consumers: Tuple[str, ...]
    collective_seq: Optional[int] = None
    independent_compute: Tuple[str, ...] = ()
    kv_dependency: bool = False

    def __post_init__(self) -> None:
        _require(bool(self.tensor_role), "a DAG node needs a tensor role", field_name="tensor_role")
        _require(bool(self.producer), "a DAG node needs a producer", field_name="producer")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tensor_role": self.tensor_role,
            "producer": self.producer,
            "consumers": list(self.consumers),
            "collective_seq": self.collective_seq,
            "independent_compute": list(self.independent_compute),
            "kv_dependency": self.kv_dependency,
        }


@dataclass
class TensorDependencyDAG:
    nodes: Tuple[DagNode, ...]
    edges: Tuple[Tuple[str, str], ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        roles = [node.tensor_role for node in self.nodes]
        if len(set(roles)) != len(roles):
            problems.append("duplicate tensor roles in the DAG")
        known = set(roles)
        for src, dst in self.edges:
            if src not in known or dst not in known:
                problems.append(f"edge {src}->{dst} references an unknown tensor role")
        for node in self.nodes:
            for consumer in node.consumers:
                if consumer not in known:
                    problems.append(f"{node.tensor_role} lists unknown consumer {consumer}")
        return problems

    def independent_window(self, collective_seq: int) -> Tuple[str, ...]:
        for node in self.nodes:
            if node.collective_seq == collective_seq:
                return node.independent_compute
        return ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [node.as_dict() for node in self.nodes],
            "edges": [list(edge) for edge in self.edges],
        }


@dataclass(frozen=True)
class OverlapBound:
    """The legal overlap window and its Amdahl upper bound (details E10-06 step 3)."""

    collective_seq: int
    independent_compute_ms: float
    comm_ms: float
    theoretical_max_overlap_ms: float
    feasible: bool
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "collective_seq": self.collective_seq,
            "independent_compute_ms": self.independent_compute_ms,
            "comm_ms": self.comm_ms,
            "theoretical_max_overlap_ms": self.theoretical_max_overlap_ms,
            "feasible": self.feasible,
            "reason": self.reason,
        }


def legal_overlap_window(
    dag: TensorDependencyDAG,
    *,
    collective_seq: int,
    comm_ms: float,
    compute_window_ms: Mapping[str, float],
) -> OverlapBound:
    """Refuse an overlap candidate with no independent compute to hide behind."""
    if comm_ms <= 0:
        raise ConfigError("comm_ms must be positive", details={"field": "comm_ms"})
    independent = dag.independent_window(collective_seq)
    available = sum(compute_window_ms.get(name, 0.0) for name in independent)
    feasible = bool(independent) and available > 0
    return OverlapBound(
        collective_seq=collective_seq,
        independent_compute_ms=available,
        comm_ms=comm_ms,
        theoretical_max_overlap_ms=min(available, comm_ms) if feasible else 0.0,
        feasible=feasible,
        reason=(
            ""
            if feasible
            else "no independent compute is available for this collective; reordering dependent "
            "work to fake an overlap is not allowed"
        ),
    )


# ── schedules ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScheduleIdentity:
    """The three arms differ in exactly one place (details E10-06 step 4)."""

    kind: str
    source_hash: str
    changed_component: str
    requested_path: str
    actual_path: str
    fallback_reason: str = ""

    def __post_init__(self) -> None:
        _require(self.kind in SCHEDULE_KINDS, "unknown schedule kind", field_name="kind")
        _require(bool(self.changed_component), "name the single changed component", field_name="changed_component")
        if self.actual_path != self.requested_path:
            _require(
                self.fallback_reason,
                "an actual schedule different from the requested one needs a fallback reason",
                field_name="fallback_reason",
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "source_hash": self.source_hash,
            "changed_component": self.changed_component,
            "requested_path": self.requested_path,
            "actual_path": self.actual_path,
            "fallback_reason": self.fallback_reason,
        }


def validate_only_variable(arms: Sequence[ScheduleIdentity]) -> Dict[str, Any]:
    """All non-schedule identity fields must be equal across the arms."""
    if len(arms) < 2:
        raise ConfigError("at least two arms are needed for an A/B")
    components = {arm.changed_component for arm in arms}
    source_hashes = {arm.source_hash for arm in arms}
    problems: List[str] = []
    if len(source_hashes) > 1:
        problems.append("the arms were built from different source bases")
    if len(components) > 1:
        problems.append(f"the arms changed different components: {sorted(components)}")
    return {
        "ok": not problems,
        "problems": problems,
        "arms": [arm.as_dict() for arm in arms],
        "unique_variable": next(iter(components)) if components else "",
    }


@dataclass(frozen=True)
class ChunkCandidate:
    """One chunking candidate; unsupported ones stay visible with a reason."""

    strategy: str
    chunk_bytes: int
    chunk_count: int
    filtered_reason: str = ""

    def __post_init__(self) -> None:
        _require(self.strategy in CHUNK_STRATEGIES, "unknown chunk strategy", field_name="strategy")
        _require(self.chunk_bytes > 0 and self.chunk_count >= 1, "chunk size/count must be positive")

    @property
    def supported(self) -> bool:
        return not self.filtered_reason

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "chunk_bytes": self.chunk_bytes,
            "chunk_count": self.chunk_count,
            "supported": self.supported,
            "filtered_reason": self.filtered_reason,
        }


def chunk_candidates(
    *,
    message_bytes: int,
    natural_boundaries: Sequence[int] = (),
    memory_limit_bytes: int = 0,
    alignment_bytes: int = 16,
) -> List[ChunkCandidate]:
    """Whole message, a few large chunks, natural boundaries and smaller ones."""
    if message_bytes <= 0:
        raise ConfigError("message_bytes must be positive", details={"field": "message_bytes"})
    candidates: List[ChunkCandidate] = [
        ChunkCandidate("whole_message", message_bytes, 1),
        ChunkCandidate("few_large_chunks", message_bytes // 2, 2),
    ]
    for boundary in natural_boundaries:
        if 0 < boundary < message_bytes:
            count = message_bytes // boundary + (1 if message_bytes % boundary else 0)
            candidates.append(
                ChunkCandidate("natural_module_boundaries", boundary, count)
            )
    for divisor in (4, 8):
        size = message_bytes // divisor
        if size >= alignment_bytes:
            count = message_bytes // size + (1 if message_bytes % size else 0)
            candidates.append(ChunkCandidate("smaller_candidates", size, count))
    filtered: List[ChunkCandidate] = []
    for candidate in candidates:
        reason = ""
        if candidate.chunk_bytes % alignment_bytes:
            reason = (
                f"chunk size {candidate.chunk_bytes} is not a multiple of the alignment "
                f"{alignment_bytes}"
            )
        if memory_limit_bytes and candidate.chunk_bytes > memory_limit_bytes:
            reason = (
                f"chunk {candidate.chunk_bytes} exceeds the memory limit {memory_limit_bytes}"
            )
        filtered.append(
            ChunkCandidate(
                strategy=candidate.strategy,
                chunk_bytes=candidate.chunk_bytes,
                chunk_count=candidate.chunk_count,
                filtered_reason=reason,
            )
        )
    return filtered


@dataclass(frozen=True)
class StreamPolicy:
    """Compute/comm streams, priorities and the group ordering rule (step 6)."""

    compute_stream: str
    comm_stream: str
    priority: str = "default"
    group_ids: Tuple[str, ...] = ()
    collective_sequence_rule: str = "same order on every rank within a group"
    max_streams: int = 4

    def __post_init__(self) -> None:
        _require(self.compute_stream != self.comm_stream, "compute and comm streams must differ")
        _require(0 < self.max_streams <= 8, "keep the stream count bounded (1..8)")
        _require(
            self.priority in ("default", "high"),
            "priority must be default|high",
            field_name="priority",
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "compute_stream": self.compute_stream,
            "comm_stream": self.comm_stream,
            "priority": self.priority,
            "group_ids": list(self.group_ids),
            "collective_sequence_rule": self.collective_sequence_rule,
            "max_streams": self.max_streams,
        }


def audit_stream_policy(policy: StreamPolicy, *, created_streams: Sequence[str]) -> Dict[str, Any]:
    """Streams are declared up front; no implicit unbounded creation (step 6)."""
    problems: List[str] = []
    if len(created_streams) > policy.max_streams:
        problems.append(
            f"{len(created_streams)} streams were created but the policy allows "
            f"{policy.max_streams}"
        )
    undeclared = sorted(set(created_streams) - {policy.compute_stream, policy.comm_stream})
    if undeclared:
        problems.append(f"undeclared streams were created: {undeclared}")
    return {"ok": not problems, "problems": problems}


@dataclass(frozen=True)
class ReadinessEvent:
    tensor_role: str
    chunk_index: int
    producer_stream: str
    recorded_ns: int
    device_context_ok: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tensor_role": self.tensor_role,
            "chunk_index": self.chunk_index,
            "producer_stream": self.producer_stream,
            "recorded_ns": self.recorded_ns,
            "device_context_ok": self.device_context_ok,
        }


@dataclass(frozen=True)
class CompletionEvent:
    tensor_role: str
    chunk_index: int
    comm_stream: str
    recorded_ns: int
    handle_valid_until_ns: int


def validate_work_lifetime(
    *, completion: CompletionEvent, consumer_wait_ns: int
) -> Dict[str, Any]:
    """The work/event must stay valid until the consumer has waited (step 8)."""
    problems: List[str] = []
    if consumer_wait_ns < completion.recorded_ns:
        problems.append("the consumer waited before the completion event was recorded")
    if consumer_wait_ns > completion.handle_valid_until_ns:
        problems.append("the work handle expired before the consumer waited on it")
    return {"ok": not problems, "problems": problems}


@dataclass(frozen=True)
class ScheduleTraceEntry:
    """One planned vs actual schedule entry (details E10-06 step 9)."""

    chunk_index: int
    tensor_range: Tuple[int, int]
    producer: str
    comm: str
    consumer: str
    stream: str
    event: str
    work_handle: str
    collective_seq: int
    start_ns: int
    end_ns: int
    rank: int = -1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "tensor_range": list(self.tensor_range),
            "producer": self.producer,
            "comm": self.comm,
            "consumer": self.consumer,
            "stream": self.stream,
            "event": self.event,
            "work_handle": self.work_handle,
            "collective_seq": self.collective_seq,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "rank": self.rank,
        }


def compare_actual_vs_planned(
    planned: Sequence[ScheduleTraceEntry], actual: Sequence[ScheduleTraceEntry]
) -> Dict[str, Any]:
    """Prove the executed schedule matches the plan (step 9)."""
    planned_keys = {
        (entry.chunk_index, entry.collective_seq): entry for entry in planned
    }
    actual_keys = {(entry.chunk_index, entry.collective_seq): entry for entry in actual}
    missing = sorted(set(planned_keys) - set(actual_keys))
    extra = sorted(set(actual_keys) - set(planned_keys))
    mismatched: List[Dict[str, Any]] = []
    for key in sorted(set(planned_keys) & set(actual_keys)):
        left, right = planned_keys[key], actual_keys[key]
        if (left.stream, left.event, left.consumer) != (right.stream, right.event, right.consumer):
            mismatched.append(
                {
                    "chunk_index": key[0],
                    "collective_seq": key[1],
                    "planned": [left.stream, left.event, left.consumer],
                    "actual": [right.stream, right.event, right.consumer],
                }
            )
    return {
        "ok": not missing and not extra and not mismatched,
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
    }


# ── correctness probes ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class RaceProbePlan:
    repeats: int = 32
    delay_ladder_ns: Tuple[int, ...] = (0, 2000, 20000, 200000)
    reverse_start_order: bool = True

    def __post_init__(self) -> None:
        _require(self.repeats >= 16, "a race probe needs many repeats", field_name="repeats")
        _require(bool(self.delay_ladder_ns), "a race probe needs a delay ladder")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "repeats": self.repeats,
            "delay_ladder_ns": list(self.delay_ladder_ns),
            "reverse_start_order": self.reverse_start_order,
        }


def evaluate_race_probe(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Any non-deterministic error must be fixed before performance is measured."""
    failures = [
        row for row in results if not bool(row.get("deterministic", False)) or bool(row.get("error"))
    ]
    return {
        "ok": not failures,
        "deterministic": not failures,
        "failures": failures,
        "note": "any accidental wrong token invalidates the performance claim",
    }


def global_sync_audit(
    *,
    unrelated_stream_events: Sequence[Mapping[str, Any]],
    device_sync_calls: int,
    necessary_tensor_waits: int,
) -> Dict[str, Any]:
    """Distinguish a necessary consumer wait from a device-wide sync (step 15)."""
    blocked = [
        row for row in unrelated_stream_events if bool(row.get("blocked_by_comm", False))
    ]
    problems: List[str] = []
    if blocked:
        problems.append(
            f"{len(blocked)} events on unrelated streams were blocked (implicit global sync)"
        )
    if device_sync_calls > len(unrelated_stream_events) and device_sync_calls > necessary_tensor_waits:
        problems.append(
            f"{device_sync_calls} device-wide syncs exceed the necessary tensor waits "
            f"({necessary_tensor_waits})"
        )
    return {
        "ok": not problems,
        "problems": problems,
        "blocked_events": blocked,
        "note": "global syncs used to 'fix' a race destroy the overlap claim",
    }


# ── cost / contention ──────────────────────────────────────────────────────


def chunk_overhead(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Extra collectives/events/launches/metadata/small-kernel loss (step 24)."""
    return {
        "extra_collectives": sum(int(row.get("extra_collectives", 0)) for row in rows),
        "extra_events": sum(int(row.get("extra_events", 0)) for row in rows),
        "extra_launches": sum(int(row.get("extra_launches", 0)) for row in rows),
        "metadata_bytes": sum(int(row.get("metadata_bytes", 0)) for row in rows),
        "small_kernel_loss_ms": sum(float(row.get("small_kernel_loss_ms", 0.0)) for row in rows),
        "note": "overhead is subtracted from the coverage gain; the knee is the answer",
    }


def contention_report(
    *,
    solo_compute_ms: float,
    solo_comm_ms: float,
    concurrent_compute_ms: float,
    concurrent_comm_ms: float,
) -> Dict[str, Any]:
    """Overlap can slow both sides; only the net E2E decides (step 23)."""
    for name, value in (
        ("solo_compute_ms", solo_compute_ms),
        ("solo_comm_ms", solo_comm_ms),
        ("concurrent_compute_ms", concurrent_compute_ms),
        ("concurrent_comm_ms", concurrent_comm_ms),
    ):
        if value < 0:
            raise ConfigError(f"{name} must be >= 0", details={"field": name})
    compute_slowdown = (
        concurrent_compute_ms / solo_compute_ms if solo_compute_ms else None
    )
    comm_slowdown = concurrent_comm_ms / solo_comm_ms if solo_comm_ms else None
    return {
        "compute_slowdown": compute_slowdown,
        "comm_slowdown": comm_slowdown,
        "both_slow": bool(
            compute_slowdown and comm_slowdown and compute_slowdown > 1.0 and comm_slowdown > 1.0
        ),
        "note": "a high overlap fraction with both sides slower is a contention finding, not a win",
    }


def compare_to_bound(*, net_gain_ms: float, bound: OverlapBound, ready_late_ms: float = 0.0) -> Dict[str, Any]:
    """Explain the gap between measured net gain and the theoretical bound (step 25)."""
    if bound.theoretical_max_overlap_ms <= 0:
        return {
            "ok": False,
            "reason": "the bound is zero: there is no legal independent compute window",
        }
    shortfall = bound.theoretical_max_overlap_ms - net_gain_ms
    return {
        "theoretical_max_ms": bound.theoretical_max_overlap_ms,
        "net_gain_ms": net_gain_ms,
        "shortfall_ms": shortfall,
        "ready_late_ms": ready_late_ms,
        "explained_by_ready_late": ready_late_ms >= shortfall * 0.5,
        "note": "a shortfall must be attributed to late readiness, contention, events or waits",
    }


# ── policy / verdict ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PhaseShapePolicy:
    """Phase/shape-aware schedule policy with an explicit safe fallback (step 29)."""

    phase: str
    shape_class: str
    schedule: str
    fallback_schedule: str = "blocking_baseline"
    requested_schedule: str = ""
    actual_schedule: str = ""
    fallback_reason: str = ""

    def __post_init__(self) -> None:
        _require(self.phase in ("prefill", "decode"), "phase must be prefill|decode", field_name="phase")
        _require(self.schedule in SCHEDULE_KINDS, "unknown schedule", field_name="schedule")
        if self.actual_schedule and self.actual_schedule != self.schedule:
            _require(
                self.fallback_reason,
                "an actual schedule different from the policy needs a fallback reason",
                field_name="fallback_reason",
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "shape_class": self.shape_class,
            "schedule": self.schedule,
            "fallback_schedule": self.fallback_schedule,
            "requested_schedule": self.requested_schedule or self.schedule,
            "actual_schedule": self.actual_schedule or self.schedule,
            "fallback_reason": self.fallback_reason,
        }


def phase_policy(
    *,
    prefill_verdict: str,
    decode_verdict: str,
    shape_class: str = "default",
) -> List[PhaseShapePolicy]:
    """A benefit in one phase never generalises to the other (step 26)."""

    def schedule_for(verdict: str) -> str:
        if verdict == "PASS_POSITIVE":
            return "overlap"
        if verdict in ("PASS_NEGATIVE", "INCONCLUSIVE"):
            return "async_no_overlap"
        return "blocking_baseline"

    return [
        PhaseShapePolicy(phase="prefill", shape_class=shape_class, schedule=schedule_for(prefill_verdict)),
        PhaseShapePolicy(phase="decode", shape_class=shape_class, schedule=schedule_for(decode_verdict)),
    ]


@dataclass(frozen=True)
class AbbaConfirmation:
    order: Tuple[str, ...]
    paired_effect_ms: float
    ci_low_ms: float
    ci_high_ms: float
    runs_per_arm: int
    pre_registered: bool = True

    def __post_init__(self) -> None:
        _require(self.runs_per_arm >= 3, "ABBA confirmation needs at least three runs per arm")
        _require(
            set(self.order) == {"A", "B"},
            "the order must be an ABBA/randomised block over A and B",
            field_name="order",
        )

    @property
    def significant(self) -> bool:
        return self.ci_low_ms > 0 or self.ci_high_ms < 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "order": list(self.order),
            "paired_effect_ms": self.paired_effect_ms,
            "ci_low_ms": self.ci_low_ms,
            "ci_high_ms": self.ci_high_ms,
            "runs_per_arm": self.runs_per_arm,
            "pre_registered": self.pre_registered,
            "significant": self.significant,
        }


def abba_confirmation(
    *, paired_effect_ms: float, ci_low_ms: float, ci_high_ms: float, runs_per_arm: int
) -> AbbaConfirmation:
    order = ("A", "B", "B", "A") if runs_per_arm % 2 == 0 else ("A", "B", "A", "B", "A")
    return AbbaConfirmation(
        order=order,
        paired_effect_ms=paired_effect_ms,
        ci_low_ms=ci_low_ms,
        ci_high_ms=ci_high_ms,
        runs_per_arm=runs_per_arm,
    )


def overlap_verdict(
    *,
    correctness_ok: bool,
    no_race: bool,
    timeline_proves_overlap: bool,
    exposed_comm_reduced: bool,
    e2e_improved: bool,
    confirmation_significant: bool,
    contention_explained: bool = True,
    conclusive: bool = True,
) -> Dict[str, Any]:
    """The §11 causal chain: path → near cause → critical path → end-to-end."""
    if not correctness_ok or not no_race:
        return {"status": "FAIL", "reason": "correctness/race gate failed; any gain is void"}
    if not conclusive:
        return {
            "status": "INCONCLUSIVE",
            "reason": "the difference is within the clock uncertainty",
        }
    if not timeline_proves_overlap:
        return {
            "status": "FAIL",
            "reason": "the timeline does not show a real intersection; async submission is not overlap",
        }
    if not exposed_comm_reduced:
        return {
            "status": "PASS_NEGATIVE",
            "reason": "overlap exists but does not reduce exposed communication on the critical path",
        }
    if not e2e_improved:
        return {
            "status": "PASS_NEGATIVE",
            "reason": "exposed comm fell but the end-to-end time did not; dependency/wait moved elsewhere",
        }
    if not contention_explained:
        return {
            "status": "INCONCLUSIVE",
            "reason": "resource contention was not quantified",
        }
    if not confirmation_significant:
        return {
            "status": "INCONCLUSIVE",
            "reason": "the independent ABBA confirmation did not reproduce the effect",
        }
    return {
        "status": "PASS_POSITIVE",
        "reason": "near cause → timeline → end-to-end chain holds and confirmation reproduced it",
    }


__all__ = [
    "AbbaConfirmation",
    "CAUSAL_MATRIX",
    "CHUNK_STRATEGIES",
    "COUNTERFACTUALS",
    "ChunkCandidate",
    "CompletionEvent",
    "DagNode",
    "Interval",
    "OVERLAP_GUARDRAILS",
    "OverlapBound",
    "OverlapMetrics",
    "PhaseShapePolicy",
    "RaceProbePlan",
    "SCHEDULE_KINDS",
    "ScheduleIdentity",
    "ScheduleTraceEntry",
    "StreamPolicy",
    "TensorDependencyDAG",
    "abba_confirmation",
    "audit_stream_policy",
    "chunk_candidates",
    "chunk_overhead",
    "compare_actual_vs_planned",
    "compare_to_bound",
    "contention_report",
    "evaluate_race_probe",
    "global_sync_audit",
    "intersection_measure_ns",
    "legal_overlap_window",
    "merge_intervals",
    "overlap_metrics_from_intervals",
    "overlap_verdict",
    "phase_policy",
    "union_measure_ns",
    "validate_only_variable",
    "validate_work_lifetime",
    "wall_covered_ns",
]
