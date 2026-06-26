"""Collective sequence, parameter consistency, timeout and cleanup (E10-03).

The experiment's core claim is *bounded failure*: a legitimate collective
sequence must complete with identical results on every rank, and any mismatch
(op/count/dtype/root/order/group/skipped/delayed) must fail within a declared
bound with the first divergence identified by group/seq/rank/field — never by an
infinite hang and never by a silent wrong answer.

Structural anti-cheat decisions live here:

* the metadata handshake is explicitly *not* a replacement for backend
  timeouts and is never a global barrier in the performance path;
* a fatal communicator rejects every later call with a stable ``COMM_ABORTED``;
* the sequence namespace is tied to ``(group_id, rank_epoch)`` so a rebuilt
  communicator cannot reuse an old sequence or work handle;
* dtype equality is checked as *semantics*, not as byte count.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.distributed.faults import DrillResult, FaultOracle, TimeMetrics
from hqsb.distributed.ranks import COMMUNICATOR_STATES, GroupMembership

#: Allowed communicator transitions (details E10-03 §3.1).
ALLOWED_TRANSITIONS: Mapping[str, Tuple[str, ...]] = {
    "INIT": ("READY", "ERROR", "ABORTING"),
    "READY": ("ENQUEUED", "ERROR", "ABORTING", "DESTROYED"),
    "ENQUEUED": ("IN_FLIGHT", "ERROR", "ABORTING"),
    "IN_FLIGHT": ("COMPLETED", "ERROR", "ABORTING", "TIMEOUT"),
    "COMPLETED": ("READY", "ENQUEUED", "ERROR", "ABORTING", "DESTROYED"),
    "ERROR": ("ABORTING",),
    "ABORTING": ("ABORTED",),
    "ABORTED": ("DESTROYED",),
    "DESTROYED": (),
}

#: Stable normalized error codes the wrapper may return.
WRAPPER_ERROR_CODES: Tuple[str, ...] = (
    "COLLECTIVE_MISMATCH",
    "COLLECTIVE_TIMEOUT",
    "RANK_MISSING",
    "COMM_ABORTED",
)

#: Fields compared by the metadata preflight (details E10-03 §4).
METADATA_FIELDS: Tuple[str, ...] = (
    "group_id",
    "ordered_group_hash",
    "collective_seq",
    "op",
    "root",
    "reduce_op",
    "logical_count",
    "dtype",
    "input_shape_hash",
    "output_shape_hash",
    "inplace",
    "stream_id",
    "callsite",
)

#: Fields whose disagreement is *semantic* even when byte counts match.
SEMANTIC_CRITICAL_FIELDS: Tuple[str, ...] = ("op", "dtype", "root", "reduce_op", "logical_count")

#: The legal golden sequence of details E10-03 step 10.
GOLDEN_SEQUENCE: Tuple[str, ...] = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "all_to_all",
)

#: Timeout families: ordinary and profiler/fault runs never share thresholds.
TIMEOUT_FAMILIES: Tuple[str, ...] = ("ordinary", "fault", "profiler")

#: Injection descriptors of the collective fault matrix (details E10-03 §5).
COLLECTIVE_FAULTS: Tuple[Mapping[str, str], ...] = (
    {"fault": "rank_order", "injection": "ranks swap the op order", "expected": "preflight or bounded timeout"},
    {"fault": "op_mismatch", "injection": "AR vs AG at the same seq", "expected": "identify seq/op/ranks"},
    {"fault": "count_shape", "injection": "one rank changes the element count", "expected": "preflight/parameter validation rejects"},
    {"fault": "dtype", "injection": "FP16 vs FP32 with equal byte count", "expected": "rejected before launch"},
    {"fault": "root", "injection": "broadcast roots differ", "expected": "explicit mismatch"},
    {"fault": "group_membership", "injection": "ordered ranks differ", "expected": "group init/preflight failure"},
    {"fault": "skipped_call", "injection": "one rank does not enter", "expected": "watchdog timeout, group abort"},
    {"fault": "delayed_rank", "injection": "bounded delay ladder", "expected": "below threshold completes; above fails by policy"},
    {"fault": "duplicate_seq", "injection": "reuse a wrong buffer/sequence", "expected": "state machine rejects"},
    {"fault": "post_error_reuse", "injection": "continue after a fatal error", "expected": "wrapper rejects COMM_ABORTED"},
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── state machine ──────────────────────────────────────────────────────────


class CommunicatorStateMachine:
    """READY/IN_FLIGHT/ERROR/ABORTED/DESTROYED with post-error rejection."""

    def __init__(self, group_id: str, rank_epoch: int = 0) -> None:
        self.group_id = group_id
        self.rank_epoch = rank_epoch
        self.state = "INIT"
        self.history: List[Dict[str, Any]] = []
        self.last_seq: int = -1
        self.abort_reason = ""

    def transition(self, target: str, *, reason: str = "") -> str:
        if target not in COMMUNICATOR_STATES:
            raise ConfigError(
                f"unknown communicator state {target!r}", details={"field": "target"}
            )
        allowed = ALLOWED_TRANSITIONS.get(self.state, ())
        if target not in allowed:
            raise ConfigError(
                f"illegal transition {self.state} -> {target} for group {self.group_id}",
                details={"field": "state"},
            )
        if target == "ABORTING" and not reason:
            raise ConfigError("aborting a communicator needs a reason", details={"field": "reason"})
        self.state = target
        if target == "ABORTING":
            self.abort_reason = reason
        self.history.append({"from": self.history[-1]["to"] if self.history else "INIT",
                             "to": target, "reason": reason})
        return self.state

    def begin_collective(self, seq: int) -> None:
        """Refuse any new call once the communicator is not usable."""
        if self.state in ("ERROR", "ABORTING", "ABORTED", "DESTROYED"):
            raise ConfigError(
                f"COMM_ABORTED: group {self.group_id} is {self.state}; the communicator must be "
                "rebuilt before another collective",
                details={"field": "state"},
            )
        if self.state not in ("READY", "COMPLETED"):
            raise ConfigError(
                f"group {self.group_id} is {self.state}; cannot enqueue seq {seq}",
                details={"field": "state"},
            )
        if seq <= self.last_seq:
            raise ConfigError(
                f"sequence {seq} is not greater than the last enqueued {self.last_seq}",
                details={"field": "collective_seq"},
            )
        self.last_seq = seq
        self.transition("ENQUEUED", reason=f"seq={seq}")

    def complete(self) -> None:
        self.transition("IN_FLIGHT")
        self.transition("COMPLETED")

    def fail(self, reason: str) -> None:
        if self.state in ("COMPLETED", "READY", "INIT"):
            self.transition("ERROR", reason=reason)
        self.transition("ABORTING", reason=reason)
        self.transition("ABORTED")

    def destroy(self) -> None:
        if self.state in ("ERROR", "ABORTING"):
            raise ConfigError(
                "abort before destroy; destroying a failed communicator hides the state",
                details={"field": "state"},
            )
        self.transition("DESTROYED", reason="explicit destroy")

    def destroy_or_abort(self) -> None:
        """Destroy if safe, otherwise abort then destroy (cleanup ordering)."""
        if self.state in ("ERROR", "ABORTING"):
            if self.state == "ERROR":
                self.transition("ABORTING", reason="cleanup after error")
            self.transition("ABORTED")
        self.transition("DESTROYED", reason="cleanup")

    @property
    def usable(self) -> bool:
        return self.state in ("READY", "COMPLETED", "ENQUEUED", "IN_FLIGHT")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "rank_epoch": self.rank_epoch,
            "state": self.state,
            "last_seq": self.last_seq,
            "abort_reason": self.abort_reason,
            "history": list(self.history),
        }


# ── sequence allocation ────────────────────────────────────────────────────


class SequenceAllocator:
    """Per-(group, epoch) monotonic sequence namespace (details E10-03 step 4)."""

    def __init__(self, group_id: str, rank_epoch: int = 0) -> None:
        self.group_id = group_id
        self.rank_epoch = rank_epoch
        self._next = 0
        self._issued: List[Tuple[int, int]] = []

    def next(self) -> int:
        seq = self._next
        self._next += 1
        self._issued.append((self.rank_epoch, seq))
        return seq

    def reset_for_epoch(self, new_epoch: int) -> None:
        """A rebuilt communicator gets a *new namespace*, not a continued one."""
        if new_epoch <= self.rank_epoch:
            raise ConfigError(
                f"new epoch {new_epoch} must be greater than {self.rank_epoch}",
                details={"field": "new_epoch"},
            )
        self.rank_epoch = new_epoch
        self._next = 0

    def assert_new_epoch_seq(self, seq: int) -> None:
        if self._issued and any(
            epoch != self.rank_epoch and issued == seq for epoch, issued in self._issued
        ):
            raise ConfigError(
                f"sequence {seq} was issued under an older epoch; a rebuilt communicator must "
                "not reuse it",
                details={"field": "collective_seq"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "rank_epoch": self.rank_epoch,
            "next_seq": self._next,
            "issued": len(self._issued),
        }


# ── call records / preflight ───────────────────────────────────────────────


@dataclass(frozen=True)
class CollectiveCallRecord:
    """Per-rank record persisted *before* enqueue (details E10-03 §4)."""

    run_id: str
    rank_epoch: int
    global_rank: int
    group_rank: int
    group_id: str
    ordered_group_hash: str
    collective_seq: int
    op: str
    root: Optional[int] = None
    reduce_op: str = "sum"
    logical_count: int = 0
    dtype: str = ""
    input_shape_hash: str = ""
    output_shape_hash: str = ""
    inplace: bool = False
    stream_id: str = ""
    callsite: str = ""
    enqueue_time_ns: int = 0
    completion_time_ns: int = 0
    error: str = ""

    @property
    def metadata_hash(self) -> str:
        return _sha256_text(
            repr({name: getattr(self, name) for name in METADATA_FIELDS})
        )

    def as_dict(self) -> Dict[str, Any]:
        payload = {name: getattr(self, name) for name in METADATA_FIELDS}
        payload.update(
            {
                "run_id": self.run_id,
                "rank_epoch": self.rank_epoch,
                "global_rank": self.global_rank,
                "group_rank": self.group_rank,
                "metadata_hash": self.metadata_hash,
                "enqueue_time_ns": self.enqueue_time_ns,
                "completion_time_ns": self.completion_time_ns,
                "error": self.error,
            }
        )
        return payload


@dataclass(frozen=True)
class FieldDiff:
    """One field-level divergence between two ranks' records."""

    field: str
    left: Any
    right: Any
    semantic: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "left": self.left,
            "right": self.right,
            "semantic": self.semantic,
        }


@dataclass(frozen=True)
class Divergence:
    """The first divergence: group/seq/rank/field, not "who timed out first"."""

    group_id: str
    collective_seq: int
    left_rank: int
    right_rank: int
    field: str
    left_value: Any
    right_value: Any

    def as_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "collective_seq": self.collective_seq,
            "left_rank": self.left_rank,
            "right_rank": self.right_rank,
            "field": self.field,
            "left_value": self.left_value,
            "right_value": self.right_value,
        }


@dataclass
class PreflightResult:
    ok: bool
    divergence: Optional[Divergence] = None
    field_diffs: Tuple[FieldDiff, ...] = ()
    mode: str = "fault"
    byte_counts_match_but_semantics_differ: bool = False
    handshake_is_not_a_barrier: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "divergence": self.divergence.as_dict() if self.divergence else None,
            "field_diffs": [item.as_dict() for item in self.field_diffs],
            "byte_counts_match_but_semantics_differ": self.byte_counts_match_but_semantics_differ,
            "handshake_is_not_a_barrier": self.handshake_is_not_a_barrier,
            "note": (
                "the metadata handshake only catches development errors; it does not replace "
                "backend timeouts or async-error handling, and it must never become a global "
                "barrier in the performance path"
            ),
        }


def _dtype_bytes(dtype: str) -> int:
    from hqsb.distributed.collectives import DTYPE_BYTES

    return DTYPE_BYTES.get(dtype, 0)


def field_diffs(
    left: CollectiveCallRecord, right: CollectiveCallRecord
) -> List[FieldDiff]:
    """Field-level comparison; total byte equality never excuses a dtype change."""
    diffs: List[FieldDiff] = []
    for name in METADATA_FIELDS:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if left_value != right_value:
            diffs.append(
                FieldDiff(
                    field=name,
                    left=left_value,
                    right=right_value,
                    semantic=name in SEMANTIC_CRITICAL_FIELDS,
                )
            )
    return diffs


def byte_counts_match_but_semantics_differ(
    left: CollectiveCallRecord, right: CollectiveCallRecord
) -> bool:
    """The FP16-vs-FP32 trap: same bytes, different meaning (details E10-03 step 14)."""
    if left.dtype == right.dtype:
        return False
    return (
        left.logical_count * _dtype_bytes(left.dtype)
        == right.logical_count * _dtype_bytes(right.dtype)
        and left.logical_count * _dtype_bytes(left.dtype) > 0
    )


def preflight_check(
    records_by_rank: Mapping[int, CollectiveCallRecord], *, mode: str = "fault"
) -> PreflightResult:
    """Aggregate metadata and report the *first* divergence with field-level detail."""
    if mode not in ("debug", "fault"):
        raise ConfigError(
            "the metadata preflight runs in debug/fault mode only (never in the performance "
            "path)",
            details={"field": "mode"},
        )
    if len(records_by_rank) < 2:
        raise ConfigError("the preflight needs at least two ranks")
    ranks = sorted(records_by_rank)
    reference_rank = ranks[0]
    reference = records_by_rank[reference_rank]
    for rank in ranks[1:]:
        record = records_by_rank[rank]
        diffs = field_diffs(reference, record)
        if diffs:
            first = diffs[0]
            return PreflightResult(
                ok=False,
                divergence=Divergence(
                    group_id=reference.group_id,
                    collective_seq=min(reference.collective_seq, record.collective_seq),
                    left_rank=reference_rank,
                    right_rank=rank,
                    field=first.field,
                    left_value=first.left,
                    right_value=first.right,
                ),
                field_diffs=tuple(diffs),
                mode=mode,
                byte_counts_match_but_semantics_differ=byte_counts_match_but_semantics_differ(
                    reference, record
                ),
            )
    return PreflightResult(ok=True, mode=mode)


# ── timeouts ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TimeoutSpec:
    """The five timeouts plus the pre-registered multiplier (details E10-03 §6/§13)."""

    init_timeout_s: float
    collective_timeout_s: float
    watchdog_heartbeat_timeout_s: float
    job_kill_grace_s: float
    post_cleanup_probe_timeout_s: float
    multiplier: float
    minimum_operational_timeout_s: float
    maximum_timeout_s: float
    family: str = "ordinary"
    healthy_p99_s: float = 0.0
    frozen: bool = False

    def __post_init__(self) -> None:
        if self.family not in TIMEOUT_FAMILIES:
            raise ConfigError(
                f"family must be one of {TIMEOUT_FAMILIES}", details={"field": "family"}
            )
        if self.multiplier < 1.0:
            raise ConfigError(
                "the timeout multiplier must be >= 1 (a timeout below the healthy P99 would "
                "kill slow-but-correct runs)",
                details={"field": "multiplier"},
            )
        if self.minimum_operational_timeout_s <= 0:
            raise ConfigError("minimum operational timeout must be positive")
        if self.maximum_timeout_s < self.minimum_operational_timeout_s:
            raise ConfigError("maximum timeout must be >= the minimum")

    def compute_collective_timeout(self, healthy_p99_s: Optional[float] = None) -> float:
        """timeout = clamp(multiplier × healthy P99, min, max) (details E10-03 §13)."""
        p99 = healthy_p99_s if healthy_p99_s is not None else self.healthy_p99_s
        if p99 <= 0:
            raise ConfigError(
                "the healthy P99 must come from E10-02 before a timeout can be derived",
                details={"field": "healthy_p99_s"},
            )
        value = self.multiplier * p99
        return min(max(value, self.minimum_operational_timeout_s), self.maximum_timeout_s)

    def refuse_post_fault_edit(self, *, edited_after_fault: bool) -> None:
        if edited_after_fault:
            raise ConfigError(
                "the timeout thresholds are pre-registered; they may not be edited after the "
                "fault times were seen",
                details={"field": "frozen"},
            )

    def validate(self) -> None:
        if self.collective_timeout_s <= 0 or self.watchdog_heartbeat_timeout_s <= 0:
            raise ConfigError("timeouts must be positive")
        if self.job_kill_grace_s < 0:
            raise ConfigError("job kill grace must be >= 0")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "init_timeout_s": self.init_timeout_s,
            "collective_timeout_s": self.collective_timeout_s,
            "watchdog_heartbeat_timeout_s": self.watchdog_heartbeat_timeout_s,
            "job_kill_grace_s": self.job_kill_grace_s,
            "post_cleanup_probe_timeout_s": self.post_cleanup_probe_timeout_s,
            "multiplier": self.multiplier,
            "minimum_operational_timeout_s": self.minimum_operational_timeout_s,
            "maximum_timeout_s": self.maximum_timeout_s,
            "family": self.family,
            "healthy_p99_s": self.healthy_p99_s,
            "frozen": self.frozen,
        }


def delayed_rank_ladder(threshold_s: float) -> List[Dict[str, Any]]:
    """Delay cases on both sides of the threshold (details E10-03 step 18)."""
    if threshold_s <= 0:
        raise ConfigError("threshold must be positive", details={"field": "threshold_s"})
    return [
        {"case": "no_delay", "delay_s": 0.0, "expected": "completes"},
        {"case": "below_threshold", "delay_s": threshold_s * 0.25, "expected": "completes"},
        {"case": "near_threshold", "delay_s": threshold_s * 0.9, "expected": "completes"},
        {"case": "just_above", "delay_s": threshold_s * 1.1, "expected": "bounded failure"},
        {"case": "far_above", "delay_s": threshold_s * 3.0, "expected": "bounded failure"},
    ]


def evaluate_delayed_case(
    *, delay_s: float, threshold_s: float, observed: str
) -> Dict[str, Any]:
    """Check a delayed-rank case did not kill a healthy-but-slow rank."""
    expected = "completes" if delay_s <= threshold_s else "bounded failure"
    ok = observed == expected
    return {
        "delay_s": delay_s,
        "threshold_s": threshold_s,
        "expected": expected,
        "observed": observed,
        "ok": ok,
        "false_positive": observed == "bounded failure" and expected == "completes",
        "false_negative": observed == "completes" and expected == "bounded failure",
    }


# ── watchdog / propagation / cleanup ───────────────────────────────────────


class Watchdog:
    """External watchdog that never depends on the failing communicator."""

    def __init__(self, *, heartbeat_timeout_s: float, progress_timeout_s: float) -> None:
        if heartbeat_timeout_s <= 0 or progress_timeout_s <= 0:
            raise ConfigError("watchdog timeouts must be positive")
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.progress_timeout_s = progress_timeout_s
        self.heartbeats: List[Dict[str, Any]] = []

    def record(self, *, rank: int, pid: int, last_seq: int, state: str, at_s: float) -> None:
        self.heartbeats.append(
            {"rank": rank, "pid": pid, "last_seq": last_seq, "state": state, "at_s": at_s}
        )

    def evaluate(self, *, now_s: float) -> Dict[str, Any]:
        from hqsb.distributed.faults import ControlHeartbeat, watchdog_decision

        beats = [
            ControlHeartbeat(
                rank=item["rank"],
                pid=item["pid"],
                last_seq=item["last_seq"],
                state=item["state"],
                at_s=item["at_s"],
            )
            for item in self.heartbeats
        ]
        if not beats:
            return {"action": "none", "reason": "no heartbeats recorded"}
        return watchdog_decision(
            beats,
            now_s=now_s,
            heartbeat_timeout_s=self.heartbeat_timeout_s,
            progress_timeout_s=self.progress_timeout_s,
        )


class ErrorPropagationChannel:
    """Independent control-plane channel; completeness over speed."""

    def __init__(self, world_size: int) -> None:
        if world_size < 1:
            raise ConfigError("world_size must be >= 1")
        self.world_size = world_size
        self.reports: Dict[int, Dict[str, Any]] = {}

    def report(
        self,
        *,
        rank: int,
        error_class: str,
        group_id: str = "",
        collective_seq: int = -1,
        detected_at_s: float = 0.0,
        message: str = "",
    ) -> None:
        if not 0 <= rank < self.world_size:
            raise ConfigError(f"rank {rank} outside [0, {self.world_size})")
        self.reports.setdefault(
            rank,
            {
                "rank": rank,
                "error_class": error_class,
                "group_id": group_id,
                "collective_seq": collective_seq,
                "detected_at_s": detected_at_s,
                "message": message,
            },
        )

    def missing_ranks(self) -> List[int]:
        return sorted(set(range(self.world_size)) - set(self.reports))

    @property
    def complete(self) -> bool:
        return not self.missing_ranks()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "world_size": self.world_size,
            "reported": len(self.reports),
            "missing_ranks": self.missing_ranks(),
            "complete": self.complete,
            "reports": [self.reports[rank] for rank in sorted(self.reports)],
        }


def cleanup_plan(group_ids: Sequence[str], *, dependency_order: Sequence[str]) -> Dict[str, Any]:
    """Abort/destroy in reverse dependency order (details E10-03 step 9)."""
    missing = sorted(set(dependency_order) - set(group_ids))
    if missing:
        raise ConfigError(f"cleanup order references unknown groups {missing}")
    reversed_order = list(reversed(list(dependency_order)))
    return {
        "order": reversed_order,
        "steps": [
            "abort in-flight work handles",
            "destroy communicators in reverse dependency order",
            "release buffers/streams/events",
            "close sockets/fds and remove rendezvous artifacts",
            "run the post-cleanup probe with a new epoch",
        ],
        "note": "abort and destroy are distinct: abort marks the state unsafe, destroy frees it",
    }


# ── safety gate ────────────────────────────────────────────────────────────


def tp_safety_gate(
    *,
    golden_ok: bool,
    faults_bounded: bool,
    cleanup_ok: bool,
    recreate_ok: bool,
    false_negatives: Sequence[str] = (),
) -> Dict[str, Any]:
    """E10-03 → E10-04 gate: no false alarms, all mismatches bounded (step 30)."""
    blockers: List[str] = []
    if not golden_ok:
        blockers.append("the legal golden sequence is not verified / produced false alarms")
    if not faults_bounded:
        blockers.append("at least one mismatch was not bounded (hang/partial success)")
    if not cleanup_ok:
        blockers.append("resources were not reclaimed (process/memory/fd/rendezvous)")
    if not recreate_ok:
        blockers.append("a new-epoch communicator was not proven healthy")
    if false_negatives:
        blockers.append(f"undetected injections: {list(false_negatives)}")
    return {
        "ready": not blockers,
        "gate": "E10-03 -> E10-04",
        "blockers": blockers,
        "reason": "" if not blockers else "TP work must not start on an unsafe communicator layer",
    }


__all__ = [
    "ALLOWED_TRANSITIONS",
    "COLLECTIVE_FAULTS",
    "CollectiveCallRecord",
    "CommunicatorStateMachine",
    "Divergence",
    "ErrorPropagationChannel",
    "FieldDiff",
    "GOLDEN_SEQUENCE",
    "METADATA_FIELDS",
    "PreflightResult",
    "SEMANTIC_CRITICAL_FIELDS",
    "SequenceAllocator",
    "TIMEOUT_FAMILIES",
    "TimeoutSpec",
    "WRAPPER_ERROR_CODES",
    "Watchdog",
    "byte_counts_match_but_semantics_differ",
    "cleanup_plan",
    "delayed_rank_ladder",
    "evaluate_delayed_case",
    "field_diffs",
    "preflight_check",
    "tp_safety_gate",
]

# Re-exported for the interface map (the drift oracle is shared with E10-10).
SHARED_ORACLE = FaultOracle
SHARED_TIME_METRICS = TimeMetrics
SHARED_DRILL = DrillResult
SHARED_GROUP = GroupMembership
