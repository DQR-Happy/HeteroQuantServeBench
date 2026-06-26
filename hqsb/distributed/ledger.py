"""Communication ledger: expected events ↔ observed events ↔ per-rank memory.

The ledger is the accounting that connects a ParallelPlan to what actually
happened.  Two rules are structural here:

* the expected side is enumerated **from the plan, shapes, phase and token
  count** — never reverse-engineered from the trace (details E10-04 step 24);
* every difference must carry a reason code (fusion, deferred gather, padding,
  fallback, instrumentation missing, duplicate gather…), and an unmapped event
  keeps the ledger *open* (details E10-04 steps 25–26).

Memory reconciliation follows the same philosophy: a measured/predicted gap is
decomposed into replication, padding, metadata, workspace and cache before the
word "fragmentation" is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.distributed.collectives import DTYPE_BYTES
from hqsb.distributed.parallel_plan import (
    ParallelPlan,
    QwenArchitectureCensus,
)

#: Reason codes a ledger difference may use (details E10-04 §3.5/§8).
LEDGER_REASON_CODES: Tuple[str, ...] = (
    "fusion",
    "deferred_gather",
    "padding",
    "fallback",
    "instrumentation_missing",
    "duplicate_gather",
    "layout_conversion",
    "trace_duplication",
    "reduce_scatter_keeps_shard",
    "vocab_parallel_logits",
    "prefix_cache_actual_rows",
)

#: Phases the ledger enumerates separately.
LEDGER_PHASES: Tuple[str, ...] = ("prefill", "decode", "both")

#: Memory buckets that must be reported separately (details E10-04 step 27).
MEMORY_BUCKETS: Tuple[str, ...] = (
    "weights",
    "kv",
    "activation",
    "workspace",
    "communicator_buffers",
    "graph_pool",
    "allocator_reserved",
    "replicated_extra",
)


def _require(condition: bool, message: str, *, field_name: str = "") -> None:
    if not condition:
        raise ConfigError(message, details={"field": field_name} if field_name else None)


@dataclass(frozen=True)
class ExpectedEvent:
    """One expected collective, enumerated from the plan (never from a trace)."""

    phase: str
    layer: Any
    submodule: str
    collective: str
    tensor_role: str
    rows: int
    hidden: int
    dtype: str
    payload_bytes: int
    group_size: int
    expected_calls: int = 1
    expected_algorithmic_bytes: int = 0
    source: str = "plan"

    def __post_init__(self) -> None:
        _require(self.phase in LEDGER_PHASES, "unknown phase", field_name="phase")
        _require(self.rows >= 0 and self.hidden >= 0, "rows/hidden must be >= 0")
        _require(self.dtype in DTYPE_BYTES, "unknown dtype", field_name="dtype")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "layer": self.layer,
            "submodule": self.submodule,
            "collective": self.collective,
            "tensor_role": self.tensor_role,
            "rows": self.rows,
            "hidden": self.hidden,
            "dtype": self.dtype,
            "payload_bytes": self.payload_bytes,
            "group_size": self.group_size,
            "expected_calls": self.expected_calls,
            "expected_algorithmic_bytes": self.expected_algorithmic_bytes,
            "source": self.source,
        }


def expected_ledger(
    plan: ParallelPlan,
    census: QwenArchitectureCensus,
    *,
    phase: str,
    token_rows: int,
    layer_range: Optional[Tuple[int, int]] = None,
    dtype: str = "fp16",
    vocab_parallel: bool = False,
) -> List[ExpectedEvent]:
    """Enumerate the expected collectives for one phase (details E10-04 §11).

    The classic column/row pairing predicts two hidden-sized reductions per
    layer; the template below is *the plan's* trigger list, so a runtime that
    uses ReduceScatter or fuses events changes only the observed side.
    """
    if phase not in ("prefill", "decode"):
        raise ConfigError("phase must be prefill|decode", details={"field": "phase"})
    if token_rows < 0:
        raise ConfigError("token_rows must be >= 0", details={"field": "token_rows"})
    if dtype not in DTYPE_BYTES:
        raise ConfigError(f"unknown dtype {dtype!r}", details={"field": "dtype"})
    start, end = layer_range if layer_range else (0, census.num_layers)
    _require(0 <= start <= end <= census.num_layers, "layer range outside the model")
    element = DTYPE_BYTES[dtype]
    rows = token_rows
    hidden_bytes = rows * census.hidden_size * element
    events: List[ExpectedEvent] = []
    for event in plan.collective_events:
        if event.layer is None:
            continue
        layer_index = int(event.layer)
        if not (start <= layer_index < end):
            continue
        events.append(
            ExpectedEvent(
                phase=phase,
                layer=layer_index,
                submodule=event.submodule,
                collective=event.collective,
                tensor_role=event.tensor_role,
                rows=rows,
                hidden=census.hidden_size,
                dtype=dtype,
                payload_bytes=hidden_bytes,
                group_size=plan.tp_degree,
                expected_algorithmic_bytes=hidden_bytes,
            )
        )
    if vocab_parallel and phase == "decode":
        vocab_rows = rows * census.vocab_size * element
        events.append(
            ExpectedEvent(
                phase=phase,
                layer="lm_head",
                submodule="lm_head",
                collective="all_gather",
                tensor_role="logits",
                rows=rows,
                hidden=census.vocab_size,
                dtype=dtype,
                payload_bytes=vocab_rows,
                group_size=plan.tp_degree,
                expected_algorithmic_bytes=vocab_rows,
            )
        )
    return events


def expected_totals(events: Sequence[ExpectedEvent]) -> Dict[str, Any]:
    """Aggregate counts/bytes by collective for the diff table."""
    by_op: Dict[str, Dict[str, int]] = {}
    for event in events:
        bucket = by_op.setdefault(event.collective, {"calls": 0, "logical_bytes": 0})
        bucket["calls"] += event.expected_calls
        bucket["logical_bytes"] += event.payload_bytes * event.expected_calls
    return {
        "total_calls": sum(item["calls"] for item in by_op.values()),
        "total_logical_bytes": sum(item["logical_bytes"] for item in by_op.values()),
        "by_collective": by_op,
    }


@dataclass(frozen=True)
class ObservedEvent:
    """One collective as recorded by the runtime/profiler (details README §17.3)."""

    rank: int
    group_id: str
    collective_seq: int
    op: str
    tensor_role: str
    phase: str
    layer: Any
    payload_bytes: int
    dtype: str
    start_ns: int = 0
    end_ns: int = 0
    stream: str = ""
    algorithm: str = ""
    protocol: str = ""
    callsite: str = ""
    source: str = "runtime"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "group_id": self.group_id,
            "collective_seq": self.collective_seq,
            "op": self.op,
            "tensor_role": self.tensor_role,
            "phase": self.phase,
            "layer": self.layer,
            "payload_bytes": self.payload_bytes,
            "dtype": self.dtype,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "stream": self.stream,
            "algorithm": self.algorithm,
            "protocol": self.protocol,
            "callsite": self.callsite,
            "source": self.source,
        }


@dataclass(frozen=True)
class LedgerDiffRow:
    phase: str
    layer: Any
    submodule: str
    collective: str
    expected_calls: int
    observed_calls: int
    expected_bytes: int
    observed_bytes: int
    reason_code: str = ""
    explanation: str = ""

    @property
    def matched(self) -> bool:
        return (
            self.expected_calls == self.observed_calls
            and self.expected_bytes == self.observed_bytes
        )

    @property
    def explained(self) -> bool:
        return bool(self.reason_code) and self.reason_code in LEDGER_REASON_CODES

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "layer": self.layer,
            "submodule": self.submodule,
            "collective": self.collective,
            "expected_calls": self.expected_calls,
            "observed_calls": self.observed_calls,
            "expected_bytes": self.expected_bytes,
            "observed_bytes": self.observed_bytes,
            "matched": self.matched,
            "reason_code": self.reason_code,
            "explained": self.explained,
            "explanation": self.explanation,
        }


@dataclass
class LedgerResult:
    rows: Tuple[LedgerDiffRow, ...]
    unmatched_expected: Tuple[Dict[str, Any], ...] = ()
    unmatched_observed: Tuple[Dict[str, Any], ...] = ()
    total_expected_bytes: int = 0
    total_observed_bytes: int = 0
    reason_codes: Tuple[str, ...] = ()

    @property
    def conservation_ok(self) -> bool:
        return not self.unmatched_expected and not self.unmatched_observed

    @property
    def closed(self) -> bool:
        """A ledger is closed when every row matches or carries a legal reason."""
        if not self.conservation_ok:
            return False
        return all(row.matched or row.explained for row in self.rows)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "closed": self.closed,
            "conservation_ok": self.conservation_ok,
            "total_expected_bytes": self.total_expected_bytes,
            "total_observed_bytes": self.total_observed_bytes,
            "rows": [row.as_dict() for row in self.rows],
            "unmatched_expected": list(self.unmatched_expected),
            "unmatched_observed": list(self.unmatched_observed),
            "reason_codes": list(self.reason_codes),
            "note": (
                "differences are grouped by op/count/bytes and must be explained; comparing "
                "only the byte total hides a small×2 + large×1 swap"
            ),
        }


def diff_ledger(
    expected: Sequence[ExpectedEvent],
    observed: Sequence[ObservedEvent],
    *,
    explanations: Optional[Mapping[Tuple[str, Any, str], Tuple[str, str]]] = None,
) -> LedgerResult:
    """Expected ↔ observed comparison with explicit reasons (details E10-04 step 26)."""
    explanations = dict(explanations or {})
    expected_index: Dict[Tuple[str, Any, str], List[ExpectedEvent]] = {}
    for event in expected:
        expected_index.setdefault((event.phase, event.layer, event.collective), []).append(event)
    observed_index: Dict[Tuple[str, Any, str], List[ObservedEvent]] = {}
    for event in observed:
        observed_index.setdefault((event.phase, event.layer, event.op), []).append(event)

    keys = sorted(set(expected_index) | set(observed_index), key=lambda item: (str(item[0]), str(item[1]), item[2]))
    rows: List[LedgerDiffRow] = []
    unmatched_expected: List[Dict[str, Any]] = []
    unmatched_observed: List[Dict[str, Any]] = []
    for key in keys:
        expected_events = expected_index.get(key, [])
        observed_events = observed_index.get(key, [])
        expected_calls = sum(item.expected_calls for item in expected_events)
        expected_bytes = sum(item.payload_bytes * item.expected_calls for item in expected_events)
        observed_calls = len(observed_events)
        observed_bytes = sum(item.payload_bytes for item in observed_events)
        reason_code, explanation = explanations.get(key, ("", ""))
        row = LedgerDiffRow(
            phase=key[0],
            layer=key[1],
            submodule=(expected_events[0].submodule if expected_events else (observed_events[0].callsite if observed_events else "")),
            collective=key[2],
            expected_calls=expected_calls,
            observed_calls=observed_calls,
            expected_bytes=expected_bytes,
            observed_bytes=observed_bytes,
            reason_code=reason_code,
            explanation=explanation,
        )
        rows.append(row)
        if expected_events and not observed_events:
            unmatched_expected.extend(item.as_dict() for item in expected_events)
        if observed_events and not expected_events:
            unmatched_observed.extend(item.as_dict() for item in observed_events)
    reasons = tuple(sorted({row.reason_code for row in rows if row.reason_code}))
    return LedgerResult(
        rows=tuple(rows),
        unmatched_expected=tuple(unmatched_expected),
        unmatched_observed=tuple(unmatched_observed),
        total_expected_bytes=sum(item.payload_bytes * item.expected_calls for item in expected),
        total_observed_bytes=sum(item.payload_bytes for item in observed),
        reason_codes=reasons,
    )


def why_observed_can_differ() -> List[Dict[str, str]]:
    """The documented reasons the observed side may legitimately differ (§11)."""
    return [
        {"reason_code": "reduce_scatter_keeps_shard", "detail": "ReduceScatter keeps the output sharded and defers the gather"},
        {"reason_code": "deferred_gather", "detail": "a later consumer performs the gather instead"},
        {"reason_code": "fusion", "detail": "bias/residual/norm fusion moves the collective"},
        {"reason_code": "padding", "detail": "physical counts exceed the logical count"},
        {"reason_code": "vocab_parallel_logits", "detail": "vocab-parallel logits add reduce/gather/top-k traffic"},
        {"reason_code": "layout_conversion", "detail": "an extra layout conversion was inserted"},
        {"reason_code": "fallback", "detail": "a fallback path ran; requested/actual must be recorded"},
        {"reason_code": "instrumentation_missing", "detail": "the trace did not capture the event (not a real difference)"},
        {"reason_code": "duplicate_gather", "detail": "the same tensor was gathered twice (layout state was not carried)"},
        {"reason_code": "trace_duplication", "detail": "the trace recorded one logical collective more than once"},
        {"reason_code": "prefix_cache_actual_rows", "detail": "prefix cache changed the real token rows that participated"},
    ]


# ── memory reconciliation ──────────────────────────────────────────────────


@dataclass(frozen=True)
class MemoryRow:
    """One rank's measured/predicted memory breakdown (details E10-04 step 27)."""

    rank: int
    weights: int
    kv: int
    activation: int
    workspace: int
    communicator_buffers: int
    graph_pool: int = 0
    allocator_reserved: int = 0
    replicated_extra: int = 0

    def buckets(self) -> Dict[str, int]:
        return {
            "weights": self.weights,
            "kv": self.kv,
            "activation": self.activation,
            "workspace": self.workspace,
            "communicator_buffers": self.communicator_buffers,
            "graph_pool": self.graph_pool,
            "allocator_reserved": self.allocator_reserved,
            "replicated_extra": self.replicated_extra,
        }

    @property
    def total(self) -> int:
        return sum(self.buckets().values())

    def as_dict(self) -> Dict[str, Any]:
        payload = {"rank": self.rank}
        payload.update(self.buckets())
        payload["total"] = self.total
        return payload


def reconcile_memory(
    predicted: Mapping[str, int],
    measured: MemoryRow,
    *,
    tolerance_fraction: float = 0.1,
) -> Dict[str, Any]:
    """Decompose a predicted/measured gap bucket by bucket before explaining it."""
    mismatches: List[Dict[str, Any]] = []
    for bucket, predicted_value in predicted.items():
        if bucket not in MEMORY_BUCKETS:
            raise ConfigError(
                f"unknown memory bucket {bucket!r}", details={"field": "bucket"}
            )
        measured_value = measured.buckets()[bucket]
        if predicted_value <= 0:
            continue
        delta = measured_value - predicted_value
        if abs(delta) > predicted_value * tolerance_fraction:
            mismatches.append(
                {
                    "bucket": bucket,
                    "predicted": predicted_value,
                    "measured": measured_value,
                    "delta": delta,
                    "needs_decomposition": True,
                }
            )
    return {
        "rank": measured.rank,
        "ok": not mismatches,
        "mismatches": mismatches,
        "note": (
            "a gap must be decomposed into replication, padding, metadata, workspace and cache "
            "before it may be called fragmentation"
        ),
    }


def rank_skew(rows: Sequence[MemoryRow]) -> Dict[str, Any]:
    """The max rank decides capacity — never the mean (details E10-04 §8)."""
    if not rows:
        raise ConfigError("rank_skew needs at least one row")
    totals = {row.rank: row.total for row in rows}
    max_rank = max(totals, key=lambda rank: totals[rank])
    mean = sum(totals.values()) / len(totals)
    return {
        "max_rank": max_rank,
        "max_bytes": totals[max_rank],
        "mean_bytes": mean,
        "skew_fraction": (totals[max_rank] - mean) / mean if mean else 0.0,
        "totals": {str(rank): totals[rank] for rank in sorted(totals)},
        "note": "an even shard plan with uneven measured memory points at workspace/allocator skew",
    }


# ── derived helpers ────────────────────────────────────────────────────────


def kv_bytes(
    census: QwenArchitectureCensus,
    *,
    tokens: int,
    degree: int,
    kv_dtype_bytes: int = 2,
) -> int:
    """Global KV bytes for a token count (prediction side of the memory model)."""
    if tokens < 0:
        raise ConfigError("tokens must be >= 0", details={"field": "tokens"})
    head_groups_local = census.num_kv_heads
    del degree  # the global formula does not depend on the shard count
    return (
        census.num_layers * tokens * head_groups_local * census.head_dim * 2 * kv_dtype_bytes
    )


def ledger_gate(result: LedgerResult, *, require_reason_for_every_diff: bool = True) -> Dict[str, Any]:
    """Close the ledger or list the unexplained rows (used before scaling)."""
    unexplained = [
        row.as_dict()
        for row in result.rows
        if not row.matched and (require_reason_for_every_diff and not row.explained)
    ]
    blockers: List[str] = []
    if result.unmatched_expected:
        blockers.append(f"{len(result.unmatched_expected)} expected events have no observed counterpart")
    if result.unmatched_observed:
        blockers.append(f"{len(result.unmatched_observed)} observed events are not in the plan")
    if unexplained:
        blockers.append(f"{len(unexplained)} differing rows have no reason code")
    return {
        "ready": not blockers,
        "closed": result.closed,
        "blockers": blockers,
        "unexplained_rows": unexplained,
    }


__all__ = [
    "ExpectedEvent",
    "LEDGER_PHASES",
    "LEDGER_REASON_CODES",
    "LedgerDiffRow",
    "LedgerResult",
    "MEMORY_BUCKETS",
    "MemoryRow",
    "ObservedEvent",
    "diff_ledger",
    "expected_ledger",
    "expected_totals",
    "kv_bytes",
    "ledger_gate",
    "rank_skew",
    "reconcile_memory",
    "why_observed_can_differ",
]
