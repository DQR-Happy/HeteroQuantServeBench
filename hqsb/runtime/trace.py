"""Request state machine, spans, iteration ledger and hot-path reporting (E07-02).

E07-02 is the experiment that turns a runtime from a black box into a causal
chain.  This module supplies the four artefacts that chain needs:

* :class:`RequestStateMachine` — the unified states of details README §7 plus the
  per-transition record (request, old/new, timestamp, iteration, reason, tokens,
  KV blocks/refcounts, batch, runner span, error, cleanup);
* :class:`SpanCollector` — the nested span model of §3 with an explicit trace
  context (``request_id`` is a *parameter*, never a thread-local guess) and
  attribute redaction so prompts cannot leak into a trace file;
* :class:`IterationLedgerEntry` — the per-iteration accounting table of §9 with
  the token-conservation identity
  ``previous_computed + scheduled_computed - rollback == new_computed``;
* report builders (:func:`state_machine_graph`, :func:`call_chain_tree`,
  :func:`hot_path_table`, :func:`instrumentation_overhead`) that are generated
  **from the raw records** — a hand-drawn state machine is a listed anti-pattern.

Clock discipline follows §4: host and device clocks are never subtracted
directly; :class:`ClockCalibration` must exist before a host/device delta is
computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError, SchemaError

# ── request state machine (details README §7) ──────────────────────────────


class RequestState:
    """Unified request states; runtime-private names are mapped onto these."""

    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    WAITING = "WAITING"
    ADMITTED = "ADMITTED"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    FINISHED = "FINISHED"
    PREEMPTED = "PREEMPTED"
    RESUMED = "RESUMED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    CLEANED = "CLEANED"


#: Legal transitions.  Terminal states only advance to ``CLEANED``.
ALLOWED_TRANSITIONS: Mapping[str, Tuple[str, ...]] = {
    RequestState.CREATED: (RequestState.VALIDATED, RequestState.REJECTED, RequestState.FAILED),
    RequestState.VALIDATED: (RequestState.WAITING, RequestState.REJECTED, RequestState.FAILED),
    RequestState.WAITING: (
        RequestState.ADMITTED,
        RequestState.CANCEL_REQUESTED,
        RequestState.TIMED_OUT,
        RequestState.REJECTED,
        RequestState.FAILED,
    ),
    RequestState.ADMITTED: (
        RequestState.PREFILLING,
        RequestState.CANCEL_REQUESTED,
        RequestState.PREEMPTED,
        RequestState.FAILED,
    ),
    RequestState.PREFILLING: (
        RequestState.DECODING,
        RequestState.CANCEL_REQUESTED,
        RequestState.TIMED_OUT,
        RequestState.PREEMPTED,
        RequestState.FINISHED,
        RequestState.FAILED,
    ),
    RequestState.DECODING: (
        RequestState.FINISHED,
        RequestState.CANCEL_REQUESTED,
        RequestState.TIMED_OUT,
        RequestState.PREEMPTED,
        RequestState.FAILED,
    ),
    RequestState.PREEMPTED: (
        RequestState.RESUMED,
        RequestState.CANCEL_REQUESTED,
        RequestState.FAILED,
        RequestState.TIMED_OUT,
    ),
    RequestState.RESUMED: (RequestState.PREFILLING, RequestState.DECODING, RequestState.FAILED),
    RequestState.CANCEL_REQUESTED: (RequestState.CANCELLED, RequestState.FAILED),
    RequestState.FINISHED: (RequestState.CLEANED,),
    RequestState.CANCELLED: (RequestState.CLEANED,),
    RequestState.TIMED_OUT: (RequestState.CLEANED,),
    RequestState.FAILED: (RequestState.CLEANED,),
    RequestState.REJECTED: (RequestState.CLEANED,),
    RequestState.CLEANED: (),
}

TERMINAL_STATES = (
    RequestState.FINISHED,
    RequestState.CANCELLED,
    RequestState.TIMED_OUT,
    RequestState.FAILED,
    RequestState.REJECTED,
)


@dataclass(frozen=True)
class StateTransitionRecord:
    """One state change with everything a postmortem needs (§7)."""

    request_id: str
    old: str
    new: str
    timestamp_ns: int
    scheduler_iteration: int = -1
    reason: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    kv_block_ids: Tuple[int, ...] = ()
    kv_refcounts: Tuple[int, ...] = ()
    batch_id: str = ""
    model_runner_span: str = ""
    error: str = ""
    cleanup: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "old": self.old,
            "new": self.new,
            "timestamp_ns": self.timestamp_ns,
            "scheduler_iteration": self.scheduler_iteration,
            "reason": self.reason,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "kv_block_ids": list(self.kv_block_ids),
            "kv_refcounts": list(self.kv_refcounts),
            "batch_id": self.batch_id,
            "model_runner_span": self.model_runner_span,
            "error": self.error,
            "cleanup": self.cleanup,
        }


class RequestStateMachine:
    """Validates transitions and records them; refuses an illegal path."""

    def __init__(self, request_id: str, *, clock_ns: int = 0) -> None:
        if not request_id:
            raise ConfigError("state machine needs a request id")
        self.request_id = request_id
        self.state = RequestState.CREATED
        self.transitions: List[StateTransitionRecord] = []
        self._clock = clock_ns

    def transition(self, new_state: str, **kwargs: Any) -> StateTransitionRecord:
        allowed = ALLOWED_TRANSITIONS.get(self.state, ())
        if new_state not in allowed:
            raise SchemaError(
                f"illegal request transition {self.state} → {new_state} for "
                f"{self.request_id}; allowed: {allowed}",
                details={"request_id": self.request_id, "state": self.state},
            )
        self._clock += 1
        record = StateTransitionRecord(
            request_id=self.request_id,
            old=self.state,
            new=new_state,
            timestamp_ns=kwargs.pop("timestamp_ns", self._clock),
            **kwargs,
        )
        self.transitions.append(record)
        self.state = new_state
        return record

    # ── invariants ────────────────────────────────────────────────────────

    def cancel_seen(self) -> bool:
        return any(
            item.new == RequestState.CANCEL_REQUESTED for item in self.transitions
        )

    def re_admitted_after_cancel(self) -> bool:
        """A cancelled/timed-out request must never be admitted again (§4)."""
        seen_terminal = False
        for item in self.transitions:
            if item.new in (RequestState.CANCELLED, RequestState.TIMED_OUT):
                seen_terminal = True
            if seen_terminal and item.new in (
                RequestState.ADMITTED,
                RequestState.PREFILLING,
                RequestState.DECODING,
            ):
                return True
        return False

    def emitted_after_cancel(self, token_events: Sequence[Mapping[str, Any]]) -> List[int]:
        """Token indices emitted strictly after the cancel was observed (§4)."""
        cancel_stamp = next(
            (
                item.timestamp_ns
                for item in self.transitions
                if item.new == RequestState.CANCEL_REQUESTED
            ),
            None,
        )
        if cancel_stamp is None:
            return []
        return [
            int(event["token_index"])
            for event in token_events
            if int(event.get("timestamp_ns", -1)) > cancel_stamp
            and not event.get("in_flight_allowed", False)
        ]

    def require_clean_finish(self) -> None:
        if self.state not in TERMINAL_STATES + (RequestState.CLEANED,):
            raise SchemaError(
                f"request {self.request_id} ended in {self.state}, which is not a "
                "terminal state; a request left mid-flight hides a cleanup bug",
                details={"request_id": self.request_id, "state": self.state},
            )
        if self.re_admitted_after_cancel():
            raise SchemaError(
                f"request {self.request_id} was re-admitted after cancel/timeout",
                details={"request_id": self.request_id},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "state": self.state,
            "transitions": [item.as_dict() for item in self.transitions],
        }


# ── spans (details README §3) ──────────────────────────────────────────────

SPAN_KINDS: Tuple[str, ...] = (
    "request",
    "validate",
    "scheduler_wait",
    "scheduler_iteration",
    "schedule_cpu",
    "kv_lookup_allocate_free",
    "prepare_batch",
    "model_runner",
    "input_pack",
    "graph_route",
    "attention",
    "mlp_custom_quant",
    "sample",
    "emit_token",
    "cleanup",
)

#: Attribute names whose values must never be written into a trace (E07-02 §12:
#: "敏感 prompt 写入 trace").
_SENSITIVE_KEYS = ("prompt", "prompt_text", "text", "token_text", "messages", "raw_text")


def redact_attributes(attributes: Mapping[str, Any]) -> Dict[str, Any]:
    """Replace sensitive attribute values with a length-only placeholder."""
    redacted: Dict[str, Any] = {}
    for name, value in attributes.items():
        if name.lower() in _SENSITIVE_KEYS:
            redacted[name] = f"<redacted:{len(str(value))}>"
        elif isinstance(value, str) and len(value) > 512:
            redacted[name] = f"<truncated:{len(value)}>"
        else:
            redacted[name] = value
    return redacted


@dataclass(frozen=True)
class Span:
    """One instrumented interval with its trace context and source symbol."""

    span_id: str
    kind: str
    request_id: str = ""
    parent_span_id: str = ""
    batch_id: str = ""
    iteration: int = -1
    start_ns: int = 0
    end_ns: int = 0
    source_symbol: str = ""
    source_file: str = ""
    source_commit: str = ""
    thread: str = ""
    process: str = ""
    stream: str = ""
    device_events: Tuple[str, ...] = ()
    state_before: str = ""
    state_after: str = ""
    token_index: Optional[int] = None
    kv_blocks: Tuple[int, ...] = ()
    error: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in SPAN_KINDS:
            raise SchemaError(
                f"unknown span kind {self.kind!r}; add it to SPAN_KINDS so the C7 "
                "projection stays complete",
                details={"field": "kind"},
            )
        if self.end_ns < self.start_ns:
            raise SchemaError(
                f"span {self.span_id} ends before it starts",
                details={"span_id": self.span_id},
            )

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    def as_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind,
            "request_id": self.request_id,
            "batch_id": self.batch_id,
            "iteration": self.iteration,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.duration_ns,
            "source_symbol": self.source_symbol,
            "source_file": self.source_file,
            "source_commit": self.source_commit,
            "thread": self.thread,
            "process": self.process,
            "stream": self.stream,
            "device_events": list(self.device_events),
            "state_before": self.state_before,
            "state_after": self.state_after,
            "token_index": self.token_index,
            "kv_blocks": list(self.kv_blocks),
            "error": self.error,
            "attributes": redact_attributes(self.attributes),
        }


@dataclass
class SpanCollector:
    """Monotonic span store with an explicit (never thread-local) context."""

    run_id: str
    spans: List[Span] = field(default_factory=list)
    _clock: int = 0

    def emit(self, kind: str, **kwargs: Any) -> Span:
        self._clock += 1
        span_id = kwargs.pop("span_id", "") or f"{kind}-{self._clock}"
        start_ns = kwargs.pop("start_ns", self._clock)
        end_ns = kwargs.pop("end_ns", start_ns)
        if "attributes" in kwargs:
            kwargs["attributes"] = redact_attributes(kwargs["attributes"])
        span = Span(span_id=span_id, kind=kind, start_ns=start_ns, end_ns=end_ns, **kwargs)
        self.spans.append(span)
        self._clock = max(self._clock, end_ns)
        return span

    def for_request(self, request_id: str) -> List[Span]:
        return [span for span in self.spans if span.request_id == request_id]

    def children_of(self, span_id: str) -> List[Span]:
        return [span for span in self.spans if span.parent_span_id == span_id]

    def join_audit(self) -> Dict[str, Any]:
        """Verify the chain can actually be walked (E07-02 §13 item 12)."""
        by_id = {span.span_id: span for span in self.spans}
        dangling = [
            span.span_id
            for span in self.spans
            if span.parent_span_id and span.parent_span_id not in by_id
        ]
        mixed_request = [
            span.span_id
            for span in self.spans
            if span.parent_span_id
            and by_id.get(span.parent_span_id) is not None
            and by_id[span.parent_span_id].request_id != span.request_id
        ]
        missing_symbol = [
            span.span_id for span in self.spans if not span.source_symbol
        ]
        return {
            "ok": not dangling and not mixed_request and not missing_symbol,
            "spans": len(self.spans),
            "dangling_parent": dangling,
            "cross_request_parent": mixed_request,
            "missing_source_symbol": missing_symbol,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "spans": [span.as_dict() for span in self.spans],
            "join_audit": self.join_audit(),
        }


# ── iteration ledger (details README §9) ───────────────────────────────────


@dataclass(frozen=True)
class IterationLedgerEntry:
    """One scheduler iteration; the shared fact table of E07-02…E07-07."""

    iteration: int
    waiting: Tuple[str, ...] = ()
    running: Tuple[str, ...] = ()
    preempted: Tuple[str, ...] = ()
    finished: Tuple[str, ...] = ()
    scheduled_tokens: Mapping[str, int] = field(default_factory=dict)
    prefill_tokens: int = 0
    decode_tokens: int = 0
    spec_tokens: int = 0
    token_budget: int = 0
    sequence_budget: int = 0
    kv_free_blocks: int = 0
    kv_used_blocks: int = 0
    kv_cached_blocks: int = 0
    prefix_hits: int = 0
    graph_bucket: str = ""
    attention_backend: str = ""
    model_input_m: int = 0
    scheduler_cpu_ms: float = 0.0
    model_runner_ms: float = 0.0
    sample_ms: float = 0.0
    kernels: Tuple[str, ...] = ()
    oom_or_preemption: str = ""
    previous_computed_positions: int = 0
    new_computed_positions: int = 0
    rollback_positions: int = 0

    @property
    def scheduled_total(self) -> int:
        return sum(int(value) for value in self.scheduled_tokens.values())

    @property
    def budget_utilization(self) -> float:
        if self.token_budget <= 0:
            raise ConfigError("token_budget must be positive to compute utilization")
        return self.scheduled_total / self.token_budget

    def conservation_residual(self) -> int:
        """``previous + scheduled - rollback - new``; must be zero (E07-02 §6)."""
        return (
            self.previous_computed_positions
            + self.scheduled_total
            - self.rollback_positions
            - self.new_computed_positions
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "waiting": list(self.waiting),
            "running": list(self.running),
            "preempted": list(self.preempted),
            "finished": list(self.finished),
            "scheduled_tokens": dict(self.scheduled_tokens),
            "scheduled_total": self.scheduled_total,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "spec_tokens": self.spec_tokens,
            "token_budget": self.token_budget,
            "sequence_budget": self.sequence_budget,
            "budget_utilization": self.budget_utilization,
            "kv_free_blocks": self.kv_free_blocks,
            "kv_used_blocks": self.kv_used_blocks,
            "kv_cached_blocks": self.kv_cached_blocks,
            "prefix_hits": self.prefix_hits,
            "graph_bucket": self.graph_bucket,
            "attention_backend": self.attention_backend,
            "model_input_m": self.model_input_m,
            "scheduler_cpu_ms": self.scheduler_cpu_ms,
            "model_runner_ms": self.model_runner_ms,
            "sample_ms": self.sample_ms,
            "kernels": list(self.kernels),
            "oom_or_preemption": self.oom_or_preemption,
            "previous_computed_positions": self.previous_computed_positions,
            "new_computed_positions": self.new_computed_positions,
            "rollback_positions": self.rollback_positions,
            "conservation_residual": self.conservation_residual(),
        }


def conservation_audit(
    entries: Sequence[IterationLedgerEntry],
) -> Dict[str, Any]:
    """Every iteration must satisfy the token-conservation identity."""
    offenders = [
        {
            "iteration": entry.iteration,
            "residual": entry.conservation_residual(),
            "scheduled_total": entry.scheduled_total,
        }
        for entry in entries
        if entry.conservation_residual() != 0
    ]
    return {"ok": not offenders, "offenders": offenders, "iterations": len(entries)}


# ── clocks (details README §4) ─────────────────────────────────────────────


@dataclass(frozen=True)
class ClockCalibration:
    """Host↔device clock relation; a delta is refused without one."""

    host_reference_ns: int
    device_reference_ns: int
    skew_ns: int
    method: str

    def __post_init__(self) -> None:
        if not self.method:
            raise ConfigError(
                "clock calibration must name its method; subtracting two different "
                "clocks without calibration is a listed anti-pattern (E07-02 §12)"
            )
        if abs(self.skew_ns) < 0:
            raise ConfigError("skew must be a signed value")

    def device_to_host_ns(self, device_ns: int) -> int:
        """Map a device timestamp onto the host timeline."""
        return device_ns + (self.host_reference_ns - self.device_reference_ns)

    def device_span_ns(self, start_device_ns: int, end_device_ns: int) -> int:
        """A device-internal span needs no host correction, only ordering."""
        span = end_device_ns - start_device_ns
        if span < 0:
            raise ConfigError("device span is negative; device events are out of order")
        return span

    def host_device_delta_ns(
        self,
        *,
        host_start_ns: int,
        host_end_ns: int,
        device_start_ns: int,
        device_end_ns: int,
    ) -> int:
        """Difference between the host span and the (calibrated) device span.

        Positive means the host window is longer than the device work — the
        signature of launch gaps and host-side preparation; negative means the
        calibration is wrong and must be redone rather than reported as speedup.
        """
        host_span = host_end_ns - host_start_ns
        if host_span < 0:
            raise ConfigError("host span is negative")
        device_span = self.device_span_ns(device_start_ns, device_end_ns)
        return host_span - device_span - abs(self.skew_ns)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "host_reference_ns": self.host_reference_ns,
            "device_reference_ns": self.device_reference_ns,
            "skew_ns": self.skew_ns,
            "method": self.method,
        }


# ── instrumentation overhead (E07-02 §7) ───────────────────────────────────

INSTRUMENTATION_LEVELS: Tuple[str, ...] = ("off", "minimal", "full", "profiler")


@dataclass(frozen=True)
class OverheadObservation:
    """Cost of instrumentation; a full trace may not be used as a benchmark."""

    level: str
    cpu_ms: float
    gpu_ms: float
    ttft_ms: float
    tpot_ms: float
    log_bytes: int = 0

    def __post_init__(self) -> None:
        if self.level not in INSTRUMENTATION_LEVELS:
            raise ConfigError(
                f"unknown instrumentation level {self.level!r}",
                details={"field": "level", "allowed": list(INSTRUMENTATION_LEVELS)},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "cpu_ms": self.cpu_ms,
            "gpu_ms": self.gpu_ms,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "log_bytes": self.log_bytes,
        }


@dataclass(frozen=True)
class OverheadThresholds:
    """Pre-registered bounds; ordinary timing runs use the minimal level."""

    max_cpu_ratio: float = 0.05
    max_ttft_ratio: float = 0.05
    max_tpot_ratio: float = 0.05

    def __post_init__(self) -> None:
        for name in ("max_cpu_ratio", "max_ttft_ratio", "max_tpot_ratio"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive")


def instrumentation_overhead(
    observations: Sequence[OverheadObservation],
    *,
    thresholds: OverheadThresholds,
) -> Dict[str, Any]:
    """Compare each level against ``off`` and decide which may time a run."""
    by_level = {item.level: item for item in observations}
    if "off" not in by_level:
        raise ConfigError("instrumentation overhead needs an 'off' baseline")
    baseline = by_level["off"]
    rows: List[Dict[str, Any]] = []
    for level in INSTRUMENTATION_LEVELS:
        item = by_level.get(level)
        if item is None:
            rows.append({"level": level, "observed": False, "usable_for_timing": False})
            continue
        row = {
            "level": level,
            "observed": True,
            "cpu_ratio": _ratio(item.cpu_ms, baseline.cpu_ms),
            "ttft_ratio": _ratio(item.ttft_ms, baseline.ttft_ms),
            "tpot_ratio": _ratio(item.tpot_ms, baseline.tpot_ms),
            "log_bytes": item.log_bytes,
        }
        row["usable_for_timing"] = (
            level in ("off", "minimal")
            and row["cpu_ratio"] <= thresholds.max_cpu_ratio
            and row["ttft_ratio"] <= thresholds.max_ttft_ratio
            and row["tpot_ratio"] <= thresholds.max_tpot_ratio
        )
        if level in ("full", "profiler"):
            row["usable_for_timing"] = False
        rows.append(row)
    return {
        "rows": rows,
        "timing_level": next(
            (row["level"] for row in rows if row.get("usable_for_timing")), ""
        ),
    }


def _ratio(value: float, baseline: float) -> float:
    if baseline <= 0:
        raise ConfigError("overhead ratios need a positive baseline")
    return max(value - baseline, 0.0) / baseline


# ── report builders (generated from raw, never hand-drawn) ─────────────────


def state_machine_graph(machines: Sequence[RequestStateMachine]) -> Dict[str, Any]:
    """Aggregate observed transitions into a graph derived from raw records."""
    edges: Dict[Tuple[str, str], int] = {}
    states = sorted(
        {machine.state for machine in machines}
        | {
            record.new
            for machine in machines
            for record in machine.transitions
        }
        | {RequestState.CREATED}
    )
    for machine in machines:
        for record in machine.transitions:
            edges[(record.old, record.new)] = edges.get((record.old, record.new), 0) + 1
    return {
        "states": states,
        "edges": [
            {"from": old, "to": new, "count": count}
            for (old, new), count in sorted(edges.items())
        ],
        "requests": len(machines),
        "generated_from_raw": True,
    }


def call_chain_tree(spans: Sequence[Span], request_id: str) -> Dict[str, Any]:
    """Nest spans of one request into a tree; dangling parents are reported."""
    by_id = {span.span_id: span for span in spans}
    roots = [
        span
        for span in spans
        if span.request_id == request_id
        and (not span.parent_span_id or span.parent_span_id not in by_id)
    ]
    dangling = [
        span.span_id
        for span in spans
        if span.request_id == request_id
        and span.parent_span_id
        and span.parent_span_id not in by_id
        and span.parent_span_id != ""
    ]

    def build(span: Span) -> Dict[str, Any]:
        children = [
            child
            for child in spans
            if child.parent_span_id == span.span_id and child.request_id == request_id
        ]
        return {
            "span_id": span.span_id,
            "kind": span.kind,
            "symbol": span.source_symbol,
            "duration_ns": span.duration_ns,
            "iteration": span.iteration,
            "children": [build(child) for child in children],
        }

    return {
        "request_id": request_id,
        "roots": [build(root) for root in roots],
        "dangling_parents": dangling,
        "ok": not dangling and bool(roots),
    }


@dataclass(frozen=True)
class HotPathRow:
    """One hot-path row; CPU self/inclusive and GPU time stay separate."""

    kind: str
    source_symbol: str
    calls: int
    cpu_self_ms: float
    cpu_inclusive_ms: float
    gpu_ms: float
    p95_ms: float
    queue_wait_ms: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "source_symbol": self.source_symbol,
            "calls": self.calls,
            "cpu_self_ms": self.cpu_self_ms,
            "cpu_inclusive_ms": self.cpu_inclusive_ms,
            "gpu_ms": self.gpu_ms,
            "p95_ms": self.p95_ms,
            "queue_wait_ms": self.queue_wait_ms,
        }


def hot_path_table(rows: Sequence[HotPathRow], *, phase: str) -> Dict[str, Any]:
    """Build the hot-path table and flag the call-count fallacy (E07-02 §8)."""
    if phase not in ("prefill", "decode"):
        raise ConfigError("hot-path tables are reported per phase")
    ordered = sorted(rows, key=lambda row: row.cpu_inclusive_ms + row.gpu_ms, reverse=True)
    notes: List[str] = []
    if ordered:
        most_called = max(ordered, key=lambda row: row.calls)
        biggest = ordered[0]
        if most_called.kind != biggest.kind and most_called.calls > 10 * max(
            biggest.calls, 1
        ):
            notes.append(
                f"{most_called.kind} has the highest call count but "
                f"{biggest.kind} dominates wall time; call count is not a hotspot "
                "criterion"
            )
    return {
        "phase": phase,
        "rows": [row.as_dict() for row in ordered],
        "notes": notes,
    }


def kernel_join(
    spans: Sequence[Span], kernel_events: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Attach kernel events to the iteration/span they belong to (E07-02 §12)."""
    span_by_id = {span.span_id: span for span in spans}
    unbound: List[str] = []
    bound: List[Dict[str, Any]] = []
    for event in kernel_events:
        span_id = str(event.get("span_id", ""))
        span = span_by_id.get(span_id)
        if span is None:
            unbound.append(str(event.get("kernel", "")))
            continue
        bound.append(
            {
                "kernel": event.get("kernel", ""),
                "span_id": span_id,
                "kind": span.kind,
                "iteration": span.iteration,
                "batch_id": span.batch_id,
                "device_ns": event.get("device_ns", 0),
            }
        )
    return {"ok": not unbound, "bound": bound, "unbound_kernels": unbound}


__all__ = [
    "ALLOWED_TRANSITIONS",
    "INSTRUMENTATION_LEVELS",
    "IterationLedgerEntry",
    "OverheadObservation",
    "OverheadThresholds",
    "RequestState",
    "RequestStateMachine",
    "SPAN_KINDS",
    "Span",
    "SpanCollector",
    "StateTransitionRecord",
    "TERMINAL_STATES",
    "ClockCalibration",
    "HotPathRow",
    "call_chain_tree",
    "conservation_audit",
    "hot_path_table",
    "instrumentation_overhead",
    "kernel_join",
    "redact_attributes",
    "state_machine_graph",
]
