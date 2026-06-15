"""Load generator: open-loop and closed-loop models, validity gates, calibration.

E08-02's capacity statement is only meaningful if the *offered* load is an
external quantity: an open-loop generator sends at pre-generated ``t_sched``
instants and never waits for the previous response.  A closed-loop generator
(concurrency-bounded) cannot replace it, because a slow service automatically
reduces the arrival rate and hides the cliff.

The generator's own fidelity is part of the evidence: if the client saturates
(lag, drops, CPU/fd limits), the load point is labelled ``LOADGEN_INVALID`` and
may not be attributed to the service.  :func:`noop_calibration` records the
client-side capacity before a formal run, and :func:`merge_client_timestamps`
joins the client clock with the service ledger through an explicit calibration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.serving.arrival import ActualArrival, ArrivalSpec, ArrivalTrace, fidelity_gate
from hqsb.serving.timing import ClockDomain, TimestampLedger

#: The two feedback models (details README §12).
OPEN_LOOP = "open_loop"
CLOSED_LOOP = "closed_loop"

#: Why the client could not keep the schedule.
LOADGEN_FAILURE_REASONS: Tuple[str, ...] = (
    "event_loop_saturated",
    "thread_pool_exhausted",
    "connection_pool_exhausted",
    "fd_limit",
    "cpu_limit",
    "network_limit",
    "parser_backlog",
    "timer_resolution",
)

INVALID_LABEL = "LOADGEN_INVALID"


@dataclass(frozen=True)
class LoadgenSpec:
    """Frozen client configuration (must match A/B runs exactly)."""

    mode: str
    model: str = "dummy-model"
    stream: bool = True
    no_retry: bool = True
    concurrency: int = 0  # closed-loop only
    max_connections: int = 64
    max_open_files: int = 4096
    cpu_budget_ratio: float = 0.5
    max_client_rss_bytes: int = 2 << 30
    timeout_ms: float = 30_000.0
    payload_trace_hash: str = ""
    arrival_trace_hash: str = ""
    topology: str = ""

    def __post_init__(self) -> None:
        if self.mode not in (OPEN_LOOP, CLOSED_LOOP):
            raise ConfigError(f"unknown loadgen mode {self.mode!r}")
        if self.mode == CLOSED_LOOP and self.concurrency <= 0:
            raise ConfigError("closed-loop loadgen needs a positive concurrency")
        if not self.no_retry:
            raise ConfigError(
                "the client must not retry in a capacity run: retries would multiply "
                "the offered load and make the curve a function of the client "
                "(E08-02 §5)"
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "model": self.model,
            "stream": self.stream,
            "no_retry": self.no_retry,
            "concurrency": self.concurrency,
            "max_connections": self.max_connections,
            "max_open_files": self.max_open_files,
            "cpu_budget_ratio": self.cpu_budget_ratio,
            "max_client_rss_bytes": self.max_client_rss_bytes,
            "timeout_ms": self.timeout_ms,
            "payload_trace_hash": self.payload_trace_hash,
            "arrival_trace_hash": self.arrival_trace_hash,
            "topology": self.topology,
        }


@dataclass
class SendRecord:
    """One request as the client actually sent (or failed to send) it."""

    request_index: int
    payload_id: int
    scheduled_ns: int
    sent_ns: Optional[int] = None
    dropped: bool = False
    drop_reason: str = ""
    connection_id: str = ""

    @property
    def lag_ns(self) -> Optional[int]:
        return None if self.sent_ns is None else self.sent_ns - self.scheduled_ns

    def as_actual(self) -> ActualArrival:
        return ActualArrival(
            scheduled_ns=self.scheduled_ns,
            sent_ns=self.sent_ns,
            dropped=self.dropped,
            drop_reason=self.drop_reason,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_index": self.request_index,
            "payload_id": self.payload_id,
            "scheduled_ns": self.scheduled_ns,
            "sent_ns": self.sent_ns,
            "lag_ns": self.lag_ns,
            "dropped": self.dropped,
            "drop_reason": self.drop_reason,
            "connection_id": self.connection_id,
        }


def run_open_loop(
    trace: ArrivalTrace,
    *,
    base_ns: int,
    send: Callable[[int, int, int], Tuple[Optional[int], str]],
) -> List[SendRecord]:
    """Walk the frozen schedule; ``send(index, payload_id, scheduled_ns)``.

    The callback returns ``(sent_ns, reason)``.  A ``None`` timestamp means the
    client could not send in time: the record is kept with its reason rather
    than silently removed from the offered load.
    """
    records: List[SendRecord] = []
    for index, offset in enumerate(trace.scheduled_ns):
        scheduled_ns = base_ns + offset
        payload_id = trace.payload_ids[index] if trace.payload_ids else index
        sent_ns, reason = send(index, payload_id, scheduled_ns)
        records.append(
            SendRecord(
                request_index=index,
                payload_id=payload_id,
                scheduled_ns=scheduled_ns,
                sent_ns=sent_ns,
                dropped=sent_ns is None,
                drop_reason=reason if sent_ns is None else "",
                connection_id=f"conn-{index % 8}",
            )
        )
    return records


@dataclass
class ClosedLoopRecord:
    """One closed-loop iteration: concurrency ≈ completed rate × response time."""

    request_index: int
    payload_id: int
    started_ns: int
    finished_ns: int

    @property
    def response_time_ns(self) -> int:
        return self.finished_ns - self.started_ns

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_index": self.request_index,
            "payload_id": self.payload_id,
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
            "response_time_ns": self.response_time_ns,
        }


def closed_loop_records(
    *, concurrency: int, runs: Sequence[Sequence[Any]]
) -> List[ClosedLoopRecord]:
    """Summarise a fixed-concurrency run (times come from the real run)."""
    if concurrency <= 0:
        raise ConfigError("concurrency must be positive")
    records: List[ClosedLoopRecord] = []
    for index, run in enumerate(runs):
        if len(run) != 3:
            raise ConfigError("each closed-loop run is (payload_id, started_ns, finished_ns)")
        payload_id, started_ns, finished_ns = run
        if int(finished_ns) < int(started_ns):
            raise ConfigError("a closed-loop iteration cannot finish before it starts")
        records.append(
            ClosedLoopRecord(
                request_index=index,
                payload_id=int(payload_id),
                started_ns=int(started_ns),
                finished_ns=int(finished_ns),
            )
        )
    return records


def lock_little_check(records: Sequence[ClosedLoopRecord], *, concurrency: int) -> Dict[str, Any]:
    """Little's relation as a *sanity* check, never as a substitute for evidence."""
    if not records:
        return {"ok": False, "reason": "no closed-loop records"}
    span = max(item.finished_ns for item in records) - min(item.started_ns for item in records)
    if span <= 0:
        return {"ok": False, "reason": "zero-length observation window"}
    mean_response = sum(item.response_time_ns for item in records) / len(records)
    rate = len(records) / (span / 1e9)
    in_system = rate * (mean_response / 1e9)
    return {
        "ok": abs(in_system - concurrency) / concurrency < 0.5,
        "measured_in_system": in_system,
        "configured_concurrency": concurrency,
        "note": (
            "Little's law is a bookkeeping check; it cannot turn a closed-loop run into "
            "an open-loop capacity statement"
        ),
    }


# ── client-side validity ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ClientResources:
    """Client-side resource readings taken during a load point."""

    cpu_ratio: float = 0.0
    rss_bytes: int = 0
    open_fds: int = 0
    connection_count: int = 0
    parser_backlog: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cpu_ratio": self.cpu_ratio,
            "rss_bytes": self.rss_bytes,
            "open_fds": self.open_fds,
            "connection_count": self.connection_count,
            "parser_backlog": self.parser_backlog,
        }


def loadgen_validity(
    records: Sequence[SendRecord],
    *,
    spec: LoadgenSpec,
    resources: ClientResources,
    lag_p95_limit_ms: float = 20.0,
    drop_ratio_limit: float = 0.001,
) -> Dict[str, Any]:
    """A saturated client labels the point; it never explains it away."""
    problems: List[str] = []
    lags = sorted(item.lag_ns for item in records if item.lag_ns is not None)
    lags = [float(value) for value in lags]
    drop_ratio = (
        sum(1 for item in records if item.dropped) / len(records) if records else 0.0
    )
    lag_p95_ms = None
    if lags:
        position = 0.95 * (len(lags) - 1)
        low, high = int(position), min(int(position) + 1, len(lags) - 1)
        lag_p95_ms = (
            lags[low] * (1 - (position - low)) + lags[high] * (position - low)
        ) / 1e6
    if lag_p95_ms is not None and lag_p95_ms > lag_p95_limit_ms:
        problems.append(f"loadgen lag p95 {lag_p95_ms:.2f} ms exceeds the limit")
    if drop_ratio > drop_ratio_limit:
        problems.append(f"drop ratio {drop_ratio:.4f} exceeds the limit")
    if resources.cpu_ratio > spec.cpu_budget_ratio:
        problems.append("client CPU budget exceeded")
    if resources.rss_bytes > spec.max_client_rss_bytes:
        problems.append("client RSS budget exceeded")
    if resources.open_fds > spec.max_open_files:
        problems.append("client file-descriptor budget exceeded")
    if resources.connection_count > spec.max_connections:
        problems.append("client connection budget exceeded")
    if resources.parser_backlog > 0:
        problems.append("client parser backlog is non-zero: the reader is the bottleneck")
    return {
        "valid": not problems,
        "label": "" if not problems else INVALID_LABEL,
        "problems": problems,
        "lag_p95_ms": lag_p95_ms,
        "drop_ratio": drop_ratio,
        "resources": resources.as_dict(),
        "note": (
            "an invalid point is a statement about the load generator, not about the "
            "service; it must be fixed on the client side and re-run"
        ),
    }


def offered_load_funnel(
    trace: ArrivalTrace,
    records: Sequence[SendRecord],
    *,
    spec: ArrivalSpec,
) -> Dict[str, Any]:
    """Intended vs. actual arrival for one load point (E08-03 step 16)."""
    mean_rate = trace.mean_rate
    return fidelity_gate(
        [record.as_actual() for record in records], spec=spec, mean_rate=mean_rate
    )


# ── observer self-verification (details README §25) ────────────────────────


@dataclass(frozen=True)
class NoopCalibrationPoint:
    """One measured client-capacity point against a no-op server."""

    target_qps: float
    achieved_qps: float
    lag_p95_ms: float
    drop_ratio: float
    client_cpu_ratio: float
    parser_errors: int = 0
    timer_resolution_ns: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target_qps": self.target_qps,
            "achieved_qps": self.achieved_qps,
            "lag_p95_ms": self.lag_p95_ms,
            "drop_ratio": self.drop_ratio,
            "client_cpu_ratio": self.client_cpu_ratio,
            "parser_errors": self.parser_errors,
            "timer_resolution_ns": self.timer_resolution_ns,
        }


def noop_calibration(points: Sequence[NoopCalibrationPoint]) -> Dict[str, Any]:
    """How fast can the client itself go, and where does it stop being exact?"""
    problems: List[str] = []
    if not points:
        return {"ok": False, "problems": ["no calibration point recorded"], "ceiling_qps": None}
    for point in points:
        if point.achieved_qps > point.target_qps * 1.05:
            problems.append(
                f"achieved {point.achieved_qps} qps above the target {point.target_qps}"
            )
        if point.parser_errors:
            problems.append("the client SSE parser produced errors against a no-op server")
    ceiling = max(point.achieved_qps for point in points)
    best = [point for point in points if point.achieved_qps == ceiling][0]
    if best.client_cpu_ratio > 0.8:
        problems.append("the client is CPU-bound at its own ceiling")
    return {
        "ok": not problems,
        "problems": problems,
        "ceiling_qps": ceiling,
        "ceiling_lag_p95_ms": best.lag_p95_ms,
        "note": (
            "every service load point must stay clearly below this client ceiling, "
            "otherwise the curve measures the client"
        ),
    }


def service_points_below_client_ceiling(
    offered_qps: Sequence[float], calibration: Mapping[str, Any], *, margin: float = 0.5
) -> Dict[str, Any]:
    ceiling = calibration.get("ceiling_qps")
    if not ceiling:
        return {"ok": False, "reason": "no client ceiling was measured"}
    limit = float(ceiling) * margin
    too_high = [value for value in offered_qps if value > limit]
    return {
        "ok": not too_high,
        "client_ceiling_qps": ceiling,
        "allowed_offered_qps": limit,
        "offered_above_limit": too_high,
        "note": "an offered load too close to the client ceiling cannot prove a service limit",
    }


# ── clock joining ──────────────────────────────────────────────────────────


def merge_client_timestamps(
    ledger: TimestampLedger,
    record: SendRecord,
    *,
    client_clock: ClockDomain,
    service_clock: ClockDomain,
) -> Dict[str, Any]:
    """Join the loadgen's ``t_sched``/``t_send`` with the service ledger.

    The two clocks are different processes: without a measured offset the join
    is refused, and the returned uncertainty is carried into every derived
    duration.
    """
    if ledger.clock_of.get("t_gateway_recv") and not service_clock.calibrated:
        raise ConfigError(
            "cannot join the client and service timelines: the service clock has no "
            "measured offset (E08-10 §6)"
        )
    if not client_clock.calibrated:
        raise ConfigError(
            "cannot join the client and service timelines: the client clock has no "
            "measured offset"
        )
    ledger.record("t_sched", record.scheduled_ns, clock=client_clock.name)
    if record.sent_ns is not None:
        ledger.record("t_send", record.sent_ns, clock=client_clock.name)
    return {
        "client_clock": client_clock.as_dict(),
        "service_clock": service_clock.as_dict(),
        "uncertainty_ns": client_clock.uncertainty_ns + service_clock.uncertainty_ns,
        "loadgen_lag_ms": (
            None if record.lag_ns is None else record.lag_ns / 1e6
        ),
    }


def schedule_hash(records: Sequence[SendRecord]) -> str:
    """Hash the *actual* schedule so a run can be compared with its intention."""
    payload = [
        {
            "i": record.request_index,
            "s": record.scheduled_ns,
            "sent": record.sent_ns,
            "drop": record.drop_reason,
        }
        for record in records
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def payload_trace_hash(payloads: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(list(payloads), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


__all__ = [
    "CLOSED_LOOP",
    "ClientResources",
    "ClosedLoopRecord",
    "INVALID_LABEL",
    "LOADGEN_FAILURE_REASONS",
    "LoadgenSpec",
    "NoopCalibrationPoint",
    "OPEN_LOOP",
    "SendRecord",
    "closed_loop_records",
    "loadgen_validity",
    "lock_little_check",
    "merge_client_timestamps",
    "noop_calibration",
    "offered_load_funnel",
    "payload_trace_hash",
    "run_open_loop",
    "schedule_hash",
    "service_points_below_client_ceiling",
]
