"""Scripted client behaviours: slow readers, stalls, disconnects, resets.

E08-08 requires the *client* side to be scripted and replayable ("不能依赖人工
关终端"): every behaviour is an explicit object, and the module can prove that
the behaviour really happened (delay/bandwidth/stall/disconnect) before a run is
trusted (E08-08 steps 4/6/7/8/9/10/11/14).

:func:`simulate_client_read` is a deterministic *model*, not a measurement: it
derives the client read schedule from the frame ready times and the script.  A
real run uses the same script over a socket; the model is used for interface
self-checks and for planning slow-client ratios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: The scripted behaviours of the S08 workload suite (E08-08 §5).
CLIENT_BEHAVIORS: Tuple[str, ...] = (
    "normal_fast_reader",
    "fixed_delay_reader",
    "token_bucket_bandwidth_reader",
    "periodic_stall_resume",
    "never_read_after_headers",
    "read_n_frames_then_fin",
    "abrupt_reset",
    "half_close_idle_timeout",
    "many_simultaneous_slow_readers",
    "slow_plus_normal_mix",
    "non_stream_control",
)

_MS = 1_000_000


@dataclass(frozen=True)
class ClientBehavior:
    """One scripted client; all timings are explicit and replayable."""

    name: str
    description: str
    read_delay_ms: float = 0.0
    bandwidth_bytes_per_s: float = 0.0  # 0 = unlimited
    stall_period_ms: float = 0.0
    stall_duration_ms: float = 0.0
    never_read_after_headers: bool = False
    read_frames_then_close: int = 0  # 0 = read everything
    abort_after_frames: int = 0  # 0 = graceful
    idle_timeout_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in CLIENT_BEHAVIORS:
            raise ConfigError(
                f"unknown client behaviour {self.name!r}; expected one of "
                f"{list(CLIENT_BEHAVIORS)}"
            )
        if self.read_delay_ms < 0 or self.bandwidth_bytes_per_s < 0:
            raise ConfigError("client timings must not be negative")
        if self.stall_duration_ms > self.stall_period_ms > 0 and self.stall_period_ms == 0:
            raise ConfigError("a stall duration needs a stall period")

    @property
    def is_slow(self) -> bool:
        return bool(
            self.read_delay_ms
            or self.bandwidth_bytes_per_s
            or self.stall_duration_ms
            or self.never_read_after_headers
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "read_delay_ms": self.read_delay_ms,
            "bandwidth_bytes_per_s": self.bandwidth_bytes_per_s,
            "stall_period_ms": self.stall_period_ms,
            "stall_duration_ms": self.stall_duration_ms,
            "never_read_after_headers": self.never_read_after_headers,
            "read_frames_then_close": self.read_frames_then_close,
            "abort_after_frames": self.abort_after_frames,
            "idle_timeout_ms": self.idle_timeout_ms,
            "is_slow": self.is_slow,
        }


def behavior_matrix() -> Dict[str, ClientBehavior]:
    """The frozen behaviour catalogue used by E08-08."""
    return {
        "normal_fast_reader": ClientBehavior(
            "normal_fast_reader", "reads every frame as soon as it is ready"
        ),
        "fixed_delay_reader": ClientBehavior(
            "fixed_delay_reader", "adds a fixed delay per frame", read_delay_ms=50.0
        ),
        "token_bucket_bandwidth_reader": ClientBehavior(
            "token_bucket_bandwidth_reader",
            "reads at a fixed byte rate",
            bandwidth_bytes_per_s=4096.0,
        ),
        "periodic_stall_resume": ClientBehavior(
            "periodic_stall_resume",
            "stalls periodically and resumes",
            stall_period_ms=200.0,
            stall_duration_ms=100.0,
        ),
        "never_read_after_headers": ClientBehavior(
            "never_read_after_headers",
            "never reads the body",
            never_read_after_headers=True,
        ),
        "read_n_frames_then_fin": ClientBehavior(
            "read_n_frames_then_fin", "reads N frames then closes gracefully",
            read_frames_then_close=3,
        ),
        "abrupt_reset": ClientBehavior(
            "abrupt_reset", "resets the connection mid-stream", abort_after_frames=2
        ),
        "half_close_idle_timeout": ClientBehavior(
            "half_close_idle_timeout",
            "half-closes and idles until the idle timeout",
            idle_timeout_ms=1000.0,
        ),
        "many_simultaneous_slow_readers": ClientBehavior(
            "many_simultaneous_slow_readers",
            "the fixed-delay behaviour applied to many connections",
            read_delay_ms=100.0,
        ),
        "slow_plus_normal_mix": ClientBehavior(
            "slow_plus_normal_mix",
            "a slow minority next to fast readers (collateral check)",
            read_delay_ms=80.0,
        ),
        "non_stream_control": ClientBehavior(
            "non_stream_control", "non-stream control: one body, no frames"
        ),
    }


@dataclass(frozen=True)
class ClientScript:
    """A behaviour applied to one connection with an explicit identity."""

    connection_id: str
    behavior: ClientBehavior
    headers_received_ns: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "behavior": self.behavior.as_dict(),
            "headers_received_ns": self.headers_received_ns,
        }


@dataclass
class ClientReadReport:
    """What the simulated client did (a model output, labelled simulated)."""

    connection_id: str
    behavior: str
    simulated: bool = True
    frames_read: int = 0
    bytes_read: int = 0
    client_read_ns: Tuple[int, ...] = ()
    disconnect_ns: Optional[int] = None
    disconnect_reason: str = ""
    stall_windows_applied: int = 0
    buffered_bytes_at_end: int = 0
    effects_observed: Mapping[str, bool] = None  # type: ignore[assignment]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "behavior": self.behavior,
            "simulated": self.simulated,
            "frames_read": self.frames_read,
            "bytes_read": self.bytes_read,
            "client_read_ns": list(self.client_read_ns),
            "disconnect_ns": self.disconnect_ns,
            "disconnect_reason": self.disconnect_reason,
            "stall_windows_applied": self.stall_windows_applied,
            "buffered_bytes_at_end": self.buffered_bytes_at_end,
            "effects_observed": dict(self.effects_observed or {}),
        }


def simulate_client_read(
    frames: Sequence[Tuple[int, bytes, int]],
    script: ClientScript,
) -> ClientReadReport:
    """Apply the script to ``(frame_index, raw_bytes, ready_ns)`` observations.

    The function is deterministic and side-effect free; it is a planning /
    self-check model and must never be quoted as a measured client timeline.
    """
    behavior = script.behavior
    read_times: List[int] = []
    bytes_read = 0
    disconnect_ns: Optional[int] = None
    disconnect_reason = ""
    stalls = 0
    t = max(script.headers_received_ns, frames[0][2] if frames else 0)
    pending_bytes = 0

    for position, (index, raw, ready_ns) in enumerate(frames):
        t = max(t, ready_ns)
        pending_bytes += len(raw)
        if behavior.never_read_after_headers:
            disconnect_ns = disconnect_ns or t + int(behavior.idle_timeout_ms * _MS)
            disconnect_reason = disconnect_reason or "never_read_after_headers"
            break
        if behavior.abort_after_frames and position >= behavior.abort_after_frames:
            disconnect_ns = t
            disconnect_reason = "abrupt_reset"
            break
        if behavior.read_frames_then_close and position >= behavior.read_frames_then_close:
            disconnect_ns = t
            disconnect_reason = "graceful_fin"
            break
        if behavior.stall_period_ms > 0:
            phase = (t / _MS) % behavior.stall_period_ms
            if phase < behavior.stall_duration_ms:
                wait_ms = behavior.stall_duration_ms - phase
                t += int(wait_ms * _MS)
                stalls += 1
        service_ms = behavior.read_delay_ms
        if behavior.bandwidth_bytes_per_s:
            service_ms += len(raw) / behavior.bandwidth_bytes_per_s * 1000.0
        t += int(service_ms * _MS)
        read_times.append(t)
        bytes_read += len(raw)
        pending_bytes -= len(raw)

    effects = {
        "delay_applied": bool(behavior.read_delay_ms) and len(read_times) > 0,
        "bandwidth_limited": bool(behavior.bandwidth_bytes_per_s),
        "stalled": stalls > 0,
        "disconnected": disconnect_ns is not None,
        "never_read": behavior.never_read_after_headers,
    }
    if behavior.read_delay_ms and read_times:
        first_ready = frames[0][2]
        effects["delay_applied"] = read_times[0] - first_ready >= int(
            behavior.read_delay_ms * _MS
        ) - _MS
    return ClientReadReport(
        connection_id=script.connection_id,
        behavior=behavior.name,
        frames_read=len(read_times),
        bytes_read=bytes_read,
        client_read_ns=tuple(read_times),
        disconnect_ns=disconnect_ns,
        disconnect_reason=disconnect_reason,
        stall_windows_applied=stalls,
        buffered_bytes_at_end=max(0, pending_bytes),
        effects_observed=effects,
    )


def verify_script_effects(
    report: ClientReadReport,
    *,
    expected: Mapping[str, bool],
) -> Dict[str, Any]:
    """Step 4 of E08-08: prove the behaviour actually took place."""
    problems: List[str] = []
    observed = dict(report.effects_observed or {})
    for key, required in expected.items():
        if required and not observed.get(key, False):
            problems.append(f"behaviour {report.behavior!r} did not produce {key!r}")
    return {"ok": not problems, "problems": problems, "observed": observed}


def slow_ratio_plan(
    *,
    total_connections: int,
    slow_ratios: Sequence[float],
) -> List[Dict[str, Any]]:
    """Plan the slow/normal mix used by the collateral check (E08-08 step 14)."""
    plan: List[Dict[str, Any]] = []
    for ratio in slow_ratios:
        if not 0.0 <= ratio <= 1.0:
            raise ConfigError(f"slow ratio {ratio} is outside [0, 1]")
        slow = int(round(total_connections * ratio))
        plan.append(
            {
                "slow_ratio": ratio,
                "slow_connections": slow,
                "normal_connections": total_connections - slow,
            }
        )
    return plan


def collateral_report(
    *,
    normal_requests: Sequence[Mapping[str, Any]],
    slow_requests: Sequence[Mapping[str, Any]],
    metric: str,
) -> Dict[str, Any]:
    """Compare the collateral damage on normal requests (E08-08 §9)."""

    def _quantiles(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        values = sorted(float(row.get(metric, 0.0)) for row in rows)
        if not values:
            return {"count": 0, "p50": None, "p95": None, "p99": None}

        def quantile(q: float) -> float:
            if len(values) == 1:
                return values[0]
            position = q * (len(values) - 1)
            low = int(position)
            high = min(low + 1, len(values) - 1)
            fraction = position - low
            return values[low] * (1 - fraction) + values[high] * fraction

        return {
            "count": len(values),
            "p50": quantile(0.5),
            "p95": quantile(0.95),
            "p99": quantile(0.99),
        }

    return {
        "metric": metric,
        "normal": _quantiles(normal_requests),
        "slow": _quantiles(slow_requests),
        "note": (
            "a slow client that only hurts itself is not evidence of isolation; the "
            "normal-client tail/goodput is the collateral that matters"
        ),
    }


__all__ = [
    "CLIENT_BEHAVIORS",
    "ClientBehavior",
    "ClientReadReport",
    "ClientScript",
    "behavior_matrix",
    "collateral_report",
    "simulate_client_read",
    "slow_ratio_plan",
    "verify_script_effects",
]
