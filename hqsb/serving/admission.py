"""Admission, backpressure, overload rejection and retry budgets (E08-05).

The point of this module is that the service **says no before the machine does**:
every queue has a hard cap, admission decides from signals that exist at that
moment (never from the future output length), the pressure state machine has
hysteresis and a minimum dwell so it cannot oscillate, and retries carry a
budget so they cannot multiply the offered load into a storm.

The unbounded queue exists only as a *guarded counter-example*: it refuses to
run longer than its guard, and it exists so the experiment can show what
unbounded admission does — not so the service can be run that way.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Pressure states (details README §17).
PRESSURE_STATES: Tuple[str, ...] = (
    "NORMAL",
    "SOFT_PRESSURE",
    "HARD_PRESSURE",
    "SHEDDING",
    "RECOVERING",
)

#: Reject reasons and the code each one maps to in the frozen catalog.
REJECT_CODES: Mapping[str, str] = {
    "queue_full": "queue_full",
    "token_budget": "token_budget_exceeded",
    "kv_budget": "kv_budget_exceeded",
    "deadline": "deadline_infeasible",
    "health": "backend_unavailable",
    "tenant_rate": "tenant_rate_limited",
    "shedding": "service_overloaded",
}

#: Safety invariants of the overload policy (§11).
SAFETY_INVARIANTS: Tuple[str, ...] = (
    "every_queue_has_a_hard_cap",
    "admission_reject_creates_no_backend_or_kv_state",
    "expired_requests_do_not_keep_running",
    "shed_order_follows_the_fairness_policy",
    "rejects_do_not_change_model_or_quality_semantics",
    "retries_never_duplicate_output_tokens",
    "pressure_state_returns_to_normal_after_load_drops",
    "health_endpoint_is_not_starved_by_the_business_queue",
    "emergency_stop_leaves_no_half_state",
)


@dataclass(frozen=True)
class PressureTransition:
    """One declared edge of the pressure state machine."""

    source: str
    target: str
    signal: str
    threshold: float
    direction: str  # "enter" (>=) or "exit" (<=)

    def __post_init__(self) -> None:
        if self.source not in PRESSURE_STATES or self.target not in PRESSURE_STATES:
            raise ConfigError(f"unknown pressure state in {self.source} -> {self.target}")
        if self.direction not in ("enter", "exit"):
            raise ConfigError("direction must be 'enter' or 'exit'")

    def satisfied(self, value: float) -> bool:
        if self.direction == "enter":
            return value >= self.threshold
        return value <= self.threshold

    def as_dict(self) -> Dict[str, Any]:
        return {
            "from": self.source,
            "to": self.target,
            "signal": self.signal,
            "threshold": self.threshold,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class PressureSpec:
    """Thresholds, hysteresis and dwell of the frozen admission spec."""

    transitions: Tuple[PressureTransition, ...]
    min_dwell_windows: int
    window_ms: float
    hard_caps: Mapping[str, int]
    recovery_ramp_windows: int
    retry: Mapping[str, Any]

    @classmethod
    def from_document(cls, payload: Mapping[str, Any]) -> "PressureSpec":
        transitions: List[PressureTransition] = []
        for raw in payload["transitions"]:
            item = dict(raw)
            transitions.append(
                PressureTransition(
                    source=str(item["from"]),
                    target=str(item["to"]),
                    signal=str(item["enter_signal"] if "enter_signal" in item else item["exit_signal"]),
                    threshold=float(item["enter_threshold"] if "enter_threshold" in item else item["exit_threshold"]),
                    direction="enter" if "enter_signal" in item else "exit",
                )
            )
        return cls(
            transitions=tuple(transitions),
            min_dwell_windows=int(payload["min_dwell_windows"]),
            window_ms=float(payload["window_ms"]),
            hard_caps={str(k): int(v) for k, v in dict(payload["hard_caps"]).items() if isinstance(v, (int, float))},
            recovery_ramp_windows=int(payload["recovery"]["ramp_windows"]),
            retry=dict(payload["retry_policy"]),
        )


@dataclass(frozen=True)
class PressureSignal:
    """The signals a decision may look at (a snapshot, not a history)."""

    queue_depth_ratio: float = 0.0
    queue_age_ms: float = 0.0
    inflight_requests: int = 0
    inflight_tokens: int = 0
    predicted_kv_ratio: float = 0.0
    deadline_slack_ms: float = 0.0
    recent_slo_violation_ratio: float = 0.0
    buffered_socket_bytes: int = 0
    process_rss_bytes: int = 0
    backend_health: str = "healthy"

    def value(self, name: str) -> float:
        if not hasattr(self, name):
            raise ConfigError(
                f"signal {name!r} is not part of the frozen pressure signal set",
                details={"signal": name},
            )
        return float(getattr(self, name))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "queue_depth_ratio": self.queue_depth_ratio,
            "queue_age_ms": self.queue_age_ms,
            "inflight_requests": self.inflight_requests,
            "inflight_tokens": self.inflight_tokens,
            "predicted_kv_ratio": self.predicted_kv_ratio,
            "deadline_slack_ms": self.deadline_slack_ms,
            "recent_slo_violation_ratio": self.recent_slo_violation_ratio,
            "buffered_socket_bytes": self.buffered_socket_bytes,
            "process_rss_bytes": self.process_rss_bytes,
            "backend_health": self.backend_health,
        }


@dataclass
class PressureEvent:
    monotonic_ns: int
    source: str
    target: str
    signal: str
    value: float
    reason: str
    dwell_windows: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "monotonic_ns": self.monotonic_ns,
            "from": self.source,
            "to": self.target,
            "signal": self.signal,
            "value": self.value,
            "reason": self.reason,
            "dwell_windows": self.dwell_windows,
        }


class PressureStateMachine:
    """Enter/exit with hysteresis and a minimum dwell in *windows*, not samples."""

    def __init__(self, spec: PressureSpec, *, state: str = "NORMAL") -> None:
        if state not in PRESSURE_STATES:
            raise ConfigError(f"unknown pressure state {state!r}")
        self.spec = spec
        self.state = state
        self.events: List[PressureEvent] = []
        self._dwell = 0
        self._pending: Optional[PressureTransition] = None

    def observe(self, signal: PressureSignal, *, monotonic_ns: int) -> List[PressureEvent]:
        produced: List[PressureEvent] = []
        candidates = [item for item in self.spec.transitions if item.source == self.state]
        chosen: Optional[PressureTransition] = None
        for item in candidates:
            if item.satisfied(signal.value(item.signal)):
                chosen = item
                break
        if chosen is None:
            self._pending = None
            self._dwell = 0
            return produced
        if self._pending is not chosen:
            self._pending = chosen
            self._dwell = 1
            return produced
        self._dwell += 1
        if self._dwell < self.spec.min_dwell_windows:
            return produced
        event = PressureEvent(
            monotonic_ns=monotonic_ns,
            source=self.state,
            target=chosen.target,
            signal=chosen.signal,
            value=signal.value(chosen.signal),
            reason=(
                f"{chosen.direction} {chosen.signal} threshold {chosen.threshold} held for "
                f"{self._dwell} windows"
            ),
            dwell_windows=self._dwell,
        )
        self.state = chosen.target
        self.events.append(event)
        produced.append(event)
        self._pending = None
        self._dwell = 0
        return produced

    def oscillation_report(self) -> Dict[str, Any]:
        """Rapid direction changes indicate a missing hysteresis (§11)."""
        reversals = 0
        for previous, current in zip(self.events, self.events[1:]):
            if previous.target == current.source and current.target == previous.source:
                reversals += 1
        return {
            "transitions": len(self.events),
            "reversals": reversals,
            "oscillating": reversals > max(1, len(self.events) // 4),
            "events": [event.as_dict() for event in self.events],
            "note": "threshold changes after seeing the curve are forbidden; fix the hysteresis instead",
        }


# ── admission cost model ───────────────────────────────────────────────────


@dataclass(frozen=True)
class AdmissionRequest:
    """Only the fields visible *before* execution may be used."""

    request_id: str
    tenant: str
    request_class: str
    priority: str
    input_tokens: int
    max_tokens: int
    expected_prefix_hit: int = 0
    deadline_ns: Optional[int] = None
    class_default_output: int = 256
    history_mean_output: Optional[float] = None


@dataclass(frozen=True)
class AdmissionEstimate:
    estimated_prefill_tokens: int
    estimated_decode_tokens: float
    estimated_kv_tokens: int
    deadline_slack_ms: Optional[float]
    feasible: bool
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "estimated_prefill_tokens": self.estimated_prefill_tokens,
            "estimated_decode_tokens": self.estimated_decode_tokens,
            "estimated_kv_tokens": self.estimated_kv_tokens,
            "deadline_slack_ms": self.deadline_slack_ms,
            "feasible": self.feasible,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AdmissionCostModel:
    """The frozen estimate (never an oracle: the real OSL is unknown)."""

    conservative: bool = True
    class_default_output: int = 256
    kv_bytes_per_token: float = 0.0

    def estimate(
        self,
        request: AdmissionRequest,
        *,
        now_ns: int,
        estimated_queue_ms: float = 0.0,
        estimated_service_ms: float = 0.0,
    ) -> AdmissionEstimate:
        prefill = max(0, request.input_tokens - request.expected_prefix_hit)
        reserve = min(request.max_tokens, request.class_default_output)
        if request.history_mean_output is not None:
            # history is allowed: it exists before the request runs
            reserve = int(min(request.max_tokens, math.ceil(request.history_mean_output)))
        kv_tokens = request.input_tokens + reserve
        slack_ms: Optional[float] = None
        feasible = True
        reason = ""
        if request.deadline_ns is not None:
            remaining_ms = (request.deadline_ns - now_ns) / 1e6
            slack_ms = remaining_ms - estimated_queue_ms - estimated_service_ms
            if slack_ms < 0:
                feasible = False
                reason = "estimated queue plus service exceeds the remaining deadline"
        return AdmissionEstimate(
            estimated_prefill_tokens=prefill,
            estimated_decode_tokens=float(reserve),
            estimated_kv_tokens=kv_tokens,
            deadline_slack_ms=slack_ms,
            feasible=feasible,
            reason=reason,
        )

    def calibration(self, pairs: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
        """Predicted vs. actual cost/OSL: over- and under-estimation both hurt."""
        if not pairs:
            return {"status": "NO_PAIRS", "note": "calibration needs completed requests"}
        signed = [actual - predicted for predicted, actual in pairs]
        over = sum(1 for value in signed if value < 0)
        under = sum(1 for value in signed if value > 0)
        return {
            "status": "measured",
            "pairs": len(pairs),
            "mean_error": sum(signed) / len(signed),
            "over_estimate_ratio": over / len(signed),
            "under_estimate_ratio": under / len(signed),
            "note": (
                "over-estimation wastes capacity, under-estimation risks OOM/SLO "
                "violation; both must appear in the calibration curve"
            ),
        }


# ── bounded queue and admission decisions ──────────────────────────────────


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    code: str
    reason: str
    queue_class: str = ""
    queue_depth: int = 0
    queue_position: int = 0
    retry_after_ms: int = 0
    backlog_slope_per_sec: float = 0.0
    created_backend_state: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "admitted": self.admitted,
            "code": self.code,
            "reason": self.reason,
            "queue_class": self.queue_class,
            "queue_depth": self.queue_depth,
            "queue_position": self.queue_position,
            "retry_after_ms": self.retry_after_ms,
            "backlog_slope_per_sec": self.backlog_slope_per_sec,
            "created_backend_state": self.created_backend_state,
        }


class BoundedQueue:
    """Hard caps on requests, tokens and bytes (the safety invariant)."""

    def __init__(
        self,
        *,
        max_requests: int,
        max_tokens: int,
        max_bytes: int,
        token_aware: bool = True,
    ) -> None:
        for name, value in (
            ("max_requests", max_requests),
            ("max_tokens", max_tokens),
            ("max_bytes", max_bytes),
        ):
            if value <= 0:
                raise ConfigError(
                    f"{name} must be positive: an unbounded queue is a counter-example, "
                    "not a policy (E08-05 §11)"
                )
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.max_bytes = max_bytes
        self.token_aware = token_aware
        self.requests: List[str] = []
        self.queued_tokens = 0
        self.queued_bytes = 0

    @property
    def depth(self) -> int:
        return len(self.requests)

    @property
    def depth_ratio(self) -> float:
        return self.depth / self.max_requests

    def try_admit(
        self,
        *,
        request_id: str,
        token_cost: int,
        byte_cost: int = 0,
        timed_out: bool = False,
    ) -> AdmissionDecision:
        if timed_out:
            return AdmissionDecision(
                admitted=False,
                code=REJECT_CODES["deadline"],
                reason="the request is already past its deadline",
            )
        if self.depth >= self.max_requests:
            return AdmissionDecision(
                admitted=False,
                code=REJECT_CODES["queue_full"],
                reason=f"queue is full ({self.depth}/{self.max_requests} requests)",
                queue_depth=self.depth,
                retry_after_ms=1000,
            )
        if self.token_aware:
            if self.queued_tokens + token_cost > self.max_tokens:
                return AdmissionDecision(
                    admitted=False,
                    code=REJECT_CODES["token_budget"],
                    reason=(
                        f"token budget: queued {self.queued_tokens} + {token_cost} > "
                        f"{self.max_tokens}"
                    ),
                    queue_depth=self.depth,
                    retry_after_ms=1000,
                )
            if self.queued_bytes + byte_cost > self.max_bytes:
                return AdmissionDecision(
                    admitted=False,
                    code=REJECT_CODES["kv_budget"],
                    reason="byte/KV budget for queued work is exhausted",
                    queue_depth=self.depth,
                    retry_after_ms=1000,
                )
        self.requests.append(request_id)
        self.queued_tokens += token_cost
        self.queued_bytes += byte_cost
        return AdmissionDecision(
            admitted=True,
            code="",
            reason="admitted inside the bounded queue",
            queue_class="bounded",
            queue_depth=self.depth,
            queue_position=self.depth - 1,
        )

    def release(self, *, request_id: str, token_cost: int = 0, byte_cost: int = 0) -> None:
        if request_id in self.requests:
            self.requests.remove(request_id)
        self.queued_tokens = max(0, self.queued_tokens - token_cost)
        self.queued_bytes = max(0, self.queued_bytes - byte_cost)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "max_requests": self.max_requests,
            "queued_tokens": self.queued_tokens,
            "queued_bytes": self.queued_bytes,
            "depth_ratio": self.depth_ratio,
        }


class UnboundedQueueGuard:
    """The *counter-example* queue: it must be killed by its own guard."""

    def __init__(self, *, max_duration_s: float, kill_guard_requests: int) -> None:
        if max_duration_s <= 0 or kill_guard_requests <= 0:
            raise ConfigError("the unbounded counter-example needs a duration and a kill guard")
        self.max_duration_s = max_duration_s
        self.kill_guard_requests = kill_guard_requests
        self.admitted = 0
        self.started_ns: Optional[int] = None
        self.stopped = False
        self.stop_reason = ""

    def guard(self, *, now_ns: int) -> Dict[str, Any]:
        if self.started_ns is None:
            self.started_ns = now_ns
        elapsed_s = (now_ns - self.started_ns) / 1e9
        if elapsed_s > self.max_duration_s or self.admitted > self.kill_guard_requests:
            self.stopped = True
            self.stop_reason = (
                "duration guard" if elapsed_s > self.max_duration_s else "request kill guard"
            )
        return {
            "stopped": self.stopped,
            "reason": self.stop_reason,
            "admitted": self.admitted,
            "elapsed_s": elapsed_s,
            "note": (
                "this queue exists only to show what unbounded admission does; it may "
                "never be used for a service run"
            ),
        }


# ── retry budgets and amplification ────────────────────────────────────────


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    attempt: int
    delay_ms: float
    reason: str
    budget_remaining_ms: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "retry": self.retry,
            "attempt": self.attempt,
            "delay_ms": self.delay_ms,
            "reason": self.reason,
            "budget_remaining_ms": self.budget_remaining_ms,
        }


class RetryBudget:
    """Attempts, elapsed budget and jitter are all bounded (E08-05 §8)."""

    def __init__(self, spec: Mapping[str, Any], *, seed: int = 0) -> None:
        self.max_attempts = int(spec["max_attempts_per_request"])
        self.max_elapsed_ms = float(spec["max_elapsed_budget_ms"])
        self.base_backoff_ms = float(spec["base_backoff_ms"])
        self.max_backoff_ms = float(spec["max_backoff_ms"])
        self.jitter_ratio = float(spec["jitter_ratio"])
        self.server_wide_ratio = float(spec["server_wide_retry_budget_ratio"])
        if self.max_attempts < 1:
            raise ConfigError("max_attempts_per_request must be at least 1")
        self._rng = random.Random(seed)
        self._spent_ms: Dict[str, float] = {}
        self.attempts = 0
        self.originals = 0
        self.server_budget_ms = 0.0

    def register_original(self) -> None:
        self.originals += 1
        self.server_budget_ms = self.originals * self.max_elapsed_ms * self.server_wide_ratio

    def decide(
        self,
        *,
        request_id: str,
        attempt: int,
        elapsed_ms: float,
        committed: bool,
        retryable: bool,
        server_spent_ms: float = 0.0,
    ) -> RetryDecision:
        remaining = self.max_elapsed_ms - elapsed_ms
        if committed:
            return RetryDecision(
                False, attempt, 0.0, "stream already committed: transparent retry is forbidden", remaining
            )
        if not retryable:
            return RetryDecision(False, attempt, 0.0, "the failure class is not retryable", remaining)
        if attempt >= self.max_attempts:
            return RetryDecision(False, attempt, 0.0, "attempt budget exhausted", remaining)
        if remaining <= 0:
            return RetryDecision(False, attempt, 0.0, "elapsed retry budget exhausted", remaining)
        if self.server_budget_ms and server_spent_ms >= self.server_budget_ms:
            return RetryDecision(False, attempt, 0.0, "server-wide retry budget exhausted", remaining)
        delay = min(self.base_backoff_ms * (2 ** (attempt - 1)), self.max_backoff_ms)
        jitter = delay * self.jitter_ratio * (self._rng.random() * 2 - 1)
        delay = max(0.0, delay + jitter)
        self.attempts += 1
        self._spent_ms[request_id] = self._spent_ms.get(request_id, 0.0) + delay
        return RetryDecision(True, attempt, delay, "retry inside every budget", remaining - delay)

    def amplification(self) -> Dict[str, Any]:
        if not self.originals:
            return {"amplification": None, "note": "no original request registered"}
        return {
            "originals": self.originals,
            "attempts": self.attempts,
            "amplification": (self.originals + self.attempts) / self.originals,
            "bounded": True,
            "note": "amplification counts transport attempts per original request",
        }


def retry_after_header(*, retry_after_ms: int) -> str:
    if retry_after_ms <= 0:
        raise ConfigError(
            "Retry-After may only be sent when the policy has a meaningful recovery "
            "estimate (E08-05 §7)"
        )
    return str(max(1, int(math.ceil(retry_after_ms / 1000.0))))


# ── overload timeline and recovery ─────────────────────────────────────────


@dataclass(frozen=True)
class OverloadPhase:
    """One phase of a pre-declared overload waveform."""

    name: str
    offered_ratio: float
    duration_sec: float
    policy: str = "bounded"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "offered_ratio": self.offered_ratio,
            "duration_sec": self.duration_sec,
            "policy": self.policy,
        }


WAVEFORMS: Mapping[str, Tuple[OverloadPhase, ...]] = {
    "ramp": (
        OverloadPhase("ramp_low", 0.8, 30.0),
        OverloadPhase("ramp_cross", 1.0, 30.0),
        OverloadPhase("ramp_over", 1.5, 30.0),
    ),
    "step_up": (
        OverloadPhase("step_120", 1.2, 20.0),
        OverloadPhase("step_150", 1.5, 20.0),
        OverloadPhase("step_down", 0.5, 20.0),
    ),
    "burst": (
        OverloadPhase("burst_peak", 2.0, 5.0),
        OverloadPhase("burst_base", 0.5, 25.0),
    ),
    "sustained": (OverloadPhase("sustained", 1.4, 60.0),),
    "backend_capacity_drop": (
        OverloadPhase("drop", 1.0, 30.0, policy="backend_slow"),
        OverloadPhase("restore", 1.0, 30.0),
    ),
    "slow_client_pressure": (OverloadPhase("slow_clients", 1.0, 30.0, policy="slow_clients"),),
    "step_down": (
        OverloadPhase("overload", 1.5, 20.0),
        OverloadPhase("recover", 0.4, 40.0),
    ),
}


def overload_plan(name: str, *, g_star_qps: float) -> Dict[str, Any]:
    """Waveforms are expressed as ratios of ``G*``, never of a guess."""
    if name not in WAVEFORMS:
        raise ConfigError(f"unknown overload waveform {name!r}; expected one of {sorted(WAVEFORMS)}")
    if g_star_qps <= 0:
        raise ConfigError("the overload plan needs a measured G* to scale from")
    phases = WAVEFORMS[name]
    return {
        "waveform": name,
        "g_star_qps": g_star_qps,
        "phases": [
            {**phase.as_dict(), "offered_qps": phase.offered_ratio * g_star_qps}
            for phase in phases
        ],
        "note": "percentages come from the E08-02 G*, not from a feeling",
    }


def recovery_report(
    *,
    state_timeline: Sequence[Tuple[int, str]],
    queue_timeline: Sequence[Tuple[int, float]],
    slo_timeline: Sequence[Tuple[int, bool]],
    memory_timeline: Sequence[Tuple[int, float]],
    drop_at_ns: int,
    window_ns: int,
    queue_baseline: float = 0.0,
    queue_tolerance: float = 1.0,
    memory_tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Queue, SLO, resource and *state* must all come back inside the window."""
    def first_after(timeline: Sequence[Tuple[int, float]], predicate) -> Optional[int]:
        for timestamp, value in timeline:
            if timestamp >= drop_at_ns and predicate(value):
                return timestamp
        return None

    queue_ns = first_after(queue_timeline, lambda value: abs(value - queue_baseline) <= queue_tolerance)
    slo_ns = None
    for timestamp, ok in slo_timeline:
        if timestamp >= drop_at_ns and ok:
            slo_ns = timestamp
            break
    memory_ns = first_after(
        memory_timeline,
        lambda value: value <= max((item for _, item in memory_timeline), default=0.0)
        + memory_tolerance,
    )
    state_ns = None
    for timestamp, state in state_timeline:
        if timestamp >= drop_at_ns and state == "NORMAL":
            state_ns = timestamp
            break
    deadlines = {
        "queue": queue_ns,
        "slo": slo_ns,
        "memory": memory_ns,
        "state": state_ns,
    }
    within = {
        name: (value is not None and value - drop_at_ns <= window_ns)
        for name, value in deadlines.items()
    }
    return {
        "ok": all(within.values()),
        "drop_at_ns": drop_at_ns,
        "window_ns": window_ns,
        "recovery_ns": deadlines,
        "within_window": within,
        "time_to_recovery_ms": {
            name: (None if value is None else (value - drop_at_ns) / 1e6)
            for name, value in deadlines.items()
        },
        "note": (
            "an average queue that looks fine while the oldest age keeps growing is not "
            "recovery; all four signals must return"
        ),
    }


def safety_audit(
    *,
    queue_snapshot: Mapping[str, Any],
    rejects_with_backend_state: int,
    expired_still_running: int,
    duplicate_tokens: int,
    emergency_stop_half_state: bool,
) -> Dict[str, Any]:
    """Check the §11 invariants that a run can actually violate."""
    problems: List[str] = []
    if rejects_with_backend_state:
        problems.append("an admission reject created Backend/KV state")
    if expired_still_running:
        problems.append("requests kept running past their deadline")
    if duplicate_tokens:
        problems.append("retries produced duplicated output tokens")
    if emergency_stop_half_state:
        problems.append("the emergency stop left half-state behind")
    if not queue_snapshot.get("max_requests"):
        problems.append("a queue without a hard cap was active")
    return {
        "ok": not problems,
        "problems": problems,
        "invariants": list(SAFETY_INVARIANTS),
        "queue": dict(queue_snapshot),
    }


__all__ = [
    "AdmissionCostModel",
    "AdmissionDecision",
    "AdmissionEstimate",
    "AdmissionRequest",
    "BoundedQueue",
    "OverloadPhase",
    "PRESSURE_STATES",
    "PressureEvent",
    "PressureSignal",
    "PressureSpec",
    "PressureStateMachine",
    "PressureTransition",
    "REJECT_CODES",
    "RetryBudget",
    "RetryDecision",
    "SAFETY_INVARIANTS",
    "UnboundedQueueGuard",
    "WAVEFORMS",
    "overload_plan",
    "recovery_report",
    "retry_after_header",
    "safety_audit",
]
