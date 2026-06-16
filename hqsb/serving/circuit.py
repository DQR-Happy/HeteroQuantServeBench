"""Health classification, circuit breaker, quarantine and recovery timing.

Three separations matter (E08-09 §6):

* **process alive ≠ ready ≠ model epoch warm** — readiness is about *this*
  artifact being servable, not about a PID existing;
* **counted failures ≠ everything that went wrong** — a client 4xx, a user
  cancel or an admission reject must never trip the breaker, or the service
  punishes the Backend for the caller's behaviour;
* **transient open ≠ correctness quarantine** — a wrong model or a corrupt
  cache identity is quarantined until a human clears it, not retried after a
  backoff.

Recovery is reported as several distinct times; "the process restarted" is not
"the service recovered".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Circuit states.
CIRCUIT_STATES: Tuple[str, ...] = ("CLOSED", "OPEN", "HALF_OPEN")

#: Failure classes that may trip the breaker.
COUNTED_FAILURE_CLASSES: Tuple[str, ...] = (
    "backend_unavailable",
    "backend_oom",
    "backend_timeout",
    "malformed_response",
    "protocol_failure",
)

#: Failures that must *not* be charged to the Backend.
EXCLUDED_FAILURE_CLASSES: Tuple[str, ...] = (
    "client_4xx",
    "user_cancel",
    "admission_reject",
    "deadline_infeasible",
)

#: Correctness problems: quarantine, not backoff.
QUARANTINE_CLASSES: Tuple[str, ...] = (
    "model_identity_mismatch",
    "precision_identity_mismatch",
    "token_quality_identity_mismatch",
    "corrupt_cache_identity",
)

#: Recovery milestones, each measured from a different origin.
RECOVERY_METRICS: Tuple[str, ...] = (
    "MTTD",
    "isolation_time",
    "request_resolution",
    "backend_recovery",
    "traffic_recovery",
    "SLO_recovery",
)


@dataclass(frozen=True)
class HealthState:
    """Every layer of "is it usable" is kept apart."""

    process_alive: bool = False
    adapter_ready: bool = False
    model_epoch_warm: bool = False
    capability_probe_fresh: bool = False
    transient_overload: bool = False
    circuit_state: str = "CLOSED"
    quarantined: bool = False
    quarantine_reason: str = ""
    model_epoch: str = ""

    def classification(self) -> str:
        if self.quarantined:
            return "quarantined"
        if not self.process_alive:
            return "down"
        if not self.adapter_ready:
            return "not_ready"
        if not self.model_epoch_warm:
            return "model_cold"
        if not self.capability_probe_fresh:
            return "probe_stale"
        if self.circuit_state == "OPEN":
            return "circuit_open"
        if self.transient_overload:
            return "degraded"
        return "ready"

    @property
    def routable(self) -> bool:
        return self.classification() == "ready"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "process_alive": self.process_alive,
            "adapter_ready": self.adapter_ready,
            "model_epoch_warm": self.model_epoch_warm,
            "capability_probe_fresh": self.capability_probe_fresh,
            "transient_overload": self.transient_overload,
            "circuit_state": self.circuit_state,
            "quarantined": self.quarantined,
            "quarantine_reason": self.quarantine_reason,
            "model_epoch": self.model_epoch,
            "classification": self.classification(),
            "routable": self.routable,
        }


@dataclass(frozen=True)
class CircuitSpec:
    """The frozen breaker configuration (``fault_spec.yaml``)."""

    window_samples: int
    min_samples: int
    failure_rate_threshold: float
    consecutive_failure_threshold: int
    open_duration_ms: float
    max_open_duration_ms: float
    half_open_probes_required: int
    half_open_probe_concurrency: int
    telemetry_freshness_ms: float

    @classmethod
    def from_document(cls, payload: Mapping[str, Any]) -> "CircuitSpec":
        breaker = dict(payload["circuit_breaker"])
        return cls(
            window_samples=int(breaker["sliding_window_samples"]),
            min_samples=int(breaker["min_samples"]),
            failure_rate_threshold=float(breaker["failure_rate_threshold"]),
            consecutive_failure_threshold=int(breaker["consecutive_failure_threshold"]),
            open_duration_ms=float(breaker["open_duration_ms"]),
            max_open_duration_ms=float(breaker["max_open_duration_ms"]),
            half_open_probes_required=int(breaker["half_open_probes_required"]),
            half_open_probe_concurrency=int(breaker["half_open_probe_concurrency"]),
            telemetry_freshness_ms=float(breaker["telemetry_freshness_ms"]),
        )


@dataclass
class FailureObservation:
    monotonic_ns: int
    failure_class: str
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "monotonic_ns": self.monotonic_ns,
            "failure_class": self.failure_class,
            "detail": self.detail,
        }


@dataclass
class CircuitEvent:
    monotonic_ns: int
    source: str
    target: str
    reason: str
    counters: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "monotonic_ns": self.monotonic_ns,
            "from": self.source,
            "to": self.target,
            "reason": self.reason,
            "counters": dict(self.counters),
        }


class CircuitBreaker:
    """Per-instance breaker with a sliding window and half-open probes."""

    def __init__(self, spec: CircuitSpec, *, instance_id: str) -> None:
        self.spec = spec
        self.instance_id = instance_id
        self.state = "CLOSED"
        self.observations: List[FailureObservation] = []
        self.events: List[CircuitEvent] = []
        self.successes = 0
        self.consecutive_failures = 0
        self.opened_at_ns: Optional[int] = None
        self.open_count = 0
        self.half_open_probes = 0
        self.half_open_successes = 0
        self.quarantined = False
        self.quarantine_reason = ""

    # -- counting -------------------------------------------------------
    def record_failure(
        self, failure_class: str, *, monotonic_ns: int, detail: str = ""
    ) -> Optional[CircuitEvent]:
        if failure_class in EXCLUDED_FAILURE_CLASSES:
            # the Backend is not responsible for caller behaviour
            return None
        if failure_class in QUARANTINE_CLASSES:
            self.quarantined = True
            self.quarantine_reason = failure_class
            return self._event(monotonic_ns, "QUARANTINED", f"correctness failure: {failure_class}")
        if failure_class not in COUNTED_FAILURE_CLASSES:
            raise ConfigError(
                f"failure class {failure_class!r} is neither counted, excluded nor a "
                "quarantine class; an unclassified failure must not silently influence "
                "the breaker",
                details={"failure_class": failure_class},
            )
        self.observations.append(
            FailureObservation(monotonic_ns=monotonic_ns, failure_class=failure_class, detail=detail)
        )
        window = self.observations[-self.spec.window_samples :]
        failures = len(window)
        self.consecutive_failures += 1
        if self.state == "CLOSED":
            # two independent triggers: a failure *rate* over a full window, or a
            # run of consecutive failures (which fires before the window fills)
            rate = failures / len(window) if window else 0.0
            rate_exceeded = (
                len(window) >= self.spec.min_samples
                and rate >= self.spec.failure_rate_threshold
            )
            run_exceeded = (
                self.consecutive_failures >= self.spec.consecutive_failure_threshold
            )
            if rate_exceeded or run_exceeded:
                reason = (
                    f"failure rate {rate:.2f}"
                    if rate_exceeded
                    else f"{self.consecutive_failures} consecutive failures"
                )
                return self._open(monotonic_ns, reason=reason)
        elif self.state == "HALF_OPEN":
            return self._open(monotonic_ns, reason="half-open probe failed")
        return None

    def record_success(self, *, monotonic_ns: int) -> Optional[CircuitEvent]:
        self.successes += 1
        self.consecutive_failures = 0
        if self.state == "HALF_OPEN":
            self.half_open_successes += 1
            if self.half_open_successes >= self.spec.half_open_probes_required:
                return self._close(monotonic_ns, reason="half-open probes succeeded")
        return None

    def allow_request(self, *, monotonic_ns: int) -> Dict[str, Any]:
        """Should a new request be routed to this instance?"""
        if self.quarantined:
            return {"allowed": False, "reason": f"quarantined: {self.quarantine_reason}"}
        if self.state == "CLOSED":
            return {"allowed": True, "reason": ""}
        if self.state == "OPEN":
            opened_at = self.opened_at_ns or monotonic_ns
            duration_ms = (self.spec.open_duration_ms)
            if monotonic_ns - opened_at >= duration_ms * 1e6:
                self.state = "HALF_OPEN"
                self.half_open_probes = 0
                self.half_open_successes = 0
                self.events.append(
                    self._event(monotonic_ns, "HALF_OPEN", "open duration elapsed")
                )
                return {"allowed": True, "reason": "half-open probe admitted"}
            return {"allowed": False, "reason": "circuit is open"}
        # HALF_OPEN: only the declared number of concurrent probes
        if self.half_open_probes >= self.spec.half_open_probe_concurrency:
            return {"allowed": False, "reason": "half-open probe concurrency reached"}
        self.half_open_probes += 1
        return {"allowed": True, "reason": "half-open probe admitted"}

    # -- lifecycle ------------------------------------------------------
    def _open(self, monotonic_ns: int, *, reason: str) -> CircuitEvent:
        self.state = "OPEN"
        self.opened_at_ns = monotonic_ns
        self.open_count += 1
        self.half_open_probes = 0
        self.half_open_successes = 0
        return self._event(monotonic_ns, "OPEN", reason)

    def _close(self, monotonic_ns: int, *, reason: str) -> CircuitEvent:
        self.state = "CLOSED"
        self.opened_at_ns = None
        self.observations.clear()
        return self._event(monotonic_ns, "CLOSED", reason)

    def _event(self, monotonic_ns: int, target: str, reason: str) -> CircuitEvent:
        event = CircuitEvent(
            monotonic_ns=monotonic_ns,
            source="",
            target=target,
            reason=reason,
            counters=self.counters(),
        )
        self.events.append(event)
        return event

    def clear_quarantine(self, *, monotonic_ns: int, operator: str) -> CircuitEvent:
        if not self.quarantined:
            raise ConfigError("no quarantine to clear")
        if not operator:
            raise ConfigError(
                "a correctness quarantine must be cleared explicitly by an operator, "
                "never by a timeout (E08-09 §6)"
            )
        self.quarantined = False
        self.quarantine_reason = ""
        return self._event(monotonic_ns, "CLEARED", f"quarantine cleared by {operator}")

    def counters(self) -> Dict[str, Any]:
        window = self.observations[-self.spec.window_samples :]
        return {
            "window_samples": len(window),
            "window_failures": len(window),
            "consecutive_failures": self.consecutive_failures,
            "successes": self.successes,
            "state": self.state,
            "open_count": self.open_count,
            "quarantined": self.quarantined,
        }


def transition_is_recomputable(breaker: CircuitBreaker) -> Dict[str, Any]:
    """Every transition must follow from the recorded counters (§16)."""
    problems: List[str] = []
    open_events = [event for event in breaker.events if event.target == "OPEN"]
    if len(open_events) != breaker.open_count:
        problems.append(
            f"open_count={breaker.open_count} but {len(open_events)} OPEN transitions recorded"
        )
    for event in open_events:
        if event.counters.get("window_failures", 0) < breaker.spec.min_samples and (
            event.counters.get("consecutive_failures", 0)
            < breaker.spec.consecutive_failure_threshold
            and event.reason != "half-open probe failed"
        ):
            problems.append("an OPEN transition does not follow from its counters")
    return {"ok": not problems, "problems": problems, "events": len(breaker.events)}


def breaker_matrix(spec: CircuitSpec) -> Dict[str, Any]:
    """A table-driven exercise of the state machine (dummy target, no model)."""
    breaker = CircuitBreaker(spec, instance_id="dummy")
    events: List[Dict[str, Any]] = []
    now = 0
    for _ in range(spec.consecutive_failure_threshold):
        now += 1_000_000
        event = breaker.record_failure("backend_unavailable", monotonic_ns=now)
        if event:
            events.append(event.as_dict())
    opened = breaker.state == "OPEN"
    denied = not breaker.allow_request(monotonic_ns=now)["allowed"]
    probe_time = now + int(spec.open_duration_ms * 1e6) + 1
    probe = breaker.allow_request(monotonic_ns=probe_time)
    for _ in range(spec.half_open_probes_required):
        now = probe_time + 1_000_000
        event = breaker.record_success(monotonic_ns=now)
        if event and event.target == "CLOSED":
            events.append(event.as_dict())
    return {
        "opened_after_threshold": opened,
        "denied_while_open": denied,
        "half_open_probe_admitted": probe["allowed"],
        "closed_after_probes": breaker.state == "CLOSED",
        "events": events,
        "recomputable": transition_is_recomputable(breaker),
    }


# ── recovery timeline ──────────────────────────────────────────────────────


@dataclass
class RecoveryTimeline:
    """The distinct recovery milestones of one fault (E08-09 §7)."""

    fault_injected_ns: int
    fault_detected_ns: Optional[int] = None
    no_new_routes_ns: Optional[int] = None
    requests_resolved_ns: Optional[int] = None
    recovery_started_ns: Optional[int] = None
    ready_new_epoch_ns: Optional[int] = None
    normal_route_share_ns: Optional[int] = None
    slo_back_in_band_ns: Optional[int] = None

    def _delta_ms(self, value: Optional[int], origin: Optional[int] = None) -> Optional[float]:
        if value is None:
            return None
        base = self.fault_injected_ns if origin is None else origin
        return (value - base) / 1e6

    def metrics(self) -> Dict[str, Any]:
        return {
            "MTTD": self._delta_ms(self.fault_detected_ns),
            "isolation_time": self._delta_ms(self.no_new_routes_ns),
            "request_resolution": self._delta_ms(self.requests_resolved_ns),
            "backend_recovery": self._delta_ms(self.ready_new_epoch_ns, self.recovery_started_ns),
            "traffic_recovery": self._delta_ms(self.normal_route_share_ns, self.ready_new_epoch_ns),
            "SLO_recovery": self._delta_ms(self.slo_back_in_band_ns),
        }

    def problems(self) -> List[str]:
        issues: List[str] = []
        order = [
            ("fault_detected_ns", self.fault_detected_ns),
            ("no_new_routes_ns", self.no_new_routes_ns),
            ("requests_resolved_ns", self.requests_resolved_ns),
            ("ready_new_epoch_ns", self.ready_new_epoch_ns),
            ("normal_route_share_ns", self.normal_route_share_ns),
            ("slo_back_in_band_ns", self.slo_back_in_band_ns),
        ]
        previous_name, previous_value = "fault_injected_ns", self.fault_injected_ns
        for name, value in order:
            if value is None:
                continue
            if value < previous_value:
                issues.append(f"{name} precedes {previous_name}")
            previous_name, previous_value = name, value
        return issues

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fault_injected_ns": self.fault_injected_ns,
            "milestones": {
                "fault_detected_ns": self.fault_detected_ns,
                "no_new_routes_ns": self.no_new_routes_ns,
                "requests_resolved_ns": self.requests_resolved_ns,
                "ready_new_epoch_ns": self.ready_new_epoch_ns,
                "normal_route_share_ns": self.normal_route_share_ns,
                "slo_back_in_band_ns": self.slo_back_in_band_ns,
            },
            "metrics": self.metrics(),
            "problems": self.problems(),
            "note": (
                "process restart is not service recovery: cache warmup, readiness, route "
                "share and SLO each have their own time"
            ),
        }


def flap_report(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Repeated fault/recovery must not oscillate the breaker forever (§21)."""
    opens = [event for event in events if event.get("target") == "OPEN"]
    closes = [event for event in events if event.get("target") == "CLOSED"]
    intervals = [
        int(second.get("monotonic_ns", 0)) - int(first.get("monotonic_ns", 0))
        for first, second in zip(opens, closes)
    ]
    shrinking = any(
        later < earlier for earlier, later in zip(intervals, intervals[1:])
    )
    return {
        "opens": len(opens),
        "closes": len(closes),
        "open_intervals_ms": [value / 1e6 for value in intervals],
        "shrinking_intervals": shrinking,
        "note": "open intervals that shrink with each flap are a sign of an over-eager breaker",
    }


__all__ = [
    "CIRCUIT_STATES",
    "COUNTED_FAILURE_CLASSES",
    "CircuitBreaker",
    "CircuitEvent",
    "CircuitSpec",
    "EXCLUDED_FAILURE_CLASSES",
    "FailureObservation",
    "HealthState",
    "QUARANTINE_CLASSES",
    "RECOVERY_METRICS",
    "RecoveryTimeline",
    "breaker_matrix",
    "flap_report",
    "transition_is_recomputable",
]
