"""Service time boundaries: five TTFTs, one timestamp ledger, clock domains.

The S08 details README (§8) is strict about this: ``runtime_TTFT`` is *not* an
online TTFT, the report must always carry a prefix, and cross-process
timestamps may not be subtracted without a measured offset.  This module owns
the timestamp vocabulary, the derived durations and the calibration objects.

Nothing here measures anything by itself: a run fills the ledger with real
monotonic readings, or leaves the field unset — in which case the derived
duration is reported as *missing*, never as zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from hqsb.core.errors import ConfigError

#: The canonical timestamp vocabulary (details README §8), in causal order.
TIMESTAMP_FIELDS: Tuple[str, ...] = (
    "t_sched",
    "t_send",
    "t_gateway_recv",
    "t_validate_done",
    "t_enqueue",
    "t_dequeue",
    "t_backend_submit",
    "t_runtime_first",
    "t_first_frame_write",
    "t_first_byte_client",
    "t_runtime_last",
    "t_terminal_write",
    "t_client_done",
    "t_cleanup_done",
)

#: Derived durations and the two timestamps each one needs.
DERIVED_DURATIONS: Mapping[str, Tuple[str, str]] = {
    "loadgen_lag": ("t_sched", "t_send"),
    "ingress": ("t_send", "t_gateway_recv"),
    "gateway_prequeue": ("t_gateway_recv", "t_enqueue"),
    "service_queue": ("t_enqueue", "t_dequeue"),
    "backend_submit": ("t_dequeue", "t_backend_submit"),
    "runtime_ttft": ("t_backend_submit", "t_runtime_first"),
    "server_ttft": ("t_gateway_recv", "t_first_frame_write"),
    "client_ttft": ("t_send", "t_first_byte_client"),
    "stream_egress_first": ("t_runtime_first", "t_first_byte_client"),
    "client_e2e": ("t_send", "t_client_done"),
    "cleanup_lag": ("t_terminal_write", "t_cleanup_done"),
}

#: The five TTFTs the S08 report may distinguish (never merged into "TTFT").
TTFT_NAMES: Tuple[str, ...] = (
    "runtime_TTFT",
    "server_TTFT",
    "client_TTFT",
    "loadgen_TTFT_send_lag",
    "gateway_TTFT_prequeue",
)

#: Clock domains that must never be mixed implicitly.
CLOCK_DOMAINS: Tuple[str, ...] = (
    "loadgen_monotonic",
    "gateway_monotonic",
    "backend_monotonic",
    "wall_clock_utc",
    "gpu_event",
    "profiler",
)

_NS_PER_MS = 1_000_000.0


@dataclass(frozen=True)
class ClockDomain:
    """One process/timeline clock with an optional measured offset."""

    name: str
    offset_ns: int = 0
    uncertainty_ns: int = 0
    method: str = ""
    calibrated: bool = False

    def __post_init__(self) -> None:
        if self.name not in CLOCK_DOMAINS:
            raise ConfigError(
                f"unknown clock domain {self.name!r}; expected one of {list(CLOCK_DOMAINS)}",
                details={"clock": self.name},
            )
        if self.uncertainty_ns < 0:
            raise ConfigError("uncertainty must not be negative")

    def to_reference_ns(self, value_ns: int) -> int:
        if not self.calibrated:
            raise ConfigError(
                f"clock domain {self.name!r} is not calibrated; cross-domain "
                "subtraction without a measured offset is forbidden (E08-10 §6)",
                details={"clock": self.name},
            )
        return value_ns + self.offset_ns

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "offset_ns": self.offset_ns,
            "uncertainty_ns": self.uncertainty_ns,
            "method": self.method,
            "calibrated": self.calibrated,
        }


def cross_domain_delta_ns(
    start: Tuple[str, int],
    end: Tuple[str, int],
    domains: Mapping[str, ClockDomain],
) -> Dict[str, Any]:
    """Difference between two readings *with* the offset uncertainty reported."""
    start_clock, start_ns = start
    end_clock, end_ns = end
    if start_clock not in domains or end_clock not in domains:
        raise ConfigError("both clock domains must be registered before a delta")
    start_domain = domains[start_clock]
    end_domain = domains[end_clock]
    raw = end_ns - start_ns
    if start_clock == end_clock:
        return {"delta_ns": raw, "uncertainty_ns": 0, "same_clock": True}
    delta = end_domain.to_reference_ns(end_ns) - start_domain.to_reference_ns(start_ns)
    return {
        "delta_ns": delta,
        "uncertainty_ns": start_domain.uncertainty_ns + end_domain.uncertainty_ns,
        "same_clock": False,
    }


@dataclass
class TimestampLedger:
    """One request's timestamps, tagged with the clock they were read from."""

    request_id: str
    clocks: Dict[str, int] = field(default_factory=dict)
    clock_of: Dict[str, str] = field(default_factory=dict)

    def record(self, field_name: str, value_ns: int, *, clock: str = "gateway_monotonic") -> None:
        if field_name not in TIMESTAMP_FIELDS:
            raise ConfigError(
                f"unknown timestamp {field_name!r}; the vocabulary is frozen "
                f"(details README §8): {list(TIMESTAMP_FIELDS)}",
                details={"field": field_name},
            )
        if clock not in CLOCK_DOMAINS:
            raise ConfigError(f"unknown clock domain {clock!r}")
        if field_name in self.clocks:
            raise ConfigError(
                f"timestamp {field_name!r} was already recorded for {self.request_id}; "
                "overwriting a boundary would hide the original observation"
            )
        self.clocks[field_name] = int(value_ns)
        self.clock_of[field_name] = clock

    def value(self, field_name: str) -> Optional[int]:
        return self.clocks.get(field_name)

    def duration_ns(self, name: str) -> Optional[int]:
        if name not in DERIVED_DURATIONS:
            raise ConfigError(
                f"unknown derived duration {name!r}; expected one of "
                f"{sorted(DERIVED_DURATIONS)}"
            )
        start_field, end_field = DERIVED_DURATIONS[name]
        start, end = self.clocks.get(start_field), self.clocks.get(end_field)
        if start is None or end is None:
            return None
        if self.clock_of[start_field] != self.clock_of[end_field]:
            raise ConfigError(
                f"{name}: {start_field} and {end_field} were read from different clocks; "
                "subtract them only through an explicit calibration",
                details={"start_clock": self.clock_of[start_field], "end_clock": self.clock_of[end_field]},
            )
        return end - start

    def duration_ms(self, name: str) -> Optional[float]:
        value = self.duration_ns(name)
        return None if value is None else value / _NS_PER_MS

    def monotonicity_problems(self) -> List[str]:
        """Same-clock timestamps must not go backwards in causal order."""
        problems: List[str] = []
        ordered = [name for name in TIMESTAMP_FIELDS if name in self.clocks]
        for left, right in zip(ordered, ordered[1:]):
            if self.clock_of[left] != self.clock_of[right]:
                continue
            if self.clocks[right] < self.clocks[left]:
                problems.append(
                    f"{right} ({self.clocks[right]}) precedes {left} ({self.clocks[left]})"
                )
        return problems

    def missing_fields(self) -> List[str]:
        return [name for name in TIMESTAMP_FIELDS if name not in self.clocks]

    def derived(self, *, only_present: bool = True) -> Dict[str, Optional[float]]:
        report: Dict[str, Optional[float]] = {}
        for name in DERIVED_DURATIONS:
            value = self.duration_ms(name)
            if value is None and only_present:
                continue
            report[name] = value
        return report

    def ttft_report(self) -> Dict[str, Optional[float]]:
        """TTFTs with their mandatory prefixes (never a bare ``ttft``)."""
        return {
            "runtime_TTFT_ms": self.duration_ms("runtime_ttft"),
            "server_TTFT_ms": self.duration_ms("server_ttft"),
            "client_TTFT_ms": self.duration_ms("client_ttft"),
            "loadgen_lag_ms": self.duration_ms("loadgen_lag"),
            "gateway_prequeue_ms": self.duration_ms("gateway_prequeue"),
        }

    def primary_slo_ttft_ms(self, *, mode: str = "client") -> Optional[float]:
        """The SLO clock is the client-visible one (or a pre-registered server one)."""
        if mode == "client":
            return self.duration_ms("client_ttft")
        if mode == "server":
            return self.duration_ms("server_ttft")
        raise ConfigError(
            f"unknown SLO TTFT mode {mode!r}; runtime TTFT is not an online TTFT "
            "(details README §8)",
            details={"mode": mode},
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "clocks": dict(sorted(self.clocks.items())),
            "clock_of": dict(sorted(self.clock_of.items())),
            "derived_ms": self.derived(only_present=False),
            "ttft_ms": self.ttft_report(),
            "missing": self.missing_fields(),
            "monotonicity_problems": self.monotonicity_problems(),
        }


def require_prefixed_ttft(name: str) -> str:
    """Reject a bare ``TTFT`` in any report field or metric name."""
    if name == "ttft" or name.endswith("_ttft") and not name.endswith(
        ("runtime_TTFT", "server_TTFT", "client_TTFT")
    ):
        # allow stage-prefixed names such as runtime_ttft/server_ttft/client_ttft
        if name in ("runtime_ttft", "server_ttft", "client_ttft"):
            return name
    if name.lower() == "ttft":
        raise ConfigError(
            "'TTFT' without a prefix is ambiguous (runtime/server/client); the report "
            "must use one of runtime_TTFT / server_TTFT / client_TTFT (details README §8)",
            details={"name": name},
        )
    return name


@dataclass(frozen=True)
class TimeConservation:
    """client_E2E vs. the sum of its parts, with overlap/uncertainty reported."""

    accounted_ns: int
    total_ns: int
    overlap_ns: int = 0
    clock_uncertainty_ns: int = 0

    @property
    def unaccounted_ns(self) -> int:
        return self.total_ns - self.accounted_ns + self.overlap_ns

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accounted_ns": self.accounted_ns,
            "total_ns": self.total_ns,
            "overlap_ns": self.overlap_ns,
            "clock_uncertainty_ns": self.clock_uncertainty_ns,
            "unaccounted_ns": self.unaccounted_ns,
            "note": (
                "span durations may overlap on the critical path; a sum that exceeds "
                "the total is expected and must not be 'fixed' by rescaling"
            ),
        }


def time_conservation(ledger: TimestampLedger, *, overlap_ns: int = 0) -> TimeConservation:
    stages = (
        "ingress",
        "gateway_prequeue",
        "service_queue",
        "backend_submit",
        "runtime_ttft",
        "stream_egress_first",
        "cleanup_lag",
    )
    total = ledger.duration_ns("client_e2e")
    if total is None:
        raise ConfigError("client_e2e requires t_send and t_client_done")
    accounted = sum(
        value for value in (ledger.duration_ns(stage) for stage in stages) if value
    )
    return TimeConservation(accounted_ns=accounted, total_ns=total, overlap_ns=overlap_ns)


__all__ = [
    "CLOCK_DOMAINS",
    "ClockDomain",
    "DERIVED_DURATIONS",
    "TIMESTAMP_FIELDS",
    "TTFT_NAMES",
    "TimeConservation",
    "TimestampLedger",
    "cross_domain_delta_ns",
    "require_prefixed_ttft",
    "time_conservation",
]
