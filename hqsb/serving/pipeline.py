"""Token/byte delivery ledger, layered buffers and cancel linearisation.

Two different quantities must never be conflated (details README §9, E08-08 §4):

    generated ≥ committed ≥ emitted ≥ flushed ≥ client_received

Generation is cheap to count and expensive to waste; delivery is what the
client actually saw.  Between the two sit five buffers, each with an owner, a
cap and a defined action when the cap is reached — a write that returns says
nothing about what the client received, so socket-level backlog is tracked
separately from the application queue.

The module is deliberately *pure*: it records observations, checks invariants
and refuses conclusions that the numbers do not support.  It does not generate
tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Delivery stages, in the order the inequality must hold.
LEDGER_STAGES: Tuple[str, ...] = (
    "generated",
    "committed",
    "emitted",
    "flushed",
    "client_received",
)

#: The five buffer layers between Runtime and client (E08-08 §2/§4).
LAYER_NAMES: Tuple[str, ...] = (
    "application_queue",
    "serializer_buffer",
    "server_buffer",
    "socket_send_buffer",
    "client_receive_buffer",
)

#: Cancel injection windows (E08-08 §7); the linearisation point is per window.
CANCEL_WINDOWS: Tuple[str, ...] = (
    "queued",
    "backend_submitted_no_first_token",
    "token_ready_not_serialized",
    "frame_queued_not_written",
    "write_in_progress",
    "client_read_n_frames",
    "runtime_generating_next_token",
    "final_frame_race",
    "shared_prefix_kv",
    "multi_request_same_batch",
)

#: Stages that may still complete for a frame that was already in flight.
ALLOWED_AFTER_CANCEL: Mapping[str, Tuple[str, ...]] = {
    "queued": (),
    "backend_submitted_no_first_token": (),
    "token_ready_not_serialized": (),
    "frame_queued_not_written": ("write_start", "write_end"),
    "write_in_progress": ("write_end", "client_read"),
    "client_read_n_frames": ("client_read",),
    "runtime_generating_next_token": (),
    "final_frame_race": ("write_start", "write_end", "terminal_write"),
    "shared_prefix_kv": (),
    "multi_request_same_batch": (),
}

#: Pipeline stages recorded per frame.
FRAME_STAGES: Tuple[str, ...] = (
    "ready",
    "enqueued",
    "serialize_start",
    "serialize_end",
    "write_start",
    "write_end",
    "client_read",
)


@dataclass
class DeliveryLedger:
    """Per-request token accounting across the five delivery stages."""

    request_id: str
    generated: int = 0
    committed: int = 0
    emitted: int = 0
    flushed: int = 0
    client_received: int = 0
    client_received_observed: bool = False
    discarded_after_disconnect: int = 0
    refused_to_emit_after_cancel: int = 0
    terminal_clean: bool = False
    disconnected: bool = False
    cancelled: bool = False

    def counts(self) -> Dict[str, int]:
        return {stage: getattr(self, stage) for stage in LEDGER_STAGES}

    def audit(self) -> Dict[str, Any]:
        problems: List[str] = []
        previous = None
        previous_name = ""
        for stage in LEDGER_STAGES:
            value = getattr(self, stage)
            if value < 0:
                problems.append(f"{stage} must not be negative")
            if previous is not None and value > previous:
                problems.append(
                    f"{stage} ({value}) exceeds {previous_name} ({previous}); the "
                    "delivery ledger may never grow downstream"
                )
            previous, previous_name = value, stage
        if (
            self.terminal_clean
            and not (self.disconnected or self.cancelled)
            and self.client_received_observed
        ):
            if not (self.committed == self.emitted == self.client_received):
                problems.append(
                    "after a clean terminal the client must have received the whole "
                    f"answer (committed={self.committed}, emitted={self.emitted}, "
                    f"client_received={self.client_received})"
                )
        if (self.disconnected or self.cancelled) and self.client_received_observed:
            shortfall = self.committed - self.client_received
            if shortfall != self.discarded_after_disconnect and shortfall != 0:
                problems.append(
                    "the difference between committed and client_received must be "
                    f"explained by discarded_after_disconnect "
                    f"(shortfall={shortfall}, recorded={self.discarded_after_disconnect})"
                )
        return {
            "ok": not problems,
            "problems": problems,
            "counts": self.counts(),
            "waste": self.waste(),
        }

    def waste(self) -> Dict[str, int]:
        """Work that was paid for and never delivered (never deleted from stats)."""
        return {
            "generated_not_committed": self.generated - self.committed,
            "committed_not_emitted": self.committed - self.emitted,
            "emitted_not_received": self.emitted - self.client_received,
            "discarded_after_disconnect": self.discarded_after_disconnect,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            **self.counts(),
            "client_received_observed": self.client_received_observed,
            "discarded_after_disconnect": self.discarded_after_disconnect,
            "refused_to_emit_after_cancel": self.refused_to_emit_after_cancel,
            "terminal_clean": self.terminal_clean,
            "disconnected": self.disconnected,
            "cancelled": self.cancelled,
        }


@dataclass
class LayerState:
    """One buffer layer with a hard cap and an explicit overflow action."""

    name: str
    cap_bytes: int
    cap_frames: int
    owner: str
    on_cap: str
    buffered_bytes: int = 0
    buffered_frames: int = 0
    high_water_bytes: int = 0
    high_water_frames: int = 0

    def __post_init__(self) -> None:
        if self.name not in LAYER_NAMES:
            raise ConfigError(
                f"unknown buffer layer {self.name!r}; expected one of {list(LAYER_NAMES)}"
            )
        if self.cap_bytes <= 0 or self.cap_frames <= 0:
            raise ConfigError(f"{self.name}: every layer needs a positive hard cap")
        if not self.on_cap:
            raise ConfigError(
                f"{self.name}: a cap without a defined overflow action is a silent drop"
            )

    def enqueue(self, *, bytes_value: int, frames: int = 1) -> None:
        if bytes_value < 0 or frames < 0:
            raise ConfigError("negative accounting is not an observation")
        self.buffered_bytes += bytes_value
        self.buffered_frames += frames
        self.high_water_bytes = max(self.high_water_bytes, self.buffered_bytes)
        self.high_water_frames = max(self.high_water_frames, self.buffered_frames)

    def dequeue(self, *, bytes_value: int, frames: int = 1) -> None:
        self.buffered_bytes = max(0, self.buffered_bytes - bytes_value)
        self.buffered_frames = max(0, self.buffered_frames - frames)

    @property
    def over_cap(self) -> bool:
        return self.buffered_bytes > self.cap_bytes or self.buffered_frames > self.cap_frames

    @property
    def pressure_ratio(self) -> float:
        return max(self.buffered_bytes / self.cap_bytes, self.buffered_frames / self.cap_frames)

    def audit(self) -> Dict[str, Any]:
        problems: List[str] = []
        if self.over_cap:
            problems.append(
                f"{self.name} exceeded its cap "
                f"({self.buffered_bytes}/{self.cap_bytes} bytes, "
                f"{self.buffered_frames}/{self.cap_frames} frames); the overflow action "
                f"({self.on_cap}) must have been applied"
            )
        return {"ok": not problems, "problems": problems, "pressure_ratio": self.pressure_ratio}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "owner": self.owner,
            "cap_bytes": self.cap_bytes,
            "cap_frames": self.cap_frames,
            "on_cap": self.on_cap,
            "buffered_bytes": self.buffered_bytes,
            "buffered_frames": self.buffered_frames,
            "high_water_bytes": self.high_water_bytes,
            "high_water_frames": self.high_water_frames,
            "pressure_ratio": self.pressure_ratio,
        }


def default_layers() -> Dict[str, LayerState]:
    """The layer set of the frozen stream lifecycle spec (defaults mirror YAML)."""
    return {
        "application_queue": LayerState(
            name="application_queue",
            cap_bytes=4_194_304,
            cap_frames=256,
            owner="gateway",
            on_cap="cancel_or_close_with_explicit_protocol",
        ),
        "serializer_buffer": LayerState(
            name="serializer_buffer",
            cap_bytes=1_048_576,
            cap_frames=64,
            owner="gateway",
            on_cap="block_writer",
        ),
        "server_buffer": LayerState(
            name="server_buffer",
            cap_bytes=1_048_576,
            cap_frames=1_048_576,
            owner="transport",
            on_cap="apply_backpressure",
        ),
        "socket_send_buffer": LayerState(
            name="socket_send_buffer",
            cap_bytes=262_144,
            cap_frames=262_144,
            owner="kernel",
            on_cap="await_writable",
        ),
        "client_receive_buffer": LayerState(
            name="client_receive_buffer",
            cap_bytes=1_048_576,
            cap_frames=1_048_576,
            owner="loadgen",
            on_cap="read_or_disconnect",
        ),
    }

@dataclass
class FramePipelineEvent:
    """One frame's journey from token-ready to client-read."""

    frame_index: int
    token_id: int
    bytes_written: int = 0
    committed: bool = True
    times_ns: Dict[str, int] = field(default_factory=dict)

    def mark(self, stage: str, value_ns: int) -> None:
        if stage not in FRAME_STAGES:
            raise ConfigError(
                f"unknown frame stage {stage!r}; expected one of {list(FRAME_STAGES)}"
            )
        if stage in self.times_ns:
            raise ConfigError(
                f"frame {self.frame_index}: stage {stage!r} recorded twice"
            )
        self.times_ns[stage] = int(value_ns)

    def problems(self) -> List[str]:
        issues: List[str] = []
        order = [stage for stage in FRAME_STAGES if stage in self.times_ns]
        for left, right in zip(order, order[1:]):
            if self.times_ns[right] < self.times_ns[left]:
                issues.append(
                    f"frame {self.frame_index}: {right} precedes {left} "
                    f"({self.times_ns[right]} < {self.times_ns[left]})"
                )
        if "client_read" in self.times_ns and "write_end" not in self.times_ns:
            issues.append(f"frame {self.frame_index}: client read before write completed")
        return issues

    def as_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "token_id": self.token_id,
            "bytes_written": self.bytes_written,
            "committed": self.committed,
            "times_ns": dict(sorted(self.times_ns.items())),
        }


@dataclass
class StreamPipelineRecorder:
    """Per-frame pipeline trace plus the cancel/terminal audits."""

    request_id: str
    events: List[FramePipelineEvent] = field(default_factory=list)
    layers: Dict[str, LayerState] = field(default_factory=default_layers)
    terminal_frames: List[int] = field(default_factory=list)
    cancel_window: str = ""
    cancel_observed_ns: Optional[int] = None
    frames_in_flight_at_cancel: Sequence[int] = ()
    bytes_already_in_socket_buffer: int = 0

    def record(self, event: FramePipelineEvent) -> None:
        if any(existing.frame_index == event.frame_index for existing in self.events):
            raise ConfigError(
                f"frame {event.frame_index} was recorded twice; a duplicated frame "
                "index would hide a duplicated token"
            )
        self.events.append(event)

    def mark_terminal(self, frame_index: int) -> None:
        if self.terminal_frames:
            raise ConfigError(
                "a second terminal frame was recorded; the terminal frame must be "
                "written at most once (E08-08 §10)"
            )
        self.terminal_frames.append(frame_index)

    def declare_cancel(self, *, window: str, observed_ns: int, in_flight: Sequence[int] = ()) -> None:
        if window not in CANCEL_WINDOWS:
            raise ConfigError(
                f"unknown cancel window {window!r}; E08-08 §7 freezes the list "
                f"{list(CANCEL_WINDOWS)}",
                details={"window": window},
            )
        self.cancel_window = window
        self.cancel_observed_ns = int(observed_ns)
        self.frames_in_flight_at_cancel = tuple(in_flight)

    def audit(self) -> Dict[str, Any]:
        problems: List[str] = []
        indices = [event.frame_index for event in self.events]
        if indices != sorted(indices):
            problems.append("frame indices are not monotonic")
        if len(set(indices)) != len(indices):
            problems.append("frame indices are not unique")
        for event in self.events:
            problems.extend(event.problems())
        if len(self.terminal_frames) > 1:
            problems.append("more than one terminal frame")
        for layer in self.layers.values():
            audit = layer.audit()
            problems.extend(audit["problems"])
        if self.cancel_observed_ns is not None:
            allowed = ALLOWED_AFTER_CANCEL.get(self.cancel_window, ())
            in_flight = set(self.frames_in_flight_at_cancel)
            for event in self.events:
                if event.frame_index in in_flight:
                    continue
                for stage, value in event.times_ns.items():
                    if stage in ("terminal_write", "write_start", "write_end", "client_read") and (
                        value > self.cancel_observed_ns and stage not in allowed
                    ):
                        problems.append(
                            f"frame {event.frame_index}: {stage} happened after cancel "
                            f"({value} > {self.cancel_observed_ns}) outside the "
                            f"'{self.cancel_window}' window; no new client-visible "
                            "token may be committed after the linearisation point"
                        )
        return {
            "ok": not problems,
            "problems": problems,
            "frames": len(self.events),
            "terminal_frames": list(self.terminal_frames),
            "layers": {name: layer.as_dict() for name, layer in sorted(self.layers.items())},
            "bytes_already_in_socket_buffer": self.bytes_already_in_socket_buffer,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "cancel_window": self.cancel_window,
            "cancel_observed_ns": self.cancel_observed_ns,
            "events": [event.as_dict() for event in self.events],
            "audit": self.audit(),
        }


def resource_slope(
    series: Mapping[str, Sequence[float]],
    *,
    warmup: int = 3,
    tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Steady-state growth check (fd/RSS/KV/task/socket slopes)."""
    rows: List[Dict[str, Any]] = []
    ok = True
    for name, values in sorted(series.items()):
        if len(values) <= warmup + 1:
            rows.append(
                {
                    "resource": name,
                    "ok": False,
                    "slope_per_cycle": None,
                    "reason": "not enough post-warmup cycles to fit a slope",
                }
            )
            ok = False
            continue
        tail = [float(value) for value in values[warmup:]]
        n = len(tail)
        mean_x = (n - 1) / 2.0
        mean_y = sum(tail) / n
        denominator = sum((index - mean_x) ** 2 for index in range(n))
        slope = (
            sum((index - mean_x) * (value - mean_y) for index, value in enumerate(tail))
            / denominator
            if denominator
            else 0.0
        )
        within = slope <= tolerance
        ok = ok and within
        rows.append(
            {
                "resource": name,
                "ok": within,
                "slope_per_cycle": slope,
                "tolerance": tolerance,
                "reason": "" if within else "resource keeps growing in steady state",
            }
        )
    return {"ok": ok, "rows": rows}


def long_run_pass(slopes: Mapping[str, Any]) -> Dict[str, Any]:
    """Growth in steady state blocks a PASS (never 'small leak' as a pass)."""
    growth = [row for row in slopes.get("rows", []) if not row["ok"]]
    return {
        "pass_allowed": not growth,
        "blocking_resources": [row["resource"] for row in growth],
        "note": "a positive slope in steady state is a leak, not noise",
    }


def token_shortfall_explained(ledger: DeliveryLedger) -> bool:
    """The client-visible shortfall must be attributed, not merely tolerated."""
    return ledger.audit()["ok"]


def summarize_layers(layers: Mapping[str, LayerState]) -> Dict[str, Any]:
    return {
        "max_pressure_ratio": max(layer.pressure_ratio for layer in layers.values()),
        "layers": {name: layer.as_dict() for name, layer in sorted(layers.items())},
        "note": (
            "a flat application queue with growing RSS/socket means the pressure "
            "moved to another buffer layer"
        ),
    }


def delivered_token_ids(events: Iterable[FramePipelineEvent]) -> Tuple[int, ...]:
    return tuple(event.token_id for event in events if "write_end" in event.times_ns)


__all__ = [
    "ALLOWED_AFTER_CANCEL",
    "CANCEL_WINDOWS",
    "DeliveryLedger",
    "FRAME_STAGES",
    "FramePipelineEvent",
    "LAYER_NAMES",
    "LEDGER_STAGES",
    "LayerState",
    "StreamPipelineRecorder",
    "default_layers",
    "delivered_token_ids",
    "long_run_pass",
    "resource_slope",
    "summarize_layers",
    "token_shortfall_explained",
]
