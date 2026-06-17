"""Transport boundary: what a gateway needs from an HTTP/SSE connection.

The gateway never touches ``wfile``/``socket`` directly; it writes through a
:class:`TransportWriter` that can report backpressure (a write that cannot make
progress) and client disconnect.  Two implementations exist:

* :class:`InProcessTransport` — deterministic, scripted, no sockets.  Used by
  unit tests and the smoke self-check, and by slow-client planning; every
  timestamp it produces is *modelled* (``simulated=True``);
* :class:`hqsb.serving.transport_http.StdlibHttpTransport` — a real HTTP/1.1 +
  SSE binding on the standard library, so the service can actually be driven
  end-to-end on a development host.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple

from hqsb.core.errors import ConfigError

from hqsb.serving.clients import ClientBehavior, ClientScript, simulate_client_read

#: Methods the frozen subset serves.
ALLOWED_METHODS: Tuple[str, ...] = ("GET", "POST")


@dataclass(frozen=True)
class TransportRequest:
    """One request as it arrived from the wire."""

    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes
    received_ns: int
    connection_id: str = ""
    remote: str = ""

    def header(self, name: str, default: str = "") -> str:
        lowered = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lowered:
                return value
        return default

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "path": self.path,
            "headers": {key.lower(): value for key, value in sorted(self.headers.items())},
            "body_bytes": len(self.body),
            "received_ns": self.received_ns,
            "connection_id": self.connection_id,
            "remote": self.remote,
        }


class TransportWriter(Protocol):
    """What the gateway may do with a connection."""

    def write_head(self, status: int, headers: Mapping[str, str]) -> None: ...

    def write_body(self, chunk: bytes) -> bool:
        """Write bytes; ``False`` means "no progress, apply backpressure"."""

    def flush(self) -> None: ...

    def close(self) -> None: ...

    def client_connected(self) -> bool: ...

    def buffered_bytes(self) -> int: ...

    def now_ns(self) -> int: ...


@dataclass
class WriteRecord:
    """One write as the transport observed it (raw evidence, not a summary)."""

    index: int
    chunk: bytes
    start_ns: int
    end_ns: int
    accepted: bool
    backpressured: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "bytes": len(self.chunk),
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "accepted": self.accepted,
            "backpressured": self.backpressured,
        }


@dataclass
class ClientOutcome:
    """The simulated client's view of one exchange."""

    request: TransportRequest
    status: int
    headers: Mapping[str, str]
    writes: List[WriteRecord] = field(default_factory=list)
    body: bytes = b""
    backpressure_events: int = 0
    client_read_report: Optional[Mapping[str, Any]] = None
    disconnected_by_client: bool = False
    simulated: bool = True

    @property
    def raw_response(self) -> bytes:
        return self.body

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request": self.request.as_dict(),
            "status": self.status,
            "headers": {key.lower(): value for key, value in sorted(self.headers.items())},
            "writes": [write.as_dict() for write in self.writes],
            "body_bytes": len(self.body),
            "backpressure_events": self.backpressure_events,
            "client_read": dict(self.client_read_report or {}),
            "disconnected_by_client": self.disconnected_by_client,
            "simulated": self.simulated,
        }


class _InProcessWriter:
    """Writer that applies a scripted client to the byte stream (simulated)."""

    def __init__(
        self,
        transport: "InProcessTransport",
        behavior: ClientBehavior,
        script: ClientScript,
    ) -> None:
        self._transport = transport
        self._behavior = behavior
        self._script = script
        self._head: Optional[Tuple[int, Mapping[str, str]]] = None
        self._writes: List[WriteRecord] = []
        self._body = bytearray()
        self._closed = False
        self._connected = True
        self._backpressure = 0
        self._frames: List[Tuple[int, bytes, int]] = []
        self._frame_index = 0
        self._cursor_ns = script.headers_received_ns

    # -- API ------------------------------------------------------------
    def write_head(self, status: int, headers: Mapping[str, str]) -> None:
        if self._head is not None:
            raise ConfigError("write_head called twice for the same response")
        self._head = (status, dict(headers))
        self._cursor_ns = max(self._cursor_ns, self._script.headers_received_ns)
        if self._behavior.never_read_after_headers:
            self._connected = False  # headers delivered, body never read

    def write_body(self, chunk: bytes, *, is_frame: bool = False) -> bool:
        if self._closed:
            return False
        if not self._connected:
            self._backpressure += 1
            return False
        start = self.now_ns()
        # the scripted client may be gone before the socket cap is reached (abort,
        # read-N-then-close, never-read): the write must fail, not silently succeed
        if not self._transport.client_accepts(self):
            self._connected = False
            self._writes.append(
                WriteRecord(len(self._writes), chunk, start, self.now_ns(), False, True)
            )
            return False
        # socket-level backpressure: the transport refuses to buffer more than the
        # configured high watermark until the client reads (E08-08 §6)
        if self.buffered_bytes() + len(chunk) > self._transport.socket_high_watermark_bytes:
            self._backpressure += 1
            self._transport.record_backpressure(self._script.connection_id, len(chunk))
            self._connected = self._transport.client_drains(self)
            if not self._connected:
                self._writes.append(
                    WriteRecord(len(self._writes), chunk, start, self.now_ns(), False, True)
                )
                return False
        end = self.now_ns()
        record = WriteRecord(len(self._writes), chunk, start, end, True, False)
        self._writes.append(record)
        self._body.extend(chunk)
        if is_frame or b"\n\n" in chunk:
            self._frames.append((self._frame_index, chunk, end))
            self._frame_index += 1
        return True

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self._closed = True

    def client_connected(self) -> bool:
        return self._connected and not self._closed

    def buffered_bytes(self) -> int:
        return len(self._body)

    def now_ns(self) -> int:
        # deterministic, ordered clock: the transport owns time in simulation
        self._cursor_ns += self._transport.tick_ns
        return self._cursor_ns

    # -- outcome --------------------------------------------------------
    def outcome(self, request: TransportRequest) -> ClientOutcome:
        status, headers = self._head or (0, {})
        report = None
        if self._frames:
            read_report = simulate_client_read(self._frames, self._script)
            report = read_report.as_dict()
        return ClientOutcome(
            request=request,
            status=status,
            headers=headers,
            writes=list(self._writes),
            body=bytes(self._body),
            backpressure_events=self._backpressure,
            client_read_report=report,
            disconnected_by_client=not self._connected,
        )


class InProcessTransport:
    """A deterministic transport with a scripted client (``simulated=True``)."""

    def __init__(
        self,
        *,
        script: ClientScript,
        tick_ns: int = 1_000_000,
        socket_high_watermark_bytes: int = 262_144,
    ) -> None:
        self.script = script
        self.tick_ns = tick_ns
        self.socket_high_watermark_bytes = socket_high_watermark_bytes
        self.backpressure_records: List[Dict[str, Any]] = []
        #: Whatever the last handler call returned (the gateway's outcome).
        self.last_handler_result: Any = None

    def record_backpressure(self, connection_id: str, bytes_value: int) -> None:
        self.backpressure_records.append(
            {"connection_id": connection_id, "bytes": bytes_value}
        )

    def client_accepts(self, writer: _InProcessWriter) -> bool:
        """Is the scripted client still able to receive another frame?"""
        behavior = self.script.behavior
        if behavior.never_read_after_headers:
            return False
        if behavior.abort_after_frames and writer._frame_index >= behavior.abort_after_frames:
            return False
        if (
            behavior.read_frames_then_close
            and writer._frame_index >= behavior.read_frames_then_close
        ):
            return False
        return True

    def client_drains(self, writer: _InProcessWriter) -> bool:
        """Ask the scripted client to drain a full socket buffer."""
        del writer
        behavior = self.script.behavior
        if behavior.never_read_after_headers:
            return False
        return True

    def run(
        self,
        request: TransportRequest,
        handler: Callable[[TransportRequest, TransportWriter], Any],
    ) -> ClientOutcome:
        """Run one exchange; the handler's return value is kept for the caller.

        The gateway returns its :class:`~hqsb.serving.gateway.RequestOutcome`;
        keeping it next to the client-side view lets a test assert on *both* the
        wire bytes and the service-side evidence.
        """
        writer = _InProcessWriter(self, self.script.behavior, self.script)
        self.last_handler_result = handler(request, writer)
        return writer.outcome(request)


def build_request(
    *,
    method: str,
    path: str,
    body: bytes = b"",
    headers: Optional[Mapping[str, str]] = None,
    received_ns: int = 0,
    connection_id: str = "conn-0",
) -> TransportRequest:
    if method not in ALLOWED_METHODS:
        raise ConfigError(f"method {method!r} is outside the frozen transport subset")
    return TransportRequest(
        method=method,
        path=path,
        headers=dict(headers or {}),
        body=body,
        received_ns=received_ns,
        connection_id=connection_id,
    )


def monotonic_ns() -> int:
    """The default clock for real transports (monotonic, never wall clock)."""
    return time.monotonic_ns()


__all__ = [
    "ALLOWED_METHODS",
    "ClientOutcome",
    "InProcessTransport",
    "TransportRequest",
    "TransportWriter",
    "WriteRecord",
    "build_request",
    "monotonic_ns",
]
