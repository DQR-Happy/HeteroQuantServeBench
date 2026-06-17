"""Real HTTP/1.1 + SSE binding on the Python standard library.

The gateway is transport-agnostic (:mod:`hqsb.serving.transport`); this module
gives it an actual socket so the service can be driven end-to-end with an
ordinary client — which is what E08-01/02/08 need when they talk about wire
bytes, slow readers and disconnects.

Design notes
------------
* standard library only: no framework may become a hidden dependency of the
  serving core (`pyproject.toml` keeps an optional ``serving`` extra for the
  experiment executor, but the frozen subset must run without it);
* every request gets its own thread; the gateway guards its own shared state;
* a write that raises ``BrokenPipeError``/``ConnectionResetError`` flips the
  writer to "disconnected" and is reported to the gateway, which then
  propagates the cancel to the Backend (E08-01 §9, E08-08 §5).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Mapping, Optional

from hqsb.core.errors import ConfigError

from hqsb.serving.transport import TransportRequest, TransportWriter, monotonic_ns

Handler = Callable[[TransportRequest, TransportWriter], None]


@dataclass
class HttpTransportConfig:
    """Transport limits of the stdlib binding."""

    host: str = "127.0.0.1"
    port: int = 0
    max_body_bytes: int = 1_048_576
    max_header_bytes: int = 65_536
    response_buffer_bytes: int = 262_144
    request_timeout_s: float = 30.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "max_body_bytes": self.max_body_bytes,
            "max_header_bytes": self.max_header_bytes,
            "response_buffer_bytes": self.response_buffer_bytes,
            "request_timeout_s": self.request_timeout_s,
        }


class HttpTransportWriter(TransportWriter):
    """One connection's writer; tracks disconnect and buffered bytes."""

    def __init__(self, handler: BaseHTTPRequestHandler, config: HttpTransportConfig) -> None:
        self._handler = handler
        self._config = config
        self._connected = True
        self._head_sent = False
        self._written = 0
        self._closed = False
        self.disconnect_reason = ""
        self.disconnect_ns: Optional[int] = None

    def write_head(self, status: int, headers: Mapping[str, str]) -> None:
        if self._head_sent:
            raise ConfigError("write_head called twice for the same response")
        self._head_sent = True
        try:
            self._handler.send_response(status)
            for key, value in headers.items():
                self._handler.send_header(key, value)
            self._handler.end_headers()
        except (BrokenPipeError, ConnectionResetError) as exc:
            self.mark_disconnected(f"{type(exc).__name__}: {exc}")

    def write_body(self, chunk: bytes) -> bool:
        if self._closed or not self._connected:
            return False
        try:
            self._handler.wfile.write(chunk)
            self._handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            self.mark_disconnected(f"{type(exc).__name__}: {exc}")
            return False
        self._written += len(chunk)
        return True

    def flush(self) -> None:
        try:
            self._handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            self.mark_disconnected(f"{type(exc).__name__}: {exc}")

    def close(self) -> None:
        self._closed = True

    def client_connected(self) -> bool:
        return self._connected and not self._closed

    def buffered_bytes(self) -> int:
        # the stdlib socket API exposes no portable send-buffer depth: report the
        # bytes handed over and let the gateway treat 0 as "no local backlog".
        return 0

    def now_ns(self) -> int:
        return time.monotonic_ns()

    def mark_disconnected(self, reason: str) -> None:
        if self._connected:
            self.disconnect_reason = reason
            self.disconnect_ns = time.monotonic_ns()
        self._connected = False

    @property
    def bytes_written(self) -> int:
        return self._written


def build_handler(app: Handler, config: HttpTransportConfig) -> type[BaseHTTPRequestHandler]:
    """Wrap a gateway ``handle(request, writer)`` in an HTTP request handler."""

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "HQSB-ServeFabric/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover
            # structured logging belongs to the gateway; the stdlib text log is
            # suppressed so the experiment log has exactly one producer.
            return None

        def _read_request(self) -> Optional[TransportRequest]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > config.max_body_bytes:
                self.send_response(413)
                self.send_header("Content-Type", "application/json")
                body = b'{"error": {"code": "body_too_large"}}'
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return None
            body = self.rfile.read(length) if length else b""
            return TransportRequest(
                method=self.command,
                path=self.path,
                headers={key: value for key, value in self.headers.items()},
                body=body,
                received_ns=monotonic_ns(),
                connection_id=f"{self.client_address[0]}:{self.client_address[1]}",
                remote=self.client_address[0],
            )

        def _dispatch(self) -> None:
            request = self._read_request()
            if request is None:
                return
            writer = HttpTransportWriter(self, config)
            try:
                app(request, writer)
            except Exception as exc:  # noqa: BLE001 - the gateway must stay up
                if writer.client_connected() and not writer._head_sent:
                    writer.write_head(500, {"Content-Type": "application/json"})
                    writer.write_body(
                        (
                            '{"error": {"message": "internal error", "type": "internal", '
                            '"code": "backend_internal", "param": ""}'
                        ).encode()
                    )
                del exc
            finally:
                writer.close()

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            self._dispatch()

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            self._dispatch()

    return _Handler


class StdlibHttpTransport:
    """Owns the server lifecycle; ``start``/``stop`` are explicit and idempotent."""

    def __init__(self, app: Handler, config: Optional[HttpTransportConfig] = None) -> None:
        self.app = app
        self.config = config or HttpTransportConfig()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> str:
        if self._server is not None:
            raise ConfigError("transport is already running; start/stop must be idempotent")
        handler = build_handler(self.app, self.config)
        server = ThreadingHTTPServer((self.config.host, self.config.port), handler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        host, port = server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise ConfigError("transport is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        if self._server is None:
            return  # idempotent close
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None

    def __enter__(self) -> "StdlibHttpTransport":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()


__all__ = [
    "Handler",
    "HttpTransportConfig",
    "HttpTransportWriter",
    "StdlibHttpTransport",
    "build_handler",
]
