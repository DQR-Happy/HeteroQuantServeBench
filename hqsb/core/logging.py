"""Structured logging with run/trace context propagation.

HQSB emits structured logs (JSON lines) so logs can be parsed and joined
with benchmark results and traces. A ``trace_id`` (and optional ``span_id``)
is carried through :mod:`contextvars` so every log record produced while
handling a request or benchmark pass is automatically tagged, without
threading those identifiers through every function signature.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from typing import Any, Dict, Optional

# Context variables holding the ambient trace/span for the current
# async/task-local execution context. They propagate across threads spawned
# by ``asyncio``/``concurrent.futures`` but not across process boundaries.
_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "hqsb_trace_id", default=None
)
_span_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "hqsb_span_id", default=None
)


def set_trace_context(trace_id: str, span_id: Optional[str] = None) -> None:
    """Set the ambient trace/span identifiers for the current context."""
    _trace_id_var.set(trace_id)
    _span_id_var.set(span_id)


def get_trace_id() -> Optional[str]:
    """Return the ambient trace identifier, if any."""
    return _trace_id_var.get()


def get_span_id() -> Optional[str]:
    """Return the ambient span identifier, if any."""
    return _span_id_var.get()


class JsonLineFormatter(logging.Formatter):
    """Format log records as single-line JSON objects.

    The emitted object always contains ``ts`` (epoch seconds), ``level``,
    ``logger``, and ``message``. ``trace_id``/``span_id`` are included when
    set in the current context. Any extra keys passed to the log call (e.g.
    ``logger.info("x", extra={"metric": 1})``) are merged in, allowing
    structured metrics to flow into the log stream.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        trace_id = _trace_id_var.get()
        span_id = _span_id_var.get()
        if trace_id is not None:
            payload["trace_id"] = trace_id
        if span_id is not None:
            payload["span_id"] = span_id

        extra = getattr(record, "hqsb_extra", None)
        if extra:
            payload.update(extra)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    level: int = logging.INFO,
    *,
    stream=None,
    json_format: bool = True,
) -> None:
    """Configure the root logger for HQSB.

    Args:
        level: Logging level for the root logger.
        stream: Output stream (defaults to ``sys.stderr``).
        json_format: When True (default), emit JSON lines; otherwise a
            human-readable format is used.
    """
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    if json_format:
        handler.setFormatter(JsonLineFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


# Re-export ``time`` to keep a stable module surface for tests that patch
# clock behavior indirectly. (Kept private; not part of public API.)
__all__ = [
    "JsonLineFormatter",
    "configure_logging",
    "get_span_id",
    "get_trace_id",
    "set_trace_context",
]
