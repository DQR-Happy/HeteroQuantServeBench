"""C7 — TraceEvent contract.

Unified event model for model load, queue, prefill, decode, cache,
collective, kernel, network, and output events, using monotonic timestamps
and trace/span identifiers (top-level architecture §5 C7).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field

from hqsb.core.contracts.base import VersionedModel


class TraceEventType(str, Enum):
    """Canonical trace event kinds."""

    MODEL_LOAD = "model_load"
    QUEUE = "queue"
    PREFILL = "prefill"
    DECODE = "decode"
    CACHE = "cache"
    COLLECTIVE = "collective"
    KERNEL = "kernel"
    NETWORK = "network"
    OUTPUT = "output"


class TraceEvent(VersionedModel):
    """A single timestamped event within a trace."""

    SCHEMA_VERSION = "1.0.0"

    event_type: TraceEventType = Field(..., description="Event kind.")
    timestamp_ns: int = Field(
        ...,
        ge=0,
        description="Monotonic timestamp in nanoseconds (time.monotonic_ns).",
    )
    trace_id: str = Field(..., description="Correlating trace identifier.")
    span_id: str = Field(..., description="Unique span identifier.")
    parent_span_id: Optional[str] = Field(
        default=None, description="Parent span identifier, if nested."
    )
    name: Optional[str] = Field(default=None, description="Human-readable name.")
    attributes: dict = Field(
        default_factory=dict, description="Structured event attributes."
    )


__all__ = ["TraceEvent", "TraceEventType"]
