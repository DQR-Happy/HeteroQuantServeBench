"""Identifiers for run, trace, and span correlation.

HQSB benchmarks and services are expected to be auditable end-to-end: a
single ``run_id`` identifies one experiment, a ``trace_id`` identifies one
request (or one benchmark pass), and a ``span_id`` identifies a unit of
work within that trace. These identifiers are embedded in
:class:`hqsb.core.contracts.BenchmarkResult` and
:class:`hqsb.core.contracts.TraceEvent` so raw samples, logs, and reports
can be correlated after the fact.
"""

from __future__ import annotations

import time
import uuid


def new_run_id() -> str:
    """Generate a globally unique, sortable experiment run identifier.

    The format is ``run_<unix_ms>_<uuid8>``, e.g. ``run_1723456789123_a1b2c3d4``.
    The millisecond epoch prefix keeps identifiers lexicographically
    orderable by creation time, which is convenient for file/database
    layouts.
    """
    return f"run_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def new_trace_id() -> str:
    """Generate a globally unique request/benchmark-pass trace identifier."""
    return uuid.uuid4().hex


def new_span_id() -> str:
    """Generate a globally unique span identifier (16 hex characters)."""
    return uuid.uuid4().hex[:16]
