"""Unit tests for run/trace/span identifier generation."""

from __future__ import annotations

import pytest

from hqsb.core.ids import new_run_id, new_span_id, new_trace_id


@pytest.mark.unit
class TestIdentifiers:
    def test_run_id_format(self):
        run_id = new_run_id()
        assert run_id.startswith("run_")
        # run_<ms>_<8 hex chars>
        prefix, ts, suffix = run_id.split("_")
        assert ts.isdigit()
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_run_ids_are_unique(self):
        ids = {new_run_id() for _ in range(100)}
        assert len(ids) == 100

    def test_trace_id_is_32_hex(self):
        trace_id = new_trace_id()
        assert len(trace_id) == 32
        assert all(c in "0123456789abcdef" for c in trace_id)

    def test_span_id_is_16_hex(self):
        span_id = new_span_id()
        assert len(span_id) == 16
