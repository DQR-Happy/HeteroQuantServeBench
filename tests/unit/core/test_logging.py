"""Unit tests for structured logging and trace context."""

from __future__ import annotations

import json
import logging

import pytest

from hqsb.core.logging import (
    JsonLineFormatter,
    configure_logging,
    get_span_id,
    get_trace_id,
    set_trace_context,
)


@pytest.mark.unit
class TestTraceContext:
    def test_defaults_to_none(self):
        assert get_trace_id() is None
        assert get_span_id() is None

    def test_set_and_get(self):
        set_trace_context("abc123", span_id="span1")
        assert get_trace_id() == "abc123"
        assert get_span_id() == "span1"
        set_trace_context("other")
        assert get_trace_id() == "other"
        assert get_span_id() is None


@pytest.mark.unit
class TestJsonLineFormatter:
    def _record(self, msg: str, **extra) -> logging.LogRecord:
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname=__file__,
            lineno=1, msg=msg, args=(), exc_info=None,
        )
        record.created = 12345.0
        record.hqsb_extra = extra if extra else None
        return record

    def test_emits_json_with_basic_fields(self):
        formatter = JsonLineFormatter()
        payload = json.loads(formatter.format(self._record("hello")))
        assert payload["level"] == "INFO"
        assert payload["logger"] == "test"
        assert payload["message"] == "hello"
        assert payload["ts"] == 12345.0

    def test_includes_trace_context_when_set(self):
        set_trace_context("trace1", span_id="span2")
        formatter = JsonLineFormatter()
        payload = json.loads(formatter.format(self._record("x")))
        assert payload["trace_id"] == "trace1"
        assert payload["span_id"] == "span2"
        set_trace_context("clear")  # reset span

    def test_merges_extra_structured_metrics(self):
        formatter = JsonLineFormatter()
        payload = json.loads(formatter.format(self._record("m", metric=42)))
        assert payload["metric"] == 42


@pytest.mark.unit
class TestConfigureLogging:
    def test_configure_replaces_handlers(self):
        configure_logging(level=logging.INFO)
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert root.level == logging.INFO
