"""Unit tests for profiler table extraction and shape normalization."""

from __future__ import annotations

import pytest

from hqsb.benchmark.profiling import _norm_shapes, extract_operator_table
from hqsb.core.errors import BenchmarkError


@pytest.mark.unit
class TestNormShapes:
    def test_empty(self):
        assert _norm_shapes(None) == []

    def test_normalizes_tuples(self):
        assert _norm_shapes([(1, 128), (1, 256)]) == ["(1, 128)", "(1, 256)"]

    def test_dedupes_and_bounds(self):
        shapes = [(1, 1)] * 100
        assert _norm_shapes(shapes, limit=8) == ["(1, 1)"]

    def test_unhashable_fallback(self):
        assert _norm_shapes(42) == []


class _FakeEvent:
    def __init__(self, key, count, cpu, cuda, mem, shapes):
        self.key = key
        self.count = count
        self.self_cpu_time_total = cpu
        # PyTorch >= 2.0 uses ``self_device_time_total`` (device-agnostic).
        self.self_device_time_total = cuda
        self.self_device_memory_usage = mem
        self.input_shapes = shapes


class _FakeProfiler:
    def __init__(self, events):
        self._events = events

    def key_averages(self):
        return self._events

    def events(self):
        return self._events


@pytest.mark.unit
class TestExtractOperatorTable:
    def test_sorts_by_cuda_time(self):
        prof = _FakeProfiler(
            [
                _FakeEvent("aten::add", 10, 1.0, 5.0, 0, (1, 1)),
                _FakeEvent("aten::mm", 2, 2.0, 50.0, 100, (1, 128)),
            ]
        )
        table = extract_operator_table(prof)
        assert table[0]["name"] == "aten::mm"
        assert table[0]["count"] == 2
        assert table[0]["cuda_time_us"] == 50.0
        assert table[0]["device_memory_bytes"] == 100
        assert table[0]["input_shapes"] == ["(1, 128)"]

    def test_legacy_cuda_time_fallback(self):
        # Older PyTorch exposes ``self_cuda_time_total`` instead of
        # ``self_device_time_total``; ensure the fallback path works.
        class LegacyEvent:
            key = "aten::mm"
            count = 1
            self_cpu_time_total = 1.0
            self_cuda_time_total = 42.0
            self_device_memory_usage = 0

        class LegacyProfiler:
            def key_averages(self):
                return [LegacyEvent()]

            def events(self):
                return []

        table = extract_operator_table(LegacyProfiler())
        assert table[0]["cuda_time_us"] == 42.0

    def test_failure_raises_benchmark_error(self):
        class BadProfiler:
            def key_averages(self):
                raise RuntimeError("boom")

        with pytest.raises(BenchmarkError):
            extract_operator_table(BadProfiler())
