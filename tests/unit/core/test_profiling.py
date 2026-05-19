"""Unit tests for profiler table extraction and shape normalization."""

from __future__ import annotations

import pytest

from hqsb.benchmark.profiling import (
    _norm_shapes,
    attach_time_share,
    cumulative_kernel_time_us,
    extract_operator_table,
    scope_totals_us,
    split_by_scope,
)
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

    def test_scope_distinguishes_kernel_from_cpu_op(self):
        # Device kernels have no host self time (cpu == 0); host-side rows
        # (ATen ops and CUDA runtime API calls) do.
        prof = _FakeProfiler(
            [
                _FakeEvent("aten::mm", 1, 100.0, 50.0, 0, ((1, 128), (128, 128))),
                _FakeEvent("ampere_fp16_gemm", 1, 0.0, 50.0, 0, ()),
                _FakeEvent("cudaLaunchKernel", 1, 5.0, 0.0, 0, ()),
            ]
        )
        table = {r["name"]: r for r in extract_operator_table(prof)}
        assert table["aten::mm"]["scope"] == "cpu"
        assert table["ampere_fp16_gemm"]["scope"] == "kernel"
        assert table["cudaLaunchKernel"]["scope"] == "cpu"


@pytest.mark.unit
class TestCumulativeKernelTime:
    """The op and kernel scopes must never be summed together."""

    def test_counts_kernel_scope_only(self):
        rows = [
            {"name": "aten::mm", "cuda_time_us": 100.0, "scope": "cpu", "count": 1},
            {"name": "gemm_kernel", "cuda_time_us": 100.0, "scope": "kernel", "count": 1},
        ]
        # Naive sum would be 200 (2x); the cumulative kernel work is 100.
        assert cumulative_kernel_time_us(rows) == 100.0

    def test_falls_back_to_all_rows_without_scope(self):
        rows = [
            {"name": "a", "cuda_time_us": 10.0},
            {"name": "b", "cuda_time_us": 5.0},
        ]
        assert cumulative_kernel_time_us(rows) == 15.0

    def test_empty_table_is_zero(self):
        assert cumulative_kernel_time_us([]) == 0.0

    def test_scope_totals_are_separate(self):
        rows = [
            {"name": "aten::mm", "cuda_time_us": 100.0, "scope": "cpu"},
            {"name": "gemm_kernel", "cuda_time_us": 100.0, "scope": "kernel"},
            {"name": "gemm2", "cuda_time_us": 30.0, "scope": "kernel"},
        ]
        totals = scope_totals_us(rows)
        assert totals["cpu"] == 100.0
        assert totals["kernel"] == 130.0


@pytest.mark.unit
class TestScopeSplitAndShare:
    def test_split_by_scope(self):
        rows = [
            {"name": "aten::mm", "cuda_time_us": 100.0, "scope": "cpu"},
            {"name": "gemm_kernel", "cuda_time_us": 100.0, "scope": "kernel"},
            {"name": "aten::mul", "cuda_time_us": 50.0, "scope": "cpu"},
        ]
        ops, kernels = split_by_scope(rows)
        assert [r["name"] for r in ops] == ["aten::mm", "aten::mul"]
        assert [r["name"] for r in kernels] == ["gemm_kernel"]

    def test_share_normalised_within_each_scope(self):
        rows = [
            {"name": "aten::mm", "cuda_time_us": 90.0, "scope": "cpu"},
            {"name": "aten::mul", "cuda_time_us": 10.0, "scope": "cpu"},
            {"name": "gemm_kernel", "cuda_time_us": 300.0, "scope": "kernel"},
            {"name": "copy_kernel", "cuda_time_us": 100.0, "scope": "kernel"},
        ]
        out = {r["name"]: r["time_share"] for r in attach_time_share(rows)}
        # cpu scope normalises by 100, kernel scope by 400 -> no cross-scope 2x.
        assert out["aten::mm"] == pytest.approx(0.9)
        assert out["gemm_kernel"] == pytest.approx(0.75)
        assert out["copy_kernel"] == pytest.approx(0.25)
