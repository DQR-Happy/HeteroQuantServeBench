"""Unit tests for :mod:`hqsb.benchmark.metrics`.

These tests are pure-Python (no PyTorch, no GPU, no model weights) and
validate the statistical contracts used by every benchmark report:
percentile interpolation, latency summary fields, and numerical error
metrics for operator replacement regression.
"""

from __future__ import annotations

import math

import pytest

from hqsb.benchmark.metrics import (
    latency_summary,
    model_core_timings,
    numerical_diff_summary,
    percentile,
)


class TestPercentile:
    def test_median_of_odd_count(self):
        assert percentile([1.0, 2.0, 3.0], 0.50) == pytest.approx(2.0)

    def test_median_of_even_count_interpolates(self):
        # Position (4 - 1) * 0.5 = 1.5 -> midpoint of elements 1 and 2.
        assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == pytest.approx(2.5)

    def test_p95_matches_numpy_linear(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        # numpy.percentile(values, 95, method="linear") == 48.0
        assert percentile(values, 0.95) == pytest.approx(48.0)

    def test_single_element(self):
        assert percentile([42.0], 0.95) == pytest.approx(42.0)

    def test_unsorted_input(self):
        assert percentile([3.0, 1.0, 2.0], 0.50) == pytest.approx(2.0)

    def test_empty_returns_nan(self):
        assert math.isnan(percentile([], 0.50))

    @pytest.mark.parametrize("quantile", [-0.01, 1.01])
    def test_out_of_range_raises(self, quantile):
        with pytest.raises(ValueError):
            percentile([1.0], quantile)


class TestLatencySummary:
    def test_complete_fields(self):
        summary = latency_summary([100.0, 200.0, 300.0])
        assert summary["count"] == 3
        assert summary["mean_ms"] == pytest.approx(200.0)
        assert summary["median_ms"] == pytest.approx(200.0)
        assert summary["min_ms"] == pytest.approx(100.0)
        assert summary["max_ms"] == pytest.approx(300.0)
        assert summary["p50_ms"] == pytest.approx(200.0)
        # Population stddev of {100, 200, 300} is sqrt(20000/3).
        assert summary["stddev_ms"] == pytest.approx(math.sqrt(20000 / 3))

    def test_empty_returns_empty_dict(self):
        assert latency_summary([]) == {}


class TestModelCoreTimings:
    """E02-01: the model-core clock convention must be a single function.

    The derived quantities (decode_total / TTFT / E2E) are fixed by one
    relationship and must be reproduced exactly here, so model_core and the
    benchmark engine can never drift.
    """

    def test_decode_total_is_sum_of_itl(self):
        timings = model_core_timings(100.0, 0.5, [2.0, 3.0, 4.0])
        assert timings["decode_total_ms"] == pytest.approx(9.0)

    def test_ttft_is_prefill_plus_selection(self):
        timings = model_core_timings(100.0, 0.5, [2.0, 3.0, 4.0])
        assert timings["model_core_ttft_ms"] == pytest.approx(100.5)

    def test_e2e_is_ttft_plus_decode(self):
        timings = model_core_timings(100.0, 0.5, [2.0, 3.0, 4.0])
        assert timings["model_core_e2e_ms"] == pytest.approx(109.5)

    def test_empty_itl(self):
        timings = model_core_timings(50.0, 1.0, [])
        assert timings["decode_total_ms"] == pytest.approx(0.0)
        assert timings["model_core_ttft_ms"] == pytest.approx(51.0)
        assert timings["model_core_e2e_ms"] == pytest.approx(51.0)

    def test_single_step_e2e_equals_ttft_plus_one_itl(self):
        # OSL=2 -> one decode step; E2E must be prefill+selection+that step.
        timings = model_core_timings(200.0, 0.0, [10.0])
        assert timings["model_core_e2e_ms"] == pytest.approx(210.0)


class TestNumericalDiffSummary:
    def test_identical_sequences(self):
        result = numerical_diff_summary([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert result["max_abs_error"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["cosine_similarity"] == pytest.approx(1.0)
        assert result["l2_relative_error"] == pytest.approx(0.0)

    def test_known_difference(self):
        result = numerical_diff_summary([0.0, 1.0], [1.0, 1.0])
        assert result["max_abs_error"] == pytest.approx(1.0)
        assert result["mean_abs_error"] == pytest.approx(0.5)
        assert result["rmse"] == pytest.approx(math.sqrt(0.5))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            numerical_diff_summary([1.0, 2.0], [1.0])

    def test_empty_returns_empty_dict(self):
        assert numerical_diff_summary([], []) == {}
