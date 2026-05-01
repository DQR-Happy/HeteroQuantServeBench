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
