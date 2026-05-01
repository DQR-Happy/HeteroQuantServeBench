"""Property-style tests for percentile/latency summary invariants.

These tests do not require the Hypothesis library; they assert invariants
that must hold for *any* input, using randomized and adversarial inputs.
"""

from __future__ import annotations

import random

import pytest

from hqsb.benchmark.metrics import latency_summary, percentile


def _random_values(rng, n):
    return [rng.uniform(0.0, 1000.0) for _ in range(n)]


@pytest.mark.property
class TestPercentileInvariants:
    def test_percentile_is_monotonic_in_quantile(self):
        rng = random.Random(0)
        for _ in range(50):
            values = _random_values(rng, rng.randint(1, 50))
            lo = percentile(values, 0.25)
            hi = percentile(values, 0.75)
            assert lo <= hi

    def test_percentile_within_data_range(self):
        rng = random.Random(1)
        for _ in range(50):
            values = _random_values(rng, rng.randint(1, 50))
            p = percentile(values, rng.random())
            assert min(values) <= p <= max(values)

    def test_min_and_max_quantiles(self):
        rng = random.Random(2)
        for _ in range(20):
            values = _random_values(rng, 10)
            assert percentile(values, 0.0) == min(values)
            assert percentile(values, 1.0) == max(values)

    def test_duplicate_values(self):
        assert percentile([5.0] * 10, 0.5) == 5.0


@pytest.mark.property
class TestLatencySummaryInvariants:
    def test_summary_fields_consistent(self):
        rng = random.Random(3)
        for _ in range(50):
            values = _random_values(rng, rng.randint(2, 50))
            summary = latency_summary(values)
            assert summary["min_ms"] == min(values)
            assert summary["max_ms"] == max(values)
            assert summary["count"] == len(values)
            assert summary["min_ms"] <= summary["mean_ms"] <= summary["max_ms"]
            assert summary["p95_ms"] <= summary["max_ms"]
            assert summary["p50_ms"] >= summary["min_ms"]
