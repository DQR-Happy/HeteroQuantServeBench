"""Statistical metrics for benchmark analysis.

Provides latency distribution summaries (percentiles, mean, stddev)
and utility functions for numerical comparison between baseline
and optimized model runs.
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Sequence, Union


def percentile(values: Sequence[float], quantile: float) -> float:
    """Compute a percentile using linear interpolation.

    Uses the same method as NumPy's ``np.percentile`` with
    ``method="linear"``: for a sorted array of length N, the
    position is ``(N - 1) * quantile``, and the result is
    linearly interpolated between adjacent elements.

    Args:
        values: Sequence of numeric values.
        quantile: Percentile in [0.0, 1.0] (e.g., 0.50 for median,
            0.95 for P95).

    Returns:
        The interpolated percentile value. Returns ``float("nan")``
        if the input is empty.

    Raises:
        ValueError: If ``quantile`` is outside [0.0, 1.0].
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0.0, 1.0], got {quantile}")

    if not values:
        return float("nan")

    ordered = sorted(values)
    n = len(ordered)

    if n == 1:
        return ordered[0]

    position = (n - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(values_ms: Sequence[float]) -> Dict[str, float]:
    """Compute comprehensive latency statistics.

    Produces a dictionary with count, mean, median, stddev (population),
    min, max, and key percentiles (P50, P95, P99).

    Args:
        values_ms: Sequence of latency measurements in milliseconds.

    Returns:
        Dictionary with keys:
            - ``count``: Number of samples.
            - ``mean_ms``: Arithmetic mean.
            - ``median_ms``: Median (50th percentile).
            - ``stddev_ms``: Population standard deviation.
            - ``min_ms``: Minimum value.
            - ``max_ms``: Maximum value.
            - ``p50_ms``: 50th percentile (same as median).
            - ``p95_ms``: 95th percentile.
            - ``p99_ms``: 99th percentile.
        Returns an empty dict if ``values_ms`` is empty.
    """
    if not values_ms:
        return {}

    return {
        "count": len(values_ms),
        "mean_ms": statistics.mean(values_ms),
        "median_ms": statistics.median(values_ms),
        "stddev_ms": statistics.pstdev(values_ms),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
        "p50_ms": percentile(values_ms, 0.50),
        "p95_ms": percentile(values_ms, 0.95),
        "p99_ms": percentile(values_ms, 0.99),
    }


def numerical_diff_summary(
    baseline: Sequence[float],
    optimized: Sequence[float],
) -> Dict[str, float]:
    """Compute numerical error metrics between baseline and optimized outputs.

    Used for comparing logits, hidden states, or token probabilities
    when replacing PyTorch operators with custom CUDA/Triton kernels.

    Args:
        baseline: Reference values (e.g., FP32 PyTorch output).
        optimized: Values from the optimized implementation.

    Returns:
        Dictionary with:
            - ``max_abs_error``: Maximum absolute error.
            - ``mean_abs_error``: Mean absolute error.
            - ``rmse``: Root mean square error.
            - ``cosine_similarity``: Cosine similarity (1.0 = identical).
            - ``l2_relative_error``: Relative L2 error norm.

    Raises:
        ValueError: If input sequences have different lengths.
    """
    if len(baseline) != len(optimized):
        raise ValueError(
            f"Sequence length mismatch: {len(baseline)} vs {len(optimized)}"
        )

    n = len(baseline)
    if n == 0:
        return {}

    abs_errors = [abs(b - o) for b, o in zip(baseline, optimized)]
    max_abs = max(abs_errors)
    mean_abs = statistics.mean(abs_errors)

    mse = sum(e**2 for e in abs_errors) / n
    rmse = math.sqrt(mse)

    # Cosine similarity
    dot = sum(b * o for b, o in zip(baseline, optimized))
    norm_b = math.sqrt(sum(b**2 for b in baseline))
    norm_o = math.sqrt(sum(o**2 for o in optimized))
    cos_sim = dot / (norm_b * norm_o) if norm_b > 0 and norm_o > 0 else 0.0

    # Relative L2 error
    l2_diff = math.sqrt(sum((b - o) ** 2 for b, o in zip(baseline, optimized)))
    l2_ref = math.sqrt(sum(b**2 for b in baseline))
    l2_rel = l2_diff / l2_ref if l2_ref > 0 else float("inf")

    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rmse": rmse,
        "cosine_similarity": cos_sim,
        "l2_relative_error": l2_rel,
    }
