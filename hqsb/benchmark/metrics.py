"""Statistical metrics for benchmark analysis.

Provides latency distribution summaries (percentiles, mean, stddev)
and utility functions for numerical comparison between baseline
and optimized model runs.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


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


def model_core_timings(
    prefill_forward_ms: float,
    first_token_selection_ms: float,
    itl_ms: Sequence[float],
) -> Dict[str, float]:
    """Single source of truth for model-core timing derivation (E02-01).

    The reference clock convention defines exactly three derived quantities
    from the three raw phase measurements:

        decode_total_ms      = sum(itl_ms)
        model_core_ttft_ms   = prefill_forward_ms + first_token_selection_ms
        model_core_e2e_ms    = model_core_ttft_ms + decode_total_ms

    This is the **only** implementation of that relationship. Both the
    model-core engine and the benchmark engine must derive these values
    through this function so the convention can never drift (S02 execution
    step 3, E02-01 pass criterion "公式只有一个实现").
    """
    decode_total_ms = float(sum(itl_ms))
    ttft_ms = float(prefill_forward_ms) + float(first_token_selection_ms)
    e2e_ms = ttft_ms + decode_total_ms
    return {
        "decode_total_ms": decode_total_ms,
        "model_core_ttft_ms": ttft_ms,
        "model_core_e2e_ms": e2e_ms,
    }


def _scalar_summary(values: Sequence[float]) -> Dict[str, float]:
    """Aggregate a scalar series into ``{count, mean, p50, p95, min, max}``.

    ``NaN`` entries are dropped before aggregation (they represent undefined
    per-request metrics, e.g. TPOT for ``G == 1``), so ``count`` reports the
    number of *valid* samples. An empty series yields ``NaN`` statistics with
    ``count == 0``.
    """
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {
            "count": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": len(clean),
        "mean": statistics.mean(clean),
        "p50": percentile(clean, 0.50),
        "p95": percentile(clean, 0.95),
        "min": min(clean),
        "max": max(clean),
    }


def request_summary(requests: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-request scaling metrics into a P50/P95 summary.

    E02-03 requires request-level TTFT/TPOT/E2E and the three throughputs to
    be reported *separately from* step-level ITL. Each request mapping must
    provide the raw E02-01 clock fields: ``prefill_forward_ms``,
    ``first_token_selection_ms``, ``raw_itl_ms`` (list of per-step ms),
    ``input_tokens``, and ``output_tokens``.

    The per-request derivations follow the single clock convention in
    :func:`model_core_timings` and the E02-03 §5 metric definitions:

        TTFT    = prefill_forward + selection
        Tdecode = sum(ITL)
        TPOT    = Tdecode / (G - 1),           G > 1
        Tgen    = TTFT + Tdecode
        prefill TPS   = I / prefill_forward
        decode-tail   = (G - 1) / Tdecode
        output TPS    = G / Tgen

    Returns:
        Dict with ``ttft_ms``, ``tpot_ms``, ``e2e_ms``,
        ``prefill_tokens_per_s``, ``decode_tokens_per_s``,
        ``output_tokens_per_s`` (each a :func:`_scalar_summary`), and
        ``pooled_itl_ms`` (step-level :func:`latency_summary` across every
        step of every request).
    """
    ttft: List[float] = []
    tpot: List[float] = []
    e2e: List[float] = []
    prefill_tps: List[float] = []
    decode_tps: List[float] = []
    output_tps: List[float] = []
    all_itl: List[float] = []

    for req in requests:
        timings = model_core_timings(
            float(req["prefill_forward_ms"]),
            float(req["first_token_selection_ms"]),
            [float(x) for x in req["raw_itl_ms"]],
        )
        decode_total_ms = timings["decode_total_ms"]
        g = int(req["output_tokens"])
        i = int(req["input_tokens"])
        prefill_ms = float(req["prefill_forward_ms"])
        e2e_ms = timings["model_core_e2e_ms"]

        ttft.append(timings["model_core_ttft_ms"])
        e2e.append(e2e_ms)
        tpot.append(decode_total_ms / (g - 1) if g > 1 else float("nan"))
        prefill_tps.append(i / (prefill_ms / 1000.0) if prefill_ms > 0 else 0.0)
        decode_tps.append(
            (g - 1) / (decode_total_ms / 1000.0)
            if decode_total_ms > 0 and g > 1
            else 0.0
        )
        output_tps.append(g / (e2e_ms / 1000.0) if e2e_ms > 0 else 0.0)
        all_itl.extend([float(x) for x in req["raw_itl_ms"]])

    return {
        "ttft_ms": _scalar_summary(ttft),
        "tpot_ms": _scalar_summary(tpot),
        "e2e_ms": _scalar_summary(e2e),
        "prefill_tokens_per_s": _scalar_summary(prefill_tps),
        "decode_tokens_per_s": _scalar_summary(decode_tps),
        "output_tokens_per_s": _scalar_summary(output_tps),
        "pooled_itl_ms": latency_summary(all_itl),
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
