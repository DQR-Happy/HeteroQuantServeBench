"""Statistics for quantization experiments: distributions, outliers, bootstrap CI.

The S05 protocol demands more than point estimates (details README §7/§9.4,
E05-10 §6/§9):

* percentile summaries and outlier rates per activation/weight channel;
* *paired* comparisons against the FP16 baseline (each evaluation item is
  compared with itself, never two independent samples);
* bootstrap confidence intervals, including cluster bootstrap (one document
  produces several windows) and seed-level bootstrap (several calibration
  subsets);
* effect sizes, so "smaller than the noise" cannot be reported as a win.

Everything is pure Python (no numpy) so the CPU-minimal installation can run
the full analysis; the sample counts here are experiment-sized (thousands),
not tensor-sized.

All functions take *raw samples* and return machine-readable summaries; no
function invents a value for a missing sample. Missing metric values are
represented explicitly (``None``/``NaN``) and reduce the effective sample
count instead of being imputed with zero.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.benchmark.metrics import percentile

#: Direction of a metric relative to the baseline (E05-10 §6.1).
HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"
DIRECTIONS = (HIGHER_IS_BETTER, LOWER_IS_BETTER)

#: Gate verdicts (E05-10 §6.3). INCONCLUSIVE is never silently promoted.
GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_INCONCLUSIVE = "INCONCLUSIVE"

DEFAULT_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95


def _clean(values: Iterable[Optional[float]]) -> List[float]:
    """Drop ``None``/NaN entries, keeping the effective sample count honest."""
    out: List[float] = []
    for value in values:
        if value is None:
            continue
        as_float = float(value)
        if math.isnan(as_float):
            continue
        out.append(as_float)
    return out


@dataclass
class DistributionSummary:
    """Percentile/outlier summary of one metric series (E05-03 §7.1)."""

    count: int
    mean: float
    std: float
    minimum: float
    maximum: float
    absmax: float
    p50: float
    p90: float
    p99: float
    p999: float
    outlier_threshold: float
    outlier_count: int
    outlier_rate: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "absmax": self.absmax,
            "p50": self.p50,
            "p90": self.p90,
            "p99": self.p99,
            "p999": self.p999,
            "outlier_threshold": self.outlier_threshold,
            "outlier_count": self.outlier_count,
            "outlier_rate": self.outlier_rate,
        }


def summarize_distribution(
    values: Iterable[Optional[float]],
    *,
    outlier_threshold: Optional[float] = None,
    outlier_sigma: float = 6.0,
) -> DistributionSummary:
    """Summarize a series with percentiles and an outlier rate.

    The outlier threshold defaults to ``mean + outlier_sigma * std``; when the
    series is empty every statistic is ``NaN`` with ``count == 0`` (never a
    fabricated zero).
    """
    clean = _clean(values)
    if not clean:
        nan = float("nan")
        return DistributionSummary(
            count=0,
            mean=nan,
            std=nan,
            minimum=nan,
            maximum=nan,
            absmax=nan,
            p50=nan,
            p90=nan,
            p99=nan,
            p999=nan,
            outlier_threshold=nan,
            outlier_count=0,
            outlier_rate=nan,
        )
    mean = statistics.fmean(clean)
    std = statistics.pstdev(clean)
    threshold = (
        float(outlier_threshold)
        if outlier_threshold is not None
        else mean + outlier_sigma * std
    )
    outliers = [value for value in clean if abs(value) > threshold]
    return DistributionSummary(
        count=len(clean),
        mean=mean,
        std=std,
        minimum=min(clean),
        maximum=max(clean),
        absmax=max(abs(value) for value in clean),
        p50=percentile(clean, 0.50),
        p90=percentile(clean, 0.90),
        p99=percentile(clean, 0.99),
        p999=percentile(clean, 0.999),
        outlier_threshold=threshold,
        outlier_count=len(outliers),
        outlier_rate=len(outliers) / len(clean),
    )


@dataclass
class Interval:
    """A bootstrap confidence interval for one statistic."""

    statistic: str
    point: float
    low: float
    high: float
    confidence: float
    resamples: int
    n_units: int
    method: str = "percentile_bootstrap"

    @property
    def crosses_zero(self) -> bool:
        return self.low <= 0.0 <= self.high

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "statistic": self.statistic,
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
            "resamples": self.resamples,
            "n_units": self.n_units,
            "method": self.method,
            "crosses_zero": self.crosses_zero,
            "half_width": self.half_width,
        }


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: str = "mean",
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI of a statistic over independent samples."""
    clean = _clean(values)
    if not 0.0 < confidence < 1.0:
        raise ConfigError(f"confidence must be in (0, 1), got {confidence}")
    if resamples < 100:
        raise ConfigError(f"resamples must be >= 100, got {resamples}")
    if not clean:
        nan = float("nan")
        return Interval(
            statistic=f"{statistic}(empty)",
            point=nan,
            low=nan,
            high=nan,
            confidence=confidence,
            resamples=resamples,
            n_units=0,
        )
    point = _statistic(clean, statistic)
    rng = random.Random(seed)
    n = len(clean)
    draws: List[float] = []
    for _ in range(resamples):
        sample = [clean[rng.randrange(n)] for _ in range(n)]
        draws.append(_statistic(sample, statistic))
    draws.sort()
    alpha = (1.0 - confidence) / 2.0
    low = percentile(draws, alpha)
    high = percentile(draws, 1.0 - alpha)
    return Interval(
        statistic=statistic,
        point=point,
        low=low,
        high=high,
        confidence=confidence,
        resamples=resamples,
        n_units=n,
    )


def paired_bootstrap_ci(
    paired_deltas: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """Bootstrap CI of the mean of *paired* deltas (E05-10 §6.2).

    The caller forms ``delta_i = metric_candidate(i) - metric_baseline(i)`` on
    the same item ``i``. Resampling the deltas (not the two series separately)
    is what preserves the pairing and keeps the CI honest for small n.
    """
    clean = _clean(paired_deltas)
    interval = bootstrap_ci(
        clean,
        statistic="mean",
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    interval.statistic = "paired_mean_delta"
    return interval


def cluster_bootstrap_ci(
    clusters: Sequence[Sequence[float]],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """Cluster bootstrap: resample whole clusters (documents, seeds).

    Used when one document produces several windows or one calibration subset
    produces several artifacts (E05-10 §6.2): treating those samples as
    independent would understate the variance.
    """
    usable = [[float(value) for value in cluster if not math.isnan(float(value))] for cluster in clusters]
    usable = [cluster for cluster in usable if cluster]
    if not usable:
        nan = float("nan")
        return Interval(
            statistic="cluster_mean(empty)",
            point=nan,
            low=nan,
            high=nan,
            confidence=confidence,
            resamples=resamples,
            n_units=0,
            method="cluster_bootstrap",
        )
    cluster_means = [statistics.fmean(cluster) for cluster in usable]
    point = statistics.fmean(cluster_means) if len(cluster_means) > 1 else cluster_means[0]
    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(resamples):
        sample = [cluster_means[rng.randrange(len(cluster_means))] for _ in cluster_means]
        draws.append(statistics.fmean(sample))
    draws.sort()
    alpha = (1.0 - confidence) / 2.0
    return Interval(
        statistic="cluster_mean",
        point=point,
        low=percentile(draws, alpha),
        high=percentile(draws, 1.0 - alpha),
        confidence=confidence,
        resamples=resamples,
        n_units=len(cluster_means),
        method="cluster_bootstrap",
    )


def _statistic(sample: Sequence[float], name: str) -> float:
    if name in ("mean", "paired_mean_delta"):
        return statistics.fmean(sample)
    if name == "median":
        return statistics.median(sample)
    if name == "p95":
        return percentile(sample, 0.95)
    if name == "sum":
        return math.fsum(sample)
    raise ConfigError(f"unsupported bootstrap statistic {name!r}")


def cohens_d_paired(deltas: Sequence[float]) -> Optional[float]:
    """Paired effect size ``mean(delta) / std(delta)``.

    ``None`` when the deltas carry no variance (a single item or identical
    values), because an effect size is undefined there — reporting ``inf``
    would be a claim, not a measurement.
    """
    clean = _clean(deltas)
    if len(clean) < 2:
        return None
    std = statistics.stdev(clean)
    if std == 0.0:
        return None
    return statistics.fmean(clean) / std


def relative_change(candidate: float, baseline: float) -> float:
    """``(candidate - baseline) / |baseline|`` with a stable zero baseline."""
    if baseline == 0.0:
        return float("nan")
    return (candidate - baseline) / abs(baseline)


def _aligned_direction(delta: float, direction: str) -> float:
    """Map a metric delta to "improvement is positive"."""
    if direction == HIGHER_IS_BETTER:
        return delta
    if direction == LOWER_IS_BETTER:
        return -delta
    raise ConfigError(f"unknown metric direction {direction!r}")


@dataclass
class NonInferiorityVerdict:
    """Outcome of a pre-registered non-inferiority check (E05-10 §6.3)."""

    metric: str
    direction: str
    margin: float
    baseline: float
    candidate: float
    delta: float
    interval: Interval
    verdict: str
    reason: str

    def as_dict(self) -> Dict[str, object]:
        payload = self.interval.as_dict()
        payload.update(
            {
                "metric": self.metric,
                "direction": self.direction,
                "margin": self.margin,
                "baseline": self.baseline,
                "candidate": self.candidate,
                "delta": self.delta,
                "verdict": self.verdict,
                "reason": self.reason,
            }
        )
        return payload


def non_inferiority_verdict(
    metric: str,
    direction: str,
    baseline: Sequence[float],
    candidate: Sequence[float],
    margin: float,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> NonInferiorityVerdict:
    """Paired non-inferiority test with a confidence-interval rule.

    Pre-registered rule (details README §9.4): the candidate passes when the
    *improvement-oriented* lower confidence bound stays within the allowed
    budget, i.e. ``lower_bound(improvement) >= -margin``. If the interval
    crosses the boundary the verdict is ``INCONCLUSIVE`` — never promoted to
    PASS (E05-10 §21 "把INCONCLUSIVE当PASS" is forbidden).
    """
    if direction not in DIRECTIONS:
        raise ConfigError(f"unknown metric direction {direction!r}")
    if margin < 0:
        raise ConfigError(f"margin must be non-negative, got {margin}")
    if len(baseline) != len(candidate):
        raise ConfigError(
            f"non-inferiority requires paired samples, got "
            f"{len(baseline)} vs {len(candidate)}"
        )
    deltas = []
    for left, right in zip(baseline, candidate):
        if left is None or right is None:
            continue
        left_float, right_float = float(left), float(right)
        if math.isnan(left_float) or math.isnan(right_float):
            continue
        deltas.append(right_float - left_float)
    if not deltas:
        nan = float("nan")
        empty = Interval("paired_mean_delta", nan, nan, nan, confidence, resamples, 0)
        return NonInferiorityVerdict(
            metric=metric,
            direction=direction,
            margin=margin,
            baseline=nan,
            candidate=nan,
            delta=nan,
            interval=empty,
            verdict=GATE_INCONCLUSIVE,
            reason="no paired samples with finite values",
        )
    oriented = [_aligned_direction(delta, direction) for delta in deltas]
    interval = paired_bootstrap_ci(
        oriented, confidence=confidence, resamples=resamples, seed=seed
    )
    baseline_mean = statistics.fmean(_clean(baseline))
    candidate_mean = statistics.fmean(_clean(candidate))
    raw_delta = candidate_mean - baseline_mean
    if interval.low >= -margin:
        verdict = GATE_PASS
        reason = (
            f"lower {confidence:.0%} bound {interval.low:.6g} >= -margin "
            f"{-margin:.6g}"
        )
    elif interval.high < -margin:
        verdict = GATE_FAIL
        reason = (
            f"upper {confidence:.0%} bound {interval.high:.6g} < -margin "
            f"{-margin:.6g}"
        )
    else:
        verdict = GATE_INCONCLUSIVE
        reason = (
            f"interval [{interval.low:.6g}, {interval.high:.6g}] crosses the "
            f"boundary -{margin:.6g}; more samples are required"
        )
    return NonInferiorityVerdict(
        metric=metric,
        direction=direction,
        margin=margin,
        baseline=baseline_mean,
        candidate=candidate_mean,
        delta=raw_delta,
        interval=interval,
        verdict=verdict,
        reason=reason,
    )


def stable_subset_agreement(
    left: Sequence[float], right: Sequence[float], *, tolerance: float
) -> Dict[str, float]:
    """Compare two statistic vectors (e.g. scale vectors across subsets).

    Returns cosine similarity, relative L2 difference, max relative change and
    the fraction of elements within ``tolerance`` — the convergence metrics
    E05-03 §8.1 requires for "statistics have stabilized".
    """
    if len(left) != len(right):
        raise ConfigError(
            f"vectors must have equal length, got {len(left)} vs {len(right)}"
        )
    if not left:
        return {
            "cosine": float("nan"),
            "relative_l2": float("nan"),
            "max_relative_change": float("nan"),
            "within_tolerance_fraction": float("nan"),
            "count": 0,
        }
    dot = math.fsum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(math.fsum(value * value for value in left))
    norm_right = math.sqrt(math.fsum(value * value for value in right))
    cosine = dot / (norm_left * norm_right) if norm_left and norm_right else float("nan")
    diff = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right)))
    relative_l2 = diff / norm_left if norm_left else float("nan")
    max_rel = 0.0
    within = 0
    for a, b in zip(left, right):
        denom = abs(a) if a != 0 else 1.0
        rel = abs(b - a) / denom
        max_rel = max(max_rel, rel)
        if rel <= tolerance:
            within += 1
    return {
        "cosine": cosine,
        "relative_l2": relative_l2,
        "max_relative_change": max_rel,
        "within_tolerance_fraction": within / len(left),
        "count": len(left),
    }


def jaccard(left: Iterable, right: Iterable) -> float:
    """Jaccard similarity of two sets (e.g. salient channel selections)."""
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return float("nan")
    return len(left_set & right_set) / len(union)


@dataclass
class HistogramSketch:
    """Fixed-bin histogram with recorded bin edges (mergeable, versioned)."""

    bin_edges: Tuple[float, float, int]
    counts: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        low, high, bins = self.bin_edges
        if not low < high:
            raise ConfigError(f"invalid histogram range [{low}, {high})")
        if bins <= 0:
            raise ConfigError(f"bins must be positive, got {bins}")
        if not self.counts:
            self.counts = [0] * bins
        if len(self.counts) != bins:
            raise ConfigError(
                f"counts length {len(self.counts)} != declared bins {bins}"
            )

    def add(self, value: float) -> None:
        low, high, bins = self.bin_edges
        if value < low or value >= high:
            raise ConfigError(
                f"value {value} outside histogram range [{low}, {high}); "
                f"out-of-range samples must be counted explicitly, not dropped"
            )
        index = int((value - low) / (high - low) * self.bin_edges[2])
        self.counts[min(index, bins - 1)] += 1

    def merge(self, other: "HistogramSketch") -> "HistogramSketch":
        if self.bin_edges != other.bin_edges:
            raise ConfigError(
                "cannot merge histograms with different bin edges "
                f"({self.bin_edges} vs {other.bin_edges})"
            )
        return HistogramSketch(
            bin_edges=self.bin_edges,
            counts=[a + b for a, b in zip(self.counts, other.counts)],
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "bin_edges": list(self.bin_edges),
            "counts": list(self.counts),
            "total": sum(self.counts),
        }


def aggregate_paired(
    rows: Sequence[Mapping[str, object]],
    *,
    metric_key: str,
    baseline_key: str,
    group_key: Optional[str] = None,
) -> Dict[str, object]:
    """Aggregate raw paired rows into overall and per-group deltas with CIs.

    ``rows`` are raw per-item records (e.g. per-position logit metrics) with
    ``metric_key`` for the candidate and ``baseline_key`` for FP16. Grouping
    is reported separately for every value of ``group_key`` (slice/length
    bucket) — a failing slice must not be averageable into a passing overall
    number (E05-10 §6.3).
    """
    deltas: List[float] = []
    groups: Dict[str, List[float]] = {}
    for row in rows:
        if metric_key not in row or baseline_key not in row:
            continue
        candidate = row[metric_key]
        baseline = row[baseline_key]
        if candidate is None or baseline is None:
            continue
        delta = float(candidate) - float(baseline)
        deltas.append(delta)
        if group_key and group_key in row:
            groups.setdefault(str(row[group_key]), []).append(delta)
    summary: Dict[str, object] = {
        "overall": paired_bootstrap_ci(deltas).as_dict(),
        "effect_size": cohens_d_paired(deltas),
        "n_items": len(deltas),
        "groups": {
            name: paired_bootstrap_ci(values).as_dict()
            for name, values in sorted(groups.items())
        },
    }
    return summary


__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_RESAMPLES",
    "DIRECTIONS",
    "DistributionSummary",
    "GATE_FAIL",
    "GATE_INCONCLUSIVE",
    "GATE_PASS",
    "HIGHER_IS_BETTER",
    "HistogramSketch",
    "Interval",
    "LOWER_IS_BETTER",
    "NonInferiorityVerdict",
    "aggregate_paired",
    "bootstrap_ci",
    "cluster_bootstrap_ci",
    "cohens_d_paired",
    "jaccard",
    "non_inferiority_verdict",
    "paired_bootstrap_ci",
    "relative_change",
    "stable_subset_agreement",
    "summarize_distribution",
]
