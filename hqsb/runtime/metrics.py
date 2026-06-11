"""Runtime model-core time and token metrics (S07 details README §8, §13.5).

Two properties of this module matter for the whole stage:

1. **Timestamps are data, not decoration.**  :class:`RequestTimeline` refuses a
   timeline whose stamps are out of order, so a "TTFT" can never be computed
   from a negative interval.  Every derived metric is a named property so the
   report can quote the exact formula.
2. **Three token denominators coexist and are never collapsed into "TPS"**
   (details README §13.5):

   ``logical_input_tokens``
       what the user submitted;
   ``useful_committed_tokens``
       logical input + finally committed output;
   ``model_computed_positions``
       positions the model actually computed (cache misses, decode, speculative
       verification, preemption recompute) — cancelled/timeout/preempted work
       stays in this denominator, it is never deleted.

Statistics follow §13.6: the **independent runtime process / run** is the unit
of repetition (thousands of tokens inside one run are not thousands of
samples), latency uses paired request differences, and the bootstrap is
deterministic (seeded) so a report can be recomputed bit-for-bit.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

# ── time ledger ────────────────────────────────────────────────────────────

#: The model-core boundary timestamps (details README §8).  ``t_first_token_ready``
#: and ``t_final_token_ready`` are the only ones strictly required; the rest are
#: optional so a cancelled request can still be described.
TIMELINE_FIELDS: Tuple[str, ...] = (
    "t_submit",
    "t_validate_done",
    "t_admit",
    "t_prefill_start",
    "t_prefill_end",
    "t_first_token_ready",
    "t_final_token_ready",
    "t_cancel_requested",
    "t_cancel_observed",
    "t_cleanup_done",
)


@dataclass(frozen=True)
class RequestTimeline:
    """Per-request host timestamps in seconds (monotonic clock)."""

    t_submit: float
    t_validate_done: Optional[float] = None
    t_admit: Optional[float] = None
    t_prefill_start: Optional[float] = None
    t_prefill_end: Optional[float] = None
    t_first_token_ready: Optional[float] = None
    t_final_token_ready: Optional[float] = None
    t_cancel_requested: Optional[float] = None
    t_cancel_observed: Optional[float] = None
    t_cleanup_done: Optional[float] = None
    output_tokens: int = 0
    token_ready_times: Tuple[float, ...] = ()
    logical_input_tokens: int = 0
    cached_prefix_tokens: int = 0
    speculative_verified_positions: int = 0
    preemption_recompute_positions: int = 0
    cancelled: bool = False
    timed_out: bool = False

    def __post_init__(self) -> None:
        stamps = self.ordered_stamps()
        for (left_name, left), (right_name, right) in zip(stamps, stamps[1:]):
            if right < left:
                raise ConfigError(
                    f"timeline is not monotonic: {right_name}={right} < "
                    f"{left_name}={left}; a metric computed from this ordering "
                    "would be meaningless",
                    details={"field": right_name},
                )
        if self.output_tokens < 0:
            raise ConfigError("output_tokens must be non-negative")
        if self.token_ready_times and len(self.token_ready_times) != self.output_tokens:
            raise ConfigError(
                "token_ready_times must have exactly one entry per output token "
                f"({len(self.token_ready_times)} != {self.output_tokens})",
                details={"field": "token_ready_times"},
            )
        if self.t_first_token_ready is None and self.output_tokens:
            raise ConfigError(
                "a request with output tokens must record t_first_token_ready; "
                "otherwise TTFT/ITL/TPOT would be undefined",
                details={"field": "t_first_token_ready"},
            )

    # ── helpers ───────────────────────────────────────────────────────────

    def ordered_stamps(self) -> List[Tuple[str, float]]:
        """Present (name, value) pairs sorted by the declared field order."""
        return [
            (name, getattr(self, name))
            for name in TIMELINE_FIELDS
            if getattr(self, name) is not None
        ]

    def stamp(self, name: str) -> float:
        value = getattr(self, name)
        if value is None:
            raise ConfigError(
                f"timeline has no {name!r}; the metric that needs it is undefined "
                "for this request (report it as NA, do not substitute a default)",
                details={"field": name},
            )
        return float(value)

    # ── metrics (details README §8) ───────────────────────────────────────

    @property
    def queue_delay(self) -> float:
        """``t_prefill_start - t_submit`` (scheduling + KV reservation wait)."""
        return self.stamp("t_prefill_start") - self.stamp("t_submit")

    @property
    def runtime_ttft(self) -> float:
        """``t_first_token_ready - t_submit``."""
        return self.stamp("t_first_token_ready") - self.stamp("t_submit")

    @property
    def prefill_service(self) -> float:
        """``t_first_token_ready - t_prefill_start``."""
        return self.stamp("t_first_token_ready") - self.stamp("t_prefill_start")

    @property
    def e2e_core(self) -> float:
        """``t_final_token_ready - t_submit`` (model-core, no HTTP)."""
        return self.stamp("t_final_token_ready") - self.stamp("t_submit")

    @property
    def tpot(self) -> float:
        """``(t_final - t_first) / max(output_tokens - 1, 1)``."""
        span = self.stamp("t_final_token_ready") - self.stamp("t_first_token_ready")
        return span / max(self.output_tokens - 1, 1)

    @property
    def itl_ms(self) -> List[float]:
        """Inter-token latencies in milliseconds, in emission order."""
        times = self.token_ready_times
        if len(times) < 2:
            return []
        return [(right - left) * 1000.0 for left, right in zip(times, times[1:])]

    @property
    def cancel_observation_latency(self) -> Optional[float]:
        if self.t_cancel_requested is None or self.t_cancel_observed is None:
            return None
        return self.t_cancel_observed - self.t_cancel_requested

    @property
    def cleanup_time(self) -> Optional[float]:
        """Cleanup time measured from the event that ended the request.

        A cancelled request ends when the cancel is observed; a completed one
        ends at its final token.  Anchoring both on the final token would report a
        negative "cleanup" for a cancel that happened earlier in the run.
        """
        if self.t_cleanup_done is None:
            return None
        if self.cancelled and self.t_cancel_observed is not None:
            anchor = self.t_cancel_observed
        elif self.t_final_token_ready is not None:
            anchor = self.t_final_token_ready
        else:
            anchor = self.t_cancel_observed or self.t_cancel_requested
        if anchor is None:
            return None
        return self.t_cleanup_done - anchor

    def token_ledger(self) -> "TokenLedger":
        """Build the three-denominator ledger for this request.

        ``logical_input_tokens`` is an explicit field: leaving it implicit would
        let a caller forget the submitted prompt and silently shrink the
        denominator that §13.5 protects.
        """
        return TokenLedger(
            logical_input_tokens=self.logical_input_tokens,
            committed_output_tokens=self.output_tokens,
            cached_prefix_tokens=self.cached_prefix_tokens,
            speculative_verified_positions=self.speculative_verified_positions,
            preemption_recompute_positions=self.preemption_recompute_positions,
        )


@dataclass(frozen=True)
class TokenLedger:
    """The three S07 token denominators (details README §13.5).

    ``model_computed_positions`` deliberately includes speculative
    verification, preemption recompute and cancelled work: those positions cost
    the model time, so removing them from the denominator would inflate
    throughput.
    """

    logical_input_tokens: int
    committed_output_tokens: int
    cached_prefix_tokens: int = 0
    speculative_verified_positions: int = 0
    preemption_recompute_positions: int = 0
    draft_tokens_proposed: int = 0
    cancelled_or_failed_positions: int = 0

    def __post_init__(self) -> None:
        for name in (
            "logical_input_tokens",
            "committed_output_tokens",
            "cached_prefix_tokens",
            "speculative_verified_positions",
            "preemption_recompute_positions",
            "draft_tokens_proposed",
            "cancelled_or_failed_positions",
        ):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must be non-negative")
        if self.cached_prefix_tokens > self.logical_input_tokens:
            raise ConfigError(
                "cached_prefix_tokens cannot exceed logical_input_tokens; a cache "
                "hit reuses submitted tokens, it does not create them",
                details={"field": "cached_prefix_tokens"},
            )

    @property
    def useful_committed_tokens(self) -> int:
        """logical input + finally committed output (details README §13.5)."""
        return self.logical_input_tokens + self.committed_output_tokens

    @property
    def model_computed_positions(self) -> int:
        """Positions the model really computed in this run."""
        freshly_prefilled = self.logical_input_tokens - self.cached_prefix_tokens
        return (
            freshly_prefilled
            + self.committed_output_tokens
            + self.speculative_verified_positions
            + self.preemption_recompute_positions
            + self.cancelled_or_failed_positions
        )

    def output_tps(self, makespan_s: float) -> float:
        return _rate(self.committed_output_tokens, makespan_s)

    def logical_token_tps(self, makespan_s: float) -> float:
        return _rate(self.useful_committed_tokens, makespan_s)

    def compute_position_tps(self, model_time_s: float) -> float:
        return _rate(self.model_computed_positions, model_time_s)

    def audit(self) -> "TokenConservationReport":
        """Conservation audit required before any of the rates is published."""
        problems: List[str] = []
        if self.committed_output_tokens == 0 and self.logical_input_tokens == 0:
            problems.append("empty ledger: no input and no output token")
        if self.cached_prefix_tokens > self.logical_input_tokens:
            problems.append("cached prefix larger than the submitted prompt")
        if self.speculative_verified_positions < 0:
            problems.append("negative speculative verification count")
        if self.draft_tokens_proposed and (
            self.speculative_verified_positions > self.draft_tokens_proposed
        ):
            problems.append(
                "accepted speculative tokens exceed proposed draft tokens"
            )
        return TokenConservationReport(problems=tuple(problems), ledger=self)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "logical_input_tokens": self.logical_input_tokens,
            "useful_committed_tokens": self.useful_committed_tokens,
            "committed_output_tokens": self.committed_output_tokens,
            "model_computed_positions": self.model_computed_positions,
            "cached_prefix_tokens": self.cached_prefix_tokens,
            "speculative_verified_positions": self.speculative_verified_positions,
            "preemption_recompute_positions": self.preemption_recompute_positions,
            "draft_tokens_proposed": self.draft_tokens_proposed,
            "cancelled_or_failed_positions": self.cancelled_or_failed_positions,
        }


@dataclass(frozen=True)
class TokenConservationReport:
    """Result of :meth:`TokenLedger.audit`."""

    problems: Tuple[str, ...]
    ledger: TokenLedger

    @property
    def ok(self) -> bool:
        return not self.problems

    def require_ok(self) -> None:
        if not self.ok:
            raise ConfigError(
                "token conservation audit failed: " + "; ".join(self.problems),
                details={"fields": list(self.problems)},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "problems": list(self.problems),
            "ledger": self.ledger.as_dict(),
        }


def _rate(numerator: int, denominator_s: float) -> float:
    if denominator_s <= 0:
        raise ConfigError(
            "throughput denominator must be positive; a zero or negative window "
            "cannot produce a rate",
            details={"denominator_s": denominator_s},
        )
    return numerator / denominator_s


# ── distributions and paired statistics ────────────────────────────────────


def percentile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolation percentile on a sorted copy (matches §13.6 usage)."""
    if not values:
        raise ConfigError("percentile of an empty sample is undefined")
    if not 0.0 <= quantile <= 1.0:
        raise ConfigError("quantile must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


@dataclass(frozen=True)
class DistributionSummary:
    """Median / P50 / P95 / spread for run-level samples (never per-token)."""

    count: int
    median: float
    p50: float
    p95: float
    minimum: float
    maximum: float
    spread: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "median": self.median,
            "p50": self.p50,
            "p95": self.p95,
            "min": self.minimum,
            "max": self.maximum,
            "spread": self.spread,
        }


def distribution_summary(values: Sequence[float]) -> DistributionSummary:
    if not values:
        raise ConfigError("distribution summary needs at least one sample")
    ordered = sorted(float(value) for value in values)
    return DistributionSummary(
        count=len(ordered),
        median=percentile(ordered, 0.5),
        p50=percentile(ordered, 0.5),
        p95=percentile(ordered, 0.95),
        minimum=ordered[0],
        maximum=ordered[-1],
        spread=ordered[-1] - ordered[0],
    )


@dataclass(frozen=True)
class PairedEffect:
    """Paired difference A/B with a deterministic bootstrap confidence interval."""

    n_pairs: int
    mean_difference: float
    ci_low: float
    ci_high: float
    ci_level: float

    @property
    def crosses_zero(self) -> bool:
        return self.ci_low <= 0.0 <= self.ci_high

    def exceeds_guard_band(self, guard_band: float) -> bool:
        """A benefit claim needs the *whole* CI beyond the guard band (§5.5)."""
        if guard_band < 0:
            raise ConfigError("guard band must be non-negative")
        return self.ci_low > guard_band

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "mean_difference": self.mean_difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "crosses_zero": self.crosses_zero,
        }


def paired_effect(
    left: Sequence[float],
    right: Sequence[float],
    *,
    ci_level: float = 0.95,
    resamples: int = 2000,
    seed: int = 20260918,
) -> PairedEffect:
    """Paired bootstrap CI of ``mean(left - right)`` (details README §13.6).

    ``left`` and ``right`` must be the *same requests* under two conditions; a
    paired analysis is the only one that removes request-to-request variance.
    """
    if len(left) != len(right):
        raise ConfigError(
            "paired statistics need two equal-length series (the same requests "
            f"under two conditions): {len(left)} != {len(right)}",
            details={"field": "pairs"},
        )
    if not left:
        raise ConfigError("paired statistics need at least one pair")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    mean = sum(differences) / len(differences)
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(len(differences)):
            total += differences[rng.randrange(len(differences))]
        means.append(total / len(differences))
    tail = (1.0 - ci_level) / 2.0
    return PairedEffect(
        n_pairs=len(differences),
        mean_difference=mean,
        ci_low=percentile(means, tail),
        ci_high=percentile(means, 1.0 - tail),
        ci_level=ci_level,
    )


@dataclass(frozen=True)
class EquivalenceCheck:
    """Pre-registered equivalence bound (§13.6: CI crossing zero is not proof)."""

    bound: float
    effect: PairedEffect

    def __post_init__(self) -> None:
        if self.bound <= 0:
            raise ConfigError(
                "an equivalence claim needs a pre-registered positive bound; "
                "otherwise 'no difference detected' is silently read as equal",
                details={"field": "bound"},
            )

    @property
    def equivalent(self) -> bool:
        return -self.bound < self.effect.ci_low and self.effect.ci_high < self.bound

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bound": self.bound,
            "equivalent": self.equivalent,
            "effect": self.effect.as_dict(),
        }


# ── run-level aggregation ──────────────────────────────────────────────────


@dataclass
class RunLevelSamples:
    """Samples grouped by independent run (process), then by request class.

    §13.6: the independent runtime process is the repetition unit, so a metrics
    table is built from run-level values (median per run) rather than from
    individual requests.
    """

    metric: str
    per_run: Dict[str, List[float]] = field(default_factory=dict)

    def record(self, run_id: str, values: Iterable[float]) -> None:
        self.per_run.setdefault(run_id, []).extend(float(value) for value in values)

    def run_medians(self) -> List[float]:
        return [percentile(values, 0.5) for values in self.per_run.values()]

    def summary(self) -> DistributionSummary:
        return distribution_summary(self.run_medians())

    def independent_runs(self) -> int:
        return len(self.per_run)


def token_accounting_from_requests(
    requests: Sequence[Mapping[str, Any]],
) -> TokenLedger:
    """Aggregate raw per-request records into one system-level ledger.

    Cancelled / timed-out / preempted requests contribute their computed
    positions (``cancelled_or_failed_positions``) and are never dropped from the
    denominator (§13.5).
    """
    logical = committed = cached = spec = recompute = cancelled = 0
    for request in requests:
        logical += int(request.get("logical_input_tokens", 0))
        committed += int(request.get("committed_output_tokens", 0))
        cached += int(request.get("cached_prefix_tokens", 0))
        spec += int(request.get("speculative_verified_positions", 0))
        recompute += int(request.get("preemption_recompute_positions", 0))
        if request.get("cancelled") or request.get("timed_out") or request.get("failed"):
            cancelled += int(request.get("computed_positions", 0))
    return TokenLedger(
        logical_input_tokens=logical,
        committed_output_tokens=committed,
        cached_prefix_tokens=cached,
        speculative_verified_positions=spec,
        preemption_recompute_positions=recompute,
        cancelled_or_failed_positions=cancelled,
    )


__all__ = [
    "DistributionSummary",
    "EquivalenceCheck",
    "PairedEffect",
    "RequestTimeline",
    "RunLevelSamples",
    "TIMELINE_FIELDS",
    "TokenConservationReport",
    "TokenLedger",
    "distribution_summary",
    "paired_effect",
    "percentile",
    "token_accounting_from_requests",
]
