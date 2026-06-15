"""Replayable arrival processes and the intended-vs-actual fidelity gate.

E08-03 only means something if the generator is *frozen before the run*: every
trace here is produced offline from an explicit seed, carries its distribution
metadata and a content hash, and never changes when the service slows down.
"A loop that suddenly sleeps" is not a burst model, so the burst generators
below fix peak/base/duty (or batch size/period) and **re-normalise so that the
long-run mean rate matches the constant and Poisson traces** — otherwise the
three distributions would not be comparable.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.serving.slo import percentile_from_raw

#: Distribution identifiers of the frozen arrival spec.
DISTRIBUTIONS: Tuple[str, ...] = (
    "constant",
    "poisson",
    "on_off_burst",
    "batch_burst",
    "compound_poisson",
    "trace_replay",
)

#: Peak windows the descriptive statistics must report (E08-03 §7).
PEAK_WINDOWS_MS: Tuple[float, ...] = (10.0, 100.0, 1000.0, 10000.0)

INVALID_LABEL = "LOADGEN_INVALID"


@dataclass(frozen=True)
class ArrivalSpec:
    """The frozen arrival protocol (``configs/serving/arrival_spec.yaml``)."""

    seeds: Tuple[int, ...]
    distributions: Mapping[str, Mapping[str, Any]]
    fidelity: Mapping[str, float]
    comparability: Mapping[str, Any]
    load_bands: Tuple[str, ...]
    schema_version: str = "1.0.0"

    @classmethod
    def from_document(cls, payload: Mapping[str, Any]) -> "ArrivalSpec":
        return cls(
            seeds=tuple(int(item) for item in payload["seeds"]),
            distributions={
                str(name): dict(body)
                for name, body in dict(payload["distributions"]).items()
            },
            fidelity={str(k): float(v) for k, v in dict(payload["fidelity_gate"]).items() if isinstance(v, (int, float))},
            comparability=dict(payload["comparability"]),
            load_bands=tuple(str(item) for item in payload["load_bands"]),
            schema_version=str(payload.get("schema_version", "1.0.0")),
        )

    def burst_cfg(self, name: str) -> Mapping[str, Any]:
        if name not in self.distributions:
            raise ConfigError(f"distribution {name!r} is not declared in the arrival spec")
        return self.distributions[name]


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


# ── generators (pure functions of the seed) ────────────────────────────────


def constant_deltas(mean_rate: float, count: int) -> List[float]:
    if mean_rate <= 0:
        raise ConfigError("the mean arrival rate must be positive")
    return [1.0 / mean_rate] * count


def poisson_deltas(mean_rate: float, count: int, *, seed: int) -> List[float]:
    """Exponential inter-arrival times: ``-ln(U_i) / λ`` (E08-03 §4)."""
    if mean_rate <= 0:
        raise ConfigError("the mean arrival rate must be positive")
    rng = random.Random(seed)
    deltas: List[float] = []
    for _ in range(count):
        u = rng.random()
        while u <= 0.0:  # never log(0)
            u = rng.random()
        deltas.append(-math.log(u) / mean_rate)
    return deltas


def on_off_deltas(
    mean_rate: float,
    count: int,
    *,
    peak_multiplier: float,
    base_multiplier: float,
    on_sec: float,
    off_sec: float,
    seed: int,
) -> Tuple[List[float], List[int], List[str]]:
    """Periodic on/off bursts re-normalised to keep the requested mean rate."""
    if on_sec <= 0 or off_sec <= 0:
        raise ConfigError("on/off burst needs positive durations")
    duty = on_sec / (on_sec + off_sec)
    raw_mean = peak_multiplier * duty + base_multiplier * (1 - duty)
    if raw_mean <= 0:
        raise ConfigError("on/off multipliers must be positive")
    factor = 1.0 / raw_mean
    peak_rate = mean_rate * peak_multiplier * factor
    base_rate = mean_rate * base_multiplier * factor
    rng = random.Random(seed)
    deltas: List[float] = []
    bursts: List[int] = []
    phases: List[str] = []
    elapsed = 0.0
    burst_id = 0
    while len(deltas) < count:
        on_phase = (elapsed % (on_sec + off_sec)) < on_sec
        rate = peak_rate if on_phase else base_rate
        delta = -math.log(max(rng.random(), 1e-12)) / rate
        deltas.append(delta)
        bursts.append(burst_id if on_phase else -1)
        phases.append("on" if on_phase else "off")
        elapsed += delta
        if not on_phase:
            burst_id += 1
    return deltas, bursts, phases


def batch_burst_deltas(
    mean_rate: float,
    count: int,
    *,
    batch_size: int,
    seed: int,
) -> Tuple[List[float], List[int]]:
    """``B`` requests per period; the period is derived from the mean rate.

    ``period = B / λ`` keeps the long-run rate equal to the other traces, which
    is exactly what the comparability rules require.
    """
    if batch_size <= 0:
        raise ConfigError("batch size must be positive")
    del seed  # the batch model is deterministic; the seed is kept for the trace hash
    period = batch_size / mean_rate
    deltas: List[float] = []
    bursts: List[int] = []
    burst_id = 0
    while len(deltas) < count:
        for position in range(batch_size):
            if len(deltas) >= count:
                break
            # B requests at the same instant, then wait one period: the long-run
            # rate is B / (B / λ) = λ, so the mean stays comparable
            deltas.append(period if position == batch_size - 1 else 0.0)
            bursts.append(burst_id)
        burst_id += 1
    return deltas, bursts


def compound_poisson_deltas(
    mean_rate: float,
    count: int,
    *,
    batch_mean: float,
    seed: int,
) -> Tuple[List[float], List[int]]:
    """Events at ``λ / mean_batch``; each event carries a random batch."""
    if batch_mean <= 1.0:
        raise ConfigError("compound Poisson needs mean batch size > 1")
    rng = random.Random(seed)
    event_rate = mean_rate / batch_mean
    probability = 1.0 / batch_mean
    deltas: List[float] = []
    bursts: List[int] = []
    event = 0
    while len(deltas) < count:
        # geometric batch size via inverse transform: E[size] = 1/p = b exactly,
        # so the realised arrival rate stays comparable with the constant and
        # Poisson traces (rounding a scaled exponential is biased low)
        size = 1 + int(
            math.log(max(rng.random(), 1e-12)) / math.log(1.0 - probability)
        )
        for position in range(size):
            if len(deltas) >= count:
                break
            delta = -math.log(max(rng.random(), 1e-12)) / event_rate if position == 0 else 0.0
            deltas.append(delta)
            bursts.append(event)
        event += 1
    return deltas, bursts


def trace_replay_deltas(timestamps_sec: Sequence[float], *, scale: float = 1.0) -> List[float]:
    """Replay recorded timestamps (scaled); representativeness must be documented."""
    if scale <= 0:
        raise ConfigError("scale must be positive")
    ordered = sorted(float(item) for item in timestamps_sec)
    deltas: List[float] = []
    for left, right in zip(ordered, ordered[1:]):
        deltas.append(max(0.0, (right - left)) * scale)
    return deltas


# ── trace artifact ─────────────────────────────────────────────────────────


@dataclass
class ArrivalTrace:
    """A frozen, hashed arrival schedule (offsets only, never content)."""

    distribution: str
    seed: int
    mean_rate: float
    deltas_sec: Tuple[float, ...]
    scheduled_ns: Tuple[int, ...]
    burst_id: Tuple[int, ...] = ()
    phase: Tuple[str, ...] = ()
    version: str = "s08.1"
    payload_ids: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.distribution not in DISTRIBUTIONS:
            raise ConfigError(f"unknown arrival distribution {self.distribution!r}")
        if len(self.deltas_sec) != len(self.scheduled_ns):
            raise ConfigError("arrival trace: deltas and schedule must have equal length")
        if any(value < 0 for value in self.scheduled_ns):
            raise ConfigError("arrival trace offsets must be monotonically increasing")

    @property
    def count(self) -> int:
        return len(self.scheduled_ns)

    @property
    def duration_sec(self) -> float:
        return self.scheduled_ns[-1] / 1e9 if self.scheduled_ns else 0.0

    def content_hash(self) -> str:
        return _canonical_hash(
            {
                "distribution": self.distribution,
                "version": self.version,
                "seed": self.seed,
                "mean_rate": self.mean_rate,
                "scheduled_ns": list(self.scheduled_ns),
            }
        )

    def statistics(self) -> Dict[str, Any]:
        return arrival_statistics(self.deltas_sec)

    def skeleton_rows(self) -> List[Dict[str, Any]]:
        """One row per request: arrival metadata only (content stays separate)."""
        rows: List[Dict[str, Any]] = []
        for index, offset in enumerate(self.scheduled_ns):
            rows.append(
                {
                    "request_index": index,
                    "payload_id": self.payload_ids[index] if self.payload_ids else index,
                    "distribution": self.distribution,
                    "version": self.version,
                    "seed": self.seed,
                    "scheduled_offset_ns": offset,
                    "inter_arrival_intended_ns": int(self.deltas_sec[index] * 1e9),
                    "burst_id": self.burst_id[index] if self.burst_id else -1,
                    "phase": self.phase[index] if self.phase else "",
                }
            )
        return rows

    def as_dict(self) -> Dict[str, Any]:
        return {
            "distribution": self.distribution,
            "version": self.version,
            "seed": self.seed,
            "mean_rate": self.mean_rate,
            "count": self.count,
            "duration_sec": self.duration_sec,
            "content_hash": self.content_hash(),
        }


def generate_arrival_trace(
    spec: ArrivalSpec,
    *,
    distribution: str,
    mean_rate: float,
    count: int,
    seed: int,
    duration_sec: Optional[float] = None,
) -> ArrivalTrace:
    """Generate one frozen trace; the service can never change the schedule."""
    cfg = spec.burst_cfg(distribution)
    burst_ids: List[int] = []
    phases: List[str] = []
    if distribution == "constant":
        deltas = constant_deltas(mean_rate, count)
    elif distribution == "poisson":
        deltas = poisson_deltas(mean_rate, count, seed=seed)
    elif distribution == "on_off_burst":
        deltas, burst_ids, phases = on_off_deltas(
            mean_rate,
            count,
            peak_multiplier=float(cfg["peak_rate_multiplier"]),
            base_multiplier=float(cfg["base_rate_multiplier"]),
            on_sec=float(cfg["burst_duration_sec"]),
            off_sec=float(cfg["off_duration_sec"]),
            seed=seed,
        )
    elif distribution == "batch_burst":
        deltas, burst_ids = batch_burst_deltas(
            mean_rate, count, batch_size=int(cfg["batch_size"]), seed=seed
        )
    elif distribution == "compound_poisson":
        deltas, burst_ids = compound_poisson_deltas(
            mean_rate, count, batch_mean=float(cfg["batch_size_mean"]), seed=seed
        )
    elif distribution == "trace_replay":
        raise ConfigError(
            "trace_replay needs recorded timestamps: call replay_trace(...) with the "
            "desensitised source instead of generating synthetic arrivals"
        )
    else:  # pragma: no cover - guarded by DISTRIBUTIONS
        raise ConfigError(f"unsupported distribution {distribution!r}")

    if duration_sec is not None:
        # scale the schedule so the measurement window matches across traces
        actual = sum(deltas)
        if actual > 0:
            factor = duration_sec / actual
            deltas = [delta * factor for delta in deltas]

    offsets: List[int] = []
    elapsed = 0.0
    for delta in deltas:
        elapsed += delta
        offsets.append(int(round(elapsed * 1e9)))
    return ArrivalTrace(
        distribution=distribution,
        seed=seed,
        mean_rate=mean_rate,
        deltas_sec=tuple(deltas),
        scheduled_ns=tuple(offsets),
        burst_id=tuple(burst_ids),
        phase=tuple(phases),
        payload_ids=tuple(range(count)),
    )


def replay_trace(
    timestamps_sec: Sequence[float],
    *,
    mean_rate: float,
    seed: int,
    scale: float = 1.0,
) -> ArrivalTrace:
    deltas = trace_replay_deltas(timestamps_sec, scale=scale)
    offsets: List[int] = []
    elapsed = 0.0
    for delta in deltas:
        elapsed += delta
        offsets.append(int(round(elapsed * 1e9)))
    return ArrivalTrace(
        distribution="trace_replay",
        seed=seed,
        mean_rate=mean_rate,
        deltas_sec=tuple(deltas),
        scheduled_ns=tuple(offsets),
        payload_ids=tuple(range(len(offsets))),
    )


# ── descriptive statistics (E08-03 §7) ─────────────────────────────────────


def arrival_statistics(deltas_sec: Sequence[float]) -> Dict[str, Any]:
    values = [float(item) for item in deltas_sec]
    if not values:
        return {"count": 0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    stddev = math.sqrt(variance)
    offsets: List[float] = []
    elapsed = 0.0
    for value in values:
        elapsed += value
        offsets.append(elapsed)
    peak_qps: Dict[str, float] = {}
    for window_ms in PEAK_WINDOWS_MS:
        window_sec = window_ms / 1000.0
        counts_per_window: List[int] = []
        if window_sec > 0:
            for start in offsets:
                counts_per_window.append(
                    sum(1 for item in offsets if start <= item < start + window_sec)
                )
        peak_qps[f"max_qps_{int(window_ms)}ms"] = (
            max(counts_per_window) / window_sec if counts_per_window and window_sec else 0.0
        )
    # Fano factor over one-second buckets: variance / mean of counts
    buckets: Dict[int, int] = {}
    for offset in offsets:
        bucket = int(offset)
        buckets[bucket] = buckets.get(bucket, 0) + 1
    counts = list(buckets.values())
    bucket_mean = sum(counts) / len(counts) if counts else 0.0
    bucket_variance = (
        sum((value - bucket_mean) ** 2 for value in counts) / len(counts) if counts else 0.0
    )
    fano = bucket_variance / bucket_mean if bucket_mean else 0.0
    # lag-1 autocorrelation of the inter-arrival series
    autocorr = None
    if len(values) > 2 and variance > 0:
        numerator = sum(
            (values[index] - mean) * (values[index + 1] - mean)
            for index in range(len(values) - 1)
        )
        autocorr = numerator / ((len(values) - 1) * variance)
    return {
        "count": len(values),
        "mean_sec": mean,
        "median_sec": percentile_from_raw(values, 0.5),
        "p95_sec": percentile_from_raw(values, 0.95),
        "p99_sec": percentile_from_raw(values, 0.99),
        "stddev_sec": stddev,
        "cv": stddev / mean if mean else None,
        "peak_to_mean": (max(values) / mean) if mean else None,
        "peak_qps": peak_qps,
        "fano_factor": fano,
        "autocorrelation_lag1": autocorr,
        "note": "mean QPS equality alone never proves the generator is correct",
    }


def burst_summary(deltas_sec: Sequence[float], bursts: Sequence[int]) -> Dict[str, Any]:
    if not bursts:
        return {"bursts": 0, "note": "this trace has no burst structure"}
    grouped: Dict[int, List[float]] = {}
    for delta, burst in zip(deltas_sec, bursts):
        if burst < 0:
            continue
        grouped.setdefault(burst, []).append(delta)
    sizes = [len(values) for values in grouped.values()]
    durations = [sum(values) for values in grouped.values()]
    total = sum(deltas_sec)
    return {
        "bursts": len(grouped),
        "burst_size_mean": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "burst_duration_mean_sec": (sum(durations) / len(durations)) if durations else 0.0,
        "duty_cycle": (sum(durations) / total) if total else 0.0,
    }


# ── comparability and fidelity gates ───────────────────────────────────────


def compare_arrival_traces(
    traces: Sequence[ArrivalTrace], *, tolerance_rel: float = 0.05
) -> Dict[str, Any]:
    """The three distributions must be comparable before they are compared."""
    problems: List[str] = []
    if len({trace.count for trace in traces}) > 1:
        problems.append("the traces do not offer the same number of requests")
    requested_rates = {trace.mean_rate for trace in traces}
    if len(requested_rates) > 1:
        problems.append(
            f"the traces were generated with different mean rates {sorted(requested_rates)}; "
            "comparability requires one pre-registered rate"
        )
    reference = next(iter(requested_rates)) if requested_rates else None
    # ``1 / mean(inter-arrival)`` is the generator's mean rate; ``count / span``
    # would be biased by the last (random) gap and would flag sampling noise.
    rates: List[float] = []
    durations: List[float] = []
    for trace in traces:
        stats = trace.statistics()
        mean_delta = float(stats.get("mean_sec") or 0.0)
        if mean_delta <= 0:
            continue
        rates.append(1.0 / mean_delta)
        durations.append(trace.count * mean_delta)
    if reference:
        for trace, rate in zip(traces, rates):
            stats = trace.statistics()
            cv = float(stats.get("cv") or 0.0)
            # standard error of the mean inter-arrival over n samples
            effective = max(tolerance_rel, 3.0 * cv / math.sqrt(max(trace.count, 1)))
            if abs(rate - reference) / reference > effective:
                problems.append(
                    f"{trace.distribution}: mean rate {rate:.3f}/s differs from the "
                    f"pre-registered rate {reference:.3f}/s by more than {effective:.1%}"
                )
    if durations:
        spread = (max(durations) - min(durations)) / max(durations)
        # a fixed request count plus bursty arrivals implies a spread in the
        # implied duration: allow the statistical amount, flag anything beyond
        worst_cv = max(float(trace.statistics().get("cv") or 0.0) for trace in traces)
        smallest = max(min(trace.count for trace in traces), 1)
        allowed_spread = max(tolerance_rel, 3.0 * worst_cv / math.sqrt(smallest))
        if spread > allowed_spread:
            problems.append(
                f"implied measurement durations differ by {spread:.1%} "
                f"(allowed {allowed_spread:.1%} for this arrival variance)"
            )
    return {
        "ok": not problems,
        "problems": problems,
        "rates_per_sec": rates,
        "implied_durations_sec": durations,
        "note": "same payload sequence, same warmup/drain and same cache state are required too",
    }


@dataclass(frozen=True)
class ActualArrival:
    """What the loadgen really sent (the intended schedule is separate)."""

    scheduled_ns: int
    sent_ns: Optional[int]
    dropped: bool = False
    drop_reason: str = ""

    @property
    def lag_ns(self) -> Optional[int]:
        if self.sent_ns is None:
            return None
        return self.sent_ns - self.scheduled_ns

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scheduled_ns": self.scheduled_ns,
            "sent_ns": self.sent_ns,
            "lag_ns": self.lag_ns,
            "dropped": self.dropped,
            "drop_reason": self.drop_reason,
        }


def fidelity_gate(
    actual: Sequence[ActualArrival],
    *,
    spec: ArrivalSpec,
    mean_rate: float,
) -> Dict[str, Any]:
    """Intended vs. actual arrival: a client-side failure is *labelled*, not hidden."""
    problems: List[str] = []
    sent = [item for item in actual if not item.dropped]
    lags = sorted(item.lag_ns for item in sent if item.lag_ns is not None)
    drop_ratio = (len(actual) - len(sent)) / len(actual) if actual else 0.0
    duration = (actual[-1].scheduled_ns - actual[0].scheduled_ns) / 1e9 if len(actual) > 1 else 0.0
    realised_rate = len(sent) / duration if duration > 0 else 0.0
    rate_error = abs(realised_rate - mean_rate) / mean_rate if mean_rate else 0.0
    lag_p95 = percentile_from_raw(lags, 0.95) if lags else 0.0
    if rate_error > spec.fidelity.get("mean_rate_rel_tolerance", 0.05):
        problems.append(f"realised rate error {rate_error:.3f} exceeds tolerance")
    if lag_p95 is not None and lag_p95 > spec.fidelity.get("lag_p95_max_ms", 20.0) * 1e6:
        problems.append("loadgen lag p95 exceeds the tolerance")
    if drop_ratio > spec.fidelity.get("drop_ratio_max", 0.001):
        problems.append(f"drop ratio {drop_ratio:.4f} exceeds tolerance")
    return {
        "ok": not problems,
        "problems": problems,
        "label": INVALID_LABEL if problems else "",
        "realised_rate_per_sec": realised_rate,
        "rate_error_rel": rate_error,
        "lag_p95_ms": (lag_p95 / 1e6) if lag_p95 is not None else None,
        "drop_ratio": drop_ratio,
        "note": (
            "if the loadgen itself saturates, the point is LOADGEN_INVALID and must not "
            "be attributed to the service"
        ),
    }


# ── transient and recovery ─────────────────────────────────────────────────


@dataclass(frozen=True)
class QueueSample:
    """One queue observation (depth/age) on the service timeline."""

    monotonic_ns: int
    queue_depth: float
    oldest_age_ms: float = 0.0
    slo_ok: bool = True
    inflight: float = 0.0
    memory_bytes: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "monotonic_ns": self.monotonic_ns,
            "queue_depth": self.queue_depth,
            "oldest_age_ms": self.oldest_age_ms,
            "slo_ok": self.slo_ok,
            "inflight": self.inflight,
            "memory_bytes": self.memory_bytes,
        }


def burst_recovery_metrics(
    samples: Sequence[QueueSample],
    *,
    burst_end_ns: int,
    baseline_depth: float = 0.0,
    depth_tolerance: float = 1.0,
    memory_tolerance_bytes: float = 0.0,
) -> Dict[str, Any]:
    """Peak, decay and recovery times of one burst (E08-03 §10)."""
    if not samples:
        return {"status": "NO_SAMPLES"}
    peak = max(samples, key=lambda item: item.queue_depth)
    after = [item for item in samples if item.monotonic_ns >= burst_end_ns]
    queue_baseline_ns: Optional[int] = None
    for item in after:
        if abs(item.queue_depth - baseline_depth) <= depth_tolerance:
            queue_baseline_ns = item.monotonic_ns
            break
    slo_recovery_ns: Optional[int] = None
    for item in after:
        if item.slo_ok:
            slo_recovery_ns = item.monotonic_ns
            break
    memory_peak = max((item.memory_bytes for item in samples), default=0.0)
    memory_recovered = all(
        item.memory_bytes <= memory_peak + memory_tolerance_bytes for item in after[-3:]
    ) if len(after) >= 3 else False
    return {
        "status": "measured",
        "queue_peak": peak.queue_depth,
        "time_to_peak_ms": (peak.monotonic_ns - samples[0].monotonic_ns) / 1e6,
        "burst_end_ns": burst_end_ns,
        "time_to_queue_baseline_ms": (
            (queue_baseline_ns - burst_end_ns) / 1e6 if queue_baseline_ns else None
        ),
        "time_to_slo_recovery_ms": (
            (slo_recovery_ns - burst_end_ns) / 1e6 if slo_recovery_ns else None
        ),
        "oldest_age_peak_ms": max(item.oldest_age_ms for item in samples),
        "memory_recovered": memory_recovered,
        "censored": queue_baseline_ns is None or slo_recovery_ns is None,
        "note": "an unrecovered burst is reported as censored, never as balanced",
    }


@dataclass(frozen=True)
class BurstinessPoint:
    mean_load_qps: float
    burstiness: float  # peak-to-mean of the arrival process
    goodput: float
    reject_ratio: float
    p99_ttft_ms: Optional[float]
    within_slo: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mean_load_qps": self.mean_load_qps,
            "burstiness": self.burstiness,
            "goodput": self.goodput,
            "reject_ratio": self.reject_ratio,
            "p99_ttft_ms": self.p99_ttft_ms,
            "within_slo": self.within_slo,
        }


def arrival_sensitivity_map(points: Sequence[BurstinessPoint]) -> Dict[str, Any]:
    """Safe / sensitive / reject zones over (mean load × burstiness) (E08-03 §24)."""
    zones: Dict[str, List[Dict[str, Any]]] = {"safe": [], "sensitive": [], "reject": []}
    for point in points:
        if point.reject_ratio > 0.01:
            zone = "reject"
        elif point.within_slo:
            zone = "safe"
        else:
            zone = "sensitive"
        zones[zone].append(point.as_dict())
    return {
        "zones": zones,
        "counts": {name: len(rows) for name, rows in zones.items()},
        "note": (
            "the map is conditional on the frozen payload/prefix/tenant mix; it is not a "
            "production traffic statement"
        ),
    }


__all__ = [
    "ActualArrival",
    "ArrivalSpec",
    "ArrivalTrace",
    "BurstinessPoint",
    "DISTRIBUTIONS",
    "INVALID_LABEL",
    "PEAK_WINDOWS_MS",
    "QueueSample",
    "arrival_sensitivity_map",
    "arrival_statistics",
    "batch_burst_deltas",
    "burst_recovery_metrics",
    "burst_summary",
    "compare_arrival_traces",
    "compound_poisson_deltas",
    "constant_deltas",
    "fidelity_gate",
    "generate_arrival_trace",
    "on_off_deltas",
    "poisson_deltas",
    "replay_trace",
    "trace_replay_deltas",
]
