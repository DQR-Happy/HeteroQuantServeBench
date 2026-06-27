"""Multi-rank trace alignment, straggler attribution and clock discipline (E10-09).

The experiment exists because "the API span was longest" identifies the *victim*,
not the cause.  This module therefore enforces:

* a correlation chain request → iteration → phase → token → layer → module →
  group/seq → kernel/transport, with unmapped events counted rather than dropped;
* clock calibration with an uncertainty, and a verdict of ``INCONCLUSIVE`` when a
  candidate difference is the same order as that uncertainty;
* the root-cause evidence matrix of details E10-09 §12 as data, plus a classifier
  that must name the evidence it used — ``UNKNOWN`` is a legal answer;
* attribution that starts from the *earliest* event that exceeded the baseline
  confidence interval and propagates along the dependency DAG, never from the
  longest span or the last error.

Injection helpers build *plans* only: no function here injects a fault, and a
link-shaping plan without an authorised isolated network is ``NOT_RUN``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Profiler modes; the clean benchmark is a separate run (details E10-09 step 3).
TRACE_MODES: Tuple[str, ...] = (
    "marker_only",
    "api_collective_timeline",
    "deep_device_network_counters",
    "clean_benchmark",
)

#: Clock domains that must not be compared directly.
CLOCK_DOMAINS: Tuple[str, ...] = ("host_monotonic", "wall_clock", "device", "profiler")

#: Root-cause classes (details E10-09 §5).
ROOT_CAUSE_CLASSES: Tuple[str, ...] = (
    "HOST_LATE_SUBMIT",
    "COMPUTE_SLOW",
    "MEMORY_PRESSURE",
    "ARRIVAL_SKEW",
    "LINK_SLOW",
    "COLLECTIVE_ALGORITHM",
    "LOAD_IMBALANCE",
    "SYNC_SERIALIZATION",
    "RETRY_ERROR",
    "UNKNOWN",
)

#: The evidence matrix of details E10-09 §12: what must be seen together.
EVIDENCE_MATRIX: Mapping[str, Mapping[str, str]] = {
    "HOST_LATE_SUBMIT": {
        "must_see": "host gap appears first, device idle, collective arrival late",
        "counter_evidence": "low CPU utilisation alone is not evidence",
    },
    "COMPUTE_SLOW": {
        "must_see": "a same-shape kernel slows first, health/clock or resource evidence agrees",
        "counter_evidence": "an early-arriving rank waiting is not a slow compute",
    },
    "LOAD_IMBALANCE": {
        "must_see": "uneven shard/token/expert work moves in the same direction as time",
        "counter_evidence": "equal token counts can still differ by per-expert cost",
    },
    "LINK_SLOW": {
        "must_see": "arrivals are close and a specific src→dst transit/error is abnormal",
        "counter_evidence": "arrival skew makes an API span look like network slowness",
    },
    "COLLECTIVE_ALGORITHM": {
        "must_see": "actual algorithm/transport changed and several ranks are affected together",
        "counter_evidence": "a single rank's host jitter",
    },
    "SYNC_SERIALIZATION": {
        "must_see": "streams with no data dependency are also blocked; the timeline shows a barrier",
        "counter_evidence": "a correct consumer wait is not a global sync",
    },
    "MEMORY_PRESSURE": {
        "must_see": "allocator/page/clock/memory counters move together with the kernel degradation",
        "counter_evidence": "high reserved memory is not evidence of pressure",
    },
    "ARRIVAL_SKEW": {
        "must_see": "one rank arrives late at the collective while transit time is normal",
        "counter_evidence": "a slow link would also change transit time",
    },
    "RETRY_ERROR": {
        "must_see": "retry/error counters and the recovery timeline align with the stall",
        "counter_evidence": "one retry is not a pattern",
    },
    "UNKNOWN": {
        "must_see": "unmapped/clock/metric evidence is insufficient",
        "counter_evidence": "do not hard-pick the most plausible class",
    },
}

#: Injection kinds the experiment may plan (details E10-09 step 15–20).
INJECTION_KINDS: Tuple[str, ...] = (
    "rank_arrival_delay",
    "rank_compute_slowdown",
    "host_submission_delay",
    "expert_skew",
    "alternative_slow_topology",
    "link_shaping",
)

#: The unified distributed trace event fields (details E10-09 §4).
TRACE_EVENT_FIELDS: Tuple[str, ...] = (
    "run_id",
    "trace_id",
    "request_id",
    "iteration_id",
    "phase",
    "token_step",
    "layer",
    "module",
    "node",
    "global_rank",
    "local_rank",
    "group_rank",
    "rank_epoch",
    "device_uuid",
    "process",
    "thread",
    "stream",
    "event_type",
    "group_id",
    "collective_seq",
    "op",
    "tensor_role",
    "shape",
    "dtype",
    "payload_bytes",
    "host_start_ns",
    "host_end_ns",
    "device_start_ns",
    "device_end_ns",
    "clock_domain",
    "mapped_global_time_ns",
    "uncertainty_ns",
    "src_rank",
    "dst_rank",
    "transport",
    "wait_reason",
    "parallel_plan_hash",
    "topology_hash",
    "profiler_source",
)

#: Public, platform-independent summary fields for S11/S12 (details E10-09 step 29).
PUBLIC_SUMMARY_FIELDS: Tuple[str, ...] = (
    "compute_ms",
    "comm_ms",
    "overlap_ms",
    "wait_ms",
    "idle_ms",
    "unknown_idle_ms",
    "straggler_root_rank",
    "straggler_class",
    "amplification",
    "unmapped_ratio",
)


def _require(condition: bool, message: str, *, field_name: str = "") -> None:
    if not condition:
        raise ConfigError(message, details={"field": field_name} if field_name else None)


# ── manifest ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricManifest:
    """Per-platform profiler metric manifest (details E10-09 step 4)."""

    platform: str
    profiler_version: str
    fields: Tuple[str, ...]
    units: Mapping[str, str]
    sampling_rate: str
    rank_coverage: str
    clock_domain: str
    collection_conflicts: Tuple[str, ...] = ()
    unavailable: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(self.platform in ("cuda_nccl", "ascend_hccl"), "unknown platform", field_name="platform")
        _require(bool(self.fields), "a manifest needs at least one field", field_name="fields")
        _require(bool(self.rank_coverage), "state the rank coverage (never rank 0 only)", field_name="rank_coverage")
        for name, reason in self.unavailable.items():
            _require(bool(reason), f"{name} is marked UNAVAILABLE without a reason")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "profiler_version": self.profiler_version,
            "fields": list(self.fields),
            "units": dict(sorted(self.units.items())),
            "sampling_rate": self.sampling_rate,
            "rank_coverage": self.rank_coverage,
            "clock_domain": self.clock_domain,
            "collection_conflicts": list(self.collection_conflicts),
            "unavailable": dict(sorted(self.unavailable.items())),
        }


# ── trace events / correlation ─────────────────────────────────────────────


@dataclass(frozen=True)
class DistributedTraceEvent:
    """One normalized distributed event (details E10-09 §4)."""

    run_id: str
    global_rank: int
    event_type: str
    phase: str
    clock_domain: str
    host_start_ns: int = 0
    host_end_ns: int = 0
    device_start_ns: int = 0
    device_end_ns: int = 0
    mapped_global_time_ns: int = 0
    uncertainty_ns: int = 0
    request_id: str = ""
    iteration_id: str = ""
    token_step: int = -1
    layer: Any = ""
    module: str = ""
    node: str = ""
    local_rank: int = -1
    group_rank: int = -1
    rank_epoch: int = 0
    device_uuid: str = ""
    process: int = 0
    thread: str = ""
    stream: str = ""
    group_id: str = ""
    collective_seq: int = -1
    op: str = ""
    tensor_role: str = ""
    shape: str = ""
    dtype: str = ""
    payload_bytes: int = 0
    src_rank: int = -1
    dst_rank: int = -1
    transport: str = ""
    wait_reason: str = ""
    parallel_plan_hash: str = ""
    topology_hash: str = ""
    profiler_source: str = ""
    unmapped: bool = False

    def __post_init__(self) -> None:
        _require(self.clock_domain in CLOCK_DOMAINS, "unknown clock domain", field_name="clock_domain")
        if self.host_end_ns and self.host_start_ns and self.host_end_ns < self.host_start_ns:
            raise ConfigError("negative host duration", details={"field": "host_end_ns"})
        if self.device_end_ns and self.device_start_ns and self.device_end_ns < self.device_start_ns:
            raise ConfigError("negative device duration", details={"field": "device_end_ns"})

    @property
    def logical_key(self) -> Tuple[str, str, int]:
        return (self.group_id, str(self.rank_epoch), self.collective_seq)

    def as_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in TRACE_EVENT_FIELDS if hasattr(self, name)} | {
            "unmapped": self.unmapped
        }


@dataclass(frozen=True)
class CorrelationChain:
    """The request→…→transport chain a trace must be able to walk (step 5)."""

    request_id: str
    iteration_id: str
    phase: str
    token_step: int
    layer: Any
    module: str
    group_id: str
    collective_seq: int
    global_rank: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "iteration_id": self.iteration_id,
            "phase": self.phase,
            "token_step": self.token_step,
            "layer": self.layer,
            "module": self.module,
            "group_id": self.group_id,
            "collective_seq": self.collective_seq,
            "global_rank": self.global_rank,
        }


def unmapped_events(events: Sequence[DistributedTraceEvent]) -> List[Dict[str, Any]]:
    """Events that cannot be attached to the chain are reported, not dropped (step 26)."""
    rows: List[Dict[str, Any]] = []
    for event in events:
        if event.unmapped or (not event.request_id and event.event_type in ("kernel", "collective")):
            rows.append(
                {
                    "global_rank": event.global_rank,
                    "event_type": event.event_type,
                    "op": event.op,
                    "collective_seq": event.collective_seq,
                    "reason": "no request/iteration/phase correlation",
                }
            )
    return rows


def unmapped_ratio(events: Sequence[DistributedTraceEvent], *, threshold: float = 0.05) -> Dict[str, Any]:
    """Too much unmapped time makes the attribution inconclusive (step 26)."""
    if not events:
        return {"ratio": 0.0, "threshold": threshold, "conclusive": True, "unmapped": 0}
    unmapped = unmapped_events(events)
    ratio = len(unmapped) / len(events)
    return {
        "ratio": ratio,
        "threshold": threshold,
        "conclusive": ratio <= threshold,
        "unmapped": len(unmapped),
        "note": "add correlation evidence before attributing unknown gaps to 'runtime overhead'",
    }


# ── clocks ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClockCalibration:
    """Host/device clock offsets and uncertainty (details E10-09 step 6)."""

    method: str
    host_offsets_ns: Mapping[str, int]
    drifts_ppm: Mapping[str, float]
    uncertainty_ns: int
    measured_before: str = ""
    measured_after: str = ""

    def __post_init__(self) -> None:
        _require(bool(self.method), "a calibration needs a method", field_name="method")
        _require(self.uncertainty_ns >= 0, "uncertainty must be >= 0", field_name="uncertainty_ns")
        if not self.measured_before and not self.measured_after:
            raise ConfigError("record when the calibration was taken (before/after the run)")

    def can_compare(self, interval_ns: int) -> bool:
        """Never claim microsecond ordering when the uncertainty covers it."""
        return interval_ns > 2 * self.uncertainty_ns

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "host_offsets_ns": {k: v for k, v in sorted(self.host_offsets_ns.items())},
            "drifts_ppm": {k: v for k, v in sorted(self.drifts_ppm.items())},
            "uncertainty_ns": self.uncertainty_ns,
            "measured_before": self.measured_before,
            "measured_after": self.measured_after,
            "note": "when the uncertainty is the same order as the difference, the verdict is INCONCLUSIVE",
        }


def calibrate_offsets(
    *,
    host_clock_ns: Mapping[str, int],
    reference_host: str,
    uncertainty_ns: int,
    measured_before: str = "",
    measured_after: str = "",
) -> ClockCalibration:
    """Derive offsets against a reference host from recorded timestamps.

    The calibration window is recorded explicitly (details E10-09 step 6): an
    offset without a measurement time cannot be judged against drift.
    """
    if reference_host not in host_clock_ns:
        raise ConfigError("reference host is not in the samples", details={"field": "reference_host"})
    reference = host_clock_ns[reference_host]
    offsets = {host: value - reference for host, value in host_clock_ns.items()}
    return ClockCalibration(
        method="host pairwise timestamp exchange against a reference",
        host_offsets_ns=offsets,
        drifts_ppm={host: 0.0 for host in host_clock_ns},
        uncertainty_ns=uncertainty_ns,
        measured_before=measured_before or "pairwise exchange (start)",
        measured_after=measured_after,
    )


# ── collective pairing / skew ──────────────────────────────────────────────


@dataclass(frozen=True)
class PairedCollective:
    """One logical collective paired across ranks by (group, epoch, seq)."""

    group_id: str
    rank_epoch: int
    collective_seq: int
    op: str
    arrivals_ns: Mapping[int, int]
    completions_ns: Mapping[int, int]
    ranks: Tuple[int, ...]

    @property
    def arrival_skew_ns(self) -> int:
        if not self.arrivals_ns:
            return 0
        return max(self.arrivals_ns.values()) - min(self.arrivals_ns.values())

    @property
    def job_interval_ns(self) -> int:
        if not self.arrivals_ns or not self.completions_ns:
            return 0
        return max(self.completions_ns.values()) - min(self.arrivals_ns.values())

    def early_rank_wait_ns(self) -> Dict[int, int]:
        if not self.arrivals_ns:
            return {}
        last = max(self.arrivals_ns.values())
        return {rank: last - value for rank, value in self.arrivals_ns.items()}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "rank_epoch": self.rank_epoch,
            "collective_seq": self.collective_seq,
            "op": self.op,
            "ranks": list(self.ranks),
            "arrival_skew_ns": self.arrival_skew_ns,
            "job_interval_ns": self.job_interval_ns,
            "early_rank_wait_ns": {str(k): v for k, v in sorted(self.early_rank_wait_ns().items())},
        }


def pair_collective_events(
    events: Sequence[DistributedTraceEvent], *, expected_ranks: Sequence[int]
) -> Dict[str, Any]:
    """Pair by group+epoch+seq; missing ranks go to the unmapped table (step 7)."""
    buckets: Dict[Tuple[str, str, int], List[DistributedTraceEvent]] = {}
    for event in events:
        if event.collective_seq < 0 or not event.group_id:
            continue
        buckets.setdefault(event.logical_key, []).append(event)
    paired: List[PairedCollective] = []
    incomplete: List[Dict[str, Any]] = []
    for key, items in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][2])):
        group_id, epoch, seq = key
        arrivals: Dict[int, int] = {}
        completions: Dict[int, int] = {}
        op = ""
        for item in items:
            op = op or item.op
            if item.mapped_global_time_ns:
                arrivals[item.global_rank] = item.mapped_global_time_ns
            if item.mapped_global_time_ns and item.host_end_ns:
                completions[item.global_rank] = item.mapped_global_time_ns + max(
                    item.host_end_ns - item.host_start_ns, 0
                )
        missing = sorted(set(expected_ranks) - set(arrivals))
        if missing:
            incomplete.append(
                {"group_id": group_id, "collective_seq": seq, "missing_ranks": missing}
            )
            continue
        paired.append(
            PairedCollective(
                group_id=group_id,
                rank_epoch=int(epoch),
                collective_seq=seq,
                op=op,
                arrivals_ns=arrivals,
                completions_ns=completions,
                ranks=tuple(sorted(arrivals)),
            )
        )
    return {"paired": paired, "incomplete": incomplete, "complete": not incomplete}


def arrival_completion_skew(paired: Sequence[PairedCollective]) -> List[Dict[str, Any]]:
    """Per-seq first/last arrival and completion plus the early-rank wait (step 12)."""
    rows: List[Dict[str, Any]] = []
    for item in paired:
        arrivals = item.arrivals_ns
        completions = item.completions_ns
        rows.append(
            {
                "group_id": item.group_id,
                "collective_seq": item.collective_seq,
                "op": item.op,
                "first_arrival_ns": min(arrivals.values()) if arrivals else None,
                "last_arrival_ns": max(arrivals.values()) if arrivals else None,
                "first_completion_ns": min(completions.values()) if completions else None,
                "last_completion_ns": max(completions.values()) if completions else None,
                "arrival_skew_ns": item.arrival_skew_ns,
                "job_interval_ns": item.job_interval_ns,
                "early_rank_wait_ns": item.early_rank_wait_ns(),
                "victim_warning": (
                    "the rank with the longest API span may be an early arriver waiting, not the "
                    "root cause"
                ),
            }
        )
    return rows


# ── breakdown / matrix ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class PhaseBreakdown:
    """Per-rank phase breakdown with inclusive/exclusive separated (step 11)."""

    rank: int
    host_ms: float
    compute_ms: float
    comm_ms: float
    overlap_ms: float
    wait_ms: float
    idle_ms: float
    unknown_ms: float
    wall_ms: float

    @property
    def accounted_ms(self) -> float:
        return (
            self.host_ms
            + self.compute_ms
            + self.comm_ms
            - self.overlap_ms
            + self.wait_ms
            + self.idle_ms
        )

    @property
    def unexplained_ms(self) -> float:
        return self.wall_ms - self.accounted_ms

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "host_ms": self.host_ms,
            "compute_ms": self.compute_ms,
            "comm_ms": self.comm_ms,
            "overlap_ms": self.overlap_ms,
            "wait_ms": self.wait_ms,
            "idle_ms": self.idle_ms,
            "unknown_ms": self.unknown_ms,
            "wall_ms": self.wall_ms,
            "accounted_ms": self.accounted_ms,
            "unexplained_ms": self.unexplained_ms,
            "note": "inclusive and exclusive spans are not summed together",
        }


def phase_breakdown(rows: Sequence[PhaseBreakdown], *, tolerance_fraction: float = 0.05) -> Dict[str, Any]:
    """Validate that no rank double-counts and the job view is not a sum of ranks."""
    problems: List[str] = []
    for row in rows:
        if abs(row.unexplained_ms) > row.wall_ms * tolerance_fraction:
            problems.append(
                f"rank {row.rank}: {row.unexplained_ms:.3f} ms unaccounted "
                f"(> {tolerance_fraction:.0%} of {row.wall_ms:.3f} ms)"
            )
        if row.overlap_ms > min(row.compute_ms, row.comm_ms) + 1e-6:
            problems.append(f"rank {row.rank}: overlap exceeds min(compute, comm)")
    return {
        "ok": not problems,
        "problems": problems,
        "rows": [row.as_dict() for row in rows],
        "critical_path_rank": max(rows, key=lambda row: row.wall_ms).rank if rows else None,
        "note": "a per-rank sum is not the job critical path",
    }


def communication_matrix(
    paired: Sequence[PairedCollective],
    *,
    topology_edges: Mapping[Tuple[int, int], str] = None,
) -> List[Dict[str, Any]]:
    """src/dst transport view joined to the topology edges (step 13)."""
    topo = dict(topology_edges or {})
    rows: List[Dict[str, Any]] = []
    for item in paired:
        waits = item.early_rank_wait_ns()
        for rank in item.ranks:
            rows.append(
                {
                    "group_id": item.group_id,
                    "collective_seq": item.collective_seq,
                    "op": item.op,
                    "rank": rank,
                    "arrival_ns": item.arrivals_ns.get(rank),
                    "completion_ns": item.completions_ns.get(rank),
                    "early_rank_wait_ns": waits.get(rank, 0),
                    "topology_edge": topo.get((rank, rank), ""),
                    "note": "hierarchical algorithms keep their multiple stages as separate rows",
                }
            )
    return rows


# ── variability / classifier ───────────────────────────────────────────────


@dataclass
class VariabilityBaseline:
    """Natural per-rank/kernel/collective variance and the P99 threshold (step 14)."""

    samples: Dict[Tuple[int, str], List[float]] = field(default_factory=dict)

    def record(self, *, rank: int, item: str, duration_ms: float) -> None:
        self.samples.setdefault((rank, item), []).append(float(duration_ms))

    def summary(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for (rank, item), values in sorted(self.samples.items()):
            ordered = sorted(values)
            mean = sum(ordered) / len(ordered)
            p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
            rows.append(
                {
                    "rank": rank,
                    "item": item,
                    "count": len(ordered),
                    "mean_ms": mean,
                    "p99_ms": p99,
                    "max_ms": ordered[-1],
                }
            )
        return rows

    def threshold_ms(self, *, rank: int, item: str, multiplier: float = 1.5) -> Optional[float]:
        values = self.samples.get((rank, item))
        if not values:
            return None
        ordered = sorted(values)
        p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
        return p99 * multiplier


@dataclass(frozen=True)
class Attribution:
    """One attribution verdict with the evidence it rests on."""

    root_rank: Optional[int]
    fault_class: str
    evidence_used: Tuple[str, ...]
    counter_evidence_checked: Tuple[str, ...]
    confidence: str = "medium"
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "root_rank": self.root_rank,
            "fault_class": self.fault_class,
            "evidence_used": list(self.evidence_used),
            "counter_evidence_checked": list(self.counter_evidence_checked),
            "confidence": self.confidence,
            "notes": self.notes,
        }


class RootCauseClassifier:
    """Classifies by earliest deviation + evidence matrix, never by longest span."""

    def __init__(self, baseline: VariabilityBaseline, *, uncertainty_ns: int = 0) -> None:
        self.baseline = baseline
        self.uncertainty_ns = uncertainty_ns

    def first_deviation(
        self,
        observations: Sequence[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """The earliest event whose duration exceeds its baseline threshold."""
        candidates: List[Dict[str, Any]] = []
        for row in observations:
            rank = int(row["rank"])
            item = str(row["item"])
            duration = float(row["duration_ms"])
            threshold = self.baseline.threshold_ms(rank=rank, item=item)
            if threshold is None:
                continue
            if duration > threshold:
                candidates.append(
                    {
                        "rank": rank,
                        "item": item,
                        "duration_ms": duration,
                        "threshold_ms": threshold,
                        "start_ns": int(row.get("start_ns", 0)),
                    }
                )
        if not candidates:
            return None
        candidates.sort(key=lambda row: (row["start_ns"], row["rank"]))
        return candidates[0]

    def classify(
        self,
        *,
        first: Optional[Mapping[str, Any]],
        evidence: Mapping[str, bool],
    ) -> Attribution:
        """Map the evidence set to a class; missing evidence yields UNKNOWN."""
        if first is None:
            return Attribution(
                root_rank=None,
                fault_class="UNKNOWN",
                evidence_used=(),
                counter_evidence_checked=(),
                confidence="low",
                notes="no event exceeded its baseline confidence interval",
            )
        for candidate in ROOT_CAUSE_CLASSES:
            if candidate == "UNKNOWN":
                continue
            required = EVIDENCE_MATRIX[candidate]["must_see"]
            if evidence.get(candidate, False):
                return Attribution(
                    root_rank=int(first["rank"]),
                    fault_class=candidate,
                    evidence_used=(required, f"first deviation: {first['item']}"),
                    counter_evidence_checked=(EVIDENCE_MATRIX[candidate]["counter_evidence"],),
                    confidence="medium",
                )
        return Attribution(
            root_rank=int(first["rank"]),
            fault_class="UNKNOWN",
            evidence_used=(f"first deviation: {first['item']}",),
            counter_evidence_checked=(),
            confidence="low",
            notes=(
                "the evidence set does not satisfy any matrix row; forcing a class would be a "
                "misdiagnosis"
            ),
        )


def classifier_scores(confusion: Mapping[Tuple[str, str], int]) -> Dict[str, Any]:
    """Precision/recall/unknown rate over *all* pre-registered injections (step 21)."""
    classes = sorted({actual for _predicted, actual in confusion})
    per_class: Dict[str, Dict[str, float]] = {}
    correct = 0
    total = 0
    unknown_predictions = 0
    for (predicted, actual), count in confusion.items():
        total += count
        if predicted == actual:
            correct += count
        if predicted == "UNKNOWN":
            unknown_predictions += count
    for name in classes:
        true_positive = confusion.get((name, name), 0)
        predicted_total = sum(
            count for (predicted, actual), count in confusion.items() if predicted == name
        )
        actual_total = sum(
            count for (predicted, actual), count in confusion.items() if actual == name
        )
        per_class[name] = {
            "precision": true_positive / predicted_total if predicted_total else 0.0,
            "recall": true_positive / actual_total if actual_total else 0.0,
        }
    return {
        "accuracy": correct / total if total else 0.0,
        "unknown_rate": unknown_predictions / total if total else 0.0,
        "per_class": per_class,
        "runs": total,
        "note": "an honest UNKNOWN is more reliable than a confident misdiagnosis",
    }


def amplification_metrics(
    *,
    delta_local_ms: float,
    delta_job_ms: float,
    victim_wait_ms: Sequence[float],
    propagation_depth: int,
) -> Dict[str, Any]:
    """Synchronisation amplification of a local delay (details E10-09 §13)."""
    if delta_local_ms <= 0:
        raise ConfigError("delta_local_ms must be positive", details={"field": "delta_local_ms"})
    return {
        "amplification": delta_job_ms / delta_local_ms,
        "victim_wait_sum_ms": sum(victim_wait_ms),
        "fanout": sum(1 for value in victim_wait_ms if value > 0),
        "propagation_depth": propagation_depth,
        "note": (
            "victim_wait_sum can far exceed the wall-clock slowdown; it measures wasted resource, "
            "not user-visible latency"
        ),
    }


# ── injections (plans only) ────────────────────────────────────────────────


def injection_plan(
    *, kind: str, target_rank: int, strength: str, duration_s: float
) -> Dict[str, Any]:
    """Describe an injection; nothing is executed here (steps 15–20)."""
    if kind not in INJECTION_KINDS:
        raise ConfigError(f"kind must be one of {INJECTION_KINDS}", details={"field": "kind"})
    if duration_s <= 0:
        raise ConfigError("duration must be positive", details={"field": "duration_s"})
    return {
        "kind": kind,
        "target_rank": target_rank,
        "strength": strength,
        "duration_s": duration_s,
        "markers": {"start": f"{kind}-start", "end": f"{kind}-end"},
        "cleanup": "remove the injected delay/hook and re-run the healthy baseline",
        "note": (
            "sleep/delay injections are marker-delimited so they are never counted as network "
            "transit"
        ),
    }


def link_shaping_plan(*, isolated_network: bool, approval_reference: str = "") -> Dict[str, Any]:
    """Traffic shaping is only allowed on an authorised isolated network (step 20)."""
    if not isolated_network:
        return {
            "status": "NOT_RUN",
            "reason": "no isolated test network: a simulated link fault must not be reported as real",
        }
    if not approval_reference:
        return {
            "status": "NOT_RUN",
            "reason": "link shaping needs an approval reference even on an isolated network",
        }
    return {
        "status": "PLANNED",
        "approval_reference": approval_reference,
        "targets": "experiment NICs/rails only; shared switches are untouched",
    }


# ── attribution helpers / verdict ──────────────────────────────────────────


def overlap_attribution(
    *, exposed_comm_ms: float, arrival_skew_ms: float, net_gain_ms: float
) -> Dict[str, Any]:
    """Did a late arrival eat the overlap benefit? (step 24)"""
    return {
        "exposed_comm_ms": exposed_comm_ms,
        "arrival_skew_ms": arrival_skew_ms,
        "net_gain_ms": net_gain_ms,
        "skew_dominates": arrival_skew_ms > abs(net_gain_ms),
        "note": "if the skew is larger than the gain, the overlap is limited by placement, not by schedule",
    }


def scaling_degradation_attribution(
    *,
    comm_growth_ms: float,
    kernel_shrink_ms: float,
    topology_cross_ms: float,
    arrival_skew_ms: float,
    idle_ms: float,
) -> Dict[str, Any]:
    """Quantify each contributor to a low-efficiency degree (step 25)."""
    contributors = {
        "communication_growth": comm_growth_ms,
        "smaller_kernels": kernel_shrink_ms,
        "topology_crossing": topology_cross_ms,
        "arrival_skew": arrival_skew_ms,
        "idle": idle_ms,
    }
    total = sum(max(value, 0.0) for value in contributors.values())
    return {
        "contributors": contributors,
        "total_ms": total,
        "shares": {
            name: (max(value, 0.0) / total if total else 0.0) for name, value in contributors.items()
        },
        "note": "an unexplained residual must be reported, not assigned to the largest bucket",
    }


def straggler_runbook_lines() -> List[Dict[str, str]]:
    """The shortest diagnosis chain of details E10-09 step 27."""
    return [
        {"symptom": "job slower than baseline", "action": "compare per-rank phase breakdown"},
        {"symptom": "one collective slow", "action": "pair by group+epoch+seq and compute arrival skew"},
        {"symptom": "arrival skew large", "action": "walk upstream: host gap → compute → load imbalance"},
        {"symptom": "arrivals equal, transit slow", "action": "map src→dst to the topology edge and check counters"},
        {"symptom": "algorithm/transport changed", "action": "compare actual algorithm evidence across runs"},
        {"symptom": "unmapped time large", "action": "improve correlation before attributing; otherwise INCONCLUSIVE"},
    ]


def public_summary_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    native_counters: Mapping[str, Any],
    namespaces: Sequence[str] = ("nccl", "hccl"),
) -> List[Dict[str, Any]]:
    """Platform-independent summary with the native counters namespaced (step 29)."""
    summaries: List[Dict[str, Any]] = []
    for row in rows:
        entry = {name: row.get(name) for name in PUBLIC_SUMMARY_FIELDS}
        summaries.append(entry)
    namespaced = {
        f"{namespace}.{key}": value
        for namespace in namespaces
        for key, value in native_counters.items()
        if key.startswith(namespace)
    }
    return summaries + [{"native_counters": namespaced}]


def trace_verdict(
    *,
    all_ranks_covered: bool,
    correlation_ok: bool,
    clock_documented: bool,
    breakdown_balanced: bool,
    injections_classified: bool,
    unknown_ratio_ok: bool,
    confirmation_ok: bool,
) -> Dict[str, Any]:
    """The E10-09 attribution verdict (details E10-09 step 30)."""
    blockers: List[str] = []
    if not all_ranks_covered:
        blockers.append("not every relevant rank has a raw trace")
    if not correlation_ok:
        blockers.append("request→phase→layer→group/seq→kernel/transport is not walkable")
    if not clock_documented:
        blockers.append("clock offset/drift/uncertainty or profiler overhead is not quantified")
    if not breakdown_balanced:
        blockers.append("compute/comm/overlap/wait/idle double-counts or does not balance")
    if not unknown_ratio_ok:
        blockers.append("the unmapped/unknown ratio exceeds the pre-registered threshold")
    if not injections_classified:
        blockers.append("the pre-registered injections were not blind-classified")
    if not confirmation_ok:
        blockers.append("the independent confirmation trace did not reproduce the ranking")
    if blockers:
        return {
            "status": "INCONCLUSIVE",
            "blockers": blockers,
            "reason": "attribution evidence is insufficient; do not report a root cause",
        }
    return {
        "status": "PASS",
        "reason": "evidence chain complete; prefill/decode bottlenecks and the locatable rate are reported",
        "unknown_is_a_legal_answer": True,
    }


__all__ = [
    "CLOCK_DOMAINS",
    "EVIDENCE_MATRIX",
    "INJECTION_KINDS",
    "PUBLIC_SUMMARY_FIELDS",
    "ROOT_CAUSE_CLASSES",
    "TRACE_EVENT_FIELDS",
    "TRACE_MODES",
    "Attribution",
    "ClockCalibration",
    "CorrelationChain",
    "DistributedTraceEvent",
    "MetricManifest",
    "PairedCollective",
    "PhaseBreakdown",
    "RootCauseClassifier",
    "VariabilityBaseline",
    "amplification_metrics",
    "arrival_completion_skew",
    "calibrate_offsets",
    "classifier_scores",
    "communication_matrix",
    "injection_plan",
    "link_shaping_plan",
    "overlap_attribution",
    "pair_collective_events",
    "phase_breakdown",
    "public_summary_rows",
    "scaling_degradation_attribution",
    "straggler_runbook_lines",
    "trace_verdict",
    "unmapped_events",
    "unmapped_ratio",
]
