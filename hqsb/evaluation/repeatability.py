"""E12-04: repetition hierarchy, drift/noise control and exclusion ledger.

"Run it a few more times and take the best" is the failure mode this module
exists to prevent:

* the *unit of inference* is the process/block/day, never the token or request
  inside one run;
* thresholds and outlier rules are inputs (frozen before the data) — the code
  cannot derive them from the results it is judging;
* an exclusion decision must cite a pre-registered rule, a performance-blind
  basis and its effect on the estimate, and the raw run stays in the ledger;
* instability is a first-class verdict: an unstable cell keeps its interval and
  blocks high-confidence recommendations instead of being retried until it looks
  stable.

Statistics are pure Python and deterministic (seeded bootstrap), so the module
stays importable on the CPU-minimal installation.

Nothing here runs a benchmark.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.evaluation.identity import stable_id
from hqsb.evaluation.records import TABLE_SCHEMAS

EXPERIMENT_ID = "E12-04"
TITLE = "多进程、多时段、多日重复性、漂移与噪声控制"
CLAIM_BOUNDARY = (
    "本实验通过证明测量不确定性得到诚实量化，"
    "不保证所有平台都达到相同稳定水平，也不能用 best-of-N 替代稳定性。"
)

STABILITY_DIMENSIONS: Tuple[str, ...] = (
    "within_run",
    "between_run",
    "between_process",
    "between_block",
    "between_day",
    "between_device",
)

#: Health/anomaly labels (§6).  Multi-select; ``UNKNOWN_ANOMALY`` is not healthy.
HEALTH_LABELS: Tuple[str, ...] = (
    "HEALTHY_INCLUDED",
    "HEALTHY_BUT_HIGH_VARIANCE",
    "THERMAL_THROTTLE",
    "POWER_LIMIT_THROTTLE",
    "CLOCK_DEVIATION",
    "CPU_CONTENTION",
    "MEMORY_PRESSURE",
    "SWAP_ACTIVITY",
    "IO_CONTENTION",
    "NEIGHBOR_PROCESS",
    "ECC_RAS_OR_LINK_EVENT",
    "OOM_OR_RECOVERY",
    "SERVICE_BACKLOG_CONTAMINATION",
    "LOADGEN_BOTTLENECK",
    "TELEMETRY_GAP",
    "UNKNOWN_ANOMALY",
)

#: Labels that make a run non-representative of production behaviour.
CONTAMINATION_LABELS: Tuple[str, ...] = tuple(
    label for label in HEALTH_LABELS if label not in ("HEALTHY_INCLUDED", "HEALTHY_BUT_HIGH_VARIANCE")
)

EXCLUSION_DECISIONS: Tuple[str, ...] = ("include", "exclude", "quarantine")
STABILITY_VERDICTS: Tuple[str, ...] = ("stable", "unstable", "conditional", "insufficient")


def is_healthy(labels: Sequence[str]) -> bool:
    """A run is healthy only when no contamination label applies."""
    return not any(label in CONTAMINATION_LABELS for label in labels)


def combine_labels(*groups: Sequence[str]) -> Tuple[str, ...]:
    """Union of labels, order-preserving and de-duplicated."""
    seen: List[str] = []
    for group in groups:
        for label in group:
            if label not in HEALTH_LABELS:
                raise ConfigError(f"unknown health label {label!r}")
            if label not in seen:
                seen.append(label)
    if not seen:
        return ("HEALTHY_INCLUDED",)
    if len(seen) > 1 and "HEALTHY_INCLUDED" in seen:
        seen.remove("HEALTHY_INCLUDED")
    return tuple(seen)


# ── replication design ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReplicationLevel:
    level: str
    unit_name: str
    min_count: int
    role: str

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.level not in STABILITY_DIMENSIONS:
            problems.append(f"unknown replication level {self.level!r}")
        if self.min_count < 1:
            problems.append(f"replication level {self.level!r} needs at least one unit")
        return problems


@dataclass
class ReplicationHierarchy:
    """Which units are nested, how many are required and what is inferred."""

    levels: Tuple[ReplicationLevel, ...]
    primary_inference_unit: str
    notes: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for level in self.levels:
            problems.extend(level.validate())
        present = {level.level for level in self.levels}
        for required in ("between_process", "between_block", "between_day"):
            if required not in present:
                problems.append(f"the hierarchy must cover {required!r} (process/block/day)")
        process = next((level for level in self.levels if level.level == "between_process"), None)
        if process is not None and process.min_count < 3:
            problems.append("at least three independent processes are required by the handbook")
        if self.primary_inference_unit in ("token", "request", "iteration"):
            problems.append(
                "tokens/requests inside one run are not independent hardware repetitions; "
                "the primary inference unit must be run/process/block/day"
            )
        if self.primary_inference_unit not in present and self.primary_inference_unit not in ("run", "process", "block", "day"):
            problems.append(f"unknown primary inference unit {self.primary_inference_unit!r}")
        return problems

    def as_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "level": level.level,
                "unit_name": level.unit_name,
                "min_count": level.min_count,
                "role": level.role,
                "primary_inference_unit": level.level == self.primary_inference_unit,
            }
            for level in self.levels
        ]


def default_hierarchy() -> ReplicationHierarchy:
    return ReplicationHierarchy(
        levels=(
            ReplicationLevel("within_run", "iteration/request", 100, "distribution shape only"),
            ReplicationLevel("between_run", "run", 3, "within-process repeatability"),
            ReplicationLevel("between_process", "process", 3, "software-instance repeatability"),
            ReplicationLevel("between_block", "time block", 2, "environment drift"),
            ReplicationLevel("between_day", "day", 2, "long-term repeatability"),
            ReplicationLevel("between_device", "device", 1, "instance extrapolation risk"),
        ),
        primary_inference_unit="process",
        notes="block/day/process are the formal levels; within-run samples only describe the local state",
    )


# ── preregistered thresholds and rules ────────────────────────────────────


@dataclass(frozen=True)
class StabilityThresholds:
    """Preregistered two-condition thresholds: statistical precision + drift."""

    metric_name: str
    layer: str
    precision_half_width_rel: float
    practical_cv: float
    max_drift: float
    anomaly_rate_max: float
    practical_effect: float
    pilot_source: str
    frozen_at: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.metric_name:
            problems.append("thresholds need a metric name")
        for name in (
            "precision_half_width_rel",
            "practical_cv",
            "max_drift",
            "anomaly_rate_max",
            "practical_effect",
        ):
            value = getattr(self, name)
            if value <= 0:
                problems.append(f"{name} must be positive (a zero threshold would always fail)")
        if not self.pilot_source:
            problems.append(
                "thresholds must cite their pilot basis; deriving them from the results they judge "
                "is not allowed"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "layer": self.layer,
            "precision_half_width_rel": self.precision_half_width_rel,
            "practical_cv": self.practical_cv,
            "max_drift": self.max_drift,
            "anomaly_rate_max": self.anomaly_rate_max,
            "practical_effect": self.practical_effect,
            "pilot_source": self.pilot_source,
            "frozen_at": self.frozen_at,
        }


#: Example metric/layer thresholds.  Values are *documented defaults for the
#: template*, not measurements; a real campaign replaces them before running.
THRESHOLDS_BY_METRIC: Mapping[Tuple[str, str], StabilityThresholds] = {
    ("operator_latency", "operator"): StabilityThresholds(
        metric_name="operator_latency",
        layer="operator",
        precision_half_width_rel=0.05,
        practical_cv=0.05,
        max_drift=0.10,
        anomaly_rate_max=0.05,
        practical_effect=0.03,
        pilot_source="pilot campaign (variance estimation only)",
    ),
    ("core_tpot", "model_core"): StabilityThresholds(
        metric_name="core_tpot",
        layer="model_core",
        precision_half_width_rel=0.05,
        practical_cv=0.08,
        max_drift=0.15,
        anomaly_rate_max=0.10,
        practical_effect=0.05,
        pilot_source="pilot campaign (variance estimation only)",
    ),
    ("token_goodput", "service"): StabilityThresholds(
        metric_name="token_goodput",
        layer="service",
        precision_half_width_rel=0.10,
        practical_cv=0.15,
        max_drift=0.20,
        anomaly_rate_max=0.15,
        practical_effect=0.08,
        pilot_source="pilot campaign (variance estimation only)",
    ),
    ("device_seconds", "distributed"): StabilityThresholds(
        metric_name="device_seconds",
        layer="distributed",
        precision_half_width_rel=0.10,
        practical_cv=0.15,
        max_drift=0.20,
        anomaly_rate_max=0.15,
        practical_effect=0.08,
        pilot_source="pilot campaign (variance estimation only)",
    ),
}


def threshold_for(metric_name: str, layer: str) -> StabilityThresholds:
    try:
        return THRESHOLDS_BY_METRIC[(metric_name, layer)]
    except KeyError as exc:
        raise ConfigError(
            f"no preregistered threshold for ({metric_name!r}, {layer!r}): thresholds must be "
            "frozen before the run, never derived from it"
        ) from exc


@dataclass(frozen=True)
class AnomalyRule:
    """One preregistered anomaly detector (performance-blind by construction)."""

    rule_id: str
    rule_version: str
    metric: str
    comparator: str
    threshold: float
    window: str
    label: str
    action: str = "mark"

    SYSTEM_METRICS: Tuple[str, ...] = (
        "temperature_c",
        "clock_mhz",
        "power_w",
        "throttle_flags",
        "cpu_load",
        "swap_used_bytes",
        "ram_used_ratio",
        "io_wait",
        "ecc_errors",
        "neighbor_processes",
        "telemetry_gap_s",
        "oom_events",
        "backlog_requests",
    )

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.metric not in self.SYSTEM_METRICS:
            problems.append(
                f"an anomaly rule may only watch system/telemetry metrics, not {self.metric!r}: "
                "rules must be performance-blind"
            )
        if self.label not in HEALTH_LABELS:
            problems.append(f"unknown health label {self.label!r}")
        if self.comparator not in (">", ">=", "<", "<=", "==", "!="):
            problems.append(f"unknown comparator {self.comparator!r}")
        if self.action not in ("mark", "quarantine", "wait"):
            problems.append(f"unknown rule action {self.action!r}")
        return problems


@dataclass
class AnomalyRuleSet:
    """Frozen rule set; evaluation is a pure function of the telemetry rows."""

    rules: Tuple[AnomalyRule, ...]
    version: str = "v1"

    def validate(self) -> List[str]:
        problems: List[str] = []
        seen: set = set()
        for rule in self.rules:
            problems.extend(f"{rule.rule_id}: {item}" for item in rule.validate())
            if rule.rule_id in seen:
                problems.append(f"duplicate anomaly rule id {rule.rule_id!r}")
            seen.add(rule.rule_id)
        if not self.rules:
            problems.append("an empty rule set cannot detect anomalies")
        return problems

    def evaluate(self, run_rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
        if self.validate():
            raise ConfigError("invalid anomaly rule set: " + "; ".join(self.validate()))
        events: List[Dict[str, Any]] = []
        for row in run_rows:
            for rule in self.rules:
                if rule.metric not in row:
                    continue
                value = row.get(rule.metric)
                if not isinstance(value, (int, float)):
                    continue
                fired = _compare(float(value), rule.comparator, rule.threshold)
                if fired:
                    events.append(
                        {
                            "event_id": stable_id(
                                "anom", {"run": row.get("run_id", ""), "rule": rule.rule_id}
                            ),
                            "run_id": str(row.get("run_id", "")),
                            "label": rule.label,
                            "rule_id": rule.rule_id,
                            "rule_version": f"{rule.rule_version}/{self.version}",
                            "threshold": rule.threshold,
                            "observed_value": float(value),
                            "action": rule.action,
                        }
                    )
        return tuple(events)


def _compare(value: float, comparator: str, threshold: float) -> bool:
    if comparator == ">":
        return value > threshold
    if comparator == ">=":
        return value >= threshold
    if comparator == "<":
        return value < threshold
    if comparator == "<=":
        return value <= threshold
    if comparator == "==":
        return value == threshold
    if comparator == "!=":
        return value != threshold
    raise ConfigError(f"unknown comparator {comparator!r}")


def default_rule_set() -> AnomalyRuleSet:
    """The reference rule set; a campaign freezes its own copy before running."""
    rules = (
        AnomalyRule("temp_high", "1.0.0", "temperature_c", ">", 85.0, "run", "THERMAL_THROTTLE", "mark"),
        AnomalyRule("power_capped", "1.0.0", "throttle_flags", ">", 0, "run", "POWER_LIMIT_THROTTLE", "mark"),
        AnomalyRule("clock_drop", "1.0.0", "clock_mhz", "<", 1.0, "run", "CLOCK_DEVIATION", "mark"),
        AnomalyRule("cpu_busy", "1.0.0", "cpu_load", ">", 1.5, "run", "CPU_CONTENTION", "mark"),
        AnomalyRule("ram_pressure", "1.0.0", "ram_used_ratio", ">", 0.9, "run", "MEMORY_PRESSURE", "mark"),
        AnomalyRule("swap_used", "1.0.0", "swap_used_bytes", ">", 0, "run", "SWAP_ACTIVITY", "mark"),
        AnomalyRule("io_wait", "1.0.0", "io_wait", ">", 0.2, "run", "IO_CONTENTION", "mark"),
        AnomalyRule("ecc_event", "1.0.0", "ecc_errors", ">", 0, "run", "ECC_RAS_OR_LINK_EVENT", "quarantine"),
        AnomalyRule("oom", "1.0.0", "oom_events", ">", 0, "run", "OOM_OR_RECOVERY", "quarantine"),
        AnomalyRule("telemetry_gap", "1.0.0", "telemetry_gap_s", ">", 1.0, "run", "TELEMETRY_GAP", "mark"),
        AnomalyRule("backlog", "1.0.0", "backlog_requests", ">", 0, "run", "SERVICE_BACKLOG_CONTAMINATION", "mark"),
    )
    return AnomalyRuleSet(rules=rules, version="1.0.0")


# ── schedule ──────────────────────────────────────────────────────────────


@dataclass
class BlockSchedule:
    """Balanced run order: candidates are interleaved, not run in one block."""

    schedule_version: str
    blocks: Tuple[Mapping[str, Any], ...]

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.blocks:
            problems.append("a schedule needs at least one block")
        for block in self.blocks:
            for name in ("block_id", "day", "time_block", "candidate_order", "seed"):
                if block.get(name) in (None, "", ()):
                    problems.append(f"block {block.get('block_id', '?')} is missing {name!r}")
        return problems

    def as_rows(self) -> List[Dict[str, Any]]:
        return [dict(block) for block in self.blocks]


def balanced_schedule(
    candidates: Sequence[str],
    *,
    blocks: int,
    days: int = 1,
    seed: int = 0,
    design: str = "latin_square",
) -> BlockSchedule:
    """Interleave candidates across blocks with a reproducible order."""
    if len(candidates) < 2:
        raise ConfigError("a balanced schedule needs at least two candidates")
    if blocks < len(candidates):
        raise ConfigError(
            "every candidate must appear in every block: blocks must be >= candidate count"
        )
    if design not in ("latin_square", "randomized"):
        raise ConfigError(f"unknown schedule design {design!r}")
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for day in range(1, days + 1):
        for block in range(1, blocks + 1):
            order = list(candidates)
            if design == "randomized":
                rng.shuffle(order)
            else:
                offset = (block - 1) % len(order)
                order = order[offset:] + order[:offset]
            rows.append(
                {
                    "block_id": f"d{day}_b{block}",
                    "day": day,
                    "time_block": block,
                    "candidate_order": order,
                    "seed": seed,
                    "host_id": "",
                    "device_ids": (),
                    "schedule_version": f"{design}:{seed}:{blocks}x{days}",
                }
            )
    return BlockSchedule(schedule_version=f"{design}:{seed}:{blocks}x{days}", blocks=tuple(rows))


def schedule_balance(schedule: BlockSchedule) -> Dict[str, Any]:
    """Position bias check: each candidate should occupy each rank equally often."""
    positions: Dict[str, Dict[int, int]] = {}
    for block in schedule.blocks:
        for index, candidate in enumerate(block.get("candidate_order", ())):
            positions.setdefault(str(candidate), {}).setdefault(index, 0)
            positions[str(candidate)][index] += 1
    balanced = all(
        (max(counts.values()) - min(counts.values())) <= 1 for counts in positions.values()
    ) if positions else False
    return {
        "candidates": sorted(positions),
        "positions": {name: dict(sorted(counts.items())) for name, counts in sorted(positions.items())},
        "balanced": balanced,
    }


def assert_schedule_immutable(schedule: BlockSchedule, other: BlockSchedule) -> None:
    """Re-ordering after seeing results invalidates the campaign."""
    if schedule.blocks != other.blocks:
        raise ConfigError(
            "the schedule changed after it was frozen: candidate order must not be re-drawn "
            "in response to results"
        )


# ── exclusion ledger ──────────────────────────────────────────────────────


@dataclass
class ExclusionEntry:
    run_id: str
    rule_id: str
    rule_version: str
    detected_event: str
    threshold: float
    observed_value: Optional[float]
    pre_registered: bool
    decision: str
    performance_blind_basis: str
    reviewer: str = ""
    artifact_refs: Tuple[str, ...] = ()
    effect_on_estimate: str = "unquantified"
    sensitivity_result: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.decision not in EXCLUSION_DECISIONS:
            problems.append(f"unknown exclusion decision {self.decision!r}")
        if not self.rule_id or not self.rule_version:
            problems.append("an exclusion must cite the rule id and version")
        if not self.performance_blind_basis:
            problems.append(
                "an exclusion must state its performance-blind basis (a slow run is not a reason)"
            )
        if not self.artifact_refs:
            problems.append("an exclusion must reference the raw artefacts it was based on")
        if self.decision != "include" and self.effect_on_estimate == "unquantified":
            problems.append("excluding a run must quantify its effect on the estimate")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "detected_event": self.detected_event,
            "threshold": self.threshold,
            "observed_value": self.observed_value,
            "pre_registered": self.pre_registered,
            "decision": self.decision,
            "performance_blind_basis": self.performance_blind_basis,
            "reviewer": self.reviewer,
            "artifact_refs": list(self.artifact_refs),
            "effect_on_estimate": self.effect_on_estimate,
            "sensitivity_result": self.sensitivity_result,
        }


class ExclusionLedger:
    def __init__(self) -> None:
        self._rows: List[ExclusionEntry] = []

    def record(self, entry: ExclusionEntry) -> None:
        problems = entry.validate()
        if problems:
            raise ConfigError("invalid exclusion entry: " + "; ".join(problems))
        self._rows.append(entry)

    def rows(self) -> Tuple[ExclusionEntry, ...]:
        return tuple(self._rows)

    def included(self) -> Tuple[str, ...]:
        return tuple(sorted(row.run_id for row in self._rows if row.decision == "include"))

    def excluded(self) -> Tuple[str, ...]:
        return tuple(sorted(row.run_id for row in self._rows if row.decision == "exclude"))

    def quarantined(self) -> Tuple[str, ...]:
        return tuple(sorted(row.run_id for row in self._rows if row.decision == "quarantine"))

    def post_hoc_entries(self) -> Tuple[ExclusionEntry, ...]:
        return tuple(row for row in self._rows if not row.pre_registered)

    def confirmatory_rows(self) -> Tuple[ExclusionEntry, ...]:
        """Post-hoc exclusions may not enter the confirmatory main result."""
        return tuple(row for row in self._rows if row.pre_registered or row.decision == "include")


def apply_inclusion_policy(
    run_labels: Mapping[str, Sequence[str]],
    *,
    ledger: Optional[ExclusionLedger] = None,
) -> Dict[str, Any]:
    """Decide include/exclude from *system events*, then report the ledger."""
    rows: List[Dict[str, Any]] = []
    for run_id, labels in sorted(run_labels.items()):
        healthy = is_healthy(labels)
        decision = "include" if healthy else "exclude"
        rows.append(
            {
                "run_id": run_id,
                "labels": list(labels),
                "decision": decision,
                "rule_id": "health_labels",
                "pre_registered": True,
                "performance_blind_basis": "health labels come from telemetry, not from the metric",
            }
        )
    if ledger is not None:
        for entry in ledger.rows():
            rows.append(dict(entry.as_dict(), decision=entry.decision, labels=[]))
    return {
        "rows": rows,
        "included": sorted(row["run_id"] for row in rows if row["decision"] == "include"),
        "excluded": sorted(row["run_id"] for row in rows if row["decision"] == "exclude"),
        "quarantined": sorted(row["run_id"] for row in rows if row["decision"] == "quarantine"),
        "note": "raw runs stay in the ledger; only the analysis view is filtered",
    }


# ── statistics (pure Python, deterministic) ───────────────────────────────


def median(values: Sequence[float]) -> float:
    if not values:
        raise ConfigError("median needs at least one value")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ConfigError("quantile needs at least one value")
    if not 0.0 < q <= 1.0:
        raise ConfigError(f"quantile must be inside (0, 1], got {q}")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def cv(values: Sequence[float]) -> float:
    """Coefficient of variation; refuses near-zero means (use log scale instead)."""
    if len(values) < 2:
        raise ConfigError("cv needs at least two values")
    mean = sum(float(value) for value in values) / len(values)
    if abs(mean) < 1e-12:
        raise ConfigError("cv is undefined near zero: analyse log-ratios instead")
    return sample_std(values) / abs(mean)


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ConfigError("sample standard deviation needs at least two values")
    mean = sum(float(value) for value in values) / len(values)
    variance = sum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def mad(values: Sequence[float]) -> float:
    """Median absolute deviation (robust, no normal assumption)."""
    centre = median(values)
    return median([abs(float(value) - centre) for value in values])


def iqr(values: Sequence[float]) -> float:
    return quantile(values, 0.75) - quantile(values, 0.25)


def robust_cv_like(values: Sequence[float]) -> float:
    centre = median(values)
    if abs(centre) < 1e-12:
        raise ConfigError("robust CV is undefined near zero")
    return mad(values) / abs(centre)


def autocorrelation(values: Sequence[float], *, lag: int = 1) -> float:
    if lag < 1:
        raise ConfigError("lag must be >= 1")
    if len(values) <= lag + 1:
        raise ConfigError("autocorrelation needs more samples than the lag")
    ordered = [float(value) for value in values]
    mean = sum(ordered) / len(ordered)
    numerator = sum(
        (ordered[index] - mean) * (ordered[index + lag] - mean)
        for index in range(len(ordered) - lag)
    )
    denominator = sum((value - mean) ** 2 for value in ordered)
    if denominator == 0:
        raise ConfigError("autocorrelation is undefined for a constant series")
    return numerator / denominator


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: str = "median",
    B: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Seeded run-level bootstrap interval (replication unit = run)."""
    if len(values) < 2:
        raise ConfigError("a bootstrap interval needs at least two runs")
    if B < 100:
        raise ConfigError("B must be at least 100 for a usable interval")
    if not 0.0 < confidence < 1.0:
        raise ConfigError("confidence must be inside (0, 1)")
    if statistic not in ("median", "mean"):
        raise ConfigError(f"unknown statistic {statistic!r}")
    rng = random.Random(seed)
    ordered = [float(value) for value in values]
    draws: List[float] = []
    for _ in range(B):
        sample = [ordered[rng.randrange(len(ordered))] for _ in ordered]
        draws.append(median(sample) if statistic == "median" else sum(sample) / len(sample))
    alpha = (1.0 - confidence) / 2.0
    return {
        "statistic": statistic,
        "value": median(ordered) if statistic == "median" else sum(ordered) / len(ordered),
        "interval_low": quantile(draws, alpha),
        "interval_high": quantile(draws, 1.0 - alpha),
        "confidence_level": confidence,
        "method": "run_level_bootstrap",
        "B": B,
        "seed": seed,
        "replication_unit": "run",
    }


def variance_components(rows: Sequence[Mapping[str, Any]], *, level: str) -> Tuple[Dict[str, Any], ...]:
    """Grouped (between/within) variance summary; no normal-model assumption."""
    if level not in ("block", "process", "day", "device"):
        raise ConfigError(f"unknown variance level {level!r}")
    groups: Dict[str, List[float]] = {}
    for row in rows:
        value = row.get("value")
        if not isinstance(value, (int, float)):
            continue
        groups.setdefault(str(row.get(level, row.get("run_id", ""))), []).append(float(value))
    if len(groups) < 2:
        return (
            {
                "component": level,
                "estimate": 0.0,
                "method": "grouped_summary",
                "block_count": len(groups),
                "detail": "fewer than two groups: no between-group component estimable",
            },
        )
    group_means = [sum(values) / len(values) for values in groups.values()]
    pooled_within = 0.0
    for values in groups.values():
        if len(values) > 1:
            pooled_within += sum((value - sum(values) / len(values)) ** 2 for value in values)
    within_df = sum(max(0, len(values) - 1) for values in groups.values())
    within = pooled_within / within_df if within_df else 0.0
    between = sample_std(group_means) ** 2 if len(group_means) > 1 else 0.0
    return (
        {
            "component": f"between_{level}",
            "estimate": between,
            "method": "grouped_summary",
            "block_count": len(groups),
            "detail": "variance of group means",
        },
        {
            "component": f"within_{level}",
            "estimate": within,
            "method": "grouped_summary",
            "block_count": len(groups),
            "detail": "pooled within-group variance",
        },
    )


def drift_slope(
    rows: Sequence[Mapping[str, Any]], *, covariate: str = "run_order", rule_version: str = "v1"
) -> Dict[str, Any]:
    """Least-squares drift of the metric against a covariate (diagnostic)."""
    pairs = [
        (float(row[covariate]), float(row["value"]))
        for row in rows
        if isinstance(row.get("value"), (int, float)) and isinstance(row.get(covariate), (int, float))
    ]
    if len(pairs) < 3:
        raise ConfigError("drift estimation needs at least three (covariate, value) pairs")
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ConfigError("the covariate does not vary: drift is not identifiable")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / denominator
    return {
        "metric_name": str(rows[0].get("metric_name", "")),
        "covariate": covariate,
        "slope": slope,
        "slope_rel": slope / abs(mean_y) if mean_y else 0.0,
        "rule_version": rule_version,
        "samples": len(pairs),
    }


def change_point(values: Sequence[float]) -> Dict[str, Any]:
    """Largest mean shift between split points (diagnostic only, versioned)."""
    if len(values) < 4:
        raise ConfigError("change-point diagnostics need at least four values")
    ordered = [float(value) for value in values]
    best_index = 0
    best_gap = -1.0
    for index in range(1, len(ordered)):
        left = ordered[:index]
        right = ordered[index:]
        gap = abs(sum(right) / len(right) - sum(left) / len(left))
        if gap > best_gap:
            best_gap = gap
            best_index = index
    return {
        "index": best_index,
        "gap": best_gap,
        "note": "diagnostic only: a change point does not explain itself",
    }


def paired_log_ratio(
    a_rows: Sequence[Mapping[str, Any]],
    b_rows: Sequence[Mapping[str, Any]],
    *,
    key: str = "block_id",
) -> Dict[str, Any]:
    """Block-aware paired comparison on the log scale (latency ratios)."""
    a_by_key = {str(row.get(key)): float(row["value"]) for row in a_rows if row.get(key)}
    b_by_key = {str(row.get(key)): float(row["value"]) for row in b_rows if row.get(key)}
    shared = sorted(set(a_by_key) & set(b_by_key))
    if not shared:
        raise ConfigError(
            "no shared blocks: a cross-block comparison would be observational, not paired"
        )
    for block in shared:
        if a_by_key[block] <= 0 or b_by_key[block] <= 0:
            raise ConfigError("log ratios need strictly positive values")
    ratios = [math.log(a_by_key[block] / b_by_key[block]) for block in shared]
    return {
        "blocks": shared,
        "paired_log_ratio_median": median(ratios),
        "ratio": math.exp(median(ratios)),
        "samples": len(ratios),
    }


def frontier_membership_probability(
    draws: Sequence[Sequence[Mapping[str, Any]]],
    *,
    objectives: Sequence[Tuple[str, str]],
    seed: int = 0,
) -> Dict[str, Any]:
    """Probability that each candidate is on the frontier across resamples."""
    if not draws:
        raise ConfigError("membership needs at least one bootstrap draw")
    for name, direction in objectives:
        if direction not in ("maximize", "minimize"):
            raise ConfigError(f"unknown objective direction {direction!r} for {name}")
    counts: Dict[str, int] = {}
    ranks: Dict[str, List[int]] = {}
    for draw in draws:
        members = _frontier([dict(row) for row in draw], objectives)
        for row in draw:
            candidate = str(row.get("candidate_id", ""))
            counts.setdefault(candidate, 0)
            ranks.setdefault(candidate, [])
        for candidate in members:
            counts[candidate] = counts.get(candidate, 0) + 1
    return {
        "draws": len(draws),
        "seed": seed,
        "membership_probability": {
            name: counts.get(name, 0) / len(draws) for name in sorted(counts)
        },
        "objectives": [list(item) for item in objectives],
        "kind": "algorithm_diagnostic",
    }


def _frontier(rows: List[Dict[str, Any]], objectives: Sequence[Tuple[str, str]]) -> List[str]:
    def transformed(row: Mapping[str, Any], name: str, direction: str) -> float:
        value = float(row[name])
        return value if direction == "maximize" else -value

    members: List[str] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            non_worse = all(
                transformed(other, name, direction) >= transformed(candidate, name, direction)
                for name, direction in objectives
            )
            strictly_better = any(
                transformed(other, name, direction) > transformed(candidate, name, direction)
                for name, direction in objectives
            )
            if non_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            members.append(str(candidate.get("candidate_id", "")))
    return members


def sensitivity_report(
    rows: Sequence[Mapping[str, Any]], *, statistic: str = "median"
) -> Tuple[Dict[str, Any], ...]:
    """all-runs / healthy-only / robust / leave-one-block-out views together."""
    values = [float(row["value"]) for row in rows if isinstance(row.get("value"), (int, float))]
    if not values:
        raise ConfigError("sensitivity needs numeric values")
    healthy = [
        float(row["value"])
        for row in rows
        if isinstance(row.get("value"), (int, float)) and is_healthy(row.get("labels", ()) or ("HEALTHY_INCLUDED",))
    ]
    views: List[Dict[str, Any]] = [
        {"scenario": "all_runs", "estimate": median(values) if statistic == "median" else sum(values) / len(values), "runs": len(values)},
        {
            "scenario": "healthy_only",
            "estimate": (median(healthy) if statistic == "median" else sum(healthy) / len(healthy)) if healthy else None,
            "runs": len(healthy),
        },
        {"scenario": "robust_cv_like", "estimate": robust_cv_like(values), "runs": len(values)},
    ]
    blocks = sorted({str(row.get("block_id", "")) for row in rows if row.get("block_id")})
    for block in blocks:
        kept = [
            float(row["value"])
            for row in rows
            if isinstance(row.get("value"), (int, float)) and str(row.get("block_id", "")) != block
        ]
        if not kept:
            continue
        views.append(
            {
                "scenario": f"leave_out_block:{block}",
                "estimate": median(kept) if statistic == "median" else sum(kept) / len(kept),
                "runs": len(kept),
            }
        )
    estimates = [view["estimate"] for view in views if view["estimate"] is not None]
    spread = (max(estimates) - min(estimates)) / abs(median(estimates)) if estimates and median(estimates) else 0.0
    return tuple(views + [{"scenario": "dependence_on_exclusion", "estimate": spread, "runs": len(values)}])


@dataclass
class PilotConfirmationSplit:
    pilot_run_ids: Tuple[str, ...]
    confirmation_run_ids: Tuple[str, ...]

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.pilot_run_ids or not self.confirmation_run_ids:
            problems.append("both a pilot and a confirmation set are required")
        overlap = sorted(set(self.pilot_run_ids) & set(self.confirmation_run_ids))
        if overlap:
            problems.append(
                f"pilot and confirmation share runs {overlap}: thresholds must not be tuned on the "
                "data that accepts them"
            )
        return problems


def confirmation_gate(
    *,
    thresholds: StabilityThresholds,
    confirmation_values: Sequence[float],
    protocol_deviation: str = "",
) -> Dict[str, Any]:
    """Confirm with the frozen thresholds; a deviation is recorded, not hidden."""
    interval = bootstrap_ci(confirmation_values, statistic="median")
    half_width_rel = (interval["interval_high"] - interval["interval_low"]) / interval["value"]
    passes = half_width_rel <= thresholds.precision_half_width_rel
    return {
        "metric_name": thresholds.metric_name,
        "interval": interval,
        "half_width_rel": half_width_rel,
        "threshold": thresholds.precision_half_width_rel,
        "passes": passes,
        "protocol_deviation": protocol_deviation,
        "threshold_revised": False,
    }


def stability_verdicts(
    cell_rows: Sequence[Mapping[str, Any]],
    *,
    thresholds: StabilityThresholds,
    anomaly_rate: float = 0.0,
) -> Tuple[Dict[str, Any], ...]:
    """Per-cell verdict: stable / unstable / conditional / insufficient."""
    buckets: Dict[str, List[Mapping[str, Any]]] = {}
    for row in cell_rows:
        buckets.setdefault(str(row.get("cell_id", "")), []).append(row)
    verdicts: List[Dict[str, Any]] = []
    for cell_id, rows in sorted(buckets.items()):
        values = [float(row["value"]) for row in rows if isinstance(row.get("value"), (int, float))]
        if len(values) < 3:
            verdicts.append(
                {
                    "cell_id": cell_id,
                    "metric_name": thresholds.metric_name,
                    "stability_verdict": "insufficient",
                    "interval_low": None,
                    "interval_high": None,
                    "dominant_variation_source": "not_estimable",
                    "anomaly_rate": anomaly_rate,
                    "evidence_level": "L1",
                    "remeasure_condition": "run at least three independent processes",
                }
            )
            continue
        try:
            relative_cv = cv(values)
        except ConfigError:
            relative_cv = float("inf")
        interval = bootstrap_ci(values, statistic="median")
        half_width_rel = (interval["interval_high"] - interval["interval_low"]) / interval["value"]
        unstable_reasons = []
        if relative_cv > thresholds.practical_cv:
            unstable_reasons.append(f"cv {relative_cv:.3f} > {thresholds.practical_cv}")
        if half_width_rel > thresholds.precision_half_width_rel:
            unstable_reasons.append(
                f"interval half-width {half_width_rel:.3f} > {thresholds.precision_half_width_rel}"
            )
        if anomaly_rate > thresholds.anomaly_rate_max:
            unstable_reasons.append(f"anomaly rate {anomaly_rate} > {thresholds.anomaly_rate_max}")
        if not unstable_reasons:
            verdict = "stable"
        elif anomaly_rate > thresholds.anomaly_rate_max and relative_cv <= thresholds.practical_cv:
            verdict = "conditional"
        else:
            verdict = "unstable"
        verdicts.append(
            {
                "cell_id": cell_id,
                "metric_name": thresholds.metric_name,
                "stability_verdict": verdict,
                "interval_low": interval["interval_low"],
                "interval_high": interval["interval_high"],
                "dominant_variation_source": _dominant_source(rows),
                "anomaly_rate": anomaly_rate,
                "evidence_level": "L3" if verdict == "stable" else "L2",
                "remeasure_condition": "; ".join(unstable_reasons) if unstable_reasons else "",
            }
        )
    return tuple(verdicts)


def _dominant_source(rows: Sequence[Mapping[str, Any]]) -> str:
    """Cheap attribution of the largest spread among the available levels."""
    best = ("unknown", 0.0)
    for level in ("process", "block", "day", "device"):
        groups: Dict[str, List[float]] = {}
        for row in rows:
            value = row.get("value")
            if isinstance(value, (int, float)) and row.get(level) not in (None, ""):
                groups.setdefault(str(row[level]), []).append(float(value))
        if len(groups) < 2:
            continue
        means = [sum(values) / len(values) for values in groups.values()]
        spread = (max(means) - min(means)) / abs(sum(means) / len(means)) if sum(means) else 0.0
        if spread > best[1]:
            best = (level, spread)
    return best[0]


def injection_cases() -> Tuple[Dict[str, Any], ...]:
    """Pre-registered anomaly injections with their expected labels and limits."""
    cases = (
        ("thermal_soak", "THERMAL_THROTTLE", "keep within the vendor thermal limit"),
        ("power_cap", "POWER_LIMIT_THROTTLE", "only legal caps; record before/after"),
        ("neighbor_load", "CPU_CONTENTION", "within our own allocation only"),
        ("memory_pressure", "MEMORY_PRESSURE", "no uncontrollable OOM risk"),
        ("swap_pressure", "SWAP_ACTIVITY", "host memory only, reversible"),
        ("recoverable_oom", "OOM_OR_RECOVERY", "small case only, never the model"),
        ("loadgen_bottleneck", "LOADGEN_BOTTLENECK", "calibrate against an empty service"),
    )
    return tuple(
        {
            "injection_id": name,
            "kind": name,
            "expected_labels": (label,),
            "safety_limit": limit,
            "control_parameters": {},
        }
        for name, label, limit in cases
    )


def evaluate_injections(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for row in results:
        expected = set(row.get("expected_labels", ()))
        observed = set(row.get("observed_labels", ()))
        rows.append(
            {
                "injection_id": str(row.get("injection_id", "")),
                "expected_labels": sorted(expected),
                "observed_labels": sorted(observed),
                "detected": bool(expected & observed),
                "false_negative": not bool(expected & observed),
                "artifact_refs": list(row.get("artifact_refs", ())),
            }
        )
    return {
        "rows": rows,
        "cases": len(rows),
        "detected": sum(1 for row in rows if row["detected"]),
        "ok": bool(rows) and all(row["detected"] for row in rows),
    }


def table_schemas() -> Mapping[str, Tuple[str, ...]]:
    return {name: TABLE_SCHEMAS[name] for name in sorted(TABLE_SCHEMAS) if name.startswith("e12_04.")}


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only callability check; never an experiment."""
    values = [100.0, 102.0, 98.0, 101.0]
    schedule = balanced_schedule(["a", "b", "c"], blocks=3, seed=7)
    ledger = ExclusionLedger()
    ledger.record(
        ExclusionEntry(
            run_id="run-1",
            rule_id="temp_high",
            rule_version="1.0.0",
            detected_event="THERMAL_THROTTLE",
            threshold=85.0,
            observed_value=91.0,
            pre_registered=True,
            decision="exclude",
            performance_blind_basis="telemetry temperature, not the metric",
            artifact_refs=("hqsb://S12/c/raw/1",),
            effect_on_estimate="median moves by 1.5% (sensitivity reported)",
            sensitivity_result="reported",
        )
    )
    best_of_n_blocked = _expect_config_error(
        lambda: ExclusionLedger().record(
            ExclusionEntry(
                run_id="run-2",
                rule_id="looked_slow",
                rule_version="1.0.0",
                detected_event="UNKNOWN_ANOMALY",
                threshold=0.0,
                observed_value=1.0,
                pre_registered=False,
                decision="exclude",
                performance_blind_basis="",
                artifact_refs=(),
            )
        )
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "schedule_balanced": schedule_balance(schedule)["balanced"],
        "rules_valid": default_rule_set().validate() == [],
        "exclusion_requires_blind_basis": best_of_n_blocked,
        "bootstrap_interval_deterministic": bootstrap_ci(values, seed=1)["interval_low"]
        == bootstrap_ci(values, seed=1)["interval_low"],
        "cv_refuses_zero_mean": _expect_config_error(lambda: cv([0.0, 0.0])),
        "tables": len(table_schemas()),
    }


def _expect_config_error(callable_: Any) -> bool:
    try:
        callable_()
    except ConfigError:
        return True
    return False


PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "选择代表 cells", ("benchmark:plan_cells", "repeatability:StabilityThresholds")),
    (2, "定义 replication hierarchy", ("repeatability:ReplicationHierarchy", "repeatability:ReplicationLevel")),
    (3, "定义 pilot 与 confirmation", ("repeatability:PilotConfirmationSplit", "repeatability:confirmation_gate")),
    (4, "预注册稳定性阈值", ("repeatability:StabilityThresholds", "repeatability:threshold_for")),
    (5, "冻结异常规则", ("repeatability:AnomalyRule", "repeatability:AnomalyRuleSet")),
    (6, "生成平衡 schedule", ("repeatability:balanced_schedule", "repeatability:schedule_balance")),
    (7, "冻结 host/device allocation", ("repeatability:BlockSchedule", "repeatability:assert_schedule_immutable")),
    (8, "部署持续 telemetry collector", ("benchmark:run_health_check", "records:TABLE_SCHEMAS")),
    (9, "校验 clock synchronization", ("records:TABLE_SCHEMAS", "campaign:ArtifactLayout")),
    (10, "执行 block 前 health snapshot", ("repeatability:HEALTH_LABELS", "benchmark:thermal_gate")),
    (11, "从干净进程启动", ("benchmark:RunPlan", "repeatability:ReplicationHierarchy")),
    (12, "验证 artifact/backend identity", ("capability:CapabilityEvidence", "benchmark:Observation")),
    (13, "执行 warmup 状态机", ("benchmark:WarmupStateMachine", "benchmark:steady_state_excludes_cold_start")),
    (14, "执行基线重复 run", ("repeatability:default_hierarchy", "benchmark:ObservationStore")),
    (15, "执行进程重启重复", ("repeatability:ReplicationLevel", "repeatability:variance_components")),
    (16, "执行时段重复", ("repeatability:BlockSchedule", "repeatability:variance_components")),
    (17, "执行跨日重复", ("repeatability:variance_components", "repeatability:drift_slope")),
    (18, "执行多实例重复", ("repeatability:ReplicationHierarchy", "repeatability:_dominant_source")),
    (19, "注入 thermal 条件", ("repeatability:injection_cases", "repeatability:default_rule_set")),
    (20, "注入 clock/power condition", ("repeatability:AnomalyRuleSet.evaluate", "repeatability:HEALTH_LABELS")),
    (21, "注入受控 neighbor load", ("repeatability:injection_cases", "repeatability:evaluate_injections")),
    (22, "注入 memory/swap pressure", ("repeatability:AnomalyRuleSet", "repeatability:is_healthy")),
    (23, "注入 recoverable failure", ("repeatability:ExclusionLedger", "repeatability:apply_inclusion_policy")),
    (24, "验证 load generator headroom", ("repeatability:HEALTH_LABELS", "benchmark:overload_recovery_check")),
    (25, "验证 queue/backlog 排空", ("benchmark:overload_recovery_check", "repeatability:is_healthy")),
    (26, "计算 run-level statistics", ("repeatability:median", "repeatability:quantile")),
    (27, "计算短期 repeatability", ("repeatability:cv", "repeatability:robust_cv_like", "repeatability:autocorrelation")),
    (28, "计算跨进程/时段/日变异", ("repeatability:variance_components", "repeatability:bootstrap_ci")),
    (29, "检测趋势与 change point", ("repeatability:drift_slope", "repeatability:change_point")),
    (30, "应用预注册 inclusion policy", ("repeatability:apply_inclusion_policy", "repeatability:ExclusionEntry")),
    (31, "执行异常处理敏感性", ("repeatability:sensitivity_report", "repeatability:mad")),
    (32, "构建 telemetry 归因模型", ("repeatability:drift_slope", "repeatability:AnomalyRuleSet")),
    (33, "评估 candidate effect 稳定性", ("repeatability:paired_log_ratio", "repeatability:StabilityThresholds")),
    (34, "评估排序/Pareto 稳定性", ("repeatability:frontier_membership_probability", "repeatability:bootstrap_ci")),
    (35, "独立 confirmation run", ("repeatability:confirmation_gate", "repeatability:PilotConfirmationSplit")),
    (36, "形成稳定性 verdict", ("repeatability:stability_verdicts", "campaign:StatusMatrix")),
)
