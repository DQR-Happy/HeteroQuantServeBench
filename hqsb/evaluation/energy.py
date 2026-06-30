"""E12-06: power-window alignment, energy integration and efficiency metrics.

Power is not energy, and a watt reading is not a measurement capability.  This
module therefore:

* distinguishes device / accelerator-set / node / facility boundaries and
  refuses to put two boundaries in one ranking;
* keeps ``total`` and ``incremental-over-idle`` energy side by side (publishing
  only the better-looking one is forbidden);
* derives every denominator from the run's own token/request accounting, using
  *quality- and SLO-compliant* work only — draft, rejected and over-SLO tokens
  never sneak into ``J/token``;
* reports an unavailable meter as ``MEASUREMENT_UNAVAILABLE`` with the missing
  fields, never a TDP-based estimate.

Nothing here samples hardware or runs a benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.evaluation.layers import TokenAccounting
from hqsb.evaluation.records import (
    MISSING_MEASUREMENT_UNAVAILABLE,
    TABLE_SCHEMAS,
)

EXPERIMENT_ID = "E12-06"
TITLE = "功耗窗口对齐、能量积分与请求/Token 能效"
CLAIM_BOUNDARY = (
    "本实验通过证明同一有效工作量的能量被可靠测得，"
    "不证明设备级能效等同于数据中心能效；不可测平台不得填估计值。"
)

ENERGY_BOUNDARIES: Tuple[str, ...] = ("device", "accelerators", "node", "facility_adjusted")

BOUNDARY_INCLUDES: Mapping[str, Tuple[str, ...]] = {
    "device": ("accelerator board",),
    "accelerators": ("all accelerators of the job",),
    "node": ("accelerators", "host CPU", "host RAM", "fans/PSU (where instrumented)"),
    "facility_adjusted": ("node energy", "cooling/overhead model (PUE scenario)"),
}

BOUNDARY_LEGAL_CLAIMS: Mapping[str, str] = {
    "device": "device energy only",
    "accelerators": "accelerator-set energy",
    "node": "node energy",
    "facility_adjusted": "scenario energy (a model, not a measurement)",
}

WINDOW_POLICIES: Tuple[str, ...] = (
    "cold_total",
    "ready_to_complete",
    "steady_measurement",
    "phase_prefill",
    "phase_decode",
    "cooldown_tail",
)

METER_SEMANTICS: Tuple[str, ...] = ("instantaneous", "averaged", "accumulator_derived")

IDLE_STATES: Tuple[str, ...] = ("model_loaded", "service_ready", "same_device_allocation")

POWER_QUALITY_CHECKS: Tuple[str, ...] = (
    "meter_identity_unit_boundary",
    "timestamps_monotonic_and_alignable",
    "gap_fraction_within_threshold",
    "no_unknown_reset_or_wrap",
    "performance_and_power_same_run",
    "request_token_accounting_closed",
    "idle_policy_identical",
    "no_unexplained_throttle_or_contamination",
    "interval_not_falsely_precise",
    "comparison_quality_and_slo_pass",
)


@dataclass(frozen=True)
class WindowPolicy:
    policy_id: str
    start_event: str
    end_event: str
    includes_backlog: bool
    includes_cooldown: bool
    phase_filter: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.policy_id not in WINDOW_POLICIES:
            problems.append(f"unknown window policy {self.policy_id!r}")
        if not self.start_event or not self.end_event:
            problems.append(f"policy {self.policy_id!r} needs start and end events")
        return problems


WINDOW_POLICY_DEFINITIONS: Mapping[str, WindowPolicy] = {
    "cold_total": WindowPolicy("cold_total", "process_start", "task_complete", True, False),
    "ready_to_complete": WindowPolicy(
        "ready_to_complete", "service_ready", "all_admitted_requests_terminal", True, False
    ),
    "steady_measurement": WindowPolicy(
        "steady_measurement", "steady_window_start", "steady_window_end", False, False
    ),
    "phase_prefill": WindowPolicy("phase_prefill", "prefill_start", "prefill_end", False, False, "prefill"),
    "phase_decode": WindowPolicy("phase_decode", "decode_start", "decode_end", False, False, "decode"),
    "cooldown_tail": WindowPolicy("cooldown_tail", "load_end", "cooldown_end", False, True),
}


def policy(policy_id: str) -> WindowPolicy:
    try:
        return WINDOW_POLICY_DEFINITIONS[policy_id]
    except KeyError as exc:
        raise ConfigError(f"unknown window policy {policy_id!r}") from exc


@dataclass(frozen=True)
class IdlePolicy:
    idle_state: str
    idle_window_s: float
    idle_power_estimator: str
    clip: str = "none"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.idle_state not in IDLE_STATES:
            problems.append(f"unknown idle state {self.idle_state!r}")
        if self.idle_window_s <= 0:
            problems.append("the idle window must be positive")
        if self.idle_power_estimator not in ("median", "time_weighted_mean"):
            problems.append(f"unknown idle estimator {self.idle_power_estimator!r}")
        if self.clip not in ("none", "clip_negative"):
            problems.append(f"unknown incremental-energy clip policy {self.clip!r}")
        return problems


def assert_same_boundary(rows: Sequence[Mapping[str, Any]]) -> None:
    boundaries = {str(row.get("boundary", "")) for row in rows if row.get("boundary")}
    if len(boundaries) > 1:
        raise ConfigError(
            f"energy rows from different boundaries ({sorted(boundaries)}) must not share one "
            "ranking: device, node and facility energies are not comparable"
        )


# ── meters and series ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class MeterIdentity:
    meter_id: str
    platform_instance_id: str
    api_or_device: str
    firmware_version: str = ""
    tool_version: str = ""
    channel: str = ""
    unit: str = "W"
    calibrated_at: str = ""
    calibration_status: str = "uncalibrated"
    task_mapping: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("platform_instance_id", "api_or_device", "channel"):
            if not getattr(self, name):
                problems.append(f"meter identity is missing {name!r}")
        if self.unit not in ("W", "mW"):
            problems.append(f"unexpected meter unit {self.unit!r}")
        if self.calibration_status not in ("calibrated", "uncalibrated", "factory"):
            problems.append(f"unknown calibration status {self.calibration_status!r}")
        return problems


@dataclass(frozen=True)
class MeterCapability:
    meter_id: str
    field: str
    semantics: str
    boundary: str
    unit: str
    resolution: float
    sample_period_s: float
    availability: str
    permission: str
    wrap_or_reset: str = "unknown"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.semantics not in METER_SEMANTICS:
            problems.append(f"unknown semantics {self.semantics!r}: a W value without its semantics is unusable")
        if self.boundary not in ENERGY_BOUNDARIES:
            problems.append(f"unknown boundary {self.boundary!r}")
        if not self.unit:
            problems.append("a meter field needs a unit")
        if self.resolution <= 0:
            problems.append("a meter field needs a positive resolution")
        if self.sample_period_s <= 0:
            problems.append("a meter field needs a positive sample period")
        if self.availability not in ("available", "unavailable", "permission_denied", "unknown"):
            problems.append(f"unknown availability {self.availability!r}")
        return problems


def can_measure(capability: MeterCapability, boundary: str) -> bool:
    return capability.availability == "available" and capability.boundary == boundary


@dataclass(frozen=True)
class EnergySample:
    t_ns: int
    power_w: Optional[float] = None
    energy_j: Optional[float] = None
    status: str = "OK"
    quality_flags: Tuple[str, ...] = ()


@dataclass
class PowerSeries:
    run_id: str
    meter_id: str
    samples: Tuple[EnergySample, ...]
    boundary: str = "device"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.run_id or not self.meter_id:
            problems.append("a power series needs a run id and a meter id")
        if self.boundary not in ENERGY_BOUNDARIES:
            problems.append(f"unknown boundary {self.boundary!r}")
        timestamps = [sample.t_ns for sample in self.samples]
        if timestamps != sorted(timestamps):
            problems.append("telemetry timestamps must be monotonic; out-of-order raw data is flagged, not reordered")
        if len(set(timestamps)) != len(timestamps):
            problems.append("duplicate timestamps in the raw series")
        if any(sample.status != "OK" for sample in self.samples):
            problems.append("non-OK samples are kept and flagged, never dropped")
        return problems

    def gap_fraction(self, *, expected_period_s: float) -> float:
        if len(self.samples) < 2:
            return 1.0
        span_s = (self.samples[-1].t_ns - self.samples[0].t_ns) / 1e9
        expected = max(1, int(span_s / expected_period_s) + 1)
        return max(0.0, 1.0 - (len(self.samples) / expected))


# ── integration and alignment ─────────────────────────────────────────────


def trapezoid_integral(
    series: PowerSeries, *, t0_ns: int, t1_ns: int, expected_period_s: float = 0.1
) -> Dict[str, Any]:
    """``Σ (P_i + P_{i+1})/2 x Δt`` over the window (no extrapolation)."""
    if t1_ns <= t0_ns:
        raise ConfigError("the integration window must be positive")
    points = [
        (sample.t_ns, sample.power_w)
        for sample in series.samples
        if t0_ns <= sample.t_ns <= t1_ns and sample.power_w is not None
    ]
    if len(points) < 2:
        return {
            "method": "integration",
            "status": MISSING_MEASUREMENT_UNAVAILABLE,
            "reason": "fewer than two power samples inside the window",
            "sample_count": len(points),
            "window_s": (t1_ns - t0_ns) / 1e9,
        }
    total = 0.0
    for (t_a, p_a), (t_b, p_b) in zip(points, points[1:]):
        total += (p_a + p_b) / 2.0 * ((t_b - t_a) / 1e9)
    return {
        "method": "trapezoid",
        "status": "OK",
        "energy_j": total,
        "sample_count": len(points),
        "gap_fraction": series.gap_fraction(expected_period_s=expected_period_s),
        "window_s": (t1_ns - t0_ns) / 1e9,
        "uncertainty_j": total * series.gap_fraction(expected_period_s=expected_period_s),
    }


def accumulator_delta(
    series: PowerSeries, *, t0_ns: int, t1_ns: int
) -> Dict[str, Any]:
    """First-minus-last of a cumulative energy counter (unit: joule)."""
    points = [
        (sample.t_ns, sample.energy_j)
        for sample in series.samples
        if sample.energy_j is not None
    ]
    inside = [(t, value) for t, value in points if t0_ns <= t <= t1_ns]
    if len(inside) < 2:
        return {
            "method": "accumulator",
            "status": MISSING_MEASUREMENT_UNAVAILABLE,
            "reason": "fewer than two accumulator samples inside the window",
        }
    deltas = [b - a for (_, a), (_, b) in zip(inside, inside[1:])]
    if any(delta < 0 for delta in deltas):
        return {
            "method": "accumulator",
            "status": MISSING_MEASUREMENT_UNAVAILABLE,
            "reason": "counter decreased inside the window: reset/wrap must be resolved before use",
        }
    return {
        "method": "accumulator",
        "status": "OK",
        "energy_j": inside[-1][1] - inside[0][1],
        "sample_count": len(inside),
        "resolution_note": "resolution limits the smallest resolvable difference",
    }


def cross_check_accumulator_vs_integral(
    *, accumulator_j: float, integral_j: float, tolerance_j: float, explained_by: str = ""
) -> Dict[str, Any]:
    difference = abs(accumulator_j - integral_j)
    ok = difference <= tolerance_j
    return {
        "accumulator_delta_j": accumulator_j,
        "integral_j": integral_j,
        "difference_j": difference,
        "tolerance_j": tolerance_j,
        "ok": ok,
        "explained_by": explained_by if not ok else "",
        "status": "OK" if ok else "INCONSISTENT",
    }


def align_windows(
    series: PowerSeries,
    *,
    events: Mapping[str, int],
    policy_id: str,
    offset_s: float = 0.0,
    averaging_window_s: float = 0.0,
) -> Dict[str, Any]:
    """Map raw events onto the sample grid, keeping raw and aligned timestamps."""
    window_policy = policy(policy_id)
    start = events.get(window_policy.start_event)
    end = events.get(window_policy.end_event)
    if start is None or end is None:
        raise ConfigError(
            f"window policy {policy_id!r} needs events {window_policy.start_event!r} and "
            f"{window_policy.end_event!r}"
        )
    offset_ns = int(offset_s * 1e9)
    return {
        "policy_id": policy_id,
        "raw_t0_ns": start,
        "raw_t1_ns": end,
        "aligned_t0_ns": start + offset_ns,
        "aligned_t1_ns": end + offset_ns,
        "offset_s": offset_s,
        "averaging_window_s": averaging_window_s,
        "alignment_note": "offset comes from calibration; the raw window is preserved",
        "includes_backlog": window_policy.includes_backlog,
        "includes_cooldown": window_policy.includes_cooldown,
    }


def power_quality_gate(
    *,
    meter: MeterCapability,
    window: Mapping[str, Any],
    accounting_closed: bool,
    same_run: bool,
    idle_policy_identical: bool,
    unexplained_contamination: bool,
    interval_j: Optional[float],
    resolution_j: Optional[float],
    comparison_ok: bool,
) -> Dict[str, Any]:
    """The ten §12 power-quality checks, as explicit statuses."""
    checks: List[Dict[str, Any]] = []

    def add(check_id: str, ok: bool, detail: str = "") -> None:
        checks.append({"check_id": check_id, "status": "PASS" if ok else "FAIL", "detail": detail})

    add(
        "meter_identity_unit_boundary",
        meter.validate() == [],
        "; ".join(meter.validate()),
    )
    add(
        "timestamps_monotonic_and_alignable",
        bool(window.get("aligned_t0_ns") is not None and window.get("aligned_t1_ns") is not None),
    )
    gap = float(window.get("gap_fraction", 1.0))
    add("gap_fraction_within_threshold", gap <= 0.05, f"gap_fraction={gap:.3f}")
    add("no_unknown_reset_or_wrap", meter.wrap_or_reset in ("none",), meter.wrap_or_reset)
    add("performance_and_power_same_run", same_run)
    add("request_token_accounting_closed", accounting_closed)
    add("idle_policy_identical", idle_policy_identical)
    add("no_unexplained_throttle_or_contamination", not unexplained_contamination)
    precise = True
    if interval_j is not None and resolution_j:
        precise = interval_j >= resolution_j
    add(
        "interval_not_falsely_precise",
        precise,
        f"interval={interval_j} resolution={resolution_j}",
    )
    add("comparison_quality_and_slo_pass", comparison_ok)
    return {
        "checks": checks,
        "passed": sum(1 for row in checks if row["status"] == "PASS"),
        "ok": all(row["status"] == "PASS" for row in checks),
    }


# ── energy results ────────────────────────────────────────────────────────


def idle_reference(
    pre_series: PowerSeries, post_series: PowerSeries, *, idle: IdlePolicy
) -> Dict[str, Any]:
    problems = idle.validate()
    if problems:
        raise ConfigError("invalid idle policy: " + "; ".join(problems))
    pre = [sample.power_w for sample in pre_series.samples if sample.power_w is not None]
    post = [sample.power_w for sample in post_series.samples if sample.power_w is not None]
    if not pre or not post:
        return {
            "status": MISSING_MEASUREMENT_UNAVAILABLE,
            "reason": "idle windows before/after the load are required",
            "idle_policy": idle.idle_state,
        }
    estimator = sorted(pre + post)
    midpoint = len(estimator) // 2
    median = (
        estimator[midpoint]
        if len(estimator) % 2
        else (estimator[midpoint - 1] + estimator[midpoint]) / 2.0
    )
    drift = abs(sum(pre) / len(pre) - sum(post) / len(post)) / (median or 1.0)
    return {
        "status": "OK" if drift <= 0.1 else "UNSTABLE",
        "idle_power_w": median if idle.idle_power_estimator == "median" else (sum(pre) + sum(post)) / (len(pre) + len(post)),
        "estimator": idle.idle_power_estimator,
        "drift_rel": drift,
        "widen_or_mark_unstable": drift > 0.1,
        "idle_state": idle.idle_state,
    }


def incremental_energy(
    *, load_energy_j: float, idle_power_w: float, window_s: float, idle: IdlePolicy
) -> Dict[str, Any]:
    """``∫ (P_load - P_idle) dt`` (total energy is always reported as well)."""
    raw = load_energy_j - idle_power_w * window_s
    if idle.clip == "clip_negative":
        value = max(0.0, raw)
    else:
        value = raw
    return {
        "incremental_energy_j": value,
        "raw_incremental_j": raw,
        "clip_policy": idle.clip,
        "note": "incremental answers 'beyond staying ready'; the total energy stays the primary physical quantity",
    }


def total_energy(
    *,
    accumulator: Optional[Mapping[str, Any]] = None,
    integral: Optional[Mapping[str, Any]] = None,
    boundary: str = "device",
) -> Dict[str, Any]:
    if boundary not in ENERGY_BOUNDARIES:
        raise ConfigError(f"unknown energy boundary {boundary!r}")
    if accumulator and accumulator.get("status") == "OK":
        chosen = accumulator
        method = "accumulator"
    elif integral and integral.get("status") == "OK":
        chosen = integral
        method = "integration"
    else:
        return {
            "status": MISSING_MEASUREMENT_UNAVAILABLE,
            "reason": "neither the accumulator difference nor the power integral is usable",
            "boundary": boundary,
        }
    return {
        "status": "OK",
        "method": method,
        "total_energy_j": chosen["energy_j"],
        "sample_count": chosen.get("sample_count", 0),
        "gap_fraction": chosen.get("gap_fraction", 0.0),
        "boundary": boundary,
        "legal_claim": BOUNDARY_LEGAL_CLAIMS[boundary],
    }


def efficiency_metrics(
    *,
    accounting: TokenAccounting,
    total_energy_j: Optional[float],
    boundary: str,
    token_goodput: Optional[float] = None,
    average_power_w: Optional[float] = None,
) -> Dict[str, Any]:
    """J/request, J/token, tokens/J and goodput/W on compliant work only."""
    problems = accounting.validate()
    if problems:
        raise ConfigError("token accounting is inconsistent: " + "; ".join(problems))
    if not accounting.closed():
        return {
            "status": MISSING_MEASUREMENT_UNAVAILABLE,
            "reason": "request/token accounting does not close: denominators are not usable",
            "missing_reason": "RUN_FAILED",
        }
    if total_energy_j is None:
        return {
            "status": MISSING_MEASUREMENT_UNAVAILABLE,
            "reason": "no usable energy measurement for this window",
            "missing_reason": MISSING_MEASUREMENT_UNAVAILABLE,
        }
    if accounting.completed_requests <= 0 or accounting.accepted_output_tokens <= 0:
        return {
            "status": MISSING_MEASUREMENT_UNAVAILABLE,
            "reason": "no compliant requests/tokens in the window: a ratio would be fabricated work",
            "missing_reason": "QUALITY_GATE_FAILED",
        }
    metrics = {
        "status": "OK",
        "boundary": boundary,
        "total_energy_j": total_energy_j,
        "j_per_request": total_energy_j / accounting.completed_requests,
        "j_per_token": total_energy_j / accounting.accepted_output_tokens,
        "tokens_per_j": accounting.accepted_output_tokens / total_energy_j,
        "denominator_requests": accounting.completed_requests,
        "denominator_tokens": accounting.accepted_output_tokens,
        "excluded_from_denominator": {
            "draft_tokens": accounting.speculative_draft_tokens,
            "rejected_tokens": accounting.rejected_tokens,
            "failed_requests": accounting.failed_requests,
            "timed_out_requests": accounting.timed_out_requests,
        },
    }
    if token_goodput is not None and average_power_w:
        metrics["goodput_per_w"] = token_goodput / average_power_w
    else:
        metrics["goodput_per_w"] = None
        metrics["goodput_per_w_reason"] = "token goodput or average power unavailable"
    return metrics


def phase_decomposition(
    *, series: PowerSeries, phase_windows: Mapping[str, Tuple[int, int]], min_resolution_ns: int
) -> Dict[str, Any]:
    """Only decompose when the trace resolution supports it."""
    for name, (start, end) in phase_windows.items():
        if (end - start) < min_resolution_ns:
            return {
                "status": "NOT_APPLICABLE",
                "reason": f"phase {name!r} is shorter than the trace resolution: report the whole window instead",
                "phases": {},
            }
    phases = {
        name: trapezoid_integral(series, t0_ns=start, t1_ns=end) for name, (start, end) in sorted(phase_windows.items())
    }
    return {"status": "OK", "phases": phases}


def throttle_association(
    measurement: Mapping[str, Any], telemetry: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    flags = sorted(
        {
            str(row.get("label", ""))
            for row in telemetry
            if row.get("throttled") or row.get("power_limited") or row.get("clock_below_reference")
        }
    )
    return {
        "throttle_flags": flags,
        "clean": not flags,
        "action": "hand to the E12-04 exclusion rules" if flags else "",
    }


def uncertainty_budget(measurement: Mapping[str, Any]) -> Dict[str, Any]:
    """Interval from the real uncertainty sources; the resolution is a floor.

    ``falsely_precise`` is set when the reported half-width is *below* the meter
    resolution: an interval smaller than what the sensor can resolve is not
    precision, it is fiction.
    """
    components = {
        "sample_gap_j": float(measurement.get("gap_fraction", 0.0)) * float(measurement.get("total_energy_j", 0.0)),
        "alignment_j": float(measurement.get("alignment_uncertainty_j", 0.0)),
        "idle_drift_j": float(measurement.get("idle_drift_j", 0.0)),
        "run_variation_j": float(measurement.get("run_variation_j", 0.0)),
    }
    half_width = sum(components.values())
    total = float(measurement.get("total_energy_j", 0.0))
    resolution = float(measurement.get("resolution_j", 0.0))
    interval = 2.0 * half_width
    return {
        "components": dict(sorted(components.items())),
        "half_width_j": half_width,
        "interval_low": total - half_width,
        "interval_high": total + half_width,
        "interval_j": interval,
        "resolution_j": resolution,
        "interval_wider_than_resolution": interval >= resolution,
        "falsely_precise": interval < resolution,
    }


def boundary_sensitivity(
    *, total: Optional[float], incremental: Optional[float], pre_idle: Optional[float], post_idle: Optional[float]
) -> Dict[str, Any]:
    values = [value for value in (incremental, pre_idle, post_idle) if value is not None]
    spread = (max(values) - min(values)) / abs(total) if values and total else None
    return {
        "total_j": total,
        "incremental_j": incremental,
        "pre_idle_w": pre_idle,
        "post_idle_w": post_idle,
        "spread_rel": spread,
        "conclusion_changes": bool(spread is not None and spread > 0.1),
        "status": "CONDITIONAL" if spread is not None and spread > 0.1 else "OK",
    }


def collector_overhead_check(
    *, on_latency_rel_delta: float, on_cpu_rel_delta: float, threshold_rel: float
) -> Dict[str, Any]:
    exceeded = max(abs(on_latency_rel_delta), abs(on_cpu_rel_delta)) > threshold_rel
    return {
        "latency_delta_rel": on_latency_rel_delta,
        "cpu_delta_rel": on_cpu_rel_delta,
        "threshold_rel": threshold_rel,
        "exceeded": exceeded,
        "action": "lower the sampling rate or use an independent collection path" if exceeded else "",
    }


def unavailable_energy(*, meter_id: str, reason: str, missing_fields: Sequence[str]) -> Dict[str, Any]:
    """The only legal representation of "we cannot measure energy here"."""
    if not reason:
        raise ConfigError("an unavailable energy result must state its reason")
    if not missing_fields:
        raise ConfigError("an unavailable energy result must list what is missing")
    return {
        "status": MISSING_MEASUREMENT_UNAVAILABLE,
        "meter_id": meter_id,
        "reason": reason,
        "missing_fields": list(missing_fields),
        "estimate": None,
        "note": "TDP or vendor typical values are not a substitute for a measurement",
    }


@dataclass
class EnergyMeasurement:
    """``EnergyMeasurement`` (§11) — the row E12-07/E12-08 consume."""

    energy_measurement_id: str
    run_id: str
    comparison_id: str
    meter_id: str
    boundary: str
    channels: Tuple[str, ...] = ()
    window_policy_id: str = ""
    raw_t0_ns: int = 0
    raw_t1_ns: int = 0
    aligned_t0_ns: int = 0
    aligned_t1_ns: int = 0
    method: str = ""
    sample_interval_s: float = 0.0
    sample_count: int = 0
    gap_fraction: float = 0.0
    total_energy_j: Optional[float] = None
    incremental_energy_j: Optional[float] = None
    idle_power_w: Optional[float] = None
    completed_compliant_requests: int = 0
    accepted_compliant_tokens: int = 0
    j_per_request: Optional[float] = None
    j_per_token: Optional[float] = None
    tokens_per_j: Optional[float] = None
    goodput_per_w: Optional[float] = None
    temperature_range: Tuple[float, float] = ()
    clock_range: Tuple[float, float] = ()
    throttle_flags: Tuple[str, ...] = ()
    uncertainty_low: Optional[float] = None
    uncertainty_high: Optional[float] = None
    quality_status: str = "not_run"
    missing_reason: str = ""
    artifact_refs: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.boundary not in ENERGY_BOUNDARIES:
            problems.append(f"unknown boundary {self.boundary!r}")
        if self.method and self.method not in ("accumulator", "integration"):
            problems.append(f"unknown energy method {self.method!r}")
        if self.total_energy_j is None:
            if not self.missing_reason:
                problems.append("an energy row without total energy must carry a missing reason")
            if any(value is not None for value in (self.j_per_request, self.j_per_token)):
                problems.append("efficiency ratios without a total energy are fabricated")
        else:
            if self.total_energy_j < 0:
                problems.append("total energy cannot be negative")
            if self.incremental_energy_j is None:
                problems.append(
                    "total and incremental energy are published together: reporting only one hides "
                    "the idle policy"
                )
            if self.j_per_token is not None and not self.accepted_compliant_tokens:
                problems.append("J/token needs the compliant-token denominator")
        if self.throttle_flags and self.quality_status == "pass":
            problems.append("a throttled measurement cannot be quality-pass without explanation")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "energy_measurement_id": self.energy_measurement_id,
            "run_id": self.run_id,
            "comparison_id": self.comparison_id,
            "meter_id": self.meter_id,
            "boundary": self.boundary,
            "channels": list(self.channels),
            "window_policy_id": self.window_policy_id,
            "raw_t0_ns": self.raw_t0_ns,
            "raw_t1_ns": self.raw_t1_ns,
            "aligned_t0_ns": self.aligned_t0_ns,
            "aligned_t1_ns": self.aligned_t1_ns,
            "method": self.method,
            "sample_interval_s": self.sample_interval_s,
            "sample_count": self.sample_count,
            "gap_fraction": self.gap_fraction,
            "total_energy_j": self.total_energy_j,
            "incremental_energy_j": self.incremental_energy_j,
            "idle_power_w": self.idle_power_w,
            "completed_compliant_requests": self.completed_compliant_requests,
            "accepted_compliant_tokens": self.accepted_compliant_tokens,
            "j_per_request": self.j_per_request,
            "j_per_token": self.j_per_token,
            "tokens_per_j": self.tokens_per_j,
            "goodput_per_w": self.goodput_per_w,
            "temperature_range": list(self.temperature_range),
            "clock_range": list(self.clock_range),
            "throttle_flags": list(self.throttle_flags),
            "uncertainty_low": self.uncertainty_low,
            "uncertainty_high": self.uncertainty_high,
            "quality_status": self.quality_status,
            "missing_reason": self.missing_reason,
            "artifact_refs": list(self.artifact_refs),
        }


def table_schemas() -> Mapping[str, Tuple[str, ...]]:
    return {name: TABLE_SCHEMAS[name] for name in sorted(TABLE_SCHEMAS) if name.startswith("e12_06.")}


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only callability check; never an experiment."""
    series = PowerSeries(
        run_id="run-1",
        meter_id="meter-1",
        samples=(
            EnergySample(t_ns=0, power_w=100.0, energy_j=0.0),
            EnergySample(t_ns=100_000_000, power_w=200.0, energy_j=20.0),
            EnergySample(t_ns=200_000_000, power_w=150.0, energy_j=35.0),
        ),
    )
    integral = trapezoid_integral(series, t0_ns=0, t1_ns=200_000_000)
    accumulator = accumulator_delta(series, t0_ns=0, t1_ns=200_000_000)
    accounting = TokenAccounting(
        run_id="run-1",
        prompt_tokens=10,
        requested_output_tokens=20,
        generated_tokens_before_stop=20,
        accepted_output_tokens=20,
        served_output_tokens=20,
        speculative_draft_tokens=5,
        admitted_requests=1,
        completed_requests=1,
    )
    efficiency = efficiency_metrics(
        accounting=accounting, total_energy_j=integral["energy_j"], boundary="device", average_power_w=150.0
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "integral_method": integral["method"],
        "integral_positive": integral["energy_j"] > 0,
        "accumulator_positive": accumulator["energy_j"] > 0,
        "denominator_excludes_draft": efficiency["excluded_from_denominator"]["draft_tokens"] == 5,
        "boundary_mixing_rejected": _expect_config_error(
            lambda: assert_same_boundary([{"boundary": "device"}, {"boundary": "node"}])
        ),
        "unavailable_is_structured": unavailable_energy(
            meter_id="meter-2", reason="no NVML on this platform", missing_fields=["accelerator.power_W"]
        )["estimate"]
        is None,
    }


def _expect_config_error(callable_: Any) -> bool:
    try:
        callable_()
    except ConfigError:
        return True
    return False


PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "读取 telemetry capability", ("platform:energy_capability", "energy:MeterCapability")),
    (2, "冻结 claim boundary", ("energy:ENERGY_BOUNDARIES", "energy:BOUNDARY_LEGAL_CLAIMS")),
    (3, "冻结能量窗口", ("energy:WINDOW_POLICY_DEFINITIONS", "energy:WindowPolicy")),
    (4, "冻结 idle policy", ("energy:IdlePolicy", "energy:IDLE_STATES")),
    (5, "冻结分母和质量/SLO", ("layers:TokenAccounting", "layers:SloSpec")),
    (6, "冻结采样与误差预算", ("energy:uncertainty_budget", "energy:MeterCapability")),
    (7, "建立 meter identity", ("energy:MeterIdentity", "energy:MeterCapability")),
    (8, "验证单位和时间戳", ("energy:PowerSeries.validate", "platform:TelemetryFieldAdapter")),
    (9, "测采样开销", ("energy:collector_overhead_check", "platform:telemetry_smoke_check")),
    (10, "校准时间延迟", ("energy:align_windows", "platform:TelemetryFieldAdapter")),
    (11, "校验累计计数", ("energy:accumulator_delta", "energy:MeterCapability")),
    (12, "校验功率积分", ("energy:cross_check_accumulator_vs_integral", "energy:trapezoid_integral")),
    (13, "无负载 baseline", ("energy:idle_reference", "energy:IdlePolicy")),
    (14, "验证系统背景负载", ("energy:PowerSeries", "repeatability:HEALTH_LABELS")),
    (15, "warmup/thermal gate", ("benchmark:thermal_gate", "repeatability:default_rule_set")),
    (16, "启动同步采集", ("energy:align_windows", "campaign:ArtifactLayout")),
    (17, "operator 能量 case", ("energy:trapezoid_integral", "energy:PowerSeries")),
    (18, "model-core workload", ("energy:efficiency_metrics", "layers:TokenAccounting")),
    (19, "service load sweep", ("energy:efficiency_metrics", "layers:goodput")),
    (20, "distributed workload", ("energy:total_energy", "energy:ENERGY_BOUNDARIES")),
    (21, "后置 idle/cooldown", ("energy:idle_reference", "energy:WINDOW_POLICY_DEFINITIONS")),
    (22, "验证 request/token 闭合", ("energy:efficiency_metrics", "benchmark:token_accounting_closes")),
    (23, "清洗 telemetry 只标记不覆盖", ("energy:PowerSeries.validate", "records:MISSINGNESS_CODES")),
    (24, "对齐 power 与 events", ("energy:align_windows", "energy:WindowPolicy")),
    (25, "计算 total energy", ("energy:total_energy", "energy:trapezoid_integral")),
    (26, "计算 idle reference", ("energy:idle_reference", "energy:IdlePolicy")),
    (27, "计算 incremental energy", ("energy:incremental_energy", "energy:total_energy")),
    (28, "计算效率指标", ("energy:efficiency_metrics", "energy:EnergyMeasurement")),
    (29, "分解 phase/active-idle", ("energy:phase_decomposition", "energy:trapezoid_integral")),
    (30, "检测 throttle/thermal/clock", ("energy:throttle_association", "repeatability:default_rule_set")),
    (31, "重复与跨日验证", ("repeatability:bootstrap_ci", "energy:EnergyMeasurement")),
    (32, "load/power sensitivity", ("energy:boundary_sensitivity", "platform:energy_capability")),
    (33, "传播计量不确定性", ("energy:uncertainty_budget", "energy:EnergyMeasurement")),
    (34, "boundary/idle sensitivity", ("energy:boundary_sensitivity", "energy:total_energy")),
    (35, "验证跨候选公平性", ("energy:assert_same_boundary", "energy:power_quality_gate")),
    (36, "形成能效 verdict", ("energy:EnergyMeasurement", "campaign:AcceptanceDecision")),
)
