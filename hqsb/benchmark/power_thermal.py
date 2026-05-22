"""E02-08 power / energy / thermal evidence toolkit.

This module is **pure logic**: it never imports ``torch``, never spawns a
process and never touches hardware. It turns raw ``tegrastats`` lines and
sysfs snapshots into auditable evidence for the S02 power/energy/thermal
baseline (protocol: ``docs/stage_experiments/details/S02/E02-08_power_energy_and_thermal.md``).

The four ideas the module encodes:

1. **Energy is an integral, not a mean.**  ``integrate_energy`` uses the
   trapezoidal rule over the *actual* per-sample timestamps; a plain
   ``mean_power x duration`` is never produced as an energy figure.
2. **A measurement window must be covered.**  ``align_and_integrate`` truncates
   the integration to the samples that actually fall inside the window, reports
   the coverage ratio, the largest sample gap and the guard samples on both
   sides, and refuses to mark a window usable when coverage is short.  A short
   window yields ``usable=False`` with a reason; it is never silently reported
   as ``energy = 0``.
3. **Rail scope must be stated.**  ``RAIL_SCOPE`` freezes which electrical node
   is measured and forbids adding the total rail to its own sub-rails.
4. **Thermal drift is not throttling.**  ``assess_thermal`` separates
   "temperature rose", "frequency dropped" and "latency rose" and only labels a
   window ``thermal_suspect`` when the temperature trend actually explains the
   frequency drop; the classification also consumes independent cooling-device
   state (``*-throttle-alert``), which is real thermal-management evidence
   rather than correlation.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ── Pre-registered protocol (frozen before execution) ──────────────────────
#
# Every threshold below is a *pre-registration*: it is declared here, before
# any measurement, and the verifier consumes it from this single place so a
# number can never be re-tuned after seeing the data.

POWER_THERMAL_PROTOCOL: Dict[str, Any] = {
    # Sampling
    "monitor_interval_ms": 250,
    "sysfs_interval_ms": 250,
    # Rail scope
    "primary_rail": "VDD_IN",
    "sub_rails": ["VDD_CPU_GPU_CV", "VDD_SOC"],
    # Window coverage rules
    "min_power_samples": 8,
    "max_sample_gap_s": 1.5,          # 6 x nominal interval
    "min_coverage_ratio": 0.90,
    "guard_lookback_s": 2.0,          # samples this close outside the window count
    "min_guard_samples": 1,
    # Steady-window shape
    "requests_per_window": 3,
    "idle_window_s": 60.0,
    "independent_runs": 3,
    # Thermal rules: `cooldown_target_c` is the single start-of-window gate
    # (tj must be at or below it before a block begins) and is also the
    # post-block recovery target.
    "cooldown_target_c": 55.0,
    "cooldown_timeout_s": 900.0,
    "thermal_suspect_c": 74.0,        # tj-thermal active trip #1
    "thermal_hard_limit_c": 95.0,     # tj-thermal active trip #2 (E02-03 convention)
    "temp_rise_threshold_c": 3.0,
    "freq_drop_frac_threshold": 0.20,
    "latency_rise_frac_threshold": 0.10,
    "throttle_alert_cooling_types": [
        "cpu-throttle-alert",
        "gpu-throttle-alert",
        "soc0-throttle-alert",
        "soc1-throttle-alert",
        "soc2-throttle-alert",
        "cv0-throttle-alert",
        "cv1-throttle-alert",
        "cv2-throttle-alert",
        "hot-surface-alert",
        "devfreq-17000000.gpu",
    ],
    # Numerical closure
    "closure_rel_tol": 0.02,
}

# ── Rail scope declaration ─────────────────────────────────────────────────

RAIL_SCOPE: Dict[str, Any] = {
    "primary_rail": "VDD_IN",
    "primary_scope": "board",
    "primary_description": (
        "VDD_IN is the total module input reported by tegrastats: it covers the "
        "CPU, GPU, DRAM/EMC and I/O domains of the Orin module. It is a "
        "board/module scope figure and must not be called 'GPU-core energy'."
    ),
    "sub_rails": {
        "VDD_CPU_GPU_CV": (
            "CPU+GPU+CV domain input; reported by the same tegrastats line and "
            "well below VDD_IN, i.e. a subset of the module input."
        ),
        "VDD_SOC": (
            "SoC domain input; also a subset of the module input on this board."
        ),
    },
    "additive": False,
    "additivity_note": (
        "Sub-rails are subsets of the total module input, so VDD_IN + "
        "VDD_CPU_GPU_CV + VDD_SOC would double count. Only the primary rail is "
        "integrated into J/request; every sub-rail is reported separately and "
        "never summed with the primary."
    ),
    "average_field_policy": (
        "tegrastats' second value (<cur>mW/<avg>mW) is a running average whose "
        "start point is unknown, so it is stored as rail_average_mw and never "
        "integrated; only the instantaneous field is integrated over this "
        "run's own window."
    ),
}


# ── tegrastats parsing ─────────────────────────────────────────────────────

_RAM_RE = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
_LFB_RE = re.compile(r"\(lfb\s+(\d+)x(\d+)MB\)")
_SWAP_RE = re.compile(r"\bSWAP\s+(\d+)/(\d+)MB(?:\s+\(cached\s+(\d+)MB\))?")
_CPU_BRACKET_RE = re.compile(r"\bCPU\s+\[([^\]]*)\]")
_CPU_ENTRY_RE = re.compile(r"(\d+)%@(\d+)")
_GR3D_RE = re.compile(r"\bGR3D_FREQ\s+(\d+)%(?:@(\d+))?")
_RAIL_RE = re.compile(r"\b(VDD_[A-Z0-9_]+)\s+(\d+)mW(?:/(\d+)mW)?")
_TEMP_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]{0,15})@([\d.]+)C\b")
_TIME_RE = re.compile(r"^(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})\b")


def parse_tegrastats_line_v3(line: str) -> Dict[str, Any]:
    """Parse one raw ``tegrastats`` line into structured, status-tagged fields.

    Unlike the legacy :func:`hqsb.benchmark.tegrastats_parser.parse_tegrastats_line`,
    this parser keeps **every** power rail separately (so rail scope can be
    audited), tags the line with a ``parse_status`` and lists the fields that
    were expected but missing.  A missing rail is *never* defaulted to 0.

    Returns:
        Dict with ``rails_mw`` (instantaneous per rail), ``rail_average_mw``
        (the tool's cumulative average field, kept apart), ``temperatures_c``,
        ``cpu_utils_pct``, ``cpu_freqs_mhz``, ``gpu_util_pct``,
        ``gpu_freq_mhz``, RAM/SWAP fields, and:
            - ``parse_status``: ``"ok"`` / ``"degraded"`` / ``"failed"``
            - ``missing_fields``: expected-but-absent groups
            - ``unparsed_rails``: ``VDD_*`` names present but not parseable
              (format/unit drift evidence)
            - ``unknown_temperature_fields``: same idea for temperatures
    """
    result: Dict[str, Any] = {
        "raw": line,
        "rails_mw": {},
        "rail_average_mw": {},
        "temperatures_c": {},
        "cpu_utils_pct": [],
        "cpu_freqs_mhz": [],
        "gpu_util_pct": None,
        "gpu_freq_mhz": None,
    }

    time_match = _TIME_RE.match(line.strip())
    if time_match:
        result["time_text"] = time_match.group(1)

    ram_match = _RAM_RE.search(line)
    if ram_match:
        result["ram_used_mb"] = int(ram_match.group(1))
        result["ram_total_mb"] = int(ram_match.group(2))

    lfb_match = _LFB_RE.search(line)
    if lfb_match:
        result["lfb_free_blocks"] = int(lfb_match.group(1))
        result["lfb_block_mb"] = int(lfb_match.group(2))

    swap_match = _SWAP_RE.search(line)
    if swap_match:
        result["swap_used_mb"] = int(swap_match.group(1))
        result["swap_total_mb"] = int(swap_match.group(2))
        if swap_match.group(3) is not None:
            result["swap_cached_mb"] = int(swap_match.group(3))

    cpu_match = _CPU_BRACKET_RE.search(line)
    if cpu_match:
        entries = _CPU_ENTRY_RE.findall(cpu_match.group(1))
        result["cpu_utils_pct"] = [int(u) for u, _ in entries]
        result["cpu_freqs_mhz"] = [int(f) for _, f in entries]

    gr3d_match = _GR3D_RE.search(line)
    if gr3d_match:
        result["gpu_util_pct"] = int(gr3d_match.group(1))
        if gr3d_match.group(2) is not None:
            result["gpu_freq_mhz"] = int(gr3d_match.group(2))

    for name, current, average in _RAIL_RE.findall(line):
        result["rails_mw"][name] = int(current)
        if average:
            result["rail_average_mw"][name] = int(average)

    for name, value in _TEMP_RE.findall(line):
        result["temperatures_c"][name.lower()] = float(value)

    # ── status: what was expected but not found ───────────────────────────
    missing: List[str] = []
    if not result["rails_mw"]:
        missing.append("rails")
    if not result["cpu_utils_pct"]:
        missing.append("cpu_bracket")
    if not result["temperatures_c"]:
        missing.append("temperatures")
    if result["gpu_util_pct"] is None:
        missing.append("gpu_util")

    # Format/unit drift: a VDD_* token exists but the mW form did not match.
    declared_rails = set(re.findall(r"\b(VDD_[A-Z0-9_]+)\b", line))
    result["unparsed_rails"] = sorted(declared_rails - set(result["rails_mw"]))
    declared_temps = {
        name.lower() for name in re.findall(r"\b([A-Za-z]\w{0,15})@[\d.]+C\b", line)
    }
    result["unknown_temperature_fields"] = sorted(
        declared_temps - set(result["temperatures_c"])
    )
    result["missing_fields"] = missing

    if "rails" in missing and "temperatures" in missing and "cpu_bracket" in missing:
        result["parse_status"] = "failed"
    elif missing:
        result["parse_status"] = "degraded"
    else:
        result["parse_status"] = "ok"
    return result


def parse_telemetry_stream(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse a ``{"time_ns", "raw"}`` stream and audit the parser itself.

    Returns ``(parsed, parser_audit)`` where ``parser_audit`` reports the line
    count, the status histogram, the observed sample interval statistics and
    the timestamp monotonicity violations.  A non-empty
    ``timestamp_violations`` or a ``parse_error_ratio`` above the frozen
    threshold is evidence that the monitor (not the model) misbehaved.
    """
    parsed: List[Dict[str, Any]] = []
    status_counts = {"ok": 0, "degraded": 0, "failed": 0}
    for record in records:
        fields = parse_tegrastats_line_v3(str(record.get("raw", "")))
        time_ns = record.get("time_ns")
        fields["time_ns"] = int(time_ns) if isinstance(time_ns, int) else None
        status_counts[fields["parse_status"]] += 1
        parsed.append(fields)

    timestamps = [p["time_ns"] for p in parsed if p["time_ns"] is not None]
    intervals_s = [
        (timestamps[i + 1] - timestamps[i]) / 1e9
        for i in range(len(timestamps) - 1)
    ]
    violations = sum(1 for d in intervals_s if d <= 0)

    total = len(parsed)
    parser_audit = {
        "num_lines": total,
        "status_counts": status_counts,
        "parse_error_ratio": (status_counts["failed"] / total) if total else 0.0,
        "degraded_ratio": (status_counts["degraded"] / total) if total else 0.0,
        "num_timestamps": len(timestamps),
        "timestamp_violations": violations,
        "monotonic": violations == 0,
        "interval_s": _interval_stats(intervals_s),
    }
    return parsed, parser_audit


def _interval_stats(intervals: Sequence[float]) -> Dict[str, Any]:
    if not intervals:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "max": None,
            "mean": None,
            "stdev": None,
        }
    return {
        "count": len(intervals),
        "min": min(intervals),
        "median": statistics.median(intervals),
        "max": max(intervals),
        "mean": statistics.mean(intervals),
        "stdev": statistics.pstdev(intervals) if len(intervals) > 1 else 0.0,
    }


def rail_availability(
    parsed: Sequence[Mapping[str, Any]],
    rail: str,
) -> Dict[str, Any]:
    """Audit how well one rail is present in a parsed stream.

    A rail that appears on fewer lines than the primary rail is a *format
    drift* event, not a zero.  ``status`` is ``"usable"`` when the rail is
    present on at least the frozen ``min_coverage_ratio`` of lines that carry
    the primary rail.
    """
    primary = POWER_THERMAL_PROTOCOL["primary_rail"]
    primary_lines = sum(1 for p in parsed if primary in p.get("rails_mw", {}))
    rail_lines = sum(1 for p in parsed if rail in p.get("rails_mw", {}))
    ratio = (rail_lines / primary_lines) if primary_lines else 0.0
    return {
        "rail": rail,
        "lines_with_primary_rail": primary_lines,
        "lines_with_rail": rail_lines,
        "presence_ratio": ratio,
        "status": (
            "usable"
            if ratio >= POWER_THERMAL_PROTOCOL["min_coverage_ratio"]
            else ("absent" if rail_lines == 0 else "insufficient")
        ),
    }


def build_power_series(
    parsed: Sequence[Mapping[str, Any]],
    rail: str,
) -> Tuple[List[int], List[float]]:
    """Extract ``(timestamps_ns, power_mw)`` pairs for one rail.

    Records without the rail are skipped (never imputed).  The series is sorted
    by timestamp so a late-arriving line cannot create a negative interval.
    """
    pairs = [
        (int(p["time_ns"]), float(p["rails_mw"][rail]))
        for p in parsed
        if p.get("time_ns") is not None and rail in p.get("rails_mw", {})
    ]
    pairs.sort(key=lambda item: item[0])
    if not pairs:
        return [], []
    times = [t for t, _ in pairs]
    powers = [p for _, p in pairs]
    return times, powers


# ── Energy integration ─────────────────────────────────────────────────────


def integrate_energy(
    power_mw: Sequence[float],
    time_ns: Sequence[int],
    *,
    max_gap_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Trapezoidal energy integral over real sample timestamps.

    ``E = sum 0.5 * (P_i + P_{i+1}) * (t_{i+1} - t_i)`` with ``P`` in mW and
    ``t`` in ns; the result is converted to joules.

    Non-positive intervals (duplicate or out-of-order timestamps) are counted
    and *skipped* rather than integrated, and never repaired by interpolation.
    ``covered_s`` is the sum of the integrated intervals, which equals
    ``t_last - t_first`` only when the timestamps are strictly increasing.
    """
    if len(power_mw) != len(time_ns):
        raise ValueError(
            f"power/time length mismatch: {len(power_mw)} vs {len(time_ns)}"
        )

    energy_mj = 0.0
    integrated_s = 0.0
    negative = 0
    zero = 0
    oversized = 0
    max_gap_seen = 0.0

    for i in range(len(power_mw) - 1):
        dt_ns = int(time_ns[i + 1]) - int(time_ns[i])
        if dt_ns < 0:
            negative += 1
            continue
        if dt_ns == 0:
            zero += 1
            continue
        dt_s = dt_ns / 1e9
        energy_mj += 0.5 * (float(power_mw[i]) + float(power_mw[i + 1])) * dt_s
        integrated_s += dt_s
        max_gap_seen = max(max_gap_seen, dt_s)
        if max_gap_s is not None and dt_s > max_gap_s:
            oversized += 1

    energy_j = energy_mj / 1000.0
    span_s = (
        (int(time_ns[-1]) - int(time_ns[0])) / 1e9 if len(time_ns) >= 2 else 0.0
    )
    return {
        "num_samples": len(power_mw),
        "num_intervals": len(power_mw) - 1,
        "energy_j": energy_j,
        "covered_s": integrated_s,
        "span_s": span_s,
        "avg_power_w": (energy_j / integrated_s) if integrated_s > 0 else None,
        "max_gap_s": max_gap_seen,
        "gap_violations": oversized,
        "negative_dt_intervals": negative,
        "zero_dt_intervals": zero,
        "valid": (
            len(power_mw) >= 2 and integrated_s > 0 and negative == 0
        ),
    }


def trapezoid_energy_j(
    power_mw: Sequence[float],
    time_ns: Sequence[int],
) -> float:
    """Oracle-friendly wrapper returning only the integral in joules."""
    return float(integrate_energy(power_mw, time_ns)["energy_j"])


def align_and_integrate(
    parsed: Sequence[Mapping[str, Any]],
    begin_ns: int,
    end_ns: int,
    rail: str,
    *,
    protocol: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Integrate one rail over a host-monotonic window and audit the coverage.

    Frozen boundary policy: **truncate, never extrapolate**.  Integration runs
    from the first to the last in-window sample, so the reported energy is
    always backed by real samples; ``coverage_ratio`` exposes what fraction of
    the requested window that represents, and guard samples just outside both
    edges prove the monitor was running continuously across the boundary.

    ``usable`` is False when the window is short of samples, contains a gap
    larger than the frozen maximum, is under-covered, or lacks a guard sample.
    An unusable window carries a ``reason`` and an energy value that callers
    must not publish as a measurement.
    """
    proto = protocol or POWER_THERMAL_PROTOCOL
    times, powers = build_power_series(parsed, rail)

    window_s = max(0.0, (int(end_ns) - int(begin_ns)) / 1e9)
    in_window = [
        (t, p) for t, p in zip(times, powers) if int(begin_ns) <= t <= int(end_ns)
    ]
    guard_lookback_ns = int(float(proto["guard_lookback_s"]) * 1e9)
    guard_before = sum(
        1
        for t in times
        if int(begin_ns) - guard_lookback_ns <= t < int(begin_ns)
    )
    guard_after = sum(
        1
        for t in times
        if int(end_ns) < t <= int(end_ns) + guard_lookback_ns
    )

    audit: Dict[str, Any] = {
        "rail": rail,
        "window_begin_ns": int(begin_ns),
        "window_end_ns": int(end_ns),
        "window_s": window_s,
        "num_rail_samples_total": len(times),
        "num_samples_in_window": len(in_window),
        "guard_samples_before": guard_before,
        "guard_samples_after": guard_after,
        "boundary_policy": "truncate_no_extrapolation",
    }

    if len(in_window) < 2:
        audit.update(
            {
                "usable": False,
                "reason": (
                    f"only {len(in_window)} sample(s) inside the window; "
                    f"need >= 2 to integrate"
                ),
                "energy_j": None,
                "covered_s": 0.0,
                "coverage_ratio": 0.0,
                "avg_power_w": None,
                "j_per_request": None,
                "output_tok_per_j": None,
            }
        )
        return audit

    win_times = [t for t, _ in in_window]
    win_powers = [p for _, p in in_window]
    integral = integrate_energy(
        win_powers, win_times, max_gap_s=float(proto["max_sample_gap_s"])
    )
    covered_s = integral["covered_s"]
    coverage_ratio = (covered_s / window_s) if window_s > 0 else 0.0

    reasons: List[str] = []
    if len(in_window) < int(proto["min_power_samples"]):
        reasons.append(
            f"num_samples {len(in_window)} < min_power_samples "
            f"{proto['min_power_samples']}"
        )
    if integral["gap_violations"] > 0:
        reasons.append(
            f"{integral['gap_violations']} sample gap(s) exceed "
            f"max_sample_gap_s={proto['max_sample_gap_s']}"
        )
    if coverage_ratio < float(proto["min_coverage_ratio"]):
        reasons.append(
            f"coverage {coverage_ratio:.3f} < min_coverage_ratio "
            f"{proto['min_coverage_ratio']}"
        )
    if guard_before < int(proto["min_guard_samples"]):
        reasons.append("no guard sample before the window")
    if guard_after < int(proto["min_guard_samples"]):
        reasons.append("no guard sample after the window")
    if integral["negative_dt_intervals"] > 0:
        reasons.append(
            f"{integral['negative_dt_intervals']} non-monotonic timestamp pair(s)"
        )

    audit.update(
        {
            "usable": not reasons,
            "reason": "; ".join(reasons) if reasons else None,
            "energy_j": integral["energy_j"],
            "covered_s": covered_s,
            "integration_span_s": integral["span_s"],
            "coverage_ratio": coverage_ratio,
            "avg_power_w": integral["avg_power_w"],
            "first_sample_ns": win_times[0],
            "last_sample_ns": win_times[-1],
            "truncation_start_s": max(0.0, (win_times[0] - int(begin_ns)) / 1e9),
            "truncation_end_s": max(0.0, (int(end_ns) - win_times[-1]) / 1e9),
            "max_gap_s": integral["max_gap_s"],
            "gap_violations": integral["gap_violations"],
            "min_power_mw": min(win_powers),
            "max_power_mw": max(win_powers),
            "mean_power_mw": statistics.mean(win_powers),
        }
    )
    return audit


def energy_metrics(
    energy_j: Optional[float],
    *,
    num_requests: int,
    output_tokens: int,
    processed_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Derive the frozen per-request / per-token energy figures.

    ``output_tok/J`` and ``total_processed_tok/J`` answer different questions
    (decode work vs. total model work) and are deliberately *not* summed into
    one undefined "token/J".  Both are ``None`` when the integral is unusable.
    """
    metrics: Dict[str, Any] = {
        "energy_j": energy_j,
        "num_requests": num_requests,
        "output_tokens": output_tokens,
        "processed_tokens": processed_tokens,
        "j_per_request": None,
        "output_tok_per_j": None,
        "processed_tok_per_j": None,
    }
    if energy_j is None or energy_j <= 0 or num_requests <= 0:
        return metrics
    metrics["j_per_request"] = energy_j / num_requests
    metrics["output_tok_per_j"] = output_tokens / energy_j if output_tokens else None
    if processed_tokens is not None:
        metrics["processed_tok_per_j"] = processed_tokens / energy_j
    return metrics


def net_energy(
    active_energy_j: float,
    active_duration_s: float,
    idle_power_w: float,
) -> Dict[str, Any]:
    """Subtract a *matched* idle baseline: ``E_net = E_active - P_idle * T``.

    The absolute energy is always returned too, and a negative net energy is
    flagged rather than clipped: a negative value means the idle reference is
    not representative (different mode, different temperature, model unloaded),
    which is a finding, not a zero.
    """
    idle_energy_j = idle_power_w * active_duration_s
    net = active_energy_j - idle_energy_j
    return {
        "active_energy_j": active_energy_j,
        "active_duration_s": active_duration_s,
        "idle_power_w": idle_power_w,
        "idle_energy_j": idle_energy_j,
        "net_energy_j": net,
        "negative": net < 0,
        "net_fraction": (net / active_energy_j) if active_energy_j > 0 else None,
    }


# ── Thermal / throttle assessment ──────────────────────────────────────────


def _mean(values: Sequence[float]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def assess_thermal(
    per_request: Sequence[Mapping[str, Any]],
    *,
    cooling_engaged: bool = False,
    cooling_detail: Optional[Mapping[str, Any]] = None,
    protocol: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify a measured window as stable / drifting / thermally managed.

    Each element of ``per_request`` must carry at least ``latency_ms`` and,
    when available, ``temp_c`` (the hottest zone in that request's window) and
    ``gpu_freq_hz`` (mean observed GPU frequency in that window).

    Classification is deliberately staged so the common misreading
    "low frequency therefore thermal throttling" cannot happen:

    * ``thermal_limit_exceeded``  – peak temperature past the hard limit.
    * ``thermal_management_engaged`` – temperature rose **and** an independent
      thermal-management indicator (``*-throttle-alert`` cooling device) engaged.
    * ``thermal_suspect`` – temperature rise, frequency drop and latency rise
      all exceed their thresholds, but no independent state change was seen.
    * ``frequency_drift_non_thermal`` – frequency dropped without a temperature
      rise (governor/power-cap behaviour, not heat).
    * ``stable`` – none of the above.
    """
    proto = protocol or POWER_THERMAL_PROTOCOL
    temps = [float(r["temp_c"]) for r in per_request if r.get("temp_c") is not None]
    freqs = [
        float(r["gpu_freq_hz"]) for r in per_request if r.get("gpu_freq_hz")
    ]
    latencies = [
        float(r["latency_ms"]) for r in per_request if r.get("latency_ms") is not None
    ]

    temp_first = temps[0] if temps else None
    temp_last = temps[-1] if temps else None
    temp_delta = (
        (temp_last - temp_first) if temps and len(temps) >= 2 else 0.0
    )
    freq_first = freqs[0] if freqs else None
    freq_last = freqs[-1] if freqs else None
    freq_drop_frac = (
        (freq_first - freq_last) / freq_first
        if freq_first and len(freqs) >= 2
        else 0.0
    )
    lat_first = latencies[0] if latencies else None
    lat_last = latencies[-1] if latencies else None
    latency_rise_frac = (
        (lat_last - lat_first) / lat_first
        if lat_first and len(latencies) >= 2
        else 0.0
    )
    peak_temp = max(temps) if temps else None

    temp_thr = float(proto["temp_rise_threshold_c"])
    freq_thr = float(proto["freq_drop_frac_threshold"])
    lat_thr = float(proto["latency_rise_frac_threshold"])

    if peak_temp is not None and peak_temp >= float(proto["thermal_hard_limit_c"]):
        label = "thermal_limit_exceeded"
    elif (
        temp_delta >= temp_thr
        and freq_drop_frac >= freq_thr
        and cooling_engaged
    ):
        label = "thermal_management_engaged"
    elif (
        temp_delta >= temp_thr
        and freq_drop_frac >= freq_thr
        and latency_rise_frac >= lat_thr
    ):
        label = "thermal_suspect"
    elif freq_drop_frac >= freq_thr and temp_delta < temp_thr:
        label = "frequency_drift_non_thermal"
    else:
        label = "stable"

    return {
        "label": label,
        "isolate_from_baseline": label
        in {
            "thermal_limit_exceeded",
            "thermal_management_engaged",
            "thermal_suspect",
        },
        "peak_temp_c": peak_temp,
        "temp_first_c": temp_first,
        "temp_last_c": temp_last,
        "temp_delta_c": temp_delta,
        "gpu_freq_first_hz": freq_first,
        "gpu_freq_last_hz": freq_last,
        "gpu_freq_drop_frac": freq_drop_frac,
        "latency_first_ms": lat_first,
        "latency_last_ms": lat_last,
        "latency_rise_frac": latency_rise_frac,
        "cooling_engaged": cooling_engaged,
        "cooling_detail": dict(cooling_detail) if cooling_detail else None,
        "thresholds": {
            "temp_rise_threshold_c": temp_thr,
            "freq_drop_frac_threshold": freq_thr,
            "latency_rise_frac_threshold": lat_thr,
            "thermal_suspect_c": proto["thermal_suspect_c"],
            "thermal_hard_limit_c": proto["thermal_hard_limit_c"],
        },
    }


def cooling_state_summary(
    samples: Sequence[Mapping[str, Any]],
    *,
    protocol: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Reduce cooling-device ``cur_state`` samples to independent throttle evidence.

    Only the frozen ``throttle_alert_cooling_types`` count as thermal-management
    engagement; the always-running ``pwm-fan`` cooling device is reported
    separately so a spinning fan is never mistaken for throttling.
    """
    proto = protocol or POWER_THERMAL_PROTOCOL
    alert_types = set(proto["throttle_alert_cooling_types"])
    engaged_samples = 0
    per_type_max: Dict[str, int] = {}
    fan_states: List[int] = []

    for sample in samples:
        states = sample.get("cooling_cur_state", {}) or {}
        engaged = False
        for name, state in states.items():
            value = int(state)
            if name in alert_types:
                per_type_max[name] = max(per_type_max.get(name, 0), value)
                if value > 0:
                    engaged = True
            elif name == "pwm-fan":
                fan_states.append(value)
        if engaged:
            engaged_samples += 1

    return {
        "num_samples": len(samples),
        "alert_samples": engaged_samples,
        "engaged": engaged_samples > 0,
        "engaged_ratio": (engaged_samples / len(samples)) if samples else 0.0,
        "per_type_max_state": per_type_max,
        "fan_cur_state_min": min(fan_states) if fan_states else None,
        "fan_cur_state_max": max(fan_states) if fan_states else None,
        "note": (
            "throttle-alert cooling devices and the GPU devfreq cooling device "
            "are independent thermal-management state; the PWM fan is excluded "
            "because it runs at idle too."
        ),
    }


def temperature_summary(
    series: Sequence[Tuple[int, float]],
) -> Dict[str, Any]:
    """Summary of a ``(time_ns, celsius)`` series (one zone)."""
    values = [v for _, v in series]
    if not values:
        return {
            "count": 0,
            "start_c": None,
            "end_c": None,
            "min_c": None,
            "max_c": None,
            "mean_c": None,
            "delta_c": None,
        }
    return {
        "count": len(values),
        "start_c": values[0],
        "end_c": values[-1],
        "min_c": min(values),
        "max_c": max(values),
        "mean_c": statistics.mean(values),
        "delta_c": values[-1] - values[0],
    }


# ── Numerical closure ──────────────────────────────────────────────────────


def dimensional_consistency(
    *,
    energy_j: Optional[float],
    duration_s: Optional[float],
    output_tokens: Optional[int],
    output_tokens_per_s: Optional[float],
    protocol: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Close ``E/T``, ``tok/J`` and ``tok/s`` against each other.

    The three identities

        average_power_from_energy = E / T
        output_tok_per_j          = total_output_tokens / E
        output_tok_per_j          = output_tokens_per_s / average_power

    must agree for one window under one token convention.  A closure failure
    only proves an internal inconsistency (mW/W, ms/s, J/mJ mix-ups); it does
    **not** prove the sensor is accurate, and the report says so.
    """
    proto = protocol or POWER_THERMAL_PROTOCOL
    tol = float(proto["closure_rel_tol"])
    out: Dict[str, Any] = {
        "energy_j": energy_j,
        "duration_s": duration_s,
        "closure_rel_tol": tol,
        "comparable": (
            energy_j is not None
            and energy_j > 0
            and duration_s is not None
            and duration_s > 0
        ),
    }
    if not out["comparable"]:
        out.update(
            {
                "avg_power_from_energy_w": None,
                "output_tok_per_j_from_energy": None,
                "output_tok_per_j_from_throughput": None,
                "rel_gap": None,
                "closed": None,
            }
        )
        return out

    avg_power = energy_j / duration_s
    from_energy = (output_tokens / energy_j) if output_tokens else None
    from_throughput = (
        output_tokens_per_s / avg_power
        if output_tokens_per_s is not None and avg_power > 0
        else None
    )
    rel_gap = None
    if from_energy is not None and from_throughput is not None and from_energy > 0:
        rel_gap = abs(from_throughput - from_energy) / from_energy

    out.update(
        {
            "avg_power_from_energy_w": avg_power,
            "output_tok_per_j_from_energy": from_energy,
            "output_tok_per_j_from_throughput": from_throughput,
            "rel_gap": rel_gap,
            "closed": (rel_gap is not None and rel_gap <= tol),
        }
    )
    return out


# ── Device state parsing (nvpmodel / jetson_clocks text) ───────────────────

_NVPMODEL_MODE_RE = re.compile(r"Current mode:\s*NV Power Mode:\s*(\S+)")
_NVPMODEL_ID_RE = re.compile(r"^\s*(\d+)\s*$", re.MULTILINE)
_NVPMODEL_PARAM_RE = re.compile(
    r"PARAM\s+(\S+):\s*ARG\s+(\S+):\s*PATH\s+(\S+):\s*"
    r"REAL_VAL:\s*(\S+)\s+CONF_VAL:\s*(\S+)"
)


def parse_nvpmodel_query(text: str) -> Dict[str, Any]:
    """Parse ``nvpmodel -q --verbose`` into mode identity + parameter table."""
    name_match = _NVPMODEL_MODE_RE.search(text)
    id_match = _NVPMODEL_ID_RE.search(text)
    params: List[Dict[str, str]] = []
    for group, arg, path, real, conf in _NVPMODEL_PARAM_RE.findall(text):
        params.append(
            {
                "param": group,
                "arg": arg,
                "path": path,
                "real_value": real,
                "conf_value": conf,
            }
        )
    return {
        "mode_name": name_match.group(1) if name_match else None,
        "mode_id": int(id_match.group(1)) if id_match else None,
        "num_params": len(params),
        "params": params,
    }


_JC_CPU_RE = re.compile(
    r"^(cpu\d+):\s+Online=(\d+)\s+Governor=(\S+)\s+MinFreq=(\d+)\s+"
    r"MaxFreq=(\d+)\s+CurrentFreq=(\d+)",
    re.MULTILINE,
)
_JC_GPU_RE = re.compile(
    r"^GPU\s+MinFreq=(\d+)\s+MaxFreq=(\d+)\s+CurrentFreq=(\d+)", re.MULTILINE
)
_JC_EMC_RE = re.compile(
    r"^EMC\s+MinFreq=(\d+)\s+MaxFreq=(\d+)\s+CurrentFreq=(\d+)\s+FreqOverride=(\d+)",
    re.MULTILINE,
)
_JC_FAN_RE = re.compile(r"^FAN\s+(.+)$", re.MULTILINE)
_JC_MODE_RE = re.compile(r"^NV Power Mode:\s*(\S+)", re.MULTILINE)
_JC_TPC_RE = re.compile(r"^Active GPU TPCs:\s*(\d+)", re.MULTILINE)


_NVPMODEL_CONF_MODE_RE = re.compile(
    r"<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=(\S+)\s*>(.*?)(?=<|$)",
    re.DOTALL,
)
_NVPMODEL_CONF_LIMIT_RE = re.compile(r"^(\S+)\s+(MIN_FREQ|MAX_FREQ)\s+(-?\d+)$", re.MULTILINE)


def parse_nvpmodel_conf(text: str) -> Dict[str, Any]:
    """Enumerate the power modes the *device* actually supports.

    The protocol forbids inferring modes from historical ids, so the list is
    read from ``/etc/nvpmodel.conf`` on the machine under test rather than
    hard-coded.
    """
    modes: List[Dict[str, Any]] = []
    for mode_id, name, body in _NVPMODEL_CONF_MODE_RE.findall(text):
        limits: Dict[str, Dict[str, int]] = {}
        for group, kind, value in _NVPMODEL_CONF_LIMIT_RE.findall(body):
            limits.setdefault(group, {})[kind.lower()] = int(value)
        modes.append({"mode_id": int(mode_id), "name": name, "limits": limits})
    default_match = re.search(r"< PM_CONFIG DEFAULT=(\d+) >", text)
    return {
        "modes": sorted(modes, key=lambda m: m["mode_id"]),
        "default_mode_id": int(default_match.group(1)) if default_match else None,
    }


def parse_jetson_clocks_show(text: str) -> Dict[str, Any]:
    """Parse ``jetson_clocks --show`` into requested/observed clock state.

    A non-root invocation prints ``Error: Run this script(...) as a root user``
    and nothing else; that is reported as ``available=False`` with the raw
    error, never as "all clocks are zero".
    """
    if "as a root user" in text or "Permission denied" in text:
        return {
            "available": False,
            "error": text.strip().splitlines()[0] if text.strip() else "",
            "cpufreq": {},
            "gpu": None,
            "emc": None,
            "fan": None,
            "nvpmodel_name": None,
            "active_tpcs": None,
        }

    cpufreq: Dict[str, Any] = {}
    for cpu, online, governor, minf, maxf, curf in _JC_CPU_RE.findall(text):
        cpufreq[cpu] = {
            "online": int(online),
            "governor": governor,
            "min_freq_hz": int(minf),
            "max_freq_hz": int(maxf),
            "current_freq_hz": int(curf),
        }

    gpu_match = _JC_GPU_RE.search(text)
    emc_match = _JC_EMC_RE.search(text)
    fan_match = _JC_FAN_RE.search(text)
    mode_match = _JC_MODE_RE.search(text)
    tpc_match = _JC_TPC_RE.search(text)

    return {
        "available": True,
        "error": None,
        "cpufreq": cpufreq,
        "gpu": (
            {
                "min_freq_hz": int(gpu_match.group(1)),
                "max_freq_hz": int(gpu_match.group(2)),
                "current_freq_hz": int(gpu_match.group(3)),
            }
            if gpu_match
            else None
        ),
        "emc": (
            {
                "min_freq_hz": int(emc_match.group(1)),
                "max_freq_hz": int(emc_match.group(2)),
                "current_freq_hz": int(emc_match.group(3)),
                "freq_override": int(emc_match.group(4)),
            }
            if emc_match
            else None
        ),
        "fan": fan_match.group(1).strip() if fan_match else None,
        "nvpmodel_name": mode_match.group(1) if mode_match else None,
        "active_tpcs": int(tpc_match.group(1)) if tpc_match else None,
    }


def _flatten_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten a device-state snapshot to ``dotted.path -> scalar`` pairs."""
    flat: Dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, sub in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), sub)
        elif isinstance(value, list):
            for index, sub in enumerate(value):
                walk(f"{prefix}[{index}]", sub)
        else:
            flat[prefix] = value

    walk("", state)
    return flat


def device_state_diff(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Dict[str, Any]:
    """Diff two device-state snapshots and decide whether the original returned.

    Clock frequencies drift continuously, so this compares the *configured*
    fields (nvpmodel id/name, governors, min/max limits, frequency override)
    and additionally reports observed current-frequency deltas separately.
    """
    flat_before = _flatten_state(before)
    flat_after = _flatten_state(after)
    keys = sorted(set(flat_before) | set(flat_after))

    changed: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        old = flat_before.get(key)
        new = flat_after.get(key)
        if old != new:
            changed[key] = {"before": old, "after": new}

    volatile = ("current_freq_hz", "cur_freq", "pwm", "temperature", "temp")
    configured_changes = {
        key: value
        for key, value in changed.items()
        if not any(token in key for token in volatile)
    }

    return {
        "changed_count": len(changed),
        "changed": changed,
        "configured_changes": configured_changes,
        "restored": not configured_changes,
    }


# ── Cross-run aggregation ──────────────────────────────────────────────────


def cross_run_reduce(per_run_values: Sequence[Optional[float]]) -> Dict[str, Any]:
    """Reduce one metric across independent runs to median / min / max.

    With three independent processes this is an honest min-max spread, not a
    confidence interval; the report states that limitation explicitly.
    """
    clean = sorted(v for v in per_run_values if v is not None)
    if not clean:
        return {"runs": 0, "median": None, "min": None, "max": None, "values": []}
    return {
        "runs": len(clean),
        "median": statistics.median(clean),
        "min": clean[0],
        "max": clean[-1],
        "values": list(clean),
    }


def relative_spread(values: Sequence[Optional[float]]) -> Optional[float]:
    """``(max - min) / median`` for a metric, or None when undefined."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    med = statistics.median(clean)
    if med == 0:
        return None
    return (max(clean) - min(clean)) / abs(med)


def power_spectrum_note() -> str:
    """One-line reminder of what the integrated energy is and is not."""
    return (
        "Energy = integral of the instantaneous VDD_IN rail over this run's own "
        "window; the tool's cumulative average field is never integrated, and "
        "no per-request estimate is extrapolated from a single sample."
    )


__all__ = [
    "POWER_THERMAL_PROTOCOL",
    "RAIL_SCOPE",
    "align_and_integrate",
    "assess_thermal",
    "build_power_series",
    "cooling_state_summary",
    "cross_run_reduce",
    "device_state_diff",
    "dimensional_consistency",
    "energy_metrics",
    "integrate_energy",
    "net_energy",
    "parse_jetson_clocks_show",
    "parse_nvpmodel_conf",
    "parse_nvpmodel_query",
    "parse_tegrastats_line_v3",
    "parse_telemetry_stream",
    "power_spectrum_note",
    "rail_availability",
    "relative_spread",
    "temperature_summary",
    "trapezoid_energy_j",
]
