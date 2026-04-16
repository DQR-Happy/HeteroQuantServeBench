"""Tegrastats output parser for Jetson resource metrics.

Parses raw tegrastats output lines into structured metrics including
RAM usage, GPU utilization, temperature, and power consumption.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List, Optional


# ── Regex patterns for tegrastats output fields ────────────────────

_RAM_PATTERN = re.compile(r"RAM\s+(\d+)/(\d+)MB")

_GPU_UTIL_PATTERN = re.compile(r"GR3D_FREQ\s+(\d+)%")

# tegrastats may report GPU temperature as "gpu@XX.XC" or "GPU@XX.XC"
# depending on JetPack version
_TEMP_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"gpu@([\d.]+)C", re.IGNORECASE),
    re.compile(r"GPU@([\d.]+)C", re.IGNORECASE),
]

# Power rail: VDD_IN is the main SoC input power on Orin platforms
# Other possible rails: VDD_CPU_CV, VDD_GPU_SOC, VDD_SOC
_POWER_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"VDD_IN\s+(\d+)/(\d+)mW"),
    re.compile(r"VDD_IN\s+(\d+)mW"),
]

_CPU_TEMP_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"CPU@([\d.]+)C", re.IGNORECASE),
    re.compile(r"cpu@([\d.]+)C", re.IGNORECASE),
    re.compile(r"thermal@([\d.]+)C", re.IGNORECASE),
]

_SWAP_PATTERN = re.compile(r"SWAP\s+(\d+)/(\d+)MB")

_CPU_UTIL_PATTERN = re.compile(r"CPU\s+\[([^\]]+)\]")


def parse_tegrastats_line(line: str) -> Dict[str, Any]:
    """Parse a single tegrastats output line into structured metrics.

    Extracts: RAM usage, SWAP usage, GPU utilization, GPU/CPU
    temperature, and VDD_IN power consumption.

    Args:
        line: A raw tegrastats output line (e.g.,
            ``"RAM 1234/7890MB ... GR3D_FREQ 45% ..."``).

    Returns:
        Dictionary with any successfully parsed fields. Possible keys:
            - ``ram_used_mb``: RAM in use (MiB).
            - ``ram_total_mb``: Total RAM (MiB).
            - ``swap_used_mb``: SWAP in use (MiB).
            - ``swap_total_mb``: Total SWAP (MiB).
            - ``gpu_util_pct``: GPU utilization percentage.
            - ``gpu_temp_c``: GPU temperature (Celsius).
            - ``cpu_temp_c``: CPU temperature (Celsius).
            - ``power_mw``: Instantaneous VDD_IN power (mW).
            - ``power_avg_mw``: Average VDD_IN power (mW) if reported.
            - ``cpu_util_pct``: CPU utilization string.

        Missing fields are simply omitted from the result.
    """
    result: Dict[str, Any] = {}

    # RAM
    ram_match = _RAM_PATTERN.search(line)
    if ram_match:
        result["ram_used_mb"] = int(ram_match.group(1))
        result["ram_total_mb"] = int(ram_match.group(2))

    # SWAP
    swap_match = _SWAP_PATTERN.search(line)
    if swap_match:
        result["swap_used_mb"] = int(swap_match.group(1))
        result["swap_total_mb"] = int(swap_match.group(2))

    # GPU utilization
    gpu_match = _GPU_UTIL_PATTERN.search(line)
    if gpu_match:
        result["gpu_util_pct"] = int(gpu_match.group(1))

    # GPU temperature
    for pattern in _TEMP_PATTERNS:
        temp_match = pattern.search(line)
        if temp_match:
            result["gpu_temp_c"] = float(temp_match.group(1))
            break

    # CPU temperature
    for pattern in _CPU_TEMP_PATTERNS:
        temp_match = pattern.search(line)
        if temp_match:
            result["cpu_temp_c"] = float(temp_match.group(1))
            break

    # Power (VDD_IN)
    for pattern in _POWER_PATTERNS:
        power_match = pattern.search(line)
        if power_match:
            result["power_mw"] = int(power_match.group(1))
            if power_match.lastindex and power_match.lastindex >= 2:
                result["power_avg_mw"] = int(power_match.group(2))
            break

    # CPU utilization
    cpu_match = _CPU_UTIL_PATTERN.search(line)
    if cpu_match:
        result["cpu_util_pct"] = cpu_match.group(1)

    return result


def compute_power_summary(
    records: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute power and energy statistics from parsed tegrastats records.

    Uses trapezoidal integration over monotonic timestamps to
    calculate total energy consumption.

    Args:
        records: List of parsed tegrastats records, each must have
            ``time_ns`` and optionally ``power_mw`` fields.

    Returns:
        Dictionary with:
            - ``num_samples``: Number of records with power data.
            - ``avg_power_w``: Average power in Watts.
            - ``peak_power_w``: Peak power in Watts.
            - ``energy_j``: Total energy in Joules (trapezoidal integration).
            - ``duration_s``: Monitoring duration in seconds.
    """
    power_values: List[float] = []
    timestamps_ns: List[int] = []

    for rec in records:
        if "power_mw" in rec:
            power_values.append(rec["power_mw"])
            timestamps_ns.append(rec.get("time_ns", 0))

    if not power_values:
        return {
            "num_samples": 0,
            "avg_power_w": 0.0,
            "peak_power_w": 0.0,
            "energy_j": 0.0,
            "duration_s": 0.0,
        }

    # Trapezoidal integration
    energy_mj: float = 0.0  # mW * s -> mJ
    for i in range(len(power_values) - 1):
        dt_s = (timestamps_ns[i + 1] - timestamps_ns[i]) / 1e9
        avg_power_mw = (power_values[i] + power_values[i + 1]) / 2.0
        energy_mj += avg_power_mw * dt_s

    energy_j = energy_mj / 1000.0

    duration_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9

    return {
        "num_samples": len(power_values),
        "avg_power_w": statistics.mean(power_values) / 1000.0,
        "peak_power_w": max(power_values) / 1000.0,
        "energy_j": energy_j,
        "duration_s": duration_s,
    }


def compute_resource_summary(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute comprehensive resource utilization summary.

    Args:
        records: List of parsed tegrastats records.

    Returns:
        Dictionary with aggregate GPU utilization, temperature,
        RAM usage, and power statistics.
    """
    result: Dict[str, Any] = {}

    # GPU utilization
    gpu_vals = [r["gpu_util_pct"] for r in records if "gpu_util_pct" in r]
    if gpu_vals:
        result["avg_gpu_util_pct"] = statistics.mean(gpu_vals)
        result["peak_gpu_util_pct"] = max(gpu_vals)

    # GPU temperature
    gpu_temp_vals = [r["gpu_temp_c"] for r in records if "gpu_temp_c" in r]
    if gpu_temp_vals:
        result["avg_gpu_temp_c"] = statistics.mean(gpu_temp_vals)
        result["peak_gpu_temp_c"] = max(gpu_temp_vals)

    # CPU temperature
    cpu_temp_vals = [r["cpu_temp_c"] for r in records if "cpu_temp_c" in r]
    if cpu_temp_vals:
        result["avg_cpu_temp_c"] = statistics.mean(cpu_temp_vals)
        result["peak_cpu_temp_c"] = max(cpu_temp_vals)

    # RAM
    ram_vals = [r["ram_used_mb"] for r in records if "ram_used_mb" in r]
    if ram_vals:
        result["avg_ram_used_mb"] = statistics.mean(ram_vals)
        result["peak_ram_used_mb"] = max(ram_vals)

    # Power & Energy
    power_summary = compute_power_summary(records)
    result.update(power_summary)

    return result
