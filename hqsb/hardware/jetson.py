"""Jetson experiment protocol helpers.

Implements the S02 experimental protocol: read power mode, read thermal
zones, and cooldown between runs so consecutive benchmarks start from a
controlled thermal state. All functions are defensive — on non-Jetson or
unsupported systems they return ``None``/no-op rather than raising.
"""

from __future__ import annotations

import glob
import logging
import os
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

_NVPMODEL_MODE_PATH = "/etc/nvpmodel.conf"
_ACTIVE_MODE_PATH = "/sys/devices/gpu.0/status"


def read_thermal_zones() -> List[float]:
    """Read all thermal zone temperatures in Celsius from sysfs.

    Returns a list of temperatures (one per zone). Empty on non-Linux or
    when no thermal zones are present.
    """
    temps: List[float] = []
    for zone_path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try:
            with open(zone_path, "rb") as fh:
                raw_bytes = fh.read()
            # Some Jetson sysfs files return None/empty on a read race;
            # skip those defensively.
            if not raw_bytes:
                continue
            # sysfs thermal zone "temp" is millidegrees Celsius (ASCII).
            raw = raw_bytes.decode("ascii").strip()
            temps.append(int(raw) / 1000.0)
        except (OSError, ValueError, TypeError, UnicodeDecodeError, AttributeError):
            continue
    return temps


def max_temperature_c() -> Optional[float]:
    """Return the maximum thermal zone temperature in Celsius, if any."""
    temps = read_thermal_zones()
    return max(temps) if temps else None


def cooldown(
    target_c: float,
    *,
    timeout_s: float = 600.0,
    poll_interval_s: float = 2.0,
) -> bool:
    """Block until all thermal zones are <= ``target_c``.

    Args:
        target_c: Target temperature ceiling in Celsius.
        timeout_s: Maximum time to wait (seconds).
        poll_interval_s: Poll interval (seconds).

    Returns:
        True if cooled to target; False on timeout (or when no thermal
        sensors are available, which is treated as "nothing to cool").
    """
    deadline = time.monotonic() + timeout_s
    while True:
        temps = read_thermal_zones()
        if not temps:
            logger.debug("No thermal zones detected; skipping cooldown.")
            return True
        current = max(temps)
        if current <= target_c:
            logger.info("Cooldown reached: %.1f C <= %.1f C", current, target_c)
            return True
        if time.monotonic() >= deadline:
            logger.warning(
                "Cooldown timeout: %.1f C > %.1f C after %.0f s",
                current, target_c, timeout_s,
            )
            return False
        time.sleep(poll_interval_s)


def active_power_mode() -> Optional[int]:
    """Return the active nvpmodel power mode, if detectable.

    Returns None when the sysfs interface is unavailable (non-Jetson).
    """
    for candidate in (
        "/sys/devices/gpu.0/status",
        "/proc/device-tree/",
    ):
        if os.path.exists(candidate):
            break
    else:
        return None

    # The nvpmodel mode is not always exposed as a single sysfs file;
    # fall back to the marker used by JetPack's `nvpmodel -q`.
    marker = "/var/lib/nvpmodel/status"
    if os.path.isfile(marker):
        try:
            with open(marker, encoding="utf-8") as fh:
                text = fh.read()
            import re

            match = re.search(r"Mode:\s*(\d+)", text)
            if match:
                return int(match.group(1))
        except OSError:
            return None
    return None


def is_jetson() -> bool:
    """Return True when running on a Jetson device (device-tree compatible)."""
    try:
        with open("/proc/device-tree/compatible", "rb") as fh:
            content = fh.read().decode("utf-8", errors="ignore")
        return "jetson" in content.lower() or "tegra" in content.lower()
    except OSError:
        return False


__all__ = [
    "active_power_mode",
    "cooldown",
    "is_jetson",
    "max_temperature_c",
    "read_thermal_zones",
]
