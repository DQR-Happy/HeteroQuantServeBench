"""Hardware-specific helpers (Jetson experiment protocol)."""

from hqsb.hardware.jetson import (
    active_power_mode,
    cooldown,
    is_jetson,
    max_temperature_c,
    read_thermal_zones,
)
from hqsb.hardware.probe import cuda_device_probe

__all__ = [
    "active_power_mode",
    "cooldown",
    "cuda_device_probe",
    "is_jetson",
    "max_temperature_c",
    "read_thermal_zones",
]
