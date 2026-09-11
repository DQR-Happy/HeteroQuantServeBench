"""Board sensors and Jetson SoC sensors have different measurement scopes."""
import subprocess

import pytest

from hqsb.console.telemetry import nvidia_sample, parse_nvidia_sample


def test_board_scope_and_memory_units():
    sample = parse_nvidia_sample("GPU-test, NVIDIA RTX 4090 D, 42, 50, 123.4, 1024, 24564\n")
    assert sample["power_w"] == 123.4 and sample["power_scope"] == "GPU board"
    assert sample["gpu_memory_used_bytes"] == 1024**3
    assert sample["gpu_util_pct"] == 42 and sample["sensor_device"] == "physical_gpu_0"


def test_unsupported_readings_are_missing():
    sample = parse_nvidia_sample("GPU-test, test, N/A, [Not Supported], nan, -1, 24564\n")
    assert sample["gpu_util_pct"] is None
    assert sample["gpu_temp_c"] is None and sample["power_w"] is None
    assert sample["gpu_memory_used_bytes"] is None
    with pytest.raises(ValueError, match="SHAPE"):
        parse_nvidia_sample("bad row")


def test_sensor_timeout_is_bounded_and_not_a_fake_zero(monkeypatch):
    def fail(command, **kwargs):
        assert kwargs["timeout"] == 2
        raise subprocess.TimeoutExpired(command, 2)
    monkeypatch.setattr(subprocess, "run", fail)
    assert nvidia_sample() == {}
