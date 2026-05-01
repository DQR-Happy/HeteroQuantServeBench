"""Unit tests for :mod:`hqsb.benchmark.tegrastats_parser`.

Pure-Python tests validating the Jetson ``tegrastats`` line parser and the
trapezoidal energy integration. No GPU or tegrastats binary is required.
"""

from __future__ import annotations

import pytest

from hqsb.benchmark.tegrastats_parser import (
    compute_power_summary,
    compute_resource_summary,
    parse_tegrastats_line,
)

# A representative tegrastats line modeled on JetPack 6 Orin output.
_SAMPLE_LINE = (
    "RAM 1234/7890MB (lfb 1234x4MB) SWAP 40/11264MB (cached 40MB) "
    "CPU [45%@1497,30%@1190,0%@729,0%@729,0%@729,0%@729] "
    "GR3D_FREQ 62%@612 "
    "gpu@55.5C CPU@52.3C thermal@51.2C "
    "VDD_IN 12345/13456mW VDD_CPU_CV 1234mW VDD_GPU_SOC 5678mW"
)


class TestParseTegrastatsLine:
    def test_parses_all_fields(self):
        result = parse_tegrastats_line(_SAMPLE_LINE)
        assert result["ram_used_mb"] == 1234
        assert result["ram_total_mb"] == 7890
        assert result["swap_used_mb"] == 40
        assert result["swap_total_mb"] == 11264
        assert result["gpu_util_pct"] == 62
        assert result["gpu_temp_c"] == pytest.approx(55.5)
        assert result["cpu_temp_c"] == pytest.approx(52.3)
        assert result["power_mw"] == 12345
        assert result["power_avg_mw"] == 13456
        assert result["cpu_util_pct"] == "45%@1497,30%@1190,0%@729,0%@729,0%@729,0%@729"

    def test_missing_fields_are_omitted(self):
        result = parse_tegrastats_line("RAM 100/200MB")
        assert result == {"ram_used_mb": 100, "ram_total_mb": 200}

    def test_empty_line(self):
        assert parse_tegrastats_line("") == {}

    def test_gpu_temperature_case_insensitive(self):
        assert parse_tegrastats_line("GPU@61.2C")["gpu_temp_c"] == pytest.approx(61.2)

    def test_power_single_value_fallback(self):
        # Some JetPack versions report VDD_IN with a single value.
        result = parse_tegrastats_line("VDD_IN 7000mW")
        assert result["power_mw"] == 7000
        assert "power_avg_mw" not in result


class TestComputePowerSummary:
    def _records(self, times_ms, powers_mw):
        return [
            {"time_ns": t * 1_000_000, "power_mw": p}
            for t, p in zip(times_ms, powers_mw)
        ]

    def test_trapezoidal_integration(self):
        records = self._records([0, 1000], [1000, 2000])
        summary = compute_power_summary(records)
        assert summary["num_samples"] == 2
        assert summary["avg_power_w"] == pytest.approx(1.5)
        assert summary["peak_power_w"] == pytest.approx(2.0)
        # Trapezoid of 1.0s with avg 1500 mW = 1.5 J.
        assert summary["energy_j"] == pytest.approx(1.5)
        assert summary["duration_s"] == pytest.approx(1.0)

    def test_no_power_records(self):
        summary = compute_power_summary([{"time_ns": 0}])
        assert summary["num_samples"] == 0
        assert summary["energy_j"] == 0.0

    def test_empty_input(self):
        summary = compute_power_summary([])
        assert summary["num_samples"] == 0


class TestComputeResourceSummary:
    def test_aggregates_gpu_metrics(self):
        records = [
            {"gpu_util_pct": 50, "gpu_temp_c": 55.0, "ram_used_mb": 1000},
            {"gpu_util_pct": 70, "gpu_temp_c": 57.0, "ram_used_mb": 1100},
        ]
        summary = compute_resource_summary(records)
        assert summary["avg_gpu_util_pct"] == pytest.approx(60.0)
        assert summary["peak_gpu_util_pct"] == 70
        assert summary["avg_gpu_temp_c"] == pytest.approx(56.0)
        assert summary["peak_ram_used_mb"] == 1100

    def test_empty_records_yields_zero_power_summary(self):
        # With no records, the power summary is still attached (all zeros).
        summary = compute_resource_summary([])
        assert summary["num_samples"] == 0
        assert summary["avg_power_w"] == 0.0
        assert summary["energy_j"] == 0.0
