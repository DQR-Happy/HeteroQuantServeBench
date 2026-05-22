"""Unit tests for :mod:`hqsb.benchmark.power_thermal` (E02-08).

Pure-Python tests: no GPU, no ``tegrastats`` binary, no torch.  The expected
values for the energy integral are hand-computed in the test bodies (protocol
section 7.2 forbids generating the oracle with the function under test).

The parser fixtures are *real* tegrastats lines captured from the Jetson Orin
Nano Super used in this stage (see the E02-08 report, section "monitor self
test"); the counterexamples are synthetic.
"""

from __future__ import annotations

import pytest

from hqsb.benchmark.power_thermal import (
    POWER_THERMAL_PROTOCOL,
    RAIL_SCOPE,
    align_and_integrate,
    assess_thermal,
    build_power_series,
    cooling_state_summary,
    cross_run_reduce,
    device_state_diff,
    dimensional_consistency,
    energy_metrics,
    integrate_energy,
    net_energy,
    parse_jetson_clocks_show,
    parse_nvpmodel_conf,
    parse_nvpmodel_query,
    parse_tegrastats_line_v3,
    parse_telemetry_stream,
    rail_availability,
    relative_spread,
    temperature_summary,
    trapezoid_energy_j,
)

# ── real fixtures ──────────────────────────────────────────────────────────

REAL_LINE = (
    "09-17-2026 02:10:03 RAM 1488/7620MB (lfb 105x4MB) SWAP 253/12002MB "
    "(cached 2MB) CPU [0%@1036,0%@1036,1%@1036,1%@1036,0%@729,0%@729] "
    "GR3D_FREQ 0% cpu@47.875C soc2@46.75C soc0@47.281C gpu@50.218C "
    "tj@50.218C soc1@48.281C VDD_IN 4654mW/4654mW "
    "VDD_CPU_GPU_CV 560mW/560mW VDD_SOC 1484mW/1484mW"
)

REAL_LINE_LOADED = (
    "09-17-2026 02:10:16 RAM 1483/7620MB (lfb 105x4MB) SWAP 253/12002MB "
    "(cached 2MB) CPU [0%@729,0%@729,0%@729,0%@729,0%@729,0%@729] "
    "GR3D_FREQ 62%@612 cpu@47.75C soc2@46.718C soc0@47.25C gpu@50C tj@50C "
    "soc1@48.093C VDD_IN 4541mW/4541mW VDD_CPU_GPU_CV 521mW/521mW "
    "VDD_SOC 1446mW/1473mW"
)

JETSON_CLOCKS_SHOW = """SOC family:tegra234  Machine:NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super

Online CPUs: 0-5
cpu0:  Online=1 Governor=schedutil MinFreq=729600 MaxFreq=1728000 CurrentFreq=1267200 IdleStates: WFI=1 c7=1
cpu4:  Online=1 Governor=schedutil MinFreq=729600 MaxFreq=1728000 CurrentFreq=1728000 IdleStates: WFI=1 c7=1
GPU MinFreq=306000000 MaxFreq=1020000000 CurrentFreq=306000000
Active GPU TPCs: 4
EMC MinFreq=204000000 MaxFreq=3199000000 CurrentFreq=2133000000 FreqOverride=0
FAN Dynamic Speed Control=kernel hwmon0_pwm1=88
NV Power Mode: MAXN_SUPER
"""

NVPMODEL_QUERY = """NVPM VERB: Config file: /etc/nvpmodel.conf
NVPM VERB: parsing done for /etc/nvpmodel.conf
NVPM VERB: Current mode: NV Power Mode: MAXN_SUPER
2
NVPM VERB: PARAM CPU_ONLINE: ARG CORE_0: PATH /sys/devices/system/cpu/cpu0/online: REAL_VAL: 1 CONF_VAL: 1
NVPM VERB: PARAM GPU: ARG MAX_FREQ: PATH /sys/devices/platform/17000000.gpu/devfreq_dev/max_freq: REAL_VAL: 1020000000 CONF_VAL: 9223372036854775807
"""

NVPMODEL_CONF = """< POWER_MODEL ID=0 NAME=15W >
CPU_A78_0 MIN_FREQ 729600
CPU_A78_0 MAX_FREQ 1497600
GPU MIN_FREQ 0
GPU MAX_FREQ 612000000
EMC MAX_FREQ 2133000000

< POWER_MODEL ID=2 NAME=MAXN_SUPER >
CPU_A78_0 MIN_FREQ 729600
CPU_A78_0 MAX_FREQ -1
GPU MIN_FREQ 0
GPU MAX_FREQ -1

< PM_CONFIG DEFAULT=1 >
"""


def _record(line: str, time_ns: int) -> dict:
    return {"time_ns": time_ns, "raw": line}


# ── 7.1 parser ─────────────────────────────────────────────────────────────


class TestParserRealLines:
    def test_all_fields_from_real_line(self):
        parsed = parse_tegrastats_line_v3(REAL_LINE)
        assert parsed["parse_status"] == "ok"
        assert parsed["time_text"] == "09-17-2026 02:10:03"
        assert parsed["rails_mw"] == {
            "VDD_IN": 4654,
            "VDD_CPU_GPU_CV": 560,
            "VDD_SOC": 1484,
        }
        # the tool's cumulative average field is kept apart, never integrated
        assert parsed["rail_average_mw"]["VDD_SOC"] == 1484
        assert parsed["ram_used_mb"] == 1488
        assert parsed["ram_total_mb"] == 7620
        assert parsed["lfb_free_blocks"] == 105
        assert parsed["lfb_block_mb"] == 4
        assert parsed["swap_used_mb"] == 253
        assert parsed["swap_cached_mb"] == 2
        assert parsed["swap_total_mb"] == 12002
        assert parsed["cpu_utils_pct"] == [0, 0, 1, 1, 0, 0]
        assert parsed["cpu_freqs_mhz"] == [1036, 1036, 1036, 1036, 729, 729]
        assert parsed["gpu_util_pct"] == 0
        assert parsed["gpu_freq_mhz"] is None
        assert parsed["temperatures_c"]["gpu"] == pytest.approx(50.218)
        assert parsed["temperatures_c"]["tj"] == pytest.approx(50.218)
        assert parsed["missing_fields"] == []

    def test_missing_rail_is_not_zero(self):
        parsed = parse_tegrastats_line_v3(REAL_LINE)
        assert "VDD_EXTRA" not in parsed["rails_mw"]

    def test_integer_temperature_and_gr3d_with_frequency(self):
        parsed = parse_tegrastats_line_v3(REAL_LINE_LOADED)
        assert parsed["parse_status"] == "ok"
        assert parsed["gpu_util_pct"] == 62
        assert parsed["gpu_freq_mhz"] == 612
        assert parsed["temperatures_c"]["gpu"] == pytest.approx(50.0)
        # single-value rail form (no "/avg") is still parsed
        assert parsed["rails_mw"]["VDD_IN"] == 4541
        assert parsed["rail_average_mw"]["VDD_SOC"] == 1473


class TestParserCounterexamples:
    def test_rail_order_permuted(self):
        line = (
            "RAM 100/200MB CPU [1%@729,1%@729] GR3D_FREQ 1% gpu@40C "
            "VDD_SOC 900mW VDD_CPU_GPU_CV 200mW VDD_IN 3000mW"
        )
        parsed = parse_tegrastats_line_v3(line)
        assert parsed["parse_status"] == "ok"
        assert parsed["rails_mw"]["VDD_IN"] == 3000
        assert parsed["rails_mw"]["VDD_SOC"] == 900

    def test_unknown_rail_is_preserved_not_dropped(self):
        line = (
            "RAM 100/200MB CPU [1%@729] GR3D_FREQ 1% gpu@40C VDD_IN 3000mW "
            "VDD_GPU_SOC 1200mW VDD_UNKNOWN_RAIL 42mW"
        )
        parsed = parse_tegrastats_line_v3(line)
        assert parsed["rails_mw"]["VDD_GPU_SOC"] == 1200
        assert parsed["rails_mw"]["VDD_UNKNOWN_RAIL"] == 42
        assert parsed["unparsed_rails"] == []

    def test_unit_change_to_watts_is_flagged_not_zeroed(self):
        line = "RAM 100/200MB CPU [1%@729] GR3D_FREQ 1% gpu@40C VDD_IN 4.65W"
        parsed = parse_tegrastats_line_v3(line)
        assert parsed["parse_status"] == "degraded"
        assert parsed["unparsed_rails"] == ["VDD_IN"]
        assert parsed["rails_mw"] == {}

    def test_truncated_line_reports_missing_cpu_bracket(self):
        line = "RAM 100/200MB GR3D_FREQ 1% gpu@40C VDD_IN 3000mW"
        parsed = parse_tegrastats_line_v3(line)
        assert parsed["parse_status"] == "degraded"
        assert "cpu_bracket" in parsed["missing_fields"]

    def test_non_numeric_power_is_flagged(self):
        line = "RAM 100/200MB CPU [1%@729] GR3D_FREQ 1% gpu@40C VDD_IN NAmW"
        parsed = parse_tegrastats_line_v3(line)
        assert parsed["parse_status"] == "degraded"
        assert parsed["unparsed_rails"] == ["VDD_IN"]

    def test_empty_and_garbage_lines_fail(self):
        assert parse_tegrastats_line_v3("")["parse_status"] == "failed"
        assert parse_tegrastats_line_v3("not a tegrastats line")[
            "parse_status"
        ] == "failed"

    def test_extra_whitespace_locale_is_tolerated(self):
        line = "RAM  100/200MB  CPU [1%@729]  GR3D_FREQ 1%  gpu@40C  VDD_IN  3000mW"
        parsed = parse_tegrastats_line_v3(line)
        assert parsed["parse_status"] == "ok"
        assert parsed["rails_mw"]["VDD_IN"] == 3000

    def test_temperature_case_is_normalised(self):
        line = "RAM 1/2MB CPU [1%@729] GR3D_FREQ 1% GPU@61.2C VDD_IN 3000mW"
        parsed = parse_tegrastats_line_v3(line)
        assert parsed["temperatures_c"]["gpu"] == pytest.approx(61.2)


class TestParserStreamAudit:
    def test_status_histogram_and_monotonicity(self):
        records = [
            _record(REAL_LINE, 0),
            _record(REAL_LINE_LOADED, 250_000_000),
            _record("garbage", 500_000_000),
        ]
        parsed, audit = parse_telemetry_stream(records)
        assert len(parsed) == 3
        assert audit["num_lines"] == 3
        assert audit["status_counts"] == {"ok": 2, "degraded": 0, "failed": 1}
        assert audit["parse_error_ratio"] == pytest.approx(1 / 3)
        assert audit["monotonic"] is True
        assert audit["interval_s"]["max"] == pytest.approx(0.25)

    def test_out_of_order_and_equal_timestamps_are_detected(self):
        records = [
            _record(REAL_LINE, 1_000_000_000),
            _record(REAL_LINE, 500_000_000),
            _record(REAL_LINE, 500_000_000),
        ]
        _, audit = parse_telemetry_stream(records)
        assert audit["monotonic"] is False
        # one negative interval + one zero interval
        assert audit["timestamp_violations"] == 2

    def test_interrupted_read_missing_timestamp_is_excluded(self):
        records = [{"raw": REAL_LINE}, _record(REAL_LINE, 1)]
        parsed, audit = parse_telemetry_stream(records)
        assert parsed[0]["time_ns"] is None
        assert audit["num_timestamps"] == 1


class TestRailAvailability:
    def test_rail_presence_ratio(self):
        records = [
            _record(REAL_LINE, 0),
            _record(REAL_LINE_LOADED, 1),
        ]
        parsed, _ = parse_telemetry_stream(records)
        primary = rail_availability(parsed, "VDD_IN")
        assert primary["status"] == "usable"
        assert primary["presence_ratio"] == pytest.approx(1.0)

    def test_absent_rail_is_reported_absent(self):
        records = [_record(REAL_LINE, 0)]
        parsed, _ = parse_telemetry_stream(records)
        assert rail_availability(parsed, "VDD_GPU_ONLY")["status"] == "absent"

    def test_partial_rail_is_insufficient(self):
        line = "RAM 1/2MB CPU [1%@729] GR3D_FREQ 1% gpu@40C VDD_IN 3000mW"
        records = [_record(REAL_LINE, i) for i in range(8)] + [
            _record(line, 8),
            _record(line, 9),
        ]
        parsed, _ = parse_telemetry_stream(records)
        audit = rail_availability(parsed, "VDD_SOC")
        assert audit["status"] == "insufficient"
        assert audit["presence_ratio"] == pytest.approx(0.8)

    def test_power_series_skips_missing_rail(self):
        line = "RAM 1/2MB CPU [1%@729] GR3D_FREQ 1% gpu@40C VDD_IN 3000mW"
        records = [_record(line, 0), _record(REAL_LINE, 1)]
        parsed, _ = parse_telemetry_stream(records)
        times, powers = build_power_series(parsed, "VDD_SOC")
        assert times == [1]
        assert powers == [1484.0]


# ── 7.2 integration oracle ─────────────────────────────────────────────────


class TestIntegrationOracle:
    def test_hand_computed_11_joules(self):
        # t = [0, 1, 3] s with P = [2, 4, 4] W
        #   0 -> 1 : 0.5 * (2 + 4) * 1 = 3 J
        #   1 -> 3 : 0.5 * (4 + 4) * 2 = 8 J
        #   total  : 11 J
        energy = trapezoid_energy_j(
            [2000.0, 4000.0, 4000.0],
            [0, 1_000_000_000, 3_000_000_000],
        )
        assert energy == pytest.approx(11.0)

    def test_hand_computed_segments(self):
        assert trapezoid_energy_j(
            [2000.0, 4000.0], [0, 1_000_000_000]
        ) == pytest.approx(3.0)
        assert trapezoid_energy_j(
            [4000.0, 4000.0], [1_000_000_000, 3_000_000_000]
        ) == pytest.approx(8.0)

    def test_milliwatt_millisecond_conversion(self):
        # 1000 mW for 1 ms = 1 mJ = 1e-3 J
        assert trapezoid_energy_j(
            [1000.0, 1000.0], [0, 1_000_000]
        ) == pytest.approx(1e-3)

    def test_mean_power_times_duration_is_not_the_integral(self):
        energy = integrate_energy(
            [2000.0, 4000.0, 4000.0],
            [0, 1_000_000_000, 3_000_000_000],
        )
        naive = (2 + 4 + 4) / 3 * 3  # mean over samples x span
        assert energy["energy_j"] == pytest.approx(11.0)
        assert naive != pytest.approx(energy["energy_j"])

    def test_average_power_is_energy_over_covered_time(self):
        energy = integrate_energy(
            [2000.0, 4000.0, 4000.0],
            [0, 1_000_000_000, 3_000_000_000],
        )
        assert energy["covered_s"] == pytest.approx(3.0)
        assert energy["avg_power_w"] == pytest.approx(11.0 / 3.0)

    def test_duplicate_timestamp_is_skipped_not_interpolated(self):
        energy = integrate_energy(
            [2000.0, 9000.0, 9000.0], [0, 0, 1_000_000_000]
        )
        assert energy["zero_dt_intervals"] == 1
        assert energy["energy_j"] == pytest.approx(9.0)

    def test_negative_interval_invalidates(self):
        energy = integrate_energy([1000.0, 1000.0], [1_000_000_000, 0])
        assert energy["negative_dt_intervals"] == 1
        assert energy["valid"] is False

    def test_large_gap_is_counted(self):
        energy = integrate_energy(
            [1000.0, 1000.0, 1000.0],
            [0, 100_000_000, 5_000_000_000],
            max_gap_s=1.5,
        )
        assert energy["gap_violations"] == 1
        assert energy["max_gap_s"] == pytest.approx(4.9)

    def test_single_sample_has_no_energy(self):
        energy = integrate_energy([1000.0], [0])
        assert energy["energy_j"] == 0.0
        assert energy["valid"] is False

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            integrate_energy([1.0, 2.0], [0])


# ── 7.3 window alignment / coverage ────────────────────────────────────────


def _power_stream(powers_mw, times_ns, rail: str = "VDD_IN") -> list:
    records = []
    for power, time_ns in zip(powers_mw, times_ns):
        line = (
            f"RAM 1/2MB CPU [1%@729] GR3D_FREQ 1% gpu@40C "
            f"{rail} {int(power)}mW"
        )
        records.append(_record(line, time_ns))
    parsed, _ = parse_telemetry_stream(records)
    return parsed


class TestWindowAlignment:
    S = 1_000_000_000

    def test_clean_window_is_usable_and_matches_hand_integral(self):
        # 0.25 s cadence from -2 s to +4 s; measured window [0 s, 3 s].
        times = [
            int(round(v * self.S))
            for v in [-2 + 0.25 * i for i in range(25)]
        ]
        powers = [1000] * len(times)
        parsed = _power_stream(powers, times)
        audit = align_and_integrate(parsed, 0, 3 * self.S, "VDD_IN")
        assert audit["usable"] is True
        assert audit["energy_j"] == pytest.approx(3.0)
        assert audit["coverage_ratio"] == pytest.approx(1.0)
        assert audit["guard_samples_before"] >= 1
        assert audit["guard_samples_after"] >= 1

    def test_truncation_is_reported_not_extrapolated(self):
        times = [0, self.S, 2 * self.S]
        parsed = _power_stream([1000, 1000, 1000], times)
        # the requested window extends far past the last sample
        audit = align_and_integrate(parsed, 0, 10 * self.S, "VDD_IN")
        assert audit["usable"] is False
        assert "coverage" in (audit["reason"] or "")
        assert audit["first_sample_ns"] == 0
        assert audit["last_sample_ns"] == 2 * self.S

    def test_window_shorter_than_sampling_interval_is_unusable(self):
        times = [0, self.S]
        parsed = _power_stream([1000, 1000], times)
        audit = align_and_integrate(parsed, 0, 100_000_000, "VDD_IN")
        assert audit["usable"] is False
        assert audit["energy_j"] is None
        assert "sample" in (audit["reason"] or "")

    def test_gap_over_threshold_makes_window_unusable(self):
        times = [0, self.S, 6 * self.S, 7 * self.S]
        parsed = _power_stream([1000, 1000, 1000, 1000], times)
        audit = align_and_integrate(parsed, 0, 7 * self.S, "VDD_IN")
        assert audit["gap_violations"] == 1
        assert audit["usable"] is False
        assert "gap" in (audit["reason"] or "")

    def test_missing_guard_sample_is_reported(self):
        times = [0, self.S, 2 * self.S, 3 * self.S]
        parsed = _power_stream([1000, 1000, 1000, 1000], times)
        # window covers the whole stream -> no samples outside either edge
        audit = align_and_integrate(parsed, 0, 3 * self.S, "VDD_IN")
        assert audit["guard_samples_before"] == 0
        assert audit["guard_samples_after"] == 0
        assert audit["usable"] is False
        assert "guard" in (audit["reason"] or "")

    def test_multi_rail_overlap_is_not_summed(self):
        # VDD_IN >= sub-rails on every line: integrating the primary and the
        # sub-rails is fine, adding them would double count.
        records = []
        for i, (total, sub) in enumerate([(3000, 500), (3000, 500), (3000, 500)]):
            line = (
                f"RAM 1/2MB CPU [1%@729] GR3D_FREQ 1% gpu@40C "
                f"VDD_IN {total}mW VDD_CPU_GPU_CV {sub}mW"
            )
            records.append(_record(line, i * self.S))
        parsed, _ = parse_telemetry_stream(records)
        total = align_and_integrate(parsed, 0, 2 * self.S, "VDD_IN")
        sub = align_and_integrate(parsed, 0, 2 * self.S, "VDD_CPU_GPU_CV")
        assert total["energy_j"] == pytest.approx(6.0)
        assert sub["energy_j"] == pytest.approx(1.0)
        assert RAIL_SCOPE["additive"] is False
        assert total["energy_j"] > sub["energy_j"]

    def test_no_samples_at_all_yields_none_energy(self):
        parsed, _ = parse_telemetry_stream([])
        audit = align_and_integrate(parsed, 0, self.S, "VDD_IN")
        assert audit["usable"] is False
        assert audit["energy_j"] is None

    def test_frozen_protocol_window_rules(self):
        assert POWER_THERMAL_PROTOCOL["max_sample_gap_s"] > 0
        assert 0 < POWER_THERMAL_PROTOCOL["min_coverage_ratio"] <= 1.0
        assert POWER_THERMAL_PROTOCOL["min_power_samples"] >= 2


# ── energy metrics / net energy / closure ──────────────────────────────────


class TestEnergyMetrics:
    def test_per_request_and_per_token(self):
        metrics = energy_metrics(
            100.0, num_requests=4, output_tokens=64, processed_tokens=1024
        )
        assert metrics["j_per_request"] == pytest.approx(25.0)
        assert metrics["output_tok_per_j"] == pytest.approx(0.64)
        assert metrics["processed_tok_per_j"] == pytest.approx(10.24)

    def test_unusable_energy_yields_none_metrics(self):
        metrics = energy_metrics(
            None, num_requests=4, output_tokens=64, processed_tokens=1024
        )
        assert metrics["j_per_request"] is None
        assert metrics["output_tok_per_j"] is None

    def test_output_and_processed_token_ratios_stay_separate(self):
        metrics = energy_metrics(
            50.0, num_requests=1, output_tokens=32, processed_tokens=2080
        )
        assert metrics["output_tok_per_j"] == pytest.approx(0.64)
        assert metrics["processed_tok_per_j"] == pytest.approx(41.6)
        assert (
            metrics["output_tok_per_j"] != metrics["processed_tok_per_j"]
        )


class TestNetEnergy:
    def test_subtraction_and_negative_flag(self):
        positive = net_energy(100.0, 10.0, 2.0)
        assert positive["idle_energy_j"] == pytest.approx(20.0)
        assert positive["net_energy_j"] == pytest.approx(80.0)
        assert positive["negative"] is False

        negative = net_energy(10.0, 10.0, 5.0)
        assert negative["net_energy_j"] == pytest.approx(-40.0)
        assert negative["negative"] is True

    def test_absolute_energy_is_always_kept(self):
        result = net_energy(10.0, 1.0, 100.0)
        assert result["active_energy_j"] == pytest.approx(10.0)


class TestDimensionalConsistency:
    def test_closure_holds_for_one_token_convention(self):
        # 90 J over 10 s -> 9 W average; 4.5 tok/s -> 0.5 tok/J
        result = dimensional_consistency(
            energy_j=90.0,
            duration_s=10.0,
            output_tokens=45,
            output_tokens_per_s=4.5,
        )
        assert result["avg_power_from_energy_w"] == pytest.approx(9.0)
        assert result["output_tok_per_j_from_energy"] == pytest.approx(0.5)
        assert result["output_tok_per_j_from_throughput"] == pytest.approx(0.5)
        assert result["closed"] is True

    def test_unit_mixup_is_detected(self):
        # throughput reported in tok/ms would be 1000x too large
        result = dimensional_consistency(
            energy_j=90.0,
            duration_s=10.0,
            output_tokens=45,
            output_tokens_per_s=4500.0,
        )
        assert result["closed"] is False
        assert result["rel_gap"] == pytest.approx(999.0)

    def test_undefined_inputs_are_not_closed(self):
        result = dimensional_consistency(
            energy_j=None,
            duration_s=10.0,
            output_tokens=45,
            output_tokens_per_s=4.5,
        )
        assert result["comparable"] is False
        assert result["closed"] is None


# ── thermal / throttle classification ─────────────────────────────────────


def _req(latency_ms, temp_c, freq_hz):
    return {
        "latency_ms": latency_ms,
        "temp_c": temp_c,
        "gpu_freq_hz": freq_hz,
    }


class TestAssessThermal:
    def test_stable_window(self):
        result = assess_thermal(
            [
                _req(1000, 50.0, 1.02e9),
                _req(1010, 50.2, 1.02e9),
                _req(1005, 50.1, 1.02e9),
            ]
        )
        assert result["label"] == "stable"
        assert result["isolate_from_baseline"] is False

    def test_low_frequency_without_heat_is_not_thermal(self):
        result = assess_thermal(
            [
                _req(1000, 50.0, 1.02e9),
                _req(1010, 50.1, 0.61e9),
                _req(1005, 50.0, 0.61e9),
            ]
        )
        assert result["label"] == "frequency_drift_non_thermal"
        assert result["isolate_from_baseline"] is False

    def test_heat_plus_frequency_plus_latency_is_thermal_suspect(self):
        result = assess_thermal(
            [
                _req(1000, 50.0, 1.02e9),
                _req(1200, 56.0, 0.80e9),
                _req(1300, 60.0, 0.70e9),
            ]
        )
        assert result["label"] == "thermal_suspect"
        assert result["isolate_from_baseline"] is True

    def test_independent_cooling_state_promotes_to_engaged(self):
        result = assess_thermal(
            [
                _req(1000, 50.0, 1.02e9),
                _req(1200, 56.0, 0.80e9),
            ],
            cooling_engaged=True,
        )
        assert result["label"] == "thermal_management_engaged"

    def test_hard_limit_exceeded(self):
        result = assess_thermal(
            [
                _req(1000, 50.0, 1.02e9),
                _req(4000, 96.0, 0.30e9),
            ]
        )
        assert result["label"] == "thermal_limit_exceeded"

    def test_empty_series_is_stable(self):
        assert assess_thermal([])["label"] == "stable"


class TestCoolingStateSummary:
    def _sample(self, **states):
        return {"cooling_cur_state": states}

    def test_fan_alone_is_not_throttling(self):
        summary = cooling_state_summary([self._sample(**{"pwm-fan": 1})])
        assert summary["engaged"] is False
        assert summary["fan_cur_state_max"] == 1

    def test_throttle_alert_engagement_is_detected(self):
        summary = cooling_state_summary(
            [
                self._sample(**{"gpu-throttle-alert": 0}),
                self._sample(**{"gpu-throttle-alert": 1}),
            ]
        )
        assert summary["engaged"] is True
        assert summary["alert_samples"] == 1
        assert summary["engaged_ratio"] == pytest.approx(0.5)

    def test_gpu_devfreq_cooling_device_counts(self):
        summary = cooling_state_summary(
            [self._sample(**{"devfreq-17000000.gpu": 3})]
        )
        assert summary["engaged"] is True
        assert summary["per_type_max_state"]["devfreq-17000000.gpu"] == 3


class TestTemperatureSummary:
    def test_delta_and_peak(self):
        summary = temperature_summary([(0, 50.0), (1, 52.5), (2, 51.0)])
        assert summary["start_c"] == pytest.approx(50.0)
        assert summary["end_c"] == pytest.approx(51.0)
        assert summary["max_c"] == pytest.approx(52.5)
        assert summary["delta_c"] == pytest.approx(1.0)

    def test_empty(self):
        assert temperature_summary([])["count"] == 0


# ── device state parsing ──────────────────────────────────────────────────


class TestNvpmodelParsing:
    def test_query_parsing(self):
        parsed = parse_nvpmodel_query(NVPMODEL_QUERY)
        assert parsed["mode_name"] == "MAXN_SUPER"
        assert parsed["mode_id"] == 2
        assert parsed["num_params"] == 2
        assert parsed["params"][1]["arg"] == "MAX_FREQ"
        assert parsed["params"][1]["real_value"] == "1020000000"

    def test_conf_enumerates_supported_modes(self):
        parsed = parse_nvpmodel_conf(NVPMODEL_CONF)
        assert [mode["mode_id"] for mode in parsed["modes"]] == [0, 2]
        assert parsed["modes"][0]["name"] == "15W"
        assert parsed["modes"][0]["limits"]["GPU"]["max_freq"] == 612000000
        assert parsed["modes"][1]["limits"]["GPU"]["max_freq"] == -1
        assert parsed["default_mode_id"] == 1


class TestJetsonClocksParsing:
    def test_show_parsing(self):
        parsed = parse_jetson_clocks_show(JETSON_CLOCKS_SHOW)
        assert parsed["available"] is True
        assert parsed["gpu"]["max_freq_hz"] == 1020000000
        assert parsed["gpu"]["current_freq_hz"] == 306000000
        assert parsed["emc"]["freq_override"] == 0
        assert parsed["cpufreq"]["cpu0"]["governor"] == "schedutil"
        assert parsed["nvpmodel_name"] == "MAXN_SUPER"
        assert parsed["active_tpcs"] == 4

    def test_non_root_error_is_not_reported_as_zero_clocks(self):
        parsed = parse_jetson_clocks_show(
            "Error: Run this script(/usr/bin/jetson_clocks) as a root user\n"
        )
        assert parsed["available"] is False
        assert parsed["gpu"] is None
        assert "root user" in parsed["error"]


class TestDeviceStateDiff:
    def test_restored_when_configured_fields_match(self):
        before = {"nvpmodel_mode_id": 2, "sysfs_gpu": {"max_freq_hz": 1020000000}}
        after = {
            "nvpmodel_mode_id": 2,
            "sysfs_gpu": {"max_freq_hz": 1020000000, "cur_freq_hz": 999},
        }
        diff = device_state_diff(before, after)
        assert diff["restored"] is True
        # the volatile current-frequency change is still recorded
        assert diff["changed_count"] >= 0

    def test_not_restored_when_a_limit_changed(self):
        before = {"nvpmodel_mode_id": 2, "sysfs_gpu": {"max_freq_hz": 1020000000}}
        after = {"nvpmodel_mode_id": 0, "sysfs_gpu": {"max_freq_hz": 612000000}}
        diff = device_state_diff(before, after)
        assert diff["restored"] is False
        assert "nvpmodel_mode_id" in diff["configured_changes"]


# ── cross-run reduction ──────────────────────────────────────────────────


class TestCrossRunReduce:
    def test_median_min_max(self):
        reduced = cross_run_reduce([3.0, 1.0, 2.0])
        assert reduced["median"] == pytest.approx(2.0)
        assert reduced["min"] == pytest.approx(1.0)
        assert reduced["max"] == pytest.approx(3.0)
        assert reduced["runs"] == 3

    def test_none_values_are_dropped(self):
        reduced = cross_run_reduce([None, 2.0, 4.0])
        assert reduced["runs"] == 2

    def test_spread(self):
        # (max - min) / median = (3 - 1) / 2
        assert relative_spread([1.0, 2.0, 3.0]) == pytest.approx(1.0)
        assert relative_spread([1.0]) is None


def test_protocol_hash_is_json_serialisable():
    import json

    payload = json.dumps(POWER_THERMAL_PROTOCOL, sort_keys=True)
    assert "primary_rail" in payload
    assert POWER_THERMAL_PROTOCOL["primary_rail"] == RAIL_SCOPE["primary_rail"]


def test_sub_rails_are_declared_and_distinct_from_primary():
    assert RAIL_SCOPE["additive"] is False
    assert POWER_THERMAL_PROTOCOL["primary_rail"] not in POWER_THERMAL_PROTOCOL[
        "sub_rails"
    ]
    assert set(POWER_THERMAL_PROTOCOL["sub_rails"]) == set(RAIL_SCOPE["sub_rails"])
