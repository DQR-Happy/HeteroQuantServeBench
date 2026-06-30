"""Unit tests for the E12-06 energy windows, integration and efficiency metrics."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.evaluation import energy as en
from hqsb.evaluation.layers import TokenAccounting

pytestmark = pytest.mark.unit


def _series(**overrides: object) -> en.PowerSeries:
    payload = {
        "run_id": "run-1",
        "meter_id": "meter-1",
        "samples": (
            en.EnergySample(t_ns=0, power_w=100.0, energy_j=0.0),
            en.EnergySample(t_ns=100_000_000, power_w=200.0, energy_j=20.0),
            en.EnergySample(t_ns=200_000_000, power_w=150.0, energy_j=35.0),
        ),
    }
    payload.update(overrides)
    return en.PowerSeries(**payload)  # type: ignore[arg-type]


def _meter(**overrides: object) -> en.MeterCapability:
    payload = {
        "meter_id": "meter-1",
        "field": "accelerator.power_W",
        "semantics": "instantaneous",
        "boundary": "device",
        "unit": "W",
        "resolution": 0.001,
        "sample_period_s": 0.1,
        "availability": "available",
        "permission": "user",
        "wrap_or_reset": "none",
    }
    payload.update(overrides)
    return en.MeterCapability(**payload)  # type: ignore[arg-type]


class TestBoundariesAndWindows:
    def test_boundary_mixing_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            en.assert_same_boundary([{"boundary": "device"}, {"boundary": "node"}])
        en.assert_same_boundary([{"boundary": "device"}, {"boundary": "device"}])

    def test_each_boundary_has_its_legal_claim(self) -> None:
        assert en.BOUNDARY_LEGAL_CLAIMS["device"] == "device energy only"
        assert "scenario" in en.BOUNDARY_LEGAL_CLAIMS["facility_adjusted"]

    def test_window_policies_declare_backlog_and_cooldown(self) -> None:
        assert en.policy("ready_to_complete").includes_backlog is True
        assert en.policy("steady_measurement").includes_backlog is False
        assert en.policy("cooldown_tail").includes_cooldown is True
        with pytest.raises(ConfigError):
            en.policy("whenever")

    def test_idle_policy_validation(self) -> None:
        assert en.IdlePolicy("model_loaded", 30.0, "median").validate() == []
        assert en.IdlePolicy("idle-ish", 30.0, "median").validate()
        assert en.IdlePolicy("model_loaded", 0.0, "median").validate()
        assert en.IdlePolicy("model_loaded", 30.0, "guess").validate()


class TestMetersAndSeries:
    def test_meter_capability_requires_semantics_boundary_unit_resolution(self) -> None:
        assert _meter().validate() == []
        problems = _meter(semantics="", boundary="planet", resolution=0.0).validate()
        assert any("semantics" in item for item in problems)
        assert any("boundary" in item for item in problems)
        assert any("resolution" in item for item in problems)

    def test_can_measure_requires_the_matching_boundary(self) -> None:
        assert en.can_measure(_meter(), "device") is True
        assert en.can_measure(_meter(), "node") is False
        assert en.can_measure(_meter(availability="permission_denied"), "device") is False

    def test_series_flags_unsorted_and_duplicate_timestamps(self) -> None:
        unsorted_series = _series(
            samples=(
                en.EnergySample(t_ns=10, power_w=1.0),
                en.EnergySample(t_ns=5, power_w=1.0),
            )
        )
        assert any("monotonic" in item for item in unsorted_series.validate())
        duplicated = _series(
            samples=(
                en.EnergySample(t_ns=10, power_w=1.0),
                en.EnergySample(t_ns=10, power_w=1.0),
            )
        )
        assert any("duplicate" in item for item in duplicated.validate())

    def test_non_ok_samples_are_kept_and_flagged(self) -> None:
        series = _series(
            samples=(
                en.EnergySample(t_ns=0, power_w=1.0),
                en.EnergySample(t_ns=1, power_w=None, status="GAP"),
            )
        )
        assert any("flagged" in item for item in series.validate())


class TestIntegration:
    def test_trapezoid_integral_matches_hand_calculation(self) -> None:
        report = en.trapezoid_integral(_series(), t0_ns=0, t1_ns=200_000_000)
        # 0.1s * (100+200)/2 + 0.1s * (200+150)/2 = 15 + 17.5 = 32.5 J
        assert report["energy_j"] == pytest.approx(32.5)
        assert report["method"] == "trapezoid"

    def test_integral_without_samples_is_unavailable(self) -> None:
        sparse = _series(samples=(en.EnergySample(t_ns=0, power_w=1.0),))
        report = en.trapezoid_integral(sparse, t0_ns=0, t1_ns=1)
        assert report["status"] == en.MISSING_MEASUREMENT_UNAVAILABLE

    def test_accumulator_delta_and_reset_detection(self) -> None:
        report = en.accumulator_delta(_series(), t0_ns=0, t1_ns=200_000_000)
        assert report["energy_j"] == pytest.approx(35.0)
        reset = _series(
            samples=(
                en.EnergySample(t_ns=0, energy_j=35.0),
                en.EnergySample(t_ns=1_000_000, energy_j=1.0),
            )
        )
        broken = en.accumulator_delta(reset, t0_ns=0, t1_ns=1_000_000)
        assert "reset/wrap" in broken["reason"]

    def test_accumulator_vs_integral_cross_check(self) -> None:
        ok = en.cross_check_accumulator_vs_integral(accumulator_j=32.0, integral_j=32.5, tolerance_j=1.0)
        assert ok["ok"] is True
        bad = en.cross_check_accumulator_vs_integral(accumulator_j=10.0, integral_j=32.5, tolerance_j=1.0)
        assert bad["status"] == "INCONSISTENT"

    def test_alignment_maps_raw_events_and_needs_both_events(self) -> None:
        window = en.align_windows(
            _series(),
            events={"steady_window_start": 1000, "steady_window_end": 2000},
            policy_id="steady_measurement",
            offset_s=0.5,
        )
        assert window["raw_t0_ns"] == 1000
        assert window["aligned_t0_ns"] == 1000 + 500_000_000
        with pytest.raises(ConfigError):
            en.align_windows(_series(), events={"steady_window_start": 1}, policy_id="steady_measurement")


class TestQualityGate:
    def _gate(self, **overrides: object) -> dict:
        payload = {
            "meter": _meter(),
            "window": {"aligned_t0_ns": 0, "aligned_t1_ns": 10, "gap_fraction": 0.0},
            "accounting_closed": True,
            "same_run": True,
            "idle_policy_identical": True,
            "unexplained_contamination": False,
            "interval_j": 1.0,
            "resolution_j": 0.001,
            "comparison_ok": True,
        }
        payload.update(overrides)
        return en.power_quality_gate(**payload)  # type: ignore[arg-type]

    def test_clean_measurement_passes_every_check(self) -> None:
        report = self._gate()
        assert report["ok"] is True
        assert report["passed"] == len(en.POWER_QUALITY_CHECKS)

    def test_each_failure_mode_is_reported(self) -> None:
        gaps = self._gate(window={"aligned_t0_ns": 0, "aligned_t1_ns": 10, "gap_fraction": 0.5})
        assert gaps["ok"] is False
        unsynced = self._gate(same_run=False)
        assert any(row["check_id"] == "performance_and_power_same_run" and row["status"] == "FAIL" for row in unsynced["checks"])
        imprecise = self._gate(interval_j=0.0001, resolution_j=0.001)
        assert any(row["check_id"] == "interval_not_falsely_precise" and row["status"] == "FAIL" for row in imprecise["checks"])


class TestEnergyAndEfficiency:
    def test_idle_reference_reports_drift(self) -> None:
        pre = _series(
            samples=(en.EnergySample(t_ns=0, power_w=50.0), en.EnergySample(t_ns=1, power_w=50.0))
        )
        post = _series(
            samples=(en.EnergySample(t_ns=2, power_w=52.0), en.EnergySample(t_ns=3, power_w=52.0))
        )
        report = en.idle_reference(pre, post, idle=en.IdlePolicy("model_loaded", 30.0, "median"))
        assert report["idle_power_w"] == pytest.approx(51.0)
        assert report["drift_rel"] < 0.1
        unstable = en.idle_reference(
            pre, _series(samples=(en.EnergySample(t_ns=2, power_w=90.0),)), idle=en.IdlePolicy("model_loaded", 30.0, "median")
        )
        assert unstable["status"] == "UNSTABLE"

    def test_incremental_energy_keeps_raw_value_and_clip_policy(self) -> None:
        report = en.incremental_energy(
            load_energy_j=32.5, idle_power_w=50.0, window_s=0.2, idle=en.IdlePolicy("model_loaded", 30.0, "median")
        )
        assert report["raw_incremental_j"] == pytest.approx(22.5)
        clipped = en.incremental_energy(
            load_energy_j=1.0,
            idle_power_w=50.0,
            window_s=1.0,
            idle=en.IdlePolicy("model_loaded", 30.0, "median", clip="clip_negative"),
        )
        assert clipped["incremental_energy_j"] == 0.0

    def test_total_energy_prefers_accumulator_then_integral(self) -> None:
        accumulator = {"status": "OK", "energy_j": 30.0, "sample_count": 3}
        integral = {"status": "OK", "energy_j": 32.5, "sample_count": 3, "gap_fraction": 0.0}
        assert en.total_energy(accumulator=accumulator, integral=integral)["method"] == "accumulator"
        assert en.total_energy(integral=integral)["method"] == "integration"
        unavailable = en.total_energy()
        assert unavailable["status"] == en.MISSING_MEASUREMENT_UNAVAILABLE

    def _accounting(self, **overrides: object) -> TokenAccounting:
        payload = {
            "run_id": "run-1",
            "prompt_tokens": 10,
            "requested_output_tokens": 20,
            "generated_tokens_before_stop": 20,
            "accepted_output_tokens": 20,
            "served_output_tokens": 20,
            "speculative_draft_tokens": 5,
            "admitted_requests": 2,
            "completed_requests": 2,
        }
        payload.update(overrides)
        return TokenAccounting(**payload)  # type: ignore[arg-type]

    def test_efficiency_uses_compliant_denominators_only(self) -> None:
        metrics = en.efficiency_metrics(
            accounting=self._accounting(), total_energy_j=40.0, boundary="device", average_power_w=100.0
        )
        assert metrics["j_per_request"] == pytest.approx(20.0)
        assert metrics["j_per_token"] == pytest.approx(2.0)
        assert metrics["excluded_from_denominator"]["draft_tokens"] == 5

    def test_efficiency_refuses_unclosed_or_empty_accounting(self) -> None:
        unclosed = self._accounting(admitted_requests=5)
        report = en.efficiency_metrics(accounting=unclosed, total_energy_j=1.0, boundary="device")
        assert report["status"] == en.MISSING_MEASUREMENT_UNAVAILABLE
        empty = self._accounting(
            accepted_output_tokens=0,
            generated_tokens_before_stop=0,
            requested_output_tokens=0,
            completed_requests=0,
            admitted_requests=0,
        )
        report = en.efficiency_metrics(accounting=empty, total_energy_j=1.0, boundary="device")
        assert report["missing_reason"] == "QUALITY_GATE_FAILED"

    def test_efficiency_without_energy_is_unavailable(self) -> None:
        report = en.efficiency_metrics(accounting=self._accounting(), total_energy_j=None, boundary="device")
        assert report["status"] == en.MISSING_MEASUREMENT_UNAVAILABLE

    def test_phase_decomposition_refuses_too_coarse_resolution(self) -> None:
        report = en.phase_decomposition(
            series=_series(),
            phase_windows={"prefill": (0, 10)},
            min_resolution_ns=1_000_000,
        )
        assert report["status"] == "NOT_APPLICABLE"
        ok = en.phase_decomposition(
            series=_series(),
            phase_windows={"steady": (0, 200_000_000)},
            min_resolution_ns=1_000,
        )
        assert ok["status"] == "OK"

    def test_uncertainty_never_smaller_than_resolution(self) -> None:
        coarse = en.uncertainty_budget(
            {"total_energy_j": 32.5, "gap_fraction": 0.0, "resolution_j": 5.0}
        )
        assert coarse["falsely_precise"] is True
        assert coarse["interval_wider_than_resolution"] is False
        fine = en.uncertainty_budget(
            {"total_energy_j": 32.5, "resolution_j": 0.001, "alignment_uncertainty_j": 2.0}
        )
        assert fine["falsely_precise"] is False

    def test_boundary_sensitivity_flags_conditional_conclusions(self) -> None:
        report = en.boundary_sensitivity(total=32.5, incremental=10.0, pre_idle=50.0, post_idle=52.0)
        assert report["status"] == "CONDITIONAL"

    def test_collector_overhead_and_unavailable_path(self) -> None:
        overhead = en.collector_overhead_check(on_latency_rel_delta=0.2, on_cpu_rel_delta=0.01, threshold_rel=0.05)
        assert overhead["exceeded"] is True
        unavailable = en.unavailable_energy(
            meter_id="m", reason="no telemetry permission", missing_fields=["accelerator.power_W"]
        )
        assert unavailable["estimate"] is None
        with pytest.raises(ConfigError):
            en.unavailable_energy(meter_id="m", reason="", missing_fields=["x"])

    def test_energy_measurement_row_validation(self) -> None:
        row = en.EnergyMeasurement(
            energy_measurement_id="e1",
            run_id="run-1",
            comparison_id="cmp",
            meter_id="meter-1",
            boundary="device",
            method="integration",
            total_energy_j=32.5,
            incremental_energy_j=22.5,
            accepted_compliant_tokens=20,
            j_per_token=32.5 / 20,
        )
        assert row.validate() == []
        only_total = en.EnergyMeasurement(
            energy_measurement_id="e2",
            run_id="run-1",
            comparison_id="cmp",
            meter_id="meter-1",
            boundary="device",
            total_energy_j=1.0,
        )
        assert any("incremental" in item for item in only_total.validate())
        fabricated = en.EnergyMeasurement(
            energy_measurement_id="e3",
            run_id="run-1",
            comparison_id="cmp",
            meter_id="m",
            boundary="device",
            j_per_token=1.0,
        )
        assert any("fabricated" in item for item in fabricated.validate())
        throttled = en.EnergyMeasurement(
            energy_measurement_id="e4",
            run_id="run-1",
            comparison_id="cmp",
            meter_id="m",
            boundary="device",
            total_energy_j=1.0,
            incremental_energy_j=0.5,
            throttle_flags=("THERMAL_THROTTLE",),
            quality_status="pass",
        )
        assert any("throttled" in item for item in throttled.validate())


class TestSmoke:
    def test_protocol_steps_are_complete(self) -> None:
        assert len(en.PROTOCOL_STEPS) == 36
        assert all(interfaces for _, _, interfaces in en.PROTOCOL_STEPS)

    def test_smoke_self_check_is_labelled(self) -> None:
        result = en.smoke_self_check()
        assert result["status"] == "smoke"
        assert result["claim_allowed"] is False
        assert result["integral_positive"] is True
        assert result["denominator_excludes_draft"] is True
        assert result["boundary_mixing_rejected"] is True
