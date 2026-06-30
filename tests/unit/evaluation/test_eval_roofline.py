"""Unit tests for the E12-05 roofline / Amdahl prediction machinery."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.evaluation import roofline as rl

pytestmark = pytest.mark.unit


def _compute_roof(level: str = "SUSTAINABLE_MICRO") -> rl.Roof:
    if level == "SUSTAINABLE_MICRO":
        return rl.sustainable_roof(
            kind="compute",
            dtype="fp16",
            value=1.0e14,
            unit="op/s",
            interval_low=0.9e14,
            interval_high=1.05e14,
            conditions={"power_cap_w": 250},
        )
    return rl.theoretical_roof(
        kind="compute",
        dtype="fp16",
        value=2.0e14,
        unit="op/s",
        source_id="vendor",
        assumptions=("dense", "boost clock"),
    )


def _bandwidth_roof() -> rl.Roof:
    return rl.sustainable_roof(
        kind="bandwidth",
        dtype="fp16",
        value=1.0e12,
        unit="byte/s",
        interval_low=0.95e12,
        interval_high=1.02e12,
        conditions={"level": "HBM"},
    )


class TestRoofs:
    def test_theoretical_roof_cannot_carry_a_prediction(self) -> None:
        with pytest.raises(ConfigError):
            rl.assert_roof_usable_for_prediction("THEORETICAL_VENDOR")
        rl.assert_roof_usable_for_prediction("SUSTAINABLE_MICRO")

    def test_theoretical_roof_requires_assumptions(self) -> None:
        with pytest.raises(ConfigError):
            rl.theoretical_roof(
                kind="compute", dtype="fp16", value=1.0, unit="op/s", source_id="vendor", assumptions=()
            )

    def test_sustainable_roof_requires_interval_and_conditions(self) -> None:
        roof = rl.Roof(
            roof_id="r1",
            kind="compute",
            dtype="fp16",
            value=1.0e14,
            unit="op/s",
            evidence_level="SUSTAINABLE_MICRO",
        )
        problems = roof.validate()
        assert any("conditions" in item for item in problems)
        assert any("interval" in item for item in problems)

    def test_vendor_peak_without_assumptions_is_invalid(self) -> None:
        roof = rl.Roof(
            roof_id="r2",
            kind="compute",
            dtype="fp16",
            value=1.0e14,
            unit="op/s",
            evidence_level="THEORETICAL_VENDOR",
        )
        assert any("assumptions" in item for item in roof.validate())

    def test_unknown_memory_level_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            rl.is_device_local("CACHE")


class TestOperationsAndTraffic:
    def test_gemm_useful_operations(self) -> None:
        report = rl.useful_operations("gemm", dims={"M": 2, "N": 3, "K": 4})
        assert report["useful_operations"] == pytest.approx(48.0)
        assert "2 * M * N * K" in report["definition"]

    def test_unknown_semantic_op_and_missing_dims_are_rejected(self) -> None:
        with pytest.raises(ConfigError):
            rl.useful_operations("mystery", dims={})
        with pytest.raises(ConfigError):
            rl.useful_operations("gemm", dims={"M": 1})

    def test_logical_and_expected_traffic_differ_by_conversions(self) -> None:
        logical = rl.logical_bytes("elementwise", dims={"N": 1000.0})
        expected = rl.expected_bytes("elementwise", dims={"N": 1000.0}, layout_conversions=1)
        assert expected["bytes"] > logical["bytes"]

    def test_measured_traffic_requires_counter_provenance(self) -> None:
        with pytest.raises(ConfigError):
            rl.measured_bytes(counter_value=1.0, counter_source="", counter_semantics="", level="HBM")
        row = rl.measured_bytes(
            counter_value=1024.0, counter_source="ncu:l2_bytes", counter_semantics="read+write", level="L2"
        )
        assert row["level"] == "L2"

    def test_mixed_traffic_kinds_are_rejected(self) -> None:
        with pytest.raises(ConfigError):
            rl.assert_bytes_not_mixed({"traffic_kind": "logic"})
        rl.assert_bytes_not_mixed(
            rl.logical_bytes("elementwise", dims={"N": 1.0}),
            rl.measured_bytes(counter_value=1.0, counter_source="s", counter_semantics="rw", level="HBM"),
        )

    def test_arithmetic_intensity_flavors_are_separate(self) -> None:
        assert rl.arithmetic_intensity(100.0, 10.0, flavor="logical")["arithmetic_intensity"] == 10.0
        with pytest.raises(ConfigError):
            rl.arithmetic_intensity(100.0, 10.0, flavor="combined")
        with pytest.raises(ConfigError):
            rl.arithmetic_intensity(100.0, 0.0, flavor="logical")


class TestRooflineBoundAndPoints:
    def test_bound_picks_the_lower_of_compute_and_bandwidth(self) -> None:
        compute = _compute_roof()
        bandwidth = _bandwidth_roof()
        bound = rl.roofline_bound(
            arithmetic_intensity_value=10.0, compute_roof=compute, bandwidth_roof=bandwidth
        )
        assert bound["bound"] == pytest.approx(1.0e13)
        assert bound["bound_kind"] == "bandwidth"

    def test_bound_rejects_wrong_kinds_and_units(self) -> None:
        compute = _compute_roof()
        bandwidth = _bandwidth_roof()
        with pytest.raises(ConfigError):
            rl.roofline_bound(
                arithmetic_intensity_value=1.0, compute_roof=bandwidth, bandwidth_roof=compute
            )

    def test_point_requires_backend_and_positive_ai(self) -> None:
        with pytest.raises(ConfigError):
            rl.place_point(
                cell_id="c",
                ai_logical=0.0,
                ai_measured=None,
                compute_roof=_compute_roof(),
                bandwidth_roof=_bandwidth_roof(),
                measured_latency_ns=None,
                useful_ops=1.0,
                actual_backend="",
                quality_status="pass",
            )

    def test_figure_visibility_is_status_gated(self) -> None:
        point = rl.place_point(
            cell_id="c",
            ai_logical=10.0,
            ai_measured=None,
            compute_roof=_compute_roof(),
            bandwidth_roof=_bandwidth_roof(),
            measured_latency_ns=1.0,
            useful_ops=1.0,
            actual_backend="cuda",
            quality_status="QUALITY_GATE_FAILED",
        )
        assert rl.point_visible_in_main_figure(point) is False

    def test_capacity_model_and_error(self) -> None:
        model = rl.predict_capacity(candidate_id="c", weights_bytes=100, kv_bytes=50, margin_bytes=10)
        assert model.predicted_peak_bytes() == 160
        underestimated = rl.capacity_error(model, measured_peak_bytes=200)
        assert underestimated["underestimated"] is True
        assert "safety margin" in underestimated["advice"]
        missing = rl.capacity_error(model, measured_peak_bytes=None)
        assert missing["missing_reason"] == "MEASUREMENT_UNAVAILABLE"


class TestPhaseAndAmdahl:
    def test_phase_shares_must_sum_to_one(self) -> None:
        rows = [
            {"phase": "prefill", "component": "gemm", "time_ns": 3.0},
            {"phase": "prefill", "component": "attn", "time_ns": 1.0},
            {"phase": "decode", "component": "gemm", "time_ns": 1.0},
        ]
        shares = rl.phase_shares(rows)
        assert sum(row["share"] for row in shares if row["phase"] == "prefill") == pytest.approx(1.0)
        assert len(rl.prefill_op_mix(rows)) == 2
        assert len(rl.decode_op_mix(rows)) == 1

    def test_amdahl_upper_bound_is_wrapped_not_reimplemented(self) -> None:
        report = rl.amdahl_prediction(fraction=0.4, local_speedup=2.0)
        assert report["upper_bound"] == pytest.approx(1.25)
        with pytest.raises(ConfigError):
            rl.amdahl_prediction(fraction=2.0, local_speedup=2.0)

    def test_dispatch_adjusted_effect_reduces_by_coverage(self) -> None:
        adjusted = rl.dispatch_adjusted_effect(
            op_effect=2.0, op_time_share=0.5, dispatch_hit_rate=0.5, graph_break_rate=0.0
        )
        assert adjusted["effective_share"] == pytest.approx(0.25)
        assert adjusted["predicted_effect"] < 1.25
        with pytest.raises(ConfigError):
            rl.dispatch_adjusted_effect(op_effect=2.0, op_time_share=1.5, dispatch_hit_rate=1.0)

    def test_service_capacity_requires_measured_inputs(self) -> None:
        unavailable = rl.service_capacity_prediction(
            service_time_s=None,
            batch_efficiency=0.8,
            kv_capacity_tokens=None,
            queue_model="M/M/1",
            assumptions=("poisson arrivals",),
        )
        assert unavailable["status"] == "MEASUREMENT_UNAVAILABLE"
        assert "service_time_s" in unavailable["missing"]
        ok = rl.service_capacity_prediction(
            service_time_s=0.01,
            batch_efficiency=0.9,
            kv_capacity_tokens=4096,
            queue_model="M/M/1",
            assumptions=("poisson arrivals",),
        )
        assert ok["status"] == "OK"
        with pytest.raises(ConfigError):
            rl.service_capacity_prediction(
                service_time_s=0.01,
                batch_efficiency=0.9,
                kv_capacity_tokens=1,
                queue_model="M/M/1",
                assumptions=(),
            )

    def test_distributed_critical_path_accounts_every_segment(self) -> None:
        report = rl.distributed_critical_path(
            [
                {"kind": "compute", "duration_ns": 10.0},
                {"kind": "collective", "duration_ns": 5.0},
                {"kind": "wait", "duration_ns": 1.0},
                {"kind": "idle", "duration_ns": 1.0},
                {"kind": "overlap", "duration_ns": 2.0},
            ]
        )
        assert report["accounted_ns"] == pytest.approx(15.0)
        with pytest.raises(ConfigError):
            rl.distributed_critical_path([{"kind": "quantum", "duration_ns": 1.0}])


class TestPredictionsAndResiduals:
    def test_prediction_record_keeps_predicted_and_measured_apart(self) -> None:
        record = rl.PredictionRecord(
            prediction_id="p1",
            model_version="v1",
            target_cell_id="c1",
            split="calibration",
            operations_definition_id="gemm_v1",
            traffic_definition_id="logical_v1",
            compute_roof_id="r1",
            bandwidth_roof_ids=("r2",),
            predicted_value=100.0,
            unit="ns",
            measured_value=110.0,
        )
        assert record.validate() == []
        assert record.as_dict()["predicted_value"] == 100.0
        assert record.as_dict()["measured_value"] == 110.0

    def test_prediction_requires_a_split_and_evidence_for_explained_residuals(self) -> None:
        record = rl.PredictionRecord(
            prediction_id="p2",
            model_version="v1",
            target_cell_id="c1",
            split="somewhere",
            operations_definition_id="o",
            traffic_definition_id="t",
            compute_roof_id="r",
            bandwidth_roof_ids=(),
            predicted_value=1.0,
            unit="ns",
            residual_status="EXPLAINED",
        )
        problems = record.validate()
        assert any("calibration or validation" in item for item in problems)
        assert any("evidence" in item for item in problems)

    def test_prediction_error_metrics(self) -> None:
        error = rl.prediction_error(predicted=110.0, measured=100.0)
        assert error["relative_error"] == pytest.approx(0.1)
        assert error["log_error"] > 0
        with pytest.raises(ConfigError):
            rl.prediction_error(predicted=1.0, measured=0.0)

    def test_calibration_and_validation_must_be_disjoint(self) -> None:
        split = rl.CalibrationValidationSplit(("c1", "c2"), ("c2", "c3"))
        assert any("overlap" in item for item in split.validate())
        with pytest.raises(ConfigError):
            rl.assert_no_leakage(split, ["c2"])

    def test_validation_split_is_consumed_once(self) -> None:
        split = rl.CalibrationValidationSplit(("c1",), ("c2",))
        assert split.consume_validation() == ("c2",)
        with pytest.raises(ConfigError):
            split.consume_validation()

    def test_residual_ledger_keeps_unexplained_residuals(self) -> None:
        ledger = rl.ResidualLedger()
        ledger.record(
            prediction_id="p1",
            residual_class="layout",
            evidence_refs=("hqsb://trace/1",),
            confidence="medium",
            resolved=True,
        )
        ledger.record(
            prediction_id="p1", residual_class="kernel", evidence_refs=(), confidence="low", resolved=False
        )
        assert len(ledger.unresolved()) == 1
        with pytest.raises(ConfigError):
            ledger.record(
                prediction_id="p1",
                residual_class="kernel",
                evidence_refs=(),
                confidence="low",
                resolved=True,
            )

    def test_residual_without_evidence_is_inconclusive(self) -> None:
        assert (
            rl.classify_residual(residual_class="launch", evidence_refs=(), supports_class=False)["status"]
            == "INCONCLUSIVE"
        )
        assert (
            rl.classify_residual(residual_class="launch", evidence_refs=("x",), supports_class=False)["status"]
            == "PARTIALLY_EXPLAINED"
        )

    def test_bottleneck_requires_an_evidence_vector(self) -> None:
        unresolved = rl.classify_bottleneck({"COMPUTE_THROUGHPUT": "absent"})
        assert unresolved["bottleneck_class"] == "MIXED_OR_UNRESOLVED"
        resolved = rl.classify_bottleneck(
            {"COMPUTE_THROUGHPUT": "support", "HBM_OR_DDR_BANDWIDTH": "against"}, confidence="high"
        )
        assert resolved["bottleneck_class"] == "COMPUTE_THROUGHPUT"
        with pytest.raises(ConfigError):
            rl.classify_bottleneck({"COMPUTE_THROUGHPUT": "maybe"})

    def test_ablation_must_change_one_factor_and_list_alternatives(self) -> None:
        plan = rl.ablation_plan(cell_id="c", changed_factor="tile", expected_shift="+5%")
        with pytest.raises(ConfigError):
            rl.evaluate_ablation(
                ablation=plan, changed_factors=["tile", "layout"], observed_shift="+5%", alternative_explanations=["clock"]
            )
        with pytest.raises(ConfigError):
            rl.evaluate_ablation(
                ablation=plan, changed_factors=["tile"], observed_shift="+5%", alternative_explanations=[]
            )
        report = rl.evaluate_ablation(
            ablation=plan, changed_factors=["tile"], observed_shift="+5%", alternative_explanations=["clock drift"]
        )
        assert report["status"] == "SUPPORTS_MODEL"

    def test_uncertainty_propagation_and_figure_and_transferability(self) -> None:
        propagated = rl.propagate_uncertainty(
            {"ops": (1.0, 2.0), "bw": (10.0, 20.0)}, model_fn=lambda values: values["ops"] * values["bw"]
        )
        assert propagated["interval_low"] == pytest.approx(10.0)
        assert propagated["interval_high"] == pytest.approx(40.0)
        with pytest.raises(ConfigError):
            rl.propagate_uncertainty({"x": (2.0, 1.0)}, model_fn=lambda values: values["x"])

        problems = rl.validate_figure_metadata({"precision_path": "fp16"})
        assert len(problems) == len(rl.figure_contract()["required_metadata"]) - 1
        labelled_wrong = rl.validate_figure_metadata(
            {
                "precision_path": "fp16",
                "memory_level": "HBM",
                "roof_source": "datasheet",
                "roof_evidence_level": "THEORETICAL_VENDOR",
                "theoretical_vs_sustainable": "sustainable",
                "error_bars": True,
                "result_ids": ("r1",),
                "prediction_error": 0.1,
            }
        )
        assert any("vendor peak" in item for item in labelled_wrong)

        transfer = rl.transferability_check({"shape_small": 0.05, "shape_large": 0.4})
        assert transfer["worst_group"] == "shape_large"


class TestSmoke:
    def test_protocol_steps_are_complete(self) -> None:
        assert len(rl.PROTOCOL_STEPS) == 36
        assert all(interfaces for _, _, interfaces in rl.PROTOCOL_STEPS)

    def test_smoke_self_check_is_labelled(self) -> None:
        result = rl.smoke_self_check()
        assert result["status"] == "smoke"
        assert result["claim_allowed"] is False
        assert result["theoretical_roof_rejected_for_prediction"] is True
        assert result["point_has_two_utilizations"] is True
        assert result["ablation_needs_single_factor"] is True
