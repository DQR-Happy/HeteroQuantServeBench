"""Unit tests for the E12-07 TCO / cost-per-token machinery (synthetic values)."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.evaluation import cost

pytestmark = pytest.mark.unit


def _source(**overrides: object) -> cost.PriceSource:
    payload = {
        "price_source_id": "src-1",
        "provider": "demo",
        "source_type": "calculator_export",
        "url": "https://example.invalid/demo",
        "retrieved_at": "2026-09-19",
        "effective_from": "2026-09-01",
        "effective_to": "2026-12-31",
        "region": "us-demo",
        "currency": "USD",
    }
    payload.update(overrides)
    return cost.PriceSource(**payload)  # type: ignore[arg-type]


def _snapshot(**overrides: object) -> cost.PriceSnapshot:
    payload = {
        "snapshot_id": "snap-1",
        "price_source_id": "src-1",
        "sku": "demo-gpu",
        "instance": "demo-node",
        "included_resources": ("gpu", "host", "ram"),
        "purchase_model": "on_demand",
        "unit_price": 2.0,
        "unit": "USD/hour",
        "content_hash": "a" * 64,
    }
    payload.update(overrides)
    return cost.PriceSnapshot(**payload)  # type: ignore[arg-type]


def _qualified_rows() -> list:
    return [
        {
            "result_id": "r1",
            "goodput": 100.0,
            "slo_compliant": True,
            "quality_status": "pass",
            "stability_status": "stable",
            "peak_goodput": 200.0,
        }
    ]


class TestPriceEvidence:
    def test_source_without_date_region_currency_is_rejected(self) -> None:
        problems = _source(url="", region="", currency="").validate()
        assert any("url" in item for item in problems)
        assert any("region" in item for item in problems)
        assert any("currency" in item for item in problems)

    def test_snapshot_requires_sku_instance_hash_and_resources(self) -> None:
        assert _snapshot().validate() == []
        problems = _snapshot(sku="", content_hash="", included_resources=()).validate()
        assert any("sku" in item for item in problems)
        assert any("content_hash" in item for item in problems)
        assert any("included resources" in item for item in problems)

    def test_stale_snapshot_is_unusable(self) -> None:
        assert cost.snapshot_valid(_snapshot(), _source(), as_of="2026-10-01") is True
        assert cost.snapshot_valid(_snapshot(), _source(), as_of="2027-01-01") is False

    def test_unavailable_price_is_structured(self) -> None:
        report = cost.price_unavailable(
            scenario_id="owned_base_case",
            reason="no dated electricity tariff",
            missing_fields=["electricity_price_per_kwh"],
        )
        assert report["estimate"] is None
        assert report["status"] == cost.MISSING_PRICE_UNAVAILABLE
        with pytest.raises(ConfigError):
            cost.price_unavailable(scenario_id="s", reason="", missing_fields=["x"])

    def test_fx_rate_needs_a_dated_source(self) -> None:
        with pytest.raises(ConfigError):
            cost.normalize_price(_snapshot(), base_currency="CNY", fx_rate=7.2)
        normalized = cost.normalize_price(_snapshot(), base_currency="CNY", fx_rate=7.2, fx_source_id="fx-1")
        assert normalized["per_hour"] == pytest.approx(14.4)

    def test_sku_mapping_flags_mismatch(self) -> None:
        matched = cost.validate_sku_mapping(_snapshot(), {"hardware_sku": "demo-gpu"})
        assert matched["matched"] is True
        mismatched = cost.validate_sku_mapping(_snapshot(), {"hardware_sku": "other-gpu"})
        assert mismatched["matched"] is False


class TestDeploymentAndCostModels:
    def test_host_resources_are_required_for_node_units(self) -> None:
        bare = cost.DeploymentUnit(
            deployment_unit_id="u1",
            candidate_id="c1",
            kind="node",
            device_count=1,
            host_resources={},
        )
        assert any("host resources" in item for item in bare.validate())

    def test_annualized_capex_is_a_scenario_not_a_fact(self) -> None:
        report = cost.annualized_capex(10000.0, years=5, residual_rel=0.2, discount_rate=0.0)
        assert report["annualized_capex"] == pytest.approx(1600.0)
        with pytest.raises(ConfigError):
            cost.annualized_capex(10000.0, years=0)
        with pytest.raises(ConfigError):
            cost.annualized_capex(10000.0, years=5, residual_rel=1.0)

    def test_pue_applies_to_it_energy_once(self) -> None:
        cost_j = cost.facility_energy_cost(
            3.6e6, electricity_price_per_kwh=0.1, pue=1.5
        )
        assert cost_j == pytest.approx(0.15)
        with pytest.raises(ConfigError):
            cost.facility_energy_cost(3.6e6, electricity_price_per_kwh=0.1, pue=0.9)

    def test_cost_models_reject_unknown_components(self) -> None:
        assert cost.cloud_cost_period({"compute_instance": 1.0})["total"] == 1.0
        with pytest.raises(ConfigError):
            cost.cloud_cost_period({"mystery": 1.0})
        with pytest.raises(ConfigError):
            cost.owned_cost_period({"mystery": 1.0})


class TestQualifiedCapacity:
    def test_peak_is_not_the_denominator(self) -> None:
        report = cost.qualified_operating_point(
            _qualified_rows(), candidate_id="cand_demo", profile_id="throughput"
        )
        assert report["status"] == "OK"
        assert report["goodput"] == 100.0
        assert report["peak_goodput"] == 200.0
        assert report["optimism_gap_rel"] == pytest.approx(1.0)

    def test_unqualified_point_yields_no_capacity(self) -> None:
        report = cost.qualified_operating_point(
            [
                {
                    "result_id": "r1",
                    "goodput": 300.0,
                    "slo_compliant": False,
                    "quality_status": "pass",
                    "stability_status": "stable",
                    "peak_goodput": 300.0,
                }
            ],
            candidate_id="c",
            profile_id="p",
        )
        assert report["status"] == cost.MISSING_PRICE_UNAVAILABLE

    def test_quality_failed_point_cannot_anchor_capacity(self) -> None:
        capacity = cost.QualifiedCapacity(
            capacity_id="c",
            candidate_id="cand_demo",
            profile_id="p",
            goodput=100.0,
            slo_compliant=True,
            quality_status="QUALITY_GATE_FAILED",
            stability_status="stable",
            source_result_id="r1",
        )
        assert any("quality" in item for item in capacity.validate())

    def test_peak_gap_is_quantified_not_recommended(self) -> None:
        point = cost.qualified_operating_point(_qualified_rows(), candidate_id="c", profile_id="p")
        gap = cost.peak_vs_qualified_gap(point)
        assert gap["gap_rel"] == pytest.approx(1.0)
        assert "understates cost" in gap["note"]


class TestReplicasAndDensity:
    def test_replicas_include_redundancy_and_stranding(self) -> None:
        plan = cost.plan_replicas(
            demand_goodput=100.0, capacity_per_replica=50.0, redundancy="N+1", utilization_target=0.8
        )
        # base = 100 / (50*0.8) = 2.5 -> ceil 3 -> +1 = 4
        assert plan["replicas"] == 4
        assert plan["stranded_capacity"] > 0

    def test_replicas_validate_inputs(self) -> None:
        with pytest.raises(ConfigError):
            cost.plan_replicas(demand_goodput=10.0, capacity_per_replica=0.0)
        with pytest.raises(ConfigError):
            cost.plan_replicas(demand_goodput=10.0, capacity_per_replica=1.0, utilization_target=1.5)

    def test_density_reports_unavailable_dimensions(self) -> None:
        report = cost.capacity_density(goodput=100.0, devices=4)
        assert report["metrics"]["goodput_per_device"] == pytest.approx(25.0)
        assert "goodput_per_rack_unit" in report["unavailable"]

    def test_cost_per_metrics_requires_a_denominator(self) -> None:
        empty = cost.cost_per_metrics(total_cost=10.0, compliant_requests=0, compliant_tokens=0, period="h")
        assert empty["status"] == cost.MISSING_PRICE_UNAVAILABLE
        report = cost.cost_per_metrics(total_cost=10.0, compliant_requests=2, compliant_tokens=100, period="h")
        assert report["cost_per_token"] == pytest.approx(0.1)
        assert report["cost_per_million_tokens"] == pytest.approx(100000.0)


class TestAuditsAndSensitivity:
    def test_double_count_audit_is_explicit(self) -> None:
        findings = cost.double_count_audit({"goodput_already_includes_downtime": "ok"})
        assert len(findings) == len(cost.DOUBLE_COUNT_CHECKS)
        statuses = {row["check_id"]: row["status"] for row in findings}
        assert statuses["goodput_already_includes_downtime"] == "PASS"
        assert statuses["pue_applies_to_it_energy_only"] == "FAIL"

    def test_one_way_sensitivity_ranks_by_elasticity(self) -> None:
        rows = cost.one_way_sensitivity(
            base_cost=100.0,
            parameters={"price": (1.0, 2.0), "utilization": (0.5, 0.8)},
            cost_fn=lambda values: values["base_cost"] * values["price"] if "price" in values else values["base_cost"] / values["utilization"],
        )
        assert rows[0]["parameter"] == "price"

    def test_break_even_finds_the_crossing(self) -> None:
        report = cost.break_even(
            candidate_a="a",
            candidate_b="b",
            profile_id="p",
            parameter="utilization",
            a_cost_fn=lambda values: 100.0 / values["utilization"],
            b_cost_fn=lambda values: 50.0 / values["utilization"] + 30.0,
            scan=[0.5, 0.75, 1.0],
        )
        assert "threshold_value" in report

    def test_leave_one_source_out(self) -> None:
        report = cost.leave_one_source_out(
            sources=["s1", "s2", "s3"],
            cost_by_source={"s1": 100.0, "s2": 110.0, "s3": 90.0},
            threshold_rel=0.1,
        )
        assert report["spread_rel"] > 0
        with pytest.raises(ConfigError):
            cost.leave_one_source_out(sources=["s1"], cost_by_source={"s1": 1.0}, threshold_rel=0.1)

    def test_workload_mix_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ConfigError):
            cost.workload_mix_cost(
                weights={"interactive": 0.6, "throughput": 0.6},
                cost_by_workload={"interactive": 1.0, "throughput": 2.0},
            )
        report = cost.workload_mix_cost(
            weights={"interactive": 0.4, "throughput": 0.6},
            cost_by_workload={"interactive": 1.0, "throughput": 2.0},
        )
        assert report["weighted_cost"] == pytest.approx(1.6)

    def test_integer_and_capacity_check(self) -> None:
        ok = cost.integer_and_capacity_check(replicas=2, devices_per_replica=4, memory_fit=True, topology_floor=4)
        assert ok["status"] == "OK"
        infeasible = cost.integer_and_capacity_check(replicas=1, devices_per_replica=2, memory_fit=False, topology_floor=4)
        assert infeasible["status"] == "INFEASIBLE"
        assert any("topology floor" in item for item in infeasible["problems"])

    def test_independent_recalculation_detects_tampering(self) -> None:
        ok = cost.independent_recalculation(
            result_id="r", components={"a": 10.0, "b": 20.0}, recomputed_total=30.0
        )
        assert ok["status"] == "PASS"
        tampered = cost.independent_recalculation(
            result_id="r", components={"a": 10.0, "b": 20.0}, recomputed_total=99.0
        )
        assert tampered["status"] == "FAIL"

    def test_cost_uncertainty_propagates(self) -> None:
        report = cost.propagate_cost_uncertainty(
            cost=100.0, capacity_interval=(90.0, 110.0), energy_interval=(10.0, 12.0)
        )
        assert report["interval_high"] >= report["interval_low"]


class TestCostResult:
    def _result(self, **overrides: object) -> cost.CostResult:
        payload = {
            "cost_result_id": "cr1",
            "candidate_id": "cand_demo",
            "profile_id": "throughput",
            "scenario_id": "cloud_on_demand_low_utilization",
            "effective_date": "2026-09-19",
            "period": "month",
            "region": "us-demo",
            "currency": "USD",
            "deployment_unit": "cloud_instance",
            "replicas": 1,
            "utilization": 0.5,
            "qualified_goodput_result_id": "r1",
            "price_source_ids": ("src-1",),
            "assumption_ids": ("a1",),
            "total_cost": 100.0,
            "compliant_requests": 10,
            "compliant_tokens": 1000,
            "cost_per_token": 0.1,
        }
        payload.update(overrides)
        return cost.CostResult(**payload)  # type: ignore[arg-type]

    def test_valid_result(self) -> None:
        assert self._result().validate() == []

    def test_missing_price_source_or_assumptions_is_rejected(self) -> None:
        problems = self._result(price_source_ids=()).validate()
        assert any("price source" in item for item in problems)
        problems = self._result(assumption_ids=()).validate()
        assert any("assumptions" in item for item in problems)

    def test_peak_anchored_capacity_is_rejected(self) -> None:
        problems = self._result(qualified_goodput_result_id="").validate()
        assert any("qualified goodput" in item for item in problems)

    def test_cost_verdict_with_refresh_window(self) -> None:
        verdict = cost.cost_verdict(
            self._result(),
            valid_until="2026-12-31",
            refresh_triggers=("price change",),
            main_sensitivity="utilization",
        )
        assert verdict["status"] == "OK"
        assert verdict["valid_until"] == "2026-12-31"


class TestSmoke:
    def test_protocol_steps_are_complete(self) -> None:
        assert len(cost.PROTOCOL_STEPS) == 36
        assert all(interfaces for _, _, interfaces in cost.PROTOCOL_STEPS)

    def test_no_hardcoded_prices_in_the_module(self) -> None:
        import hqsb.evaluation.cost as module

        source = open(module.__file__, encoding="utf-8").read()  # noqa: SIM115
        # The module must not bake in any electricity/price constant.
        assert "0.12" not in source and "electricity_price=" not in source

    def test_smoke_self_check_is_labelled(self) -> None:
        result = cost.smoke_self_check()
        assert result["status"] == "smoke"
        assert result["claim_allowed"] is False
        assert result["peak_never_anchors_cost"] is True
        assert result["unavailable_price_is_structured"] is True
