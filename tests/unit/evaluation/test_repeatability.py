"""Unit tests for the E12-04 repeatability, drift and exclusion machinery."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.evaluation import repeatability as rep

pytestmark = pytest.mark.unit


class TestVocabulary:
    def test_unknown_label_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            rep.combine_labels(("SOMETHING_NEW",))

    def test_combining_healthy_with_contamination_drops_healthy(self) -> None:
        labels = rep.combine_labels(("HEALTHY_INCLUDED",), ("SWAP_ACTIVITY",))
        assert labels == ("SWAP_ACTIVITY",)
        assert rep.is_healthy(labels) is False

    def test_unknown_anomaly_is_not_healthy(self) -> None:
        assert rep.is_healthy(("UNKNOWN_ANOMALY",)) is False
        assert rep.is_healthy(("HEALTHY_INCLUDED",)) is True


class TestHierarchy:
    def test_default_hierarchy_is_valid(self) -> None:
        assert rep.default_hierarchy().validate() == []

    def test_token_level_inference_is_rejected(self) -> None:
        hierarchy = rep.default_hierarchy()
        hierarchy.primary_inference_unit = "token"
        assert any("not independent" in item for item in hierarchy.validate())

    def test_missing_day_level_is_rejected(self) -> None:
        hierarchy = rep.default_hierarchy()
        hierarchy.levels = tuple(level for level in hierarchy.levels if level.level != "between_day")
        assert any("between_day" in item for item in hierarchy.validate())


class TestThresholds:
    def test_thresholds_must_be_positive_and_cite_a_pilot(self) -> None:
        good = rep.threshold_for("core_tpot", "model_core")
        assert good.validate() == []
        broken = rep.StabilityThresholds(
            metric_name="m",
            layer="operator",
            precision_half_width_rel=0.0,
            practical_cv=0.1,
            max_drift=0.1,
            anomaly_rate_max=0.1,
            practical_effect=0.1,
            pilot_source="",
        )
        problems = broken.validate()
        assert any("precision_half_width_rel" in item for item in problems)
        assert any("pilot basis" in item for item in problems)

    def test_unregistered_metric_is_refused(self) -> None:
        with pytest.raises(ConfigError):
            rep.threshold_for("made_up_metric", "operator")


class TestAnomalyRules:
    def test_rules_must_watch_system_metrics_only(self) -> None:
        rule = rep.AnomalyRule(
            rule_id="sneaky", rule_version="1", metric="latency_ns", comparator=">", threshold=1.0, window="run", label="TELEMETRY_GAP"
        )
        assert any("performance-blind" in item for item in rule.validate())

    def test_rule_set_detects_events(self) -> None:
        events = rep.default_rule_set().evaluate(
            [{"run_id": "r1", "temperature_c": 91.0, "swap_used_bytes": 1024}]
        )
        labels = {event["label"] for event in events}
        assert labels == {"THERMAL_THROTTLE", "SWAP_ACTIVITY"}
        assert all(event["rule_version"] for event in events)

    def test_clean_run_produces_no_events(self) -> None:
        assert rep.default_rule_set().evaluate([{"run_id": "r1", "temperature_c": 40.0}]) == ()

    def test_empty_rule_set_is_invalid(self) -> None:
        assert rep.AnomalyRuleSet(rules=()).validate()


class TestSchedule:
    def test_balanced_schedule_is_position_balanced(self) -> None:
        schedule = rep.balanced_schedule(["a", "b", "c"], blocks=6, seed=3)
        report = rep.schedule_balance(schedule)
        assert report["balanced"] is True

    def test_schedule_requires_every_candidate_per_block(self) -> None:
        with pytest.raises(ConfigError):
            rep.balanced_schedule(["a", "b", "c"], blocks=2)

    def test_schedule_is_reproducible_and_immutable(self) -> None:
        first = rep.balanced_schedule(["a", "b"], blocks=4, seed=11)
        second = rep.balanced_schedule(["a", "b"], blocks=4, seed=11)
        rep.assert_schedule_immutable(first, second)
        changed = rep.balanced_schedule(["a", "b"], blocks=4, seed=12)
        with pytest.raises(ConfigError):
            rep.assert_schedule_immutable(first, changed)


class TestExclusionLedger:
    def _entry(self, **overrides: object) -> rep.ExclusionEntry:
        payload = {
            "run_id": "run-1",
            "rule_id": "temp_high",
            "rule_version": "1.0.0",
            "detected_event": "THERMAL_THROTTLE",
            "threshold": 85.0,
            "observed_value": 91.0,
            "pre_registered": True,
            "decision": "exclude",
            "performance_blind_basis": "telemetry temperature",
            "artifact_refs": ("hqsb://S12/c/raw/1",),
            "effect_on_estimate": "median moves 1.5%",
            "sensitivity_result": "reported",
        }
        payload.update(overrides)
        return rep.ExclusionEntry(**payload)  # type: ignore[arg-type]

    def test_exclusion_needs_a_blind_basis_and_effect(self) -> None:
        broken = self._entry(performance_blind_basis="", effect_on_estimate="unquantified", artifact_refs=())
        problems = broken.validate()
        assert any("performance-blind" in item for item in problems)
        assert any("effect on the estimate" in item for item in problems)
        assert any("raw artefacts" in item for item in problems)

    def test_post_hoc_entries_are_kept_out_of_confirmatory_results(self) -> None:
        ledger = rep.ExclusionLedger()
        ledger.record(self._entry())
        ledger.record(self._entry(run_id="run-2", pre_registered=False))
        assert ledger.excluded() == ("run-1", "run-2")
        assert len(ledger.post_hoc_entries()) == 1
        assert "run-2" not in [row.run_id for row in ledger.confirmatory_rows()]

    def test_inclusion_policy_is_event_driven(self) -> None:
        report = rep.apply_inclusion_policy(
            {
                "run-1": ("HEALTHY_INCLUDED",),
                "run-2": ("SWAP_ACTIVITY",),
            }
        )
        assert report["included"] == ["run-1"]
        assert report["excluded"] == ["run-2"]


class TestStatistics:
    def test_median_and_quantile_definitions(self) -> None:
        assert rep.median([1.0, 3.0]) == pytest.approx(2.0)
        assert rep.quantile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
        with pytest.raises(ConfigError):
            rep.quantile([], 0.5)

    def test_cv_refuses_near_zero_mean(self) -> None:
        with pytest.raises(ConfigError):
            rep.cv([0.0, 0.0])

    def test_robust_statistics_are_order_independent(self) -> None:
        values = [10.0, 11.0, 12.0, 13.0, 100.0]
        assert rep.mad(values) == pytest.approx(rep.mad(list(reversed(values))))
        assert rep.iqr(values) == pytest.approx(2.0)
        assert rep.robust_cv_like(values) > 0

    def test_bootstrap_interval_is_deterministic_and_ordered(self) -> None:
        values = [100.0, 104.0, 98.0, 102.0, 101.0]
        first = rep.bootstrap_ci(values, seed=5)
        second = rep.bootstrap_ci(values, seed=5)
        assert first["interval_low"] == second["interval_low"]
        assert first["interval_low"] <= first["value"] <= first["interval_high"]
        assert first["method"] == "run_level_bootstrap"

    def test_bootstrap_requires_independent_runs(self) -> None:
        with pytest.raises(ConfigError):
            rep.bootstrap_ci([1.0])

    def test_variance_components_report_between_and_within(self) -> None:
        rows = [
            {"value": 100.0, "block": "b1"},
            {"value": 101.0, "block": "b1"},
            {"value": 110.0, "block": "b2"},
            {"value": 111.0, "block": "b2"},
        ]
        components = rep.variance_components(rows, level="block")
        names = {row["component"] for row in components}
        assert "between_block" in names and "within_block" in names

    def test_drift_slope_requires_variation(self) -> None:
        rows = [{"value": float(value), "run_order": 0} for value in (1, 2, 3)]
        with pytest.raises(ConfigError):
            rep.drift_slope(rows)
        drift = rep.drift_slope(
            [{"value": 1.0 * index, "run_order": index} for index in range(4)]
        )
        assert drift["slope"] == pytest.approx(1.0)

    def test_paired_log_ratio_needs_shared_blocks(self) -> None:
        with pytest.raises(ConfigError):
            rep.paired_log_ratio([{"block_id": "b1", "value": 1.0}], [{"block_id": "b2", "value": 1.0}])
        report = rep.paired_log_ratio(
            [{"block_id": "b1", "value": 2.0}], [{"block_id": "b1", "value": 1.0}]
        )
        assert report["ratio"] == pytest.approx(2.0)

    def test_frontier_membership_is_a_probability(self) -> None:
        draws = [
            [
                {"candidate_id": "a", "goodput": 10.0, "cost": 1.0},
                {"candidate_id": "b", "goodput": 9.0, "cost": 2.0},
            ],
            [
                {"candidate_id": "a", "goodput": 1.0, "cost": 5.0},
                {"candidate_id": "b", "goodput": 2.0, "cost": 4.0},
            ],
        ]
        report = rep.frontier_membership_probability(
            draws, objectives=(("goodput", "maximize"), ("cost", "minimize"))
        )
        probabilities = report["membership_probability"]
        assert set(probabilities) == {"a", "b"}
        assert all(0.0 <= value <= 1.0 for value in probabilities.values())

    def test_sensitivity_report_includes_leave_one_out(self) -> None:
        rows = [
            {"value": 10.0, "block_id": "b1", "labels": ("HEALTHY_INCLUDED",)},
            {"value": 11.0, "block_id": "b1", "labels": ("HEALTHY_INCLUDED",)},
            {"value": 40.0, "block_id": "b2", "labels": ("SWAP_ACTIVITY",)},
        ]
        report = rep.sensitivity_report(rows)
        scenarios = {row["scenario"] for row in report}
        assert "all_runs" in scenarios
        assert "healthy_only" in scenarios
        assert any(name.startswith("leave_out_block:") for name in scenarios)
        assert "dependence_on_exclusion" in scenarios


class TestConfirmationAndVerdicts:
    def test_pilot_and_confirmation_must_not_overlap(self) -> None:
        split = rep.PilotConfirmationSplit(("r1", "r2"), ("r2", "r3"))
        assert any("share runs" in item for item in split.validate())
        assert rep.PilotConfirmationSplit(("r1",), ("r2",)).validate() == []

    def test_confirmation_gate_uses_frozen_thresholds(self) -> None:
        thresholds = rep.threshold_for("core_tpot", "model_core")
        report = rep.confirmation_gate(
            thresholds=thresholds, confirmation_values=[100.0, 101.0, 99.0, 100.5]
        )
        assert report["threshold_revised"] is False
        assert "half_width_rel" in report

    def test_verdicts_mark_insufficient_below_three_runs(self) -> None:
        thresholds = rep.threshold_for("core_tpot", "model_core")
        verdicts = rep.stability_verdicts(
            [{"cell_id": "c1", "value": 1.0}, {"cell_id": "c1", "value": 1.1}], thresholds=thresholds
        )
        assert verdicts[0]["stability_verdict"] == "insufficient"

    def test_verdicts_flag_high_variance_as_unstable(self) -> None:
        thresholds = rep.StabilityThresholds(
            metric_name="core_tpot",
            layer="model_core",
            precision_half_width_rel=0.01,
            practical_cv=0.01,
            max_drift=0.01,
            anomaly_rate_max=0.01,
            practical_effect=0.01,
            pilot_source="pilot",
        )
        rows = [
            {"cell_id": "c1", "value": 100.0, "process": "p1"},
            {"cell_id": "c1", "value": 160.0, "process": "p2"},
            {"cell_id": "c1", "value": 90.0, "process": "p3"},
        ]
        verdict = rep.stability_verdicts(rows, thresholds=thresholds)[0]
        assert verdict["stability_verdict"] == "unstable"
        assert verdict["remeasure_condition"]

    def test_injection_results_must_be_detected(self) -> None:
        report = rep.evaluate_injections(
            [{"injection_id": "thermal_soak", "expected_labels": ("THERMAL_THROTTLE",), "observed_labels": ("THERMAL_THROTTLE",)}]
        )
        assert report["ok"] is True
        missed = rep.evaluate_injections(
            [{"injection_id": "thermal_soak", "expected_labels": ("THERMAL_THROTTLE",), "observed_labels": ()}]
        )
        assert missed["rows"][0]["false_negative"] is True


class TestSmoke:
    def test_protocol_steps_are_complete(self) -> None:
        assert len(rep.PROTOCOL_STEPS) == 36
        assert all(interfaces for _, _, interfaces in rep.PROTOCOL_STEPS)

    def test_smoke_self_check_is_labelled(self) -> None:
        result = rep.smoke_self_check()
        assert result["status"] == "smoke"
        assert result["claim_allowed"] is False
        assert result["schedule_balanced"] is True
        assert result["exclusion_requires_blind_basis"] is True
        assert result["cv_refuses_zero_mean"] is True
