"""Coverage for the E11-07 cost-model, regret and safe-selection interfaces."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.compiler import costmodel as cm


def _units() -> list:
    return [
        cm.DatasetUnit("t1", "g1", "sm86", "default", {"H": 4096, "phase": "decode", "input_dtype": "fp16"}, (2.0, 2.1, 1.9)),
        cm.DatasetUnit("t1", "g1", "sm86", "cuda_v2", {"H": 4096, "phase": "decode", "input_dtype": "fp16"}, (1.0, 1.1, 0.9)),
        cm.DatasetUnit("t2", "g2", "sm86", "default", {"H": 4096, "phase": "prefill", "input_dtype": "fp16"}, (3.0, 3.1)),
        cm.DatasetUnit("t2", "g2", "sm86", "cuda_v2", {"H": 4096, "phase": "prefill", "input_dtype": "fp16"}, (3.0, 3.05)),
    ]


@pytest.mark.unit
class TestFeatureSchema:
    def test_default_schema_validates(self) -> None:
        schema = cm.default_feature_schema()
        assert schema.validate() == []
        assert schema.digest()

    def test_leakage_field_is_rejected(self) -> None:
        field = cm.FeatureField(name="latency_ms", group="workload_shape", availability="post_run")
        assert any("leakage" in problem for problem in field.validate())

    def test_unknown_group_and_availability_are_rejected(self) -> None:
        field = cm.FeatureField(name="x", group="vibes", availability="someday")
        problems = field.validate()
        assert any("group" in problem for problem in problems)
        assert any("availability" in problem for problem in problems)

    def test_missing_critical_features_force_fallback(self) -> None:
        schema = cm.default_feature_schema()
        assert schema.missing_critical({"H": 4096, "phase": "decode", "input_dtype": "fp16"}) == []
        assert set(schema.missing_critical({"H": None, "phase": "decode"})) == {"H", "input_dtype"}

    def test_duplicate_features_are_rejected(self) -> None:
        schema = cm.FeatureSchema(
            version="1.0.0",
            fields=(
                cm.FeatureField("a", "workload_shape", "pre_compile"),
                cm.FeatureField("a", "workload_shape", "pre_compile"),
            ),
        )
        assert any("duplicate" in problem for problem in schema.validate())

    def test_leakage_audit_flags_forbidden_and_undeclared_fields(self) -> None:
        schema = cm.default_feature_schema()
        report = cm.leakage_audit(
            [{"H": 1, "phase": "decode", "input_dtype": "fp16", "latency_ms": 1.0, "mystery": 2}], schema
        )
        assert report["clean"] is False
        issues = {row["field"]: row["issue"] for row in report["problems"]}
        assert "latency_ms" in issues
        assert "mystery" in issues


@pytest.mark.unit
class TestLabelsAndSplits:
    def test_build_labels_marks_ci_overlaps_as_ties(self) -> None:
        labels = cm.build_labels(_units())
        assert len(labels["rows"]) == 4
        assert labels["ties"]
        assert all(row["label_median"] > 0 for row in labels["rows"])

    def test_unit_requires_samples(self) -> None:
        with pytest.raises(ConfigError):
            cm.DatasetUnit("t", "g", "arch", "c", {}, ()).median()

    def test_split_manifest_must_be_grouped_and_locked(self) -> None:
        manifest = cm.SplitManifest(
            strategy="random_row_diagnostic", train=("a",), validation=("b",), final_test=("c",), locked=False
        )
        problems = manifest.validate()
        assert any("random-row" in problem for problem in problems)
        assert any("locked" in problem for problem in problems)

    def test_build_split_rejects_overlap(self) -> None:
        with pytest.raises(ConfigError):
            cm.build_split(
                strategy="leave_shape_group_out",
                group_keys={"u1": "g1", "u2": "g2"},
                holdout_keys=["g1"],
                validation_keys=["g1"],
            )

    def test_final_test_can_only_be_consumed_once(self) -> None:
        manifest = cm.build_split(
            strategy="leave_shape_group_out",
            group_keys={"u1": "g1", "u2": "g2"},
            holdout_keys=["g2"],
        )
        manifest.consume_final_test()
        with pytest.raises(ConfigError):
            manifest.consume_final_test()

    def test_primary_split_must_be_group_based(self) -> None:
        with pytest.raises(ConfigError):
            cm.build_split(
                strategy="random_row_diagnostic", group_keys={"u1": "g1"}, holdout_keys=["g1"]
            )


@pytest.mark.unit
class TestBaselinesAndModels:
    def test_baselines_include_strong_references(self) -> None:
        report = cm.evaluate_baselines(
            candidates=["default", "cuda_v2"],
            features={"phase": "decode"},
            latencies={"default": 2.0, "cuda_v2": 1.0},
        )
        rows = {row["baseline"]: row for row in report["rows"]}
        assert rows["oracle"]["chosen"] == "cuda_v2"
        assert rows["oracle"]["regret"] == 0.0
        assert rows["default"]["regret"] == pytest.approx(1.0)

    def test_unknown_baseline_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            cm.evaluate_baseline(name="magic", candidates=["a"], features={}, latencies={"a": 1.0})

    def test_heuristic_prefers_small_tile_for_decode(self) -> None:
        assert cm.heuristic_pick(["default", "small_tile"], {"phase": "decode"}) == "small_tile"
        assert cm.heuristic_pick(["default", "small_tile"], {"phase": "prefill"}) == "default"

    def test_constant_and_linear_predictors_fit_and_predict(self) -> None:
        rows = [
            {"H": 1024, "label": 1.0},
            {"H": 2048, "label": 2.0},
            {"H": 4096, "label": 4.0},
        ]
        constant = cm.ConstantPredictor().fit(rows)
        assert constant.predict({"H": 1}) == pytest.approx(7.0 / 3.0)
        linear = cm.LinearRanker().fit(rows)
        assert linear.predict({"H": 4096}) == pytest.approx(4.0, abs=0.1)
        assert linear.describe()["trained"] is True

    def test_pairwise_ranker_orders_higher_feature_as_cheaper(self) -> None:
        rows = [
            {"tile": 256, "task_id": "t1", "label": 2.0},
            {"tile": 1024, "task_id": "t1", "label": 1.0},
        ]
        ranker = cm.PairwiseRanker(seed=1).fit(rows)
        assert ranker.predict({"tile": 1024}) < ranker.predict({"tile": 256})

    def test_fit_and_predict_on_empty_inputs_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            cm.LinearRanker().fit([])
        with pytest.raises(ConfigError):
            cm.ConstantPredictor().predict({})


@pytest.mark.unit
class TestMetrics:
    def test_regret_report_reports_tail_metrics(self) -> None:
        report = cm.RegretReport(
            per_case=({"regret": 0.01}, {"regret": 0.02}, {"regret": 0.8}),
            weights=(1.0, 1.0, 1.0),
            catastrophic_threshold=0.5,
        )
        summary = report.summary()
        assert summary["max"] == 0.8
        assert summary["catastrophic_rate"] == pytest.approx(1 / 3)
        assert summary["weighted_mean"] == pytest.approx(round((0.01 + 0.02 + 0.8) / 3, 6))

    def test_empty_regret_is_unavailable_not_zero(self) -> None:
        summary = cm.RegretReport(per_case=({"regret": None},)).summary()
        assert summary["cases"] == 0
        assert "UNAVAILABLE" in summary["note"]

    def test_weight_count_must_match_cases(self) -> None:
        with pytest.raises(ConfigError):
            cm.RegretReport(per_case=({"regret": 0.1},), weights=(1.0, 2.0)).summary()

    def test_speedup_capture_handles_zero_headroom(self) -> None:
        assert cm.speedup_capture(baseline=1.0, selected=0.9, oracle=1.0) is None
        assert cm.speedup_capture(baseline=2.0, selected=1.5, oracle=1.0) == pytest.approx(0.5)

    def test_ranking_metrics_are_tie_aware(self) -> None:
        report = cm.ranking_metrics(
            predictions=[("a", 1.0), ("b", 0.5)],
            true_latencies={"a": 2.0, "b": 1.0},
        )
        assert report["top1"] is True
        assert report["pairwise_accuracy"] == 1.0
        tied = cm.ranking_metrics(predictions=[("a", 1.0), ("b", 0.5)], true_latencies={"a": 1.0, "b": 1.0})
        assert tied["ties"] == 1


@pytest.mark.unit
class TestConfidenceAndPolicy:
    def test_risk_coverage_requires_calibration(self) -> None:
        with pytest.raises(ConfigError):
            cm.ConfidenceModel().risk_coverage([])

    def test_risk_coverage_reports_monotonicity(self) -> None:
        model = cm.ConfidenceModel().calibrate([{"top2_margin": 0.1}], threshold=0.2)
        report = model.risk_coverage(
            [
                {"confidence": 0.9, "regret": 0.01},
                {"confidence": 0.1, "regret": 0.5},
            ]
        )
        coverages = [row["coverage"] for row in report["rows"]]
        # thresholds include the calibrated value 0.2 as well, so the curve has
        # three points; coverage must decrease as the threshold rises
        assert coverages[0] == pytest.approx(1.0)
        assert coverages[-1] == pytest.approx(0.5)
        assert coverages == sorted(coverages, reverse=True)
        assert report["risk_monotone"] is True

    def test_ood_detection_uses_train_ranges(self) -> None:
        model = cm.ConfidenceModel().fit_ranges([{"H": 4096}], ["H"])
        assert model.ood_score({"H": 4096})[1] is False
        assert model.ood_score({"H": 16384})[1] is True

    def test_policy_chain_is_fail_closed(self) -> None:
        empty = cm.select_with_policy(
            eligible=[], predictions={}, confidence=0.9, confidence_threshold=0.5, ood=False
        )
        assert empty["action"] == "semantic_fallback"
        ood = cm.select_with_policy(
            eligible=["a"], predictions={"a": 1.0}, confidence=0.9, confidence_threshold=0.5, ood=True
        )
        assert ood["action"] == "heuristic_default"
        missing = cm.select_with_policy(
            eligible=["a"],
            predictions={"a": 1.0},
            confidence=0.9,
            confidence_threshold=0.5,
            ood=False,
            missing_critical=["H"],
        )
        assert missing["action"] == "heuristic_default" and missing["fallback_triggered"] is True
        low = cm.select_with_policy(
            eligible=["a"],
            predictions={"a": 1.0},
            confidence=0.1,
            confidence_threshold=0.5,
            ood=False,
            topk_benchmark_candidates=["a", "b"],
        )
        assert low["action"] == "benchmark_topk"
        selected = cm.select_with_policy(
            eligible=["a", "b"], predictions={"a": 1.0, "b": 2.0}, confidence=0.9, confidence_threshold=0.5, ood=False
        )
        assert selected["action"] == "model_selected" and selected["chosen"] == "a"

    def test_illegal_candidate_is_blocked(self) -> None:
        report = cm.illegal_candidate_injection(
            eligible=["default"],
            illegal="illegal_cheap",
            mock_predictions={"illegal_cheap": 0.0001, "default": 1.0},
            policy_kwargs={"confidence": 0.99, "confidence_threshold": 0.5, "ood": False},
        )
        assert report["blocked"] is True
        assert report["decision"]["chosen"] == "default"

    def test_model_artifact_guard_disables_stale_models(self) -> None:
        guard = cm.ModelArtifactGuard(
            expected_hash="h", expected_schema_version="1.0.0", expected_feature_version="1.0.0"
        )
        ok = guard.check(artifact_hash="h", schema_version="1.0.0", feature_version="1.0.0")
        assert ok["ok"] is True
        stale = guard.check(artifact_hash="other", schema_version="1.0.0", feature_version="1.0.0")
        assert stale["ok"] is False
        assert stale["action"] == "use heuristic/default"

    def test_selection_overhead_ratio(self) -> None:
        report = cm.selection_overhead(
            feature_extraction_us=1.0,
            model_load_us=0.0,
            inference_us=1.0,
            candidate_filter_us=0.5,
            decision_us=0.5,
            kernel_latency_us=100.0,
        )
        assert report["total_us"] == pytest.approx(3.0)
        assert report["selection_overhead_ratio"] == pytest.approx(0.03)
        unavailable = cm.selection_overhead(
            feature_extraction_us=1.0, model_load_us=0.0, inference_us=0.0,
            candidate_filter_us=0.0, decision_us=0.0, kernel_latency_us=None,
        )
        assert unavailable["selection_overhead_ratio"] is None

    def test_deployment_criteria_separate_pass_from_fail(self) -> None:
        criteria = cm.DeploymentCriteria(
            max_weighted_regret=0.05,
            max_p95_regret=0.1,
            max_catastrophic_rate=0.01,
            min_improvement_over_heuristic=0.05,
            confidence_risk_monotone=True,
        )
        passing = criteria.evaluate(
            {"weighted_mean": 0.01, "p95": 0.05, "catastrophic_rate": 0.0, "improvement_over_heuristic": 0.2},
            overhead_ratio=0.01,
        )
        assert passing["deploy"] is True
        failing = criteria.evaluate(
            {"weighted_mean": 0.2, "p95": 0.5, "catastrophic_rate": 0.1, "improvement_over_heuristic": 0.0},
            overhead_ratio=0.5,
        )
        assert failing["deploy"] is False
        assert {row["check"] for row in failing["failures"]} >= {"weighted_regret", "p95_regret"}

    def test_methodology_verdict_requires_every_check(self) -> None:
        report = cm.methodology_verdict(
            dataset_reproducible=True,
            final_test_untouched_until_frozen=True,
            baselines_complete=True,
            regret_reported=True,
            risk_coverage_reported=True,
            illegal_injection_blocked=True,
            corrupt_model_fallback=True,
            dispatch_replay_ok=True,
            overhead_reported=False,
        )
        assert report["methodology_pass"] is False
        assert report["failing"] == ["overhead_reported"]

    def test_error_case_study_kinds(self) -> None:
        study = cm.error_case_study(
            case_id="c", kind="high_confidence_failure", missing_feature="registers"
        )
        assert study["missing_feature"] == "registers"
        with pytest.raises(ConfigError):
            cm.error_case_study(case_id="c", kind="magic", missing_feature="x")

    def test_dataset_digest_is_order_independent_of_key_order(self) -> None:
        first = cm.dataset_digest([{"b": 1, "a": 2}])
        second = cm.dataset_digest([{"a": 2, "b": 1}])
        assert first == second
