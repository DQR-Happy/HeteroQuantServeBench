"""Unit tests for the E12-09 software-maturity rubric machinery."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.evaluation import maturity as mat

pytestmark = pytest.mark.unit


def _session(**overrides: object) -> mat.SessionRecord:
    payload = {
        "session_id": "s1",
        "candidate_id": "cand_demo",
        "operator_or_session_id": "op1",
        "experience_band": "novice",
        "clean_or_warm": "clean",
        "task_id": "load_model",
        "start": "2026-09-19T00:00:00Z",
        "end": "2026-09-19T00:30:00Z",
        "wall_s": 1800.0,
        "active_s": 600.0,
        "blocked_s": 1000.0,
        "rework_s": 200.0,
        "success_status": "success",
        "correctness_status": "pass",
        "actual_backend": "cuda",
    }
    payload.update(overrides)
    return mat.SessionRecord(**payload)  # type: ignore[arg-type]


class TestRubric:
    def test_anchors_exist_and_are_observable(self) -> None:
        assert mat.anchor_for("installability", 4).startswith("stable")
        with pytest.raises(ConfigError):
            mat.anchor_for("made_up", 4)
        with pytest.raises(ConfigError):
            mat.anchor_for("installability", 99)

    def test_not_evaluated_is_not_a_zero(self) -> None:
        with pytest.raises(ConfigError):
            mat.rate(
                session_id="s",
                task_id="t",
                dimension="installability",
                level=0,
                rationale="never ran it",
                reviewer="r",
                evidence_refs=(),
                not_evaluated=True,
            )

    def test_rating_needs_a_reviewer_and_rationale(self) -> None:
        rating = mat.rate(
            session_id="s",
            task_id="t",
            dimension="installability",
            level=3,
            rationale="reproduced from docs",
            reviewer="r1",
            evidence_refs=("e1",),
        )
        assert rating.validate() == []

    def test_aggregate_keeps_raw_vector_and_reports_coverage(self) -> None:
        ratings = [
            mat.rate(session_id="s1", task_id="t", dimension="installability", level=3, rationale="x", reviewer="r", evidence_refs=()),
            mat.rate(session_id="s2", task_id="t", dimension="installability", level=None, rationale="x", reviewer="r", evidence_refs=(), not_evaluated=True),
        ]
        report = mat.aggregate_scores(ratings)
        assert report["coverage"] == 0.5
        assert report["not_evaluated_count"] == 1
        assert report["raw_vector"] == {"installability": [3]}

    def test_coverage_penalty_for_scoring_only_successes(self) -> None:
        ratings = [mat.rate(session_id="s1", task_id="t", dimension="installability", level=4, rationale="x", reviewer="r", evidence_refs=())]
        sessions = [_session(), _session(session_id="s2", success_status="fail", correctness_status="fail")]
        report = mat.coverage_penalty_check(ratings, sessions)
        assert report["penalty_applies"] is True
        assert report["unrated_failed_sessions"] == ["s2"]

    def test_agreement_reports_disputes_not_averages(self) -> None:
        agree = [
            mat.rate(session_id="s", task_id="t", dimension="installability", level=3, rationale="x", reviewer="r1", evidence_refs=()),
            mat.rate(session_id="s", task_id="t", dimension="installability", level=3, rationale="x", reviewer="r2", evidence_refs=()),
        ]
        assert mat.agreement(agree)["exact_agreement"] == 1
        disagree = [
            mat.rate(session_id="s", task_id="t", dimension="installability", level=1, rationale="x", reviewer="r1", evidence_refs=()),
            mat.rate(session_id="s", task_id="t", dimension="installability", level=4, rationale="x", reviewer="r2", evidence_refs=()),
        ]
        report = mat.agreement(disagree)
        assert report["disputed_items"]


class TestTasksAndSessions:
    def test_standard_tasks_cover_the_fifteen_areas(self) -> None:
        assert len(mat.STANDARD_TASKS) == 15
        assert all(task.validate() == [] for task in mat.STANDARD_TASKS)

    def test_not_applicable_needs_justification(self) -> None:
        task = mat.TaskSpec(
            task_id="multi_device",
            dimension="framework_model_compatibility",
            description="d",
            inputs=(),
            expected_output="e",
            success_criteria="s",
            time_budget_s=10,
            allowed_help=(),
            termination_condition="t",
            applicability="not_applicable",
        )
        report = mat.validate_applicability(task, capability_evidence=[], profile_requires=False)
        assert report["applicable"] is False
        forced = mat.validate_applicability(task, capability_evidence=[], profile_requires=True)
        assert forced["applicable"] is True
        assert "profile requires" in forced["reason"]

    def test_timing_breakdown_keeps_wall_and_active_apart(self) -> None:
        breakdown = mat.split_times(
            [
                {"kind": "active", "duration_s": 100.0},
                {"kind": "blocked", "duration_s": 50.0},
                {"kind": "rework", "duration_s": 10.0},
            ]
        )
        assert breakdown.wall_s == pytest.approx(160.0)
        assert breakdown.active_s == 100.0
        with pytest.raises(ConfigError):
            mat.split_times([{"kind": "thinking", "duration_s": 1.0}])

    def test_session_rejects_inconsistent_time_or_success(self) -> None:
        assert _session().validate() == []
        assert _session(clean_or_warm="lukewarm").validate()
        assert _session(success_status="success", correctness_status="fail").validate()
        assert _session(actual_backend="").validate()

    def test_task_metrics_preserves_the_four_buckets(self) -> None:
        metrics = mat.task_metrics(_session())
        assert metrics["blocked_s"] == 1000.0
        assert metrics["first_correct_inference_s"] == 600.0
        failed = mat.task_metrics(_session(success_status="fail", correctness_status="fail"))
        assert failed["first_correct_inference_s"] is None


class TestFailuresAndWorkarounds:
    def test_failure_classification_needs_evidence(self) -> None:
        classified = mat.classify_failure(
            category="SILENT_FALLBACK",
            severity="high",
            root_cause_confidence="high",
            evidence_refs=("e1",),
        )
        assert classified["category"] == "SILENT_FALLBACK"
        with pytest.raises(ConfigError):
            mat.classify_failure(category="BUILD_OR_COMPILE", severity="high", root_cause_confidence="high", evidence_refs=())

    def test_unknown_requires_low_confidence(self) -> None:
        with pytest.raises(ConfigError):
            mat.classify_failure(category="UNKNOWN", severity="high", root_cause_confidence="high", evidence_refs=())
        ok = mat.classify_failure(category="UNKNOWN", severity="high", root_cause_confidence="low", evidence_refs=())
        assert ok["category"] == "UNKNOWN"

    def test_workaround_requires_owner_and_versions(self) -> None:
        registry = mat.WorkaroundRegistry()
        row = registry.register(
            workaround_id="w1",
            kind="patch",
            owner="o",
            upstream_issue="up-1",
            first_version="1.0",
            last_verified_version="1.2",
            risk="high",
            automation_status="manual",
            maintenance_hours=2.0,
            removal_condition="upstream fix",
        )
        assert row["risk"] == "high"
        assert [row["workaround_id"] for row in registry.at_risk()] == ["w1"]
        with pytest.raises(ConfigError):
            registry.register(
                workaround_id="w2",
                kind="patch",
                owner="",
                upstream_issue="",
                first_version="",
                last_verified_version="",
                risk="high",
                automation_status="manual",
                maintenance_hours=1.0,
                removal_condition="",
            )


class TestToolsAndChecks:
    def test_silent_fallback_is_high_risk(self) -> None:
        silent = mat.silent_fallback_check(requested_backend="triton", actual_backend="eager", fallback_reason="")
        assert silent["risk"] == "high"
        explained = mat.silent_fallback_check(requested_backend="triton", actual_backend="eager", fallback_reason="guard false")
        assert explained["risk"] == "none"
        with pytest.raises(ConfigError):
            mat.silent_fallback_check(requested_backend="", actual_backend="eager", fallback_reason="")

    def test_tool_coverage_summary(self) -> None:
        rows = [
            mat.tool_coverage_row(candidate_id="c", area="build", tool="cmake", can_observe=True, can_export=True, can_correlate=True, can_automate=True, limitation=""),
            mat.tool_coverage_row(candidate_id="c", area="power", tool="nvml", can_observe=True, can_export=False, can_correlate=False, can_automate=False, limitation="no energy"),
        ]
        report = mat.coverage_summary(rows)
        assert "power" in report["coverage"]
        assert len(report["missing_areas"]) == 5
        with pytest.raises(ConfigError):
            mat.tool_coverage_row(candidate_id="c", area="quantum", tool="x", can_observe=True, can_export=False, can_correlate=False, can_automate=False, limitation="")

    def test_learning_curve_and_maintenance_estimate(self) -> None:
        curve = mat.learning_curve([_session(), _session(session_id="s2", start="2026-09-20T00:00:00Z", active_s=200.0)], task_id="load_model")
        assert curve["first_session_active_s"] == 600.0
        assert curve["repeat_session_active_s"] == 200.0
        with pytest.raises(ConfigError):
            mat.learning_curve([], task_id="load_model")

        estimate = mat.maintenance_estimate(
            workarounds=[{"maintenance_hours": 2.0}],
            incidents=[{"recovery_hours": 1.0}],
            upgrades=[{"migration_hours": 0.5}],
        )
        assert estimate["active_hours_low"] == 0.5
        assert estimate["active_hours_high"] == 2.0
        assert estimate["single_incident_exaggeration_guard"] is False

    def test_upgrade_must_rerun_correctness_smoke(self) -> None:
        with pytest.raises(ConfigError):
            mat.upgrade_record(component="torch", from_version="1", to_version="2", correctness_smoke="not_run", breaking_changes=(), migration_hours=1.0)
        ok = mat.upgrade_record(component="torch", from_version="1", to_version="2", correctness_smoke="pass", breaking_changes=(), migration_hours=1.0)
        assert ok["correctness_smoke"] == "pass"

    def test_rollback_verification_checks_all_three(self) -> None:
        assert mat.rollback_verification(artifact_identity_restored=True, cache_restored=True, environment_restored=True)["status"] == "OK"
        bad = mat.rollback_verification(artifact_identity_restored=True, cache_restored=False, environment_restored=True)
        assert bad["status"] == "FAIL"
        assert bad["problems"] == ["cache"]

    def test_deployable_constraints_and_verdict(self) -> None:
        constraints = mat.deployable_constraints(silent_fallbacks=["triton->eager"], undiagnosable_failures=[], missing_capabilities=[], maintenance_risk="medium")
        assert constraints["status"] == "not_deployable"
        verdict = mat.maturity_verdict(
            candidate_id="c",
            ratings=[mat.rate(session_id="s1", task_id="t", dimension="installability", level=3, rationale="x", reviewer="r", evidence_refs=())],
            sessions=[_session()],
            coverage={},
        )
        assert verdict["task_completion_rate"] == 1.0
        assert "production reliability" in verdict["forbidden_claims"][0]


class TestSmoke:
    def test_protocol_steps_are_complete(self) -> None:
        assert len(mat.PROTOCOL_STEPS) == 36
        assert all(interfaces for _, _, interfaces in mat.PROTOCOL_STEPS)

    def test_smoke_self_check_is_labelled(self) -> None:
        result = mat.smoke_self_check()
        assert result["status"] == "smoke"
        assert result["claim_allowed"] is False
        assert result["session_valid"] is True
        assert result["not_evaluated_is_not_zero"] is True
        assert result["silent_fallback_is_high_risk"] is True
        assert result["workaround_needs_owner"] is True
