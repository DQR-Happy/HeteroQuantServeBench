"""Unit tests for the E12-08 Pareto frontier and recommendation machinery."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.evaluation import pareto

pytestmark = pytest.mark.unit


def _profile(**overrides: object) -> pareto.BusinessProfile:
    payload = {
        "profile_id": "interactive",
        "version": "v1",
        "owner": "o",
        "effective_date": "2026-09-19",
        "workload_weights": {"interactive": 1.0},
        "quality_gate_id": "g1",
        "latency_slos": {"p99_ttft": {"p99": 500.0}},
        "goodput_demand": 10.0,
        "context_distribution": {"short": 1.0},
        "output_distribution": {"short": 1.0},
        "capacity": {"max_context_tokens": 4096},
        "power_constraints": {},
        "cost_scenario_id": "s1",
        "budget": 100.0,
        "availability": 0.99,
        "redundancy": "N+1",
        "required_capabilities": (),
        "risk_tolerance": "medium",
        "confidence_requirement": 0.9,
        "objectives": (("goodput", "maximize"), ("cost", "minimize")),
        "preference_policy": "lexicographic",
    }
    payload.update(overrides)
    return pareto.BusinessProfile(**payload)  # type: ignore[arg-type]


OBJECTIVES = (("goodput", "maximize"), ("cost", "minimize"))


class TestProfiles:
    def test_all_six_profiles_are_registered(self) -> None:
        assert set(pareto.profile_templates()) == set(pareto.PROFILE_IDS)

    def test_a_name_without_fields_cannot_run(self) -> None:
        problems = _profile(quality_gate_id="", objectives=()).validate()
        assert any("quality gate" in item for item in problems)
        assert any("objectives" in item for item in problems)

    def test_weights_must_sum_to_one(self) -> None:
        problems = _profile(workload_weights={"a": 0.5, "b": 0.6}).validate()
        assert any("sum to 1.0" in item for item in problems)

    def test_unknown_profile_id_is_rejected(self) -> None:
        assert _profile(profile_id="interactive").validate() == []
        assert any("unknown profile" in item for item in _profile(profile_id="gaming").validate())


class TestDominance:
    def test_clear_dominance(self) -> None:
        a = {"candidate_id": "a", "goodput": 10.0, "cost": 1.0}
        b = {"candidate_id": "b", "goodput": 9.0, "cost": 2.0}
        assert pareto.dominates(a, b, OBJECTIVES)["dominates"] is True
        assert pareto.dominates(b, a, OBJECTIVES)["dominates"] is False

    def test_tradeoff_is_not_dominance(self) -> None:
        a = {"candidate_id": "a", "goodput": 10.0, "cost": 2.0}
        b = {"candidate_id": "b", "goodput": 9.0, "cost": 1.0}
        assert pareto.dominates(a, b, OBJECTIVES)["dominates"] is False
        assert pareto.dominates(b, a, OBJECTIVES)["dominates"] is False

    def test_missing_objective_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            pareto.dominates({"candidate_id": "a", "goodput": 1.0}, {"candidate_id": "b", "goodput": 1.0}, OBJECTIVES)

    def test_frontier_and_ledger(self) -> None:
        rows = [
            {"candidate_id": "a", "goodput": 10.0, "cost": 1.0},
            {"candidate_id": "b", "goodput": 9.0, "cost": 2.0},
            {"candidate_id": "c", "goodput": 5.0, "cost": 0.5},
        ]
        assert set(pareto.pareto_frontier(rows, OBJECTIVES)) == {"a", "c"}
        ledger = pareto.dominance_ledger(rows, OBJECTIVES)
        dominated_b = next(row for row in ledger if row["dominated"] == "b")
        assert dominated_b["dominated_by"] == ["a"]

    def test_tolerance_keeps_nearly_tied_points(self) -> None:
        rows = [
            {"candidate_id": "a", "goodput": 10.0, "cost": 1.0},
            {"candidate_id": "b", "goodput": 9.99, "cost": 1.0},
        ]
        assert set(pareto.pareto_frontier(rows, OBJECTIVES)) == {"a"}
        assert set(pareto.pareto_frontier(rows, OBJECTIVES, tolerance=0.1)) == {"a", "b"}

    def test_near_frontier_and_conservative(self) -> None:
        rows = [
            {"candidate_id": "a", "goodput": 10.0, "cost": 1.0},
            {"candidate_id": "b", "goodput": 9.0, "cost": 1.0},
        ]
        near = pareto.near_frontier(rows, OBJECTIVES, tolerance=2.0)
        assert any(row["candidate_id"] == "b" and row["near"] for row in near)
        conservative = pareto.conservative_frontier(
            [
                {"candidate_id": "a", "goodput": 10.0, "goodput_low": 8.0, "cost": 1.0, "cost_high": 3.0},
                {"candidate_id": "b", "goodput": 9.0, "goodput_low": 9.0, "cost": 1.0, "cost_high": 1.0},
            ],
            OBJECTIVES,
        )
        assert "b" in conservative

    def test_membership_probability_is_bounded(self) -> None:
        draws = [
            [{"candidate_id": "a", "goodput": 10.0, "cost": 1.0}, {"candidate_id": "b", "goodput": 9.0, "cost": 2.0}],
            [{"candidate_id": "a", "goodput": 1.0, "cost": 5.0}, {"candidate_id": "b", "goodput": 2.0, "cost": 4.0}],
        ]
        report = pareto.membership_probability(draws, objectives=OBJECTIVES)
        assert set(report["membership_probability"]) == {"a", "b"}
        assert all(0.0 <= value <= 1.0 for value in report["membership_probability"].values())


class TestEvidenceAndFeasibility:
    def _evidence(self) -> list:
        return [
            {"candidate_id": "a", "status": "EVIDENCE_VALID"},
            {"candidate_id": "b", "status": "EVIDENCE_QUALITY_FAIL"},
        ]

    def test_quality_fail_is_eliminated_even_if_fastest(self) -> None:
        evidence = pareto.evidence_gate(
            [
                {
                    "candidate_id": "fast",
                    "goodput": 100.0,
                    "quality_status": "QUALITY_GATE_FAILED",
                    "comparability_status": "COMPARABLE",
                    "source_result_id": "r",
                }
            ],
            _profile(),
        )
        assert evidence[0]["status"] == "EVIDENCE_QUALITY_FAIL"

    def test_feasible_set_separates_all_four_statuses(self) -> None:
        rows = [
            {"candidate_id": "a", "p99_ttft": 100.0},
            {"candidate_id": "b", "p99_ttft": 100.0},
            {"candidate_id": "c", "p99_ttft": 900.0},
            {"candidate_id": "d", "p99_ttft": 100.0},
        ]
        evidence = [
            {"candidate_id": "a", "status": "EVIDENCE_VALID"},
            {"candidate_id": "b", "status": "EVIDENCE_QUALITY_FAIL"},
            {"candidate_id": "c", "status": "EVIDENCE_VALID"},
            {"candidate_id": "d", "status": "EVIDENCE_INCOMPLETE"},
        ]
        constraints = [pareto.Constraint("slo", "p99_ttft", "<=", 500.0, "ms")]
        report = pareto.feasible_set(rows, constraints=constraints, evidence=evidence)
        assert report["feasible"] == ["a"]
        assert report["infeasible"] == ["b", "c"]
        assert report["insufficient"] == ["d"]

    def test_interval_crossing_is_conditional(self) -> None:
        constraints = [pareto.Constraint("slo", "p99_ttft", "<=", 500.0, "ms")]
        checked = pareto.evaluate_constraints(
            {"candidate_id": "x", "p99_ttft": 500.0, "p99_ttft_low": 400.0, "p99_ttft_high": 600.0},
            constraints,
        )
        assert any(row["verdict"] == "conditional" for row in checked["constraints"])

    def test_constraint_slack_flags_borderline(self) -> None:
        rows = [{"candidate_id": "a", "p99_ttft": 499.0}]
        slacks = pareto.constraint_slack_rows(rows, [pareto.Constraint("slo", "p99_ttft", "<=", 500.0, "ms")])
        assert slacks[0]["borderline"] is True

    def test_no_feasible_candidate_is_a_first_class_outcome(self) -> None:
        report = pareto.no_feasible_candidate(
            profile=_profile(), infeasible_rows=[{"candidate_id": "x", "reason": "over SLO"}]
        )
        assert report["status"] == "NO_FEASIBLE_CANDIDATE"


class TestMaturityAndSelection:
    def test_maturity_has_three_legal_roles(self) -> None:
        hard = pareto.maturity_as_constraint(
            ("service_open_loop",), [{"capability": "service_open_loop", "status": "VERIFIED"}]
        )
        assert hard["missing"] == []
        hard_missing = pareto.maturity_as_constraint(("service_open_loop",), [])
        assert hard_missing["eliminates_on_missing"] is True
        objective = pareto.maturity_as_objective(hours=20.0, failure_rate=0.05)
        assert objective["role"] == "independent_objective"
        risk = pareto.maturity_as_risk_label(level="medium", evidence=("e1",))
        assert risk["role"] == "risk_label"
        with pytest.raises(ConfigError):
            pareto.maturity_as_risk_label(level="extreme", evidence=())

    def test_secondary_selection_explicit_weights_only(self) -> None:
        rows = [
            {"candidate_id": "a", "goodput": 10.0, "cost": 1.0},
            {"candidate_id": "c", "goodput": 5.0, "cost": 0.5},
        ]
        with pytest.raises(ConfigError):
            pareto.secondary_selection(rows, objectives=OBJECTIVES, policy="explicit_weights", weights={"goodput": 1.0})
        report = pareto.secondary_selection(
            rows, objectives=OBJECTIVES, policy="explicit_weights", weights={"goodput": 0.2, "cost": 0.8}
        )
        # score_a = 0.2*10 + 0.8*(-1) = 1.2; score_c = 0.2*5 + 0.8*(-0.5) = 0.6
        assert report["selected"] == "a"
        assert report["alternative_if_preferences_change"] == "c"
        # different weights flip the choice: a legal, documented reversal
        flipped = pareto.secondary_selection(
            rows, objectives=OBJECTIVES, policy="explicit_weights", weights={"goodput": 0.0, "cost": 1.0}
        )
        assert flipped["selected"] == "c"

    def test_lexicographic_selection_reports_ties(self) -> None:
        rows = [
            {"candidate_id": "a", "goodput": 10.0, "cost": 1.0},
            {"candidate_id": "b", "goodput": 10.0, "cost": 2.0},
        ]
        report = pareto.secondary_selection(rows, objectives=(("goodput", "maximize"),), policy="lexicographic")
        assert report["remaining_ties"]  # both tie on goodput


class TestRecommendation:
    def _card(self, **overrides: object) -> pareto.Recommendation:
        payload = {
            "profile_id": "interactive",
            "version": "v1",
            "effective_date": "2026-09-19",
            "candidate_id": "a",
            "role": "primary",
            "feasibility_status": "feasible",
            "frontier_status": "on_frontier",
            "membership_probability": 0.9,
            "binding_constraints": ("slo",),
            "constraint_slacks": {"slo": 100.0},
            "objective_values": {"goodput": 10.0, "cost": 1.0},
            "quality_status": "pass",
            "comparability_status": "COMPARABLE",
            "stability_status": "stable",
            "evidence_status": "EVIDENCE_VALID",
            "cost_scenario_id": "s1",
            "energy_boundary": "device",
            "maturity_record_ids": (),
            "selection_policy": "lexicographic",
            "preference_assumptions": ("slo first",),
            "risks": ("queue explosion",),
            "limitations": ("device-level energy only",),
            "forbidden_claims": ("no absolute hardware ranking",),
            "refresh_triggers": ("price change",),
            "source_result_ids": ("r1",),
        }
        payload.update(overrides)
        return pareto.recommendation_card(**payload)  # type: ignore[arg-type]

    def test_valid_card(self) -> None:
        card = self._card()
        assert card.validate() == []
        assert card.role == "primary"

    def test_infeasible_candidate_cannot_be_primary(self) -> None:
        with pytest.raises(ConfigError):
            self._card(feasibility_status="infeasible")

    def test_card_needs_sources_claims_and_triggers(self) -> None:
        with pytest.raises(ConfigError):
            self._card(source_result_ids=())
        with pytest.raises(ConfigError):
            self._card(forbidden_claims=())
        with pytest.raises(ConfigError):
            self._card(refresh_triggers=())

    def test_independent_review_reproduces_frontier(self) -> None:
        card = self._card()
        rows = [
            {"candidate_id": "a", "goodput": 10.0, "cost": 1.0},
            {"candidate_id": "b", "goodput": 9.0, "cost": 2.0},
        ]
        review = pareto.independent_decision_review(card, rows, OBJECTIVES, reviewer="r2")
        assert review["reproduced_frontier_membership"] is True
        disagree = pareto.independent_decision_review(self._card(candidate_id="b", role="alternative"), rows, OBJECTIVES, reviewer="r2")
        assert disagree["reproduced_frontier_membership"] is False


class TestDecisionRegression:
    def test_suite_covers_all_ten_cases(self) -> None:
        report = pareto.run_decision_regression_suite()
        assert {row["case_id"] for row in report["cases"]} == set(pareto.DECISION_REGRESSION_CASES)
        assert report["ok"] is True

    def test_suite_is_algorithm_regression_not_experiment(self) -> None:
        report = pareto.run_decision_regression_suite()
        assert report["kind"] == "algorithm_regression"
        assert report["claim_allowed"] is False


class TestSmoke:
    def test_protocol_steps_are_complete(self) -> None:
        assert len(pareto.PROTOCOL_STEPS) == 36
        assert all(interfaces for _, _, interfaces in pareto.PROTOCOL_STEPS)

    def test_smoke_self_check_is_labelled(self) -> None:
        result = pareto.smoke_self_check()
        assert result["status"] == "smoke"
        assert result["claim_allowed"] is False
        assert result["profiles_have_all_six"] is True
        assert result["weights_must_sum_to_one"] is True
        assert result["regression_suite_passes"] is True
