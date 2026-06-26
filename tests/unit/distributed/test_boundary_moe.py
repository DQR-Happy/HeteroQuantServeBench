"""E10-07 / E10-08 interface tests: PP/CP/SP gating and MoE permutation semantics."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.distributed import boundary as bd
from hqsb.distributed import moe as mo


@pytest.mark.unit
class TestBoundaryActivation:
    def test_inactive_p1_is_not_run_not_claimed(self):
        decision = bd.activation_decision(claiming_parallelism=False, tp_satisfies_goal=True)
        assert decision.activated is False
        assert decision.status == "NOT_RUN_NOT_CLAIMED"
        with pytest.raises(ConfigError):
            bd.ActivationDecision(activated=False, reason="x", status="PASS")

    def test_activated_needs_a_claim_scope(self):
        decision = bd.activation_decision(claiming_parallelism=True, tp_satisfies_goal=False)
        assert decision.activated is True and decision.claim_scope


@pytest.mark.unit
class TestBoundarySelection:
    def test_candidate_without_algorithm_is_refused(self):
        with pytest.raises(ConfigError):
            bd.CandidateDefinition(
                kind="sp", algorithm_name="  ", framework_impl="x",
                inference_semantics="y", communication_pattern="z",
            )

    def test_cp_sp_algorithm_must_be_named(self):
        with pytest.raises(ConfigError):
            bd.CandidateDefinition(
                kind="cp", algorithm_name="some_unknown_thing", framework_impl="x",
                inference_semantics="y", communication_pattern="z",
            )

    def test_capability_gate_and_rubric_selection(self):
        gate = bd.capability_gate(
            runtime_support=True, head_gqa_ok=True, kv_ok=False, collective_ok=True,
            dynamic_request_ok=True, hardware_ok=True,
        )
        assert gate["ok"] is False and "kv_ok" in gate["failed"]
        scores = bd.score_candidates(capability={"pp": True, "cp": False, "sp": False})
        selection = bd.select_primary(scores)
        assert selection["selected"] == "pp"
        assert selection["scores"][1]["total"] == 0

    def test_no_candidate_gives_a_negative_selection(self):
        scores = bd.score_candidates(capability={"pp": False, "cp": False, "sp": False})
        selection = bd.select_primary(scores)
        assert selection["selected"] is None
        assert selection["status"] == "PASS_NEGATIVE"


@pytest.mark.unit
class TestBoundaryPlans:
    def _census(self) -> "object":
        from hqsb.distributed.parallel_plan import QwenArchitectureCensus

        return QwenArchitectureCensus(
            family="qwen3", revision="fixture", num_layers=4, hidden_size=64,
            intermediate_size=128, num_attention_heads=8, num_kv_heads=4, head_dim=8,
            vocab_size=128, rms_norm_eps=1e-6,
        )

    def test_bubble_ideal_formula(self):
        ideal = bd.pipeline_bubble_ideal(stages=4, microbatches=4)
        assert ideal["ideal_utilization"] == pytest.approx(4 / 7)
        assert ideal["ideal_bubble_fraction"] == pytest.approx(3 / 7)

    def test_partition_conservation(self):
        good = bd.verify_partition(partitions={0: (0, 2), 1: (2, 4)}, total=4, name="layers")
        assert good["ok"] is True
        overlap = bd.verify_partition(partitions={0: (0, 3), 1: (2, 4)}, total=4, name="layers")
        assert overlap["ok"] is False
        gap = bd.verify_partition(partitions={0: (0, 1), 1: (2, 4)}, total=4, name="layers")
        assert gap["ok"] is False

    def test_stage_balance_uses_the_slowest_stage(self):
        rows = (
            bd.StageBalanceRow(0, 0, 2, 10.0, 100, 10, 5, 5),
            bd.StageBalanceRow(1, 1, 2, 25.0, 100, 10, 5, 5),
        )
        report = bd.stage_balance(rows)
        assert report["slowest_stage"] == 1
        assert report["imbalance_fraction"] > 0

    def test_bubble_metrics_expose_the_ideal_difference(self):
        metrics = bd.bubble_metrics(measured_idle_ms=30.0, total_ms=100.0, stages=2, microbatches=2)
        assert metrics["measured_bubble_fraction"] == pytest.approx(0.3)
        assert metrics["explained_by"]

    def test_context_plan_requires_a_named_algorithm_and_mask_note(self):
        plan = bd.ContextPlan(
            kind="cp", algorithm_name="ring_attention", degree=2,
            sequence_shards=((0, 64), (64, 128)), mask_correctness_note="causal mask per shard",
            kv_sharded=True,
        )
        metrics = bd.cp_sp_metrics(
            plan=plan, sequence_length=128, hidden=64, dtype_bytes=2, kv_heads_local=2,
            head_dim=8, layers=4,
        )
        assert metrics["per_rank_tokens"] == 64
        assert metrics["steps"] == 1

    def test_analytical_costs_write_down_assumptions(self):
        costs = bd.analytical_costs(self._census(), kind="pp", degree=2, stages=2, microbatches=4)
        assert costs.ideal_bubble_fraction == pytest.approx(1 / 5)
        assert costs.assumptions


@pytest.mark.unit
class TestBoundaryVerdict:
    def test_adopt_requires_criteria_and_no_regression(self):
        blocked = bd.adopt_reject_verdict(
            kind="pp", checklist={}, primary_improved=True, capacity_problem_solved=False
        )
        assert blocked["decision"] == "REJECT"
        adopted = bd.adopt_reject_verdict(
            kind="pp",
            checklist={name: True for name in bd.ADOPT_REJECT_CRITERIA},
            primary_improved=True,
            capacity_problem_solved=False,
        )
        assert adopted["decision"] == "ADOPT"
        regressed = bd.adopt_reject_verdict(
            kind="pp",
            checklist={name: True for name in bd.ADOPT_REJECT_CRITERIA},
            primary_improved=True,
            capacity_problem_solved=False,
            regressions=("TPOT +40%",),
        )
        assert regressed["decision"] == "REJECT"

    def test_boundary_verdict_when_not_activated(self):
        verdict = bd.boundary_verdict(
            activated=False, primary=None, correctness_ok=False, ledger_closed=False,
            decision="REJECT", confirmation_ok=False,
        )
        assert verdict["status"] == "NOT_RUN_NOT_CLAIMED"
        assert verdict["allowed_claim"] == "none"

    def test_health_probe_plan_lists_the_chain(self):
        plan = bd.health_probe_plan("pp")
        assert plan["chain"] == list(bd.PROBE_CHAIN)
        assert "timeout" in plan["unsupported_cases"]


@pytest.mark.unit
class TestMoeRouting:
    def test_claim_gate_downgrades_without_a_real_model(self):
        gate = mo.claim_gate(
            level="L4", has_real_moe_artifact=False, runtime_closed=True, model_quality_rows=False
        )
        assert gate["allowed_level"] == "L3"
        assert "model quality" in gate["must_not_claim"]

    def test_route_artifact_conservation_and_hash(self):
        route = mo.generate_route_artifact(
            artifact_id="r", token_count=16, world_size=2, experts_per_rank=2, top_k=2,
            profile="mild_zipf", seed=3,
        )
        assert len(route.assignments) == 32
        assert route.sha256
        placement = mo.round_robin_placement(num_experts=4, world_size=2)
        matrix = mo.count_matrix(route, expert_to_rank=placement.expert_to_rank, world_size=2)
        audit = matrix.audit()
        assert audit["ok"] is True
        assert sum(audit["send_tokens"]) == 32

    def test_route_rejects_a_wrong_assignment_count(self):
        with pytest.raises(ConfigError):
            mo.RouteArtifact(
                artifact_id="r", token_count=2, top_k=2, num_experts=2, assignments=(),
                profile="uniform", seed=0,
            )

    def test_skew_profiles_are_available_and_deterministic(self):
        for profile in mo.SKEW_PROFILES:
            route = mo.generate_route_artifact(
                artifact_id=f"r-{profile}", token_count=8, world_size=2, experts_per_rank=2,
                top_k=1, profile=profile, seed=5,
            )
            assert len(route.assignments) == 8
        first = mo.generate_route_artifact(
            artifact_id="a", token_count=8, world_size=2, experts_per_rank=2, top_k=1,
            profile="uniform", seed=1,
        )
        second = mo.generate_route_artifact(
            artifact_id="a", token_count=8, world_size=2, experts_per_rank=2, top_k=1,
            profile="uniform", seed=1,
        )
        assert first.sha256 == second.sha256


@pytest.mark.unit
class TestMoeOracle:
    def test_dispatch_and_combine_restore_token_order(self):
        route = mo.generate_route_artifact(
            artifact_id="r", token_count=4, world_size=2, experts_per_rank=2, top_k=2,
            profile="uniform", seed=11,
        )
        dispatch = mo.dispatch_oracle(route, alignment=2)
        assert dispatch.packed_count == 8
        outputs = {
            position: mo.deterministic_stub_expert(
                position=position, expert_id=dispatch.expert_order[position], hidden_size=4
            )
            for position in range(dispatch.packed_count)
        }
        combined = mo.combine_oracle(dispatch, expert_outputs=outputs, hidden_size=4)
        assert combined["token_conservation"] is True
        assert len(combined["tokens"]) == 4

    def test_combine_rejects_a_missing_expert_output(self):
        route = mo.generate_route_artifact(
            artifact_id="r", token_count=2, world_size=2, experts_per_rank=2, top_k=1,
            profile="uniform", seed=2,
        )
        dispatch = mo.dispatch_oracle(route)
        with pytest.raises(ConfigError):
            mo.combine_oracle(dispatch, expert_outputs={}, hidden_size=4)

    def test_compute_offsets_rejects_negative_counts(self):
        with pytest.raises(ConfigError):
            mo.compute_offsets([1, -1])
        assert mo.compute_offsets([1, 2], alignment=4) == [0, 4]

    def test_imbalance_metrics_cover_the_required_fields(self):
        metrics = mo.imbalance_metrics(
            tokens_per_expert=[10, 2, 2, 0], tokens_per_rank=[7, 7], expert_unit_costs=[1.0] * 4,
        )
        assert metrics["max_per_mean_expert"] > 1
        assert metrics["empty_experts"] == 1
        assert metrics["gini"] > 0 and metrics["entropy"] > 0

    def test_padding_waste_is_counted(self):
        # an odd token count makes the per-source rows unequal, so a fixed
        # AllToAll must pad; the accounting identity must hold either way
        route = mo.generate_route_artifact(
            artifact_id="r", token_count=5, world_size=2, experts_per_rank=2, top_k=1,
            profile="single_hot_expert", seed=4,
        )
        placement = mo.round_robin_placement(num_experts=4, world_size=2)
        waste = mo.padding_waste(
            route=route, expert_to_rank=placement.expert_to_rank, world_size=2, top_k=1
        )
        assert waste["fixed_slots"] - waste["actual_slots"] == waste["padding_slots"]
        assert waste["padding_slots"] > 0
        assert waste["fixed_slots"] >= waste["actual_slots"]

    def test_capacity_policy_reports_quality_semantics_change(self):
        operator = mo.MoeOperatorSpec(
            hidden_size=8, num_experts=4, top_k=2, capacity_factor=1.0, overflow_policy="drop"
        )
        rows = mo.capacity_policy_rows(operator=operator, tokens=8, assignments=[5, 1, 1, 1])
        dropping = [row for row in rows if row["policy"] == "drop"][0]
        assert dropping["quality_semantics_changed"] is True


@pytest.mark.unit
class TestMoePlacementAndVerdict:
    def test_placement_a_b_and_holdout(self):
        tuning = mo.generate_route_artifact(
            artifact_id="t", token_count=16, world_size=2, experts_per_rank=2, top_k=1,
            profile="single_hot_expert", seed=6, distribution_params={"hot_expert": 0},
        )
        baseline = mo.round_robin_placement(num_experts=4, world_size=2)
        treatment = mo.topology_aware_placement(route=tuning, world_size=2, num_experts=4)
        report = mo.placement_ab_rows(
            route=tuning, baseline=baseline, treatment=treatment, world_size=2
        )
        assert len(report["rows"]) == 2
        holdout = mo.holdout_check(tuning_route=tuning, holdout_route=tuning, policy=treatment)
        assert holdout["overfits"] is False and holdout["policy"] == treatment.name
        # a holdout route whose hot expert moved: the tuning placement must be
        # flagged as overfit with a robust fallback
        drifted = mo.generate_route_artifact(
            artifact_id="h", token_count=16, world_size=2, experts_per_rank=2, top_k=1,
            profile="single_hot_expert", seed=7, distribution_params={"hot_expert": 3},
        )
        assert drifted.tokens_per_expert() != tuning.tokens_per_expert()
        overfit = mo.holdout_check(
            tuning_route=tuning, holdout_route=drifted, policy=treatment, divergence_threshold=0.05
        )
        assert overfit["overfits"] is True and overfit["fallback_recommended"] is True

    def test_topology_aware_placement_respects_memory(self):
        route = mo.generate_route_artifact(
            artifact_id="r", token_count=8, world_size=2, experts_per_rank=2, top_k=1,
            profile="uniform", seed=8,
        )
        with pytest.raises(ConfigError):
            mo.topology_aware_placement(
                route=route, world_size=2, num_experts=4, max_memory_bytes_per_rank=10,
                expert_bytes=100,
            )

    def test_l4_gate_and_graded_verdict(self):
        gate = mo.l4_gate(level="L3", has_real_moe_artifact=False, model_rows=False)
        assert gate["ok"] is False
        verdict = mo.moe_verdict(
            highest_level="L2", conservation_ok=True, l2_or_l3_closed=True, skew_covered=True,
            placement_conclusion="round_robin kept", holdout_ok=True,
        )
        assert verdict["status"] == "PASS"
        assert "L4" in verdict["remaining_gap"]
        failed = mo.moe_verdict(
            highest_level="L2", conservation_ok=False, l2_or_l3_closed=True, skew_covered=True,
            placement_conclusion="x", holdout_ok=True,
        )
        assert failed["status"] == "FAIL"

    def test_phase_cases_and_expert_mlp_plan(self):
        cases = mo.phase_cases(prefill_tokens=64, decode_tokens=1, hidden_size=8)
        assert {case["phase"] for case in cases} == {"prefill", "decode"}
        operator = mo.MoeOperatorSpec(hidden_size=8, num_experts=2, top_k=1)
        plan = mo.expert_mlp_plan(operator=operator, tokens_per_expert=[3, 0])
        assert plan["per_expert"][1]["empty"] is True
