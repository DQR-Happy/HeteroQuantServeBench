"""E07-10 fair comparison and E07-07 strict A/B gates."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.runtime import comparison as C
from hqsb.runtime import policy_ab as P


def _identity(**overrides) -> C.BackendIdentity:
    payload = {
        "backend_id": "cloud",
        "role": "cloud_primary",
        "version": "1.0.0",
        "commit": "c0",
        "model_id": "Qwen/Qwen3-1.7B",
        "precision": "float16",
        "hardware": "host-a",
    }
    payload.update(overrides)
    return C.BackendIdentity(**payload)


def _spec(**overrides) -> C.ComparisonSpec:
    payload = {
        "model_id": "Qwen/Qwen3-1.7B",
        "model_manifest_sha256": "a" * 64,
        "precision": "float16",
        "hardware": "host-a",
        "request_trace_hash": "trace",
    }
    payload.update(overrides)
    return C.ComparisonSpec(**payload)


def _row(**overrides) -> C.ComparisonRow:
    payload = {
        "backend": _identity(),
        "workload": "short",
        "tier": C.TIER_A,
        "mode": "common_denominator",
        "metrics": {"ttft_ms": 10.0, "throughput": 100.0},
    }
    payload.update(overrides)
    return C.ComparisonRow(**payload)


@pytest.mark.unit
class TestComparisonSpec:
    def test_identity_fields_are_required(self):
        with pytest.raises(ConfigError):
            _spec(request_trace_hash="")

    def test_at_least_three_independent_processes(self):
        with pytest.raises(ConfigError):
            _spec(independent_processes=2)


@pytest.mark.unit
class TestTiers:
    def test_tier_is_derived_from_the_controlled_variables(self):
        spec = _spec()
        assert _identity().tier(spec) == C.TIER_A
        assert _identity(precision="w4a16").tier(spec) == C.TIER_B
        assert _identity(hardware="host-b").tier(spec) == C.TIER_C
        assert _identity(model_id="other", hardware="host-b").tier(spec) == C.TIER_D

    def test_tier_claims_are_explicit(self):
        assert set(C.TIER_CLAIMS) == {"A", "B", "C", "D"}

    def test_unknown_tier_refused(self):
        with pytest.raises(ConfigError):
            _row(tier="E")


@pytest.mark.unit
class TestRowsAndModes:
    def test_incomparable_row_needs_a_reason(self):
        with pytest.raises(ConfigError):
            _row(comparable=False)

    def test_na_row_keeps_its_reason(self):
        row = _row(comparable=False, na_reason="backend has no streaming")
        assert row.as_dict()["na_reason"] == "backend has no streaming"

    def test_quality_failure_blocks_a_comparable_row(self):
        with pytest.raises(ConfigError):
            _row(quality_gate_passed=False)

    def test_unknown_mode_refused(self):
        with pytest.raises(ConfigError):
            _row(mode="best_effort")

    def test_missing_capability_matrix_refused(self):
        with pytest.raises(ConfigError):
            C.common_denominator_rows([_row()], {})

    def test_common_denominator_records_excluded_features(self):
        features = {
            "cloud": {feature: True for feature in C.COMMON_DENOMINATOR_FEATURES},
            "edge": {feature: feature == "streaming" for feature in C.COMMON_DENOMINATOR_FEATURES},
        }
        rows = C.common_denominator_rows(
            [_row(backend=_identity(backend_id="cloud")), _row(backend=_identity(backend_id="edge"))],
            features,
        )
        excluded = {feature for row in rows for feature in row.excluded_features}
        assert "cuda_graph" in excluded
        assert all(row.mode == "common_denominator" for row in rows)

    def test_best_valid_table_must_be_preregistered(self):
        with pytest.raises(ConfigError):
            C.best_valid_rows([_row()], preregistered=False)
        rows = C.best_valid_rows([_row()], preregistered=True)
        assert rows[0].mode == "best_valid"


@pytest.mark.unit
class TestMetricRecompute:
    def test_every_rate_is_recomputed_from_raw(self):
        report = C.recompute_metrics(
            requests=[
                {"logical_input_tokens": 32, "committed_output_tokens": 8, "ttft_ms": 10.0, "tpot_ms": 2.0},
                {"logical_input_tokens": 16, "committed_output_tokens": 4, "ttft_ms": 12.0, "tpot_ms": 3.0},
            ],
            iterations=[
                {"model_runner_ms": 100.0, "scheduler_cpu_ms": 5.0, "sample_ms": 5.0},
                {"model_runner_ms": 90.0, "scheduler_cpu_ms": 5.0, "sample_ms": 5.0},
            ],
        )
        assert report["requests"] == 2
        assert report["token_ledger"]["useful_committed_tokens"] == 60
        assert report["output_tps"] == pytest.approx(12 / 0.21)
        assert report["ttft_ms"]["p50"] == pytest.approx(11.0)

    def test_inconsistent_ledger_blocks_recomputation(self):
        with pytest.raises(ConfigError):
            C.recompute_metrics(requests=[], iterations=[])

    def test_token_denominator_audit_compares_reported_with_recomputed(self):
        raw = C.recompute_metrics(
            requests=[{"logical_input_tokens": 8, "committed_output_tokens": 2}],
            iterations=[{"model_runner_ms": 10.0}],
        )
        audit = C.token_denominator_audit(
            raw=raw, reported={"output_tps": raw["output_tps"] * 2}
        )
        assert not audit["ok"]
        assert audit["rows"][0]["delta"] > 0
        assert audit["denominators"]["model_computed_positions"] == 10

    def test_missing_recomputed_value_is_reported(self):
        audit = C.token_denominator_audit(
            raw={"token_ledger": {"logical_input_tokens": 1}}, reported={"output_tps": 5.0}
        )
        assert not audit["ok"]


@pytest.mark.unit
class TestColdWarmAndPareto:
    def test_all_phases_must_be_reported(self):
        records = [
            C.PhaseRecord(phase=phase, backend_id="cloud", wall_ms=1.0)
            for phase in C.COLD_WARM_PHASES
        ]
        assert C.cold_warm_report(records)["ok"]
        report = C.cold_warm_report(records[:-1])
        assert not report["ok"]
        assert "close" in report["missing_phases"]["cloud"]

    def test_build_time_may_not_be_folded_into_steady(self):
        with pytest.raises(ConfigError):
            C.PhaseRecord(
                phase="install_build", backend_id="edge", wall_ms=1000.0, included_in_steady=True
            )

    def test_unknown_phase_refused(self):
        with pytest.raises(ConfigError):
            C.PhaseRecord(phase="warmup", backend_id="edge", wall_ms=1.0)

    def test_pareto_front_keeps_dominated_points_visible(self):
        report = C.pareto_front(
            [
                C.ParetoPoint(
                    backend_id="fast",
                    workload="short",
                    hardware="host-a",
                    objectives={"ttft_ms": 5.0, "throughput": 200.0},
                ),
                C.ParetoPoint(
                    backend_id="slow",
                    workload="short",
                    hardware="host-a",
                    objectives={"ttft_ms": 10.0, "throughput": 100.0},
                ),
            ]
        )
        front = report["fronts"][0]
        assert front["front"] == ["fast"]
        assert front["dominated"] == ["slow"]

    def test_fronts_are_computed_per_hardware_and_workload(self):
        report = C.pareto_front(
            [
                C.ParetoPoint(
                    backend_id="a", workload="short", hardware="host-a", objectives={"ttft_ms": 1.0}
                ),
                C.ParetoPoint(
                    backend_id="b", workload="long", hardware="host-a", objectives={"ttft_ms": 9.0}
                ),
            ]
        )
        assert len(report["fronts"]) == 2

    def test_run_comparison_requires_the_same_class(self):
        left = C.RunSamples(backend_id="a", workload="short", request_class="short", values=(1.0,))
        right = C.RunSamples(backend_id="b", workload="short", request_class="long", values=(1.0,))
        with pytest.raises(ConfigError):
            C.compare_runs(left, right)

    def test_no_winner_when_the_interval_crosses_zero(self):
        left = C.RunSamples(backend_id="a", workload="short", request_class="short", values=(1.0, 2.0, 3.0))
        right = C.RunSamples(backend_id="b", workload="short", request_class="short", values=(1.0, 2.0, 3.0))
        assert C.compare_runs(left, right)["winner"] == ""

    def test_limitations_report_keeps_na_rows(self):
        report = C.limitations_report(
            [
                _row(comparable=False, na_reason="no prefix cache", observability_limitation="black box")
            ]
        )
        assert report["na_rows"]
        assert report["observability_limitations"]

    def test_s08_surface_excludes_private_objects(self):
        surface = C.s08_interface_surface()
        assert "close" in surface["backend_operations"]
        assert any("private" in item for item in surface["forbidden"])


@pytest.mark.unit
class TestSelectionGateAndAdr:
    def _gate(self, **overrides) -> P.SelectionGate:
        payload = {
            "baseline_bottleneck_quantified": True,
            "mechanism_explainable": True,
            "metric_preregistered": True,
            "not_duplicating_upstream": True,
            "risk_and_fallback_controllable": True,
            "counterexample_constructible": True,
            "scope_consistent_with_s08_s11": True,
            "evidence_refs": ("E07-02:kv",),
        }
        payload.update(overrides)
        return P.SelectionGate(**payload)

    def test_ineligible_change_is_refused(self):
        gate = self._gate(baseline_bottleneck_quantified=False)
        assert not gate.ok
        with pytest.raises(ConfigError):
            gate.require_ok()

    def test_eligible_change_needs_evidence_references(self):
        with pytest.raises(ConfigError):
            self._gate(evidence_refs=()).require_ok()
        self._gate().require_ok()

    def test_adr_requires_metrics_workloads_and_rollback(self):
        base = dict(
            decision_id="ADR-1",
            candidate="kv_block_size_or_allocation",
            problem="fragmentation",
            baseline_evidence="E07-03",
            hypothesis="larger blocks reduce metadata",
            mechanism="fewer block table entries",
            change_scope="kv allocator",
            expected_benefit="lower tpot",
            expected_regressions=("larger tail waste",),
            invariants=("token conservation",),
            metrics=("tpot_ms",),
            workloads=("short",),
            rollback="restore block size 16",
            stop_rule="stop after 3 blocks",
            non_goals=("prefix cache",),
        )
        adr = P.Adr(**base)
        assert adr.primary_metric == "tpot_ms"
        with pytest.raises(ConfigError):
            P.Adr(**{**base, "metrics": ()})
        with pytest.raises(ConfigError):
            P.Adr(**{**base, "rollback": ""})

    def test_unknown_candidate_refused(self):
        with pytest.raises(ConfigError):
            P.Adr(
                decision_id="ADR-1",
                candidate="rewrite_everything",
                problem="p",
                baseline_evidence="b",
                hypothesis="h",
                mechanism="m",
                change_scope="s",
                expected_benefit="e",
                expected_regressions=(),
                invariants=("i",),
                metrics=("tpot_ms",),
                workloads=("short",),
                rollback="r",
                stop_rule="s",
                non_goals=(),
            )


@pytest.mark.unit
class TestAbIdentityAndSchedule:
    def _identity(self, patch: str = "p1", **overrides) -> P.AbIdentity:
        payload = {
            "model_id": "Qwen/Qwen3-1.7B",
            "tokenizer_id": "Qwen/Qwen3-1.7B",
            "precision": "float16",
            "runtime_base_commit": "base",
            "hardware": "host-a",
            "request_trace_hash": "trace",
            "scheduler_config_hash": "sched",
            "kv_graph_attention_config_hash": "kv",
            "warmup_policy": "warm",
            "measurement_policy": "measure",
            "seed": 0,
            "patch_hash": patch,
            "build_hash": "build",
        }
        payload.update(overrides)
        return P.AbIdentity(**payload)

    def test_only_the_patch_may_differ(self):
        report = P.identity_equal(self._identity(), self._identity("p2"))
        assert report["ok"]
        P.require_identity_equal(self._identity(), self._identity("p2"))

    def test_other_differences_are_refused(self):
        report = P.identity_equal(self._identity(), self._identity("p2", seed=1))
        assert not report["ok"]
        assert report["differing_fields"] == ["seed"]

    def test_identical_artifacts_are_refused(self):
        with pytest.raises(ConfigError):
            P.require_identity_equal(self._identity(), self._identity())

    def test_abba_schedule_is_balanced_and_interleaved(self):
        order = P.block_schedule(blocks=3, scheme="ABBA")
        report = P.schedule_balance(order)
        assert report["ok"]
        assert report["arm_a"] == report["arm_b"]

    def test_random_block_schedule_needs_a_seed(self):
        with pytest.raises(ConfigError):
            P.block_schedule(blocks=3, scheme="RANDOM_BLOCKS")
        order = P.block_schedule(blocks=3, scheme="RANDOM_BLOCKS", seed=7)
        assert P.schedule_balance(order)["ok"]

    def test_too_few_blocks_refused(self):
        with pytest.raises(ConfigError):
            P.block_schedule(blocks=2)

    def test_long_consecutive_run_is_flagged(self):
        assert not P.schedule_balance(["A", "A", "A", "A", "B", "B", "B", "B"])["ok"]


@pytest.mark.unit
class TestCausalChainAndDecision:
    def _adr(self) -> P.Adr:
        return P.Adr(
            decision_id="ADR-1",
            candidate="token_budget_or_chunk",
            problem="decode stall",
            baseline_evidence="E07-02",
            hypothesis="smaller chunks reduce ITL",
            mechanism="decode shares the token budget",
            change_scope="scheduler",
            expected_benefit="lower ITL",
            expected_regressions=("long TTFT",),
            invariants=("token conservation",),
            metrics=("tpot_ms",),
            workloads=("short",),
            rollback="restore chunk size",
            stop_rule="stop after 3 blocks",
            non_goals=("prefix cache",),
        )

    def test_attributable_chain_requires_all_conditions(self):
        chain = P.causal_chain_check(
            patch_changed_mechanism=True,
            proximate_metric_changed=True,
            phase_metric_changed=True,
            end_to_end_changed=True,
            ablation_isolates_mechanism=True,
        )
        assert chain["attributable"]

    def test_end_to_end_change_without_proximate_change_is_not_attributable(self):
        chain = P.causal_chain_check(
            patch_changed_mechanism=True,
            proximate_metric_changed=False,
            phase_metric_changed=True,
            end_to_end_changed=True,
            ablation_isolates_mechanism=True,
        )
        assert not chain["attributable"]
        assert any("not attributable" in item for item in chain["problems"])

    def test_amdahl_case_is_reported_as_a_limit(self):
        chain = P.causal_chain_check(
            patch_changed_mechanism=True,
            proximate_metric_changed=True,
            phase_metric_changed=True,
            end_to_end_changed=False,
            ablation_isolates_mechanism=True,
        )
        assert not chain["attributable"]
        assert any("Amdahl" in item for item in chain["problems"])

    def test_decision_uses_the_preregistered_primary(self):
        adr = self._adr()
        effect = P.measure_ab(
            metric="latency_ms",
            arm_a_values=[10.0, 11.0, 12.0],
            arm_b_values=[1.0, 1.0, 1.0],
            guard_band=0.5,
            better="lower",
        )
        with pytest.raises(ConfigError):
            P.decide(
                adr=adr,
                effect=effect,
                causal={"attributable": True},
                regressions={},
                correctness_ok=True,
                safety_ok=True,
            )

    def test_benefit_merges_when_the_chain_is_attributable(self):
        adr = self._adr()
        effect = P.measure_ab(
            metric="tpot_ms",
            arm_a_values=[10.0, 11.0, 12.0],
            arm_b_values=[1.0, 1.0, 1.0],
            guard_band=0.5,
            better="lower",
        )
        decision = P.decide(
            adr=adr,
            effect=effect,
            causal={"attributable": True},
            regressions={},
            correctness_ok=True,
            safety_ok=True,
        )
        assert decision["decision"] == "MERGE"

    def test_correctness_failure_forces_a_rollback(self):
        adr = self._adr()
        effect = P.measure_ab(
            metric="tpot_ms",
            arm_a_values=[10.0, 11.0, 12.0],
            arm_b_values=[1.0, 1.0, 1.0],
            guard_band=0.5,
            better="lower",
        )
        decision = P.decide(
            adr=adr,
            effect=effect,
            causal={"attributable": True},
            regressions={},
            correctness_ok=False,
            safety_ok=True,
        )
        assert decision["decision"] == "ROLLBACK"
        assert "correctness" in decision["reason"]

    def test_guardrail_regression_forces_a_rollback(self):
        adr = self._adr()
        effect = P.measure_ab(
            metric="tpot_ms",
            arm_a_values=[10.0, 11.0, 12.0],
            arm_b_values=[1.0, 1.0, 1.0],
            guard_band=0.5,
            better="lower",
        )
        decision = P.decide(
            adr=adr,
            effect=effect,
            causal={"attributable": True},
            regressions={"regressions": ["long/decode_heavy"]},
            correctness_ok=True,
            safety_ok=True,
        )
        assert decision["decision"] == "ROLLBACK"

    def test_attributable_but_insignificant_is_a_negative_result(self):
        adr = self._adr()
        effect = P.measure_ab(
            metric="tpot_ms",
            arm_a_values=[10.0, 10.0, 10.0],
            arm_b_values=[10.0, 10.0, 10.0],
            guard_band=0.1,
            better="lower",
        )
        decision = P.decide(
            adr=adr,
            effect=effect,
            causal={"attributable": True},
            regressions={},
            correctness_ok=True,
            safety_ok=True,
        )
        assert decision["decision"] == "PASS_NEGATIVE"

    def test_incomplete_chain_is_inconclusive(self):
        adr = self._adr()
        effect = P.measure_ab(
            metric="tpot_ms",
            arm_a_values=[10.0, 10.5, 11.0],
            arm_b_values=[1.0, 1.0, 1.0],
            guard_band=0.1,
            better="lower",
        )
        decision = P.decide(
            adr=adr,
            effect=effect,
            causal={"attributable": False, "problems": ["no phase metric"]},
            regressions={},
            correctness_ok=True,
            safety_ok=True,
        )
        assert decision["decision"] == "INCONCLUSIVE"

    def test_regression_envelope_flags_guardrail_breaches(self):
        effect = P.measure_ab(
            metric="tpot_ms",
            arm_a_values=[10.0, 11.0, 12.0],
            arm_b_values=[1.0, 1.0, 1.0],
            guard_band=0.5,
            better="lower",
        )
        envelope = P.regression_envelope(
            [P.RegressionRow(workload="long", request_class="long_prefill", effect=effect, guardrail=0.5)]
        )
        assert envelope["regressions"] == ["long/long_prefill"]

    def test_pilot_runs_may_not_enter_the_final_matrix(self):
        assert P.pilot_separation(["p1"], ["f1"])["ok"]
        assert not P.pilot_separation(["shared"], ["shared"])["ok"]

    def test_rollback_plan_keeps_the_reference_path(self):
        plan = P.rollback_plan(self._adr())
        assert plan["feature_flag_required"]
        assert plan["reference_path_preserved"]
