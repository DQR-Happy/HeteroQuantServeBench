"""Coverage for the E11-06 autotune search/budget/holdout interfaces."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.compiler import autotune as at
from hqsb.compiler import records as rec


def _space(**overrides: object) -> at.SearchSpaceSpec:
    payload = dict(
        space_id="space_v1",
        semantic_op="hqsb::fused_add_rms_norm",
        version="1.0.0",
        target_id="sm86",
        shape_domain_id="decode_m1",
        knobs=(
            at.Knob("BLOCK", "int", (256, 512, 1024), "hidden tile"),
            at.Knob("warps", "int", (4, 8), "warp count"),
        ),
        constraints=(
            at.SearchConstraint(
                "BLOCK <= hidden",
                "shape",
                "a tile larger than the row wastes lanes",
                predicate=lambda config: config["BLOCK"] <= 4096,
            ),
        ),
        default_candidate={"BLOCK": 256, "warps": 4},
        heuristic_policy="BLOCK=512,warps=4",
        max_candidates=64,
        seed=7,
        correctness_policy_id="common_s06",
        benchmark_policy_id="common_timing",
    )
    payload.update(overrides)
    return at.SearchSpaceSpec(**payload)  # type: ignore[arg-type]


def _trial(**overrides: object) -> rec.AutotuneTrialRecord:
    payload = dict(
        session_id="s1",
        task_id="t1",
        shape_group="decode_m1",
        split="tuning",
        candidate_config={"BLOCK": 256, "warps": 4},
        legality_status="legal",
        compile_status="ok",
        correctness_status="pass",
        raw_timing_samples=(1.0, 1.1, 0.9),
    )
    payload.update(overrides)
    return rec.AutotuneTrialRecord(**payload)  # type: ignore[arg-type]


@pytest.mark.unit
class TestSearchSpace:
    def test_valid_space_passes(self) -> None:
        assert _space().validate() == []

    def test_knobs_need_meaning_and_values(self) -> None:
        knob = at.Knob("BLOCK", "int", (), "")
        problems = knob.validate()
        assert any("values" in problem for problem in problems)
        assert any("mechanism" in problem for problem in problems)

    def test_unknown_knob_type_is_rejected(self) -> None:
        assert any("knob type" in problem for problem in at.Knob("x", "magic", (1,), "why").validate())

    def test_constraints_need_reasons_and_predicates(self) -> None:
        constraint = at.SearchConstraint("BLOCK <= hidden", "shape", "")
        assert any("reason" in problem for problem in constraint.validate())
        with pytest.raises(ConfigError):
            constraint.allows({"BLOCK": 1})

    def test_space_requires_policies_and_budget(self) -> None:
        problems = _space(correctness_policy_id="", max_candidates=0).validate()
        assert any("correctness" in problem for problem in problems)
        assert any("max_candidates" in problem for problem in problems)

    def test_digest_is_stable(self) -> None:
        assert _space().digest() == _space().digest()

    def test_default_candidate_must_be_declared(self) -> None:
        assert any("default candidate" in problem for problem in _space(default_candidate={}).validate())


@pytest.mark.unit
class TestCandidateGeneration:
    def test_generation_applies_static_constraints(self) -> None:
        space = _space(
            constraints=(
                at.SearchConstraint(
                    "BLOCK <= 512",
                    "shape",
                    "simulated shape limit",
                    predicate=lambda config: config["BLOCK"] <= 512,
                ),
            )
        )
        rows = at.generate_candidates(space)
        assert len(rows) == 6
        invalid = [row for row in rows if row.status == "invalid"]
        assert len(invalid) == 2
        assert all(row.reject_reason == "AUTOTUNE_INVALID:shape" for row in invalid)

    def test_candidate_identity_depends_on_config_and_build(self) -> None:
        space = _space()
        first = at.candidate_identity(space, {"BLOCK": 256, "warps": 4}, kernel_build_id="b1")
        second = at.candidate_identity(space, {"BLOCK": 512, "warps": 4}, kernel_build_id="b1")
        third = at.candidate_identity(space, {"BLOCK": 256, "warps": 4}, kernel_build_id="b2")
        assert len({first, second, third}) == 3

    def test_generation_refuses_to_exceed_the_declared_budget(self) -> None:
        with pytest.raises(ConfigError):
            at.generate_candidates(_space(max_candidates=2))

    def test_filter_false_reject_audit(self) -> None:
        space = _space(
            constraints=(
                at.SearchConstraint(
                    "BLOCK <= 512", "shape", "limit", predicate=lambda config: config["BLOCK"] <= 512
                ),
            )
        )
        rows = at.generate_candidates(space)
        rejected = next(row for row in rows if row.status == "invalid")
        audit = at.filter_false_reject_audit(rows, compile_fn={rejected.identity: "compile_failed"})
        assert audit["false_rejects"] == []
        false_reject = at.filter_false_reject_audit(rows, compile_fn={rejected.identity: "compiled"})
        assert false_reject["false_rejects"]


@pytest.mark.unit
class TestBudgets:
    def test_ladder_is_monotone(self) -> None:
        ladder = at.budget_ladder(space_size=32)
        generated = [budget.max_generated for budget in ladder]
        measured = [budget.max_measured for budget in ladder]
        assert generated == sorted(generated)
        assert measured == sorted(measured)
        assert [budget.name for budget in ladder] == [
            "B0_default",
            "B1_small",
            "B2_medium",
            "B3_exhaustive",
        ]

    def test_budget_requires_early_stop_and_sane_bounds(self) -> None:
        budget = at.Budget("B1", 4, 3, 1, 1.0, 1.0, 1.0, 1, early_stop="")
        assert any("early-stop" in problem for problem in budget.validate())
        assert at.Budget("B1", 4, 3, 1, 1.0, 1.0, 1.0, 1, "stop").validate() == []

    def test_budget_fairness_detects_mismatch(self) -> None:
        ladder = at.budget_ladder(space_size=8)
        fair = at.check_budget_fairness([("a", ladder[1]), ("b", ladder[1])])
        assert fair["fair"] is True
        unfair = at.check_budget_fairness([("a", ladder[1]), ("b", ladder[2])])
        assert unfair["fair"] is False


@pytest.mark.unit
class TestSplits:
    def _groups(self) -> list:
        return [
            at.ShapeGroup(f"g{index}", "family", "prefill", str(index), "fp16", "contig", "sm_86")
            for index in range(10)
        ]

    def test_group_split_has_no_leakage(self) -> None:
        manifest = at.build_split_manifest(self._groups())
        report = at.split_leak_check(manifest["manifest"])
        assert report["leak_free"] is True
        assert manifest["manifest"]["holdout"]

    def test_duplicate_group_keys_are_rejected(self) -> None:
        groups = self._groups()
        duplicated = list(groups) + [at.ShapeGroup("g10", "family", "prefill", "0", "fp16", "contig", "sm_86")]
        with pytest.raises(ConfigError):
            at.build_split_manifest(duplicated)

    def test_leave_one_group_out_needs_a_declared_holdout(self) -> None:
        with pytest.raises(ConfigError):
            at.build_split_manifest(self._groups(), strategy="leave_one_shape_group_out")
        manifest = at.build_split_manifest(
            self._groups(), strategy="leave_one_shape_group_out", holdout_id="g3"
        )
        assert manifest["manifest"]["holdout"] == ["g3"]

    def test_unknown_strategy_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            at.build_split_manifest(self._groups(), strategy="random")

    def test_holdout_kinds_table_documents_all_kinds(self) -> None:
        assert {row["kind"] for row in at.holdout_kinds_table()} == set(at.HOLDOUT_KINDS)


@pytest.mark.unit
class TestSandboxAndOrder:
    def test_sandbox_requires_limits_and_isolated_output(self) -> None:
        sandbox = at.TrialSandbox(timeout_s=0.0, memory_limit_mb=0, output_dir="")
        assert len(sandbox.validate()) == 3
        assert at.TrialSandbox(timeout_s=1.0, memory_limit_mb=64, output_dir="tmp").validate() == []

    def test_reset_protocol_and_verification(self) -> None:
        protocol = at.reset_protocol(tensors=["x", "residual"], kv=True)
        assert "residual" in protocol["reset_tensors"]
        assert at.verify_reset({"x": "h"}, {"x": "h"})["ok"] is True
        assert at.verify_reset({"x": "h"}, {"x": "other"})["ok"] is False

    def test_measurement_schedule_is_seeded_and_rotated(self) -> None:
        first = at.measurement_schedule(["a", "b", "c"], seed=1, batches=2)
        second = at.measurement_schedule(["a", "b", "c"], seed=1, batches=2)
        assert first["order"] == second["order"]
        assert sorted(first["order"]) == sorted(["a", "b", "c"] * 2)

    def test_measurement_schedule_rejects_empty_input(self) -> None:
        with pytest.raises(ConfigError):
            at.measurement_schedule([], seed=1)


@pytest.mark.unit
class TestTrials:
    def test_invalid_candidate_is_never_timed(self) -> None:
        trial = _trial(
            legality_status="invalid",
            reject_reason="AUTOTUNE_INVALID:resource_infeasible",
            raw_timing_samples=(1.0,),
        )
        assert any("latency=inf" in problem for problem in trial.validate())

    def test_selected_candidate_requires_confirmation(self) -> None:
        trial = _trial(selected=True)
        assert any("confirmation" in problem for problem in trial.validate())
        ok = _trial(selected=True, confirmation_status="confirmed")
        assert ok.validate() == []

    def test_correctness_before_benchmark(self) -> None:
        trial = _trial(correctness_status="not_run", raw_timing_samples=(1.0,))
        report = at.correctness_before_benchmark(trial)
        assert report["compliant"] is False

    def test_metrics_separate_failure_classes(self) -> None:
        rows = [
            _trial(),
            _trial(
                candidate_config={"BLOCK": 512, "warps": 4},
                legality_status="invalid",
                reject_reason="AUTOTUNE_INVALID:shape",
            ),
            _trial(
                candidate_config={"BLOCK": 1024, "warps": 8},
                compile_status="failed",
            ),
            _trial(
                candidate_config={"BLOCK": 512, "warps": 8},
                correctness_status="fail",
                raw_timing_samples=(),
            ),
        ]
        metrics = at.autotune_metrics(rows)
        assert metrics["generated"] == 4
        assert metrics["statically_legal"] == 3
        assert metrics["compiled"] == 2
        assert metrics["correct"] == 1
        assert metrics["invalid_reasons"] == {"AUTOTUNE_INVALID:shape": 1}


@pytest.mark.unit
class TestOracleAndWinners:
    def test_oracle_picks_the_best_measured_candidate(self) -> None:
        rows = [
            _trial(raw_timing_samples=(2.0, 2.1)),
            _trial(candidate_config={"BLOCK": 512, "warps": 4}, raw_timing_samples=(1.0, 1.05)),
        ]
        oracle = at.per_shape_oracle(
            shape_group="decode_m1", rows=rows, oracle_scope="frozen candidate set"
        )
        assert oracle["best_median"] == pytest.approx(1.025)
        assert oracle["candidates_considered"] == 2

    def test_oracle_reports_missing_measurements(self) -> None:
        oracle = at.per_shape_oracle(
            shape_group="decode_m1",
            rows=[_trial(raw_timing_samples=())],
            oracle_scope="frozen",
        )
        assert oracle["best"] == ""
        assert oracle["reason"]

    def test_winner_selection_reports_ties(self) -> None:
        rows = [
            _trial(raw_timing_samples=(1.0, 1.0, 1.0)),
            _trial(candidate_config={"BLOCK": 512, "warps": 4}, raw_timing_samples=(1.001, 1.0, 1.0)),
            _trial(candidate_config={"BLOCK": 1024, "warps": 8}, raw_timing_samples=(5.0, 5.0, 5.0)),
        ]
        report = at.select_winners(rows, top_k=3)
        assert report["provisional"][0]["median"] <= 1.01
        assert report["requires_confirmation"] is True
        assert report["ties"]

    def test_confirmation_plan_requires_three_runs(self) -> None:
        with pytest.raises(ConfigError):
            at.confirmation_plan(top_k=["a"], default_candidates=["d"], runs_per_cell=1)
        plan = at.confirmation_plan(top_k=["a"], default_candidates=["d"])
        assert plan["runs_per_cell"] == 3
        assert plan["process"] == "new process"


@pytest.mark.unit
class TestHoldoutAndCost:
    def test_holdout_verdict_requires_fallback_on_failure(self) -> None:
        report = at.holdout_evaluate(
            kind="interpolation",
            shared_winner_id="w",
            rows=[
                {"case": "s1", "default_latency_ms": 1.0, "winner_latency_ms": 0.9, "oracle_latency_ms": 0.85},
                {"case": "s2", "default_latency_ms": 1.0, "winner_latency_ms": 1.4, "oracle_latency_ms": 0.9},
            ],
        )
        assert report["verdict"] == "FALLBACK_REQUIRED"
        assert report["exceeded"] == ["s2"]
        assert "kept" in report["fallback"]

    def test_unknown_holdout_kind_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            at.holdout_evaluate(kind="magic", shared_winner_id="w", rows=[])

    def test_search_cost_breakdown_sums_stages(self) -> None:
        report = at.search_cost_breakdown(
            generation_s=0.5,
            static_filter_s=0.1,
            build_compile_s=10.0,
            correctness_s=1.0,
            benchmark_s=2.0,
            confirmation_s=1.5,
            cache_write_s=0.2,
            host_cpu_seconds=12.0,
            device_seconds=3.0,
            disk_bytes=1 << 20,
        )
        assert report["total_wall_s"] == pytest.approx(15.3)
        assert report["device_seconds"] == 3.0

    def test_amortization_is_never_when_there_is_no_saving(self) -> None:
        report = at.amortization(phase="decode", search_cost_s=100.0, per_call_saving_s=0.0, call_count=1000)
        assert report["break_even"] == "NEVER"

    def test_amortization_is_finite_with_a_saving(self) -> None:
        report = at.amortization(phase="prefill", search_cost_s=10.0, per_call_saving_s=0.01, call_count=10000)
        assert report["break_even_calls"] == pytest.approx(1000.0)

    def test_quality_cost_curve_reports_regret(self) -> None:
        curve = at.quality_cost_curve(
            rows=[
                {"budget": "B0_default", "best_confirmed_median_ms": 2.0, "device_seconds": 1, "trial_count": 1},
                {"budget": "B1_small", "best_confirmed_median_ms": 1.2, "device_seconds": 8, "trial_count": 6},
            ],
            oracle_latency_ms=1.0,
        )
        assert curve["curve"][1]["regret_vs_oracle"] == pytest.approx(0.2)
        assert "frozen candidate set" in curve["oracle_scope"]


@pytest.mark.unit
class TestDatabaseAndPolicy:
    def test_tuning_database_key_must_be_complete(self) -> None:
        with pytest.raises(ConfigError):
            at.tuning_database_record(
                key={"semantic_op": "x"},
                winner_config={"BLOCK": 256},
                winner_identity="w",
                confirmation_status="confirmed",
                invalidation_dependencies=["compiler_version"],
            )
        record = at.tuning_database_record(
            key={name: "v" for name in at.TUNING_DB_KEY_FIELDS},
            winner_config={"BLOCK": 256},
            winner_identity="w",
            confirmation_status="confirmed",
            invalidation_dependencies=["compiler_version"],
        )
        assert record["record_digest"]
        assert set(record["key"]) == set(at.TUNING_DB_KEY_FIELDS)

    def test_policy_document_forbids_online_tuning_of_stateful_kernels(self) -> None:
        document = at.autotune_policy_document(
            per_shape_tune_when=["tail shapes"],
            shared_config_when=["same bucket"],
            budget=at.budget_ladder(space_size=4)[1],
            holdout_failure_action="per-shape tune or default",
        )
        assert document["online_tuning_forbidden"] is True
        assert "stateful" in document["stateful_kernel_rule"]

    def test_split_summary_counts_every_split(self) -> None:
        summary = at.split_summary([_trial(), _trial(split="holdout")])
        assert summary["counts"]["tuning"] == 1
        assert summary["counts"]["holdout"] == 1
        assert summary["total"] == 2
