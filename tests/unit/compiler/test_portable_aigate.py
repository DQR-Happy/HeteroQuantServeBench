"""Coverage for the E11-09 second-stack and E11-10 AI-gate interfaces."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.compiler import aigate as ag
from hqsb.compiler import portable as pt


@pytest.mark.unit
class TestStackSelection:
    def test_valid_selection(self) -> None:
        selection = pt.StackSelection(
            primary_stack="tvm",
            versions={"tvm": "0.19.0", "relax": "0.19.0", "tir": "0.19.0", "llvm": "17.0.6"},
            scope=pt.SCOPE_CEILING,
            secondary_stack="mlir",
        )
        assert selection.validate() == []
        assert selection.as_dict()["secondary_status"] == "NOT_RUN_SCOPE_LIMITED"

    def test_unknown_stack_and_vague_versions_are_rejected(self) -> None:
        selection = pt.StackSelection(
            primary_stack="magic", versions={"tvm": "latest"}, scope=""
        )
        problems = selection.validate()
        assert any("stack" in problem for problem in problems)
        assert any("frozen" in problem for problem in problems)
        assert any("scope" in problem for problem in problems)


@pytest.mark.unit
class TestSemanticMapping:
    def test_mapping_table_covers_every_row(self) -> None:
        table = pt.semantic_mapping_table("tvm")
        assert len(table["rows"]) == len(pt.MAPPING_ROWS)
        assert all(row["must_preserve"] for row in table["rows"])
        assert table["ok"] is True

    def test_unmapped_semantics_block_by_default(self) -> None:
        blocked = pt.semantic_mapping_table("tvm", unmapped=["kv_page_table"])
        assert blocked["ok"] is False
        assert blocked["unmapped"][0]["action"] == "block"
        external = pt.semantic_mapping_table(
            "mlir", unmapped=["kv_page_table"], unmapped_policy="external_call"
        )
        assert external["ok"] is True
        assert external["unmapped"][0]["action"] == "explicit external call"

    def test_unknown_stack_and_policy_are_rejected(self) -> None:
        with pytest.raises(ConfigError):
            pt.semantic_mapping_table("magic")
        with pytest.raises(ConfigError):
            pt.semantic_mapping_table("tvm", unmapped_policy="magic")


@pytest.mark.unit
class TestLegalization:
    def test_analysis_only_reports_illegal_ops_before_codegen(self) -> None:
        target = pt.LegalizationTarget(
            legal_ops=("relax.add", "tir.call"), dynamic_legal_ops=("relax.nn.rms_norm",)
        )
        report = pt.analysis_only_legality(["relax.add", "relax.nn.rms_norm", "relax.mystery"], target)
        assert report["illegal"] == ["relax.mystery"]
        assert report["analysis_ok"] is False
        assert any(row["status"] == "dynamic_legal" for row in report["rows"])

    def test_unknown_ops_are_never_legal_by_default(self) -> None:
        target = pt.LegalizationTarget(legal_ops=("relax.add",))
        assert target.classify("relax.unknown") == "illegal"

    def test_overlapping_legal_and_illegal_sets_are_rejected(self) -> None:
        target = pt.LegalizationTarget(legal_ops=("a",), illegal_ops=("a",))
        assert any("legal and illegal" in problem for problem in target.validate())

    def test_unknown_mode_is_rejected(self) -> None:
        assert any("mode" in problem for problem in pt.LegalizationTarget(legal_ops=("a",), mode="magic").validate())


@pytest.mark.unit
class TestLoopIRAndSchedule:
    def test_loop_ir_validation_and_verifier(self) -> None:
        spec = pt.LoopIRSpec(
            block_id="b",
            buffers=("x", "w", "y"),
            loops=(("i", 64, 1), ("j", 128, 1)),
            reads={"x": ("i", "j"), "w": ("j",)},
            writes={"y": ("i", "j")},
            reduction={"axis": "j", "init": "zero", "update": "sum"},
            tail_policy="masked",
        )
        assert spec.validate() == []
        assert pt.verify_loop_ir(spec)["ok"] is True
        broken = pt.LoopIRSpec(
            block_id="b", buffers=("x",), loops=(("i", 0, 1),), reads={}, writes={}, tail_policy=""
        )
        problems = pt.verify_loop_ir(broken)["problems"]
        assert any("extent" in problem for problem in problems)
        assert any("tail" in problem for problem in problems)

    def test_loop_ir_rejects_unknown_buffers(self) -> None:
        spec = pt.LoopIRSpec(
            block_id="b", buffers=("x",), loops=(("i", 4, 1),), reads={"ghost": ("i",)}, writes={},
            tail_policy="none",
        )
        assert any("ghost" in problem for problem in spec.validate())

    def test_schedule_steps_need_hardware_reasons(self) -> None:
        step = pt.ScheduleStep(kind="tile", params={"tile": 128}, hardware_reason="")
        assert any("hardware reason" in problem for problem in step.validate())
        noop = pt.ScheduleStep(
            kind="tile",
            params={"tile": 128},
            hardware_reason="fit in shared memory",
            before_hash="h",
            after_hash="h",
        )
        assert any("did not change" in problem for problem in noop.validate())

    def test_schedule_trace_replay_structure(self) -> None:
        trace = pt.ScheduleTrace(
            trace_id="t",
            steps=(
                pt.ScheduleStep(
                    kind="tile",
                    params={"tile": 128},
                    hardware_reason="shared memory budget",
                    before_hash="a",
                    after_hash="b",
                ),
            ),
        )
        assert trace.validate() == []
        report = pt.replay_schedule(trace)
        assert report["status"] == "STRUCTURE_ONLY"
        assert trace.as_dict()["digest"]

    def test_empty_schedule_trace_is_rejected(self) -> None:
        assert any("steps" in problem for problem in pt.ScheduleTrace(trace_id="t", steps=()).validate())

    def test_pass_pipeline_trace_checks_hash_chain(self) -> None:
        entries = (
            pt.PassTraceEntry(name="legalize", options={}, before_hash="a", after_hash="b"),
            pt.PassTraceEntry(name="lower", options={"target": "cuda"}, before_hash="b", after_hash="c"),
        )
        report = pt.pass_pipeline_trace(entries)
        assert report["ok"] is True
        broken = (
            pt.PassTraceEntry(name="legalize", options={}, before_hash="a", after_hash="b"),
            pt.PassTraceEntry(name="lower", options={}, before_hash="other", after_hash="c"),
        )
        assert pt.pass_pipeline_trace(broken)["ok"] is False

    def test_failed_pass_is_reported(self) -> None:
        entries = (pt.PassTraceEntry(name="lower", options={}, before_hash="a", after_hash="a", status="failed"),)
        report = pt.pass_pipeline_trace(entries)
        assert report["failed_passes"] == ["lower"]


@pytest.mark.unit
class TestBridge:
    def test_bridge_contract_validation(self) -> None:
        contract = pt.BridgeContract(
            kind="dlpack",
            ownership="caller owns the tensor",
            stride_policy="require contiguous or copy",
            device_policy="same device only",
            stream_policy="current torch stream",
            sync_policy="no implicit sync",
            error_policy="raise structured error",
        )
        assert contract.validate() == []
        assert pt.bridge_audit_plan(contract)["ok"] is True
        broken = pt.BridgeContract(
            kind="magic",
            ownership="",
            stride_policy="copy_to_contiguous",
            device_policy="d",
            stream_policy="s",
            sync_policy="y",
            error_policy="e",
            zero_copy_claimed=True,
        )
        problems = broken.validate()
        assert any("kind" in problem for problem in problems)
        assert any("zero-copy" in problem for problem in problems)

    def test_bridge_overhead_is_unavailable_without_a_timeline(self) -> None:
        report = pt.bridge_overhead(end_to_end_ms=None, device_kernel_ms=1.0)
        assert report["status"] == "UNAVAILABLE"
        measured = pt.bridge_overhead(end_to_end_ms=1.5, device_kernel_ms=1.0)
        assert measured["bridge_overhead_ms"] == pytest.approx(0.5)


@pytest.mark.unit
class TestCostAndAdoption:
    def test_development_cost_rubric_requires_every_item(self) -> None:
        report = pt.development_cost_rubric(
            implementation_loc=500,
            integration_effort="medium",
            debugging_hours=12.0,
            dependency_weight="tvm 40MB",
            artifact_footprint_bytes=1 << 20,
            replay_success=True,
            maintenance_risk="medium",
        )
        assert report["complete"] is True
        assert set(report["rubric"]) == set(pt.COST_RUBRIC_ITEMS)

    def test_role_comparison_requires_all_dimensions(self) -> None:
        report = pt.role_comparison(
            hqsb={name: "fx/inductor" for name in pt.ROLE_DIMENSIONS},
            second_stack={name: "tvm" for name in pt.ROLE_DIMENSIONS},
        )
        assert report["complete"] is True
        partial = pt.role_comparison(hqsb={"capture": "dynamo"}, second_stack={})
        assert partial["complete"] is False
        assert partial["missing"]

    def test_adoption_decision_blocks_unsupported_claims(self) -> None:
        blocked = pt.adoption_decision(
            option="second_stack_as_target_backend",
            correctness_ok=True,
            performance_evidence=False,
            dev_cost_rubric={},
        )
        assert blocked["decided"] is False
        assert any("performance" in item for item in blocked["blockers"])
        allowed = pt.adoption_decision(
            option="concept_validation_only",
            correctness_ok=True,
            performance_evidence=False,
            dev_cost_rubric={},
        )
        assert allowed["decided"] is True
        assert len(allowed["allowed_public_conclusions"]) == 4

    def test_unknown_adoption_option_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            pt.adoption_decision(
                option="magic", correctness_ok=True, performance_evidence=False, dev_cost_rubric={}
            )

    def test_negative_capability_case_and_concept_rows(self) -> None:
        case = pt.negative_capability_case(kind="unsupported_dtype", detail="fp8 not legalised")
        assert case["action"] == "explicit failure or fallback"
        assert len(pt.concept_summary_rows()) >= 5


def _task_package() -> ag.TaskPackage:
    return ag.TaskPackage(
        task_id="t1",
        operator_spec_id="hqsb::fused_add_rms_norm",
        allowed_apis=("triton", "torch"),
        public_examples=({"shape": [1, 4096]},),
        forbidden_actions=("network", "filesystem", "modify_reference"),
        resource_limits={"cpu_seconds": 60, "gpu_seconds": 30, "memory_mb": 4096},
        output_contract={"dtype": "fp16", "tolerance_policy": "common_s06"},
        reference_hash="ref-hash",
        harness_hash="harness-hash",
    )


def _provenance(**overrides: object) -> ag.ProvenanceRecord:
    payload = dict(
        candidate_id="c1",
        prompt_hash="p",
        model="agent-x",
        model_version="2026-09",
        patch="+def kernel(): ...",
        license_declaration="own work",
        source_sha256="s" * 64,
    )
    payload.update(overrides)
    return ag.ProvenanceRecord(**payload)  # type: ignore[arg-type]


def _sandbox(**overrides: object) -> ag.SandboxPolicy:
    payload = dict(
        network=False,
        secrets_visible=False,
        reference_read_only=True,
        registry_writable=False,
        official_cache_writable=False,
        scratch_dir="/tmp/scratch",
        cpu_seconds_limit=60.0,
        gpu_seconds_limit=30.0,
        memory_mb_limit=4096,
        process_limit=8,
        file_size_mb_limit=256,
    )
    payload.update(overrides)
    return ag.SandboxPolicy(**payload)  # type: ignore[arg-type]


@pytest.mark.unit
class TestTaskAndProvenance:
    def test_task_package_validation(self) -> None:
        assert _task_package().validate() == []
        broken = ag.TaskPackage(
            task_id="t",
            operator_spec_id="op",
            allowed_apis=(),
            public_examples=(),
            forbidden_actions=(),
            resource_limits={},
            output_contract={},
        )
        problems = broken.validate()
        assert any("allowed APIs" in problem for problem in problems)
        assert any("forbidden" in problem for problem in problems)

    def test_task_digest_is_stable(self) -> None:
        assert _task_package().digest() == _task_package().digest()

    def test_generation_config_validation(self) -> None:
        config = ag.GenerationConfig(
            provider="local",
            model="agent-x",
            model_version="2026-09",
            prompt_template_hash="p",
            temperature=0.2,
            seed=1,
            max_iterations=3,
            max_tokens=4096,
            feedback_policy="correctness feedback only",
        )
        assert config.validate() == []
        assert ag.GenerationConfig(
            provider="", model="", model_version="", prompt_template_hash="",
            temperature=3.0, seed=0, max_iterations=0, max_tokens=0, feedback_policy="",
        ).validate()

    def test_provenance_missing_fields_are_reported(self) -> None:
        assert _provenance().missing_fields() == []
        assert _provenance(license_declaration="").missing_fields() == ["license_declaration"]


@pytest.mark.unit
class TestSandboxAndHarness:
    def test_sandbox_defaults_pass(self) -> None:
        assert _sandbox().validate() == []

    def test_unsafe_sandbox_is_rejected(self) -> None:
        problems = _sandbox(
            network=True, secrets_visible=True, reference_read_only=False,
            registry_writable=True, official_cache_writable=True,
        ).validate()
        assert len(problems) >= 4

    def test_harness_lock_rejects_leaked_api_surface(self) -> None:
        lock = ag.HarnessLock(
            runner_hash="r",
            reference_hash="ref",
            tests_hash="t",
            timer_hash="timer",
            build_wrapper_hash="b",
            candidate_api_surface=("tensors", "timer"),
        )
        assert any("leaks" in problem for problem in lock.validate())
        assert ag.HarnessLock("r", "ref", "t", "timer", "b").validate() == []

    def test_hidden_distributions_validate(self) -> None:
        shapes = ag.HiddenShapeDistribution(
            interpolation=({"B": 1, "H": 4096},),
            boundary=({"B": 1, "H": 4096},),
            extrapolation=({"B": 4, "H": 4096},),
        )
        assert shapes.validate() == []
        assert ag.HiddenShapeDistribution().validate()
        values = ag.HiddenValueDistribution(seeds=(1, 2, 3))
        assert values.validate() == []
        assert ag.HiddenValueDistribution(seeds=(1,)).validate()

    def test_metamorphic_tests_have_relations(self) -> None:
        tests = ag.metamorphic_tests()
        assert tests and all(row["relation"] for row in tests)


@pytest.mark.unit
class TestGateChain:
    def test_every_gate_accepts_a_clean_candidate(self) -> None:
        results = [
            ag.gate_g0_provenance(_provenance(), _task_package()),
            ag.gate_g1_static_policy(
                source_text="@triton.jit\ndef k(): pass",
                imports=("triton", "torch"),
                license_declaration="own work",
            ),
            ag.gate_g2_isolated_compile(
                compile_status="ok", artifact_hash="h", sandbox=_sandbox()
            ),
            ag.gate_g3_sanitizer_memory(sanitizer_status="clean"),
            ag.gate_g4_public_correctness(max_abs_error=0.01, tolerance=0.02, cases=5),
            ag.gate_g5_hidden_correctness(hidden_cases=12),
            ag.gate_g6_state_concurrency(repeated_runs=3),
            ag.gate_g7_measurement_integrity(
                timer_hash_unchanged=True,
                inputs_unchanged=True,
                reference_unchanged=True,
                device_sync_used=True,
                outputs_validated_after_timing=True,
            ),
            ag.gate_g8_performance_resource(
                correct_and_faster=True, speedup=1.5, min_speedup=1.2
            ),
            ag.gate_g9_compiler_integration(
                registry_schema_ok=True,
                guards_ok=True,
                fallback_ok=True,
                cache_identity_ok=True,
                actual_dispatch_ok=True,
            ),
            ag.gate_g10_review(
                reviewer_id="alice", checklist={"math": True, "bounds": True}, approved_commit="abc"
            ),
        ]
        chain = ag.run_gate_chain("c1", results)
        assert chain["first_reject"] == ""
        assert chain["chain_ok"] is True
        assert chain["executed"] == list(ag.GATE_ORDER)
        assert ag.candidate_verdict(results, "c1", faster=False)["status"] == "QUALIFIED_NOT_FASTER"
        assert ag.candidate_verdict(results, "c1", faster=True)["status"] == "ADMITTED"

    def test_later_accept_cannot_overrule_an_earlier_reject(self) -> None:
        results = [
            ag.gate_g2_isolated_compile(compile_status="failed", artifact_hash=""),
            ag.gate_g8_performance_resource(correct_and_faster=True, speedup=9.0, min_speedup=1.0),
        ]
        chain = ag.run_gate_chain("c1", results)
        assert chain["first_reject"] == "G2_isolated_compile"
        assert chain["later_accepts_after_reject"] == ["G8_performance_resource"]
        verdict = ag.candidate_verdict(results, "c1", faster=True)
        assert verdict["status"] == "REJECTED_G2_isolated_compile"

    def test_wrong_candidate_classes_are_rejected_at_the_expected_gate(self) -> None:
        cases = {
            "C1_syntax_compile_wrong": ag.gate_g2_isolated_compile(
                compile_status="failed", artifact_hash=""
            ),
            "C2_math_wrong": ag.gate_g5_hidden_correctness(
                hidden_cases=5, failures=[{"case": "eps_diff"}]
            ),
            "C4_value_hardcode_skip": ag.gate_g5_hidden_correctness(
                hidden_cases=5, hardcode_suspected=True
            ),
            "C5_memory_unsafe": ag.gate_g3_sanitizer_memory(
                sanitizer_status="failures", findings=("oob read",)
            ),
            "C6_race_nondeterministic": ag.gate_g6_state_concurrency(
                repeated_runs=3, nondeterministic=True
            ),
            "C7_timing_exploit": ag.gate_g7_measurement_integrity(
                timer_hash_unchanged=False,
                inputs_unchanged=True,
                reference_unchanged=True,
                device_sync_used=False,
                outputs_validated_after_timing=False,
            ),
            "C8_state_corrupt": ag.gate_g6_state_concurrency(
                repeated_runs=3, state_mutated=("residual",)
            ),
            "C9_slow_resource_bomb": ag.gate_g8_performance_resource(
                correct_and_faster=True, speedup=1.5, min_speedup=1.2,
                resource_regression="workspace 8GB", compile_time_s=900.0, compile_time_limit_s=600.0,
            ),
            "C10_provenance_violation": ag.gate_g1_static_policy(
                source_text="import socket", imports=("socket",), license_declaration=""
            ),
        }
        for candidate_class, result in cases.items():
            assert result.outcome == "reject", candidate_class
            assert result.reason_code, candidate_class
            assert ag.EXPECTED_GATE_BY_CLASS[candidate_class] == result.gate, candidate_class

    def test_control_candidate_is_not_killed(self) -> None:
        control = ag.gate_g2_isolated_compile(compile_status="ok", artifact_hash="h", sandbox=_sandbox())
        assert control.outcome == "accept"

    def test_sanitizer_unavailable_lowers_the_claim_but_accepts_with_canary(self) -> None:
        report = ag.gate_g3_sanitizer_memory(
            sanitizer_status="unavailable", tool_available=False, canary_ok=True
        )
        assert report.outcome == "accept"
        assert report.reason_code == "SANITIZER_UNAVAILABLE_CANARY_ONLY"
        blocked = ag.gate_g3_sanitizer_memory(sanitizer_status="unavailable", tool_available=False)
        assert blocked.outcome == "reject"

    def test_gate_result_validation(self) -> None:
        assert any("unknown gate" in problem for problem in ag.GateResult("G99", "accept").validate())
        assert any("reason" in problem for problem in ag.GateResult("G2_isolated_compile", "reject").validate())

    def test_wrong_corpus_template_is_complete_and_reviewed(self) -> None:
        template = ag.wrong_corpus_template()
        assert template["complete"] is True
        assert not template["missing_classes"]

    def test_wrong_corpus_entry_requires_human_review_and_isolation(self) -> None:
        entry = ag.WrongCorpusEntry(
            candidate_id="w1",
            candidate_class="C2_math_wrong",
            expected_gate="G5_hidden_correctness",
            expected_reason="eps error",
            isolated_from_hidden_tests=False,
        )
        problems = entry.validate()
        assert any("human-reviewed" in problem for problem in problems)
        assert any("isolated" in problem for problem in problems)
        wrong_gate = ag.WrongCorpusEntry(
            candidate_id="w2",
            candidate_class="C2_math_wrong",
            expected_gate="G8_performance_resource",
            expected_reason="eps error",
            human_reviewed=True,
        )
        assert any("expected gate" in problem for problem in wrong_gate.validate())

    def test_adversarial_matrix_flags_false_accepts(self) -> None:
        expected = ag.wrong_corpus_template()["entries"]
        observed = [
            {
                "candidate_id": "wrong_math_wrong",
                "status": "REJECTED_G5_hidden_correctness",
                "earliest_failure": "G5_hidden_correctness",
            },
            {"candidate_id": "wrong_memory_unsafe", "status": "ADMITTED", "earliest_failure": ""},
        ]
        report = ag.adversarial_detection_matrix(observed, expected)
        assert "wrong_memory_unsafe" in report["false_accept"]
        assert report["ok"] is False


@pytest.mark.unit
class TestAdmissionAndMetrics:
    def test_fast_p_excludes_correctness_failures_from_speedup(self) -> None:
        rows = [
            {"candidate_id": "a", "correct": True, "speedup": 2.0},
            {"candidate_id": "b", "correct": True, "speedup": 0.9},
            {"candidate_id": "c", "correct": False, "speedup": 9.0},
        ]
        report = ag.fast_p(rows, p=1.2)
        assert report["correct"] == 2
        assert report["fast_p"] == pytest.approx(1 / 3, abs=1e-4)

    def test_fast_p_without_correctness_is_unavailable(self) -> None:
        report = ag.fast_p([{"speedup": 1.0}], p=1.2)
        assert report["fast_p"] is None

    def test_resource_pareto_keeps_trade_offs(self) -> None:
        rows = [
            {"candidate_id": "fast_heavy", "speedup": 2.0, "peak_memory_mb": 8000, "compile_time_s": 10, "code_size_kb": 90},
            {"candidate_id": "balanced", "speedup": 1.5, "peak_memory_mb": 4000, "compile_time_s": 5, "code_size_kb": 80},
            {"candidate_id": "dominated", "speedup": 1.0, "peak_memory_mb": 9000, "compile_time_s": 20, "code_size_kb": 120},
        ]
        front = {row["candidate_id"] for row in ag.resource_pareto(rows)["pareto_front"]}
        assert "dominated" not in front
        assert {"fast_heavy", "balanced"} <= front

    def test_admission_schema_fails_on_missing_fields(self) -> None:
        report = ag.validate_admission({"candidate_id": "c1"})
        assert report["status"] == "SCHEMA_FAIL"
        assert "license" in report["missing_fields"]
        complete = ag.validate_admission({name: "x" for name in ag.ADMISSION_REQUIRED_FIELDS})
        assert complete["status"] == "QUALIFIED"
        assert complete["admitted"] is False

    def test_iteration_lineage_requires_equal_budget(self) -> None:
        report = ag.iteration_lineage(
            one_shot=[{"generation_seconds": 10, "accepted": True}],
            feedback=[{"generation_seconds": 10, "accepted": True, "regression": 1}],
            equal_budget_ok=False,
        )
        assert report["comparable"] is False
        assert report["feedback"]["regressions"] == 1

    def test_human_review_checklist_is_complete(self) -> None:
        checklist = ag.human_review_checklist()
        assert {row["item"] for row in checklist} >= {"math", "bounds", "license"}

    def test_claim_boundary_matches_the_available_evidence(self) -> None:
        empty = ag.claim_boundary(admitted_candidates=0, methodology_pass=True)
        assert empty["allowed"] is True
        assert "no candidate reached admission" in empty["text"]
        admitted = ag.claim_boundary(admitted_candidates=1, methodology_pass=True)
        assert "bound to the stated semantic op" in admitted["text"]
        incomplete = ag.claim_boundary(admitted_candidates=0, methodology_pass=False)
        assert incomplete["allowed"] is False
