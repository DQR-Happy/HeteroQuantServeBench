"""Coverage for the E11-03 backend contract and E11-04 cross-layer attribution."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.compiler import backend as bk
from hqsb.compiler import codegen as cg
from hqsb.compiler import ir as ir
from hqsb.compiler import lowering as lw
from hqsb.compiler import pattern_library as pl
from hqsb.compiler import rewrite as rw
from hqsb.compiler import targets as tg


def _decision() -> lw.LoweringDecision:
    registry = lw.LoweringRegistry()
    registry.register(lw.reference_lowering_entry())
    evidence = lw.EvidenceIndex(
        [
            lw.CorrectnessEvidence(
                evidence_id="ev",
                level="operator",
                implementation_id="hqsb.reference.fused_add_rms_norm",
                dtypes=("fp16", "fp32", "bf16"),
                tolerance_policy_id="common_s06",
                status="pass",
                raw_ref="raw://operator",
            )
        ]
    )
    return lw.evaluate_candidates(
        registry=registry,
        semantic_op="hqsb::fused_add_rms_norm",
        schema_version="1.0.0",
        target=tg.cpu_target_snapshot(),
        evidence=evidence,
        inputs={},
        policy="reference",
    )


class _GraphModule:
    def forward(self) -> float:  # pragma: no cover - trivial callable
        return 1.0


@pytest.mark.unit
class TestBackendPlan:
    def _hooks(self) -> bk.BackendHooks:
        return bk.BackendHooks(
            capture_adapter=lambda state: {"graph_id": "g1", "graph_break_count": 2},
            canonicalize_and_verify=lambda state: {"canonical_ir_id": "ir_c"},
            run_semantic_passes=lambda state: {"rewrite_count": 1},
            analyze_target=lambda state: {"target_id": "cpu"},
            enumerate_lowerings=lambda state: {"candidate_ids": ["reference"]},
            build_guards=lambda state: {"guard_set_id": "gs"},
            select_candidate=lambda state: {"lowering": _decision()},
            materialize_callable=lambda state: {
                "materialized": lw.materialize(_decision(), _registry_with_reference()),
                "materialize_status": "materialized",
            },
            emit_artifact_manifest=lambda state: {"artifact_manifest_id": "m1"},
        )

    def test_full_compile_records_every_step(self) -> None:
        backend = bk.HQSBBackend(hooks=self._hooks(), capture_only=False)
        plan = backend.compile(_GraphModule(), [])
        assert [record.step for record in plan.steps] == list(bk.BACKEND_STEPS)
        assert all(record.status == "ok" for record in plan.steps)
        assert plan.is_debug_backend is False
        assert plan.artifact_manifest_id == "m1"
        assert plan.validate() == []

    def test_capture_only_plan_is_a_debug_backend(self) -> None:
        hooks = bk.BackendHooks(
            capture_adapter=lambda state: {"graph_id": "g"},
            materialize_callable=lambda state: {"materialize_status": "materialized"},
        )
        backend = bk.HQSBBackend(hooks=hooks, capture_only=True)
        plan = backend.compile(_GraphModule(), [])
        assert plan.is_debug_backend is True
        report = bk.assert_not_debug_backend(plan)
        assert report["ok"] is False
        assert "capture/debug backend" in report["reason"]

    def test_compiled_backend_refuses_to_return_forward(self) -> None:
        hooks = bk.BackendHooks(capture_adapter=lambda state: {"graph_id": "g"})
        backend = bk.HQSBBackend(hooks=hooks, capture_only=False)
        with pytest.raises(ConfigError):
            backend.compile(_GraphModule(), [])

    def test_failing_step_is_recorded_and_stops_the_chain(self) -> None:
        def boom(state: dict) -> dict:
            raise ConfigError("compile exploded")

        hooks = self._hooks()
        hooks.run_semantic_passes = boom
        backend = bk.HQSBBackend(hooks=hooks)
        plan = backend.compile(_GraphModule(), [])
        failed = [record for record in plan.steps if record.status == "failed"]
        assert failed and failed[0].step == "run_semantic_passes"
        assert "compile exploded" in plan.error
        # steps after the failure must not claim success
        assert all(record.status != "ok" for record in plan.steps[3:])

    def test_compile_run_projection_uses_cost_keys(self) -> None:
        backend = bk.HQSBBackend(hooks=self._hooks())
        plan = backend.compile(_GraphModule(), [])
        run = plan.as_compile_run()
        assert run.validate() == []
        assert set(run.timing_breakdown) <= {
            "capture_time",
            "graph_transform_time",
            "lowering_selection_time",
            "codegen_time",
            "artifact_write_time",
        }
        assert run.record_hash()

    def test_backend_contract_document_lists_the_nine_steps(self) -> None:
        document = bk.backend_contract_document()
        assert document["steps"] == list(bk.BACKEND_STEPS)
        assert any("gm.forward" in rule for rule in document["rules"])

    def test_torch_compile_invocation_is_descriptive_only(self) -> None:
        description = bk.torch_compile_invocation(model_name="qwen3")
        assert "torch.compile" in description["call"]
        assert "fullgraph" in description["notes"]


def _registry_with_reference() -> lw.LoweringRegistry:
    registry = lw.LoweringRegistry()
    registry.register(lw.reference_lowering_entry())
    return registry


@pytest.mark.unit
class TestArtifactManifest:
    def test_manifest_flags_incomplete_lineage(self) -> None:
        backend = bk.HQSBBackend(
            hooks=bk.BackendHooks(
                capture_adapter=lambda state: {"graph_id": "g"},
                materialize_callable=lambda state: {"materialize_status": "materialized"},
            ),
            capture_only=True,
        )
        plan = backend.compile(_GraphModule(), [])
        manifest = bk.emit_artifact_manifest(plan, artifacts=[{"ir_level": "HQSB_TARGETED"}])
        assert manifest["lineage_complete"] is False
        complete = bk.emit_artifact_manifest(
            plan, artifacts=[{"ir_level": "HQSB_TARGETED", "canonical_hash": "h"}]
        )
        assert complete["lineage_complete"] is True
        assert manifest["manifest_id"] != complete["manifest_id"]


@pytest.mark.unit
class TestGeneratedSource:
    def test_generated_source_requires_parent_ir_and_symbol(self) -> None:
        artifact = cg.GeneratedSourceArtifact(
            artifact_id="a", language="triton", source_text="code", entry_symbol="k", parent_ir_id="ir"
        )
        assert artifact.validate() == []
        broken = cg.GeneratedSourceArtifact(
            artifact_id="b", language="magic", source_text="", entry_symbol="", parent_ir_id=""
        )
        problems = broken.validate()
        assert any("language" in problem for problem in problems)
        assert any("entry symbol" in problem for problem in problems)
        assert any("IR" in problem for problem in problems)

    def test_hashes_separate_raw_and_canonical(self) -> None:
        artifact = cg.GeneratedSourceArtifact(
            artifact_id="a",
            language="triton",
            source_text="@triton.jit\ndef k(): pass  # 0x7ffe1234",
            entry_symbol="k",
            parent_ir_id="ir",
        )
        hashes = artifact.hashes()
        assert hashes["raw_hash"] != hashes["canonical_hash"]
        assert hashes["stripped_keys"]

    def test_source_digest_is_stable(self) -> None:
        artifact = cg.GeneratedSourceArtifact(
            artifact_id="a", language="triton", source_text="code", entry_symbol="k", parent_ir_id="ir"
        )
        assert cg.hash_all_sources([artifact])["combined"] == cg.hash_all_sources([artifact])["combined"]


@pytest.mark.unit
class TestToolchainAndResources:
    def test_export_plan_marks_missing_tools(self) -> None:
        plan = cg.toolchain_export_plan(tg.cpu_target_snapshot())
        assert {layer["layer"] for layer in plan["layers"]} == set(cg.EXPORT_LAYERS)
        for layer in plan["layers"]:
            if not layer["available"]:
                assert layer["status"] == "NOT_RUN_TOOL_UNAVAILABLE"
                assert layer["reason"]
        assert "binary hash" in plan["sass_binding_rule"]

    def test_resource_parser_keeps_missing_values_unknown(self) -> None:
        report = cg.parse_resource_usage("REG:64 SHARED:1024 STACK:0 LDL:3 STL:1")
        assert report["fields"]["registers_per_thread"] == 64
        assert report["fields"]["spill_loads"] == 3
        assert "dynamic_shared_bytes" in report["missing_fields"]
        assert report["parse_status"] in ("OK", "PARTIAL")

    def test_occupancy_limits_are_limits_not_goals(self) -> None:
        report = cg.occupancy_limits(
            registers_per_thread=64,
            static_shared_bytes=1024,
            max_registers_per_sm=65536,
            shared_per_sm_bytes=65536,
            max_threads_per_sm=2048,
            block_size=256,
        )
        assert report["estimated_blocks_per_sm"] is not None
        assert "not automatically faster" in report["warning"]

    def test_memory_ledger_diff_counts_savings(self) -> None:
        before = [
            cg.MemoryLedgerRow("a", "add", ("norm",), 100, "strided"),
            cg.MemoryLedgerRow("b", "norm", ("out",), 50, "strided"),
        ]
        after = [cg.MemoryLedgerRow("b", "fused", ("out",), 50, "strided")]
        report = cg.memory_plan_diff(before, after)
        assert report["eliminated_buffers"] == ["a"]
        assert report["bytes_saved"] == 100

    def test_launch_ledger_diff_detects_mismatch(self) -> None:
        expected = [cg.LaunchLedgerRow(0, "kernel", "k_fused", phase="prefill")]
        observed = [
            cg.LaunchLedgerRow(0, "kernel", "k_fused", observed=True),
            cg.LaunchLedgerRow(1, "kernel", "k_extra", observed=True),
        ]
        report = cg.launch_ledger_diff(expected, observed)
        assert report["balanced"] is False
        assert report["extra_symbols"] == ["k_extra"]


@pytest.mark.unit
class TestMechanisms:
    def _hypothesis(self, **overrides: object) -> cg.MechanismHypothesis:
        payload = dict(
            mechanism_id="m1",
            case_pair="original↔fused",
            changed_factor="fusion",
            ir_prediction="one fewer intermediate write",
            generated_code_observation="single kernel",
            binary_observation="register count 64→72",
            counter_prediction="DRAM store bytes down",
            counter_observation="store bytes down",
            runtime_observation="latency down",
            ablation_observation="single-factor trend matches",
            alternative_explanations=("launch overhead only",),
            artifact_refs=("codegen/a1",),
        )
        payload.update(overrides)
        return cg.MechanismHypothesis(**payload)  # type: ignore[arg-type]

    def test_supported_needs_three_layers_plus_ablation(self) -> None:
        report = cg.mechanism_verdict_table([self._hypothesis()])
        assert report["at_least_one_supported"] is True
        assert report["problems"] == []

    def test_supported_without_alternatives_is_rejected(self) -> None:
        row = self._hypothesis(alternative_explanations=()).evaluate()
        assert any("alternative" in problem for problem in row.validate())

    def test_weak_evidence_is_inconclusive(self) -> None:
        row = self._hypothesis(
            generated_code_observation="",
            binary_observation="",
            counter_observation="",
            runtime_observation="",
            ablation_observation="",
        ).evaluate()
        assert row.verdict == "INCONCLUSIVE"
        assert row.confidence == "low"

    def test_refutation_requires_a_counter_observation(self) -> None:
        row = self._hypothesis(
            refuting_observation="ablation shows the same latency with and without fusion"
        ).evaluate()
        assert row.verdict == "REFUTED"
        assert row.validate() == []
        unsupported = cg.MechanismHypothesis(
            mechanism_id="m2",
            case_pair="p",
            changed_factor="f",
            ir_prediction="x",
        )
        unsupported.verdict = "REFUTED"
        assert any("refuting observation" in problem for problem in unsupported.validate())

    def test_ablation_matrix_is_single_factor(self) -> None:
        cases = cg.ablation_matrix({"fusion": False, "tile": 128}, factors={"fusion": [True], "tile": [64]})
        report = cg.evaluate_ablation(cases)
        assert report["problems"] == []
        assert {row["changed_factor"] for row in report["trends"]} == {"fusion", "tile"}

    def test_ablation_requires_a_base_case(self) -> None:
        cases = [cg.AblationCase(case_id="x", factors={"fusion": True})]
        with pytest.raises(ConfigError):
            cg.evaluate_ablation(cases)

    def test_amdahl_residual_is_reported(self) -> None:
        report = cg.amdahl_attribution(
            kernel_saving_ms=1.0,
            call_share_in_phase=0.1,
            phase_latency_ms=100.0,
            measured_phase_saving_ms=0.1,
        )
        assert report["predicted_phase_saving_ms"] == 0.1
        assert report["residual_ms"] == 0.0
        missing = cg.amdahl_attribution(
            kernel_saving_ms=1.0,
            call_share_in_phase=0.1,
            phase_latency_ms=100.0,
            measured_phase_saving_ms=None,
        )
        assert missing["attribution_ok"] is None


@pytest.mark.unit
class TestProfilingAndReconstruction:
    def test_profile_plan_requires_symbols_and_refuses_primary_profile_timing(self) -> None:
        plan = cg.ProfilePlan(target="sm_86", symbols=())
        assert any("symbol" in problem for problem in plan.validate())
        primary = cg.ProfilePlan(target="sm_86", symbols=("k",), report_timing_is_primary=True)
        assert any("primary performance" in problem for problem in primary.validate())

    def test_mutating_kernel_profile_drops_replay_commands(self) -> None:
        safe = cg.ProfilePlan(target="sm_86", symbols=("k",))
        risky = cg.ProfilePlan(target="sm_86", symbols=("k",), mutates_state=True)
        assert len(risky.commands()) < len(safe.commands())
        assert "isolated inputs" in risky.as_dict()["replay_risk"]

    def test_reconstruction_check_distinguishes_semantic_and_bitwise(self) -> None:
        original = {"canonical_hash": "h", "raw_hash": "r1", "resource_report": {"registers": 64}}
        rebuilt = {"canonical_hash": "h", "raw_hash": "r2", "resource_report": {"registers": 64}}
        report = cg.reconstruction_check(manifest_id="m", original=original, rebuilt=rebuilt)
        assert report["semantic_rebuild_ok"] is True
        assert report["bitwise_identical"] is False
        divergent = cg.reconstruction_check(
            manifest_id="m", original=original, rebuilt={"canonical_hash": "other", "raw_hash": "r3"}
        )
        assert divergent["semantic_rebuild_ok"] is False

    def test_cross_layer_diff_rows_cover_each_layer(self) -> None:
        artifact = cg.GeneratedSourceArtifact(
            artifact_id="a", language="triton", source_text="code", entry_symbol="k", parent_ir_id="ir"
        )
        rows = cg.cross_layer_diff_rows(
            graph_diff={"ops_removed": ["o2"]},
            generated_source=[artifact],
            resource_before={"registers_per_thread": 64},
            resource_after={"registers_per_thread": 72},
            timing_before_ms=1.0,
            timing_after_ms=0.8,
        )
        assert {row["layer"] for row in rows} == {"ir", "generated_source", "binary", "runtime"}

    def test_reconstruction_plan_mentions_nondeterminism(self) -> None:
        plan = cg.reconstruction_plan(
            manifest_id="m", toolchain_versions={"nvcc": "12.4"}, source_artifacts=["a"]
        )
        assert "bytewise-identical binaries are not guaranteed" in plan["binary_nondeterminism_note"]
        assert len(plan["steps"]) >= 4


@pytest.mark.unit
class TestIRToCodegenChain:
    def test_rewrite_to_manifest_chain_links_ir_and_source(self) -> None:
        graph = rw.build_graph(pl.residual_add_rmsnorm_graph(graph_id="chain", variant="canonical"))
        outcome = rw.run_fused_add_rmsnorm_pass(graph)
        assert outcome.rewrite_count == 1
        source = cg.GeneratedSourceArtifact(
            artifact_id="a1",
            language="triton",
            source_text="@triton.jit def fused(): pass",
            entry_symbol="fused_add_rms_norm",
            parent_ir_id=outcome.graph.graph_id,
        )
        assert ir.verify_graph(outcome.graph).ok is True
        assert source.validate() == []
        chain = cg.cross_layer_diff_rows(
            graph_diff=ir.diff_graphs(graph, outcome.graph).as_dict(),
            generated_source=[source],
            resource_before={"registers_per_thread": None},
            resource_after={"registers_per_thread": None},
            timing_before_ms=None,
            timing_after_ms=None,
        )
        assert {row["layer"] for row in chain} >= {"ir", "generated_source"}
