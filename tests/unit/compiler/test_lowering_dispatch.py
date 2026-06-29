"""Coverage for the E11-03 lowering registry, selection, dispatch and fallback."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.compiler import lowering as lw
from hqsb.compiler import targets as tg


def _cuda_snapshot(**overrides: object) -> tg.TargetSnapshot:
    payload = dict(
        target_id="sm86",
        device_kind="cuda",
        device_name="RTX 3090",
        device_uuid="GPU-1",
        arch="sm_86",
        driver_version="550.54.15",
        runtime_version="12.4",
        compiler_version="12.4.131",
        backend_versions={"torch": "2.8.0", "triton": "3.4.0"},
        dtypes=("fp16", "fp32", "bf16"),
        features=("tensor_core",),
        shared_memory_per_block_bytes=48 * 1024,
        registers_per_thread=255,
        abi_version="1",
        target_triple="x86_64-linux-gnu",
    )
    payload.update(overrides)
    return tg.TargetSnapshot(**payload)  # type: ignore[arg-type]


def _evidence(implementation_id: str = "hqsb.cuda.fused_add_rms_norm", **overrides: object) -> lw.CorrectnessEvidence:
    payload = dict(
        evidence_id="ev",
        level="operator",
        implementation_id=implementation_id,
        dtypes=("fp16",),
        shapes=("*",),
        archs=("sm_86",),
        tolerance_policy_id="common_s06",
        status="pass",
        raw_ref="raw://operator",
    )
    payload.update(overrides)
    return lw.CorrectnessEvidence(**payload)  # type: ignore[arg-type]


def _registry(*, with_kernel: bool = True, archs: tuple = ("sm_86",)) -> lw.LoweringRegistry:
    registry = lw.LoweringRegistry()
    registry.register(lw.reference_lowering_entry())
    if with_kernel:
        registry.register(
            lw.custom_kernel_entry(
                candidate_id="cuda_v2",
                implementation_id="hqsb.cuda.fused_add_rms_norm",
                build_id="build-1",
                artifact_locator="build/cuda-sm86-release/libhqsb.so",
                artifact_hash="a" * 64,
                archs=archs,
                priority=10,
                fallback_id="reference",
            )
        )
    return registry


@pytest.mark.unit
class TestTargetSnapshot:
    def test_valid_snapshot_passes(self) -> None:
        assert _cuda_snapshot().validate() == []

    def test_vague_versions_are_rejected(self) -> None:
        problems = _cuda_snapshot(compiler_version="latest").validate()
        assert any("compiler_version" in problem for problem in problems)

    def test_unavailable_entries_need_reasons(self) -> None:
        problems = _cuda_snapshot(unavailable={"arch": ""}).validate()
        assert any("reason" in problem for problem in problems)

    def test_fingerprint_is_stable_and_short(self) -> None:
        snapshot = _cuda_snapshot()
        assert snapshot.fingerprint == snapshot.sha256()[:12]
        assert len(snapshot.fingerprint) == 12

    def test_field_diff_lists_changes(self) -> None:
        rows = tg.target_field_diff(_cuda_snapshot(), _cuda_snapshot(arch="sm_87"))
        assert rows == [{"field": "arch", "before": "sm_86", "after": "sm_87"}]

    def test_cpu_and_ascend_probes_never_raise(self) -> None:
        assert tg.cpu_target_snapshot().validate() == []
        ascend = tg.ascend_target_snapshot()
        assert ascend.unavailable


@pytest.mark.unit
class TestCapability:
    def test_requirement_checks_arch_dtype_feature_abi(self) -> None:
        requirement = tg.CapabilityRequirement(
            requirement_id="r",
            backends=("cuda",),
            dtypes=("fp16",),
            archs=("sm_86",),
            features=("tensor_core",),
            abi_version="1",
            evidence_scope="operator",
        )
        assert requirement.check(_cuda_snapshot()).available is True
        assert requirement.check(_cuda_snapshot(arch="sm_70")).reason_code == "TARGET_UNSUPPORTED"
        assert requirement.check(_cuda_snapshot(dtypes=("fp32",))).reason_code == "DTYPE_UNSUPPORTED"
        assert requirement.check(_cuda_snapshot(features=())).reason_code == "FEATURE_MISSING"
        assert requirement.check(_cuda_snapshot(abi_version="2")).reason_code == "ABI_MISMATCH"

    def test_backend_mismatch_is_reported(self) -> None:
        requirement = tg.CapabilityRequirement(
            requirement_id="r", backends=("cuda",), evidence_scope="operator"
        )
        outcome = requirement.check(tg.cpu_target_snapshot())
        assert outcome.available is False
        assert outcome.reason_code == "TARGET_UNSUPPORTED"

    def test_resource_limits_are_enforced(self) -> None:
        requirement = tg.CapabilityRequirement(
            requirement_id="r",
            shared_memory_bytes=64 * 1024,
            evidence_scope="operator",
        )
        assert requirement.check(_cuda_snapshot()).reason_code == "RESOURCE_INSUFFICIENT"

    def test_requirement_needs_an_evidence_scope(self) -> None:
        requirement = tg.CapabilityRequirement(requirement_id="r")
        assert any("evidence scope" in problem for problem in requirement.validate())

    def test_capability_matrix_aggregates(self) -> None:
        matrix = tg.capability_matrix(
            [
                tg.CapabilityRequirement(requirement_id="ok", evidence_scope="operator"),
                tg.CapabilityRequirement(requirement_id="bad", dtypes=("fp8_e4m3",), evidence_scope="operator"),
            ],
            _cuda_snapshot(),
        )
        assert matrix["available"] == ["ok"]
        assert matrix["rejected"][0]["reason"] == "DTYPE_UNSUPPORTED"

    def test_toolchain_export_capability_reports_missing_tools(self) -> None:
        report = tg.toolchain_export_capability(_cuda_snapshot())
        assert report["status"] in ("OK", "NOT_RUN_TOOL_UNAVAILABLE")
        assert isinstance(report["missing"], list)


@pytest.mark.unit
class TestRegistry:
    def test_kernel_entry_requires_artifact_identity(self) -> None:
        entry = lw.LoweringEntry(
            candidate_id="c",
            semantic_op="hqsb::fused_add_rms_norm",
            schema_version="1.0.0",
            execution_kind="custom_kernel",
            implementation_id="impl",
            requirement=tg.CapabilityRequirement(requirement_id="r", evidence_scope="operator"),
        )
        problems = entry.validate()
        assert any("artifact locator" in problem for problem in problems)
        assert any("build id" in problem for problem in problems)

    def test_registry_rejects_unknown_fallback_and_duplicates(self) -> None:
        registry = lw.LoweringRegistry()
        with pytest.raises(ConfigError):
            registry.register(
                lw.custom_kernel_entry(
                    candidate_id="c",
                    implementation_id="impl",
                    build_id="b",
                    artifact_locator="l",
                    artifact_hash="h" * 64,
                    archs=("sm_86",),
                    fallback_id="missing",
                )
            )
        registry = _registry(with_kernel=False)
        with pytest.raises(ConfigError):
            registry.register(lw.reference_lowering_entry())
        with pytest.raises(ConfigError):
            registry.get("nope")

    def test_snapshot_digest_is_stable(self) -> None:
        assert _registry().snapshot()["digest"] == _registry().snapshot()["digest"]

    def test_entries_are_ordered_by_priority(self) -> None:
        entries = _registry().entries_for("hqsb::fused_add_rms_norm", "1.0.0")
        assert [entry.candidate_id for entry in entries] == ["cuda_v2", "reference"]


@pytest.mark.unit
class TestSelection:
    def _evidence_index(self) -> lw.EvidenceIndex:
        return lw.EvidenceIndex(
            [
                _evidence(),
                _evidence("hqsb.reference.fused_add_rms_norm", dtypes=("fp16", "fp32", "bf16")),
            ]
        )

    def test_reference_policy_selects_reference(self) -> None:
        decision = lw.evaluate_candidates(
            registry=_registry(),
            semantic_op="hqsb::fused_add_rms_norm",
            schema_version="1.0.0",
            target=_cuda_snapshot(),
            evidence=self._evidence_index(),
            inputs={},
            policy="reference",
        )
        assert decision.selected == "reference"
        assert decision.validate() == []

    def test_auto_heuristic_prefers_the_kernel(self) -> None:
        decision = lw.evaluate_candidates(
            registry=_registry(),
            semantic_op="hqsb::fused_add_rms_norm",
            schema_version="1.0.0",
            target=_cuda_snapshot(),
            evidence=self._evidence_index(),
            inputs={},
            policy="auto_heuristic",
            predicted_costs={"cuda_v2": 1.0, "reference": 5.0},
        )
        assert decision.selected == "cuda_v2"
        assert decision.fallback_id == "reference"

    def test_forced_unsupported_never_silently_switches(self) -> None:
        decision = lw.evaluate_candidates(
            registry=_registry(archs=("sm_90",)),
            semantic_op="hqsb::fused_add_rms_norm",
            schema_version="1.0.0",
            target=_cuda_snapshot(),
            evidence=self._evidence_index(),
            inputs={},
            policy="forced:cuda_v2",
        )
        assert decision.forced_unsupported is True
        assert decision.selected == ""
        assert decision.reject_reason.startswith("FORCED_UNSUPPORTED")
        plan = lw.materialize(decision, _registry(archs=("sm_90",)))
        assert plan.status == "fallback_only"

    def test_missing_evidence_removes_the_candidate(self) -> None:
        registry = _registry()
        decision = lw.evaluate_candidates(
            registry=registry,
            semantic_op="hqsb::fused_add_rms_norm",
            schema_version="1.0.0",
            target=_cuda_snapshot(),
            evidence=lw.EvidenceIndex([_evidence("hqsb.reference.fused_add_rms_norm", dtypes=("fp16",))]),
            inputs={},
            policy="auto_heuristic",
        )
        row = next(item for item in decision.candidates if item.candidate_id == "cuda_v2")
        assert row.evidence_covered is False
        assert decision.selected == "reference"

    def test_guard_false_row_records_the_reason(self) -> None:
        decision = lw.evaluate_candidates(
            registry=_registry(),
            semantic_op="hqsb::fused_add_rms_norm",
            schema_version="1.0.0",
            target=_cuda_snapshot(),
            evidence=self._evidence_index(),
            inputs={"B": 32},
            policy="auto_heuristic",
            guard_evaluator=lambda entry, values: entry.execution_kind == "reference"
            or values.get("B", 0) <= 8,
        )
        row = next(item for item in decision.candidates if item.candidate_id == "cuda_v2")
        assert row.guard_outcome is False
        assert row.first_failure == "GUARD_FALSE"

    def test_unknown_policy_and_schema_are_rejected(self) -> None:
        with pytest.raises(ConfigError):
            lw.parse_policy("vibes")
        registry = _registry()
        with pytest.raises(ConfigError):
            lw.evaluate_candidates(
                registry=registry,
                semantic_op="hqsb::fused_add_rms_norm",
                schema_version="9.9.9",
                target=_cuda_snapshot(),
                evidence=self._evidence_index(),
                inputs={},
                policy="reference",
            )

    def test_candidate_table_always_contains_every_entry(self) -> None:
        decision = lw.evaluate_candidates(
            registry=_registry(),
            semantic_op="hqsb::fused_add_rms_norm",
            schema_version="1.0.0",
            target=_cuda_snapshot(),
            evidence=self._evidence_index(),
            inputs={},
            policy="reference",
        )
        assert {row.candidate_id for row in decision.candidates} == {"cuda_v2", "reference"}
        assert sum(1 for row in decision.candidates if row.selected) == 1


@pytest.mark.unit
class TestMaterialisationAndPreflight:
    def test_materialize_requires_a_reachable_fallback(self) -> None:
        registry = _registry()
        decision = lw.evaluate_candidates(
            registry=registry,
            semantic_op="hqsb::fused_add_rms_norm",
            schema_version="1.0.0",
            target=_cuda_snapshot(),
            evidence=lw.EvidenceIndex(
                [_evidence(), _evidence("hqsb.reference.fused_add_rms_norm", dtypes=("fp16",))]
            ),
            inputs={},
            policy="auto_heuristic",
        )
        plan = lw.materialize(decision, registry)
        assert plan.status == "materialized"
        assert plan.fallback_implementation == "hqsb.reference.fused_add_rms_norm"
        assert plan.validate() == []

    def test_per_call_selection_is_rejected(self) -> None:
        plan = lw.MaterializedPlan(
            compile_id="c",
            selected="s",
            fallback="reference",
            selected_implementation="impl",
            fallback_implementation="ref",
            python_per_call_selection=True,
        )
        assert any("per call" in problem for problem in plan.validate())

    def test_rebuild_plan_detects_registry_drift(self) -> None:
        registry = _registry()
        snapshot = registry.snapshot()["digest"]
        first = lw.rebuild_plan(
            registry,
            semantic_op="hqsb::fused_add_rms_norm",
            selected="cuda_v2",
            expected_registry_digest=snapshot,
            expected_plan_digest="",
        )
        assert first["registry_matches"] is True
        second = lw.rebuild_plan(
            registry,
            semantic_op="hqsb::fused_add_rms_norm",
            selected="cuda_v2",
            expected_registry_digest=snapshot,
            expected_plan_digest=first["plan_digest"],
        )
        assert second["plan_matches"] is True and second["ok"] is True

    def test_preflight_flags_failed_checks(self) -> None:
        decision = lw.evaluate_candidates(
            registry=_registry(),
            semantic_op="hqsb::fused_add_rms_norm",
            schema_version="1.0.0",
            target=_cuda_snapshot(),
            evidence=lw.EvidenceIndex(
                [_evidence(), _evidence("hqsb.reference.fused_add_rms_norm", dtypes=("fp16",))]
            ),
            inputs={},
            policy="auto_heuristic",
        )
        report = lw.preflight_checks(
            decision,
            guard_true=False,
            artifact_hash_matches=True,
            evidence_covered=True,
            workspace_fits=True,
            stream_matches=True,
        )
        assert report["proceed"] is False
        assert report["failed"] == ["guard"]
        assert report["action"] == "fallback_before_launch"


@pytest.mark.unit
class TestDispatch:
    def test_dispatch_row_rejects_silent_fallback(self) -> None:
        row = lw.DispatchRow(
            dispatch_id="d",
            compile_id="c",
            variant_id="v",
            requested="cuda_v2",
            eligible=("cuda_v2", "reference"),
            selected="cuda_v2",
            actual="hqsb.reference.fused_add_rms_norm",
            actual_candidate_id="reference",
        )
        assert any("fallback" in problem for problem in row.validate())

    def test_dispatch_row_rejects_ineligible_selection(self) -> None:
        row = lw.DispatchRow(
            dispatch_id="d",
            compile_id="c",
            variant_id="v",
            requested="x",
            eligible=("reference",),
            selected="cuda_v2",
            actual="cuda_v2",
        )
        assert any("not eligible" in problem for problem in row.validate())

    def test_telemetry_summary_counts_fallbacks(self) -> None:
        telemetry = lw.DispatchTelemetry()
        telemetry.record(
            lw.DispatchRow(
                dispatch_id="d1",
                compile_id="c",
                variant_id="v",
                requested="auto",
                eligible=("reference",),
                selected="reference",
                actual="hqsb.reference.fused_add_rms_norm",
                actual_candidate_id="reference",
            )
        )
        telemetry.record(
            lw.DispatchRow(
                dispatch_id="d2",
                compile_id="c",
                variant_id="v",
                requested="cuda_v2",
                eligible=("cuda_v2", "reference"),
                selected="cuda_v2",
                actual="hqsb.reference.fused_add_rms_norm",
                actual_candidate_id="reference",
                fallback_reason="GUARD_FALSE",
            )
        )
        summary = telemetry.summary()
        assert summary["dispatches"] == 2
        assert summary["fallback_count"] == 1
        assert summary["fallback_reasons"] == ["GUARD_FALSE"]

    def test_actual_dispatch_needs_two_independent_evidence_kinds(self) -> None:
        weak = lw.ActualDispatchEvidence(dispatch_id="d", kinds={"compiled_code_symbol": "sym"})
        assert weak.confirm()["enough_evidence"] is False
        strong = lw.ActualDispatchEvidence(
            dispatch_id="d",
            kinds={"compiled_code_symbol": "sym", "profiler_kernel_symbol": "sym"},
        )
        assert strong.confirm()["enough_evidence"] is True

    def test_unknown_evidence_kind_is_flagged(self) -> None:
        report = lw.ActualDispatchEvidence(dispatch_id="d", kinds={"log_line": "selected=x"}).confirm()
        assert report["unknown_kinds"] == ["log_line"]
        assert report["enough_evidence"] is False

    def test_disable_ablation_plan_lists_expected_signals(self) -> None:
        plan = lw.disable_ablation_plan("cuda_v2")
        assert {"trace", "kernel_count", "latency", "output"} <= set(plan["expect"])


@pytest.mark.unit
class TestSideEffectsAndFailures:
    def test_state_mutating_failure_fails_the_request(self) -> None:
        gate = lw.SideEffectGate(mutates_state=True)
        policy = gate.on_runtime_error()
        assert policy["policy"] == "FAIL_REQUEST"
        assert policy["retry_allowed"] is False

    def test_transactional_kernel_may_retry(self) -> None:
        gate = lw.SideEffectGate(mutates_state=True, transactional=True)
        assert gate.on_runtime_error()["retry_allowed"] is True

    def test_pre_launch_order_is_documented(self) -> None:
        gate = lw.SideEffectGate(mutates_state=False)
        plan = gate.plan("cuda_v2")
        assert plan["pre_launch_checks"] == list(lw.PRE_LAUNCH_CHECKS)
        assert "never launch" in plan["order"]

    def test_injected_failures_never_load_or_launch(self) -> None:
        registry = _registry()
        entry = registry.get("cuda_v2")
        for kind in lw.INJECTABLE_LOWERING_FAILURES:
            report = lw.inject_lowering_failure(
                kind, entry=entry, target=_cuda_snapshot(), mutate_state=True
            )
            assert report["attempted_load"] is False
            assert report["attempted_launch"] is False
            assert report["reason_code"]

    def test_unknown_injection_kind_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            lw.inject_lowering_failure("magic", entry=_registry().get("cuda_v2"), target=_cuda_snapshot())


@pytest.mark.unit
class TestOrdersAndCost:
    def test_gate_sequence_must_be_a_prefix(self) -> None:
        assert lw.gate_sequence_ok(list(lw.CORRECTNESS_ORDER[:2]), lw.CORRECTNESS_ORDER)["ok"] is True
        out_of_order = [lw.CORRECTNESS_ORDER[1], lw.CORRECTNESS_ORDER[0]]
        assert lw.gate_sequence_ok(out_of_order, lw.CORRECTNESS_ORDER)["ok"] is False

    def test_next_gate_is_the_first_incomplete_one(self) -> None:
        assert lw.next_gate([], lw.CORRECTNESS_ORDER) == "ir_verifier"
        assert lw.next_gate(["ir_verifier"], lw.CORRECTNESS_ORDER) == "reference_lowering_vs_original"

    def test_performance_requires_all_correctness_gates(self) -> None:
        rows = {gate: "pass" for gate in lw.CORRECTNESS_ORDER}
        rows["qwen_block_selected_tensors"] = "fail"
        assert lw.performance_allowed(rows)["allowed"] is False
        rows["qwen_block_selected_tensors"] = "pass"
        assert lw.performance_allowed(rows)["allowed"] is True

    def test_compile_breakdown_stays_out_of_steady_state(self) -> None:
        report = lw.compile_breakdown(
            capture_time=0.1,
            graph_transform_time=0.2,
            lowering_selection_time=0.05,
            codegen_time=0.3,
            native_compile_link_time=0.4,
            artifact_write_time=0.01,
        )
        assert report["unknown_keys"] == []
        assert "warm_steady_runtime" not in report["breakdown"]
        assert report["cold_total_s"] > 0

    def test_success_status_for_fallback_only_is_a_pass_path(self) -> None:
        decision = lw.LoweringDecision(
            compile_id="c",
            op_instance_id="op",
            semantic_op="hqsb::fused_add_rms_norm",
            schema_version="1.0.0",
            source_ir_id="s",
            targeted_ir_id="t",
            target_snapshot_id="tgt",
            candidates=(
                lw.CandidateEvaluation(
                    candidate_id="cuda_v2",
                    semantic_legal=True,
                    capability_outcome=None,
                    artifact_compatible=True,
                    guard_outcome=True,
                    evidence_covered=True,
                    predicted_cost=None,
                ),
            ),
            selection_policy="forced:cuda_v2",
            selected="",
            fallback_id="reference",
            forced_unsupported=True,
        )
        plan = lw.MaterializedPlan(
            compile_id="c",
            selected="",
            fallback="reference",
            selected_implementation="",
            fallback_implementation="hqsb.reference",
            status="fallback_only",
        )
        assert lw.success_status_for(decision, plan) == "PASS"

    def test_evidence_index_requires_passing_raw_reference(self) -> None:
        with pytest.raises(ConfigError):
            lw.EvidenceIndex([_evidence(raw_ref="", status="pass")])
        with pytest.raises(ConfigError):
            lw.EvidenceIndex([_evidence(tolerance_policy_id="")])

    def test_evidence_scope_covers_shapes(self) -> None:
        index = lw.EvidenceIndex([_evidence()])
        assert index.lookups("hqsb.cuda.fused_add_rms_norm", dtype="fp16", shape="prefill_2048", arch="sm_86", level="operator")["covered"]
        assert not index.lookups("hqsb.cuda.fused_add_rms_norm", dtype="fp8_e4m3", shape="prefill_2048", arch="sm_86", level="operator")["covered"]
