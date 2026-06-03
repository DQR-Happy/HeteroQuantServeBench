"""Lowering decisions, allocation accounting and C6/C7 projection (E06-07)."""

from __future__ import annotations

import pytest

from hqsb.core.contracts.result import BenchmarkResult
from hqsb.core.errors import ConfigError, RegistryError
from hqsb.core.contracts.trace import TraceEvent, TraceEventType
from hqsb.integration import lowering, telemetry


def _fused_request(**overrides) -> lowering.LoweringRequest:
    payload = {
        "pattern_id": "hqsb.pattern.residual_add_rms_norm",
        "pattern_version": "1.0.0",
        "node": "fused_norm",
        "op": "hqsb::fused_add_rms_norm",
        "dtype": "float16",
        "layout": "contiguous",
        "rank": 2,
        "m": 1,
        "arch": "sm_86",
    }
    payload.update(overrides)
    return lowering.LoweringRequest(**payload)


@pytest.mark.unit
class TestLoweringRegistry:
    def test_selection_is_capability_driven(self):
        registry = lowering.frozen_registry()
        decision = registry.select(_fused_request())
        assert decision.ok
        assert decision.selected_name == "hqsb.cuda.fused_add_rms_norm"
        assert decision.selected is not None
        assert decision.selected.kernel_symbol == "hqsb_fused_add_rms_norm_v1"

    def test_rejected_targets_carry_reasons(self):
        registry = lowering.frozen_registry()
        decision = registry.select(_fused_request(dtype="int8"))
        assert decision.fallback
        assert decision.fallback_reason == "NO_ELIGIBLE_LOWERING_TARGET"
        assert all(item.reason for item in decision.rejected)

    def test_arch_mismatch_rejects_the_kernels_but_keeps_inductor(self):
        registry = lowering.frozen_registry()
        decision = registry.select(_fused_request(arch="sm_120"))
        reasons = {item.reason for item in decision.rejected}
        assert "BACKEND_CAPABILITY" in reasons
        # The arch-agnostic Inductor route stays eligible: a capability refusal
        # must not become "no path at all" when a reference target exists.
        assert decision.selected_name == "hqsb.inductor.generated"

    def test_group_size_requirement_is_enforced(self):
        registry = lowering.frozen_registry()
        decision = registry.select(
            lowering.LoweringRequest(
                pattern_id="hqsb.pattern.rmsnorm_quantize",
                op="hqsb::dequant_linear",
                dtype="float16",
                rank=2,
                m=1,
                group_size=7,
            )
        )
        rejected = {item.target: item.reason for item in decision.rejected}
        assert rejected.get("hqsb.triton.dequant_linear") == "QUANT_POLICY"
        assert rejected.get("hqsb.cuda.dequant_linear") == "QUANT_POLICY"

    def test_priority_rule_prefers_the_configured_target(self):
        registry = lowering.frozen_registry()
        decode = registry.select(
            lowering.LoweringRequest(
                pattern_id="hqsb.pattern.rmsnorm_quantize",
                op="hqsb::dequant_linear",
                dtype="float16",
                rank=2,
                m=1,
                group_size=128,
            )
        )
        prefill = registry.select(
            lowering.LoweringRequest(
                pattern_id="hqsb.pattern.rmsnorm_quantize",
                op="hqsb::dequant_linear",
                dtype="float16",
                rank=2,
                m=256,
                group_size=128,
            )
        )
        assert decode.selected_name == "hqsb.triton.dequant_linear"
        assert prefill.selected_name == "hqsb.cuda.dequant_linear"
        assert decode.rule is not None

    def test_duplicate_target_registration_refused(self):
        registry = lowering.frozen_registry()
        with pytest.raises(RegistryError):
            registry.register(
                lowering.LoweringTarget(
                    name="hqsb.cuda.fused_add_rms_norm", provider="cuda_shared_lib"
                )
            )

    def test_selection_audit_rejects_model_name_selectors(self):
        registry = lowering.LoweringRegistry(
            rules=(
                lowering.PriorityRule(
                    name="qwen_special", tags=("layers.0.self_attn",), prefer=("a",)
                ),
            )
        )
        audit = registry.selection_audit()
        assert not audit["ok"]
        assert audit["findings"][0]["reason"] == "MODEL_NAME_SELECTOR"

    def test_decision_serialises_every_required_field(self):
        payload = lowering.frozen_registry().select(_fused_request()).as_dict()
        for key in ("request", "selected", "rejected", "rule", "generated_artifact", "observed_kernel"):
            assert key in payload


@pytest.mark.unit
class TestAllocationAccounting:
    def test_tensor_bytes_uses_dtype_sizes(self):
        assert lowering.tensor_bytes((2, 64), "float16") == 256
        assert lowering.tensor_bytes((2, 64), "float32") == 512
        with pytest.raises(ConfigError):
            lowering.tensor_bytes((2, 64), "complex128")

    def test_fusion_saving_model_lists_removed_tensors(self):
        model = lowering.fuse_saving_model(
            [("add_out", (1, 64)), ("norm_out", (1, 64))],
            [("fused_out", (1, 64))],
            dtype="float16",
        )
        assert model["before_bytes"] == 256
        assert model["after_bytes"] == 128
        assert model["saving_bytes"] == 128
        assert "add_out" in model["removed_tensors"]

    def test_reconcile_explains_the_residual(self):
        account = lowering.AllocationAccount(
            intermediate_bytes_before=1024,
            intermediate_bytes_after=0,
            workspace_bytes=128,
            materialized_dequant_bytes=64,
            contiguous_copies_bytes=64,
        )
        report = account.reconcile(measured_saving_bytes=768)
        assert report["residual_bytes"] == 0
        assert report["explained"] is True

    def test_unexplained_residual_stays_visible(self):
        account = lowering.AllocationAccount(
            intermediate_bytes_before=1024, intermediate_bytes_after=0
        )
        report = account.reconcile(measured_saving_bytes=512)
        assert report["residual_bytes"] == -512
        assert report["explained"] is False

    def test_launch_delta_is_signed(self):
        account = lowering.AllocationAccount(
            intermediate_bytes_before=0,
            intermediate_bytes_after=0,
            launch_count_before=8,
            launch_count_after=3,
        )
        assert account.launch_delta == -5


@pytest.mark.unit
class TestAmdahlAndAttribution:
    def test_amdahl_bound(self):
        prediction = lowering.AmdahlPrediction(
            target_fraction=0.5, speedup=2.0, baseline_phase_ms=100.0
        )
        assert prediction.predicted_model_speedup == pytest.approx(1.3333, rel=1e-3)
        assert prediction.predicted_delta_ms == pytest.approx(75.0)

    def test_invalid_parameters_refused(self):
        with pytest.raises(ConfigError):
            lowering.AmdahlPrediction(target_fraction=1.5, speedup=2.0, baseline_phase_ms=1.0)
        with pytest.raises(ConfigError):
            lowering.AmdahlPrediction(target_fraction=0.5, speedup=0.0, baseline_phase_ms=1.0)

    def test_attribution_residual_must_be_explained(self):
        prediction = lowering.AmdahlPrediction(
            target_fraction=0.5, speedup=2.0, baseline_phase_ms=100.0
        )
        explained = lowering.AttributionReport(
            prediction=prediction, actual_delta_ms=71.0, factors={"fewer_hits": -4.0}
        )
        assert explained.residual_ms == 0.0
        assert explained.as_dict()["explained"] is True
        unexplained = lowering.AttributionReport(prediction=prediction, actual_delta_ms=60.0)
        assert not unexplained.as_dict()["explained"]

    def test_ablation_matrix_disables_one_factor_at_a_time(self):
        cases = lowering.ablation_matrix()
        assert len(cases) == 5
        disabled = [case for case in cases if not (case.pattern_enabled and case.lowering_enabled and case.kernel_enabled)]
        assert len(disabled) == 3
        assert any(not case.epilogue_enabled for case in cases)


@pytest.mark.unit
class TestC6Projection:
    def test_projection_validates_against_the_frozen_c6(self):
        result = telemetry.project_c6(
            "run-1",
            telemetry.S06ResultFields(
                graph_identity="g",
                compile_identity="c",
                cache_layer="dynamo_code",
                requested_lowering="hqsb.cuda.fused_add_rms_norm",
                actual_lowering="hqsb.cuda.fused_add_rms_norm",
                observed_kernel="hqsb_fused_add_rms_norm_v1",
                raw_artifacts={"graph": "artifact://graph"},
            ),
        )
        assert isinstance(result, BenchmarkResult)
        assert result.summary["s06"]["observed_kernel"] == "hqsb_fused_add_rms_norm_v1"

    def test_field_coverage_is_complete(self):
        coverage = telemetry.c6_field_coverage()
        assert coverage["ok"] is True
        assert coverage["missing"] == []
        assert len(coverage["fields"]) == len(telemetry.S06_C6_FIELDS)

    def test_fallback_without_a_reason_is_refused(self):
        with pytest.raises(ConfigError):
            telemetry.S06ResultFields(
                requested_lowering="hqsb.cuda.fused_add_rms_norm",
                actual_lowering="torch.eager.reference",
                fallback_reason="",
            )

    def test_fallback_with_a_reason_is_accepted(self):
        fields = telemetry.S06ResultFields(
            requested_lowering="hqsb.cuda.fused_add_rms_norm",
            actual_lowering="torch.eager.reference",
            fallback_reason="STRIDE_LAYOUT",
        )
        assert fields.fallback_reason == "STRIDE_LAYOUT"

    def test_unknown_correctness_status_refused(self):
        with pytest.raises(ConfigError):
            telemetry.S06ResultFields(correctness_status="probably_fine")

    def test_negative_counts_refused(self):
        with pytest.raises(ConfigError):
            telemetry.S06ResultFields(graph_count=-1)


@pytest.mark.unit
class TestC7Projection:
    def test_every_s06_kind_maps_to_a_valid_frozen_event(self):
        coverage = telemetry.c7_kind_coverage()
        assert coverage["ok"] is True
        assert coverage["invalid"] == []

    def test_records_project_onto_trace_events(self):
        collector = telemetry.TraceCollector(run_id="run-1")
        collector.emit("capture", request_id="r0", nodes=12)
        collector.emit("cache_lookup", request_id="r0", layer="dynamo_code")
        events = telemetry.to_trace_events(collector.records, run_id="run-1", trace_id="t1")
        assert len(events) == 2
        assert isinstance(events[0], TraceEvent)
        assert events[0].attributes[telemetry.C7_KIND_ATTRIBUTE] == "capture"
        assert events[1].event_type == TraceEventType.CACHE

    def test_unknown_kind_refused(self):
        with pytest.raises(ConfigError):
            telemetry.TraceCollector(run_id="r").emit("something_new")

    def test_trace_join_check_passes_for_wellformed_records(self):
        collector = telemetry.TraceCollector(run_id="run-1")
        first = collector.emit("capture", span_id="s1")
        collector.emit("kernel", span_id="s2", parent_span_id=first.span_id)
        events = telemetry.to_trace_events(collector.records, run_id="run-1", trace_id="t1")
        assert telemetry.trace_join_check(events)["ok"] is True

    def test_dangling_parent_is_detected(self):
        collector = telemetry.TraceCollector(run_id="run-1")
        collector.emit("kernel", span_id="s2", parent_span_id="missing")
        events = telemetry.to_trace_events(collector.records, run_id="run-1", trace_id="t1")
        report = telemetry.trace_join_check(events)
        assert report["ok"] is False
        assert report["dangling_parent"] == ["s2"]

    def test_monotonic_timestamps(self):
        collector = telemetry.TraceCollector(run_id="run-1")
        for kind in ("capture", "pattern_hit", "lowering_select", "kernel"):
            collector.emit(kind)
        assert collector.monotonic() is True

    def test_summary_reports_both_schemas(self):
        summary = telemetry.c6_c7_summary()
        assert summary["c6_schema_version"] == BenchmarkResult.SCHEMA_VERSION
        assert summary["c7_schema_version"] == TraceEvent.SCHEMA_VERSION
