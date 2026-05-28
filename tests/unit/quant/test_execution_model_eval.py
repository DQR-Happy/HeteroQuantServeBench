"""Tests for execution reality labels, phase timing and unified tables."""

from __future__ import annotations


import pytest

from hqsb.core.errors import ConfigError
from hqsb.quant.execution import (
    EXPLICIT_FALLBACK,
    ExecutionRecord,
    FAKE_QUANT,
    FP16_REFERENCE,
    FUSED_DEQUANT_WEIGHT_ONLY,
    INTEGER_TENSOR_OP,
    STORAGE_ONLY,
    claim_audit,
    classify_execution,
    records_to_jsonl,
    parse_records,
)
from hqsb.quant import model_eval as me


@pytest.mark.unit
class TestClassifyExecution:
    def test_fp16_reference_default(self):
        assert classify_execution() == FP16_REFERENCE

    def test_storage_only(self):
        assert classify_execution(weights_materialized_to_fp16=True) == STORAGE_ONLY

    def test_fused_dequant_requires_observed_symbol(self):
        assert (
            classify_execution(packed_layout_id="x", observed_kernel_symbol="sym", kernel_provider="triton")
            == FUSED_DEQUANT_WEIGHT_ONLY
        )

    def test_packed_without_symbol_is_not_low_bit(self):
        # The "API said int4" trap: a layout without an observed symbol must
        # not be labelled low-bit.
        assert classify_execution(packed_layout_id="x") == FP16_REFERENCE

    def test_integer_tensor_op_requires_evidence(self):
        assert (
            classify_execution(
                packed_layout_id="x",
                observed_kernel_symbol="sym",
                kernel_provider="triton",
                integer_dot_evidence=True,
            )
            == INTEGER_TENSOR_OP
        )

    def test_fallback_takes_precedence(self):
        assert classify_execution(fallback_reason="layout_mismatch") == EXPLICIT_FALLBACK

    def test_declared_fake_quant_accepted_only_when_consistent(self):
        # fake_quant without a packed layout is the consistent case.
        assert classify_execution(declared_label=FAKE_QUANT) == FAKE_QUANT
        # fake_quant *with* a packed layout is contradictory: the packed path
        # wins and is classified from its own evidence, not from the claim.
        assert classify_execution(declared_label=FAKE_QUANT, packed_layout_id="x") == FP16_REFERENCE


@pytest.mark.unit
class TestExecutionRecord:
    def test_record_round_trips(self):
        record = ExecutionRecord(
            module="q_proj",
            phase="steady_prefill",
            expected_kernel="hqsb_dequant_gemm_kernel",
            observed_kernel_symbol="hqsb_dequant_gemm_kernel",
            kernel_provider="triton",
            packed_layout_id="hqsb.w4a16.rowmajor.nk.v1",
            m=1,
            n=1024,
            k=2048,
        )
        label = record.execution_label()
        assert label == FUSED_DEQUANT_WEIGHT_ONLY
        assert record.low_bit_executed is True
        assert record.kernel_matches_expected() is True
        rebuilt = parse_records([record.as_dict()])
        assert rebuilt[0].as_dict()["execution_label"] == label

    def test_claim_audit_flags_missing_symbol(self):
        record = ExecutionRecord(
            module="m", phase="p", packed_layout_id="hqsb.w4a16.rowmajor.nk.v1"
        )
        audit = claim_audit([record])
        assert "m" in audit["missing_observed_symbol"]

    def test_records_jsonl_round_trip(self):
        record = ExecutionRecord(module="m", phase="p", duration_ms=1.0)
        text = records_to_jsonl([record])
        assert parse_records([__import__("json").loads(text)])[0].module == "m"

    def test_unknown_field_refused(self):
        with pytest.raises(ConfigError):
            parse_records([{"module": "m", "phase": "p", "extra": 1}])


@pytest.mark.unit
class TestPhaseTimer:
    def test_timer_collects_samples(self):
        timer = me.PhaseTimer(warmup=1, repeats=3, use_cuda_events=False)
        result = timer.time_phase("steady_prefill", lambda: None, tokens=5)
        assert len(result.samples) == 3
        assert result.samples[0].tokens == 5
        assert result.summary()["count"] == 3

    def test_unknown_phase_refused(self):
        timer = me.PhaseTimer(use_cuda_events=False)
        with pytest.raises(ConfigError):
            timer.time_phase("not_a_phase", lambda: None)

    def test_repeats_must_be_positive(self):
        with pytest.raises(ConfigError):
            me.PhaseTimer(repeats=0)


@pytest.mark.unit
class TestMemoryLadder:
    def test_artifact_memory_breakdown(self):
        from hqsb.quant.fixtures import build_tiny_artifact

        doc = build_tiny_artifact()
        measured = me.measure_artifact_memory(doc)
        assert measured["canonical_total_bytes"] > 0
        assert measured["packed_variant_bytes"] > 0
        assert measured["scale_bytes"] > 0


@pytest.mark.unit
class TestPerplexity:
    def test_perfect_prediction(self):
        # One-hot logits where the target has the largest value.
        rows = [[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]]
        report = me.perplexity_from_logits(rows, [1, 1])
        assert report["perplexity"] == pytest.approx(1.0, abs=1e-3)

    def test_denominator_excludes_first_position(self):
        rows = [[0.0, 10.0], [10.0, 0.0], [10.0, 0.0]]
        report = me.perplexity_from_logits(rows, [1, 0, 0])
        assert report["denominator"] == 2
        assert report["excluded_positions"] == 1

    def test_shape_mismatch(self):
        with pytest.raises(ConfigError):
            me.perplexity_from_logits([[1.0, 2.0]], [1, 2])


@pytest.mark.unit
class TestUnifiedTable:
    def test_unknown_field_rejected(self):
        with pytest.raises(ConfigError, match="unknown"):
            me.build_unified_table([{"run_id": "r", "made_up": 1}])

    def test_table_hash_stable(self):
        row = {"run_id": "r", "metric": "ttft_ms", "value": 1.0}
        a = me.build_unified_table([row])
        b = me.build_unified_table([row])
        assert a["table_hash"] == b["table_hash"]
        assert a["rows"][0]["run_id"] == "r"


@pytest.mark.unit
class TestGateOrder:
    def test_correctness_failure_dominates(self):
        report = me.gate_order(
            correctness={"passed": False},
            artifact={"passed": True},
            quality={"verdict": "PASS"},
            execution={"passed": True},
            measurement={"passed": True},
        )
        assert report["classification"] == "rejected"

    def test_fake_quant_is_quality_only(self):
        report = me.gate_order(
            correctness={"passed": True},
            artifact={"passed": True},
            quality={"verdict": "PASS"},
            execution={"passed": False},
            measurement={"passed": True},
        )
        assert report["classification"] == "quality_only"

    def test_all_pass_is_deployment_candidate(self):
        report = me.gate_order(
            correctness={"passed": True},
            artifact={"passed": True},
            quality={"verdict": "PASS"},
            execution={"passed": True},
            measurement={"passed": True},
        )
        assert report["classification"] == "deployment_candidate"

    def test_measurement_failure_is_research_only(self):
        report = me.gate_order(
            correctness={"passed": True},
            artifact={"passed": True},
            quality={"verdict": "PASS"},
            execution={"passed": True},
            measurement={"passed": False},
        )
        assert report["classification"] == "research_only"
