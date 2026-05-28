"""Tests for coverage, reversible weight application and quality metrics."""

from __future__ import annotations


import pytest

from hqsb.core.errors import ConfigError
from hqsb.quant import quality as Q
from hqsb.quant.coverage import CoveragePolicy, classify_weight, coverage_summary
from hqsb.quant.spec import MAIN_W4_GROUP

torch = pytest.importorskip("torch")


@pytest.mark.unit
class TestCoverage:
    def test_explicit_include_wins(self):
        policy = CoveragePolicy(include=("lm_head",))
        quantized, selected_by, reason = classify_weight("lm_head.weight", "Linear", policy)
        assert quantized and selected_by == "explicit_include"

    def test_norm_is_excluded_by_type(self):
        policy = CoveragePolicy()
        quantized, _by, reason = classify_weight("model.norm.weight", "RMSNorm", policy)
        assert not quantized and reason == "excluded_type:RMSNorm"

    def test_embedding_is_excluded(self):
        policy = CoveragePolicy()
        quantized, _by, reason = classify_weight("model.embed_tokens.weight", "Embedding", policy)
        assert not quantized and "Embedding" in reason

    def test_bias_excluded_by_default(self):
        policy = CoveragePolicy()
        quantized, _by, reason = classify_weight("model.layers.0.q_proj.bias", "Linear", policy, is_bias=True)
        assert not quantized and reason == "bias_excluded"

    def test_linear_by_default(self):
        policy = CoveragePolicy()
        quantized, selected_by, reason = classify_weight("model.layers.0.mlp.down_proj.weight", "Linear", policy)
        assert quantized and selected_by == "default_linear"

    def test_not_linear_projection_excluded(self):
        policy = CoveragePolicy()
        quantized, _by, reason = classify_weight("model.some_random.weight", "Linear", policy)
        assert not quantized and reason == "not_linear_projection"

    def test_coverage_summary_fraction(self):
        from hqsb.quant.coverage import WeightRow

        rows = [
            WeightRow("a.q_proj.weight", "Linear", (2, 2), "float16", 4, True, "default_linear"),
            WeightRow("b.norm.weight", "RMSNorm", (4,), "float16", 4, False, "", "excluded_type:RMSNorm"),
        ]
        summary = coverage_summary(rows)
        assert summary["parameter_coverage"] == pytest.approx(0.5)


@pytest.mark.unit
class TestApplySwap:
    def test_swap_and_restore_is_bitwise(self):
        import torch.nn as nn

        from hqsb.quant import apply as A
        from hqsb.quant.artifact import ModelIdentity, QuantArtifactDocument
        from hqsb.quant.packing import pack_kernel_variant
        from hqsb.quant.rtn import quantize

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(16, 8, bias=False)

            def forward(self, x):
                return self.q_proj(x)

        torch.manual_seed(0)
        model = Tiny()
        weight = [float(v) for v in model.q_proj.weight.detach().reshape(-1).tolist()]
        qt = quantize(weight, MAIN_W4_GROUP, shape=tuple(model.q_proj.weight.shape))
        doc = QuantArtifactDocument.from_quantized(
            qt, tensor_name="q_proj.weight", model=ModelIdentity(model_id="m", revision="r")
        )
        doc.add_variant(
            pack_kernel_variant(qt.q, qt.scales, qt.zeros, qt.scheme, model.q_proj.weight.shape[0], model.q_proj.weight.shape[1])
        )
        x = torch.randn(2, 16)
        reference = model(x).detach().clone()
        swaps, backups = A.swap_weights(model, {"q_proj.weight": doc}, mode=A.MODE_FAKE_DEQUANT)
        assert A.application_report(swaps)["low_bit_claim_allowed"] is False
        records = A.restore_weights(model, swaps, backups)
        assert all(record["restore_ok"] for record in records)
        assert torch.equal(model(x).detach(), reference)

    def test_kernel_mode_requires_executor(self):
        import torch.nn as nn

        from hqsb.quant import apply as A
        from hqsb.quant.artifact import ModelIdentity, QuantArtifactDocument
        from hqsb.quant.packing import pack_kernel_variant
        from hqsb.quant.rtn import quantize

        model = nn.Linear(8, 4, bias=False)
        weight = [float(v) for v in model.weight.detach().reshape(-1).tolist()]
        qt = quantize(weight, MAIN_W4_GROUP, shape=tuple(model.weight.shape))
        doc = QuantArtifactDocument.from_quantized(qt, tensor_name="weight", model=ModelIdentity(model_id="m", revision="r"))
        doc.add_variant(pack_kernel_variant(qt.q, qt.scales, qt.zeros, qt.scheme, 4, 8))
        with pytest.raises(ConfigError):
            A.swap_weights(model, {"weight": doc}, mode=A.MODE_KERNEL)

    def test_mismatched_module_is_refused(self):
        import torch.nn as nn

        from hqsb.quant import apply as A
        from hqsb.quant.artifact import ModelIdentity, QuantArtifactDocument
        from hqsb.quant.rtn import quantize

        model = nn.Linear(8, 4, bias=False)
        weight = [float(v) for v in model.weight.detach().reshape(-1).tolist()]
        qt = quantize(weight, MAIN_W4_GROUP, shape=tuple(model.weight.shape))
        doc = QuantArtifactDocument.from_quantized(qt, tensor_name="other.weight", model=ModelIdentity(model_id="m", revision="r"))
        with pytest.raises(Exception):
            A.swap_weights(model, {"other.weight": doc}, mode=A.MODE_FAKE_DEQUANT)

    def test_fake_quant_weight_matches_dtype(self):
        from hqsb.quant import apply as A

        weight = torch.randn(4, 8, dtype=torch.float16)
        out = A.fake_quant_weight(weight, MAIN_W4_GROUP)
        assert out.dtype == torch.float16
        assert out.shape == weight.shape


@pytest.mark.unit
class TestQualityMetrics:
    def test_kl_and_js(self):
        p = [1.0, 2.0, 3.0]
        q = [1.0, 2.0, 3.0]
        assert Q.kl_divergence(p, q) == pytest.approx(0.0, abs=1e-9)
        assert Q.jensen_shannon(p, q) == pytest.approx(0.0, abs=1e-9)
        assert Q.kl_divergence(p, q) >= 0

    def test_topk_and_overlap(self):
        assert Q.topk_indices([1.0, 3.0, 2.0, 0.0], 2) == [1, 2]
        assert Q.overlap([1, 2, 3], [1, 2, 9]) == pytest.approx(2 / 3)

    def test_position_metrics_identical(self):
        metrics = Q.position_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], position=0)
        assert metrics.logit_max_abs_error == 0.0
        assert metrics.top1_flipped is False
        assert metrics.logit_cosine == pytest.approx(1.0)

    def test_position_metrics_flip_detected(self):
        metrics = Q.position_metrics([3.0, 2.0, 1.0], [2.0, 3.0, 1.0], position=0)
        assert metrics.top1_flipped is True

    def test_teacher_forcing_report(self):
        report = Q.teacher_forcing_report(
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
            [[1.0, 2.0, 3.0], [2.0, 3.0, 1.0]],
            sample_id="s",
        )
        assert report["positions"] == 2
        assert report["first_divergence_position"] == 1
        assert "kl_fp16_quant" in report["means"]

    def test_teacher_forcing_requires_equal_length(self):
        with pytest.raises(ConfigError):
            Q.teacher_forcing_report([[1.0, 2.0]], [[1.0, 2.0], [3.0, 4.0]])

    def test_generation_report(self):
        steps = [Q.GenerationStep(step=i, context_length=10 + i, token=1, kv_bytes=100, kl_fp16_quant=0.01) for i in range(5)]
        report = Q.generation_report(steps)
        assert report["steps"] == 5
        assert report["kv_bytes_final"] == 100


@pytest.mark.unit
class TestQualityGate:
    def _budget(self, margins, slices=()):
        return Q.QualityBudget(name="b", margins=margins, slices=slices)

    def test_unknown_metric_is_refused(self):
        with pytest.raises(ConfigError):
            Q.evaluate_quality_gate(
                self._budget({}),
                paired_metrics={"made_up_metric": ([1.0], [1.0])},
            )

    def test_missing_slice_is_inconclusive(self):
        budget = self._budget({"kl_fp16_quant": 0.1}, slices=("long",))
        report = Q.evaluate_quality_gate(
            budget,
            paired_metrics={"kl_fp16_quant": ([0.1, 0.2], [0.11, 0.21])},
        )
        assert report["slices"]["long"]["verdict"] == "INCONCLUSIVE"

    def test_hard_safety_failure_forces_fail(self):
        budget = self._budget({"kl_fp16_quant": 1.0})
        report = Q.evaluate_quality_gate(
            budget,
            paired_metrics={"kl_fp16_quant": ([0.1], [0.11])},
            safety={"no_nan_inf": False},
        )
        assert report["overall_verdict"] == "FAIL"

    def test_worst_slice_report(self):
        budget = self._budget({"kl_fp16_quant": 1.0}, slices=("short",))
        report = Q.evaluate_quality_gate(
            budget,
            paired_metrics={"kl_fp16_quant": ([0.1, 0.2], [0.11, 0.21])},
            slice_metrics={"short": {"kl_fp16_quant": ([0.1, 0.2], [0.11, 0.21])}},
        )
        assert Q.worst_slice_report(report) == "short"
