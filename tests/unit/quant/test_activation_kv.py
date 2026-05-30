"""Tests for activation quantization (E05-07) and KV cache (E05-08)."""

from __future__ import annotations


import pytest

from hqsb.core.errors import ConfigError
from hqsb.quant import activation as act
from hqsb.quant import kv


@pytest.mark.unit
class TestActivationSpec:
    def test_per_group_requires_group_size(self):
        with pytest.raises(ConfigError, match="group_size"):
            act.ActivationQuantSpec(granularity="per_group")

    def test_non_8bit_must_declare_range(self):
        with pytest.raises(ConfigError, match="default"):
            act.ActivationQuantSpec(bits=4)

    def test_clip_percentile_range(self):
        with pytest.raises(ConfigError):
            act.ActivationQuantSpec(clip_percentile=1.5)

    def test_scheme_mapping_per_token(self):
        scheme = act.scheme_for_activation(act.ActivationQuantSpec(granularity="per_token"), cols=64)
        assert scheme.axis == -1
        assert scheme.granularity == "per_channel"


@pytest.mark.unit
class TestQuantizeActivation:
    def test_dynamic_per_token(self):
        spec = act.ActivationQuantSpec(granularity="per_token", mode=act.DYNAMIC)
        result = act.quantize_activation([1.5, -2.0, 0.5, 0.1, 0.2, -0.3], spec, shape=(2, 3))
        assert len(result.scales) == 2  # one per token
        assert result.saturation_rate >= 0
        assert result.scale_utilization >= 0

    def test_dynamic_refuses_static_scales(self):
        spec = act.ActivationQuantSpec(mode=act.DYNAMIC)
        with pytest.raises(ConfigError, match="static"):
            act.quantize_activation([1.0], spec, shape=(1, 1), static_scales=[0.01])

    def test_static_requires_scales(self):
        spec = act.ActivationQuantSpec(mode=act.STATIC)
        with pytest.raises(ConfigError, match="static"):
            act.quantize_activation([1.0], spec, shape=(1, 1))

    def test_shape_mismatch_refused(self):
        spec = act.ActivationQuantSpec(mode=act.DYNAMIC)
        with pytest.raises(ConfigError, match="implies"):
            act.quantize_activation([1.0, 2.0], spec, shape=(1, 3))

    def test_per_channel_axis_zero(self):
        spec = act.ActivationQuantSpec(granularity="per_channel", mode=act.DYNAMIC)
        result = act.quantize_activation([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], spec, shape=(2, 3))
        assert len(result.scales) == 3  # one per column


@pytest.mark.unit
class TestSmoothQuant:
    def test_transform_is_equivariant(self):
        smooth = act.compute_smooth_scale([10.0, 1.0], [1.0, 1.0], alpha=0.5)
        weight = smooth.apply_to_weight([1.0, 1.0, 1.0, 1.0], cols=2)
        activations = smooth.apply_to_activation([2.0, 2.0], cols=2)
        # X W = (X / s) (s W) elementwise
        assert weight[0] * activations[0] == pytest.approx(2.0)
        assert weight[1] * activations[1] == pytest.approx(2.0)

    def test_alpha_out_of_range(self):
        with pytest.raises(ConfigError):
            act.compute_smooth_scale([1.0], [1.0], alpha=1.5)


@pytest.mark.unit
class TestW8A8Reference:
    def test_exact_integer_accumulation(self):
        out = act.w8a8_reference(
            [1, -2], [1, 0, 0, 1], x_scales=[0.5], w_scales=[0.25], m=1, n=2, k=2
        )
        # col 0: 1*1 + (-2)*0 = 1  -> 0.5*0.25*1 = 0.125
        # col 1: 1*0 + (-2)*1 = -2 -> 0.5*0.25*(-2) = -0.25
        assert out == pytest.approx([0.125, -0.25])

    def test_shape_mismatch(self):
        with pytest.raises(ConfigError):
            act.int32_accumulate([1, 2, 3], [1, 0, 0, 1], m=1, n=2, k=2)

    def test_zero_point_correction_must_be_explicit(self):
        with pytest.raises(ConfigError, match="zero"):
            act.w8a8_reference(
                [1, 2], [1, 0], x_scales=[1.0], w_scales=[1.0],
                m=1, n=1, k=2, x_zeros=[0], w_zeros=[0],
            )


@pytest.mark.unit
class TestKvSpec:
    def test_rope_point_validated(self):
        with pytest.raises(ConfigError, match="rope"):
            kv.KvQuantSpec(rope_point="during_rope")

    def test_bits_validated(self):
        with pytest.raises(ConfigError):
            kv.KvQuantSpec(k_bits=0)

    def test_per_group_requires_group_size(self):
        with pytest.raises(ConfigError, match="group_size"):
            kv.KvQuantSpec(k_granularity="per_group")


@pytest.mark.unit
class TestKvCapacity:
    def test_capacity_includes_metadata(self):
        spec = kv.KvQuantSpec(k_bits=8, v_bits=8, residual_window=32)
        capacity = kv.kv_capacity(spec, layers=28, kv_heads=8, head_dim=128, tokens=4096)
        assert capacity.payload_bytes > 0
        assert capacity.scale_bytes > 0
        assert capacity.residual_bytes > 0
        assert capacity.total_bytes > capacity.payload_bytes

    def test_bytes_per_token_positive(self):
        spec = kv.KvQuantSpec(k_bits=4, v_bits=8)
        capacity = kv.kv_capacity(spec, layers=1, kv_heads=1, head_dim=64, tokens=128)
        assert capacity.as_dict()["bytes_per_token"] > 0

    def test_reconcile_reports_residual(self):
        spec = kv.KvQuantSpec(k_bits=8, v_bits=8)
        capacity = kv.kv_capacity(spec, layers=1, kv_heads=1, head_dim=64, tokens=128)
        report = capacity.reconcile(capacity.total_bytes + 100, tolerance_bytes=50)
        assert report["within_tolerance"] is False
        assert report["residual_bytes"] == 100

    def test_fp16_baseline(self):
        baseline = kv.fp16_baseline_bytes(layers=1, kv_heads=1, head_dim=64, tokens=128)
        assert baseline == 1 * 1 * 64 * 2 * 2 * 128


@pytest.mark.unit
class TestAttentionOracle:
    def test_reference_matches_hand_calc(self):
        q = [[1.0, 0.0]]
        k = [[1.0, 0.0], [0.0, 1.0]]
        v = [[1.0, 2.0], [3.0, 4.0]]
        output = kv.attention_reference(q, k, v)
        # scale = 1/sqrt(2); scores = [0.7071, 0]; softmax weights =
        # [e^0.7071, 1]/Z -> output[0][0] = w0*1 + w1*3 ~= 1.6604
        assert output[0][0] == pytest.approx(1.6604, abs=1e-3)
        assert output[0][1] == pytest.approx(2.6604, abs=1e-3)

    def test_comparison_self_is_zero_error(self):
        q = [[1.0, 0.0, 0.0]]
        k = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        v = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        report = kv.attention_comparison_report(q, k, v, k, v)
        assert report["attention_output"]["max_abs_error"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
class TestKvSweeps:
    def test_context_matrix(self):
        matrix = kv.context_generation_matrix(context_lengths=[128, 256], output_lengths=[8, 16])
        assert len(matrix) == 4
        assert matrix[0]["total_cache_length"] == 128 + 8

    def test_context_matrix_refuses_bad_inputs(self):
        with pytest.raises(ConfigError):
            kv.context_generation_matrix(context_lengths=[0], output_lengths=[1])

    def test_generation_sweep(self):
        steps = [kv.KvStepRecord(step=i, context_length=10 + i, kv_bytes=100, logit_kl=0.01, diverged=(i == 3)) for i in range(5)]
        report = kv.generation_sweep(steps, requested_output_tokens=5)
        assert report["first_divergence_step"] == 3
        assert report["completed"] is True

    def test_max_context_search_records_probes(self):
        def probe(context):
            fits = context <= 4096
            return {"fits": fits, "free_bytes": 0 if context > 4096 else 8192 - context}

        report = kv.max_context_search(probe, low=1024, high=8192, safety_margin_bytes=2048)
        assert report["max_context"] == 4096
        assert len(report["probes"]) > 0
        assert all(p["safety_margin_bytes"] == 2048 for p in report["probes"])
