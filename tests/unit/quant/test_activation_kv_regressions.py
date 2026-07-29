"""Numerical regressions found by the E05-07/E05-08 experiment collectors."""
from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.quant.activation import ActivationQuantSpec, quantize_activation, w8a8_reference
from hqsb.quant.kv import KvQuantSpec, capacity_ratio, kv_capacity


@pytest.mark.unit
class TestActivationExperimentRegressions:
    def test_distinct_row_and_output_scales_with_bias(self):
        # X @ W = [[4, 5, 5], [10, 11, 9]]. The non-square output
        # distinguishes token rows from output channels independently.
        actual = w8a8_reference(
            [1, 2, 3, 4], [2, 1, -1, 1, 2, 3],
            x_scales=[0.5, 2.0], w_scales=[0.25, 3.0, 0.1],
            bias=[-0.5, 0.25, 2.0], m=2, n=3, k=2,
        )
        assert actual == pytest.approx([0.0, 7.75, 2.25, 4.5, 66.25, 3.8])

    @pytest.mark.parametrize("x_scales,w_scales,expected", [
        ([0.5], [0.25, 3.0], [0.5, 7.5, 1.25, 16.5]),
        ([0.5, 2.0], [0.25], [0.5, 0.625, 5.0, 5.5]),
    ])
    def test_scalar_broadcast_is_independent_for_each_operand(self, x_scales, w_scales, expected):
        assert w8a8_reference([1, 2, 3, 4], [2, 1, 1, 2],
                              x_scales=x_scales, w_scales=w_scales,
                              m=2, n=2, k=2) == pytest.approx(expected)

    @pytest.mark.parametrize("override", [
        {"x_scales": []}, {"x_scales": [1, 2, 3]},
        {"w_scales": [1, 2, 3]}, {"bias": [1]},
    ])
    def test_ambiguous_scale_or_bias_dimensions_are_refused(self, override):
        options = {"x_scales": [1.0], "w_scales": [1.0], "m": 2, "n": 2, "k": 2}
        options.update(override)
        with pytest.raises(ConfigError):
            w8a8_reference([1, 2, 3, 4], [2, 1, 1, 2], **options)

    @pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_scale_cannot_silently_enter_oracle(self, scale):
        with pytest.raises(ConfigError, match="scale"):
            w8a8_reference([1], [1], x_scales=[scale], w_scales=[1], m=1, n=1, k=1)

    @pytest.mark.parametrize("operand", ["x_units_per_row", "w_units_per_row"])
    def test_k_group_scales_require_group_boundaries(self, operand):
        with pytest.raises(ConfigError, match="group"):
            w8a8_reference([1, 1], [1, 1], x_scales=[1, 2], w_scales=[1, 2],
                              m=1, n=1, k=2, **{operand: 2})

    def test_half_range_utilization_differs_from_saturation(self):
        result = quantize_activation(
            [-127, 127, -64, 64, -63, 0, 63],
            ActivationQuantSpec(mode="static", granularity="per_tensor"),
            shape=(1, 7), static_scales=[1.0],
        )
        assert result.scale_utilization == pytest.approx(4 / 7)
        assert result.saturation_rate == pytest.approx(2 / 7)


@pytest.mark.unit
class TestKvExperimentRegressions:
    def test_residual_defaults_to_the_declared_fp16_baseline(self):
        spec = KvQuantSpec(k_bits=4, v_bits=4, residual_window=32)
        dims = dict(layers=28, kv_heads=8, head_dim=128, tokens=128)
        default = kv_capacity(spec, **dims)
        explicit = kv_capacity(spec, **dims, dtype_bytes=2)
        assert default.residual_bytes == 28 * 8 * 128 * 2 * 2 * 32
        assert default.total_bytes == explicit.total_bytes

    def test_fp16_payload_has_no_quant_scale_or_zero_metadata(self):
        spec = KvQuantSpec(k_bits=16, v_bits=16, k_zero_point=True, v_zero_point=True)
        result = kv_capacity(spec, layers=2, kv_heads=3, head_dim=5, tokens=7)
        assert result.payload_bytes == 2 * 3 * 5 * 7 * 2 * 2
        assert result.scale_bytes == result.zero_bytes == 0
        assert result.page_header_bytes > 0

    def test_mixed_fp16_and_int8_only_counts_quantized_metadata(self):
        spec = KvQuantSpec(k_bits=16, v_bits=8, k_zero_point=True, v_zero_point=True)
        result = kv_capacity(spec, layers=2, kv_heads=3, head_dim=5, tokens=7)
        assert result.scale_bytes == 2 * 3 * 7 * 4
        assert result.zero_bytes == 2 * 3 * 7

    @pytest.mark.parametrize("granularity", ["per_tensor", "per_channel"])
    def test_all_residual_cache_has_no_quant_metadata(self, granularity):
        spec = KvQuantSpec(k_bits=4, v_bits=8, residual_window=8,
                           k_granularity=granularity, v_granularity=granularity,
                           k_zero_point=True, v_zero_point=True)
        result = kv_capacity(spec, layers=2, kv_heads=3, head_dim=5, tokens=7)
        assert result.payload_bytes == result.scale_bytes == result.zero_bytes == 0
        assert result.residual_bytes == 2 * 3 * 5 * 7 * 2 * 2

    def test_capacity_ratio_residual_dtype_is_explicit(self):
        spec = KvQuantSpec(k_bits=4, v_bits=4, residual_window=2)
        dims = dict(layers=2, kv_heads=3, head_dim=8, tokens=7)
        default = capacity_ratio(spec, **dims)
        fp32_residual = capacity_ratio(spec, **dims, dtype_bytes=4)
        explicit_capacity = kv_capacity(spec, **dims, dtype_bytes=4)
        assert fp32_residual["quantized_total_bytes"] == explicit_capacity.total_bytes
        assert fp32_residual["fp16_payload_bytes"] == default["fp16_payload_bytes"]
        assert fp32_residual["quantized_total_bytes"] - default["quantized_total_bytes"] == 2 * 3 * 8 * 2 * 2 * 2
