"""Unit/property tests for RTN math and the golden cross-oracle (E05-01 §9)."""

from __future__ import annotations

import math

import pytest

from hqsb.core.errors import ConfigError
from hqsb.quant.golden import (
    build_golden_vectors,
    cross_check_with_reference,
    golden_package_hash,
    golden_package_json,
    quantize_exact,
)
from hqsb.quant.spec import (
    GRANULARITY_PER_CHANNEL,
    GRANULARITY_PER_GROUP,
    GRANULARITY_PER_TENSOR,
    MAIN_W4_GROUP,
    MAIN_W8,
    ConstantGroupPolicy,
    NaNPolicy,
    QuantScheme,
    RangePolicy,
    ZeroGroupPolicy,
)
from hqsb.quant.rtn import (
    QuantizedTensor,
    clamp_free_error_bound,
    dequantize,
    quantize,
    reconstruction_report,
    requantize_idempotence_check,
)


def _scheme(**overrides) -> QuantScheme:
    return MAIN_W4_GROUP.with_overrides(**overrides)


@pytest.mark.unit
class TestMathInvariants:
    def test_codes_always_inside_the_declared_range(self):
        values = [0.0, 1.0, -1.0, 1e3, -1e3, 0.3333, -0.7777]
        for scheme in (MAIN_W8, _scheme(), _scheme(symmetric=False, range_policy=RangePolicy.TWOS_COMPLEMENT)):
            qt = quantize(values, scheme, shape=(1, len(values)))
            assert all(scheme.qmin <= code <= scheme.qmax for code in qt.q)

    def test_zero_maps_to_zero(self):
        qt = quantize([0.0, 0.0, 0.0, 0.0], _scheme(), shape=(1, 4))
        assert qt.q == [0, 0, 0, 0]
        assert qt.values_dequant == [0.0, 0.0, 0.0, 0.0]
        assert qt.stats.zero_group_count == 1

    def test_zero_group_policy_min_positive(self):
        scheme = _scheme(zero_group_policy=ZeroGroupPolicy.SCALE_MIN_POSITIVE)
        qt = quantize([0.0] * 4, scheme, shape=(1, 4))
        assert qt.scales[0] > 0.0
        assert math.isfinite(qt.scales[0])

    def test_scales_are_positive_and_finite(self):
        values = [float(index) - 8.0 for index in range(64)]
        qt = quantize(values, _scheme(), shape=(1, 64))
        assert all(scale > 0 and math.isfinite(scale) for scale in qt.scales)
        assert not qt.stats.invalid_scale_units

    def test_reconstruction_error_bounded_by_half_scale(self):
        values = [0.1 * index for index in range(64)]
        qt = quantize(values, _scheme(), shape=(1, 64))
        bound = clamp_free_error_bound(qt)
        worst = max(abs(a - b) for a, b in zip(values, qt.values_dequant))
        assert worst <= bound + 1e-9

    def test_symmetric_scheme_is_antisymmetric(self):
        values = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]
        qt = quantize(values, _scheme(), shape=(1, 8))
        positives = [code for code, value in zip(qt.q, values) if value > 0]
        negatives = [code for code, value in zip(qt.q, values) if value < 0]
        assert positives == [-code for code in negatives]

    def test_requantization_is_idempotent_without_clamping(self):
        values = [0.05 * index for index in range(64)]
        qt = quantize(values, _scheme(), shape=(1, 64))
        assert qt.stats.clamp_count == 0
        assert requantize_idempotence_check(qt)

    def test_monotonicity_within_one_scale(self):
        values = [index * 0.01 for index in range(32)]
        qt = quantize(values, _scheme(), shape=(1, 32))
        assert qt.q == sorted(qt.q)

    def test_input_is_not_mutated(self):
        values = [0.25, -0.5, 0.75, -1.0]
        snapshot = list(values)
        quantize(values, _scheme(), shape=(1, 4))
        assert values == snapshot

    def test_dequantize_matches_stored_values(self):
        values = [0.3 * index for index in range(32)]
        qt = quantize(values, _scheme(), shape=(1, 32))
        assert dequantize(qt) == qt.values_dequant


@pytest.mark.unit
class TestGranularityAndAxis:
    def test_per_tensor_has_exactly_one_scale(self):
        scheme = QuantScheme(bits=8, granularity=GRANULARITY_PER_TENSOR)
        qt = quantize([1.0, 2.0, 3.0, 4.0], scheme, shape=(2, 2))
        assert qt.num_units == 1
        assert len(qt.scales) == 1
        assert qt.scales[0] == pytest.approx(4.0 / 127.0)

    def test_per_channel_one_scale_per_row(self):
        scheme = QuantScheme(bits=8, granularity=GRANULARITY_PER_CHANNEL, axis=-1)
        qt = quantize([float(index) for index in range(8)], scheme, shape=(2, 4))
        assert qt.num_units == 2
        assert len(qt.scales) == 2

    def test_per_channel_axis_zero_is_per_column(self):
        scheme = QuantScheme(bits=8, granularity=GRANULARITY_PER_CHANNEL, axis=0)
        qt = quantize([float(index) for index in range(8)], scheme, shape=(2, 4))
        assert qt.num_units == 4
        # column 0 holds values 0 and 4 → its scale must cover both.
        assert qt.scales[0] == pytest.approx(4.0 / 127.0)

    def test_tail_group_is_not_dropped(self):
        scheme = _scheme(group_size=32)
        values = [0.1 * index for index in range(40)]  # K=40 → groups [32, 8]
        qt = quantize(values, scheme, shape=(1, 40))
        assert qt.units_per_row == 2
        assert qt.tail == 8
        assert all(code is not None for code in qt.q)
        assert len(qt.q) == 40

    def test_group_larger_than_k_single_unit(self):
        scheme = _scheme(group_size=128)
        qt = quantize([1.0, 2.0, 3.0], scheme, shape=(1, 3))
        assert qt.units_per_row == 1
        assert qt.tail == 3

    def test_unit_of_index_is_axis_aware(self):
        scheme = _scheme(group_size=4)
        qt = quantize([float(index) for index in range(8)], scheme, shape=(1, 8))
        assert [qt.unit_of(index) for index in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]

    def test_per_tensor_scheme_reports_zero_group_once(self):
        qt = quantize([0.0] * 6, QuantScheme(bits=8, granularity=GRANULARITY_PER_TENSOR), shape=(2, 3))
        assert qt.stats.zero_group_count == 1


@pytest.mark.unit
class TestPoliciesAndErrors:
    def test_constant_group_reject_policy(self):
        scheme = QuantScheme(
            bits=8,
            symmetric=False,
            range_policy=RangePolicy.TWOS_COMPLEMENT,
            granularity=GRANULARITY_PER_TENSOR,
            constant_group_policy=ConstantGroupPolicy.REJECT,
        )
        with pytest.raises(ConfigError, match="constant group"):
            quantize([7.0, 7.0, 7.0, 7.0], scheme, shape=(1, 4))

    def test_constant_group_encoded_with_zero_code(self):
        scheme = QuantScheme(
            bits=8,
            symmetric=False,
            range_policy=RangePolicy.TWOS_COMPLEMENT,
            granularity=GRANULARITY_PER_TENSOR,
        )
        qt = quantize([7.0, 7.0, 7.0, 7.0], scheme, shape=(1, 4))
        assert qt.stats.constant_group_count == 1
        assert qt.q == [0, 0, 0, 0]

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_nan_inf_rejected_by_default(self, bad):
        with pytest.raises(ConfigError, match="non-finite"):
            quantize([1.0, bad], _scheme(), shape=(1, 2))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_nan_inf_propagate_marks_the_stats(self, bad):
        scheme = _scheme(nan_policy=NaNPolicy.PROPAGATE)
        qt = quantize([1.0, bad], scheme, shape=(1, 2))
        assert qt.stats.nonfinite_count == 1

    def test_shape_mismatch_is_refused(self):
        with pytest.raises(ConfigError, match="implies"):
            quantize([1.0, 2.0, 3.0], _scheme(), shape=(1, 2))

    def test_axis_out_of_range_is_refused(self):
        with pytest.raises(ConfigError, match="axis"):
            quantize([1.0, 2.0], _scheme(), shape=(1, 2), axis=5)


@pytest.mark.unit
class TestReconstructionReport:
    def test_report_contains_the_protocol_metrics(self):
        report = reconstruction_report([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
        for key in (
            "max_abs_error",
            "mean_abs_error",
            "rmse",
            "cosine_similarity",
            "l2_relative_error",
            "sqnr_db",
            "first_mismatch",
        ):
            assert key in report
        assert report["first_mismatch"] == 0.0

    def test_identical_vectors_have_infinite_sqnr(self):
        report = reconstruction_report([1.0, 2.0], [1.0, 2.0])
        assert report["max_abs_error"] == 0.0
        assert report["sqnr_db"] == float("inf")


@pytest.mark.unit
class TestGoldenVectors:
    def test_exact_oracle_agrees_with_the_hypothesis(self):
        report = cross_check_with_reference()
        assert report["mismatches"] == []
        assert report["checked"] >= 10

    def test_golden_package_hash_is_stable(self):
        assert golden_package_hash() == golden_package_hash()
        assert '"vectors"' in golden_package_json()

    def test_vectors_cover_the_required_cases(self):
        tags = {tag for vector in build_golden_vectors() for tag in vector.tags}
        assert {"codes", "round", "tie", "zero-group", "tail"} <= tags
        assert any("asymmetric" in vector.name for vector in build_golden_vectors())

    def test_exact_oracle_rejects_shape_mismatch(self):
        with pytest.raises(ConfigError):
            quantize_exact([1, 2, 3], MAIN_W4_GROUP, shape=(1, 2))

    def test_exact_oracle_tie_results(self):
        from fractions import Fraction

        scheme = QuantScheme(bits=4, granularity=GRANULARITY_PER_TENSOR)
        result = quantize_exact(
            [Fraction(7), Fraction(5, 2), Fraction(-5, 2)], scheme, shape=(3,)
        )
        # amax = 7 → scale = 1 → 2.5 ties to even 2, -2.5 ties to even -2
        assert result.q == [7, 2, -2]


@pytest.mark.unit
class TestPropertySweeps:
    @pytest.mark.parametrize("bits", [4, 8])
    @pytest.mark.parametrize("symmetric", [True, False])
    def test_quantize_dequantize_within_range(self, bits, symmetric):
        scheme = QuantScheme(
            bits=bits,
            granularity=GRANULARITY_PER_GROUP,
            group_size=16,
            symmetric=symmetric,
            range_policy=RangePolicy.SYMMETRIC if symmetric else RangePolicy.TWOS_COMPLEMENT,
        )
        values = [math.sin(index * 0.37) * (index % 11) * 0.5 for index in range(64)]
        qt = quantize(values, scheme, shape=(2, 32))
        assert all(scheme.qmin <= code <= scheme.qmax for code in qt.q)
        assert len(qt.values_dequant) == len(values)
        assert all(math.isfinite(value) for value in qt.values_dequant)

    @pytest.mark.parametrize("shape", [(1, 1), (1, 7), (3, 5), (2, 64)])
    def test_odd_shapes_round_trip(self, shape):
        values = [0.13 * index for index in range(shape[0] * shape[1])]
        qt = quantize(values, _scheme(group_size=8), shape=shape)
        assert isinstance(qt, QuantizedTensor)
        assert len(qt.q) == shape[0] * shape[1]
        assert dequantize(qt) == qt.values_dequant
