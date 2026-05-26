"""Property-based tests for the quantization math and packing (E05-01 §12).

These are *self-checks*: they prove the interfaces behave correctly across a
generated input space, not that any model was quantized.
"""

from __future__ import annotations

import math

import pytest

from hqsb.quant.packing import pack_canonical, pack_kernel_variant, unpack_canonical, unpack_kernel_variant
from hqsb.quant.rtn import quantize, reconstruction_report
from hqsb.quant.spec import QuantScheme, RangePolicy


@pytest.mark.property
class TestPackingProperties:
    @pytest.mark.parametrize("count", [1, 2, 3, 7, 16, 31, 64, 100])
    def test_nibble_round_trip_is_identity(self, count):
        codes = [(i % 15) - 7 for i in range(count)]
        payload = pack_canonical(codes, 4)
        assert unpack_canonical(payload, count, 4) == codes

    @pytest.mark.parametrize("count", [1, 5, 64, 129])
    def test_byte_round_trip_is_identity(self, count):
        codes = [(i % 255) - 127 for i in range(count)]
        payload = pack_canonical(codes, 8)
        assert unpack_canonical(payload, count, 8) == codes

    def test_independent_decoder_agrees_for_odd_counts(self):
        for count in range(1, 40):
            codes = [(i % 15) - 7 for i in range(count)]
            payload = pack_canonical(codes, 4)
            assert unpack_canonical(payload, count, 4) == codes


@pytest.mark.property
class TestQuantizationProperties:
    @pytest.mark.parametrize("bits", [4, 8])
    @pytest.mark.parametrize("symmetric", [True, False])
    def test_codes_always_within_range(self, bits, symmetric):
        scheme = QuantScheme(
            bits=bits,
            granularity="per_group",
            group_size=16,
            symmetric=symmetric,
            range_policy=RangePolicy.SYMMETRIC if symmetric else RangePolicy.TWOS_COMPLEMENT,
        )
        for seed in range(20):
            values = [
                math.sin((seed * 100 + i) * 0.31) * ((i % 9) - 4) * 0.7
                for i in range(64)
            ]
            qt = quantize(values, scheme, shape=(2, 32))
            assert all(scheme.qmin <= code <= scheme.qmax for code in qt.q)
            assert len(qt.q) == len(values)

    def test_scales_positive_and_finite(self):
        scheme = QuantScheme(bits=4, granularity="per_group", group_size=32, symmetric=True)
        for seed in range(20):
            values = [float(((seed + i) % 17) - 8) for i in range(128)]
            qt = quantize(values, scheme, shape=(2, 64))
            assert all(scale > 0 and math.isfinite(scale) for scale in qt.scales)

    def test_error_never_exceeds_half_scale_of_group(self):
        scheme = QuantScheme(bits=4, granularity="per_group", group_size=32, symmetric=True)
        for seed in range(10):
            values = [float(((seed + i) % 23) - 11) * 0.4 for i in range(128)]
            qt = quantize(values, scheme, shape=(2, 64))
            report = reconstruction_report(values, qt.values_dequant)
            # Worst error is bounded by scale/2 of the group the element sits in.
            assert report["max_abs_error"] <= 1e-6 + max(scale / 2.0 for scale in qt.scales)


@pytest.mark.property
class TestKernelVariantProperties:
    @pytest.mark.parametrize("n,k,group", [(2, 32, 16), (3, 40, 8), (5, 100, 32), (1, 17, 128)])
    def test_kernel_variant_round_trip(self, n, k, group):
        scheme = QuantScheme(bits=4, granularity="per_group", group_size=group, symmetric=True)
        values = [float((i % 31) - 15) * 0.3 for i in range(n * k)]
        qt = quantize(values, scheme, shape=(n, k))
        packed = pack_kernel_variant(qt.q, qt.scales, qt.zeros, qt.scheme, n, k)
        assert unpack_kernel_variant(packed, scheme) == qt.q
        assert unpack_kernel_variant(packed, scheme, independent=True) == qt.q

    def test_padding_is_never_decoded_as_data(self):
        scheme = QuantScheme(bits=4, granularity="per_group", group_size=32, symmetric=True)
        values = [1.0] * 40  # K=40 -> one group of 32 + tail of 8
        qt = quantize(values, scheme, shape=(1, 40))
        packed = pack_kernel_variant(qt.q, qt.scales, qt.zeros, qt.scheme, 1, 40)
        recovered = unpack_kernel_variant(packed, scheme, independent=True)
        assert len(recovered) == 40
        assert recovered == qt.q


@pytest.mark.property
class TestSpecProperty:
    def test_scheme_hash_is_collision_free_for_distinct_schemes(self):
        seen = set()
        from hqsb.quant.spec import factorial_matrix

        for case in factorial_matrix():
            if case.expected_reject is not None:
                continue
            scheme = case.resolve(k=64)
            digest = scheme.scheme_hash()
            assert digest not in seen, f"hash collision for {scheme.label}"
            seen.add(digest)
