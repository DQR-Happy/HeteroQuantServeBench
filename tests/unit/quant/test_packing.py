"""Unit tests for canonical/kernel packing (E05-01 §10/§12, E05-06 §9)."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ArtifactError
from hqsb.quant import packing
from hqsb.quant.spec import MAIN_W4_GROUP, MAIN_W8, QuantScheme


@pytest.mark.unit
class TestCanonicalNibble:
    def test_round_trip_all_codes(self):
        codes = list(range(-8, 8))
        payload = packing.pack_canonical(codes, 4)
        assert packing.unpack_canonical(payload, len(codes), 4) == codes
        assert packing.unpack_independent(payload, len(codes), 4) == codes

    def test_payload_length(self):
        assert len(packing.pack_canonical([1, 2, 3], 4)) == 2  # ceil(3/2)
        assert len(packing.pack_canonical([1, 2, 3, 4], 4)) == 2
        assert len(packing.pack_canonical([1], 8)) == 1

    def test_odd_count_padding_nibble_is_zero(self):
        payload = packing.pack_canonical([1, -2, 3], 4)
        assert payload[-1] >> 4 == 0  # high nibble of the trailing byte is 0
        assert packing.unpack_canonical(payload, 3, 4) == [1, -2, 3]

    def test_nonzero_padding_is_refused(self):
        payload = bytearray(packing.pack_canonical([1], 4))
        payload[0] |= 0xF0  # corrupt the padding nibble
        with pytest.raises(ArtifactError, match="padding"):
            packing.unpack_canonical(bytes(payload), 1, 4)

    def test_byte_packing_is_little_endian_int8(self):
        codes = [127, -128, 0, 1]
        payload = packing.pack_canonical(codes, 8)
        assert list(payload) == [127, 128, 0, 1]  # -128 stored as 0x80
        assert packing.unpack_canonical(payload, 4, 8) == codes

    def test_unsupported_bits(self):
        with pytest.raises(ArtifactError, match="no canonical packing"):
            packing.pack_canonical([1, 2], 5)

    def test_unsupported_nibble_order(self):
        with pytest.raises(ArtifactError):
            packing.pack_canonical([1, 2], 4, nibble_order="weird")

    def test_validate_codes_enforces_range(self):
        scheme = MAIN_W4_GROUP
        packing.validate_codes([-7, 0, 7], scheme)
        with pytest.raises(ArtifactError, match="outside"):
            packing.validate_codes([-8], scheme)
        with pytest.raises(ArtifactError, match="outside"):
            packing.validate_codes([8], scheme)
        with pytest.raises(ArtifactError, match="outside"):
            packing.validate_codes([-129], MAIN_W8)


@pytest.mark.unit
class TestKnownLayouts:
    def test_known_layouts_are_declared(self):
        assert packing.LAYOUT_W4A16_ROWMAJOR_NK_V1 in packing.KNOWN_LAYOUTS
        assert packing.LAYOUT_W4A16_HIFIRST_NK_V1 in packing.KNOWN_LAYOUTS
        assert packing.LAYOUT_W8A16_ROWMAJOR_NK_V1 in packing.KNOWN_LAYOUTS
        bits, order = packing.KNOWN_LAYOUTS[packing.LAYOUT_W4A16_HIFIRST_NK_V1]
        assert bits == 4 and order == packing.NIBBLE_HIGH_FIRST

    def test_hifirst_payload_differs_and_round_trips(self):
        codes = [1, -2, 3, -4]
        low = packing.pack_canonical(codes, 4, nibble_order=packing.NIBBLE_LOW_FIRST)
        high = packing.pack_canonical(codes, 4, nibble_order=packing.NIBBLE_HIGH_FIRST)
        assert low != high
        assert packing.unpack_independent(high, 4, 4, nibble_order=packing.NIBBLE_HIGH_FIRST) == codes

    def test_unknown_layout_is_refused(self):
        with pytest.raises(ArtifactError, match="unknown kernel layout"):
            packing.plan_kernel_layout(MAIN_W4_GROUP, 4, 64, layout_id="hqsb.unknown")

    def test_layout_bit_mismatch_is_refused(self):
        with pytest.raises(ArtifactError, match="bit"):
            packing.plan_kernel_layout(
                MAIN_W8, 4, 64, layout_id=packing.LAYOUT_W4A16_ROWMAJOR_NK_V1
            )


@pytest.mark.unit
class TestKernelVariant:
    def _pack(self, n=4, k=64, group=32, bits=4, symmetric=True):
        from hqsb.quant.rtn import quantize
        from hqsb.quant.spec import RangePolicy

        scheme = QuantScheme(
            bits=bits,
            granularity="per_group",
            group_size=group,
            symmetric=symmetric,
            range_policy=RangePolicy.SYMMETRIC if symmetric else RangePolicy.TWOS_COMPLEMENT,
        )
        values = [float((i % 29) - 14) * 0.25 for i in range(n * k)]
        qt = quantize(values, scheme, shape=(n, k))
        layout_id = (
            packing.LAYOUT_W4A16_ROWMAJOR_NK_V1
            if bits == 4
            else packing.LAYOUT_W8A16_ROWMAJOR_NK_V1
        )
        packed = packing.pack_kernel_variant(
            qt.q, qt.scales, qt.zeros, qt.scheme, n, k, layout_id=layout_id
        )
        return packed, qt, scheme

    def test_round_trip_kernel_variant(self):
        packed, qt, scheme = self._pack()
        recovered = packing.unpack_kernel_variant(packed, scheme)
        assert recovered == qt.q
        assert packing.unpack_kernel_variant(packed, scheme, independent=True) == qt.q

    def test_asymmetric_round_trip(self):
        packed, qt, scheme = self._pack(symmetric=False)
        assert packing.unpack_kernel_variant(packed, scheme) == qt.q

    def test_tail_shape(self):
        packed, qt, scheme = self._pack(n=3, k=40, group=16)
        assert packing.unpack_kernel_variant(packed, scheme) == qt.q

    def test_row_stride_is_aligned(self):
        packed, _qt, _scheme = self._pack()
        assert packed.layout.row_stride_bytes % 16 == 0
        assert packed.layout.row_stride_bytes >= packing.packed_row_bytes(
            4, packed.layout.padded_cols
        )

    def test_variant_hash_is_stable(self):
        packed_a, _qt, _scheme = self._pack()
        packed_b, _qt, _scheme = self._pack()
        assert packed_a.variant_hash() == packed_b.variant_hash()

    def test_shape_mismatch_is_refused(self):
        from hqsb.quant.rtn import quantize

        qt = quantize([float(i) for i in range(8)], MAIN_W4_GROUP, shape=(2, 4))
        with pytest.raises(ArtifactError, match="values"):
            packing.pack_kernel_variant(qt.q, qt.scales, qt.zeros, qt.scheme, 3, 4)

    def test_packed_row_bytes(self):
        assert packing.packed_row_bytes(4, 64) == 32
        assert packing.packed_row_bytes(8, 64) == 64
        assert packing.packed_row_bytes(4, 65) == 33  # ceil(65/2)


@pytest.mark.unit
class TestNegativePacking:
    def test_pack_refuses_nan_scale(self):
        with pytest.raises(ArtifactError, match="scale"):
            packing.pack_kernel_variant(
                [1, 2, 3, 4], [float("nan")], [], MAIN_W4_GROUP, 1, 4
            )

    def test_pack_refuses_non_positive_scale(self):
        with pytest.raises(ArtifactError, match="scale"):
            packing.pack_kernel_variant(
                [1, 2, 3, 4], [0.0], [], MAIN_W4_GROUP, 1, 4
            )

    def test_unpack_refuses_payload_length_mismatch(self):
        scheme = QuantScheme(bits=4, granularity="per_group", group_size=4)
        packed = packing.PackedTensor(
            layout=packing.plan_kernel_layout(scheme, 1, 4),
            payload=b"\x00",
            scales=[1.0],
            zeros=[],
        )
        with pytest.raises(ArtifactError):
            packing.unpack_kernel_variant(packed, scheme, independent=True)

    def test_pack_canonical_rejects_non_integer_code(self):
        with pytest.raises(ArtifactError, match="not an integer"):
            packing.pack_canonical([1.5], 4)
