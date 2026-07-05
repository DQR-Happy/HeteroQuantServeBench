from __future__ import annotations

import math

import pytest

from hqsb.quant.legacy_e05_01.rtn import (
    RtnSpec,
    dequantize,
    pack_int4,
    quantize,
    round_nearest_even,
    unpack_int4,
)


@pytest.mark.unit
@pytest.mark.property
def test_round_nearest_even_halfway_both_signs():
    values = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    assert [round_nearest_even(value) for value in values] == [
        -4,
        -2,
        -2,
        0,
        0,
        2,
        2,
        4,
    ]


@pytest.mark.unit
@pytest.mark.property
def test_int4_every_code_and_odd_row_round_trip():
    every_code = tuple(range(-8, 8))
    packed = pack_int4(every_code, (1, 16))
    assert packed.hex() == "98badcfe10325476"
    assert unpack_int4(packed, (1, 16)) == every_code

    odd = (-8, -1, 0, 1, 7)
    odd_packed = pack_int4(odd, (1, 5))
    assert odd_packed.hex() == "f81007"
    assert unpack_int4(odd_packed, (1, 5)) == odd
    with pytest.raises(ValueError, match="padding"):
        unpack_int4(odd_packed[:-1] + bytes([0xF7]), (1, 5))


@pytest.mark.unit
@pytest.mark.property
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize(
    "granularity,group_size",
    [("per-tensor", None), ("per-channel", None), ("per-group", 4)],
)
def test_quantize_tail_invariants(bits, symmetric, granularity, group_size):
    source = [
        [-7.0, -3.5, -1.0, 0.0, 0.5, 2.5, 7.0],
        [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
    ]
    before = [row[:] for row in source]
    spec = RtnSpec(
        bits=bits,
        symmetric=symmetric,
        granularity=granularity,
        group_size=group_size,
    )
    result = quantize(source, spec)
    reconstructed = dequantize(result)
    assert source == before
    assert result.original_shape == (2, 7)
    assert all(spec.qmin <= value <= spec.qmax for value in result.qvalues)
    assert all(math.isfinite(value) and value > 0 for value in result.scales)
    assert all(math.isfinite(value) for row in reconstructed for value in row)
    if granularity == "per-group":
        assert result.parameter_shape == (2, 2)
        assert result.group_valid_sizes == (4, 3, 4, 3)


@pytest.mark.unit
def test_zero_and_asymmetric_constant_groups_are_finite_and_exact():
    zero = quantize(
        [[0.0, 0.0]],
        RtnSpec(bits=4, symmetric=False, granularity="per-tensor"),
    )
    assert zero.scales == (1.0,)
    assert zero.zero_points == (0,)
    assert dequantize(zero) == [[0.0, 0.0]]

    constant = quantize(
        [[-0.25, -0.25, -0.25]],
        RtnSpec(bits=4, symmetric=False, granularity="per-tensor"),
    )
    assert constant.qvalues == (-1, -1, -1)
    assert constant.zero_points == (0,)
    assert dequantize(constant) == [[-0.25, -0.25, -0.25]]


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"bits": 3, "symmetric": True, "granularity": "per-tensor"},
        {"bits": 4, "symmetric": True, "granularity": "bad"},
        {"bits": 4, "symmetric": True, "granularity": "per-group"},
        {
            "bits": 4,
            "symmetric": True,
            "granularity": "per-channel",
            "group_size": 4,
        },
        {"bits": 4, "symmetric": True, "granularity": "per-tensor", "axis": 0},
    ],
)
def test_illegal_specs_rejected(kwargs):
    with pytest.raises(ValueError):
        RtnSpec(**kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_input_rejected(bad):
    with pytest.raises(ValueError, match="NaN and Inf"):
        quantize(
            [[0.0, bad]],
            RtnSpec(bits=4, symmetric=True, granularity="per-tensor"),
        )
