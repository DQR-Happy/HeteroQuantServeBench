"""Unit tests for the frozen quantization spec and round modes (E05-01 §3–§5)."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.quant.rounding import (
    ROUND_FLOOR,
    ROUND_HALF_AWAY_FROM_ZERO,
    ROUND_NEAREST_EVEN,
    ROUND_TRUNC,
    coerce_round_value,
    get_rounder,
    is_rtn,
    round_half_away_from_zero,
    round_nearest_even,
)
from hqsb.quant.spec import (
    GRANULARITY_PER_CHANNEL,
    GRANULARITY_PER_GROUP,
    GRANULARITY_PER_TENSOR,
    MAIN_W4_GROUP,
    MAIN_W8,
    NaNPolicy,
    QuantScheme,
    RangePolicy,
    factorial_matrix,
    groups_per_row,
    integer_range,
    tail_length,
)


@pytest.mark.unit
class TestIntegerRange:
    def test_symmetric_range_drops_most_negative_code(self):
        assert integer_range(8, RangePolicy.SYMMETRIC) == (-127, 127)
        assert integer_range(4, RangePolicy.SYMMETRIC) == (-7, 7)

    def test_twos_complement_range(self):
        assert integer_range(8, RangePolicy.TWOS_COMPLEMENT) == (-128, 127)
        assert integer_range(4, RangePolicy.TWOS_COMPLEMENT) == (-8, 7)

    @pytest.mark.parametrize("bits,policy", [(0, RangePolicy.SYMMETRIC), (9, RangePolicy.SYMMETRIC), (8, "nope")])
    def test_illegal_range_arguments(self, bits, policy):
        with pytest.raises(ConfigError):
            integer_range(bits, policy)


@pytest.mark.unit
class TestGroupArithmetic:
    def test_groups_per_row_covers_tail(self):
        assert groups_per_row(256, 128) == 2
        assert groups_per_row(260, 128) == 3
        assert groups_per_row(100, 128) == 1  # G > K: one partial group
        assert groups_per_row(10, 4) == 3
        assert groups_per_row(10, None) == 1  # per-channel: one unit per row
        assert groups_per_row(0, 128) == 0

    def test_tail_length(self):
        assert tail_length(256, 128) == 128
        assert tail_length(260, 128) == 4
        assert tail_length(100, 128) == 100
        assert tail_length(10, None) == 10
        assert tail_length(0, 4) == 0

    @pytest.mark.parametrize("k,group", [(-1, 4), (10, 0), (10, -2)])
    def test_illegal_group_arguments(self, k, group):
        with pytest.raises(ConfigError):
            groups_per_row(k, group)
        with pytest.raises(ConfigError):
            tail_length(k, group)


@pytest.mark.unit
class TestSchemeValidation:
    def test_main_schemes_are_valid_and_hashable(self):
        assert MAIN_W8.scheme_hash() != MAIN_W4_GROUP.scheme_hash()
        assert MAIN_W8.qmin == -127 and MAIN_W8.qmax == 127
        assert MAIN_W4_GROUP.group_size == 128
        assert MAIN_W8.scheme_hash() == MAIN_W8.scheme_hash()  # stable

    def test_per_group_requires_group_size(self):
        with pytest.raises(ConfigError, match="group_size"):
            QuantScheme(bits=4, granularity=GRANULARITY_PER_GROUP, group_size=None)

    def test_per_channel_must_not_declare_group_size(self):
        with pytest.raises(ConfigError, match="group_size"):
            QuantScheme(
                bits=8,
                granularity=GRANULARITY_PER_CHANNEL,
                group_size=128,
            )

    def test_per_tensor_must_not_declare_group_size(self):
        with pytest.raises(ConfigError):
            QuantScheme(
                bits=8,
                granularity=GRANULARITY_PER_TENSOR,
                group_size=8,
            )

    def test_symmetric_with_twos_complement_is_ambiguous(self):
        with pytest.raises(ConfigError, match="ambiguous"):
            QuantScheme(bits=8, symmetric=True, range_policy=RangePolicy.TWOS_COMPLEMENT)

    def test_asymmetric_requires_full_range(self):
        with pytest.raises(ConfigError):
            QuantScheme(bits=8, symmetric=False, range_policy=RangePolicy.SYMMETRIC)

    def test_unknown_fields_are_rejected_on_load(self):
        payload = MAIN_W8.as_dict()
        payload["mystery"] = 1
        with pytest.raises(ConfigError, match="unknown"):
            QuantScheme.from_mapping(payload)

    def test_nan_policy_values(self):
        scheme = MAIN_W8.with_overrides(nan_policy=NaNPolicy.PROPAGATE)
        assert scheme.nan_policy == NaNPolicy.PROPAGATE


@pytest.mark.unit
class TestFactorialMatrix:
    def test_legal_cells_resolve_and_illegal_cells_are_refused(self):
        cases = factorial_matrix()
        assert len(cases) >= 30
        illegal = [case for case in cases if case.expected_reject]
        assert illegal, "the matrix must pre-register illegal combinations"
        for case in illegal:
            with pytest.raises(ConfigError):
                case.resolve(k=64)

    def test_shape_relative_group_sizes_need_k(self):
        cases = [
            case
            for case in factorial_matrix(bits=(4,), granularities=(GRANULARITY_PER_GROUP,))
            if case.group_size_expression
        ]
        assert cases
        with pytest.raises(ConfigError):
            cases[0].resolve()
        resolved_k = [
            case.resolve(k=96) for case in cases if case.group_size_expression == "K"
        ]
        assert resolved_k and resolved_k[0].group_size == 96

    def test_every_legal_cell_has_a_distinct_label(self):
        labels = [
            case.resolve(k=64).label
            for case in factorial_matrix()
            if case.expected_reject is None
        ]
        assert len(labels) == len(set(labels))


@pytest.mark.unit
class TestRoundModes:
    def test_nearest_even_ties(self):
        assert round_nearest_even(2.5) == 2.0
        assert round_nearest_even(3.5) == 4.0
        assert round_nearest_even(-2.5) == -2.0
        assert round_nearest_even(-3.5) == -4.0
        assert round_nearest_even(0.5) == 0.0
        assert round_nearest_even(-0.5) == 0.0

    def test_half_away_from_zero_ties(self):
        assert round_half_away_from_zero(2.5) == 3.0
        assert round_half_away_from_zero(-2.5) == -3.0
        assert round_half_away_from_zero(0.5) == 1.0
        assert round_half_away_from_zero(-0.5) == -1.0

    def test_non_tie_values_agree(self):
        for value in (1.2, -1.2, 1.8, -1.8, 0.0, 7.0):
            assert round_nearest_even(value) == float(round(value))
            assert round_half_away_from_zero(value) == float(round(value))

    @pytest.mark.parametrize("mode", [ROUND_TRUNC, ROUND_FLOOR])
    def test_non_rtn_modes_are_refused_by_default(self, mode):
        with pytest.raises(ConfigError, match="not a round-to-nearest"):
            get_rounder(mode)
        assert get_rounder(mode, allow_non_rtn=True) is not None
        assert not is_rtn(mode)

    def test_unknown_round_mode(self):
        with pytest.raises(ConfigError):
            get_rounder("magic")

    def test_coerce_round_value_rejects_non_integers(self):
        assert coerce_round_value(3.0) == 3
        with pytest.raises(ConfigError):
            coerce_round_value(3.5)

    def test_rounder_rejects_non_finite(self):
        for rounder in (round_nearest_even, round_half_away_from_zero):
            with pytest.raises(ConfigError):
                rounder(float("nan"))
            with pytest.raises(ConfigError):
                rounder(float("inf"))

    def test_round_modes_are_declared(self):
        assert is_rtn(ROUND_NEAREST_EVEN) and is_rtn(ROUND_HALF_AWAY_FROM_ZERO)
