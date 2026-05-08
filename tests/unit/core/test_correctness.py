"""Unit tests for correctness/determinism/golden comparison."""

from __future__ import annotations

import pytest

from hqsb.benchmark.correctness import (
    compare_against_golden,
    compare_first_token_logits,
    compare_token_sequence,
    hash_token_sequence,
    verify_determinism,
)
from hqsb.core.errors import SchemaError


@pytest.mark.unit
class TestHashTokenSequence:
    def test_stable_hash(self):
        assert hash_token_sequence([1, 2, 3]) == hash_token_sequence([1, 2, 3])

    def test_distinguishes_sequences(self):
        assert hash_token_sequence([1, 2]) != hash_token_sequence([12])

    def test_distinguishes_order(self):
        assert hash_token_sequence([1, 2]) != hash_token_sequence([2, 1])


@pytest.mark.unit
class TestCompareTokenSequence:
    def test_match(self):
        result = compare_token_sequence([1, 2, 3], [1, 2, 3])
        assert result.passed is True
        # On match, no first_mismatch_index key is present.
        assert "first_mismatch_index" not in result.details

    def test_first_mismatch_localized(self):
        result = compare_token_sequence([1, 2, 9, 4], [1, 2, 3, 4])
        assert result.passed is False
        assert result.details["first_mismatch_index"] == 2
        assert result.details["actual_token"] == 9
        assert result.details["expected_token"] == 3

    def test_prefix_mismatch(self):
        result = compare_token_sequence([1, 2, 3], [1, 2])
        assert result.passed is False
        assert result.details["first_mismatch_index"] == 2


@pytest.mark.unit
class TestCompareLogits:
    def test_close(self):
        assert compare_first_token_logits([1.0, 2.0], [1.0, 2.0]).passed

    def test_difference_detected(self):
        result = compare_first_token_logits([1.0], [2.0], rtol=0.0, atol=0.1)
        assert result.passed is False

    def test_length_mismatch(self):
        result = compare_first_token_logits([1.0], [1.0, 2.0])
        assert result.passed is False
        assert result.details["reason"] == "length_mismatch"


@pytest.mark.unit
class TestVerifyDeterminism:
    def test_all_identical(self):
        result = verify_determinism([[1, 2], [1, 2], [1, 2]])
        assert result.passed is True
        assert result.details["distinct_sequence_hashes"] == 1

    def test_diverged(self):
        result = verify_determinism([[1, 2], [1, 3]])
        assert result.passed is False
        assert result.details["distinct_sequence_hashes"] == 2

    def test_empty(self):
        assert verify_determinism([]).passed is False


@pytest.mark.unit
class TestCompareAgainstGolden:
    def test_full_match(self):
        golden = {
            "schema_version": "1.0.0",
            "generated_tokens": [1, 2, 3],
            "first_token": {"logits": [0.1, 0.2]},
        }
        result = compare_against_golden(
            golden, generated_tokens=[1, 2, 3], first_token_logits=[0.1, 0.2]
        )
        assert result.passed is True
        assert result.method == "golden"
        assert "tokens" in result.details
        assert "first_token_logits" in result.details

    def test_token_mismatch(self):
        golden = {"schema_version": "1.0.0", "generated_tokens": [1, 2]}
        result = compare_against_golden(golden, generated_tokens=[1, 9])
        assert result.passed is False

    def test_missing_version_rejected(self):
        with pytest.raises(SchemaError):
            compare_against_golden({}, generated_tokens=[1])

    def test_missing_generated_tokens_rejected(self):
        golden = {"schema_version": "1.0.0"}
        with pytest.raises(SchemaError):
            compare_against_golden(golden, generated_tokens=[1])

    def test_no_inputs_rejected(self):
        golden = {"schema_version": "1.0.0", "generated_tokens": [1]}
        with pytest.raises(SchemaError):
            compare_against_golden(golden)
