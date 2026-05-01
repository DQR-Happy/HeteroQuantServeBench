"""Unit tests for :mod:`hqsb.benchmark.workload`.

Uses a lightweight fake tokenizer so the deterministic input-length
contract can be validated on CPU without model weights or a GPU.
"""

from __future__ import annotations

import pytest
import torch

from hqsb.benchmark.workload import make_fixed_token_input


class _FakeTokenizer:
    """Minimal tokenizer returning a fixed token id sequence."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [7, 8, 9]}


class TestMakeFixedTokenInput:
    def test_exact_token_count(self):
        tokenizer = _FakeTokenizer()
        inputs = make_fixed_token_input(tokenizer, 10, device="cpu")
        assert inputs["input_ids"].shape == (1, 10)
        assert inputs["attention_mask"].shape == (1, 10)
        assert torch.all(inputs["attention_mask"] == 1)

    def test_truncates_to_exact_length(self):
        inputs = make_fixed_token_input(_FakeTokenizer(), 5, device="cpu")
        assert inputs["input_ids"].shape[1] == 5

    def test_longer_than_seed_repeats(self):
        inputs = make_fixed_token_input(_FakeTokenizer(), 30, device="cpu")
        assert inputs["input_ids"].shape[1] == 30

    def test_dtype_is_long(self):
        inputs = make_fixed_token_input(_FakeTokenizer(), 3, device="cpu")
        assert inputs["input_ids"].dtype == torch.long
        assert inputs["attention_mask"].dtype == torch.long

    def test_zero_tokens_raises(self):
        with pytest.raises(ValueError):
            make_fixed_token_input(_FakeTokenizer(), 0, device="cpu")

    def test_empty_tokenizer_output_raises(self):
        class _EmptyTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": []}

        with pytest.raises(RuntimeError):
            make_fixed_token_input(_EmptyTokenizer(), 3, device="cpu")
