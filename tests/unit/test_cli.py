"""Unit tests for the shared CLI argument validators.

Ensures every public benchmark CLI rejects malformed numeric arguments
with a diagnostic ``argparse.ArgumentTypeError`` (which argparse surfaces
as a non-zero exit with a usage message) instead of failing downstream.
"""

from __future__ import annotations

import argparse

import pytest

from hqsb.benchmark.cli import non_negative_int, positive_int


class TestPositiveInt:
    @pytest.mark.parametrize("value", ["1", "32", "2048"])
    def test_accepts_positive(self, value):
        assert positive_int(value) == int(value)

    @pytest.mark.parametrize("value", ["0", "-1", "-32"])
    def test_rejects_non_positive(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int(value)

    @pytest.mark.parametrize("value", ["abc", "", "1.5", "1e3"])
    def test_rejects_non_integer(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int(value)


class TestNonNegativeInt:
    def test_accepts_zero(self):
        assert non_negative_int("0") == 0

    def test_rejects_negative(self):
        with pytest.raises(argparse.ArgumentTypeError):
            non_negative_int("-1")


class TestArgparseIntegration:
    def _parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input-tokens", type=positive_int, required=True)
        return parser

    def test_valid_argument_parses(self):
        args = self._parser().parse_args(["--input-tokens", "128"])
        assert args.input_tokens == 128

    def test_invalid_argument_exits_nonzero(self):
        with pytest.raises(SystemExit) as excinfo:
            self._parser().parse_args(["--input-tokens", "0"])
        assert excinfo.value.code != 0
