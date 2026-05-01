"""Validates the C3 OperatorSpec reference example (RMSNorm V0).

Confirms the checked-in RMSNorm operator metadata conforms to the
OperatorSpec contract, satisfying the S01 acceptance requirement that
existing RMSNorm metadata is migrated to the new schema.
"""

from __future__ import annotations

import json
import os

import pytest

from hqsb.core.contracts import OperatorSpec

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_SPEC_PATH = os.path.join(
    _REPO_ROOT, "configs", "operators", "rmsnorm_v0.json"
)


@pytest.mark.unit
class TestRmsnormOperatorSpec:
    def test_file_exists_and_parses(self):
        with open(_SPEC_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        spec = OperatorSpec.model_validate(data)
        assert spec.name == "rmsnorm"
        assert spec.schema_version == "1.0.0"
        assert spec.implementation == "v0_shared"

    def test_tolerance_matches_baseline_gate(self):
        with open(_SPEC_PATH, encoding="utf-8") as fh:
            spec = OperatorSpec.model_validate(json.load(fh))
        # RMSNorm baseline correctness gate is max_abs_error <= 5e-4.
        assert spec.tolerance == 0.0005
        assert spec.deterministic is True

    def test_has_input_and_output_tensors(self):
        with open(_SPEC_PATH, encoding="utf-8") as fh:
            spec = OperatorSpec.model_validate(json.load(fh))
        assert len(spec.inputs) == 2
        assert len(spec.outputs) == 1
        assert spec.outputs[0].dtype == "float32"
