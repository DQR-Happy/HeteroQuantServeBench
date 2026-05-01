"""Unit tests for legacy golden/result migration to the current schema."""

from __future__ import annotations

import json
import os

import pytest

from hqsb.core.errors import SchemaError
from hqsb.core.schema.migrate import (
    migrate_any,
    migrate_legacy_golden,
    migrate_legacy_result,
)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

_GOLDEN_PATH = os.path.join(
    _REPO_ROOT, "benchmarks", "workloads", "golden", "isl128_osl32.json"
)


def _legacy_result() -> dict:
    return {
        "schema_version": "1.0.0",
        "timestamp": 1786500000.0,
        "hardware": {"device": "Orin", "compute_capability": [8, 7]},
        "software": {"torch": "2.5.0", "cuda": "12.6"},
        "model": {"id": "Qwen/Qwen3-1.7B", "backend": "modelscope-transformers"},
        "workload": {"input_tokens": 128, "output_tokens": 32},
        "deterministic": True,
        "generated_token_sha256": "deadbeef",
        "repetitions": [
            {
                "input_tokens": 128,
                "output_tokens": 32,
                "prefill_forward_ms": 117.0,
                "first_token_selection_ms": 0.2,
                "generated_token_ids": [1, 2, 3],
                "itl": {"count": 2, "mean_ms": 98.0},
                "peak_cuda_allocated_mb": 3300.0,
                "peak_cuda_reserved_mb": 3900.0,
            }
        ],
    }


@pytest.mark.unit
class TestMigrateLegacyGolden:
    def test_migrates_real_golden_file(self):
        with open(_GOLDEN_PATH, encoding="utf-8") as fh:
            document = json.load(fh)
        result = migrate_legacy_golden(document)
        assert result.workload.input_tokens == 128
        assert result.workload.output_tokens == 32
        assert "first_token" in result.summary
        assert result.artifact_links["legacy_kind"] == "golden"
        assert result.raw_samples[0]["generated_token_ids"]

    def test_rejects_non_golden(self):
        with pytest.raises(SchemaError):
            migrate_legacy_golden({"schema_version": "1.0.0"})


@pytest.mark.unit
class TestMigrateLegacyResult:
    def test_migrates_result(self):
        result = migrate_legacy_result(_legacy_result())
        assert result.correctness.passed is True
        assert result.correctness.method == "determinism"
        assert len(result.raw_samples) == 1
        assert result.workload.input_tokens == 128
        # ITL reconstructed from summary stats: count * mean
        assert result.raw_samples[0]["itl_ms"] == [98.0, 98.0]

    def test_rejects_non_result(self):
        with pytest.raises(SchemaError):
            migrate_legacy_result({"schema_version": "1.0.0"})


@pytest.mark.unit
class TestMigrateAny:
    def test_auto_detects_golden(self):
        document = {
            "schema_version": "1.0.0",
            "timestamp": 1.0,
            "input_tokens": 8,
            "output_tokens": 4,
            "input_token_ids": [1, 2],
            "generated_tokens": [3],
            "first_token": {"token_id": 3},
            "model": {"id": "m"},
        }
        result = migrate_any(document)
        assert result.artifact_links["legacy_kind"] == "golden"

    def test_auto_detects_result(self):
        result = migrate_any(_legacy_result())
        assert result.artifact_links["legacy_kind"] == "result"

    def test_unknown_shape_raises(self):
        with pytest.raises(SchemaError):
            migrate_any({"schema_version": "1.0.0", "foo": "bar"})
