"""Unit tests for legacy golden/result migration to the current schema.

The assertions follow the E01-07 migration contract:

* no per-sample data is fabricated from aggregate statistics (ITL summary is
  preserved, ``itl_ms`` stays empty);
* ``model.config_hash`` is not promoted to a C1 ``model_artifact_hash``;
* a ``deterministic`` repetition flag is not reinterpreted as a reference
  correctness gate;
* migration provenance / loss rows are attached to the C6 summary.
"""

from __future__ import annotations

import json
import os

import pytest

from hqsb.core.errors import (
    SchemaError,
    UnsupportedSchemaVersionError,
)
from hqsb.core.schema.migrate import (
    MIGRATOR_VERSION,
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
        # config_hash is NOT promoted to a C1 artifact hash.
        assert result.model_artifact_hash is None
        assert (
            result.artifact_links["model_config_sha256"]
            == document["model"]["config_hash"]
        )
        assert result.summary["migration"]["migrator_version"] == MIGRATOR_VERSION

    def test_rejects_non_golden(self):
        with pytest.raises(SchemaError):
            migrate_legacy_golden({"schema_version": "1.0.0"})

    def test_golden_never_fabricates_timing(self):
        with open(_GOLDEN_PATH, encoding="utf-8") as fh:
            document = json.load(fh)
        result = migrate_legacy_golden(document)
        assert result.raw_samples[0]["itl_ms"] == []
        classes = {
            row["loss_class"] for row in result.summary["migration_losses"]
        }
        assert "insufficient" in classes


@pytest.mark.unit
class TestMigrateLegacyResult:
    def test_migrates_result(self):
        result = migrate_legacy_result(_legacy_result())
        assert len(result.raw_samples) == 1
        assert result.workload.input_tokens == 128
        # ITL summary is preserved; per-sample itl_ms is NOT fabricated.
        assert result.raw_samples[0]["itl_ms"] == []
        assert result.summary["itl_summary"] == [{"count": 2, "mean_ms": 98.0}]
        # deterministic flag is preserved, not promoted to a correctness gate.
        assert result.summary["deterministic"] is True
        assert result.correctness is None
        # per-rep measured timings are kept verbatim.
        assert result.raw_samples[0]["prefill_forward_ms"] == 117.0
        assert result.raw_samples[0]["first_token_selection_ms"] == 0.2

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

    def test_future_version_is_rejected(self):
        document = {
            "schema_version": "2.0.0",
            "timestamp": 1.0,
            "input_tokens": 8,
            "output_tokens": 4,
            "input_token_ids": [1, 2],
            "generated_tokens": [3],
            "first_token": {"token_id": 3},
            "model": {"id": "m"},
        }
        with pytest.raises(UnsupportedSchemaVersionError):
            migrate_any(document)

    def test_ambiguous_family_is_rejected(self):
        document = {
            "schema_version": "1.0.0",
            "timestamp": 1.0,
            "input_tokens": 8,
            "output_tokens": 4,
            "input_token_ids": [1, 2],
            "first_token": {"token_id": 3},
            "repetitions": [],
        }
        with pytest.raises(SchemaError):
            migrate_any(document)


@pytest.mark.unit
class TestMigrateCurrentC6Noop:
    def _current_c6(self) -> dict:
        from hqsb.core.contracts.result import (
            BenchmarkResult,
            EnvironmentInfo,
        )
        from hqsb.core.ids import new_run_id

        base = BenchmarkResult(
            run_id=new_run_id(),
            timestamp=1786500000.0,
            environment=EnvironmentInfo(platform="test"),
            raw_samples=[{"input_tokens": 128, "output_tokens": 32}],
        )
        return base.model_dump(mode="json")

    def test_current_c6_is_noop(self):
        payload = self._current_c6()
        migrated = migrate_any(payload)
        assert migrated.run_id == payload["run_id"]
        # No nested migration metadata is added for an already-current object.
        assert "migration" not in migrated.summary
        assert len(migrated.raw_samples) == len(payload["raw_samples"])

    def test_future_c6_version_is_rejected(self):
        payload = self._current_c6()
        payload["schema_version"] = "2.0.0"
        with pytest.raises(UnsupportedSchemaVersionError):
            migrate_any(payload)
