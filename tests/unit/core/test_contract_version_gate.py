"""Unit tests for the C1–C7 schema version gate (E01-01).

The version gate lives in :class:`hqsb.core.contracts.base.VersionedModel`
and turns "a versioned payload is silently coerced" into a stable,
machine-classifiable rejection:

* a future version raises :class:`UnsupportedSchemaVersionError`;
* an older version raises :class:`SchemaMigrationRequiredError`;
* omitting ``schema_version`` is legal and defaults to the class version.

These tests guard the contract boundary independently of the E01-01 runner
so a regression is caught by CI (E01-01 §9 PASS criterion).
"""

from __future__ import annotations

import pytest

from hqsb.core.contracts import (
    BackendCapability,
    BenchmarkResult,
    EnvironmentInfo,
    ModelArtifact,
    OperatorSpec,
    QuantArtifact,
    TraceEvent,
    WorkloadSpec,
)
from hqsb.core.contracts.registry import load_contract_registry, registry_checks
from hqsb.core.errors import (
    SchemaMigrationRequiredError,
    UnsupportedSchemaVersionError,
)

#: Every public C1–C7 contract document class.
_CONTRACTS = [
    ModelArtifact,
    WorkloadSpec,
    OperatorSpec,
    BackendCapability,
    QuantArtifact,
    BenchmarkResult,
    TraceEvent,
]


@pytest.mark.unit
class TestVersionGate:
    @pytest.mark.parametrize("model_cls", _CONTRACTS)
    def test_future_version_rejected(self, model_cls):
        payload = _minimal_payload(model_cls)
        payload["schema_version"] = "2.0.0"
        with pytest.raises(UnsupportedSchemaVersionError):
            model_cls.model_validate(payload)

    @pytest.mark.parametrize("model_cls", _CONTRACTS)
    def test_old_version_requires_migration(self, model_cls):
        payload = _minimal_payload(model_cls)
        payload["schema_version"] = "0.9.0"
        with pytest.raises(SchemaMigrationRequiredError):
            model_cls.model_validate(payload)

    @pytest.mark.parametrize("model_cls", _CONTRACTS)
    def test_omitted_version_defaults_to_current(self, model_cls):
        payload = _minimal_payload(model_cls)
        payload.pop("schema_version", None)
        obj = model_cls.model_validate(payload)
        assert obj.schema_version == model_cls.SCHEMA_VERSION == "1.0.0"


@pytest.mark.unit
class TestContractRegistry:
    def test_registry_has_all_seven_and_is_complete(self):
        entries = load_contract_registry()
        checks = registry_checks(entries)
        assert len(entries) == 7
        assert checks == {
            "all_seven_registered": True,
            "no_duplicate_names": True,
            "every_contract_has_version": True,
            "every_contract_has_parser_serializer": True,
            "every_contract_has_canonical_fixture": True,
        }

    def test_every_contract_has_stable_schema_digest(self):
        for entry in load_contract_registry():
            assert len(entry.schema_sha256) == 64
            assert entry.unknown_field_policy == "forbid"
            assert entry.legacy_policy == "requires_migration"


def _minimal_payload(model_cls):
    """Return a minimal valid payload for each contract class."""
    if model_cls is ModelArtifact:
        return {
            "schema_version": "1.0.0",
            "model_id": "m",
            "source": "local",
            "architecture": "A",
            "dtype": "float16",
        }
    if model_cls is WorkloadSpec:
        return {
            "schema_version": "1.0.0",
            "name": "short",
            "input_tokens": 8,
            "output_tokens": 4,
        }
    if model_cls is OperatorSpec:
        return {
            "schema_version": "1.0.0",
            "name": "op",
            "semantic_version": "1.0.0",
            "inputs": [],
            "outputs": [],
            "implementation": "v0",
        }
    if model_cls is BackendCapability:
        return {"schema_version": "1.0.0", "name": "dummy"}
    if model_cls is QuantArtifact:
        return {
            "schema_version": "1.0.0",
            "algorithm": "rtn",
            "bits": 4,
            "granularity": "per-channel",
        }
    if model_cls is BenchmarkResult:
        return {
            "schema_version": "1.0.0",
            "run_id": "run_1",
            "timestamp": 1.0,
            "environment": EnvironmentInfo().model_dump(mode="json"),
        }
    if model_cls is TraceEvent:
        return {
            "schema_version": "1.0.0",
            "event_type": "prefill",
            "timestamp_ns": 1,
            "trace_id": "t",
            "span_id": "s",
        }
    raise AssertionError(model_cls)
