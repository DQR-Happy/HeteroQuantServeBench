"""Unit tests for schema versioning and document migration.

Covers version parsing/ordering, migration chaining, and forward-version
rejection.
"""

from __future__ import annotations

import pytest

from hqsb.core.errors import SchemaVersionError
from hqsb.core.schema import SchemaVersion, migrate_document


@pytest.mark.unit
class TestSchemaVersion:
    def test_parse_valid(self):
        v = SchemaVersion.parse("1.2.3")
        assert (v.major, v.minor, v.patch) == (1, 2, 3)

    @pytest.mark.parametrize("bad", ["1.2", "1.2.3.4", "abc", "", "1.2.x"])
    def test_parse_invalid(self, bad):
        with pytest.raises(SchemaVersionError):
            SchemaVersion.parse(bad)

    def test_ordering(self):
        assert SchemaVersion.parse("1.0.0") < SchemaVersion.parse("1.0.1")
        assert SchemaVersion.parse("1.0.1") < SchemaVersion.parse("1.1.0")
        assert SchemaVersion.parse("1.1.0") < SchemaVersion.parse("2.0.0")
        assert SchemaVersion.parse("1.0.0") == SchemaVersion.parse("1.0.0")

    def test_str_roundtrip(self):
        assert str(SchemaVersion.parse("1.0.0")) == "1.0.0"


@pytest.mark.unit
class TestMigrateDocument:
    def test_current_version_passthrough(self):
        doc = {"schema_version": "1.0.0", "x": 1}
        result = migrate_document(doc, SchemaVersion.parse("1.0.0"), {})
        assert result == doc

    def test_missing_version_rejected(self):
        with pytest.raises(SchemaVersionError):
            migrate_document({}, SchemaVersion.parse("1.0.0"), {})

    def test_future_version_rejected(self):
        doc = {"schema_version": "2.0.0"}
        with pytest.raises(SchemaVersionError):
            migrate_document(doc, SchemaVersion.parse("1.0.0"), {})

    def test_single_step_migration(self):
        def _rename(doc):
            result = dict(doc)
            result["new_name"] = result.pop("old_name")
            return result

        doc = {"schema_version": "1.0.0", "old_name": "value"}
        migrations = {SchemaVersion.parse("1.0.0"): _rename}
        result = migrate_document(doc, SchemaVersion.parse("1.0.1"), migrations)
        assert result["schema_version"] == "1.0.1"
        assert result["new_name"] == "value"
        assert "old_name" not in result

    def test_multi_step_migration_chains(self):
        def _step1(doc):
            doc = dict(doc)
            doc["step1"] = True
            return doc

        def _step2(doc):
            doc = dict(doc)
            doc["step2"] = True
            return doc

        doc = {"schema_version": "1.0.0"}
        migrations = {
            SchemaVersion.parse("1.0.0"): _step1,
            SchemaVersion.parse("1.0.1"): _step2,
        }
        result = migrate_document(doc, SchemaVersion.parse("1.0.2"), migrations)
        assert result["schema_version"] == "1.0.2"
        assert result["step1"] is True
        assert result["step2"] is True

    def test_missing_migration_step_rejected(self):
        doc = {"schema_version": "1.0.0"}
        with pytest.raises(SchemaVersionError):
            migrate_document(doc, SchemaVersion.parse("1.0.1"), {})
