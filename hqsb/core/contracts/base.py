"""Shared base types for versioned HQSB contracts.

Every public contract is a Pydantic model that:

* declares its own ``schema_version`` (defaulting to the contract's
  ``SCHEMA_VERSION`` class constant),
* rejects unknown fields by default (``extra="forbid"``), so typos and
  unsupported keys fail fast instead of being silently ignored,
* carries a machine-readable description of each field for schema
  generation and documentation.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from hqsb.core.errors import (
    SchemaMigrationRequiredError,
    SchemaVersionError,
    UnsupportedSchemaVersionError,
)
from hqsb.core.schema.versioning import SchemaVersion


class VersionedModel(BaseModel):
    """Base class for all versioned HQSB contract documents.

    Construction is itself a version gate (E01-01): a payload that
    *explicitly* declares a ``schema_version`` different from the class's
    ``SCHEMA_VERSION`` is rejected at parse time.  A future version is never
    guessed; an older version must first be migrated (see
    :func:`hqsb.core.schema.versioning.migrate_document`) and is otherwise
    refused with a ``requires_migration``-style error.  Omitting the field is
    legal: the object then simply carries the current class version.
    """

    #: The schema version this contract class currently emits/validates.
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"

    schema_version: str = "1.0.0"

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    def model_post_init(self, __context) -> None:  # type: ignore[override]
        """Default ``schema_version`` to the class version when omitted.

        Uses ``model_fields_set`` so that an *explicit* ``schema_version``
        value (even one equal to the default) is never overwritten.

        Raises:
            SchemaVersionError: If ``schema_version`` was explicitly set to
                anything other than :attr:`SCHEMA_VERSION`.  Versions newer
                than the implementation are refused (forward-incompatible,
                :class:`UnsupportedSchemaVersionError`); older versions are
                refused with a migration hint unless a migration has already
                lifted the payload to the current version
                (:class:`SchemaMigrationRequiredError`).
        """
        if "schema_version" not in self.model_fields_set:
            self.schema_version = self.SCHEMA_VERSION
            return
        explicit = self.schema_version
        if explicit == self.SCHEMA_VERSION:
            return
        try:
            received = SchemaVersion.parse(explicit)
        except SchemaVersionError:
            received = None
        if received is not None and received > SchemaVersion.parse(self.SCHEMA_VERSION):
            raise UnsupportedSchemaVersionError(
                f"{type(self).__name__}: schema_version {explicit!r} is newer "
                f"than the supported version {self.SCHEMA_VERSION}; refusing to "
                f"interpret a future schema (field path: schema_version)"
            )
        raise SchemaMigrationRequiredError(
            f"{type(self).__name__}: schema_version {explicit!r} is not the "
            f"current version {self.SCHEMA_VERSION}; migrate to the current "
            f"schema first or the payload is refused (field path: schema_version)"
        )


__all__ = ["VersionedModel"]
