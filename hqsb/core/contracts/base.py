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


class VersionedModel(BaseModel):
    """Base class for all versioned HQSB contract documents."""

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
        """
        if "schema_version" not in self.model_fields_set:
            self.schema_version = self.SCHEMA_VERSION


__all__ = ["VersionedModel"]
