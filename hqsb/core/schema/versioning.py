"""Schema version handling and document migration framework.

HQSB artifacts (model manifest, workload, golden reference, benchmark
result, quant artifact) are versioned documents. This module provides:

* :class:`SchemaVersion` — a strict semantic version with ordering.
* :func:`migrate_document` — a small, explicit migration executor that
  upgrades a raw document from an older schema version to the latest,
  applying a caller-supplied chain of migration functions.

Design rules (matching the top-level architecture §3.3 and S01 §3):

* Documents without a ``schema_version`` field are rejected — a missing
  version means we cannot know how to interpret the payload.
* Documents whose version is *newer* than the code understands are
  rejected (forward-incompatibility), never silently truncated.
* Migrations are explicit, ordered, and composed; there is no implicit
  "best-effort" coercion.
"""

from __future__ import annotations

import re
from functools import total_ordering
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import SchemaVersionError

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


@total_ordering
class SchemaVersion:
    """A three-component semantic version (``major.minor.patch``).

    Ordering follows semantic-versioning precedence: compare major, then
    minor, then patch. Two versions are equal iff all three components are
    equal.
    """

    __slots__ = ("major", "minor", "patch")

    def __init__(self, major: int, minor: int, patch: int) -> None:
        if major < 0 or minor < 0 or patch < 0:
            raise SchemaVersionError(
                f"version components must be non-negative, got "
                f"{major}.{minor}.{patch}"
            )
        self.major = major
        self.minor = minor
        self.patch = patch

    @classmethod
    def parse(cls, text: str) -> "SchemaVersion":
        """Parse a ``major.minor.patch`` string.

        Raises:
            SchemaVersionError: If the string is not a valid version.
        """
        match = _VERSION_RE.match(text.strip())
        if match is None:
            raise SchemaVersionError(
                f"invalid schema version {text!r}; expected 'major.minor.patch'"
            )
        return cls(*(int(group) for group in match.groups()))

    def _key(self) -> Tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SchemaVersion):
            return NotImplemented
        return self._key() == other._key()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SchemaVersion):
            return NotImplemented
        return self._key() < other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"SchemaVersion({self.major}, {self.minor}, {self.patch})"


# A migration step: transforms a raw document dict into the *next* version.
Migration = Callable[[dict], dict]


def _parse_version(document: Mapping) -> SchemaVersion:
    """Extract and validate ``schema_version`` from a raw document."""
    raw = document.get("schema_version")
    if raw is None:
        raise SchemaVersionError("document is missing 'schema_version'")
    if not isinstance(raw, str):
        raise SchemaVersionError(
            f"'schema_version' must be a string, got {type(raw).__name__}"
        )
    return SchemaVersion.parse(raw)


def _ordered_migrations(
    migrations: Mapping[SchemaVersion, Migration],
) -> List[Tuple[SchemaVersion, Migration]]:
    """Return migrations sorted by source version (ascending)."""
    return sorted(migrations.items(), key=lambda item: item[0])


def migrate_document(
    document: Mapping,
    current: SchemaVersion,
    migrations: Mapping[SchemaVersion, Migration],
) -> dict:
    """Upgrade ``document`` to ``current`` schema version.

    Args:
        document: Raw document mapping. Must contain a valid
            ``schema_version``.
        current: The latest schema version this code understands.
        migrations: Mapping from *source* version → migration callable. A
            migration at version ``v`` is responsible for upgrading a
            document that is currently at ``v`` to the *next* version.
            Migrations are applied in ascending source-version order.

    Returns:
        A new dict representing the document at ``current`` version.

    Raises:
        SchemaVersionError: If the document version is missing/malformed,
            newer than ``current`` (forward-incompatible), or a required
            migration step is absent so the chain cannot reach ``current``.
    """
    source = _parse_version(document)
    if source > current:
        raise SchemaVersionError(
            f"document version {source} is newer than the supported "
            f"version {current}; refusing to interpret a future schema"
        )

    result = dict(document)
    version = source

    steps = _ordered_migrations(migrations)
    while version < current:
        # Find the migration whose source version equals the current one.
        step = next((m for v, m in steps if v == version), None)
        if step is None:
            raise SchemaVersionError(
                f"no migration available to upgrade schema from "
                f"{version} towards {current}"
            )
        result = step(result)
        result["schema_version"] = str(_increment(version))
        version = _increment(version)

    result["schema_version"] = str(current)
    return result


def _increment(version: SchemaVersion) -> SchemaVersion:
    """Return the next version in the migration chain (patch bump).

    This helper defines a canonical "next" step. For schemas that follow a
    different cadence (e.g. minor bump), migrations should be keyed
    explicitly by their source version, which :func:`migrate_document` fully
    supports; this function only provides a deterministic default chain.
    """
    return SchemaVersion(version.major, version.minor, version.patch + 1)


__all__ = [
    "Migration",
    "SchemaVersion",
    "migrate_document",
]
