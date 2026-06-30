"""Identity, hashing and canonicalisation primitives for S12 (E12-10 §4–§5).

Three different hashes answer three different questions, and S12 must never
confuse them:

``byte_hash``
    ``raw_sha256``: did the *bytes* of a file change?  Used for raw samples,
    binaries and source snapshots.
``canonical_hash``
    A hash over structured data after a *versioned* canonicalisation (stable
    field order, normalised units/types, volatile fields removed).  Two
    payloads with the same canonical hash have the same semantics under that
    policy — which never replaces a correctness check.
``aggregate_root``
    ``H(schema_version, ordered [(role, path, byte_hash, size)], parents)``
    answers "is this *set* of files the same?" quickly; locating a tampered
    field still requires a per-file/field diff, because a single root hash
    cannot be inverted.

Logical URIs (``hqsb://S12/<campaign>/<entity_type>/<entity_id>``) are the
permanent identity of an entity; absolute paths are storage locations and may
move without changing identity.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.core.fingerprint import canonical_json as _core_canonical_json

SCHEMA_VERSION = "1.0.0"

#: Canonicalisation policy version.  Changing the policy invalidates every
#: canonical hash computed with the previous version, so it is part of the
#: payload of every canonical hash.
CANONICALIZATION_VERSION = "s12-canon-1.0.0"

#: Fields that carry no semantics and are therefore dropped before hashing.
VOLATILE_FIELDS: Tuple[str, ...] = (
    "collected_at",
    "created_at",
    "written_at",
    "retrieved_at",
    "started_at",
    "ended_at",
    "host",
    "hostname",
    "local_path",
    "storage_location",
    "storage_locations",
    "abs_path",
    "cwd",
    "pid",
    "duration_s",
    "log",
)

URI_SCHEME = "hqsb"

#: Hash algorithms allowed by the policy (base64/hex excluded: hex only).
HASH_ALGORITHMS: Tuple[str, ...] = ("sha256",)


def canonical_json(payload: Any) -> str:
    """Stable JSON serialisation (sorted keys, no insignificant whitespace)."""
    return _core_canonical_json(payload)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def byte_hash_file(path: str, *, chunk_size: int = 1 << 20) -> str:
    """Stream ``raw_sha256`` of a file (never loads the whole file)."""
    if not os.path.isfile(path):
        raise ConfigError(f"cannot hash missing file: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CanonicalPolicy:
    """Versioned canonicalisation policy (E12-10 step 3)."""

    policy_id: str = "s12_default"
    version: str = CANONICALIZATION_VERSION
    volatile_fields: Tuple[str, ...] = VOLATILE_FIELDS
    drop_none: bool = True
    unit_normalisation: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "volatile_fields": list(self.volatile_fields),
            "drop_none": self.drop_none,
            "unit_normalisation": self.unit_normalisation,
        }


DEFAULT_POLICY = CanonicalPolicy()


def _canonicalize(value: Any, policy: CanonicalPolicy, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for name in sorted(value):
            if name in policy.volatile_fields:
                continue
            child = value[name]
            if policy.drop_none and child is None:
                continue
            out[name] = _canonicalize(child, policy, key=name)
        return out
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item, policy, key=key) for item in value]
    if isinstance(value, float):
        # NaN/Inf are not valid JSON and must never silently become a number.
        if value != value or value in (float("inf"), float("-inf")):
            raise ConfigError(f"non-finite float in canonical payload at {key!r}: {value!r}")
        # Normalise -0.0 and float noise representation without changing value.
        return round(value, 12) if policy.unit_normalisation else value
    if isinstance(value, (set, frozenset)):
        return sorted(_canonicalize(item, policy, key=key) for item in value)
    return value


def canonicalize_payload(payload: Any, policy: CanonicalPolicy = DEFAULT_POLICY) -> Any:
    """Return the canonical form used for ``canonical_hash``."""
    return _canonicalize(payload, policy)


def canonical_hash(payload: Any, policy: CanonicalPolicy = DEFAULT_POLICY) -> str:
    """Semantic hash: canonical form + policy version + schema version."""
    canonical = canonicalize_payload(payload, policy)
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "canonicalization_version": policy.version,
        "payload": canonical,
    }
    return sha256_text(canonical_json(envelope))


def stable_id(prefix: str, payload: Mapping[str, Any], *, length: int = 16) -> str:
    """Deterministic identifier from canonical fields (never a display name)."""
    if not prefix:
        raise ConfigError("stable_id requires a non-empty prefix")
    digest = canonical_hash(payload)[:length]
    return f"{prefix}_{digest}"


def logical_uri(campaign_id: str, entity_type: str, entity_id: str) -> str:
    """``hqsb://S12/<campaign>/<entity_type>/<entity_id>`` (no local paths)."""
    for name, value in (("campaign_id", campaign_id), ("entity_type", entity_type), ("entity_id", entity_id)):
        if not value:
            raise ConfigError(f"logical_uri requires {name}")
    return f"{URI_SCHEME}://S12/{campaign_id}/{entity_type}/{entity_id}"


def parse_logical_uri(uri: str) -> Dict[str, str]:
    prefix = f"{URI_SCHEME}://S12/"
    if not uri.startswith(prefix):
        raise ConfigError(f"not an S12 logical URI: {uri!r}")
    parts = uri[len(prefix) :].split("/")
    if len(parts) != 3:
        raise ConfigError(f"logical URI must have 3 path segments: {uri!r}")
    return {"campaign_id": parts[0], "entity_type": parts[1], "entity_id": parts[2]}


@dataclass
class EntityRef:
    """One evidence entity (E12-10 §13 ``EvidenceEntity`` minimum fields)."""

    entity_id: str
    entity_type: str
    logical_uri: str = ""
    byte_hash: str = ""
    canonical_hash: str = ""
    aggregate_root: str = ""
    size: Optional[int] = None
    mime: str = ""
    schema_version: str = SCHEMA_VERSION
    storage_locations: Tuple[str, ...] = ()
    created_at: str = ""
    status: str = "REGISTERED"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.entity_id:
            problems.append("entity_id is required")
        if not self.entity_type:
            problems.append("entity_type is required")
        for name in ("byte_hash", "canonical_hash", "aggregate_root"):
            value = getattr(self, name)
            if value and (len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)):
                problems.append(f"{name} is not a sha256 hex digest: {value!r}")
        if self.size is not None and self.size < 0:
            problems.append("size must not be negative")
        if self.logical_uri:
            try:
                parsed = parse_logical_uri(self.logical_uri)
            except ConfigError as exc:
                problems.append(str(exc))
            else:
                if parsed["entity_id"] != self.entity_id:
                    problems.append("logical_uri entity_id does not match entity_id")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "logical_uri": self.logical_uri,
            "byte_hash": self.byte_hash,
            "canonical_hash": self.canonical_hash,
            "aggregate_root": self.aggregate_root,
            "size": self.size,
            "mime": self.mime,
            "schema_version": self.schema_version,
            "storage_locations": list(self.storage_locations),
            "created_at": self.created_at,
            "status": self.status,
        }


def aggregate_root(
    schema_version: str,
    entries: Sequence[Sequence[Any]],
    *,
    parents: Sequence[str] = (),
) -> str:
    """``H(schema_version, ordered [(role, path, byte_hash, size)], parents)``.

    ``entries`` are ``(logical_role, relative_path, byte_hash, size)`` rows; they
    are sorted by ``(role, path)`` so directory listing order cannot change the
    root.
    """
    rows: List[Tuple[str, str, str, int]] = []
    for row in entries:
        if len(row) != 4:
            raise ConfigError(f"aggregate_root entries need 4 fields, got {row!r}")
        role, path, digest, size = row
        if not str(digest):
            raise ConfigError(f"aggregate_root entry without a byte hash: {row!r}")
        rows.append((str(role), str(path), str(digest), int(size)))
    envelope = {
        "schema_version": schema_version,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "entries": sorted(rows),
        "parents": sorted({parent for parent in parents if parent}),
    }
    return sha256_text(canonical_json(envelope))


def manifest_aggregate_root(rows: Iterable[Mapping[str, Any]], *, schema_version: str = SCHEMA_VERSION) -> str:
    """Aggregate root from manifest rows carrying role/path/hash/size keys."""
    entries: List[Tuple[str, str, str, int]] = []
    for row in rows:
        entries.append(
            (
                str(row.get("role", "")),
                str(row.get("path", "")),
                str(row.get("byte_hash", row.get("sha256", ""))),
                int(row.get("size", 0)),
            )
        )
    return aggregate_root(schema_version, entries)


@dataclass
class IdentityChain:
    """Ordered identity chain used to check that a claim has all its parents."""

    stages: Tuple[str, ...]
    values: Dict[str, str] = field(default_factory=dict)

    def add(self, stage: str, value: str) -> None:
        if not value:
            raise ConfigError(f"identity chain stage {stage!r} requires a non-empty value")
        self.values[stage] = value

    def missing(self) -> List[str]:
        return [stage for stage in self.stages if not self.values.get(stage)]

    def complete(self) -> bool:
        return not self.missing()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stages": list(self.stages),
            "values": dict(sorted(self.values.items())),
            "missing": self.missing(),
            "complete": self.complete(),
        }


def verify_byte_hash(path: str, expected: str) -> Tuple[bool, str]:
    """Return ``(ok, actual_hash)``; never raises for a mismatch."""
    actual = byte_hash_file(path)
    return actual == expected, actual


def hash_directory(root: str, *, roles: Optional[Mapping[str, str]] = None) -> List[Dict[str, Any]]:
    """Hash every file under ``root`` in stable relative-path order."""
    rows: List[Dict[str, Any]] = []
    if not os.path.isdir(root):
        raise ConfigError(f"not a directory: {root}")
    for base, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(base, name)
            relative = os.path.relpath(path, root)
            rows.append(
                {
                    "role": (roles or {}).get(relative, "artifact"),
                    "path": relative,
                    "byte_hash": byte_hash_file(path),
                    "size": os.path.getsize(path),
                }
            )
    return rows
