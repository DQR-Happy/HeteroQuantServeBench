"""Canonical identity for every S15 artifact (release / open-source / evidence layer).

S15 has to answer lineage questions that a path name cannot answer
(``details/S15/README.md`` §7, §8, §9):

* which *release candidate* does a claim belong to, and is that candidate still
  the frozen one (``ReleaseCandidateSnapshot``);
* which raw entity does a public number resolve to, and is the digest of that
  entity the digest the claim recorded (evidence URI *and* hash are both
  required — ``E15-01`` §3.2);
* which figure point was derived from which normalized row and raw sample
  (``PointLineageRecord``, ``E15-06``);
* which public rendering of a claim (README / report / demo / resume) carries the
  same fact fields (``E15-01`` §3.5).

Everything here is *content addressed*: a value is serialised canonically
(sorted keys, no NaN, explicit units), hashed with SHA-256 and wrapped in a typed
id of the form ``<kind>::sha256:<hex>``.  Two payloads that differ only in
whitespace or dict ordering share an id; two payloads that differ in any recorded
field never do.

Two S15-specific rules are enforced here rather than left to review:

* **stable claim ids** — a claim id must survive wording edits (``E15-01`` step 13)
  and change when the *fact object* changes (step 37/45).  :func:`stable_claim_id`
  therefore hashes the fact signature, never the rendering text;
* **no machine-local URI** — an evidence URI pointing at an author's home
  directory makes a claim non-auditable for everyone else (``E15-01`` step 22,
  ``E15-04`` step 35, ``E15-05`` step 32).  :func:`assert_public_uri` refuses
  ``file://``, absolute home prefixes and private-network hosts, because a claim
  that can only be resolved on the author's laptop is an orphan by construction.

The module is pure stdlib and CPU-only: no framework import, no device access.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Schema-version namespace for every S15 payload.
SCHEMA_PREFIX = "hqsb.s15"

#: Artifact kinds that may be content addressed by :func:`artifact_id`.
ARTIFACT_KINDS: Tuple[str, ...] = (
    "candidate",
    "claim",
    "claim-ledger",
    "evidence-bundle",
    "evidence-graph",
    "public-surface",
    "figure",
    "figure-spec",
    "plot-point",
    "release-artifact",
    "artifact-manifest",
    "provenance",
    "sbom",
    "license-report",
    "vulnerability-report",
    "notice",
    "scan-report",
    "documentation-site",
    "quickstart-session",
    "hero-replay",
    "demo-asset",
    "demo-video",
    "narrative",
    "faq",
    "resume-bullet",
    "contribution-record",
    "reviewer-session",
    "reproduction-record",
    "upstream-contribution",
    "interview-session",
    "first-impression-study",
    "audit",
    "action-log",
    "acceptance",
    "run",
    "environment",
    "config",
    "manifest",
)

#: Evidence levels in their partial order (``E15-01`` §3.3, §11).
#: Higher levels must cite the lower evidence they depend on; skipping a level is
#: the "overclaim" failure the audit exists to catch.
EVIDENCE_LEVELS: Tuple[str, ...] = (
    "PLANNED",
    "SOURCE",
    "TEST",
    "RUNTIME",
    "MODEL",
    "SERVICE",
    "PORTABLE",
)

#: Claim lifecycle states (``details/S15/README.md`` §8.2).
CLAIM_STATES: Tuple[str, ...] = ("DRAFT", "VERIFIED", "STALE", "REJECTED", "RETRACTED")

#: Terminological discipline of ``details/S15/README.md`` §6 — "I ran it again"
#: is not "a third party reproduced it".
REPRODUCTION_TERMS: Tuple[str, ...] = (
    "repeatability",
    "portability_replay",
    "reproducibility",
    "replication",
    "artifact_availability",
    "artifact_functionality",
    "artifact_reusability",
)

#: Public channels a claim can be rendered into (``E15-01`` §11).
PUBLIC_CHANNELS: Tuple[str, ...] = (
    "roadmap",
    "design_document",
    "test_status",
    "readme_results",
    "service_dashboard",
    "cross_hardware_guide",
    "demo",
    "resume_bullet",
)

#: Minimum evidence level each channel is allowed to carry (``E15-01`` §11).
CHANNEL_MINIMUM_LEVEL: Mapping[str, str] = {
    "roadmap": "PLANNED",
    "design_document": "SOURCE",
    "test_status": "TEST",
    "readme_results": "RUNTIME",
    "service_dashboard": "SERVICE",
    "cross_hardware_guide": "PORTABLE",
    "demo": "PLANNED",  # must match the original claim; checked per-claim, not per-channel
    "resume_bullet": "PLANNED",  # same: bound to the claim's own level
}

#: Channels whose minimum is *exactly* the claim's own level (E15-01 §11 rows
#: "Demo | 与原 claim 相同" and "简历 bullet | 与所述层级相同").
CHANNEL_MIRRORS_CLAIM: Tuple[str, ...] = ("demo", "resume_bullet")

#: URI schemes and host patterns that are never publicly resolvable.
_LOCAL_URI_PREFIXES: Tuple[str, ...] = ("file://", "ftp://localhost")
_LOCAL_PATH_PREFIXES: Tuple[str, ...] = ("/root/", "/home/", "/Users/", "/mnt/", "/data/", "C:\\")
_PRIVATE_HOST_PATTERNS: Tuple[str, ...] = (
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "10.",
    "192.168.",
    "172.16.",
    ".internal",
    ".local:",
)

_HEX = frozenset("0123456789abcdef")

#: Characters allowed in a single path segment of a run id / candidate id.
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def canonical_json(value: Any) -> str:
    """Serialise ``value`` deterministically (sorted keys, compact separators).

    ``float('nan')``/``inf`` are refused instead of emitted as non-JSON tokens: a
    NaN that reaches a digest silently makes two different payloads collide
    (the same reasoning as ``S14`` ``identity.canonical_json``; here it also
    protects the claim ledger, where a NaN effect would poison every digest
    computed above it).
    """
    _reject_non_finite(value, path="$")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _reject_non_finite(value: Any, *, path: str) -> None:
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ConfigError(f"non-finite number at {path} cannot be canonicalised")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, path=f"{path}[{index}]")


def canonical_digest(value: Any) -> str:
    """SHA-256 of the canonical serialisation, as ``sha256:<hex>``."""
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_text(text: str) -> str:
    """SHA-256 of a raw string (used for claim texts, templates and commands)."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def is_digest(value: str) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    body = value[len("sha256:") :]
    return len(body) == 64 and all(char in _HEX for char in body)


def digest_file(path: str, *, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return "sha256:" + hasher.hexdigest()


def content_address_aggregate(entries: Mapping[str, str]) -> str:
    """Aggregate a ``{relative path: digest}`` inventory into one root digest.

    The aggregate is order independent by construction (sorted keys) but changes
    when a file is added, removed, renamed or modified — the property
    ``E15-01`` step 23 and ``E15-06`` step 23 rely on.  An empty inventory is
    refused: "the directory exists" is not a complete inventory.
    """
    if not entries:
        raise ConfigError("cannot aggregate an empty inventory: an empty inventory is not a complete one")
    for name, digest in entries.items():
        if not is_digest(digest):
            raise ConfigError(f"inventory entry {name!r} is not a sha256 digest: {digest!r}")
    return canonical_digest({str(name): str(digest) for name, digest in entries.items()})


def file_inventory(root: str, *, suffixes: Sequence[str] = (), limit: int = 200_000) -> Dict[str, str]:
    """Hash every regular file under ``root`` into ``{relpath: sha256:...}``.

    ``limit`` guards against hashing a directory that accidentally points at a
    whole filesystem; exceeding it raises instead of truncating silently (a
    silently truncated inventory is the failure mode ``E15-01`` step 23 forbids).
    """
    if not os.path.isdir(root):
        raise ConfigError(f"inventory root is not a directory: {root!r}")
    entries: Dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in sorted(filenames):
            if suffixes and not filename.endswith(tuple(suffixes)):
                continue
            absolute = os.path.join(dirpath, filename)
            if not os.path.isfile(absolute):
                continue
            if len(entries) >= limit:
                raise ConfigError(f"inventory under {root!r} exceeds {limit} files; refusing to truncate")
            entries[os.path.relpath(absolute, root)] = digest_file(absolute)
    if not entries:
        raise ConfigError(f"inventory under {root!r} is empty")
    return entries


def artifact_id(kind: str, digest: str) -> str:
    """Build ``<kind>::<sha256>``; an unknown kind is refused, never inferred."""
    if kind not in ARTIFACT_KINDS:
        raise ConfigError(f"unknown S15 artifact kind {kind!r}; known: {', '.join(ARTIFACT_KINDS)}")
    if not is_digest(digest):
        raise ConfigError(f"artifact digest must be sha256:<hex>, got {digest!r}")
    return f"{kind}::{digest}"


def parse_artifact_id(value: str) -> Tuple[str, str]:
    if "::" not in value:
        raise ConfigError(f"artifact id must be '<kind>::sha256:<hex>', got {value!r}")
    kind, digest = value.split("::", 1)
    return kind, artifact_id(kind, digest)[len(kind) + 2 :]


def assert_safe_segment(value: str, *, what: str = "segment") -> str:
    """Refuse a path segment that could escape its directory (``../``, ``/``)."""
    if not isinstance(value, str) or not _SAFE_SEGMENT.match(value):
        raise ConfigError(f"{what} must be a single safe path segment, got {value!r}")
    return value


# ── evidence references (URI + digest are both mandatory) ───────────────────


def assert_public_uri(uri: str) -> str:
    """Refuse a URI that is only resolvable on the author's machine.

    ``E15-01`` step 22 requires the URI to resolve *in a consumer environment
    without the author's local mounts*; a ``file://`` URI, an absolute home path
    or a private-network host is exactly the "orphan with a link" the gate must
    catch before a reviewer finds it.
    """
    if not isinstance(uri, str) or not uri.strip():
        raise ConfigError("evidence URI must be a non-empty string")
    candidate = uri.strip()
    lowered = candidate.lower()
    for prefix in _LOCAL_URI_PREFIXES:
        if lowered.startswith(prefix):
            raise ConfigError(f"evidence URI {uri!r} is machine-local (scheme {prefix!r}); use a public/relative URI")
    for prefix in _LOCAL_PATH_PREFIXES:
        if candidate.startswith(prefix) or f" {prefix}" in candidate:
            raise ConfigError(f"evidence URI {uri!r} contains a machine-local absolute path ({prefix!r})")
    for host in _PRIVATE_HOST_PATTERNS:
        if host in lowered:
            raise ConfigError(f"evidence URI {uri!r} points at a private/loopback host ({host!r})")
    if "\\" in candidate:
        raise ConfigError(f"evidence URI {uri!r} looks like a Windows filesystem path, not a URI")
    return candidate


@dataclass(frozen=True)
class EvidenceRef:
    """A typed pointer to one content-addressed entity (``E15-01`` §3.2).

    Both halves are mandatory: the URI answers "where do I fetch it", the digest
    answers "is this the same file that was audited".  Either one alone is a
    known failure mode (only-hash ⇒ unfetchable; only-URI ⇒ silently replaced).
    """

    uri: str
    digest: str
    kind: str = "manifest"
    media_type: str = ""
    license: str = ""
    sensitivity: str = "public"

    def __post_init__(self) -> None:
        assert_public_uri(self.uri)
        if not is_digest(self.digest):
            raise ConfigError(f"evidence digest must be sha256:<hex>, got {self.digest!r}")
        if self.kind not in ARTIFACT_KINDS:
            raise ConfigError(f"evidence kind {self.kind!r} is not an S15 artifact kind")
        if self.sensitivity not in ("public", "redacted", "withheld"):
            raise ConfigError(
                f"evidence sensitivity {self.sensitivity!r} unknown; a withheld artifact must say so explicitly"
            )

    @property
    def artifact_id(self) -> str:
        return artifact_id(self.kind, self.digest)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "uri": self.uri,
            "digest": self.digest,
            "media_type": self.media_type,
            "license": self.license,
            "sensitivity": self.sensitivity,
        }


# ── stable claim ids ────────────────────────────────────────────────────────


#: Fields of a claim that make it a *different fact object* when they change
#: (``E15-01`` step 13: the id must not move when the wording is tweaked, and
#: must move when the fact changes).
FACT_SIGNATURE_FIELDS: Tuple[str, ...] = (
    "claim_type",
    "model_artifact_id",
    "workload_spec_id",
    "operator_spec_id",
    "backend_id",
    "hardware_ids",
    "precision",
    "baseline_id",
    "estimand",
    "direction",
)


def stable_claim_id(fact: Mapping[str, Any]) -> str:
    """Derive ``CLM-<12 hex>`` from the claim's *fact signature*, not its text.

    The signature is the projection of the claim onto :data:`FACT_SIGNATURE_FIELDS`
    (missing fields are recorded as ``null`` — an absent scope is part of the
    fact).  Editing a sentence therefore keeps the id; re-scoping the claim to a
    different model or baseline yields a new id, which is the revision rule of
    ``E15-01`` step 45.
    """
    projection = {name: fact.get(name) for name in FACT_SIGNATURE_FIELDS}
    digest = canonical_digest(projection)
    return "CLM-" + digest[len("sha256:") :][:12]


def claim_revision(fact: Mapping[str, Any], *, parent_revision: int = 0) -> Dict[str, Any]:
    """Wrap a fact signature into an explicit revision record."""
    if parent_revision < 0:
        raise ConfigError("claim revision must be non-negative")
    return {
        "claim_id": stable_claim_id(fact),
        "revision": parent_revision + 1,
        "fact_signature_sha256": canonical_digest({name: fact.get(name) for name in FACT_SIGNATURE_FIELDS}),
        "rendering_text_sha256": digest_text(str(fact.get("text", ""))),
    }


# ── evidence level partial order ─────────────────────────────────────────────


def level_index(level: str) -> int:
    if level not in EVIDENCE_LEVELS:
        raise ConfigError(f"unknown evidence level {level!r}; known: {', '.join(EVIDENCE_LEVELS)}")
    return EVIDENCE_LEVELS.index(level)


def level_at_least(level: str, minimum: str) -> bool:
    """Is ``level`` at or above ``minimum`` in the evidence partial order?"""
    return level_index(level) >= level_index(minimum)


def required_support(level: str) -> Tuple[str, ...]:
    """The lower evidence a claim at ``level`` must be able to cite.

    ``RUNTIME`` is a measurement on a real device, so it needs ``SOURCE`` code and
    ``TEST`` results underneath it; a claim cannot jump to ``RUNTIME`` while
    citing only a source file (``E15-01`` §3.3: 高层必须引用低层所需证据，不能跳级).
    """
    index = level_index(level)
    return EVIDENCE_LEVELS[:index]


def channel_level_ok(channel: str, level: str, claim_level: str = "") -> bool:
    """May a claim at ``level`` be rendered into ``channel``?

    Mirrors the channel table of ``E15-01`` §11.  ``demo`` and ``resume_bullet``
    mirror the claim's own level, so they require ``claim_level`` to be supplied.
    """
    if channel not in PUBLIC_CHANNELS:
        raise ConfigError(f"unknown public channel {channel!r}; known: {', '.join(PUBLIC_CHANNELS)}")
    if channel in CHANNEL_MIRRORS_CLAIM:
        if not claim_level:
            raise ConfigError(f"channel {channel!r} mirrors the claim level; claim_level is required")
        return level_at_least(level, claim_level) and level_index(level) == level_index(claim_level)
    return level_at_least(level, CHANNEL_MINIMUM_LEVEL[channel])


@dataclass
class AuditLogEntry:
    """One entry of the ``command_or_action_log.jsonl`` (§22 of the S15 manual)."""

    actor: str
    action: str
    started_at_utc: str
    ended_at_utc: str = ""
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()
    exit_code: Optional[int] = None
    duration_s: Optional[float] = None
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "actor": self.actor,
            "action": self.action,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "exit_code": self.exit_code,
            "duration_s": self.duration_s,
            "notes": self.notes,
        }
        payload["entry_sha256"] = canonical_digest(payload)
        return payload


def sort_actions(entries: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """UTC-monotonic ordering of an action log (``E15-01`` §22 requirement)."""
    return sorted(entries, key=lambda item: (str(item.get("started_at_utc", "")), str(item.get("action", ""))))


@dataclass
class FrozenInputs:
    """The S15 inputs frozen *before* any audit runs (``E15-01`` step 1)."""

    candidate_id: str
    source_commit: str
    tree_dirty: bool
    contract_versions: Mapping[str, str] = field(default_factory=dict)
    upstream_stage_acceptance: Tuple[str, ...] = ()
    claim_ledger_id: str = ""
    evidence_graph_root: str = ""
    release_policy_id: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.candidate_id:
            problems.append("FrozenInputs: candidate_id is required")
        if not self.source_commit:
            problems.append("FrozenInputs: a full source commit is required (tag names are not identity)")
        if self.tree_dirty:
            problems.append("FrozenInputs: a dirty tree cannot be frozen as an audit input (E15-05 step 2)")
        expected = tuple(f"C{index}" for index in range(1, 8))
        missing = [name for name in expected if name not in self.contract_versions]
        if missing:
            problems.append(f"FrozenInputs: contract versions missing for {', '.join(missing)}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": f"{SCHEMA_PREFIX}.frozen-inputs.v1",
            "candidate_id": self.candidate_id,
            "source_commit": self.source_commit,
            "tree_dirty": self.tree_dirty,
            "contract_versions": {key: self.contract_versions[key] for key in sorted(self.contract_versions)},
            "upstream_stage_acceptance": list(self.upstream_stage_acceptance),
            "claim_ledger_id": self.claim_ledger_id,
            "evidence_graph_root": self.evidence_graph_root,
            "release_policy_id": self.release_policy_id,
        }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the identity layer (labelled smoke, not an experiment)."""
    problems: List[str] = []
    fact = {
        "claim_type": "performance",
        "model_artifact_id": "model::sha256:" + "0" * 64,
        "hardware_ids": ["dev-a"],
        "estimand": "tokens/s",
        "direction": "up",
        "text": "RMSNorm 更快",
    }
    first = stable_claim_id(fact)
    reworded = dict(fact, text="RMSNorm 在本 shape 上更快")
    rescoped = dict(fact, hardware_ids=["dev-b"])
    if stable_claim_id(reworded) != first:
        problems.append("a wording change must not move the claim id")
    if stable_claim_id(rescoped) == first:
        problems.append("a fact change must move the claim id")
    refused = 0
    for uri in ("file:///root/x.json", "/root/artifacts/x.json", "http://10.0.0.4/x.json"):
        try:
            assert_public_uri(uri)
        except ConfigError:
            refused += 1
    if refused != 3:
        problems.append(f"machine-local URI detection missed {3 - refused} of 3 cases")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "claim_id_example": first,
        "levels": list(EVIDENCE_LEVELS),
        "channels": list(PUBLIC_CHANNELS),
        "local_uri_refusals": refused,
        "problems": problems,
    }
