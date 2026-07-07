"""Canonical identity for every S14 artifact.

S14 has to answer lineage questions that a path name cannot answer
(``details/S14/README.md`` §7, §9, §28 errors 1/2/3):

* which training run produced this checkpoint, from which base model, data and
  seed bundle;
* which checkpoint, converter and adapter produced this serving artifact;
* which policy snapshot produced this trajectory.

Everything here is *content addressed*: a value is serialised canonically
(sorted keys, no NaN, explicit units), hashed with SHA-256 and wrapped in a
typed id of the form ``train::<sha256>``.  Two runs that differ only in
whitespace or dict ordering therefore share an id, while two runs that differ in
any recorded field never do.

The module is pure stdlib and CPU-only: no framework import, no filesystem
access outside the explicit ``*_inventory`` helpers.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Schema-version namespace for every S14 payload.
SCHEMA_PREFIX = "hqsb.s14"

#: Artifact kinds that may be content addressed by :func:`artifact_id`.
ARTIFACT_KINDS: Tuple[str, ...] = (
    "training-run",
    "checkpoint",
    "serving-model",
    "policy-snapshot",
    "trajectory",
    "reward",
    "dataset",
    "conversion-graph",
    "frontier-contract",
    "adoption-decision",
    "dependency-boundary",
    "run",
    "quality",
    "seed-bundle",
    "topology",
    "generation-config",
    "sparse-artifact",
)

#: Seed roles that must be recorded separately (E14-02 step 7): one global seed
#: is not a reproducible experiment.
SEED_ROLES: Tuple[str, ...] = (
    "python",
    "numpy",
    "framework",
    "device",
    "sampler",
    "dataloader",
    "dropout",
)

_HEX = frozenset("0123456789abcdef")


def canonical_json(value: Any) -> str:
    """Serialise ``value`` deterministically (sorted keys, compact separators).

    ``float('nan')``/``inf`` are refused instead of emitted as non-JSON tokens:
    a NaN that reaches a digest silently makes two different runs collide
    (``E14-02`` §11: special values must be an explicit failure policy).
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
    """SHA-256 of a raw string (used for prompts, templates and commands)."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def is_digest(value: str) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    body = value[len("sha256:") :]
    return len(body) == 64 and all(char in _HEX for char in body)


def content_address_aggregate(entries: Mapping[str, str]) -> str:
    """Aggregate a ``{relative path: digest}`` inventory into one root digest.

    The aggregate is order independent by construction (sorted keys) but changes
    when a file is added, removed, renamed or modified — the property
    ``E14-02`` step 30 and ``E14-03`` step 20 rely on.  A directory existing is
    never accepted as proof of completeness: the caller must supply the full
    inventory (``inventory_completeness`` checks that separately).
    """
    if not entries:
        raise ConfigError("cannot aggregate an empty inventory: an empty inventory is not a complete one")
    for name, digest in entries.items():
        if not is_digest(digest):
            raise ConfigError(f"inventory entry {name!r} is not a sha256 digest: {digest!r}")
    return canonical_digest({str(name): str(digest) for name, digest in entries.items()})


def file_inventory(root: str, *, suffixes: Sequence[str] = (), limit: int = 200_000) -> Dict[str, str]:
    """Hash every regular file under ``root`` into ``{relpath: sha256:...}``.

    ``limit`` guards against hashing a checkpoint directory that accidentally
    points at a whole filesystem; exceeding it raises instead of truncating
    silently (a silently truncated inventory is exactly the "directory exists"
    failure mode ``E14-02`` step 30 forbids).
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


def digest_file(path: str, *, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return "sha256:" + hasher.hexdigest()


def artifact_id(kind: str, digest: str) -> str:
    """Build ``<kind>::<sha256>``; an unknown kind is refused, never inferred."""
    if kind not in ARTIFACT_KINDS:
        raise ConfigError(f"unknown S14 artifact kind {kind!r}; known: {', '.join(ARTIFACT_KINDS)}")
    if not is_digest(digest):
        raise ConfigError(f"artifact digest must be sha256:<hex>, got {digest!r}")
    return f"{kind}::{digest}"


def parse_artifact_id(value: str) -> Tuple[str, str]:
    if "::" not in value:
        raise ConfigError(f"artifact id must be '<kind>::sha256:<hex>', got {value!r}")
    kind, digest = value.split("::", 1)
    return kind, artifact_id(kind, digest)[len(kind) + 2 :]


@dataclass(frozen=True)
class ArtifactRef:
    """A typed pointer to one content-addressed artifact."""

    kind: str
    digest: str
    uri: str = ""
    schema_version: str = ""

    @property
    def artifact_id(self) -> str:
        return artifact_id(self.kind, self.digest)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"artifact_id": self.artifact_id, "kind": self.kind, "digest": self.digest}
        if self.uri:
            payload["uri"] = self.uri
        if self.schema_version:
            payload["schema_version"] = self.schema_version
        return payload


def artifact_ref(kind: str, payload: Any, *, uri: str = "", schema_version: str = "") -> ArtifactRef:
    """Content-address an in-memory payload."""
    if schema_version and not schema_version.startswith(SCHEMA_PREFIX):
        raise ConfigError(f"schema_version {schema_version!r} is not in the {SCHEMA_PREFIX}.* namespace")
    return ArtifactRef(kind=kind, digest=canonical_digest(payload), uri=uri, schema_version=schema_version)


# ── seed bundle ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SeedBundle:
    """Per-role seeds with a documented derivation rule (E14-02 step 7).

    The bundle stores the *base* seeds; derived seeds are computed by
    :meth:`derive` so a resumed run can reproduce the stream of a given
    ``(role, rank, step)`` without replaying the whole run.
    """

    base_seeds: Mapping[str, int]
    derivation: str = "sha256(base:role:rank:step) mod 2**31"
    notes: str = ""

    def __post_init__(self) -> None:
        missing = [role for role in SEED_ROLES if role not in self.base_seeds]
        if missing:
            raise ConfigError(
                "seed bundle must record every role separately (a single global seed is not "
                f"reproducible); missing: {', '.join(missing)}"
            )
        for role, seed in self.base_seeds.items():
            if not isinstance(seed, int) or seed < 0:
                raise ConfigError(f"seed for role {role!r} must be a non-negative int, got {seed!r}")

    def derive(self, role: str, *, rank: int = 0, step: int = 0) -> int:
        if role not in self.base_seeds:
            raise ConfigError(f"unknown seed role {role!r}; known: {', '.join(sorted(self.base_seeds))}")
        if rank < 0 or step < 0:
            raise ConfigError("rank and step must be non-negative when deriving a seed")
        material = f"{self.base_seeds[role]}:{role}:{rank}:{step}".encode("utf-8")
        return int(hashlib.sha256(material).hexdigest(), 16) % (2**31)

    @property
    def seed_bundle_id(self) -> str:
        return artifact_id("seed-bundle", canonical_digest(self.as_dict()))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": f"{SCHEMA_PREFIX}.seed-bundle.v1",
            "base_seeds": {role: self.base_seeds[role] for role in sorted(self.base_seeds)},
            "derivation": self.derivation,
            "notes": self.notes,
        }


def seed_bundle(**seeds: int) -> SeedBundle:
    """Build a bundle from keyword seeds, e.g. ``seed_bundle(python=1, numpy=1, ...)``."""
    return SeedBundle(base_seeds=dict(seeds))


# ── parallel topology / rank map ───────────────────────────────────────────


@dataclass(frozen=True)
class RankIdentity:
    """One rank's immutable identity (E14-02 step 14)."""

    rank: int
    local_rank: int
    world_size: int
    host: str
    device_id: str = ""
    node_rank: int = 0
    process_group: str = "default"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "host": self.host,
            "device_id": self.device_id,
            "node_rank": self.node_rank,
            "process_group": self.process_group,
        }


def assert_rank_identities(identities: Sequence[RankIdentity]) -> List[str]:
    """Return the list of identity violations (empty means a well-formed group).

    Two ranks bound to the same ``(host, device_id)`` would keep training while
    silently computing the same shard twice (``E14-02`` step 14).
    """
    problems: List[str] = []
    if not identities:
        return ["no rank identities supplied"]
    world = identities[0].world_size
    ranks = sorted(item.rank for item in identities)
    if ranks != list(range(world)):
        problems.append(f"ranks {ranks} are not 0..{world - 1}")
    if len(identities) != world:
        problems.append(f"{len(identities)} identities for world_size {world}")
    for item in identities:
        if item.world_size != world:
            problems.append(f"rank {item.rank} declares world_size {item.world_size}, expected {world}")
    devices = [(item.host, item.device_id) for item in identities if item.device_id]
    duplicates = {pair for pair in devices if devices.count(pair) > 1}
    for host, device in sorted(duplicates):
        problems.append(f"device {host}:{device} is claimed by more than one rank")
    return problems


def topology_digest(identities: Sequence[RankIdentity], *, parallel_plan: Optional[Mapping[str, Any]] = None) -> str:
    """Content address a topology so a checkpoint can prove where it came from."""
    payload = {
        "ranks": [item.as_dict() for item in sorted(identities, key=lambda entry: entry.rank)],
        "parallel_plan": dict(parallel_plan or {}),
    }
    return canonical_digest(payload)


# ── lineage ────────────────────────────────────────────────────────────────


@dataclass
class LineageEdge:
    """One edge of the artifact DAG (``E14-03`` §3.1)."""

    node: str
    tool: str
    inputs: Tuple[str, ...]
    output: str
    config_digest: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "tool": self.tool,
            "inputs": list(self.inputs),
            "output": self.output,
            "config_digest": self.config_digest,
        }


@dataclass
class LineageChain:
    """A conversion/lineage DAG that can be walked and audited."""

    edges: List[LineageEdge] = field(default_factory=list)

    def add(self, node: str, tool: str, inputs: Iterable[str], output: str, config_digest: str = "") -> LineageEdge:
        edge = LineageEdge(node=node, tool=tool, inputs=tuple(inputs), output=output, config_digest=config_digest)
        self.edges.append(edge)
        return edge

    def roots(self) -> List[str]:
        produced = {edge.output for edge in self.edges}
        consumed = {item for edge in self.edges for item in edge.inputs}
        return sorted(consumed - produced)

    def leaves(self) -> List[str]:
        produced = {edge.output for edge in self.edges}
        consumed = {item for edge in self.edges for item in edge.inputs}
        return sorted(produced - consumed)

    def producers(self, artifact: str) -> List[LineageEdge]:
        return [edge for edge in self.edges if edge.output == artifact]

    def consumers(self, artifact: str) -> List[LineageEdge]:
        return [edge for edge in self.edges if artifact in edge.inputs]

    def reachable_from(self, artifact: str) -> List[str]:
        """Every artifact derived (transitively) from ``artifact``."""
        seen: set = set()
        frontier = [artifact]
        while frontier:
            current = frontier.pop()
            for edge in self.consumers(current):
                if edge.output not in seen:
                    seen.add(edge.output)
                    frontier.append(edge.output)
        return sorted(seen)

    def validate(self, *, strict_roots: bool = False) -> List[str]:
        """Structural problems: duplicate producers, cycles, and (optionally) dangling inputs.

        A conversion DAG legitimately *starts* from external artifacts — the source
        checkpoint, the tokenizer, the config — so an input with no producer is a
        **root**, not an error.  ``strict_roots=True`` additionally reports every
        such input, which is what a "the DAG must be self-contained" audit wants
        (``E14-03`` step 3 freezes the node list, so an unexpected external input
        is worth flagging when the caller knows there should be none).
        """
        problems: List[str] = []
        produced: Dict[str, List[LineageEdge]] = {}
        for edge in self.edges:
            produced.setdefault(edge.output, []).append(edge)
        for output, producers in sorted(produced.items()):
            if len(producers) > 1:
                problems.append(f"artifact {output} has {len(producers)} producers (the DAG is not a function)")
        consumers_of = {item for edge in self.edges for item in edge.inputs}
        dangling = sorted(consumers_of - set(produced))
        if strict_roots:
            for artifact in dangling:
                problems.append(f"input {artifact} is consumed but never produced")
        if self._has_cycle():
            problems.append("lineage graph contains a cycle")
        return problems

    def _has_cycle(self) -> bool:
        adjacency: Dict[str, List[str]] = {}
        for edge in self.edges:
            for item in edge.inputs:
                adjacency.setdefault(item, []).append(edge.output)
        state: Dict[str, int] = {}

        def visit(node: str) -> bool:
            if state.get(node) == 1:
                return True
            if state.get(node) == 2:
                return False
            state[node] = 1
            for nxt in adjacency.get(node, ()):
                if visit(nxt):
                    return True
            state[node] = 2
            return False

        return any(visit(node) for node in list(adjacency))

    def digest(self) -> str:
        return canonical_digest([edge.as_dict() for edge in self.edges])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": f"{SCHEMA_PREFIX}.lineage.v1",
            "roots": self.roots(),
            "leaves": self.leaves(),
            "edges": [edge.as_dict() for edge in self.edges],
            "digest": self.digest(),
        }
