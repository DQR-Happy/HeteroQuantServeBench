"""Artifact identity, multi-level lineage and the three S11 identities.

Protocol anchors (``docs/stage_experiments/details/S11/README.md``):

* §5 — every compiled artifact records an :class:`ArtifactIdentity` with
  ``raw_hash`` **and** ``canonical_hash`` plus parent links, versions, target
  features, guards and the environment fingerprint;
* §6 — ``graph_identity`` / ``semantic_identity`` / ``compile_identity`` are
  explicit, auditable keys (they are *not* claimed to equal any framework
  internal cache key);
* §13/E11-08 — identities feed the cache key spec.

Nothing in this module compiles, loads or probes anything: it only builds,
normalises, hashes and validates identity records.  Hashing never silently
drops non-semantic noise: the caller declares which keys are noise via
``NOISE_KEYS`` (or an explicit override), and the canonicaliser records what it
removed so the report can cite it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from hqsb.core.errors import ConfigError

# ── IR levels (protocol §5) ────────────────────────────────────────────────

IR_SOURCE = "SOURCE"
IR_DYNAMO_FX = "DYNAMO_FX"
IR_EXPORT_ATEN = "EXPORT_ATEN"
IR_HQSB_CANONICAL = "HQSB_CANONICAL"
IR_HQSB_TARGETED = "HQSB_TARGETED"
IR_INDUCTOR_LOOP = "INDUCTOR_LOOP"
IR_TIR = "TIR"
IR_MLIR = "MLIR"
IR_TTIR = "TTIR"
IR_TTGIR = "TTGIR"
IR_BACKEND_NATIVE = "BACKEND_NATIVE"
IR_LLVM_IR = "LLVM_IR"
IR_PTX = "PTX"
IR_CUBIN_SASS = "CUBIN_SASS"
IR_NPU_BINARY = "NPU_BINARY"

IR_LEVELS: Tuple[str, ...] = (
    IR_SOURCE,
    IR_DYNAMO_FX,
    IR_EXPORT_ATEN,
    IR_HQSB_CANONICAL,
    IR_HQSB_TARGETED,
    IR_INDUCTOR_LOOP,
    IR_TIR,
    IR_MLIR,
    IR_TTIR,
    IR_TTGIR,
    IR_BACKEND_NATIVE,
    IR_LLVM_IR,
    IR_PTX,
    IR_CUBIN_SASS,
    IR_NPU_BINARY,
)

#: Levels whose payload may legitimately contain addresses, temporary symbol
#: names or timestamps (they must therefore be canonicalised before hashing).
VOLATILE_IR_LEVELS: Tuple[str, ...] = (
    IR_DYNAMO_FX,
    IR_EXPORT_ATEN,
    IR_INDUCTOR_LOOP,
    IR_LLVM_IR,
    IR_PTX,
    IR_TTIR,
    IR_TTGIR,
    IR_BACKEND_NATIVE,
)

#: Key names stripped by the canonicaliser.  Chosen conservatively: only
#: fields that cannot change semantics may appear here, and the strip list is
#: recorded in every report (``stripped_keys``) so a silent over-strip would
#: be visible.
NOISE_KEYS: Tuple[str, ...] = (
    "created_at",
    "written_at",
    "timestamp",
    "host_clock",
    "address",
    "addr",
    "ptr",
    "pointer",
    "temp_name",
    "tmp_path",
    "temp_path",
    "absolute_path",
    "output_root",
    "pid",
    "duration_us",
    "elapsed_us",
    "_path",
)

#: Keys that must never be stripped (guarded by a test): they change identity.
SEMANTIC_KEYS: Tuple[str, ...] = (
    "op",
    "semantic_op",
    "target",
    "dtype",
    "shape",
    "stride",
    "layout",
    "guards",
    "constraints",
    "effects",
    "constants",
    "pass",
    "version",
    "artifact_id",
    "ir_level",
)

_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{4,}")
_TEMP_NAME_RE = re.compile(r"\b(tmp|temp|t\d+|_t\d+)\b")


def sha256_text(text: str) -> str:
    """SHA-256 of UTF-8 text (hex, lowercase)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, ``ensure_ascii=False``."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonicalize_payload(
    payload: Any, *, noise_keys: Sequence[str] = NOISE_KEYS
) -> Tuple[Any, List[str]]:
    """Return ``(canonical_payload, stripped_paths)``.

    Only *noise* keys are removed; addresses and temporary symbol names inside
    string values are normalised to placeholders so that two runs of the same
    semantic artifact hash equal.  The returned ``stripped_paths`` is recorded
    in reports, so an over-aggressive canonicaliser cannot hide a semantic
    difference silently.
    """
    stripped: List[str] = []
    noise = set(noise_keys)

    def _walk(value: Any, path: str) -> Any:
        if isinstance(value, Mapping):
            out: Dict[str, Any] = {}
            for key in sorted(value):
                child = f"{path}.{key}" if path else str(key)
                if key in noise:
                    stripped.append(child)
                    continue
                out[key] = _walk(value[key], child)
            return out
        if isinstance(value, (list, tuple)):
            return [_walk(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, str):
            cleaned = _ADDRESS_RE.sub("0xADDR", value)
            cleaned = _TEMP_NAME_RE.sub("TMP", cleaned)
            if cleaned != value:
                stripped.append(f"{path}:normalised")
            return cleaned
        return value

    return _walk(payload, ""), stripped


def content_hash(
    payload: Any, *, noise_keys: Sequence[str] = NOISE_KEYS
) -> Tuple[str, List[str]]:
    """Canonical hash of a payload plus the list of stripped paths."""
    canonical, stripped = canonicalize_payload(payload, noise_keys=noise_keys)
    return sha256_text(canonical_json(canonical)), stripped


# ── artifact identity (protocol §5) ────────────────────────────────────────

IDENTITY_REQUIRED_FIELDS: Tuple[str, ...] = (
    "artifact_id",
    "run_id",
    "compile_id",
    "ir_level",
    "format",
    "schema_version",
    "source_commit",
    "target_triple",
    "target_arch",
)

_IDENTITY_SEQUENCE_FIELDS: Tuple[str, ...] = (
    "parent_artifact_ids",
    "compiler_versions",
    "target_features",
    "symbolic_constraints",
    "guards",
)


@dataclass
class ArtifactIdentity:
    """One compiled artifact at one IR level (protocol §5 table)."""

    artifact_id: str
    ir_level: str
    format: str
    schema_version: str
    run_id: str = ""
    compile_id: str = ""
    parent_artifact_ids: Tuple[str, ...] = ()
    source_commit: str = ""
    dirty_patch_hash: str = ""
    capture_version: str = ""
    pass_pipeline_version: str = ""
    lowering_version: str = ""
    compiler_versions: Tuple[str, ...] = ()
    target_triple: str = ""
    target_arch: str = ""
    target_features: Tuple[str, ...] = ()
    symbolic_constraints: Tuple[str, ...] = ()
    guards: Tuple[str, ...] = ()
    raw_hash: str = ""
    canonical_hash: str = ""
    created_at: str = ""
    command: str = ""
    environment_fingerprint: str = ""
    verification_status: str = "UNVERIFIED"
    consumer: str = ""

    def validate(self) -> List[str]:
        """Return a list of schema problems (empty list means valid)."""
        problems: List[str] = []
        for name in IDENTITY_REQUIRED_FIELDS:
            if not getattr(self, name):
                problems.append(f"missing required field {name!r}")
        if self.ir_level and self.ir_level not in IR_LEVELS:
            problems.append(f"unknown ir_level {self.ir_level!r}")
        if not (self.raw_hash or self.canonical_hash):
            problems.append("at least one of raw_hash/canonical_hash must be set")
        for name in ("raw_hash", "canonical_hash"):
            value = getattr(self, name)
            if value and not re.fullmatch(r"[0-9a-f]{64}", value):
                problems.append(f"{name} must be a 64-char lowercase hex digest")
        if self.ir_level in VOLATILE_IR_LEVELS and not self.canonical_hash:
            problems.append(
                f"ir_level {self.ir_level} is volatile: canonical_hash is mandatory "
                "(raw_hash alone cannot define identity)"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "run_id": self.run_id,
            "compile_id": self.compile_id,
            "parent_artifact_ids": list(self.parent_artifact_ids),
            "ir_level": self.ir_level,
            "format": self.format,
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
            "dirty_patch_hash": self.dirty_patch_hash,
            "capture_version": self.capture_version,
            "pass_pipeline_version": self.pass_pipeline_version,
            "lowering_version": self.lowering_version,
            "compiler_versions": list(self.compiler_versions),
            "target_triple": self.target_triple,
            "target_arch": self.target_arch,
            "target_features": list(self.target_features),
            "symbolic_constraints": list(self.symbolic_constraints),
            "guards": list(self.guards),
            "raw_hash": self.raw_hash,
            "canonical_hash": self.canonical_hash,
            "created_at": self.created_at,
            "command": self.command,
            "environment_fingerprint": self.environment_fingerprint,
            "verification_status": self.verification_status,
            "consumer": self.consumer,
        }

    def identity_hash(self) -> str:
        """Hash of the *semantic* part only: timestamps and command excluded."""
        payload = self.as_dict()
        for key in ("created_at", "command", "environment_fingerprint", "raw_hash"):
            payload.pop(key, None)
        return sha256_text(canonical_json(payload))

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, verify: bool = True
    ) -> "ArtifactIdentity":
        """Build from a mapping (unknown keys rejected — no silent drop)."""
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ConfigError(f"unknown ArtifactIdentity fields: {unknown}")
        data: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in _IDENTITY_SEQUENCE_FIELDS:
                data[key] = tuple(value)
            else:
                data[key] = value
        identity = cls(**data)
        if verify:
            problems = identity.validate()
            if problems:
                raise ConfigError("invalid artifact identity: " + "; ".join(problems))
        return identity


def artifact_index_rows(artifacts: Iterable[ArtifactIdentity]) -> List[Dict[str, Any]]:
    """Flatten identities into ``artifact_index.jsonl`` rows."""
    rows = []
    for artifact in artifacts:
        row = artifact.as_dict()
        row["identity_hash"] = artifact.identity_hash()
        rows.append(row)
    return rows


# ── lineage (E11-03 step 17 / hypothesis H2) ───────────────────────────────


@dataclass
class LineageGraph:
    """Parent/child closure over artifacts; validates and measures coverage."""

    artifacts: Dict[str, ArtifactIdentity] = field(default_factory=dict)
    children: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, artifact: ArtifactIdentity) -> None:
        if artifact.artifact_id in self.artifacts:
            raise ConfigError(f"duplicate artifact_id {artifact.artifact_id!r}")
        self.artifacts[artifact.artifact_id] = artifact
        for parent in artifact.parent_artifact_ids:
            self.children.setdefault(parent, []).append(artifact.artifact_id)

    def validate(self) -> List[str]:
        problems: List[str] = []
        for artifact_id, artifact in self.artifacts.items():
            for parent in artifact.parent_artifact_ids:
                if parent not in self.artifacts:
                    problems.append(f"{artifact_id}: unknown parent {parent!r}")
            if artifact_id in artifact.parent_artifact_ids:
                problems.append(f"{artifact_id}: self-parent")
        problems.extend(f"cycle detected: {' -> '.join(cycle)}" for cycle in self.cycles())
        return problems

    def cycles(self) -> List[List[str]]:
        """Find cycles via iterative DFS (small graphs: clarity over speed)."""
        found: List[List[str]] = []
        color: Dict[str, int] = {}

        def visit(node: str, stack: List[str]) -> None:
            color[node] = 1
            stack.append(node)
            for child in self.children.get(node, []):
                if color.get(child, 0) == 1:
                    start = stack.index(child)
                    found.append(stack[start:] + [child])
                elif color.get(child, 0) == 0:
                    visit(child, stack)
            stack.pop()
            color[node] = 2

        for node in list(self.artifacts):
            if color.get(node, 0) == 0:
                visit(node, [])
        return found

    def ancestors(self, artifact_id: str) -> List[str]:
        seen: List[str] = []
        frontier = list(self.artifacts[artifact_id].parent_artifact_ids)
        while frontier:
            node = frontier.pop()
            if node in seen:
                continue
            seen.append(node)
            frontier.extend(self.artifacts[node].parent_artifact_ids)
        return seen

    def levels_reachable(self, artifact_id: str) -> List[str]:
        ids = [artifact_id, *self.ancestors(artifact_id)]
        levels = {self.artifacts[item].ir_level for item in ids}
        return [level for level in IR_LEVELS if level in levels]

    def coverage(self, artifact_id: str, expected_levels: Sequence[str]) -> Dict[str, Any]:
        """Lineage completeness: which expected levels are reachable via parents."""
        reachable = self.levels_reachable(artifact_id)
        missing = [level for level in expected_levels if level not in reachable]
        return {
            "artifact_id": artifact_id,
            "expected_levels": list(expected_levels),
            "reachable_levels": reachable,
            "missing_levels": missing,
            "complete": not missing,
        }


# ── the three S11 identities (protocol §6) ─────────────────────────────────


def graph_identity(
    *,
    normalized_graph: Any,
    op_schema_versions: Mapping[str, str],
    constants_hash: str,
    tensor_metadata: Any,
    symbolic_constraints: Sequence[str],
    effects: Sequence[str],
    capture_mode: str,
    capture_version: str,
) -> Dict[str, Any]:
    """``graph_identity = H(structure, schemas, constants, metadata, …)``."""
    payload = {
        "normalized_graph": normalized_graph,
        "op_schema_versions": dict(sorted(op_schema_versions.items())),
        "constants_hash": constants_hash,
        "tensor_metadata": tensor_metadata,
        "symbolic_constraints": list(symbolic_constraints),
        "effects": list(effects),
        "capture_mode": capture_mode,
        "capture_version": capture_version,
    }
    digest, stripped = content_hash(payload)
    return {
        "kind": "graph_identity",
        "digest": digest,
        "payload": payload,
        "stripped_keys": stripped,
        "note": "audit key; not claimed to equal any framework cache key",
    }


def semantic_identity(
    *,
    canonical_graph_identity: str,
    model_artifact_id: str,
    quant_artifact_id: str,
    numerical_contract: str,
    state_contract: str,
) -> Dict[str, Any]:
    """``semantic_identity`` = canonicalised graph identity + policy contracts."""
    payload = {
        "canonical_graph_identity": canonical_graph_identity,
        "model_artifact_id": model_artifact_id,
        "quant_artifact_id": quant_artifact_id,
        "numerical_contract": numerical_contract,
        "state_contract": state_contract,
    }
    digest, stripped = content_hash(payload)
    return {
        "kind": "semantic_identity",
        "digest": digest,
        "payload": payload,
        "stripped_keys": stripped,
        "note": "audit key; not claimed to equal any framework cache key",
    }


def compile_identity(
    *,
    semantic_identity_digest: str,
    pass_pipeline: Sequence[Mapping[str, Any]],
    pass_options: Mapping[str, Any],
    lowering_registry_digest: str,
    selected_candidate_id: str,
    kernel_build_ids: Sequence[str],
    compiler_versions: Mapping[str, str],
    compiler_flags: Sequence[str],
    target_triple: str,
    target_arch: str,
    target_features: Sequence[str],
    abi_version: str,
    guard_domain: Any,
    autotune_id: str,
    cost_model_id: str,
) -> Dict[str, Any]:
    """``compile_identity`` = everything that can change the produced binary."""
    payload = {
        "semantic_identity": semantic_identity_digest,
        "pass_pipeline": list(pass_pipeline),
        "pass_options": dict(sorted(pass_options.items())),
        "lowering_registry_digest": lowering_registry_digest,
        "selected_candidate_id": selected_candidate_id,
        "kernel_build_ids": list(kernel_build_ids),
        "compiler_versions": dict(sorted(compiler_versions.items())),
        "compiler_flags": list(compiler_flags),
        "target_triple": target_triple,
        "target_arch": target_arch,
        "target_features": list(target_features),
        "abi_version": abi_version,
        "guard_domain": guard_domain,
        "autotune_id": autotune_id,
        "cost_model_id": cost_model_id,
    }
    digest, stripped = content_hash(payload)
    return {
        "kind": "compile_identity",
        "digest": digest,
        "payload": payload,
        "stripped_keys": stripped,
        "note": "audit key; not claimed to equal any framework cache key",
    }


def identity_field_diff(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Field-level diff used by cache invalidation and drift reports."""
    rows: List[Dict[str, Any]] = []
    for key in sorted(set(before) | set(after)):
        left, right = before.get(key), after.get(key)
        if left != right:
            rows.append({"field": key, "before": left, "after": right})
    return rows


def version_is_frozen(version: str) -> bool:
    """Reject vague versions (``latest``/``main``/empty) — identity must be exact."""
    if not version or not version.strip():
        return False
    lowered = version.strip().lower()
    if lowered in {"latest", "main", "master", "head", "unknown", "any", "stable"}:
        return False
    return bool(re.search(r"\d", version))


def require_frozen_versions(versions: Mapping[str, str]) -> List[str]:
    """Return the names of versions that are not frozen."""
    return [name for name, value in sorted(versions.items()) if not version_is_frozen(value)]


def check_noise_keys_do_not_touch_semantics() -> List[str]:
    """Self-audit used by tests: no semantic key may be in ``NOISE_KEYS``."""
    return sorted(set(NOISE_KEYS) & set(SEMANTIC_KEYS))


def normalized_dirty_patch_hash(diff_text: str) -> str:
    """Hash of a working-tree diff with volatile headers removed."""
    lines = [
        line
        for line in diff_text.splitlines()
        if not line.startswith(("index ", "--- ", "+++ ", "@@ "))
    ]
    return sha256_text("\n".join(lines).strip())


#: Minimum lineage chain an experiment's hero path must be able to show.
IDENTITY_CHAINS: Mapping[str, Tuple[str, ...]] = {
    "E11-01": (IR_SOURCE, IR_DYNAMO_FX, IR_EXPORT_ATEN),
    "E11-02": (IR_DYNAMO_FX, IR_HQSB_CANONICAL),
    "E11-03": (
        IR_SOURCE,
        IR_DYNAMO_FX,
        IR_HQSB_CANONICAL,
        IR_HQSB_TARGETED,
        IR_CUBIN_SASS,
    ),
    "E11-04": (IR_HQSB_TARGETED, IR_INDUCTOR_LOOP, IR_PTX, IR_CUBIN_SASS),
    "E11-05": (IR_DYNAMO_FX, IR_HQSB_TARGETED),
    "E11-06": (IR_HQSB_TARGETED, IR_TTIR),
    "E11-07": (IR_HQSB_TARGETED,),
    "E11-08": (IR_HQSB_TARGETED, IR_CUBIN_SASS),
    "E11-09": (IR_HQSB_CANONICAL, IR_TIR, IR_LLVM_IR, IR_CUBIN_SASS),
    "E11-10": (IR_HQSB_TARGETED, IR_CUBIN_SASS),
}


def identity_required_levels_for_experiment(experiment_id: str) -> Tuple[str, ...]:
    """Minimum lineage chain an experiment's hero path must be able to show."""
    if experiment_id not in IDENTITY_CHAINS:
        raise ConfigError(f"unknown experiment id {experiment_id!r}")
    return IDENTITY_CHAINS[experiment_id]


def identity_chain_status(
    lineage: LineageGraph, experiment_id: str, artifact_id: str
) -> Dict[str, Any]:
    """Attach the experiment-specific expected chain to ``LineageGraph.coverage``."""
    report = lineage.coverage(artifact_id, identity_required_levels_for_experiment(experiment_id))
    report["experiment_id"] = experiment_id
    return report
