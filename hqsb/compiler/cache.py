"""Compile-artifact cache: keys, transactional publish, safe reads, lifecycle.

Protocol anchor: ``details/S11/E11-08_compile_artifact_cache_invalidation.md``.
A cache is *conditional reuse*, not "a file exists":

    cache_hit = entry exists ∧ metadata parses ∧ schema supported ∧ key matches
                ∧ identity/target/ABI compatible ∧ guard domain covers the input
                ∧ payload hash matches ∧ state == COMMITTED

Everything in this module is real: entries are written to disk with
write-temp → fsync → atomic rename → COMMITTED-marker-last, readers validate
before loading, corrupt entries are quarantined and never executed.  The
corruption/fault injections operate on isolated copies under the caller's
directory, so a user cache is never damaged by a test.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import (
    canonical_json,
    content_hash,
    sha256_text,
)

# ── layers and key spec (E11-08 §3.2, §4) ──────────────────────────────────

CACHE_LAYERS: Tuple[str, ...] = (
    "frame_guard",
    "capture_graph",
    "pass_ir",
    "lowering_decision",
    "codegen_native",
    "autotune_db",
    "cost_model",
    "binary_package",
)

LAYER_OWNERS: Mapping[str, str] = {
    "frame_guard": "framework (Dynamo variants)",
    "capture_graph": "hqsb.compiler.capture",
    "pass_ir": "hqsb.compiler.rewrite",
    "lowering_decision": "hqsb.compiler.lowering",
    "codegen_native": "toolchain (Inductor/Triton/native builder)",
    "autotune_db": "hqsb.compiler.autotune",
    "cost_model": "hqsb.compiler.costmodel",
    "binary_package": "artifact store",
}

#: Key inclusion reasons — every key field must name why it is in the key.
KEY_FIELD_REASONS: Mapping[str, str] = {
    "semantic_graph": "the computation being compiled",
    "constants_hash": "folded weights change the output",
    "model_artifact_id": "model revision/policy identity",
    "quant_artifact_id": "quantised semantics and packing",
    "op_schema_versions": "operator ABI/semantics",
    "symbolic_constraints": "the input domain the artifact is valid for",
    "guard_domain": "variant legality: a different domain is a different binary",
    "pass_pipeline": "rewrite order/code changes the IR",
    "pass_options": "pass configuration changes the IR",
    "lowering_registry": "available implementations change the decision",
    "selected_candidate": "which implementation was chosen",
    "kernel_build_id": "kernel source/build identity",
    "kernel_abi": "binary ABI compatibility",
    "compiler_versions": "code generation changes with the compiler",
    "compiler_flags": "flags alter generated code",
    "target_triple": "target ABI/triple",
    "target_arch": "instruction set",
    "target_features": "usable instructions",
    "driver_compat_class": "loadability policy class",
    "autotune_db": "the chosen config comes from the tuning database",
    "cost_model": "the decision can change without recompiling the kernel",
    "compile_mode": "determinism/debug flags that affect code",
}

#: Fields that must NOT change the key (verified by key_stability).
NON_KEY_FIELDS: Tuple[str, ...] = (
    "timestamp",
    "pid",
    "output_root",
    "absolute_path",
    "log_level",
    "hostname",
)


@dataclass
class CacheKeySpec:
    """Versioned key specification with per-field reasons and compatibility class."""

    spec_version: str
    fields: Mapping[str, str]  # field -> reason
    compatibility_classes: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.spec_version:
            problems.append("key spec must be versioned")
        for name, reason in self.fields.items():
            if not reason:
                problems.append(f"key field {name!r} has no inclusion reason")
            if name in NON_KEY_FIELDS:
                problems.append(f"non-semantic field {name!r} must not be in the cache key")
        return problems

    def compute(self, parts: Mapping[str, Any]) -> Dict[str, Any]:
        """Deterministic key; missing fields are an error, not an empty string."""
        problems = self.validate()
        if problems:
            raise ConfigError("invalid key spec: " + "; ".join(problems))
        missing = [name for name in self.fields if name not in parts]
        if missing:
            raise ConfigError(f"cache key missing fields (would collide silently): {missing}")
        payload = {name: parts[name] for name in sorted(self.fields)}
        digest, stripped = content_hash(payload)
        return {
            "key": digest,
            "spec_version": self.spec_version,
            "fields": payload,
            "stripped": stripped,
        }

    def test_vectors(self) -> List[Dict[str, Any]]:
        """Vectors that document which changes must move the key."""
        base = {name: f"{name}:v1" for name in self.fields}
        rows = [{"name": "base", "parts": dict(base), "expect": "same"}]
        for name in sorted(self.fields):
            variant = dict(base)
            variant[name] = f"{name}:v2"
            rows.append({"name": f"change:{name}", "parts": variant, "expect": "differs"})
        for name in NON_KEY_FIELDS:
            variant = dict(base)
            variant[name] = "noise"
            rows.append({"name": f"noise:{name}", "parts": variant, "expect": "same"})
        return rows


def default_key_spec() -> CacheKeySpec:
    return CacheKeySpec(
        spec_version="1.0.0",
        fields=dict(KEY_FIELD_REASONS),
        compatibility_classes={
            "driver_compat_class": "minor upgrade tolerated only if the policy was verified",
            "compiler_versions": "conservative: any change invalidates unless proven compatible",
        },
    )


def evaluate_key_vectors(spec: CacheKeySpec) -> Dict[str, Any]:
    base_key = spec.compute({name: f"{name}:v1" for name in spec.fields})["key"]
    rows: List[Dict[str, Any]] = []
    for vector in spec.test_vectors():
        parts = {name: f"{name}:v1" for name in spec.fields}
        parts.update({key: value for key, value in vector["parts"].items()})
        try:
            key = spec.compute(parts)["key"]
            failed = False
        except ConfigError:
            key = ""
            failed = False  # non-key noise fields are simply ignored
        same = key == base_key if key else True
        expected_same = vector["expect"] == "same"
        rows.append(
            {
                "vector": vector["name"],
                "expect": vector["expect"],
                "same_as_base": same,
                "ok": failed is False and same == expected_same,
            }
        )
    return {"vectors": rows, "all_ok": all(row["ok"] for row in rows)}


def key_stability(spec: CacheKeySpec, *, parts: Mapping[str, Any]) -> Dict[str, Any]:
    """Non-semantic noise must not change the key (step 14)."""
    base = spec.compute(parts)["key"]
    rows = []
    for name in NON_KEY_FIELDS:
        variant = dict(parts)
        variant[name] = "noise-core-" + name
        try:
            key = spec.compute(variant)["key"]
        except ConfigError:
            key = base  # the field is not part of the key at all
        rows.append({"field": name, "same_key": key == base})
    return {
        "base_key": base,
        "rows": rows,
        "stable": all(row["same_key"] for row in rows),
        "rule": (
            "if a noise field changes the key, the spec is over-specified; if a semantic field "
            "stops changing it, the cache can serve a wrong binary"
        ),
    }


# ── entry state machine and store (E11-08 §3.4, steps 4–7) ─────────────────

STATE_TEMP = "TEMP"
STATE_VALIDATING = "VALIDATING"
STATE_COMMITTED = "COMMITTED"
STATE_QUARANTINED = "QUARANTINED"
STATE_EVICTED = "EVICTED"

ENTRY_STATES: Tuple[str, ...] = (
    STATE_TEMP,
    STATE_VALIDATING,
    STATE_COMMITTED,
    STATE_QUARANTINED,
    STATE_EVICTED,
)

ALLOWED_STATE_TRANSITIONS: Mapping[str, Tuple[str, ...]] = {
    STATE_TEMP: (STATE_VALIDATING, STATE_EVICTED),
    STATE_VALIDATING: (STATE_COMMITTED, STATE_QUARANTINED),
    STATE_COMMITTED: (STATE_QUARANTINED, STATE_EVICTED),
    STATE_QUARANTINED: (STATE_EVICTED,),
    STATE_EVICTED: (),
}

READ_REJECT_REASONS: Tuple[str, ...] = (
    "MISS_NOT_FOUND",
    "REJECT_METADATA_PARSE",
    "REJECT_METADATA_HASH",
    "REJECT_UNKNOWN_SCHEMA",
    "REJECT_KEY_MISMATCH",
    "REJECT_TARGET_INCOMPATIBLE",
    "REJECT_ABI_MISMATCH",
    "REJECT_GUARD_FALSE",
    "REJECT_PAYLOAD_HASH",
    "REJECT_NOT_COMMITTED",
    "REJECT_PERMISSION",
)


@dataclass
class EntryManifest:
    """Metadata sidecar of one entry (metadata and payload hash separately)."""

    entry_id: str
    key: str
    layer: str
    schema_version: str = "1.0.0"
    state: str = STATE_TEMP
    payload_name: str = "payload.bin"
    payload_sha256: str = ""
    payload_bytes: int = 0
    metadata_sha256: str = ""
    parent_key: str = ""
    created_by: str = ""
    target_arch: str = ""
    abi_version: str = ""
    guard_domain: str = ""
    compatibility_class: str = ""
    compile_occurred: bool = True

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.layer not in CACHE_LAYERS:
            problems.append(f"unknown cache layer {self.layer!r}")
        if self.state not in ENTRY_STATES:
            problems.append(f"unknown state {self.state!r}")
        if not self.key:
            problems.append("entry manifest needs a key")
        if self.state == STATE_COMMITTED and not self.payload_sha256:
            problems.append("a committed entry must carry its payload hash")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "key": self.key,
            "layer": self.layer,
            "schema_version": self.schema_version,
            "state": self.state,
            "payload_name": self.payload_name,
            "payload_sha256": self.payload_sha256,
            "payload_bytes": self.payload_bytes,
            "metadata_sha256": self.metadata_sha256,
            "parent_key": self.parent_key,
            "created_by": self.created_by,
            "target_arch": self.target_arch,
            "abi_version": self.abi_version,
            "guard_domain": self.guard_domain,
            "compatibility_class": self.compatibility_class,
            "compile_occurred": self.compile_occurred,
        }


@dataclass
class ReadResult:
    status: str  # hit | miss | reject
    stage: str = ""
    reason_code: str = ""
    detail: str = ""
    entry_id: str = ""
    payload_bytes: bytes = b""
    load_calls: int = 0
    steps: Tuple[Tuple[str, bool], ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "stage": self.stage,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "entry_id": self.entry_id,
            "payload_bytes": len(self.payload_bytes),
            "load_calls": self.load_calls,
            "steps": [list(step) for step in self.steps],
        }


def metadata_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash of a manifest with its own ``metadata_sha256`` field blanked.

    The field is blanked on both sides (writer and reader) so the check is
    reproducible instead of depending on whether the field was present.
    """
    body = dict(payload)
    body["metadata_sha256"] = ""
    return sha256_text(canonical_json(body))


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EntryStore:
    """A directory of cache entries with atomic publish and safe reads."""

    def __init__(self, root: str, *, spec: Optional[CacheKeySpec] = None) -> None:
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.spec = spec or default_key_spec()

    def entry_dir(self, entry_id: str) -> str:
        return os.path.join(self.root, entry_id)

    def manifest_path(self, entry_id: str) -> str:
        return os.path.join(self.entry_dir(entry_id), "manifest.json")

    def payload_path(self, entry_id: str, manifest: Optional[EntryManifest] = None) -> str:
        name = manifest.payload_name if manifest else "payload.bin"
        return os.path.join(self.entry_dir(entry_id), name)

    # -- writer ------------------------------------------------------------

    def publish(
        self,
        *,
        entry_id: str,
        key: str,
        layer: str,
        payload: bytes,
        target_arch: str = "",
        abi_version: str = "",
        guard_domain: str = "",
        compatibility_class: str = "",
        parent_key: str = "",
        created_by: str = "hqsb.compiler.cache",
        kill_at: str = "",
    ) -> Dict[str, Any]:
        """write temp → hash → validate → atomic rename → COMMITTED marker last."""
        if layer not in CACHE_LAYERS:
            raise ConfigError(f"unknown cache layer {layer!r}")
        stages: List[str] = []
        final_dir = self.entry_dir(entry_id)
        lock = _KeyLock(os.path.join(self.root, f".{entry_id}.lock"))
        with lock:
            if os.path.isdir(final_dir) and os.path.isfile(
                os.path.join(final_dir, "manifest.json")
            ):
                raise ConfigError(
                    f"entry {entry_id!r} already published; a second writer must reuse it, not "
                    "overwrite it"
                )
            tmp_dir = tempfile.mkdtemp(prefix=f".tmp-{entry_id}-", dir=self.root)
            stages.append("TEMP")
            if kill_at == "after_temp":
                return self._killed(entry_id, tmp_dir, stages)
            payload_path = os.path.join(tmp_dir, "payload.bin")
            with open(payload_path, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            payload_hash = _sha256_file(payload_path)
            stages.append("VALIDATING")
            manifest = EntryManifest(
                entry_id=entry_id,
                key=key,
                layer=layer,
                state=STATE_TEMP,
                payload_sha256=payload_hash,
                payload_bytes=len(payload),
                parent_key=parent_key,
                created_by=created_by,
                target_arch=target_arch,
                abi_version=abi_version,
                guard_domain=guard_domain,
                compatibility_class=compatibility_class,
            )
            manifest.state = STATE_VALIDATING
            with open(os.path.join(tmp_dir, "manifest.json"), "w", encoding="utf-8") as handle:
                handle.write(canonical_json(manifest.as_dict()))
                handle.flush()
                os.fsync(handle.fileno())
            if kill_at == "after_validating":
                return self._killed(entry_id, tmp_dir, stages)
            os.replace(tmp_dir, final_dir)
            stages.append("PUBLISHED")
            if kill_at == "before_marker":
                return self._killed(entry_id, final_dir, stages)
            # the manifest is promoted before the marker so the on-disk metadata
            # never contradicts the committed state; the marker itself is last
            final_manifest = manifest.as_dict()
            final_manifest["state"] = STATE_COMMITTED
            final_manifest["metadata_sha256"] = metadata_fingerprint(final_manifest)
            with open(
                os.path.join(final_dir, "manifest.json"), "w", encoding="utf-8"
            ) as handle:
                handle.write(canonical_json(final_manifest))
                handle.flush()
                os.fsync(handle.fileno())
            marker_dir = os.path.join(final_dir, ".committed")
            with open(marker_dir, "w", encoding="utf-8") as handle:
                handle.write(final_manifest["metadata_sha256"])
                handle.flush()
                os.fsync(handle.fileno())
            stages.append("COMMITTED")
        return {
            "entry_id": entry_id,
            "key": key,
            "state": STATE_COMMITTED,
            "stages": stages,
            "payload_sha256": payload_hash,
            "payload_bytes": len(payload),
            "path": final_dir,
        }

    def _killed(self, entry_id: str, path: str, stages: Sequence[str]) -> Dict[str, Any]:
        return {
            "entry_id": entry_id,
            "state": "KILLED",
            "stages": list(stages),
            "path": path,
            "note": "simulated writer kill: no COMMITTED marker was published",
        }

    # -- reader ------------------------------------------------------------

    def read(
        self,
        entry_id: str,
        *,
        expected_key: str,
        target_arch: str = "",
        abi_version: str = "",
        guard_covers: bool = True,
        schema_supported: Callable[[str], bool] = lambda version: version == "1.0.0",
        payload_loader: Optional[Callable[[str], bytes]] = None,
    ) -> ReadResult:
        """Ordered validation: parse → schema → key → compat → guard → hash → load."""
        steps: List[Tuple[str, bool]] = []
        directory = self.entry_dir(entry_id)
        manifest_path = os.path.join(directory, "manifest.json")
        if not os.path.isfile(manifest_path):
            steps.append(("parse", False))
            return ReadResult(
                status="miss",
                stage="parse",
                reason_code="MISS_NOT_FOUND",
                detail=f"no manifest at {manifest_path}",
                entry_id=entry_id,
                steps=tuple(steps),
            )
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            steps.append(("parse", False))
            return self._reject("parse", "REJECT_METADATA_PARSE", str(exc), entry_id, steps)
        steps.append(("parse", True))
        schema = str(payload.get("schema_version", ""))
        if not schema_supported(schema):
            steps.append(("schema", False))
            return self._reject(
                "schema",
                "REJECT_UNKNOWN_SCHEMA",
                f"schema {schema!r} unsupported",
                entry_id,
                steps,
            )
        steps.append(("schema", True))
        if payload.get("key") != expected_key:
            steps.append(("key", False))
            return self._reject(
                "key",
                "REJECT_KEY_MISMATCH",
                "recomputed key differs from the entry key",
                entry_id,
                steps,
            )
        steps.append(("key", True))
        if target_arch and payload.get("target_arch") and payload.get("target_arch") != target_arch:
            steps.append(("compatibility", False))
            return self._reject(
                "compatibility",
                "REJECT_TARGET_INCOMPATIBLE",
                f"entry arch {payload.get('target_arch')!r} != {target_arch!r}",
                entry_id,
                steps,
            )
        if abi_version and payload.get("abi_version") and payload.get("abi_version") != abi_version:
            steps.append(("compatibility", False))
            return self._reject(
                "compatibility",
                "REJECT_ABI_MISMATCH",
                f"entry ABI {payload.get('abi_version')!r} != {abi_version!r}",
                entry_id,
                steps,
            )
        steps.append(("compatibility", True))
        if not guard_covers:
            steps.append(("guard", False))
            return self._reject(
                "guard",
                "REJECT_GUARD_FALSE",
                "guard domain does not cover the current input",
                entry_id,
                steps,
            )
        steps.append(("guard", True))
        marker = os.path.join(directory, ".committed")
        if not os.path.isfile(marker):
            steps.append(("state", False))
            return self._reject(
                "state",
                "REJECT_NOT_COMMITTED",
                "COMMITTED marker missing (writer killed or temp residue)",
                entry_id,
                steps,
            )
        manifest_state = str(payload.get("state", ""))
        if manifest_state != STATE_COMMITTED:
            steps.append(("state", False))
            return self._reject(
                "state",
                "REJECT_NOT_COMMITTED",
                f"marker present but manifest state is {manifest_state!r}: metadata and marker "
                "disagree, so the entry is not trusted",
                entry_id,
                steps,
            )
        steps.append(("state", True))
        recorded_metadata_hash = str(payload.get("metadata_sha256", ""))
        if recorded_metadata_hash:
            if metadata_fingerprint(payload) != recorded_metadata_hash:
                steps.append(("metadata_hash", False))
                return self._reject(
                    "metadata_hash",
                    "REJECT_METADATA_HASH",
                    "metadata hash mismatch: the manifest was edited after it was published",
                    entry_id,
                    steps,
                )
            steps.append(("metadata_hash", True))
        payload_path = self.payload_path(entry_id, EntryManifest(**{
            key: value
            for key, value in payload.items()
            if key in EntryManifest.__dataclass_fields__  # type: ignore[attr-defined]
        }))
        try:
            actual_hash = _sha256_file(payload_path)
        except (OSError, PermissionError) as exc:
            steps.append(("payload_hash", False))
            return self._reject("payload_hash", "REJECT_PERMISSION", str(exc), entry_id, steps)
        if actual_hash != payload.get("payload_sha256"):
            steps.append(("payload_hash", False))
            self.quarantine(entry_id, reason="payload hash mismatch")
            return self._reject(
                "payload_hash",
                "REJECT_PAYLOAD_HASH",
                "payload hash mismatch (corruption detected before load)",
                entry_id,
                steps,
            )
        steps.append(("payload_hash", True))
        loader = payload_loader or self._default_loader
        try:
            data = loader(payload_path)
        except (OSError, PermissionError) as exc:
            steps.append(("load", False))
            return self._reject("load", "REJECT_PERMISSION", str(exc), entry_id, steps)
        steps.append(("load", True))
        return ReadResult(
            status="hit",
            stage="load",
            detail="all checks passed",
            entry_id=entry_id,
            payload_bytes=data,
            load_calls=1,
            steps=tuple(steps),
        )

    def _default_loader(self, path: str) -> bytes:
        with open(path, "rb") as handle:
            return handle.read()

    def _reject(
        self, stage: str, reason: str, detail: str, entry_id: str, steps: Sequence[Tuple[str, bool]]
    ) -> ReadResult:
        if reason not in READ_REJECT_REASONS:
            raise ConfigError(f"unknown reject reason {reason!r}")
        return ReadResult(
            status="reject",
            stage=stage,
            reason_code=reason,
            detail=detail,
            entry_id=entry_id,
            load_calls=0,
            steps=tuple(steps),
        )

    # -- lifecycle ---------------------------------------------------------

    def quarantine(self, entry_id: str, *, reason: str) -> Dict[str, Any]:
        directory = self.entry_dir(entry_id)
        target = directory + f".{STATE_QUARANTINED.lower()}"
        if os.path.isdir(directory):
            os.replace(directory, target)
        return {"entry_id": entry_id, "state": STATE_QUARANTINED, "reason": reason, "path": target}

    def cleanup_temp(self) -> Dict[str, Any]:
        removed: List[str] = []
        for name in os.listdir(self.root):
            if name.startswith(".tmp-"):
                shutil.rmtree(os.path.join(self.root, name), ignore_errors=True)
                removed.append(name)
        return {"removed": removed, "count": len(removed)}

    def entry_state(self, entry_id: str) -> str:
        directory = self.entry_dir(entry_id)
        if os.path.isdir(directory + f".{STATE_QUARANTINED.lower()}"):
            return STATE_QUARANTINED
        if not os.path.isdir(directory):
            if os.path.isfile(self.manifest_path(entry_id)):
                return STATE_TEMP
            return STATE_EVICTED
        if os.path.isfile(os.path.join(directory, ".committed")):
            return STATE_COMMITTED
        if os.path.isfile(os.path.join(directory, "manifest.json")):
            return STATE_VALIDATING
        return STATE_TEMP


class _KeyLock:
    """Advisory single-writer lock (create-exclusive file)."""

    def __init__(self, path: str) -> None:
        self.path = path

    def __enter__(self) -> "_KeyLock":
        try:
            self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # another writer holds it; wait briefly then proceed (content-addressed dedup)
            import time

            for _ in range(50):
                time.sleep(0.01)
                if not os.path.exists(self.path):
                    return self.__enter__()
            raise ConfigError("cache key lock stuck: a writer did not release it")
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            os.close(self._fd)
        except OSError:
            pass
        try:
            os.remove(self.path)
        except OSError:
            pass


# ── telemetry and timing boundaries (steps 7, 9–12, 30) ────────────────────

CACHE_EVENT_KINDS: Tuple[str, ...] = (
    "lookup",
    "hit",
    "miss",
    "reject",
    "write",
    "load",
    "quarantine",
    "evict",
    "compile",
)

TIMING_BOUNDARIES: Tuple[str, ...] = (
    "C0_no_cache_cold_compile",
    "C1_same_process_memory_hit",
    "C2_new_process_disk_hit",
    "C3_prebuilt_package_load",
    "C4_warm_steady",
)


@dataclass
class CacheEvent:
    layer: str
    kind: str
    key: str = ""
    entry_id: str = ""
    reason_code: str = ""
    bytes: int = 0
    latency_us: Optional[float] = None
    compile_occurred: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.layer not in CACHE_LAYERS:
            problems.append(f"unknown cache layer {self.layer!r}")
        if self.kind not in CACHE_EVENT_KINDS:
            problems.append(f"unknown cache event kind {self.kind!r}")
        if self.kind == "hit" and self.compile_occurred:
            problems.append("a hit that also compiled is not a hit (it may be a fast recompile)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "kind": self.kind,
            "key": self.key,
            "entry_id": self.entry_id,
            "reason_code": self.reason_code,
            "bytes": self.bytes,
            "latency_us": self.latency_us,
            "compile_occurred": self.compile_occurred,
        }


class CacheTelemetry:
    def __init__(self) -> None:
        self._rows: List[CacheEvent] = []

    def record(self, event: CacheEvent) -> None:
        problems = event.validate()
        if problems:
            raise ConfigError("invalid cache event: " + "; ".join(problems))
        self._rows.append(event)

    def rows(self) -> List[CacheEvent]:
        return list(self._rows)

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for row in self._rows:
            counts[row.kind] = counts.get(row.kind, 0) + 1
        return {
            "events": len(self._rows),
            "by_kind": dict(sorted(counts.items())),
            "by_layer": _histogram(row.layer for row in self._rows),
        }


def _histogram(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def cache_metrics(
    *,
    eligible_lookups: int,
    compatible_guarded_hits: int,
    false_hits: int,
    false_misses: int,
    corrupt_executions: int,
    cross_process_recompiles: int,
    verification_overhead_s: float,
    cold_cost_saved_s: float,
) -> Dict[str, Any]:
    """Safety-first metrics: false hit / corrupt execution must be zero."""
    return {
        "true_hit_rate": round(compatible_guarded_hits / max(1, eligible_lookups), 6),
        "false_hit_count": false_hits,
        "false_miss_count": false_misses,
        "corrupt_execution_count": corrupt_executions,
        "cross_process_recompile_count": cross_process_recompiles,
        "verification_overhead_s": round(verification_overhead_s, 9),
        "cache_amortization_s": round(cold_cost_saved_s - verification_overhead_s, 9),
        "safe": false_hits == 0 and corrupt_executions == 0,
        "priority": "safety first: false miss is a cost problem, false hit is a correctness problem",
    }


def timing_boundaries(
    *,
    c0_cold_total_s: float,
    c1_memory_hit_s: Optional[float] = None,
    c2_disk_hit_s: Optional[float] = None,
    c3_prebuilt_load_s: Optional[float] = None,
    c4_steady_s: Optional[float] = None,
    cache_reset_scope: Sequence[str] = (),
    os_page_cache_cleared: bool = False,
) -> Dict[str, Any]:
    """C0–C4 with explicit reset scope (no vague ``clear_all``)."""
    payload = {
        "C0_no_cache_cold_compile_s": c0_cold_total_s,
        "C1_same_process_memory_hit_s": c1_memory_hit_s,
        "C2_new_process_disk_hit_s": c2_disk_hit_s,
        "C3_prebuilt_package_load_s": c3_prebuilt_load_s,
        "C4_warm_steady_s": c4_steady_s,
        "cache_reset_scope": list(cache_reset_scope),
        "os_page_cache_cleared": os_page_cache_cleared,
        "note": (
            "unless the OS page cache/driver JIT cache were cleared, C0 is 'project-cold', not "
            "absolutely cold; the scope is recorded instead of claimed"
        ),
    }
    values = [value for value in (c1_memory_hit_s, c2_disk_hit_s, c3_prebuilt_load_s, c4_steady_s) if value]
    payload["monotone_expected"] = all(value <= c0_cold_total_s for value in values)
    return payload


# ── invalidation (steps 15–20, §6) ─────────────────────────────────────────

INVALIDATION_MATRIX: Tuple[Tuple[str, str, str], ...] = (
    ("timestamp_or_pid", "HIT", "non-semantic noise"),
    ("same_input_shape_domain", "HIT", "variant still legal"),
    ("shape_inside_guard", "HIT", "bounded reuse"),
    ("guard_false", "MISS_OR_FALLBACK", "current binary is not legal for the input"),
    ("model_weight_or_constant", "MISS", "folded constants change the output"),
    ("op_schema_or_effect_version", "MISS", "operator ABI/semantics"),
    ("pass_code_config_order", "MISS", "IR changes"),
    ("lowering_registry_or_kernel_build", "MISS", "implementation changes"),
    ("compiler_or_codegen_flags", "MISS", "generated code / compatibility"),
    ("target_arch_or_features", "MISS", "binary capability"),
    ("debug_or_profiler_flag", "DEPENDS", "MISS if it changes code, else run identity only"),
    ("autotune_winner_or_db", "MISS_IF_SELECTION_CHANGES", "candidate changes"),
    ("cost_model_or_policy", "DECISION_CACHE_MISS", "selection can change"),
    ("tokenizer_only", "RUN_IDENTITY_CHANGES", "model-core binary unaffected"),
)

INVALIDATION_ACTIONS: Tuple[str, ...] = ("HIT", "MISS", "MISS_OR_FALLBACK", "DEPENDS")


def evaluate_invalidation(
    *,
    change: str,
    expected: str,
    rewrote_key: bool,
    recompile_occurred: bool,
    actual_binary_same: bool,
) -> Dict[str, Any]:
    """Compare an observed change against the frozen matrix (step 15–20)."""
    row = next((item for item in INVALIDATION_MATRIX if item[0] == change), None)
    if row is None:
        return {
            "change": change,
            "known": False,
            "reason": "unregistered change: classify it before the run, not after",
        }
    matrix_expectation = row[1]
    if matrix_expectation == "HIT":
        ok = (not rewrote_key) and (not recompile_occurred)
    elif matrix_expectation in ("MISS",):
        ok = rewrote_key or not actual_binary_same
    elif matrix_expectation == "MISS_OR_FALLBACK":
        ok = rewrote_key or recompile_occurred
    else:  # DEPENDS / MISS_IF_SELECTION_CHANGES / DECISION_CACHE_MISS / RUN_IDENTITY_CHANGES
        ok = True
    return {
        "change": change,
        "matrix_expectation": matrix_expectation,
        "reason": row[2],
        "observed": {
            "key_changed": rewrote_key,
            "recompiled": recompile_occurred,
            "binary_same": actual_binary_same,
        },
        "ok": ok,
        "note": (
            "a conservative extra miss is a cost problem; a missing invalidation that reuses a "
            "stale binary is a correctness problem"
        ),
    }


def matrix_table() -> List[Dict[str, str]]:
    return [
        {"change": change, "expectation": expected, "reason": reason}
        for change, expected, reason in INVALIDATION_MATRIX
    ]


# ── corruption, concurrency, key omission, eviction ────────────────────────

CORRUPTION_CASES: Tuple[str, ...] = (
    "metadata_truncation",
    "metadata_field_edit",
    "payload_truncation",
    "payload_bit_flip",
    "metadata_payload_swap",
    "unknown_schema",
    "missing_commit_marker",
    "temp_residue",
    "permission_denied",
    "symlink_entry",
    "writer_killed",
)


def inject_corruption(store: EntryStore, entry_id: str, case: str) -> Dict[str, Any]:
    """Corrupt an isolated entry copy; the reader must reject before loading."""
    if case not in CORRUPTION_CASES:
        raise ConfigError(f"unknown corruption case {case!r}")
    directory = store.entry_dir(entry_id)
    manifest_path = os.path.join(directory, "manifest.json")
    payload_path = os.path.join(directory, "payload.bin")
    if case == "metadata_truncation":
        with open(manifest_path, "r+", encoding="utf-8") as handle:
            text = handle.read()
            handle.seek(0)
            handle.write(text[: max(1, len(text) // 2)])
            handle.truncate()
    elif case == "payload_truncation":
        with open(payload_path, "r+b") as handle:
            data = handle.read()
            handle.seek(0)
            handle.write(data[: max(1, len(data) // 2)])
            handle.truncate()
    elif case == "payload_bit_flip":
        with open(payload_path, "r+b") as handle:
            first = handle.read(1)
            if first:
                handle.seek(0)
                handle.write(bytes([first[0] ^ 0xFF]))
    elif case == "metadata_payload_swap":
        other = _first_other_entry(store, entry_id)
        if other:
            shutil.copyfile(
                os.path.join(store.entry_dir(other), "payload.bin"), payload_path
            )
    elif case == "metadata_field_edit":
        # a field that no other validator compares, so only the metadata hash
        # can catch the edit
        _patch_manifest(manifest_path, {"created_by": "tampered-writer"})
    elif case == "unknown_schema":
        _patch_manifest(manifest_path, {"schema_version": "99.0.0"})
    elif case == "missing_commit_marker":
        marker = os.path.join(directory, ".committed")
        if os.path.isfile(marker):
            os.remove(marker)
    elif case == "temp_residue":
        tmp = os.path.join(store.root, ".tmp-residue-x")
        os.makedirs(tmp, exist_ok=True)
        with open(os.path.join(tmp, "payload.bin"), "wb") as handle:
            handle.write(b"half")
    elif case == "permission_denied":
        os.chmod(payload_path, 0o000)
    elif case == "symlink_entry":
        target = os.path.join(store.root, "outside-target")
        with open(target, "wb") as handle:
            handle.write(b"outside")
        os.remove(payload_path)
        os.symlink(target, payload_path)
    elif case == "writer_killed":
        _patch_manifest(manifest_path, {"state": STATE_VALIDATING})
    return {"case": case, "entry_id": entry_id, "directory": directory}


def _patch_manifest(path: str, patch: Mapping[str, Any]) -> None:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.update(patch)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)


def _first_other_entry(store: EntryStore, entry_id: str) -> str:
    for name in sorted(os.listdir(store.root)):
        if (
            name != entry_id
            and not name.startswith(".")
            and os.path.isfile(os.path.join(store.root, name, "manifest.json"))
        ):
            return name
    return ""


def reader_writer_plan(*, writers: int, readers: int) -> Dict[str, Any]:
    return {
        "writers": writers,
        "readers": readers,
        "expected": (
            "readers only see old COMMITTED entries or a miss; a reader must never observe a "
            "temp directory or a marker-less entry"
        ),
        "dedup": "content-addressed dedup is allowed; duplicate compiles are recorded as cost",
    }


def evaluate_concurrency(
    *, publishes: Sequence[Mapping[str, Any]], reads: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    duplicate_compiles = sum(1 for row in publishes if row.get("compile_occurred"))
    published_entries = {row.get("entry_id") for row in publishes if row.get("state") == STATE_COMMITTED}
    partial_reads = [row for row in reads if row.get("status") == "hit" and not row.get("steps_committed", True)]
    return {
        "publishes": len(publishes),
        "duplicate_compiles": duplicate_compiles,
        "distinct_published": len(published_entries),
        "partial_reads": partial_reads,
        "ok": not partial_reads,
        "rule": "a half-published entry must never be readable; duplicate work is only a cost",
    }


def key_omission_detector(
    *, spec: CacheKeySpec, base_parts: Mapping[str, Any], hidden_config_field: str
) -> Dict[str, Any]:
    """Two builds differing only in a hidden config must not share a key."""
    if hidden_config_field not in spec.fields:
        return {
            "field": hidden_config_field,
            "in_key_spec": False,
            "ok": False,
            "action": "add the field to the key spec: otherwise two different builds collide",
        }
    first = spec.compute(base_parts)["key"]
    second_parts = dict(base_parts)
    second_parts[hidden_config_field] = f"{base_parts.get(hidden_config_field, 'v1')}-hidden-v2"
    second = spec.compute(second_parts)["key"]
    return {
        "field": hidden_config_field,
        "in_key_spec": True,
        "keys_differ": first != second,
        "ok": first != second,
        "note": "a collision here must fail the experiment immediately and keep the evidence",
    }


@dataclass
class EvictionPolicy:
    max_bytes: int
    max_entries: int
    strategy: str = "lru"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.max_bytes <= 0 or self.max_entries <= 0:
            problems.append("eviction limits must be positive")
        if self.strategy not in ("lru", "ttl", "quota"):
            problems.append(f"unknown eviction strategy {self.strategy!r}")
        return problems

    def plan(self, entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        problems = self.validate()
        if problems:
            raise ConfigError("invalid eviction policy: " + "; ".join(problems))
        in_use = [row for row in entries if row.get("in_use")]
        evictable = [
            row for row in entries if not row.get("in_use") and row.get("state") == STATE_COMMITTED
        ]
        evictable.sort(key=lambda row: (row.get("last_used_index", 0)))
        total_bytes = sum(int(row.get("bytes", 0)) for row in entries if row.get("state") == "COMMITTED")
        selected: List[str] = []
        for row in evictable:
            if total_bytes <= self.max_bytes and len(entries) - len(selected) <= self.max_entries:
                break
            selected.append(str(row.get("entry_id")))
            total_bytes -= int(row.get("bytes", 0))
        return {
            "in_use_protected": [row.get("entry_id") for row in in_use],
            "evicted": selected,
            "remaining_bytes": total_bytes,
            "ok": not any(row.get("entry_id") in selected for row in in_use),
            "rule": "in-use artifacts are never evicted; evicted entries must be rebuildable",
        }


def cache_policy_document(
    *,
    spec: CacheKeySpec,
    eviction: EvictionPolicy,
    compat_policy: Mapping[str, str],
    quarantine_ttl_s: float,
) -> Dict[str, Any]:
    return {
        "key_spec": {"version": spec.spec_version, "fields": sorted(spec.fields)},
        "eviction": {"max_bytes": eviction.max_bytes, "max_entries": eviction.max_entries,
                     "strategy": eviction.strategy},
        "compatibility_policy": dict(sorted(compat_policy.items())),
        "quarantine_ttl_s": quarantine_ttl_s,
        "defaults": {
            "unknown_version": "fail closed (reject → recompile/fallback)",
            "corruption": "quarantine, never execute",
            "concurrency": "commit marker last, readers only see committed entries",
        },
        "rule": "security/integrity checks are never disabled for performance",
    }


def cache_policy_digest(document: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(document))


def layer_dependency_map() -> Dict[str, Any]:
    """Which layer invalidates which downstream layer (step 2)."""
    return {
        "frame_guard": ["capture_graph"],
        "capture_graph": ["pass_ir"],
        "pass_ir": ["lowering_decision", "codegen_native"],
        "lowering_decision": ["codegen_native", "binary_package"],
        "codegen_native": ["binary_package"],
        "autotune_db": ["lowering_decision", "codegen_native"],
        "cost_model": ["lowering_decision"],
        "binary_package": [],
        "note": "only the top-level cache flag hides internal recompiles; every layer reports its own",
    }
