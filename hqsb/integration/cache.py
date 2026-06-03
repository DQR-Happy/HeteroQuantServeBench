"""Compile identity, cache layers, invalidation and integrity (E06-06).

"Its second call was faster" says nothing about *which* cache answered
(E06-06 §2).  This module separates the layers, freezes the identity inputs,
validates entries **before** anything is deserialised or executed, provides
injection helpers that only ever touch a scratch copy, and computes the
break-even point without hiding the cold-start cost.

Two identities are kept distinct on purpose (E06-06 §5 / details README §8):

* ``graph_identity`` — normalised graph + op schema versions + tensor metadata
  + model/quant policy + rewrite pass/config;
* ``compile_identity`` — graph identity + PyTorch/Inductor/Triton versions +
  compiler flags + backend/ABI/arch + guard set + relevant environment.

Neither is assumed equal to PyTorch's internal cache key; the protocol requires
storing both, which :class:`CacheEntry` supports through ``internal_key``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ArtifactError, ConfigError

# ── compile phases (E06-06 §4) ────────────────────────────────────────────


class CompilePhase:
    IMPORT_EXTENSION_LOAD = "import_extension_load"
    DYNAMO_CAPTURE = "dynamo_capture"
    GUARD_CONSTRUCTION = "guard_construction"
    FX_DECOMP_FUNCTIONALIZATION = "fx_decomp_functionalization"
    HQSB_REWRITE = "hqsb_rewrite"
    AOT_AUTOGRAD = "aot_autograd"
    INDUCTOR_LOWERING = "inductor_lowering"
    CODEGEN = "codegen"
    AUTOTUNE = "autotune"
    HOST_COMPILE_LINK = "host_compile_link"
    BINARY_LOAD = "binary_load"
    FIRST_KERNEL_LOAD = "first_kernel_load"
    FIRST_EXECUTION = "first_execution"
    CACHE_LOOKUP = "cache_lookup"
    CACHE_READ_VALIDATE = "cache_read_validate"

    ALL = (
        IMPORT_EXTENSION_LOAD,
        DYNAMO_CAPTURE,
        GUARD_CONSTRUCTION,
        FX_DECOMP_FUNCTIONALIZATION,
        HQSB_REWRITE,
        AOT_AUTOGRAD,
        INDUCTOR_LOWERING,
        CODEGEN,
        AUTOTUNE,
        HOST_COMPILE_LINK,
        BINARY_LOAD,
        FIRST_KERNEL_LOAD,
        FIRST_EXECUTION,
        CACHE_LOOKUP,
        CACHE_READ_VALIDATE,
    )


@dataclass
class PhaseTimer:
    """Per-phase timing; unknown phases are refused, not silently accepted."""

    timings_ms: Dict[str, float] = field(default_factory=dict)
    started: Dict[str, int] = field(default_factory=dict)

    def start(self, phase: str, timestamp_ns: Optional[int] = None) -> None:
        if phase not in CompilePhase.ALL:
            raise ConfigError(
                f"unknown compile phase {phase!r}",
                details={"field": "phase", "allowed": list(CompilePhase.ALL)},
            )
        self.started[phase] = timestamp_ns if timestamp_ns is not None else time.monotonic_ns()

    def stop(self, phase: str, timestamp_ns: Optional[int] = None) -> float:
        if phase not in self.started:
            raise ConfigError(
                f"phase {phase!r} was never started",
                details={"field": "phase"},
            )
        now = timestamp_ns if timestamp_ns is not None else time.monotonic_ns()
        elapsed_ms = (now - self.started.pop(phase)) / 1_000_000.0
        self.timings_ms[phase] = self.timings_ms.get(phase, 0.0) + elapsed_ms
        return elapsed_ms

    def record(self, phase: str, duration_ms: float) -> None:
        if phase not in CompilePhase.ALL:
            raise ConfigError(f"unknown compile phase {phase!r}", details={"field": "phase"})
        self.timings_ms[phase] = self.timings_ms.get(phase, 0.0) + float(duration_ms)

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phases_ms": dict(sorted(self.timings_ms.items())),
            "total_ms": self.total_ms,
            "unclosed": sorted(self.started),
        }


# ── identities ────────────────────────────────────────────────────────────


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GraphIdentity:
    """HQSB's *audit* graph identity (not an internal framework key)."""

    graph_hash: str
    op_schema_versions: Tuple[Tuple[str, str], ...] = ()
    tensor_metadata: Tuple[str, ...] = ()
    model_id: str = ""
    quant_policy_id: str = ""
    rewrite_spec_id: str = ""

    @property
    def digest(self) -> str:
        return _digest(
            {
                "graph_hash": self.graph_hash,
                "op_schema_versions": list(self.op_schema_versions),
                "tensor_metadata": list(self.tensor_metadata),
                "model_id": self.model_id,
                "quant_policy_id": self.quant_policy_id,
                "rewrite_spec_id": self.rewrite_spec_id,
            }
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "graph_hash": self.graph_hash,
            "op_schema_versions": [list(item) for item in self.op_schema_versions],
            "tensor_metadata": list(self.tensor_metadata),
            "model_id": self.model_id,
            "quant_policy_id": self.quant_policy_id,
            "rewrite_spec_id": self.rewrite_spec_id,
            "digest": self.digest,
        }


def graph_identity(
    graph_hash: str,
    *,
    op_schema_versions: Optional[Mapping[str, str]] = None,
    tensor_metadata: Sequence[str] = (),
    model_id: str = "",
    quant_policy_id: str = "",
    rewrite_spec_id: str = "",
) -> GraphIdentity:
    """Compose the audit graph identity (details README §8)."""
    return GraphIdentity(
        graph_hash=graph_hash,
        op_schema_versions=tuple(sorted((op_schema_versions or {}).items())),
        tensor_metadata=tuple(sorted(tensor_metadata)),
        model_id=model_id,
        quant_policy_id=quant_policy_id,
        rewrite_spec_id=rewrite_spec_id,
    )


@dataclass(frozen=True)
class CompileIdentity:
    """Audit compile identity: graph identity plus toolchain/ABI/guard facts."""

    graph: GraphIdentity
    torch_version: str = ""
    inductor_version: str = ""
    triton_version: str = ""
    compiler_flags: Tuple[str, ...] = ()
    backend_lowering_version: str = ""
    kernel_build_hash: str = ""
    abi_version: str = ""
    target_arch: str = ""
    guard_set_hash: str = ""
    environment_hash: str = ""

    @property
    def digest(self) -> str:
        return _digest(
            {
                "graph": self.graph.digest,
                "torch": self.torch_version,
                "inductor": self.inductor_version,
                "triton": self.triton_version,
                "flags": list(self.compiler_flags),
                "lowering": self.backend_lowering_version,
                "kernel": self.kernel_build_hash,
                "abi": self.abi_version,
                "arch": self.target_arch,
                "guards": self.guard_set_hash,
                "env": self.environment_hash,
            }
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "graph": self.graph.as_dict(),
            "torch_version": self.torch_version,
            "inductor_version": self.inductor_version,
            "triton_version": self.triton_version,
            "compiler_flags": list(self.compiler_flags),
            "backend_lowering_version": self.backend_lowering_version,
            "kernel_build_hash": self.kernel_build_hash,
            "abi_version": self.abi_version,
            "target_arch": self.target_arch,
            "guard_set_hash": self.guard_set_hash,
            "environment_hash": self.environment_hash,
            "digest": self.digest,
        }


# ── cache layers and states ───────────────────────────────────────────────


class CacheLayer:
    PYTHON_SOURCE = "python_source_bytecode"
    DYNAMO_CODE = "dynamo_code"
    AOT_AUTOGRAD = "aot_autograd"
    INDUCTOR_FX = "inductor_fx_graph"
    TRITON_BINARY = "triton_binary"
    HQSB_KERNEL = "hqsb_kernel_binary"
    AUTOTUNE = "autotune"
    GPU_MODULE = "gpu_module_driver"
    ALLOCATOR = "allocator"
    OS_PAGE = "os_page"

    ALL = (
        PYTHON_SOURCE,
        DYNAMO_CODE,
        AOT_AUTOGRAD,
        INDUCTOR_FX,
        TRITON_BINARY,
        HQSB_KERNEL,
        AUTOTUNE,
        GPU_MODULE,
        ALLOCATOR,
        OS_PAGE,
    )


class CacheState:
    COLD_EMPTY = "COLD-EMPTY"
    WARM_SAME_PROCESS = "WARM-SAME-PROCESS"
    WARM_NEW_PROCESS_LOCAL = "WARM-NEW-PROCESS-LOCAL"
    IMPORTED_REMOTE = "IMPORTED/REMOTE"
    DISABLED = "CACHE-DISABLED"
    CORRUPT_PARTIAL = "CORRUPT/PARTIAL"
    NOT_CLAIMED = "NOT_CLAIMED"

    ALL = (
        COLD_EMPTY,
        WARM_SAME_PROCESS,
        WARM_NEW_PROCESS_LOCAL,
        IMPORTED_REMOTE,
        DISABLED,
        CORRUPT_PARTIAL,
        NOT_CLAIMED,
    )


# ── entries, manifests and validation ─────────────────────────────────────


class IntegrityFailure:
    MISSING_FILE = "MISSING_FILE"
    PAYLOAD_TRUNCATED = "PAYLOAD_TRUNCATED"
    HASH_MISMATCH = "HASH_MISMATCH"
    BIT_FLIP = "BIT_FLIP"
    MANIFEST_MISMATCH = "MANIFEST_MISMATCH"
    MISSING_BINARY = "MISSING_BINARY"
    STALE_LOCK = "STALE_LOCK"
    BAD_PERMISSIONS = "BAD_PERMISSIONS"
    FOREIGN_ARCH = "FOREIGN_ARCH"
    STALE_SCHEMA_ABI = "STALE_SCHEMA_ABI"
    PARTIAL_WRITE = "PARTIAL_WRITE"
    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    UNTRUSTED_ARTIFACT = "UNTRUSTED_ARTIFACT"
    UNKNOWN = "UNKNOWN"

    ALL = (
        MISSING_FILE,
        PAYLOAD_TRUNCATED,
        HASH_MISMATCH,
        BIT_FLIP,
        MANIFEST_MISMATCH,
        MISSING_BINARY,
        STALE_LOCK,
        BAD_PERMISSIONS,
        FOREIGN_ARCH,
        STALE_SCHEMA_ABI,
        PARTIAL_WRITE,
        PATH_TRAVERSAL,
        UNTRUSTED_ARTIFACT,
        UNKNOWN,
    )


@dataclass(frozen=True)
class CacheEntry:
    """One cache entry with the identity material needed to validate it."""

    key_digest: str
    layer: str
    entry_hash: str = ""
    payload_path: str = ""
    size_bytes: int = 0
    target_arch: str = ""
    abi_version: str = ""
    schema_hash: str = ""
    lowering_version: str = ""
    complete: bool = True
    created_ns: int = 0
    internal_key: str = ""
    trusted: bool = True

    def __post_init__(self) -> None:
        if self.layer not in CacheLayer.ALL:
            raise ConfigError(
                f"unknown cache layer {self.layer!r}",
                details={"field": "layer", "allowed": list(CacheLayer.ALL)},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key_digest": self.key_digest,
            "layer": self.layer,
            "entry_hash": self.entry_hash,
            "payload_path": self.payload_path,
            "size_bytes": self.size_bytes,
            "target_arch": self.target_arch,
            "abi_version": self.abi_version,
            "schema_hash": self.schema_hash,
            "lowering_version": self.lowering_version,
            "complete": self.complete,
            "created_ns": self.created_ns,
            "internal_key": self.internal_key,
            "trusted": self.trusted,
        }


@dataclass(frozen=True)
class ValidationResult:
    """Entry validation outcome; always evaluated *before* execution."""

    ok: bool
    failure: str = ""
    detail: Any = None
    pre_execution: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "failure": self.failure,
            "detail": self.detail,
            "pre_execution": self.pre_execution,
        }


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_entry(
    entry: CacheEntry,
    root: str,
    *,
    expected_arch: str = "",
    expected_abi: str = "",
    expected_schema_hash: str = "",
) -> ValidationResult:
    """Validate an entry without deserialising or executing anything.

    Order matters: path safety → presence → completeness → platform identity →
    content hash.  A corrupt entry must be detected here, never after it ran
    (E06-06 §7).
    """
    if not entry.trusted:
        return ValidationResult(ok=False, failure=IntegrityFailure.UNTRUSTED_ARTIFACT)
    if entry.payload_path:
        resolved = os.path.abspath(os.path.join(root, entry.payload_path))
        root_abs = os.path.abspath(root)
        if not resolved.startswith(root_abs + os.sep):
            return ValidationResult(
                ok=False,
                failure=IntegrityFailure.PATH_TRAVERSAL,
                detail={"payload_path": entry.payload_path},
            )
    if not entry.complete:
        return ValidationResult(ok=False, failure=IntegrityFailure.PARTIAL_WRITE)
    if expected_arch and entry.target_arch and entry.target_arch != expected_arch:
        return ValidationResult(
            ok=False,
            failure=IntegrityFailure.FOREIGN_ARCH,
            detail={"expected": expected_arch, "actual": entry.target_arch},
        )
    if expected_abi and entry.abi_version and entry.abi_version != expected_abi:
        return ValidationResult(
            ok=False,
            failure=IntegrityFailure.STALE_SCHEMA_ABI,
            detail={"expected": expected_abi, "actual": entry.abi_version},
        )
    if expected_schema_hash and entry.schema_hash and entry.schema_hash != expected_schema_hash:
        return ValidationResult(
            ok=False,
            failure=IntegrityFailure.STALE_SCHEMA_ABI,
            detail={"expected": expected_schema_hash, "actual": entry.schema_hash},
        )
    if not entry.payload_path:
        return ValidationResult(ok=False, failure=IntegrityFailure.MISSING_BINARY)
    payload = os.path.join(root, entry.payload_path)
    if not os.path.isfile(payload):
        return ValidationResult(
            ok=False, failure=IntegrityFailure.MISSING_FILE, detail={"path": entry.payload_path}
        )
    # Check the permission bits explicitly: ``os.access`` answers "can *this*
    # process read it", which is always true for root — a root-run CI would
    # otherwise accept a mode 000 entry.
    mode = os.stat(payload).st_mode
    if not mode & 0o444:
        return ValidationResult(
            ok=False, failure=IntegrityFailure.BAD_PERMISSIONS, detail={"path": entry.payload_path}
        )
    if not os.access(payload, os.R_OK):  # pragma: no cover - non-root defensive check
        return ValidationResult(
            ok=False, failure=IntegrityFailure.BAD_PERMISSIONS, detail={"path": entry.payload_path}
        )
    actual_size = os.path.getsize(payload)
    if entry.size_bytes and actual_size != entry.size_bytes:
        return ValidationResult(
            ok=False,
            failure=IntegrityFailure.PAYLOAD_TRUNCATED,
            detail={"expected": entry.size_bytes, "actual": actual_size},
        )
    if entry.entry_hash:
        digest = sha256_file(payload)
        if digest != entry.entry_hash:
            return ValidationResult(
                ok=False,
                failure=IntegrityFailure.HASH_MISMATCH,
                detail={"expected": entry.entry_hash, "actual": digest},
            )
    return ValidationResult(ok=True)


def atomic_write_bytes(path: str, payload: bytes) -> str:
    """Write via temp file + ``os.replace``; a reader sees old or new, never half."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_text(path: str, text: str) -> str:
    return atomic_write_bytes(path, text.encode("utf-8"))


# ── corruption injection (scratch copies only) ────────────────────────────


class RefusedMutationError(ArtifactError):
    """Injection was attempted outside the scratch area."""


@dataclass
class CorruptionFixture:
    """A scratch copy of a cache entry used for integrity injection.

    The fixture **copies** the source before mutating and refuses to operate
    unless the target lives under the declared ``scratch_root``; golden data is
    therefore structurally unreachable (E06-06 §13 "do not destroy golden").
    """

    source_dir: str
    scratch_root: str
    work_dir: str = ""

    def prepare(self) -> str:
        source = os.path.abspath(self.source_dir)
        scratch = os.path.abspath(self.scratch_root)
        if not os.path.isdir(source):
            raise ArtifactError(f"source entry dir does not exist: {source}")
        os.makedirs(scratch, exist_ok=True)
        target = os.path.join(scratch, "entry-under-test")
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.copytree(source, target)
        self.work_dir = target
        return target

    def _guard(self, path: str) -> str:
        scratch = os.path.abspath(self.scratch_root)
        resolved = os.path.abspath(path)
        if not resolved.startswith(scratch + os.sep):
            raise RefusedMutationError(
                f"refusing to mutate {resolved!r}: it is outside the scratch root {scratch!r}",
                details={"path": resolved, "scratch_root": scratch},
            )
        return resolved

    def truncate(self, payload_name: str, keep_bytes: int = 8) -> str:
        path = self._guard(os.path.join(self.work_dir, payload_name))
        with open(path, "rb") as handle:
            data = handle.read()
        with open(path, "wb") as handle:
            handle.write(data[:keep_bytes])
        return path

    def bit_flip(self, payload_name: str, offset: int = 0) -> str:
        path = self._guard(os.path.join(self.work_dir, payload_name))
        with open(path, "r+b") as handle:
            handle.seek(offset)
            current = handle.read(1)
            handle.seek(offset)
            handle.write(bytes([current[0] ^ 0x01]) if current else b"\x01")
        return path

    def delete(self, payload_name: str) -> None:
        path = self._guard(os.path.join(self.work_dir, payload_name))
        if os.path.isfile(path):
            os.remove(path)

    def chmod(self, payload_name: str, mode: int = 0o000) -> str:
        path = self._guard(os.path.join(self.work_dir, payload_name))
        os.chmod(path, mode)
        return path

    def stale_lock(self, name: str = "entry.lock") -> str:
        path = self._guard(os.path.join(self.work_dir, name))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("stale\n")
        return path


@dataclass(frozen=True)
class InjectionOutcome:
    kind: str
    target: str
    detected: bool
    failure: str = ""
    detail: Any = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "detected": self.detected,
            "failure": self.failure,
            "detail": self.detail,
        }


# ── invalidation matrix ───────────────────────────────────────────────────


class InvalidationAction:
    HIT = "HIT"
    MISS = "MISS"
    REPACK = "REPACK"
    REJECT = "REJECT"

    ALL = (HIT, MISS, REPACK, REJECT)


@dataclass(frozen=True)
class InvalidationFactor:
    """One factor whose single change must produce a known cache action."""

    name: str
    expected_action: str
    semantic: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if self.expected_action not in InvalidationAction.ALL:
            raise ConfigError(
                f"factor {self.name!r}: unknown action {self.expected_action!r}",
                details={"field": "expected_action"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "expected_action": self.expected_action,
            "semantic": self.semantic,
            "note": self.note,
        }


def default_invalidation_matrix() -> Tuple[InvalidationFactor, ...]:
    """The frozen expectations from E06-06 §6 (one factor changed at a time)."""
    return (
        InvalidationFactor("python_source", InvalidationAction.MISS),
        InvalidationFactor("graph_structure", InvalidationAction.MISS),
        InvalidationFactor("pattern_pass_version", InvalidationAction.MISS),
        InvalidationFactor("operator_schema", InvalidationAction.MISS),
        InvalidationFactor("kernel_binary", InvalidationAction.MISS),
        InvalidationFactor("model_revision", InvalidationAction.MISS),
        InvalidationFactor(
            "weight_values_same_shape",
            InvalidationAction.MISS,
            note="runtime weights are not baked into a compiled artifact; the bound "
            "artifact must be re-checked",
        ),
        InvalidationFactor("quant_artifact_hash", InvalidationAction.MISS),
        InvalidationFactor("quant_group_or_layout", InvalidationAction.REPACK),
        InvalidationFactor("dynamic_bounds", InvalidationAction.MISS),
        InvalidationFactor("dtype", InvalidationAction.MISS),
        InvalidationFactor("device_or_stride", InvalidationAction.MISS),
        InvalidationFactor("pytorch_patch", InvalidationAction.MISS),
        InvalidationFactor("triton_inductor", InvalidationAction.MISS),
        InvalidationFactor("cuda_toolkit_driver", InvalidationAction.MISS),
        InvalidationFactor("gpu_arch", InvalidationAction.REJECT),
        InvalidationFactor("compiler_flags", InvalidationAction.MISS),
        InvalidationFactor("capability_env", InvalidationAction.MISS),
        InvalidationFactor("backend_preference", InvalidationAction.MISS),
        InvalidationFactor("quality_policy", InvalidationAction.MISS),
        InvalidationFactor(
            "irrelevant_metadata_control",
            InvalidationAction.HIT,
            semantic=False,
            note="a control that must NOT invalidate anything",
        ),
    )


@dataclass(frozen=True)
class InvalidationObservation:
    factor: str
    expected_action: str
    actual_action: str
    new_key: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.expected_action == self.actual_action

    def as_dict(self) -> Dict[str, Any]:
        return {
            "factor": self.factor,
            "expected_action": self.expected_action,
            "actual_action": self.actual_action,
            "new_key": self.new_key,
            "detail": self.detail,
            "ok": self.ok,
        }


def evaluate_invalidation(
    factors: Sequence[InvalidationFactor],
    observations: Sequence[InvalidationObservation],
) -> Dict[str, Any]:
    """Compare expected vs actual for every factor; unfavourable rows stay."""
    by_factor = {item.factor: item for item in observations}
    rows = []
    for factor in factors:
        observed = by_factor.get(factor.name)
        row = dict(factor.as_dict())
        row["factor"] = row.pop("name")
        row["actual_action"] = observed.actual_action if observed else "<NOT_OBSERVED>"
        row["ok"] = bool(observed and observed.ok)
        row["detail"] = observed.detail if observed else "no observation recorded"
        rows.append(row)
    mismatches = [row for row in rows if not row["ok"]]
    return {
        "rows": rows,
        "mismatches": mismatches,
        "ok": not mismatches,
        "observed": len(observations),
        "required": len(factors),
    }


# ── break-even ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BreakEven:
    """Amortisation point; ``None`` means no positive amortisation exists."""

    total_extra_compile_cost_ms: float
    eager_latency_ms: float
    compiled_steady_latency_ms: float
    requests: Optional[int]
    reason: str = ""

    @property
    def per_request_saving_ms(self) -> float:
        return self.eager_latency_ms - self.compiled_steady_latency_ms

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_extra_compile_cost_ms": self.total_extra_compile_cost_ms,
            "eager_latency_ms": self.eager_latency_ms,
            "compiled_steady_latency_ms": self.compiled_steady_latency_ms,
            "per_request_saving_ms": self.per_request_saving_ms,
            "requests": self.requests,
            "reason": self.reason,
        }


def compute_break_even(
    total_extra_compile_cost_ms: float,
    eager_latency_ms: float,
    compiled_steady_latency_ms: float,
) -> BreakEven:
    """``N = extra_compile_cost / (eager - compiled_steady)`` with an explicit
    "no amortisation point" result when the denominator is not positive."""
    saving = eager_latency_ms - compiled_steady_latency_ms
    if saving <= 0:
        return BreakEven(
            total_extra_compile_cost_ms=total_extra_compile_cost_ms,
            eager_latency_ms=eager_latency_ms,
            compiled_steady_latency_ms=compiled_steady_latency_ms,
            requests=None,
            reason="DENOMINATOR_NON_POSITIVE: compiled steady state is not faster, so no "
            "request count amortises the compile cost",
        )
    if total_extra_compile_cost_ms <= 0:
        return BreakEven(
            total_extra_compile_cost_ms=total_extra_compile_cost_ms,
            eager_latency_ms=eager_latency_ms,
            compiled_steady_latency_ms=compiled_steady_latency_ms,
            requests=0,
            reason="NO_EXTRA_COMPILE_COST_RECORDED",
        )
    return BreakEven(
        total_extra_compile_cost_ms=total_extra_compile_cost_ms,
        eager_latency_ms=eager_latency_ms,
        compiled_steady_latency_ms=compiled_steady_latency_ms,
        requests=int(math.ceil(total_extra_compile_cost_ms / saving)),
    )


# ── cache directory + concurrency ─────────────────────────────────────────


@dataclass
class CacheManifest:
    """Manifest of the entries in one cache directory."""

    state: str
    entries: List[CacheEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state not in CacheState.ALL:
            raise ConfigError(
                f"unknown cache state {self.state!r}",
                details={"field": "state", "allowed": list(CacheState.ALL)},
            )

    def add(self, entry: CacheEntry) -> None:
        self.entries.append(entry)

    def find(self, key_digest: str, layer: str = "") -> Optional[CacheEntry]:
        for entry in self.entries:
            if entry.key_digest == key_digest and (not layer or entry.layer == layer):
                return entry
        return None

    @property
    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries)

    @property
    def layers(self) -> Tuple[str, ...]:
        return tuple(sorted({entry.layer for entry in self.entries}))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "entry_count": len(self.entries),
            "total_bytes": self.total_bytes,
            "layers": list(self.layers),
            "entries": [entry.as_dict() for entry in self.entries],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=2)


def save_manifest(path: str, manifest: CacheManifest) -> str:
    return atomic_write_text(path, manifest.to_json())


def load_manifest(path: str) -> CacheManifest:
    if not os.path.isfile(path):
        raise ArtifactError(f"cache manifest not found: {path}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    state = payload.get("state", CacheState.COLD_EMPTY)
    if state not in CacheState.ALL:
        raise ArtifactError(
            f"{path}: unknown cache state {state!r}",
            details={"field": "state", "allowed": list(CacheState.ALL)},
        )
    manifest = CacheManifest(state=state)
    for item in payload.get("entries", ()):
        manifest.add(
            CacheEntry(
                key_digest=str(item["key_digest"]),
                layer=str(item["layer"]),
                entry_hash=str(item.get("entry_hash", "")),
                payload_path=str(item.get("payload_path", "")),
                size_bytes=int(item.get("size_bytes", 0)),
                target_arch=str(item.get("target_arch", "")),
                abi_version=str(item.get("abi_version", "")),
                schema_hash=str(item.get("schema_hash", "")),
                lowering_version=str(item.get("lowering_version", "")),
                complete=bool(item.get("complete", True)),
                created_ns=int(item.get("created_ns", 0)),
                internal_key=str(item.get("internal_key", "")),
                trusted=bool(item.get("trusted", True)),
            )
        )
    return manifest


class CacheLockError(ArtifactError):
    """A lock could not be acquired (timeout, or a stale lock was refused)."""


@dataclass
class CacheLock:
    """A simple lock file with stale detection (atomic create, no hidden state)."""

    path: str
    stale_after_s: float = 300.0

    def acquire(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(self.path):
            age = time.time() - os.path.getmtime(self.path)
            if age <= self.stale_after_s:
                raise CacheLockError(
                    f"cache lock {self.path!r} is held (age {age:.1f}s); refusing to "
                    "proceed without an explicit stale decision",
                    details={"path": self.path, "age_s": age},
                )
            # Stale locks are removed deliberately, with the reason recorded by
            # the caller; never silently ignored.
            os.remove(self.path)
        handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(handle, str(os.getpid()).encode("utf-8"))
        os.close(handle)

    def release(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)

    def __enter__(self) -> "CacheLock":
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


@dataclass
class CacheSizeTracker:
    """Tracks entry/byte growth against a cap (E06-06 §10 step 16)."""

    cap_entries: int
    cap_bytes: int

    def evaluate(self, manifest: CacheManifest) -> Dict[str, Any]:
        over_entries = len(manifest.entries) > self.cap_entries
        over_bytes = manifest.total_bytes > self.cap_bytes
        return {
            "entries": len(manifest.entries),
            "bytes": manifest.total_bytes,
            "cap_entries": self.cap_entries,
            "cap_bytes": self.cap_bytes,
            "over_cap": bool(over_entries or over_bytes),
            "reason": (
                "CACHE_CAP_EXCEEDED" if (over_entries or over_bytes) else "WITHIN_CAP"
            ),
        }


def evict(manifest: CacheManifest, cap_entries: int) -> List[CacheEntry]:
    """Deterministic eviction: oldest ``created_ns`` first; returns evicted entries."""
    if len(manifest.entries) <= cap_entries:
        return []
    ordered = sorted(manifest.entries, key=lambda entry: (entry.created_ns, entry.key_digest))
    evicted = ordered[: len(manifest.entries) - cap_entries]
    remaining = [entry for entry in ordered if entry not in evicted]
    manifest.entries = remaining
    return evicted


__all__ = [
    "BreakEven",
    "CacheEntry",
    "CacheLayer",
    "CacheLock",
    "CacheLockError",
    "CacheManifest",
    "CacheSizeTracker",
    "CacheState",
    "CompileIdentity",
    "CompilePhase",
    "CorruptionFixture",
    "GraphIdentity",
    "InjectionOutcome",
    "IntegrityFailure",
    "InvalidationAction",
    "InvalidationFactor",
    "InvalidationObservation",
    "PhaseTimer",
    "RefusedMutationError",
    "ValidationResult",
    "atomic_write_bytes",
    "atomic_write_text",
    "compute_break_even",
    "default_invalidation_matrix",
    "evaluate_invalidation",
    "evict",
    "graph_identity",
    "load_manifest",
    "save_manifest",
    "sha256_file",
    "validate_entry",
]
