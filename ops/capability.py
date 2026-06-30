"""Unified backend capability detection for the operator dispatcher (S04).

S04's contract is explicit: *"Jetson 对 Triton 支持需能力检测，不能默认"* and
*"未安装/不支持的 DSL 走明确 fallback，不影响 CPU/CUDA 核心包"*. This module
is the single source of truth for what is actually usable at runtime, and it
never raises — an unavailable backend simply reports ``False``.

Each probe does *real* work where cheap (e.g. Triton compiles a 1-element
kernel to prove the backend works, not merely that the package imports),
because ``import triton`` succeeding does not guarantee the JIT backend can
compile for the installed GPU.

The legacy :func:`detect_capabilities` aggregate is retained for the S04
operator dispatcher.  E04-01 additionally requires a structured, replayable
failure contract; :class:`CapabilityResult`, :class:`CapabilityCache`, and
:func:`resolve_backend` provide that contract without importing an optional
backend at module import time.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import json
import importlib.util
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

from hqsb.core.errors import BackendError, CapabilityError, ExitCode


CAPABILITY_SCHEMA_VERSION = "hqsb.capability/v1"
CAPABILITY_POLICY_VERSION = "e04-01-v1"


class CapabilityStage(str, Enum):
    """Ordered stages at which an optional backend probe can stop."""

    DISCOVERY = "DISCOVERY"
    IMPORT = "IMPORT"
    VERSION = "VERSION"
    DEVICE = "DEVICE"
    COMPILER = "COMPILER"
    COMPILE = "COMPILE"
    LOAD = "LOAD"
    EXECUTE = "EXECUTE"
    RESOURCE = "RESOURCE"
    POLICY = "POLICY"


class CapabilityReason(str, Enum):
    """Stable reason codes; human-readable ``detail`` is never a key."""

    AVAILABLE = "AVAILABLE"
    PACKAGE_NOT_INSTALLED = "PACKAGE_NOT_INSTALLED"
    SHARED_LIBRARY_NOT_FOUND = "SHARED_LIBRARY_NOT_FOUND"
    VERSION_INCOMPATIBLE = "VERSION_INCOMPATIBLE"
    ABI_MISMATCH = "ABI_MISMATCH"
    DEVICE_UNAVAILABLE = "DEVICE_UNAVAILABLE"
    ARCH_UNSUPPORTED = "ARCH_UNSUPPORTED"
    COMPILER_UNAVAILABLE = "COMPILER_UNAVAILABLE"
    COMPILE_FAILED = "COMPILE_FAILED"
    MODULE_LOAD_FAILED = "MODULE_LOAD_FAILED"
    SYMBOL_MISSING = "SYMBOL_MISSING"
    RUNTIME_FAILED = "RUNTIME_FAILED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    TIMEOUT = "TIMEOUT"
    DISABLED_BY_POLICY = "DISABLED_BY_POLICY"
    PROBE_INTERNAL_ERROR = "PROBE_INTERNAL_ERROR"


_DETERMINISTIC_REASONS = frozenset(
    {
        CapabilityReason.PACKAGE_NOT_INSTALLED,
        CapabilityReason.SHARED_LIBRARY_NOT_FOUND,
        CapabilityReason.VERSION_INCOMPATIBLE,
        CapabilityReason.ABI_MISMATCH,
        CapabilityReason.DEVICE_UNAVAILABLE,
        CapabilityReason.ARCH_UNSUPPORTED,
        CapabilityReason.COMPILER_UNAVAILABLE,
        CapabilityReason.SYMBOL_MISSING,
        CapabilityReason.DISABLED_BY_POLICY,
    }
)


@dataclass(frozen=True)
class CapabilityIdentity:
    """All identities that can invalidate a capability decision."""

    device_identity: str
    arch: Optional[Tuple[int, int]]
    package_version: Optional[str]
    runtime_version: Optional[str]
    compiler_version: Optional[str]
    build_identity: str
    policy_version: str = CAPABILITY_POLICY_VERSION

    def as_dict(self) -> dict:
        return {
            "device_identity": self.device_identity,
            "arch": list(self.arch) if self.arch else None,
            "package_version": self.package_version,
            "runtime_version": self.runtime_version,
            "compiler_version": self.compiler_version,
            "build_identity": self.build_identity,
            "policy_version": self.policy_version,
        }

    def cache_key(self, backend: str) -> str:
        payload = {"backend": backend, **self.as_dict()}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProbeStageResult:
    stage: CapabilityStage
    duration_ms: float
    status: str

    def as_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "duration_ms": self.duration_ms,
            "status": self.status,
        }


@dataclass(frozen=True)
class CapabilityResult:
    """Structured result for one backend capability probe."""

    backend: str
    available: bool
    stage: CapabilityStage
    reason_code: CapabilityReason
    detail: str
    retryable: bool
    identity: CapabilityIdentity
    supported_dtypes: Tuple[str, ...] = ()
    supported_layouts: Tuple[str, ...] = ()
    supported_features: Tuple[str, ...] = ()
    probe_stages: Tuple[ProbeStageResult, ...] = ()
    cause_chain: Tuple[str, ...] = ()
    probed_at: float = field(default_factory=time.time)
    from_cache: bool = False

    def __post_init__(self) -> None:
        if self.available and self.reason_code is not CapabilityReason.AVAILABLE:
            raise ValueError("an available backend must use reason AVAILABLE")
        if not self.available and self.reason_code is CapabilityReason.AVAILABLE:
            raise ValueError("an unavailable backend needs a failure reason")

    @property
    def deterministic(self) -> bool:
        return self.reason_code in _DETERMINISTIC_REASONS

    @property
    def exit_code(self) -> int:
        if self.available:
            return ExitCode.SUCCESS
        if self.stage in {
            CapabilityStage.DISCOVERY,
            CapabilityStage.IMPORT,
            CapabilityStage.VERSION,
            CapabilityStage.DEVICE,
            CapabilityStage.COMPILER,
            CapabilityStage.POLICY,
        }:
            return ExitCode.CAPABILITY
        return ExitCode.BACKEND

    def as_dict(self) -> dict:
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION,
            "backend": self.backend,
            "available": self.available,
            "stage": self.stage.value,
            "reason_code": self.reason_code.value,
            "detail": self.detail,
            "retryable": self.retryable,
            **self.identity.as_dict(),
            "binary_or_cache_identity": self.identity.build_identity,
            "supported_dtypes": list(self.supported_dtypes),
            "supported_layouts": list(self.supported_layouts),
            "supported_features": list(self.supported_features),
            "probe_stages": [item.as_dict() for item in self.probe_stages],
            "cause_chain": list(self.cause_chain),
            "probed_at": self.probed_at,
            "from_cache": self.from_cache,
            "deterministic": self.deterministic,
            "exit_code": self.exit_code,
        }

    def with_cache_hit(self) -> "CapabilityResult":
        return CapabilityResult(
            **{
                **self.__dict__,
                "from_cache": True,
            }
        )


@dataclass(frozen=True)
class BackendDecision:
    """Auditable forced/auto selection result."""

    requested: str
    actual: Optional[str]
    reason_code: str
    fallback: bool
    candidate_results: Tuple[CapabilityResult, ...]

    def as_dict(self) -> dict:
        return {
            "requested": self.requested,
            "actual": self.actual,
            "reason_code": self.reason_code,
            "fallback": self.fallback,
            "candidate_results": [result.as_dict() for result in self.candidate_results],
        }


class CapabilityCacheCorruption(ValueError):
    """Raised when a serialized capability cache fails integrity checks."""


class CapabilityCache:
    """Identity-bound, single-flight cache with failure-aware persistence.

    Successful probes and deterministic incompatibilities are cached.  A
    transient compile timeout/runtime/OOM result is returned to the caller but
    deliberately not retained, so the next request can recover.
    """

    def __init__(self) -> None:
        self._values: Dict[str, CapabilityResult] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock_for(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def get_or_probe(
        self,
        backend: str,
        identity: CapabilityIdentity,
        probe: Callable[[], CapabilityResult],
    ) -> CapabilityResult:
        key = identity.cache_key(backend)
        cached = self._values.get(key)
        if cached is not None:
            return cached.with_cache_hit()
        with self._lock_for(key):
            cached = self._values.get(key)
            if cached is not None:
                return cached.with_cache_hit()
            result = probe()
            if result.backend != backend or result.identity != identity:
                raise ValueError("probe result identity does not match cache request")
            if result.available or result.deterministic:
                self._values[key] = result
            return result

    def clear(self) -> None:
        with self._guard:
            self._values.clear()

    def to_payload(self) -> dict:
        values = {key: value.as_dict() for key, value in sorted(self._values.items())}
        digest = hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION,
            "values": values,
            "sha256": digest,
        }

    @staticmethod
    def validate_payload(payload: Mapping[str, object]) -> None:
        if payload.get("schema_version") != CAPABILITY_SCHEMA_VERSION:
            raise CapabilityCacheCorruption("capability cache schema mismatch")
        values = payload.get("values")
        if not isinstance(values, dict):
            raise CapabilityCacheCorruption("capability cache values are not an object")
        actual = hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if payload.get("sha256") != actual:
            raise CapabilityCacheCorruption("capability cache digest mismatch")


def unavailable_result(
    backend: str,
    identity: CapabilityIdentity,
    stage: CapabilityStage,
    reason_code: CapabilityReason,
    detail: str,
    *,
    retryable: bool,
    cause_chain: Iterable[str] = (),
) -> CapabilityResult:
    """Construct an unavailable result while enforcing the stable taxonomy."""
    return CapabilityResult(
        backend=backend,
        available=False,
        stage=stage,
        reason_code=reason_code,
        detail=detail,
        retryable=retryable,
        identity=identity,
        cause_chain=tuple(cause_chain),
    )


def _forced_error(result: CapabilityResult) -> Exception:
    details = result.as_dict()
    message = (
        f"forced backend {result.backend!r} unavailable at {result.stage.value}: "
        f"{result.reason_code.value}: {result.detail}"
    )
    if result.exit_code == ExitCode.CAPABILITY:
        return CapabilityError(message, details=details)
    return BackendError(message, details=details)


def resolve_backend(
    requested: str,
    candidates: Iterable[CapabilityResult],
    *,
    reference_backend: str = "reference",
) -> BackendDecision:
    """Resolve an ``auto`` or forced request without silent fallback.

    ``auto`` evaluates every candidate in order and records every rejection.
    A forced request either returns that exact backend or raises a stable HQSB
    error (exit 7 for unsupported capability, exit 6 for operational failure).
    """
    results = tuple(candidates)
    by_name = {result.backend: result for result in results}
    if requested != "auto":
        result = by_name.get(requested)
        if result is None:
            identity = CapabilityIdentity("unknown", None, None, None, None, "unknown")
            raise CapabilityError(
                f"forced backend {requested!r} is not registered",
                details=unavailable_result(
                    requested,
                    identity,
                    CapabilityStage.DISCOVERY,
                    CapabilityReason.PACKAGE_NOT_INSTALLED,
                    "backend not registered",
                    retryable=False,
                ).as_dict(),
            )
        if not result.available:
            raise _forced_error(result)
        return BackendDecision(requested, requested, "FORCED_AVAILABLE", False, results)

    for result in results:
        if result.available:
            return BackendDecision("auto", result.backend, "FIRST_AVAILABLE", False, results)
    primary = results[0].reason_code.value if results else "NO_CANDIDATE"
    return BackendDecision("auto", reference_backend, primary, True, results)

# Default search path for the CUDA RMSNorm shared library (built by CMake).
_DEFAULT_CUDA_LIB_GLOB = "build/*/ops/cuda/rmsnorm/libhqsb_rmsnorm_shared.so"

# CUTLASS is header-only; these are the canonical include locations, plus an
# override environment variable so a CI/cloud machine can point at its own
# checkout. The in-repo ``third_party/cutlass`` is resolved at detection time
# (relative to the repo root), so it is appended programmatically below.
_DEFAULT_CUTLASS_PATHS = (
    "/usr/local/cutlass/include",
    "/opt/cutlass/include",
)


@dataclass(frozen=True)
class BackendCapabilities:
    """The runtime-usability snapshot consumed by the dispatcher."""

    cuda_available: bool
    device_capability: Optional[Tuple[int, int]]
    triton_available: bool
    triton_version: Optional[str]
    cutlass_available: bool
    cutlass_include_dir: Optional[str]
    tilelang_available: bool
    tilelang_version: Optional[str]
    cublas_available: bool
    cuda_rmsnorm_available: bool
    cuda_rmsnorm_lib: Optional[str]
    notes: Tuple[str, ...] = field(default_factory=tuple)
    # Compute capability the CUDA shared library was actually *compiled* for,
    # read from the library's own C ABI. ``None`` means the build arch could
    # not be determined, which callers must treat as "not proven usable"
    # rather than as a match (see ``ops.dispatcher``).
    cuda_rmsnorm_build_arch: Optional[Tuple[int, int]] = None

    def as_dict(self) -> dict:
        """Render as a JSON-serializable dict for reports."""
        return {
            "cuda_available": self.cuda_available,
            "device_capability": (
                list(self.device_capability) if self.device_capability else None
            ),
            "triton_available": self.triton_available,
            "triton_version": self.triton_version,
            "cutlass_available": self.cutlass_available,
            "cutlass_include_dir": self.cutlass_include_dir,
            "tilelang_available": self.tilelang_available,
            "tilelang_version": self.tilelang_version,
            "cublas_available": self.cublas_available,
            "cuda_rmsnorm_available": self.cuda_rmsnorm_available,
            "cuda_rmsnorm_lib": self.cuda_rmsnorm_lib,
            "cuda_rmsnorm_build_arch": (
                list(self.cuda_rmsnorm_build_arch)
                if self.cuda_rmsnorm_build_arch
                else None
            ),
            "notes": list(self.notes),
        }


def _detect_cuda() -> Tuple[bool, Optional[Tuple[int, int]]]:
    """Detect CUDA and return (available, capability) without importing torch."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, None
        cap = torch.cuda.get_device_capability(0)
        return True, (int(cap[0]), int(cap[1]))
    except Exception:
        return False, None


def _detect_triton() -> Tuple[bool, Optional[str], str]:
    """Prove Triton works by compiling a trivial kernel, not just importing."""
    if importlib.util.find_spec("triton") is None:
        return False, None, "triton package not installed"

    try:
        import torch
        import triton
        import triton.language as tl

        if not torch.cuda.is_available():
            return False, str(triton.__version__), "CUDA unavailable for Triton"

        # Real probe: compile + run a 1-element add kernel.
        @triton.jit
        def _probe_kernel(x_ptr, o_ptr):
            tl.store(o_ptr, tl.load(x_ptr) + 1.0)

        x = torch.ones(1, device="cuda")
        o = torch.empty(1, device="cuda")
        _probe_kernel[(1,)](x, o)
        torch.cuda.synchronize()
        if o.item() != 2.0:
            return False, str(triton.__version__), "Triton probe produced wrong result"

        return True, str(triton.__version__), ""
    except Exception as exc:  # pragma: no cover - depends on environment
        version = None
        try:
            import triton

            version = str(triton.__version__)
        except Exception:
            pass
        return False, version, f"Triton probe failed: {exc}"


def _detect_cutlass() -> Tuple[bool, Optional[str], str]:
    """Detect a CUTLASS header checkout (header-only; needs no linking)."""
    override = os.environ.get("HQSB_CUTLASS_INCLUDE_DIR")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    in_repo = os.path.join(repo_root, "third_party", "cutlass", "include")

    candidates = (
        ([override] if override else [])
        + [in_repo]
        + list(_DEFAULT_CUTLASS_PATHS)
    )

    for path in candidates:
        if path and os.path.isdir(os.path.join(path, "cutlass", "gemm")):
            return True, path, ""
    return False, None, "CUTLASS headers not found (set HQSB_CUTLASS_INCLUDE_DIR)"


def _detect_tilelang() -> Tuple[bool, Optional[str], str]:
    """Prove TileLang works by compiling + running a trivial kernel.

    Mirrors the Triton probe: ``import tilelang`` succeeding does not prove
    the TVM-based JIT can lower to the installed GPU, so we actually compile
    and run a 2-element add kernel.
    """
    if importlib.util.find_spec("tilelang") is None:
        return False, None, "TileLang package not installed"

    try:
        import torch
        import tilelang

        if not torch.cuda.is_available():
            return False, str(tilelang.__version__), "CUDA unavailable for TileLang"

        # The probe lives in a module WITHOUT `from __future__ import
        # annotations` because TVM requires concrete type annotations.
        from ops._tilelang_probe import run_probe

        if not run_probe():
            return False, str(tilelang.__version__), "TileLang probe produced wrong result"
        return True, str(tilelang.__version__), ""
    except Exception as exc:  # pragma: no cover - depends on environment
        version = None
        try:
            import tilelang

            version = str(tilelang.__version__)
        except Exception:
            pass
        return False, version, f"TileLang probe failed: {exc}"


def _detect_cublas() -> Tuple[bool, str]:
    """Detect cuBLAS availability (torch.matmul uses it under the hood)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA unavailable for cuBLAS"
        # torch linear/matmul dispatches to cuBLAS/cuBLASLt on CUDA.
        return True, ""
    except Exception:
        return False, "torch unavailable"


def _find_cuda_rmsnorm_lib() -> Optional[str]:
    """Locate the CUDA RMSNorm shared library built by CMake."""
    override = os.environ.get("HQSB_CUDA_RMSNORM_LIB")
    if override and os.path.isfile(override):
        return override

    # Resolve relative to the repository root (this file lives in ops/).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for path in glob.glob(os.path.join(repo_root, _DEFAULT_CUDA_LIB_GLOB)):
        if os.path.isfile(path):
            return path
    return None


def _probe_cuda_lib_build_arch(
    lib_path: str,
) -> Tuple[Optional[Tuple[int, int]], str]:
    """Read the compute capability the CUDA shared library was built for.

    The dispatcher must not assume the precompiled kernels match the runtime
    device, and it must not hard-code one platform either. The library exports
    ``hqsb_rmsnorm_query_build_arch`` for exactly this purpose; when the symbol
    is missing (an older build) the caller gets ``None`` plus an actionable
    note instead of a guess.
    """
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError as exc:
        return None, f"CUDA RMSNorm shared library failed to load: {exc}"

    fn = getattr(lib, "hqsb_rmsnorm_query_build_arch", None)
    if fn is None:
        return None, (
            "CUDA RMSNorm shared library does not export its build arch; "
            "rebuild it so arch-gated dispatch can verify compatibility"
        )

    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    fn.restype = ctypes.c_int
    major = ctypes.c_int(0)
    minor = ctypes.c_int(0)
    rc = fn(ctypes.byref(major), ctypes.byref(minor))
    if rc != 0:
        return None, (
            f"CUDA RMSNorm shared library could not report its build arch "
            f"(hqsb_rmsnorm_query_build_arch rc={rc}); dispatcher will not "
            f"assume the precompiled kernels match this device"
        )
    return (int(major.value), int(minor.value)), ""


@lru_cache(maxsize=1)
def detect_capabilities() -> BackendCapabilities:
    """Detect and cache the full backend capability snapshot."""
    notes: list = []

    cuda_available, capability = _detect_cuda()

    triton_available, triton_version, triton_note = _detect_triton()
    if triton_note:
        notes.append(triton_note)

    cutlass_available, cutlass_include, cutlass_note = _detect_cutlass()
    if cutlass_note:
        notes.append(cutlass_note)

    tilelang_available, tilelang_version, tilelang_note = _detect_tilelang()
    if tilelang_note:
        notes.append(tilelang_note)

    cublas_available, cublas_note = _detect_cublas()
    if cublas_note:
        notes.append(cublas_note)

    lib = _find_cuda_rmsnorm_lib()
    cuda_rmsnorm_available = cuda_available and lib is not None
    build_arch: Optional[Tuple[int, int]] = None
    if cuda_available and lib is None:
        notes.append(
            "CUDA RMSNorm shared library not found; build it with "
            "`cmake --build build/<preset>` (or set HQSB_CUDA_RMSNORM_LIB)"
        )
    elif lib is not None:
        build_arch, arch_note = _probe_cuda_lib_build_arch(lib)
        if arch_note:
            notes.append(arch_note)

    return BackendCapabilities(
        cuda_available=cuda_available,
        device_capability=capability,
        triton_available=triton_available,
        triton_version=triton_version,
        cutlass_available=cutlass_available,
        cutlass_include_dir=cutlass_include,
        tilelang_available=tilelang_available,
        tilelang_version=tilelang_version,
        cublas_available=cublas_available,
        cuda_rmsnorm_available=cuda_rmsnorm_available,
        cuda_rmsnorm_lib=lib,
        notes=tuple(notes),
        cuda_rmsnorm_build_arch=build_arch,
    )


__all__ = [
    "BackendDecision",
    "BackendCapabilities",
    "CAPABILITY_POLICY_VERSION",
    "CAPABILITY_SCHEMA_VERSION",
    "CapabilityCache",
    "CapabilityCacheCorruption",
    "CapabilityIdentity",
    "CapabilityReason",
    "CapabilityResult",
    "CapabilityStage",
    "ProbeStageResult",
    "detect_capabilities",
    "resolve_backend",
    "unavailable_result",
]
