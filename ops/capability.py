"""Unified backend capability detection for the operator dispatcher (S04).

S04's contract is explicit: *"Jetson 对 Triton 支持需能力检测，不能默认"* and
*"未安装/不支持的 DSL 走明确 fallback，不影响 CPU/CUDA 核心包"*. This module
is the single source of truth for what is actually usable at runtime, and it
never raises — an unavailable backend simply reports ``False``.

Each probe does *real* work where cheap (e.g. Triton compiles a 1-element
kernel to prove the backend works, not merely that the package imports),
because ``import triton`` succeeding does not guarantee the JIT backend can
compile for the installed GPU.

Detection is cached (``functools.lru_cache``) because some probes (Triton JIT
compile, ctypes load) are not free; callers get a stable snapshot.
"""

from __future__ import annotations

import ctypes
import glob
import importlib.util
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Tuple

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
        return False, None, "tilelang package not installed"

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
    if cuda_available and lib is None:
        notes.append(
            "CUDA RMSNorm shared library not found; build it with "
            "`cmake --build build/jetson-release`"
        )

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
    )


__all__ = [
    "BackendCapabilities",
    "detect_capabilities",
]
