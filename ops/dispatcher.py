"""Unified operator dispatcher (S04 step 7).

Selects an implementation for ``rmsnorm`` and ``gemm`` from the available
backends — hand-written CUDA, Triton, cuBLAS (via torch), and (when present)
CUTLASS — using a deterministic, auditable policy:

1. **capability** — a backend that is not installed/compiled is never chosen;
2. **arch** — the hand-written CUDA shared library is built for a specific
   compute capability; on a mismatched arch it is skipped in favor of
   Triton's JIT (which adapts to any arch);
3. **shape/dtype** — dispatches to the fastest supported variant (e.g. CUDA
   V2 for aligned shapes, V1 for non-multiple-of-4 FP32);
4. **explicit fallback** — the final torch reference is always available on
   CPU, so a core benchmark can still *run* (with a recorded degradation)
   even when every accelerator backend is missing.

Every decision carries a human-readable ``reason`` so a benchmark report can
explain *why* an implementation was chosen — never a silent heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from ops.capability import BackendCapabilities, detect_capabilities

# The hand-written CUDA shared library is compiled for this arch (see
# CMakeLists -DCMAKE_CUDA_ARCHITECTURES=87). If the runtime device differs,
# the precompiled kernel is not usable and Triton JIT is preferred.
_CUDA_LIB_ARCH = (8, 7)

# CUDA variant codes (mirror ops/cuda/rmsnorm/src/rmsnorm_c_api.cu).
_VARIANT_CODE = {
    "v0_shared": 2,
    "v1_warp_shuffle": 3,
    "v2_vectorized": 4,
}


@dataclass(frozen=True)
class DispatchDecision:
    """The outcome of a dispatch: which backend/variant and why."""

    backend: str
    variant: str
    reason: str

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "variant": self.variant,
            "reason": self.reason,
        }


class OperatorDispatcher:
    """Resolves rmsnorm/gemm to a concrete implementation."""

    def __init__(self, capabilities: Optional[BackendCapabilities] = None) -> None:
        self.capabilities = capabilities or detect_capabilities()

    # ── RMSNorm ────────────────────────────────────────────────────────

    def select_rmsnorm(self, dtype: str, hidden: int) -> DispatchDecision:
        """Choose an RMSNorm implementation for ``dtype``/``hidden``.

        Args:
            dtype: ``"fp32"`` or ``"fp16"``.
            hidden: number of columns (drives vectorization choice).
        """
        cap = self.capabilities

        # 1. Hand-written CUDA (fastest, arch-locked).
        if cap.cuda_rmsnorm_available and self._cuda_arch_matches():
            if dtype == "fp16":
                return DispatchDecision(
                    "cuda", "v2_vectorized",
                    "CUDA V2: FP16 always vectorized (half2 + scalar tail)",
                )
            if hidden % 4 == 0:
                return DispatchDecision(
                    "cuda", "v2_vectorized",
                    "CUDA V2: FP32 aligned to float4",
                )
            return DispatchDecision(
                "cuda", "v1_warp_shuffle",
                "CUDA V1: FP32 non-multiple-of-4 falls back to warp shuffle",
            )

        # 2. Triton (JIT, arch-agnostic).
        if cap.triton_available:
            return DispatchDecision(
                "triton", "autotuned",
                "Triton: CUDA shared lib unavailable or arch mismatch; "
                "autotuned BLOCK_SIZE",
            )

        # 3. torch reference (CPU/CUDA fallback, always present).
        return DispatchDecision(
            "torch", "reference",
            "torch: no accelerator backend; correctness-only fallback",
        )

    def run_rmsnorm(self, x, weight, epsilon: float = 1e-5) -> Tuple:
        """Run RMSNorm via the selected backend and return ``(output, decision)``."""
        import torch

        dtype = "fp16" if x.dtype == torch.float16 else "fp32"
        decision = self.select_rmsnorm(dtype, int(x.shape[1]))

        if decision.backend == "cuda":
            from ops.cuda_bridge import rmsnorm_forward as cuda_forward

            variant_code = _VARIANT_CODE.get(decision.variant, 0)
            out = cuda_forward(
                x, weight, dtype=dtype, variant=variant_code, epsilon=epsilon
            )
            return out, decision

        if decision.backend == "triton":
            from ops.triton.rmsnorm import rmsnorm_optimized

            return rmsnorm_optimized(x, weight, epsilon=epsilon), decision

        from ops.triton.rmsnorm import rmsnorm_torch

        return rmsnorm_torch(x, weight, epsilon=epsilon), decision

    # ── GEMM ──────────────────────────────────────────────────────────

    def select_gemm(self, dtype: str) -> DispatchDecision:
        """Choose a GEMM implementation. cuBLAS wins when CUDA is present."""
        cap = self.capabilities

        if cap.cublas_available:
            return DispatchDecision(
                "cublas", "torch_matmul",
                "cuBLAS/cuBLASLt via torch.matmul: vendor-tuned GEMM is the "
                "fastest general path (S02: GEMM is the dominant hotspot)",
            )
        if cap.triton_available:
            return DispatchDecision(
                "triton", "autotuned_tiled",
                "Triton tiled GEMM: cuBLAS unavailable; DSL fallback",
            )
        return DispatchDecision(
            "torch", "reference",
            "torch: CPU fallback for GEMM (no accelerator)",
        )

    def run_gemm(self, a, b) -> Tuple:
        """Run GEMM via the selected backend and return ``(output, decision)``."""
        import torch

        dtype = "fp16" if a.dtype == torch.float16 else "fp32"
        decision = self.select_gemm(dtype)

        if decision.backend == "cublas":
            from ops.triton.gemm import gemm_cublas

            return gemm_cublas(a, b), decision

        if decision.backend == "triton":
            from ops.triton.gemm import gemm_optimized

            return gemm_optimized(a, b), decision

        return a @ b, decision

    # ── helpers ───────────────────────────────────────────────────────

    def _cuda_arch_matches(self) -> bool:
        cap = self.capabilities.device_capability
        return cap is not None and tuple(cap) == _CUDA_LIB_ARCH


def run_rmsnorm(x, weight, epsilon: float = 1e-5) -> Tuple:
    """Module-level convenience wrapper."""
    return OperatorDispatcher().run_rmsnorm(x, weight, epsilon=epsilon)


def run_gemm(a, b) -> Tuple:
    """Module-level convenience wrapper."""
    return OperatorDispatcher().run_gemm(a, b)


__all__ = [
    "DispatchDecision",
    "OperatorDispatcher",
    "run_gemm",
    "run_rmsnorm",
]
