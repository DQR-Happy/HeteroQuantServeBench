"""Optional-backend-free operator references.

This module intentionally has no module-scope torch, Triton, CUTLASS, or
TileLang import.  The reference path therefore remains importable when every
optional accelerator package is absent.  ``torch`` is imported only when the
caller actually asks to execute a tensor operation.
"""

from __future__ import annotations


def rmsnorm_torch(x, weight, epsilon: float = 1e-5):
    """PyTorch CPU/CUDA reference for the frozen RMSNorm semantic."""
    rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(epsilon).sqrt()
    return (x / rms * weight).to(x.dtype)


def gemm_torch(a, b):
    """PyTorch reference GEMM."""
    return a @ b


__all__ = ["gemm_torch", "rmsnorm_torch"]
