"""Triton RMSNorm: reference and autotuned variants (S04).

These kernels implement the *same* OperatorSpec (C3 ``rmsnorm``) and numeric
contract as the S03 CUDA kernels — same semantic, dtype, and tolerance — so
CUDA vs Triton is an apples-to-apples comparison, never a comparison with a
weakened reference.

Correctness contract (learned during S04): one program owns one row, but the
row may be wider than ``BLOCK`` (e.g. Qwen3 ``hidden=2048`` with
``BLOCK=1024``). Both kernels therefore **loop** over the row in ``BLOCK``
chunks — first accumulating the mean-square, then a second pass to normalize
and store. This mirrors the CUDA kernels' strided loop
(``for col = tid; col < hidden; col += blockDim``).

Autotune cache contract (S04 test standard): the autotuner's cache key is
bound to the device arch *and* the kernel args (``hidden``), and we surface a
``config_hash`` through :func:`rmsnorm_config_hash` for tests and reports.
"""

from __future__ import annotations

import hashlib
import json

import torch

import triton
import triton.language as tl


# Autotune search space: block size per row. Kept modest on purpose so the
# search is cheap on an edge device; the point is the mechanism (S04 §3:
# "训练/测试 shape 分离").
_BLOCK_SIZES = [256, 512, 1024, 2048]


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, hidden, eps, BLOCK: tl.constexpr):
    """One program per row, looping over ``hidden`` in ``BLOCK`` chunks."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    square_sum = 0.0
    for off in range(0, hidden, BLOCK):
        c = off + cols
        mask = c < hidden
        x = tl.load(x_ptr + row * hidden + c, mask=mask, other=0.0)
        square_sum += tl.sum(x * x, axis=0)

    inv_rms = 1.0 / tl.sqrt(square_sum / hidden + eps)

    for off in range(0, hidden, BLOCK):
        c = off + cols
        mask = c < hidden
        x = tl.load(x_ptr + row * hidden + c, mask=mask, other=0.0)
        w = tl.load(w_ptr + c, mask=mask, other=0.0)
        tl.store(out_ptr + row * hidden + c, x * inv_rms * w, mask=mask)


@triton.autotune(
    configs=[triton.Config({"BLOCK": b}, num_warps=4) for b in _BLOCK_SIZES],
    key=["hidden"],
)
@triton.jit
def _rmsnorm_autotuned_kernel(x_ptr, w_ptr, out_ptr, hidden, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    square_sum = 0.0
    for off in range(0, hidden, BLOCK):
        c = off + cols
        mask = c < hidden
        x = tl.load(x_ptr + row * hidden + c, mask=mask, other=0.0)
        square_sum += tl.sum(x * x, axis=0)

    inv_rms = 1.0 / tl.sqrt(square_sum / hidden + eps)

    for off in range(0, hidden, BLOCK):
        c = off + cols
        mask = c < hidden
        x = tl.load(x_ptr + row * hidden + c, mask=mask, other=0.0)
        w = tl.load(w_ptr + c, mask=mask, other=0.0)
        tl.store(out_ptr + row * hidden + c, x * inv_rms * w, mask=mask)


def rmsnorm_config_hash(rows: int, hidden: int, dtype: str, epsilon: float) -> str:
    """Return a stable hash binding a workload + dtype + epsilon.

    Used in autotune cache keys and reports so cache invalidation is
    deterministic and auditable.
    """
    payload = {
        "rows": int(rows),
        "hidden": int(hidden),
        "dtype": str(dtype),
        "epsilon": float(epsilon),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-5):
    """Naive Triton RMSNorm (1 program per row, fixed BLOCK=1024)."""
    rows, hidden = x.shape
    out = torch.empty_like(x)
    _rmsnorm_kernel[(rows,)](x, weight, out, hidden, epsilon, BLOCK=1024, num_warps=4)
    return out


def rmsnorm_optimized(x: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-5):
    """Autotuned Triton RMSNorm (BLOCK_SIZE searched per ``hidden``)."""
    rows, hidden = x.shape
    out = torch.empty_like(x)
    _rmsnorm_autotuned_kernel[(rows,)](x, weight, out, hidden, epsilon)
    return out


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-5):
    """PyTorch reference (cuBLAS-independent elementwise) for correctness."""
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return (x / rms * weight).to(x.dtype)


__all__ = [
    "rmsnorm_config_hash",
    "rmsnorm_optimized",
    "rmsnorm_reference",
    "rmsnorm_torch",
]
