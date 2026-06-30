"""Triton RMSNorm backends used by the S04 comparison experiments.

The two entry points implement the frozen S03 contract: contiguous
``(rows, hidden)`` FP16/FP32 tensors, FP32 accumulation, epsilon inside
``rsqrt`` and output in the input dtype.  They never make a hidden contiguous
copy or silently fall back to PyTorch/CUDA.

``rmsnorm_reference`` uses a fixed four-warp configuration.  The tuned kernel
searches only the number of warps (2/4/8); the program model and arithmetic do
not change.  One program owns one row and uses a power-of-two masked vector,
which covers every S03 odd/tail width through H=8192.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Optional

import torch
import triton
import triton.language as tl


SEARCH_SPACE = (
    {"config_id": "warps2", "num_warps": 2, "num_stages": 1},
    {"config_id": "warps4", "num_warps": 4, "num_stages": 1},
    {"config_id": "warps8", "num_warps": 8, "num_stages": 1},
)
MAX_HIDDEN = 8192


@triton.jit
def _rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr, hidden, eps, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden
    row_start = row * hidden
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    square_sum = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(square_sum / hidden + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row_start + cols, x * inv_rms * w, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=item["num_warps"], num_stages=item["num_stages"])
        for item in SEARCH_SPACE
    ],
    key=["hidden"],
)
@triton.jit
def _rmsnorm_tuned_kernel(
    x_ptr, w_ptr, out_ptr, hidden, eps, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden
    row_start = row * hidden
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    square_sum = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(square_sum / hidden + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row_start + cols, x * inv_rms * w, mask=mask)


def _validate(x: torch.Tensor, weight: torch.Tensor, epsilon: float, out):
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("x and weight must be torch.Tensor instances")
    if x.ndim != 2 or weight.ndim != 1:
        raise ValueError("x must be 2-D and weight must be 1-D")
    rows, hidden = int(x.shape[0]), int(x.shape[1])
    if rows < 1 or hidden < 1 or hidden > MAX_HIDDEN:
        raise ValueError(f"rows must be >=1 and hidden must be in [1, {MAX_HIDDEN}]")
    if int(weight.numel()) != hidden:
        raise ValueError("weight length must equal hidden")
    if not x.is_cuda or not weight.is_cuda or x.device != weight.device:
        raise ValueError("x and weight must be CUDA tensors on the same device")
    if x.dtype not in (torch.float16, torch.float32) or weight.dtype != x.dtype:
        raise ValueError("x and weight must share dtype float16 or float32")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("non-contiguous input is outside the frozen contract")
    try:
        eps = float(epsilon)
    except (TypeError, ValueError):
        raise ValueError("epsilon must be finite and > 0") from None
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be finite and > 0")
    if out is not None:
        if (
            not isinstance(out, torch.Tensor)
            or out.shape != x.shape
            or out.dtype != x.dtype
            or out.device != x.device
            or not out.is_contiguous()
        ):
            raise ValueError("out must match x shape/dtype/device and be contiguous")
    return rows, hidden, eps


def _block_size(hidden: int) -> int:
    return int(triton.next_power_of_2(hidden))


def rmsnorm_config_hash(rows: int, hidden: int, dtype: str, epsilon: float) -> str:
    """Stable workload/config-space identity for evidence and cache audits."""
    payload = {
        "rows": int(rows),
        "hidden": int(hidden),
        "dtype": str(dtype),
        "epsilon": float(epsilon),
        "block_size": _block_size(int(hidden)),
        "search_space": SEARCH_SPACE,
        "triton_version": str(triton.__version__),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rmsnorm_last_tuned_config() -> Optional[Dict[str, Any]]:
    """Return the most recently selected autotune config when Triton exposes it."""
    config = getattr(_rmsnorm_tuned_kernel, "best_config", None)
    if config is None:
        return None
    return {
        "kwargs": dict(getattr(config, "kwargs", {}) or {}),
        "num_warps": int(getattr(config, "num_warps", 0) or 0),
        "num_stages": int(getattr(config, "num_stages", 0) or 0),
    }


def rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-5,
    *,
    out: Optional[torch.Tensor] = None,
):
    """Fixed, readable Triton implementation (four warps, no autotune)."""
    rows, hidden, eps = _validate(x, weight, epsilon, out)
    if out is None:
        out = torch.empty_like(x)
    _rmsnorm_kernel[(rows,)](
        x, weight, out, hidden, eps,
        BLOCK_SIZE=_block_size(hidden), num_warps=4, num_stages=1,
    )
    return out


def rmsnorm_optimized(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-5,
    *,
    out: Optional[torch.Tensor] = None,
):
    """Autotuned Triton implementation with the frozen search space above."""
    rows, hidden, eps = _validate(x, weight, epsilon, out)
    if out is None:
        out = torch.empty_like(x)
    _rmsnorm_tuned_kernel[(rows,)](
        x, weight, out, hidden, eps, BLOCK_SIZE=_block_size(hidden)
    )
    return out


# Compatibility re-export.  Production fallback imports this from
# ``ops.reference`` directly, so missing Triton cannot break reference import.
from ops.reference import rmsnorm_torch


__all__ = [
    "MAX_HIDDEN",
    "SEARCH_SPACE",
    "rmsnorm_config_hash",
    "rmsnorm_last_tuned_config",
    "rmsnorm_optimized",
    "rmsnorm_reference",
    "rmsnorm_torch",
]
