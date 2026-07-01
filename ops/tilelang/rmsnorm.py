"""Experimental TileLang RMSNorm for the E04-11 portability study.

The kernel implements the frozen HQSB RMSNorm contract with FP32 square,
reduction, epsilon, reciprocal-square-root and weight multiplication.  Shape
and dtype are compile-time specializations.  This module deliberately omits
``from __future__ import annotations`` because TileLang inspects concrete
``T.Tensor`` annotations while constructing TIR.

This is not registered in the production dispatcher.  Importing it is an
explicit opt-in and therefore cannot make the core/reference path depend on
TileLang.
"""

import functools
import math

import torch

import tilelang
import tilelang.language as T


MAX_HIDDEN = 8192
DEFAULT_THREADS = 128


def _dtype_name(dtype):
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    raise ValueError("TileLang RMSNorm supports only torch.float16 and torch.float32")


def _validate(x, weight, epsilon):
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
    return rows, hidden, eps, _dtype_name(x.dtype)


@functools.lru_cache(maxsize=128)
def compile_rmsnorm(rows, hidden, dtype="float16", epsilon=1.0e-6,
                    threads=DEFAULT_THREADS):
    """Compile and cache one static TileLang RMSNorm specialization."""
    rows = int(rows)
    hidden = int(hidden)
    threads = int(threads)
    epsilon = float(epsilon)
    if rows < 1 or hidden < 1 or hidden > MAX_HIDDEN:
        raise ValueError("invalid static RMSNorm shape")
    if dtype not in ("float16", "float32"):
        raise ValueError("dtype must be float16 or float32")
    if threads not in (32, 64, 128, 256):
        raise ValueError("threads must be one of 32, 64, 128, 256")
    block_hidden = max(threads, 1 << (hidden - 1).bit_length())

    @T.prim_func
    def _kernel(
        X: T.Tensor((rows, hidden), dtype),
        W: T.Tensor((hidden,), dtype),
        Y: T.Tensor((rows, hidden), dtype),
    ):
        with T.Kernel(rows, threads=threads) as bx:
            # The reduction fragment is a power-of-two multiple of the thread
            # group.  Explicit zero padding makes odd widths a real masked
            # tail case and avoids backend-dependent fragment layouts.
            x_fp32 = T.alloc_fragment((block_hidden,), "float32")
            square = T.alloc_fragment((block_hidden,), "float32")
            square_sum = T.alloc_fragment((1,), "float32")

            for j in T.Parallel(block_hidden):
                if j < hidden:
                    value = T.cast(X[bx, j], "float32", sat=False)
                    x_fp32[j] = value
                    square[j] = value * value
                else:
                    x_fp32[j] = 0.0
                    square[j] = 0.0

            T.reduce_sum(square, square_sum, dim=0)
            square_sum[0] = T.rsqrt(square_sum[0] / hidden + epsilon)

            for j in T.Parallel(block_hidden):
                if j < hidden:
                    Y[bx, j] = T.cast(
                        x_fp32[j] * square_sum[0]
                        * T.cast(W[j], "float32", sat=False),
                        dtype,
                        sat=False,
                    )

    return tilelang.compile(
        _kernel,
        out_idx=[2],
        pass_configs={"tl.disable_tma_lower": True},
    )


def rmsnorm_tilelang(x, weight, epsilon=1.0e-6, *, threads=DEFAULT_THREADS):
    """Execute the experimental implementation on PyTorch CUDA tensors."""
    rows, hidden, eps, dtype = _validate(x, weight, epsilon)
    kernel = compile_rmsnorm(rows, hidden, dtype, eps, int(threads))
    return kernel(x, weight)


def clear_compile_cache():
    """Clear only HQSB's in-process specialization handle cache."""
    compile_rmsnorm.cache_clear()


__all__ = [
    "DEFAULT_THREADS",
    "MAX_HIDDEN",
    "clear_compile_cache",
    "compile_rmsnorm",
    "rmsnorm_tilelang",
]
