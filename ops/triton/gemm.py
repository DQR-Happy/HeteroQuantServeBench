"""Triton GEMM for the cuBLAS/Triton comparison (S04).

Implements the canonical tiled GEMM (blocked register tiling with a K-loop
accumulator) from the Triton model, plus an autotuned variant over tile
sizes. This is the "DSL 手写" data point; the "厂商库" data point is
``torch.matmul`` / ``torch.nn.functional.linear`` (which dispatch to cuBLAS/
cuBLASLt on CUDA).

Per S02/S03 convention we do **not** claim to beat cuBLAS at general GEMM —
the point is to quantify the gap and explain *why* (S04 acceptance: "至少
一个 GEMM/epilogue 完成库级调优" + "解释 CUDA 与 Triton 的性能差异").
"""

from __future__ import annotations

import torch

import triton
import triton.language as tl


# ── Fixed-tile reference GEMM (readable baseline) ─────────────────────

@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Blocked GEMM: C = A @ B (row-major A, B, C)."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # input_precision="ieee" keeps FP32 accuracy (tl.dot otherwise
        # defaults to TF32 tensor-core math, diverging ~0.2% from cuBLAS FP32).
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


def gemm_reference(a: torch.Tensor, b: torch.Tensor):
    """Triton tiled GEMM with a fixed, readable tile configuration."""
    assert a.is_cuda and b.is_cuda
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    grid = (triton.cdiv(m, 64) * triton.cdiv(n, 64),)
    _gemm_kernel[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8,
    )
    return c


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k, "GROUP_M": 8}, num_stages=3)
        for m, n, k in [(64, 64, 32), (128, 64, 32), (64, 128, 32), (128, 128, 64)]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_autotuned_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


def gemm_optimized(a: torch.Tensor, b: torch.Tensor):
    """Autotuned Triton GEMM."""
    assert a.is_cuda and b.is_cuda
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    _gemm_autotuned_kernel[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def gemm_cublas(a: torch.Tensor, b: torch.Tensor):
    """Vendor-library GEMM via torch (dispatches to cuBLAS/cuBLASLt)."""
    return a @ b


__all__ = [
    "gemm_cublas",
    "gemm_optimized",
    "gemm_reference",
]
