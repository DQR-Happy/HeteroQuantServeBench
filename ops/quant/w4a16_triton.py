"""Triton W8/W4 fused-dequant weight-only GEMM (E05-06 §6).

Implements the execution side of the ``hqsb.w4a16.rowmajor.nk.v1`` and
``hqsb.w8a16.rowmajor.nk.v1`` layouts: the kernel reads the *packed* bytes,
extracts codes, applies the per-group scale (and zero point when present) and
feeds an FP16 fragment into ``tl.dot``. The dequantization happens inside the
kernel — this is what makes the low-bit *weight bandwidth* label legitimate
(``fused_dequant_weight_only``), as opposed to a storage-only path that
materializes FP16 weights before a normal GEMM.

Layout contract (must match :mod:`hqsb.quant.packing`):

* payload is a ``uint8`` tensor ``[N, row_stride_bytes]``;
* for 4-bit, byte ``j`` of a row holds logical columns ``2j`` (low nibble) and
  ``2j+1`` (high nibble); for 8-bit, byte ``j`` holds column ``j`` as int8;
* scales are ``[N, groups_per_row]`` float32; zeros (asymmetric) are int32;
* padding columns must be masked out by ``K`` — the kernel never multiplies a
  padding column into the accumulator.

The kernel is deliberately plain (no autotune, no split-K, no persistent
scheduling): its purpose is to be a *correct, observable* low-bit path with a
real kernel symbol, not a peak-performance kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from hqsb.core.errors import CapabilityError, ConfigError
from hqsb.quant.artifact import PackedVariantRecord, QuantArtifactDocument

#: Kernel symbol fragment used to identify the observed kernel in profiles.
KERNEL_SYMBOL = "hqsb_dequant_gemm_kernel"
PROVIDER = "triton"


def _require_torch_triton():
    try:
        import torch  # noqa: F401
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - CPU-minimal CI
        raise CapabilityError(
            "the fused-dequant GEMM needs torch and triton; install the "
            "'benchmark' and 'triton' extras",
            details={"capability": "torch+triton", "reason": str(exc)},
        ) from exc
    import torch

    if not torch.cuda.is_available():
        raise CapabilityError(
            "the fused-dequant GEMM requires a CUDA device",
            details={"capability": "cuda", "reason": "no device visible"},
        )


def _kernel_source():
    """Build the Triton kernel (deferred so importing this module is free)."""
    import triton
    import triton.language as tl

    @triton.jit
    def hqsb_dequant_gemm_kernel(
        X,
        PACKED,
        SCALES,
        ZEROS,
        OUT,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_pn,
        stride_sn,
        stride_sg,
        GROUP: tl.constexpr,
        BITS: tl.constexpr,
        HAS_ZERO: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + offs_k
            k_mask = k_idx < K
            x_ptrs = X + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
            x_tile = tl.load(
                x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0
            )

            if BITS == 4:
                byte_offs = (
                    offs_n[:, None] * stride_pn + (k_idx[None, :] // 2)
                )
                packed = tl.load(
                    PACKED + byte_offs,
                    mask=(offs_n[:, None] < N) & k_mask[None, :],
                    other=0,
                ).to(tl.uint8)
                low = (k_idx[None, :] % 2) == 0
                nib = tl.where(
                    low, packed & 0xF, (packed >> 4) & 0xF
                ).to(tl.int8)
                code = tl.where(nib >= 8, nib - 16, nib).to(tl.float32)
            else:
                byte_offs = offs_n[:, None] * stride_pn + k_idx[None, :]
                packed = tl.load(
                    PACKED + byte_offs,
                    mask=(offs_n[:, None] < N) & k_mask[None, :],
                    other=0,
                ).to(tl.uint8)
                signed = packed.to(tl.int8).to(tl.float32)
                code = signed

            group_idx = k_idx // GROUP
            scale_ptrs = (
                SCALES + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg
            )
            scale = tl.load(
                scale_ptrs,
                mask=(offs_n[:, None] < N) & k_mask[None, :],
                other=1.0,
            )
            if HAS_ZERO:
                zero_ptrs = (
                    ZEROS
                    + offs_n[:, None] * stride_sn
                    + group_idx[None, :] * stride_sg
                )
                zero = tl.load(
                    zero_ptrs,
                    mask=(offs_n[:, None] < N) & k_mask[None, :],
                    other=0,
                ).to(tl.float32)
                weight = scale * (code - zero)
            else:
                weight = scale * code
            # Padding columns are masked to zero by the load mask above; the
            # explicit mask keeps them from contributing on any path.
            weight = tl.where(k_mask[None, :], weight, 0.0)
            w_tile = weight.to(x_tile.dtype)
            acc = tl.dot(x_tile, tl.trans(w_tile), acc)

        out_ptrs = OUT + offs_m[:, None] * N + offs_n[None, :]
        tl.store(
            out_ptrs,
            acc.to(OUT.dtype.element_ty),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    return hqsb_dequant_gemm_kernel


@dataclass
class PreparedWeights:
    """Device tensors and metadata for one packed variant."""

    packed: Any           # uint8 [N, row_stride_bytes]
    scales: Any           # float32 [N, groups_per_row]
    zeros: Any            # int32 [N, groups_per_row] or None
    n: int
    k: int
    row_stride_bytes: int
    groups_per_row: int
    group_size: int
    bits: int
    layout_id: str
    variant_hash: str
    kernel_symbol: str = KERNEL_SYMBOL
    provider: str = PROVIDER

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "k": self.k,
            "row_stride_bytes": self.row_stride_bytes,
            "groups_per_row": self.groups_per_row,
            "group_size": self.group_size,
            "bits": self.bits,
            "layout_id": self.layout_id,
            "variant_hash": self.variant_hash,
            "kernel_symbol": self.kernel_symbol,
            "provider": self.provider,
        }


def prepare_weights(
    document: QuantArtifactDocument,
    variant: Optional[PackedVariantRecord] = None,
    *,
    device: str = "cuda",
) -> PreparedWeights:
    """Load a packed variant into the device tensors the kernel expects.

    The function is strict: the layout must be one the kernel implements, the
    tensor must be 2-D, and the group size must divide the padded columns
    consistently with the recorded layout. Anything else is a refusal with a
    reason (never an implicit fallback).
    """
    import os

    from hqsb.quant import packing
    from hqsb.quant.artifact import _weight_matrix_shape

    record = variant or (document.variants[0] if document.variants else None)
    if record is None:
        raise ConfigError(
            "document has no packed variant; rebuild it (canonical-only "
            "artifacts cannot be executed)"
        )
    if record.layout_id not in (
        packing.LAYOUT_W4A16_ROWMAJOR_NK_V1,
        packing.LAYOUT_W8A16_ROWMAJOR_NK_V1,
    ):
        raise CapabilityError(
            f"layout {record.layout_id!r} is not implemented by this kernel",
            details={
                "capability": "layout",
                "layout_id": record.layout_id,
                "supported": [
                    packing.LAYOUT_W4A16_ROWMAJOR_NK_V1,
                    packing.LAYOUT_W8A16_ROWMAJOR_NK_V1,
                ],
            },
        )
    rows, cols = _weight_matrix_shape(document.tensor.shape)
    layout_meta = (record.layout_extra or {}).get("layout", {})
    row_stride = int(layout_meta.get("row_stride_bytes", 0) or 0)
    if row_stride <= 0 or row_stride % 2 != 0:
        raise ConfigError(
            f"variant declares an unusable row_stride_bytes={row_stride}"
        )
    with open(os.path.join(document.artifact_dir, "variants", record.filename), "rb") as stream:
        payload = stream.read()
    if len(payload) != row_stride * rows:
        raise ConfigError(
            f"payload is {len(payload)} bytes but the layout implies "
            f"{row_stride * rows}"
        )
    # Packing validation and the storage-only reference need no Triton or
    # accelerator. Device capability is checked only by the execution path.
    try:
        import torch
    except ImportError as exc:
        raise CapabilityError("preparing weight tensors requires torch") from exc
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise CapabilityError("preparing CUDA weights requires a CUDA device")
    packed = torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(
        rows, row_stride
    ).to(device)
    group_size = document.scheme.group_size or cols
    groups = document.scheme.unit_count(cols)
    if groups <= 0:
        raise ConfigError("document declares no quantization units")
    scales = torch.tensor(document.scales, dtype=torch.float32, device=device)
    if scales.numel() == 1 and rows * groups > 1:
        scales = scales.expand(rows * groups).contiguous()
    if scales.numel() != rows * groups:
        raise ConfigError(
            f"document has {scales.numel()} scales, expected rows*groups="
            f"{rows * groups}"
        )
    scales = scales.reshape(rows, groups)
    zeros = None
    if not document.scheme.symmetric:
        zeros = torch.tensor(document.zeros, dtype=torch.int32, device=device)
        if zeros.numel() != rows * groups:
            raise ConfigError(
                f"document has {zeros.numel()} zeros, expected rows*groups="
                f"{rows * groups}"
            )
        zeros = zeros.reshape(rows, groups)
    return PreparedWeights(
        packed=packed,
        scales=scales,
        zeros=zeros,
        n=rows,
        k=cols,
        row_stride_bytes=row_stride,
        groups_per_row=groups,
        group_size=group_size,
        bits=document.scheme.bits,
        layout_id=record.layout_id,
        variant_hash=record.variant_hash,
    )


def gemm_low_bit(x, prepared: PreparedWeights, *, block_m: int = 64, block_n: int = 64, block_k: int = 32):
    """Run the fused-dequant GEMM ``x @ W_hat^T``.

    ``x`` is ``[M, K]`` FP16 on the device; the result is ``[M, N]`` FP16.
    Every shape check happens before the launch so an unsupported shape is a
    structured refusal rather than a garbage kernel read.
    """
    _require_torch_triton()
    import torch

    if x.dim() != 2:
        raise ConfigError(f"x must be 2-D, got shape {tuple(x.shape)}")
    if x.dtype != torch.float16:
        raise CapabilityError(
            f"the kernel computes in float16, got activation dtype {x.dtype}",
            details={"capability": "dtype", "dtype": str(x.dtype)},
        )
    m, k = int(x.shape[0]), int(x.shape[1])
    if k != prepared.k:
        raise ConfigError(
            f"activation K={k} does not match the packed weight K={prepared.k}"
        )
    if block_k % 2 != 0 and prepared.bits == 4:
        raise ConfigError("4-bit dequant needs an even BLOCK_K")
    x = x.contiguous()
    out = torch.empty((m, prepared.n), dtype=torch.float16, device=x.device)
    grid = (
        (m + block_m - 1) // block_m,
        (prepared.n + block_n - 1) // block_n,
    )
    kernel = _kernel_source()
    kernel[grid](
        x,
        prepared.packed,
        prepared.scales,
        prepared.zeros if prepared.zeros is not None else prepared.scales,
        out,
        m,
        prepared.n,
        k,
        x.stride(0),
        x.stride(1),
        prepared.packed.stride(0),
        prepared.scales.stride(0),
        prepared.scales.stride(1),
        GROUP=prepared.group_size,
        BITS=prepared.bits,
        HAS_ZERO=prepared.zeros is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out


def dequantize_weight_reference(prepared: PreparedWeights):
    """Materialize the dequantized weight from the packed bytes (CPU, torch).

    Used by the *storage-only* executor and by tests; it exists to prove that
    the packed bytes carry the right values, not to be a fast path.
    """
    _require_torch_triton()
    import torch

    if prepared.bits == 4:
        low = (prepared.packed & 0x0F).to(torch.int16)
        high = ((prepared.packed >> 4) & 0x0F).to(torch.int16)
        codes = torch.stack([low, high], dim=-1).reshape(prepared.n, -1)[:, : prepared.k]
        codes = torch.where(codes >= 8, codes - 16, codes).to(torch.float32)
    else:
        codes = prepared.packed[:, : prepared.k].to(torch.int8).to(torch.float32)
    scales = prepared.scales.repeat_interleave(prepared.group_size, dim=1)[:, : prepared.k]
    if prepared.zeros is not None:
        zeros = prepared.zeros.repeat_interleave(prepared.group_size, dim=1)[:, : prepared.k].to(torch.float32)
        return scales * (codes - zeros)
    return scales * codes


def compile_probe() -> Dict[str, Any]:
    """Compile and run a minimal instance of the real kernel.

    The probe uses the *actual* kernel (1x1x2, 4-bit, symmetric) so a broken
    layout or a compiler/sm mismatch is reported as a capability failure, not
    discovered later inside a benchmark.
    """
    _require_torch_triton()
    import torch

    # ``tl.dot`` requires >= 16 along every dimension, so the probe uses the
    # smallest legal tile; the values are chosen so the expected result is
    # exactly K (every code is 1 and both scales are 1).
    m, n, k = 2, 8, 16
    x = torch.ones((m, k), dtype=torch.float16, device="cuda")
    packed = torch.full((n, k // 2), 0x11, dtype=torch.uint8, device="cuda")
    scales = torch.ones((n, 1), dtype=torch.float32, device="cuda")
    prepared = PreparedWeights(
        packed=packed,
        scales=scales,
        zeros=None,
        n=n,
        k=k,
        row_stride_bytes=k // 2,
        groups_per_row=1,
        group_size=k,
        bits=4,
        layout_id="probe",
        variant_hash="probe",
    )
    out = gemm_low_bit(x, prepared, block_m=16, block_n=16, block_k=16)
    torch.cuda.synchronize()
    value = float(out[0, 0])
    # Every code is 1 (0x11 -> nibbles 1,1), every activation is 1, so each
    # output element is the sum over K.
    return {
        "kernel_symbol": KERNEL_SYMBOL,
        "provider": PROVIDER,
        "value": value,
        "expected": float(k),
        "ok": abs(value - float(k)) < 1e-2,
    }


__all__ = [
    "KERNEL_SYMBOL",
    "PROVIDER",
    "PreparedWeights",
    "compile_probe",
    "dequantize_weight_reference",
    "gemm_low_bit",
    "prepare_weights",
]
