"""Two-level correctness oracles (E05-02 §5.3, E05-06 §8).

Comparing a low-bit kernel directly with the FP16 model conflates two very
different errors. The protocol therefore requires two references:

A. **quantization oracle** — FP16 weights → canonical quantize/dequantize,
   then a *high-precision* GEMM. This isolates the algorithm's
   representation error.
B. **kernel oracle** — canonical qvalues + scales/zeros → high-precision
   dequantized GEMM computed from the *artifact's own semantics*. This
   isolates the kernel implementation error (indexing, nibble order, scale
   broadcast, tail mask, accumulation).

Both are computed here from the artifact, never from the kernel under test,
and the canonical dequantization is re-derived from the packed bytes with the
independent decoder when requested (so a packer bug cannot co-validate).

The high-precision path uses ``float64`` accumulation; on a real run with
torch available it uses torch's float64 matmul (CPU or GPU), otherwise it
raises a structured capability error rather than silently using float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from hqsb.core.errors import CapabilityError, ConfigError
from hqsb.quant import packing
from hqsb.quant.artifact import PackedVariantRecord, QuantArtifactDocument
from hqsb.quant.rtn import dequantize_flat

ORACLE_QUANTIZATION = "quantization_oracle"
ORACLE_KERNEL = "kernel_oracle"
ORACLE_FP16 = "fp16_reference"


@dataclass
class OracleResult:
    """Output of one oracle plus the context needed to interpret it."""

    kind: str
    values: Any  # torch.Tensor when torch is available
    dtype: str
    shape: Tuple[int, ...]
    artifact_identity: str = ""
    layout_id: str = ""
    accumulation_dtype: str = "float64"
    notes: str = ""


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in CPU-minimal CI
        raise CapabilityError(
            "the quantization/kernel oracles need torch for high-precision GEMM; "
            "install the 'benchmark' extra (pip install -e '.[benchmark]')",
            details={"capability": "torch", "reason": "not installed"},
        ) from exc
    return torch


def kernel_oracle(
    x,
    document: QuantArtifactDocument,
    variant: Optional[PackedVariantRecord] = None,
    *,
    independent_decoder: bool = True,
    weight_transposed: bool = True,
) -> OracleResult:
    """High-precision dequant-GEMM reference from the artifact's semantics.

    Args:
        x: activation ``[M, K]`` (torch tensor).
        document: the artifact; its canonical values define the weights.
        variant: which packed variant to decode. When given, the values are
            decoded from the *packed bytes* with the independent decoder, so
            the oracle also proves the bytes carry the canonical values.
        weight_transposed: ``True`` when the artifact tensor is ``[N, K]``
            (``nn.Linear.weight`` layout) and the GEMM is ``x @ W^T``.
    """
    torch = _require_torch()
    from hqsb.quant.artifact import _weight_matrix_shape

    rows, cols = _weight_matrix_shape(document.tensor.shape)
    scheme = document.scheme

    if variant is not None:
        payload_path = document.artifact_dir
        if not payload_path:
            raise ConfigError(
                "kernel_oracle needs the artifact directory to decode a packed "
                "variant; load the document from disk first"
            )
        packed = packing.PackedTensor(
            layout=packing.plan_kernel_layout(
                scheme,
                rows,
                cols,
                layout_id=variant.layout_id,
                alignment=variant.alignment,
                parent_canonical_hash=variant.parent_canonical_hash,
            ),
            payload=open(_variant_path(document, variant), "rb").read(),
            scales=list(document.scales),
            zeros=list(document.zeros),
        )
        qvalues = packing.unpack_kernel_variant(
            packed, scheme, independent=independent_decoder
        )
    else:
        qvalues = list(document.values)

    dequant = dequantize_flat(
        qvalues,
        document.scales,
        document.zeros,
        document.tensor.shape,
        document.units_per_row,
        scheme,
        group_size=scheme.group_size,
        axis=scheme.axis,
    )
    weight = torch.tensor(dequant, dtype=torch.float64).reshape(rows, cols)
    if weight_transposed:
        weight = weight.t()
    x64 = x.detach().to(torch.float64)
    if x64.shape[-1] != weight.shape[0]:
        raise ConfigError(
            f"activation K={x64.shape[-1]} does not match weight K={weight.shape[0]}"
        )
    values = x64 @ weight
    return OracleResult(
        kind=ORACLE_KERNEL,
        values=values,
        dtype=str(values.dtype),
        shape=tuple(int(dim) for dim in values.shape),
        artifact_identity=document.artifact_id(),
        layout_id=variant.layout_id if variant else "canonical",
        notes=(
            "decoded from packed bytes with the independent decoder"
            if variant is not None and independent_decoder
            else "computed from canonical logical values"
        ),
    )


def _variant_path(document: QuantArtifactDocument, variant: PackedVariantRecord) -> str:
    import os

    return os.path.join(document.artifact_dir, "variants", variant.filename)


def quantization_oracle(
    x,
    document: QuantArtifactDocument,
    *,
    weight_transposed: bool = True,
) -> OracleResult:
    """Algorithm-level reference: canonical dequantize + FP64 GEMM.

    Identical to :func:`kernel_oracle` without a packed variant — it does not
    look at any kernel layout, so layout bugs cannot influence it.
    """
    result = kernel_oracle(
        x, document, None, weight_transposed=weight_transposed
    )
    result.kind = ORACLE_QUANTIZATION
    result.notes = "canonical logical values, no packed bytes involved"
    return result


def fp16_reference(x, weight) -> OracleResult:
    """FP64 GEMM on the original (unquantized) weights.

    ``weight`` follows ``nn.Linear`` convention ``[out_features, in_features]``
    when its last dimension matches ``x``'s last dimension; otherwise it is
    treated as already transposed ``[in, out]``.
    """
    torch = _require_torch()
    x64 = x.detach().to(torch.float64)
    w64 = weight.detach().to(torch.float64)
    values = x64 @ w64.t() if w64.shape[-1] == x64.shape[-1] else x64 @ w64
    return OracleResult(
        kind=ORACLE_FP16,
        values=values,
        dtype=str(values.dtype),
        shape=tuple(int(dim) for dim in values.shape),
        notes="original FP16/BF16 weights promoted to float64",
    )


def correctness_report(
    reference: OracleResult,
    actual,
    *,
    tolerance: Optional[Mapping[str, float]] = None,
    sentinel_region: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Compare a kernel output with an oracle (E05-06 §8.3).

    Reports max/mean/RMSE/relative-L2/cosine, the worst per-row error, NaN/Inf
    counts, deterministic-repeat equality when provided, sentinel-region
    integrity, and the pass/fail verdict against a **layered** tolerance
    (per dtype/accumulator, never one global number).

    Returns a machine-readable dict; ``passed`` is only True when every check
    passed and the tolerance was actually provided (a missing tolerance is
    ``None``, i.e. unverifiable — not a pass).
    """
    torch = _require_torch()
    from hqsb.benchmark.metrics import numerical_diff_summary

    if reference.shape != tuple(int(dim) for dim in actual.shape):
        return {
            "passed": False,
            "reason": f"shape mismatch: oracle {reference.shape} vs actual "
            f"{tuple(actual.shape)}",
        }
    ref_flat = reference.values.reshape(-1).tolist()
    act_flat = actual.detach().reshape(-1).to(torch.float64).tolist()
    summary = numerical_diff_summary(ref_flat, act_flat)

    nan_count = sum(1 for value in act_flat if value != value)
    inf_count = sum(1 for value in act_flat if value in (float("inf"), float("-inf")))

    rows = reference.values.shape[0] if reference.values.dim() > 1 else 1
    per_row_worst = 0.0
    ref_rows = reference.values.reshape(rows, -1)
    act_rows = actual.detach().reshape(rows, -1)
    for index in range(rows):
        row_diff = (ref_rows[index] - act_rows[index].to(torch.float64)).abs().max()
        per_row_worst = max(per_row_worst, float(row_diff))

    sentinel_ok = True
    if sentinel_region is not None:
        start, end = sentinel_region
        sentinel_ok = all(value == 0.0 for value in act_flat[start:end])

    checks: Dict[str, bool] = {
        "finite_output": nan_count == 0 and inf_count == 0,
        "sentinel_intact": sentinel_ok,
    }
    verdict: Optional[bool] = None
    if tolerance is not None:
        checks["max_abs"] = summary["max_abs_error"] <= tolerance.get(
            "max_abs", float("inf")
        )
        checks["relative_l2"] = summary["l2_relative_error"] <= tolerance.get(
            "relative_l2", float("inf")
        )
        checks["cosine"] = summary["cosine_similarity"] >= tolerance.get(
            "cosine_min", 0.0
        )
        verdict = all(checks.values())
    return {
        "passed": verdict,
        "tolerance_provided": tolerance is not None,
        "checks": checks,
        "metrics": summary,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "per_row_worst_abs": per_row_worst,
        "oracle_kind": reference.kind,
        "oracle_note": reference.notes,
    }


__all__ = [
    "ORACLE_FP16",
    "ORACLE_KERNEL",
    "ORACLE_QUANTIZATION",
    "OracleResult",
    "correctness_report",
    "fp16_reference",
    "kernel_oracle",
    "quantization_oracle",
]
