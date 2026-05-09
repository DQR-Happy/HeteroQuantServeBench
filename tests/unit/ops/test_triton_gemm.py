"""GPU correctness tests for Triton GEMM vs cuBLAS (torch.matmul).

Skipped when Triton/CUDA is unavailable. Two comparison regimes:

* **FP32 GEMM** — tight absolute tolerance; FP32 accumulation means the two
  implementations agree to ~1e-3 regardless of tile/order.
* **FP16 GEMM** — FP16 *output* has per-element rounding that differs with
  accumulation order (tile size), so we compare with a relative tolerance
  ``|a - e| <= atol + rtol * |e|`` (FP16 is ~3 decimal digits; rtol=2e-2).

This is NOT a weakened threshold per implementation — it reflects the FP16
output's intrinsic precision, identical for cuBLAS and Triton.
"""

from __future__ import annotations

import pytest
import torch

from ops.capability import detect_capabilities

_caps = detect_capabilities()
_HAS_TRITON = _caps.triton_available and _caps.cuda_available

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not _HAS_TRITON, reason="Triton/CUDA not available"),
]


def _fp16_close(actual, expected, rtol=2e-2, atol=1e-2):
    diff = (actual - expected).abs()
    tol = atol + rtol * expected.abs()
    return (diff <= tol).all().item()


@pytest.mark.parametrize(
    ("m", "k", "n"),
    [(64, 128, 64), (512, 256, 256)],
)
def test_triton_gemm_matches_cublas_fp32(m, k, n):
    from ops.triton.gemm import gemm_optimized, gemm_reference

    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(k, n, device="cuda", dtype=torch.float32)
    expected = a @ b

    assert (gemm_reference(a, b) - expected).abs().max().item() <= 1e-2
    assert (gemm_optimized(a, b) - expected).abs().max().item() <= 1e-2


@pytest.mark.parametrize(
    ("m", "k", "n"),
    [(64, 128, 64), (1, 2048, 2048), (512, 256, 256)],
)
def test_triton_gemm_matches_cublas_fp16(m, k, n):
    from ops.triton.gemm import gemm_optimized, gemm_reference

    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    expected = a @ b  # cuBLAS

    assert _fp16_close(gemm_reference(a, b), expected)
    assert _fp16_close(gemm_optimized(a, b), expected)
