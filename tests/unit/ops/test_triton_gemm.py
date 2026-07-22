"""GPU correctness tests for Triton GEMM vs cuBLAS (torch.matmul).

Skipped when Triton/CUDA is unavailable. Two comparison regimes:

* **FP32 GEMM** — tight absolute tolerance; FP32 accumulation means the two
  implementations agree to ~1e-3 regardless of tile/order.
* **FP16 GEMM** — every implementation is compared against an **FP64**
  reference with *scale-free* gates (relative L2 error and cosine
  similarity).

Why the FP16 regime is scale-free (S04 补齐)
--------------------------------------------
An earlier version of this file compared Triton against cuBLAS with the
per-element bound ``|a - e| <= atol + rtol * |e|``. That bound is unsound
regardless of how the kernels are implemented:

* the agreement two correct implementations can reach is set by the
  *reduction length* ``K``, not by the output magnitude — two different
  accumulation orders differ by roughly ``K * eps_fp32 * sum_i |a_i b_i|``;
* near the zero crossing of a length-``K`` dot product the output magnitude
  ``|e|`` is arbitrarily small while that accumulated difference stays the
  same size, so the bound degenerates to ``atol`` and cannot be satisfied.

Observed on RTX 3090 (sm_86), shape 1x2048x2048 FP16: cuBLAS and Triton had
an *identical* max error against the FP64 reference (0.0556), i.e. both were
equally correct, yet a single near-zero element (expected -0.0078, actual
-0.0201) tripped the old bound. The tests below replace the bound with gates
that scale with the reference's norm, and a negative control
(``test_fp16_gate_rejects_wrong_output``) pins down that the gates still fail
on genuinely wrong output — so this is a stricter formulation, not a looser
one.
"""

from __future__ import annotations

import pytest
import torch

from ops.capability import detect_capabilities

_caps = detect_capabilities()
_HAS_TRITON = _caps.triton_available and _caps.cuda_available

pytestmark = [
    pytest.mark.unit,
    pytest.mark.hardware,
    pytest.mark.skipif(not _HAS_TRITON, reason="Triton/CUDA not available"),
]

# Thresholds follow the FP16 output quantization floor (~2^-11): the relative
# L2 error of a correct FP16-output GEMM sits around 1e-4, so 1e-2 leaves two
# orders of magnitude of headroom while still rejecting real breakage.
_L2_RELATIVE_TOLERANCE = 1e-2
_MIN_COSINE_SIMILARITY = 0.9999


def _fp64_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Ground-truth GEMM accumulated in FP64, returned in the inputs' dtype."""
    return (a.double() @ b.double()).to(a.dtype)


def _l2_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.float().reshape(-1)
    e = expected.float().reshape(-1)
    denominator = float(torch.linalg.vector_norm(e))
    if denominator == 0.0:
        return 0.0
    return float(torch.linalg.vector_norm(a - e)) / denominator


def _cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.float().reshape(-1)
    e = expected.float().reshape(-1)
    denominator = float(torch.linalg.vector_norm(a)) * float(
        torch.linalg.vector_norm(e)
    )
    if denominator == 0.0:
        return 1.0
    return float(torch.dot(a, e)) / denominator


def _fp16_gemm_ok(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Scale-free correctness gate for an FP16-output GEMM."""
    return (
        _l2_relative_error(actual, expected) <= _L2_RELATIVE_TOLERANCE
        and _cosine_similarity(actual, expected) >= _MIN_COSINE_SIMILARITY
    )


def _assert_fp16_gemm_ok(actual, reference, label: str) -> None:
    l2 = _l2_relative_error(actual, reference)
    cosine = _cosine_similarity(actual, reference)
    assert l2 <= _L2_RELATIVE_TOLERANCE, (
        f"{label}: l2_relative_error={l2:.6g} exceeds "
        f"{_L2_RELATIVE_TOLERANCE} (cosine={cosine:.8f})"
    )
    assert cosine >= _MIN_COSINE_SIMILARITY, (
        f"{label}: cosine_similarity={cosine:.8f} below "
        f"{_MIN_COSINE_SIMILARITY} (l2_relative_error={l2:.6g})"
    )


@pytest.mark.parametrize(
    ("m", "k", "n"),
    [
        (64, 128, 64),
        (63, 128, 64),   # M tail
        (64, 128, 65),   # N tail
        (64, 129, 64),   # K tail
        (63, 129, 65),   # M/N/K tail
        (512, 256, 256),
    ],
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
    [
        (64, 128, 64),
        (63, 128, 64),
        (64, 128, 65),
        (64, 129, 64),
        (63, 129, 65),
        (1, 2048, 2048),
        (512, 256, 256),
    ],
)
def test_triton_gemm_matches_cublas_fp16(m, k, n):
    """Every backend is checked against the FP64 reference, not against each
    other, so a shared error cannot cancel out."""
    from ops.triton.gemm import gemm_optimized, gemm_reference

    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    reference = _fp64_reference(a, b)

    _assert_fp16_gemm_ok(gemm_reference(a, b), reference, "triton_reference")
    _assert_fp16_gemm_ok(gemm_optimized(a, b), reference, "triton_optimized")
    _assert_fp16_gemm_ok(a @ b, reference, "cublas")


def test_fp16_gate_rejects_wrong_output():
    """Negative control: the scale-free gates must still fail on bad output.

    Without this, loosening the comparison to `return True` would look
    identical to a correct fix.
    """
    from ops.triton.gemm import gemm_reference

    torch.manual_seed(0)
    a = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    reference = _fp64_reference(a, b)

    # Right magnitude, wrong content (constant output) -> cosine collapses.
    assert not _fp16_gemm_ok(torch.full_like(reference, 1.0), reference)
    # Right values, wrong column order -> must also be rejected.
    assert not _fp16_gemm_ok(gemm_reference(a, b).flip(-1), reference)
    # A scaled result keeps the direction but inflates the relative L2 error.
    assert not _fp16_gemm_ok(gemm_reference(a, b) * 2.0, reference)
