"""Unit tests for the ctypes bridge and cross-backend consistency (S04).

Verifies the CUDA bridge produces results identical to Triton and torch for
the *same* input (cross-backend consistency), plus the negative contract
that the bridge raises a clear error when the shared library is missing.
"""

from __future__ import annotations

import pytest
import torch

from ops.capability import detect_capabilities
from ops.cuda_bridge import CudaRmsnormUnavailable

_caps = detect_capabilities()
_HAS_CUDA = _caps.cuda_rmsnorm_available and _caps.cuda_available

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not _HAS_CUDA, reason="CUDA RMSNorm lib not available"),
]


def test_cuda_rmsnorm_matches_torch():
    from ops.cuda_bridge import rmsnorm_forward
    from ops.triton.rmsnorm import rmsnorm_torch

    x = torch.randn(512, 2048, device="cuda", dtype=torch.float32)
    w = torch.randn(2048, device="cuda", dtype=torch.float32)
    expected = rmsnorm_torch(x, w)

    for variant in (0, 2, 3, 4):  # auto, v0, v1, v2
        out = rmsnorm_forward(x, w, dtype="fp32", variant=variant)
        assert (out - expected).abs().max().item() <= 5e-4


def test_cuda_rmsnorm_fp16():
    from ops.cuda_bridge import rmsnorm_forward
    from ops.triton.rmsnorm import rmsnorm_torch

    x = torch.randn(256, 2048, device="cuda", dtype=torch.float16)
    w = torch.randn(2048, device="cuda", dtype=torch.float16)
    expected = rmsnorm_torch(x, w)
    out = rmsnorm_forward(x, w, dtype="fp16", variant=4)
    assert (out - expected).abs().max().item() <= 2e-2


def test_cross_backend_consistency_cuda_vs_triton():
    """CUDA and Triton must agree on the SAME input within the shared tolerance."""
    from ops.cuda_bridge import rmsnorm_forward
    from ops.triton.rmsnorm import rmsnorm_optimized

    x = torch.randn(256, 2048, device="cuda", dtype=torch.float32)
    w = torch.randn(2048, device="cuda", dtype=torch.float32)

    cuda_out = rmsnorm_forward(x, w, dtype="fp32", variant=4)
    triton_out = rmsnorm_optimized(x, w)
    assert (cuda_out - triton_out).abs().max().item() <= 5e-4


def test_missing_lib_raises_clear_error(monkeypatch):
    # Simulate a missing shared library: the bridge must raise a *clear*
    # error, not a cryptic ctypes OSError (S04 negative path).
    from ops import cuda_bridge

    monkeypatch.setattr(
        cuda_bridge, "detect_capabilities",
        lambda: detect_capabilities().__class__(
            cuda_available=True,
            device_capability=(8, 7),
            triton_available=True,
            triton_version=None,
            cutlass_available=False,
            cutlass_include_dir=None,
            tilelang_available=False,
            tilelang_version=None,
            cublas_available=True,
            cuda_rmsnorm_available=False,
            cuda_rmsnorm_lib=None,
            notes=(),
        ),
    )
    monkeypatch.setattr(cuda_bridge._bridge, "_lib", None)

    with pytest.raises(CudaRmsnormUnavailable):
        cuda_bridge.rmsnorm_forward(torch.randn(1, 1, device="cuda"),
                                    torch.randn(1, device="cuda"))
