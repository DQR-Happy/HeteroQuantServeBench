"""GPU correctness tests for Triton RMSNorm vs the S03 shared reference.

Thresholds are *identical* to the S03 CUDA suite (FP32 5e-4, FP16 2e-2) —
S04's contract forbids a weakened reference per implementation. Tests are
skipped when Triton or CUDA is unavailable.
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


@pytest.fixture(scope="module")
def triton_rmsnorm():
    from ops.triton.rmsnorm import rmsnorm_optimized, rmsnorm_reference, rmsnorm_torch

    return rmsnorm_reference, rmsnorm_optimized, rmsnorm_torch


@pytest.mark.parametrize("hidden", [1024, 2048])
def test_fp32_reference_matches_torch(triton_rmsnorm, hidden):
    ref, opt, torch_ref = triton_rmsnorm
    x = torch.randn(128, hidden, device="cuda", dtype=torch.float32)
    w = torch.randn(hidden, device="cuda", dtype=torch.float32)
    expected = torch_ref(x, w)
    assert (ref(x, w) - expected).abs().max().item() <= 5e-4


@pytest.mark.parametrize("hidden", [1024, 2048])
def test_fp32_optimized_matches_torch(triton_rmsnorm, hidden):
    _, opt, torch_ref = triton_rmsnorm
    x = torch.randn(128, hidden, device="cuda", dtype=torch.float32)
    w = torch.randn(hidden, device="cuda", dtype=torch.float32)
    expected = torch_ref(x, w)
    assert (opt(x, w) - expected).abs().max().item() <= 5e-4


def test_fp16_matches_torch(triton_rmsnorm):
    _, opt, torch_ref = triton_rmsnorm
    x = torch.randn(64, 2048, device="cuda", dtype=torch.float16)
    w = torch.randn(2048, device="cuda", dtype=torch.float16)
    expected = torch_ref(x, w)
    assert (opt(x, w) - expected).abs().max().item() <= 2e-2


def test_non_aligned_hidden(triton_rmsnorm):
    # hidden=100 is not a power of two and not a multiple of 4; the loop +
    # mask must still be correct (S04 negative: non-aligned input).
    ref, _, torch_ref = triton_rmsnorm
    x = torch.randn(16, 100, device="cuda", dtype=torch.float32)
    w = torch.randn(100, device="cuda", dtype=torch.float32)
    expected = torch_ref(x, w)
    assert (ref(x, w) - expected).abs().max().item() <= 5e-4


def test_dynamic_shapes(triton_rmsnorm):
    # Same kernel handles multiple shapes without recompilation semantics
    # differing from torch (S04 negative: dynamic shape).
    _, opt, torch_ref = triton_rmsnorm
    for rows, hidden in [(1, 512), (8, 2048), (512, 1024)]:
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.float32)
        w = torch.randn(hidden, device="cuda", dtype=torch.float32)
        expected = torch_ref(x, w)
        assert (opt(x, w) - expected).abs().max().item() <= 5e-4


def test_config_hash_is_stable_and_sensitive():
    from ops.triton.rmsnorm import rmsnorm_config_hash

    h1 = rmsnorm_config_hash(128, 2048, "fp32", 1e-5)
    h2 = rmsnorm_config_hash(128, 2048, "fp32", 1e-5)
    assert h1 == h2  # stable

    assert rmsnorm_config_hash(128, 2048, "fp32", 1e-5) != rmsnorm_config_hash(
        128, 2048, "fp16", 1e-5
    )  # dtype-sensitive
    assert rmsnorm_config_hash(128, 2048, "fp32", 1e-5) != rmsnorm_config_hash(
        128, 1024, "fp32", 1e-5
    )  # shape-sensitive
