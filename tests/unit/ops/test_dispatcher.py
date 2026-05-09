"""Unit tests for the unified operator dispatcher (S04).

These tests mock :class:`BackendCapabilities` so the dispatch *policy* is
verified on CPU without a GPU — the policy must be correct regardless of
what hardware is present.
"""

from __future__ import annotations

import pytest

from ops.capability import BackendCapabilities
from ops.dispatcher import DispatchDecision, OperatorDispatcher


def _caps(**overrides) -> BackendCapabilities:
    defaults = dict(
        cuda_available=True,
        device_capability=(8, 7),
        triton_available=True,
        triton_version="3.7.1",
        cutlass_available=False,
        cutlass_include_dir=None,
        tilelang_available=False,
        tilelang_version=None,
        cublas_available=True,
        cuda_rmsnorm_available=True,
        cuda_rmsnorm_lib="/fake/lib.so",
        notes=(),
    )
    defaults.update(overrides)
    return BackendCapabilities(**defaults)


@pytest.mark.unit
class TestRmsnormDispatch:
    def test_fp32_aligned_uses_cuda_v2(self):
        d = OperatorDispatcher(_caps())
        decision = d.select_rmsnorm("fp32", hidden=2048)
        assert decision.backend == "cuda"
        assert decision.variant == "v2_vectorized"

    def test_fp32_unaligned_uses_cuda_v1(self):
        d = OperatorDispatcher(_caps())
        decision = d.select_rmsnorm("fp32", hidden=101)
        assert decision.backend == "cuda"
        assert decision.variant == "v1_warp_shuffle"

    def test_fp16_always_cuda_v2(self):
        d = OperatorDispatcher(_caps())
        decision = d.select_rmsnorm("fp16", hidden=3)
        assert decision.backend == "cuda"
        assert decision.variant == "v2_vectorized"

    def test_arch_mismatch_falls_back_to_triton(self):
        # CUDA shared lib built for sm_87; device is sm_80.
        d = OperatorDispatcher(_caps(device_capability=(8, 0)))
        decision = d.select_rmsnorm("fp32", hidden=2048)
        assert decision.backend == "triton"

    def test_no_cuda_lib_uses_triton(self):
        d = OperatorDispatcher(_caps(cuda_rmsnorm_available=False, cuda_rmsnorm_lib=None))
        decision = d.select_rmsnorm("fp32", hidden=2048)
        assert decision.backend == "triton"

    def test_no_accelerator_uses_torch(self):
        d = OperatorDispatcher(
            _caps(
                cuda_rmsnorm_available=False,
                triton_available=False,
                cublas_available=False,
            )
        )
        decision = d.select_rmsnorm("fp32", hidden=2048)
        assert decision.backend == "torch"

    def test_cutlass_never_selected_when_unavailable(self):
        d = OperatorDispatcher(_caps(cutlass_available=False))
        decision = d.select_rmsnorm("fp32", hidden=2048)
        assert decision.backend != "cutlass"


@pytest.mark.unit
class TestGemmDispatch:
    def test_cublas_preferred(self):
        d = OperatorDispatcher(_caps())
        decision = d.select_gemm("fp16")
        assert decision.backend == "cublas"

    def test_triton_when_no_cublas(self):
        d = OperatorDispatcher(_caps(cublas_available=False))
        decision = d.select_gemm("fp16")
        assert decision.backend == "triton"

    def test_torch_fallback(self):
        d = OperatorDispatcher(_caps(cublas_available=False, triton_available=False))
        decision = d.select_gemm("fp32")
        assert decision.backend == "torch"


@pytest.mark.unit
class TestDispatchDecision:
    def test_as_dict(self):
        d = DispatchDecision("cuda", "v2", "reason")
        assert d.as_dict() == {"backend": "cuda", "variant": "v2", "reason": "reason"}
