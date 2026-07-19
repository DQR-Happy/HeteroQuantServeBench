"""Python → C ABI stream, allocator and alias handoff on the target GPU."""

import ctypes

import pytest

pytestmark = pytest.mark.hardware


@pytest.fixture
def device_bridge():
    torch = pytest.importorskip("torch")
    from ops.cuda_bridge import _CudaRmsnormBridge

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    bridge = _CudaRmsnormBridge()
    bridge._ensure_loaded()  # Missing build is an actionable hardware-test failure.
    return torch, bridge


def test_current_stream_is_forwarded_and_real_kernel_is_correct(
    device_bridge, monkeypatch
):
    torch, bridge = device_bridge
    native = bridge._lib.hqsb_rmsnorm_forward_ex_c
    seen = []

    def recording_call(*args):
        seen.append(ctypes.cast(args[-1], ctypes.c_void_p).value or 0)
        return native(*args)

    monkeypatch.setattr(bridge._lib, "hqsb_rmsnorm_forward_ex_c", recording_call)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(7, 128, device="cuda")
        w = torch.randn(128, device="cuda")
        expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * w
        out = bridge.forward(x, w)
        torch.testing.assert_close(out, expected, atol=5e-4, rtol=5e-4)
    stream.synchronize()
    assert seen == [stream.cuda_stream]


def test_alias_boundaries_reject_overlap_and_allow_exact_inplace(device_bridge):
    torch, bridge = device_bridge
    from ops.cuda_bridge import RmsNormContractError

    storage = torch.randn(257, device="cuda")
    x = storage[:256].view(2, 128)
    w = torch.ones(128, device="cuda")
    with pytest.raises(RmsNormContractError, match="ALIAS_INVALID"):
        bridge.forward(x, w, out=storage[1:].view(2, 128))
    with pytest.raises(RmsNormContractError, match="ALIAS_INVALID"):
        bridge.forward(x, x[0], out=x)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
    actual = bridge.forward(x, w, out=x)
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)


def test_stale_library_cannot_fallback_to_default_stream(device_bridge):
    torch, bridge = device_bridge
    from ops.cuda_bridge import CudaRmsnormUnavailable

    bridge._has_ex = False
    with pytest.raises(CudaRmsnormUnavailable, match="stream-aware"):
        bridge.forward(
            torch.ones(2, 128, device="cuda"), torch.ones(128, device="cuda")
        )
