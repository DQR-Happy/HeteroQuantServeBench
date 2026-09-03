"""FP16 integration tolerances, frozen before execution, not adoption gates.
Norms/activation/RoPE: atol=5e-3 rtol=3e-3; linear: atol=4e-2 rtol=5e-3.
"""
import os
import pytest

pytestmark = pytest.mark.hardware


@pytest.fixture(scope="module")
def runtime():
    from pathlib import Path
    from hqsb.observation.lease import DeviceLease
    import torch
    from hqsb.integration.torch_ops import register_model_ops
    manifest = os.environ.get("HQSB_KERNEL_MANIFEST")
    if not manifest:
        pytest.skip("explicit HQSB_KERNEL_MANIFEST required")
    lease = DeviceLease(Path.home()/".cache/hqsb/device0.lock")
    lease.acquire()
    native = register_model_ops(manifest)
    torch.manual_seed(1234)
    try:
        with torch.inference_mode():
            yield torch,native,torch.ops.hqsb
    finally:
        torch.cuda.synchronize()
        native.close()
        lease.release()


def reference_norm(x,w,eps):
    import torch
    f = x.float()
    return w * (f * torch.rsqrt(f.square().mean(-1,keepdim=True)+eps)).to(x.dtype)


@pytest.mark.parametrize("shape",[(1,128),(13,2048),(512,2048),(2,7,128),(3,129)])
@pytest.mark.parametrize("variant",[1,2])
def test_norm(runtime,shape,variant):
    torch,_,ops = runtime
    x = torch.randn(shape,device="cuda",dtype=torch.float16)
    w = torch.randn(shape[-1],device="cuda",dtype=torch.float16)
    if variant == 2 and shape[-1] % 2:
        with pytest.raises(ValueError,match="ALIGNMENT"):
            ops.l2_rms_norm(x,w,1e-6,variant)
        return
    y = ops.l2_rms_norm(x,w,1e-6,variant)
    torch.testing.assert_close(y,reference_norm(x,w,1e-6),atol=5e-3,rtol=3e-3)
    assert y.data_ptr() != x.data_ptr()


def test_residual_activation_rope(runtime):
    torch,_,ops = runtime
    x = torch.randn((2,3,4,128),device="cuda",dtype=torch.float16)
    r = torch.randn_like(x)
    w = torch.randn(128,device="cuda",dtype=x.dtype)
    y,updated = ops.l2_residual_norm(x,r,w,1e-6)
    torch.testing.assert_close(updated,x+r,atol=0,rtol=0)
    torch.testing.assert_close(y,reference_norm(x+r,w,1e-6),atol=5e-3,rtol=3e-3)
    assert len({t.data_ptr() for t in (x,r,y,updated)}) == 4
    torch.testing.assert_close(ops.l2_silu_mul(x,r),torch.nn.functional.silu(x)*r,atol=5e-3,rtol=3e-3)
    cos = torch.randn((2,3,128),device="cuda",dtype=x.dtype)
    sin = torch.randn_like(cos)
    qt = x.transpose(1,2)
    rotate = torch.cat((-qt[...,64:],qt[...,:64]),dim=-1)
    torch.testing.assert_close(ops.l2_rope(x,cos,sin),qt*cos[:,None]+rotate*sin[:,None],atol=0,rtol=0)


@pytest.mark.parametrize("m,n,k",[(1,73,129),(4,512,128),(16,2048,2048),(128,6144,2048),(33,127,256)])
def test_linear(runtime,m,n,k):
    torch,_,ops = runtime
    x = torch.randn((m,k),device="cuda",dtype=torch.float16)
    w = torch.randn((n,k),device="cuda",dtype=torch.float16)
    reference = torch.nn.functional.linear(x,w)
    for small in ([False,True] if m <= 4 else [False]):
        torch.testing.assert_close(ops.l2_linear(x,w,small),reference,atol=4e-2,rtol=5e-3)


def test_offsets_guards_and_current_stream(runtime):
    torch,_,ops = runtime
    base = torch.randn(3*128+1,device="cuda",dtype=torch.float16)
    x = base[1:].view(3,128)
    w = torch.ones(128,device="cuda",dtype=x.dtype)
    with pytest.raises(ValueError,match="ALIGNMENT"):
        ops.l2_rms_norm(x,w,1e-6,2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x.fill_(2)
        result = ops.l2_rms_norm(x,w,1e-6,1)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(result,reference_norm(x,w,1e-6),atol=0,rtol=0)
    with pytest.raises(ValueError,match="SIGNATURE"):
        ops.l2_rms_norm(x.t(),w,1e-6,1)


def test_nonfinite_and_zero_norm_semantics(runtime):
    torch,_,ops = runtime
    w = torch.ones(128,device="cuda",dtype=torch.float16)
    for value in (0.,float("nan"),float("inf")):
        x = torch.full((2,128),value,device="cuda",dtype=w.dtype)
        torch.testing.assert_close(ops.l2_rms_norm(x,w,1e-6,1),reference_norm(x,w,1e-6),
                                   atol=0,rtol=0,equal_nan=True)


@pytest.mark.parametrize("t,length",[(16,16),(1,17),(3,19)])
def test_sdpa_cache_offset_gqa_and_masks(runtime,t,length):
    torch,_,_ = runtime
    from hqsb.backends.model_kernels.sdpa import run
    q = torch.randn((1,4,t,128),device="cuda",dtype=torch.float16)
    k = torch.randn((1,2,length,128),device="cuda",dtype=q.dtype)
    v = torch.randn_like(k)
    visible = torch.arange(length,device="cuda")[None,:] <= (torch.arange(t,device="cuda")+length-t)[:,None]
    mask = torch.zeros((1,1,t,length),device="cuda",dtype=q.dtype).masked_fill(~visible,float("-inf"))
    key,value = k.repeat_interleave(2,dim=1),v.repeat_interleave(2,dim=1)
    weights = (q @ key.transpose(-2,-1))*128**-0.5 + mask
    reference = torch.softmax(weights.float(),dim=-1).to(q.dtype) @ value
    out = run(q,k,v,None,128**-0.5,"torch-sdpa-math-gqa-repeat-v1")
    torch.testing.assert_close(out.transpose(1,2),reference,atol=5e-3,rtol=3e-3)
    hidden = torch.zeros_like(mask,dtype=torch.bool)
    torch.testing.assert_close(run(q,k,v,hidden,1.,"torch-sdpa-math-gqa-repeat-v1"),torch.zeros_like(out),atol=0,rtol=0)
