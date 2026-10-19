"""CUDA arithmetic of compact L5 modules; no model-quality or speedup claim."""

import pytest
import torch
from hqsb.backends.structure_modules import FactorizedLinear

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA target required"),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1, 16, 4, 16), (2, 7, 17, 5, 23)])
def test_two_real_gemms_match_float32_factor_reference(dtype, shape):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unsupported")
    torch.manual_seed(9)
    batch, tokens, inp, rank, out = shape
    module = FactorizedLinear(inp, out, rank, True, dtype=dtype).cuda().eval()
    x = torch.randn(batch, tokens, inp * 2, device="cuda", dtype=dtype)[..., ::2]
    with torch.inference_mode():
        reference = torch.nn.functional.linear(
            torch.nn.functional.linear(x.float(), module.left.weight.float()),
            module.right.weight.float(),
            module.right.bias.float(),
        )
        actual = module(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual.float(),
        reference,
        rtol=0.02,
        atol=0.003 if dtype == torch.float16 else 0.02,
    )
    assert module.left.weight.numel() + module.right.weight.numel() == rank * (
        inp + out
    )
    assert not hasattr(module, "weight")
