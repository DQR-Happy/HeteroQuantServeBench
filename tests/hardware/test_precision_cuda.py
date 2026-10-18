"""Actual CUDA arithmetic for compact weights, odd tails, bias and batch size."""

import pytest

pytestmark = pytest.mark.hardware


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("shape", [(1, 37, 19), (7, 288, 96), (17, 64, 33)])
def test_fused_packed_linear_matches_explicit_decoding(bits, shape):
    import torch
    from hqsb.backends.precision_linear import PackedLinear, encode_weight

    m, k, n = shape
    torch.manual_seed(17)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.05
    bias = torch.randn(n, device="cuda", dtype=torch.float16) * 0.01
    packed, scale, zero, group = encode_weight(weight, bits, 32)
    module = PackedLinear(packed, scale, zero, k, group, bits, bias)
    if bits == 4:
        low = (packed & 15).to(torch.int8)
        high = (packed >> 4).to(torch.int8)
        codes = torch.stack([low, high], -1).flatten(1)[:, :k]
        codes = torch.where(codes >= 8, codes - 16, codes)
    else:
        codes = packed.view(torch.int8)
    indexes = torch.arange(k, device="cuda") // group
    reconstructed = (codes.float() - zero[:, indexes].float()) * scale[:, indexes]
    reference = torch.nn.functional.linear(x, reconstructed.half(), bias)
    with torch.inference_mode():
        actual = module(x)
    torch.testing.assert_close(actual, reference, atol=2e-3, rtol=2e-3)
    assert module.calls == 1 and not list(module.parameters())


@pytest.mark.parametrize("m,k,n", [(1, 37, 19), (7, 288, 96), (17, 64, 33)])
def test_int8_linear_matches_independent_integer_reference(m, k, n):
    import torch
    from hqsb.backends.precision_linear import Int8Linear

    torch.manual_seed(17)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    codes = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    scales = torch.rand(n, device="cuda") * 0.003 + 0.0001
    bias = torch.randn(n, device="cuda", dtype=torch.float16) * 0.01
    module = Int8Linear(codes, scales, bias)
    a_scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
    a = (x.float() / a_scale).round().clamp(-127, 127)
    reference = (
        (a.double() @ codes.double().t()).float() * a_scale * scales[None]
    ).half() + bias
    with torch.inference_mode():
        actual = module(x)
    torch.testing.assert_close(actual, reference, atol=2e-3, rtol=2e-3)


def test_receipt_does_not_count_quantized_module_bias_as_remaining_float_storage():
    import torch
    import bitsandbytes as bnb
    from types import SimpleNamespace
    from hqsb.backends.precision_runtime import receipt
    from hqsb.core.precision import PrecisionPlan

    model = torch.nn.Sequential(
        bnb.nn.Linear8bitLt(32, 32, bias=True).half().cuda(),
        torch.nn.Linear(32, 4).half().cuda(),
    )
    model.config = SimpleNamespace()
    with torch.inference_mode():
        model(torch.ones(1, 32, dtype=torch.float16, device="cuda"))
    provider = SimpleNamespace(
        model=model,
        config={"precision": "float16"},
        precision_plan=PrecisionPlan(method="bnb_int8"),
        precision_load={},
    )
    observed = receipt(provider)
    expected = sum(p.untyped_storage().nbytes() for p in model[1].parameters())
    assert observed["remaining_float_parameter_bytes"] == expected
    assert observed["modules"][0]["logical_weight_shape"] == [32, 32]
    assert observed["resolved_recipe"]["method"] == "bnb_int8"
    from hqsb.backends.tensor_memory import parameter_inventory

    inventory = parameter_inventory(model)
    assert any(row["kind"] == "quantization_auxiliary" for row in inventory["entries"])
    assert (
        inventory["unique_storage_bytes"]
        == observed["quantized_storage_bytes"] + expected
    )


def test_native_bundle_reload_preserves_persistent_fp32_state(tmp_path, monkeypatch):
    import torch
    from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from hqsb.backends.precision_artifacts import export_native, load_native
    from hqsb.backends.precision_runtime import _replace, _native_records
    from hqsb.core.precision import PrecisionPlan

    class MixedState(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(32, 32, dtype=torch.float16)
            self.lm_head = torch.nn.Linear(32, 4, dtype=torch.float16)
            self.register_buffer(
                "positional_state", torch.tensor([1.0001234], dtype=torch.float32)
            )
            self.config = LlamaConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                vocab_size=4,
            )

    model = _replace(MixedState().cuda(), PrecisionPlan(method="rtn", group_size=32))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<unk>": 0, "a": 1, "b": 2, "c": 3}, unk_token="<unk>")
        ),
        unk_token="<unk>",
    )
    records, payload = _native_records(model)
    destination = tmp_path / "bundle"
    export_native(model, tokenizer, destination, records, payload, {})
    # Substitute only model construction: exercise real serialization, state
    # coverage checks, dtype copies and CUDA transfer with a persistent FP32 leaf.
    monkeypatch.setattr(
        AutoModelForCausalLM, "from_config", lambda *a, **k: MixedState()
    )
    _, restored, _ = load_native(destination, torch.float16, "eager")
    assert restored.positional_state.dtype == torch.float32
    assert torch.equal(restored.positional_state, model.positional_state)
    assert torch.equal(restored.proj.packed, model.proj.packed)


def test_asymmetric_w4_matches_independent_affine_dequantization():
    import torch
    from hqsb.backends.precision_linear import PackedLinear, encode_codes

    torch.manual_seed(31)
    m, n, k, group = 7, 19, 37, 32
    codes = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="cuda")
    zeros = torch.randint(-7, 8, (n, 2), dtype=torch.int32, device="cuda")
    scales = torch.rand(n, 2, device="cuda") * 0.01 + 0.001
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    indexes = torch.arange(k, device="cuda") // group
    expected_weight = ((codes.float() - zeros[:, indexes]) * scales[:, indexes]).half()
    linear = PackedLinear(encode_codes(codes, 4), scales, zeros, k, group, 4)
    with torch.inference_mode():
        expected = torch.nn.functional.linear(x, expected_weight)
        actual = linear(x)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
