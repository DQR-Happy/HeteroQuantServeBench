"""Small real CUDA models test masks, cache packing and independent storage."""
from types import SimpleNamespace

import pytest

from hqsb.core.contracts.scheduling import Step

pytestmark = pytest.mark.hardware


@pytest.mark.parametrize("family", ["llama", "qwen2", "qwen3"])
@pytest.mark.parametrize("precision", ["float32", "float16"])
def test_dense_variable_length_cuda_matches_isolated_teacher_forcing(family, precision):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
    from hqsb.backends.step_executor import StepExecutor

    if not torch.cuda.is_available():
        pytest.skip("CUDA target required")
    torch.manual_seed(123)
    config_cls, model_cls = {"llama": (LlamaConfig, LlamaForCausalLM), "qwen2": (Qwen2Config, Qwen2ForCausalLM), "qwen3": (Qwen3Config, Qwen3ForCausalLM)}[family]
    config = config_cls(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    config._attn_implementation = "sdpa"
    model = model_cls(config).cuda().to(getattr(torch, precision)).eval()
    provider = SimpleNamespace(model=model, config={"context_limit": 128}, kernel_session=None,
        memory=lambda: {"peak_allocated_bytes": torch.cuda.max_memory_allocated()})
    executor = StepExecutor(provider, 1024)
    histories = {"a": [1, 2, 3], "b": [4]}
    steps = [Step(k, 0, tuple(v), True, "prefill") for k, v in histories.items()]
    for iteration in range(6):
        result = executor.execute(steps, str(iteration))
        for step, token in zip(steps, result.tokens):
            with torch.inference_mode():
                reference = model(torch.tensor([histories[step.request_id]], device="cuda"), use_cache=True)
            assert token == reference.logits[:, -1].argmax(-1).item()
            layers = executor.kv.rows[step.request_id]
            assert all(t.shape[-2] == len(histories[step.request_id]) for pair in layers for t in pair)
            tolerance = 5e-3 if precision == "float16" else 1e-5
            for pair, isolated in zip(layers, reference.past_key_values.layers):
                for actual, expected in zip(pair, (isolated.keys, isolated.values)):
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            histories[step.request_id].append(token)
        if iteration == 1:
            executor.release("b")
            histories.pop("b")
            histories["c"] = [7, 8]
        steps = [Step(k, 0 if k not in executor.kv.rows else len(v) - 1,
                      tuple(v) if k not in executor.kv.rows else (v[-1],), True,
                      "prefill" if k not in executor.kv.rows else "decode") for k, v in histories.items()]
    a, c = executor.kv.rows["a"][0][0], executor.kv.rows["c"][0][0]
    assert a.untyped_storage().data_ptr() != c.untyped_storage().data_ptr()
    assert executor.close()["device_kv_rows"] == 0
