"""Real CUDA/Inductor numerics with state mutation, distinct inputs and KV reset."""
import pytest

pytestmark = pytest.mark.hardware


@pytest.mark.parametrize("diagnostic", [None, "inductor"])
def test_inductor_decode_matches_eager_distinct_prompts_and_restores(tmp_path, diagnostic):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from hqsb.backends.runtime_session import RuntimeSession
    from hqsb.core.runtime_plan import RuntimePlan

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attn_implementation="eager")).eval().cuda()

    def decode(tokens):
        ids = torch.tensor([tokens], device="cuda")
        mask = torch.ones_like(ids)
        cache, values = None, []
        for _ in range(4):
            output = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
            values.append(output.logits[:, -1].detach().clone())
            ids = output.logits[:, -1].argmax(-1)[:, None]
            mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
            cache = output.past_key_values
        return values

    with torch.inference_mode():
        references = [decode(tokens) for tokens in ([1, 3, 7], [1, 9, 12, 4])]
        session = RuntimeSession(model, RuntimePlan(mode="inductor", shape_policy="dynamic").model_dump(), diagnostic)
        for index, tokens in enumerate(([1, 3, 7], [1, 9, 12, 4])):
            with session.activate({"capture_id": f"run_{index}", "capture_dir": str(tmp_path)}):
                actual = decode(tokens)
                summary = session.summary()
                assert summary["status"] == "invoked" and summary["cuda_graphs"] is False
                for x, y in zip(actual, references[index]):
                    torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-5)
                    assert torch.equal(x.argmax(-1), y.argmax(-1))
        if diagnostic:
            assert session.views and all(v["status"] == "available" for v in session.views)
    assert "forward" not in model.__dict__
    torch._dynamo.reset()
