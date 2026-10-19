"""Independent full-sequence CE reference for the streaming CUDA PPL scorer."""

import json
import math
import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.hardware


@pytest.mark.parametrize("mode", ["incremental_decode", "full_prefill"])
def test_corpus_nll_matches_independent_cross_entropy(tmp_path, mode):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from hqsb.backends.corpus_evaluation import evaluate
    from hqsb.core.precision import PrecisionPlan

    torch.manual_seed(13)
    model = (
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
            )
        )
        .half()
        .cuda()
        .eval()
    )
    model.config._attn_implementation = "eager"
    tokens = [3, 7, 2, 8, 11, 12, 6]
    tokenizer = SimpleNamespace(
        encode=lambda *_args, **_kwargs: tokens,
        decode=lambda ids: " ".join(map(str, ids)),
    )
    provider = SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        config={"context_limit": 128, "attention": "eager", "precision": "float16"},
        source_identity={"hash": "fixture"},
        tokenizer_identity={"hash": "fixture"},
        source_environment={"source_digest": "fixture"},
        precision_plan=PrecisionPlan(),
        precision_load={},
        memory=lambda: {"allocated_bytes": torch.cuda.memory_allocated()},
    )
    plan = {
        "forward_mode": mode,
        "max_tokens": 64,
        "benchmark_id": "unit",
        "benchmark_sha256": "fixture",
        "samples": [{"id": "one", "slice": "reference", "text": "fixture"}],
    }
    request = {
        "capture_dir": str(tmp_path),
        "capture_id": "fixture",
        "remaining_ms": 30000,
        "max_output_tokens": 1,
        "input_mode": "completion",
    }
    with torch.inference_mode():
        x = torch.tensor([tokens], device="cuda")
        logits = model(input_ids=x, use_cache=False).logits[0, :-1].float()
        reference = torch.nn.functional.cross_entropy(
            logits, x[0, 1:], reduction="sum"
        ).item()
    events = list(evaluate(provider, request, threading.Event(), plan))
    result = json.loads((tmp_path / "quality-result.json").read_text())
    assert result["status"] == "complete" and result["scored_tokens"] == 6
    assert result["nll_sum"] == pytest.approx(reference, abs=0.003)
    assert result["perplexity"] == pytest.approx(math.exp(reference / 6), rel=0.001)
    assert events[-1]["metrics"]["measurement_role"] == "quality"
    assert not list(tmp_path.glob("quality-logits-*"))
    assert len(result["samples"][0]["token_nll"]) == 6


def test_corpus_cancel_keeps_partial_status_and_releases_local_references(tmp_path):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from hqsb.backends.corpus_evaluation import evaluate
    from hqsb.core.precision import PrecisionPlan

    model = (
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
            )
        )
        .half()
        .cuda()
        .eval()
    )
    model.config._attn_implementation = "eager"
    provider = SimpleNamespace(
        model=model,
        tokenizer=None,
        config={"context_limit": 128, "attention": "eager", "precision": "float16"},
        source_identity={},
        tokenizer_identity={},
        source_environment={"source_digest": "fixture"},
        precision_plan=PrecisionPlan(),
        precision_load={},
        memory=lambda: {"allocated_bytes": torch.cuda.memory_allocated()},
    )
    plan = {
        "forward_mode": "incremental_decode",
        "max_tokens": 64,
        "benchmark_id": "unit",
        "benchmark_sha256": "fixture",
        "samples": [{"id": "one", "slice": "reference", "text": "fixture"}],
    }
    cancelled = threading.Event()
    cancelled.set()
    request = {
        "capture_dir": str(tmp_path),
        "capture_id": "fixture",
        "remaining_ms": 30000,
        "max_output_tokens": 1,
    }
    events = list(evaluate(provider, request, cancelled, plan))
    assert events[-1]["finish_reason"] == "cancelled"
    assert (
        json.loads((tmp_path / "quality-result.json").read_text())["status"]
        == "partial"
    )
    assert not list(tmp_path.glob("quality-logits-*"))
    assert provider.model is model
