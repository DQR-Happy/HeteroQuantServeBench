"""Actual CUDA greedy verification and logit-selection contracts; no speed claim."""

import copy
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from hqsb.backends.decoding_engine import Decoder
from hqsb.backends.decoding_sampling import Sampler
from hqsb.core.decoding_plan import DecodingPlan

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA target required"),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("draft_size", [1, 4, 7])
@pytest.mark.parametrize("draft_kind", ["identical", "different"])
def test_cuda_block_verification_matches_serial_at_tail_and_full_acceptance(
    dtype, draft_size, draft_kind
):
    torch.manual_seed(31)
    m = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=256,
                eos_token_id=None,
            )
        )
        .to("cuda", dtype=dtype)
        .eval()
    )
    history = [1, 7, 23, 1, 7, 23]
    p = DecodingPlan()
    base = Decoder(m, p, Sampler(p, "cuda", set()), history, 23, set(), lambda: None)
    expected = [t for g in base.rounds() for t in g]
    base.close()
    p = DecodingPlan(
        method="assistant", draft_deployment_id="same", draft_tokens=draft_size
    )
    draft = copy.deepcopy(m)
    if draft_kind == "different":
        torch.manual_seed(79)
        draft = LlamaForCausalLM(copy.deepcopy(m.config)).to("cuda", dtype=dtype).eval()
    d = Decoder(
        m,
        p,
        Sampler(p, "cuda", set()),
        history,
        23,
        set(),
        lambda: None,
        draft,
    )
    assert [t for g in d.rounds() for t in g] == expected
    if draft_kind == "identical":
        assert d.summary()["accepted_tokens"] > 0
    else:
        assert d.summary()["discarded_proposal_tokens"] > 0
    assert d.cache.get_seq_length() == len(history) + 22
    d.close()


def test_cuda_sampling_noncontiguous_logits_and_independent_rng():
    x = torch.linspace(-3, 3, 256, device="cuda")[::2]
    p = DecodingPlan(sampling="multinomial", temperature=0.8, top_p=0.9, top_k=17)
    a, b = Sampler(p, "cuda", set()), Sampler(p, "cuda", set())
    assert [a.select(x, [], []) for _ in range(20)] == [
        b.select(x, [], []) for _ in range(20)
    ]
