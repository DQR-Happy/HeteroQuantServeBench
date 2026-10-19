"""Real two-GPU numerical/placement gate; never substitute one GPU for two."""

import json
import os
import subprocess
import sys
import torch
import pytest
from hqsb.observation.storage import sha256

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="Requires two distinct visible CUDA devices; CPU/rank simulation is insufficient",
    ),
]


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers

    root = tmp_path_factory.mktemp("l9-cuda-source")
    torch.manual_seed(17)
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            eos_token_id=None,
        )
    ).save_pretrained(root)
    raw = Tokenizer(models.WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="t0", bos_token="t1"
    ).save_pretrained(root)
    manifest = root.parent / "l9-cuda.sha256"
    manifest.write_text(
        "".join(
            f"{sha256(p)}  {p.name}\n" for p in sorted(root.iterdir()) if p.is_file()
        )
    )
    return {
        "model_path": str(root),
        "manifest": str(manifest),
        "attention": "eager",
        "context_limit": 128,
    }


def execute(root, source, strategy):
    from hqsb.core.parallel_execution import ParallelPlan

    root.mkdir()
    plan = ParallelPlan(
        strategy=strategy,
        devices=[0] if strategy == "single" else [0, 1],
        prompts=["t2 t3 t4", "t7 t8 t9"],
        max_output_tokens=7,
    )
    (root / "input.json").write_text(
        json.dumps({"run_id": "cuda-test", "config": source, "plan": plan.model_dump()})
    )
    args = [
        sys.executable,
        "-m",
        "hqsb.backends.parallel_launcher",
        "--standalone",
        f"--nproc-per-node={plan.ranks}",
        "--max-restarts=0",
        "--module",
        "hqsb.backends.parallel_rank",
        str(root / "input.json"),
    ]
    p = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "OMP_NUM_THREADS": "2"},
    )
    assert p.returncode == 0, p.stderr[-6000:]
    return [
        json.loads((root / f"rank-{i}.json").read_text()) for i in range(plan.ranks)
    ]


@pytest.mark.parametrize("strategy", ["dp", "tp", "layer_shard", "prefill_decode"])
def test_two_cuda_devices_match_single_tokens(tmp_path, source, strategy):
    base = execute(tmp_path / "single", source, "single")[0]
    candidate = execute(tmp_path / "candidate", source, strategy)
    if strategy == "dp":
        results = sorted(
            [r for rank in candidate for r in rank["requests"]],
            key=lambda r: r["index"],
        )
    elif strategy == "prefill_decode":
        results = candidate[1]["requests"]
    else:
        results = candidate[0]["requests"]
    assert [r["token_ids"] for r in results] == [
        r["token_ids"] for r in base["requests"]
    ]
    if strategy == "layer_shard":
        assert len({t["device"] for t in candidate[0]["parameters"]["tensors"]}) == 2
    if strategy == "tp":
        assert all(
            any(
                t["global_shape"] != t["local_shape"]
                for t in rank["parameters"]["tensors"]
            )
            for rank in candidate
        )
