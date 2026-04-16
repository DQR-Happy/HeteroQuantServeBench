#!/usr/bin/env python3
"""Generate golden numerical reference outputs for model-level regression.

Produces a structured JSON file containing:
- Input token IDs (for exact reproduction)
- Generated token IDs (for token-level agreement check)
- First-token top-K logits (for numerical error analysis)
- First-token logits L2 norm (for overall magnitude check)

This golden reference is the ground truth for future custom CUDA
operator validation. Any replacement operator must produce outputs
that match this reference within acceptable numerical tolerance.

Usage:
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    python benchmarks/scripts/generate_golden.py \\
        --input-tokens 128 \\
        --output-tokens 32 \\
        --top-k 32 \\
        --output benchmarks/workloads/golden/isl128_osl32.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time

import torch

from hqsb.benchmark.workload import make_fixed_token_input
from hqsb.models.loader import load_qwen3

logger = logging.getLogger(__name__)


def _hash_config(model_path: str) -> str:
    """Compute SHA256 hash of model config.json."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return "unavailable"
    with open(config_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@torch.inference_mode()
def generate_golden_reference(
    model: torch.nn.Module,
    inputs: dict,
    output_tokens: int,
    top_k: int = 32,
) -> dict:
    """Generate a golden reference for a given input/output configuration.

    Args:
        model: Loaded HF causal LM in eval mode.
        inputs: Dict with ``input_ids`` and ``attention_mask``.
        output_tokens: Number of tokens to generate.
        top_k: Number of top logits to record for first token.

    Returns:
        Dictionary with golden reference data.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    device = input_ids.device

    input_token_list = input_ids[0].tolist()

    # ── Prefill ──────────────────────────────────────────────────
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )

    logits = outputs.logits[0, -1, :]  # (vocab_size,)

    # First token: top-K and L2 norm
    top_k_values, top_k_indices = torch.topk(logits.float(), k=min(top_k, logits.shape[0]))

    first_token_data = {
        "token_id": int(top_k_indices[0].item()),
        "top_k_token_ids": top_k_indices.tolist(),
        "top_k_logits": [round(v.item(), 8) for v in top_k_values],
        "logits_l2_norm": round(torch.norm(logits.float(), p=2).item(), 8),
    }

    # ── Decode ───────────────────────────────────────────────────
    next_token = top_k_indices[0:1].unsqueeze(0)  # (1, 1)
    past_key_values = outputs.past_key_values

    generated_tokens = [int(next_token.item())]
    current_length = input_ids.shape[1]

    for _ in range(1, output_tokens):
        current_length += 1
        decode_mask = torch.ones(
            (1, current_length), dtype=torch.long, device=device
        )
        outputs = model(
            input_ids=next_token,
            attention_mask=decode_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        generated_tokens.append(int(next_token.item()))

    return {
        "input_token_ids": input_token_list,
        "generated_tokens": generated_tokens,
        "first_token": first_token_data,
    }


def main() -> None:
    """Parse arguments and generate golden reference."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Generate golden numerical reference for model regression",
    )
    parser.add_argument(
        "--input-tokens", type=int, required=True,
        help="Input sequence length",
    )
    parser.add_argument(
        "--output-tokens", type=int, required=True,
        help="Output sequence length",
    )
    parser.add_argument(
        "--top-k", type=int, default=32,
        help="Number of top logits to record (default: 32)",
    )
    parser.add_argument(
        "--model-path", default="~/models/hqsb/Qwen3-1.7B",
        help="Local model directory",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)

    logger.info("Loading model from: %s", model_path)
    tokenizer, model, load_time = load_qwen3(
        model_path, dtype=torch.float16, attention_backend="eager",
    )
    logger.info("Model loaded in %.2f s", load_time)

    logger.info("Generating workload: ISL=%d", args.input_tokens)
    workload = make_fixed_token_input(tokenizer, args.input_tokens)

    logger.info(
        "Generating golden reference: ISL=%d, OSL=%d, top_k=%d",
        args.input_tokens,
        args.output_tokens,
        args.top_k,
    )

    golden = generate_golden_reference(
        model, workload, args.output_tokens, top_k=args.top_k,
    )

    # ── Build full output ────────────────────────────────────────
    output = {
        "schema_version": "1.0.0",
        "timestamp": time.time(),
        "model": {
            "id": "Qwen/Qwen3-1.7B",
            "source": "modelscope",
            "dtype": "float16",
            "config_hash": _hash_config(model_path),
        },
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "input_token_ids": golden["input_token_ids"],
        "generated_tokens": golden["generated_tokens"],
        "first_token": golden["first_token"],
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0),
        },
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Golden reference saved to: %s", args.output)
    logger.info(
        "First token: %d, generated: %d tokens",
        golden["first_token"]["token_id"],
        len(golden["generated_tokens"]),
    )


if __name__ == "__main__":
    main()
