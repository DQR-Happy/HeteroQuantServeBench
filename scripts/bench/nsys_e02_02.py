#!/usr/bin/env python3
"""E02-02 NSYS driver: one workload with NVTX phase ranges.

Emits an outer NVTX range ``e02_02_model_core`` plus inner ranges
``prefill``, ``first-token-selection``, ``decode-early`` and ``decode-late``
so that ``nsys profile --capture-range=nvtx --nvtx-capture=e02_02_model_core``
captures exactly the model-core region and excludes model loading and warmup.

This is an audit tool: it cross-checks the two-layer PyTorch-profiler
attribution (host op vs device kernel) against a timeline, and confirms each
kernel appears once and in the expected phase order. It is not the E02-02
shape census and not a latency baseline.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import torch

from hqsb.benchmark.workload import make_fixed_token_input
from hqsb.models.loader import load_qwen3

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _model_step(model, token, mask, past):
    outputs = model(
        input_ids=token,
        attention_mask=mask,
        past_key_values=past,
        use_cache=True,
    )
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return next_token, outputs.past_key_values


def main() -> int:
    parser = argparse.ArgumentParser(description="E02-02 NSYS NVTX driver")
    parser.add_argument("--model-path", default="~/models/hqsb/Qwen3-1.7B")
    parser.add_argument(
        "--manifest",
        default=os.path.join(_REPO_ROOT, "docs", "benchmark", "model_sha256_manifest.txt"),
    )
    parser.add_argument("--isl", type=int, default=128)
    parser.add_argument("--osl", type=int, default=32)
    parser.add_argument("--early", type=int, default=1, help="last early decode step")
    parser.add_argument("--late", type=int, default=0, help="first late decode step (0=last)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, _ = load_qwen3(
        args.model_path,
        dtype=torch.float16,
        attention_backend="eager",
        verify_manifest=args.manifest,
        allow_extra=("model_sha256_manifest.txt",),
        cpu_staging=True,
    )
    inputs = make_fixed_token_input(tokenizer, args.isl, device=device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Warm up outside the captured region with a *small* input so the 8 GiB
    # unified-memory device does not spend its headroom on a full-length
    # prefill staging buffer before the real capture starts.
    warm = make_fixed_token_input(tokenizer, 32, device=device)
    _ = model(
        input_ids=warm["input_ids"],
        attention_mask=warm["attention_mask"],
        use_cache=True,
    )
    del warm, _
    gc.collect()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    last_step = args.osl - 1
    late_step = args.late or last_step
    early_step = max(args.early, 1)

    with torch.cuda.nvtx.range("e02_02_model_core"):
        torch.cuda.profiler.start()
        with torch.cuda.nvtx.range("prefill"):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        with torch.cuda.nvtx.range("first-token-selection"):
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values

        current_length = args.isl
        for step in range(1, args.osl):
            current_length += 1
            mask = torch.ones(
                (1, current_length), dtype=torch.long, device=device
            )
            label = None
            if step <= early_step:
                label = "decode-early"
            elif step >= late_step:
                label = "decode-late"
            if label is None:
                next_token, past_key_values = _model_step(
                    model, next_token, mask, past_key_values
                )
            else:
                with torch.cuda.nvtx.range(label):
                    next_token, past_key_values = _model_step(
                        model, next_token, mask, past_key_values
                    )

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    print(f"nsys driver done: ISL={args.isl} OSL={args.osl}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
