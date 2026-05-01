#!/usr/bin/env python3
"""Smoke test for Qwen3-1.7B model loading and generation.

Performs a minimal end-to-end test: loads the model from local
files, runs a short generation, and validates that the model
produces coherent output without OOM or CUDA errors.

This is the first checkpoint in Phase 2 and must pass before
running any benchmarks.

Usage:
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    python scripts/models/smoke_qwen3.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

import torch

from hqsb.models.loader import load_qwen3

logger = logging.getLogger(__name__)

MODEL_PATH = "~/models/hqsb/Qwen3-1.7B"


def main() -> None:
    """Run the smoke test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    model_path = os.path.expanduser(MODEL_PATH)

    # ── Check CUDA ──────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("FATAL: CUDA is not available. Smoke test requires GPU.", file=sys.stderr)
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA:   {torch.version.cuda}")
    print(f"Torch:  {torch.__version__}")
    print()

    # ── Load Model ──────────────────────────────────────────────────
    logger.info("Loading model...")
    try:
        tokenizer, model, load_time = load_qwen3(
            model_path,
            dtype=torch.float16,
            attention_backend="eager",
        )
    except FileNotFoundError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        print("\nModel not found. Download it first:")
        print("  python scripts/models/download_qwen3_modelscope.py")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: Model loading failed: {e}", file=sys.stderr)
        sys.exit(1)

    logger.info("Model loaded in %.2f s", load_time)

    # ── Memory Check ────────────────────────────────────────────────
    allocated_gb = torch.cuda.memory_allocated() / (1024**3)
    reserved_gb = torch.cuda.memory_reserved() / (1024**3)
    logger.info(
        "CUDA memory: allocated=%.2f GB, reserved=%.2f GB",
        allocated_gb,
        reserved_gb,
    )

    # ── Generate ────────────────────────────────────────────────────
    messages = [
        {"role": "user", "content": "Briefly explain what CUDA is."}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated = output[0, inputs["input_ids"].shape[1]:]

    # ── Results ─────────────────────────────────────────────────────
    input_count = inputs["input_ids"].shape[1]
    output_count = generated.numel()

    print()
    print("=" * 60)
    print("Generated text:")
    print("-" * 60)
    print(tokenizer.decode(generated, skip_special_tokens=True))
    print("-" * 60)
    print(f"Input tokens   : {input_count}")
    print(f"Output tokens  : {output_count}")
    print(f"Generation time: {elapsed:.2f} s")
    print(f"Tokens/second  : {output_count / elapsed:.1f}" if elapsed > 0 else "")
    print("=" * 60)

    # ── Validation ──────────────────────────────────────────────────
    checks = [
        ("Model loaded", True),
        ("GPU visible", torch.cuda.is_available()),
        ("No OOM", True),  # If we got here, no OOM
        ("Text generated", output_count > 0),
        ("CUDA no errors", True),  # If we got here, no CUDA errors
    ]

    print()
    all_pass = True
    for name, status in checks:
        status_str = "PASS" if status else "FAIL"
        if not status:
            all_pass = False
        print(f"  [{status_str}] {name}")

    print()
    if all_pass:
        print("SMOKE TEST PASSED - Ready for benchmarks.")
    else:
        print("SMOKE TEST FAILED - Fix issues before proceeding.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
