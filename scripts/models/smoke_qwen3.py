#!/usr/bin/env python3
"""Smoke test for Qwen3-1.7B model loading and generation.

Performs a minimal end-to-end test: loads the model from local files, runs a
short generation, and validates that the model produces coherent output without
OOM or device errors.

This is the first checkpoint in Phase 2 and must pass before running any
benchmarks.  It runs on whichever accelerator is present (CUDA or Ascend NPU)
and reports the backend it actually used, so the result can never be filed
under a device label the run did not touch.

Usage:
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    python scripts/models/smoke_qwen3.py [--device cuda|npu|cpu]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import torch

from hqsb.hardware import device as device_api
from hqsb.models.loader import load_qwen3

logger = logging.getLogger(__name__)

MODEL_PATH = "~/models/hqsb/Qwen3-1.7B"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-1.7B loading smoke test")
    parser.add_argument(
        "--device",
        default="",
        help="Accelerator to use (cuda/npu/cpu). Empty = auto-detect "
        "(HQSB_DEVICE overrides).",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.expanduser(MODEL_PATH),
        help=f"Local model directory (default: {MODEL_PATH})",
    )
    return parser


def main() -> None:
    """Run the smoke test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _build_parser().parse_args()
    model_path = os.path.expanduser(args.model_path)

    # ── Resolve the device actually available ───────────────────────────
    target = device_api.resolve_device(args.device)
    if not device_api.is_accelerator(target):
        print(
            f"FATAL: no accelerator available (resolved device={target}). "
            "The smoke test needs a GPU or an Ascend NPU.",
            file=sys.stderr,
        )
        sys.exit(1)

    described = device_api.describe(target)
    print(f"Backend: {described['kind']} ({described['device']})")
    print(f"Device:  {described['name'] or 'name UNAVAILABLE'}")
    print(f"Torch:   {torch.__version__}")
    print()

    # ── Load Model ──────────────────────────────────────────────────────
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

    # ── Memory Check ────────────────────────────────────────────────────
    # `reserved` is a CUDA-only concept; on Ascend it is reported as
    # UNAVAILABLE rather than being silently filled with the allocated number.
    allocated = device_api.memory_allocated(target)
    reserved = device_api.max_memory_reserved(target)
    logger.info(
        "%s memory: allocated=%s, reserved=%s",
        described["kind"].upper(),
        f"{allocated / (1024**3):.2f} GB" if allocated is not None else "UNAVAILABLE",
        f"{reserved / (1024**3):.2f} GB" if reserved is not None else "UNAVAILABLE",
    )

    # ── Generate ────────────────────────────────────────────────────────
    messages = [{"role": "user", "content": "Briefly explain what an AI accelerator is."}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(target)

    device_api.synchronize(target)
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    device_api.synchronize(target)
    elapsed = time.perf_counter() - start

    generated = output[0, inputs["input_ids"].shape[1]:]

    # ── Results ─────────────────────────────────────────────────────────
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

    # ── Validation ──────────────────────────────────────────────────────
    checks = [
        ("Model loaded", True),
        (f"{described['kind'].upper()} visible", device_api.is_accelerator(target)),
        ("No OOM", True),  # If we got here, no OOM
        ("Text generated", output_count > 0),
        ("Device no errors", True),  # If we got here, no device errors
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
        print(f"SMOKE TEST PASSED ({described['kind']}) - Ready for benchmarks.")
    else:
        print("SMOKE TEST FAILED - Fix issues before proceeding.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
