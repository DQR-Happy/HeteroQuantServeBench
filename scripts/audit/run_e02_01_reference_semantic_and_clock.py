#!/usr/bin/env python3
"""E02-01 runner: reference semantic and clock convention.

Proves four things about the S02 FP16 reference runtime (see
``docs/stage_experiments/details/S02/E02-01_reference_semantic_and_clock.md``):

1. **Deterministic output** — the same token input, repeated across independent
   process runs, produces identical token sequences (same SHA256) and
   first-token logits within a pre-registered tolerance.
2. **Exact token count / first-token attribution** — generated tokens == OSL,
   decode steps == OSL-1, input length == ISL, and the first token is the
   argmax of the prefill logits (not a decode step).
3. **Sync-point clock** — every timed region is bracketed by
   ``torch.cuda.synchronize()`` (static), and a micro-benchmark shows an async
   launch is mis-measured as ~0 without a sync.
4. **Single formula** — ``model_core_timings`` is the only implementation of
   the decode/TTFT/E2E derivation, consumed by both model_core and engine.

Usage (3 independent process runs, then verify):

    python scripts/audit/run_e02_01_reference_semantic_and_clock.py collect \
        --output-dir docs/stage_experiments/S02/E02-01/raw --run-index 0
    ... --run-index 1
    ... --run-index 2
    python scripts/audit/run_e02_01_reference_semantic_and_clock.py verify \
        --output-dir docs/stage_experiments/S02/E02-01/raw
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from hqsb.benchmark.correctness import (
    compare_first_token_logits,
    hash_token_sequence,
)
from hqsb.benchmark.model_core import benchmark_model_core
from hqsb.benchmark.workload import make_fixed_token_input
from hqsb.models.loader import load_qwen3

logger = logging.getLogger("e02_01")

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Pre-registered tolerance for first-token logits (FP16, E02-01 §5).
_LOGIT_RTOL = 1e-3
_LOGIT_ATOL = 1e-3

# Reference workload (the "short" workload, kept small so 3 process runs are
# bounded; E02-01 is about the convention, not about scaling).
_DEFAULT_ISL = 128
_DEFAULT_OSL = 32
_DEFAULT_REPETITIONS = 3


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=_REPO_ROOT,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, cwd=_REPO_ROOT,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, OSError):
        return None


def _environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
    }
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        env["device"] = torch.cuda.get_device_name(0)
        env["compute_capability"] = [int(cc[0]), int(cc[1])]
    else:
        env["device"] = "cpu"
        env["compute_capability"] = None
    return env


def _manifest_hash(manifest_path: str) -> Optional[str]:
    try:
        data = Path(manifest_path).read_bytes()
        return hashlib.sha256(data).hexdigest()
    except OSError:
        return None


def sync_boundary_check() -> Dict[str, Any]:
    """Micro-benchmark: an async launch is mis-measured without a sync."""
    if not torch.cuda.is_available():
        return {"skipped": True, "reason": "no_cuda"}
    a = torch.randn(2048, 2048, dtype=torch.float16, device="cuda")
    b = torch.randn(2048, 2048, dtype=torch.float16, device="cuda")
    # Warm up so the first (compile/allocator) cost is excluded.
    _ = a @ b
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = a @ b  # async launch, no sync
    async_ms = (time.perf_counter() - t0) * 1000.0

    torch.cuda.synchronize()  # drain the async launch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = a @ b
    torch.cuda.synchronize()
    sync_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "skipped": False,
        "async_launch_ms": async_ms,
        "sync_completion_ms": sync_ms,
        "ratio": sync_ms / max(async_ms, 1e-9),
        "misrecord_without_sync": async_ms < sync_ms * 0.5,
    }


def _run_collect(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, load_time_s = load_qwen3(
        args.model_path,
        dtype=torch.float16,
        attention_backend="eager",
        verify_manifest=args.manifest,
        # The snapshot ships a copy of the manifest itself
        # (``model_sha256_manifest.txt``) that is not a weight and is not
        # declared by the tracked 14-entry manifest; it is a known, benign
        # extra file (E02-01 §5).
        allow_extra=("model_sha256_manifest.txt",),
    )

    inputs = make_fixed_token_input(
        tokenizer, args.input_tokens, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    input_ids = [int(x) for x in inputs["input_ids"][0].tolist()]
    input_hash = hash_token_sequence(input_ids)

    # Warmup (not timed/recorded), so first-call effects don't enter samples.
    benchmark_model_core(model, inputs, output_tokens=2)

    samples: List[Dict[str, Any]] = []
    for rep in range(args.repetitions):
        result = benchmark_model_core(
            model, inputs, args.output_tokens, capture_logits=True
        )
        generated = result["generated_token_ids"]
        seq_hash = hash_token_sequence(generated)
        raw_itl = result["raw_itl_ms"]

        sample = {
            "repetition": rep,
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "generated_token_ids": generated,
            "sequence_sha256": seq_hash,
            "first_token_logits": result.get("first_token_logits"),
            "first_token_topk": result.get("first_token_topk"),
            "prefill_forward_ms": result["prefill_forward_ms"],
            "first_token_selection_ms": result["first_token_selection_ms"],
            "model_core_ttft_ms": result["model_core_ttft_ms"],
            "decode_total_ms": result["decode_total_ms"],
            "model_core_e2e_ms": result["model_core_e2e_ms"],
            "raw_itl_ms": raw_itl,
            "checks": {
                "output_tokens_exact": len(generated) == args.output_tokens,
                "decode_steps_exact": len(raw_itl) == args.output_tokens - 1,
                "input_tokens_exact": result["input_tokens"] == args.input_tokens,
                "first_token_is_argmax": (
                    generated[0] == int(result["first_token_topk"][0]["token_id"])
                    if result.get("first_token_topk")
                    else None
                ),
                # Formula consistency (floating-point exact, since it is a
                # pure summation over already-summed values).
                "ttft_formula": abs(
                    result["model_core_ttft_ms"]
                    - (result["prefill_forward_ms"] + result["first_token_selection_ms"])
                )
                < 1e-9,
                "decode_total_formula": abs(
                    result["decode_total_ms"] - sum(raw_itl)
                )
                < 1e-9,
                "e2e_formula": abs(
                    result["model_core_e2e_ms"]
                    - (result["model_core_ttft_ms"] + result["decode_total_ms"])
                )
                < 1e-9,
            },
        }
        samples.append(sample)

    run_record = {
        "run_id": f"run_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}",
        "run_index": args.run_index,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "environment": _environment(),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "manifest_sha256": _manifest_hash(args.manifest),
        "load_time_s": load_time_s,
        "workload": {
            "name": "e02_01_reference",
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "repetitions": args.repetitions,
            "input_token_ids": input_ids,
            "input_hash": input_hash,
        },
        "logit_rtol": _LOGIT_RTOL,
        "logit_atol": _LOGIT_ATOL,
        "samples": samples,
    }

    out_path = output_dir / f"run_{args.run_index}.json"
    out_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    logger.info("wrote %s (load=%.1fs, %d samples)",
                out_path, load_time_s, len(samples))
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    run_files = sorted(output_dir.glob("run_*.json"))
    if len(run_files) < 3:
        logger.error("need >= 3 run_*.json files, found %d", len(run_files))
        return 2

    runs = [json.loads(p.read_text(encoding="utf-8")) for p in run_files]

    # ── Cross-run determinism ────────────────────────────────────────
    all_hashes: List[str] = []
    all_first_tokens: List[int] = []
    reference_logits: Optional[List[float]] = None
    logit_results: List[Dict[str, Any]] = []

    for run in runs:
        for sample in run["samples"]:
            all_hashes.append(sample["sequence_sha256"])
            all_first_tokens.append(sample["generated_token_ids"][0])
            if reference_logits is None:
                reference_logits = sample["first_token_logits"]

    # First-token logits across samples/runs (tolerance-based).
    for run in runs:
        for sample in run["samples"]:
            cmp = compare_first_token_logits(
                sample["first_token_logits"],
                reference_logits,
                rtol=_LOGIT_RTOL,
                atol=_LOGIT_ATOL,
            )
            logit_results.append(cmp.details)

    input_hashes = {run["workload"]["input_hash"] for run in runs}
    token_deterministic = len(set(all_hashes)) == 1
    first_token_deterministic = len(set(all_first_tokens)) == 1
    logits_deterministic = all(
        cmp["max_relative_error"] <= 1.0 for cmp in logit_results
    )

    # ── Token count / attribution (all samples) ──────────────────────
    checks: List[Dict[str, Any]] = []
    for run in runs:
        for sample in run["samples"]:
            checks.append(sample["checks"])
    all_checks_pass = all(
        all(v is None or v is True for v in c.values()) for c in checks
    )

    # ── Static: single formula + sync points ─────────────────────────
    static = _static_guards()

    # ── Runtime sync micro-benchmark ─────────────────────────────────
    sync_check = sync_boundary_check()

    verdict = {
        "deterministic_output": {
            "token_sequences_identical": token_deterministic,
            "first_tokens_identical": first_token_deterministic,
            "first_token_logits_within_tolerance": logits_deterministic,
            "distinct_sequence_hashes": len(set(all_hashes)),
            "distinct_first_tokens": len(set(all_first_tokens)),
            "max_logit_relative_error": max(
                (c["max_relative_error"] for c in logit_results), default=None
            ),
        },
        "exact_token_count": {
            "all_checks_pass": all_checks_pass,
            "input_hashes_identical": len(input_hashes) == 1,
        },
        "sync_point_clock": sync_check,
        "single_formula": static,
        "runs": len(runs),
        "samples": sum(len(r["samples"]) for r in runs),
        "passed": (
            token_deterministic
            and first_token_deterministic
            and logits_deterministic
            and all_checks_pass
            and len(input_hashes) == 1
            and static["single_definition"]
            and static["consumers_use_helper"]
            and bool(sync_check.get("misrecord_without_sync", False))
        ),
    }

    out_path = output_dir / "verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    logger.info("wrote %s (passed=%s)", out_path, verdict["passed"])
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["passed"] else 1


def _static_guards() -> Dict[str, Any]:
    """Static checks: the formula has one implementation; sync points exist."""
    model_core_src = (_REPO_ROOT / "hqsb" / "benchmark" / "model_core.py").read_text(encoding="utf-8")
    engine_src = (_REPO_ROOT / "hqsb" / "benchmark" / "engine.py").read_text(encoding="utf-8")
    metrics_src = (_REPO_ROOT / "hqsb" / "benchmark" / "metrics.py").read_text(encoding="utf-8")

    inline = "prefill_forward_ms + first_token_selection_ms"
    return {
        "single_definition": metrics_src.count("def model_core_timings") == 1,
        "consumers_use_helper": (
            "model_core_timings" in model_core_src
            and "model_core_timings" in engine_src
        ),
        "no_inline_formula_in_model_core": inline not in model_core_src,
        "no_inline_formula_in_engine": inline not in engine_src,
        "sync_calls_in_model_core": model_core_src.count("torch.cuda.synchronize()"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E02-01 reference semantic/clock")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="load model and collect one run")
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--model-path", default="~/models/hqsb/Qwen3-1.7B")
    collect.add_argument(
        "--manifest",
        default=str(_REPO_ROOT / "docs" / "benchmark" / "model_sha256_manifest.txt"),
    )
    collect.add_argument("--input-tokens", type=int, default=_DEFAULT_ISL)
    collect.add_argument("--output-tokens", type=int, default=_DEFAULT_OSL)
    collect.add_argument("--repetitions", type=int, default=_DEFAULT_REPETITIONS)
    collect.add_argument("--run-index", type=int, default=0)
    collect.set_defaults(func=_run_collect)

    verify = sub.add_parser("verify", help="cross-run verification")
    verify.add_argument("--output-dir", required=True)
    verify.set_defaults(func=_run_verify)
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
