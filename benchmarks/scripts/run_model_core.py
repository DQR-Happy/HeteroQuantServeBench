#!/usr/bin/env python3
"""Single-case model-core benchmark runner.

Executes a single workload configuration (fixed ISL/OSL), collects
latency/throughput/memory metrics, validates determinism across
repetitions, and outputs a structured JSON result file.

Usage:
    python benchmarks/scripts/run_model_core.py \\
        --input-tokens 128 \\
        --output-tokens 32 \\
        --repetitions 3 \\
        --output reports/dev/llm/short.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
from typing import Any, Dict, List

import modelscope
import psutil
import torch
import transformers

from hqsb.benchmark.metrics import latency_summary
from hqsb.benchmark.model_core import benchmark_model_core
from hqsb.benchmark.resource_monitor import TegrastatsMonitor
from hqsb.benchmark.tegrastats_parser import (
    compute_power_summary,
    compute_resource_summary,
    parse_tegrastats_line,
)
from hqsb.benchmark.workload import make_fixed_token_input
from hqsb.models.loader import load_qwen3

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging for the benchmark runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_hardware_info() -> Dict[str, Any]:
    """Collect hardware identification information."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    if torch.cuda.is_available():
        info["device"] = torch.cuda.get_device_name(0)
        info["compute_capability"] = torch.cuda.get_device_capability(0)
        info["cuda_devices"] = torch.cuda.device_count()
    return info


def _build_software_info() -> Dict[str, Any]:
    """Collect software version information for reproducibility."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "modelscope": modelscope.__version__,
    }


def _compute_determinism(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute determinism metrics across repeated benchmark runs.

    Returns:
        Dict with ``deterministic`` (bool) and ``token_hash`` (str).
    """
    hashes: List[str] = []
    for run in runs:
        encoded = json.dumps(run["generated_token_ids"]).encode()
        hashes.append(hashlib.sha256(encoded).hexdigest())

    return {
        "deterministic": len(set(hashes)) == 1,
        "generated_token_sha256": hashes[0],
        "all_hashes": hashes,
    }


def main() -> None:
    """Parse arguments, load model, run benchmark, and write results."""
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="HQSB Model-Core Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-tokens", type=int, required=True,
        help="Input sequence length (ISL) in tokens",
    )
    parser.add_argument(
        "--output-tokens", type=int, required=True,
        help="Output sequence length (OSL) in tokens",
    )
    parser.add_argument(
        "--repetitions", type=int, default=3,
        help="Number of benchmark repetitions (default: 3)",
    )
    parser.add_argument(
        "--model-path", default="~/models/hqsb/Qwen3-1.7B",
        help="Local model directory path (default: ~/models/hqsb/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--skip-tegrastats", action="store_true",
        help="Skip tegrastats monitoring (for non-Jetson platforms)",
    )
    parser.add_argument(
        "--warmup-output-tokens", type=int, default=8,
        help="Output tokens for warmup run (default: 8)",
    )
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)

    # ── Load Model ──────────────────────────────────────────────────

    logger.info("Loading model from: %s", model_path)
    tokenizer, model, load_time_s = load_qwen3(
        model_path,
        dtype=torch.float16,
        attention_backend="eager",
    )
    logger.info("Model loaded in %.2f s", load_time_s)

    # ── Generate Workload ───────────────────────────────────────────

    workload = make_fixed_token_input(tokenizer, args.input_tokens)
    logger.info(
        "Workload: ISL=%d, OSL=%d, repetitions=%d",
        args.input_tokens,
        args.output_tokens,
        args.repetitions,
    )

    # ── Warmup ──────────────────────────────────────────────────────

    logger.info("Running warmup...")
    benchmark_model_core(
        model,
        workload,
        min(args.output_tokens, args.warmup_output_tokens),
    )
    logger.info("Warmup complete")

    # ── Benchmark Repetitions ───────────────────────────────────────

    runs: List[Dict[str, Any]] = []
    process = psutil.Process()

    for rep in range(args.repetitions):
        logger.info("Repetition %d/%d", rep + 1, args.repetitions)

        # Start tegrastats monitoring per repetition
        monitor = None
        tegrastats_parsed: List[Dict[str, Any]] = []

        if not args.skip_tegrastats:
            try:
                monitor = TegrastatsMonitor(interval_ms=100)
                monitor.start()
            except RuntimeError as e:
                logger.warning("tegrastats unavailable: %s", e)

        # Run benchmark
        result = benchmark_model_core(model, workload, args.output_tokens)

        # Stop monitoring and parse results
        if monitor is not None:
            monitor.stop()
            for record in monitor.records:
                parsed = parse_tegrastats_line(record["raw"])
                parsed["time_ns"] = record["time_ns"]
                tegrastats_parsed.append(parsed)

        # ── System memory ────────────────────────────────────────
        mem_info = process.memory_info()
        vmem = psutil.virtual_memory()

        result["process_rss_mb"] = mem_info.rss / (1024**2)
        result["system_memory_used_mb"] = vmem.used / (1024**2)
        result["system_memory_total_mb"] = vmem.total / (1024**2)
        result["system_memory_pct"] = vmem.percent

        # ── Tegrastats summary ───────────────────────────────────
        if tegrastats_parsed:
            resource_summary = compute_resource_summary(tegrastats_parsed)
            result["tegrastats_summary"] = resource_summary

            power_summary = compute_power_summary(tegrastats_parsed)
            result["tegrastats_power"] = power_summary

            # Energy per output token
            if power_summary.get("energy_j", 0) > 0 and args.output_tokens > 0:
                result["energy_j_per_output_token"] = (
                    power_summary["energy_j"] / args.output_tokens
                )

            result["tegrastats_sample_count"] = len(tegrastats_parsed)

        runs.append(result)

    # ── Determinism Check ───────────────────────────────────────────

    determinism = _compute_determinism(runs)

    # ── Aggregate ITL across repetitions ────────────────────────────

    all_itl_ms: List[float] = []
    for run in runs:
        itl_data = run.get("itl", {})
        if itl_data and "count" in itl_data:
            # Extract raw ITL values from each run's repetitions
            # We approximate by using the summary stats
            count = itl_data["count"]
            mean_ms = itl_data["mean_ms"]
            all_itl_ms.extend([mean_ms] * count)

    # ── Build Output ────────────────────────────────────────────────

    output: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "timestamp": time.time(),
        "hardware": _build_hardware_info(),
        "software": _build_software_info(),
        "model": {
            "source": "modelscope",
            "id": "Qwen/Qwen3-1.7B",
            "local_path": model_path,
            "dtype": "float16",
            "backend": "modelscope-transformers",
            "attention_backend": "eager",
            "load_time_s": load_time_s,
        },
        "workload": {
            "batch_size": 1,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
        },
        "deterministic": determinism["deterministic"],
        "generated_token_sha256": determinism["generated_token_sha256"],
        "repetitions": runs,
    }

    # ── Write Output ────────────────────────────────────────────────

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to: %s", args.output)
    logger.info("Deterministic: %s", determinism["deterministic"])

    # Quick summary
    if runs:
        avg_e2e = sum(r["model_core_e2e_ms"] for r in runs) / len(runs)
        avg_ttft = sum(r["model_core_ttft_ms"] for r in runs) / len(runs)
        logger.info(
            "Avg E2E: %.2f ms, Avg TTFT: %.2f ms",
            avg_e2e,
            avg_ttft,
        )


if __name__ == "__main__":
    main()
