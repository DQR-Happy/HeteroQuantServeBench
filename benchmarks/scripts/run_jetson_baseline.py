#!/usr/bin/env python3
"""Complete Jetson baseline benchmark orchestrator.

Runs all six workload configurations (tiny, short, balanced,
long_prefill, decode_heavy, long_balanced) sequentially against
Qwen3-1.7B FP16 on Jetson Orin Nano Super.

Each workload produces a separate JSON result file in a timestamped
output directory under ``reports/dev/llm/<run_id>/``.

Usage:
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    python benchmarks/scripts/run_jetson_baseline.py
"""

from __future__ import annotations

import datetime
import logging
import os
import subprocess
import sys
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ── Workload definitions ────────────────────────────────────────────
# (name, input_tokens, output_tokens)

WORKLOADS: List[Tuple[str, int, int]] = [
    ("tiny", 32, 16),
    ("short", 128, 32),
    ("balanced", 512, 128),
    ("long_prefill", 2048, 32),
    ("decode_heavy", 128, 256),
    ("long_balanced", 2048, 128),
]

REPETITIONS = 3


def _setup_logging() -> None:
    """Configure logging for the baseline orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _get_project_root() -> str:
    """Get the absolute path to the project root directory."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )


def _get_pythonpath(root: str) -> str:
    """Build PYTHONPATH with the project root prepended."""
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        return f"{root}:{existing}"
    return root


def main() -> None:
    """Execute all benchmark workloads sequentially."""
    _setup_logging()

    root = _get_project_root()
    os.chdir(root)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(root, "reports", "dev", "llm", run_id)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("HQSB Phase 2 Baseline Benchmark")
    logger.info("=" * 70)
    logger.info("Run ID    : %s", run_id)
    logger.info("Output dir: %s", output_dir)
    logger.info("Workloads : %d cases", len(WORKLOADS))
    logger.info("Repetitions: %d", REPETITIONS)
    logger.info("=" * 70)

    success_count = 0
    fail_count = 0

    for name, isl, osl in WORKLOADS:
        logger.info("")
        logger.info("-" * 70)
        logger.info("Case: %-16s  ISL=%5d  OSL=%5d", name, isl, osl)
        logger.info("-" * 70)

        output_path = os.path.join(output_dir, f"{name}.json")

        command = [
            sys.executable,
            os.path.join(root, "benchmarks", "scripts", "run_model_core.py"),
            "--input-tokens", str(isl),
            "--output-tokens", str(osl),
            "--repetitions", str(REPETITIONS),
            "--output", output_path,
        ]

        try:
            subprocess.run(
                command,
                check=True,
                env={
                    **os.environ,
                    "PYTHONPATH": _get_pythonpath(root),
                },
            )
            success_count += 1
            logger.info("Case '%s' completed successfully", name)
        except subprocess.CalledProcessError as e:
            fail_count += 1
            logger.error("Case '%s' FAILED with exit code %d", name, e.returncode)

    # ── Summary ──────────────────────────────────────────────────────

    logger.info("")
    logger.info("=" * 70)
    logger.info("Baseline Complete")
    logger.info("=" * 70)
    logger.info("Success: %d/%d", success_count, len(WORKLOADS))
    logger.info("Failed : %d/%d", fail_count, len(WORKLOADS))
    logger.info("Output : %s", output_dir)

    # List result files
    if os.path.isdir(output_dir):
        files = sorted(f for f in os.listdir(output_dir) if f.endswith(".json"))
        logger.info("Files  : %s", ", ".join(files))

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
