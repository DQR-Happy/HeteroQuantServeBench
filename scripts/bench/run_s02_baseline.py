#!/usr/bin/env python3
"""S02 baseline orchestrator (contract-native).

Runs every workload defined in ``configs/benchmarks/jetson_qwen3_fp16.yaml``
against the :class:`PyTorchBackend` through the contract-based
:class:`BenchmarkEngine`, producing one versioned
:class:`BenchmarkResult` JSON per workload plus a normalized CSV summary.

This replaces the legacy ``run_jetson_baseline.py`` subprocess-based
orchestrator: the YAML file is now the single source of truth, and results
are bound to a model artifact hash, config hash, environment, and git commit.

Usage:
    python scripts/bench/run_s02_baseline.py \
        --config configs/benchmarks/jetson_qwen3_fp16.yaml \
        --model-path ~/models/hqsb/Qwen3-1.7B \
        --manifest docs/benchmark/model_sha256_manifest.txt
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Optional

import torch

from hqsb.core.contracts import (
    BenchmarkResult,
    EnvironmentInfo,
    ModelArtifact,
    WorkloadSpec,
)
from hqsb.core.errors import HqsbError
from hqsb.backends import PyTorchBackend
from hqsb.benchmark.engine import BenchmarkEngine
from hqsb.benchmark.workload_config import load_workload_specs
from hqsb.models.manifest import load_manifest

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S02 contract-native baseline")
    parser.add_argument(
        "--config",
        default=os.path.join(
            _REPO_ROOT, "configs", "benchmarks", "jetson_qwen3_fp16.yaml"
        ),
        help="Workload YAML config (single source of truth)",
    )
    parser.add_argument(
        "--model-path",
        default="~/models/hqsb/Qwen3-1.7B",
        help="Local model directory",
    )
    parser.add_argument(
        "--manifest",
        default=os.path.join(
            _REPO_ROOT, "docs", "benchmark", "model_sha256_manifest.txt"
        ),
        help="SHA256 manifest for artifact binding",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: reports/dev/llm/<timestamp>)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Override repetitions (default: from config or 3)",
    )
    return parser


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


def _environment() -> EnvironmentInfo:
    cc = None
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        cc = [int(capability[0]), int(capability[1])]
    return EnvironmentInfo(
        platform=platform.platform(),
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        compute_capability=cc,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "",
        framework_versions={
            "transformers": _transformers_version(),
            "modelscope": _modelscope_version(),
        },
    )


def _transformers_version() -> str:
    try:
        import transformers

        return transformers.__version__
    except Exception:
        return ""


def _modelscope_version() -> str:
    try:
        import modelscope

        return modelscope.__version__
    except Exception:
        return ""


def _build_artifact(model_path: str, manifest_path: str) -> ModelArtifact:
    entries = load_manifest(manifest_path)
    file_hashes = {e.normalized_path: e.sha256 for e in entries}
    return ModelArtifact(
        model_id="Qwen/Qwen3-1.7B",
        source="modelscope",
        architecture="Qwen3ForCausalLM",
        dtype="float16",
        file_hashes=file_hashes,
    )


def _write_result(result: BenchmarkResult, output_dir: str, name: str) -> str:
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(result.model_dump_json(indent=2))
    return path


def _write_summary(results: List[tuple], output_dir: str) -> str:
    """Write a normalized CSV summary from (name, workload, result) tuples."""
    path = os.path.join(output_dir, "summary.csv")
    fieldnames = [
        "name", "input_tokens", "output_tokens",
        "prefill_forward_ms_mean", "model_core_ttft_ms_mean",
        "decode_total_ms_mean", "model_core_e2e_ms_mean",
        "decode_tokens_per_s", "correctness_passed", "config_hash",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for name, workload, result in results:
            summary = result.summary
            writer.writerow({
                "name": name,
                "input_tokens": workload.input_tokens,
                "output_tokens": workload.output_tokens,
                "prefill_forward_ms_mean": summary.get("prefill_forward_ms_mean", ""),
                "model_core_ttft_ms_mean": summary.get("model_core_ttft_ms_mean", ""),
                "decode_total_ms_mean": summary.get("decode_total_ms_mean", ""),
                "model_core_e2e_ms_mean": summary.get("model_core_e2e_ms_mean", ""),
                "decode_tokens_per_s": summary.get("decode_tokens_per_s", ""),
                "correctness_passed": result.correctness.passed if result.correctness else "",
                "config_hash": result.config_hash,
            })
    return path


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _build_parser().parse_args()

    os.chdir(_REPO_ROOT)

    try:
        workloads = load_workload_specs(args.config)
        artifact = _build_artifact(args.model_path, args.manifest)
        backend = PyTorchBackend(
            model_path=args.model_path,
            verify_manifest=args.manifest,
        )
    except HqsbError as exc:
        logger.error("setup failed: %s", exc)
        return exc.exit_code

    # Load once; reuse across workloads.
    backend.load(artifact)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        _REPO_ROOT, "reports", "dev", "llm", run_id
    )
    os.makedirs(output_dir, exist_ok=True)

    environment = _environment()
    engine = BenchmarkEngine(backend)

    results: List[tuple] = []
    for workload in workloads:
        logger.info("Running workload %s (ISL=%d OSL=%d)",
                    workload.name, workload.input_tokens, workload.output_tokens)
        if args.repetitions is not None:
            workload = workload.model_copy(update={"repetitions": args.repetitions})
        try:
            result = engine.run(
                workload,
                artifact=artifact,
                environment=environment,
                git_commit=_git_commit(),
                git_dirty=_git_dirty(),
                load_artifact=False,
            )
        except HqsbError as exc:
            logger.error("workload %s failed: %s", workload.name, exc)
            return exc.exit_code

        path = _write_result(result, output_dir, workload.name)
        results.append((workload.name, workload, result))
        logger.info("  -> %s (correctness=%s)",
                    path, result.correctness.passed if result.correctness else "n/a")

    summary_path = _write_summary(results, output_dir)
    logger.info("Summary: %s", summary_path)

    backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
