#!/usr/bin/env python3
"""PyTorch Profiler capture for representative prefill and decode passes.

Produces a structured operator table (JSON) and a human-readable ranking
for one representative prefill-heavy and one decode-heavy workload. The
table feeds the Roofline/Amdahl hotspot classification in S02.

Usage:
    python scripts/bench/profile_model.py \
        --model-path ~/models/hqsb/Qwen3-1.7B \
        --output-dir reports/dev/profiler/<timestamp>
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys

import torch

from hqsb.core.contracts import ModelArtifact, WorkloadSpec
from hqsb.backends import PyTorchBackend
from hqsb.benchmark.profiling import profile_model_core
from hqsb.benchmark.roofline import rank_hotspots
from hqsb.benchmark.workload import make_fixed_token_input

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch Profiler capture")
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
        help="SHA256 manifest",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: reports/dev/profiler/<timestamp>)",
    )
    return parser


def _profile_case(backend, tokenizer, model, name, isl, osl, output_dir):
    workload = WorkloadSpec(
        name=name, input_tokens=isl, output_tokens=osl, repetitions=1
    )
    inputs = make_fixed_token_input(
        tokenizer, isl, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    profiler, table = profile_model_core(model, inputs, osl)

    # Save the raw table.
    with open(os.path.join(output_dir, f"{name}_operators.json"), "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=2)

    # Classify/rank top operators by self CUDA time.
    total_cuda = sum(r["cuda_time_us"] for r in table) or 1.0
    ranked = rank_hotspots(
        [
            {
                "name": r["name"],
                "time_share": r["cuda_time_us"] / total_cuda,
            }
            for r in table[:20]
        ]
    )
    return table, ranked, total_cuda


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args()
    os.chdir(_REPO_ROOT)

    artifact = ModelArtifact(
        model_id="Qwen/Qwen3-1.7B",
        source="modelscope",
        architecture="Qwen3ForCausalLM",
        dtype="float16",
    )
    backend = PyTorchBackend(
        model_path=args.model_path, verify_manifest=args.manifest
    )
    backend.load(artifact)
    tokenizer = backend._tokenizer
    model = backend._model

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        _REPO_ROOT, "reports", "dev", "profiler", run_id
    )
    os.makedirs(output_dir, exist_ok=True)

    report = {}
    for case in (
        ("prefill_heavy", 1024, 2),
        ("decode_heavy", 128, 16),
    ):
        name, isl, osl = case
        logger.info("Profiling %s (ISL=%d OSL=%d)", name, isl, osl)
        table, ranked, total_cuda = _profile_case(
            backend, tokenizer, model, name, isl, osl, output_dir
        )
        report[name] = {
            "total_cuda_time_us": total_cuda,
            "top_operators": [
                {
                    "name": c.name,
                    "time_share": round(c.time_share, 4),
                    "classification": c.classification.value,
                    "amdahl_max": round(c.amdahl_max, 3),
                }
                for c in ranked[:10]
            ],
        }

    with open(os.path.join(output_dir, "hotspot_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))
    backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
