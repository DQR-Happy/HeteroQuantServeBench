#!/usr/bin/env python3
"""E02-02 runner: module/operator shape census across the six workloads.

Proves two things required by ``S02_实验清单.md`` E02-02 (see
``docs/stage_experiments/details/S02/E02-02_shape_census.md``):

1. **Coverage** — the census spans the complete decoder block (q/k/v/o +
   gate/up/down projections + input/post-attention norms), the KV cache, the
   LM head, and the input embedding, for every configured workload, split by
   prefill/decode.
2. **Run-generated data** — every shape/dtype/stride/layout/contiguity value
   is read off live tensors during a real forward pass (runtime hooks +
   PyTorch profiler); nothing is hand-copied.

Usage:

    python scripts/audit/run_e02_02_shape_census.py collect \
        --output-dir docs/stage_experiments/S02/E02-02/raw
    python scripts/audit/run_e02_02_shape_census.py verify \
        --output-dir docs/stage_experiments/S02/E02-02/raw
"""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

from hqsb.benchmark.shape_census import collect_shape_census
from hqsb.benchmark.workload import make_fixed_token_input
from hqsb.benchmark.workload_config import load_workload_dicts
from hqsb.models.loader import load_qwen3

logger = logging.getLogger("e02_02")

_REPO_ROOT = Path(__file__).resolve().parents[2]

_WORKLOAD_YAML = _REPO_ROOT / "configs" / "benchmarks" / "jetson_qwen3_fp16.yaml"

# Submodule paths that must be present for every decoder layer (the "complete
# decoder block" coverage criterion). Names follow the Qwen3 transformers
# architecture (``model.layers.<i>.<sub>``).
_REQUIRED_LAYER_SUBMODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "input_layernorm",
    "post_attention_layernorm",
]

# KV cache tensors are (batch=1, num_kv_heads=8, seq, head_dim=128) for
# Qwen3-1.7B. The seq dimension grows during decode, so we match the shape
# prefix/suffix rather than the exact length.
_KV_SHAPE_RE = re.compile(r"^\[1, 8, \d+, 128\]$")


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
        return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    except OSError:
        return None


def _layer_indices(module_paths: Set[str]) -> List[int]:
    """Extract decoder-layer indices from paths like ``model.layers.<N>``."""
    indices: Set[int] = set()
    for path in module_paths:
        m = re.match(r"^model\.layers\.(\d+)(?:\.|$)", path)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def _coverage_checks(census: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the E02-02 coverage facts from a single workload census."""
    paths: Set[str] = set()
    for record in census["modules"]:
        paths.add(record["module"])

    layer_indices = _layer_indices(paths)

    # Complete decoder block: every present layer must expose all 9 submodules.
    missing_per_layer: Dict[int, List[str]] = {}
    for idx in layer_indices:
        missing = [
            sub
            for sub in _REQUIRED_LAYER_SUBMODULES
            if f"model.layers.{idx}.{sub}" not in paths
        ]
        if missing:
            missing_per_layer[idx] = missing

    # LM head + final norm + input embedding.
    structural = {
        "lm_head": "lm_head" in paths,
        "final_norm": "model.norm" in paths,
        "embed_tokens": "model.embed_tokens" in paths,
    }

    # KV cache: ``past_key_values`` (key_cache/value_cache) are exposed in the
    # model/root outputs (transformers 5.x attention returns only
    # ``(attn_output, attn_weights)``), so any module output containing a
    # ``(1, 8, seq, 128)`` key/value tensor proves the KV cache was captured.
    # We require it in *both* phases (prefill seq=ISL, decode seq=ISL+step).
    kv_shapes: List[str] = []
    kv_phases: Set[str] = set()
    for record in census["modules"]:
        for shape in record.get("output_shapes", []):
            if _KV_SHAPE_RE.match(shape):
                kv_shapes.append(shape)
                kv_phases.add(record["phase"])

    kv_covered = bool(kv_shapes) and kv_phases == {"prefill", "decode"}

    return {
        "layer_count": len(layer_indices),
        "layer_indices": layer_indices,
        "complete_decoder_block": not missing_per_layer and bool(layer_indices),
        "missing_layer_submodules": missing_per_layer,
        "structural": structural,
        "kv_cache_covered": kv_covered,
        "kv_phases": sorted(kv_phases),
        "kv_shapes": sorted(set(kv_shapes)),
    }


def _run_collect(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, load_time_s = load_qwen3(
        args.model_path,
        dtype=torch.float16,
        attention_backend="eager",
        verify_manifest=args.manifest,
        allow_extra=("model_sha256_manifest.txt",),
    )

    workloads = load_workload_dicts(str(args.workload_yaml))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Warmup so first-call effects (cuBLAS workspace, lazy init) do not leak
    # into the census operator tables.
    warm_inputs = make_fixed_token_input(tokenizer, 32, device=device)
    _ = model(
        input_ids=warm_inputs["input_ids"],
        attention_mask=warm_inputs["attention_mask"],
        use_cache=True,
    )
    if device == "cuda":
        torch.cuda.synchronize()

    workload_records: List[Dict[str, Any]] = []
    for spec in workloads:
        name = spec["name"]
        isl = int(spec["input_tokens"])
        osl = int(spec["output_tokens"])
        logger.info("Census %s (ISL=%d OSL=%d)", name, isl, osl)

        inputs = make_fixed_token_input(tokenizer, isl, device=device)
        census = collect_shape_census(model, inputs, osl)

        record = {
            "workload": {"name": name, "input_tokens": isl, "output_tokens": osl},
            "provenance": {
                "modules": "runtime_forward_hook",
                "operators": "torch_profiler_events",
                "shapes": "read_from_live_tensors",
            },
            "input_len": census["input_len"],
            "output_tokens": census["output_tokens"],
            "decode_steps": census["decode_steps"],
            "decode_steps_profiled": census["decode_steps_profiled"],
            "modules": census["modules"],
            "prefill_operators": census["prefill_operators"],
            "decode_operators": census["decode_operators"],
        }
        workload_records.append(record)

        out_path = output_dir / f"census_{name}.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.info("  wrote %s (%d module records)", out_path, len(census["modules"]))

        # Release per-workload profiler/cache memory before the next workload,
        # otherwise the 8 GiB unified-memory device OOMs during long decodes.
        del record, census
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "run_id": f"run_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}",
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "environment": _environment(),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "manifest_sha256": _manifest_hash(args.manifest),
        "workload_yaml": str(args.workload_yaml),
        "load_time_s": load_time_s,
        "workloads": [spec["name"] for spec in workloads],
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("wrote %s", meta_path)
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    census_files = sorted(output_dir.glob("census_*.json"))
    if not census_files:
        logger.error("no census_*.json found in %s", output_dir)
        return 2

    per_workload: Dict[str, Dict[str, Any]] = {}
    for path in census_files:
        census = json.loads(path.read_text(encoding="utf-8"))
        name = census["workload"]["name"]
        per_workload[name] = _coverage_checks(census)

    names = sorted(per_workload)
    all_complete = all(per_workload[n]["complete_decoder_block"] for n in names)
    all_kv = all(per_workload[n]["kv_cache_covered"] for n in names)
    all_structural = all(
        all(per_workload[n]["structural"].values()) for n in names
    )
    layer_counts = {per_workload[n]["layer_count"] for n in names}
    consistent_layers = len(layer_counts) == 1

    # Data must be run-generated: non-empty module records with shapes present.
    non_empty = all(
        len(census["modules"]) > 0 and any(
            r.get("output_shapes") or r.get("input_shapes") for r in census["modules"]
        )
        for census in (
            json.loads(p.read_text(encoding="utf-8")) for p in census_files
        )
    )

    verdict = {
        "coverage": per_workload,
        "complete_decoder_block_all_workloads": all_complete,
        "kv_cache_covered_all_workloads": all_kv,
        "structural_modules_all_workloads": all_structural,
        "consistent_layer_count": consistent_layers,
        "layer_counts": sorted(layer_counts),
        "run_generated_non_empty": non_empty,
        "workloads": names,
        "passed": (
            all_complete
            and all_kv
            and all_structural
            and consistent_layers
            and non_empty
        ),
    }

    out_path = output_dir / "verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    logger.info("wrote %s (passed=%s)", out_path, verdict["passed"])
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["passed"] else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E02-02 shape census")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="load model and census six workloads")
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--model-path", default="~/models/hqsb/Qwen3-1.7B")
    collect.add_argument(
        "--manifest",
        default=str(_REPO_ROOT / "docs" / "benchmark" / "model_sha256_manifest.txt"),
    )
    collect.add_argument("--workload-yaml", default=str(_WORKLOAD_YAML))
    collect.set_defaults(func=_run_collect)

    verify = sub.add_parser("verify", help="check coverage + run-generated data")
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
