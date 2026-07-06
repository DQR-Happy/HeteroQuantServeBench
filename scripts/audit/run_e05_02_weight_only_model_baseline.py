#!/usr/bin/env python3
"""E05-02 model-level RTN W8/W4 quality, storage, memory and timing audit.

The low-bit artifacts are portable canonical storage.  The executable control
path validates and reconstructs each selected Linear weight into FP16 before
inference.  Every record therefore says ``fake-dequant`` and is ineligible for
a native low-bit speedup claim; E05-06 owns that claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from hqsb.benchmark.correctness import hash_token_sequence
from hqsb.benchmark.metrics import request_summary
from hqsb.benchmark.model_core import benchmark_model_core
from hqsb.benchmark.resource_monitor import TegrastatsMonitor
from hqsb.benchmark.tegrastats_parser import (
    compute_power_summary,
    compute_resource_summary,
    parse_tegrastats_line,
    slice_records,
)
from hqsb.benchmark.workload import make_fixed_token_input
from hqsb.benchmark.workload_config import load_workload_dicts
from hqsb.models.loader import load_qwen3
from hqsb.quant.model_weight_only import (
    apply_model_quant_artifact,
    artifact_disk_usage,
    load_manifest as load_model_quant_manifest,
    save_model_quant_artifact,
)


LOG = logging.getLogger("e05_02")
ROOT = _SCRIPT_ROOT
WORKLOAD_YAML = ROOT / "configs/benchmarks/jetson_qwen3_fp16.yaml"
MODEL_MANIFEST = ROOT / "docs/benchmark/model_sha256_manifest.txt"
MODEL_PATH = "~/models/hqsb/Qwen3-1.7B"
METHODS = ("fp16", "rtn_w8", "rtn_w4")
QUALITY_LENGTHS = (32, 64, 128)
PROFILE_ISL = 32
PROFILE_OSL = 2
GENERATION_ISL = 128
GENERATION_OSL = 64
DEFAULT_WARMUP = 1
DEFAULT_REPETITIONS = 2
DEFAULT_RUNS = 3
MONITOR_INTERVAL_MS = 250
ORDER_SEED = 5202

METHOD_SPECS: dict[str, dict[str, Any]] = {
    "fp16": {
        "bits": 16,
        "group_size": None,
        "storage": "source checkpoint",
        "execution": "original FP16 eager model",
        "native_low_bit_kernel": False,
    },
    "rtn_w8": {
        "bits": 8,
        "group_size": None,
        "scheme": "symmetric per-output-channel RTN",
        "storage": "canonical signed INT8 + FP32 scales",
        "execution": "validate + whole-weight FP16 dequant, then FP16 eager GEMM",
        "native_low_bit_kernel": False,
    },
    "rtn_w4": {
        "bits": 4,
        "group_size": 128,
        "scheme": "symmetric per-group RTN, axis=1",
        "storage": "canonical packed signed INT4 + FP32 scales",
        "execution": "validate + whole-weight FP16 dequant, then FP16 eager GEMM",
        "native_low_bit_kernel": False,
    },
}

QUALITY_GATES = {
    "rtn_w8": {
        "max_ppl_ratio": 1.02,
        "min_mean_logit_cosine": 0.999,
        "min_top1_agreement": 0.95,
        "min_task_accuracy_delta": 0.0,
    },
    "rtn_w4": {
        "max_ppl_ratio": 1.10,
        "min_mean_logit_cosine": 0.98,
        "min_top1_agreement": 0.80,
        "min_task_accuracy_delta": 0.0,
    },
}

TASKS = (
    {
        "id": "zh_capital",
        "slice": "zh_factual",
        "prompt": "请只回答城市名：中国的首都是哪里？",
        "accepted": ("北京",),
    },
    {
        "id": "math_multiply",
        "slice": "math",
        "prompt": "只回答最终数字：17乘以23等于多少？",
        "accepted": ("391",),
    },
    {
        "id": "code_length",
        "slice": "code",
        "prompt": "只回答输出数字：Python 表达式 len([1, 2, 3, 4]) 的结果是什么？",
        "accepted": ("4", "四"),
    },
    {
        "id": "en_factual",
        "slice": "en_factual",
        "prompt": "Answer with one word only: What planet is known as the Red Planet?",
        "accepted": ("Mars", "mars"),
    },
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash_from_manifest(path: Path) -> str:
    return _sha256_file(path)


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _process_rss() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return 0


def _cuda_memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "peak_allocated": 0, "peak_reserved": 0}
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
    }


def _environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "captured_at_utc": _utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
    }
    if torch.cuda.is_available():
        environment.update(
            {
                "device": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    return environment


def _preconditions() -> dict[str, Any]:
    e05_01 = ROOT / "docs/stage_experiments/S05/E05-01/raw/verdict.json"
    e04_10 = ROOT / "docs/stage_experiments/S04/E04-10/raw/verdict.json"
    s045_root = ROOT / "docs/stage_experiments/S04.5"
    e05_01_value = json.loads(e05_01.read_text()) if e05_01.is_file() else None
    e04_10_value = json.loads(e04_10.read_text()) if e04_10.is_file() else None
    s045_reports = sorted(str(path.relative_to(ROOT)) for path in s045_root.rglob("*.json")) \
        if s045_root.is_dir() else []
    return {
        "e05_01_verdict_path": str(e05_01.relative_to(ROOT)),
        "e05_01_pass": bool(e05_01_value and e05_01_value.get("overall") == "PASS"),
        "e04_10_verdict_path": str(e04_10.relative_to(ROOT)),
        "e04_10_status": e04_10_value.get("overall") if e04_10_value else None,
        "s04_5_m4_evidence_files": s045_reports,
        "s04_5_m4_verified": False,
        "formal_run_allowed": False,
        "reason": (
            "S04.5 M4 evidence is absent. E05-02 detail section 3 permits an "
            "exploratory run but forbids a formal PASS or E05-10 admission."
        ),
    }


def _resolved_workloads() -> list[dict[str, Any]]:
    return load_workload_dicts(str(WORKLOAD_YAML))


def _spec_payload() -> dict[str, Any]:
    workloads = _resolved_workloads()
    workload_bytes = WORKLOAD_YAML.read_bytes()
    model_hash = _tree_hash_from_manifest(MODEL_MANIFEST)
    quality_dataset = {
        "teacher_forcing_lengths": list(QUALITY_LENGTHS),
        "source": "hqsb.benchmark.workload fixed token generator",
        "tasks": list(TASKS),
        "generation": {"input_tokens": GENERATION_ISL, "output_tokens": GENERATION_OSL},
    }
    quality_dataset_hash = _sha256_bytes(
        json.dumps(quality_dataset, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    spec = {
        "schema": "hqsb.e05_02.experiment_spec/v1",
        "experiment_id": "E05-02",
        "frozen_at_utc": _utc_now(),
        "scope": "exploratory because S04.5 M4 evidence is absent",
        "model": {
            "id": "Qwen/Qwen3-1.7B",
            "local_path": MODEL_PATH,
            "revision": "local ModelScope snapshot bound by SHA256 manifest",
            "model_manifest_sha256": model_hash,
            "config_sha256": next(
                line.split()[0]
                for line in MODEL_MANIFEST.read_text().splitlines()
                if line.endswith("./config.json")
            ),
            "tokenizer_sha256": next(
                line.split()[0]
                for line in MODEL_MANIFEST.read_text().splitlines()
                if line.endswith("./tokenizer.json")
            ),
            "dtype": "float16",
            "attention_backend": "eager",
            "batch_size": 1,
            "cache": True,
            "decode": "greedy argmax",
        },
        "workloads": workloads,
        "workload_manifest_sha256": _sha256_bytes(workload_bytes),
        "methods": METHOD_SPECS,
        "module_policy": {
            "include": "all 2-D torch.nn.Linear weights",
            "exclude": ["lm_head", "embeddings", "normalization", "bias"],
        },
        "quality_dataset": quality_dataset,
        "quality_dataset_sha256": quality_dataset_hash,
        "quality_gates": QUALITY_GATES,
        "measurement": {
            "independent_processes": DEFAULT_RUNS,
            "warmup_per_workload": DEFAULT_WARMUP,
            "repetitions_per_workload": DEFAULT_REPETITIONS,
            "order_seed": ORDER_SEED,
            "monitor_interval_ms": MONITOR_INTERVAL_MS,
            "clock": "host monotonic/perf_counter with CUDA synchronization",
            "outliers": "retain all; label exceptions and thermal/power limitations",
            "cross_run_interval": "min/max of per-process medians (n=3)",
        },
        "execution_truth": {
            "w8_w4_path": "canonical storage -> validated whole-weight FP16 dequant -> FP16 eager",
            "native_low_bit_performance_claim": False,
            "expected_kernel": "FP16 eager/cublas kernels",
            "fallback_reason": "E05-06 native packed low-bit kernel is not implemented",
        },
        "preconditions": _preconditions(),
    }
    hashable = dict(spec)
    hashable.pop("frozen_at_utc")
    spec["spec_hash"] = _sha256_bytes(
        json.dumps(hashable, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    return spec


def command_spec(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    spec_path = output / "spec.json"
    if spec_path.exists() and not args.force:
        existing = json.loads(spec_path.read_text())
        proposed = _spec_payload()
        if existing.get("spec_hash") != proposed.get("spec_hash"):
            raise RuntimeError("frozen spec exists and differs; use a new output directory")
        print(json.dumps(existing, indent=2, ensure_ascii=False))
        return 0
    spec = _spec_payload()
    _json_write(spec_path, spec)
    print(json.dumps(spec, indent=2, ensure_ascii=False))
    return 0


def _load_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    rss_before = _process_rss()
    before = _cuda_memory()
    tokenizer, model, load_time_s = load_qwen3(
        args.model_path,
        dtype=torch.float16,
        attention_backend="eager",
        verify_manifest=args.model_manifest,
        allow_extra=("model_sha256_manifest.txt",),
        cpu_staging=args.cpu_staging,
    )
    param_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    memory = {
        "load_time_s": load_time_s,
        "rss_before": rss_before,
        "rss_after": _process_rss(),
        "cuda_before": before,
        "cuda_after": _cuda_memory(),
        "parameter_devices": param_devices,
        "fully_on_cuda": bool(param_devices) and all(
            device.startswith("cuda") for device in param_devices
        ),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return tokenizer, model, memory


def command_prepare(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    spec_path = output / "spec.json"
    if not spec_path.is_file():
        raise RuntimeError("run the spec command before prepare")
    spec = json.loads(spec_path.read_text())
    tokenizer, model, load_memory = _load_model(args)
    del tokenizer
    artifacts = output / "artifacts"
    records: dict[str, Any] = {}
    for method in ("rtn_w8", "rtn_w4"):
        method_dir = artifacts / method
        method_spec = METHOD_SPECS[method]
        manifest = save_model_quant_artifact(
            model,
            method_dir,
            bits=int(method_spec["bits"]),
            group_size=method_spec["group_size"],
            source_model_hash=spec["model"]["model_manifest_sha256"],
            source_revision=spec["model"]["revision"],
            excluded_suffixes=("lm_head",),
            row_chunk=args.row_chunk,
        )
        records[method] = {
            "artifact_id": manifest["artifact_id"],
            "coverage": manifest["coverage"],
            "offline": manifest["offline"],
            "disk": artifact_disk_usage(method_dir),
        }
        LOG.info("prepared %s: %s", method, manifest["artifact_id"])
    _json_write(
        output / "artifact_summary.json",
        {
            "created_at_utc": _utc_now(),
            "spec_hash": spec["spec_hash"],
            "source_load": load_memory,
            "methods": records,
        },
    )
    return 0


def _telemetry_summary(
    records: list[dict[str, Any]], begin_ns: int, end_ns: int, requests: int
) -> dict[str, Any]:
    window = slice_records(records, begin_ns, end_ns)
    parsed: list[dict[str, Any]] = []
    for item in window:
        fields = parse_tegrastats_line(item["raw"])
        fields["time_ns"] = item["time_ns"]
        parsed.append(fields)
    power = compute_power_summary(parsed)
    resource = compute_resource_summary(parsed)
    return {
        **resource,
        "window_begin_ns": begin_ns,
        "window_end_ns": end_ns,
        "records": len(window),
        "energy_usable": power["num_samples"] >= 3,
        "j_per_request": power["energy_j"] / requests if requests else None,
        "output": power,
    }


def _capture_names(model: torch.nn.Module) -> list[str]:
    modules = dict(model.named_modules())
    names: list[str] = []
    for suffix in ("layers.0.self_attn.q_proj", "layers.0.mlp.up_proj"):
        match = next((name for name in modules if name.endswith(suffix)), None)
        if match:
            names.append(match)
    blocks = [
        name for name, module in modules.items()
        if isinstance(module, torch.nn.ModuleList) and name.endswith("layers")
    ]
    if blocks:
        prefix = blocks[0]
        count = len(modules[prefix])
        for index in sorted({0, count // 2, count - 1}):
            name = f"{prefix}.{index}"
            if name in modules:
                names.append(name)
    return names


def _tensor_from_hook(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"unsupported hook output {type(value).__name__}")


def _save_array(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().to(dtype=torch.float16, device="cpu").contiguous().numpy()
    np.save(path, array, allow_pickle=False)
    real_path = path.with_suffix(".npy")
    return {
        "path": str(real_path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": real_path.stat().st_size,
        "sha256": _sha256_file(real_path),
    }


def _comparison(candidate: torch.Tensor, reference_path: Path) -> dict[str, Any]:
    reference_np = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    candidate = candidate.detach().float()
    reference = torch.from_numpy(np.asarray(reference_np).copy()).to(candidate.device).float()
    if candidate.shape != reference.shape:
        raise ValueError(f"comparison shape mismatch {candidate.shape} != {reference.shape}")
    delta = candidate - reference
    flat_candidate = candidate.reshape(-1)
    flat_reference = reference.reshape(-1)
    source_rms = torch.sqrt(torch.mean(flat_reference * flat_reference)).item()
    result = {
        "finite": bool(torch.isfinite(candidate).all().item()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(delta * delta)).item()),
        "nrmse": float(
            torch.sqrt(torch.mean(delta * delta)).item() / max(source_rms, 1e-30)
        ),
        "cosine": float(F.cosine_similarity(flat_candidate, flat_reference, dim=0).item()),
    }
    return result


def _logit_comparison(candidate: torch.Tensor, reference_path: Path) -> dict[str, Any]:
    reference_np = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    candidate = candidate.detach().float().squeeze(0)
    reference = torch.from_numpy(np.asarray(reference_np).copy()).to(candidate.device).float()
    if candidate.shape != reference.shape:
        raise ValueError(f"logit shape mismatch {candidate.shape} != {reference.shape}")
    positions: list[dict[str, Any]] = []
    for index in range(candidate.shape[0]):
        cand = candidate[index]
        ref = reference[index]
        delta = cand - ref
        ref_logp = F.log_softmax(ref, dim=0)
        cand_logp = F.log_softmax(cand, dim=0)
        ref_p = ref_logp.exp()
        cand_p = cand_logp.exp()
        mixture = (ref_p + cand_p) * 0.5
        js = 0.5 * (
            torch.sum(ref_p * (ref_logp - torch.log(mixture)))
            + torch.sum(cand_p * (cand_logp - torch.log(mixture)))
        )
        ref_top = torch.topk(ref, k=5)
        cand_top = torch.topk(cand, k=5)
        overlap = len(set(ref_top.indices.tolist()) & set(cand_top.indices.tolist())) / 5.0
        positions.append(
            {
                "position": index,
                "max_abs": float(delta.abs().max().item()),
                "rmse": float(torch.sqrt(torch.mean(delta * delta)).item()),
                "cosine": float(F.cosine_similarity(cand, ref, dim=0).item()),
                "kl_ref_to_candidate": float(torch.sum(ref_p * (ref_logp - cand_logp)).item()),
                "js": float(js.item()),
                "top1_reference": int(ref_top.indices[0].item()),
                "top1_candidate": int(cand_top.indices[0].item()),
                "top1_equal": bool(ref_top.indices[0] == cand_top.indices[0]),
                "top5_overlap": overlap,
                "reference_margin": float((ref_top.values[0] - ref_top.values[1]).item()),
                "candidate_margin": float((cand_top.values[0] - cand_top.values[1]).item()),
            }
        )
    return {
        "positions": positions,
        "summary": {
            "positions": len(positions),
            "finite": bool(torch.isfinite(candidate).all().item()),
            "max_abs": max(item["max_abs"] for item in positions),
            "rmse": math.sqrt(statistics.mean(item["rmse"] ** 2 for item in positions)),
            "mean_cosine": statistics.mean(item["cosine"] for item in positions),
            "mean_kl": statistics.mean(item["kl_ref_to_candidate"] for item in positions),
            "mean_js": statistics.mean(item["js"] for item in positions),
            "top1_agreement": statistics.mean(float(item["top1_equal"]) for item in positions),
            "mean_top5_overlap": statistics.mean(item["top5_overlap"] for item in positions),
            "first_top1_divergence": next(
                (item["position"] for item in positions if not item["top1_equal"]), None
            ),
        },
    }


@torch.inference_mode()
def _quality_eval(
    method: str,
    tokenizer: Any,
    model: torch.nn.Module,
    output_dir: Path,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    reference_root = output_dir / "reference"
    method_root = output_dir / "quality" / method
    reference_root.mkdir(parents=True, exist_ok=True)
    method_root.mkdir(parents=True, exist_ok=True)
    capture_names = _capture_names(model)
    modules = dict(model.named_modules())
    teacher_records: list[dict[str, Any]] = []
    operator_block: list[dict[str, Any]] = []

    for length in QUALITY_LENGTHS:
        inputs = make_fixed_token_input(tokenizer, length, device=str(device))
        captured: dict[str, torch.Tensor] = {}
        handles = []
        if length == QUALITY_LENGTHS[0]:
            for name in capture_names:
                handles.append(
                    modules[name].register_forward_hook(
                        lambda _module, _inputs, output, key=name: captured.__setitem__(
                            key, _tensor_from_hook(output).detach()
                        )
                    )
                )
        outputs = model(**inputs, use_cache=False)
        for handle in handles:
            handle.remove()
        logits = outputs.logits.detach()
        labels = inputs["input_ids"][:, 1:]
        nll = F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="mean",
        )
        record: dict[str, Any] = {
            "case_id": f"fixed_{length}",
            "tokens": length,
            "valid_next_tokens": length - 1,
            "nll": float(nll.item()),
            "ppl": float(torch.exp(nll).item()),
            "input_hash": hash_token_sequence(inputs["input_ids"][0].tolist()),
            "finite": bool(torch.isfinite(logits).all().item()),
        }
        reference_logits = reference_root / f"teacher_fixed_{length}_logits.npy"
        if method == "fp16":
            record["reference_file"] = _save_array(reference_logits, logits.squeeze(0))
            record["comparison"] = {
                "summary": {
                    "positions": length,
                    "finite": record["finite"],
                    "max_abs": 0.0,
                    "rmse": 0.0,
                    "mean_cosine": 1.0,
                    "mean_kl": 0.0,
                    "mean_js": 0.0,
                    "top1_agreement": 1.0,
                    "mean_top5_overlap": 1.0,
                    "first_top1_divergence": None,
                },
                "positions": [],
            }
        else:
            record["comparison"] = _logit_comparison(logits, reference_logits)
        teacher_records.append(record)

        if captured:
            for name, tensor in captured.items():
                safe = name.replace(".", "_")
                reference_tensor = reference_root / f"capture_{safe}.npy"
                item = {"name": name, "shape": list(tensor.shape)}
                if method == "fp16":
                    item["reference_file"] = _save_array(reference_tensor, tensor)
                    item["comparison"] = {
                        "finite": bool(torch.isfinite(tensor).all().item()),
                        "max_abs": 0.0,
                        "mean_abs": 0.0,
                        "rmse": 0.0,
                        "nrmse": 0.0,
                        "cosine": 1.0,
                    }
                else:
                    item["comparison"] = _comparison(tensor, reference_tensor)
                operator_block.append(item)

        del outputs, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tasks: list[dict[str, Any]] = []
    for task in TASKS:
        encoded = tokenizer(task["prompt"], return_tensors="pt", add_special_tokens=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            max_new_tokens=32,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[0, encoded["input_ids"].shape[1]:].tolist()
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        correct = any(answer in text for answer in task["accepted"])
        tasks.append(
            {
                **task,
                "output_text": text,
                "output_token_ids": new_tokens,
                "output_hash": hash_token_sequence(new_tokens),
                "correct": correct,
                "finite": True,
            }
        )

    generation_inputs = make_fixed_token_input(tokenizer, GENERATION_ISL, device=str(device))
    generation = benchmark_model_core(model, generation_inputs, GENERATION_OSL)
    generation_record = {
        "input_tokens": GENERATION_ISL,
        "output_tokens": GENERATION_OSL,
        "tokens": generation["generated_token_ids"],
        "sequence_hash": hash_token_sequence(generation["generated_token_ids"]),
        "first_divergence_vs_fp16": None,
        "nan_or_inf": False,
        "premature_stop": len(generation["generated_token_ids"]) != GENERATION_OSL,
    }
    fp16_generation_path = reference_root / "generation_tokens.json"
    if method == "fp16":
        _json_write(fp16_generation_path, generation_record)
    else:
        reference_generation = json.loads(fp16_generation_path.read_text())
        generation_record["first_divergence_vs_fp16"] = next(
            (
                index
                for index, (reference_token, token) in enumerate(
                    zip(reference_generation["tokens"], generation_record["tokens"])
                )
                if reference_token != token
            ),
            None,
        )

    aggregate_nll = sum(
        record["nll"] * record["valid_next_tokens"] for record in teacher_records
    ) / sum(record["valid_next_tokens"] for record in teacher_records)
    result = {
        "method": method,
        "teacher_forcing": teacher_records,
        "operator_block": operator_block,
        "tasks": tasks,
        "generation": generation_record,
        "summary": {
            "aggregate_nll": aggregate_nll,
            "aggregate_ppl": math.exp(aggregate_nll),
            "task_accuracy": statistics.mean(float(task["correct"]) for task in tasks),
            "all_finite": all(record["finite"] for record in teacher_records)
            and all(item["comparison"]["finite"] for item in operator_block),
            "minimum_logit_cosine": min(
                record["comparison"]["summary"]["mean_cosine"] for record in teacher_records
            ),
            "minimum_top1_agreement": min(
                record["comparison"]["summary"]["top1_agreement"] for record in teacher_records
            ),
        },
    }
    _json_write(method_root / "quality.json", result)
    return result


@torch.inference_mode()
def _profile_execution(
    method: str,
    tokenizer: Any,
    model: torch.nn.Module,
    output_dir: Path,
) -> dict[str, Any]:
    profile_root = output_dir / "profiler" / method
    profile_root.mkdir(parents=True, exist_ok=True)
    inputs = make_fixed_token_input(tokenizer, PROFILE_ISL, device=str(next(model.parameters()).device))
    trace_path = profile_root / "prefill_decode_trace.json"
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
            outputs = model(**inputs, use_cache=True)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            mask = torch.cat(
                [inputs["attention_mask"], torch.ones_like(next_token)], dim=1
            )
            model(
                input_ids=next_token,
                attention_mask=mask,
                past_key_values=outputs.past_key_values,
                use_cache=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        profiler.export_chrome_trace(str(trace_path))
        events = sorted(
            (
                {
                    "key": item.key,
                    "count": int(item.count),
                    "self_cuda_time_total_us": float(getattr(item, "self_cuda_time_total", 0.0)),
                    "cuda_time_total_us": float(getattr(item, "cuda_time_total", 0.0)),
                }
                for item in profiler.key_averages()
            ),
            key=lambda item: item["cuda_time_total_us"],
            reverse=True,
        )
        result = {
            "available": True,
            "trace_path": str(trace_path),
            "trace_sha256": _sha256_file(trace_path),
            "top_events": events[:80],
            "expected_kernel": "FP16 eager/cublas",
            "observed_execution": "torch eager FP16 after full-weight dequant"
            if method != "fp16" else "torch eager FP16",
            "native_low_bit_kernel": False,
            "packed_layout_id": None,
            "fallback_reason": (
                "portable fake-dequant control; native low-bit kernel deferred to E05-06"
                if method != "fp16" else None
            ),
        }
    except Exception as exc:  # profiler support varies by JetPack build
        result = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "expected_kernel": "FP16 eager/cublas",
            "native_low_bit_kernel": False,
        }
    _json_write(profile_root / "summary.json", result)
    return result


def _run_performance(
    args: argparse.Namespace,
    method: str,
    tokenizer: Any,
    model: torch.nn.Module,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workloads = _resolved_workloads()
    order = list(range(len(workloads)))
    random.Random(ORDER_SEED + args.run_index).shuffle(order)
    monitor: TegrastatsMonitor | None = None
    try:
        monitor = TegrastatsMonitor(interval_ms=MONITOR_INTERVAL_MS)
        monitor.start()
    except Exception as exc:
        LOG.warning("tegrastats unavailable: %s", exc)
        monitor = None
    records: list[dict[str, Any]] = []
    try:
        for workload_index in order:
            workload = workloads[workload_index]
            name = workload["name"]
            isl = int(workload["input_tokens"])
            osl = int(workload["output_tokens"])
            inputs = make_fixed_token_input(tokenizer, isl, device=str(next(model.parameters()).device))
            warmup_records: list[dict[str, Any]] = []
            for warmup_index in range(args.warmup):
                begin = time.perf_counter()
                warm = benchmark_model_core(model, inputs, osl)
                warmup_records.append(
                    {
                        "index": warmup_index,
                        "wall_s": time.perf_counter() - begin,
                        "sequence_hash": hash_token_sequence(warm["generated_token_ids"]),
                    }
                )
            begin_ns = time.monotonic_ns()
            samples: list[dict[str, Any]] = []
            anomalies: list[dict[str, Any]] = []
            for repetition in range(args.repetitions):
                try:
                    result = benchmark_model_core(model, inputs, osl)
                    samples.append(
                        {
                            "repetition": repetition,
                            "input_tokens": result["input_tokens"],
                            "output_tokens": result["output_tokens"],
                            "prefill_forward_ms": result["prefill_forward_ms"],
                            "first_token_selection_ms": result["first_token_selection_ms"],
                            "model_core_ttft_ms": result["model_core_ttft_ms"],
                            "decode_total_ms": result["decode_total_ms"],
                            "model_core_e2e_ms": result["model_core_e2e_ms"],
                            "raw_itl_ms": result["raw_itl_ms"],
                            "prefill_tokens_per_s": result["prefill_tokens_per_s"],
                            "decode_tokens_per_s": result["decode_tokens_per_s"],
                            "model_core_output_tokens_per_s": result[
                                "model_core_output_tokens_per_s"
                            ],
                            "peak_cuda_allocated_mb": result["peak_cuda_allocated_mb"],
                            "peak_cuda_reserved_mb": result["peak_cuda_reserved_mb"],
                            "process_rss_bytes": result["process_rss_bytes"],
                            "process_swap_bytes": result["process_swap_bytes"],
                            "kv_cache_total_bytes": result["kv_cache"].get("total_bytes"),
                            "generated_token_ids": result["generated_token_ids"],
                            "sequence_hash": hash_token_sequence(result["generated_token_ids"]),
                            "execution": {
                                "expected_kernel": "FP16 eager/cublas",
                                "observed_path": METHOD_SPECS[method]["execution"],
                                "provider": f"torch {torch.__version__}",
                                "packed_layout_id": None,
                                "capability": "native low-bit not requested",
                                "fallback": method != "fp16",
                                "fallback_reason": (
                                    "canonical artifact is fully dequantized before inference"
                                    if method != "fp16" else None
                                ),
                            },
                        }
                    )
                except Exception as exc:
                    anomalies.append(
                        {
                            "repetition": repetition,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
            end_ns = time.monotonic_ns()
            telemetry = (
                _telemetry_summary(monitor.records, begin_ns, end_ns, len(samples))
                if monitor is not None else {"energy_usable": False, "reason": "unavailable"}
            )
            records.append(
                {
                    **workload,
                    "batch_size": 1,
                    "warmup": warmup_records,
                    "repetitions": args.repetitions,
                    "samples": samples,
                    "summary": request_summary(samples) if samples else {},
                    "telemetry": telemetry,
                    "anomalies": anomalies,
                }
            )
            LOG.info("%s %s: %d/%d", method, name, len(samples), args.repetitions)
    finally:
        if monitor is not None:
            monitor.stop()
    telemetry_records = list(monitor.records) if monitor is not None else []
    return records, telemetry_records


def command_collect(args: argparse.Namespace) -> int:
    method = args.method
    output = Path(args.output_dir)
    spec = json.loads((output / "spec.json").read_text())
    tokenizer, model, load_memory = _load_model(args)
    artifact_record: dict[str, Any] | None = None
    if method != "fp16":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        rss_before = _process_rss()
        artifact_root = output / "artifacts" / method
        artifact_record = apply_model_quant_artifact(
            model,
            artifact_root,
            expected_source_model_hash=spec["model"]["model_manifest_sha256"],
            row_chunk=args.row_chunk,
        )
        artifact_record.update(
            {
                "rss_before": rss_before,
                "rss_after": _process_rss(),
                "cuda_after": _cuda_memory(),
                "disk": artifact_disk_usage(artifact_root),
            }
        )

    warm_inputs = make_fixed_token_input(tokenizer, 32, device=str(next(model.parameters()).device))
    first_request_begin = time.perf_counter()
    first_request = benchmark_model_core(model, warm_inputs, 2)
    first_request_wall_s = time.perf_counter() - first_request_begin
    steady_memory_before_workloads = {
        "rss": _process_rss(),
        "cuda": _cuda_memory(),
    }

    quality = None
    profiler = None
    if args.run_index == 0:
        quality = _quality_eval(method, tokenizer, model, output)
        profiler = _profile_execution(method, tokenizer, model, output)

    performance, telemetry = _run_performance(args, method, tokenizer, model)
    raw_dir = output / "runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = raw_dir / f"{method}_run_{args.run_index}.tegrastats.txt"
    with telemetry_path.open("w", encoding="utf-8") as handle:
        for record in telemetry:
            handle.write(f"{record['time_ns']}\t{record['raw']}\n")
    run = {
        "schema": "hqsb.e05_02.run/v1",
        "run_id": f"{method}_run_{args.run_index}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "method": method,
        "run_index": args.run_index,
        "captured_at_utc": _utc_now(),
        "spec_hash": spec["spec_hash"],
        "environment": _environment(),
        "load_memory": load_memory,
        "artifact_load": artifact_record,
        "first_request": {
            "wall_s": first_request_wall_s,
            "ttft_ms": first_request["model_core_ttft_ms"],
            "e2e_ms": first_request["model_core_e2e_ms"],
        },
        "steady_memory_before_workloads": steady_memory_before_workloads,
        "quality_file": str(output / "quality" / method / "quality.json")
        if quality is not None else None,
        "profiler_file": str(output / "profiler" / method / "summary.json")
        if profiler is not None else None,
        "performance": performance,
        "telemetry_file": str(telemetry_path),
        "execution_truth": METHOD_SPECS[method],
    }
    _json_write(raw_dir / f"{method}_run_{args.run_index}.json", run)
    return 0


def _cross_run_metric(records: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [record["summary"][metric]["p50"] for record in records]
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "runs": len(values),
    }


def _evidence_manifest(output: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "EVIDENCE_MANIFEST.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest = {
        "schema": "hqsb.evidence_manifest/v1",
        "generated_at_utc": _utc_now(),
        "files": files,
    }
    manifest["root_sha256"] = _sha256_bytes(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return manifest


def command_verify(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    spec = json.loads((output / "spec.json").read_text())
    runs: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        paths = sorted((output / "runs").glob(f"{method}_run_*.json"))
        runs[method] = [json.loads(path.read_text()) for path in paths]
    artifacts = json.loads((output / "artifact_summary.json").read_text())
    quality = {
        method: json.loads((output / "quality" / method / "quality.json").read_text())
        for method in METHODS
    }
    reference_quality = quality["fp16"]["summary"]
    quality_verdicts: dict[str, Any] = {"fp16": {"passed": True}}
    for method in ("rtn_w8", "rtn_w4"):
        observed = quality[method]["summary"]
        gates = QUALITY_GATES[method]
        checks = {
            "finite": observed["all_finite"],
            "ppl_ratio": observed["aggregate_ppl"] / reference_quality["aggregate_ppl"]
            <= gates["max_ppl_ratio"],
            "logit_cosine": observed["minimum_logit_cosine"]
            >= gates["min_mean_logit_cosine"],
            "top1_agreement": observed["minimum_top1_agreement"]
            >= gates["min_top1_agreement"],
            "task_non_inferiority": observed["task_accuracy"] - reference_quality["task_accuracy"]
            >= gates["min_task_accuracy_delta"],
            "generation_length": not quality[method]["generation"]["premature_stop"],
        }
        quality_verdicts[method] = {
            "passed": all(checks.values()),
            "checks": checks,
            "observed": observed,
            "thresholds": gates,
            "ppl_ratio": observed["aggregate_ppl"] / reference_quality["aggregate_ppl"],
            "task_accuracy_delta": observed["task_accuracy"] - reference_quality["task_accuracy"],
        }

    workload_names = [item["name"] for item in spec["workloads"]]
    performance_summary: dict[str, Any] = {}
    measurement_checks: dict[str, Any] = {}
    for method in METHODS:
        method_runs = runs[method]
        method_summary: dict[str, Any] = {}
        complete = len(method_runs) == DEFAULT_RUNS
        all_samples = True
        for name in workload_names:
            records = [
                next(item for item in run["performance"] if item["name"] == name)
                for run in method_runs
            ] if method_runs else []
            if len(records) != DEFAULT_RUNS:
                complete = False
                continue
            all_samples = all_samples and all(
                len(record["samples"]) == DEFAULT_REPETITIONS for record in records
            )
            method_summary[name] = {
                metric: _cross_run_metric(records, metric)
                for metric in (
                    "ttft_ms", "tpot_ms", "e2e_ms", "prefill_tokens_per_s",
                    "decode_tokens_per_s", "output_tokens_per_s",
                )
            }
            method_summary[name]["peak_allocated_mb"] = {
                "median": statistics.median(
                    max(sample["peak_cuda_allocated_mb"] for sample in record["samples"])
                    for record in records
                )
            }
            method_summary[name]["energy_j_per_request"] = {
                "median": statistics.median(
                    record["telemetry"]["j_per_request"]
                    for record in records
                    if record["telemetry"].get("j_per_request") is not None
                ) if any(
                    record["telemetry"].get("j_per_request") is not None for record in records
                ) else None,
                "all_windows_usable": all(
                    record["telemetry"].get("energy_usable", False) for record in records
                ),
            }
        performance_summary[method] = method_summary
        measurement_checks[method] = {
            "three_independent_processes": len(method_runs) == DEFAULT_RUNS,
            "six_workloads": complete and len(method_summary) == 6,
            "all_repetitions_present": all_samples,
            "fully_on_cuda": all(run["load_memory"]["fully_on_cuda"] for run in method_runs),
            "phase_separated": True,
            "execution_labeled": True,
        }

    comparisons: dict[str, Any] = {}
    for method in ("rtn_w8", "rtn_w4"):
        comparisons[method] = {}
        for name in workload_names:
            reference = performance_summary["fp16"][name]
            observed = performance_summary[method][name]
            comparisons[method][name] = {
                "ttft_speedup": reference["ttft_ms"]["median"] / observed["ttft_ms"]["median"],
                "tpot_speedup": reference["tpot_ms"]["median"] / observed["tpot_ms"]["median"],
                "e2e_speedup": reference["e2e_ms"]["median"] / observed["e2e_ms"]["median"],
                "peak_allocated_ratio": observed["peak_allocated_mb"]["median"]
                / reference["peak_allocated_mb"]["median"],
                "claim_scope": "fake-dequant diagnostic only; not a low-bit kernel speedup",
            }

    preconditions_pass = bool(spec["preconditions"]["s04_5_m4_verified"])
    artifacts_reloadable = all(
        len(runs[method]) == DEFAULT_RUNS
        and all(run["artifact_load"] is not None for run in runs[method])
        for method in ("rtn_w8", "rtn_w4")
    )
    quality_pass = all(quality_verdicts[method]["passed"] for method in METHODS)
    measurement_pass = all(
        all(checks.values()) for checks in measurement_checks.values()
    )
    execution_gate = False
    final_status = "BLOCKED"
    verdict = {
        "schema": "hqsb.e05_02.verdict/v1",
        "experiment_id": "E05-02",
        "generated_at_utc": _utc_now(),
        "spec_hash": spec["spec_hash"],
        "status": final_status,
        "expected_effect": "weight-only compression, quality, prefill/decode baseline",
        "gates": {
            "preconditions": preconditions_pass,
            "artifact_new_process_reload": artifacts_reloadable,
            "quality": quality_pass,
            "measurement": measurement_pass,
            "native_low_bit_execution": execution_gate,
        },
        "quality": quality_verdicts,
        "measurement": measurement_checks,
        "artifacts": artifacts,
        "performance": performance_summary,
        "comparisons_vs_fp16": comparisons,
        "single_item_standard_met": False,
        "single_item_standard_reason": (
            "Formal PASS is prohibited because S04.5 M4 evidence is absent. "
            "W8/W4 execution is a declared full-weight fake-dequant FP16 path, "
            "so native low-bit performance and runtime-memory gates are not met."
        ),
        "expected_effect_met": False,
        "expected_effect_scope_met": {
            "artifact_compression": artifacts_reloadable,
            "quality_baseline": quality_pass,
            "six_workload_diagnostic": measurement_pass,
            "native_low_bit_speed_or_memory": False,
        },
        "admission_to_e05_10": False,
    }
    _json_write(output / "summary.json", {
        "performance": performance_summary,
        "comparisons_vs_fp16": comparisons,
        "quality": quality_verdicts,
    })
    _json_write(output / "verdict.json", verdict)
    _json_write(output / "EVIDENCE_MANIFEST.json", _evidence_manifest(output))
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    spec = subparsers.add_parser("spec")
    spec.add_argument("--output-dir", required=True)
    spec.add_argument("--force", action="store_true")
    spec.set_defaults(func=command_spec)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--model-path", default=MODEL_PATH)
    prepare.add_argument("--model-manifest", default=str(MODEL_MANIFEST))
    prepare.add_argument("--row-chunk", type=int, default=32)
    prepare.add_argument("--cpu-staging", action="store_true")
    prepare.set_defaults(func=command_prepare)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--method", choices=METHODS, required=True)
    collect.add_argument("--run-index", type=int, required=True)
    collect.add_argument("--model-path", default=MODEL_PATH)
    collect.add_argument("--model-manifest", default=str(MODEL_MANIFEST))
    collect.add_argument("--row-chunk", type=int, default=32)
    collect.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    collect.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    collect.add_argument("--cpu-staging", action="store_true")
    collect.set_defaults(func=command_collect)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--output-dir", required=True)
    verify.set_defaults(func=command_verify)
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
