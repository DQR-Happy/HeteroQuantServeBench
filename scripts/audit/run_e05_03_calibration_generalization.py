#!/usr/bin/env python3
"""E05-03 calibration representativeness and generalization audit.

The measured method is deliberately narrow: static per-channel A8 fake
quantization is injected at six Qwen projection inputs.  It is a
calibration-dependent range probe, not a GPTQ/AWQ implementation and not a
low-bit kernel claim.  Its purpose is to make the data protocol, learning
curve, subset variance, frozen selection and one-shot final evaluation real
and auditable before E05-04 consumes the protocol.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from hqsb.models.loader import load_qwen3
from hqsb.quant import stats as qstats
from hqsb.quant.calibration import (
    DataSpec,
    SPLIT_CALIBRATION,
    SPLIT_FINAL_EVALUATION,
    SPLIT_POLICY_VALIDATION,
    SPLIT_STRESS,
    SampleRecord,
    SplitManifest,
    audit_leakage,
    draw_subset,
    normalized_text_hash,
)


ROOT = _SCRIPT_ROOT
MODEL_PATH = "~/models/hqsb/Qwen3-1.7B"
MODEL_MANIFEST = ROOT / "docs/benchmark/model_sha256_manifest.txt"
DEFAULT_OUTPUT = ROOT / "docs/stage_experiments/S05/E05-03/raw"
SAMPLE_COUNTS = (2, 4, 8, 16, 32)
SUBSET_SEEDS = (5300, 5301, 5302)
SHORT_TOKENS = 48
LONG_TOKENS = 160
STRESS_TOKENS = 256
CALIBRATION_PER_CELL = 48
EVALUATION_PER_CELL = 2
QMAX = 127.0
MODULE_SUFFIXES = tuple(
    f"layers.{layer}.{role}"
    for layer in (0, 14, 27)
    for role in ("self_attn.q_proj", "mlp.up_proj")
)

# Each parent file belongs to exactly one split.  That makes the parent-id
# leakage gate meaningful rather than relying only on disjoint token windows.
CORPUS_FILES: dict[str, dict[str, tuple[str, ...]]] = {
    SPLIT_CALIBRATION: {
        "technical": (
            "docs/architecture/HQSB_项目完整剖析.md",
            "docs/stages/S05_量化与低精度推理.md",
        ),
        "code": ("hqsb/quant/calibration.py", "hqsb/quant/stats.py"),
    },
    SPLIT_POLICY_VALIDATION: {
        "technical": (
            "docs/stage_experiments/details/S05/E05-01_rtn_quant_math_and_artifact.md",
            "docs/stage_experiments/details/S05/E05-02_weight_only_model_baseline.md",
        ),
        "code": ("hqsb/quant/activation.py", "hqsb/quant/quality.py"),
    },
    SPLIT_FINAL_EVALUATION: {
        "technical": (
            "docs/stage_experiments/details/S05/E05-04_industrial_method_adapters.md",
            "docs/stage_experiments/details/S05/E05-05_layer_sensitivity_mixed_precision.md",
        ),
        "code": (
            "hqsb/quant/adapters/base.py",
            "hqsb/quant/model_weight_only.py",
        ),
    },
    SPLIT_STRESS: {
        "technical": (
            "docs/reports/扩展宿主OOM故障复盘.md",
            "docs/stage_experiments/details/S14/E14-F3_long_context_kv_capacity_quality_latency.md",
        ),
        "code": (
            "tests/property/test_quant_properties.py",
            "hqsb/quant/interface_map.py",
        ),
    },
}

QUALITY_GATES = {
    "max_mean_delta_nll": 0.020,
    "max_delta_nll_ci_high": 0.025,
    "max_mean_kl": 0.005,
    "min_top1_agreement": 0.990,
    "max_saturation_rate": 0.001,
}

SELECTION_RULE = {
    "stabilization_epsilon_delta_nll": 0.002,
    "max_seed_range_delta_nll": 0.005,
    "min_scale_cosine_to_next_budget": 0.990,
    "max_policy_slice_delta_nll": 0.020,
    "offline_cost_ceiling_s": 600.0,
    "winner_order": ["sample_count", "valid_tokens_mean", "source", "length_coverage"],
    "selected_seed": SUBSET_SEEDS[0],
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    temporary.replace(path)


def _jsonl_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def _environment() -> dict[str, Any]:
    return {
        "captured_at_utc": _utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--short")),
        "model_manifest_sha256": _sha256_file(MODEL_MANIFEST),
        "pyarrow_available": False,
        "table_format": "JSONL (pyarrow is unavailable on the target)",
    }


def _load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        os.path.expanduser(model_path), local_files_only=True, trust_remote_code=True
    )


@dataclass
class Corpus:
    manifests: dict[str, SplitManifest]
    records: dict[str, SampleRecord]
    texts: dict[str, str]


def _file_revision(relative: str) -> str:
    path = ROOT / relative
    return f"git:{_git('rev-parse', 'HEAD')};sha256:{_sha256_file(path)}"


def _windows(
    token_ids: Sequence[int], *, length: int, count: int, from_tail: bool
) -> list[tuple[int, list[int]]]:
    if len(token_ids) < length:
        return []
    stride = length + 17
    offsets = list(range(0, len(token_ids) - length + 1, stride))
    if from_tail:
        offsets = list(reversed(offsets))
    return [(offset, list(token_ids[offset : offset + length])) for offset in offsets[:count]]


def _build_corpus(tokenizer) -> Corpus:
    manifests = {
        split: SplitManifest(
            split=split,
            dataset_revision=_git("rev-parse", "HEAD"),
            tokenizer_revision=_sha256_file(ROOT / "docs/benchmark/model_sha256_manifest.txt"),
            preprocess_version="hqsb-token-window-v1",
        )
        for split in CORPUS_FILES
    }
    records: dict[str, SampleRecord] = {}
    texts: dict[str, str] = {}
    for split, domains in CORPUS_FILES.items():
        for domain, files in domains.items():
            target = CALIBRATION_PER_CELL if split == SPLIT_CALIBRATION else EVALUATION_PER_CELL
            lengths = (("long", STRESS_TOKENS),) if split == SPLIT_STRESS else (
                ("short", SHORT_TOKENS),
                ("long", LONG_TOKENS),
            )
            for bucket, length in lengths:
                made = 0
                for relative in files:
                    raw = (ROOT / relative).read_text(errors="replace")
                    tokens = tokenizer(raw, add_special_tokens=False)["input_ids"]
                    available = target - made
                    for offset, chunk in _windows(
                        tokens,
                        length=length,
                        count=available,
                        from_tail=bucket == "long",
                    ):
                        sample_id = (
                            f"{split}:{domain}:{Path(relative).name}:"
                            f"{bucket}:{offset}:{made}"
                        )
                        decoded = tokenizer.decode(chunk, skip_special_tokens=False)
                        record = SampleRecord(
                            dataset=f"hqsb-repository-{domain}",
                            revision=_file_revision(relative),
                            split=split,
                            sample_id=sample_id,
                            text_hash=normalized_text_hash(decoded),
                            token_ids=chunk,
                            license="repository-local; no external redistribution claim",
                            language="zh-en-mixed" if domain == "technical" else "python-en",
                            domain=domain,
                            length_bucket=bucket,
                            task_type="next-token-language-modeling",
                            template="",
                            preprocess_version="hqsb-token-window-v1",
                            tokenizer_revision=manifests[split].tokenizer_revision,
                            truncated=False,
                            parent_id=relative,
                        )
                        manifests[split].add(record)
                        records[sample_id] = record
                        texts[sample_id] = decoded
                        made += 1
                        if made >= target:
                            break
                    if made >= target:
                        break
                if made != target:
                    raise RuntimeError(
                        f"could only create {made}/{target} {split}/{domain}/{bucket} samples"
                    )
    return Corpus(manifests=manifests, records=records, texts=texts)


def _data_spec() -> DataSpec:
    sources = []
    for split, domains in CORPUS_FILES.items():
        for domain, files in domains.items():
            for relative in files:
                sources.append(
                    {
                        "dataset": f"hqsb-repository-{domain}",
                        "split": split,
                        "path": relative,
                        "revision": _file_revision(relative),
                        "license": "repository-local; no external redistribution claim",
                    }
                )
    return DataSpec(
        name="E05-03 Qwen3 selected-module static-A8 range probe",
        sources=sources,
        length_buckets={"short": (SHORT_TOKENS, SHORT_TOKENS + 1), "long": (LONG_TOKENS, STRESS_TOKENS + 1)},
        sample_counts=SAMPLE_COUNTS,
        token_budgets=(0,),
        subset_seeds=SUBSET_SEEDS,
        ngram=13,
        near_duplicate_threshold=0.85,
        primary_quality_metric="paired_delta_nll",
        primary_direction=qstats.LOWER_IS_BETTER,
        quality_margin=SELECTION_RULE["max_policy_slice_delta_nll"],
        stabilization_epsilon=SELECTION_RULE["stabilization_epsilon_delta_nll"],
        seed_variance_threshold=SELECTION_RULE["max_seed_range_delta_nll"],
        offline_cost_ceiling_s=SELECTION_RULE["offline_cost_ceiling_s"],
        notes=(
            "Fixed-sample-count primary sweep; valid token count is recorded for "
            "every subset. Method is selected-module static A8 fake quant only."
        ),
    )


def _spec_payload() -> dict[str, Any]:
    data_spec = _data_spec()
    e05_02_verdict = ROOT / "docs/stage_experiments/S05/E05-02/raw/verdict.json"
    prior = json.loads(e05_02_verdict.read_text()) if e05_02_verdict.is_file() else {}
    prior_status = prior.get("overall", prior.get("status"))
    payload = {
        "schema": "hqsb.e05_03.experiment_spec/v1",
        "experiment_id": "E05-03",
        "frozen_at_utc": _utc_now(),
        "scope": (
            "exploratory calibration protocol: E05-02 is not PASS and the measured "
            "method is a selected-module static-A8 fake-quant range probe"
        ),
        "model": {
            "id": "Qwen/Qwen3-1.7B",
            "local_path": MODEL_PATH,
            "dtype": "float16",
            "attention_backend": "eager",
            "model_manifest_sha256": _sha256_file(MODEL_MANIFEST),
        },
        "data_spec": json.loads(data_spec.to_json()),
        "data_spec_hash": data_spec.data_spec_hash,
        "method": {
            "id": "selected_module_static_a8_fake_quant",
            "bits": 8,
            "qrange": [-127, 127],
            "granularity": "per-input-channel",
            "scale_statistic": "calibration subset channel absmax / 127",
            "modules": list(MODULE_SUFFIXES),
            "real_low_bit_kernel": False,
            "industrial_method_claim": False,
            "purpose": "calibration sensitivity and data-protocol validation",
        },
        "factors": {
            "source": ["technical", "code"],
            "sample_count": list(SAMPLE_COUNTS),
            "length_coverage": ["short-only", "balanced"],
            "subset_seed": list(SUBSET_SEEDS),
            "primary_budget_control": "sample_count",
            "secondary_budget_observation": "valid_tokens",
        },
        "quality_gates": QUALITY_GATES,
        "selection_rule": SELECTION_RULE,
        "final_policy": "final-evaluation is inaccessible until policy selection is frozen",
        "stress_policy": "descriptive only; never feeds selection",
        "preconditions": {
            "e05_02_verdict_path": str(e05_02_verdict.relative_to(ROOT)),
            "e05_02_overall": prior_status,
            "formal_pass_allowed": prior_status == "PASS",
        },
        "known_limits": [
            "repository-local corpus is not a public benchmark dataset",
            "no GPTQ Hessian or AWQ saliency is measured in this range-only probe",
            "fake quant isolates quality and cannot support latency/kernel claims",
            "JSONL replaces Parquet because pyarrow is absent on the target",
        ],
        "protocol_amendment": {
            "version": 3,
            "reason": (
                "policy-validation-only sequential extension: neither the 2/4/8 "
                "grid nor the appended 16-sample budget established a stable "
                "family because scale cosine remained below 0.99"
            ),
            "change": "append sample_count=32 and enlarge only the calibration pool",
            "unchanged": ["all thresholds", "all existing subsets", "policy set", "final set", "stress set"],
            "final_evaluation_accessed_before_amendment": False,
        },
    }
    hashable = dict(payload)
    hashable.pop("frozen_at_utc")
    payload["spec_hash"] = _sha256_bytes(
        json.dumps(hashable, ensure_ascii=False, sort_keys=True).encode()
    )
    return payload


def _balanced_subset(
    pool: Sequence[SampleRecord], *, sample_count: int, seed: int
) -> dict[str, Any]:
    short = [sample for sample in pool if sample.length_bucket == "short"]
    long = [sample for sample in pool if sample.length_bucket == "long"]
    each = sample_count // 2
    left = draw_subset(short, sample_count=each, seed=seed)
    right = draw_subset(long, sample_count=sample_count - each, seed=seed + 10000)
    ids = left["sample_ids"] + right["sample_ids"]
    hashes = left["token_hashes"] + right["token_hashes"]
    return {
        "subset_seed": seed,
        "requested_sample_count": sample_count,
        "sample_count": len(ids),
        "valid_tokens": left["valid_tokens"] + right["valid_tokens"],
        "token_budget": None,
        "sample_ids": ids,
        "token_hashes": hashes,
        "length_buckets": {
            "short": left["sample_count"],
            "long": right["sample_count"],
        },
        "subset_hash": _sha256_bytes(
            json.dumps(sorted(zip(ids, hashes)), separators=(",", ":")).encode()
        ),
    }


def _candidate_rows(
    corpus: Corpus, existing_rows: Sequence[Mapping[str, Any]] = ()
) -> list[dict[str, Any]]:
    calibration = corpus.manifests[SPLIT_CALIBRATION].samples
    existing = {
        (row["source"], row["length_coverage"], row["sample_count"], row["subset_seed"]): dict(row)
        for row in existing_rows
        if row.get("sample_count") in SAMPLE_COUNTS
    }
    rows: list[dict[str, Any]] = []
    for source in ("technical", "code"):
        source_pool = [sample for sample in calibration if sample.domain == source]
        for coverage in ("short-only", "balanced"):
            for count in SAMPLE_COUNTS:
                for seed in SUBSET_SEEDS:
                    key = (source, coverage, count, seed)
                    if key in existing:
                        preserved = existing[key]
                        for sample_id, token_hash in zip(
                            preserved["sample_ids"], preserved["token_hashes"]
                        ):
                            if sample_id not in corpus.records or corpus.records[sample_id].token_hash != token_hash:
                                raise RuntimeError(f"cannot preserve amended subset sample {sample_id}")
                        rows.append(preserved)
                        continue
                    if coverage == "short-only":
                        subset = draw_subset(
                            [s for s in source_pool if s.length_bucket == "short"],
                            sample_count=count,
                            seed=seed,
                        )
                    else:
                        subset = _balanced_subset(source_pool, sample_count=count, seed=seed)
                    subset.update(
                        {
                            "candidate_id": f"{source}__{coverage}__n{count}__seed{seed}",
                            "source": source,
                            "length_coverage": coverage,
                        }
                    )
                    rows.append(subset)
    return rows


def command_spec_data(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    if (output / "quality/final.json").exists():
        raise RuntimeError("refusing to amend data protocol after final-evaluation was consumed")
    spec_path = output / "spec.json"
    proposed = _spec_payload()
    if spec_path.exists() and not args.force:
        existing = json.loads(spec_path.read_text())
        if existing.get("spec_hash") != proposed.get("spec_hash"):
            raise RuntimeError("frozen spec differs; use a new output directory")
    else:
        _json_write(spec_path, proposed)
    tokenizer = _load_tokenizer(args.model_path)
    corpus = _build_corpus(tokenizer)
    data_dir = output / "data"
    names = {
        SPLIT_CALIBRATION: "calibration_manifest.jsonl",
        SPLIT_POLICY_VALIDATION: "policy_validation_manifest.jsonl",
        SPLIT_FINAL_EVALUATION: "final_evaluation_manifest.jsonl",
        SPLIT_STRESS: "stress_manifest.jsonl",
    }
    for split, filename in names.items():
        _jsonl_write(
            data_dir / filename,
            (sample.as_dict(include_tokens=False) for sample in corpus.manifests[split].samples),
        )
    report = audit_leakage(
        list(corpus.manifests.values()),
        ngram=13,
        near_duplicate_threshold=0.85,
        benchmark_answer_strings=("北京", "391", "Mars"),
        text_lookup=lambda sample: corpus.texts[sample.sample_id],
    )
    _json_write(data_dir / "leakage_report.json", report)
    distribution = {
        "splits": {
            split: manifest.summary() for split, manifest in corpus.manifests.items()
        },
        "sample_token_distribution": {
            split: qstats.summarize_distribution(
                [sample.num_tokens for sample in manifest.samples]
            ).as_dict()
            for split, manifest in corpus.manifests.items()
        },
        "corpus_files": CORPUS_FILES,
    }
    _json_write(data_dir / "distribution.json", distribution)
    old_subsets = _jsonl_read(data_dir / "subsets.jsonl")
    _jsonl_write(data_dir / "subsets.jsonl", _candidate_rows(corpus, old_subsets))
    _json_write(output / "environment.json", _environment())
    print(json.dumps({"spec": proposed["spec_hash"], "leakage": report}, indent=2))
    return 0


def _load_model(args: argparse.Namespace):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    before_rss = _rss_bytes()
    started = time.perf_counter()
    tokenizer, model, loader_time = load_qwen3(
        args.model_path,
        dtype=torch.float16,
        attention_backend="eager",
        verify_manifest=args.model_manifest,
        allow_extra=("model_sha256_manifest.txt",),
        cpu_staging=True,
    )
    return tokenizer, model, {
        "phase": "model_load",
        "wall_time_s": time.perf_counter() - started,
        "loader_reported_s": loader_time,
        "host_rss_before_bytes": before_rss,
        "host_rss_after_bytes": _rss_bytes(),
        "device_peak_bytes": torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else 0,
        "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
    }


def _resolve_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    modules = dict(model.named_modules())
    resolved: dict[str, torch.nn.Module] = {}
    for suffix in MODULE_SUFFIXES:
        matches = [(name, module) for name, module in modules.items() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one module ending in {suffix!r}, got {len(matches)}")
        resolved[matches[0][0]] = matches[0][1]
    return resolved


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    flat = tensor.detach().float().reshape(-1)
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    nan_count = int(torch.isnan(flat).sum().item())
    inf_count = int(torch.isinf(flat).sum().item())
    if not finite_values.numel():
        raise RuntimeError("activation tensor has no finite values")
    absolute = finite_values.abs()
    mean = float(finite_values.mean().item())
    std = float(finite_values.std(unbiased=False).item())
    threshold = abs(mean) + 6.0 * std
    quantiles = torch.quantile(
        absolute, torch.tensor([0.5, 0.9, 0.99, 0.999], device=absolute.device)
    )
    return {
        "count": int(finite_values.numel()),
        "mean": mean,
        "std": std,
        "rms": float(torch.sqrt(torch.mean(finite_values.square())).item()),
        "min": float(finite_values.min().item()),
        "max": float(finite_values.max().item()),
        "absmax": float(absolute.max().item()),
        "p50_abs": float(quantiles[0].item()),
        "p90_abs": float(quantiles[1].item()),
        "p99_abs": float(quantiles[2].item()),
        "p999_abs": float(quantiles[3].item()),
        "outlier_threshold": threshold,
        "outlier_count": int((absolute > threshold).sum().item()),
        "outlier_rate": float((absolute > threshold).float().mean().item()),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "percentile_basis": "absolute activation values",
    }


def _sample_stat_path(output: Path, sample_id: str) -> Path:
    return output / "stats/activation/per_sample" / f"{_sha256_bytes(sample_id.encode())}.npz"


@torch.inference_mode()
def _collect_one_sample(
    model: torch.nn.Module,
    modules: Mapping[str, torch.nn.Module],
    sample: SampleRecord,
    output: Path,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    input_ids = torch.tensor([list(sample.token_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    if int(attention_mask.sum().item()) != sample.num_tokens:
        raise RuntimeError("explicit mask accounting failed")
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in modules.items():
        def _hook(_module, args, key=name):
            captured[key] = args[0].detach()

        handles.append(module.register_forward_pre_hook(_hook))
    started = time.perf_counter()
    base = getattr(model, "model", model)
    base(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_time = time.perf_counter() - started
    for handle in handles:
        handle.remove()
    if set(captured) != set(modules):
        raise RuntimeError(f"capture mismatch: {sorted(captured)} vs {sorted(modules)}")
    arrays: dict[str, np.ndarray] = {}
    module_rows: list[dict[str, Any]] = []
    module_map: dict[str, str] = {}
    for index, (name, tensor) in enumerate(sorted(captured.items())):
        key = f"module_{index}"
        per_channel = tensor.detach().float().abs().amax(dim=(0, 1)).cpu().numpy().astype(np.float32)
        arrays[key] = per_channel
        module_map[key] = name
        row = {
            "sample_id": sample.sample_id,
            "module": name,
            "layer_index": int(name.split(".layers.", 1)[1].split(".", 1)[0]),
            "direction": "input",
            "shape": list(tensor.shape),
            "valid_tokens": sample.num_tokens,
            "valid_count": tensor.numel(),
            "padded_tokens": 0,
            "dtype": str(tensor.dtype),
            "merge_algorithm": "per-sample exact; candidate channel absmax uses elementwise max",
            "numeric_dtype": "float32 statistics from float16 activation",
        }
        row.update(_tensor_summary(tensor))
        module_rows.append(row)
    path = _sample_stat_path(output, sample.sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "sample_id": sample.sample_id,
        "token_hash": sample.token_hash,
        "sample_tokens": sample.num_tokens,
        "domain": sample.domain,
        "length_bucket": sample.length_bucket,
        "array_file": str(path.relative_to(output)),
        "array_sha256": _sha256_file(path),
        "module_map": module_map,
        "modules": module_rows,
        "wall_time_s": wall_time,
        "host_rss_bytes": _rss_bytes(),
        "device_peak_bytes": torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else 0,
    }


def command_collect_stats(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    if not (output / "spec.json").is_file():
        raise RuntimeError("run spec-data first")
    tokenizer, model, load_cost = _load_model(args)
    corpus = _build_corpus(tokenizer)
    modules = _resolve_modules(model)
    index_path = output / "stats/activation/index.jsonl"
    existing = {row["sample_id"]: row for row in _jsonl_read(index_path)}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sample in corpus.manifests[SPLIT_CALIBRATION].samples:
        previous = existing.get(sample.sample_id)
        if previous and (output / previous["array_file"]).is_file():
            rows.append(previous)
            continue
        try:
            row = _collect_one_sample(model, modules, sample, output)
            rows.append(row)
            _jsonl_write(index_path, rows)
            print(f"collected {sample.sample_id} ({len(rows)})", flush=True)
        except Exception as exc:
            failures.append({"sample_id": sample.sample_id, "error": repr(exc)})
            _json_write(output / "stats/activation/failures.json", failures)
            raise
    _jsonl_write(index_path, rows)
    _json_write(
        output / "stats/hessian/not_applicable.json",
        {"applicable": False, "reason": "range-only static A8 probe; no Hessian method executed"},
    )
    _json_write(
        output / "stats/saliency/not_applicable.json",
        {"applicable": False, "reason": "range-only static A8 probe; no AWQ saliency executed"},
    )
    cost = {
        "model_load": load_cost,
        "stat_collection": {
            "phase": "stat_collection",
            "samples": len(rows),
            "wall_time_s": sum(row["wall_time_s"] for row in rows),
            "host_peak_bytes": max(row["host_rss_bytes"] for row in rows),
            "device_peak_bytes": max(row["device_peak_bytes"] for row in rows),
            "failures": failures,
        },
    }
    _json_write(output / "cost/stat_collection.json", cost)
    print(json.dumps({"samples": len(rows), "modules": list(modules)}, indent=2))
    return 0


def _load_sample_channels(output: Path) -> dict[str, dict[str, np.ndarray]]:
    values: dict[str, dict[str, np.ndarray]] = {}
    for row in _jsonl_read(output / "stats/activation/index.jsonl"):
        path = output / row["array_file"]
        if _sha256_file(path) != row["array_sha256"]:
            raise RuntimeError(f"activation statistics hash mismatch: {path}")
        archive = np.load(path, allow_pickle=False)
        values[row["sample_id"]] = {
            module_name: np.asarray(archive[key], dtype=np.float32)
            for key, module_name in row["module_map"].items()
        }
    return values


def _logical_artifact_hash(
    metadata: Mapping[str, Any], scales: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")).encode())
    for module, values in sorted(scales.items()):
        digest.update(module.encode())
        digest.update(np.asarray(values, dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def _build_candidate_artifact(
    subset: Mapping[str, Any], sample_channels: Mapping[str, Mapping[str, np.ndarray]]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    chosen = [sample_channels[sample_id] for sample_id in subset["sample_ids"]]
    modules = sorted(chosen[0])
    if any(sorted(sample) != modules for sample in chosen):
        raise RuntimeError("module set changed between calibration samples")
    scales = {
        module: np.maximum.reduce([sample[module] for sample in chosen]).astype(np.float32)
        / np.float32(QMAX)
        for module in modules
    }
    for values in scales.values():
        values[values <= 0.0] = np.float32(1e-8)
    identity = {
        "schema": "hqsb.e05_03.static_a8_scale_artifact/v1",
        "method": "selected_module_static_a8_fake_quant",
        "subset_hash": subset["subset_hash"],
        "sample_ids": list(subset["sample_ids"]),
        "token_hashes": list(subset["token_hashes"]),
        "source": subset["source"],
        "length_coverage": subset["length_coverage"],
        "sample_count": subset["sample_count"],
        "valid_tokens": subset["valid_tokens"],
        "subset_seed": subset["subset_seed"],
        "qrange": [-127, 127],
        "scale_rule": "per-channel calibration absmax / 127",
        "modules": modules,
    }
    artifact_hash = _logical_artifact_hash(identity, scales)
    record = {
        **identity,
        "candidate_id": subset["candidate_id"],
        "artifact_hash": artifact_hash,
        "scale_summaries": {
            module: qstats.summarize_distribution(values.tolist()).as_dict()
            for module, values in scales.items()
        },
        "offline_cost_s": time.perf_counter() - started,
        "replay": {
            "logical_hash_algorithm": "sha256(metadata canonical JSON + sorted module float32 LE bytes)",
            "expected_hash": artifact_hash,
        },
    }
    return record, scales


def _save_artifact(output: Path, record: dict[str, Any], scales: Mapping[str, np.ndarray]) -> None:
    artifact_dir = output / "artifacts" / record["candidate_id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    keys: dict[str, np.ndarray] = {}
    module_map: dict[str, str] = {}
    for index, (module, values) in enumerate(sorted(scales.items())):
        key = f"module_{index}"
        keys[key] = np.asarray(values, dtype=np.float32)
        module_map[key] = module
    scale_path = artifact_dir / "scales.npz"
    np.savez_compressed(scale_path, **keys)
    record = dict(record)
    record["module_map"] = module_map
    record["scale_file"] = str(scale_path.relative_to(output))
    record["scale_file_sha256"] = _sha256_file(scale_path)
    _json_write(artifact_dir / "manifest.json", record)


def _load_artifact(output: Path, candidate_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = output / "artifacts" / candidate_id / "manifest.json"
    record = json.loads(manifest_path.read_text())
    scale_path = output / record["scale_file"]
    if _sha256_file(scale_path) != record["scale_file_sha256"]:
        raise RuntimeError(f"scale file hash mismatch: {scale_path}")
    archive = np.load(scale_path, allow_pickle=False)
    scales = {
        module: np.asarray(archive[key], dtype=np.float32)
        for key, module in record["module_map"].items()
    }
    identity_keys = (
        "schema", "method", "subset_hash", "sample_ids", "token_hashes", "source",
        "length_coverage", "sample_count", "valid_tokens", "subset_seed", "qrange",
        "scale_rule", "modules",
    )
    identity = {key: record[key] for key in identity_keys}
    observed = _logical_artifact_hash(identity, scales)
    if observed != record["artifact_hash"]:
        raise RuntimeError(f"logical artifact hash mismatch for {candidate_id}")
    return record, scales


def _cosine_arrays(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float64)
    b = right.astype(np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else float("nan")


def _scale_stability(output: Path, subsets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["source"], row["length_coverage"], row["sample_count"], row["subset_seed"]): row
        for row in subsets
    }
    rows: list[dict[str, Any]] = []
    for subset in subsets:
        next_counts = [count for count in SAMPLE_COUNTS if count > subset["sample_count"]]
        if not next_counts:
            continue
        next_count = min(next_counts)
        other = by_key[(subset["source"], subset["length_coverage"], next_count, subset["subset_seed"])]
        _, left = _load_artifact(output, subset["candidate_id"])
        _, right = _load_artifact(output, other["candidate_id"])
        module_rows = []
        for module in sorted(left):
            denom = float(np.linalg.norm(right[module].astype(np.float64)))
            module_rows.append(
                {
                    "module": module,
                    "cosine": _cosine_arrays(left[module], right[module]),
                    "relative_l2": float(
                        np.linalg.norm((left[module] - right[module]).astype(np.float64)) / denom
                    ) if denom else float("nan"),
                }
            )
        rows.append(
            {
                "candidate_id": subset["candidate_id"],
                "next_candidate_id": other["candidate_id"],
                "module_metrics": module_rows,
                "minimum_cosine": min(row["cosine"] for row in module_rows),
                "maximum_relative_l2": max(row["relative_l2"] for row in module_rows),
            }
        )
    return rows


def command_build_artifacts(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    subsets = _jsonl_read(output / "data/subsets.jsonl")
    if not subsets:
        raise RuntimeError("run spec-data first")
    sample_channels = _load_sample_channels(output)
    replay_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for subset in subsets:
        record, scales = _build_candidate_artifact(subset, sample_channels)
        replay_record, replay_scales = _build_candidate_artifact(subset, sample_channels)
        replay_ok = record["artifact_hash"] == replay_record["artifact_hash"] and all(
            np.array_equal(scales[module], replay_scales[module]) for module in scales
        )
        if not replay_ok:
            raise RuntimeError(f"artifact replay mismatch: {subset['candidate_id']}")
        _save_artifact(output, record, scales)
        replay_rows.append(
            {
                "candidate_id": subset["candidate_id"],
                "artifact_hash": record["artifact_hash"],
                "bitwise_replay": replay_ok,
            }
        )
    stability = _scale_stability(output, subsets)
    _jsonl_write(output / "stats/activation/scale_stability.jsonl", stability)
    _jsonl_write(output / "artifacts/replay.jsonl", replay_rows)
    prior = json.loads((ROOT / "docs/stage_experiments/S05/E05-02/raw/artifact_summary.json").read_text())
    rtn_hash = prior["methods"]["rtn_w8"]["artifact_id"]
    _json_write(
        output / "artifacts/rtn_control.json",
        {
            "control": "RTN-W8 from E05-02; calibration-free",
            "artifact_id_by_candidate": {
                row["candidate_id"]: rtn_hash for row in subsets
            },
            "unique_artifact_ids": [rtn_hash],
            "invariant": True,
        },
    )
    _json_write(
        output / "cost/artifact_generation.json",
        {
            "phase": "scale_artifact_generation_and_replay",
            "candidates": len(subsets),
            "wall_time_s": time.perf_counter() - started,
            "host_peak_bytes": _rss_bytes(),
            "device_peak_bytes": 0,
        },
    )
    print(json.dumps({"artifacts": len(subsets), "replay_all": all(r["bitwise_replay"] for r in replay_rows)}, indent=2))
    return 0


class StaticA8Injector:
    def __init__(self, modules: Mapping[str, torch.nn.Module], scales: Mapping[str, np.ndarray]):
        self.modules = modules
        self.scale_cpu = scales
        self.scale_device: dict[str, torch.Tensor] = {}
        self.handles = []
        self.metrics: dict[str, dict[str, float]] = {}

    def __enter__(self):
        for name, module in self.modules.items():
            scale = torch.from_numpy(self.scale_cpu[name]).float().to(next(module.parameters()).device)
            self.scale_device[name] = scale.reshape(1, 1, -1)
            self.metrics[name] = {
                "values": 0.0,
                "saturated": 0.0,
                "squared_error": 0.0,
                "reference_squared": 0.0,
                "dot": 0.0,
                "reference_norm2": 0.0,
                "candidate_norm2": 0.0,
            }

            def _hook(_module, args, key=name):
                x = args[0]
                x32 = x.float()
                resolved_scale = self.scale_device[key]
                codes = torch.round(x32 / resolved_scale)
                clipped = torch.clamp(codes, -QMAX, QMAX)
                dequant = clipped * resolved_scale
                metric = self.metrics[key]
                metric["values"] += float(x32.numel())
                metric["saturated"] += float((x32.abs() > resolved_scale * QMAX).sum().item())
                metric["squared_error"] += float((dequant - x32).square().sum().item())
                metric["reference_squared"] += float(x32.square().sum().item())
                metric["dot"] += float((x32 * dequant).sum().item())
                metric["reference_norm2"] += float(x32.square().sum().item())
                metric["candidate_norm2"] += float(dequant.square().sum().item())
                return (dequant.to(dtype=x.dtype), *args[1:])

            self.handles.append(module.register_forward_pre_hook(_hook))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, metric in self.metrics.items():
            reference = metric["reference_squared"]
            denom = math.sqrt(metric["reference_norm2"] * metric["candidate_norm2"])
            result[name] = {
                "values": int(metric["values"]),
                "saturated": int(metric["saturated"]),
                "saturation_rate": metric["saturated"] / metric["values"] if metric["values"] else float("nan"),
                "nrmse": math.sqrt(metric["squared_error"] / reference) if reference else float("nan"),
                "cosine": metric["dot"] / denom if denom else float("nan"),
            }
        return result


def _paired_logit_metrics(reference: torch.Tensor, candidate: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    reference = reference[:, :-1, :].float()
    candidate = candidate[:, :-1, :].float()
    labels = labels[:, 1:]
    ref_nll = F.cross_entropy(reference.reshape(-1, reference.shape[-1]), labels.reshape(-1))
    cand_nll = F.cross_entropy(candidate.reshape(-1, candidate.shape[-1]), labels.reshape(-1))
    ref_logp = F.log_softmax(reference, dim=-1)
    cand_logp = F.log_softmax(candidate, dim=-1)
    kl = F.kl_div(cand_logp, ref_logp, log_target=True, reduction="none").sum(-1).mean()
    cosine = F.cosine_similarity(reference, candidate, dim=-1).mean()
    ref_top1 = reference.argmax(dim=-1)
    cand_top1 = candidate.argmax(dim=-1)
    top1 = (ref_top1 == cand_top1).float().mean()
    ref_top5 = reference.topk(5, dim=-1).indices
    cand_top5 = candidate.topk(5, dim=-1).indices
    overlap = (
        (ref_top5.unsqueeze(-1) == cand_top5.unsqueeze(-2))
        .any(dim=-1)
        .float()
        .sum(dim=-1)
        .div(5.0)
        .mean()
    )
    return {
        "reference_nll": float(ref_nll.item()),
        "candidate_nll": float(cand_nll.item()),
        "delta_nll": float((cand_nll - ref_nll).item()),
        "reference_ppl": float(torch.exp(ref_nll).item()),
        "candidate_ppl": float(torch.exp(cand_nll).item()),
        "kl_reference_to_candidate": float(kl.item()),
        "mean_logit_cosine": float(cosine.item()),
        "top1_agreement": float(top1.item()),
        "top5_overlap": float(overlap.item()),
        "positions": int(labels.numel()),
        "finite": bool(torch.isfinite(candidate).all().item()),
    }


@torch.inference_mode()
def _evaluate_artifact(
    model: torch.nn.Module,
    modules: Mapping[str, torch.nn.Module],
    scales: Mapping[str, np.ndarray],
    samples: Sequence[SampleRecord],
    *,
    candidate_id: str,
    split: str,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    rows: list[dict[str, Any]] = []
    local_accumulator: dict[str, list[dict[str, Any]]] = {name: [] for name in modules}
    started = time.perf_counter()
    for sample in samples:
        input_ids = torch.tensor([list(sample.token_ids)], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with StaticA8Injector(modules, scales) as injector:
            candidate = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
        reference = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        metrics = _paired_logit_metrics(reference, candidate, input_ids)
        local = injector.summary()
        for name, values in local.items():
            local_accumulator[name].append(values)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "token_hash": sample.token_hash,
                "domain": sample.domain,
                "language": sample.language,
                "length_bucket": sample.length_bucket,
                "tokens": sample.num_tokens,
                **metrics,
                "module_local": local,
            }
        )
        del candidate, reference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    deltas = [row["delta_nll"] for row in rows]
    ci = qstats.paired_bootstrap_ci(deltas, resamples=1000, seed=5303).as_dict()
    slice_delta: dict[str, float] = {}
    for field in ("domain", "language", "length_bucket"):
        values = sorted({row[field] for row in rows})
        for value in values:
            selected = [row["delta_nll"] for row in rows if row[field] == value]
            slice_delta[f"{field}:{value}"] = statistics.fmean(selected)
    local_summary = {
        name: {
            "mean_nrmse": statistics.fmean(row["nrmse"] for row in values),
            "max_nrmse": max(row["nrmse"] for row in values),
            "mean_cosine": statistics.fmean(row["cosine"] for row in values),
            "saturation_rate": sum(row["saturated"] for row in values)
            / sum(row["values"] for row in values),
        }
        for name, values in local_accumulator.items()
    }
    summary = {
        "candidate_id": candidate_id,
        "split": split,
        "samples": len(rows),
        "valid_next_tokens": sum(row["positions"] for row in rows),
        "mean_reference_nll": statistics.fmean(row["reference_nll"] for row in rows),
        "mean_candidate_nll": statistics.fmean(row["candidate_nll"] for row in rows),
        "mean_delta_nll": statistics.fmean(deltas),
        "delta_nll_ci": ci,
        "ppl_ratio": math.exp(statistics.fmean(deltas)),
        "mean_kl": statistics.fmean(row["kl_reference_to_candidate"] for row in rows),
        "mean_logit_cosine": statistics.fmean(row["mean_logit_cosine"] for row in rows),
        "top1_agreement": statistics.fmean(row["top1_agreement"] for row in rows),
        "top5_overlap": statistics.fmean(row["top5_overlap"] for row in rows),
        "maximum_saturation_rate": max(
            module["saturation_rate"] for module in local_summary.values()
        ),
        "all_finite": all(row["finite"] for row in rows),
        "slice_delta_nll": slice_delta,
        "module_local": local_summary,
        "wall_time_s": time.perf_counter() - started,
        "rows": rows,
    }
    return summary


def _quality_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "mean_delta_nll": summary["mean_delta_nll"] <= QUALITY_GATES["max_mean_delta_nll"],
        "delta_nll_ci_high": summary["delta_nll_ci"]["high"] <= QUALITY_GATES["max_delta_nll_ci_high"],
        "mean_kl": summary["mean_kl"] <= QUALITY_GATES["max_mean_kl"],
        "top1_agreement": summary["top1_agreement"] >= QUALITY_GATES["min_top1_agreement"],
        "saturation_rate": summary["maximum_saturation_rate"] <= QUALITY_GATES["max_saturation_rate"],
        "finite": bool(summary["all_finite"]),
    }
    return {"checks": checks, "passed": all(checks.values()), "thresholds": QUALITY_GATES}


def _select_family(
    output: Path, subsets: Sequence[Mapping[str, Any]], results: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    stability = {
        row["candidate_id"]: row for row in _jsonl_read(output / "stats/activation/scale_stability.jsonl")
    }
    stat_cost = json.loads((output / "cost/stat_collection.json").read_text())["stat_collection"]["wall_time_s"]
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    subset_by_id = {row["candidate_id"]: row for row in subsets}
    for candidate_id, result in results.items():
        subset = subset_by_id[candidate_id]
        groups.setdefault(
            (subset["source"], subset["length_coverage"], subset["sample_count"]), []
        ).append(result)
    family_rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for (source, coverage, count), values in sorted(groups.items()):
        means = [row["mean_delta_nll"] for row in values]
        seed_ci = qstats.bootstrap_ci(means, resamples=1000, seed=5304).as_dict()
        subsets_for_family = [
            subset_by_id[row["candidate_id"]] for row in values
        ]
        row = {
            "family_id": f"{source}__{coverage}__n{count}",
            "source": source,
            "length_coverage": coverage,
            "sample_count": count,
            "seeds": sorted(subset["subset_seed"] for subset in subsets_for_family),
            "candidate_ids": sorted(row["candidate_id"] for row in values),
            "valid_tokens_mean": statistics.fmean(subset["valid_tokens"] for subset in subsets_for_family),
            "mean_delta_nll": statistics.fmean(means),
            "seed_variance": statistics.pvariance(means),
            "seed_range": max(means) - min(means),
            "seed_level_ci": seed_ci,
            "worst_slice_delta_nll": max(
                slice_value
                for result in values
                for slice_value in result["slice_delta_nll"].values()
            ),
            "minimum_scale_cosine_to_next": min(
                stability[result["candidate_id"]]["minimum_cosine"]
                for result in values
                if result["candidate_id"] in stability
            ) if all(result["candidate_id"] in stability for result in values) else None,
            "offline_cost_s": stat_cost + sum(
                json.loads(
                    (output / "artifacts" / result["candidate_id"] / "manifest.json").read_text()
                )["offline_cost_s"]
                for result in values
            ),
        }
        family_rows.append(row)
    family_by_key = {
        (row["source"], row["length_coverage"], row["sample_count"]): row
        for row in family_rows
    }
    stable_rows = []
    for row in family_rows:
        next_counts = [count for count in SAMPLE_COUNTS if count > row["sample_count"]]
        following = family_by_key.get(
            (row["source"], row["length_coverage"], min(next_counts))
        ) if next_counts else None
        checks = {
            "next_budget_exists": following is not None,
            "budget_saturation": following is not None and abs(
                following["mean_delta_nll"] - row["mean_delta_nll"]
            ) < SELECTION_RULE["stabilization_epsilon_delta_nll"],
            "seed_range": row["seed_range"] < SELECTION_RULE["max_seed_range_delta_nll"],
            "ci_overlap": following is not None
            and row["seed_level_ci"]["low"] <= following["seed_level_ci"]["high"]
            and following["seed_level_ci"]["low"] <= row["seed_level_ci"]["high"],
            "slices": row["worst_slice_delta_nll"] <= SELECTION_RULE["max_policy_slice_delta_nll"],
            "scale_stability": row["minimum_scale_cosine_to_next"] is not None
            and row["minimum_scale_cosine_to_next"] >= SELECTION_RULE["min_scale_cosine_to_next_budget"],
            "offline_cost": row["offline_cost_s"] <= SELECTION_RULE["offline_cost_ceiling_s"],
        }
        item = {**row, "checks": checks, "stable": all(checks.values())}
        trace.append(item)
        if item["stable"]:
            stable_rows.append(item)
    selected_family = min(
        stable_rows,
        key=lambda row: (
            row["sample_count"], row["valid_tokens_mean"], row["source"], row["length_coverage"]
        ),
    ) if stable_rows else None
    selected_candidate = None
    if selected_family:
        selected_candidate = next(
            candidate_id
            for candidate_id in selected_family["candidate_ids"]
            if subset_by_id[candidate_id]["subset_seed"] == SELECTION_RULE["selected_seed"]
        )
    return {
        "rule": SELECTION_RULE,
        "selected_family": selected_family["family_id"] if selected_family else None,
        "selected_candidate": selected_candidate,
        "reason": "cheapest stable family under pre-registered policy-only rule"
        if selected_family
        else "no family satisfied every pre-registered rule",
        "trace": trace,
        "stable_families": [row["family_id"] for row in stable_rows],
        "family_rows": family_rows,
    }


def command_evaluate(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    subsets = _jsonl_read(output / "data/subsets.jsonl")
    if not subsets:
        raise RuntimeError("run spec-data first")
    tokenizer, model, load_cost = _load_model(args)
    corpus = _build_corpus(tokenizer)
    modules = _resolve_modules(model)
    policy_samples = corpus.manifests[SPLIT_POLICY_VALIDATION].samples
    results: dict[str, dict[str, Any]] = {}
    policy_dir = output / "quality/policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    evaluation_started = time.perf_counter()
    for index, subset in enumerate(subsets, 1):
        path = policy_dir / f"{subset['candidate_id']}.json"
        artifact, scales = _load_artifact(output, subset["candidate_id"])
        if path.is_file():
            result = json.loads(path.read_text())
            if result.get("artifact_hash") != artifact["artifact_hash"]:
                raise RuntimeError(f"stale policy result for {subset['candidate_id']}")
        else:
            result = _evaluate_artifact(
                model,
                modules,
                scales,
                policy_samples,
                candidate_id=subset["candidate_id"],
                split=SPLIT_POLICY_VALIDATION,
            )
            result["artifact_hash"] = artifact["artifact_hash"]
            result["quality_gate"] = _quality_gate(result)
            _json_write(path, result)
        results[subset["candidate_id"]] = result
        print(
            f"policy {index}/{len(subsets)} {subset['candidate_id']} "
            f"delta_nll={result['mean_delta_nll']:.6g}",
            flush=True,
        )
    learning_rows = []
    local_rows = []
    slice_rows = []
    subset_by_id = {row["candidate_id"]: row for row in subsets}
    for candidate_id, result in sorted(results.items()):
        subset = subset_by_id[candidate_id]
        learning_rows.append(
            {
                "candidate_id": candidate_id,
                "source": subset["source"],
                "sample_count": subset["sample_count"],
                "valid_tokens": subset["valid_tokens"],
                "length_coverage": subset["length_coverage"],
                "subset_seed": subset["subset_seed"],
                "mean_delta_nll": result["mean_delta_nll"],
                "delta_nll_ci_low": result["delta_nll_ci"]["low"],
                "delta_nll_ci_high": result["delta_nll_ci"]["high"],
                "ppl_ratio": result["ppl_ratio"],
                "mean_kl": result["mean_kl"],
                "top1_agreement": result["top1_agreement"],
                "maximum_saturation_rate": result["maximum_saturation_rate"],
                "artifact_hash": result["artifact_hash"],
            }
        )
        for module, metrics in result["module_local"].items():
            local_rows.append({"candidate_id": candidate_id, "module": module, **metrics})
        for slice_name, delta in result["slice_delta_nll"].items():
            slice_rows.append(
                {"candidate_id": candidate_id, "split": SPLIT_POLICY_VALIDATION, "slice": slice_name, "delta_nll": delta}
            )
    _jsonl_write(output / "quality/learning_curve.jsonl", learning_rows)
    _jsonl_write(output / "quality/local.jsonl", local_rows)
    _jsonl_write(output / "quality/slices.jsonl", slice_rows)
    selection = _select_family(output, subsets, results)
    _json_write(output / "selection/rule.json", SELECTION_RULE)
    _json_write(
        output / "selection/trace.json",
        {key: value for key, value in selection.items() if key != "family_rows"},
    )
    _jsonl_write(output / "selection/family_summary.jsonl", selection["family_rows"])
    if selection["selected_candidate"] is None:
        _json_write(
            output / "cost/evaluation.json",
            {
                "model_load": load_cost,
                "policy_validation_wall_time_s": time.perf_counter() - evaluation_started,
                "final_evaluation_executed": False,
                "reason": selection["reason"],
            },
        )
        print(json.dumps(selection, indent=2))
        return 0
    selected = selection["selected_candidate"]
    artifact, scales = _load_artifact(output, selected)
    frozen = {
        "schema": "hqsb.e05_03.frozen_policy/v1",
        "frozen_at_utc": _utc_now(),
        "selected_family": selection["selected_family"],
        "selected_candidate": selected,
        "artifact_hash": artifact["artifact_hash"],
        "subset_hash": artifact["subset_hash"],
        "sample_ids": artifact["sample_ids"],
        "method": artifact["method"],
        "selection_trace_sha256": _sha256_file(output / "selection/trace.json"),
        "final_evaluation_consumed": False,
    }
    policy_hashable = dict(frozen)
    policy_hashable.pop("frozen_at_utc")
    policy_hashable.pop("final_evaluation_consumed")
    frozen["policy_hash"] = _sha256_bytes(
        json.dumps(policy_hashable, sort_keys=True, separators=(",", ":")).encode()
    )
    frozen_path = output / "selection/frozen_policy.json"
    if frozen_path.exists():
        existing = json.loads(frozen_path.read_text())
        if existing.get("policy_hash") != frozen["policy_hash"]:
            raise RuntimeError("frozen policy changed after selection")
        frozen = existing
    else:
        _json_write(frozen_path, frozen)

    final_path = output / "quality/final.json"
    consumption_path = output / "selection/final_consumption.json"
    if not final_path.exists():
        consumption = json.loads(consumption_path.read_text()) if consumption_path.exists() else {
            "policy_hash": frozen["policy_hash"],
            "selected_candidate": selected,
            "started_at_utc": _utc_now(),
            "attempts_same_frozen_policy": 0,
        }
        if consumption["policy_hash"] != frozen["policy_hash"]:
            raise RuntimeError("final evaluation marker belongs to a different policy")
        consumption["attempts_same_frozen_policy"] += 1
        _json_write(consumption_path, consumption)
        final = _evaluate_artifact(
            model,
            modules,
            scales,
            corpus.manifests[SPLIT_FINAL_EVALUATION].samples,
            candidate_id=selected,
            split=SPLIT_FINAL_EVALUATION,
        )
        final["artifact_hash"] = artifact["artifact_hash"]
        final["policy_hash"] = frozen["policy_hash"]
        final["quality_gate"] = _quality_gate(final)
        final["completed_at_utc"] = _utc_now()
        _json_write(final_path, final)
        consumption["completed_at_utc"] = final["completed_at_utc"]
        consumption["completed"] = True
        _json_write(consumption_path, consumption)
        frozen["final_evaluation_consumed"] = True
        frozen["final_result_sha256"] = _sha256_file(final_path)
        _json_write(frozen_path, frozen)
    else:
        final = json.loads(final_path.read_text())
        if final.get("policy_hash") != frozen["policy_hash"]:
            raise RuntimeError("existing final result does not match frozen policy")

    stress_path = output / "quality/stress.json"
    if not stress_path.exists():
        stress = _evaluate_artifact(
            model,
            modules,
            scales,
            corpus.manifests[SPLIT_STRESS].samples,
            candidate_id=selected,
            split=SPLIT_STRESS,
        )
        stress["artifact_hash"] = artifact["artifact_hash"]
        stress["selection_use_forbidden"] = True
        stress["descriptive_gate"] = _quality_gate(stress)
        _json_write(stress_path, stress)
    else:
        stress = json.loads(stress_path.read_text())
    _json_write(
        output / "cost/evaluation.json",
        {
            "model_load": load_cost,
            "policy_candidates": len(results),
            "policy_validation_wall_time_s": sum(row["wall_time_s"] for row in results.values()),
            "selected_final_wall_time_s": final["wall_time_s"],
            "stress_wall_time_s": stress["wall_time_s"],
            "host_peak_bytes": _rss_bytes(),
            "device_peak_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "failed_runs": [],
        },
    )
    print(
        json.dumps(
            {
                "selected": selected,
                "policy_hash": frozen["policy_hash"],
                "final_gate": final["quality_gate"],
                "stress_gate_descriptive": stress["descriptive_gate"],
            },
            indent=2,
        )
    )
    return 0


def _factor_effects(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    for field in ("source", "sample_count", "length_coverage", "subset_seed"):
        grouped: dict[str, list[float]] = {}
        for row in rows:
            grouped.setdefault(str(row[field]), []).append(float(row["mean_delta_nll"]))
        means = {key: statistics.fmean(values) for key, values in sorted(grouped.items())}
        effects[field] = {
            "mean_delta_nll": means,
            "range_across_levels": max(means.values()) - min(means.values()),
        }
    worst = max(rows, key=lambda row: row["mean_delta_nll"])
    best = min(rows, key=lambda row: row["mean_delta_nll"])
    effects["best_candidate"] = dict(best)
    effects["worst_candidate"] = dict(worst)
    return effects


def _evidence_manifest(output: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "EVIDENCE_MANIFEST.json":
            continue
        rows.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    root_hash = _sha256_bytes(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    )
    return {
        "schema": "hqsb.evidence_manifest/v1",
        "generated_at_utc": _utc_now(),
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "root_sha256": root_hash,
    }


def command_verify(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    spec = json.loads((output / "spec.json").read_text())
    leakage = json.loads((output / "data/leakage_report.json").read_text())
    subsets = _jsonl_read(output / "data/subsets.jsonl")
    learning = _jsonl_read(output / "quality/learning_curve.jsonl")
    activation_index = _jsonl_read(output / "stats/activation/index.jsonl")
    replay = _jsonl_read(output / "artifacts/replay.jsonl")
    selection = json.loads((output / "selection/trace.json").read_text())
    final_path = output / "quality/final.json"
    stress_path = output / "quality/stress.json"
    final = json.loads(final_path.read_text()) if final_path.is_file() else None
    stress = json.loads(stress_path.read_text()) if stress_path.is_file() else None
    frozen_path = output / "selection/frozen_policy.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.is_file() else None
    consumption_path = output / "selection/final_consumption.json"
    consumption = json.loads(consumption_path.read_text()) if consumption_path.is_file() else None
    contamination = leakage.get("benchmark_contamination", [])
    contamination_clean = all(
        item.get("checked") and item.get("hit_count", 0) == 0 for item in contamination
    )
    split_manifests = [
        output / "data/calibration_manifest.jsonl",
        output / "data/policy_validation_manifest.jsonl",
        output / "data/final_evaluation_manifest.jsonl",
        output / "data/stress_manifest.jsonl",
    ]
    all_stats_complete = len(activation_index) == CALIBRATION_PER_CELL * 4 and all(
        len(row["modules"]) == len(MODULE_SUFFIXES)
        and all(
            module["p999_abs"] >= module["p99_abs"] >= module["p90_abs"]
            and module["padded_tokens"] == 0
            and module["nan_count"] == 0
            and module["inf_count"] == 0
            for module in row["modules"]
        )
        for row in activation_index
    )
    factor_replayable = (
        len(subsets) == 2 * 2 * len(SAMPLE_COUNTS) * len(SUBSET_SEEDS)
        and {row["source"] for row in subsets} == {"technical", "code"}
        and {row["length_coverage"] for row in subsets} == {"short-only", "balanced"}
        and {row["sample_count"] for row in subsets} == set(SAMPLE_COUNTS)
        and {row["subset_seed"] for row in subsets} == set(SUBSET_SEEDS)
        and all(row["valid_tokens"] > 0 and row["token_hashes"] for row in subsets)
    )
    rtn = json.loads((output / "artifacts/rtn_control.json").read_text())
    final_after_freeze = bool(
        frozen
        and final
        and consumption
        and frozen.get("final_evaluation_consumed")
        and final.get("policy_hash") == frozen.get("policy_hash")
        and consumption.get("policy_hash") == frozen.get("policy_hash")
        and consumption.get("attempts_same_frozen_policy") == 1
    )
    costs_present = all(
        (output / relative).is_file()
        for relative in (
            "cost/stat_collection.json",
            "cost/artifact_generation.json",
            "cost/evaluation.json",
        )
    )
    criteria = [
        {
            "id": 1,
            "name": "four split identities/revisions/hashes auditable",
            "passed": all(path.is_file() and path.stat().st_size > 0 for path in split_manifests),
        },
        {
            "id": 2,
            "name": "exact/near-duplicate/parent/template/answer leakage checked",
            "passed": not leakage["leaked"] and leakage["pairs_checked"] == 6 and contamination_clean,
        },
        {"id": 3, "name": "source/count/token/length/seed factors replayable", "passed": factor_replayable},
        {"id": 4, "name": "RTN control invariant across subsets", "passed": rtn["invariant"] and len(rtn["unique_artifact_ids"]) == 1},
        {"id": 5, "name": "range method statistics and numerical stability saved", "passed": all_stats_complete},
        {
            "id": 6,
            "name": "three independent calibration seeds with mean/variance/CI",
            "passed": len(learning) == len(subsets)
            and all(len(row["seeds"]) == 3 for row in selection.get("trace", [])),
        },
        {"id": 7, "name": "minimum sufficient configuration selected by preregistered policy rule", "passed": bool(selection.get("selected_candidate"))},
        {"id": 8, "name": "final used only after freeze and exactly one attempt", "passed": final_after_freeze},
        {
            "id": 9,
            "name": "domain/length/language slices and OOD boundary reported",
            "passed": bool(final and stress and final["slice_delta_nll"] and stress["slice_delta_nll"]),
        },
        {"id": 10, "name": "offline time/memory/failures complete", "passed": costs_present},
        {
            "id": 11,
            "name": "frozen handoff sufficient for E05-04 unified calibration protocol",
            "passed": bool(frozen and frozen.get("artifact_hash") and frozen.get("sample_ids")),
        },
        {
            "id": 12,
            "name": "raw to curve to selection to final traceable",
            "passed": len(replay) == len(subsets)
            and all(row["bitwise_replay"] for row in replay)
            and bool(learning and selection and final),
        },
    ]
    factor_effects = _factor_effects(learning)
    _json_write(output / "quality/factor_effects.json", factor_effects)
    if frozen:
        handoff = {
            "schema": "hqsb.e05_03.e05_04_handoff/v1",
            "data_spec_hash": spec["data_spec_hash"],
            "calibration_manifest_sha256": _sha256_file(output / "data/calibration_manifest.jsonl"),
            "policy_validation_manifest_sha256": _sha256_file(output / "data/policy_validation_manifest.jsonl"),
            "final_evaluation_manifest_sha256": _sha256_file(output / "data/final_evaluation_manifest.jsonl"),
            "stress_manifest_sha256": _sha256_file(output / "data/stress_manifest.jsonl"),
            "minimum_sufficient_family": selection["selected_family"],
            "selected_calibration_candidate": selection["selected_candidate"],
            "sample_ids": frozen["sample_ids"],
            "artifact_hash": frozen["artifact_hash"],
            "statistics_schema": "stats/activation/index.jsonl + per_sample/*.npz",
            "leakage_rule": "data/leakage_report.json",
            "offline_cost_fields": ["phase", "wall_time_s", "host_peak_bytes", "device_peak_bytes", "failures"],
            "limitations": spec["known_limits"],
        }
        _json_write(output / "selection/e05_04_handoff.json", handoff)
    expected_effect = {
        "name": "calibration gain is not merely final-evaluation overfitting",
        "passed": bool(
            not leakage["leaked"]
            and final
            and final["quality_gate"]["passed"]
            and selection.get("selected_candidate")
        ),
        "evidence": {
            "final_quality_gate": final["quality_gate"] if final else None,
            "factor_effects": factor_effects,
            "scope_limit": "repository-local corpus and selected-module static-A8 proxy only",
        },
    }
    single_standard = {
        "name": "calibration/evaluation isolated; minimum sufficient configuration and failure distribution reported",
        "passed": bool(criteria[1]["passed"] and criteria[6]["passed"] and criteria[8]["passed"]),
    }
    scientific = "PASS" if all(row["passed"] for row in criteria) and expected_effect["passed"] and single_standard["passed"] else "FAIL"
    formal_allowed = bool(spec["preconditions"]["formal_pass_allowed"])
    overall = scientific if formal_allowed else "BLOCKED"
    verdict = {
        "schema": "hqsb.e05_03.verdict/v1",
        "verified_at_utc": _utc_now(),
        "overall": overall,
        "scientific_execution_verdict": scientific,
        "formal_pass_allowed": formal_allowed,
        "blocking_precondition": None if formal_allowed else "E05-02 overall is not PASS",
        "expected_effect": expected_effect,
        "single_item_standard": single_standard,
        "detail_pass_criteria": criteria,
        "passed_criteria": sum(row["passed"] for row in criteria),
        "total_criteria": len(criteria),
        "selected_family": selection.get("selected_family"),
        "selected_candidate": selection.get("selected_candidate"),
        "final_summary": {
            key: final[key]
            for key in (
                "mean_delta_nll", "delta_nll_ci", "ppl_ratio", "mean_kl",
                "top1_agreement", "maximum_saturation_rate", "quality_gate",
            )
        } if final else None,
        "stress_summary": {
            key: stress[key]
            for key in (
                "mean_delta_nll", "delta_nll_ci", "ppl_ratio", "mean_kl",
                "top1_agreement", "maximum_saturation_rate", "descriptive_gate",
            )
        } if stress else None,
        "format_deviation": "Required parquet tables emitted as JSONL because pyarrow is absent; no fake .parquet files were created.",
    }
    _json_write(output / "verdict.json", verdict)
    manifest = _evidence_manifest(output)
    _json_write(output / "EVIDENCE_MANIFEST.json", manifest)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print(json.dumps({"evidence_root": manifest["root_sha256"], "files": manifest["file_count"]}, indent=2))
    return 0 if overall in ("PASS", "BLOCKED") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("spec-data", "collect-stats", "build-artifacts", "evaluate", "verify"),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--model-manifest", default=str(MODEL_MANIFEST))
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    commands = {
        "spec-data": command_spec_data,
        "collect-stats": command_collect_stats,
        "build-artifacts": command_build_artifacts,
        "evaluate": command_evaluate,
        "verify": command_verify,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
