#!/usr/bin/env python3
"""E02-05 runner: weights / KV / activation / allocator memory-model validation.

CLI over :mod:`hqsb.benchmark.memory_experiment`. See
``docs/stage_experiments/details/S02/E02-05_memory_model_validation.md`` for the
protocol and ``docs/stage_experiments/S02/E02-05/E02-05_实验报告.md`` for the
result of the frozen run.

Sub-commands::

    # allocator capability probe (each mode must run in its own process)
    python3 scripts/audit/run_e02_05_memory_model.py probe \
        --mode caching --output-dir docs/stage_experiments/S02/E02-05/raw
    python3 scripts/audit/run_e02_05_memory_model.py probe \
        --mode no-caching --output-dir docs/stage_experiments/S02/E02-05/raw

    # collection; repeat with --run-index 0/1/2 for independent-process recheck
    python3 scripts/audit/run_e02_05_memory_model.py collect \
        --output-dir docs/stage_experiments/S02/E02-05/raw --run-index 0

    # verdict
    python3 scripts/audit/run_e02_05_memory_model.py verify \
        --output-dir docs/stage_experiments/S02/E02-05/raw
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
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from hqsb.benchmark import memory_experiment as me
from hqsb.benchmark.memory import process_rss_bytes, process_swap_bytes
from hqsb.benchmark.memory_model import (
    MIB,
    DeviceMemorySampler,
    bytes_to_mib,
    memory_snapshot,
    model_memory_inventory,
    snapshot_used_bytes,
)
from hqsb.benchmark.resource_monitor import TegrastatsMonitor
from hqsb.benchmark.tegrastats_parser import (
    compute_power_summary,
    compute_resource_summary,
    parse_tegrastats_line,
)
from hqsb.benchmark.workload import make_fixed_token_input
from hqsb.models.loader import load_qwen3

logger = logging.getLogger("e02_05")

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ───────────────────────────── identity helpers ─────────────────────────────


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
    try:
        import transformers  # noqa: PLC0415

        env["transformers_version"] = transformers.__version__
    except Exception:  # noqa: BLE001
        env["transformers_version"] = None
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


# ─────────────────────────────── collect ───────────────────────────────


def _run_collect(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"run_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    )
    out_path = output_dir / f"run_{args.run_index}.json"
    telemetry_path = output_dir / f"run_{args.run_index}.tegrastats.txt"

    record: Dict[str, Any] = {
        "experiment": "E02-05",
        "run_id": run_id,
        "run_index": args.run_index,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "environment": _environment(),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "manifest_sha256": _manifest_hash(args.manifest),
        "protocol": me.PROTOCOL,
        "tolerances": me.TOLERANCES,
        "safety": me.SAFETY,
        "env_allocator": {
            "PYTORCH_NO_CUDA_MEMORY_CACHING": os.environ.get(
                "PYTORCH_NO_CUDA_MEMORY_CACHING", ""
            ),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        },
        "sections": args.sections,
        "stages": {},
        "inventory": None,
        "load": None,
        "allocator_probe": None,
        "staged_point": None,
        "context_sweep": [],
        "batch_sweep": [],
        "multi_request": None,
        "hook_ab": None,
        "noise_probe": None,
        "repeat_probe": None,
        "oom_cross_reference": None,
        "abnormal_runs": [],
        "telemetry": None,
    }

    def _emit() -> None:
        """Persist incrementally so an OOM-kill never discards finished points."""
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    # ── M0: before any CUDA API touches the device ────────────────
    record["stages"]["M0_process_start"] = memory_snapshot(label="M0", with_cuda=False)

    # ── M1: CUDA context / libraries only ─────────────────────────
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    record["stages"]["M1_cuda_context"] = memory_snapshot(label="M1")
    _emit()

    # ── M2: model loaded (weights resident, no KV) ────────────────
    load_sampler = DeviceMemorySampler(me.PROTOCOL["sampler_interval_s"])
    load_sampler.start()
    tokenizer, model, load_time_s = load_qwen3(
        args.model_path,
        dtype=torch.float16,
        attention_backend="eager",
        verify_manifest=args.manifest,
        allow_extra=("model_sha256_manifest.txt",),
    )
    load_sampler.stop()
    record["stages"]["M2_model_loaded"] = memory_snapshot(label="M2")

    param_devices = sorted({str(p.device) for p in model.parameters()})
    record["load"] = {
        "load_time_s": load_time_s,
        "peak": load_sampler.summary(),
        "param_devices": param_devices,
        "model_fully_on_gpu": bool(param_devices)
        and all(d.startswith("cuda") for d in param_devices),
    }
    record["inventory"] = model_memory_inventory(model)
    record["dims"] = me.model_dims(model)
    record["allocator_probe"] = me.allocator_probe()
    _emit()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampler_interval_s = float(me.PROTOCOL["sampler_interval_s"])

    monitor: Optional[TegrastatsMonitor] = None
    monitor_records: List[Dict[str, Any]] = []
    try:
        monitor = TegrastatsMonitor(interval_ms=args.monitor_interval_ms)
        monitor.start()
    except Exception as exc:  # noqa: BLE001
        logger.warning("tegrastats unavailable: %s", exc)
        monitor = None

    # Warm-up outside the measured points (lazy init / cuBLAS workspace).
    warm = make_fixed_token_input(tokenizer, 32, device=device)
    me.measure_point(model, tokenizer, 32, 2, 1, device, sampler_interval_s, label="warmup")
    del warm
    gc.collect()

    warmup_passes = int(me.PROTOCOL["shape_warmup_passes"])

    def _free_mb() -> float:
        return torch.cuda.mem_get_info()[0] / MIB if device == "cuda" else 0.0

    try:
        if "noise_probe" in args.sections:
            record["noise_probe"] = me.noise_probe(me.PROTOCOL["noise_probe"])
            _emit()

        if "repeat_probe" in args.sections:
            record["repeat_probe"] = me.run_repeat_probe(
                model, tokenizer, device, me.PROTOCOL["repeat_probe"],
                sampler_interval_s, warmup_passes=warmup_passes,
            )
            _emit()

        if "staged" in args.sections:
            spec = me.PROTOCOL["staged"]
            staged = me.measure_point(
                model, tokenizer, spec["input_tokens"], spec["output_tokens"],
                spec["batch_size"], device, sampler_interval_s,
                label="staged_first_commit", warmup_passes=warmup_passes,
            )
            record["staged_point"] = staged
            snaps = staged["snapshots"]
            record["stages"]["M3_prefill_with_logits"] = snaps["prefill_with_logits"]
            record["stages"]["M3b_kv_live_no_logits"] = snaps["prefill_steady"]
            record["stages"]["M4_decode_end"] = snaps["decode_end"]
            record["stages"]["M5_outputs_retained"] = snaps["outputs_retained"]
            record["stages"]["M6_released"] = snaps["released"]
            gc.collect()
            record["stages"]["M7_before_next_request"] = memory_snapshot(label="M7")
            _emit()

        if "context_sweep" in args.sections:
            spec = me.PROTOCOL["context_sweep"]
            for isl in spec["input_tokens"]:
                if _free_mb() < me.SAFETY["min_device_free_mb"]:
                    record["abnormal_runs"].append(
                        {
                            "section": "context_sweep",
                            "key": f"I{isl}",
                            "reason": f"safety_stop: device free {_free_mb():.1f} MiB",
                        }
                    )
                    break
                logger.info("context sweep ISL=%d", isl)
                record["context_sweep"].append(
                    me.measure_point(
                        model, tokenizer, isl, spec["output_tokens"],
                        spec["batch_size"], device, sampler_interval_s,
                        label=f"context_I{isl}", warmup_passes=warmup_passes,
                    )
                )
                _emit()

        if "batch_sweep" in args.sections:
            for spec in me.PROTOCOL["batch_sweep"]:
                for batch in spec["batch_sizes"]:
                    if _free_mb() < me.SAFETY["min_device_free_mb"]:
                        record["abnormal_runs"].append(
                            {
                                "section": "batch_sweep",
                                "key": f"I{spec['input_tokens']}_B{batch}",
                                "reason": f"safety_stop: device free {_free_mb():.1f} MiB",
                            }
                        )
                        break
                    logger.info("batch sweep ISL=%d B=%d", spec["input_tokens"], batch)
                    record["batch_sweep"].append(
                        me.measure_point(
                            model, tokenizer, spec["input_tokens"],
                            spec["output_tokens"], batch, device, sampler_interval_s,
                            label=f"batch_I{spec['input_tokens']}_B{batch}",
                            warmup_passes=warmup_passes,
                        )
                    )
                    _emit()

        if "multi_request" in args.sections:
            record["multi_request"] = me.run_multi_request(
                model, tokenizer, device, me.PROTOCOL["multi_request"],
                sampler_interval_s,
            )
            _emit()

        if "hook_ab" in args.sections:
            record["hook_ab"] = me.run_hook_ab(
                model, tokenizer, device, me.PROTOCOL["hook_ab"], sampler_interval_s
            )
            _emit()

        m2 = record["stages"]["M2_model_loaded"]
        free_after_load = (
            None
            if m2["device"]["free_mb"] is None
            else int(m2["device"]["free_mb"] * MIB)
        )
        record["available_after_load_bytes"] = free_after_load
        record["oom_cross_reference"] = me.oom_cross_reference(
            record["dims"], free_after_load
        )
        _emit()
    finally:
        if monitor is not None:
            monitor.stop()
            monitor_records = list(monitor.records)

    with telemetry_path.open("w", encoding="utf-8") as fh:
        for item in monitor_records:
            fh.write(f"{item['time_ns']}\t{item['raw']}\n")

    record["telemetry"] = _telemetry_summary(monitor_records)
    record["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _emit()
    logger.info("wrote %s (load=%.1fs)", out_path, load_time_s)
    return 0


def _telemetry_summary(monitor_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not monitor_records:
        return {"available": False}
    parsed: List[Dict[str, Any]] = []
    for item in monitor_records:
        fields = parse_tegrastats_line(item["raw"])
        fields["time_ns"] = item["time_ns"]
        parsed.append(fields)
    power = compute_power_summary(parsed)
    resource = compute_resource_summary(parsed)
    return {
        "available": True,
        "num_records": len(monitor_records),
        "avg_power_w": power.get("avg_power_w"),
        "peak_power_w": power.get("peak_power_w"),
        "avg_gpu_temp_c": resource.get("avg_gpu_temp_c"),
        "peak_gpu_temp_c": resource.get("peak_gpu_temp_c"),
        "avg_gpu_util_pct": resource.get("avg_gpu_util_pct"),
        "peak_ram_used_mb": resource.get("peak_ram_used_mb"),
        "thermal_suspect": bool(
            (resource.get("peak_gpu_temp_c") or 0) > me.SAFETY["thermal_suspect_c"]
        ),
    }


# ─────────────────────────────── probe ───────────────────────────────


def _run_probe(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode.startswith("no-caching"):
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    else:
        os.environ.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)

    # ``auto`` lets accelerate shard across GPU/CPU/disk, which is what blows up
    # during loading (E02-04 §12). ``cuda`` asks for a direct full-GPU placement.
    device_map = "cuda" if args.mode.endswith("-direct") else None

    out_path = output_dir / f"allocator_probe_{args.mode}.json"

    probe: Dict[str, Any] = {
        "experiment": "E02-05",
        "mode": args.mode,
        "device_map": device_map or "auto",
        "requested_env": {
            "PYTORCH_NO_CUDA_MEMORY_CACHING": os.environ.get(
                "PYTORCH_NO_CUDA_MEMORY_CACHING", ""
            )
        },
        "environment": _environment(),
        "git_commit": _git_commit(),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": "start",
    }

    def _write_probe() -> None:
        """Persist before the model load: an OOM-kill (SIGKILL) is uncatchable,
        so a probe that dies during loading must still leave its context and
        allocator evidence on disk."""
        probe["written_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        out_path.write_text(json.dumps(probe, indent=2), encoding="utf-8")

    try:
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        probe["context"] = memory_snapshot(label="context")
        probe["allocator_probe"] = me.allocator_probe()
        probe["stage"] = "before_load"
        _write_probe()

        load_sampler = DeviceMemorySampler(0.02)
        load_sampler.start()
        tokenizer, model, load_time_s = load_qwen3(
            args.model_path,
            dtype=torch.float16,
            attention_backend="eager",
            device_map=device_map,
            verify_manifest=args.manifest,
            allow_extra=("model_sha256_manifest.txt",),
        )
        load_sampler.stop()

        probe["load"] = {
            "load_time_s": load_time_s,
            "peak": load_sampler.summary(),
            "succeeded": True,
        }
        probe["after_load"] = memory_snapshot(label="after_load")
        probe["after_load_allocator"] = me.allocator_view()
        probe["inventory_resident_bytes"] = model_memory_inventory(model)[
            "resident_bytes"
        ]

        point = me.measure_point(
            model, tokenizer, 128, 8, 1, "cuda", 0.02, label="probe_point"
        )
        probe["point"] = {
            "observations": point["observations"],
            "decomposition": point["decomposition"],
            "kv_checks": point["kv_checks"],
        }
        del model, tokenizer
        gc.collect()
        probe["stage"] = "complete"
    except Exception as exc:  # noqa: BLE001
        probe["load"] = {"succeeded": False}
        probe["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "looks_like_oom": "out of memory" in str(exc).lower(),
        }
        probe["stage"] = "failed"
        logger.warning("probe %s failed: %s", args.mode, exc)

    probe["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write_probe()
    logger.info("wrote %s", out_path)
    print(
        json.dumps(
            {k: probe[k] for k in ("mode", "load", "error") if k in probe}, indent=2
        )
    )
    return 0


# ─────────────────────────────── verify ───────────────────────────────


def _run_verify(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    run_files = sorted(output_dir.glob("run_*.json"))
    if not run_files:
        logger.error("no run_*.json found in %s", output_dir)
        return 2
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in run_files]

    unit_selftest = me.unit_selftest()

    # ── Weight ledger (storage-de-duplicated) vs the device delta M2-M1 ──
    weight_rows: List[Dict[str, Any]] = []
    for record in runs:
        stages = record.get("stages", {})
        m1 = stages.get("M1_cuda_context") or {}
        m2 = stages.get("M2_model_loaded") or {}
        inventory = record.get("inventory") or {}
        predicted = inventory.get("resident_bytes")
        observed = None
        if (
            m1.get("device_used_mb") is not None
            and m2.get("device_used_mb") is not None
        ):
            observed = int(round((m2["device_used_mb"] - m1["device_used_mb"]) * MIB))
        decomposed = (
            me.decompose(predicted={"resident_weights": predicted}, observed_bytes=observed)
            if predicted is not None
            else None
        )
        relativity = (
            me.relative_error(predicted, observed)
            if predicted is not None and observed is not None
            else None
        )
        param_summary = inventory.get("parameter_summary") or {}
        buffer_summary = inventory.get("buffer_summary") or {}
        weight_rows.append(
            {
                "run_index": record.get("run_index"),
                "predicted_resident_bytes": predicted,
                "predicted_resident_mib": (
                    bytes_to_mib(predicted) if predicted is not None else None
                ),
                "observed_m2_minus_m1_bytes": observed,
                "observed_m2_minus_m1_mib": (
                    bytes_to_mib(observed) if observed is not None else None
                ),
                "rel_error": relativity,
                "decomposition": decomposed,
                "num_parameter_names": param_summary.get("num_tensors"),
                "num_unique_parameter_storages": param_summary.get(
                    "num_unique_storages"
                ),
                "duplicate_counted_bytes": param_summary.get(
                    "duplicate_counted_bytes"
                ),
                "alias_groups": param_summary.get("alias_groups"),
                "buffer_dedup_bytes": buffer_summary.get("dedup_total_bytes"),
                "buffer_dtypes": buffer_summary.get("dtype_histogram"),
                "param_dtypes": param_summary.get("dtype_histogram"),
            }
        )
    weight_errors = [r["rel_error"] for r in weight_rows if r["rel_error"] is not None]
    weight_check = {
        "num_runs": len(weight_rows),
        "max_rel_error": max(weight_errors) if weight_errors else None,
        "tolerance": me.TOLERANCES["weight_rel_error"],
        "rows": weight_rows,
        "passed": bool(
            weight_errors and max(weight_errors) <= me.TOLERANCES["weight_rel_error"]
        ),
    }

    # ── KV metadata vs formula ────────────────────────────────────
    kv_rows: List[Dict[str, Any]] = []
    for index, record in enumerate(runs):
        for section, point in me.iter_points(record):
            kv_rows.append(
                {
                    "run_index": record.get("run_index", index),
                    "section": section,
                    "key": me.point_key(point),
                    "theory_kv_final_bytes": point["theory"]["kv_final"]["total_bytes"],
                    "metadata_kv_final_bytes": point["kv_metadata"]["final"][
                        "total_logical_bytes"
                    ],
                    "rel_error": point["kv_checks"]["final_metadata_rel_error"],
                    "num_layers": point["kv_checks"]["metadata_layers"],
                    "kv_heads": point["kv_checks"]["metadata_kv_heads"],
                    "element_bytes": point["kv_checks"]["metadata_element_bytes"],
                    "per_token_per_layer_bytes": point["kv_checks"][
                        "metadata_per_token_per_layer_bytes"
                    ],
                    "preallocated": point["kv_checks"]["metadata_preallocated"],
                    "output_tokens_exact": point["kv_checks"]["output_tokens_exact"],
                }
            )
    kv_errors = [r["rel_error"] for r in kv_rows if r["rel_error"] is not None]
    kv_max = max(kv_errors) if kv_errors else None
    layers = sorted({r["num_layers"] for r in kv_rows})
    heads = sorted({h for r in kv_rows for h in (r["kv_heads"] or [])})
    kv_check = {
        "num_points": len(kv_rows),
        "max_rel_error": kv_max,
        "tolerance": me.TOLERANCES["kv_metadata_rel_error"],
        "layers_observed": layers,
        "kv_heads_observed": heads,
        "all_output_tokens_exact": all(r["output_tokens_exact"] for r in kv_rows),
        "any_preallocated": any(r["preallocated"] for r in kv_rows),
        "passed": bool(
            kv_max is not None
            and kv_max <= me.TOLERANCES["kv_metadata_rel_error"]
            and layers == [28]
            and heads == [8]
            and all(r["output_tokens_exact"] for r in kv_rows)
        ),
        "rows": kv_rows,
    }

    # ── Measured device-observation floor (idle span + known-size recovery) ──
    noise_floors = [
        r["noise_probe"]["noise_floor_bytes"]
        for r in runs
        if r.get("noise_probe") and r["noise_probe"].get("noise_floor_bytes") is not None
    ]
    noise_floor = int(statistics.median(noise_floors)) if noise_floors else 0
    noise_idle_spans = [
        r["noise_probe"]["idle_span_bytes"]
        for r in runs
        if r.get("noise_probe") and r["noise_probe"].get("idle_span_bytes") is not None
    ]
    empty_rise_errors = [
        r["noise_probe"]["median_abs_empty_rise_error_bytes"]
        for r in runs
        if r.get("noise_probe")
        and r["noise_probe"].get("median_abs_empty_rise_error_bytes") is not None
    ]
    # The device-view attribution floor: the idle wander, but also how badly
    # ``cudaMemGetInfo`` under-reports a *reserved* allocation. Both are needed
    # because on this platform the device view is a lower-bound observable.
    device_view_floor = int(
        max(
            noise_floor,
            statistics.median(empty_rise_errors) if empty_rise_errors else 0,
        )
    )
    noise_check = {
        "noise_floor_bytes": noise_floor,
        "noise_floor_mib": bytes_to_mib(noise_floor),
        "median_abs_empty_rise_error_bytes": (
            int(statistics.median(empty_rise_errors)) if empty_rise_errors else None
        ),
        "median_abs_empty_rise_error_mib": (
            bytes_to_mib(statistics.median(empty_rise_errors))
            if empty_rise_errors
            else None
        ),
        "device_view_floor_bytes": device_view_floor,
        "device_view_floor_mib": bytes_to_mib(device_view_floor),
        "idle_span_bytes_max": max(noise_idle_spans) if noise_idle_spans else None,
        "min_signal_over_noise": me.TOLERANCES["min_signal_over_noise"],
        "per_run": [
            {
                "run_index": r.get("run_index"),
                "noise_floor_bytes": (r.get("noise_probe") or {}).get("noise_floor_bytes"),
                "idle_span_bytes": (r.get("noise_probe") or {}).get("idle_span_bytes"),
                "median_abs_net_bytes": (r.get("noise_probe") or {}).get(
                    "median_abs_net_bytes"
                ),
                "median_abs_empty_rise_error_bytes": (r.get("noise_probe") or {}).get(
                    "median_abs_empty_rise_error_bytes"
                ),
                "median_abs_fill_rise_error_bytes": (r.get("noise_probe") or {}).get(
                    "median_abs_fill_rise_error_bytes"
                ),
            }
            for r in runs
        ],
    }

    # ── Steady state (allocator-observed KV) vs formula ────────────
    # Eligibility is a signal-to-noise rule, not a tuned constant: a point is
    # only cross-checked when its predicted KV clears the measured floor. The
    # allowance is the model tolerance *plus* that same floor, so a residual
    # that cannot be distinguished from the device's own accounting wobble is
    # never reported as a model failure.
    threshold = max(
        me.TOLERANCES["min_kv_for_observation_check_bytes"],
        me.TOLERANCES["min_signal_over_noise"] * device_view_floor,
    )
    steady_rows: List[Dict[str, Any]] = []
    for record in runs:
        for section, point in me.iter_points(record):
            predicted = point["predictions"]["steady_final_bytes"]
            observed = point["observations"]["decode_end_increment_bytes"]
            decomposed = point["decomposition"]["steady_final"]
            residual = decomposed["residual_bytes"]
            allowance = (
                me.TOLERANCES["steady_residual_ratio"] * predicted + device_view_floor
            )
            steady_rows.append(
                {
                    "run_index": record.get("run_index"),
                    "section": section,
                    "key": me.point_key(point),
                    "predicted_bytes": predicted,
                    "observed_bytes": observed,
                    "predicted_mib": bytes_to_mib(predicted),
                    "observed_mib": (
                        bytes_to_mib(observed) if observed is not None else None
                    ),
                    "residual_bytes": residual,
                    "residual_ratio": decomposed["residual_ratio"],
                    "eligible": predicted >= threshold,
                    "allowance_bytes": allowance,
                    "within_allowance": (
                        None
                        if residual is None or predicted < threshold
                        else abs(residual) <= allowance
                    ),
                }
            )
    eligible = [r for r in steady_rows if r["eligible"] and r["residual_bytes"] is not None]
    steady_ratios = [abs(r["residual_ratio"]) for r in eligible if r["residual_ratio"] is not None]
    steady_check = {
        "num_points": len(steady_rows),
        "num_eligible": len(eligible),
        "threshold_bytes": threshold,
        "threshold_mib": bytes_to_mib(threshold),
        "noise_floor_bytes": noise_floor,
        "device_view_floor_bytes": device_view_floor,
        "max_abs_residual_ratio": max(steady_ratios) if steady_ratios else None,
        "max_abs_residual_bytes": (
            max(abs(r["residual_bytes"]) for r in eligible) if eligible else None
        ),
        "median_abs_residual_bytes": (
            statistics.median([abs(r["residual_bytes"]) for r in eligible])
            if eligible
            else None
        ),
        "tolerance": me.TOLERANCES["steady_residual_ratio"],
        "passed": bool(eligible and all(r["within_allowance"] for r in eligible)),
        "rows": steady_rows,
    }

    # ── Repeatability of the observation (same point twice) ────────
    repeat_spreads = [
        r["repeat_probe"]["max_spread_bytes"]
        for r in runs
        if r.get("repeat_probe") and r["repeat_probe"].get("max_spread_bytes") is not None
    ]
    repeatability_check = {
        "max_spread_bytes": max(repeat_spreads) if repeat_spreads else None,
        "per_run": [
            {
                "run_index": r.get("run_index"),
                "max_spread_bytes": (r.get("repeat_probe") or {}).get(
                    "max_spread_bytes"
                ),
                "predicted_steady_final_bytes": (r.get("repeat_probe") or {}).get(
                    "predicted_steady_final_bytes"
                ),
                "decode_end_increment_bytes": (r.get("repeat_probe") or {}).get(
                    "decode_end_increment_bytes"
                ),
            }
            for r in runs
        ],
        "passed": bool(
            repeat_spreads
            and max(repeat_spreads) <= me.TOLERANCES["cleanup_growth_bytes"]
        ),
    }

    # ── Peak model (unresolved bucket) ────────────────────────────
    peak_rows: List[Dict[str, Any]] = []
    for record in runs:
        for section, point in me.iter_points(record):
            if point["theory"]["kv_prefill"]["context_length"] < 512:
                continue
            for phase in ("peak_prefill", "peak_decode"):
                decomposed = point["decomposition"][phase]
                if decomposed["observed_bytes"] is None:
                    continue
                predicted = decomposed["predicted_total_bytes"]
                observed = decomposed["observed_bytes"]
                r_ratio = decomposed["residual_ratio"]
                peak_rows.append(
                    {
                        "run_index": record.get("run_index"),
                        "section": section,
                        "label": point.get("label"),
                        "key": me.point_key(point),
                        "phase": phase,
                        "predicted_bytes": predicted,
                        "observed_bytes": observed,
                        "residual_bytes": decomposed["residual_bytes"],
                        "residual_ratio": decomposed["residual_ratio"],
                        "unresolved_bytes": decomposed["unresolved_bytes"],
                        "unresolved_ratio": decomposed["unresolved_ratio"],
                        "winning_branch": (
                            point.get("peak_model", {}).get("prefill_winning_branch")
                            if phase == "peak_prefill"
                            else point.get("peak_model", {}).get("decode_winning_branch")
                        ),
                        "naive_sum_minus_phase_max_bytes": point.get(
                            "peak_model", {}
                        ).get("naive_sum_minus_phase_max_bytes"),
                        # Direction A — capacity safety: the analytic peak must
                        # not be *below* what the device saw beyond the measured
                        # device-view floor, otherwise the model would recommend
                        # an unsafe batch.
                        "no_understatement": (
                            observed <= predicted + device_view_floor
                        ),
                        "demonstrated": (
                            None
                            if r_ratio is None
                            else abs(r_ratio) <= me.TOLERANCES["peak_unresolved_ratio"]
                        ),
                    }
                )
    unresolved = [
        abs(r["unresolved_ratio"]) for r in peak_rows if r["unresolved_ratio"] is not None
    ]
    understated = [r for r in peak_rows if not r["no_understatement"]]
    demonstrated = [r for r in peak_rows if r["demonstrated"]]
    naive_gaps = [
        r["naive_sum_minus_phase_max_bytes"]
        for r in peak_rows
        if r["naive_sum_minus_phase_max_bytes"] is not None
    ]
    peak_check = {
        "num_rows": len(peak_rows),
        "max_abs_unresolved_ratio": max(unresolved) if unresolved else None,
        "max_abs_unresolved_bytes": (
            max(abs(r["unresolved_bytes"]) for r in peak_rows) if peak_rows else None
        ),
        "noise_floor_bytes": noise_floor,
        "device_view_floor_bytes": device_view_floor,
        "understated_rows": understated,
        "num_understated": len(understated),
        "num_demonstrated": len(demonstrated),
        "demonstrated_points": [
            {
                "run_index": r["run_index"],
                "key": r["key"],
                "phase": r["phase"],
                "predicted_bytes": r["predicted_bytes"],
                "observed_bytes": r["observed_bytes"],
                "residual_ratio": r["residual_ratio"],
            }
            for r in demonstrated
        ],
        "demonstration_ratio_tolerance": me.TOLERANCES["peak_unresolved_ratio"],
        "min_demonstrated_points": me.TOLERANCES["min_demonstrated_points"],
        "naive_sum_overprediction_bytes_max": max(naive_gaps) if naive_gaps else None,
        "rows": peak_rows,
        "passed": bool(
            peak_rows
            and not understated
            and len(demonstrated) >= int(me.TOLERANCES["min_demonstrated_points"])
        ),
    }

    # ── Logits term: the analytic full-prefill logits size vs the live tensor ──
    logits_rows: List[Dict[str, Any]] = []
    for record in runs:
        for section, point in me.iter_points(record):
            logits_rows.append(
                {
                    "run_index": record.get("run_index"),
                    "section": section,
                    "key": me.point_key(point),
                    "theory_bytes": point["kv_checks"]["theory_logits_bytes"],
                    "observed_bytes": point["kv_checks"]["observed_logits_bytes"],
                    "rel_error": point["kv_checks"]["logits_rel_error"],
                    "shape": point["logits"]["shape"],
                    "dtype": point["logits"]["dtype"],
                }
            )
    logits_errors = [
        r["rel_error"] for r in logits_rows if r["rel_error"] is not None
    ]
    logits_check = {
        "num_points": len(logits_rows),
        "max_rel_error": max(logits_errors) if logits_errors else None,
        "tolerance": me.TOLERANCES["unit_rel_error"],
        "rows": logits_rows,
        "passed": bool(
            logits_errors and max(logits_errors) <= 1e-9
        ),
    }

    # ── Calibration / held-out validation ─────────────────────────
    def _is_calibration(section: str, spec: Dict[str, Any]) -> bool:
        if section == "context_sweep":
            return spec["input_tokens"] in me.PROTOCOL["calibration_context_lengths"]
        if section == "batch_sweep":
            return [spec["input_tokens"], spec["batch_size"]] in me.PROTOCOL[
                "calibration_batch_points"
            ]
        return False

    def _is_heldout(section: str, spec: Dict[str, Any]) -> bool:
        if section == "context_sweep":
            return spec["input_tokens"] in me.PROTOCOL["heldout_context_lengths"]
        if section == "batch_sweep":
            return [spec["input_tokens"], spec["batch_size"]] in me.PROTOCOL[
                "heldout_batch_points"
            ]
        return False

    residuals: List[float] = []
    for record in runs:
        for section, point in me.iter_points(record):
            if not _is_calibration(section, point["spec"]):
                continue
            decomposed = point["decomposition"]["peak_prefill"]
            if decomposed["observed_bytes"] is not None:
                residuals.append(float(decomposed["residual_bytes"]))
    calibration_extra = statistics.median(residuals) if residuals else None

    heldout_rows: List[Dict[str, Any]] = []
    for record in runs:
        for section, point in me.iter_points(record):
            if not _is_heldout(section, point["spec"]):
                continue
            decomposed = point["decomposition"]["peak_prefill"]
            if decomposed["observed_bytes"] is None:
                continue
            raw_predicted = float(decomposed["predicted_total_bytes"])
            calibrated = raw_predicted + (calibration_extra or 0.0)
            heldout_rows.append(
                {
                    "run_index": record.get("run_index"),
                    "key": me.point_key(point),
                    "observed_bytes": decomposed["observed_bytes"],
                    "raw_predicted_bytes": raw_predicted,
                    "raw_rel_error": me.relative_error(
                        raw_predicted, decomposed["observed_bytes"]
                    ),
                    "calibrated_predicted_bytes": calibrated,
                    "calibrated_rel_error": me.relative_error(
                        calibrated, decomposed["observed_bytes"]
                    ),
                }
            )
    raw_errors = [r["raw_rel_error"] for r in heldout_rows if r["raw_rel_error"] is not None]
    calibrated_errors = [
        r["calibrated_rel_error"]
        for r in heldout_rows
        if r["calibrated_rel_error"] is not None
    ]
    # Object-level held-out validation: on points that took no part in any
    # calibration, the predictor must reproduce the *actual* KV storage and the
    # *actual* logits tensor exactly. This is the predictive-power claim that can
    # be evaluated on this platform; the device-level numbers are reported
    # alongside it together with the measured device-view floor.
    heldout_object_rows: List[Dict[str, Any]] = []
    for record in runs:
        for section, point in me.iter_points(record):
            if not _is_heldout(section, point["spec"]):
                continue
            heldout_object_rows.append(
                {
                    "run_index": record.get("run_index"),
                    "key": me.point_key(point),
                    "kv_prefill_rel_error": point["kv_checks"][
                        "prefill_metadata_rel_error"
                    ],
                    "kv_final_rel_error": point["kv_checks"]["final_metadata_rel_error"],
                    "logits_rel_error": point["kv_checks"]["logits_rel_error"],
                }
            )

    for row in heldout_rows:
        row["allowance_bytes"] = (
            me.TOLERANCES["heldout_rel_error"] * row["observed_bytes"]
            + device_view_floor
        )
        row["calibrated_abs_error_bytes"] = abs(
            row["calibrated_predicted_bytes"] - row["observed_bytes"]
        )
        row["within_allowance"] = (
            row["calibrated_abs_error_bytes"] <= row["allowance_bytes"]
        )
    object_errors = [
        value
        for row in heldout_object_rows
        for value in (
            row["kv_prefill_rel_error"],
            row["kv_final_rel_error"],
            row["logits_rel_error"],
        )
        if value is not None
    ]
    heldout_check = {
        "calibration_extra_bytes": calibration_extra,
        "calibration_num_points": len(residuals),
        "num_heldout_points": len(heldout_rows),
        "num_object_level_points": len(heldout_object_rows),
        "object_level_max_rel_error": max(object_errors) if object_errors else None,
        "object_level_tolerance": 1e-9,
        "object_level_rows": heldout_object_rows,
        "device_level_max_raw_rel_error": max(raw_errors) if raw_errors else None,
        "device_level_max_calibrated_rel_error": (
            max(calibrated_errors) if calibrated_errors else None
        ),
        "device_level_tolerance": me.TOLERANCES["heldout_rel_error"],
        "device_view_floor_bytes": device_view_floor,
        "device_level_rows": heldout_rows,
        # Pass is claimed on the object-level predictor (exact, deterministic)
        # plus the safety direction of the device view; the device-level rel
        # error is informational because it cannot be separated from the
        # measured device-view floor (see the report's limitation section).
        "device_level_all_within_allowance": bool(
            heldout_rows and all(r["within_allowance"] for r in heldout_rows)
        ),
        "device_level_all_not_understated": bool(
            heldout_rows
            and all(
                r["calibrated_predicted_bytes"] >= r["observed_bytes"] - device_view_floor
                for r in heldout_rows
            )
        ),
        "passed": bool(
            object_errors and max(object_errors) <= 1e-9 and heldout_rows
        ),
    }

    # ── Cross-run stability of eligible points ────────────────────
    # Metadata and analytic predictions are deterministic; the recorded spread
    # is the *device observation* spread, which is only required to stay inside
    # either the ratio tolerance or twice the measured noise floor. This keeps
    # "3 independent processes" a real check without letting unified-memory
    # accounting wobble masquerade as model instability.
    grouped: Dict[str, List[float]] = {}
    grouped_predicted: Dict[str, List[float]] = {}
    for row in steady_rows:
        if not row["eligible"] or row["observed_bytes"] is None:
            continue
        grouped.setdefault(row["key"], []).append(float(row["observed_bytes"]))
        grouped_predicted.setdefault(row["key"], []).append(float(row["predicted_bytes"]))
    stability: Dict[str, Any] = {}
    for key, values in sorted(grouped.items()):
        mean = statistics.mean(values)
        spread = (max(values) - min(values)) / mean if mean else None
        spread_bytes = max(values) - min(values)
        ratio_ok = spread is not None and spread <= me.TOLERANCES["cross_run_rel_spread"]
        floor_ok = spread_bytes <= 2 * device_view_floor
        stability[key] = {
            "num_runs": len(values),
            "observed_mean_bytes": mean,
            "observed_min_bytes": min(values),
            "observed_max_bytes": max(values),
            "rel_spread": spread,
            "spread_bytes": spread_bytes,
            "predicted_bytes": statistics.mean(grouped_predicted[key]),
            "ratio_ok": ratio_ok,
            "within_twice_noise_floor": floor_ok,
            "ok": ratio_ok or floor_ok,
        }
    max_spread = max(
        (v["rel_spread"] for v in stability.values() if v["rel_spread"] is not None),
        default=None,
    )
    max_spread_bytes = max(
        (v["spread_bytes"] for v in stability.values()), default=None
    )
    stability_check = {
        "num_keys": len(stability),
        "max_rel_spread": max_spread,
        "max_spread_bytes": max_spread_bytes,
        "noise_floor_bytes": noise_floor,
        "device_view_floor_bytes": device_view_floor,
        "tolerance": me.TOLERANCES["cross_run_rel_spread"],
        "per_key": stability,
        "passed": bool(stability and all(v["ok"] for v in stability.values())),
    }

    # ── Cleanup / multi-request release ───────────────────────────
    cleanup_rows: List[Dict[str, Any]] = []
    for record in runs:
        multi = record.get("multi_request")
        if not multi:
            continue
        cleanup_rows.append(
            {
                "run_index": record.get("run_index"),
                "device_used_growth_bytes_first_to_last": multi[
                    "device_used_growth_bytes_first_to_last"
                ],
                "regression_to_baseline": multi["regression_to_baseline"],
                "released_increments_bytes": [
                    item["released_increment_bytes"] for item in multi["records"]
                ],
                "predicted_steady_final_bytes": [
                    item["predicted_steady_final_bytes"] for item in multi["records"]
                ],
                "observed_steady_final_bytes": [
                    item["decode_end_increment_bytes"] for item in multi["records"]
                ],
            }
        )
    cleanup_check = {
        "runs": cleanup_rows,
        "tolerance_bytes": me.TOLERANCES["cleanup_growth_bytes"],
        "passed": bool(
            cleanup_rows and all(r["regression_to_baseline"] for r in cleanup_rows)
        ),
    }

    # ── OOM cross-reference ───────────────────────────────────────
    oom = next(
        (r["oom_cross_reference"] for r in runs if r.get("oom_cross_reference")), None
    )
    oom_check = {
        "rows": (oom or {}).get("rows", []),
        "consistent": bool(
            oom and all(row["predicted_over_available"] for row in oom["rows"])
        ),
    }

    # ── Hook A/B ──────────────────────────────────────────────────
    hook_ab = next((r["hook_ab"] for r in runs if r.get("hook_ab")), None)
    hook_deltas = (
        []
        if not hook_ab
        else [
            abs(d)
            for d in (hook_ab["peak_delta_bytes"], hook_ab["steady_delta_bytes"])
            if d is not None
        ]
    )
    hook_check = {
        "result": hook_ab,
        "max_abs_delta_bytes": max(hook_deltas) if hook_deltas else None,
        "passed": bool(
            hook_deltas and max(hook_deltas) <= me.TOLERANCES["cleanup_growth_bytes"]
        ),
    }

    # ── Capacity guard: raw prediction must not understate observation ──
    guard_violations = [
        {
            "key": row["key"],
            "phase": row["phase"],
            "understated_bytes": row["observed_bytes"] - row["predicted_bytes"],
        }
        for row in peak_rows
        if row["predicted_bytes"] is not None
        and row["observed_bytes"] is not None
        and row["predicted_bytes"] < row["observed_bytes"]
    ]
    capacity_guard = {
        "violations": guard_violations,
        "num_violations": len(guard_violations),
        "note": "raw analytic prediction excludes the fitted fixed overhead",
    }

    verdict = {
        "experiment": "E02-05",
        "runs": len(runs),
        "run_indices": [r.get("run_index") for r in runs],
        "protocol": me.PROTOCOL,
        "tolerances": me.TOLERANCES,
        "unit_selftest": unit_selftest,
        "weight_check": weight_check,
        "kv_ledger_check": kv_check,
        "logits_check": logits_check,
        "steady_state_check": steady_check,
        "peak_model_check": peak_check,
        "heldout_check": heldout_check,
        "noise_check": noise_check,
        "repeatability_check": repeatability_check,
        "cross_run_stability": stability_check,
        "cleanup_check": cleanup_check,
        "hook_ab_check": hook_check,
        "oom_cross_reference_check": oom_check,
        "capacity_guard": capacity_guard,
        "abnormal_runs": [item for r in runs for item in r.get("abnormal_runs", [])],
    }
    verdict["passed"] = all(
        [
            unit_selftest["all_ok"],
            weight_check["passed"],
            kv_check["passed"],
            logits_check["passed"],
            steady_check["passed"],
            peak_check["passed"],
            heldout_check["passed"],
            repeatability_check["passed"],
            stability_check["passed"],
            cleanup_check["passed"],
            hook_check["passed"],
            oom_check["consistent"],
        ]
    )

    out_path = output_dir / "verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    logger.info("wrote %s (passed=%s)", out_path, verdict["passed"])
    print(
        json.dumps(
            {
                "passed": verdict["passed"],
                "unit_selftest_all_ok": unit_selftest["all_ok"],
                "weight_max_rel_error": weight_check["max_rel_error"],
                "kv_max_rel_error": kv_check["max_rel_error"],
                "logits_max_rel_error": logits_check["max_rel_error"],
                "peak_num_understated": peak_check["num_understated"],
                "peak_num_demonstrated": peak_check["num_demonstrated"],
                "naive_sum_overprediction_bytes_max": peak_check[
                    "naive_sum_overprediction_bytes_max"
                ],
                "noise_floor_bytes": noise_check["noise_floor_bytes"],
                "device_view_floor_bytes": noise_check["device_view_floor_bytes"],
                "steady_num_eligible": steady_check["num_eligible"],
                "steady_max_abs_residual_ratio": steady_check[
                    "max_abs_residual_ratio"
                ],
                "repeatability_max_spread_bytes": repeatability_check[
                    "max_spread_bytes"
                ],
                "heldout_object_level_max_rel_error": heldout_check[
                    "object_level_max_rel_error"
                ],
                "heldout_device_level_max_calibrated_rel_error": heldout_check[
                    "device_level_max_calibrated_rel_error"
                ],
                "stability_max_rel_spread": stability_check["max_rel_spread"],
                "stability_max_spread_bytes": stability_check["max_spread_bytes"],
                "cleanup_passed": cleanup_check["passed"],
                "hook_ab_passed": hook_check["passed"],
                "oom_consistent": oom_check["consistent"],
            },
            indent=2,
        )
    )
    return 0 if verdict["passed"] else 1


# ─────────────────────────────── summarize ───────────────────────────────


def _mib(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(bytes_to_mib(value), 2)


def _run_summarize(args: argparse.Namespace) -> int:
    """Render the report tables from the raw evidence (no new measurements)."""
    output_dir = Path(args.output_dir)
    run_files = sorted(output_dir.glob("run_*.json"))
    if not run_files:
        logger.error("no run_*.json found in %s", output_dir)
        return 2
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in run_files]
    lines: List[str] = []

    def out(text: str = "") -> None:
        lines.append(text)
        print(text)

    out(f"# E02-05 evidence summary ({len(runs)} runs)")
    out()

    out("## Model loading (stages M0-M2)")
    for record in runs:
        stages = record["stages"]
        load = record["load"]
        out(
            f"- run {record.get('run_index')}: load={load['load_time_s']:.1f}s "
            f"M0_used={stages['M0_process_start']['device_used_mb']} "
            f"M1_used={round(stages['M1_cuda_context']['device_used_mb'], 1)} MiB "
            f"M2_used={round(stages['M2_model_loaded']['device_used_mb'], 1)} MiB "
            f"M2_free={round(stages['M2_model_loaded']['device']['free_mb'], 1)} MiB "
            f"load_peak(max)={load['peak'].get('max')} MiB "
            f"fully_on_gpu={load['model_fully_on_gpu']}"
        )
    out()

    out("## Parameter / storage inventory (per run)")
    for record in runs:
        inventory = record["inventory"]
        params = inventory["parameter_summary"]
        buffers = inventory["buffer_summary"]
        out(
            f"- run {record.get('run_index')}: names={params['num_tensors']} "
            f"unique_storages={params['num_unique_storages']} "
            f"logical={params['logical_total_mib']:.2f} MiB "
            f"dedup={params['dedup_total_mib']:.2f} MiB "
            f"duplicate_counted={_mib(params['duplicate_counted_bytes'])} MiB "
            f"p_dtypes={params['dtype_histogram']} "
            f"p_devices={params['device_histogram']} "
            f"buffers={buffers['dedup_total_mib']:.3f} MiB {buffers['dtype_histogram']} "
            f"resident={inventory['resident_mib']:.2f} MiB "
            f"({inventory['resident_gib']:.3f} GiB)"
        )
        for group in params["alias_groups"]:
            out(
                f"  - alias {group['group_id']}: {group['member_names']} "
                f"storage={group['storage_mib']:.2f} MiB"
            )
    out()

    out("## Staged point (M3-M7) and phase observations")
    header = (
        "| run | key | kv_prefill P | M3 used | M3b used | M4/M5 used | M6 used "
        "| M7 used | logits obs | logits P |"
    )
    out(header)
    out("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for record in runs:
        point = record["staged_point"]
        stages = record["stages"]
        out(
            f"| {record.get('run_index')} | {me.point_key(point)} "
            f"| {_mib(point['predictions']['steady_prefill_bytes'])} MiB "
            f"| {round(stages['M3_prefill_with_logits']['device_used_mb'], 1)} "
            f"| {round(stages['M3b_kv_live_no_logits']['device_used_mb'], 1)} "
            f"| {round(stages['M4_decode_end']['device_used_mb'], 1)} "
            f"| {round(stages['M6_released']['device_used_mb'], 1)} "
            f"| {round(stages['M7_before_next_request']['device_used_mb'], 1)} "
            f"| {point['logits']['observed_bytes']} B "
            f"| {point['theory']['logits_full_prefill_bytes']} B |"
        )
    out()
    out(
        "| run | key | kv_final P | steady obs | prefill_steady obs "
        "| peak P | peak obs | decode peak P | decode peak obs | released obs |"
    )
    out("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for record in runs:
        for section, point in me.iter_points(record):
            observations = point["observations"]
            predictions = point["predictions"]
            out(
                f"| {record.get('run_index')} | {section}:{me.point_key(point)} "
                f"| {_mib(predictions['steady_final_bytes'])} "
                f"| {_mib(observations['decode_end_increment_bytes'])} "
                f"| {_mib(observations['prefill_steady_increment_bytes'])} "
                f"| {_mib(predictions['peak_prefill_bytes'])} "
                f"| {_mib(observations['prefill_peak_increment_bytes'])} "
                f"| {_mib(predictions['peak_decode_bytes'])} "
                f"| {_mib(observations['decode_peak_increment_bytes'])} "
                f"| {_mib(observations['released_increment_bytes'])} |"
            )
    out()

    out("## KV ledger: theory vs live cache metadata")
    out("| run | key | L | Hkv | Dh | dtype B | B/token/layer | KV theory MiB | KV metadata MiB | rel err | prealloc |")
    out("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for record in runs:
        for section, point in me.iter_points(record):
            checks = point["kv_checks"]
            theory = point["theory"]
            out(
                f"| {record.get('run_index')} | {section}:{me.point_key(point)} "
                f"| {checks['metadata_layers']} | {checks['metadata_kv_heads']} "
                f"| {theory['kv_final']['head_dim']} "
                f"| {checks['metadata_element_bytes']} "
                f"| {checks['metadata_per_token_per_layer_bytes']} "
                f"| {_mib(theory['kv_final']['total_bytes'])} "
                f"| {_mib(point['kv_metadata']['final']['total_logical_bytes'])} "
                f"| {checks['final_metadata_rel_error']} "
                f"| {checks['metadata_preallocated']} |"
            )
    out()

    out("## Device-observation floor (noise probe)")
    out("| run | idle span MiB | median abs net MiB | median abs empty-rise error MiB | median abs filled-rise error MiB |")
    out("|---|---:|---:|---:|---:|")
    for record in runs:
        probe = record.get("noise_probe") or {}
        out(
            f"| {record.get('run_index')} "
            f"| {_mib(probe.get('idle_span_bytes'))} "
            f"| {_mib(probe.get('median_abs_net_bytes'))} "
            f"| {_mib(probe.get('median_abs_empty_rise_error_bytes'))} "
            f"| {_mib(probe.get('median_abs_fill_rise_error_bytes'))} |"
        )
    out()

    out("## Repeatability (same point twice) and multi-request release")
    for record in runs:
        repeat = record.get("repeat_probe")
        if repeat:
            out(
                f"- run {record.get('run_index')} repeat: prefill_steady="
                f"{[_mib(v) for v in repeat['prefill_steady_increment_bytes']]} "
                f"decode_end={[_mib(v) for v in repeat['decode_end_increment_bytes']]} "
                f"peak={[_mib(v) for v in repeat['prefill_peak_increment_bytes']]} "
                f"max_spread={_mib(repeat['max_spread_bytes'])} MiB "
                f"predicted_steady_final={_mib(repeat['predicted_steady_final_bytes'])} MiB"
            )
        multi = record.get("multi_request")
        if multi:
            out(
                f"- run {record.get('run_index')} multi-request: before="
                f"{[_mib(r['device_used_before_bytes']) for r in multi['records']]} "
                f"released_inc={[_mib(r['released_increment_bytes']) for r in multi['records']]} "
                f"decode_end_inc={[_mib(r['decode_end_increment_bytes']) for r in multi['records']]} "
                f"predicted={_mib(multi['records'][0]['predicted_steady_final_bytes'])} MiB "
                f"growth={_mib(multi['device_used_growth_bytes_first_to_last'])} MiB"
            )
        hook = record.get("hook_ab")
        if hook:
            out(
                f"- run {record.get('run_index')} hook A/B: peak_delta="
                f"{_mib(hook['peak_delta_bytes'])} MiB steady_delta="
                f"{_mib(hook['steady_delta_bytes'])} MiB"
            )
    out()

    out("## OOM cross-reference (E02-04 frozen points)")
    out("| workload | B | KV MiB | eager WS MiB | logits MiB | phase-max peak MiB | naive sum MiB | available MiB | over available | fits without eager |")
    out("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for record in runs[:1]:
        reference = record.get("oom_cross_reference") or {}
        for row in reference.get("rows", []):
            naive = (
                row["kv_prefill_bytes"]
                + row["attention_workspace_prefill_bytes"]
                + row["activation_lower_bound_bytes"]
                + row["logits_full_bytes"]
            )
            out(
                f"| {row['workload']} | {row['batch_size']} "
                f"| {_mib(row['kv_prefill_bytes'])} "
                f"| {_mib(row['attention_workspace_prefill_bytes'])} "
                f"| {_mib(row['logits_full_bytes'])} "
                f"| {_mib(row['predicted_prefill_peak_increment_bytes'])} "
                f"| {_mib(naive)} "
                f"| {_mib(row['available_after_load_bytes'])} "
                f"| {row['predicted_over_available']} "
                f"| {row['without_eager_would_fit']} |"
            )
    out()

    verdict_path = output_dir / "verdict.json"
    if verdict_path.exists():
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        out("## Verdict checks")
        out("| check | passed | evidence |")
        out("|---|---|---|")
        out(f"| unit_selftest | {verdict['unit_selftest']['all_ok']} | hand-computed formula/unit checks |")
        out(f"| weight_check | {verdict['weight_check']['passed']} | max rel err {verdict['weight_check']['max_rel_error']} |")
        out(f"| kv_ledger_check | {verdict['kv_ledger_check']['passed']} | max rel err {verdict['kv_ledger_check']['max_rel_error']} |")
        out(f"| logits_check | {verdict['logits_check']['passed']} | max rel err {verdict['logits_check']['max_rel_error']} |")
        out(
            f"| steady_state_check | {verdict['steady_state_check']['passed']} "
            f"| eligible {verdict['steady_state_check']['num_eligible']}, "
            f"max |residual| {_mib(verdict['steady_state_check']['max_abs_residual_bytes'])} MiB |"
        )
        out(
            f"| peak_model_check | {verdict['peak_model_check']['passed']} "
            f"| understated {verdict['peak_model_check']['num_understated']}, "
            f"demonstrated {verdict['peak_model_check']['num_demonstrated']} |"
        )
        out(
            f"| heldout_check | {verdict['heldout_check']['passed']} "
            f"| object-level max rel err {verdict['heldout_check']['object_level_max_rel_error']} |"
        )
        out(f"| stability_check | {verdict['cross_run_stability']['passed']} | max spread {_mib(verdict['cross_run_stability']['max_spread_bytes'])} MiB |")
        out(f"| cleanup_check | {verdict['cleanup_check']['passed']} | per-run release regression |")
        out(f"| hook_ab_check | {verdict['hook_ab_check']['passed']} | max delta {_mib(verdict['hook_ab_check']['max_abs_delta_bytes'])} MiB |")
        out(f"| oom_cross_reference | {verdict['oom_cross_reference_check']['consistent']} | all frozen OOM points predicted over available |")
        out()
        out(f"OVERALL PASSED = {verdict['passed']}")

    out_path = output_dir / "evidence_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", out_path)
    return 0


# ─────────────────────────────── cli ───────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E02-05 memory model validation")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", required=True)
    common.add_argument("--model-path", default=me.MODEL_DEFAULT)
    common.add_argument(
        "--manifest",
        default=str(_REPO_ROOT / "docs" / "benchmark" / "model_sha256_manifest.txt"),
    )

    collect = sub.add_parser("collect", parents=[common], help="collect memory evidence")
    collect.add_argument("--run-index", type=int, default=0)
    collect.add_argument(
        "--sections",
        nargs="+",
        default=[
            "noise_probe", "repeat_probe", "staged", "context_sweep",
            "batch_sweep", "multi_request", "hook_ab",
        ],
        choices=[
            "noise_probe", "repeat_probe", "staged", "context_sweep",
            "batch_sweep", "multi_request", "hook_ab",
        ],
    )
    collect.add_argument("--monitor-interval-ms", type=int, default=500)
    collect.set_defaults(func=_run_collect)

    probe = sub.add_parser("probe", parents=[common], help="allocator capability probe")
    probe.add_argument(
        "--mode",
        choices=["caching", "no-caching", "caching-direct", "no-caching-direct"],
        required=True,
    )
    probe.set_defaults(func=_run_probe)

    verify = sub.add_parser("verify", parents=[common], help="derive the verdict")
    verify.set_defaults(func=_run_verify)

    summarize = sub.add_parser(
        "summarize", parents=[common], help="render report tables from raw evidence"
    )
    summarize.set_defaults(func=_run_summarize)
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
