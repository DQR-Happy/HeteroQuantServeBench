#!/usr/bin/env python3
"""E02-04 runner: static batch scan, throughput gain and safe capacity boundary.

Proves the three things required by ``S02_实验清单.md`` E02-04 (see
``docs/stage_experiments/details/S02/E02-04_batch_and_capacity_boundary.md``):

1. **Static batch scan** — for each frozen workload (I/G fixed), sweep
   ``B = 1, 2, 4, 8, ...`` up to a pre-registered ceiling or the first
   failure, feeding a *real* ``[B, I]`` token tensor (never a broadcast view).
2. **Boundary + OOM not discarded** — every success and every failure (OOM
   stage / decode step / memory context / recovery state) is recorded; the
   largest successful batch, first failed batch and the safe batch are all
   derived from evidence, not asserted.
3. **Explainable throughput gain + memory growth** — batch throughput and the
   batch KV/weight memory accounting are reported side by side so the
   throughput/latency/memory trade-off is interpretable.

Usage:

    python scripts/audit/run_e02_04_batch_capacity.py collect \
        --output-dir docs/stage_experiments/S02/E02-04/raw
    python scripts/audit/run_e02_04_batch_capacity.py verify \
        --output-dir docs/stage_experiments/S02/E02-04/raw
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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from hqsb.benchmark.batch_core import (
    BatchBenchmarkOOM,
    batch_token_budgets,
    benchmark_model_core_batch,
    compute_batch_kv_cache_info,
)
from hqsb.benchmark.correctness import hash_token_sequence
from hqsb.benchmark.memory import (
    device_memory_snapshot,
    host_memory_snapshot,
    model_kv_cache_config,
    model_weight_bytes,
    process_rss_bytes,
    process_swap_bytes,
)
from hqsb.benchmark.model_core import benchmark_model_core
from hqsb.benchmark.resource_monitor import TegrastatsMonitor
from hqsb.benchmark.tegrastats_parser import (
    compute_power_summary,
    compute_resource_summary,
    extract_cpu_freqs_mhz,
    parse_tegrastats_line,
    slice_records,
)
from hqsb.benchmark.workload import make_fixed_token_input
from hqsb.benchmark.workload_config import load_workload_dicts
from hqsb.models.loader import load_qwen3

logger = logging.getLogger("e02_04")

_REPO_ROOT = Path(__file__).resolve().parents[2]

_WORKLOAD_YAML = _REPO_ROOT / "configs" / "benchmarks" / "jetson_qwen3_fp16.yaml"

# Pre-registered capacity scan protocol (E02-04 §5/§6/§7):
_BATCH_GEOMETRY = [1, 2, 4, 8, 16]

# Safety policy frozen *before* running (E02-04 §5: never tune the gate after
# seeing results). On unified memory the 3.2 GiB model itself consumes most of
# the reported "device free", so the pre-point floor is on *CUDA* free memory
# (the actual headroom for the next batch) rather than host MemAvailable,
# which is contaminated by the already-loaded weights. CUDA OOM is caught
# gracefully (see BatchBenchmarkOOM); these floors only stop a point *before*
# it would push the whole system into the OOM killer.
_SAFETY = {
    "max_batch": _BATCH_GEOMETRY[-1],
    "min_device_free_mb": 128.0,        # CUDA free floor before a point
    "thermal_suspect_c": 95.0,
    # A point is "safe" (E02-04 §5/§10) only when, after cleanup, the process
    # was NOT pushed into swap thrashing. The idle process swap baseline is
    # ~130-175 MiB; a run whose pages are evicted to swap (>= this many MiB)
    # shows memory pressure and is excluded from the recommended safe batch.
    "max_safe_process_swap_mb": 768.0,
}

_DEFAULT_WARMUP_OSL = 2
_DEFAULT_MONITOR_INTERVAL_MS = 500
_MIN_POWER_SAMPLES = 3


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


def _build_batched_inputs(
    tokenizer: Any,
    isl: int,
    batch_size: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Real ``[B, I]`` copies of the fixed input (never a broadcast view).

    E02-04 §7 forbids simulating independent request state via a tensor-view
    broadcast; ``.repeat`` materializes ``B`` independent rows.
    """
    single = make_fixed_token_input(tokenizer, isl, device=device)
    return {
        "input_ids": single["input_ids"].repeat(batch_size, 1),
        "attention_mask": single["attention_mask"].repeat(batch_size, 1),
    }


def _leading_dim(shape_str: str) -> Optional[int]:
    """Extract the leading (batch) dim from a rendered shape like ``[B, ...]``."""
    m = re.match(r"^\[(\d+),", shape_str)
    return int(m.group(1)) if m else None


def _verify_batch_propagation(
    model: Any,
    batched_inputs: Dict[str, torch.Tensor],
    batch_size: int,
) -> Dict[str, Any]:
    """E02-04 §7 step 1: confirm the batch dimension actually enters the model.

    Attaches the E02-02 runtime hook for a single prefill and checks that the
    embedding output and layer-0 projection input carry the requested batch
    dimension (not a serialized B=1 loop in disguise).
    """
    from hqsb.benchmark.shape_census import ShapeCensusCollector

    collector = ShapeCensusCollector()
    collector.attach(model)
    collector.set_phase("prefill")
    with torch.inference_mode():
        model(
            input_ids=batched_inputs["input_ids"],
            attention_mask=batched_inputs["attention_mask"],
            use_cache=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    collector.detach()

    embed = next(
        (r for r in collector.records()
         if r["module"] == "model.embed_tokens" and r["phase"] == "prefill"),
        None,
    )
    q_proj = next(
        (r for r in collector.records()
         if r["module"] == "model.layers.0.self_attn.q_proj" and r["phase"] == "prefill"),
        None,
    )
    embed_shapes = embed["output_shapes"] if embed else []
    q_proj_shapes = q_proj["input_shapes"] if q_proj else []

    embed_dim = _leading_dim(embed_shapes[0]) if embed_shapes else None
    q_proj_dim = _leading_dim(q_proj_shapes[0]) if q_proj_shapes else None

    return {
        "embed_tokens_output_shapes": embed_shapes[:4],
        "layer0_q_proj_input_shapes": q_proj_shapes[:4],
        "batch_dim_embed": embed_dim,
        "batch_dim_layer0": q_proj_dim,
        "batch_dim_confirmed": (
            embed_dim == batch_size and q_proj_dim == batch_size
        ),
    }


def _memory_prediction(model: Any, batch_size: int, isl: int, osl: int) -> Dict[str, Any]:
    """E02-04 §7 step 3: initial memory model for safe-candidate selection.

    Weights are not replicated by B; KV scales with ``B``. Prefill
    activation / attention workspace are *not* analytically predicted here
    (that is E02-05), so the prediction is a lower bound, not a substitute for
    the real run.
    """
    config = model_kv_cache_config(model)
    kv: Dict[str, Any] = {}
    if {"num_layers", "num_kv_heads", "head_dim"}.issubset(config):
        try:
            kv = compute_batch_kv_cache_info(
                num_layers=config["num_layers"],
                num_kv_heads=config["num_kv_heads"],
                head_dim=config["head_dim"],
                batch_size=batch_size,
                context_length=isl + osl - 1,
                dtype_bytes=2,
            )
        except ValueError:
            kv = {}
    weight_bytes = model_weight_bytes(model)
    return {
        "weight_bytes": weight_bytes,
        "weight_mb": weight_bytes / (1024**2),
        "kv_final_bytes": kv.get("total_bytes", 0),
        "kv_final_mb": kv.get("total_bytes", 0) / (1024**2),
        "kv_per_token_bytes": kv.get("per_token_bytes", 0),
        "token_budgets": batch_token_budgets(batch_size, isl, osl),
        "note": "prefill activation/attention workspace measured empirically (E02-05)",
    }


def _memory_context() -> Dict[str, Any]:
    """Snapshot device/host/process memory for failure & recovery evidence."""
    return {
        "cuda_allocated_mb": (
            torch.cuda.memory_allocated() / (1024**2)
            if torch.cuda.is_available() else 0.0
        ),
        "cuda_reserved_mb": (
            torch.cuda.memory_reserved() / (1024**2)
            if torch.cuda.is_available() else 0.0
        ),
        "peak_allocated_mb": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if torch.cuda.is_available() else 0.0
        ),
        "peak_reserved_mb": (
            torch.cuda.max_memory_reserved() / (1024**2)
            if torch.cuda.is_available() else 0.0
        ),
        "device": device_memory_snapshot(),
        "host": host_memory_snapshot(),
        "process_rss_bytes": process_rss_bytes(),
        "process_swap_bytes": process_swap_bytes(),
    }


def _telemetry_window_summary(
    monitor_records: List[Dict[str, Any]],
    begin_ns: int,
    end_ns: int,
) -> Dict[str, Any]:
    window = slice_records(monitor_records, begin_ns, end_ns)
    parsed: List[Dict[str, Any]] = []
    cpu_freqs: List[int] = []
    for record in window:
        fields = parse_tegrastats_line(record["raw"])
        fields["time_ns"] = record["time_ns"]
        parsed.append(fields)
        cpu_freqs.extend(extract_cpu_freqs_mhz(record["raw"]))

    power = compute_power_summary(parsed)
    resource = compute_resource_summary(parsed)
    return {
        "window_begin_ns": begin_ns,
        "window_end_ns": end_ns,
        "num_records": len(window),
        "num_power_samples": power["num_samples"],
        "avg_power_w": power["avg_power_w"],
        "peak_power_w": power["peak_power_w"],
        "energy_j": power["energy_j"],
        "avg_gpu_temp_c": resource.get("avg_gpu_temp_c"),
        "peak_gpu_temp_c": resource.get("peak_gpu_temp_c"),
        "avg_cpu_temp_c": resource.get("avg_cpu_temp_c"),
        "peak_cpu_temp_c": resource.get("peak_cpu_temp_c"),
        "avg_gpu_util_pct": resource.get("avg_gpu_util_pct"),
        "peak_gpu_util_pct": resource.get("peak_gpu_util_pct"),
        "avg_ram_used_mb": resource.get("avg_ram_used_mb"),
        "peak_ram_used_mb": resource.get("peak_ram_used_mb"),
        "cpu_freqs_mhz": sorted(set(cpu_freqs)),
        "energy_usable": power["num_samples"] >= _MIN_POWER_SAMPLES,
    }


def _point_checks(
    result: Dict[str, Any],
    isl: int,
    osl: int,
    batch_size: int,
) -> Dict[str, Any]:
    rows = result["generated_token_ids"]
    hashes = [hash_token_sequence(r) for r in rows]
    distinct = len(set(hashes))
    checks = {
        "batch_size_exact": result["batch_size"] == batch_size,
        "rows_present": len(rows) == batch_size,
        "input_tokens_exact": result["input_tokens"] == isl,
        "output_tokens_exact_all_rows": all(len(r) == osl for r in rows),
        "decode_steps_exact": len(result["raw_itl_ms"]) == osl - 1,
        "distinct_row_hashes": distinct,
        "all_rows_identical": distinct == 1,
    }
    checks["correctness_ok"] = all(
        checks[k] for k in (
            "batch_size_exact",
            "rows_present",
            "input_tokens_exact",
            "output_tokens_exact_all_rows",
            "decode_steps_exact",
            "all_rows_identical",
        )
    )
    return checks


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

    param_devices = sorted({str(p.device) for p in model.parameters()})
    model_fully_on_gpu = bool(param_devices) and all(
        d.startswith("cuda") for d in param_devices
    )

    workloads = load_workload_dicts(str(args.workload_yaml))
    if args.workloads:
        wanted = {w.strip() for w in args.workloads.split(",") if w.strip()}
        workloads = [w for w in workloads if w["name"] in wanted]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model-level warmup (B=1) so lazy init / cuBLAS workspace does not leak
    # into the first capacity point, then release any reserved-but-unused
    # allocator blocks back to the OS (unified memory: reserved == physical).
    warm_inputs = make_fixed_token_input(tokenizer, 32, device=device)
    _ = benchmark_model_core(model, warm_inputs, 2)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Background telemetry (tegrastats) for temperature/power + thermal-suspect.
    monitor: Optional[TegrastatsMonitor] = None
    monitor_records: List[Dict[str, Any]] = []
    monitor_available = False
    try:
        monitor = TegrastatsMonitor(interval_ms=args.monitor_interval_ms)
        monitor.start()
        monitor_available = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("tegrastats unavailable: %s", exc)
        monitor = None

    batch_geometry = [b for b in args.batchs if b <= _SAFETY["max_batch"]]
    workload_records: List[Dict[str, Any]] = []

    run_id = (
        f"run_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    )
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out_path = output_dir / f"run_{args.run_index}.json"
    telemetry_path = output_dir / f"run_{args.run_index}.tegrastats.txt"

    def _emit(workloads_list: List[Dict[str, Any]]) -> None:
        """Incrementally persist collected evidence so a system OOM-kill mid-scan
        (SIGKILL, uncatchable) never discards already-finished capacity points."""
        record: Dict[str, Any] = {
            "run_id": run_id,
            "run_index": args.run_index,
            "started_at": started_at,
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "environment": _environment(),
            "model_path": str(Path(args.model_path).expanduser().resolve()),
            "manifest_sha256": _manifest_hash(args.manifest),
            "workload_yaml": str(args.workload_yaml),
            "load_time_s": load_time_s,
            "actual_device": {
                "param_devices": param_devices,
                "model_fully_on_gpu": model_fully_on_gpu,
            },
            "protocol": {
                "batch_geometry": batch_geometry,
                "input_mode": "duplicate_fixed_token_input",
                "warmup_osl": _DEFAULT_WARMUP_OSL,
                "attention_backend": "eager",
                "dtype": "float16",
                "clock_domain": "host_monotonic_ns + torch.cuda.synchronize",
                "monitor_interval_ms": args.monitor_interval_ms,
                "env_pytorch_no_cuda_memory_caching": (
                    os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", "")
                ),
            },
            "safety": _SAFETY,
            "monitor": {
                "available": monitor_available,
                "num_records": len(monitor_records),
                "raw_telemetry_file": str(telemetry_path),
            },
            "workloads": workloads_list,
        }
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    try:
        for spec in workloads:
            name = spec["name"]
            isl = int(spec["input_tokens"])
            osl = int(spec["output_tokens"])
            logger.info("Workload %s (ISL=%d OSL=%d)", name, isl, osl)

            point_records: List[Dict[str, Any]] = []
            batch_propagation: Optional[Dict[str, Any]] = None
            stop_reason: Optional[str] = None

            for b in batch_geometry:
                device_free_mb = (
                    torch.cuda.mem_get_info()[0] / (1024**2)
                    if torch.cuda.is_available() else 0.0
                )
                if device_free_mb < _SAFETY["min_device_free_mb"]:
                    stop_reason = (
                        f"safety_stop: device free {device_free_mb:.1f}MB "
                        f"< {_SAFETY['min_device_free_mb']}MB"
                    )
                    logger.warning("  %s B=%d: %s", name, b, stop_reason)
                    point_records.append(
                        {"batch_size": b, "status": "safety_stop",
                         "reason": stop_reason, "memory_before": _memory_context()}
                    )
                    break

                batched = _build_batched_inputs(tokenizer, isl, b, device)
                logger.info("  B=%d", b)

                # E02-04 §7 step 1: confirm batch enters the model (once, B=2).
                if b == 2:
                    batch_propagation = _verify_batch_propagation(model, batched, b)

                point: Dict[str, Any] = {
                    "batch_size": b,
                    "input_tokens": isl,
                    "output_tokens": osl,
                    "token_budgets": batch_token_budgets(b, isl, osl),
                    "prediction": _memory_prediction(model, b, isl, osl),
                    "memory_before": _memory_context(),
                }

                measure_begin_ns = time.monotonic_ns()

                # Warmup (allocator / workspace for this batch width).
                try:
                    benchmark_model_core_batch(
                        model, batched["input_ids"], batched["attention_mask"],
                        _DEFAULT_WARMUP_OSL,
                    )
                except BatchBenchmarkOOM as exc:
                    point.update(
                        status="failure",
                        failure={
                            "category": "cuda_oom",
                            "stage": exc.stage,
                            "step": exc.step,
                            "message": str(exc),
                            "where": "warmup",
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    point.update(
                        status="failure",
                        failure={
                            "category": type(exc).__name__,
                            "stage": None,
                            "step": None,
                            "message": str(exc),
                            "where": "warmup",
                        },
                    )

                # Measurement (full I/G) — only if warmup did not already fail.
                if "status" not in point:
                    try:
                        result = benchmark_model_core_batch(
                            model, batched["input_ids"], batched["attention_mask"],
                            osl,
                        )
                        point.update(
                            status="success",
                            result=result,
                            checks=_point_checks(result, isl, osl, b),
                        )
                    except BatchBenchmarkOOM as exc:
                        point.update(
                            status="failure",
                            failure={
                                "category": "cuda_oom",
                                "stage": exc.stage,
                                "step": exc.step,
                                "message": str(exc),
                                "where": "measurement",
                            },
                        )
                    except Exception as exc:  # noqa: BLE001
                        point.update(
                            status="failure",
                            failure={
                                "category": type(exc).__name__,
                                "stage": None,
                                "step": None,
                                "message": str(exc),
                                "where": "measurement",
                            },
                        )

                measure_end_ns = time.monotonic_ns()
                point["memory_after"] = _memory_context()

                if monitor_available and monitor is not None:
                    point["telemetry"] = _telemetry_window_summary(
                        monitor.records, measure_begin_ns, measure_end_ns
                    )
                    peak_temp = point["telemetry"].get("peak_gpu_temp_c")
                    if peak_temp is not None and peak_temp > _SAFETY["thermal_suspect_c"]:
                        point["thermal_suspect"] = True
                        point["thermal_suspect_reason"] = (
                            f"peak_gpu_temp_c={peak_temp:.1f}C > "
                            f"{_SAFETY['thermal_suspect_c']}C"
                        )
                    else:
                        point["thermal_suspect"] = False

                point_records.append(point)

                # Release KV / activation for the next point; recover allocator,
                # then snapshot the *recovered* free memory (the "safe" verdict
                # uses this post-cleanup headroom, E02-04 §5/§10).
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                point["memory_recovered"] = _memory_context()

                # Persist progress before moving on (survives an OOM-kill).
                current_workload: Dict[str, Any] = {
                    "name": name,
                    "input_tokens": isl,
                    "output_tokens": osl,
                    "batch_geometry_scanned": [p["batch_size"] for p in point_records],
                    "batch_propagation": batch_propagation,
                    "stop_reason": stop_reason,
                    "points": list(point_records),
                }
                _emit(workload_records + [current_workload])

                if point["status"] == "failure":
                    stop_reason = (
                        f"failure_at_B={b}: {point['failure']['category']} "
                        f"stage={point['failure']['stage']} step={point['failure']['step']}"
                    )
                    logger.warning("  %s: %s", name, stop_reason)
                    break

            workload_records.append(
                {
                    "name": name,
                    "input_tokens": isl,
                    "output_tokens": osl,
                    "batch_geometry_scanned": [p["batch_size"] for p in point_records],
                    "batch_propagation": batch_propagation,
                    "stop_reason": stop_reason,
                    "points": point_records,
                }
            )
    finally:
        if monitor is not None:
            monitor.stop()
            monitor_records = list(monitor.records)

    with telemetry_path.open("w", encoding="utf-8") as fh:
        for record in monitor_records:
            fh.write(f"{record['time_ns']}\t{record['raw']}\n")

    _emit(workload_records)
    logger.info("wrote %s (load=%.1fs, %d workloads)",
                out_path, load_time_s, len(workload_records))
    return 0


def _capacity_summary(points: List[Dict[str, Any]], safety: Dict[str, Any]) -> Dict[str, Any]:
    """Derive success/failure/boundary/safe facts from one workload's points."""
    successes = [p for p in points if p.get("status") == "success"]
    failures = [p for p in points if p.get("status") == "failure"]
    safety_stops = [p for p in points if p.get("status") == "safety_stop"]

    max_success_batch = max((p["batch_size"] for p in successes), default=None)
    first_failed_batch = min((p["batch_size"] for p in failures), default=None)

    safe_points: List[Dict[str, Any]] = []
    swap_spike_bytes = safety["max_safe_process_swap_mb"] * (1024**2)
    for p in successes:
        checks = p.get("checks", {})
        # A point is "safe" (E02-04 §5/§10) if it passed correctness, stayed
        # within the thermal window, and did NOT push the process into swap
        # thrashing (pages evicted to swap = memory-pressure evidence).
        process_swap = p.get("result", {}).get("process_swap_bytes", 0)
        safe = (
            checks.get("correctness_ok") is True
            and not p.get("thermal_suspect", False)
            and process_swap < swap_spike_bytes
        )
        p["_safe"] = safe
        p["_safe_swap_mb"] = process_swap / (1024**2)
        if safe:
            safe_points.append(p)

    max_safe_batch = max((p["batch_size"] for p in safe_points), default=None)

    return {
        "successful_batches": sorted(p["batch_size"] for p in successes),
        "failed_batches": sorted(p["batch_size"] for p in failures),
        "safety_stop_batches": sorted(p["batch_size"] for p in safety_stops),
        "max_successful_batch": max_success_batch,
        "first_failed_batch": first_failed_batch,
        "max_safe_batch": max_safe_batch,
        "boundary": (
            None
            if first_failed_batch is None
            else {"lower": max_success_batch, "upper": first_failed_batch}
        ),
        "oom_recorded": bool(failures),
        "safe_swap_threshold_mb": safety["max_safe_process_swap_mb"],
    }


def _run_verify(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    run_files = sorted(output_dir.glob("run_*.json"))
    if not run_files:
        logger.error("no run_*.json found in %s", output_dir)
        return 2

    runs = [json.loads(p.read_text(encoding="utf-8")) for p in run_files]

    model_fully_on_gpu_all = all(
        run.get("actual_device", {}).get("model_fully_on_gpu") is True
        for run in runs
    )

    names = sorted({w["name"] for run in runs for w in run["workloads"]})
    per_workload: Dict[str, Any] = {}

    for name in names:
        # Merge points across runs (each run may carry a subset of workloads).
        points: List[Dict[str, Any]] = []
        batch_propagation: Optional[Dict[str, Any]] = None
        for run in runs:
            record = next((w for w in run["workloads"] if w["name"] == name), None)
            if record is None:
                continue
            points.extend(record["points"])
            if record.get("batch_propagation"):
                batch_propagation = record["batch_propagation"]

        capacity = _capacity_summary(points, _SAFETY)
        # All OOM / failures must carry a stage/category (never dropped).
        all_failures = [p for p in points if p.get("status") == "failure"]
        oom_points = [p for p in all_failures
                      if p.get("failure", {}).get("category") == "cuda_oom"]
        failures_reasoned = all(
            p.get("failure", {}).get("category")
            and p.get("failure", {}).get("message")
            for p in all_failures
        )

        per_workload[name] = {
            "capacity": capacity,
            "batch_propagation": batch_propagation,
            "batch_dim_confirmed": (
                batch_propagation.get("batch_dim_confirmed")
                if batch_propagation else None
            ),
            "num_failures": len(all_failures),
            "num_oom": len(oom_points),
            "failures_reasoned": failures_reasoned,
            "num_points": len(points),
        }

    all_batch_dim_confirmed = all(
        per_workload[n]["batch_dim_confirmed"] is True for n in names
    )
    all_failures_reasoned = all(
        per_workload[n]["failures_reasoned"] for n in names
    )
    all_have_safe = all(
        per_workload[n]["capacity"]["max_safe_batch"] is not None for n in names
    )

    verdict = {
        "runs": len(runs),
        "workloads": names,
        "model_fully_on_gpu_all_runs": model_fully_on_gpu_all,
        "batch_geometry": _BATCH_GEOMETRY,
        "safety": _SAFETY,
        "per_workload": per_workload,
        "batch_dim_confirmed_all": all_batch_dim_confirmed,
        "failures_reasoned_all": all_failures_reasoned,
        "all_workloads_have_safe_batch": all_have_safe,
        "passed": (
            all_batch_dim_confirmed
            and all_failures_reasoned
            and all_have_safe
        ),
    }

    out_path = output_dir / "verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    logger.info("wrote %s (passed=%s)", out_path, verdict["passed"])
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["passed"] else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E02-04 batch capacity boundary")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="scan static batch and record boundary")
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--model-path", default="~/models/hqsb/Qwen3-1.7B")
    collect.add_argument(
        "--manifest",
        default=str(_REPO_ROOT / "docs" / "benchmark" / "model_sha256_manifest.txt"),
    )
    collect.add_argument("--workload-yaml", default=str(_WORKLOAD_YAML))
    collect.add_argument("--run-index", type=int, default=0)
    collect.add_argument("--batchs", type=int, nargs="+", default=_BATCH_GEOMETRY)
    collect.add_argument(
        "--workloads",
        help="Comma-separated workload names to scan (default: all). "
             "Use to isolate a system OOM-kill (SIGKILL) to one workload.",
        default=None,
    )
    collect.add_argument(
        "--monitor-interval-ms", type=int, default=_DEFAULT_MONITOR_INTERVAL_MS
    )
    collect.set_defaults(func=_run_collect)

    verify = sub.add_parser("verify", help="derive capacity boundary + safe batch")
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
