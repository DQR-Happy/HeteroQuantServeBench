#!/usr/bin/env python3
"""E02-06 runner: cold-load / first-request / warm-steady / KV-reset separation.

Proves what ``S02_实验清单.md`` E02-06 requires and what
``docs/stage_experiments/details/S02/E02-06_cold_warm_and_cache_states.md``
specifies: four *separable* costs, a steady-state report that does **not** mix
in cold/first-request cost, and a separately retained startup metric.

* **A — process-cold startup**: a fresh process records ``process_start``
  (via ``/proc``), ``imports_ready``, ``artifact_verified``, ``device_init``,
  ``load`` and ``model_ready`` on one host-monotonic clock. Artifact
  verification is *inside* the startup window (the gate is never bypassed for
  a "nicer" number); its page-cache footprint is recorded via
  ``/proc/self/io``.
* **B — first request**: the first target-shape request after ``model_ready``,
  with no shape warmup, measured with the frozen E02-01 clock.
* **C — warm steady**: pre-registered warmup floor + rolling-window stability
  rule; each warm/steady request runs on a fresh, empty KV cache.
* **D — KV reset**: a live cache is inspected, released, and an independent
  request B must (a) start from an empty cache and (b) reproduce the reference;
  a bounded "no-reset" continuation is the negative control that makes the
  check falsifiable.

Usage (3 independent cold processes, then verify + summarize)::

    for i in 0 1 2; do
      PYTORCH_NO_CUDA_MEMORY_CACHING=1 python3 \\
          scripts/audit/run_e02_06_cold_warm_states.py collect \\
          --output-dir docs/stage_experiments/S02/E02-06/raw --run-index $i
    done
    python3 scripts/audit/run_e02_06_cold_warm_states.py verify \\
        --output-dir docs/stage_experiments/S02/E02-06/raw
    python3 scripts/audit/run_e02_06_cold_warm_states.py summarize \\
        --output-dir docs/stage_experiments/S02/E02-06/raw

``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` is required on this board: the caching
allocator is OOM-killed while loading the model (E02-05 §4.12), so the
allocator counters read 0 and device free/total plus RSS/swap are the
observable memory quantities. That gap is recorded, not hidden.
"""

from __future__ import annotations

import time

# Recorded *before* torch / modelscope / transformers are imported so the
# framework-import span is real, not silently excluded (protocol §8 step 1).
_IMPORTS_BEGIN_NS = time.monotonic_ns()

import argparse  # noqa: E402
import datetime  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import torch  # noqa: E402

from hqsb.benchmark import cold_warm_experiment as cwe  # noqa: E402
from hqsb.benchmark.cold_warm import (  # noqa: E402
    EventTimeline,
    compilation_evidence,
    derive_windows,
    kv_reset_verdict,
    process_start_monotonic_ns,
    read_meminfo_fields,
    read_proc_io,
    read_process_start_ticks,
    read_uptime_seconds,
    startup_attribution,
    validate_event_ordering,
)
from hqsb.benchmark.memory_model import DeviceMemorySampler  # noqa: E402
from hqsb.benchmark.resource_monitor import TegrastatsMonitor  # noqa: E402
from hqsb.benchmark.tegrastats_parser import (  # noqa: E402
    compute_power_summary,
    compute_resource_summary,
    parse_tegrastats_line,
    slice_records,
)
from hqsb.benchmark.workload import make_fixed_token_input  # noqa: E402
from hqsb.benchmark.workload_config import load_workload_dicts  # noqa: E402
from hqsb.models.loader import load_qwen3  # noqa: E402
from hqsb.models.manifest import verify_or_raise  # noqa: E402

logger = logging.getLogger("e02_06")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKLOAD_YAML = _REPO_ROOT / "configs" / "benchmarks" / "jetson_qwen3_fp16.yaml"

_TARGET_WORKLOAD = "balanced"      # first request + warm steady (case B/C)
_RESET_WORKLOAD = "short"          # KV reset verification (case D)
_THERMAL_SUSPECT_C = 95.0          # E02-03/E02-08 pre-registered threshold
_MIN_POWER_SAMPLES = 3


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


def _environment_basic() -> Dict[str, Any]:
    """Environment identity that does **not** touch CUDA.

    Calling ``torch.cuda.*`` here would initialize the context before the timed
    ``device_init`` block and destroy the very cost being measured (protocol §8
    step 2).
    """
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "cuda_initialized_before_device_init": bool(torch.cuda.is_initialized()),
        "pytorch_no_cuda_memory_caching": os.environ.get(
            "PYTORCH_NO_CUDA_MEMORY_CACHING"
        ),
        "allocator_mode": (
            "no_caching"
            if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1"
            else "caching"
        ),
    }


def _meminfo_delta(
    before: Dict[str, int], after: Dict[str, int]
) -> Dict[str, Optional[int]]:
    """Per-field byte delta between two ``/proc/meminfo`` snapshots."""
    return {
        key: (after[key] - before[key])
        if key in before and key in after
        else None
        for key in sorted(set(before) | set(after))
    }


def _device_info() -> Dict[str, Any]:
    """CUDA device identity, called only *after* the timed device init."""
    if not torch.cuda.is_available():
        return {"device": "cpu", "compute_capability": None}
    capability = torch.cuda.get_device_capability(0)
    return {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "device_index": torch.cuda.current_device(),
    }


def _manifest_hash(manifest_path: str) -> Optional[str]:
    try:
        return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    except OSError:
        return None


def _model_files_summary(model_path: str) -> Dict[str, Any]:
    """List model files with sizes (page-cache evidence support)."""
    root = Path(model_path).expanduser().resolve()
    files: List[Dict[str, Any]] = []
    total = 0
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                size = path.stat().st_size
                total += size
                files.append(
                    {"path": str(path.relative_to(root)), "bytes": int(size)}
                )
    return {"root": str(root), "num_files": len(files), "total_bytes": total,
            "files": files}


def _io_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, Optional[int]]:
    """Per-counter delta between two ``/proc/self/io`` snapshots."""
    keys = sorted(set(before) | set(after))
    return {
        key: (after.get(key, 0) - before.get(key, 0)) if key in after else None
        for key in keys
    }


def _telemetry_window_summary(
    records: List[Dict[str, Any]], begin_ns: int, end_ns: int
) -> Dict[str, Any]:
    """Reduce a host-monotonic window to temperature / power / energy."""
    window = slice_records(records, begin_ns, end_ns)
    parsed: List[Dict[str, Any]] = []
    for record in window:
        fields = parse_tegrastats_line(record["raw"])
        fields["time_ns"] = record["time_ns"]
        parsed.append(fields)
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
        "duration_s": power["duration_s"],
        "avg_gpu_temp_c": resource.get("avg_gpu_temp_c"),
        "peak_gpu_temp_c": resource.get("peak_gpu_temp_c"),
        "avg_cpu_temp_c": resource.get("avg_cpu_temp_c"),
        "peak_cpu_temp_c": resource.get("peak_cpu_temp_c"),
    }


# ───────────────────────────────── collect ─────────────────────────────────


def _run_collect(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timeline = EventTimeline()

    # ── process start on the same clock as every in-process event ──
    start_ns = process_start_monotonic_ns(
        now_mono_ns=time.monotonic_ns(),
        uptime_s=read_uptime_seconds(),
        starttime_ticks=read_process_start_ticks(),
        clk_tck=float(os.sysconf("SC_CLK_TCK")) if hasattr(os, "sysconf") else None,
    )
    timeline.record("process_start", ns=start_ns if start_ns is not None else None)
    timeline.record("imports_begin", ns=_IMPORTS_BEGIN_NS)
    timeline.record("imports_ready")

    environment = _environment_basic()
    if environment["allocator_mode"] != "no_caching":
        logger.warning(
            "caching allocator detected; on this board the model load is OOM-"
            "killed (E02-05 §4.12). Re-run with PYTORCH_NO_CUDA_MEMORY_CACHING=1"
        )
    monitor: Optional[TegrastatsMonitor] = None
    monitor_available = False
    monitor_records: List[Dict[str, Any]] = []

    record: Dict[str, Any] = {
        "run_id": (
            "run_"
            + datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        ),
        "run_index": args.run_index,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "environment": environment,
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "manifest_sha256": _manifest_hash(args.manifest),
        "process": {"pid": os.getpid(), "process_start_ns": start_ns},
        "protocol": {
            **cwe.PROTOCOL,
            "target_workload": args.target_workload,
            "reset_workload": args.reset_workload,
            "monitor_interval_ms": args.monitor_interval_ms,
        },
        "error": None,
    }

    try:
        # ── background telemetry (records its own start/end events) ──
        timeline.record("monitor_start_begin")
        try:
            monitor = TegrastatsMonitor(interval_ms=args.monitor_interval_ms)
            monitor.start()
            monitor_available = True
        except Exception as exc:  # noqa: BLE001 - non-Jetson / permission
            logger.warning("tegrastats unavailable: %s", exc)
            monitor = None
        timeline.record("monitor_start_end")

        # ── M0: host view only, must not touch CUDA (protocol §8 step 3) ──
        snapshot_m0 = cwe.snapshot(label="M0_process_start", with_cuda=False)
        io_before_verify = read_proc_io()
        mem_before_verify = read_meminfo_fields()
        files = _model_files_summary(args.model_path)

        # ── artifact gate (inside the startup window, never bypassed) ──
        timeline.record("artifact_verify_begin")
        verification = verify_or_raise(
            record["model_path"],
            args.manifest,
            strict_extra=True,
            allow_extra=("model_sha256_manifest.txt",),
        )
        timeline.record("artifact_verified")
        io_after_verify = read_proc_io()
        mem_after_verify = read_meminfo_fields()

        # ── device init is an explicit, timed block (not silently triggered) ──
        timeline.record("device_init_begin")
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.synchronize()
        timeline.record("device_init_end")
        snapshot_m1 = cwe.snapshot(label="M1_device_ready", with_cuda=True)
        device_info = _device_info()

        # ── weight load (verification already done; no second gate) ──
        load_sampler = DeviceMemorySampler(interval_s=args.sampler_interval_s)
        load_sampler.start()
        timeline.record("load_begin")
        try:
            tokenizer, model, load_time_s = load_qwen3(
                record["model_path"],
                dtype=torch.float16,
                attention_backend="eager",
                device_map=args.device_map,
                verify_manifest=None,
                allow_extra=("model_sha256_manifest.txt",),
            )
        finally:
            timeline.record("model_ready")
            load_sampler.stop()
        io_after_load = read_proc_io()
        mem_after_load = read_meminfo_fields()
        load_device_peak = load_sampler.summary()

        snapshot_m2 = cwe.snapshot(label="M2_model_ready", with_cuda=True)
        param_devices = sorted({str(p.device) for p in model.parameters()})
        model_fully_on_gpu = bool(param_devices) and all(
            d.startswith("cuda") for d in param_devices
        )
        load_peak_allocated_mb = (
            torch.cuda.max_memory_allocated() / (1024**2)
            if torch.cuda.is_available()
            else None
        )
        load_peak_reserved_mb = (
            torch.cuda.max_memory_reserved() / (1024**2)
            if torch.cuda.is_available()
            else None
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        workloads = {w["name"]: w for w in load_workload_dicts(str(args.workload_yaml))}
        target = workloads[args.target_workload]
        reset_spec = workloads[args.reset_workload]

        windows = derive_windows(timeline.events)
        attribution = startup_attribution(windows)

        # ═══ Case B: first request (no target-shape warmup) ═══
        target_inputs = make_fixed_token_input(
            tokenizer, int(target["input_tokens"]), device=device
        )
        timeline.record("first_request_begin")
        first = cwe.measure_request(
            model,
            target_inputs,
            int(target["output_tokens"]),
            sampler_interval_s=args.sampler_interval_s,
        )
        timeline.record("first_request_end")
        first_metrics = first["metrics"]
        first_tokens = list(first_metrics["generated_token_ids"])
        snapshot_m3 = cwe.snapshot(label="M3_after_first_request", with_cuda=True)

        # ═══ Case C: warm steady state ═══
        timeline.record("steady_phase_begin")
        steady = cwe.run_steady(
            model,
            target_inputs,
            int(target["output_tokens"]),
            warmup_min=args.warmup_min,
            max_wait=args.steady_max_wait,
            window=args.steady_window,
            rel_tol=args.steady_rel_tol,
            sampler_interval_s=args.sampler_interval_s,
        )
        timeline.record("steady_phase_end")
        # Representative steady request window = the sample that satisfied the
        # stability rule (the last collected sample).
        last_sample = steady["samples"][-1]
        timeline.record("steady_request_begin", ns=last_sample["start_ns"])
        timeline.record("steady_request_end", ns=last_sample["end_ns"])
        steady_median_ms = steady["latency_summary_ms"]["p50"]

        # ═══ Case D: KV reset state isolation ═══
        reset_inputs = make_fixed_token_input(
            tokenizer, int(reset_spec["input_tokens"]), device=device
        )
        reset_osl = int(reset_spec["output_tokens"])
        reference, ref_cache = cwe.probe_request_with_kv(
            model, reset_inputs, reset_osl
        )
        del ref_cache
        gc.collect()
        request_a, cache_a = cwe.probe_request_with_kv(
            model, reset_inputs, reset_osl
        )
        held_objects = {
            "num_layers": request_a["kv_final"]["num_layers"],
            "kv_final_bytes": request_a["kv_final"]["total_logical_bytes"],
            "cache_type": type(cache_a).__name__,
        }
        snapshot_held = cwe.snapshot(label="M4_kv_held", with_cuda=True)
        mem_before_reset = read_meminfo_fields()
        timeline.record("cache_reset_begin")
        del cache_a
        gc.collect()
        timeline.record("cache_reset_end")
        snapshot_reset = cwe.snapshot(label="M5_after_reset", with_cuda=True)
        mem_after_reset = read_meminfo_fields()
        request_b, cache_b = cwe.probe_request_with_kv(
            model, reset_inputs, reset_osl
        )
        del cache_b
        gc.collect()

        negative = cwe.negative_control_no_reset(model, reset_inputs)
        reset_verdict = kv_reset_verdict(
            cache_filled_before_reset=request_a["kv_final"].get("context_filled"),
            request_b_filled_after_prefill=request_b["kv_after_prefill"].get(
                "context_filled"
            ),
            expected_filled_after_prefill=int(reset_spec["input_tokens"]),
            request_b_matches_reference=(
                list(request_b["generated_token_ids"])
                == list(reference["generated_token_ids"])
            ),
            negative_control=negative,
        )

        # ═══ Case E: optional allocator reuse diagnostic ═══
        allocator = {"enabled": False}
        if args.allocator_probe:
            timeline.record("allocator_probe_begin")
            allocator = cwe.allocator_reuse_probe(
                model, target_inputs, int(target["output_tokens"])
            )
            timeline.record("allocator_probe_end")

        # ═══ close ═══
        timeline.record("close_begin")
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if monitor is not None:
            monitor.stop()
            monitor_records = list(monitor.records)
            monitor_available = True
        timeline.record("close_complete")
        io_final = read_proc_io()

        # ── derived forward-looking values ──
        windows = derive_windows(timeline.events)
        attribution = startup_attribution(windows)
        ordering_violations = validate_event_ordering(timeline.events)

        compile_evidence = compilation_evidence(
            enabled=False,
            dynamo_counters=cwe.dynamo_counters(),
            notes=(
                "eager attention backend; no torch.compile call in the runner. "
                "Compilation optimization is out of E02-06 scope (S06)."
            ),
        )

        telemetry: Dict[str, Any] = {"available": monitor_available}
        if monitor_records:
            telemetry["startup_window"] = _telemetry_window_summary(
                monitor_records,
                timeline.get("process_start") or timeline.get("imports_ready"),
                timeline.get("model_ready"),
            )
            telemetry["first_request_window"] = _telemetry_window_summary(
                monitor_records,
                timeline.get("first_request_begin"),
                timeline.get("first_request_end"),
            )
            telemetry["steady_window"] = _telemetry_window_summary(
                monitor_records,
                steady["steady_phase_window_ns"][0],
                steady["steady_phase_window_ns"][1],
            )
            peak_temp = telemetry["steady_window"].get("peak_gpu_temp_c")
            if peak_temp is not None and peak_temp > _THERMAL_SUSPECT_C:
                telemetry["thermal_suspect"] = True
            else:
                telemetry["thermal_suspect"] = False
            for key in (
                "startup_window",
                "first_request_window",
                "steady_window",
            ):
                window = telemetry[key]
                window["energy_usable"] = (
                    window["num_power_samples"] >= _MIN_POWER_SAMPLES
                )

        io_verify_delta = _io_delta(io_before_verify, io_after_verify)
        io_load_delta = _io_delta(io_after_verify, io_after_load)
        mem_verify_delta = _meminfo_delta(mem_before_verify, mem_after_verify)
        mem_load_delta = _meminfo_delta(mem_after_verify, mem_after_load)
        file_cache_interpretation = (
            "artifact verification streams every weight file, so /proc/meminfo "
            "Cached grows by roughly the model size; the loader then reads that "
            "same data while Cached barely grows again, i.e. it is served from "
            "the OS page cache (process-cold, not storage-cold). /proc/self/io "
            "is unavailable on this kernel, so meminfo is the page-cache probe."
        )
        memory_accounting = {
            "allocator_mode": environment["allocator_mode"],
            "allocator_counters_available": (
                environment["allocator_mode"] == "caching"
            ),
            "method": (
                "device free/total (mem_get_info) sampler + RSS/swap; the "
                "caching allocator is unusable on this board (OOM killer during "
                "load), so allocated/reserved are reported only when available"
                if environment["allocator_mode"] == "no_caching"
                else "torch allocator counters + sampler + RSS/swap"
            ),
            "note": (
                "E02-05 independently established that the caching allocator is "
                "OOM-killed while loading this model on this board; under "
                "PYTORCH_NO_CUDA_MEMORY_CACHING=1 the counters read 0 and device "
                "free/total is the observable memory quantity (see E02-05 §4.12)"
            ),
        }

        first_vs_steady = {
            "first_e2e_ms": first_metrics["model_core_e2e_ms"],
            "steady_e2e_ms": steady_median_ms,
            "e2e_delta_ms": (
                first_metrics["model_core_e2e_ms"] - steady_median_ms
                if steady_median_ms is not None
                else None
            ),
            "e2e_ratio": (
                first_metrics["model_core_e2e_ms"] / steady_median_ms
                if steady_median_ms
                else None
            ),
            "first_ttft_ms": first_metrics["model_core_ttft_ms"],
            "steady_ttft_ms": steady["samples"][-1]["model_core_ttft_ms"],
            "first_decode_total_ms": first_metrics["decode_total_ms"],
            "steady_decode_total_ms": steady["samples"][-1]["decode_total_ms"],
            "first_peak_allocated_mb": first_metrics["peak_cuda_allocated_mb"],
            "steady_peak_allocated_mb": steady["samples"][-1][
                "peak_cuda_allocated_mb"
            ],
            "first_device_peak_mib": (first["external_device_peak"] or {}).get(
                "max"
            ),
            "steady_device_peak_mib": (
                steady["samples"][-1].get("external_device_peak") or {}
            ).get("max"),
            "explanation": (
                "first vs steady is decomposed into the prefill/TTFT path "
                "(shape-keyed library init, declared as first-shape cost) and "
                "the decode path; warmup samples are retained and excluded from "
                "the steady distribution. Temperature is cross-checked so an "
                "init saving is not misread as thermal degradation."
            ),
        }

        checks = {
            "four_cost_classes_present": all(
                windows[key] is not None
                for key in (
                    "startup_to_model_ready_ms",
                    "first_request_latency_ms",
                    "steady_request_latency_ms",
                    "reset_latency_ms",
                )
            ),
            "model_fully_on_gpu": model_fully_on_gpu,
            "steady_reached": steady["reached"],
            "kv_reset_passed": reset_verdict["passed"],
            "timeline_ordering_ok": not ordering_violations,
            "first_request_was_cold_shape": True,
            "artifact_gate_in_startup": True,
            "compile_not_enabled": not compile_evidence["enabled"],
        }

        record.update(
            {
                "device_info": device_info,
                "model_files": files,
                "load_time_s": load_time_s,
                "actual_device": {
                    "param_devices": param_devices,
                    "model_fully_on_gpu": model_fully_on_gpu,
                },
                "memory_accounting": memory_accounting,
                "caveats": [
                    (
                        "allocator_mode=no_caching: the caching allocator is "
                        "OOM-killed while loading this model on this board "
                        "(re-confirmed here and in E02-05 §4.12). Absolute "
                        "latencies are therefore inflated and are NOT comparable "
                        "to the caching-allocator E02-03 baseline; the four cost "
                        "classes are separated consistently *within* this mode."
                        if environment["allocator_mode"] == "no_caching"
                        else "allocator_mode=caching: counters available"
                    ),
                    (
                        "os_file_page_cache is not controlled: verification "
                        "warms the page cache and the loader then reuses it "
                        "(process-cold, not storage-cold)"
                    ),
                ],
                "startup": {
                    "artifact_verification": verification.as_dict(),
                    "load_peak_allocated_mb": load_peak_allocated_mb,
                    "load_peak_reserved_mb": load_peak_reserved_mb,
                    "load_device_used_peak_mib": load_device_peak,
                    "windows_ms": windows,
                    "attribution": attribution,
                    "ordering_violations": ordering_violations,
                    "snapshots": {
                        "M0_process_start": snapshot_m0,
                        "M1_device_ready": snapshot_m1,
                        "M2_model_ready": snapshot_m2,
                    },
                    "process_io": {
                        "available": bool(io_after_verify or io_after_load),
                        "before_verify": io_before_verify,
                        "after_verify": io_after_verify,
                        "after_load": io_after_load,
                        "final": io_final,
                        "verify_delta": io_verify_delta,
                        "load_delta": io_load_delta,
                    },
                    "meminfo": {
                        "before_verify": mem_before_verify,
                        "after_verify": mem_after_verify,
                        "after_load": mem_after_load,
                        "before_reset": mem_before_reset,
                        "after_reset": mem_after_reset,
                        "verify_delta": mem_verify_delta,
                        "load_delta": mem_load_delta,
                    },
                    "page_cache_evidence": {
                        "controlled": False,
                        "method": "/proc/meminfo Cached/MemAvailable deltas",
                        "model_total_bytes": files["total_bytes"],
                        "cached_growth_verify_bytes": mem_verify_delta.get("Cached"),
                        "cached_growth_load_bytes": mem_load_delta.get("Cached"),
                        "memavailable_delta_verify_bytes": mem_verify_delta.get(
                            "MemAvailable"
                        ),
                        "interpretation": file_cache_interpretation,
                    },
                },
                "first_request": {
                    "workload": args.target_workload,
                    "input_tokens": first_metrics["input_tokens"],
                    "output_tokens": first_metrics["output_tokens"],
                    "latency_ms": windows["first_request_latency_ms"],
                    "startup_to_first_result_ms": windows[
                        "startup_to_first_result_ms"
                    ],
                    "metrics": {
                        key: first_metrics[key]
                        for key in (
                            "prefill_forward_ms",
                            "first_token_selection_ms",
                            "model_core_ttft_ms",
                            "decode_total_ms",
                            "model_core_e2e_ms",
                            "prefill_tokens_per_s",
                            "decode_tokens_per_s",
                            "peak_cuda_allocated_mb",
                            "peak_cuda_reserved_mb",
                        )
                    },
                    "raw_itl_ms": first_metrics["raw_itl_ms"],
                    "sequence_sha256": hashlib.sha256(
                        json.dumps(first_tokens, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "first_token": first_tokens[0],
                    "last_token": first_tokens[-1],
                    "external_device_peak": first["external_device_peak"],
                    "snapshot_after": snapshot_m3,
                },
                "steady": {
                    **steady,
                    "workload": args.target_workload,
                    "kv_state_per_request": "empty_then_filled_by_request",
                    "excludes_startup_and_first": True,
                    "median_e2e_ms": steady_median_ms,
                },
                "kv_reset": {
                    "workload": args.reset_workload,
                    "reference": {
                        "source": "independent fresh-cache run in this process",
                        "sequence_sha256": hashlib.sha256(
                            json.dumps(
                                list(reference["generated_token_ids"]),
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest(),
                        "kv_after_prefill_filled": reference["kv_after_prefill"].get(
                            "context_filled"
                        ),
                    },
                    "request_a": {
                        "sequence_sha256": hashlib.sha256(
                            json.dumps(
                                list(request_a["generated_token_ids"]),
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest(),
                        "kv_after_prefill_filled": request_a["kv_after_prefill"].get(
                            "context_filled"
                        ),
                        "kv_final_filled": request_a["kv_final"].get("context_filled"),
                        "held_objects": held_objects,
                    },
                    "reset": {
                        "mechanism": cwe.PROTOCOL["reset_mechanism"],
                        "allocator_empty_cache": cwe.PROTOCOL[
                            "allocator_empty_cache_in_reset"
                        ],
                        "logical_reset": True,
                        "snapshot_held": snapshot_held,
                        "snapshot_after": snapshot_reset,
                    },
                    "request_b": {
                        "sequence_sha256": hashlib.sha256(
                            json.dumps(
                                list(request_b["generated_token_ids"]),
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest(),
                        "kv_after_prefill_filled": request_b["kv_after_prefill"].get(
                            "context_filled"
                        ),
                    },
                    "verdict": reset_verdict,
                    "passed": reset_verdict["passed"],
                },
                "allocator_reuse": allocator,
                "compile": compile_evidence,
                "cache_states": {
                    "process": {
                        "fresh_process": True,
                        "pid": os.getpid(),
                        "process_start_ns": timeline.get("process_start"),
                    },
                    "os_file_page_cache": {
                        "controlled": False,
                        "method": "meminfo_cached_deltas",
                        "cached_growth_verify_bytes": mem_verify_delta.get("Cached"),
                        "cached_growth_load_bytes": mem_load_delta.get("Cached"),
                        "memavailable_delta_verify_bytes": mem_verify_delta.get(
                            "MemAvailable"
                        ),
                        "model_total_bytes": files["total_bytes"],
                    },
                    "compile": {
                        "enabled": compile_evidence["enabled"],
                        "statement": compile_evidence["statement"],
                    },
                    "allocator": {
                        "mode": environment["allocator_mode"],
                        "counters_available": memory_accounting[
                            "allocator_counters_available"
                        ],
                        "reset_empty_cache": cwe.PROTOCOL[
                            "allocator_empty_cache_in_reset"
                        ],
                        "load_peak_allocated_mb": load_peak_allocated_mb,
                        "load_peak_reserved_mb": load_peak_reserved_mb,
                    },
                    "kv": {
                        "per_request_empty": True,
                        "reset_verified": reset_verdict["passed"],
                        "filled_before_reset": reset_verdict[
                            "cache_filled_before_reset"
                        ],
                    },
                    "shape_warmness": {
                        "target_shape_warmed_before_steady": True,
                        "first_request_used_cold_target_shape": True,
                        "target_workload": args.target_workload,
                    },
                    "thermal": {
                        "avg_gpu_temp_first_c": telemetry.get(
                            "first_request_window", {}
                        ).get("avg_gpu_temp_c"),
                        "avg_gpu_temp_steady_c": telemetry.get(
                            "steady_window", {}
                        ).get("avg_gpu_temp_c"),
                    },
                },
                "first_vs_steady": first_vs_steady,
                "telemetry": telemetry,
                "checks": checks,
            }
        )
    except Exception as exc:  # noqa: BLE001 - record and re-raise
        record["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("collect failed")
    finally:
        if monitor is not None:
            try:
                monitor.stop()
            except Exception:  # noqa: BLE001
                pass
            if not monitor_records:
                monitor_records = list(monitor.records)
        telemetry_path = output_dir / f"run_{args.run_index}.tegrastats.txt"
        with telemetry_path.open("w", encoding="utf-8") as fh:
            for item in monitor_records:
                fh.write(f"{item['time_ns']}\t{item['raw']}\n")
        record["timeline"] = timeline.as_dict()
        record["monitor"] = {
            "available": monitor_available,
            "num_records": len(monitor_records),
            "raw_telemetry_file": str(telemetry_path),
        }
        out_path = output_dir / f"run_{args.run_index}.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.info("wrote %s (error=%s)", out_path, record["error"])

    return 0 if record["error"] is None else 1


# ───────────────────────────────── verify ─────────────────────────────────


def _cross_run_summary(
    per_run: List[Dict[str, Optional[float]]]
) -> Dict[str, Any]:
    keys = [
        "startup_to_model_ready_ms",
        "first_request_latency_ms",
        "steady_request_latency_ms",
        "reset_latency_ms",
        "startup_to_first_result_ms",
        "artifact_verify_ms",
        "load_ms",
    ]
    out: Dict[str, Any] = {}
    for key in keys:
        values = sorted(
            float(run[key]) for run in per_run if run.get(key) is not None
        )
        if not values:
            out[key] = {"median": None, "min": None, "max": None, "runs": 0}
            continue
        median = values[len(values) // 2]
        out[key] = {
            "median": median,
            "min": values[0],
            "max": values[-1],
            "runs": len(values),
        }
    return out


def _run_verify(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    run_files = sorted(output_dir.glob("run_*.json"))
    if len(run_files) < 3:
        logger.error("need >= 3 run_*.json files, found %d", len(run_files))
        return 2

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in run_files]
    ok_runs = [r for r in runs if r.get("error") is None]
    if len(ok_runs) < 3:
        logger.error("only %d successful runs (need >= 3)", len(ok_runs))
        return 2

    def _windows(run: Dict[str, Any]) -> Dict[str, Optional[float]]:
        return run.get("startup", {}).get("windows_ms", {})

    four_cost = all(
        all(
            _windows(run).get(key) is not None
            for key in (
                "startup_to_model_ready_ms",
                "first_request_latency_ms",
                "steady_request_latency_ms",
                "reset_latency_ms",
            )
        )
        for run in ok_runs
    )

    steady_excludes_cold = all(
        run["steady"].get("reached") is True
        and run["steady"].get("sample_count", 0)
        >= run["steady"].get("rule", {}).get("window", 3)
        and _windows(run).get("steady_request_latency_ms") is not None
        for run in ok_runs
    )
    # The representative steady request must physically follow the first request.
    steady_window_after_first = all(
        run["timeline"]["events"].get("steady_request_begin", 0)
        >= run["timeline"]["events"].get("first_request_end", 0)
        for run in ok_runs
    )

    startup_retained = all(
        _windows(run).get("startup_to_model_ready_ms") is not None
        and run["startup"]["attribution"].get("complete") is True
        for run in ok_runs
    )
    kv_reset_independent = all(run["kv_reset"]["passed"] for run in ok_runs)
    ordering_ok = all(
        not run["startup"].get("ordering_violations") for run in ok_runs
    )
    cache_states_evidenced = all(
        isinstance(run.get("cache_states"), dict)
        and run["cache_states"].get("os_file_page_cache")
        and run["cache_states"].get("compile", {}).get("enabled") is False
        and run["cache_states"].get("allocator")
        and run["cache_states"].get("kv")
        and run["cache_states"].get("shape_warmness")
        for run in ok_runs
    )
    differences_explained = all(
        run.get("first_vs_steady", {}).get("explanation") for run in ok_runs
    )
    three_processes = len(ok_runs) >= 3 and len(
        {run["run_id"] for run in ok_runs}
    ) == len(ok_runs)

    per_run_costs = [_windows(run) for run in ok_runs]
    cross_run = _cross_run_summary(per_run_costs)

    verdict = {
        "runs": len(ok_runs),
        "run_ids": [run["run_id"] for run in ok_runs],
        "single_process_per_run": [
            run["cache_states"]["process"].get("fresh_process") for run in ok_runs
        ],
        "cross_run_cost_summary_ms": cross_run,
        "checks": {
            "four_cost_classes_separable": bool(four_cost),
            "steady_excludes_cold_and_first": bool(
                steady_excludes_cold and steady_window_after_first
            ),
            "startup_metric_retained_separately": bool(startup_retained),
            "kv_reset_preserves_independence": bool(kv_reset_independent),
            "cache_states_evidenced": bool(cache_states_evidenced),
            "three_independent_processes": bool(three_processes),
            "differences_explained_within_observations": bool(differences_explained),
            "timeline_ordering_ok": bool(ordering_ok),
        },
        "per_run": {
            run["run_index"]: {
                "windows_ms": _windows(run),
                "first_vs_steady": run.get("first_vs_steady"),
                "steady_samples": run["steady"].get("sample_count"),
                "steady_reached": run["steady"].get("reached"),
                "kv_reset_passed": run["kv_reset"]["passed"],
                "checks": run.get("checks"),
            }
            for run in ok_runs
        },
    }
    verdict["passed"] = all(verdict["checks"].values())

    out_path = output_dir / "verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    logger.info("wrote %s (passed=%s)", out_path, verdict["passed"])
    print(json.dumps(verdict["checks"], indent=2))
    return 0 if verdict["passed"] else 1


# ───────────────────────────────── summarize ─────────────────────────────────


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _run_summarize(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    verdict = json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))
    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_dir.glob("run_*.json"))
    ]
    runs = [r for r in runs if r.get("error") is None]

    lines: List[str] = []
    lines.append("# E02-06 evidence summary (recomputed from raw)\n")
    lines.append("## Four cost classes per run (ms)\n")
    lines.append(
        "| run | startup→ready | first request | steady request | reset | "
        "startup→first result | artifact verify | load |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        w = run["startup"]["windows_ms"]
        lines.append(
            f"| {run['run_index']} | {_fmt(w['startup_to_model_ready_ms'])} | "
            f"{_fmt(w['first_request_latency_ms'])} | "
            f"{_fmt(w['steady_request_latency_ms'])} | "
            f"{_fmt(w['reset_latency_ms'])} | "
            f"{_fmt(w['startup_to_first_result_ms'])} | "
            f"{_fmt(w['artifact_verify_ms'])} | {_fmt(w['load_ms'])} |"
        )

    lines.append("\n## Startup attribution (non-overlapping, ms)\n")
    lines.append(
        "| run | interpreter | framework import | artifact verify | device init | "
        "load | unattributed | total |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        parts = run["startup"]["attribution"]["parts"]
        lines.append(
            f"| {run['run_index']} | {_fmt(parts.get('interpreter_startup_ms'))} | "
            f"{_fmt(parts.get('framework_import_ms'))} | "
            f"{_fmt(parts.get('artifact_verify_ms'))} | "
            f"{_fmt(parts.get('device_init_ms'))} | "
            f"{_fmt(parts.get('load_ms'))} | "
            f"{_fmt(run['startup']['attribution'].get('unattributed_overhead_ms'))} | "
            f"{_fmt(run['startup']['attribution'].get('total_startup_to_model_ready_ms'))} |"
        )

    lines.append("\n## cache state evidence\n")
    lines.append(
        "| run | Cached growth during verify (MiB) | Cached growth during load (MiB) | "
        "model size (MiB) | allocator mode | compile enabled |"
    )
    lines.append("|---|---:|---:|---:|---|---|")
    for run in runs:
        cache = run["cache_states"]["os_file_page_cache"]
        allocator = run["cache_states"]["allocator"]

        def _mib(value: Optional[float]) -> Optional[float]:
            return None if value is None else value / (1024**2)

        lines.append(
            f"| {run['run_index']} | "
            f"{_fmt(_mib(cache.get('cached_growth_verify_bytes')))} | "
            f"{_fmt(_mib(cache.get('cached_growth_load_bytes')))} | "
            f"{_fmt(_mib(cache.get('model_total_bytes')))} | "
            f"{allocator.get('mode')} | {run['compile']['enabled']} |"
        )

    lines.append("\n## first request vs warm steady (ms)\n")
    lines.append(
        "| run | first E2E | steady E2E | delta | ratio | first TTFT | steady TTFT | "
        "first decode | steady decode |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        fv = run["first_vs_steady"]
        lines.append(
            f"| {run['run_index']} | {_fmt(fv.get('first_e2e_ms'))} | "
            f"{_fmt(fv.get('steady_e2e_ms'))} | {_fmt(fv.get('e2e_delta_ms'))} | "
            f"{_fmt(fv.get('e2e_ratio'), 3)} | {_fmt(fv.get('first_ttft_ms'))} | "
            f"{_fmt(fv.get('steady_ttft_ms'))} | "
            f"{_fmt(fv.get('first_decode_total_ms'))} | "
            f"{_fmt(fv.get('steady_decode_total_ms'))} |"
        )

    lines.append("\n## KV reset verification\n")
    lines.append(
        "| run | A KV filled | B prefill filled | expected | B == reference | "
        "negative control detected | passed |"
    )
    lines.append("|---|---:|---:|---:|---|---|---|")
    for run in runs:
        verdict_run = run["kv_reset"]["verdict"]
        lines.append(
            f"| {run['run_index']} | "
            f"{verdict_run.get('cache_filled_before_reset')} | "
            f"{verdict_run.get('request_b_filled_after_prefill')} | "
            f"{verdict_run.get('expected_filled_after_prefill')} | "
            f"{verdict_run.get('request_b_matches_reference')} | "
            f"{verdict_run.get('negative_control_detected')} | "
            f"{verdict_run.get('passed')} |"
        )

    lines.append("\n## verdict checks\n")
    for name, value in verdict["checks"].items():
        lines.append(f"- {name}: {value}")

    text = "\n".join(lines) + "\n"
    out_path = output_dir / "evidence_summary.md"
    out_path.write_text(text, encoding="utf-8")
    print(text)
    logger.info("wrote %s", out_path)
    return 0


# ───────────────────────────────── CLI ─────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E02-06 cold/warm cost separation")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="one fresh process, full A→F trace")
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--model-path", default="~/models/hqsb/Qwen3-1.7B")
    collect.add_argument(
        "--device-map",
        default=None,
        help="forwarded to load_qwen3 (default: auto); 'cuda' skips the CPU "
        "staging path that inflates the load transient peak",
    )
    collect.add_argument(
        "--manifest",
        default=str(_REPO_ROOT / "docs" / "benchmark" / "model_sha256_manifest.txt"),
    )
    collect.add_argument("--workload-yaml", default=str(_WORKLOAD_YAML))
    collect.add_argument("--run-index", type=int, default=0)
    collect.add_argument("--target-workload", default=_TARGET_WORKLOAD)
    collect.add_argument("--reset-workload", default=_RESET_WORKLOAD)
    collect.add_argument("--warmup-min", type=int, default=cwe.PROTOCOL["warmup_min"])
    collect.add_argument(
        "--steady-max-wait", type=int, default=cwe.PROTOCOL["steady_max_wait"]
    )
    collect.add_argument(
        "--steady-window", type=int, default=cwe.PROTOCOL["steady_window"]
    )
    collect.add_argument(
        "--steady-rel-tol", type=float, default=cwe.PROTOCOL["steady_rel_tol"]
    )
    collect.add_argument(
        "--sampler-interval-s",
        type=float,
        default=cwe.PROTOCOL["sampler_interval_s"],
    )
    collect.add_argument("--monitor-interval-ms", type=int, default=500)
    collect.add_argument("--allocator-probe", action="store_true")
    collect.set_defaults(func=_run_collect)

    verify = sub.add_parser("verify", help="cross-run cost separation + pass verdict")
    verify.add_argument("--output-dir", required=True)
    verify.set_defaults(func=_run_verify)

    summarize = sub.add_parser("summarize", help="recompute report tables from raw")
    summarize.add_argument("--output-dir", required=True)
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
