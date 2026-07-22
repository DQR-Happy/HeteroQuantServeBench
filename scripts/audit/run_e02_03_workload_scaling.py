#!/usr/bin/env python3
"""E02-03 runner: six-workload latency / throughput scaling baseline.

Proves the three things required by ``S02_实验清单.md`` E02-03 (see
``docs/stage_experiments/details/S02/E02-03_workload_scaling_baseline.md``):

1. **Scaling curves** — fixed B=1, sweep the six frozen workloads
   (tiny / short / balanced / long-prefill / decode-heavy / long-balanced)
   and record TTFT / TPOT / E2E together with prefill / decode / output
   tokens/s for each request.
2. **Per-request raw timing + P50/P95 + memory + temperature + energy** —
   every request stores raw prefill/selection/ITL/Tgen timings, peak CUDA
   allocated/reserved, RSS/swap, and a host-monotonic telemetry window
   (tegrastats) is integrated for temperature and energy.
3. **Reproducibility** — at least three independent process runs, each with a
   frozen warmup + measurement repeat plan; prefill and decode are reported
   separately, and anomalous runs are kept and labelled with a reason.

Usage (3 independent process runs, then verify):

    for i in 0 1 2; do
      python scripts/audit/run_e02_03_workload_scaling.py collect \
          --output-dir docs/stage_experiments/S02/E02-03/raw --run-index $i
    done
    python scripts/audit/run_e02_03_workload_scaling.py verify \
        --output-dir docs/stage_experiments/S02/E02-03/raw
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from hqsb.benchmark.correctness import hash_token_sequence
from hqsb.benchmark.metrics import request_summary
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

logger = logging.getLogger("e02_03")

_REPO_ROOT = Path(__file__).resolve().parents[2]

_WORKLOAD_YAML = _REPO_ROOT / "configs" / "benchmarks" / "jetson_qwen3_fp16.yaml"

# Pre-registered measurement protocol (E02-03 §6): warmup repeats are never
# timed/recorded; only the R measurement repeats enter latency distributions.
_DEFAULT_WARMUP = 1
_DEFAULT_REPETITIONS = 3
_DEFAULT_MONITOR_INTERVAL_MS = 500
_DEFAULT_ORDER_SEED = 42

# Thermal rule (E02-03 §6 / E02-08 §8): a workload whose peak GPU temperature
# exceeds this threshold is marked thermal-suspect and kept (not silently
# dropped), but flagged as not part of the ordinary baseline.
_THERMAL_SUSPECT_C = 95.0
# Minimum power samples for a trustworthy energy integral over a window.
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


def _telemetry_window_summary(
    monitor_records: List[Dict[str, Any]],
    begin_ns: int,
    end_ns: int,
) -> Dict[str, Any]:
    """Slice a monitor window and reduce it to temperature / frequency / energy.

    The window is the half-open ``[begin_ns, end_ns)`` host-monotonic interval
    bracketing the R measurement requests (warmup excluded). Energy is the
    trapezoidal integral of ``VDD_IN`` power over that window (E02-08 §2).
    """
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
        "duration_s": power["duration_s"],
        "avg_gpu_temp_c": resource.get("avg_gpu_temp_c"),
        "peak_gpu_temp_c": resource.get("peak_gpu_temp_c"),
        "avg_cpu_temp_c": resource.get("avg_cpu_temp_c"),
        "peak_cpu_temp_c": resource.get("peak_cpu_temp_c"),
        "avg_gpu_util_pct": resource.get("avg_gpu_util_pct"),
        "peak_gpu_util_pct": resource.get("peak_gpu_util_pct"),
        "avg_ram_used_mb": resource.get("avg_ram_used_mb"),
        "peak_ram_used_mb": resource.get("peak_ram_used_mb"),
        "cpu_freqs_mhz": sorted(set(cpu_freqs)),
    }


def _sample_checks(
    result: Dict[str, Any],
    isl: int,
    osl: int,
    seq_hash: str,
) -> Dict[str, Any]:
    return {
        "output_tokens_exact": result["output_tokens"] == osl,
        "decode_steps_exact": len(result["raw_itl_ms"]) == osl - 1,
        "input_tokens_exact": result["input_tokens"] == isl,
        "sequence_sha256": seq_hash,
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

    # Record the *actual* device placement (E02-03 §4 control variable "无隐式
    # offload"): the loader consolidates onto CUDA when free memory allows, and
    # we must not silently claim pure-GPU if part of the model stayed offloaded.
    param_devices = sorted({str(p.device) for p in model.parameters()})
    model_fully_on_gpu = bool(param_devices) and all(
        d.startswith("cuda") for d in param_devices
    )

    workloads = load_workload_dicts(str(args.workload_yaml))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Model-level warmup (first-call / cuBLAS workspace / lazy init) ──
    warm_inputs = make_fixed_token_input(tokenizer, 32, device=device)
    _ = benchmark_model_core(model, warm_inputs, 2)

    # ── Workload order: fixed-seed shuffle per run to balance thermal drift
    #    (E02-03 §5). The exact order is recorded in the run record.
    order = list(range(len(workloads)))
    order_seed = args.seed + args.run_index
    random.Random(order_seed).shuffle(order)

    # ── Background telemetry (tegrastats) ──────────────────────────────
    monitor: Optional[TegrastatsMonitor] = None
    monitor_records: List[Dict[str, Any]] = []
    monitor_available = False
    try:
        monitor = TegrastatsMonitor(interval_ms=args.monitor_interval_ms)
        monitor.start()
        monitor_available = True
    except Exception as exc:  # noqa: BLE001 — non-Jetson / permission
        logger.warning("tegrastats unavailable: %s", exc)
        monitor = None

    workload_records: List[Dict[str, Any]] = []
    total_output_tokens = 0
    total_input_tokens = 0

    try:
        for idx in order:
            spec = workloads[idx]
            name = spec["name"]
            isl = int(spec["input_tokens"])
            osl = int(spec["output_tokens"])
            logger.info(
                "Workload %s (ISL=%d OSL=%d)", name, isl, osl
            )

            inputs = make_fixed_token_input(tokenizer, isl, device=device)

            # ── Warmup (never timed / recorded) ──────────────────────
            for _ in range(args.warmup):
                benchmark_model_core(model, inputs, osl)

            # ── Measurement window (R requests, each with fresh KV) ──
            measure_begin_ns = time.monotonic_ns()
            samples: List[Dict[str, Any]] = []
            anomalies: List[Dict[str, Any]] = []
            for rep in range(args.repetitions):
                try:
                    result = benchmark_model_core(model, inputs, osl)
                except Exception as exc:  # noqa: BLE001 — e.g. OOM
                    anomalies.append(
                        {
                            "repetition": rep,
                            "reason": f"{type(exc).__name__}: {exc}",
                            "category": "exception",
                        }
                    )
                    logger.warning("workload %s rep %d failed: %s", name, rep, exc)
                    continue

                seq_hash = hash_token_sequence(result["generated_token_ids"])
                sample = {
                    "repetition": rep,
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "generated_token_ids": result["generated_token_ids"],
                    "sequence_sha256": seq_hash,
                    "prefill_forward_ms": result["prefill_forward_ms"],
                    "first_token_selection_ms": result["first_token_selection_ms"],
                    "model_core_ttft_ms": result["model_core_ttft_ms"],
                    "decode_total_ms": result["decode_total_ms"],
                    "model_core_e2e_ms": result["model_core_e2e_ms"],
                    "prefill_tokens_per_s": result["prefill_tokens_per_s"],
                    "decode_tokens_per_s": result["decode_tokens_per_s"],
                    "model_core_output_tokens_per_s": result[
                        "model_core_output_tokens_per_s"
                    ],
                    "raw_itl_ms": result["raw_itl_ms"],
                    "peak_cuda_allocated_mb": result["peak_cuda_allocated_mb"],
                    "peak_cuda_reserved_mb": result["peak_cuda_reserved_mb"],
                    "process_rss_bytes": result["process_rss_bytes"],
                    "process_swap_bytes": result["process_swap_bytes"],
                    "kv_cache_total_bytes": result["kv_cache"]["total_bytes"],
                    "checks": _sample_checks(result, isl, osl, seq_hash),
                }
                samples.append(sample)
                total_output_tokens += result["output_tokens"]
                total_input_tokens += result["input_tokens"]

            measure_end_ns = time.monotonic_ns()

            telemetry: Dict[str, Any] = {"available": monitor_available}
            if monitor_available and monitor is not None:
                window = _telemetry_window_summary(
                    monitor.records, measure_begin_ns, measure_end_ns
                )
                window["j_per_request"] = (
                    window["energy_j"] / len(samples) if len(samples) else None
                )
                window["output_tok_per_j"] = (
                    sum(s["output_tokens"] for s in samples) / window["energy_j"]
                    if window["energy_j"] > 0 and samples
                    else None
                )
                window["prefill_tok_per_j"] = (
                    sum(s["input_tokens"] for s in samples) / window["energy_j"]
                    if window["energy_j"] > 0 and samples
                    else None
                )
                window["energy_usable"] = (
                    window["num_power_samples"] >= _MIN_POWER_SAMPLES
                )
                # Thermal-suspect detection (kept, not dropped).
                peak_temp = window.get("peak_gpu_temp_c")
                if peak_temp is not None and peak_temp > _THERMAL_SUSPECT_C:
                    anomalies.append(
                        {
                            "repetition": None,
                            "reason": (
                                f"thermal_suspect: peak_gpu_temp_c={peak_temp:.1f}C "
                                f"> {_THERMAL_SUSPECT_C}C"
                            ),
                            "category": "thermal_suspect",
                        }
                    )
                telemetry["window"] = window

            exception_count = sum(
                1 for a in anomalies if a.get("category") == "exception"
            )
            thermal_suspect = any(
                a.get("category") == "thermal_suspect" for a in anomalies
            )
            workload_records.append(
                {
                    "name": name,
                    "input_tokens": isl,
                    "output_tokens": osl,
                    "batch_size": 1,
                    "warmup": args.warmup,
                    "repetitions": args.repetitions,
                    "valid_samples": len(samples),
                    "failed_samples": exception_count,
                    "thermal_suspect": thermal_suspect,
                    "anomalies": anomalies,
                    "samples": samples,
                    "summary": request_summary(samples) if samples else {},
                    "telemetry": telemetry,
                }
            )
            logger.info(
                "  %s: %d valid / %d failed samples (thermal_suspect=%s)",
                name, len(samples), exception_count, thermal_suspect,
            )
    finally:
        if monitor is not None:
            monitor.stop()
            monitor_records = list(monitor.records)

    # ── Raw telemetry dump (full time series, E02-03 §9 resource evidence) ──
    telemetry_path = output_dir / f"run_{args.run_index}.tegrastats.txt"
    with telemetry_path.open("w", encoding="utf-8") as fh:
        for record in monitor_records:
            fh.write(f"{record['time_ns']}\t{record['raw']}\n")

    run_record: Dict[str, Any] = {
        "run_id": (
            f"run_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        ),
        "run_index": args.run_index,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
            "cuda_allocated_gb": (
                torch.cuda.memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            ),
        },
        "protocol": {
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "order_seed": order_seed,
            "execution_order": [workloads[i]["name"] for i in order],
            "monitor_interval_ms": args.monitor_interval_ms,
            "clock_domain": "host_monotonic_ns + torch.cuda.synchronize",
            "attention_backend": "eager",
            "dtype": "float16",
            "thermal_suspect_c": _THERMAL_SUSPECT_C,
            "min_power_samples": _MIN_POWER_SAMPLES,
            "measurement_reset": "per_request_fresh_kv",
        },
        "monitor": {
            "available": monitor_available,
            "num_records": len(monitor_records),
            "raw_telemetry_file": str(telemetry_path),
        },
        "totals": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
        "workloads": workload_records,
    }

    out_path = output_dir / f"run_{args.run_index}.json"
    out_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    logger.info("wrote %s (load=%.1fs, %d workloads)",
                out_path, load_time_s, len(workload_records))
    return 0


def _cross_run_summary(per_run_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce per-run request summaries to a cross-run median/min/max.

    With only 3 independent processes the cross-run dispersion is reported as
    min/max of the per-run medians rather than a formal confidence interval
    (E02-03 §6 / §9: state the sensitivity honestly).
    """
    metrics = [
        "ttft_ms", "tpot_ms", "e2e_ms",
        "prefill_tokens_per_s", "decode_tokens_per_s", "output_tokens_per_s",
    ]
    out: Dict[str, Any] = {}
    for metric in metrics:
        medians = [
            s[metric]["p50"] for s in per_run_summaries if metric in s and s[metric]["count"] > 0
        ]
        if not medians:
            out[metric] = {"median": None, "min": None, "max": None, "runs": 0}
            continue
        out[metric] = {
            "median": sorted(medians)[len(medians) // 2],
            "min": min(medians),
            "max": max(medians),
            "runs": len(medians),
        }
    return out


def _run_verify(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    run_files = sorted(output_dir.glob("run_*.json"))
    if len(run_files) < 3:
        logger.error("need >= 3 run_*.json files, found %d", len(run_files))
        return 2

    runs = [json.loads(p.read_text(encoding="utf-8")) for p in run_files]

    # Device placement across runs (E02-03 §4 "无隐式 offload" control).
    model_fully_on_gpu_all = all(
        run.get("actual_device", {}).get("model_fully_on_gpu") is True
        for run in runs
    )

    # Workload names from the first run's protocol order (all runs share the
    # same six workloads; execution order may differ by run).
    names = sorted({w["name"] for run in runs for w in run["workloads"]})

    per_workload: Dict[str, Any] = {}
    for name in names:
        all_samples: List[Dict[str, Any]] = []
        per_run: Dict[int, Dict[str, Any]] = {}
        all_hashes: List[str] = []
        anomalies: List[Dict[str, Any]] = []
        token_counts_exact = True
        for run in runs:
            record = next(w for w in run["workloads"] if w["name"] == name)
            samples = record["samples"]
            per_run[run["run_index"]] = request_summary(samples) if samples else {}
            anomalies.extend(record.get("anomalies", []))
            if len(samples) != record["repetitions"]:
                token_counts_exact = False
            for s in samples:
                all_samples.append(s)
                all_hashes.append(s["sequence_sha256"])
                ok = all(v is True for v in s["checks"].values() if isinstance(v, bool))
                if not ok:
                    token_counts_exact = False

        within_run = True
        for run in runs:
            record = next(w for w in run["workloads"] if w["name"] == name)
            hashes = {s["sequence_sha256"] for s in record["samples"]}
            if len(hashes) > 1:
                within_run = False

        distinct = set(all_hashes)
        cross_run_deterministic = len(distinct) == 1

        energy_available = any(
            next(w for w in run["workloads"] if w["name"] == name)
            .get("telemetry", {})
            .get("available")
            for run in runs
        )
        energy_usable = all(
            next(w for w in run["workloads"] if w["name"] == name)
            .get("telemetry", {})
            .get("window", {})
            .get("energy_usable", False)
            for run in runs
        )

        per_workload[name] = {
            "deterministic_within_run": within_run,
            "deterministic_cross_run": cross_run_deterministic,
            "distinct_sequence_hashes": len(distinct),
            "token_counts_exact": token_counts_exact,
            "total_samples": len(all_samples),
            "per_run": per_run,
            "cross_run": _cross_run_summary(list(per_run.values())),
            "anomalies": anomalies,
            "anomalies_have_reason": all(a.get("reason") for a in anomalies),
            "energy_available": energy_available,
            "energy_usable": energy_usable,
        }

    # ── Pass criteria (E02-03 单项通过标准) ──────────────────────────
    all_deterministic = all(
        per_workload[n]["deterministic_cross_run"] for n in names
    )
    all_within_run = all(
        per_workload[n]["deterministic_within_run"] for n in names
    )
    all_token_exact = all(per_workload[n]["token_counts_exact"] for n in names)
    all_anomalies_reasoned = all(
        per_workload[n]["anomalies_have_reason"] for n in names
    )

    verdict = {
        "runs": len(runs),
        "workloads": names,
        "repetitions_per_run": runs[0]["protocol"]["repetitions"],
        "per_workload": per_workload,
        "prefill_decode_separate": True,
        "model_fully_on_gpu_all_runs": model_fully_on_gpu_all,
        "three_independent_runs": len(runs) >= 3,
        "deterministic_cross_run_all": all_deterministic,
        "deterministic_within_run_all": all_within_run,
        "token_counts_exact_all": all_token_exact,
        "anomalies_reasoned_all": all_anomalies_reasoned,
        "passed": (
            len(runs) >= 3
            and all_deterministic
            and all_within_run
            and all_token_exact
            and all_anomalies_reasoned
        ),
    }

    out_path = output_dir / "verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    logger.info("wrote %s (passed=%s)", out_path, verdict["passed"])
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["passed"] else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E02-03 workload scaling baseline")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="load model and measure six workloads")
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--model-path", default="~/models/hqsb/Qwen3-1.7B")
    collect.add_argument(
        "--manifest",
        default=str(_REPO_ROOT / "docs" / "benchmark" / "model_sha256_manifest.txt"),
    )
    collect.add_argument("--workload-yaml", default=str(_WORKLOAD_YAML))
    collect.add_argument("--run-index", type=int, default=0)
    collect.add_argument("--warmup", type=int, default=_DEFAULT_WARMUP)
    collect.add_argument("--repetitions", type=int, default=_DEFAULT_REPETITIONS)
    collect.add_argument("--seed", type=int, default=_DEFAULT_ORDER_SEED)
    collect.add_argument(
        "--monitor-interval-ms", type=int, default=_DEFAULT_MONITOR_INTERVAL_MS
    )
    collect.set_defaults(func=_run_collect)

    verify = sub.add_parser("verify", help="cross-run reproducibility + scaling summary")
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
