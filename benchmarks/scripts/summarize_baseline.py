#!/usr/bin/env python3
"""Baseline results summarizer.

Aggregates multiple JSON benchmark result files into a CSV summary
table, computing averages across repetitions and extracting key
metrics for comparison.

Usage:
    python benchmarks/scripts/summarize_baseline.py \\
        --input-dir reports/dev/llm/20260101_120000
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional


def _avg_across_repetitions(runs: List[Dict[str, Any]], key: str) -> float:
    """Compute the arithmetic mean of a metric across repetitions."""
    values = [run[key] for run in runs if key in run]
    if not values:
        return 0.0
    return statistics.mean(values)


def _max_across_repetitions(runs: List[Dict[str, Any]], key: str) -> float:
    """Compute the maximum of a metric across repetitions."""
    values = [run[key] for run in runs if key in run]
    if not values:
        return 0.0
    return max(values)


def _safe_get(obj: Dict[str, Any], *keys: str, default: Any = 0.0) -> Any:
    """Safely navigate nested dict keys."""
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key, default)
        else:
            return default
    return obj


def summarize_directory(input_dir: str) -> List[Dict[str, Any]]:
    """Summarize all JSON benchmark results in a directory.

    Args:
        input_dir: Path to directory containing ``*.json`` result files.

    Returns:
        List of row dictionaries suitable for CSV writing.
    """
    rows: List[Dict[str, Any]] = []

    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    for path in json_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        runs = data.get("repetitions", [])
        if not runs:
            continue

        case_name = os.path.basename(path).replace(".json", "")

        # ── Core latency metrics ──────────────────────────────────
        row: Dict[str, Any] = {
            "case": case_name,
            "input_tokens": data["workload"]["input_tokens"],
            "output_tokens": data["workload"]["output_tokens"],
            "prefill_ms": _avg_across_repetitions(runs, "prefill_forward_ms"),
            "first_token_selection_ms": _avg_across_repetitions(
                runs, "first_token_selection_ms"
            ),
            "model_core_ttft_ms": _avg_across_repetitions(runs, "model_core_ttft_ms"),
            "decode_total_ms": _avg_across_repetitions(runs, "decode_total_ms"),
            "e2e_ms": _avg_across_repetitions(runs, "model_core_e2e_ms"),
            "prefill_tps": _avg_across_repetitions(runs, "prefill_tokens_per_s"),
            "decode_tps": _avg_across_repetitions(runs, "decode_tokens_per_s"),
            "output_tps": _avg_across_repetitions(
                runs, "model_core_output_tokens_per_s"
            ),
        }

        # ── ITL statistics (from last repetition) ─────────────────
        last_run = runs[-1]
        itl_data = last_run.get("itl", {})
        row["itl_mean_ms"] = itl_data.get("mean_ms", 0.0)
        row["itl_median_ms"] = itl_data.get("median_ms", 0.0)
        row["itl_p50_ms"] = itl_data.get("p50_ms", 0.0)
        row["itl_p95_ms"] = itl_data.get("p95_ms", 0.0)
        row["itl_p99_ms"] = itl_data.get("p99_ms", 0.0)
        row["itl_stddev_ms"] = itl_data.get("stddev_ms", 0.0)

        # ── Memory metrics ────────────────────────────────────────
        row["peak_cuda_allocated_mb"] = _max_across_repetitions(
            runs, "peak_cuda_allocated_mb"
        )
        row["peak_cuda_reserved_mb"] = _max_across_repetitions(
            runs, "peak_cuda_reserved_mb"
        )
        row["process_rss_mb"] = _max_across_repetitions(runs, "process_rss_mb")
        row["system_memory_used_mb"] = _max_across_repetitions(
            runs, "system_memory_used_mb"
        )

        # ── Power & Energy ────────────────────────────────────────
        row["avg_power_w"] = _safe_get(last_run, "tegrastats_power", "avg_power_w")
        row["peak_power_w"] = _safe_get(last_run, "tegrastats_power", "peak_power_w")
        row["energy_j"] = _safe_get(last_run, "tegrastats_power", "energy_j")
        row["energy_j_per_output_token"] = last_run.get("energy_j_per_output_token", 0.0)

        # ── GPU utilization ───────────────────────────────────────
        row["avg_gpu_util_pct"] = _safe_get(
            last_run, "tegrastats_summary", "avg_gpu_util_pct"
        )
        row["peak_gpu_util_pct"] = _safe_get(
            last_run, "tegrastats_summary", "peak_gpu_util_pct"
        )
        row["peak_gpu_temp_c"] = _safe_get(
            last_run, "tegrastats_summary", "peak_gpu_temp_c"
        )

        # ── Determinism ───────────────────────────────────────────
        row["deterministic"] = data.get("deterministic", False)

        # ── Reproducibility ───────────────────────────────────────
        row["model_id"] = _safe_get(data, "model", "id")
        row["dtype"] = _safe_get(data, "model", "dtype")
        row["backend"] = _safe_get(data, "model", "backend")
        row["attention_backend"] = _safe_get(data, "model", "attention_backend")
        row["torch_version"] = _safe_get(data, "software", "torch")
        row["cuda_version"] = _safe_get(data, "software", "torch_cuda")
        row["transformers_version"] = _safe_get(data, "software", "transformers")
        row["modelscope_version"] = _safe_get(data, "software", "modelscope")
        row["device"] = _safe_get(data, "hardware", "device")

        rows.append(row)

    return rows


def main() -> None:
    """Parse arguments and generate CSV summary."""
    parser = argparse.ArgumentParser(
        description="Summarize HQSB benchmark results into CSV",
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing JSON benchmark result files",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV file path (default: <input-dir>/summary.csv)",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Error: directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    rows = summarize_directory(input_dir)

    if not rows:
        print(f"No JSON result files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or os.path.join(input_dir, "summary.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary written to: {output_path}")
    print(f"  {len(rows)} cases summarized")

    # Print a quick table to stdout
    print()
    header = (
        f"{'Case':<16s} {'ISL':>5s} {'OSL':>5s} "
        f"{'TTFT(ms)':>10s} {'E2E(ms)':>10s} "
        f"{'Prefill(t/s)':>13s} {'Decode(t/s)':>12s} "
        f"{'Output(t/s)':>11s} {'Det':>5s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['case']:<16s} "
            f"{row['input_tokens']:>5d} "
            f"{row['output_tokens']:>5d} "
            f"{row['model_core_ttft_ms']:>10.2f} "
            f"{row['e2e_ms']:>10.2f} "
            f"{row['prefill_tps']:>13.1f} "
            f"{row['decode_tps']:>12.1f} "
            f"{row['output_tps']:>11.1f} "
            f"{'True' if row['deterministic'] else 'FALSE':>5s}"
        )


if __name__ == "__main__":
    main()
