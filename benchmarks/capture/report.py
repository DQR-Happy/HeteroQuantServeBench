"""Read-only aggregation: process-block uncertainty, availability, memory scope."""

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks.capture.oracle import quantiles
from benchmarks.capture.worker import RECEIPT


def interval(values, seed=20261017):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "median": None, "ci95": None}
    if len(values) < 3:
        return {"n": len(values), "median": statistics.median(values), "ci95": None}
    rng = random.Random(seed)
    sampled = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(10_000))
    return {"n": len(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "ci95": [sampled[250], sampled[9750]],
            "estimand": "median of independent process-run scalar; 3-run CI is exploratory"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", action="append", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    groups, all_rows, failures = defaultdict(list), [], []
    for campaign in args.campaign:
        campaign = campaign.resolve()
        if (campaign / "superseded.json").exists():
            raise ValueError(f"refusing superseded campaign {campaign}")
        for path in sorted(campaign.glob("*/metrics.json")):
            row = json.loads(path.read_text())
            if "cohort_completion_rate" not in row:
                continue
            row["campaign"] = campaign.name
            raw = json.loads((path.parent / "worker.json").read_text())
            terminal = raw["terminal"]
            if "published_ns" in terminal:
                row["publication_ms"] = (terminal["published_ns"] - terminal["close_start_ns"]) / 1e6
                row["supervisor_publication_ms"] = ((terminal["published_ns"] - terminal["publication_start_ns"]) / 1e6
                                                    if "publication_start_ns" in terminal else None)
            row["pacer"] = raw["case"].get("pacer", "native_clock_yield" if raw["track"] == "NATIVE_CORE" else "hybrid_spin")
            row["repeat"] = raw["case"].get("repeat", raw["case"].get("seed", 100) - 100
                                              if raw["track"] == "NATIVE_CORE" else 0)
            by_result = defaultdict(list)
            for receipt in path.parent.glob("receipts-*.bin"):
                with receipt.open("rb") as stream:
                    while block := stream.read(RECEIPT.size):
                        _i, begin, _scheduled, duration, code = RECEIPT.unpack(block)
                        if begin:
                            by_result[str(code)].append(duration)
            row["call_latency_by_result_code"] = {code: quantiles(values) for code, values in by_result.items()}
            timeline = path.parent / "process-timeline.json"
            if timeline.exists():
                samples = json.loads(timeline.read_text())
                samples = [s for s in samples if raw["t0_ns"] <= s["monotonic_ns"] <= raw["endpoint_end_ns"]]
                row["sampled_peak_PSS_KiB"] = max((s["smaps_rollup:Pss"] for s in samples if "smaps_rollup:Pss" in s), default=None)
            else:
                row["sampled_peak_PSS_KiB"] = None
            row["raw_metrics_path"] = str(path)
            key = (campaign.name, row["case_id"], row["adapter"], row["pacer"], row["measurement_pass"])
            groups[key].append(row)
            all_rows.append(row)
        for path in sorted(campaign.glob("*/failure.json")):
            failures.append(dict(json.loads(path.read_text()), campaign=campaign.name, path=str(path)))
    summary = []
    for key, rows in sorted(groups.items()):
        item = dict(zip(("campaign", "case_id", "adapter", "pacer", "measurement_pass"), key))
        item["statuses"] = [r["status"] for r in rows]
        for field in ("cohort_completion_rate", "scheduled_delivery_ratio", "retained_ratio", "cpu_ns_per_valid_event",
                      "sampled_peak_PSS_KiB", "process_maxrss_KiB", "encoding_space_ratio", "close_ms", "publication_ms"):
            item[field] = interval([r.get(field) for r in rows])
        for q in ("p50_ns", "p95_ns", "p99_ns", "p999_ns"):
            item["call_" + q] = interval([r["call_latency"].get(q) for r in rows])
        item["notes"] = "stock buffering/overload policies differ; CPU/runtime/inputs/endpoints fixed per campaign; not a maximum-capacity result"
        summary.append(item)
    (output / "all-runs-enriched.json").write_text(json.dumps(all_rows, indent=2))
    (output / "aggregate.json").write_text(json.dumps(summary, indent=2))
    (output / "failures.json").write_text(json.dumps(failures, indent=2))
    fields = ["campaign", "case_id", "adapter", "pacer", "measurement_pass", "n", "goodput_events_s",
              "scheduled_delivery_ratio", "call_p50_ns", "call_p99_ns", "sampled_PSS_MiB", "CPU_ns_per_valid", "close_ms"]
    with (output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            med = lambda key, row=row: row[key]["median"]
            pss = med("sampled_peak_PSS_KiB")
            writer.writerow({"campaign": row["campaign"], "case_id": row["case_id"], "adapter": row["adapter"],
                "pacer": row["pacer"],
                "measurement_pass": row["measurement_pass"], "n": row["cohort_completion_rate"]["n"],
                "goodput_events_s": med("cohort_completion_rate"), "scheduled_delivery_ratio": med("scheduled_delivery_ratio"),
                "call_p50_ns": med("call_p50_ns"), "call_p99_ns": med("call_p99_ns"),
                "sampled_PSS_MiB": pss / 1024 if pss is not None else None,
                "CPU_ns_per_valid": med("cpu_ns_per_valid_event"), "close_ms": med("close_ms")})
    inventory = json.loads((Path(__file__).parent / "spec/metrics.json").read_text())
    available = {"offered_rate", "attempt_rate", "scheduled_not_attempted", "admitted_rate", "cohort_completion_rate",
        "logical_bandwidth", "call_latency", "dispatch_lag", "retained_ratio", "not_retained_ratio", "counter_reconciliation",
        "duplicate_records", "semantic_corruption", "foreign_or_cross_epoch_records", "ordering_violations",
        "cpu_ns_per_valid_event", "cpu_ns_per_logical_byte", "process_memory", "compression_ratio", "system_activity",
        "normalized_jain_fairness", "startup_latency", "close_latency", "publication_latency"}
    coverage = []
    for metric in inventory["metrics"]:
        mid = metric["id"]
        coverage.append({"id": mid, "status": "measured_or_derived_when_supported" if mid in available else "not_available_in_common_stock_adapter_track",
            "scope": "admitted/counter reconciliation: HQSB only; PSS sampled, RSS process scope; system_activity only rusage counters; no general PMU; publication only compatible HQSB captures",
            "reason": None if mid in available else "requires missing stock hooks, isolated collector accounting, other retention policy, capacity search, or real GPU model; service track completion timing reported separately"})
    (output / "metric-availability.json").write_text(json.dumps(coverage, indent=2))
    print(json.dumps({"valid_metric_runs": len(all_rows), "failed_runs": len(failures), "groups": len(summary)}))


if __name__ == "__main__":
    main()
