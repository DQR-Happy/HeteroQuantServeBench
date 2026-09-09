"""Read-only SERVICE/MooBench summaries, kept separate from offer() rankings."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from benchmarks.capture.report import interval


def summarize_service(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["rate"], row["batch_events"])].append(row)
    result = []
    for (mode, rate, batch), group in sorted(groups.items()):
        item = {"mode": mode, "planned_events_s": rate, "batch_events": batch,
                "runs": len(group), "statuses": [r["status"] for r in group]}
        for name in ("durable_goodput", "scheduled_delivery_ratio", "scheduled_not_attempted"):
            item[name] = interval([r.get(name) for r in group])
        for field in ("scheduled_durable_latency", "HTTP_batch_latency", "dispatch_lag"):
            for q in ("p50_ns", "p99_ns"):
                item[f"{field}_{q}"] = interval([r.get(field, {}).get(q) for r in group])
        item["bad_records"] = sum(sum(r.get("counts", {}).get(k, 0)
                                      for k in ("corrupt", "duplicate", "foreign")) for r in group)
        item["scope"] = "sequential sender + codec + loopback HTTP + fsync sink; direct is control, not engine capacity"
        result.append(item)
    return result


def summarize_moobench(rows):
    groups = defaultdict(list)
    off = {(r["depth"], r["repeat"]): r["mean_method_ns"] for r in rows if r["mode"] == "off"}
    for row in rows:
        groups[(row["depth"], row["mode"])].append(row)
    result = []
    for (depth, mode), group in sorted(groups.items()):
        item = {"recursion_depth": depth, "mode": mode, "runs": len(group),
                "mean_method_ns": interval([r["mean_method_ns"] for r in group]),
                "top_level_p50_ns": interval([r["method_call_latency"].get("p50_ns") for r in group]),
                "paired_mean_overhead_ns": interval([r["mean_method_ns"] - off[depth, r["repeat"]] for r in group]),
                "paired_mean_ratio_to_off": interval([r["mean_method_ns"] / off[depth, r["repeat"]] for r in group]),
                "exit_codes": [r["exit_code"] for r in group],
                "recorded_calls_match": all(r["recorded_calls"] == r["calls"] for r in group),
                "spans_received": sum(r["receiver"]["spans"] for r in group),
                "span_duplicates": sum(r["receiver"]["duplicates"] for r in group),
                "expected_spans": sum(r.get("expected_spans") or 0 for r in group),
                "scope": "unmodified upstream Python recursive SuT; local Zipkin-protocol receiver, not Zipkin database or disk durability"}
        result.append(item)
    return result


def write_csv(path, rows, fields):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key]["median"] if isinstance(row[key], dict) else row[key] for key in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", type=Path, required=True, help="all-runs.json")
    parser.add_argument("--moobench", type=Path, required=True, help="all-runs.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    service = summarize_service(json.loads(args.service.read_text()))
    moo = summarize_moobench(json.loads(args.moobench.read_text()))
    (args.output / "service-aggregate.json").write_text(json.dumps(service, indent=2))
    (args.output / "moobench-aggregate.json").write_text(json.dumps(moo, indent=2))
    write_csv(args.output / "service.csv", service,
              ["mode", "planned_events_s", "batch_events", "runs", "durable_goodput",
               "scheduled_delivery_ratio", "scheduled_durable_latency_p50_ns", "scheduled_durable_latency_p99_ns", "bad_records"])
    write_csv(args.output / "moobench.csv", moo,
              ["recursion_depth", "mode", "runs", "mean_method_ns", "top_level_p50_ns",
               "paired_mean_overhead_ns", "paired_mean_ratio_to_off", "spans_received", "expected_spans", "span_duplicates"])


if __name__ == "__main__":
    main()
