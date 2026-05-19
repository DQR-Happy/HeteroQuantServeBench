#!/usr/bin/env python3
"""Read-only extractor for the E02-02 report.

Prints the compact numbers the report cites (scope totals, top ATen ops and
top device kernels per phase, coverage, identity), so the report never
hand-copies values out of the raw JSON.

Usage:
    python3 scripts/bench/e02_02_report_data.py --dir docs/stage_experiments/S02/E02-02/raw_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _top(rows, n=6):
    return [
        {
            "name": r["name"][:80],
            "count": r["count"],
            "cuda_ms": round(r["cuda_time_us"] / 1000.0, 3),
            "share_pct": round(r["time_share"] * 100, 2),
        }
        for r in sorted(rows, key=lambda r: r["cuda_time_us"], reverse=True)[:n]
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--workload", default="short")
    args = parser.parse_args()

    base = Path(args.dir)
    census = json.loads((base / "run_0" / f"census_{args.workload}.json").read_text())
    verdict = json.loads((base / "verdict.json").read_text())
    meta = json.loads((base / "run_0" / "run_meta.json").read_text())

    pre, dec = census["prefill"], census["decode"]
    out = {
        "identity": meta["identity"],
        "workload": census["workload"],
        "decode_steps": census["decode_steps"],
        "decode_probe_steps": census["decode_probe_steps"],
        "token_parity": census["token_parity"],
        "coverage": census["coverage"],
        "scope_check": census["scope_check"],
        "prefill": {
            "probe_scope": pre["probe_scope"],
            "aten_top": _top(pre["aten_ops"]),
            "kernel_top": _top(pre["kernels"]),
        },
        "decode": {
            "probe_scope": dec["probe_scope"],
            "aten_top": _top(dec["aten_ops_cumulative_over_probe_steps"]),
            "kernel_top": _top(dec["kernels_cumulative_over_probe_steps"]),
        },
        "verdict_summary": {
            "passed": verdict["passed"],
            "nsys_audit_present": verdict["nsys_audit_present"],
            "workloads": {
                n: {
                    "sig": c["module_signature_identical"],
                    "tok": c["token_hash_identical"],
                    "calls": c["call_counts_identical"],
                    "cov": c["coverage_ok"],
                    "parity": c["token_parity_all"],
                    "scopes": c["scopes_closed"],
                }
                for n, c in verdict["workloads"].items()
            },
        },
    }
    nsys_summary = base / "nsys" / "summary.json"
    if nsys_summary.exists():
        ns = json.loads(nsys_summary.read_text())
        out["nsys"] = {
            "passed": ns["passed"],
            "captures": {
                k: {
                    "cross_check_ok": v.get("cross_check_ok"),
                    "timeline_kernel_rows": v.get("timeline_kernel_rows"),
                    "timeline_total_instances": v.get("timeline_total_instances"),
                    "top": v.get("timeline_top", [])[:3],
                }
                for k, v in ns["captures"].items()
            },
        }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
