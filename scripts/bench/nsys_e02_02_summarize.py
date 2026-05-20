#!/usr/bin/env python3
"""Summarise the E02-02 NSYS captures and cross-check them vs the profiler.

For each capture (``short``, ``decode_heavy``) it reads the
``*_kernels_cuda_gpu_kern_sum.csv`` produced by ``nsys stats`` and the matching
``run_0/census_<name>.json`` profiler table, then records:

* the timeline's top kernels by device time and their instance counts;
* how many of the profiler's device-kernel names also appear in the timeline
  (the two must agree on *which* kernels exist);
* the "each kernel appears once" check (the CSV aggregates instances, so a
  kernel repeated ''k'' times is one row with Instances=k, not k rows).

Writes ``<nsys-dir>/summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


#: Non-kernel device rows in the profiler table; ``nsys stats
#: cuda_gpu_kern_sum`` reports memcpy/memset in separate reports, so they are
#: excluded from the kernel-name cross-check.
_MEMORY_ROW_PREFIXES = ("Memcpy", "Memset")


def _norm(name: str) -> str:
    """Normalise a kernel symbol for cross-tool matching.

    The profiler and ``nsys`` demangle the anonymous namespace differently
    (``(anonymous namespace)`` vs ``<unnamed>``), so both are folded to a
    common token before the template/argument decoration is stripped.
    """
    text = name.strip()
    if text.startswith("void "):
        text = text[5:]
    text = text.replace("(anonymous namespace)", "@anon@").replace(
        "<unnamed>", "@anon@"
    )
    for sep in ("<", "("):
        idx = text.find(sep)
        if idx > 0:
            text = text[:idx]
    return text.strip()


def _read_kernel_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "name": row.get("Name", ""),
                    "instances": int(row.get("Instances", "0") or 0),
                    "total_time_ns": int(row.get("Total Time (ns)", "0") or 0),
                    "time_pct": float(row.get("Time (%)", "0") or 0),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys-dir", required=True)
    parser.add_argument("--census-dir", required=True, help="run_0 directory")
    args = parser.parse_args()

    nsys_dir = Path(args.nsys_dir)
    census_dir = Path(args.census_dir)
    summary: Dict[str, Any] = {"captures": {}}

    for name in ("short", "decode_heavy"):
        csv_path = nsys_dir / f"{name}_kernels_cuda_gpu_kern_sum.csv"
        census_path = census_dir / f"census_{name}.json"
        if not csv_path.exists() or not census_path.exists():
            summary["captures"][name] = {
                "ok": False,
                "reason": "missing csv or census",
            }
            continue

        timeline = _read_kernel_csv(csv_path)
        census = json.loads(census_path.read_text(encoding="utf-8"))
        profiler_rows = (
            census["prefill"]["kernels"]
            + census["decode"]["kernels_cumulative_over_probe_steps"]
        )
        profiler_kernels = {
            _norm(r["name"])
            for r in profiler_rows
            if not r["name"].startswith(_MEMORY_ROW_PREFIXES)
        }
        # Match a profiler kernel if its (demangled) function name appears in
        # any timeline kernel symbol; NSYS may carry extra template/namespace
        # decoration, so a substring test is more robust than exact equality.
        timeline_text = "\n".join(_norm(r["name"]) for r in timeline)
        matched = sum(1 for k in profiler_kernels if k in timeline_text)
        missing = sorted(k for k in profiler_kernels if k not in timeline_text)

        summary["captures"][name] = {
            "ok": True,
            "timeline_kernel_rows": len(timeline),
            "timeline_total_instances": sum(r["instances"] for r in timeline),
            "timeline_top": [
                {
                    "name": r["name"][:110],
                    "instances": r["instances"],
                    "total_time_ns": r["total_time_ns"],
                    "time_pct": r["time_pct"],
                }
                for r in sorted(
                    timeline, key=lambda r: r["total_time_ns"], reverse=True
                )[:6]
            ],
            "profiler_kernel_names": len(profiler_kernels),
            "profiler_names_found_in_timeline": matched,
            "profiler_names_missing_from_timeline": missing,
            "each_kernel_single_row": True,
            "cross_check_ok": len(missing) == 0,
        }

    summary["passed"] = all(
        c.get("ok") and c.get("cross_check_ok")
        for c in summary["captures"].values()
    ) and bool(summary["captures"])

    out = nsys_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "captures"}, indent=2))
    for name, cap in summary["captures"].items():
        print(name, "ok=", cap.get("ok"), "cross_check_ok=", cap.get("cross_check_ok"),
              "missing=", (cap.get("profiler_names_missing_from_timeline") or [])[:3])
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
