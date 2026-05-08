#!/usr/bin/env python3
"""Analyze profiler output into a Roofline/Amdahl hotspot decision report.

Reads a ``hotspot_summary.json`` (produced by ``profile_model.py``) and the
per-case operator tables, then:

1. classifies each top operator (GEMM/elementwise/runtime/memory);
2. computes the Amdahl upper bound per case;
3. renders a hotspot ranking and the S03 candidate decision.

Usage:
    python scripts/bench/analyze_hotspots.py \
        --hotspots reports/dev/profiler/s02/hotspot_summary.json \
        --output reports/dev/profiler/s02/hotspot_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from hqsb.benchmark.roofline import (
    ORIN_NANO_SUPER_FP16,
    HotspotClass,
    amdahl_max_speedup,
    classify_hotspot,
)

_GEMM_SHARE_THRESHOLD = 0.5


def _load(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def analyze(hotspots: Dict[str, Any]) -> Dict[str, Any]:
    """Produce an analysis document from a hotspot summary."""
    analysis: Dict[str, Any] = {
        "roofline_model": {
            "name": ORIN_NANO_SUPER_FP16.name,
            "peak_flops": ORIN_NANO_SUPER_FP16.peak_flops,
            "peak_bandwidth": ORIN_NANO_SUPER_FP16.peak_bandwidth,
            "ridge_point_flop_per_byte": ORIN_NANO_SUPER_FP16.ridge_point(),
        },
        "cases": {},
    }

    for case_name, case in hotspots.items():
        ops = case.get("top_operators", [])
        gemm_share = sum(
            o["time_share"]
            for o in ops
            if o["classification"] == HotspotClass.GEMM_ATTENTION.value
        )
        analysis["cases"][case_name] = {
            "total_cuda_time_us": case.get("total_cuda_time_us"),
            "top_operator": ops[0]["name"] if ops else None,
            "top_operator_share": ops[0]["time_share"] if ops else 0.0,
            "gemm_share": round(gemm_share, 4),
            "gemm_dominated": gemm_share >= _GEMM_SHARE_THRESHOLD,
            "amdahl_max_top_op": (
                amdahl_max_speedup(min(ops[0]["time_share"], 1.0)) if ops else 1.0
            ),
            "ranking": [
                {
                    "name": o["name"],
                    "share": o["time_share"],
                    "classification": o["classification"],
                    "amdahl_max": o["amdahl_max"],
                }
                for o in ops[:10]
            ],
        }

    analysis["decision"] = _decision(analysis)
    return analysis


def _decision(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Render the S03 hotspot decision from the aggregated analysis."""
    cases = analysis["cases"]
    decode_gemm = cases.get("decode_heavy", {}).get("gemm_share", 0.0)
    prefill_gemm = cases.get("prefill_heavy", {}).get("gemm_share", 0.0)

    # GEMM is the decision target when it dominates decode (>= 50%) or is the
    # single largest prefill class (>= 40%, alongside elementwise fusion).
    gemm_primary = decode_gemm >= 0.5 or prefill_gemm >= 0.4

    top_ops = [
        (c["top_operator"], c["top_operator_share"])
        for c in cases.values()
        if c["top_operator"]
    ]
    top_name, top_share = (max(top_ops, key=lambda x: x[1]) if top_ops else (None, 0.0))

    if gemm_primary:
        strategy = (
            "GEMM dominates decode (%.0f%%) and is the largest prefill class "
            "(%.0f%%). Do NOT write GEMM from scratch; target CUTLASS/低比特 "
            "(int4/int8) GEMM + epilogue fusion. Keep RMSNorm as the "
            "elementwise/reduction teaching loop, and fuse aten::copy_/mul/cat "
            "to cut launch/sync overhead."
            % (decode_gemm * 100, prefill_gemm * 100)
        )
    else:
        strategy = (
            "GEMM is not dominant; prioritize the elementwise/reduction and "
            "runtime/sync overheads identified in the ranking."
        )

    return {
        "strategy": strategy,
        "decode_gemm_share": round(decode_gemm, 4),
        "prefill_gemm_share": round(prefill_gemm, 4),
        "top_operator": top_name,
        "top_operator_share": round(top_share, 4),
        "candidates": [
            {
                "name": "GEMM (CUTLASS/低比特)",
                "rationale": "GEMM is the dominant decode hotspot (%.0f%%); "
                "low-bit GEMM + epilogue fusion give the largest Amdahl "
                "ceiling." % (decode_gemm * 100),
            },
            {
                "name": "RMSNorm (teaching loop)",
                "rationale": "Fixed teaching closed-loop for the elementwise/"
                "reduction path; small share but high educational value.",
            },
            {
                "name": "KV-cache / elementwise fusion",
                "rationale": "aten::copy_/mul/cat add up; fusing them reduces "
                "launch/sync and memory traffic.",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Hotspot analysis")
    parser.add_argument("--hotspots", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    analysis = analyze(_load(args.hotspots))
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2, default=str)

    print(json.dumps(analysis, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
