#!/usr/bin/env python3
"""Administrator CLI only; no browser-controlled executable or shell."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hqsb.observation.tools import capture
from hqsb.observation.storage import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=["nsys", "ncu"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--launch-count", type=int, default=3)
    parser.add_argument("--kernel-filter", help="NCU demangled kernel-name regex")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = capture(
        args.tool,
        command,
        args.output,
        phase=args.phase,
        launch_count=args.launch_count,
        kernel_filter=args.kernel_filter,
    )
    csv = Path(args.output).resolve() / "metrics.csv"
    if args.tool == "ncu" and csv.exists() and csv.stat().st_size <= 16 * 1024**2:
        from hqsb.benchmark.multilevel_profiling import parse_ncu_csv

        panel = parse_ncu_csv(csv.read_text(), per_launch=True)
        panel.update(
            capture_id=result["capture_id"],
            measurement_role="diagnostic_not_performance",
            raw_sha256=next(
                (
                    f["sha256"]
                    for f in result["artifacts"]
                    if f["path"] == "metrics.csv"
                ),
                None,
            ),
        )
        atomic_json(csv.parent / "counter-panel.json", panel)
    print(json.dumps(result))
    return 0 if result["availability"] == "observed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
