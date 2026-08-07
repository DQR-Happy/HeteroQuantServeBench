#!/usr/bin/env python3
"""HQSB S09 read-only preflight and evidence campaign driver.

Examples:
    python3 scripts/ascend/run_e09.py --list --json
    python3 scripts/ascend/run_e09.py --prerequisites --json
    python3 scripts/ascend/run_e09.py --interface-map --json
    python3 scripts/ascend/run_e09.py --smoke --json
    python3 scripts/ascend/run_e09.py --collect --json

``--collect`` writes formal evidence, but cannot turn an absent Ascend device or
CANN stack into PASS.  Collector success and scientific verdict are separate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Mapping

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hqsb.ascend import experiment as exp  # noqa: E402
from hqsb.ascend import interface_map as imap  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HQSB S09 campaign collector (hardware-gated, no fabricated conclusions)"
    )
    parser.add_argument("--list", action="store_true", help="list the ten protocols")
    parser.add_argument("--prerequisites", action="store_true", help="run read-only host capability probes")
    parser.add_argument("--interface-map", action="store_true", help="verify 280 detail steps have a collection surface")
    parser.add_argument("--smoke", action="store_true", help="run CPU host-contract smoke; never hardware evidence")
    parser.add_argument("--collect", action="store_true", help="persist the preflight campaign and ten verdicts")
    parser.add_argument("--experiment", choices=exp.EXPERIMENTS, help="limit --collect to one experiment")
    parser.add_argument("--run-id", default="", help="single path component; timestamp when omitted")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def _emit(payload: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(dict(payload), sort_keys=True, indent=2, ensure_ascii=False))
        return
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            print(f"{key}:")
            print(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False))
        else:
            print(f"{key}: {value}")


def main() -> int:
    args = _parser().parse_args()
    chosen = sum(bool(value) for value in (args.list, args.prerequisites, args.interface_map, args.smoke, args.collect))
    if chosen != 1:
        _parser().error("choose exactly one of --list/--prerequisites/--interface-map/--smoke/--collect")
    if args.experiment and not args.collect:
        _parser().error("--experiment is only valid with --collect")

    if args.list:
        _emit(exp.protocol_catalog(), as_json=args.json)
    elif args.prerequisites:
        _emit(exp.collect_capability_snapshot(REPO_ROOT).as_dict(), as_json=args.json)
    elif args.interface_map:
        _emit(imap.resolve_interfaces(REPO_ROOT), as_json=args.json)
    elif args.smoke:
        _emit(exp.smoke_self_check(), as_json=args.json)
    else:
        selected = (args.experiment,) if args.experiment else exp.EXPERIMENTS
        summary = exp.collect_campaign(
            REPO_ROOT,
            experiment_ids=selected,
            run_id=args.run_id,
        )
        _emit(summary, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
