#!/usr/bin/env python3
"""Rebuild an experiment report from its frozen plan and verified attempt ledger."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hqsb.experiments.artifacts import read_json
from hqsb.experiments.comparison import compare_campaign
from hqsb.observation.contracts import RUN_ID
from hqsb.observation.storage import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.campaign.resolve()
    plan = read_json(directory / "plan.json")
    ledger = read_json(directory / "ledger.json")["attempts"]
    groups, failed = {}, []
    for attempt in ledger:
        run_id = attempt["run_id"]
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("EXPERIMENT_LEDGER_PATH_INVALID")
        if attempt["returncode"] != 0 or not attempt.get("result_sha256"):
            failed.append(attempt)
            continue
        result_path = directory / run_id / "result.json"
        if result_path.is_symlink() or sha256(result_path) != attempt["result_sha256"]:
            raise ValueError("EXPERIMENT_RESULT_INTEGRITY")
        if attempt.get("result_path") != f"{run_id}/result.json":
            raise ValueError("EXPERIMENT_LEDGER_PATH_INVALID")
        for suffix in ("stdout", "stderr"):
            path = directory / f"{run_id}.{suffix}"
            if path.is_symlink() or sha256(path) != attempt[suffix + "_sha256"]:
                raise ValueError("EXPERIMENT_LOG_INTEGRITY")
        group = groups.setdefault((attempt["role"], attempt["pair"]), {})
        if attempt["arm"] in group:
            raise ValueError("EXPERIMENT_DUPLICATE_ARM")
        group[attempt["arm"]] = (run_id, read_json(result_path))
    pairs = []
    for (role, _), group in groups.items():
        if set(group) != {"baseline", "candidate"}:
            continue
        pairs.append({"role": role, "baseline_dir": str(directory / group["baseline"][0]),
                      "candidate_dir": str(directory / group["candidate"][0]),
                      "baseline_metrics": group["baseline"][1]["metrics"], "candidate_metrics": group["candidate"][1]["metrics"]})
    result = compare_campaign(plan, pairs)
    expected = plan["spec"]["performance_pairs"] + 2
    result.update(experiment_id=plan["experiment_id"], failed_attempts=failed,
                  ledger_sha256=sha256(directory / "ledger.json"),
                  completeness="complete" if not failed and len(pairs) == expected else "partial")
    if result["completeness"] != "complete":
        result["decision"] = "hold"
    if args.output.exists():
        raise FileExistsError("Retain existing derived reports; choose a new output")
    atomic_json(args.output, result)
    print(json.dumps({"quality": result["quality_status"], "performance": result["performance"]["status"],
                      "decision": result["decision"], "completeness": result["completeness"]}))


if __name__ == "__main__":
    main()
