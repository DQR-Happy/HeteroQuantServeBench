#!/usr/bin/env python3
"""Pre-registered, independent-process G01-G08 campaign. Remote only."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hqsb.experiments.plans import role_request, validate_spec
from hqsb.experiments.comparison import compare_campaign
from hqsb.observation.storage import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--kernel-manifest")
    parser.add_argument("--model-path", default="~/models/hqsb/Qwen3-1.7B")
    parser.add_argument("--output-root", default="experiment_results/EXP")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if not 30 <= args.timeout <= 1800:
        parser.error("Timeout must be 30..1800 seconds")
    plan = validate_spec(json.loads(Path(args.spec).read_text()))
    experiment_id = "exp_" + uuid.uuid4().hex
    directory = Path(args.output_root).resolve() / experiment_id
    directory.mkdir(parents=True, exist_ok=False)
    plan.update(experiment_id=experiment_id, registered_at=time.time())
    atomic_json(directory / "plan.json", plan)
    atomic_json(directory / "quality-plan.json", plan["spec"]["quality_plan"])
    ledger, pairs = [], []
    for role, count in (("performance", plan["spec"]["performance_pairs"]), ("diagnostic", 1), ("quality", 1)):
        for index in range(count):
            runs = {}
            order = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
            for arm in order:
                run_id = "run_" + uuid.uuid4().hex
                payload = role_request(plan, arm, role)
                argv = [sys.executable, str(ROOT / "scripts/observation/run_model_observation.py"),
                        "--model-path", args.model_path, "--manifest", args.manifest,
                        "--mode", payload["observation_mode"], "--isl", str(plan["spec"]["input_tokens"]),
                        "--osl", str(payload["max_output_tokens"]), "--run-id", run_id,
                        "--output-root", str(directory), "--experiment-role", role,
                        "--optimization-json", json.dumps(payload["optimization"]),
                        "--experiment-links-json", json.dumps({"campaign_id": experiment_id, "spec_sha256": plan["spec_sha256"],
                                                              "pair_id": str(index), "arm": arm, "role": role})]
                if payload.get("observation_plan"):
                    argv.extend(["--observation-plan-json", json.dumps(payload["observation_plan"])])
                if args.kernel_manifest:
                    argv.extend(["--kernel-manifest", args.kernel_manifest])
                if role == "quality":
                    argv.extend(["--quality-plan", str(directory / "quality-plan.json")])
                stdout, stderr = directory / f"{run_id}.stdout", directory / f"{run_id}.stderr"
                started = time.monotonic()
                with stdout.open("w") as out, stderr.open("w") as err:
                    process = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=True)
                    timed_out = False
                    try:
                        code = process.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            code = process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            code = process.wait(timeout=5)
                item = {"run_id": run_id, "arm": arm, "role": role, "pair": index, "argv": argv,
                        "returncode": code, "timeout": timed_out, "elapsed_s": time.monotonic() - started,
                        "stdout_sha256": sha256(stdout), "stderr_sha256": sha256(stderr)}
                result_path = directory / run_id / "result.json"
                if code == 0 and result_path.exists():
                    item["result_sha256"] = sha256(result_path)
                    runs[arm] = json.loads(result_path.read_text())
                    item["result_path"] = str(result_path.relative_to(directory))
                else:
                    item["status"] = "failed_or_incomplete"
                    failure = directory / run_id / "failure.json"
                    item["failure_sha256"] = sha256(failure) if failure.exists() else None
                ledger.append(item)
                atomic_json(directory / "ledger.json", {"attempts": ledger})
                print(json.dumps({"run_id": run_id, "arm": arm, "role": role, "returncode": code}), flush=True)
            if set(runs) == {"baseline", "candidate"}:
                ids = {entry["arm"]: entry["run_id"] for entry in ledger if entry["role"] == role and entry["pair"] == index}
                pairs.append({"role": role, "baseline_dir": str(directory / ids["baseline"]),
                              "candidate_dir": str(directory / ids["candidate"]),
                              "baseline_metrics": runs["baseline"]["metrics"], "candidate_metrics": runs["candidate"]["metrics"]})
    result = compare_campaign(plan, pairs)
    failures = [item for item in ledger if item["returncode"] != 0 or item.get("status") == "failed_or_incomplete"]
    incomplete = any(reason in result["reasons"] for reason in ("INCOMPLETE_EXECUTION", "DIAGNOSTIC_COVERAGE_MISMATCH"))
    result.update(experiment_id=experiment_id, failed_attempts=failures,
                  completeness="partial" if failures or incomplete else "complete")
    if failures:
        result["decision"] = "hold"
    atomic_json(directory / "report.json", result)
    print(json.dumps({"directory": str(directory), "quality": result["quality_status"],
                      "performance": result["performance"]["status"], "decision": result["decision"],
                      "completeness": result["completeness"]}), flush=True)
    return 1 if failures or incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
