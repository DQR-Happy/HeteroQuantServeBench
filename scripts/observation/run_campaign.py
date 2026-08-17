#!/usr/bin/env python3
"""Paired independent-process experiments, retaining every failure and raw run."""

import argparse
import json
import os
import signal
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hqsb.observation.storage import atomic_json, sha256
from hqsb.observation.analysis import analyze, compare
from hqsb.evaluation.observation_comparison import paired_uncertainty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--isl", type=int, default=128)
    parser.add_argument("--osl", type=int, default=64)
    parser.add_argument(
        "--kind", choices=["overhead", "attention", "cleanup"], default="overhead"
    )
    parser.add_argument(
        "--mode", choices=["basic", "memory", "pages", "operators"], default="basic"
    )
    parser.add_argument("--output-root", default="experiment_results/OBS/campaigns")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    if not 1 <= args.pairs <= 100:
        parser.error("pairs must be 1..100")
    root = Path(args.output_root).resolve() / ("campaign_" + uuid.uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    spec = {
        **vars(args),
        "measurement_role": "overhead" if args.kind == "overhead" else "mechanism",
        "quality_status": "not_evaluated",
        "seed": 0,
        "allocator_environment": {
            k: os.environ.get(k)
            for k in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_NO_CUDA_MEMORY_CACHING")
        },
    }
    atomic_json(root / "comparison-spec.json", spec)
    rows, pairs = [], []
    for pair in range(args.pairs):
        members = {}
        for candidate in ["A", "B"] if pair % 2 == 0 else ["B", "A"]:
            mode = (
                ("off" if candidate == "A" else args.mode)
                if args.kind == "overhead"
                else args.mode
            )
            attention = (
                "sdpa" if args.kind == "attention" and candidate == "B" else "eager"
            )
            run_id = "run_" + uuid.uuid4().hex
            command = [
                sys.executable,
                str(ROOT / "scripts/observation/run_model_observation.py"),
                "--mode",
                mode,
                "--attention",
                attention,
                "--isl",
                str(args.isl),
                "--osl",
                str(args.osl),
                "--run-id",
                run_id,
                "--output-root",
                str(root),
            ]
            command.extend(
                [
                    "--cleanup-policy",
                    "preserve_allocator"
                    if args.kind == "cleanup" and candidate == "B"
                    else "empty_cache",
                ]
            )
            if args.manifest:
                command.extend(["--manifest", args.manifest])
            started = time.monotonic()
            with (
                (root / f"{pair}-{candidate}.stdout").open("w") as stdout,
                (root / f"{pair}-{candidate}.stderr").open("w") as stderr,
            ):
                child = subprocess.Popen(
                    command, stdout=stdout, stderr=stderr, start_new_session=True
                )
                timed_out = False
                try:
                    returncode = child.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        returncode = child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        returncode = child.wait(timeout=5)
            record = {
                "pair": pair,
                "candidate": candidate,
                "run_id": run_id,
                "command": command,
                "returncode": returncode,
                "timed_out": timed_out,
                "elapsed_s": time.monotonic() - started,
                "mode": mode,
                "attention": attention,
                "stdout_sha256": sha256(root / f"{pair}-{candidate}.stdout"),
                "stderr_sha256": sha256(root / f"{pair}-{candidate}.stderr"),
            }
            if returncode == 0 and not timed_out:
                result = json.loads((root / run_id / "result.json").read_text())
                record["metrics"] = result["metrics"]
                record["result_sha256"] = sha256(root / run_id / "result.json")
                members[candidate] = record
            else:
                failure_path = root / run_id / "failure.json"
                record["failure"] = (
                    json.loads(failure_path.read_text())
                    if failure_path.exists()
                    and failure_path.stat().st_size <= 16 * 1024
                    else None
                )
            rows.append(record)
            atomic_json(root / "ledger.json", rows)
        if len(members) == 2:
            left, right = (members[c]["metrics"] for c in ("A", "B"))
            item = {
                "pair": pair,
                "tokens_equal": left["generated_token_ids"]
                == right["generated_token_ids"],
                "runtime_e2e_delta_ms": right["runtime_e2e_ms"]
                - left["runtime_e2e_ms"],
                "runtime_e2e_relative_delta": right["runtime_e2e_ms"]
                / left["runtime_e2e_ms"]
                - 1,
            }
            item["first_token_relative_delta"] = (
                right["runtime_first_token_ms"] / left["runtime_first_token_ms"] - 1
            )
            item["worker_total_relative_delta"] = (
                right["worker_request_total_ms"] / left["worker_request_total_ms"] - 1
            )
            item["worker_cpu_relative_delta"] = (
                right["worker_request_cpu_ms"] / left["worker_request_cpu_ms"] - 1
            )
            if args.kind in {"attention", "cleanup"}:
                item["mechanisms"] = compare(
                    *(analyze(root / members[c]["run_id"]) for c in ("A", "B")),
                    intervention="cleanup_policy"
                    if args.kind == "cleanup"
                    else "attention_requested",
                )
            pairs.append(item)
    summary = {
        "comparison_status": "descriptive"
        if len(pairs) == args.pairs
        else "blocked_by_failed_or_incomplete_runs",
        "spec": spec,
        "pairs": pairs,
        "attempted": len(rows),
        "paired_uncertainty": paired_uncertainty(pairs)
        if args.kind == "overhead"
        else {
            "status": "not_applicable",
            "reason": "Instrumented mechanism comparisons are not overhead baselines",
        },
        "failed": sum(row["returncode"] != 0 for row in rows),
        "completed_pairs": len(pairs),
        "quality_status": "not_evaluated",
        "decision": "hold",
        "median_relative_e2e_delta": statistics.median(
            [p["runtime_e2e_relative_delta"] for p in pairs]
        )
        if pairs
        else None,
        "notice": "Observed paired differences; no low-overhead acceptance or speedup claim without uncertainty and quality gates.",
    }
    atomic_json(root / "summary.json", summary)
    print(
        json.dumps(
            {
                "path": str(root),
                "completed_pairs": len(pairs),
                "failed": summary["failed"],
            }
        )
    )
    return 0 if len(pairs) == args.pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
