#!/usr/bin/env python3
"""Counterbalanced off/basic capture overhead in one resident Console worker.

This measures within-session disturbance, not independent-process acceptance.
No backend changes, model reload, clock changes, or profiler fallback.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import urllib.error
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8768")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", type=int, default=20)
    args = parser.parse_args()
    if not 10 <= args.pairs <= 40 or args.pairs % 2:
        parser.error("An even 10..40 pairs required")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = args.url.rstrip("/") + "/api/console/v1"
    session_headers = {"X-HQSB-Client": "console"}
    try:
        with urllib.request.urlopen(urllib.request.Request(base + "/session", headers=session_headers), timeout=30):
            pass
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            raise
        token = (Path(args.data_dir) / "access-token").read_text().strip()
        session_headers["Authorization"] = "Bearer " + token
    ledger, rows = [], []

    def save(name, obj):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

    def api(path, body=None, binary=False):
        headers = dict(session_headers)
        data = None
        if body is not None:
            headers.update({"Content-Type": "application/json", "Idempotency-Key": uuid.uuid4().hex})
            data = json.dumps(body).encode()
        with urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers), timeout=120) as response:
            content = response.read()
        return content if binary else json.loads(content)

    def wait(run):
        limit = time.monotonic() + 180
        while time.monotonic() < limit:
            run = api("/runs/" + run["id"])
            if run["state"] in {"completed", "failed", "cancelled", "timed_out"}:
                save("runs/" + run["id"] + "/run.json", run)
                ledger.append({"id": run["id"], "state": run["state"]})
                save("ledger.json", ledger)
                if run["state"] != "completed":
                    raise RuntimeError("Execution failed: " + run["id"] + " " + str(run.get("error")))
                return run
            time.sleep(0.05)
        api("/requests/" + run["id"] + "/cancel", {})
        raise TimeoutError(run["id"])

    ds = [d for d in api("/deployments")["items"] if d["state"] == "ready"]
    if len(ds) != 1:
        raise RuntimeError("Exactly one ready deployment required")
    deployment = ds[0]
    if deployment["detail"].get("weight_verification") != "manifest_verified":
        raise RuntimeError("Verified weights required")
    if any(r["state"] in {"queued", "running", "cancel_requested"} for r in api("/runs?limit=100")["items"]):
        raise RuntimeError("Console already has active work")
    save("deployment-before.json", deployment)
    identity = {"deployment_id": deployment["id"], "expected_epoch": deployment["epoch"]}
    protocol = {"pairs": args.pairs, "input_tokens": 128, "output_tokens": 32,
                "warmup_pairs_excluded": 2, "order": "off/basic then basic/off alternating",
                "replication_unit": "two counterbalanced request pairs within one resident worker",
                "observation_plan": {"mode": "basic", "sources": [], "max_decode_steps": 8, "os_interval_ms": 250},
                "same_controls": ["loaded model/epoch", "fixed input/output tokens", "FP16/eager/greedy", "empty_cache cleanup"],
                "primary": "runtime_e2e_ms; host generation loop excluding capture setup/finalization",
                "secondary": ["worker_request_total_ms including collection finalization", "console_e2e_ms including queue/control/publication", "runtime_first_token_ms", "token_itl_ms p95", "worker_request_cpu_ms", "allocator peak"],
                "bootstrap": {"blocks": args.pairs // 2, "resamples": 10000, "seed": 20261014},
                "scope": "conditional within-session disturbance, not independent-process low-overhead acceptance",
                "thresholds_registered_before_execution": {"generation_upper_relative": 0.05, "worker_upper_relative": 0.10},
                "clock_control": "unchanged/unlocked", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save("protocol.json", protocol)

    def percentile(values, fraction):
        values = sorted(values)
        x = (len(values) - 1) * fraction
        lo = int(x)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] * (hi - x) + values[hi] * (x - lo) if hi != lo else values[lo]

    def run_pair(index):
        spec = {"baseline": {"changes": {}}, "candidate": {"changes": {}}, "declared_changes": [],
                "performance_pairs": 1, "input_tokens": 128, "output_tokens": 32,
                "quality_plan": {"samples": [{"text": "The GPU stores model weights."}], "max_tokens": 16, "generation_tokens": 4},
                "observation_plan": protocol["observation_plan"]}
        plan = api("/experiments/plans", {"spec": spec})
        save("plans/" + plan["experiment_id"] + ".json", plan)
        members = {}
        order = ["off", "basic"] if index % 2 == 0 else ["basic", "off"]
        for mode in order:
            run = wait(api("/experiments/" + plan["experiment_id"] + "/jobs", {**identity,
                       "arm": "baseline" if mode == "off" else "candidate", "role": "performance" if mode == "off" else "diagnostic", "pair": 0}))
            metrics = run["metrics"]
            if metrics["input_tokens"] != 128 or metrics["output_tokens"] != 32 or run["config"]["observation_mode"] != mode:
                raise RuntimeError("Fixed workload/mode differs")
            bundle = api("/runs/" + run["id"] + "/experiment")
            save("runs/" + run["id"] + "/bundle.json", bundle)
            if bundle["manifest"]["status"] != "complete" or bundle["receipt"]["status"] != "applied":
                raise RuntimeError("Unsealed or unapplied run")
            if mode == "basic":
                manifest = api("/runs/" + run["id"] + "/observations/manifest")
                save("runs/" + run["id"] + "/observation-manifest.json", manifest)
                if manifest["capture_status"] != "complete" or manifest["counts"]["dropped"] or manifest["counts"]["accepted_not_written"]:
                    raise RuntimeError("Incomplete capture")
                for n, item in enumerate(manifest["artifacts"]):
                    name = Path(item["path"])
                    if name.is_absolute() or ".." in name.parts:
                        raise RuntimeError("Invalid artifact path")
                    content = api("/runs/" + run["id"] + "/observations/artifacts/" + str(n), binary=True)
                    if hashlib.sha256(content).hexdigest() != item["sha256"] or len(content) != item["bytes"]:
                        raise RuntimeError("Artifact integrity mismatch")
                    target = root / "runs" / run["id"] / "artifacts" / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
            members[mode] = {"run_id": run["id"], "metrics": metrics, "receipt": bundle["receipt"]}
        a, b = members["off"], members["basic"]
        if a["metrics"]["generated_token_ids"] != b["metrics"]["generated_token_ids"] or a["metrics"]["input_token_sha256"] != b["metrics"]["input_token_sha256"]:
            raise RuntimeError("Input/output tokens differ")
        keys = ("runtime_e2e_ms", "worker_request_total_ms", "console_e2e_ms", "runtime_first_token_ms", "worker_request_cpu_ms")
        ratios = {key: b["metrics"][key] / a["metrics"][key] - 1 for key in keys}
        ratios["token_itl_p95_ms"] = percentile(b["metrics"]["token_itl_ms"], .95) / percentile(a["metrics"]["token_itl_ms"], .95) - 1
        return {"pair": index, "order": order, "members": members, "relative_deltas": ratios,
                "peak_allocated_delta_bytes": b["metrics"]["memory"]["peak_allocated_bytes"] - a["metrics"]["memory"]["peak_allocated_bytes"], "tokens_equal": True}

    try:
        for index in range(-2, args.pairs):
            row = run_pair(index)
            if index < 0:
                save("warmup-" + str(index + 2) + ".json", row)
            else:
                rows.append(row)
                save("pairs.json", rows)
                print(json.dumps({"pair": index, "generation_relative": row["relative_deltas"]["runtime_e2e_ms"],
                                  "worker_relative": row["relative_deltas"]["worker_request_total_ms"]}), flush=True)
        summary = {"scope": protocol["scope"], "completed_pairs": len(rows), "all_tokens_equal": True,
                   "all_basic_captures_complete": True, "metrics": {},
                   "peak_allocated_deltas_bytes": [r["peak_allocated_delta_bytes"] for r in rows]}
        blocks = [rows[n:n + 2] for n in range(0, len(rows), 2)]
        rng = random.Random(20261014)
        selections = [[rng.randrange(len(blocks)) for _ in blocks] for _ in range(10000)]
        for key in rows[0]["relative_deltas"]:
            draws = [statistics.median(r["relative_deltas"][key] for n in selection for r in blocks[n]) for selection in selections]
            baseline_key = key if key != "token_itl_p95_ms" else None
            summary["metrics"][key] = {"paired_median_relative": statistics.median(r["relative_deltas"][key] for r in rows),
                    "conditional_block_bootstrap_95_interval": [percentile(draws, .025), percentile(draws, .975)]}
            if baseline_key:
                summary["metrics"][key].update({mode + "_median_ms": statistics.median(r["members"][mode]["metrics"][key] for r in rows) for mode in ("off", "basic")})
        summary["independent_process_acceptance"] = "not_evaluated"
        save("summary.json", summary)
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        save("failure.json", {"error": type(exc).__name__ + ": " + str(exc), "completed_pairs": len(rows)})
        raise
    finally:
        save("deployment-after.json", api("/deployments")["items"][0])


if __name__ == "__main__":
    main()
