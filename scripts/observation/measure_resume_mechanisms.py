#!/usr/bin/env python3
"""Measure live KV layout and allocator cleanup through an existing Console.

Invoke on the target via remote_run.sh. This is a diagnostic experiment, not
an independent-process performance benchmark. It never loads/unloads a model.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
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
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error("repeats must be 1..5")
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
    attempts = []

    def save(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2))

    def request(path, body=None, binary=False):
        headers = dict(session_headers)
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers.update({"Content-Type": "application/json", "Idempotency-Key": uuid.uuid4().hex})
        req = urllib.request.Request(base + path, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as response:
            content = response.read()
        return content if binary else json.loads(content)

    def wait(run):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            current = request("/runs/" + run["id"])
            if current["state"] in {"completed", "failed", "cancelled", "timed_out"}:
                attempts.append({"run_id": current["id"], "state": current["state"]})
                save("attempts.json", attempts)
                save("runs/" + current["id"] + "/run.json", current)
                if current["state"] != "completed":
                    raise RuntimeError("Run failed: " + current["id"] + " " + str(current.get("error")))
                return current
            time.sleep(0.2)
        request("/requests/" + run["id"] + "/cancel", {})
        raise TimeoutError(run["id"])

    deployments = request("/deployments")["items"]
    ready = [d for d in deployments if d["state"] == "ready"]
    if len(ready) != 1:
        raise RuntimeError("Exactly one ready deployment required")
    deployment = ready[0]
    save("deployment-before.json", deployment)
    if deployment["detail"].get("weight_verification") != "manifest_verified":
        raise RuntimeError("Model weights must be manifest verified")
    runs = request("/runs?limit=100")["items"]
    if any(r["state"] in {"queued", "running", "cancel_requested"} for r in runs):
        raise RuntimeError("Console already has active work")
    identity = {"deployment_id": deployment["id"], "expected_epoch": deployment["epoch"]}
    output_tokens = 16
    protocol = {"scope": "same_resident_worker_diagnostic_only", "input_lengths": [32, 128, 256, 512],
                "output_tokens": output_tokens, "repeats": args.repeats,
                "cleanup_input_tokens": 128, "cleanup_policies": ["empty_cache", "preserve_allocator"],
                "cleanup_order": "AB/BA alternating; unobserved empty_cache precondition before every diagnostic",
                "performance_claim": False, "quality_scope": "one short teacher-forced fixture, not a task benchmark",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "criteria": ["complete sealed captures", "downloaded artifact hashes match", "KV inventory complete",
                             "KV shape bytes equal logical and deduplicated storage bytes", "fixed workloads applied",
                             "cleanup pair output tokens identical", "quality full-vocabulary values identical"]}
    save("protocol.json", protocol)

    def precondition():
        return wait(request("/requests", {**identity, "messages": [{"role": "user", "content": "Explain model inference and memory caching."}],
                    "max_output_tokens": 16, "deadline_ms": 120000, "observation_mode": "off",
                    "optimization": {"changes": {"cleanup_policy": "empty_cache"}}}))

    def plan(length, cleanup=False):
        spec = {"baseline": {"changes": {}}, "candidate": {"changes": {"cleanup_policy": "preserve_allocator"}} if cleanup else {"changes": {}},
                "declared_changes": ["cleanup_policy"] if cleanup else [], "performance_pairs": 1,
                "input_tokens": length, "output_tokens": output_tokens,
                "quality_plan": {"samples": [{"id": "fixed-text", "text": "The GPU stores model weights and cached attention keys."}],
                                 "max_tokens": 16, "generation_tokens": 8},
                "observation_plan": {"mode": "memory" if cleanup else "basic", "sources": ["allocator"] if cleanup else [],
                                     "max_decode_steps": 2}}
        value = request("/experiments/plans", {"spec": spec})
        save("plans/" + value["experiment_id"] + ".json", value)
        return value

    def job(value, arm="baseline", role="diagnostic"):
        precondition()
        run = wait(request("/experiments/" + value["experiment_id"] + "/jobs", {**identity, "arm": arm, "role": role, "pair": 0}))
        bundle = request("/runs/" + run["id"] + "/experiment")
        if bundle["manifest"]["status"] != "complete" or bundle["receipt"]["status"] != "applied":
            raise RuntimeError("Unsealed or unapplied run")
        save("runs/" + run["id"] + "/bundle.json", bundle)
        for index, item in enumerate(bundle["manifest"]["artifacts"]):
            relative = PurePosixPath(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid artifact path")
            content = request("/runs/" + run["id"] + "/experiment/artifacts/" + str(index), binary=True)
            if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError("Artifact integrity mismatch")
            destination = root / "runs" / run["id"] / "artifacts" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        observation = None
        if role == "diagnostic":
            observation = request("/runs/" + run["id"] + "/observations")
            save("runs/" + run["id"] + "/observation.json", observation)
            manifest = request("/runs/" + run["id"] + "/observations/manifest")
            save("runs/" + run["id"] + "/observation-manifest.json", manifest)
            for index, item in enumerate(manifest["artifacts"]):
                relative = PurePosixPath(item["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Invalid observation path")
                content = request("/runs/" + run["id"] + "/observations/artifacts/" + str(index), binary=True)
                if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
                    raise ValueError("Observation integrity mismatch")
                destination = root / "runs" / run["id"] / "artifacts" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            if observation["capture_status"] != "complete":
                raise RuntimeError("Partial observation")
            if run["metrics"]["input_tokens"] != value["spec"]["input_tokens"] or run["metrics"]["output_tokens"] != output_tokens:
                raise RuntimeError("Workload differs")
        return run, bundle, observation

    kv_rows, cleanup_rows = [], []
    try:
        for repeat in range(args.repeats):
            lengths = protocol["input_lengths"] if repeat % 2 == 0 else list(reversed(protocol["input_lengths"]))
            for length in lengths:
                run, _, obs = job(plan(length))
                kv = next(row for row in obs["kv"] if row["label"] == "after_generation")
                if kv["availability"] != "complete" or not kv["entries"]:
                    raise RuntimeError("KV metadata unavailable")
                entries = kv["entries"]
                lengths_seen = {entry["shape"][-2] for entry in entries}
                if len(lengths_seen) != 1:
                    raise RuntimeError("KV lengths inconsistent")
                cache_tokens = lengths_seen.pop()
                shape_bytes = sum(math.prod(e["shape"]) * 2 for e in entries)
                if any(e["dtype"] != "torch.float16" for e in entries) or shape_bytes != kv["logical_bytes"] or shape_bytes != kv["unique_storage_bytes"]:
                    raise RuntimeError("KV shape/logical/storage mismatch")
                item = {"repeat": repeat, "run_id": run["id"], "input_tokens": length, "output_tokens": output_tokens,
                        "cache_tokens": cache_tokens, "layers": len({e["layer"] for e in entries}), "entries": len(entries),
                        "logical_bytes": kv["logical_bytes"], "unique_storage_bytes": kv["unique_storage_bytes"],
                        "shape_bytes": shape_bytes, "bytes_per_cache_token": shape_bytes / cache_tokens}
                kv_rows.append(item)
                save("kv-runs.json", kv_rows)
                print(json.dumps({"kind": "kv", **item}), flush=True)
        for repeat in range(args.repeats):
            value = plan(128, cleanup=True)
            members = {}
            for arm in (["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]):
                run, _, obs = job(value, arm)
                after = next(row for row in obs["allocator"] if row["label"] == "after_cleanup")
                before = next(row for row in obs["allocator"] if row["label"] == "before_cleanup")
                metrics = {row["metric_id"]: row["value"] for row in obs["metrics"] if row.get("label") == "after_cleanup"}
                attribution = obs["allocation_attribution"]
                lifetimes = obs["allocation_lifetimes"]
                members[arm] = {"run_id": run["id"], "generated_token_ids": run["metrics"]["generated_token_ids"],
                                "allocated_bytes": metrics["allocated_bytes"], "reserved_bytes": metrics["reserved_bytes"],
                                "peak_allocated_bytes": metrics["peak_allocated_bytes"], "before_cleanup": before, "after_cleanup": after,
                                "attribution_total": len(attribution), "attribution_matched": sum(r["confidence"] == "snapshot_correlated" for r in attribution),
                                "lifetime_items": lifetimes["total_items"], "history_prefix_may_be_overwritten": lifetimes["prefix_may_be_overwritten"]}
            a, b = members["baseline"], members["candidate"]
            if a["generated_token_ids"] != b["generated_token_ids"]:
                raise RuntimeError("Cleanup pair output differs")
            item = {"repeat": repeat, "experiment_id": value["experiment_id"], "members": members,
                    "reserved_delta_bytes": b["reserved_bytes"] - a["reserved_bytes"],
                    "allocated_delta_bytes": b["allocated_bytes"] - a["allocated_bytes"],
                    "peak_allocated_delta_bytes": b["peak_allocated_bytes"] - a["peak_allocated_bytes"], "tokens_equal": True}
            cleanup_rows.append(item)
            save("cleanup-pairs.json", cleanup_rows)
            print(json.dumps({"kind": "cleanup", "repeat": repeat, "reserved_delta_bytes": item["reserved_delta_bytes"]}), flush=True)
        quality = {}
        for arm in ("baseline", "candidate"):
            run, bundle, _ = job(value, arm, "quality")
            quality[arm] = {"run_id": run["id"], "quality": bundle["quality"],
                            "logits_sha256": [r["sha256"] for r in bundle["manifest"]["artifacts"] if r["path"].endswith(".f32")]}
        if not quality["baseline"]["logits_sha256"] or quality["baseline"]["logits_sha256"] != quality["candidate"]["logits_sha256"]:
            raise RuntimeError("Quality logits differ")
        save("quality.json", quality)
        summary = {"status": "complete", "scope": protocol["scope"], "kv_runs": len(kv_rows),
                   "kv_bytes_per_token": sorted({r["bytes_per_cache_token"] for r in kv_rows}),
                   "kv_layers": sorted({r["layers"] for r in kv_rows}), "kv_shape_storage_max_error_bytes": 0,
                   "cleanup_pairs": len(cleanup_rows), "cleanup_reserved_deltas_bytes": [r["reserved_delta_bytes"] for r in cleanup_rows],
                   "cleanup_reserved_median_delta_bytes": statistics.median(r["reserved_delta_bytes"] for r in cleanup_rows),
                   "cleanup_allocated_deltas_bytes": [r["allocated_delta_bytes"] for r in cleanup_rows],
                   "cleanup_peak_allocated_deltas_bytes": [r["peak_allocated_delta_bytes"] for r in cleanup_rows],
                   "quality_logits_byte_identical": True, "performance_claim": False}
        save("summary.json", summary)
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        save("failure.json", {"error": type(exc).__name__ + ": " + str(exc), "completed_kv_runs": len(kv_rows), "completed_cleanup_pairs": len(cleanup_rows)})
        raise
    finally:
        save("deployment-after.json", request("/deployments")["items"][0])


if __name__ == "__main__":
    main()
