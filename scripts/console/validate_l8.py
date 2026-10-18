"""Target-only real-weight L8 acceptance. Uses the running no-login console.

Run through scripts/remote_run.sh; never on the editing workstation.
Evidence is retained, including numerical failures. Prior ready model restored.
"""
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = "http://127.0.0.1:8768/api/console/v1"
OUT = Path("reports/console/l8")
TERMINAL = {"completed", "failed", "cancelled", "timed_out", "interrupted"}


def call(path, body=None, key=None):
    request = urllib.request.Request(ROOT + path, data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-HQSB-Client": "console", "Idempotency-Key": key or uuid.uuid4().hex})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def wait(run):
    for _ in range(6000):
        result = call("/runs/" + run["id"])
        if result["state"] in TERMINAL:
            return result
        time.sleep(.05)
    raise TimeoutError(run["id"])


def lifecycle(deployment, action):
    run = wait(call(f"/deployments/{deployment}/{action}", {}))
    assert run["state"] == "completed", (action, run)
    return run


def save(name, value):
    (OUT / (name + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def live_controls(deployment):
    state = next(r for r in call("/deployments")["items"] if r["id"] == deployment)
    plan = {"deployment_id": deployment, "expected_epoch": state["epoch"], "modes": ["continuous"],
        "save_input": True, "requests": [{"id": "seed", "prompt": "Once upon a time", "output_tokens": 192}]}
    accepted = call("/scheduling-benchmarks", plan)
    for _ in range(100):
        if call("/runs/" + accepted["id"])["state"] == "running":
            break
        time.sleep(.01)
    time.sleep(.08)
    call(f"/scheduling-benchmarks/{accepted['id']}/requests", {"id": "http-join", "prompt": "A little dog", "output_tokens": 128})
    time.sleep(.08)
    call(f"/scheduling-benchmarks/{accepted['id']}/requests/http-join/cancel", {})
    run = wait(accepted)
    save("tinyllama-http-ingress", run)
    assert run["state"] == "completed", (run["id"], run.get("error"))
    variant = run["result"]["variants"][0]
    assert "http-join" in variant["dynamic_join_requests"]
    assert {r["id"]: r["reason"] for r in variant["requests"]} == {"seed": "length", "http-join": "cancelled"}
    assert variant["cleanup"]["device_kv_rows"] == 0
    cancelled = call("/scheduling-benchmarks", plan)
    time.sleep(.1)
    call(f"/scheduling-benchmarks/{cancelled['id']}/cancel", {})
    final = wait(cancelled)
    save("tinyllama-session-cancel", final)
    assert final["state"] == "cancelled" and final["cleanup"] == "succeeded", final["id"]
    return [run["id"], final["id"]]


def experiment(deployment, requests, *, modes=None, limits=None, repeats=1, name, **extra):
    row = next(r for r in call("/deployments")["items"] if r["id"] == deployment)
    body = {"deployment_id": deployment, "expected_epoch": row["epoch"], "requests": requests,
        "modes": modes or ["serial", "static", "continuous"], "repeats": repeats, "save_input": True,
        "limits": {"prefill_chunk_tokens": 8, "max_batched_tokens": 64, **(limits or {})}, **extra}
    key = uuid.uuid4().hex
    accepted = call("/scheduling-benchmarks", body, key)
    assert call("/scheduling-benchmarks", body, key)["id"] == accepted["id"]
    run = wait(accepted)
    save(name, run)
    assert run["state"] == "completed" and run["cleanup"] == "succeeded", (run["id"], run["state"], run.get("error"))
    evidence = call(f"/scheduling-benchmarks/{run['id']}/records?kind=forward&limit=200")
    save(name + "-forwards", evidence)
    assert evidence["capture_status"] == "complete"
    assert all(v["cleanup"]["device_kv_rows"] == 0 for v in run["result"]["variants"])
    return run


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert all(r["state"] in TERMINAL for r in call("/runs?limit=100")["items"])
    before = call("/deployments")["items"]
    ready = [r for r in before if r["state"] == "ready"]
    save("acceptance-before", [{k: r.get(k) for k in ("id", "state", "kernel_profile")} for r in before])
    summary = {"status": "RUNNING", "runs": [], "restored": [], "quality_failures": []}
    try:
        for row in ready:
            if row["id"] == "gemma3-1b":
                cap = call(f"/deployments/{row['id']}/scheduling-capabilities")
                assert not cap["supported"] and cap["reasons"]
                summary["unsupported_gemma"] = cap
            lifecycle(row["id"], "unload")
        for deployment, label in [("ms-586060631c81c17023c0", "tinyllama"), ("ms-6d9b6a4a3f944e041fd6", "qwen3")]:
            lifecycle(deployment, "load")
            assert call(f"/deployments/{deployment}/scheduling-capabilities")["supported"]
            requests = [{"id": "long", "prompt": "Once upon a time a little fox found a red ball. The fox", "output_tokens": 128},
                {"id": "short", "prompt": "One day a cat", "output_tokens": 8},
                {"id": "join", "prompt": "A dog saw the fox. The dog", "arrival_ms": 40, "output_tokens": 24},
                {"id": "later", "prompt": "The little girl wanted to help her friend. She", "arrival_ms": 100, "output_tokens": 32}]
            run = experiment(deployment, requests, repeats=3, name=label + "-paired")
            for variant in run["result"]["variants"]:
                for request in variant["requests"]:
                    if request["quality"] == "exact_tokens_fail":
                        summary["quality_failures"].append({"run_id": run["id"], "session_id": variant["session_id"],
                            "request_id": request["id"], "first_difference": request.get("first_difference")})
            continuous = [v for v in run["result"]["variants"] if v["mode"] == "continuous"]
            assert all(v["max_actual_batch"] > 1 and v["dynamic_join_requests"] for v in continuous)
            assert all(v["loadgen"]["validity"] == "VALID" for v in run["result"]["variants"])
            summary["runs"].append({"name": label + "-paired", "id": run["id"], "quality": run["result"]["quality"],
                "variants": [{k: v[k] for k in ("mode", "repeat", "output_tokens_per_s", "ttft_p95_ms", "max_actual_batch", "membership_changes_with_overlap", "kv_copy_bytes", "peak_allocator_bytes")} for v in run["result"]["variants"]]})
            # Legacy generation remains callable immediately after session cleanup.
            state = next(r for r in call("/deployments")["items"] if r["id"] == deployment)
            single = wait(call("/requests", {"deployment_id": deployment, "expected_epoch": state["epoch"],
                "prompt": "Once upon a time", "max_output_tokens": 8, "save_input": True, "observation_mode": "basic"}))
            save(label + "-legacy-single", single)
            assert single["state"] == "completed" and single["metrics"]["output_tokens"] > 0
            parity = experiment(deployment, [{"id": "legacy", "prompt": "Once upon a time", "output_tokens": 8, "fixed_output_length": False}],
                modes=["serial"], limits={"chunked_prefill": False}, name=label + "-legacy-parity")
            tokens = call(f"/scheduling-benchmarks/{parity['id']}/records?kind=request.result&limit=1")["items"][0]["payload"]["output_token_ids"]
            assert tokens == single["metrics"]["generated_token_ids"], (label, "legacy/new B1 mismatch")
            summary.setdefault("legacy_parity", []).append({"deployment": deployment, "legacy_run": single["id"], "step_run": parity["id"], "exact_tokens": True})
            if label == "tinyllama":
                stress_items = [{"id": f"r{i}", "prompt": "Once upon a time" if i % 2 else "A little dog", "output_tokens": 2,
                    **({"cancel_after_ms": 0} if i % 32 == 0 else {"deadline_ms": 1} if i % 32 == 1 else
                       {"prompt": "Once upon a time " * 80} if i % 32 == 2 else {})} for i in range(128)]
                stress = experiment(deployment, stress_items,
                    modes=["serial", "continuous"], repeats=4, limits={"max_active_sequences": 8, "max_batch_sequences": 8}, name="tinyllama-lifecycle-1024")
                assert stress["result"]["quality"] == "partial_exact_tokens_pass"
                assert stress["result"]["quality_coverage"]["failures"] == 0
                summary["lifecycle_request_runs"] = sum(len(v["requests"]) for v in stress["result"]["variants"])
                final_allocations = [v["cleanup"]["memory"]["allocated_bytes"] for v in stress["result"]["variants"]]
                assert max(final_allocations) - min(final_allocations) <= 2 * 1024**2, "GPU allocations did not plateau"
                summary["lifecycle_cleanup_allocated_bytes"] = final_allocations
                summary["lifecycle_reason_counts"] = {reason: sum(v["reason_counts"].get(reason, 0) for v in stress["result"]["variants"])
                    for reason in ("length", "cancelled", "timed_out", "rejected")}
                summary["runs"].append({"name": "lifecycle-1024", "id": stress["id"]})
                preempt = experiment(deployment, [
                    {"id": "low", "prompt": "Once upon a time a fox", "output_tokens": 96},
                    {"id": "high", "prompt": "One day", "output_tokens": 8, "arrival_ms": 40, "priority": 9}],
                    modes=["serial", "continuous"], limits={"max_active_sequences": 1, "max_batch_sequences": 1, "preemption": "recompute"}, name="tinyllama-preempt")
                assert preempt["result"]["quality"] == "exact_tokens_pass"
                assert any(r["preemptions"] and r["recomputed_positions"] for v in preempt["result"]["variants"] for r in v["requests"])
                summary["runs"].append({"name": "preempt", "id": preempt["id"]})
                negative = experiment(deployment, [
                    {"id": "normal", "prompt": "Once upon a time", "output_tokens": 8},
                    {"id": "cancel", "prompt": "A dog", "output_tokens": 64, "cancel_after_ms": 10},
                    {"id": "expired", "prompt": "A cat", "output_tokens": 64, "deadline_ms": 1},
                    {"id": "reject", "prompt": "Once upon a time " * 80, "output_tokens": 8}],
                    modes=["continuous"], name="tinyllama-lifecycle-negative")
                reasons = {r["id"]: r["reason"] for r in negative["result"]["variants"][0]["requests"]}
                assert reasons == {"normal": "length", "cancel": "cancelled", "expired": "timed_out", "reject": "rejected"}, reasons
                summary["runs"].append({"name": "negative", "id": negative["id"], "reasons": reasons})
                fair = experiment(deployment, [
                    {"id": "a", "prompt": "Once upon a time", "output_tokens": 16, "tenant": "a"},
                    {"id": "b", "prompt": "A little dog " * 8, "output_tokens": 8, "tenant": "b"},
                    {"id": "c", "prompt": "A cat", "output_tokens": 8, "arrival_ms": 20, "tenant": "a"}],
                    modes=["serial", "continuous"], arrival_mode="closed_loop", concurrency=2,
                    limits={"policy": "tenant_fair", "tenant_weights": {"a": 2, "b": 1},
                        "phase_policy": "decode_first", "prefill_quota_tokens": 4, "max_batched_tokens": 16}, name="tinyllama-weighted-decode-first")
                assert fair["result"]["quality"] == "exact_tokens_pass"
                summary["runs"].append({"name": "weighted-decode-first-closed-loop", "id": fair["id"]})
                summary["http_control_runs"] = live_controls(deployment)
            lifecycle(deployment, "unload")
        summary["status"] = "FUNCTIONAL_PASS_WITH_QUALITY_FAILURES" if summary["quality_failures"] else "PASS"
    finally:
        for row in call("/deployments")["items"]:
            if row["state"] == "ready" and row["id"] not in {r["id"] for r in ready}:
                lifecycle(row["id"], "unload")
        for row in ready:
            state = next(r for r in call("/deployments")["items"] if r["id"] == row["id"])
            if state["state"] != "ready":
                run = wait(call(f"/deployments/{row['id']}/load", {"kernel_profile": row["kernel_profile"]}))
                assert run["state"] == "completed", run
            summary["restored"].append(row["id"])
        save("acceptance", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
