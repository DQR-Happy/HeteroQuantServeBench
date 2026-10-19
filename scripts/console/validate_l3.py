"""Target-only L3 real-model acceptance through the production console API.

Requires an idle service. Writes full runs/graphs and restores the saved original
deployment/precision/KV artifact in finally. Does not download or modify weights.
"""
import argparse
import json
import time
import urllib.request
import urllib.error
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8768/api/console/v1"
OUT = Path("reports/console/l3-runtime")


def call(path, body=None):
    request = urllib.request.Request(BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", "X-HQSB-Client": "console", "Idempotency-Key": uuid.uuid4().hex})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{path}:HTTP {exc.code}:{exc.read().decode()[:1600]}") from exc


def save(name, value):
    (OUT / (name + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def wait(run, label):
    end = time.monotonic() + 700
    while run["state"] not in {"completed", "failed", "cancelled", "timed_out", "interrupted"}:
        if time.monotonic() > end:
            raise TimeoutError(run["id"])
        time.sleep(1)
        run = call("/runs/" + run["id"])
    save(label, run)
    assert run["state"] == "completed" and run["cleanup"] == "succeeded", (label, run.get("error"), run["state"])
    return run


def unload_all(label):
    assert call("/overview")["pending"] == 0
    for d in call("/deployments")["items"]:
        if d["provider"] == "pytorch" and d["state"] == "ready":
            wait(call(f'/deployments/{d["id"]}/unload', {}), label + "-unload")


def main(snapshot, models):
    OUT.mkdir(parents=True, exist_ok=True)
    assert call("/overview")["pending"] == 0
    assert call("/scheduling-benchmarks")["active_id"] is None
    original = json.loads(Path(snapshot).read_text())
    assert call("/benchmark-catalog?deployment_id=" + original["id"])["active_id"] is None
    recipe = {k: v for k, v in original["detail"]["precision_receipt"]["resolved_recipe"].items() if k != "calibration"}
    if original.get("quantization", {}).get("artifact_id"):
        recipe["artifact_id"] = original["quantization"]["artifact_id"]
    summary = {"acceptance": {"max_abs_mean_nll_delta": 0.01, "reference": "same loaded model eager", "speedup_claim": False}, "models": {}}
    deadline_ms = call("/runtime/catalog")["limits"]["max_deadline_ms"]
    try:
        for identity in models:
            print("MODEL", identity, flush=True)
            unload_all(identity)
            wait(call(f"/deployments/{identity}/load", {"precision": "float16", "kernel_profile": "framework-native", "quantization": {"method": "none"}}), identity + "-load")
            deployment = next(d for d in call("/deployments")["items"] if d["id"] == identity)
            results = summary["models"][identity] = {}
            for mode in ("eager", "dynamo_eager", "inductor"):
                plan = {"mode": mode, "shape_policy": "auto", "fullgraph": False}
                body = {"deployment_id": identity, "expected_epoch": deployment["epoch"], "runtime_plan": plan,
                        "prompt": "Once upon a time, there was a little", "max_output_tokens": 16,
                        "deadline_ms": deadline_ms, "observation_mode": "off"}
                runs = []
                for repeat in range(2):
                    run = wait(call("/requests", body), f"{identity}-{mode}-{repeat}")
                    graph = call(f'/runs/{run["id"]}/runtime-graph')
                    save(f"{identity}-{mode}-{repeat}-graph", graph)
                    receipt = run["metrics"]["optimization_receipt"]
                    assert receipt["status"] == "applied", receipt["runtime_receipt"]
                    if mode != "eager":
                        assert graph["graph_invocations"] > 0 and graph["graphs"]
                        assert graph["analysis"]["mapped_call_nodes"] > 0
                        assert any(n.get("module", "") and n["module"].endswith(".self_attn.q_proj")
                                   for g in graph["graphs"] for n in g["nodes"]), "L3_NESTED_MODULE_JOIN_MISSING"
                    runs.append({"run_id": run["id"], "tokens": run["metrics"]["generated_token_ids"],
                                 "token_s": run["metrics"]["output_tokens_per_s"], "runtime": receipt["runtime_receipt"]})
                    print(mode, repeat, run["id"], "new", graph["new_graphs"], "calls", graph["graph_invocations"], flush=True)
                quality = wait(call("/precision/quality-jobs", {"deployment_id": identity, "expected_epoch": deployment["epoch"],
                    "runtime_plan": plan, "quality_plan": {"benchmark_id": "wikitext2-mini-v1"}}), f"{identity}-{mode}-quality")
                results[mode] = {"generation": runs, "quality_run_id": quality["id"], "quality": quality["metrics"]["quality_evaluation"]}
                if mode != "eager":
                    comparison = call("/runtime/comparisons", {"baseline_run_id": results["eager"]["quality_run_id"], "candidate_run_id": quality["id"]})
                    save(f"{identity}-{mode}-quality-comparison", comparison)
                    assert comparison["comparable"], comparison
                    assert abs(comparison["quality"]["mean_nll_increase"]) <= summary["acceptance"]["max_abs_mean_nll_delta"], comparison
                    results[mode]["quality_comparison"] = comparison
                save("summary", summary)
            # A request without runtime_plan remains the original generation path.
            plain = wait(call("/requests", {"deployment_id": identity, "expected_epoch": deployment["epoch"],
                "prompt": "Once upon a time", "max_output_tokens": 4, "observation_mode": "off"}), identity + "-legacy")
            assert plain["metrics"]["optimization_receipt"]["runtime_receipt"]["plan"]["mode"] == "eager"
    finally:
        print("RESTORING", original["id"], flush=True)
        unload_all("restore")
        wait(call(f'/deployments/{original["id"]}/load', {"precision": original["precision"],
            "kernel_profile": original.get("kernel_profile", "framework-native"), "quantization": recipe}), "restored-original-load")
        current = next(d for d in call("/deployments")["items"] if d["id"] == original["id"])
        assert current["quantization"]["method"] == original["quantization"]["method"]
        assert current["quantization"]["kv"] == original["quantization"]["kv"]
        assert current["quantization"].get("artifact_id") == original["quantization"].get("artifact_id")
        save("restored-original", current)
        summary["restored"] = True
        save("summary", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    main(args.snapshot, args.models)
