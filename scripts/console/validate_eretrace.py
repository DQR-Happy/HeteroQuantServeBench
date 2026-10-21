"""Target-only real-model ERETrace acceptance through the public console API."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8768")
    parser.add_argument("--deployment", default="gemma3-1b-sglang")
    parser.add_argument("--kernel", default="_fwd_kernel")
    parser.add_argument("--source", help="Use an existing completed single-request baseline")
    parser.add_argument("--codec", choices=["reference", "raw"], default="reference")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--mode", choices=["warp", "memory"], default="warp")
    parser.add_argument("--memory-capacity", type=int, default=32)
    parser.add_argument("--attention-backend", help="Explicit upstream option for the new baseline")
    parser.add_argument("--launch-count", type=int, default=3)
    parser.add_argument("--launch-skip", type=int, default=0)
    parser.add_argument("--cancel", action="store_true", help="Validate cancellation during engine loading")
    args = parser.parse_args()
    root = Path("reports/console/dmat" if args.mode == "memory" else "reports/console/eretrace")
    root.mkdir(parents=True, exist_ok=True)

    def call(path, body=None):
        request = urllib.request.Request(args.url + "/api/console/v1" + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json", "X-HQSB-Client": "console", "Idempotency-Key": uuid.uuid4().hex})
        with urllib.request.urlopen(request, timeout=35) as response:
            return json.load(response)

    def task(path, body):
        run = call(path, body)
        print("submitted", path, run["id"], flush=True)
        deadline = time.monotonic() + 950
        last = None
        while time.monotonic() < deadline:
            run = call("/runs/" + run["id"])
            state = (run["state"], run.get("progress", {}).get("stage"))
            if state != last:
                print(state, flush=True)
                last = state
            if run["state"] in {"completed", "failed", "cancelled", "interrupted", "timed_out"}:
                (root / (run["id"] + ".json")).write_text(json.dumps(run, ensure_ascii=False, indent=2))
                if run["state"] != "completed":
                    raise RuntimeError(run.get("error"))
                return run
            time.sleep(1)
        raise TimeoutError(run["id"])

    assert not call("/overview")["pending"]
    if args.source:
        source = args.source
    else:
        state = next(x for x in call("/deployments")["items"] if x["id"] == args.deployment)
        if state["state"] not in {"unloaded", "failed"}:
            raise RuntimeError("Unload the deployment before acceptance")
        options = {**state["engine_options"], "enforce_eager": True, "tensor_parallel_size": 1, "pipeline_parallel_size": 1}
        if state["provider"] == "sglang":
            options.update(attention_backend="triton", block_size=1)
        if args.attention_backend:
            options["attention_backend"] = args.attention_backend
        try:
            task(f"/deployments/{args.deployment}/load", {} if state["provider"] == "pytorch" else {"engine_options": options})
            state = next(x for x in call("/deployments")["items"] if x["id"] == args.deployment)
            if state["provider"] == "pytorch":
                baseline = task("/requests", {"deployment_id": args.deployment, "expected_epoch": state["epoch"],
                    "messages": [{"role": "user", "content": "Compute 17 + 25. Give only the number."}],
                    "max_output_tokens": 8, "save_input": True, "observation_mode": "off"})
            else:
                baseline = task("/native-engines/benchmarks", {"deployment_id": args.deployment, "expected_epoch": state["epoch"],
                    "prompts": ["Compute 17 + 25. Give only the number."], "expected_outputs": ["42"],
                    "concurrency": 1, "rounds": 1, "max_output_tokens": 8, "warmup_requests": 1})
            source = baseline["id"]
            if state["provider"] != "pytorch":
                assert baseline["metrics"]["native_benchmark"]["summary"]["quality_exact_match_rate"] == 1
        finally:
            state = next(x for x in call("/deployments")["items"] if x["id"] == args.deployment)
            if state["state"] == "ready":
                task(f"/deployments/{args.deployment}/unload", {})
    print("source", source, flush=True)
    plan = {"source_run_id": source, "tool": "eretrace", "ere_kernel": args.kernel,
        "ere_codec": args.codec, "ere_verify": args.verify, "launch_count": args.launch_count, "launch_skip": args.launch_skip}
    if args.mode == "memory":
        plan.update(ere_mode="memory", ere_codec="raw", ere_verify=False, memory_capacity=args.memory_capacity)
    if args.cancel:
        run = call("/diagnostics/jobs", plan)
        print("cancel target", run["id"], flush=True)
        deadline, started, requested = time.monotonic() + 100, time.monotonic(), False
        while time.monotonic() < deadline:
            run = call("/runs/" + run["id"])
            if not requested and run.get("progress", {}).get("stage") == "loading" and time.monotonic() - started > 8:
                call("/requests/" + run["id"] + "/cancel", {})
                requested = True
            if run["state"] in {"cancelled", "failed", "completed", "timed_out"}:
                break
            time.sleep(.5)
        (root / (run["id"] + ".json")).write_text(json.dumps(run, ensure_ascii=False, indent=2))
        assert requested and run["state"] == "cancelled" and run["cleanup"] == "succeeded", run
        assert not call("/overview")["pending"]
        print("cancellation and cleanup passed", run["id"], flush=True)
        return
    run = task("/diagnostics/jobs", plan)
    receipt = run["metrics"]["diagnostic"]
    assert run["cleanup"] == "succeeded"
    assert receipt["replay_workloads_match_source"] and receipt["replay_environment_matches_source"]
    assert receipt["replay_outputs_match_source"] is True
    panel = receipt["panel"]
    if args.mode == "memory":
        assert panel["counts"]["events"] > 0 and panel["counts"]["dropped_events"] == 0
        if "replay_token_ids_match_source" in receipt:
            assert receipt["replay_token_ids_match_source"] is True
        (root / "latest.json").write_text(json.dumps({"source": source, "run": run["id"], "counts": panel["counts"]}, indent=2))
        print("verified DMAT", json.dumps({"source": source, "run": run["id"], "counts": panel["counts"]}), flush=True)
        return
    assert panel["counts"]["observed_records"] > 0
    assert not call("/overview")["pending"]
    (root / "latest.json").write_text(json.dumps({"source": source, "run": run["id"], "counts": panel["counts"], "codec": args.codec}, indent=2))
    print("verified", json.dumps({"source": source, "run": run["id"], "counts": panel["counts"], "encoding": panel["encoding_counts"]}), flush=True)


if __name__ == "__main__":
    main()
