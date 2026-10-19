"""Target acceptance with optional registered Jetson peer; restores original model."""

import json
from pathlib import Path
import time
from scripts.console import validate_l3 as api, validate_graph_views as graph

OUT = Path("reports/console/l9-parallel")


def idle():
    end = time.monotonic() + 40
    while api.call("/overview")["pending"]:
        if time.monotonic() > end:
            raise TimeoutError("L9 supervisor cleanup pending")
        time.sleep(0.3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    api.OUT = graph.OUT = OUT
    idle()
    original = next(
        d
        for d in api.call("/deployments")["items"]
        if d["provider"] == "pytorch" and d["state"] == "ready"
    )
    api.save("original", original)
    summary = {"restored": False, "speedup_claim": False}
    try:
        api.unload_all("initial")
        graph.restart()
        hardware = api.call("/parallel/hardware")
        api.save("hardware", hardware)
        if len(hardware["devices"]) == 1:
            try:
                api.call(
                    "/parallel/jobs",
                    {
                        "deployment_id": "qwen-reference",
                        "plan": {"strategy": "tp", "devices": [0, 1]},
                    },
                )
            except RuntimeError as exc:
                assert "L9_REQUESTED_GPU_NOT_VISIBLE" in str(exc)
                summary["two_gpu_rejected"] = True
            else:
                raise AssertionError("single GPU must not pretend TP ranks")
        plan = {
            "max_output_tokens": 16,
            "trace_collectives": False,
            "prompts": [
                "1 2 3 4 5 1 2 3 4 5",
                "Once upon a time, there was a little fox.",
            ],
        }
        baseline = None
        for model in ["gemma3-1b", "qwen-reference"]:
            run = api.wait(
                api.call("/parallel/jobs", {"deployment_id": model, "plan": plan}),
                model + "-single",
            )
            doc = api.call("/runs/" + run["id"] + "/parallel")
            api.save(model + "-receipt", doc)
            assert doc["status"] == "applied" and len(doc["requests"]) == 2
            assert (
                doc["ranks"][0]["parameters"]["local_parameter_bytes"] > 0
                and doc["ranks"][0]["allocator"][0]["peak_allocated_bytes"] > 0
            )
            summary[model] = {"run_id": run["id"], "receipt": doc}
            if model == "qwen-reference":
                baseline = run
            idle()
        nodes = api.call("/machines")
        api.save("machines-before", nodes)
        # A loopback endpoint reached through the task's explicit SSH tunnels.
        peer_id = "jetson-l9"
        if not any(d.get("id") == peer_id for d in nodes["items"]):
            api.call(
                "/machines",
                {
                    "id": peer_id,
                    "name": "Jetson L9 · 经 Mac SSH 隧道",
                    "transport": "http",
                    "base_url": "http://127.0.0.1:18879",
                },
            )
        peer = api.wait(
            api.call(
                "/parallel/federated-jobs",
                {
                    "deployment_id": "qwen-reference",
                    "peer_machine_id": peer_id,
                    "peer_deployment_id": "qwen-reference",
                    "plan": plan,
                },
            ),
            "3090-jetson-dp",
        )
        paired = api.call("/runs/" + peer["id"] + "/parallel")
        api.save("3090-jetson-receipt", paired)
        assert len(paired["ranks"]) == 2 and [
            r["index"] for r in paired["requests"]
        ] == [0, 1]
        assert all(
            r["devices"][0]["name"] != "CPU diagnostic process; not a GPU"
            for r in paired["ranks"]
        )
        comparison = api.call(
            "/parallel/comparisons",
            {"baseline_run_id": baseline["id"], "candidate_run_id": peer["id"]},
        )
        api.save("3090-jetson-comparison", comparison)
        # Heterogeneous greedy equality is measured, never made a prerequisite
        # for recording genuine hardware results or relaxed by tolerance.
        summary["peer"] = {"run_id": peer["id"], "comparison": comparison}
        idle()
        cancelled = api.call(
            "/parallel/jobs",
            {
                "deployment_id": "qwen-reference",
                "plan": {
                    **plan,
                    "prompts": plan["prompts"] * 4,
                    "max_output_tokens": 128,
                },
            },
        )
        deadline = time.monotonic() + 100
        while True:
            current = api.call("/runs/" + cancelled["id"])
            if current["state"] in {"completed", "failed", "cancelled", "timed_out"}:
                raise AssertionError("cancel test ended before cancellation")
            if any(r["completed_requests"] >= 1 for r in current.get("progress", [])):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("wait actual GPU workload before cancelling")
            time.sleep(0.4)
        api.call("/requests/" + cancelled["id"] + "/cancel", {})
        while current["state"] not in {"completed", "failed", "cancelled", "timed_out"}:
            time.sleep(0.3)
            current = api.call("/runs/" + cancelled["id"])
        api.save("cancelled", current)
        assert current["state"] == "cancelled" and current["cleanup"] == "succeeded"
        idle()
        summary["cancel_run_id"] = current["id"]
    finally:
        idle()
        api.unload_all("restore")
        recipe = {
            k: v
            for k, v in original["detail"]["precision_receipt"][
                "resolved_recipe"
            ].items()
            if k != "calibration"
        }
        api.wait(
            api.call(
                "/deployments/" + original["id"] + "/load",
                {
                    "precision": original["precision"],
                    "kernel_profile": original.get(
                        "kernel_profile", "framework-native"
                    ),
                    "quantization": recipe,
                },
            ),
            "restored-load",
        )
        summary["restored"] = True
        api.save("summary", summary)
        print(
            json.dumps(
                {
                    "restored": True,
                    "peer": summary.get("peer"),
                    "cancel": summary.get("cancel_run_id"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
