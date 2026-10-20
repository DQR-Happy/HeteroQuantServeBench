"""Real GPU L10 acceptance through production API, restores original recipe."""

import argparse
import json
from pathlib import Path
import time
from scripts.console import validate_l3 as api, validate_graph_views as graph

OUT = Path("reports/console/l10-application")


def idle():
    end = time.monotonic() + 60
    while api.call("/overview")["pending"]:
        if time.monotonic() > end:
            raise TimeoutError("console cleanup pending")
        time.sleep(0.1)


def load(d, label):
    return api.wait(
        api.call(
            f"/deployments/{d}/load",
            {
                "precision": "float16",
                "kernel_profile": "framework-native",
                "quantization": {"method": "none"},
            },
        ),
        label,
    )


def run(plan, label):
    row = api.wait(api.call("/application/jobs", {"plan": plan}), label)
    receipt = api.call(f"/runs/{row['id']}/application")
    api.save(label + "-receipt", receipt)
    assert receipt["status"] == "applied" and receipt["cleanup"] == "succeeded"
    print(
        label,
        row["id"],
        json.dumps([v["metrics"] for v in receipt["variants"]]),
        flush=True,
    )
    idle()
    return row, receipt


def main(managed):
    OUT.mkdir(parents=True, exist_ok=True)
    api.OUT = graph.OUT = OUT
    idle()
    assert api.call("/overview").get("benchmark_owner") is None
    original = next(
        d
        for d in api.call("/deployments")["items"]
        if d["provider"] == "pytorch" and d["state"] == "ready"
    )
    api.save("original", original)
    summary = {"restored": False, "universal_speedup_claim": False}
    try:
        api.unload_all("initial")
        graph.restart()
        load("qwen-reference", "qwen-load")
        target = {
            "id": "primary",
            "deployment_id": "qwen-reference",
            "machine_id": "local",
            "lifecycle": "existing",
        }
        requests = [
            {
                "id": f"r{i}",
                "prompt": "Once upon a time, there was a little fox.",
                "tenant": "other" if i == 3 else "default",
                "max_output_tokens": 16,
            }
            for i in range(4)
        ]
        plan = {
            "targets": [target],
            "requests": requests,
            "fixed_output_length": True,
            "concurrency": 2,
        }
        row, doc = run(plan, "cache")
        a, b = doc["variants"]
        assert (
            a["metrics"]["backend_invocations"] == 4
            and b["metrics"]["backend_invocations"] == 2
        )
        assert b["cache"]["hits"] == 2 and doc["comparison"]["all_tokens_exact"]
        assert (
            next(r for r in b["requests"] if r["tenant"] == "other")["cache"] == "miss"
        )
        summary["cache_run_id"] = row["id"]
        row, doc = run(
            {**plan, "requests": requests[:2], "candidate": {"prefix_reuse": True}},
            "prefix",
        )
        prefix = [
            (r.get("backend_metrics") or {}).get("memory_mechanism", {}).get("prefix")
            or {}
            for r in doc["variants"][1]["requests"]
        ]
        assert any(p.get("reused_tokens", 0) > 0 for p in prefix), prefix
        assert doc["comparison"]["all_tokens_exact"]
        summary["prefix_run_id"] = row["id"]
        row, doc = run(
            {
                **plan,
                "requests": [
                    {
                        "id": "long",
                        "prompt": "The cat sat on the mat. " * 30,
                        "max_output_tokens": 16,
                    }
                ],
                "candidate": {
                    "input_transform": "head_tail_tokens",
                    "input_token_cap": 32,
                    "output_cap": 8,
                },
            },
            "input-budget",
        )
        r = doc["variants"][1]["requests"][0]
        assert (
            r["input"]["input_tokens_before"] > r["input"]["input_tokens_after"]
            and r["input"]["input_tokens_after"] <= 32
        )
        assert (
            r["output_tokens"] == 8
            and not doc["comparison"]["pairs"][0]["same_effective_input"]
        )
        summary["input_budget_run_id"] = row["id"]
        row, doc = run(
            {
                **plan,
                "requests": [
                    {
                        "id": "chat-window",
                        "messages": [
                            {"role": "system", "content": "Answer briefly in English."},
                            {"role": "user", "content": "Tell me about a fox."},
                            {
                                "role": "assistant",
                                "content": "A fox is a small wild animal.",
                            },
                            {"role": "user", "content": "What is two plus two?"},
                        ],
                        "max_output_tokens": 8,
                    }
                ],
                "candidate": {
                    "input_transform": "recent_messages",
                    "recent_messages": 1,
                },
            },
            "chat-window",
        )
        transformed = doc["variants"][1]["requests"][0]
        assert (
            transformed["input"]["input_tokens_before"]
            > transformed["input"]["input_tokens_after"]
        )
        assert (
            transformed["output_tokens"] == 8
            and not doc["comparison"]["pairs"][0]["same_effective_input"]
        )
        summary["chat_window_run_id"] = row["id"]
        if managed:
            api.unload_all("before-managed")
            cold = {
                "id": "managed",
                "deployment_id": "gemma3-1b",
                "machine_id": "local",
                "lifecycle": "managed",
            }
            row, doc = run(
                {
                    "targets": [cold],
                    "requests": [
                        {
                            "id": "first",
                            "prompt": "Once upon a time",
                            "max_output_tokens": 8,
                        },
                        {
                            "id": "later",
                            "prompt": "Once upon a time",
                            "arrival_ms": 22000,
                            "max_output_tokens": 8,
                        },
                    ],
                    "candidate": {"autoscale": True, "idle_unload_ms": 100},
                    "fixed_output_length": True,
                    "timeout_s": 300,
                },
                "managed-autoscale",
            )
            assert sum(e["kind"] == "replica.loaded" for e in doc["events"]) >= 3
            assert sum(e["kind"] == "replica.unloaded" for e in doc["events"]) >= 3
            assert any(e.get("reason") == "idle_scale_down" for e in doc["events"])
            assert all(
                d["state"] in {"unloaded", "failed"}
                for d in api.call("/deployments")["items"]
                if d["provider"] == "pytorch"
            )
            summary["managed_run_id"] = row["id"]
            load("qwen-reference", "qwen-after-managed")
        request = api.call(
            "/application/jobs",
            {
                "plan": {
                    **plan,
                    "requests": [
                        {**requests[0], "id": f"long-{i}", "max_output_tokens": 128}
                        for i in range(8)
                    ],
                }
            },
        )
        end = time.monotonic() + 40
        while True:
            current = api.call("/runs/" + request["id"])
            if (current.get("progress") or {}).get("stage") == "backend.accepted":
                break
            assert current["state"] not in {"failed", "completed"}, current.get("error")
            if time.monotonic() > end:
                raise TimeoutError("child dispatch not observed")
            time.sleep(0.1)
        api.call("/requests/" + request["id"] + "/cancel", {})
        while current["state"] not in {"completed", "failed", "cancelled", "timed_out"}:
            time.sleep(0.2)
            current = api.call("/runs/" + request["id"])
        api.save("cancelled", current)
        assert current["state"] == "cancelled" and current["cleanup"] == "succeeded", (
            current.get("error")
        )
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
        if original.get("quantization", {}).get("artifact_id"):
            recipe["artifact_id"] = original["quantization"]["artifact_id"]
        api.wait(
            api.call(
                f"/deployments/{original['id']}/load",
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
        d = next(
            d for d in api.call("/deployments")["items"] if d["id"] == original["id"]
        )
        api.wait(
            api.call(
                "/requests",
                {
                    "deployment_id": d["id"],
                    "expected_epoch": d["epoch"],
                    "prompt": "Once upon a time",
                    "max_output_tokens": 4,
                },
            ),
            "legacy-request",
        )
        summary["restored"] = True
        api.save("restored-original", d)
        api.save("summary", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--managed", action="store_true")
    main(parser.parse_args().managed)
