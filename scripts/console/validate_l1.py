"""Target-only L1 acceptance. Restores original resident model and board cap."""

import argparse
import json
from pathlib import Path
import time
import urllib.request
from scripts.console import validate_l3 as api, validate_graph_views as graph

OUT = Path("reports/console/l1-system")


def idle():
    end = time.monotonic() + 60
    while api.call("/overview")["pending"]:
        if time.monotonic() > end:
            raise TimeoutError("L1 cleanup pending")
        time.sleep(0.3)


def terminal(run):
    end = time.monotonic() + 650
    while run["state"] not in {"completed", "failed", "cancelled", "timed_out"}:
        if time.monotonic() > end:
            raise TimeoutError(run["id"])
        time.sleep(0.5)
        run = api.call("/runs/" + run["id"])
    idle()
    return run


def main(models, rounds):
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
        hw = api.call("/system/hardware")
        api.save("hardware", hw)
        original_power = hw["devices"][0]["power_limit_w"]
        cpus = hw["allowed_cpus"][:2]
        plan = {
            "rounds": rounds,
            "max_output_tokens": 16,
            "candidate": {"cpu_ids": cpus, "intra_threads": 1, "interop_threads": 1},
        }
        for model in models:
            run = api.wait(
                api.call("/system/jobs", {"deployment_id": model, "plan": plan}), model
            )
            doc = api.call(f"/runs/{run['id']}/system")
            api.save(model + "-receipt", doc)
            assert doc["cleanup"] == "succeeded" and len(doc["trials"]) == rounds * 2
            assert doc["comparison"]["greedy_tokens_exact"], (
                "L1 exact greedy tokens changed"
            )
            assert all(
                t["metrics"]["token_s"] > 0 and t["metrics"]["peak_allocated_bytes"] > 0
                for t in doc["trials"]
            )
            assert all(
                t["effective"]["cpu_ids"] == cpus
                and t["effective"]["intra_threads"] == 1
                for t in doc["trials"]
                if t["label"] == "candidate"
            )
            summary[model] = {"run_id": run["id"], "comparison": doc["comparison"]}
            print(model, run["id"], json.dumps(doc["comparison"]), flush=True)
            idle()
        # Driver permission is tested as a real request, never assumed from root.
        power_plan = {
            **plan,
            "rounds": 1,
            "max_output_tokens": 8,
            "baseline": {"power_limit_w": int(original_power)},
            "candidate": {"power_limit_w": max(100, int(original_power) - 100)},
        }
        power = terminal(
            api.call(
                "/system/jobs", {"deployment_id": "qwen-reference", "plan": power_plan}
            )
        )
        api.save("power", power)
        assert power["cleanup"] == "succeeded", power.get("error")
        power_doc = api.call(f"/runs/{power['id']}/system")
        api.save("power-receipt", power_doc)
        assert power_doc.get("power_restored_w") == original_power
        summary["power"] = {
            "run_id": power["id"],
            "state": power["state"],
            "error": power.get("error"),
        }
        # Cancel a long real model job, including board restoration if supported.
        cancel_plan = {**plan, "rounds": 4, "max_output_tokens": 128}
        if power["state"] == "completed":
            cancel_plan["baseline"] = {
                "power_limit_w": max(100, int(original_power) - 100)
            }
        run = api.call(
            "/system/jobs", {"deployment_id": "qwen-reference", "plan": cancel_plan}
        )
        time.sleep(4)
        api.call(f"/requests/{run['id']}/cancel", {})
        cancelled = terminal(run)
        api.save("cancelled", cancelled)
        assert cancelled["state"] == "cancelled" and cancelled["cleanup"] == "succeeded"
        summary["cancel_run_id"] = run["id"]
        assert not Path(".console/l1-power-journal.json").exists()
        # Explicit no-auth and unsupported/admission checks through production API.
        with urllib.request.urlopen(
            api.BASE + "/system/catalog", timeout=15
        ) as response:
            assert response.status == 200
        for patch, reason in [
            ({"device": 63}, "GPU_NOT_VISIBLE"),
            ({"candidate": {"cpu_ids": [8191]}}, "OUTSIDE_EFFECTIVE"),
        ]:
            try:
                api.call(
                    "/system/jobs", {"deployment_id": "qwen-reference", "plan": patch}
                )
            except RuntimeError as exc:
                assert reason in str(exc)
            else:
                raise AssertionError("invalid resource must be rejected")
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
        current = next(
            d for d in api.call("/deployments")["items"] if d["id"] == original["id"]
        )
        assert current["quantization"] == original["quantization"]
        api.save("restored-original", current)
        legacy = api.wait(
            api.call(
                "/requests",
                {
                    "deployment_id": current["id"],
                    "expected_epoch": current["epoch"],
                    "prompt": "Once upon a time",
                    "max_output_tokens": 4,
                    "observation_mode": "basic",
                },
            ),
            "restored-single-request",
        )
        assert legacy["metrics"]["output_tokens"] > 0
        summary["single_request_preserved"] = True
        summary["restored"] = True
        api.save("summary", summary)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in summary.items()
                    if k not in {"qwen-reference", "gemma3-1b"}
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen-reference", "gemma3-1b"])
    parser.add_argument("--rounds", type=int, choices=[1, 2, 3, 4], default=2)
    args = parser.parse_args()
    main(args.models, args.rounds)
