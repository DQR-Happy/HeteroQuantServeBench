"""Production L7 acceptance on the configured target; restores original model."""

import json
import time
from pathlib import Path
from scripts.console import validate_l3 as api, validate_graph_views as graph

OUT = Path("reports/console/l7-decoding")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    api.OUT = graph.OUT = OUT
    assert api.call("/overview")["pending"] == 0
    original = next(
        d
        for d in api.call("/deployments")["items"]
        if d["state"] == "ready" and d["provider"] == "pytorch"
    )
    api.save("original", original)
    recipe = {
        k: v
        for k, v in original["detail"]["precision_receipt"]["resolved_recipe"].items()
        if k != "calibration"
    }
    if original.get("quantization", {}).get("artifact_id"):
        recipe["artifact_id"] = original["quantization"]["artifact_id"]
    summary = {
        "models": {},
        "restored": False,
        "speedup_claim": False,
        "sampling_distribution_claim": False,
    }
    prompt = "1 2 3 4 5 1 2 3 4 5 1 2 3 4 5"

    def d(identity):
        return next(d for d in api.call("/deployments")["items"] if d["id"] == identity)

    def generation(dep, label, plan=None, **kwargs):
        body = {
            "deployment_id": dep["id"],
            "expected_epoch": dep["epoch"],
            "prompt": prompt,
            "max_output_tokens": 32,
            "fixed_output_length": True,
            "save_input": True,
            "observation_mode": "off",
            "optimization": {"changes": {}},
            **kwargs,
        }
        if plan is not None:
            body["decoding_plan"] = plan
        return api.wait(api.call("/requests", body), label)

    def compare(a, b, label):
        result = api.call(
            "/decoding/comparisons",
            {"baseline_run_id": a["id"], "candidate_run_id": b["id"]},
        )
        api.save(label + "-comparison", result)
        assert result["comparable"], result
        return result

    try:
        api.unload_all("initial")
        graph.restart()
        for identity in ("qwen-reference", "gemma3-1b"):
            print("MODEL", identity, flush=True)
            api.unload_all("switch")
            api.wait(
                api.call(
                    "/deployments/" + identity + "/load",
                    {
                        "precision": "float16",
                        "kernel_profile": "framework-native",
                        "quantization": {"method": "none"},
                    },
                ),
                identity + "-load",
            )
            dep = d(identity)
            row = summary["models"][identity] = {}
            legacy = generation(dep, identity + "-legacy")
            base = generation(dep, identity + "-greedy", {})
            assert (
                legacy["metrics"]["generated_token_ids"]
                == base["metrics"]["generated_token_ids"]
            )
            lookup = generation(dep, identity + "-lookup", {"method": "prompt_lookup"})
            row["greedy_run"] = base["id"]
            row["lookup_run"] = lookup["id"]
            row["lookup"] = compare(base, lookup, identity + "-lookup")
            sample = {
                "sampling": "multinomial",
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.95,
                "seed": 7,
                "repetition_penalty": 1.1,
                "min_new_tokens": 2,
            }
            a = generation(dep, identity + "-sample-a", sample)
            b = generation(dep, identity + "-sample-b", sample)
            assert (
                a["metrics"]["generated_token_ids"]
                == b["metrics"]["generated_token_ids"]
            )
            row["seed_reproducible_runs"] = [a["id"], b["id"]]
            choice = {"choices": ["positive", "negative", "neutral"]}
            args = {
                "prompt": "Classify the sentiment. Reply with positive, negative, or neutral.\nText: I love this movie!\nAnswer:",
                "fixed_output_length": False,
            }
            a = generation(
                dep,
                identity + "-mask",
                {**choice, "constraint_backend": "mask"},
                **args,
            )
            b = generation(
                dep,
                identity + "-gather",
                {**choice, "constraint_backend": "gather"},
                **args,
            )
            assert a["metrics"]["decoding_receipt"]["constraint_satisfied"] is True
            row["constraints"] = compare(a, b, identity + "-constraints")
            basic = generation(
                dep,
                identity + "-basic",
                {},
                observation_mode="basic",
                max_output_tokens=8,
            )
            assert (
                basic["metrics"]["observation_bundle"]["publication"]
                == "supervisor_acknowledged"
            )
            row["basic_run"] = basic["id"]
            if identity == "qwen-reference":
                # Current node contains only Qwen 1.7B and Gemma 1B after user cleanup.
                # A separate identical draft validates all-accept, not small-draft speed.
                failed = api.call(
                    "/requests",
                    {
                        "deployment_id": identity,
                        "expected_epoch": dep["epoch"],
                        "prompt": prompt,
                        "max_output_tokens": 8,
                        "observation_mode": "off",
                        "decoding_plan": {
                            "method": "assistant",
                            "draft_deployment_id": "gemma3-1b",
                        },
                    },
                )
                end = time.monotonic() + 90
                while failed["state"] not in {
                    "completed",
                    "failed",
                    "cancelled",
                    "timed_out",
                }:
                    assert time.monotonic() < end
                    time.sleep(0.1)
                    failed = api.call("/runs/" + failed["id"])
                api.save("mismatched-tokenizer", failed)
                assert (
                    failed["state"] == "failed"
                    and "TOKENIZER_MISMATCH" in failed["error"]
                )
                assert (
                    d(identity)["state"] == "ready"
                    and d(identity)["epoch"] == dep["epoch"]
                )
                cold = generation(
                    dep,
                    "assistant-cold",
                    {"method": "assistant", "draft_deployment_id": identity},
                )
                warm = generation(
                    dep,
                    "assistant-warm",
                    {"method": "assistant", "draft_deployment_id": identity},
                )
                assert (
                    cold["metrics"]["decoding_receipt"]["draft_loaded_this_request"]
                    is True
                )
                assert (
                    warm["metrics"]["decoding_receipt"]["draft_loaded_this_request"]
                    is False
                )
                assert (
                    warm["metrics"]["decoding_receipt"]["statistics"]["accepted_tokens"]
                    > 0
                )
                row["assistant_cold"] = compare(base, cold, "assistant-cold")
                row["assistant_warm"] = compare(base, warm, "assistant-warm")
                resident = api.call("/decoding/status?deployment_id=" + identity)
                assert resident["status"] == "loaded"
                release = api.wait(
                    api.call(
                        "/decoding/draft/unload",
                        {"deployment_id": identity, "expected_epoch": dep["epoch"]},
                    ),
                    "assistant-unload",
                )
                assert (
                    release["metrics"]["decoding_draft_release"]["current"]["status"]
                    == "not_loaded"
                )
                assert (
                    d(identity)["state"] == "ready"
                    and d(identity)["epoch"] == dep["epoch"]
                )
                row["release_run"] = release["id"]
                cancel = api.call(
                    "/requests",
                    {
                        "deployment_id": identity,
                        "expected_epoch": dep["epoch"],
                        "prompt": prompt,
                        "max_output_tokens": 512,
                        "observation_mode": "off",
                        "fixed_output_length": True,
                        "decoding_plan": {},
                    },
                )
                end = time.monotonic() + 30
                while (cancel["metrics"].get("output_tokens") or 0) < 2 and cancel[
                    "state"
                ] not in {"failed", "completed"}:
                    assert time.monotonic() < end
                    time.sleep(0.05)
                    cancel = api.call("/runs/" + cancel["id"])
                api.call("/requests/" + cancel["id"] + "/cancel", {})
                while cancel["state"] not in {
                    "cancelled",
                    "failed",
                    "completed",
                    "timed_out",
                }:
                    time.sleep(0.05)
                    cancel = api.call("/runs/" + cancel["id"])
                assert (
                    cancel["state"] == "cancelled" and cancel["cleanup"] == "succeeded"
                )
                assert cancel["metrics"]["decoding_receipt"]["cleanup"][
                    "target_request_cache_released"
                ]
                api.save("cancelled", cancel)
                row["cancel_run"] = cancel["id"]
            # Existing L6 and ordinary generation retain their contracts.
            after = generation(
                dep, identity + "-legacy-after", memory_plan={"mode": "dynamic"}
            )
            assert (
                after["metrics"]["generated_token_ids"]
                == legacy["metrics"]["generated_token_ids"]
            )
            row["legacy_and_l6_preserved"] = True
            api.save("summary", summary)
            print(
                identity,
                "lookup accept",
                lookup["metrics"]["decoding_receipt"]["statistics"]["acceptance_rate"],
                flush=True,
            )
    finally:
        print("RESTORE", original["id"], flush=True)
        if "cancel" in locals():
            owned = api.call("/runs/" + cancel["id"])
            if owned["state"] not in {
                "completed",
                "failed",
                "cancelled",
                "timed_out",
                "interrupted",
            }:
                api.call("/requests/" + owned["id"] + "/cancel", {})
                end = time.monotonic() + 30
                while api.call("/overview")["pending"]:
                    assert time.monotonic() < end
                    time.sleep(0.1)
        api.unload_all("restore")
        api.wait(
            api.call(
                "/deployments/" + original["id"] + "/load",
                {
                    "precision": original["precision"],
                    "kernel_profile": original["kernel_profile"],
                    "quantization": recipe,
                    **(
                        {"structure_artifact_id": original["structure_artifact_id"]}
                        if original.get("structure_artifact_id")
                        else {}
                    ),
                },
            ),
            "restored-load",
        )
        current = d(original["id"])
        assert current["quantization"] == original["quantization"] and current.get(
            "structure_artifact_id"
        ) == original.get("structure_artifact_id")
        summary["restored"] = True
        api.save("restored", current)
        api.save("summary", summary)
    print(
        json.dumps(
            {"restored": summary["restored"], "models": list(summary["models"])}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
