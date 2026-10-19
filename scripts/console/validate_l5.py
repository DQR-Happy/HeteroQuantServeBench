"""Target-only L5 production acceptance. Original resident recipe is restored."""

import time
import hashlib
from hqsb.core.structure_plan import StructurePlan
from pathlib import Path
from scripts.console import validate_l3 as api
from scripts.console import validate_graph_views as graph

OUT = Path("reports/console/l5-structure")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    api.OUT = OUT
    graph.OUT = OUT
    assert api.call("/overview")["pending"] == 0
    assert api.call("/scheduling-benchmarks")["active_id"] is None
    original = next(
        d
        for d in api.call("/deployments")["items"]
        if d["state"] == "ready" and d["provider"] == "pytorch"
    )
    assert (
        api.call("/benchmark-catalog?deployment_id=" + original["id"])["active_id"]
        is None
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
        "quality_preservation_claim": False,
    }

    def load(identity, artifact=None):
        api.unload_all("switch")
        result = api.wait(
            api.call(
                f"/deployments/{identity}/load",
                {
                    "precision": "float16",
                    "kernel_profile": "framework-native",
                    "quantization": {"method": "none"},
                    **({"structure_artifact_id": artifact} if artifact else {}),
                },
            ),
            identity + "-" + (artifact or "original") + "-load",
        )
        return next(
            d for d in api.call("/deployments")["items"] if d["id"] == identity
        ), result

    def quality(d, label):
        return api.wait(
            api.call(
                "/precision/quality-jobs",
                {
                    "deployment_id": d["id"],
                    "expected_epoch": d["epoch"],
                    "quality_plan": {"benchmark_id": "wikitext2-mini-v1"},
                },
            ),
            label + "-quality",
        )

    def generation(d, label, **extra):
        return api.wait(
            api.call(
                "/requests",
                {
                    "deployment_id": d["id"],
                    "expected_epoch": d["epoch"],
                    "prompt": "Once upon a time, there was a little fox. The fox",
                    "max_output_tokens": 16,
                    "fixed_output_length": True,
                    "save_input": True,
                    "observation_mode": "off",
                    "optimization": {"changes": {}},
                    **extra,
                },
            ),
            label + "-generation",
        )

    try:
        api.unload_all("initial")
        graph.restart()
        for identity, methods in [
            (
                "ms-586060631c81c17023c0",
                ["magnitude", "wanda", "mlp_channels", "svd_low_rank"],
            ),
            ("ms-6d9b6a4a3f944e041fd6", ["mlp_channels", "svd_low_rank"]),
            ("gemma3-1b", ["mlp_channels"]),
        ]:
            print("MODEL", identity, flush=True)
            d, _ = load(identity)
            base_quality = quality(d, identity + "-original")
            base_generation = generation(d, identity + "-original")
            caps = api.call("/structure/capabilities?deployment_id=" + identity)
            results = summary["models"][identity] = {
                "baseline_quality": base_quality["id"],
                "baseline_generation": base_generation["id"],
                "methods": {},
            }
            for method in methods:
                if d.get("structure_artifact_id"):
                    d, _ = load(identity)
                before_epoch = d["epoch"]
                name = (
                    caps["groups"][0]["name"]
                    if method == "mlp_channels"
                    else next(
                        x["name"]
                        for x in caps["linears"]
                        if x["name"].endswith("down_proj")
                    )
                )
                plan = {
                    "method": method,
                    "targets": [name],
                    "max_modules": 1,
                    "keep_ratio": 0.5,
                    "sparsity": 0.5,
                    **(
                        {
                            "calibration": [
                                "A small rabbit lived near a river and visited its friends.",
                                "The children planted flowers and watched them grow.",
                            ],
                            "calibration_max_tokens": 32,
                        }
                        if method == "wanda"
                        else {}
                    ),
                }
                existing = api.call("/structure/artifacts?deployment_id=" + identity)[
                    "items"
                ]
                expected_recipe = StructurePlan.model_validate(plan).model_dump(
                    exclude={"calibration"}
                )
                reuse = next(
                    (
                        a
                        for a in existing
                        if a["source_model_identity"]
                        == base_quality["metrics"]["optimization_receipt"][
                            "model_identity"
                        ]
                        and a["recipe"] == expected_recipe
                    ),
                    None,
                )
                if reuse and method == "wanda":
                    hashes = {
                        hashlib.sha256(t.encode()).hexdigest()
                        for t in plan["calibration"]
                    }
                    details = api.call("/structure/artifacts/" + reuse["artifact_id"])
                    if not hashes.issubset(details["calibration_text_sha256"]):
                        reuse = None
                if reuse:
                    job = api.call("/runs/" + reuse["creation_run_id"])
                else:
                    job = api.wait(
                        api.call(
                            "/structure/jobs",
                            {
                                "deployment_id": identity,
                                "expected_epoch": d["epoch"],
                                "plan": plan,
                            },
                        ),
                        identity + "-" + method + "-build",
                    )
                artifact = job["metrics"]["structure_artifact"]["artifact_id"]
                still = next(
                    x for x in api.call("/deployments")["items"] if x["id"] == identity
                )
                assert (
                    still["state"] == "ready"
                    and still["epoch"] == before_epoch
                    and not still.get("structure_artifact_id")
                )
                doc = api.call("/structure/artifacts/" + artifact)
                assert doc["publication"] == "supervisor_acknowledged"
                d, _ = load(identity, artifact)
                assert d["structure_artifact_id"] == artifact
                if method in {"mlp_channels", "svd_low_rank"}:
                    assert (
                        d["detail"]["parameter_bytes"]
                        < doc["statistics"]["same_dtype_parameter_bytes_before"]
                    )
                else:
                    assert (
                        doc["statistics"]["parameters_before"]
                        == doc["statistics"]["parameters_after"]
                    )
                q = quality(d, identity + "-" + method)
                g = generation(d, identity + "-" + method)
                assert (
                    q["metrics"]["optimization_receipt"]["structure_receipt"]["status"]
                    == "invoked"
                )
                comparison = api.call(
                    "/structure/comparisons",
                    {
                        "baseline_run_id": base_quality["id"],
                        "candidate_run_id": q["id"],
                    },
                )
                perf = api.call(
                    "/structure/comparisons",
                    {
                        "baseline_run_id": base_generation["id"],
                        "candidate_run_id": g["id"],
                    },
                )
                api.save(identity + "-" + method + "-quality-comparison", comparison)
                api.save(identity + "-" + method + "-performance-comparison", perf)
                assert comparison["comparable"], comparison
                assert perf["comparable"], perf
                eligibility = api.call("/runs/" + q["id"] + "/quality-baseline")
                assert not eligibility["reference_eligible"]
                architecture = api.call("/deployments/" + identity + "/architecture")
                assert (
                    architecture["runtime"]["structure_receipt"]["artifact_id"]
                    == artifact
                )
                result = results["methods"][method] = {
                    "artifact_id": artifact,
                    "build_run_id": job["id"],
                    "quality_run_id": q["id"],
                    "generation_run_id": g["id"],
                    "statistics": doc["statistics"],
                    "quality": comparison["quality"],
                    "performance": perf["performance"],
                }
                if method == "mlp_channels" and identity == "ms-586060631c81c17023c0":
                    memory = generation(
                        d, identity + "-l6", memory_plan={"mode": "dynamic"}
                    )
                    assert memory["metrics"]["memory_receipt"]["status"] == "applied"
                    result["l6_run_id"] = memory["id"]
                    spec = {
                        "deployment_id": identity,
                        "expected_epoch": d["epoch"],
                        "modes": ["serial", "continuous"],
                        "requests": [
                            {
                                "id": "a",
                                "prompt": "Once upon a time",
                                "output_tokens": 8,
                            },
                            {"id": "b", "prompt": "A little fox", "output_tokens": 8},
                        ],
                        "limits": {
                            "kv_layout": "block_pool",
                            "kv_capacity_tokens": 128,
                            "kv_pool_max_bytes": 1024**2,
                            "max_batched_tokens": 32,
                            "prefill_chunk_tokens": 8,
                        },
                    }
                    e = api.call("/scheduling-benchmarks", spec)
                    while e["state"] not in {
                        "completed",
                        "failed",
                        "cancelled",
                        "timed_out",
                        "interrupted",
                    }:
                        time.sleep(0.5)
                        e = api.call("/scheduling-benchmarks/" + e["id"])
                    assert e["state"] == "completed", e.get("error")
                    assert all(
                        v["structure_receipt"]["status"] == "invoked"
                        and v["cleanup"]["device_kv_rows"] == 0
                        for v in e["result"]["variants"]
                    )
                    api.save(identity + "-l8", e)
                    result["l8_experiment_id"] = e["id"]
                print(
                    method,
                    artifact,
                    "PPL",
                    comparison["quality"].get("baseline_perplexity"),
                    "->",
                    comparison["quality"].get("candidate_perplexity"),
                    "parameters",
                    round(doc["statistics"]["parameter_reduction"] * 100, 3),
                    "%",
                    flush=True,
                )
                api.save("summary", summary)
    finally:
        print("RESTORE", original["id"], flush=True)
        api.unload_all("restore")
        api.wait(
            api.call(
                f"/deployments/{original['id']}/load",
                {
                    "precision": original["precision"],
                    "kernel_profile": original.get(
                        "kernel_profile", "framework-native"
                    ),
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
        current = next(
            d for d in api.call("/deployments")["items"] if d["id"] == original["id"]
        )
        assert current["quantization"] == original["quantization"] and current.get(
            "structure_artifact_id"
        ) == original.get("structure_artifact_id")
        summary["restored"] = True
        api.save("restored", current)
        api.save("summary", summary)


if __name__ == "__main__":
    main()
