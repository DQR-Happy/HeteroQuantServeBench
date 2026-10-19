"""Bounded L4 output ablation through the API; restore the resident recipe.

Run via remote_run.sh on the configured execution target. This diagnoses one
artifact and explicit prompts, and does not certify model quality or speed.
"""

import json
import time
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8768/api/console/v1"
OUT = Path("reports/console/l4-output-diagnosis")
TERMINAL = {"completed", "failed", "cancelled", "timed_out", "interrupted"}


def call(path, body=None):
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Content-Type": "application/json",
            "X-HQSB-Client": "console",
            "Idempotency-Key": uuid.uuid4().hex,
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def wait(run, label):
    deadline = time.monotonic() + 360
    while run["state"] not in TERMINAL:
        if time.monotonic() > deadline:
            raise TimeoutError(run["id"])
        time.sleep(0.3)
        run = call("/runs/" + run["id"])
    (OUT / (label + ".json")).write_text(json.dumps(run, ensure_ascii=False, indent=2))
    if run["state"] != "completed" or run["cleanup"] != "succeeded":
        raise RuntimeError((label, run["state"], run.get("error")))
    return run


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    overview = call("/overview")
    if overview["pending"] or call("/scheduling-benchmarks")["active_id"]:
        raise RuntimeError("Wait for existing device tasks")
    original = next(
        d
        for d in overview["deployments"]
        if d["state"] == "ready" and d["provider"] == "pytorch"
    )
    if original.get("quantization", {}).get("method") != "gptq":
        raise RuntimeError("Diagnostic expects the reported GPTQ deployment")
    (OUT / "original-deployment.json").write_text(
        json.dumps(original, ensure_ascii=False, indent=2)
    )
    identity = original["id"]
    recipe = original["detail"]["precision_receipt"]["resolved_recipe"]
    restore = {
        "method": "gptq",
        "artifact_id": original["quantization"]["artifact_id"],
        **{
            k: recipe[k]
            for k in ("kv_bits", "kv_group_size", "kv_residual_length", "allow_tf32")
        },
    }
    variants = [
        ("gptq_kv2", restore),
        ("gptq_default_kv", {**restore, "kv_bits": None}),
        ("fp16_default_kv", {"method": "none"}),
        (
            "fp16_kv2",
            {
                "method": "none",
                **{
                    k: recipe[k]
                    for k in ("kv_bits", "kv_group_size", "kv_residual_length")
                },
            },
        ),
    ]
    summary = {"source_run": "run_bc60343d9a2b4d26a1338100fa5eab8a", "variants": {}}

    def snapshot():
        (OUT / "ablation-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
        )

    def apply(plan, label):
        current = next(d for d in call("/deployments")["items"] if d["id"] == identity)
        if current["state"] == "ready":
            wait(call(f"/deployments/{identity}/unload", {}), label + "-unload")
        elif current["state"] not in {"unloaded", "failed"}:
            raise RuntimeError("Unexpected lifecycle owner")
        wait(
            call(
                f"/deployments/{identity}/load",
                {
                    "precision": "float16",
                    "kernel_profile": "framework-native",
                    "quantization": plan,
                },
            ),
            label + "-load",
        )

    try:
        for index, (label, plan) in enumerate(variants):
            print("START", label, flush=True)
            if index:
                apply(plan, label)
            current = next(
                d for d in call("/deployments")["items"] if d["id"] == identity
            )
            result = {
                "epoch": current["epoch"],
                "quantization": current.get("quantization"),
                "runs": {},
            }
            summary["variants"][label] = result
            for mode, content in [
                (
                    "completion",
                    {"prompt": "Once upon a time a little fox found a ball. The fox"},
                ),
                (
                    "chat",
                    {
                        "messages": [
                            {"role": "user", "content": "请用一句话解释什么是变压器。"}
                        ]
                    },
                ),
            ]:
                run = wait(
                    call(
                        "/requests",
                        {
                            "deployment_id": identity,
                            "expected_epoch": current["epoch"],
                            "max_output_tokens": 16 if mode == "completion" else 48,
                            "observation_mode": "basic",
                            **content,
                        },
                    ),
                    label + "-" + mode,
                )
                result["runs"][mode] = {
                    "id": run["id"],
                    "output": run["output"],
                    "tokens_per_s": run["metrics"].get("output_tokens_per_s"),
                }
                print(label, mode, repr(run["output"]), flush=True)
                snapshot()
            run = wait(
                call(
                    "/precision/quality-jobs",
                    {
                        "deployment_id": identity,
                        "expected_epoch": current["epoch"],
                        "quality_plan": {
                            "samples": [
                                {
                                    "id": "heldout-0",
                                    "text": "A red cat sat beside the window.",
                                },
                                {
                                    "id": "heldout-1",
                                    "text": "The young girl smiled at her sister.",
                                },
                            ],
                            "max_tokens": 64,
                            "generation_tokens": 4,
                        },
                    },
                ),
                label + "-quality",
            )
            result["runs"]["quality"] = {
                "id": run["id"],
                **run["metrics"]["quality_evaluation"],
            }
            print(
                label,
                "PPL",
                run["metrics"]["quality_evaluation"]["perplexity"],
                flush=True,
            )
            snapshot()
    finally:
        print("RESTORING", flush=True)
        apply(restore, "restore-original")
        actual = next(d for d in call("/deployments")["items"] if d["id"] == identity)
        summary["restored"] = {
            "state": actual["state"],
            "quantization": actual["quantization"],
            "epoch": actual["epoch"],
        }
        snapshot()
        print("RESTORED", actual["state"], flush=True)


if __name__ == "__main__":
    main()
