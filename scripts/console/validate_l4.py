"""Real-model L4 API validation on an isolated control port; restore prior model.

Run exclusively through scripts/remote_run.sh. Calibration/evaluation texts are
separate explicit fixtures, not a benchmark dataset or a universal quality gate.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
import yaml

OUT = Path("reports/console/l4")
BASE = "http://127.0.0.1:8769/api/console/v1"
MAIN = "http://127.0.0.1:8768/api/console/v1"
END = {"completed", "failed", "cancelled", "timed_out", "interrupted"}
CALIBRATION = [
    "Once there was a small rabbit who lived near a river. Every morning it visited its friends.",
    "A boy found an old key in the garden. He asked his mother what the key could open.",
    "The blue bird flew over the mountains and returned home before sunset.",
    "A kind teacher explained how plants grow when they have water and sunlight.",
    "The little puppy was hungry. It found a bowl of food beside the kitchen door.",
    "Three children built a small house out of wood and painted it yellow.",
    "Rain fell on the village all night. In the morning the sky was clear again.",
    "The farmer carried fresh apples to the market and shared them with a friend.",
]
VARIANTS = {
    "fp16": ("float16", {}),
    "fp32": ("float32", {}),
    "bf16": ("bfloat16", {}),
    "rtn4": ("float16", {"method": "rtn", "bits": 4, "group_size": 32}),
    "rtn8": ("float16", {"method": "rtn", "bits": 8}),
    "gptq": (
        "float16",
        {
            "method": "gptq",
            "bits": 4,
            "group_size": 32,
            "calibration": CALIBRATION,
            "calibration_tokens": 64,
        },
    ),
    "awq": (
        "float16",
        {
            "method": "awq",
            "bits": 4,
            "group_size": 32,
            "calibration": CALIBRATION,
            "calibration_tokens": 64,
        },
    ),
    "smoothquant": (
        "float16",
        {"method": "smoothquant", "calibration": CALIBRATION, "calibration_tokens": 64},
    ),
    "bnb_int8": ("float16", {"method": "bnb_int8"}),
    "bnb_nf4": ("float16", {"method": "bnb_nf4"}),
    "bnb_fp4": ("float16", {"method": "bnb_fp4"}),
    "kv4": ("float16", {"kv_bits": 4, "kv_group_size": 32, "kv_residual_length": 4}),
}
VARIANTS["gptq8"] = ("float16", {**VARIANTS["gptq"][1], "bits": 8})


def call(path, body=None, base=BASE):
    req = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "X-HQSB-Client": "console",
            "Idempotency-Key": uuid.uuid4().hex,
        },
    )
    try:
        with urllib.request.urlopen(
            req, timeout=180 if path == "/precision/comparisons" else 30
        ) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{path}: HTTP {exc.code}: {exc.read().decode()}") from exc


def wait(run, base=BASE):
    for _ in range(1500):
        run = call("/runs/" + run["id"], base=base)
        if run["state"] in END:
            return run
        time.sleep(0.2)
    raise TimeoutError(run["id"])


def lifecycle(identity, action, body=None, base=BASE):
    run = wait(call(f"/deployments/{identity}/{action}", body or {}, base), base)
    assert run["state"] == "completed", (action, identity, run.get("error"))
    return run


def save(name, body):
    (OUT / (name + ".json")).write_text(json.dumps(body, ensure_ascii=False, indent=2))


def main(methods, model_name="LLM-Research/tinyllama-15M-stories", exclusions=None):
    OUT.mkdir(parents=True, exist_ok=True)
    before = call("/overview", base=MAIN)
    assert before["pending"] == 0
    assert call("/scheduling-benchmarks", base=MAIN)["active_id"] is None
    ready = [
        d
        for d in before["deployments"]
        if d["provider"] == "pytorch" and d["state"] == "ready"
    ]
    save(
        "original-models",
        [
            {k: d.get(k) for k in ("id", "kernel_profile", "precision", "quantization")}
            for d in ready
        ],
    )
    configs = json.loads(Path(".console/model-hub/deployments.json").read_text())
    tiny = next(c for c in configs.values() if c["model"] == model_name)
    qwen = next(c for c in configs.values() if c["model"] == "Qwen/Qwen3-0.6B")
    state = Path(".console/l4-validation").resolve()
    config_path = OUT / "control.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "deployments": list({c["id"]: c for c in (tiny, qwen)}.values()),
                "auth_required": False,
                "machine_name": "3090 L4 isolated validation",
            }
        )
    )
    process = None
    summary = {
        "variants": {},
        "restored": [],
        "scope": "explicit_heldout_fixture_not_general_quality_claim",
    }
    try:
        for row in ready:
            lifecycle(row["id"], "unload", base=MAIN)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/hardware/test_precision_cuda.py",
                "-q",
                "--tb=short",
            ],
            check=True,
        )
        log = (OUT / "service.log").open("w")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "hqsb.console",
                "--port",
                "8769",
                "--config",
                str(config_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "HQSB_CONSOLE_DATA": str(state)},
        )
        for _ in range(200):
            try:
                call("/session")
                break
            except (OSError, RuntimeError):
                if process.poll() is not None:
                    raise RuntimeError("Validation service exited; inspect service.log")
                time.sleep(0.1)
        else:
            raise TimeoutError("Validation service startup")
        quality_ids = []
        for label in methods:
            dtype, recipe = VARIANTS[label]
            recipe = dict(recipe)
            if exclusions and recipe.get("method"):
                recipe["exclude_modules"] = exclusions
            print("START", label, flush=True)
            loaded = lifecycle(
                tiny["id"],
                "load",
                {
                    "kernel_profile": "framework-native",
                    "precision": dtype,
                    "quantization": recipe,
                },
            )
            save(label + "-load", loaded)
            deployment = next(
                d for d in call("/deployments")["items"] if d["id"] == tiny["id"]
            )
            if label == methods[0]:
                save(
                    "capabilities",
                    call(f"/deployments/{tiny['id']}/precision-capabilities"),
                )
            run = wait(
                call(
                    "/requests",
                    {
                        "deployment_id": tiny["id"],
                        "expected_epoch": deployment["epoch"],
                        "prompt": "Once upon a time a little fox found a ball. The fox",
                        "max_output_tokens": 16,
                        "observation_mode": "basic",
                    },
                )
            )
            save(label + "-inference", run)
            assert run["state"] == "completed", run.get("error")
            assert run["metrics"]["output_tokens_per_s"] > 0
            repeats = [run]
            for repeat in range(2):
                repeated = wait(
                    call(
                        "/requests",
                        {
                            "deployment_id": tiny["id"],
                            "expected_epoch": deployment["epoch"],
                            "prompt": "Once upon a time a little fox found a ball. The fox",
                            "max_output_tokens": 16,
                            "observation_mode": "basic",
                        },
                    )
                )
                assert repeated["state"] == "completed", repeated.get("error")
                save(f"{label}-inference-warm-{repeat}", repeated)
                repeats.append(repeated)
            quality = wait(
                call(
                    "/precision/quality-jobs",
                    {
                        "deployment_id": tiny["id"],
                        "expected_epoch": deployment["epoch"],
                        "quality_plan": {
                            "samples": [
                                {
                                    "id": "heldout-a",
                                    "text": "A red cat sat beside the window.",
                                },
                                {
                                    "id": "heldout-b",
                                    "text": "The young girl smiled at her sister.",
                                },
                            ],
                            "max_tokens": 32,
                            "generation_tokens": 4,
                        },
                    },
                )
            )
            save(label + "-quality", quality)
            assert quality["state"] == "completed", quality.get("error")
            quality_ids.append(quality["id"])
            summary["variants"][label] = {
                "load_run_id": loaded["id"],
                "inference_run_id": run["id"],
                "quality_run_id": quality["id"],
                "tokens_per_s": run["metrics"]["output_tokens_per_s"],
                "quality": quality["metrics"]["quality_evaluation"],
                "performance_repeats": [
                    {
                        "run_id": r["id"],
                        "tokens_per_s": r["metrics"]["output_tokens_per_s"],
                        "output_tokens": r["metrics"]["output_tokens"],
                        "ttft_ms": r["metrics"]["runtime_first_token_ms"],
                    }
                    for r in repeats
                ],
                "receipt": run["metrics"]["optimization_receipt"]["precision_receipt"],
                "memory": run["metrics"]["memory"],
                "quantized_kv_inventory": run["metrics"].get("quantized_kv_inventory"),
            }
            if label in {"rtn4", "bnb_nf4"}:
                schedule = wait(
                    call(
                        "/scheduling-benchmarks",
                        {
                            "deployment_id": tiny["id"],
                            "expected_epoch": deployment["epoch"],
                            "modes": ["serial", "continuous"],
                            "requests": [
                                {
                                    "id": "a",
                                    "prompt": "Once upon a time",
                                    "output_tokens": 8,
                                },
                                {
                                    "id": "b",
                                    "prompt": "A little dog",
                                    "output_tokens": 8,
                                },
                            ],
                        },
                    )
                )
                save(label + "-l8", schedule)
                assert schedule["state"] == "completed", schedule.get("error")
            artifact_id = (
                loaded.get("result", {}).get("precision_receipt", {}).get("artifact_id")
            )
            if artifact_id:
                lifecycle(tiny["id"], "unload")
                reloaded = lifecycle(
                    tiny["id"],
                    "load",
                    {
                        "precision": dtype,
                        "quantization": {
                            "method": recipe["method"],
                            "artifact_id": artifact_id,
                        },
                    },
                )
                save(label + "-reload", reloaded)
                state_now = next(
                    d for d in call("/deployments")["items"] if d["id"] == tiny["id"]
                )
                replayed = wait(
                    call(
                        "/requests",
                        {
                            "deployment_id": tiny["id"],
                            "expected_epoch": state_now["epoch"],
                            "prompt": "Once upon a time a little fox found a ball. The fox",
                            "max_output_tokens": 16,
                            "observation_mode": "basic",
                        },
                    )
                )
                save(label + "-reload-inference", replayed)
                assert replayed["state"] == "completed", replayed.get("error")
                assert (
                    replayed["metrics"]["generated_token_ids"]
                    == run["metrics"]["generated_token_ids"]
                ), "persisted model changed output"
            lifecycle(tiny["id"], "unload")
            print("DONE", label, run["id"], flush=True)
            save("summary", summary)
        if len(quality_ids) >= 2:
            for start in range(1, len(quality_ids), 8):
                save(
                    "quality-comparison-" + str(start),
                    call(
                        "/precision/comparisons",
                        {
                            "baseline_run_id": quality_ids[0],
                            "candidate_run_ids": quality_ids[start : start + 8],
                        },
                    ),
                )
        failed = wait(
            call(f"/deployments/{tiny['id']}/load", {"quantization": {"method": "fp8"}})
        )
        save("fp8-unsupported", failed)
        assert (
            failed["state"] == "failed"
            and "DEVICE_SM_BELOW_REQUIRED" in failed["error"]
        )
        # Recover immediately after unsupported load; old single-request path stays usable.
        lifecycle(tiny["id"], "load")
        lifecycle(tiny["id"], "unload")
        summary["status"] = "completed"
    finally:
        if process:
            try:
                for d in call("/deployments")["items"]:
                    if d["state"] == "ready":
                        lifecycle(d["id"], "unload")
            finally:
                process.terminate()
                process.wait(timeout=30)
        for row in ready:
            previous_plan = (
                row.get("detail", {}).get("precision_artifact", {}).get("requested", {})
            )
            previous_quant = row.get("quantization", {})
            if previous_quant.get("artifact_id"):
                previous_plan = {
                    **previous_plan,
                    "method": previous_quant["method"],
                    "artifact_id": previous_quant["artifact_id"],
                }
            lifecycle(
                row["id"],
                "load",
                {
                    "kernel_profile": row.get("kernel_profile", "framework-native"),
                    "precision": row["precision"],
                    **({"quantization": previous_plan} if previous_plan else {}),
                },
                base=MAIN,
            )
            summary["restored"].append(row["id"])
        save("summary", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(VARIANTS))
    parser.add_argument("--model", default="LLM-Research/tinyllama-15M-stories")
    parser.add_argument("--output", default=str(OUT))
    parser.add_argument("--exclude", default="")
    args = parser.parse_args()
    OUT = Path(args.output)
    main(
        args.methods.split(","),
        args.model,
        args.exclude.split(",") if args.exclude else None,
    )
