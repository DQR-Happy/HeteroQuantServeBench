"""Target-only real-model corpus PPL acceptance; restore the original artifact.

The optional original snapshot supports a controlled API service restart. Data
and protocols are fixed by the server, never selected by observed model scores.
"""

import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8768/api/console/v1"
OUT = Path("reports/console/quality-benchmark")


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
    end = time.monotonic() + 360
    while run["state"] not in {
        "completed",
        "failed",
        "cancelled",
        "timed_out",
        "interrupted",
    }:
        if time.monotonic() > end:
            raise TimeoutError(run["id"])
        time.sleep(0.5)
        run = call("/runs/" + run["id"])
    (OUT / (label + ".json")).write_text(json.dumps(run, ensure_ascii=False, indent=2))
    assert run["state"] == "completed" and run["cleanup"] == "succeeded", (
        label,
        run.get("error"),
        run["state"],
    )
    return run


def main(snapshot=None):
    OUT.mkdir(parents=True, exist_ok=True)
    before = call("/overview")
    assert before["pending"] == 0
    assert call("/scheduling-benchmarks")["active_id"] is None
    original = (
        json.loads(Path(snapshot).read_text())[0]
        if snapshot
        else next(
            d
            for d in before["deployments"]
            if d["state"] == "ready" and d["provider"] == "pytorch"
        )
    )
    recipe = original["detail"]["precision_receipt"]["resolved_recipe"]
    identity = original["id"]
    assert call("/benchmark-catalog?deployment_id=" + identity)["active_id"] is None
    restore = {k: v for k, v in recipe.items() if k != "calibration"}
    if original["quantization"].get("artifact_id"):
        restore["artifact_id"] = original["quantization"]["artifact_id"]
    assert restore["method"] == "gptq", (
        "This acceptance needs the reported GPTQ artifact"
    )
    summary = {
        "benchmark": call("/precision/quality-benchmarks")["items"][0],
        "variants": {},
    }

    def apply(plan, label):
        current = next(d for d in call("/deployments")["items"] if d["id"] == identity)
        if current["state"] == "ready":
            wait(call(f"/deployments/{identity}/unload", {}), label + "-unload")
        wait(
            call(
                f"/deployments/{identity}/load",
                {
                    "precision": original["precision"],
                    "kernel_profile": "framework-native",
                    "quantization": plan,
                },
            ),
            label + "-load",
        )
        return next(d for d in call("/deployments")["items"] if d["id"] == identity)

    try:
        for label, plan in [
            ("fp16", {"method": "none"}),
            ("gptq", {**restore, "kv_bits": None}),
            ("gptq-kv2", restore),
        ]:
            print("START", label, flush=True)
            current = apply(plan, label)
            run = wait(
                call(
                    "/precision/quality-jobs",
                    {
                        "deployment_id": identity,
                        "expected_epoch": current["epoch"],
                        "quality_plan": {"benchmark_id": summary["benchmark"]["id"]},
                    },
                ),
                label + "-quality",
            )
            metrics = run["metrics"]["quality_evaluation"]
            assert (
                metrics["status"] == "complete"
                and metrics["score_mode"] == "corpus_nll"
            )
            assert metrics["document_count"] == 8 and metrics["scored_tokens"] > 400
            assert metrics["perplexity"] > 0
            folder = Path(".console/captures") / run["id"]
            assert not list(folder.glob("quality-logits-*"))
            summary["variants"][label] = {
                "run_id": run["id"],
                **metrics,
                "evidence_bytes": sum(
                    p.stat().st_size for p in folder.iterdir() if p.is_file()
                ),
            }
            print(
                label,
                json.dumps(summary["variants"][label], ensure_ascii=False),
                flush=True,
            )
            (OUT / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2)
            )
        comparison = call(
            "/precision/comparisons",
            {
                "baseline_run_id": summary["variants"]["fp16"]["run_id"],
                "candidate_run_ids": [
                    summary["variants"][k]["run_id"] for k in ("gptq", "gptq-kv2")
                ],
            },
        )
        assert all(
            c["quality"]["status"] == "measured_no_acceptance_limits"
            for c in comparison["candidates"]
        )
        summary["comparison"] = comparison
        (OUT / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2)
        )
    finally:
        print("RESTORING", flush=True)
        restored = apply(restore, "restore")
        summary["restored"] = {
            "id": restored["id"],
            "state": restored["state"],
            "quantization": restored["quantization"],
        }
        (OUT / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
        )
        print("RESTORED", restored["state"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-snapshot")
    args = parser.parse_args()
    main(args.original_snapshot)
