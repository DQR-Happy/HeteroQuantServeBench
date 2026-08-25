#!/usr/bin/env python3
"""Real Jetson API acceptance for plans, receipts, quality and comparisons."""

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from hqsb.console.app import PREFIX, create_app
from hqsb.console.config import load_settings
from hqsb.console.store import TERMINAL
from hqsb.observation.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--legacy-quantize", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    settings.data_dir = Path(args.data_dir).resolve()
    if settings.data_dir.exists():
        parser.error("A fresh directory is required; retain old evidence")
    settings.data_dir.mkdir(parents=True)
    settings.privileged_worker = True
    settings.deployments[0].manifest = str(ROOT / "docs/benchmark/model_sha256_manifest.txt")
    token = secrets.token_urlsafe(32)
    token_file = settings.data_dir / "access-token"
    token_file.write_text(token)
    token_file.chmod(0o600)
    report = {"status": "FAIL", "checks": [], "runs": [], "scope": "real_Qwen3_FP16_G01_G08_fixture"}

    def check(name, value, details=None):
        report["checks"].append({"name": name, "passed": bool(value), "details": details})
        print(json.dumps({"check": name, "passed": bool(value)}), flush=True)
        if not value:
            raise AssertionError(name)

    try:
        with TestClient(create_app(settings, token, monitor=False), headers={"Authorization": "Bearer " + token}) as client:
            def get(path):
                response = client.get(PREFIX + path)
                response.raise_for_status()
                return response.json()

            def post(path, body=None):
                response = client.post(PREFIX + path, json=body or {}, headers={"Idempotency-Key": uuid.uuid4().hex})
                response.raise_for_status()
                return response.json()

            def wait(run):
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    current = get("/runs/" + run["id"])
                    if current["state"] in TERMINAL:
                        report["runs"].append(current)
                        return current
                    time.sleep(0.1)
                post("/requests/" + run["id"] + "/cancel")
                raise TimeoutError(run["id"])

            deployment = settings.deployments[0].id
            loaded = wait(post(f"/deployments/{deployment}/load"))
            check("manifest_verified_model_load", loaded["state"] == "completed", loaded.get("error"))
            epoch = get("/deployments")["items"][0]["epoch"]
            catalogue = get("/optimization-catalog")
            check("ten_layers_declared", len(catalogue["layers"]) == 10)
            normal = {"deployment_id": deployment, "expected_epoch": epoch,
                      "messages": [{"role": "user", "content": "Explain GPU cache."}],
                      "max_output_tokens": 4, "observation_mode": "off", "deadline_ms": 120000}
            before = wait(post("/requests", normal))
            check("legacy_generation_preserved", before["state"] == "completed", before.get("error"))
            rejected = client.post(PREFIX + "/requests", json={**normal, "optimization": {"changes": {"paged_kv": True}}},
                                   headers={"Idempotency-Key": uuid.uuid4().hex})
            check("unsupported_measure_rejected", rejected.status_code == 422)
            rejected = client.post(PREFIX + "/requests", json={**normal, "optimization": {"changes": {"attention_backend": "sdpa"}}},
                                   headers={"Idempotency-Key": uuid.uuid4().hex})
            check("deployment_reload_required", rejected.status_code == 422)
            spec = {"baseline": {"changes": {}}, "candidate": {"changes": {"cleanup_policy": "preserve_allocator"}},
                    "declared_changes": ["cleanup_policy"], "performance_pairs": 1, "input_tokens": 32, "output_tokens": 4,
                    "quality_plan": {"samples": [{"id": "arithmetic-fixture", "text": "1 + 1 =", "expected": "2", "slice": "fixture"}],
                                     "max_tokens": 16, "generation_tokens": 4},
                    "observation_plan": {"mode": "basic", "sources": ["allocator", "pages", "operators"],
                                         "max_pages": 16, "max_decode_steps": 1}}
            plan = post("/experiments/plans", {"spec": spec})
            experiment = plan["experiment_id"]
            check("pre_registered_spec_identity", len(plan["spec_sha256"]) == 64)
            for role in ("performance", "diagnostic", "quality"):
                for arm in ("baseline", "candidate"):
                    run = wait(post(f"/experiments/{experiment}/jobs", {"deployment_id": deployment, "expected_epoch": epoch,
                                                                      "role": role, "arm": arm, "pair": 0}))
                    check(f"{role}_{arm}_completed", run["state"] == "completed", run.get("error"))
                    bundle = get(f"/runs/{run['id']}/experiment")
                    check(f"{role}_{arm}_applied", bundle["receipt"]["status"] == "applied")
                    check(f"{role}_{arm}_sealed", bundle["manifest"]["status"] == "complete")
                    files = bundle["manifest"]["artifacts"]
                    for index in range(len(files)):
                        response = client.get(PREFIX + f"/runs/{run['id']}/experiment/artifacts/{index}")
                        response.raise_for_status()
                        check(f"{role}_{arm}_artifact_{index}_hash", hashlib.sha256(response.content).hexdigest() == files[index]["sha256"])
                    if role == "diagnostic":
                        observation = get(f"/runs/{run['id']}/observations")
                        check(f"{arm}_composite_capture_complete", observation["capture_status"] == "complete", observation.get("source_problems"))
                        check(f"{arm}_allocator_observed", bool(observation["allocator"]))
                        check(f"{arm}_pages_bounded", bool(observation["page_samples"]) and all(row["sampled_pages"] <= 16 for row in observation["page_samples"]))
                        check(f"{arm}_linked_context", observation["context"]["experiment_links"]["campaign_id"] == experiment)
                    if role != "quality":
                        check(f"{role}_{arm}_fixed_workload_applied", run["metrics"]["input_tokens"] == 32 and run["metrics"]["output_tokens"] == 4)
                    if role == "quality":
                        check(f"{arm}_quality_measured", bundle["quality"]["scored_tokens"] > 0 and bundle["quality"]["perplexity"] > 0)
                        check(f"{arm}_quality_excluded_from_performance", run["metrics"].get("runtime_e2e_ms") is None and run["metrics"]["measurement_role"] == "quality")
                        check(f"{arm}_input_not_in_run_config", "text" not in run["config"]["quality_plan"]["samples"][0])
            comparison = get(f"/experiments/{experiment}/report")
            check("complete_report", comparison["completeness"] == "complete", comparison.get("failed_pairs"))
            check("comparison_controlled", comparison["comparable"], comparison["reasons"])
            check("quality_contract_passed", comparison["quality_status"] == "passed", comparison["quality"])
            check("mechanism_differences_present", bool(comparison["mechanisms"][0]["differences"]))
            check("shared_worker_is_not_independent_performance", comparison["performance"]["status"] == "insufficient_evidence")
            check("no_unearned_adoption", comparison["decision"] == "hold")
            report["comparison"] = comparison
            # Corruption must fail through the actual download endpoint.
            quality_run = next(run for run in reversed(report["runs"]) if run["config"].get("experiment_role") == "quality")
            bundle = get(f"/runs/{quality_run['id']}/experiment")
            index = next(i for i, item in enumerate(bundle["manifest"]["artifacts"]) if item["path"].endswith(".f32"))
            path = settings.data_dir / "captures" / quality_run["id"] / bundle["manifest"]["artifacts"][index]["path"]
            saved = path.read_bytes()
            try:
                path.write_bytes(b"corruption-check")
                check("corrupt_artifact_rejected", client.get(PREFIX + f"/runs/{quality_run['id']}/experiment/artifacts/{index}").status_code == 409)
            finally:
                path.write_bytes(saved)
            cancellation_spec = {**spec, "quality_plan": {
                "samples": [{"text": "The GPU stores data. " * 8}], "max_tokens": 64, "generation_tokens": 32}}
            cancellation_plan = post("/experiments/plans", {"spec": cancellation_spec})
            cancelled = post(f"/experiments/{cancellation_plan['experiment_id']}/jobs", {
                "deployment_id": deployment, "expected_epoch": epoch, "role": "quality", "arm": "baseline", "pair": 0})
            deadline = time.monotonic() + 30
            while get("/runs/" + cancelled["id"])["state"] == "queued" and time.monotonic() < deadline:
                time.sleep(0.05)
            time.sleep(0.5)
            post("/requests/" + cancelled["id"] + "/cancel")
            cancelled = wait(cancelled)
            check("quality_cancel_acknowledged", cancelled["state"] == "cancelled", cancelled.get("error"))
            cancelled_bundle = get(f"/runs/{cancelled['id']}/experiment")
            check("cancelled_quality_is_partial", cancelled_bundle["manifest"]["status"] == "partial")
            after = wait(post("/requests", normal))
            check("generation_preserved_after_quality", after["metrics"]["generated_token_ids"] == before["metrics"]["generated_token_ids"])
            if args.legacy_quantize:
                quantized = wait(post("/quantization/jobs", {"deployment_id": deployment, "expected_epoch": epoch,
                                                           "bits": 4, "group_size": 128}))
                check("legacy_w4_artifact_completed", quantized["state"] == "completed", quantized.get("error"))
                summary = quantized["metrics"]["quantization"]
                check("legacy_w4_is_storage_only", summary["execution_path"] == "storage_only" and not summary["resident_model_modified"] and not summary["native_deployment_available"])
                downloaded = client.get(PREFIX + f"/quantization/artifacts/{quantized['id']}/manifest")
                downloaded.raise_for_status()
                check("legacy_quantization_manifest_hash", hashlib.sha256(downloaded.content).hexdigest() == downloaded.headers["x-content-sha256"])
                w8 = post("/quantization/jobs", {"deployment_id": deployment, "expected_epoch": epoch, "bits": 8})
                deadline = time.monotonic() + 30
                while get("/runs/" + w8["id"])["state"] == "queued" and time.monotonic() < deadline:
                    time.sleep(0.05)
                time.sleep(0.2)
                post("/requests/" + w8["id"] + "/cancel")
                w8 = wait(w8)
                check("legacy_w8_cancelled", w8["state"] == "cancelled", w8.get("error"))
                check("legacy_w8_partial_removed", not (settings.data_dir / "artifacts" / (w8["id"] + ".partial")).exists())
                restored = wait(post("/requests", normal))
                check("generation_preserved_after_quantization", restored["metrics"]["generated_token_ids"] == before["metrics"]["generated_token_ids"])
            unloaded = wait(post(f"/deployments/{deployment}/unload"))
            check("worker_unloaded", unloaded["state"] == "completed")
            report["status"] = "PASS"
    finally:
        atomic_json(Path(args.output), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
