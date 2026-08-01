"""Real HTTP/device acceptance. Run only through scripts/remote_run.sh.

No credentials are written to evidence. Performance deltas are measurements,
not pass/fail thresholds; unavailable device profiling is explicitly reported.
"""

import argparse
import hashlib
import io
import json
import statistics
import time
import uuid
import zipfile
from pathlib import Path

import httpx

TERMINAL = {"completed", "failed", "cancelled", "timed_out", "interrupted"}
PROMPT = "请简要解释 GPU 内存带宽为什么影响大模型逐 token 解码。"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--data-dir", type=Path, default=Path(".console"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/console/observability.json")
    )
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--overhead-repeats", type=int, default=3)
    args = parser.parse_args()
    if not 3 <= args.overhead_repeats <= 20:
        parser.error("--overhead-repeats must be between 3 and 20")
    evidence = {
        "started_at": time.time(),
        "mode": "real_device_http",
        "verdict": "FAIL",
        "checks": [],
        "runs": [],
        "limitations": [],
        "test_prompt": PROMPT,
        "quantization_requested": args.quantize,
    }
    client = None

    def check(name, condition, details=None):
        evidence["checks"].append(
            {"name": name, "passed": bool(condition), "details": details}
        )
        print(json.dumps({"check": name, "passed": bool(condition)}), flush=True)
        if not condition:
            raise AssertionError(name)

    def response(path):
        result = client.get(path)
        result.raise_for_status()
        return result

    def get(path):
        return response(path).json()

    def post(path, payload=None):
        result = client.post(
            path, json=payload or {}, headers={"Idempotency-Key": uuid.uuid4().hex}
        )
        result.raise_for_status()
        return result.json()

    def wait(run, timeout=900):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            current = get("/runs/" + run["id"])
            if current["state"] in TERMINAL:
                evidence["runs"].append(current)
                return current
            time.sleep(0.4)
        post("/requests/" + run["id"] + "/cancel")
        raise TimeoutError("Timed out waiting for " + run["id"])

    def infer(body, mode):
        run = wait(post("/requests", {**body, "observation_mode": mode}))
        check(
            mode + "_request_completed", run["state"] == "completed", run.get("error")
        )
        return run

    try:
        token = (args.data_dir / "access-token").read_text().strip()
        client = httpx.Client(
            base_url=args.url.rstrip("/") + "/api/console/v1",
            headers={"Authorization": "Bearer " + token},
            timeout=60,
        )
        session = get("/session")
        check("api_v0_2", session.get("version", "").startswith("0.2."))
        resources, research = get("/resources"), get("/research")
        evidence["resources_before"] = resources
        evidence["research_status"] = {
            key: value.get("status")
            for key, value in research.items()
            if isinstance(value, dict)
        }
        check(
            "resource_accounting",
            resources["host"].get("available_bytes", 0) > 0
            and bool(resources["limitations"]),
        )
        check("research_schema", bool(research.get("schema_version")))
        evidence["capabilities"] = get("/capabilities")
        deployment = next(
            row for row in get("/deployments")["items"] if row["provider"] == "pytorch"
        )
        if deployment["state"] != "ready":
            check(
                "deployment_can_load",
                deployment["state"] in {"unloaded", "failed"},
                deployment["state"],
            )
            loaded = wait(post("/deployments/" + deployment["id"] + "/load"))
            check("model_loaded", loaded["state"] == "completed", loaded.get("error"))
        deployment = next(
            row for row in get("/deployments")["items"] if row["id"] == deployment["id"]
        )
        evidence["deployment"] = deployment
        body = {
            "deployment_id": deployment["id"],
            "expected_epoch": deployment["epoch"],
            "messages": [{"role": "user", "content": PROMPT}],
            "max_output_tokens": 12,
            "deadline_ms": min(600000, session["limits"]["max_deadline_ms"]),
            "save_input": True,
        }
        samples, baseline_tokens = {"off": [], "basic": []}, None
        for pair in range(args.overhead_repeats):
            for mode in ["off", "basic"] if pair % 2 == 0 else ["basic", "off"]:
                run = infer(body, mode)
                metrics = run["metrics"]
                tokens = metrics["generated_token_ids"]
                if baseline_tokens is None:
                    baseline_tokens = tokens
                check(mode + "_token_identity", tokens == baseline_tokens)
                observation = metrics["observation"]
                check(
                    mode + "_observation_contract",
                    observation["mode"] == mode
                    and (
                        bool(observation["tokens"])
                        if mode == "basic"
                        else not observation["tokens"]
                    ),
                )
                samples[mode].append(
                    {
                        "run_id": run["id"],
                        "pair": pair,
                        **{
                            key: metrics[key]
                            for key in (
                                "runtime_e2e_ms",
                                "runtime_first_token_ms",
                                "console_e2e_ms",
                                "console_first_content_ms",
                            )
                        },
                    }
                )
        medians = {
            mode: {
                key: statistics.median(row[key] for row in values)
                for key in values[0]
                if key not in {"pair", "run_id"}
            }
            for mode, values in samples.items()
        }
        evidence["observation_overhead"] = {
            "samples": samples,
            "medians": medians,
            "basic_vs_off_percent": {
                key: (value / medians["off"][key] - 1) * 100
                for key, value in medians["basic"].items()
                if medians["off"][key] > 0
            },
            "acceptance_threshold": None,
            "interpretation": "Small interleaved sample; includes runtime/thermal noise. Negative deltas do not prove speedup; no zero-overhead claim.",
        }
        profiled = infer(body, "operators")
        obs = profiled["metrics"]["observation"]
        profile = obs["profile"]
        check(
            "profile_token_identity",
            profiled["metrics"]["generated_token_ids"] == baseline_tokens,
        )
        check(
            "phases_and_token_count",
            bool(obs["phases"])
            and len(obs["tokens"]) == profiled["metrics"]["output_tokens"],
        )
        check(
            "bounded_operator_window",
            profile["coverage"]["prefill"] and profile["coverage"]["decode_steps"] <= 8,
        )
        check(
            "operator_cpu_evidence",
            "CPU" in profile["activities"] and bool(profile["operators"]),
        )
        if "CUDA" in profile["activities"]:
            check(
                "operator_cuda_evidence",
                any((row.get("self_cuda_ms") or 0) > 0 for row in profile["operators"]),
            )
            evidence["cuda_profile"] = "measured"
        else:
            check(
                "cuda_unavailable_honestly",
                profile["status"] == "partial"
                and bool(profile["limitations"])
                and all(
                    row.get("self_cuda_ms") is None for row in profile["operators"]
                ),
            )
            evidence["cuda_profile"] = "unavailable"
            evidence["limitations"].append(
                "CUDA profiling unavailable; CPU capture was validated, not GPU coverage."
            )
        check(
            "module_window_declared",
            profile["coverage"].get("module_scope") == "first_decoder_layer_only",
        )
        check(
            "worker_memory_snapshots",
            {row["label"] for row in obs["memory_snapshots"]}
            >= {"before_request", "after_prefill", "after_cleanup"},
        )
        prefix = "/runs/" + profiled["id"]
        analysis = get(prefix + "/analysis")
        evidence["profile_analysis"] = analysis
        check("trace_export_available", profile["trace_available"])
        trace = response(prefix + "/trace")
        check(
            "trace_download_hash",
            hashlib.sha256(trace.content).hexdigest()
            == trace.headers.get("x-content-sha256"),
        )
        parsed = trace.json()
        check(
            "trace_json_and_module_ranges",
            bool(parsed.get("traceEvents"))
            and any(
                str(row.get("name", "")).startswith("hqsb.module:")
                for row in parsed["traceEvents"]
            ),
        )
        del trace, parsed
        page = get(prefix + "/trace/events?limit=5")
        if page["summary"]["status"] == "download_only":
            evidence["limitations"].append(
                "Trace exceeds API parse budget; raw download verified, pagination unavailable."
            )
        else:
            check(
                "trace_pagination",
                0 < len(page["items"]) <= 5 and page["total"] >= len(page["items"]),
            )
            if page["next_offset"] is not None:
                next_page = get(
                    prefix + "/trace/events?limit=5&offset=" + str(page["next_offset"])
                )
                check(
                    "trace_pages_distinct",
                    not (
                        {row["id"] for row in page["items"]}
                        & {row["id"] for row in next_page["items"]}
                    ),
                )
        report = get(prefix + "/optimization")
        check(
            "optimization_is_unverified_hypothesis",
            report["status"] == "hypotheses_only" and bool(report["findings"]),
        )
        check(
            "markdown_export",
            "待验证" in response(prefix + "/optimization?format=markdown").text,
        )
        bundle = response(prefix + "/bundle")
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            check(
                "bundle_identity",
                manifest["run_id"] == profiled["id"]
                and manifest["raw_trace_included"] is False,
            )
            check(
                "bundle_member_integrity",
                all(
                    len(archive.read(item["path"])) == item["bytes"]
                    and hashlib.sha256(archive.read(item["path"])).hexdigest()
                    == item["sha256"]
                    for item in manifest["files"]
                ),
            )
        if args.quantize:
            payload = {
                "deployment_id": deployment["id"],
                "expected_epoch": deployment["epoch"],
                "bits": 4,
                "group_size": 128,
            }
            job = wait(post("/quantization/jobs", payload), timeout=1200)
            check("rtn_w4_job_completed", job["state"] == "completed", job.get("error"))
            quant = job["metrics"]["quantization"]
            check(
                "quantization_progress",
                job.get("progress", {}).get("total_tensors", 0) > 0,
            )
            check(
                "storage_only_truth",
                quant["execution_path"] == "storage_only"
                and quant["quality"] == "not_evaluated"
                and quant["native_deployment_available"] is False
                and quant["resident_model_modified"] is False,
            )
            raw_manifest = response(
                "/quantization/artifacts/" + job["id"] + "/manifest"
            )
            manifest = raw_manifest.json()
            check(
                "quant_manifest_hash",
                hashlib.sha256(raw_manifest.content).hexdigest()
                == raw_manifest.headers.get("x-content-sha256"),
            )
            identity_keys = (
                "schema",
                "version",
                "source",
                "quantization",
                "module_policy",
                "packing_version",
                "coverage",
                "tensors",
                "runtime_contract",
            )
            canonical = json.dumps(
                {key: manifest[key] for key in identity_keys},
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            check(
                "canonical_artifact_identity",
                manifest["artifact_id"]
                == quant["artifact_id"]
                == "sha256:" + hashlib.sha256(canonical).hexdigest(),
            )
            check(
                "quant_artifact_bytes",
                quant["bytes"]["qvalues"] > 0
                and quant["bytes"]["total"] > quant["bytes"]["qvalues"],
            )
            source_hash = manifest["source"]["model_sha256"]
            expected_scope = (
                "verified_manifest_at_load"
                if deployment["detail"].get("weight_verification")
                == "manifest_verified"
                else "metadata_only_unverified_weights"
            )
            check(
                "source_identity_not_overclaimed",
                quant["source_identity_scope"] == expected_scope
                and (
                    not source_hash.startswith("unverified-metadata:")
                    if expected_scope == "verified_manifest_at_load"
                    else source_hash.startswith("unverified-metadata:")
                ),
            )
            after = infer(body, "basic")
            check(
                "fp16_tokens_unchanged_after_quantization",
                after["metrics"]["generated_token_ids"] == baseline_tokens,
            )
            cancel_job = post(
                "/quantization/jobs", {**payload, "bits": 8, "group_size": None}
            )
            started_by = time.monotonic() + 30
            while time.monotonic() < started_by:
                current = get("/runs/" + cancel_job["id"])
                if (
                    current.get("progress", {}).get("rows_processed")
                    or current["state"] in TERMINAL
                ):
                    break
                time.sleep(0.1)
            post("/requests/" + cancel_job["id"] + "/cancel")
            cancelled = wait(cancel_job)
            check(
                "quant_cancel_cleanup",
                cancelled["state"] == "cancelled"
                and cancelled["cleanup"] == "succeeded",
                cancelled.get("error"),
            )
            check(
                "cancelled_artifact_inaccessible",
                client.get(
                    "/quantization/artifacts/" + cancel_job["id"] + "/manifest"
                ).status_code
                == 404,
            )
            check(
                "cancelled_partial_removed",
                not (
                    args.data_dir / "artifacts" / (cancel_job["id"] + ".partial")
                ).exists(),
            )
            followup = infer(body, "basic")
            check(
                "reuse_after_quant_cancel",
                followup["metrics"]["generated_token_ids"] == baseline_tokens,
            )
        evidence["resources_after"] = get("/resources")
        evidence["verdict"] = (
            "PASS" if not evidence["limitations"] else "PASS_WITH_LIMITATIONS"
        )
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        evidence["finished_at"] = time.time()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if client is not None:
            client.close()
        print(
            json.dumps(
                {
                    "verdict": evidence["verdict"],
                    "checks": len(evidence["checks"]),
                    "output": str(args.output),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
