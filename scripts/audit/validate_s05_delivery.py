#!/usr/bin/env python3
"""Verify archived S05 delivery through the real read-only Console HTTP routes.

Run on Jetson using a Console-enabled Python. TestClient has no telemetry monitor,
deployment actor, model loader or GPU work. Manifest verification only reads raw
files; this script writes its own validation result, never upstream verdicts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
import httpx  # noqa: E402
from hqsb.console.app import PREFIX, create_app  # noqa: E402
from hqsb.console.config import Settings  # noqa: E402


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_manifest(raw):
    path = raw / "EVIDENCE_MANIFEST.json"
    result = {"experiment": raw.parent.name, "manifest_path": str(path), "checked_files": 0,
              "checked_bytes": 0, "failures": [], "passed": False}
    if not path.is_file():
        result["failures"].append({"reason": "manifest missing"})
        return result
    result["manifest_sha256"] = sha(path)
    manifest = json.loads(path.read_text())
    listed = manifest.get("files", [])
    if not listed:
        result["failures"].append({"reason": "empty manifest"})
    seen = set()
    for row in listed:
        relative = row.get("path", "")
        artifact = (raw / relative).resolve()
        failure = None
        if not relative or relative in seen:
            failure = "missing or duplicate manifest path"
        elif not artifact.is_relative_to(raw.resolve()):
            failure = "manifest path escaped experiment raw"
        elif not artifact.is_file():
            failure = "listed file missing"
        else:
            size, measured_hash = artifact.stat().st_size, sha(artifact)
            result["checked_files"] += 1
            result["checked_bytes"] += size
            if size != row.get("bytes") or measured_hash != row.get("sha256"):
                failure = "byte count or SHA-256 mismatch"
        seen.add(relative)
        if failure:
            result["failures"].append({"path": relative, "reason": failure})
    if sha(path) != result["manifest_sha256"]:
        result["failures"].append({"reason": "manifest changed while verifying"})
    report = manifest.get("report")
    if report:
        report_path = (raw / report["path"]).resolve()
        valid = (report_path.is_relative_to(raw.parent.resolve()) and report_path.is_file()
                 and sha(report_path) == report["sha256"])
        result["linked_report_hash_passed"] = valid
        if not valid:
            result["failures"].append({"path": report["path"], "reason": "linked report missing, escaped or SHA-256 mismatch"})
    result["passed"] = not result["failures"]
    return result


def audit_statistic(root):
    base = root / "docs/stage_experiments/S05"
    means, energy = [], []
    for index in range(3):
        value = json.loads((base / f"E05-02/raw/runs/fp16_run_{index}.json").read_text())
        workload = next(row for row in value["performance"] if row["name"] == "tiny")
        means.append(statistics.mean(row["model_core_ttft_ms"] for row in workload["samples"]))
        energy.append(workload["telemetry"]["j_per_request"])
    summaries = [json.loads(line) for line in (base / "E05-10/raw/metrics/summary.jsonl").read_text().splitlines() if line.strip()]
    target = next(row for row in summaries if row["method"] == "fp16" and row["workload"] == "tiny" and row["metric"] == "ttft_ms")
    e_target = next(row for row in summaries if row["method"] == "fp16" and row["workload"] == "tiny" and row["metric"] == "energy_j_per_request")
    mean = statistics.mean(means)
    half = 4.302652729911275 * statistics.stdev(means) / math.sqrt(3)
    expected = [mean - half, mean + half]
    registry = [json.loads(line) for line in (base / "E05-10/raw/candidates/registry.jsonl").read_text().splitlines() if line.strip()]
    release = json.loads((base / "E05-10/raw/release/manifest.json").read_text())
    checks = {"process_count_is_three": target["n_processes"] == 3,
              "ttft_mean_recomputed": math.isclose(target["mean"], mean, rel_tol=1e-12),
              "ttft_t_interval_recomputed": all(math.isclose(left, right, rel_tol=1e-12) for left, right in zip(target["ci95"], expected)),
              "energy_window_mean_recomputed": math.isclose(e_target["mean"], statistics.mean(energy), rel_tol=1e-12),
              "all_candidates_five_gates": all(set(row["gates"]) == {"correctness", "artifact", "quality", "execution", "measurement"} for row in registry),
              "fake_dequant_excluded": all(not row["eligible_low_bit_deployment"] for row in registry if row["method"] in {"rtn_w4", "rtn_w8"}),
              "release_denied": release["release_allowed"] is False and release["s06_admission"] == "DENIED"}
    return {"passed": all(checks.values()), "checks": checks, "fp16_tiny_ttft_process_means": means,
            "recomputed_mean": mean, "recomputed_ci95": expected, "unit": "independent process; not token or launch"}


def audit_new_measurements(root):
    base = root / "docs/stage_experiments/S05"
    weights = [json.loads(line) for line in (base / "E05-05/raw/stats/weight.jsonl").read_text().splitlines() if line]
    identities = [json.loads((base / f"E05-06/raw/micro/process_{run}_identity.json").read_text()) for run in range(3)]
    matrices = [[json.loads(line) for line in (base / f"E05-06/raw/micro/process_{run}.jsonl").read_text().splitlines() if line]
                for run in range(3)]
    checks = {
        "weight_records_392_unique": len(weights) == len({(r["module"], r["bits"]) for r in weights}) == 392,
        "weight_cosine_finite_and_in_range": all(math.isfinite(r["cosine"]) and -1 <= r["cosine"] <= 1 for r in weights),
        "offline_quant_separate_from_stats": all(0 <= r["offline_cpu_quant_ms"] <= r["offline_cpu_quant_and_stats_ms"] for r in weights),
        "micro_process_ids_distinct": len({r["pid"] for r in identities}) == 3,
        "micro_56_unique_cases_each": all(len(r) == len({(v["module"], v["bits"], v["M"]) for v in r}) == 56 for r in matrices),
        "micro_gates_recomputed": all(v["kernel_error"]["finite"] and v["kernel_error"]["relative_l2"] <= .002
                                     and v["kernel_error"]["max_abs"] <= .125 for group in matrices for v in group),
    }
    model = json.loads((base / "E05-06/raw/model/probe.json").read_text())
    verdict = json.loads((base / "E05-06/raw/verdict.json").read_text())
    decode = model["decode_kernel_logit_error"]
    decode_pass = decode["finite"] and decode["relative_l2"] <= .002 and decode["max_abs"] <= .125
    checks["model_decode_failure_preserved"] = verdict["model_decode_same_kernel_error_gate"] == decode_pass
    checks["model_observed_prefill_and_decode"] = {r["M"] for r in model["calls"]} == {1, 32} and model["observed_fused_kernel_count"] == 2
    return {"passed": all(checks.values()), "checks": checks, "weight_records": len(weights),
            "micro_records": sum(map(len, matrices)), "model_decode_error": decode,
            "scope": "collection integrity; this PASS never changes scientific BLOCKED/quality failure"}


def audit_model_screening(root):
    directory = root / "docs/stage_experiments/S05/E05-05/model_diagnostics"
    read = lambda name: json.loads((directory / name).read_text())
    rows = lambda name: [json.loads(line) for line in (directory / name).read_text().splitlines() if line.strip()]
    spec, summary, status = read("spec.json"), read("summary.json"), read("execution_status.json")
    samples = read("policy_samples.json")["samples"]
    by_condition = {row["condition"]: row for row in summary["conditions"]}
    checks = {
        "58_unique_conditions": len(spec["conditions"]) == len(set(spec["conditions"])) == len(by_condition) == 58,
        "completed_232_forwards": status["status"] == "COMPLETED" and status["forward_passes"] == 232,
        "policy_only_four_samples": len(samples) == 4 and all(s["split"] == "policy-validation" for s in samples),
        "final_not_read": summary["final_evaluation_read"] is False and status["final_evaluation_read"] is False,
        "no_policy_or_formal_pass": summary["deployable_policy_selected"] is False and summary["formal_experiment_overall"] == "BLOCKED",
        "restored_196_projections": read("restore_validation.json")["all_fp16_restored"] is True
            and read("restore_validation.json")["restored_projection_count"] == 196,
        "source_392_payloads_validated": read("source_artifact_validation.json")["all_passed"] is True
            and read("source_artifact_validation.json")["validated_payload_count"] == 392,
    }
    checks["token_hashes_recomputed"] = all(
        hashlib.sha256(b"".join(int(token).to_bytes(4, "little", signed=False) for token in s["token_ids"])).hexdigest() == s["token_hash"]
        and sha(directory / s["source_snapshot"]) == s["source_sha256_verified"] for s in samples)
    checks["quality_and_hidden_matrix_complete"] = True
    checks["summary_means_recomputed"] = True
    checks["intervention_sets_correct"] = True
    for condition in spec["conditions"]:
        quality, hidden = rows(f"quality/{condition}.jsonl"), rows(f"hidden/{condition}.jsonl")
        checks["quality_and_hidden_matrix_complete"] &= (
            len(quality) == 4 and {r["sample_index"] for r in quality} == set(range(4))
            and len(hidden) == len({(r["sample_index"], r["block"]) for r in hidden}) == 112
            and all(r["scored_tokens"] == len(r["positions"]) == r["input_tokens"] - 1 for r in quality))
        expected_blocks = (set() if condition == "fp16" else set(range(28)) if condition == "all_w4"
                           else {int(condition.rsplit("_", 1)[1])} if condition.startswith("single_")
                           else set(range(28)) - {int(condition.rsplit("_", 1)[1])})
        checks["intervention_sets_correct"] &= all(set(r["active_w4_blocks"]) == expected_blocks for r in quality)
        for field in ("next_token_nll", "delta_nll", "teacher_kl_mean", "mean_logit_cosine", "top1_agreement"):
            checks["summary_means_recomputed"] &= math.isclose(statistics.mean(r[field] for r in quality), by_condition[condition][field], abs_tol=1e-12)
        checks["summary_means_recomputed"] &= by_condition[condition]["scored_tokens"] == sum(r["scored_tokens"] for r in quality) == 412
    activation = rows("activation_fp16.jsonl")
    checks["784_fp16_activation_records"] = len(activation) == len({(r["sample_index"], r["module"]) for r in activation}) == 784
    checks["activation_all_finite"] = all(r["finite"] for r in activation)
    all_kl = by_condition["all_w4"]["teacher_kl_mean"]
    checks["recovery_not_confused_with_absolute_kl"] = all(
        math.isclose(r["kl_recovery"], all_kl - by_condition[r["condition"]]["teacher_kl_mean"], abs_tol=1e-12)
        for r in summary["leave_one_out_ranking"])
    return {"passed": all(checks.values()), "checks": checks, "conditions": 58, "forwards": 232,
            "scope": "four policy samples; integrity verification only, not full E05-05 acceptance"}


def audit_live(root, url, token_file, expected_nested):
    """Read-only requests to the already-running loopback API; no restarts."""
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.username or parsed.password:
        raise ValueError("Live validation accepts an HTTP loopback URL without embedded credentials only")
    token = token_file.read_text().strip()
    if not token:
        raise ValueError("Live access token is empty")
    result = {"url": url, "requests_are_read_only": True, "service_restarted": False,
              "token_recorded": False, "experiments": [], "passed_top_level": False}
    with httpx.Client(base_url=url.rstrip("/"), headers={"Authorization": "Bearer " + token}, timeout=30) as client:
        response = client.get(PREFIX + "/evidence")
        if response.status_code != 200:
            result["error"] = f"Live evidence list HTTP {response.status_code}"
            return result
        items = {row["experiment"]: row for row in response.json()["items"] if row["stage"] == "S05"}
        live_nested = set()
        for index in range(1, 11):
            experiment = f"E05-{index:02d}"
            row = items.get(experiment)
            check = {"experiment": experiment, "passed": False, "attachments": []}
            if row is not None:
                raw = root / f"docs/stage_experiments/S05/{experiment}/raw/verdict.json"
                verdict = json.loads(raw.read_text())
                expected_status = verdict.get("overall") or verdict.get("verdict") or verdict.get("status") or "UNKNOWN"
                check["raw_status"], check["api_status"] = expected_status, row["status"]
                selected = [ref for ref in row["files"] if ref["relative_path"] in expected_nested
                            or ref["name"] in {"verdict.json", f"{experiment}_实验报告.md"}]
                for ref in selected:
                    expected = (root / ref["relative_path"]).read_bytes()
                    expected_hash = hashlib.sha256(expected).hexdigest()
                    detail = client.get(PREFIX + f"/evidence/{ref['id']}")
                    download = client.get(PREFIX + f"/evidence/{ref['id']}/download")
                    passed = (detail.status_code == download.status_code == 200
                              and detail.json().get("sha256") == expected_hash
                              and download.content == expected
                              and download.headers.get("X-Content-SHA256") == expected_hash)
                    check["attachments"].append({"path": ref["relative_path"], "sha256": expected_hash, "passed": passed})
                top_level = [ref for ref in check["attachments"] if ref["path"].endswith((f"/{experiment}_实验报告.md", "/raw/verdict.json"))]
                check["passed"] = expected_status == row["status"] and len(top_level) == 2 and all(ref["passed"] for ref in top_level)
                live_nested.update(ref["path"] for ref in check["attachments"] if ref["passed"] and ref["path"] in expected_nested)
            result["experiments"].append(check)
        bundle_path = root / "docs/stage_experiments/S05/delivery_20260921/browser_manifest.json"
        if bundle_path.is_file():
            bundles = json.loads(bundle_path.read_text())["bundles"]
            refs = {ref["relative_path"]: ref for item in items.values() for ref in item["files"]}
            covered, bundle_results = set(), []
            for bundle in bundles:
                ref = refs.get(bundle["path"])
                check = {"path": bundle["path"], "passed": False, "sources_verified": 0}
                if ref:
                    detail = client.get(PREFIX + f"/evidence/{ref['id']}")
                    download = client.get(PREFIX + f"/evidence/{ref['id']}/download")
                    data = download.content
                    valid = (detail.status_code == download.status_code == 200
                             and len(data) == bundle["bytes"] and hashlib.sha256(data).hexdigest() == bundle["sha256"]
                             and detail.json().get("sha256") == bundle["sha256"]
                             and detail.json().get("content", "").encode("utf-8") == data
                             and download.headers.get("X-Content-SHA256") == bundle["sha256"])
                    for source in bundle["sources"]:
                        original = root / source["source"]
                        embedded = data[source["start_byte"]:source["end_byte"]]
                        source_valid = (len(embedded) == source["bytes"]
                                        and hashlib.sha256(embedded).hexdigest() == source["sha256"]
                                        and original.read_bytes() == embedded)
                        valid &= source_valid
                        if source_valid:
                            check["sources_verified"] += 1
                    check["passed"] = valid
                    if valid:
                        covered.update(s["source"] for s in bundle["sources"])
                bundle_results.append(check)
            result["packaged_text_delivery"] = {
                "passed": bool(bundles) and all(b["passed"] for b in bundle_results) and set(expected_nested) <= covered,
                "manifest_sha256": sha(bundle_path), "bundles": bundle_results,
                "covered_source_files": len(covered), "missing_sources": sorted(set(expected_nested) - covered),
                "download_format": "Markdown container with original UTF-8 text, source paths and hashes; not individual original files"}
    result["passed_top_level"] = all(row["passed"] for row in result["experiments"])
    missing = sorted(set(expected_nested) - live_nested)
    result["nested_text_delivery"] = {"all_available": not missing, "missing_count": len(missing), "missing_paths": missing,
                                      "boundary": "Live service may still use the prior top-level index. New-code TestClient validation is separate; no restart was performed."}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=Path("/home/jetson/work/HeteroQuantServeBench"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--unit-test-log", type=Path, help="Optional saved output from the real remote pytest invocation")
    parser.add_argument("--live-url", help="Optional running loopback Console base URL; read-only validation")
    parser.add_argument("--token-file", type=Path, help="Existing access-token file, read in memory and never saved or printed")
    args = parser.parse_args()
    if bool(args.live_url) != bool(args.token_file):
        parser.error("--live-url and --token-file must be supplied together")
    root = args.evidence_root.resolve()
    output = args.output or root / "docs/stage_experiments/S05/E05-10/raw/frontend_api_validation.json"
    result = {"schema": "hqsb.s05.delivery-validation/v1", "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "evidence_root": str(root), "api_code_root": str(ROOT), "monitor_enabled": False,
              "deployment_actors_created": 0, "gpu_or_model_work": False,
              "scope": "Actual archived data through TestClient; not a live browser screenshot or hardware experiment",
              "api_code_sha256": {name: sha(ROOT / name) for name in ("hqsb/console/evidence.py", "hqsb/console/app.py")},
              "experiments": [], "manifest_integrity": [], "errors": []}
    settings = Settings(data_dir=Path("/tmp/hqsb-s05-delivery-state"), evidence_root=root,
                        web_dist=Path("/tmp/hqsb-no-web-dist"), deployments=[])
    app = create_app(settings, "ephemeral-s05-readonly-audit", service=SimpleNamespace(close=lambda: None), monitor=False)
    headers = {"Authorization": "Bearer ephemeral-s05-readonly-audit"}
    with TestClient(app) as client:
        result["authentication_required"] = client.get(PREFIX + "/evidence").status_code == 401
        response = client.get(PREFIX + "/evidence", headers=headers)
        response.raise_for_status()
        items = {row["experiment"]: row for row in response.json()["items"] if row["stage"] == "S05"}
        result["indexed_s05_experiments"] = sorted(items)
        for index in range(1, 11):
            experiment = f"E05-{index:02d}"
            raw = root / "docs/stage_experiments/S05" / experiment / "raw"
            checked = {"experiment": experiment, "passed": False, "attachments": [], "failures": []}
            row = items.get(experiment)
            if row is None:
                checked["failures"].append("experiment absent from evidence API")
                result["experiments"].append(checked)
                continue
            verdict = json.loads((raw / "verdict.json").read_text())
            status = verdict.get("overall") or verdict.get("verdict") or verdict.get("status") or "UNKNOWN"
            checked["raw_status"], checked["api_status"] = status, row["status"]
            if status != row["status"]:
                checked["failures"].append("raw/API verdict mismatch")
            selected = [ref for ref in row["files"] if ref["name"] == "verdict.json" or ref["name"] == f"{experiment}_实验报告.md"
                        or (index >= 5 and ref["relative_path"].endswith(".jsonl") and "/raw/" in ref["relative_path"])
                        or "/post_fix/" in ref["relative_path"]
                        or ("/model_diagnostics/" in ref["relative_path"] and ref["relative_path"].endswith((".json", ".jsonl", ".md")))]
            if not any(ref["name"] == f"{experiment}_实验报告.md" for ref in selected):
                checked["failures"].append("experiment report not indexed")
            if index >= 5 and not any(ref["relative_path"].endswith(".jsonl") for ref in selected):
                checked["failures"].append("no actual JSONL result indexed")
            if len({ref["id"] for ref in row["files"]}) != len(row["files"]):
                checked["failures"].append("duplicate file IDs")
            for ref in selected:
                path = root / ref["relative_path"]
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                detail = client.get(PREFIX + f"/evidence/{ref['id']}", headers=headers)
                download = client.get(PREFIX + f"/evidence/{ref['id']}/download", headers=headers)
                okay = detail.status_code == download.status_code == 200
                if okay:
                    body = detail.json()
                    expected = json.loads(data) if path.suffix == ".json" else data.decode("utf-8")
                    okay = (body["sha256"] == digest and body["content"] == expected
                            and body["bytes"] == len(data) and download.content == data
                            and download.headers.get("X-Content-SHA256") == digest)
                checked["attachments"].append({"id": ref["id"], "path": ref["relative_path"], "sha256": digest,
                                                "bytes": len(data), "preview_status": detail.status_code,
                                                "download_status": download.status_code, "passed": okay})
                if not okay:
                    checked["failures"].append(f"preview/download mismatch: {ref['relative_path']}")
            checked["passed"] = not checked["failures"]
            result["experiments"].append(checked)
    for index in range(1, 11):
        try:
            result["manifest_integrity"].append(audit_manifest(root / f"docs/stage_experiments/S05/E05-{index:02d}/raw"))
        except (OSError, ValueError, KeyError) as exc:
            result["manifest_integrity"].append({"experiment": f"E05-{index:02d}", "passed": False, "error": str(exc)})
    for index in (7, 8):
        result["manifest_integrity"].append(audit_manifest(root / f"docs/stage_experiments/S05/E05-{index:02d}/post_fix"))
    result["manifest_integrity"].append(audit_manifest(root / "docs/stage_experiments/S05/E05-05/model_diagnostics"))
    try:
        result["independent_statistical_spot_check"] = audit_statistic(root)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        result["independent_statistical_spot_check"] = {"passed": False, "error": str(exc)}
    try:
        result["new_measurement_audit"] = audit_new_measurements(root)
    except (OSError, ValueError, KeyError) as exc:
        result["new_measurement_audit"] = {"passed": False, "error": str(exc)}
    try:
        result["model_screening_audit"] = audit_model_screening(root)
    except (OSError, ValueError, KeyError) as exc:
        result["model_screening_audit"] = {"passed": False, "error": str(exc)}
    if args.unit_test_log:
        result["unit_test_log"] = {"path": str(args.unit_test_log), "sha256": sha(args.unit_test_log),
                                   "content": args.unit_test_log.read_text(), "scope": "Prior remote unit tests, separate from actual-data delivery validation"}
    if args.live_url:
        try:
            expected_nested = {row["path"] for item in result["experiments"] for row in item["attachments"]
                               if not row["path"].endswith((f"/{item['experiment']}_实验报告.md", "/raw/verdict.json"))}
            result["live_service"] = audit_live(root, args.live_url, args.token_file, expected_nested)
        except Exception as exc:
            # Do not persist HTTP request objects, headers, or token-related IO
            # error text. Exception type is enough to distinguish failure class.
            result["live_service"] = {"passed_top_level": False, "error_type": type(exc).__name__, "token_recorded": False}
    result["archive_and_updated_api_passed"] = (result["authentication_required"] and len(result["experiments"]) == 10
                        and all(row["passed"] for row in result["experiments"] + result["manifest_integrity"])
                        and result["independent_statistical_spot_check"]["passed"]
                        and result["new_measurement_audit"]["passed"]
                        and result["model_screening_audit"]["passed"])
    result["passed"] = (result["archive_and_updated_api_passed"] and (not args.live_url or
                        (result["live_service"]["passed_top_level"] and
                         (result["live_service"].get("nested_text_delivery", {}).get("all_available", False)
                          or result["live_service"].get("packaged_text_delivery", {}).get("passed", False)))))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "archive_and_updated_api_passed": result["archive_and_updated_api_passed"],
                      "live_top_level_passed": result.get("live_service", {}).get("passed_top_level"),
                      "live_missing_attachments": result.get("live_service", {}).get("nested_text_delivery", {}).get("missing_count"),
                      "experiments": len(result["experiments"]),
                      "attachments": sum(len(row["attachments"]) for row in result["experiments"]),
                      "manifest_files": sum(row.get("checked_files", 0) for row in result["manifest_integrity"]),
                      "output": str(output)}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
