"""Compose sealed execution receipts, quality evidence and mechanism facts."""

import array
from dataclasses import asdict
from pathlib import Path
import sys
import statistics

from hqsb.core.optimization import fingerprint, resolve_plan
from hqsb.evaluation.optimization_comparison import performance_gate
from hqsb.experiments.artifacts import load_bundle, read_json
from hqsb.observation.analysis import analyze
from hqsb.observation.storage import sha256
from hqsb.quant.quality import position_metrics


def _kernel_metrics(document):
    """Project explicit worker facts; host dispatch never becomes GPU timing."""
    summary = document.get("kernel_receipt")
    if not summary or summary.get("status") not in {"invoked", "reference"}:
        return []
    values = [("kernel.logical_site_calls", phase, sum(
        site["calls"].get(phase, 0) for site in summary["sites"])) for phase in ("prefill", "decode")]
    values += [("kernel." + name, "request", summary[name]) for name in (
        "gqa_repeat_peak_logical_bytes", "workspace_bytes", "weight_packing_bytes")]
    return [{"metric_id":name,"label":label,"value":value,"status":"measured",
             "scope":"request_worker_kernel_evidence","unit":"count" if name.endswith("calls") else "bytes",
             "evidence_scope":"worker_dispatch_or_tensor_shape_not_hardware_counter"}
            for name,label,value in values]


def _quality(left_dir, right_dir, left, right, limits, *, allow_generation_budget_change=False):
    if not left or not right or left["status"] != "complete" or right["status"] != "complete":
        return {"status": "not_evaluated", "reason": "QUALITY_INCOMPLETE"}
    protocols = [{key: value for key, value in item["protocol"].items()
                  if not (allow_generation_budget_change and key == "generation_tokens")} for item in (left, right)]
    if left["dataset_sha256"] != right["dataset_sha256"] or protocols[0] != protocols[1]:
        return {"status": "not_comparable", "reason": "QUALITY_DATA_OR_PROTOCOL_MISMATCH"}
    a, b = left["positions"], right["positions"]
    if not a or len(a) != len(b):
        return {"status": "not_comparable", "reason": "QUALITY_POSITION_MISMATCH"}
    rows = []
    for x, y in zip(a, b):
        if any(x[key] != y[key] for key in ("sample_id", "position", "vocab_size", "slice")):
            return {"status": "not_comparable", "reason": "QUALITY_POSITION_MISMATCH"}
        values = []
        for directory, item in ((left_dir, x), (right_dir, y)):
            if item["path"] != Path(item["path"]).name or item["bytes"] != item["vocab_size"] * 4:
                raise ValueError("QUALITY_ROW_INVALID")
            row = array.array("f")
            row.frombytes((directory / item["path"]).read_bytes())
            if sys.byteorder != "little":
                row.byteswap()
            values.append(row.tolist())
        metric = asdict(position_metrics(*values, position=x["position"], sample_id=x["sample_id"],
                                         top_k=left["protocol"]["top_k"], slice_name=x["slice"]))
        metric["nll_increase"] = y["nll"] - x["nll"]
        metric["source_logits"] = {"baseline": {"run_id": left["run_id"], "artifact": x["path"], "sha256": x["sha256"]},
                                   "candidate": {"run_id": right["run_id"], "artifact": y["path"], "sha256": y["sha256"]}}
        metric["passed"] = (metric["logit_max_abs_error"] <= limits["max_logit_abs_error"]
                             and metric["kl_fp16_quant"] <= limits["max_kl"]
                             and metric["nll_increase"] <= limits["max_nll_increase"])
        rows.append(metric)
    accuracy_drop = left["task_accuracy"] - right["task_accuracy"] if left["task_accuracy"] is not None and right["task_accuracy"] is not None else None
    passed = all(row["passed"] for row in rows) and (accuracy_drop is None or accuracy_drop <= limits["max_accuracy_drop"])
    divergence = {row["sample_id"]: -1 for row in rows}
    for row in rows:
        if row["top1_flipped"] and divergence[row["sample_id"]] == -1:
            divergence[row["sample_id"]] = row["position"]
    return {"status": "passed" if passed else "failed", "scope": "declared_dataset_and_quality_limits_only",
            "first_divergence_by_sample": divergence,
            "positions": rows, "baseline_perplexity": left["perplexity"], "candidate_perplexity": right["perplexity"],
            "baseline_task_accuracy": left["task_accuracy"], "candidate_task_accuracy": right["task_accuracy"],
            "task_accuracy_status": "measured" if accuracy_drop is not None else "not_evaluated",
            "accuracy_drop": accuracy_drop, "limits": limits,
            "generation_budgets": [item["protocol"].get("generation_tokens") for item in (left, right)],
            "free_generation_tokens_equal": [s["output_token_sha256"] for s in left["samples"]] == [s["output_token_sha256"] for s in right["samples"]]}


def compare_campaign(plan, pairs):
    """Pairs contain supervisor-validated directories and measured result metrics."""
    spec = plan["spec"]
    if fingerprint(spec) != plan["spec_sha256"]:
        raise ValueError("EXPERIMENT_PLAN_INTEGRITY")
    reasons, quality, mechanisms, deltas, processes, inputs = [], [], [], [], [], []
    role_costs = {role: {arm: [] for arm in ("baseline", "candidate")} for role in ("performance", "diagnostic")}
    for pair in pairs:
        left_dir, right_dir = Path(pair["baseline_dir"]), Path(pair["candidate_dir"])
        left, right = load_bundle(left_dir), load_bundle(right_dir)
        inputs.extend([left["manifest_sha256"], right["manifest_sha256"]])
        a, b = left["receipt"], right["receipt"]
        if not a or not b or left["manifest"]["status"] != "complete" or right["manifest"]["status"] != "complete":
            reasons.append("INCOMPLETE_EXECUTION")
            continue
        role = pair["role"]
        if role in role_costs:
            for arm, document in (("baseline", a), ("candidate", b)):
                measured = pair.get(arm + "_metrics", {})
                role_costs[role][arm].append({"runtime_e2e_ms": document["proof"].get("runtime_e2e_ms"),
                                            "worker_request_total_ms": measured.get("worker_request_total_ms"),
                                            "worker_request_cpu_ms": measured.get("worker_request_cpu_ms")})
        for arm, document in (("baseline", a), ("candidate", b)):
            defaults = {"cleanup_policy": "empty_cache", "attention_backend": "eager",
                        "output_budget": spec["quality_plan"]["generation_tokens"] if role == "quality" else spec["output_tokens"]}
            expected_effective = resolve_plan(spec[arm], defaults)["resolved"]
            if any(document["effective"].get(key) != value for key, value in expected_effective.items()):
                reasons.append("REGISTERED_EFFECT_NOT_MATCHED:" + arm)
            expected_cleanup = "torch.cuda.empty_cache" if expected_effective["cleanup_policy"] == "empty_cache" else "empty_cache_not_called"
            if document["proof"].get("cleanup_call") != expected_cleanup:
                reasons.append("CLEANUP_PROOF_MISMATCH:" + arm)
            expected_hash = fingerprint({"schema_version": "1.0.0", "changes": spec[arm].get("changes", {})})
            if document["plan_sha256"] != expected_hash:
                reasons.append("REGISTERED_PLAN_MISMATCH:" + arm)
            links = document.get("experiment_links", {})
            if links.get("spec_sha256") != plan["spec_sha256"] or links.get("arm") != arm or links.get("role") != role or links.get("campaign_id") != plan.get("experiment_id"):
                reasons.append("EXPERIMENT_LINK_MISMATCH")
        expected_role = "diagnostic" if role == "diagnostic" else role
        if a["measurement_role"] != expected_role or b["measurement_role"] != expected_role:
            reasons.append("MEASUREMENT_ROLE_MISMATCH")
        if a["status"] != "applied" or b["status"] != "applied":
            reasons.append("OPTIMIZATION_NOT_APPLIED")
        controls = spec.get("control_changes", {})
        for key in ("model_identity", "tokenizer_identity"):
            if key in controls:
                if a[key] != controls[key]["baseline"] or b[key] != controls[key]["candidate"]:
                    reasons.append("DECLARED_CONTROL_NOT_MATCHED:" + key)
            elif a[key] != b[key]:
                reasons.append("CONTROL_MISMATCH:" + key)
        if any(item["model_identity"].get("scope") != "verified_manifest_at_load" for item in (a, b)):
            reasons.append("WEIGHTS_NOT_VERIFIED")
        for key in ("source_digest", "packages", "kernel", "machine", "allocator_environment"):
            if key == "source_digest" and controls.get("source_files"):
                af, bf = a["source"].get("source_files", {}), b["source"].get("source_files", {})
                changed = {path for path in af.keys() | bf.keys() if af.get(path) != bf.get(path)}
                if changed != set(controls["source_files"]) or any(af.get(path) != hashes["baseline"] or bf.get(path) != hashes["candidate"] for path, hashes in controls["source_files"].items()):
                    reasons.append("SOURCE_PATCH_SCOPE_MISMATCH")
            elif a["source"].get(key) != b["source"].get(key):
                reasons.append("CONTROL_MISMATCH:" + key)
        for key in ("dtype", "kernel"):
            av, bv = a.get("runtime", {}).get(key), b.get("runtime", {}).get(key)
            if key == "kernel" and "kernel_profile" in spec["declared_changes"]:
                if key in controls and (av != controls[key]["baseline"] or bv != controls[key]["candidate"]):
                    reasons.append("DECLARED_CONTROL_NOT_MATCHED:" + key)
                for arm, document in (("baseline", a), ("candidate", b)):
                    expected = spec[arm]["changes"]["kernel_profile"]
                    evidence = document.get("kernel_receipt") or {}
                    if document.get("runtime", {}).get("kernel") != expected or evidence.get("profile_id") != expected or evidence.get("status") not in {"reference", "invoked"}:
                        reasons.append("KERNEL_EXECUTION_PROOF_MISMATCH:" + arm)
                    directory = left_dir if arm == "baseline" else right_dir
                    try:
                        bundle = left if arm == "baseline" else right
                        if not {"kernel-summary.json", "kernel-bindings.json", "kernel-plan.json"}.issubset(
                            {item["path"] for item in bundle["manifest"]["artifacts"]}
                        ):
                            raise ValueError("kernel artifacts not sealed")
                        summary = read_json(directory / "kernel-summary.json")
                        binding = read_json(directory / "kernel-bindings.json")
                        kernel_plan = read_json(directory / "kernel-plan.json")
                        if (summary != evidence or fingerprint(binding) != summary["binding_manifest_sha256"]
                                or fingerprint(kernel_plan) != summary["kernel_plan_sha256"]
                                or kernel_plan["profile_id"] != expected):
                            raise ValueError("kernel evidence identity")
                        if expected != "framework-native" and (summary["not_exercised"] or not summary["sites"]):
                            raise ValueError("kernel sites not exercised")
                    except (OSError, ValueError, KeyError):
                        reasons.append("KERNEL_ARTIFACT_PROOF_MISMATCH:" + arm)
                continue
            if key in controls:
                if av != controls[key]["baseline"] or bv != controls[key]["candidate"]:
                    reasons.append("DECLARED_CONTROL_NOT_MATCHED:" + key)
            elif av != bv:
                reasons.append("RUNTIME_CONTROL_MISMATCH:" + key)
        for key in a["effective"]:
            if key not in spec["declared_changes"] and a["effective"][key] != b["effective"].get(key):
                reasons.append("UNDECLARED_EFFECTIVE_CHANGE:" + key)
        if role == "quality":
            quality.append(_quality(left_dir, right_dir, left["quality"], right["quality"], spec["quality_limits"],
                                    allow_generation_budget_change="output_budget" in spec["declared_changes"]))
        elif role == "diagnostic":
            x, y = analyze(left_dir), analyze(right_dir)
            if x["capture_status"] != "complete" or y["capture_status"] != "complete" or x["context"]["plan"] != y["context"]["plan"]:
                reasons.append("DIAGNOSTIC_COVERAGE_MISMATCH")
            context_equal = a["output_token_hash"] == b["output_token_hash"]
            if x["context"].get("workload_hash") != y["context"].get("workload_hash"):
                reasons.append("WORKLOAD_MISMATCH")
            lm = {(m["metric_id"], m.get("label")): m for m in x["metrics"] + _kernel_metrics(a)}
            rm = {(m["metric_id"], m.get("label")): m for m in y["metrics"] + _kernel_metrics(b)}
            diffs = []
            for key in sorted(lm.keys() & rm.keys(), key=str):
                p, q = lm[key], rm[key]
                if p["value"] is not None and q["value"] is not None and p["scope"] == q["scope"] and p["unit"] == q["unit"]:
                    diffs.append({"metric_id": key[0], "label": key[1], "delta": q["value"] - p["value"], "unit": p["unit"], "scope": p["scope"],
                                  "baseline_source_record_seqs": p.get("source_record_seqs", []),
                                  "candidate_source_record_seqs": q.get("source_record_seqs", [])})
            mechanisms.append({"differences": diffs, "generated_context_equal": context_equal,
                               "baseline_run_id": a["run_id"], "candidate_run_id": b["run_id"],
                               "baseline_observation_manifest_sha256": sha256(left_dir / "observations/manifest.json"),
                               "candidate_observation_manifest_sha256": sha256(right_dir / "observations/manifest.json"),
                               "interpretation": "same_context" if context_equal else "context_diverged_descriptive_only"})
        elif role == "performance":
            am, bm = a["proof"], b["proof"]
            if am.get("input_token_sha256") != bm.get("input_token_sha256") or not am.get("input_token_sha256"):
                reasons.append("PERFORMANCE_WORKLOAD_IDENTITY_MISMATCH")
            if a["output_token_hash"] != b["output_token_hash"] and spec.get("output_contract", "exact_tokens") == "exact_tokens":
                reasons.append("PERFORMANCE_GENERATED_CONTEXT_DIVERGED")
            if am["output_tokens"] != bm["output_tokens"]:
                reasons.append("PERFORMANCE_WORK_AMOUNT_CHANGED")
            if am["runtime_e2e_ms"] <= 0:
                reasons.append("PERFORMANCE_TIME_INVALID")
            else:
                deltas.append(bm["runtime_e2e_ms"] / am["runtime_e2e_ms"] - 1)
            processes.extend([fingerprint(a["source"]["process"]), fingerprint(b["source"]["process"])])
    performance = performance_gate(deltas, maximum=spec["max_relative_e2e"], independent=bool(processes) and len(set(processes)) == len(processes))
    quality_status = ("not_evaluated" if not quality else "failed" if any(q["status"] == "failed" for q in quality)
                      else "not_comparable" if any(q["status"] == "not_comparable" for q in quality)
                      else "not_evaluated" if any(q["status"] == "not_evaluated" for q in quality) else "passed")
    mechanism_passed = bool(spec["mechanism_expectations"] and mechanisms)
    for expectation in spec["mechanism_expectations"]:
        for mechanism in mechanisms:
            matches = [d for d in mechanism["differences"] if d["metric_id"] == expectation["metric_id"] and d["label"] == expectation["label"]]
            sign = 1 if expectation["direction"] == "increase" else -1
            mechanism_passed &= bool(mechanism["generated_context_equal"] and matches and sign * matches[0]["delta"] >= expectation["minimum_delta"])
    observer_cost = {"scope": "descriptive_separate_role_samples_not_low_overhead_acceptance", "arms": {}}
    for arm in ("baseline", "candidate"):
        reference = role_costs["performance"][arm]
        diagnostics = role_costs["diagnostic"][arm]
        deltas_by_metric = {}
        for metric_id in ("runtime_e2e_ms", "worker_request_total_ms", "worker_request_cpu_ms"):
            controls = [row[metric_id] for row in reference if row[metric_id] is not None]
            captured = [row[metric_id] for row in diagnostics if row[metric_id] is not None]
            if controls and captured and statistics.median(controls) > 0:
                deltas_by_metric[metric_id] = {"relative_delta": statistics.median(captured) / statistics.median(controls) - 1,
                                             "reference_samples": len(controls), "diagnostic_samples": len(captured)}
        observer_cost["arms"][arm] = deltas_by_metric
    adoption_reasons = []
    if spec["declared_changes"] == ["cleanup_policy"] and not spec.get("control_changes"):
        adoption_reasons.append("CLEANUP_OUTSIDE_PRIMARY_GENERATION_TIME_WINDOW:no_repeated_request_benefit_measured")
    return {"schema_version": "1.0.0", "spec_sha256": plan["spec_sha256"],
            "analysis_modules_sha256": {name: sha256(Path(__file__).resolve().parents[1] / name) for name in
                                         ("experiments/comparison.py", "quant/quality.py", "benchmark/metrics.py", "evaluation/optimization_comparison.py")},
            "performance_scope": "generation_loop_host_e2e_excludes_load_quality_and_archive",
            "input_manifest_hashes": inputs, "comparable": not reasons, "reasons": sorted(set(reasons)),
            "quality_status": quality_status, "quality": quality, "performance": performance,
            "observer_cost": observer_cost,
            "mechanisms": mechanisms, "mechanism_gate": "passed" if mechanism_passed else "not_passed",
            "adoption_reasons": adoption_reasons,
            "decision": "eligible_under_declared_contract" if not reasons and not adoption_reasons and quality_status == "passed" and performance["status"] == "passed" and mechanism_passed else "hold",
            "notice": "Only the declared dataset/workload and actual executor controls are covered; no general optimization claim."}
