"""Fixed, versioned interactive comparison tasks and bounded result projection.

Task accuracy is scored only where a reference answer exists. Token agreement
describes change from a baseline, never substitutes for task correctness.
"""
import json
import math
import statistics

from hqsb.core.optimization import fingerprint

VERSION = "typical-inputs-v1"
SYSTEM = "Follow the requested output format exactly. Do not include reasoning or extra commentary."
CASES = [
    {"id": "arithmetic", "name": "短答案 · 算术", "category": "短输入 / 短输出", "scoring": "exact",
     "prompt": "计算 17 × 19。只输出十进制整数，不要解释。", "expected": "323"},
    {"id": "structured", "name": "结构化提取", "category": "格式与事实", "scoring": "json",
     "prompt": '从记录“产品代号 AX17，购买数量 3”提取信息。只输出一个 JSON 对象，键为 sku（字符串）和 quantity（整数），不要输出代码围栏。',
     "expected": {"sku": "AX17", "quantity": 3}},
    {"id": "retrieval", "name": "上下文检索", "category": "较长输入 / 短输出", "scoring": "exact",
     "prompt": "Read the following lookup table.\n" + "\n".join(
         f"item{i:02d} = {('amber' if i == 19 else 'blue' + str(i))}" for i in range(40))
         + "\nWhat is the value of item19? Output the value only.", "expected": "amber"},
    {"id": "explanation", "name": "解释生成", "category": "较长输出 / 开放任务", "scoring": "unscored",
     "prompt": "用三句话解释大模型推理中 KV Cache 的作用、内存代价和适用条件。", "expected": None},
]


def suite(input_mode="chat"):
    cases = [{**case, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": case["prompt"]}]} for case in CASES]
    if input_mode == "completion":
        for case in cases:
            case.update(messages=[], input_mode="completion")
    for case in cases:
        case["input_sha256"] = fingerprint({"input_mode": "completion", "prompt": case["prompt"]}) if input_mode == "completion" else fingerprint(case["messages"])
    return {"version": VERSION + "-completion-v1" if input_mode == "completion" else VERSION, "cases": cases, "suite_sha256": fingerprint(cases),
            "quality_scope": "fixed_tasks_only_not_general_model_accuracy"}


def score(case, output):
    if case["scoring"] == "unscored":
        return None
    if case["scoring"] == "exact":
        return output.strip() == case["expected"]
    try:
        value = json.loads(output.strip())
        return isinstance(value, dict) and value == case["expected"] and type(value.get("quantity")) is int
    except (ValueError, TypeError):
        return False


METRICS = ("runtime_first_token_ms", "runtime_e2e_ms", "decode_tail_tokens_per_s", "input_tokens", "output_tokens")


def project_run(run, *, scheme_id, case, role, repeat):
    metrics = run.get("metrics", {})
    receipt = metrics.get("optimization_receipt") or {}
    return {"run_id": run["id"], "scheme_id": scheme_id, "case_id": case["id"], "role": role,
            "repeat": repeat, "state": run["state"], "cleanup": run["cleanup"], "error": run.get("error"),
            "output": run.get("output", ""), "finish_reason": run.get("finish_reason"),
            "score": score(case, run.get("output", "")) if run["state"] == "completed" else None,
            "metrics": {**{key: metrics.get(key) for key in METRICS},
                        "peak_allocated_bytes": metrics.get("memory", {}).get("peak_allocated_bytes")},
            "input_token_sha256": metrics.get("input_token_sha256"),
            "generated_token_ids": metrics.get("generated_token_ids"),
            "receipt": {key: receipt.get(key) for key in ("status", "effective", "runtime", "kernel_receipt", "source_digest", "model_identity", "tokenizer_identity")}}


def summarize(plan, records):
    measured = [row for row in records if row["role"] == "measurement"]
    rows = []
    for case in plan["cases"]:
        baseline = [row for row in measured if row["case_id"] == case["id"] and row["scheme_id"] == plan["baseline_id"]]
        for scheme in plan["schemes"]:
            attempts = [row for row in measured if row["case_id"] == case["id"] and row["scheme_id"] == scheme["id"]]
            complete = [row for row in attempts if row["state"] == "completed" and row["cleanup"] == "succeeded"]
            reasons, agreement, same_text = [], [], []
            if len(complete) != plan["repeats"] or len(baseline) != plan["repeats"]:
                reasons.append("INCOMPLETE_MEASUREMENTS")
            for item in complete:
                peer = next((row for row in baseline if row["repeat"] == item["repeat"] and row["state"] == "completed" and row["cleanup"] == "succeeded"), None)
                if not peer:
                    reasons.append("BASELINE_INCOMPLETE")
                    continue
                for key, reason in (("input_token_sha256", "INPUT_TOKENS_DIFFER"),):
                    if not item[key] or item[key] != peer[key]:
                        reasons.append(reason)
                if item["metrics"]["output_tokens"] != peer["metrics"]["output_tokens"]:
                    reasons.append("OUTPUT_LENGTH_CHANGED")
                for sample in (item, peer):
                    if any(type(sample["metrics"].get(key)) is not int or sample["metrics"][key] <= 0 for key in ("input_tokens", "output_tokens")):
                        reasons.append("TOKEN_COUNTS_UNAVAILABLE")
                    elapsed = sample["metrics"].get("runtime_e2e_ms")
                    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
                        reasons.append("TIMING_UNAVAILABLE")
                for key in ("model_identity", "source_digest", "tokenizer_identity"):
                    if not item["receipt"].get(key) or item["receipt"][key] != peer["receipt"].get(key):
                        reasons.append(key.upper() + "_DIFFERS")
                effective = item["receipt"].get("effective") or {}
                if item["receipt"].get("status") != "applied" or effective.get("kernel_profile", "framework-native") != scheme["kernel_profile"] or effective.get("cleanup_policy") != scheme["cleanup_policy"]:
                    reasons.append("EXECUTION_NOT_CONFIRMED")
                baseline_scheme = next(row for row in plan["schemes"] if row["id"] == plan["baseline_id"])
                baseline_effective = peer["receipt"].get("effective") or {}
                if (peer["receipt"].get("status") != "applied"
                        or baseline_effective.get("kernel_profile", "framework-native") != baseline_scheme["kernel_profile"]
                        or baseline_effective.get("cleanup_policy") != baseline_scheme["cleanup_policy"]):
                    reasons.append("BASELINE_EXECUTION_NOT_CONFIRMED")
                a, b = item.get("generated_token_ids"), peer.get("generated_token_ids")
                if isinstance(a, list) and isinstance(b, list) and max(len(a), len(b)):
                    agreement.append(sum(x == y for x, y in zip(a, b)) / max(len(a), len(b)))
                same_text.append(item["output"] == peer["output"])
            medians = {}
            for key in (*METRICS, "peak_allocated_bytes"):
                values = [row["metrics"][key] for row in complete if isinstance(row["metrics"].get(key), (int, float)) and math.isfinite(row["metrics"][key])]
                medians[key] = statistics.median(values) if values and len(values) == len(complete) else None
            base_times = [row["metrics"]["runtime_e2e_ms"] for row in baseline if row["state"] == "completed" and row["metrics"].get("runtime_e2e_ms") is not None]
            base_time = statistics.median(base_times) if base_times else None
            comparable = not reasons and bool(base_time) and medians["runtime_e2e_ms"] is not None
            scores = [row["score"] for row in complete if row["score"] is not None]
            rows.append({"case_id": case["id"], "scheme_id": scheme["id"], "completed": len(complete), "expected": plan["repeats"],
                         "metrics": medians, "task_accuracy": sum(scores) / len(scores) if scores else None,
                         "scored_samples": len(scores), "token_agreement": statistics.mean(agreement) if agreement else None,
                         "text_equal_rate": statistics.mean(same_text) if same_text else None,
                         "comparable": comparable, "reasons": sorted(set(reasons)),
                         "relative_e2e": medians["runtime_e2e_ms"] / base_time - 1 if comparable else None,
                         "runs": [row["run_id"] for row in attempts]})
    return {"rows": rows, "performance_scope": "resident_worker_quick_comparison_not_independent_process_acceptance",
            "quality_scope": "reference_task_scores_and_baseline_agreement_are_distinct",
            "timing_scope": "generation_loop_excludes_load_warmup_tokenization_archival_and_quality_scoring",
            "warmups_in_statistics": False, "diagnostics_in_statistics": False}
