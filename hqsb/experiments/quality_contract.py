"""Comparable quality evidence and eligibility of an unquantized reference."""

import math

from hqsb.core.corpus_benchmark import compare as compare_corpus
from hqsb.core.optimization import fingerprint


def source_digest(receipt):
    return receipt.get("source_digest") or receipt.get("source", {}).get(
        "source_digest"
    )


def check_executor(left, right):
    if (
        not left
        or not right
        or any(
            not left.get(k) or left.get(k) != right.get(k)
            for k in ("model_identity", "tokenizer_identity")
        )
    ):
        raise ValueError("L4_SOURCE_MODEL_OR_TOKENIZER_MISMATCH")
    if not source_digest(left) or source_digest(left) != source_digest(right):
        raise ValueError("L4_RUNTIME_SOURCE_CHANGED")
    if left.get("effective", {}).get("attention_backend") != right.get(
        "effective", {}
    ).get("attention_backend"):
        raise ValueError("L4_ATTENTION_BACKEND_MISMATCH")


def context(bundle):
    r, q = bundle["receipt"], bundle["quality"]
    if (
        not r
        or not q
        or q.get("status") != "complete"
        or bundle["manifest"]["status"] != "complete"
    ):
        raise ValueError("BASELINE_COMPLETE_SEALED_QUALITY_REQUIRED")
    if (
        r.get("status") != "applied"
        or r.get("execution_status") != "completed"
        or r.get("cleanup_status") != "succeeded"
    ):
        raise ValueError("BASELINE_APPLIED_EXECUTION_REQUIRED")
    if (
        not isinstance(r.get("model_identity"), dict)
        or not isinstance(r["model_identity"].get("hash"), str)
        or not r["model_identity"]["hash"]
        or not r.get("tokenizer_identity")
        or not source_digest(r)
    ):
        raise ValueError("BASELINE_IDENTITY_REQUIRED")
    if not q.get("dataset_sha256") or not q.get("protocol") or not q.get("samples"):
        raise ValueError("BASELINE_SCORING_IDENTITY_REQUIRED")
    samples = []
    for s in q["samples"]:
        if not s.get("input_sha256") or not s.get("text_sha256"):
            raise ValueError("BASELINE_SAMPLE_IDENTITY_REQUIRED")
        samples.append(
            {
                k: s.get(k)
                for k in (
                    "id",
                    "input_sha256",
                    "text_sha256",
                    "input_tokens",
                    "scored_tokens",
                )
            }
        )
    if type(q.get("mean_nll")) not in (int, float) or not math.isfinite(q["mean_nll"]):
        raise ValueError("BASELINE_FINITE_SCORE_REQUIRED")
    if (
        q["mean_nll"] < 0
        or type(q.get("scored_tokens")) is not int
        or q["scored_tokens"] < 1
    ):
        raise ValueError("BASELINE_FINITE_SCORE_REQUIRED")
    return {
        "model_identity": r["model_identity"],
        "tokenizer_identity": r["tokenizer_identity"],
        "source_digest": source_digest(r),
        "runtime_packages": r.get("source", {}).get("packages", {}),
        "machine_architecture": r.get("source", {}).get("machine"),
        "attention_backend": r.get("effective", {}).get("attention_backend"),
        "benchmark_id": q.get("benchmark_id"),
        "score_mode": q.get("score_mode", "full_logits"),
        "dataset_sha256": q["dataset_sha256"],
        "protocol": q["protocol"],
        "samples_sha256": fingerprint(samples),
        "scored_tokens": q.get("scored_tokens"),
    }


def is_floating_reference(bundle):
    r = bundle["receipt"] or {}
    p = r.get("precision_receipt") or {}
    dtype = p.get("compute_dtype", "").removeprefix("torch.")
    return (
        (r.get("structure_receipt") or {}).get("artifact_id") is None
        and
        p.get("method") == "none"
        and p.get("quantized_module_count") == 0
        and not p.get("artifact_id")
        and p.get("execution") == "floating_reference"
        and isinstance(p.get("kv"), dict)
        and p["kv"].get("backend") == "model_default"
        and "bits" in p["kv"]
        and p["kv"]["bits"] is None
        and dtype in {"float16", "bfloat16", "float32"}
        and r.get("runtime", {}).get("kernel") == "framework-native"
        and r.get("runtime_receipt", {}).get("plan", {}).get("mode", "eager") == "eager"
        and (dtype != "float32" or p.get("matmul_allow_tf32") is False)
    )


def compare_scores(left, right):
    check_executor(left["receipt"], right["receipt"])
    if context(left) != context(right):
        raise ValueError("BASELINE_SCORING_CONTEXT_MISMATCH")
    a, b = left["quality"], right["quality"]
    evaluation = {
        s.get(k)
        for s in b["samples"]
        for k in ("text_sha256", "scored_text_sha256")
        if s.get(k)
    }
    for bundle in (left, right):
        if evaluation & set(
            bundle["receipt"]
            .get("precision_receipt", {})
            .get("calibration_text_sha256", [])
        ):
            raise ValueError("L4_EXACT_CALIBRATION_EVALUATION_LEAKAGE")
    if a.get("score_mode") == "corpus_nll":
        result = compare_corpus(a, b)
    else:
        increase = b["mean_nll"] - a["mean_nll"]
        ratio = math.exp(increase) if increase < 709 else None
        result = {
            "metric": "token_perplexity",
            "status": "measured_no_acceptance_limits",
            "baseline_perplexity": a.get("perplexity"),
            "candidate_perplexity": b.get("perplexity"),
            "baseline_mean_nll": a["mean_nll"],
            "candidate_mean_nll": b["mean_nll"],
            "mean_nll_increase": increase,
            "ppl_ratio": ratio,
            "ppl_relative_increase_percent": (ratio - 1) * 100
            if ratio is not None
            else None,
        }
    result["ppl_delta"] = (
        b["perplexity"] - a["perplexity"]
        if a.get("perplexity") is not None and b.get("perplexity") is not None
        else None
    )
    result["task_accuracy_drop_percentage_points"] = (
        (a["task_accuracy"] - b["task_accuracy"]) * 100
        if a.get("task_accuracy") is not None and b.get("task_accuracy") is not None
        else None
    )
    overflow = []
    for key in ("ppl_ratio", "ppl_delta", "ppl_relative_increase_percent"):
        if result.get(key) is not None and not math.isfinite(result[key]):
            result[key] = None
            overflow.append(key)
    result["numeric_overflow_fields"] = overflow
    return result
