"""Pinned corpus/protocol contracts and token-weighted PPL; no model imports."""

import hashlib
import json
import math
from pathlib import Path

BENCHMARK_ID = "wikitext2-mini-v1"
DATA_SHA256 = "7a7ea1ae33195598a02cbf9877751166ef016ec8c745a76bdd2e009b44cd04d9"
TOKEN_LIMIT = 64


def load_benchmark():
    raw = (Path(__file__).with_name("data") / (BENCHMARK_ID + ".json")).read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("QUALITY_BENCHMARK_INTEGRITY")
    document = json.loads(raw)
    if document["id"] != BENCHMARK_ID or len(document["samples"]) != 8:
        raise ValueError("QUALITY_BENCHMARK_SCHEMA")
    return document


def catalogue():
    data = load_benchmark()
    return [
        {
            "id": BENCHMARK_ID,
            "name": "WikiText-2 Mini · 8 段 / 64 Token",
            "metric": "token_perplexity",
            "language": "en",
            "split": "test",
            "sample_count": len(data["samples"]),
            "max_tokens": TOKEN_LIMIT,
            "dataset_sha256": DATA_SHA256,
            "source": data["source"],
            "selection": data["selection"],
            "articles": [s["article"] for s in data["samples"]],
            "scope": "fixed_english_regression_subset_not_full_wikitext_or_general_task_accuracy",
            "default_forward_mode": "incremental_decode",
        }
    ]


def resolve_plan(raw):
    if raw.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("QUALITY_BENCHMARK_UNKNOWN")
    mode = raw.get("forward_mode", "incremental_decode")
    if mode not in {"incremental_decode", "full_prefill"}:
        raise ValueError("QUALITY_FORWARD_MODE_INVALID")
    data = load_benchmark()
    plan = {
        "benchmark_id": BENCHMARK_ID,
        "score_mode": "corpus_nll",
        "forward_mode": mode,
        "samples": [
            {"id": s["id"], "text": s["text"], "slice": s["article"]}
            for s in data["samples"]
        ],
        "max_tokens": TOKEN_LIMIT,
        "generation_tokens": 1,
        "benchmark_sha256": DATA_SHA256,
    }
    # This contract is validated at API, admission and worker boundaries. Only
    # the exact server-resolved plan may cross with the otherwise private fields.
    if set(raw) - {"benchmark_id", "forward_mode"} and raw != plan:
        raise ValueError("QUALITY_BENCHMARK_PLAN_MISMATCH")
    return plan


def aggregate(samples):
    if not samples or any(
        type(s["scored_tokens"]) is not int
        or s["scored_tokens"] < 1
        or not math.isfinite(s["nll_sum"])
        or s["nll_sum"] < 0
        for s in samples
    ):
        raise ValueError("QUALITY_CORPUS_SCORE_INVALID")
    tokens = sum(s["scored_tokens"] for s in samples)
    total = math.fsum(s["nll_sum"] for s in samples)
    mean = total / tokens
    return {
        "scored_tokens": tokens,
        "nll_sum": total,
        "mean_nll": mean,
        "perplexity": math.exp(mean) if mean < 709 else None,
        "perplexity_overflow": mean >= 709,
        "document_count": len(samples),
    }


def compare(left, right, max_mean_nll_increase=None):
    if any(
        q.get("status") != "complete" or q.get("score_mode") != "corpus_nll"
        for q in (left, right)
    ):
        raise ValueError("QUALITY_CORPUS_COMPLETE_RUNS_REQUIRED")
    if (
        left["benchmark_id"] != right["benchmark_id"]
        or left["dataset_sha256"] != right["dataset_sha256"]
        or left["protocol"] != right["protocol"]
    ):
        raise ValueError("QUALITY_CORPUS_DATA_OR_PROTOCOL_MISMATCH")
    a, b = left["samples"], right["samples"]
    if len(a) != len(b) or any(
        any(
            x[k] != y[k] for k in ("id", "input_sha256", "scored_tokens", "text_sha256")
        )
        for x, y in zip(a, b)
    ):
        raise ValueError("QUALITY_CORPUS_TOKEN_OR_SAMPLE_MISMATCH")
    baseline, candidate = aggregate(a), aggregate(b)
    increase = candidate["mean_nll"] - baseline["mean_nll"]
    ratio = math.exp(increase) if increase < 709 else None
    return {
        "metric": "token_perplexity",
        "status": "measured_no_acceptance_limits"
        if max_mean_nll_increase is None
        else "passed"
        if increase <= max_mean_nll_increase
        else "failed",
        "scope": "fixed_subset_same_model_tokenizer_and_protocol_only",
        "benchmark_id": left["benchmark_id"],
        "dataset_sha256": left["dataset_sha256"],
        "baseline_perplexity": baseline["perplexity"],
        "candidate_perplexity": candidate["perplexity"],
        "baseline_mean_nll": baseline["mean_nll"],
        "candidate_mean_nll": candidate["mean_nll"],
        "mean_nll_increase": increase,
        "ppl_ratio": ratio,
        "ppl_relative_increase_percent": (ratio - 1) * 100
        if ratio is not None
        else None,
        "scored_tokens": baseline["scored_tokens"],
        "document_count": len(a),
        "max_mean_nll_increase": max_mean_nll_increase,
        "documents": [
            {
                "id": x["id"],
                "article": x["slice"],
                "scored_tokens": x["scored_tokens"],
                "mean_nll_increase": y["nll_sum"] / y["scored_tokens"]
                - x["nll_sum"] / x["scored_tokens"],
            }
            for x, y in zip(a, b)
        ],
        "limitations": [
            "8 English passages; no general accuracy claim or full-corpus ranking",
            "not comparable with word-level PPL or different context protocols",
            "near-duplicate calibration leakage and pretraining contamination not checked",
        ],
    }
