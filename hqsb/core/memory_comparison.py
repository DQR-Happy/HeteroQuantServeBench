"""Strict descriptive L6 pairs. Never turn one prompt into a model quality score."""
def compare(a, b, ra, rb, ma, mb):
    reasons = []
    for label, run, receipt, manifest in (("baseline", a, ra, ma), ("candidate", b, rb, mb)):
        if run["state"] != "completed" or manifest.get("status") != "complete" or receipt.get("status") != "applied":
            reasons.append(label + "_incomplete")
        if not run["metrics"].get("memory_receipt"):
            reasons.append(label + "_memory_receipt_missing")
        if run["config"].get("graph_capture") or run["config"].get("capture_architecture") or run["config"].get("experiment_role"):
            reasons.append(label + "_not_generation_sample")
    for key in ("model_identity", "tokenizer_identity"):
        if not ra.get(key) or ra.get(key) != rb.get(key):
            reasons.append(key + "_mismatch")
    if not (ra.get("source") or {}).get("source_digest") or (ra.get("source") or {}).get("source_digest") != (rb.get("source") or {}).get("source_digest"):
        reasons.append("source_mismatch")
    for key in ("input_token_sha256", "requested_token_limit"):
        value = ra.get("proof", {}).get(key)
        if value is None or value != rb.get("proof", {}).get(key):
            reasons.append(key + "_mismatch")
    for key in ("attention_backend", "cleanup_policy"):
        if ra.get("effective", {}).get(key) != rb.get("effective", {}).get(key):
            reasons.append(key + "_mismatch")
    if ra.get("runtime") != rb.get("runtime"):
        reasons.append("runtime_identity_mismatch")
    if ra.get("precision_receipt", {}).get("plan_sha256") != rb.get("precision_receipt", {}).get("plan_sha256"):
        reasons.append("precision_mismatch")
    for key in ("epoch", "observation_mode", "observation_plan", "runtime_plan", "fixed_output_length"):
        if a["config"].get(key) != b["config"].get(key):
            reasons.append(key + "_mismatch")
    for key in ("output_tokens", "finish_reason"):
        if (a["metrics"].get(key) if key == "output_tokens" else a.get(key)) != (b["metrics"].get(key) if key == "output_tokens" else b.get(key)):
            reasons.append(key + "_mismatch")
    x, y = (r["metrics"].get("generated_token_ids") for r in (a, b))
    exact = bool(x and y and x == y)
    logs = [r["metrics"].get("memory_receipt", {}).get("selected_token_logprobs", []) for r in (a, b)]
    delta = max((abs(i - j) for i, j in zip(*logs)), default=0) if exact and all(logs) and len(logs[0]) == len(logs[1]) else None
    fields = ("worker_request_total_ms", "runtime_first_token_ms", "runtime_e2e_ms", "output_tokens_per_s", "decode_tail_tokens_per_s")
    return {"schema_version": "l6-v1", "baseline_run_id": a["id"], "candidate_run_id": b["id"],
            "comparable": not reasons, "reasons": reasons,
            "generation_check": {"scope": "same_prompt_greedy_output_only_not_benchmark_quality", "exact_tokens": exact,
                                 "max_abs_selected_token_logprob_delta": delta} if not reasons else None,
            "performance": {"status": "descriptive_pair_no_speedup_claim", **{k: [r["metrics"].get(k) for r in (a, b)] for k in fields}} if not reasons else None,
            "memory": [r["metrics"].get("memory_receipt") for r in (a, b)],
            "limitations": ["Generation timing excludes prefix restoration/snapshot and cleanup; worker_request_total_ms includes these costs.",
                            "Compare repeated cold/warm samples separately; no general performance or quality conclusion from one pair.",
                            "Output token agreement is a correctness check, not a replacement for an independent quality benchmark."]}
