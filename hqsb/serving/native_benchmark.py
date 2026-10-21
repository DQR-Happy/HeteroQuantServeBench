"""One common HTTP workload for both upstream engines, outside API serial ingress."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import statistics
import time

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from hqsb.core.optimization import fingerprint


class NativeBenchmark(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    expected_epoch: str
    prompts: list[str] = Field(default_factory=lambda: [
        "Compute 17 + 25. Give only the number.",
        "Explain why memory bandwidth matters for LLM decoding in two sentences.",
        "Continue the story: A small fox found a key beside the old tree.",
        "List three practical ways to reduce inference latency.",
    ], min_length=1, max_length=32)
    expected_outputs: list[str] | None = None
    rounds: int = Field(2, ge=1, le=128)
    concurrency: int = Field(4, ge=1, le=64)
    arrival_mode: Literal["closed_loop", "open_loop"] = "closed_loop"
    request_rate: float = Field(2.0, gt=0, le=1000)
    max_output_tokens: int = Field(32, ge=1, le=4096)
    fixed_output_length: bool = False
    warmup_requests: int = Field(1, ge=0, le=16)
    ttft_slo_ms: float = Field(2000, gt=0)
    e2e_slo_ms: float = Field(30000, gt=0)
    sender_lag_limit_ms: float = Field(50, gt=0)
    deadline_ms: int = Field(120000, ge=1000, le=600000)


def validate(raw):
    plan = NativeBenchmark.model_validate(raw).model_dump()
    if any(not p.strip() or len(p) > 32768 for p in plan["prompts"]):
        raise ValueError("NATIVE_BENCHMARK_PROMPT_LIMIT")
    if plan["expected_outputs"] is not None and len(plan["expected_outputs"]) != len(plan["prompts"]):
        raise ValueError("NATIVE_BENCHMARK_EXPECTED_OUTPUT_COUNT")
    if len(plan["prompts"]) * plan["rounds"] > 4096:
        raise ValueError("NATIVE_BENCHMARK_REQUEST_LIMIT")
    if len(plan["prompts"]) * plan["rounds"] * plan["max_output_tokens"] > 65536:
        raise ValueError("NATIVE_BENCHMARK_OUTPUT_EVIDENCE_LIMIT")
    return plan


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def execute(provider, raw, cancel, *, capture_window=None):
    import httpx
    plan = validate({k: v for k, v in raw.items() if k != "remaining_ms"})
    def send(index, prompt, planned_at=None):
        sent = time.monotonic()
        request = {"max_output_tokens": plan["max_output_tokens"], "remaining_ms": max(1, (deadline - sent) * 1000),
                   "fixed_output_length": plan["fixed_output_length"]}
        request.update({"prompt": prompt} if provider.config["input_mode"] == "completion" else {"messages": [{"role": "user", "content": prompt}]})
        row = {"index": index, "sent_at_s": sent - start, "planned_at_s": planned_at - start if planned_at is not None else None,
               "sender_lag_ms": max(0, (sent - planned_at) * 1000) if planned_at is not None else 0}
        try:
            if cancel.is_set() or sent >= deadline:
                return {**row, "status": "cancelled" if cancel.is_set() else "timed_out"}
            for event in provider.stream(request, cancel, client=client):
                if event["kind"] == "result":
                    row.update(event)
            row["status"] = "completed" if row.get("finish_reason") in {"stop", "length"} else row.get("finish_reason", "failed")
            expected = plan["expected_outputs"]
            row["quality_exact_match"] = row["text"].strip() == expected[index % len(expected)].strip() if expected else None
            return row
        except Exception as exc:
            return {**row, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:1200]}
    with httpx.Client(timeout=httpx.Timeout(plan["deadline_ms"] / 1000), trust_env=False,
                      limits=httpx.Limits(max_connections=plan["concurrency"], max_keepalive_connections=plan["concurrency"])) as client:
        start = time.monotonic()
        deadline = start + min(plan["deadline_ms"], raw.get("remaining_ms", plan["deadline_ms"])) / 1000
        for i in range(plan["warmup_requests"]):
            row = send(i, plan["prompts"][i % len(plan["prompts"])])
            if row["status"] != "completed":
                if cancel.is_set():
                    return {"finish_reason": "cancelled", "metrics": {"native_warmup_partial": row}}
                raise RuntimeError("NATIVE_BENCHMARK_WARMUP_FAILED:" + str(row))
        # Warmup is excluded from both metrics delta and elapsed throughput window.
        before = provider.snapshot()
        with capture_window() if capture_window else nullcontext():
            start = time.monotonic()
            prompts = plan["prompts"] * plan["rounds"]
            with ThreadPoolExecutor(max_workers=plan["concurrency"]) as pool:
                futures = []
                for index, prompt in enumerate(prompts):
                    planned = start + index / plan["request_rate"] if plan["arrival_mode"] == "open_loop" else None
                    if planned is not None:
                        while time.monotonic() < min(planned, deadline) and not cancel.is_set():
                            time.sleep(min(.01, max(0, planned - time.monotonic())))
                    futures.append(pool.submit(send, index, prompt, planned))
                rows = [future.result() for future in futures]
            elapsed = time.monotonic() - start
    after = provider.snapshot()
    completed = [row for row in rows if row["status"] == "completed"]
    timed = [row for row in completed if row["metrics"].get("proxy_first_content_ms") is not None]
    output_tokens = sum(row["metrics"].get("output_tokens") or 0 for row in completed)
    usage_complete = all(row["metrics"].get("output_tokens") is not None for row in completed)
    good = [row for row in timed if row["metrics"]["proxy_first_content_ms"] <= plan["ttft_slo_ms"]
            and row["metrics"]["runtime_e2e_ms"] <= plan["e2e_slo_ms"]]
    good_quality = [row for row in good if row["quality_exact_match"] is True]
    lag = max((row["sender_lag_ms"] for row in rows), default=0)
    summary = {"offered": len(rows), "completed": len(completed), "failed_or_cancelled": len(rows) - len(completed),
               "elapsed_s": elapsed, "timing_boundary": "first_send_through_last_response_including_drain",
               "output_tokens": output_tokens, "output_tokens_per_s": output_tokens / elapsed if usage_complete else None,
               "completed_requests_per_s": len(completed) / elapsed, "latency_goodput_requests_per_s": len(good) / elapsed,
               "quality_gated_goodput_requests_per_s": len(good_quality) / elapsed if plan["expected_outputs"] else None,
               "quality": "exact_match_fixture" if plan["expected_outputs"] else "outputs_recorded_not_scored",
               "quality_exact_match_rate": statistics.mean(row["quality_exact_match"] for row in completed) if completed and plan["expected_outputs"] else None,
               "sender_max_lag_ms": lag, "arrival_valid": plan["arrival_mode"] == "closed_loop" or lag <= plan["sender_lag_limit_ms"],
               "usage_complete": usage_complete, "cache_state": "engine_current_state_after_explicit_warmup"}
    for field, metric in [("ttft", "proxy_first_content_ms"), ("e2e", "runtime_e2e_ms")]:
        values = [row["metrics"][metric] for row in timed]
        summary[field + "_ms"] = {"p50": percentile(values, .5), "p95": percentile(values, .95),
                                  "p99": percentile(values, .99) if len(values) >= 1000 else None}
    plan_hash = fingerprint({k: v for k, v in plan.items() if k != "expected_epoch"})
    receipt = {"schema_version": "1.0.0", "plan": plan, "workload_sha256": plan_hash,
               "engine": provider.receipt, "summary": summary, "requests": rows,
               "native_counters": {"before": before, "after": after},
               "comparison_scope": "same_artifact_input_sampling_workload_required",
               "measurement_profile": "native-common-http-diagnostic-v1" if capture_window else "native-common-http-loadgen-v1"}
    result = {"finish_reason": "cancelled" if cancel.is_set() else "timed_out" if time.monotonic() >= deadline else "stop",
            "metrics": {"native_benchmark": receipt, "output_tokens_per_s": summary["output_tokens_per_s"]}}
    if not cancel.is_set() and len(completed) != len(rows):
        result["request_error"] = {"code": "NATIVE_BENCHMARK_REQUEST_FAILURE", "message": "部分请求失败，逐请求错误与已完成的输出已记录。"}
    return result
