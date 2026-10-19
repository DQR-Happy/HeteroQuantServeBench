"""Composition root: resident model, pure scheduler, timed sender and evidence."""
import hashlib
import json
import queue
import resource
import time

from hqsb.backends.step_executor import StepExecutor, capabilities
from hqsb.core.contracts.scheduling import SchedulingExperiment, Sequence
from hqsb.observation.storage import EvidenceWriter
from hqsb.observation.procfs import process_key, sample_process
from hqsb.runtime.live_engine import LiveEngine
from hqsb.serving.scheduling_loadgen import WorkloadSender


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * p))] if values else None


def execute(provider, payload, cancel, controls):
    from hqsb.backends.precision_runtime import receipt as precision_receipt, reset_calls
    from hqsb.backends.structure_runtime import receipt as structure_receipt
    spec = SchedulingExperiment.model_validate(payload["experiment"])
    capability = capabilities(provider)
    if not capability["supported"]:
        raise ValueError("SCHEDULING_UNSUPPORTED: " + ",".join(capability["reasons"]))
    if spec.limits.prefill_chunk_tokens > spec.limits.max_batched_tokens and spec.limits.chunked_prefill:
        raise ValueError("CHUNK_EXCEEDS_TOKEN_BUDGET")
    prepared = []
    for item in spec.requests:
        ids = provider.tokenizer.encode(item.prompt, add_special_tokens=True)
        if not ids:
            raise ValueError("EMPTY_TOKENIZED_PROMPT")
        if item.output_tokens > provider.config["max_output_tokens"]:
            raise ValueError("DEPLOYMENT_OUTPUT_LIMIT")
        prepared.append({**item.model_dump(), "input_ids": ids})
    identity = payload["capture_id"]
    writer = EvidenceWriter(payload["capture_dir"], {
        "run_id": identity, "schema": "l8-v1", "workload_sha256": digest([x.model_dump() for x in spec.requests]),
        "model_identity": provider.source_identity, "source_environment": provider.source_environment,
        "capability": capability, "limits": spec.limits.model_dump(), "save_input": spec.save_input},
        queue_records=1024)
    variants, baseline = [], {}
    cleanup, status, problems = "unknown", "failed", []
    current_session = None
    try:
        writer.offer("execution.receipt", {"run_id": identity, "epoch": spec.expected_epoch,
            "capability": capability, "model_identity": provider.source_identity,
            "tokenizer_identity": provider.tokenizer_identity,
            "attention": provider.config["attention"], "precision": provider.config["precision"],
            "precision_receipt": precision_receipt(provider),
            "structure_receipt": structure_receipt(provider),
            "kernel_profile": provider.kernel_session.profile if provider.kernel_session else None,
            "sampling": "greedy", "cleanup_policy": "session_empty_cache", "paged_kv": False,
            "kv_layout": spec.limits.kv_layout, "kv_limits": spec.limits.model_dump()})
        for repeat in range(spec.repeats):
            for mode in spec.modes:
                if cancel.is_set():
                    break
                current_session = f"{identity}:{repeat}:{mode}"
                reset_calls(provider)
                executor = StepExecutor(provider, spec.limits.max_padded_positions, spec.limits.max_temporary_bytes, limits=spec.limits)
                receipts = []

                def emit(kind, body):
                    writer.offer(kind, body)
                    if kind == "forward":
                        receipts.append(body)

                eos = provider.model.generation_config.eos_token_id
                engine = LiveEngine(executor, spec.limits, mode, emit, current_session,
                                    eos if isinstance(eos, list) else [eos])
                sender = WorkloadSender(prepared, spec.arrival_mode, spec.concurrency,
                                        lag_limit_ms=spec.sender_lag_limit_ms)
                usage_before = resource.getrusage(resource.RUSAGE_SELF)
                process = process_key()
                system_before = sample_process(process)
                sender.start()
                origin = sender.origin
                def receive():
                    incoming = sender.drain()
                    for _ in range(256):
                        try:
                            control = controls.get_nowait()
                        except queue.Empty:
                            break
                        if control.get("run_id") != identity:
                            continue  # stale messages cannot act on a new session
                        if control["kind"] == "cancel_request":
                            incoming.append(("cancel", control["request_id"]))
                        elif control["kind"] == "submit_request":
                            item = control["item"]
                            now = time.monotonic()
                            ids = provider.tokenizer.encode(item["prompt"], add_special_tokens=True)
                            if not ids or item["output_tokens"] > provider.config["max_output_tokens"]:
                                writer.offer("control.refused", {"session_id": current_session,
                                    "request_id": item["id"], "reason": "TOKENIZATION_OR_OUTPUT_LIMIT"})
                            else:
                                incoming.append(("submit", {**item, "input_ids": ids, "sent": now,
                                                             "planned": now, "live_ingress": True}))
                    result = []
                    for kind, value in incoming:
                        if kind != "submit":
                            result.append((kind, value))
                            continue
                        now = time.monotonic()
                        row = Sequence(value["id"], value["input_ids"], value["output_tokens"], now,
                            now + value["deadline_ms"] / 1000, priority=value["priority"], tenant=value["tenant"],
                            cancel_at=None if value["cancel_after_ms"] is None else value["sent"] + value["cancel_after_ms"] / 1000,
                            fixed=value["fixed_output_length"])
                        writer.offer("workload", {"session_id": current_session, "request_id": row.id,
                            "request_run_id": f"{current_session}:{row.id}",
                            "planned": value["planned"], "sent": value["sent"], "received": now,
                            "prompt": value["prompt"] if spec.save_input else None,
                            "input_ids": row.input_ids if spec.save_input else None,
                            "input_sha256": digest(row.input_ids), "input_storage": "saved" if spec.save_input else "hash_only"})
                        result.append(("submit", row))
                    return result

                try:
                    rows = engine.run(receive, cancel.is_set, origin + spec.session_deadline_s, sender.complete)
                    elapsed = time.monotonic() - origin
                finally:
                    try:
                        fidelity = sender.close()
                    finally:
                        final_memory = engine.close()
                usage_after = resource.getrusage(resource.RUSAGE_SELF)
                summaries = []
                for row in rows:
                    first, last = (row.times[0], row.times[-1]) if row.times else (None, None)
                    itl = [(b - a) * 1000 for a, b in zip(row.times, row.times[1:])]
                    summary = {"id": row.id, "reason": row.reason, "error": row.error,
                        "input_tokens": len(row.input_ids), "output_tokens": len(row.output),
                        "ttft_ms": (first - row.received) * 1000 if first else None,
                        "queue_ms": (row.admitted - row.received) * 1000 if row.admitted else None,
                        "e2e_ms": (row.ended - row.received) * 1000 if row.ended else None,
                        "tpot_ms": (last - first) * 1000 / (len(row.output) - 1) if len(row.output) > 1 else None,
                        "decode_tokens_per_s": (len(row.output) - 1) / (last - first) if len(row.output) > 1 and last > first else None,
                        "itl_p95_ms": percentile(itl, .95), "recomputed_positions": row.recomputed,
                        "output_tokens_per_s": len(row.output) / (row.ended - row.received) if row.ended and row.ended > row.received else None,
                        "selected_token_nll": -sum(row.logprobs) / len(row.logprobs) if row.logprobs else None,
                        "preemptions": row.preemptions, "output_sha256": digest(row.output),
                        "max_unscheduled_ms": row.max_unscheduled_ms,
                        "text": provider.tokenizer.decode(row.output, skip_special_tokens=True),
                        "tenant": row.tenant}
                    eligible = row.reason in {"length", "stop"}
                    if mode == "serial" and eligible:
                        baseline[(repeat, row.id)] = (row.output, row.reason, row.logprobs, row.margins)
                    ref = baseline.get((repeat, row.id))
                    summary["quality"] = "reference" if mode == "serial" else (
                        "exact_tokens_pass" if ref[:2] == (row.output, row.reason) else "exact_tokens_fail") if ref and eligible else "not_comparable"
                    summary["selected_logprob_max_abs_delta"] = max((abs(a - b) for a, b in zip(ref[2], row.logprobs)), default=0) if ref and ref[:2] == (row.output, row.reason) else None
                    if ref:
                        first_difference = next((i for i, (a, b) in enumerate(zip(ref[0], row.output)) if a != b), None)
                        common = first_difference if first_difference is not None else min(len(ref[0]), len(row.output))
                        summary["common_prefix_tokens"] = common
                        summary["prefix_logprob_max_abs_delta"] = max((abs(a - b) for a, b in zip(ref[2][:common], row.logprobs[:common])), default=0)
                        summary["first_difference"] = None if first_difference is None else {
                            "output_index": first_difference + 1, "baseline_token": ref[0][first_difference],
                            "candidate_token": row.output[first_difference],
                            "baseline_top2_logit_margin": ref[3][first_difference],
                            "candidate_top2_logit_margin": row.margins[first_difference]}
                    summary["slo_pass"] = bool(eligible and summary["ttft_ms"] is not None
                        and summary["ttft_ms"] <= spec.ttft_slo_ms and all(v <= spec.itl_slo_ms for v in itl))
                    summaries.append(summary)
                    writer.offer("request.result", {"session_id": current_session, **summary,
                        "output_token_ids": row.output, "token_ready_times": row.times, "selected_logprobs": row.logprobs,
                        "top2_logit_margins": row.margins,
                        "input_consumed": row.consumed})
                dynamic = sum(bool(set(a["members"]) & set(b["members"])) and set(a["members"]) != set(b["members"])
                              for a, b in zip(receipts, receipts[1:]))
                outputs = sum(len(r.output) for r in rows)
                failed = any(r.reason == "rejected" for r in rows)
                variant = {"session_id": current_session, "repeat": repeat, "mode": mode,
                    "elapsed_s": elapsed, "output_tokens": outputs, "output_tokens_per_s": outputs / elapsed,
                    "completed_requests_per_s": sum(r.reason in {"length", "stop"} for r in rows) / elapsed,
                    "slo_goodput_requests_per_s": sum(s["slo_pass"] for s in summaries) / elapsed,
                    "reason_counts": {reason: sum(r.reason == reason for r in rows) for reason in {r.reason for r in rows}},
                    "tenant_service_positions": dict(engine.service),
                    "ttft_p95_ms": percentile([s["ttft_ms"] for s in summaries if s["ttft_ms"] is not None], .95),
                    "forwards": len(receipts), "max_actual_batch": max((len(r["members"]) for r in receipts), default=0),
                    "membership_changes_with_overlap": dynamic,
                    "dynamic_join_requests": sorted(engine.confirmed_join_ids),
                    "logical_positions": sum(r["logical_positions"] for r in receipts),
                    "padding_positions": sum(r["padding_positions"] for r in receipts),
                    "kv_padding_positions": sum(r["kv_padding_positions"] for r in receipts),
                    "kv_copy_bytes": sum(r["pack_copy_bytes"] + r["scatter_copy_bytes"] for r in receipts),
                    "kv_layout": spec.limits.kv_layout,
                    "memory_management": {"peak_pool_storage_bytes": max((r.get("memory_management", {}).get("pool_storage_bytes", 0) for r in receipts), default=0),
                        "peak_tail_waste_tokens": max((r.get("memory_management", {}).get("tail_waste_tokens", 0) for r in receipts), default=0),
                        "pack_copy_bytes": sum(r["pack_copy_bytes"] for r in receipts),
                        "scatter_copy_bytes": sum(r["scatter_copy_bytes"] for r in receipts)},
                    "gpu_forward_ms": sum(r["gpu_forward_ms"] for r in receipts),
                    "peak_allocator_bytes": max((r["memory"]["peak_allocated_bytes"] for r in receipts), default=0),
                    "peak_reserved_kv_tokens": engine.kv.peak, "cleanup": final_memory,
                    "system": {"scope": "worker_session", "minor_faults": usage_after.ru_minflt - usage_before.ru_minflt,
                        "major_faults": usage_after.ru_majflt - usage_before.ru_majflt,
                        "max_rss_kib_process_lifetime": usage_after.ru_maxrss,
                        "before": system_before, "after": sample_process(process)},
                    "loadgen": fidelity, "requests": summaries,
                    "precision_receipt": precision_receipt(provider),
                    "structure_receipt": structure_receipt(provider),
                    "outcome": "invalid" if fidelity["validity"] != "VALID" else "rejected_requests" if failed else "completed"}
                writer.offer("session.result", variant)
                variants.append(variant)
                yield {"kind": "progress", "session_id": current_session,
                       "mode": mode, "output_tokens_per_s": variant["output_tokens_per_s"],
                       "completed_variants": len(variants)}
        cleanup = "succeeded"
        status = "cancelled" if cancel.is_set() else "completed"
    except Exception as exc:
        problems.append(f"{type(exc).__name__}:{exc}"[:800])
        raise
    finally:
        report = writer.close(execution_status=status, cleanup_status=cleanup, source_problems=problems)
    quality_failures = sum(s["quality"] == "exact_tokens_fail" for v in variants for s in v["requests"])
    candidates = [s for v in variants if v["mode"] != "serial" for s in v["requests"]]
    matches = sum(s["quality"] == "exact_tokens_pass" for s in candidates)
    incomparable = sum(s["quality"] == "not_comparable" for s in candidates)
    quality = "failed" if quality_failures else (
        "partial_exact_tokens_pass" if incomparable else "exact_tokens_pass") if matches else "not_evaluated"
    yield {"kind": "result", "finish_reason": status, "variants": variants,
        "quality": quality,
        "quality_coverage": {"candidate_requests": len(candidates), "exact_matches": matches,
            "failures": quality_failures, "not_comparable": incomparable},
        "quality_failures": quality_failures, "capability": capability,
        "evidence": {"capture_status": report["capture_status"], "counts": report["counts"], "queue": report["queue"]},
        "cleanup": cleanup}
