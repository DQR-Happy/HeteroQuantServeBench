"""L7 request lifecycle, real streaming, observations and sealed execution proof."""

import gc
from pathlib import Path
import time
from hqsb.core.decoding_plan import DecodingPlan, public_plan, plan_hash
from hqsb.core.optimization import fingerprint
from hqsb.backends.observation import ObservationRecorder
from hqsb.backends.decoding_engine import Decoder, DecodingStopped
from hqsb.backends.decoding_sampling import Sampler, ChoiceTrie
from hqsb.backends.decoding_draft import acquire, summary as draft_summary


def generate(provider, request, cancel):
    import torch
    from hqsb.backends.prompting import prepare_prompt, ContextLimitExceeded
    from hqsb.backends.optimization_receipt import resolve_request, receipt
    from hqsb.backends.precision_runtime import reset_calls
    from hqsb.backends.tensor_memory import cache_inventory
    from hqsb.observation.storage import atomic_json
    from hqsb.backends.observation_session import CaptureSession

    request, opt = resolve_request(provider.config, request)
    p = DecodingPlan.model_validate(request["decoding_plan"])
    reset_calls(provider)
    if provider.kernel_session:
        provider.kernel_session.reset()
    start = time.monotonic()
    cpu = time.process_time()
    deadline = start + request["remaining_ms"] / 1000
    status, cleanup, finish = "failed", "unknown", "length"
    limit = request["max_output_tokens"]
    ids = []
    times = []
    text = ""
    decoder = stream = None
    result = None
    input_hash = None
    count = 0
    runtime_start = None
    draft_loaded = False
    session = None
    measurement = ObservationRecorder(
        request.get("observation_mode", "basic"),
        start=start,
        execution={
            "decoding": p.method,
            "sampling": p.sampling,
            "dtype": provider.config["precision"],
            "provider": "pytorch",
            "use_cache": True,
        },
    )
    proof = {
        "schema_version": "l7-v1",
        "plan": public_plan(p),
        "plan_sha256": plan_hash(p),
        "status": "not_invoked",
        "algorithm": "longest_greedy_prefix_target_correction_and_cache_crop"
        if p.method != "autoregressive"
        else "serial_logit_processing",
        "timing_scope": "host wall clock including CUDA selection waits; burst commits are not independent decode forwards",
        "sample_class": "single_request_no_speedup_claim",
        "cleanup": {},
        "statistics": {},
    }

    def check():
        if cancel.is_set():
            raise DecodingStopped("cancelled")
        if time.monotonic() >= deadline:
            raise DecodingStopped("timed_out")

    try:
        check()
        torch.cuda.reset_peak_memory_stats()
        if measurement.mode != "off" and request.get("capture_dir"):
            session = CaptureSession(
                torch,
                request["capture_dir"],
                {
                    "run_id": request.get("capture_id"),
                    "source": provider.source_environment,
                    "model_identity": provider.source_identity,
                    "input_mode": request.get("input_mode"),
                    "decoding_plan_sha256": plan_hash(p),
                },
                "basic",
                None,
            )
            session.start(provider.model, provider.load_inventory)
            measurement.sink = session.emit
        measurement.snapshot("before_request", provider.memory)
        with measurement.phase("format_and_tokenize"):
            prompt, special = prepare_prompt(
                provider.tokenizer, request, provider.model.config.model_type
            )
            input_ids = provider.tokenizer.encode(prompt, add_special_tokens=special)
            if not input_ids:
                raise ValueError("L7_EMPTY_INPUT")
            count = len(input_ids)
            input_hash = fingerprint(input_ids)
            if count + limit > provider.config["context_limit"]:
                raise ContextLimitExceeded(
                    count, limit, provider.config["context_limit"]
                )
            eos = provider.model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos]) - {None}
            if any(t < 0 or t >= provider.model.config.vocab_size for t in eos):
                raise ValueError("L7_EOS_OUTSIDE_VOCABULARY")
            trie = (
                ChoiceTrie(provider.tokenizer, p.choices, eos, limit, p.min_new_tokens)
                if p.choices
                else None
            )
        draft = None
        with measurement.phase("draft_prepare"):
            if p.method == "assistant":
                draft, draft_loaded = acquire(
                    provider, request["_l7_draft_config"], count + limit, check
                )
        proof["draft_loaded_this_request"] = draft_loaded
        proof["resident_draft"] = draft_summary(provider)
        measurement.snapshot("after_draft_prepare", provider.memory)
        sampler = Sampler(p, provider.model.device, eos, trie)
        decoder = Decoder(
            provider.model, p, sampler, input_ids, limit, eos, check, draft
        )
        decoder.fixed = request.get("fixed_output_length", False)
        runtime_start = time.monotonic()
        stream = decoder.rounds()
        while len(ids) < limit:
            check()
            before = measurement.now_ms()
            try:
                with measurement.phase(
                    "prefill_and_verify" if not ids else "draft_and_verify"
                ):
                    emitted = next(stream)
            except StopIteration:
                break
            committed = time.monotonic()
            for token in emitted:
                check()
                ids.append(token)
                times.append((committed - runtime_start) * 1000)
                text = provider.tokenizer.decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ).rstrip("\ufffd")
                measurement.token(
                    index=len(ids),
                    phase="prefill" if len(ids) == 1 else "decode",
                    start_ms=before,
                    duration_ms=measurement.now_ms() - before,
                    model_host_ms=None,
                    selection_ms=None,
                    detokenize_ms=None,
                    commit_group=len(decoder.stats["rounds"]),
                    burst_tokens=len(emitted),
                )
                yield {
                    "kind": "output",
                    "text": text,
                    "token_id": token,
                    "output_tokens": len(ids),
                    "input_tokens": count,
                    "runtime_first_token_ms": times[0],
                }
            if emitted[-1] in eos and not decoder.fixed:
                finish = "stop"
                break
        status = "completed"
        proof["constraint_satisfied"] = trie.valid(ids) if trie else None
        if trie and not proof["constraint_satisfied"]:
            raise ValueError("L7_CONSTRAINT_DID_NOT_FINISH_AT_AN_ALLOWED_TERMINAL")
    except DecodingStopped as exc:
        finish = status = str(exc)
    except (ValueError, ContextLimitExceeded) as exc:
        finish = "rejected"
        result = {
            "request_error": {"code": "DECODING_PLAN_EXECUTION", "message": str(exc)}
        }
    finally:
        elapsed = (time.monotonic() - runtime_start) * 1000 if runtime_start else None
        if stream:
            stream.close()
        if decoder:
            proof["statistics"] = decoder.summary()
            proof["target_kv_before_cleanup"] = cache_inventory(decoder.cache)
            proof["draft_kv_before_cleanup"] = (
                cache_inventory(decoder.draft.cache) if decoder.draft else None
            )
        if session and decoder:
            session.boundary(
                "after_generation",
                cache=decoder.cache,
                step=len(ids) - 1,
                detailed=True,
            )
        memory = provider.memory()
        try:
            with measurement.phase("request_cleanup"):
                if decoder:
                    decoder.close()
                stream = None
                decoder = None
                gc.collect()
                torch.cuda.synchronize()
                if request.get("cleanup_policy", "empty_cache") == "empty_cache":
                    torch.cuda.empty_cache()
                cleanup = "succeeded"
                proof["cleanup"] = {
                    "target_request_cache_released": True,
                    "draft_request_cache_released": True,
                    "resident_draft_retained": draft_summary(provider).get("status")
                    == "loaded",
                }
        finally:
            proof.update(
                status="invoked" if status == "completed" else status,
                execution_status=status,
                cleanup_status=cleanup,
                input_token_sha256=input_hash,
                output_token_hash=fingerprint(ids),
                generated_tokens=len(ids),
                target_model_identity=provider.source_identity,
                tokenizer_identity=provider.tokenizer_identity,
                source_digest=provider.source_environment["source_digest"],
                run_id=request.get("capture_id"),
                epoch=request.get("expected_epoch"),
            )
            request["_decoding_receipt"] = proof
            if request.get("capture_dir"):
                atomic_json(
                    Path(request["capture_dir"]) / "decoding-receipt.json", proof
                )
            r = receipt(
                provider,
                request,
                opt,
                role="performance" if measurement.mode == "off" else "diagnostic",
                output_hash=fingerprint(ids),
                status=status,
                cleanup=cleanup,
                proof={
                    "input_token_sha256": input_hash,
                    "requested_token_limit": limit,
                    "generated_tokens": len(ids),
                    "output_tokens": len(ids),
                    "after_cleanup_allocator": provider.memory(),
                },
            )
            measurement.snapshot("after_cleanup", provider.memory)
            metrics = {
                "input_tokens": count,
                "output_tokens": len(ids),
                "generated_token_ids": ids,
                "input_token_sha256": input_hash,
                "runtime_first_token_ms": times[0] if times else None,
                "runtime_e2e_ms": elapsed,
                "output_tokens_per_s": len(ids) * 1000 / elapsed if elapsed else None,
                "decode_tail_tokens_per_s": (len(ids) - 1)
                * 1000
                / (times[-1] - times[0])
                if len(ids) > 1 and times[-1] > times[0]
                else None,
                "token_itl_ms": [b - a for a, b in zip(times, times[1:])],
                "memory": memory,
                "measurement_profile": "l7-verified-commit-host-monotonic-v1",
                "worker_request_total_ms": (time.monotonic() - start) * 1000,
                "worker_request_cpu_ms": (time.process_time() - cpu) * 1000,
                "observation": measurement.data,
                "decoding_receipt": proof,
                "optimization_receipt": r,
                "quality": "not_evaluated",
                "kv_inventory": proof.get("target_kv_before_cleanup"),
            }
            if session:
                session.emit("optimization_receipt", r)
                report = session.finish(status, cleanup)
                metrics["observation_bundle"] = {
                    "capture_status": report["capture_status"],
                    "counts": report["counts"],
                    "publication": report["publication"],
                    "problems": session.problems,
                }
            result = {
                "kind": "result",
                "text": text,
                "finish_reason": finish,
                "metrics": metrics,
                **(result or {}),
            }
    yield result
