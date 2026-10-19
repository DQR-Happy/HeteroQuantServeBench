"""Bounded teacher-forced corpus scoring on the resident model and actual KV."""

import gc
import math
import time
from pathlib import Path

from hqsb.core.corpus_benchmark import aggregate
from hqsb.core.optimization import fingerprint
from hqsb.observation.storage import atomic_json
from hqsb.backends.runtime_session import set_phase


class CorpusStopped(Exception):
    pass


def evaluate(provider, request, cancel, plan):
    import torch

    from hqsb.backends.optimization_receipt import receipt, resolve_request
    from hqsb.backends.precision_runtime import new_cache, reset_calls

    if provider.config["context_limit"] < plan["max_tokens"]:
        raise ValueError("QUALITY_BENCHMARK_CONTEXT_TOO_SMALL")
    mode = plan["forward_mode"]
    if mode == "full_prefill" and getattr(
        getattr(provider, "precision_plan", None), "kv_bits", None
    ):
        raise ValueError(
            "QUALITY_FULL_PREFILL_BYPASSES_QUANTIZED_KV:use_incremental_decode"
        )
    request, optimization = resolve_request(provider.config, request)
    directory = Path(request["capture_dir"])
    deadline = time.monotonic() + request["remaining_ms"] / 1000
    kernels = getattr(provider, "kernel_session", None)
    if kernels:
        kernels.reset()
    reset_calls(provider)
    result = {
        "schema_version": "1.0.0",
        "run_id": request["capture_id"],
        "status": "partial",
        "measurement_role": "quality_not_performance",
        "score_mode": "corpus_nll",
        "benchmark_id": plan["benchmark_id"],
        "dataset_sha256": plan["benchmark_sha256"],
        "protocol": {
            "name": "hqs-bounded-token-ppl-v1",
            "forward_mode": mode,
            "max_tokens": plan["max_tokens"],
            "text_selection": "prefix_tokens",
            "add_special_tokens": False,
            "chat_template": False,
            "context": "reset_per_passage_first_token_not_scored",
            "aggregation": "exp(sum_nll/sum_scored_tokens)",
            "log_softmax_dtype": "float32",
            "kv_measured": mode == "incremental_decode",
        },
        "samples": [],
        "positions": [],
        "task_accuracy": None,
        "limitations": [
            "fixed English subset, not full WikiText-2 or task accuracy",
            "not comparable across tokenizers or context protocols",
        ],
    }
    finish, execution = "length", "failed"
    cache = output = current = mask = rows = None
    atomic_json(
        directory / "quality-pending.json",
        {"status": "running", "run_id": request["capture_id"]},
    )

    def check():
        if cancel.is_set() or time.monotonic() >= deadline:
            raise CorpusStopped()

    try:
        with torch.inference_mode():
            for sample_index, sample in enumerate(plan["samples"]):
                check()
                original = provider.tokenizer.encode(
                    sample["text"], add_special_tokens=False
                )
                tokens = original[: plan["max_tokens"]]
                if len(tokens) < 2:
                    raise ValueError("QUALITY_BENCHMARK_TOKENIZATION_TOO_SHORT")
                losses = []
                if mode == "full_prefill":
                    set_phase(provider, "prefill")
                    if kernels:
                        kernels.set_phase("prefill")
                    current = torch.tensor(
                        [tokens[:-1]], dtype=torch.long, device="cuda:0"
                    )
                    output = provider.model(
                        input_ids=current,
                        attention_mask=torch.ones_like(current),
                        use_cache=False,
                    )
                    rows = output.logits[0].float()
                    if not bool(torch.isfinite(rows).all()):
                        raise ValueError("QUALITY_NONFINITE_LOGITS")
                    targets = torch.tensor(tokens[1:], device=rows.device)
                    losses = (
                        torch.nn.functional.cross_entropy(
                            rows, targets, reduction="none"
                        )
                        .double()
                        .cpu()
                        .tolist()
                    )
                    output = rows = None
                else:
                    cache = new_cache(provider)
                    for position, token in enumerate(tokens[:-1]):
                        check()
                        set_phase(provider, "prefill" if position == 0 else "decode")
                        if kernels:
                            kernels.set_phase("prefill" if position == 0 else "decode")
                        current = torch.tensor(
                            [[token]], dtype=torch.long, device="cuda:0"
                        )
                        mask = torch.ones(
                            (1, position + 1), dtype=torch.long, device="cuda:0"
                        )
                        output = provider.model(
                            input_ids=current,
                            attention_mask=mask,
                            past_key_values=cache,
                            use_cache=True,
                        )
                        cache = output.past_key_values
                        row = output.logits[0, -1].float()
                        if not bool(torch.isfinite(row).all()):
                            raise ValueError("QUALITY_NONFINITE_LOGITS")
                        losses.append(
                            float(
                                -torch.log_softmax(row, dim=-1)[
                                    tokens[position + 1]
                                ].item()
                            )
                        )
                        output = row = None
                        if position % 16 == 15:
                            yield {
                                "kind": "progress",
                                "stage": "corpus_ppl",
                                "completed_samples": sample_index,
                                "scored_tokens": sum(
                                    s["scored_tokens"] for s in result["samples"]
                                )
                                + len(losses),
                            }
                if not all(math.isfinite(loss) and loss >= 0 for loss in losses):
                    raise ValueError("QUALITY_NONFINITE_NLL")
                result["samples"].append(
                    {
                        "id": sample["id"],
                        "slice": sample["slice"],
                        "text_sha256": fingerprint(" ".join(sample["text"].split())),
                        "scored_text_sha256": fingerprint(
                            " ".join(provider.tokenizer.decode(tokens).split())
                        ),
                        "input_sha256": fingerprint(tokens),
                        "original_tokens": len(original),
                        "input_tokens": len(tokens),
                        "prefix_truncated": len(original) > len(tokens),
                        "scored_tokens": len(losses),
                        "nll_sum": math.fsum(losses),
                        "token_nll": losses,
                    }
                )
                cache = None
                yield {
                    "kind": "progress",
                    "stage": "corpus_ppl",
                    "completed_samples": sample_index + 1,
                    "total_samples": len(plan["samples"]),
                }
            result.update(status="complete", **aggregate(result["samples"]))
            execution = "completed"
    except CorpusStopped:
        finish = "cancelled" if cancel.is_set() else "timed_out"
        execution = finish
    finally:
        cache = output = current = mask = rows = None
        gc.collect()
        if request["cleanup_policy"] == "empty_cache":
            torch.cuda.empty_cache()
        applied = receipt(
            provider,
            request,
            optimization,
            role="quality",
            output_hash=fingerprint(result["samples"]),
            status=execution,
            cleanup="succeeded",
            proof={
                "score_mode": "corpus_nll",
                "benchmark_id": plan["benchmark_id"],
                "scored_positions": sum(s["scored_tokens"] for s in result["samples"]),
                "full_vocab_logits_saved": False,
                "after_cleanup_allocator": provider.memory(),
            },
        )
        atomic_json(directory / "quality-result.json", result)
        atomic_json(
            directory / "quality-pending.json",
            {
                "status": result["status"],
                "execution_status": execution,
                "run_id": request["capture_id"],
            },
        )
    yield {
        "kind": "result",
        "text": "",
        "finish_reason": finish,
        "metrics": {
            "measurement_profile": "corpus-ppl-not-performance-v1",
            "measurement_role": "quality",
            "optimization_receipt": applied,
            "quality_evaluation": {
                "status": result["status"],
                "score_mode": "corpus_nll",
                "benchmark_id": plan["benchmark_id"],
                "dataset_sha256": result["dataset_sha256"],
                **{
                    key: result.get(key)
                    for key in (
                        "mean_nll",
                        "perplexity",
                        "perplexity_overflow",
                        "scored_tokens",
                        "document_count",
                    )
                },
                "task_accuracy": None,
                "protocol": result["protocol"],
            },
        },
    }
