"""Bounded, separate teacher-forced and free-generation quality execution.

Full-vocabulary logits are CPU float32 files, one position at a time. No
second model, retained activation hooks, or performance claim lives here.
"""

import gc
import math
import time
from pathlib import Path

from hqsb.backends.optimization_receipt import receipt, resolve_request
from hqsb.core.optimization import fingerprint, validate_quality_plan
from hqsb.observation.storage import atomic_json, sha256


class QualityCancelled(Exception):
    pass


def evaluate(provider, request, cancel):
    import torch

    if provider.model is None:
        raise RuntimeError("QUALITY_MODEL_NOT_LOADED")
    plan = validate_quality_plan(request["quality_plan"])
    request, optimization = resolve_request(provider.config, request)
    kernels = getattr(provider, "kernel_session", None)
    if kernels:
        kernels.reset()
    from hqsb.backends.precision_runtime import reset_calls
    reset_calls(provider)
    mode = plan.get("forward_mode", "incremental_decode")
    generation_limit = optimization["resolved"]["output_budget"]
    if not 1 <= generation_limit <= 32:
        raise ValueError("QUALITY_GENERATION_LIMIT")
    directory = Path(request["capture_dir"])
    deadline = time.monotonic() + request["remaining_ms"] / 1000
    encoded = [provider.tokenizer.encode(sample["text"], add_special_tokens=False) for sample in plan["samples"]]
    if any(not 2 <= len(tokens) <= plan["max_tokens"] for tokens in encoded):
        raise ValueError("QUALITY_TOKEN_BOUND_EXCEEDED:no_silent_truncation")
    required = sum(len(tokens) - 1 for tokens in encoded) * int(provider.model.config.vocab_size) * 4
    if required > plan["max_bytes"]:
        raise ValueError("QUALITY_LOGITS_BUDGET_EXCEEDED")
    result = {"schema_version": "1.0.0", "run_id": request["capture_id"],
              "status": "partial", "measurement_role": "quality_not_performance",
              "dataset_sha256": fingerprint(plan["samples"]),
              "protocol": {"name": "raw-full-prefill-v1" if mode == "full_prefill" else "raw-token-next-token-v1", "max_tokens": plan["max_tokens"],
                           "generation_tokens": generation_limit, "top_k": plan["top_k"], "dtype": "little_endian_float32"},
              "samples": [], "positions": [], "max_bytes": plan["max_bytes"]}
    atomic_json(directory / "quality-pending.json", {"status": "running", "run_id": request["capture_id"]})
    cache = output = current = mask = full_output = None
    bytes_written, losses = 0, []
    finish, execution, cleanup = "length", "failed", "unknown"

    def check():
        if cancel.is_set() or time.monotonic() >= deadline:
            raise QualityCancelled()

    try:
        with torch.inference_mode():
            for sample_index, (sample, tokens) in enumerate(zip(plan["samples"], encoded)):
                from hqsb.backends.precision_runtime import new_cache
                cache = new_cache(provider)
                sample_id = sample.get("id", str(sample_index))
                if mode == "full_prefill":
                    check()
                    current = torch.tensor([tokens[:-1]], dtype=torch.long, device="cuda:0")
                    mask = torch.ones_like(current)
                    if kernels:
                        kernels.set_phase("prefill")
                    full_output = provider.model(input_ids=current, attention_mask=mask, use_cache=False)
                for position, token in enumerate(tokens[:-1]):
                    check()
                    if mode == "incremental_decode":
                        current = torch.tensor([[token]], dtype=torch.long, device="cuda:0")
                        mask = torch.ones((1, position + 1), dtype=torch.long, device="cuda:0")
                        if kernels:
                            kernels.set_phase("prefill" if position == 0 else "decode")
                        output = provider.model(input_ids=current, attention_mask=mask, past_key_values=cache, use_cache=True)
                        cache = output.past_key_values
                    row = (full_output.logits[0, position] if full_output is not None else output.logits[0, -1]).float().cpu()
                    if not bool(torch.isfinite(row).all()):
                        raise ValueError("QUALITY_NONFINITE_LOGITS")
                    raw = row.numpy().astype("<f4", copy=False).tobytes()
                    if bytes_written + len(raw) > plan["max_bytes"]:
                        raise ValueError("QUALITY_LOGITS_BUDGET_EXCEEDED")
                    name = f"quality-logits-{len(result['positions']):04d}.f32"
                    path = directory / name
                    with path.open("xb") as stream:
                        stream.write(raw)
                    nll = float(-torch.log_softmax(row, dim=-1)[tokens[position + 1]].item())
                    losses.append(nll)
                    result["positions"].append({"sample_id": sample_id, "slice": sample.get("slice", "all"),
                                                "position": position, "path": name, "bytes": len(raw),
                                                "sha256": sha256(path), "vocab_size": len(row), "nll": nll})
                    bytes_written += len(raw)
                    output = row = raw = None
                # A fresh cache makes free generation independent of teacher forcing.
                cache = new_cache(provider)
                full_output = None
                generated = []
                current = torch.tensor([tokens], dtype=torch.long, device="cuda:0")
                mask = torch.ones_like(current)
                eos = provider.model.generation_config.eos_token_id
                eos_ids = set(eos if isinstance(eos, list) else [eos])
                for step in range(generation_limit):
                    check()
                    if kernels:
                        kernels.set_phase("prefill" if step == 0 else "decode")
                    output = provider.model(input_ids=current, attention_mask=mask, past_key_values=cache, use_cache=True)
                    if not bool(torch.isfinite(output.logits[:, -1, :]).all()):
                        raise ValueError("QUALITY_NONFINITE_GENERATION_LOGITS")
                    token = int(output.logits[:, -1, :].argmax(-1).item())
                    cache = output.past_key_values
                    output = None
                    generated.append(token)
                    if token in eos_ids:
                        break
                    current = torch.tensor([[token]], device="cuda:0", dtype=torch.long)
                    mask = torch.cat((mask, mask.new_ones((1, 1))), dim=1)
                text = provider.tokenizer.decode(generated, skip_special_tokens=True)
                result["samples"].append({"id": sample_id, "slice": sample.get("slice", "all"),
                                           "text_sha256": fingerprint(" ".join(sample["text"].split())),
                                           "input_sha256": fingerprint(tokens), "input_tokens": len(tokens),
                                           "output_tokens": len(generated), "output_token_sha256": fingerprint(generated),
                                           "exact_match": text.strip() == sample["expected"].strip() if "expected" in sample else None})
                cache = None
                yield {"kind": "progress", "stage": "quality", "completed_samples": sample_index + 1}
        result.update(status="complete", scored_tokens=len(losses), mean_nll=sum(losses) / len(losses),
                      perplexity=math.exp(sum(losses) / len(losses)), bytes_written=bytes_written)
        scored = [sample["exact_match"] for sample in result["samples"] if sample["exact_match"] is not None]
        result["task_accuracy"] = sum(scored) / len(scored) if scored else None
        result["task_scored_samples"] = len(scored)
        execution = "completed"
    except QualityCancelled:
        finish = "cancelled" if cancel.is_set() else "timed_out"
        execution = finish
    finally:
        cache = output = current = mask = full_output = None
        gc.collect()
        if request["cleanup_policy"] == "empty_cache":
            torch.cuda.empty_cache()
        cleanup = "succeeded"
        applied = receipt(provider, request, optimization, role="quality", output_hash=fingerprint(result["samples"]),
                          status=execution, cleanup=cleanup,
                          proof={"scored_positions": len(result["positions"]), "generation_limit": generation_limit,
                                 "cleanup_call": "torch.cuda.empty_cache" if request["cleanup_policy"] == "empty_cache" else "empty_cache_not_called",
                                 "after_cleanup_allocator": provider.memory()})
        atomic_json(directory / "quality-result.json", result)
        atomic_json(directory / "quality-pending.json", {"status": result["status"], "execution_status": execution,
                                                          "run_id": request["capture_id"]})
    yield {"kind": "result", "text": "", "finish_reason": finish,
           "metrics": {"measurement_profile": "quality-evaluation-not-performance-v1", "measurement_role": "quality",
                       "optimization_receipt": applied, "quality_evaluation": {
                           "status": result["status"], "dataset_sha256": result["dataset_sha256"],
                           "perplexity": result.get("perplexity"), "task_accuracy": result.get("task_accuracy"),
                           "scored_tokens": len(result["positions"])}}}
