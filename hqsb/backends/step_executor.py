"""Resident-model step executor. Tensor/cache layout changes stay here."""
import inspect
import math
import time

from hqsb.core.contracts.scheduling import StepResult
from hqsb.backends.device_kv import DenseKV


def capabilities(provider):
    cfg = provider.model.config.get_text_config(decoder=True)
    reasons = []
    if getattr(getattr(provider, "precision_plan", None), "kv_bits", None):
        reasons.append("QUANTIZED_KV_STEP_ADAPTER_NOT_VERIFIED")
    types = getattr(cfg, "layer_types", None)
    if getattr(cfg, "sliding_window", None) or getattr(cfg, "attention_chunk_size", None) or (types and any(x != "full_attention" for x in types)):
        reasons.append("SLIDING_OR_HYBRID_CACHE_NOT_ADAPTED")
    # This is a verified execution adapter boundary, not an architecture-reader gate.
    # Full attention alone does not establish positional/mask/cache semantics.
    if type(provider.model).__name__ not in {"LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"}:
        reasons.append("FORWARD_MASK_POSITION_ADAPTER_NOT_VERIFIED")
    if "position_ids" not in inspect.signature(provider.model.forward).parameters:
        reasons.append("EXPLICIT_POSITION_IDS_REQUIRED")
    profile = provider.kernel_session.profile["profile_id"] if provider.kernel_session else "framework-native"
    if profile != "framework-native":
        reasons.append("CUSTOM_KERNEL_PROFILE_NOT_BATCH_VERIFIED")
    return {"schema_version": "l8-v1", "supported": not reasons, "reasons": reasons,
            "executor": "dense-full-attention-v1", "kv_layout": "independent_compact_rows",
            "paged_kv": False, "model_class": type(provider.model).__name__,
            "layer_types": types, "policies": ["fcfs", "priority_aging", "tenant_fair"],
            "modes": ["serial", "static", "continuous"], "sampling": "greedy",
            "limits": ["single GPU, one forward in flight", "no prefix sharing/offload/KV quantization",
                       "dense pack/scatter copies and padding are measured", "no per-request GPU L2/page/power attribution"]}


class StepExecutor:
    def __init__(self, provider, max_padded_positions, max_temporary_bytes=512 * 1024**2):
        import torch

        self.torch, self.provider = torch, provider
        self.kv = DenseKV(torch)
        self.max_padded_positions = max_padded_positions
        self.context_limit = provider.config["context_limit"]
        self.max_temporary_bytes = max_temporary_bytes
        config = provider.model.config.get_text_config(decoder=True)
        self.kv_token_bytes = (2 * config.num_hidden_layers * getattr(config, "num_key_value_heads", config.num_attention_heads)
                               * getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
                               * provider.model.dtype.itemsize)
        self.capability = capabilities(provider)
        if not self.capability["supported"]:
            raise ValueError("SCHEDULING_UNSUPPORTED: " + ",".join(self.capability["reasons"]))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def prepare(self, steps):
        """Conservative estimate, not a promise about opaque kernel workspace."""
        batch, past, query = len(steps), max(s.start for s in steps), max(len(s.token_ids) for s in steps)
        config = self.provider.model.config.get_text_config(decoder=True)
        # Packed cache, updated cache, compact copies, mask and major activations.
        estimate = (3 * batch * (past + query) * self.kv_token_bytes
                    + batch * query * (past + query) * (self.provider.model.dtype.itemsize + 2)
                    + batch * query * (config.hidden_size * 16 + config.vocab_size) * 4)
        free, _ = self.torch.cuda.mem_get_info()
        available = free + self.torch.cuda.memory_reserved() - self.torch.cuda.memory_allocated()
        if estimate > self.max_temporary_bytes:
            return "TEMPORARY_MEMORY_ESTIMATE_BUDGET"
        if estimate + 256 * 1024**2 > available:
            return "DEVICE_HEADROOM_ESTIMATE"
        return None

    def execute(self, steps, forward_id):
        torch = self.torch
        start = time.monotonic()
        identities = [s.request_id for s in steps]
        lengths = [s.start for s in steps]
        additions = [len(s.token_ids) for s in steps]
        past, query, batch = max(lengths), max(additions), len(steps)
        if batch * (past + query) > self.max_padded_positions:
            raise ValueError("PADDED_POSITION_BUDGET")
        before = self.kv.bytes()
        with torch.inference_mode():
            cache, packed_bytes = self.kv.pack(identities, lengths)
            inputs = torch.zeros((batch, query), dtype=torch.long, device="cuda")
            positions = torch.zeros_like(inputs)
            valid = torch.zeros((batch, past + query), dtype=torch.bool, device="cuda")
            for row, step in enumerate(steps):
                count, added = lengths[row], additions[row]
                inputs[row, query - added:] = torch.tensor(step.token_ids, dtype=torch.long, device="cuda")
                positions[row, query - added:] = torch.arange(count, count + added, device="cuda")
                valid[row, past - count:past] = True
                valid[row, past + query - added:] = True
            # Explicit 4-D mask avoids inference of packed sequences from reset
            # position_ids and gives the same causal relation as isolated rows.
            causal = torch.arange(past + query, device="cuda")[None, :] <= (past + torch.arange(query, device="cuda"))[:, None]
            allowed = valid[:, None, None, :] & causal[None, None, :, :]
            mask = torch.zeros(allowed.shape, device="cuda", dtype=self.provider.model.dtype)
            mask.masked_fill_(~allowed, torch.finfo(mask.dtype).min)
            assembled = time.monotonic()
            gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            if self.provider.kernel_session:
                self.provider.kernel_session.set_phase("decode" if all(s.phase == "decode" for s in steps) else "prefill")
            gpu_start.record()
            output = self.provider.model(input_ids=inputs, attention_mask=mask, position_ids=positions,
                                         past_key_values=cache, use_cache=True)
            gpu_end.record()
            # One bounded D2H selection for the whole batch; this establishes
            # device completion before publishing tokens or releasing storage.
            logits = output.logits[:, -1, :].float()
            chosen = logits.argmax(-1)
            top_two = logits.topk(2, dim=-1).values
            margin = top_two[:, 0] - top_two[:, 1]
            logprobs = logits.log_softmax(-1).gather(-1, chosen[:, None]).squeeze(-1)
            decisions = torch.stack((chosen.float(), logprobs, margin), dim=-1).tolist()
            if not all(math.isfinite(v[1]) for v in decisions):
                raise ValueError("NONFINITE_BATCH_LOGPROBS")
            tokens = tuple(int(v[0]) for v in decisions)
            forward_end = time.monotonic()
            scatter_bytes = self.kv.commit(output.past_key_values, identities, lengths, additions)
            torch.cuda.synchronize()
            end = time.monotonic()
            receipt = {"forward_id": forward_id, "members": identities,
                       "input_shape": [batch, query], "past_width": past,
                       "valid_past": lengths, "valid_inputs": additions,
                       "logical_positions": sum(additions), "padding_positions": batch * query - sum(additions),
                       "kv_padding_positions": batch * past - sum(lengths),
                       "pack_copy_bytes": packed_bytes, "scatter_copy_bytes": scatter_bytes,
                       "resident_kv_before_bytes": before, "resident_kv_after_bytes": self.kv.bytes(),
                       "kv_token_bytes_geometry_estimate": self.kv_token_bytes,
                       "assembly_host_ms": (assembled - start) * 1000,
                       "forward_selection_host_ms": (forward_end - assembled) * 1000,
                       "scatter_completion_host_ms": (end - forward_end) * 1000,
                       "gpu_forward_ms": gpu_start.elapsed_time(gpu_end),
                       "completion": "cuda_synchronized", "memory": self.provider.memory(),
                       "selected_logprobs": [v[1] for v in decisions],
                       "top2_logit_margins": [v[2] for v in decisions],
                       "phases": [s.phase for s in steps]}
            del output, cache, inputs, mask, valid, positions, allowed, logits, chosen, logprobs, top_two, margin
        return StepResult(tokens=tokens, receipt=receipt)

    def release(self, identity):
        self.kv.release(identity)

    def quiesce(self):
        self.torch.cuda.synchronize()

    def close(self):
        import gc

        self.torch.cuda.synchronize()
        self.kv.clear()
        gc.collect()
        self.torch.cuda.empty_cache()
        return {"device_kv_rows": len(self.kv.rows), "resident_kv_bytes": self.kv.bytes(),
                "memory": self.provider.memory(), "completion": "cuda_synchronized"}
