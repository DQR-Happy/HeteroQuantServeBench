"""L9 native model loading, explicit placement and bounded greedy execution."""

import os
import time
from pathlib import Path
from hqsb.core.optimization import fingerprint


def load(config, plan, rank):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from hqsb.models.manifest import verify_or_raise

    root = Path(os.path.expandvars(config["model_path"])).expanduser()
    verify_or_raise(
        str(root),
        os.path.expandvars(os.path.expanduser(config["manifest"])),
        strict_extra=True,
        allow_extra=tuple(config.get("manifest_allow_extra", [])),
    )
    cfg = AutoConfig.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    if getattr(cfg, "quantization_config", None):
        raise ValueError("L9_NATIVE_UNQUANTIZED_SOURCE_REQUIRED")
    if plan.strategy == "tp" and not getattr(cfg, "base_model_tp_plan", None):
        raise ValueError("L9_MODEL_HAS_NO_NATIVE_TP_PLAN")
    with torch.device("meta"):
        shape_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=False)
    full_shapes = {name: list(p.shape) for name, p in shape_model.named_parameters()}
    del shape_model
    if plan.strategy == "prefill_decode" and cfg.model_type not in {
        "llama",
        "qwen2",
        "qwen3",
    }:
        raise ValueError("L9_KV_HANDOFF_ADAPTER_REQUIRES_DENSE_LLAMA_QWEN2_QWEN3")
    if plan.strategy == "prefill_decode" and (
        getattr(cfg, "sliding_window", None)
        or any(t != "full_attention" for t in (getattr(cfg, "layer_types", None) or []))
    ):
        raise ValueError(
            "L9_KV_HANDOFF_REQUIRES_FULL_ATTENTION_NO_SLIDING_OR_HYBRID_STATE"
        )
    device = "cpu" if plan.device_type == "cpu" else f"cuda:{rank}"
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "dtype": getattr(torch, plan.precision),
        "attn_implementation": config["attention"],
    }
    placement = None
    if plan.strategy == "tp":
        kwargs["tp_plan"] = "auto"
    elif plan.strategy == "layer_shard":
        if cfg.model_type not in {"llama", "qwen2", "qwen3", "gemma3_text"}:
            raise ValueError("L9_LAYER_PLACEMENT_ADAPTER_UNAVAILABLE")
        n = len(plan.devices)
        if cfg.num_hidden_layers < n:
            raise ValueError("L9_MORE_DEVICES_THAN_LAYERS")
        placement = {
            "model.embed_tokens": 0,
            "model.norm": 0,
            "model.rotary_emb": 0,
            "lm_head": 0,
        }
        for i in range(cfg.num_hidden_layers):
            placement[f"model.layers.{i}"] = min(n - 1, i * n // cfg.num_hidden_layers)
        kwargs["device_map"] = placement
        device = "cuda:0"
    else:
        # Ordinary, swappable CPU staging avoids the pinned-memory peak of
        # direct CUDA checkpoint dispatch on Jetson's shared DRAM. This is
        # an explicit path, not an OOM retry or partial CPU offload fallback.
        kwargs["device_map"] = "cpu"
    model = AutoModelForCausalLM.from_pretrained(root, **kwargs).eval()
    if plan.strategy not in {"tp", "layer_shard"} and plan.device_type == "cuda":
        model.to(device)
        if {str(p.device) for p in model.parameters()} != {device}:
            raise ValueError("L9_FULL_DEVICE_PLACEMENT_REQUIRED")
    model._l9_load_strategy = (
        "native_tp_shard_load"
        if plan.strategy == "tp"
        else "explicit_layer_device_map"
        if plan.strategy == "layer_shard"
        else "cpu_staged_then_full_device"
        if plan.device_type == "cuda"
        else "cpu_reference"
    )
    model._l9_full_shapes = full_shapes
    tok = AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    return model, tok, device, placement


def parameters(model):
    entries = []
    count = 0
    for name, p in model.named_parameters():
        local = p.to_local() if hasattr(p, "to_local") else p
        size = local.numel() * local.element_size()
        count += size
        if len(entries) < 256:
            entries.append(
                {
                    "name": name,
                    "global_shape": model._l9_full_shapes.get(name, list(p.shape)),
                    "local_shape": list(local.shape),
                    "bytes": size,
                    "device": str(local.device),
                    "placements": [str(x) for x in getattr(p, "placements", ())],
                }
            )
    return {
        "local_parameter_bytes": count,
        "tensors": entries,
        "truncated": len(list(model.named_parameters())) > 256,
    }


def kv_summary(cache):
    rows = []
    if cache is not None:
        for i, layer in enumerate(cache.layers):
            k, v = layer.keys, layer.values
            if k is not None:
                rows.append(
                    {
                        "layer": i,
                        "key_shape": list(k.shape),
                        "value_shape": list(v.shape),
                        "bytes": k.numel() * k.element_size()
                        + v.numel() * v.element_size(),
                        "device": str(k.device),
                    }
                )
    return {"layers": rows, "bytes": sum(r["bytes"] for r in rows)}


def prefill(model, ids, device):
    import torch
    from transformers import DynamicCache

    x = torch.tensor([ids], device=device)
    out = model(
        input_ids=x, past_key_values=DynamicCache(config=model.config), use_cache=True
    )
    return out.past_key_values, int(out.logits[0, -1].argmax().item())


def generate(model, tok, ids, device, plan, cache=None, first=None):
    import torch

    start = time.monotonic()
    times = []
    tokens = []
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos] if eos is not None else [])
    with torch.inference_mode():
        if cache is None:
            cache, first = prefill(model, ids, device)
        for i in range(plan.max_output_tokens):
            tokens.append(first)
            times.append((time.monotonic() - start) * 1000)
            if first in eos and not plan.fixed_output_length:
                break
            if i + 1 < plan.max_output_tokens:
                out = model(
                    input_ids=torch.tensor([[first]], device=device),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                first = int(out.logits[0, -1].argmax().item())
        if plan.device_type == "cuda":
            for i in (
                range(len(plan.devices))
                if plan.strategy == "layer_shard"
                else [torch.cuda.current_device()]
            ):
                torch.cuda.synchronize(i)
        ms = (time.monotonic() - start) * 1000
        inventory = kv_summary(cache)
    return {
        "token_ids": tokens,
        "output": tok.decode(tokens, skip_special_tokens=True),
        "input_token_hash": fingerprint(ids),
        "input_tokens": len(ids),
        "first_token_ms": times[0],
        "generation_ms": ms,
        "token_s": len(tokens) * 1000 / ms,
        "kv": inventory,
    }, cache
