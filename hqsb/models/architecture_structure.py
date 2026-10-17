"""Family-independent artifact containment and optional semantic annotations.

Containment is derived from actual tensor paths, never presented as forward edges.
Family names annotate evidence; they never admit or reject structural inspection.
"""

FAMILIES = {
    "Llama": ("llama", "llama4", "llama4_text", "mllama", "diffllama"),
    "Qwen": ("qwen2", "qwen2_moe", "qwen2_vl", "qwen2_5_vl", "qwen3", "qwen3_moe", "qwen3_next", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"),
    "Gemma": ("gemma", "gemma2", "gemma3", "gemma3_text", "gemma3n", "gemma3n_text", "gemma4", "gemma4_text", "gemma4_assistant", "recurrent_gemma", "vaultgemma"),
    "DeepSeek": ("deepseek_v2", "deepseek_v3", "deepseek_v32", "deepseek_v4"),
    "GLM": ("glm", "glm4", "glm4_moe", "glm4_moe_lite", "glm4v", "glm4v_moe", "glm_moe_dsa"),
    "Mistral / Mixtral": ("mistral", "mixtral", "mistral3", "ministral", "ministral3"),
    "Phi": ("phi", "phi3", "phi4_multimodal", "phimoe"),
    "GPT-OSS": ("gpt_oss",),
    "MiniMax": ("minimax", "minimax_m2"),
    "InternLM": ("internlm", "internlm2", "internlm3"),
    "Falcon": ("falcon", "falcon_mamba", "falcon_h1"),
    "OLMo": ("olmo", "olmo2", "olmo3", "olmoe"),
    "GPT / OPT / Bloom": ("gpt2", "gpt_neox", "gptj", "opt", "bloom"),
}


def semantic_family(config):
    kind = config.get("model_type", "unknown")
    family = next((label for label, types in FAMILIES.items() if kind in types), None)
    return {"family": family or kind, "recognized": family is not None,
            "declared_features": {key: config[key] for key in (
                "layer_types", "sliding_window", "q_lora_rank", "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim",
                "num_local_experts", "n_routed_experts", "num_experts", "num_experts_per_tok", "n_shared_experts",
                "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim") if key in config},
            "model_type": kind, "source": "config.model_type",
            "scope": "semantic_annotation_only_not_execution_or_kernel_certification",
            "coverage": [{"family": label, "model_types": list(types)} for label, types in FAMILIES.items()]}


def module_role(path, class_name=""):
    value = (path.rsplit(".", 1)[-1] + " " + class_name).lower()
    if "attention" in value or "self_attn" in value or value.startswith("attn "):
        return "attention"
    if "expert" in value or "moe" in value:
        return "experts"
    if "mlp" in value or "feedforward" in value or "feed_forward" in value or value.startswith("ffn "):
        return "mlp"
    if "norm" in value:
        return "normalization"
    if "embed" in value or value.startswith("wte "):
        return "embedding"
    if "linear" in value or any(value.startswith(name + " ") for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head")):
        return "linear"
    return "module"


def artifact_structure(tensors, limit=20000):
    modules = {"": {"path": "", "parent": None, "role": "root", "tensor_count": 0, "stored_bytes": 0}}
    truncated = False
    for tensor in tensors:
        path = tensor["name"].rsplit(".", 1)[0] if "." in tensor["name"] else ""
        parts = path.split(".") if path else []
        for end in range(1, len(parts) + 1):
            name = ".".join(parts[:end])
            if name not in modules:
                if len(modules) >= limit:
                    truncated = True
                    continue
                modules[name] = {"path": name, "parent": ".".join(parts[:end-1]),
                                 "role": module_role(name), "tensor_count": 0, "stored_bytes": 0}
        if path in modules:
            modules[path]["tensor_count"] += 1
            modules[path]["stored_bytes"] += tensor["stored_bytes"]
    return {"status": "partial" if truncated else "complete", "source": "safetensors_tensor_paths",
            "modules": list(modules.values()), "limit": limit,
            "scope": "weight_path_containment_not_forward_dependency_or_runtime_module_classes"}
