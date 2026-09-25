"""Reviewed logical dependencies, separate from artifact facts and GPU scheduling."""
from importlib.metadata import distribution, PackageNotFoundError
import hashlib

# Whole implementation files inspected on the target, not merely model names.
IMPLEMENTATIONS = {
    "qwen2": ("qwen2", "99fa98c5676604cf6ef505892b70fda5c1c4cd835971459f38d61090cccab1e4"),
    "qwen3": ("qwen3", "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab"),
    "gemma3_text": ("gemma3", "5c3dab4bdd834084649c0cdbc0d03aa4090ddf5471649c3ee6b72b7feca2d6eb"),
}


def implementation(model_type):
    if model_type not in IMPLEMENTATIONS:
        return {"verified": False, "reason": "此模型族尚未建立经过核对的计算依赖适配器"}
    family, expected = IMPLEMENTATIONS[model_type]
    relative = f"transformers/models/{family}/modeling_{family}.py"
    try:
        dist = distribution("transformers")
        actual = hashlib.sha256(dist.locate_file(relative).read_bytes()).hexdigest()
    except (OSError, PackageNotFoundError):
        return {"verified": False, "reason": "本机缺少可核对的 Transformers 模型实现，仅展示制品元数据"}
    return {"verified": dist.version == "5.8.0" and actual == expected,
            "version": dist.version, "file": relative, "sha256": actual,
            "reason": None if dist.version == "5.8.0" and actual == expected else "安装的模型实现与已核对版本不同，仅展示制品结构"}


def graphs(model_type, config):
    gemma = model_type == "gemma3_text"
    qk_norm = model_type in {"qwen3", "gemma3_text"}

    def node(identity, label, col, row=1, module=None, kind="operation", note=""):
        return {"id": identity, "label": label, "column": col, "row": row,
                "module": module, "kind": kind, "note": note}

    def edge(source, target, kind="dependency"):
        return {"source": source, "target": target, "kind": kind}

    block = [node("x", "Block 输入 x", 0), node("r1", "残差 x（恒等路径）", 1, 0, kind="residual"),
             node("norm1", "输入 RMSNorm", 1, module="input_layernorm"),
             node("attn", "Self Attention ↗", 2, module="self_attn"),
             node("sum1", "残差相加", 4), node("r2", "残差 h（恒等路径）", 5, 0, kind="residual"),
             node("norm2", "FFN 前 RMSNorm", 5, module="pre_feedforward_layernorm" if gemma else "post_attention_layernorm"),
             node("mlp", "门控 MLP ↗", 6, module="mlp"), node("sum2", "残差相加 / Block 输出", 8)]
    block_edges = [edge("x", "norm1"), edge("x", "r1", "residual"), edge("norm1", "attn"),
                   edge("r1", "sum1", "residual"), edge("sum1", "r2", "residual"),
                   edge("sum1", "norm2"), edge("norm2", "mlp"), edge("r2", "sum2", "residual")]
    if gemma:
        block += [node("an", "Attention 后 RMSNorm", 3, module="post_attention_layernorm"),
                  node("mn", "FFN 后 RMSNorm", 7, module="post_feedforward_layernorm")]
        block_edges += [edge("attn", "an"), edge("an", "sum1"), edge("mlp", "mn"), edge("mn", "sum2")]
    else:
        block_edges += [edge("attn", "sum1"), edge("mlp", "sum2")]
    attention = [node("x", "归一化输入", 0),
        node("q", "Q Linear", 1, 0, "self_attn.q_proj"),
        node("k", "K Linear", 1, 1, "self_attn.k_proj"),
        node("v", "V Linear", 1, 2, "self_attn.v_proj"),
        node("qr", "Q RoPE / reshape", 3, 0), node("kr", "K RoPE / reshape", 3, 1),
        node("kv", "K/V Cache 更新或读取", 4, 2, kind="state",
             note="use_cache 时的逻辑状态依赖；不表示实测缓存命中或物理分页"),
        node("scores", "Q·Kᵀ / scaling", 5, 0, note="GQA 按 KV 组关联 Query heads；具体融合取决于 attention 后端"),
        node("mask", "Causal / 局部 mask", 6, 0), node("softmax", "Softmax", 7, 0),
        node("pv", "P·V / 合并 heads", 8, 1), node("o", "输出 Linear", 9, 1, "self_attn.o_proj")]
    ae = [edge("x", part) for part in ("q", "k", "v")]
    if qk_norm:
        attention += [node("qn", "Q RMSNorm", 2, 0, "self_attn.q_norm"), node("kn", "K RMSNorm", 2, 1, "self_attn.k_norm")]
        ae += [edge("q", "qn"), edge("qn", "qr"), edge("k", "kn"), edge("kn", "kr")]
    else:
        ae += [edge("q", "qr"), edge("k", "kr")]
    ae += [edge("qr", "scores"), edge("kr", "kv", "state"), edge("v", "kv", "state"),
           edge("kv", "scores", "state"), edge("scores", "mask"), edge("mask", "softmax"),
           edge("softmax", "pv"), edge("kv", "pv", "state"), edge("pv", "o")]
    activation = config.get("hidden_activation") or config.get("hidden_act") or ("gelu_pytorch_tanh" if gemma else "silu")
    mlp = [node("x", "FFN 输入", 0), node("gate", "Gate Linear", 1, 0, "mlp.gate_proj"),
           node("up", "Up Linear", 1, 2, "mlp.up_proj"), node("act", activation, 2, 0, "mlp.act_fn"),
           node("mul", "逐元素相乘", 3), node("down", "Down Linear", 4, 1, "mlp.down_proj")]
    me = [edge("x", "gate"), edge("x", "up"), edge("gate", "act"), edge("act", "mul"), edge("up", "mul"), edge("mul", "down")]
    return {"block": {"nodes": block, "edges": block_edges},
            "attention": {"nodes": attention, "edges": ae}, "mlp": {"nodes": mlp, "edges": me}}


def layer_kinds(config, count):
    if config.get("layer_types"):
        return config["layer_types"]
    if config.get("model_type") == "gemma3_text":
        pattern = config.get("sliding_window_pattern", 6)
        if type(pattern) is not int or pattern < 1:
            return ["unknown"] * count
        return ["full_attention" if (i + 1) % pattern == 0 else "sliding_attention" for i in range(count)]
    if config.get("use_sliding_window"):
        boundary = config.get("max_window_layers", count)
        return ["sliding_attention" if i >= boundary else "full_attention" for i in range(count)]
    return ["full_attention"] * count
