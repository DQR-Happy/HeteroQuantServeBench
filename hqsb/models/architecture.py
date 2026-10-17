"""Bounded config and safetensors-header inspection; no torch/model imports."""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import time

from hqsb.models.architecture_graphs import graphs, implementation, layer_kinds
from hqsb.models.architecture_structure import artifact_structure, semantic_family


def read_json(path, limit=2 * 1024**2):
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("架构元数据超过读取预算")
    return json.loads(data), hashlib.sha256(data).hexdigest()


def inspect_architecture(model_path):
    root = Path(os.path.expandvars(model_path)).expanduser().resolve()
    config, config_hash = read_json(root / "config.json")
    tensors, shards, issues = {}, [], []
    files = sorted(root.glob("*.safetensors"))
    if len(files) > 256:
        raise ValueError("权重分片数超过 256 的元数据预算")
    budget = 32 * 1024**2
    for path in files:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("Safetensors header 截断：" + path.name)
            length = struct.unpack("<Q", prefix)[0]
            if length > min(16 * 1024**2, budget):
                raise ValueError("Safetensors header 超过读取预算")
            data = stream.read(length)
            if len(data) != length:
                raise ValueError("Safetensors header 截断：" + path.name)
            budget -= length
        header = json.loads(data)
        for name, row in header.items():
            if name == "__metadata__":
                continue
            shape, offsets = row["shape"], row["data_offsets"]
            if not all(type(dim) is int and dim >= 0 for dim in shape) or len(offsets) != 2 or not 0 <= offsets[0] <= offsets[1] <= path.stat().st_size - 8 - length:
                raise ValueError("Safetensors shape / offsets 非法：" + name)
            if name in tensors:
                raise ValueError("跨分片出现重复 tensor：" + name)
            bits = {"F64":64,"F32":32,"F16":16,"BF16":16,"I64":64,"I32":32,"I16":16,
                    "I8":8,"U8":8,"BOOL":8,"F8_E4M3":8,"F8_E5M2":8}.get(row["dtype"])
            if bits is not None and offsets[1] - offsets[0] != (math.prod(shape) * bits + 7) // 8:
                raise ValueError("Safetensors 字节范围与 shape / dtype 不一致：" + name)
            tensors[name] = {"name": name, "shape": shape, "dtype": row["dtype"],
                             "stored_bytes": offsets[1] - offsets[0], "file": path.name}
        shards.append({"file": path.name, "header_sha256": hashlib.sha256(data).hexdigest(), "file_bytes": path.stat().st_size})
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index, _ = read_json(index_path, 8 * 1024**2)
        for name, filename in index.get("weight_map", {}).items():
            if name not in tensors or tensors[name]["file"] != filename:
                issues.append("索引中的 tensor 缺失或分片不匹配：" + name)
    if not files:
        issues.append("未发现 Safetensors；只能展示 config 声明，不能确认权重模块")
    model_type = config.get("model_type", "unknown")
    support = implementation(model_type)
    count = config.get("num_hidden_layers")
    if type(count) is not int or not 1 <= count <= 1024:
        count = 0
        issues.append("层数缺失或超过视图预算")
    if model_type in {"llama", "qwen2", "qwen3", "gemma3_text"} and count:
        hidden = config.get("hidden_size")
        heads, kv_heads = config.get("num_attention_heads"), config.get("num_key_value_heads")
        dim = config.get("head_dim") or (hidden // heads if type(hidden) is int and type(heads) is int and heads > 0 else None)
        intermediate = config.get("intermediate_size")
        if not all(type(value) is int and value > 0 for value in (hidden, heads, kv_heads, dim, intermediate)):
            issues.append("缺少有效的 Attention / MLP 维度配置")
        expected_shapes = ({"self_attn.q_proj.weight": [heads * dim, hidden],
            "self_attn.k_proj.weight": [kv_heads * dim, hidden], "self_attn.v_proj.weight": [kv_heads * dim, hidden],
            "self_attn.o_proj.weight": [hidden, heads * dim], "mlp.gate_proj.weight": [intermediate, hidden],
            "mlp.up_proj.weight": [intermediate, hidden], "mlp.down_proj.weight": [hidden, intermediate],
            "input_layernorm.weight": [hidden], "post_attention_layernorm.weight": [hidden]}
            if not issues else {})
        for i in range(count):
            prefix = f"model.layers.{i}."
            for suffix in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                           "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight"):
                if prefix + suffix not in tensors:
                    issues.append("缺少预期权重：" + prefix + suffix)
                elif suffix in expected_shapes and tensors[prefix + suffix]["shape"] != expected_shapes[suffix]:
                    issues.append("权重 shape 与 config 不一致：" + prefix + suffix)
            query = tensors.get(prefix + "self_attn.q_proj.weight")
            if query and (len(query["shape"]) != 2 or query["shape"][1] != hidden):
                issues.append("Q projection 与 config 隐藏维度不一致：" + str(i))
            if model_type in {"qwen3", "gemma3_text"}:
                for suffix in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
                    if prefix + suffix not in tensors:
                        issues.append("缺少 Q/K norm 权重：" + prefix + suffix)
            if model_type == "gemma3_text":
                for suffix in ("pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"):
                    if prefix + suffix not in tensors:
                        issues.append("缺少 Gemma norm 权重：" + prefix + suffix)
    for name in tensors:
        match = re.match(r"model\.layers\.(\d+)\.", name)
        if match and int(match[1]) >= count:
            issues.append("权重层号超过 config 层数：" + name)
    if config.get("quantization_config") or config.get("auto_map"):
        issues.append("量化或远程实现尚未建立计算依赖适配")
    if config.get("vision_config") or config.get("num_local_experts") or config.get("n_routed_experts"):
        issues.append("多模态或 MoE 尚未建立计算依赖适配")
    kinds = layer_kinds(config, count) if support["verified"] and count and not issues else []
    if kinds and len(kinds) != count:
        issues.append("layer_types 与层数不一致")
        kinds = []
    layers = []
    for i in range(count):
        entries = [row for name, row in tensors.items() if name.startswith(f"model.layers.{i}.")]
        layers.append({"index": i, "path": f"model.layers.{i}", "tensor_count": len(entries),
                       "stored_bytes": sum(row["stored_bytes"] for row in entries),
                       "attention_type": kinds[i] if kinds else "unknown"})
    visible_config = {key: config.get(key) for key in ("model_type", "architectures", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings", "hidden_act", "hidden_activation", "tie_word_embeddings", "sliding_window")}
    graph = graphs(model_type, config) if support["verified"] and count and not issues else None
    structure = artifact_structure(list(tensors.values()))
    semantic = semantic_family(config)
    identity = hashlib.sha256(json.dumps({"config": config_hash, "headers": shards, "implementation": support,
                                         "graphs": graph, "structure": structure, "semantic": semantic}, sort_keys=True).encode()).hexdigest()
    return {"schema_version": "1.0", "snapshot_sha256": identity, "observed_at": time.time(),
            "config": visible_config, "config_sha256": config_hash,
            "implementation": support, "status": "verified_static" if graph else "metadata_only", "issues": issues[:100],
            "layers": layers, "tensors": list(tensors.values()), "shards": shards,
            "stored_tensor_count": len(tensors), "stored_parameter_elements": sum(math.prod(row["shape"]) for row in tensors.values()),
            "stored_bytes": sum(row["stored_bytes"] for row in tensors.values()), "dtypes": dict(Counter(row["dtype"] for row in tensors.values())),
            "graphs": graph, "evidence_scope": "artifact containment from config + safetensors headers; optional reviewed static semantic diagrams; no forward execution or weight values loaded",
            "structure": structure, "semantic": semantic,
            "limitations": ["这是逻辑计算依赖图，不是 GPU 执行时间线；分支独立不等于 stream 或 kernel 同时执行。",
                "Safetensors header 哈希不等于完整权重校验；存储 dtype/字节不等于加载后的运行 dtype/显存。",
                "权重名称不能确定 forward；专用静态示意图要求实现指纹匹配，实际算子依赖需另行显式采集，不受模型族识别限制。",
                "Attention 展示数学子步骤；SDPA 或自定义融合实现可能合并节点。无实际 trace 不推断融合、Tensor Core、SM 映射或 Cache 命中。",
                "层间存在依赖；标准自回归 decode 的下一 token 依赖上一步结果。Prefill 中 token/head 并行属于逻辑机会，实际调度另需运行证据。"]}
