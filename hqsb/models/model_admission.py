"""Conservative estimates for one two-byte floating-point CUDA deployment."""
from __future__ import annotations

GIB = 1024**3


def estimate(weights, config=None, context=2048):
    config = config or {}
    # Shape-derived KV estimate. Unknown shapes retain a conservative reserve.
    layers, heads = config.get("num_hidden_layers"), config.get("num_key_value_heads") or config.get("num_attention_heads")
    dim = config.get("head_dim")
    if not dim and config.get("hidden_size") and config.get("num_attention_heads"):
        dim = config["hidden_size"] // config["num_attention_heads"]
    kv = 2 * layers * heads * dim * context * 2 if all(type(x) is int and x > 0 for x in (layers, heads, dim)) else GIB
    if type(weights) is not int or weights <= 0:
        return {"estimated_runtime_bytes": None, "kv_bytes": kv, "context_tokens": context}
    return {"estimated_runtime_bytes": weights + kv + max(2 * GIB, weights // 5),
            "kv_bytes": kv, "context_tokens": context}


def admission(plan, capacity, reserve):
    precision = plan.get("runtime_precision", "float16")
    estimate_result = estimate(plan["weight_bytes"], plan.get("config"))
    reasons = list(plan.get("runtime_reasons", []))
    total = capacity.get("gpu_total_bytes")
    host_available = capacity.get("host_available_bytes")
    required = estimate_result["estimated_runtime_bytes"]
    if total is None:
        reasons.append("设备显存容量未知；尚不能判断全 GPU 部署可用性")
    elif required is None or required > total:
        reasons.append("16 位浮点权重、KV 与运行余量预估超过 GPU 总容量")
    if host_available is None or host_available < plan["weight_bytes"] + GIB:
        reasons.append("主机可用内存不足以保守支持 CPU 权重暂存")
    disk_ok = capacity["disk_free_bytes"] >= plan["download_bytes"] + reserve
    if not disk_ok:
        reasons.append("下载磁盘空间不足，需保留节点配置的空闲余量")
    return {**estimate_result, "eligible": not reasons, "reasons": reasons,
            "disk_ok": disk_ok, "status": "estimated_fit" if not reasons else "blocked",
            "scope": f"single_gpu_{'bf16' if precision == 'bfloat16' else 'fp16'}_batch1_context2048_after_unloading_current_model",
            "limitations": ["容量预估不是运行认证；下载后仍需加载、有限 logits 预热和实际推理验证。",
                            "不使用 CPU/disk offload，不把量化存储体积当作 FP16 驻留体积。",
                            "显存总量用于卸载当前模型后的估算；其他负载、分配碎片与瞬时余量仍可能导致失败。"]}


def effective_host_available(host_available, cgroup_root):
    """Respect the container's memory ceiling instead of advertising host RAM."""
    for limit_name, used_name in (("memory.max", "memory.current"),
                                  ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes")):
        try:
            limit = int((cgroup_root / limit_name).read_text().strip())
            used = int((cgroup_root / used_name).read_text().strip())
        except (OSError, ValueError):
            continue
        if 0 < limit < 2**60 and used >= 0:
            available = max(0, limit - used)
            return min(host_available, available) if host_available is not None else available
    return host_available
