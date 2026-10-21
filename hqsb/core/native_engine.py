"""Versioned upstream controls, grouped by HQSB optimization layers.

Only engine-owned public arguments are translated here. Local PyTorch patches
have a separate execution contract and are never injected into upstream models.
"""
from pydantic import BaseModel, ConfigDict, Field

VERSIONS = {"vllm": "0.11.0", "sglang": "0.5.5"}


class EngineOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    attention_backend: str = "auto"
    enforce_eager: bool = True
    dtype: str = "float16"
    quantization: str = "none"
    kv_cache_dtype: str = "auto"
    memory_fraction: float = Field(0.7, ge=0.1, le=0.95)
    prefix_cache: bool = False
    block_size: int = Field(16, ge=1, le=64)
    cpu_offload_gb: float = Field(0.0, ge=0, le=64)
    max_running_requests: int = Field(16, ge=1, le=256)
    max_batch_tokens: int = Field(2048, ge=32, le=131072)
    chunked_prefill: bool = True
    scheduling_policy: str = "fcfs"
    tensor_parallel_size: int = Field(1, ge=1, le=8)
    pipeline_parallel_size: int = Field(1, ge=1, le=8)
    seed: int = Field(0, ge=0, le=2147483647)
    enable_thinking: bool = False


FIELDS = {
    "attention_backend": ([2], "原生 Attention 实现"),
    "enforce_eager": ([3], "Eager 执行；关闭后由引擎使用编译/图优化"),
    "dtype": ([4], "计算精度"), "quantization": ([4], "量化加载器（none 沿用制品声明）"),
    "kv_cache_dtype": ([4, 6], "KV 存储精度"),
    "memory_fraction": ([6], "引擎显存预算比例（各引擎预算口径独立）"),
    "prefix_cache": ([6], "前缀缓存"), "block_size": ([6], "KV 分块/分页大小"),
    "cpu_offload_gb": ([6], "CPU 权重卸载预算 GiB"),
    "max_running_requests": ([8], "最大同时运行请求数"),
    "max_batch_tokens": ([8], "批次 Token 预算 / SGLang prefill Token 上限"),
    "chunked_prefill": ([8], "分块 Prefill"), "scheduling_policy": ([8], "调度策略"),
    "tensor_parallel_size": ([9], "张量并行度"), "pipeline_parallel_size": ([9], "流水线并行度"),
    "seed": ([7], "随机种子"),
    "enable_thinking": ([7, 10], "聊天模板 thinking 参数（适用模型）"),
}


def option_choices(engine):
    return {
        "dtype": ["float16", "bfloat16", "float32"],
        "quantization": ["none", "awq", "gptq", "bitsandbytes"],
        "kv_cache_dtype": ["auto", "fp8_e4m3", "fp8_e5m2"],
        "attention_backend": ["auto", "FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"] if engine == "vllm" else ["auto", "triton", "flashinfer", "torch_native"],
        "scheduling_policy": ["fcfs", "priority"] if engine == "vllm" else ["fcfs", "lpm", "random", "dfs-weight", "lof"],
        "block_size": [8, 16, 32] if engine == "vllm" else [1, 16, 32, 64],
    }


def resolve_options(engine, raw):
    if engine not in VERSIONS:
        raise ValueError("NATIVE_ENGINE_UNKNOWN")
    options = EngineOptions.model_validate(raw).model_dump()
    for name, allowed in option_choices(engine).items():
        if options[name] not in allowed:
            raise ValueError("NATIVE_OPTION_INVALID:" + name)
    if engine == "sglang" and options["cpu_offload_gb"] != int(options["cpu_offload_gb"]):
        raise ValueError("SGLANG_CPU_OFFLOAD_INTEGER_GB_REQUIRED")
    return options


def catalogue(engine):
    defaults = resolve_options(engine, {})
    properties = EngineOptions.model_json_schema()["properties"]
    return {"engine": engine, "version": VERSIONS[engine], "scope": "deployment_reload",
            "items": [{"id": name, "layers": layers, "label": label, "default": defaults[name],
                       "implementation_status": "upstream_public_argument",
                       "runtime_capability": "requires_engine_model_hardware_validation",
                       "api": "/deployments/{id}/load", "field": "engine_options." + name,
                       "entrypoint": "/engines", "proof_required": True}
                       | {k: properties[name][k] for k in ("minimum", "maximum") if k in properties[name]}
                       | ({"values": option_choices(engine)[name]} if name in option_choices(engine) else {})
                      for name, (layers, label) in FIELDS.items()],
            "unmapped_layers": {1: "记录硬件信息；系统配置实验当前使用 PyTorch 执行端", 5: "通过兼容模型制品接入",
                                7: "本次接入贪心生成与随机种子；推测解码需独立适配",
                                10: "原生流式服务；L10 策略执行接口待适配"}}


def launch_arguments(engine, config, options, port):
    o = resolve_options(engine, options)
    args = ["-m", "vllm.entrypoints.openai.api_server" if engine == "vllm" else "sglang.launch_server",
            "--host", "127.0.0.1", "--port", str(port),
            "--model" if engine == "vllm" else "--model-path", config["model_path"],
            "--served-model-name", config["model"], "--dtype", o["dtype"],
            "--kv-cache-dtype", o["kv_cache_dtype"], "--seed" if engine == "vllm" else "--random-seed", str(o["seed"])]
    if engine == "vllm":
        args += ["--max-model-len", str(config["context_limit"]), "--gpu-memory-utilization", str(o["memory_fraction"]),
                 "--block-size", str(o["block_size"]), "--max-num-seqs", str(o["max_running_requests"]),
                 "--max-num-batched-tokens", str(o["max_batch_tokens"]),
                 "--scheduling-policy", o["scheduling_policy"], "--cpu-offload-gb", str(o["cpu_offload_gb"]),
                 "--tensor-parallel-size", str(o["tensor_parallel_size"]),
                 "--pipeline-parallel-size", str(o["pipeline_parallel_size"]),
                 "--enable-prefix-caching" if o["prefix_cache"] else "--no-enable-prefix-caching",
                 "--enable-chunked-prefill" if o["chunked_prefill"] else "--no-enable-chunked-prefill",
                 "--generation-config", "vllm"]
        if o["enforce_eager"]:
            args += ["--enforce-eager"]
    else:
        args += ["--context-length", str(config["context_limit"]), "--mem-fraction-static", str(o["memory_fraction"]),
                 "--page-size", str(o["block_size"]), "--max-running-requests", str(o["max_running_requests"]),
                 "--max-prefill-tokens", str(o["max_batch_tokens"]),
                 "--chunked-prefill-size", str(o["max_batch_tokens"] if o["chunked_prefill"] else -1),
                 "--schedule-policy", o["scheduling_policy"], "--tp-size", str(o["tensor_parallel_size"]),
                 "--pp-size", str(o["pipeline_parallel_size"]), "--enable-metrics", "--sampling-defaults", "openai",
                 "--cpu-offload-gb", str(int(o["cpu_offload_gb"]))]
        if not o["prefix_cache"]:
            args += ["--disable-radix-cache"]
        if o["enforce_eager"]:
            args += ["--disable-cuda-graph"]
    if o["attention_backend"] != "auto":
        # vLLM 0.11 selects attention through VLLM_ATTENTION_BACKEND.
        if engine == "sglang":
            args += ["--attention-backend", o["attention_backend"]]
    if o["quantization"] != "none":
        args += ["--quantization", o["quantization"]]
    return args
