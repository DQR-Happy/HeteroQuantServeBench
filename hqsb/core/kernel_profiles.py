"""Server-owned L2 profiles. Pure data: importing this never probes a device.

Versions describe executable implementations, not speed or quality rankings.
Unlisted families retain their original framework implementation.
"""

PROFILES = {
    "framework-native": {},
    "qwen3-norm-scalar-v1": {"rms_norm": "cuda-hf-scalar-v1"},
    "qwen3-norm-vector-v2": {"rms_norm": "cuda-hf-half2-v2"},
    "qwen3-residual-norm-v1": {"residual_rms_norm": "cuda-hf-residual-v1"},
    "qwen3-silu-mul-v1": {"silu_mul": "cuda-half-rounded-v1"},
    "qwen3-rope-v1": {"rope": "cuda-split-half-v1"},
    "qwen3-linear-lt-v1": {"linear": "cublaslt-fp32-acc-v1"},
    "qwen3-linear-gemv-v1": {"linear": "cuda-small-m-gemv-v1"},
    "qwen3-attention-math-v1": {"attention": "torch-sdpa-math-gqa-repeat-v1"},
    "qwen3-attention-flash-v1": {"attention": "torch-sdpa-flash-gqa-repeat-v1"},
    "qwen3-attention-triton-n32-v1": {"attention": "triton-online-softmax-n32-v1"},
    "qwen3-attention-triton-n64-v1": {"attention": "triton-online-softmax-n64-v1"},
    "qwen3-local-fusions-v1": {
        "rms_norm": "cuda-hf-half2-v2",
        "residual_rms_norm": "cuda-hf-residual-v1",
        "silu_mul": "cuda-half-rounded-v1", "rope": "cuda-split-half-v1",
    },
    "qwen3-l2-math-v1": {
        "rms_norm": "cuda-hf-half2-v2",
        "residual_rms_norm": "cuda-hf-residual-v1",
        "silu_mul": "cuda-half-rounded-v1", "rope": "cuda-split-half-v1",
        "linear": "cublaslt-fp32-acc-v1",
        "attention": "torch-sdpa-math-gqa-repeat-v1",
    },
}

# Presentation metadata belongs to the catalogue, never to the hashed execution plan.
LABELS = {
    "framework-native": ("框架原生", "Framework native · 模型默认实现"),
    "qwen3-norm-scalar-v1": ("RMSNorm", "CUDA scalar · v1"),
    "qwen3-norm-vector-v2": ("RMSNorm", "CUDA half2 向量化 · v2"),
    "qwen3-residual-norm-v1": ("残差与归一化", "CUDA residual + RMSNorm · v1"),
    "qwen3-silu-mul-v1": ("激活", "CUDA SiLU × up · v1"),
    "qwen3-rope-v1": ("位置编码", "CUDA split-half RoPE · v1"),
    "qwen3-linear-lt-v1": ("Linear", "cuBLASLt FP32 累加 · v1"),
    "qwen3-linear-gemv-v1": ("Linear", "CUDA small-M GEMV + cuBLASLt · v1"),
    "qwen3-attention-math-v1": ("Attention", "PyTorch SDPA math · v1"),
    "qwen3-attention-flash-v1": ("Attention", "PyTorch SDPA flash · v1"),
    "qwen3-attention-triton-n32-v1": ("Attention", "Triton online softmax · K/V tile 32 · v1"),
    "qwen3-attention-triton-n64-v1": ("Attention", "Triton online softmax · K/V tile 64 · v1"),
    "qwen3-local-fusions-v1": ("组合方案", "Norm + residual + SiLU + RoPE · v1"),
    "qwen3-l2-math-v1": ("组合方案", "局部融合 + cuBLASLt + SDPA math · v1"),
}


def requires_native(name):
    return any(not variant.startswith(("torch-", "triton-")) for variant in profile(name)["operators"].values())


def profile(name):
    if not isinstance(name, str) or name not in PROFILES:
        raise ValueError("KERNEL_PROFILE_UNKNOWN:" + str(name))
    return {"schema_version": "l2-profile-v1", "profile_id": name,
            "operators": dict(PROFILES[name]), "fallback_policy": "strict",
            "workspace_max_bytes": 0, "recommendation": "research_candidate" if PROFILES[name] else "baseline"}


def catalogue():
    return {"schema_version": "l2-catalogue-v1", "profiles": [
        {**profile(name), "family": LABELS[name][0], "label": LABELS[name][1],
         "requires_native_artifact": requires_native(name), "capability": "requires_worker_validation"}
        for name in PROFILES], "not_integrated": [
            "flash-attn-external", "cutlass-model-linear", "qk-norm-rope-fusion",
            "qkv-weight-packing", "gate-up-weight-packing", "quantized-linear", "moe"]}
