"""L9 execution contracts, independent of accelerators and HTTP."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ParallelPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["single", "dp", "tp", "layer_shard", "prefill_decode"] = "single"
    device_type: Literal["cuda", "cpu"] = "cuda"
    devices: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=4)
    precision: Literal["float16", "bfloat16", "float32"] = "float16"
    prompts: list[str] = Field(
        default_factory=lambda: [
            "Once upon a time, there was a little fox.",
            "Explain why the sky is blue in one sentence.",
        ],
        min_length=1,
        max_length=8,
    )
    max_output_tokens: int = Field(32, ge=1, le=128)
    fixed_output_length: bool = True
    trace_collectives: bool = False
    timeout_s: int = Field(120, ge=15, le=300)

    @model_validator(mode="after")
    def valid(self):
        if len(set(self.devices)) != len(self.devices) or any(
            i < 0 or i > 63 for i in self.devices
        ):
            raise ValueError("L9_DEVICES_MUST_BE_UNIQUE_VALID_INDICES")
        n = len(self.devices)
        if self.strategy == "single" and n != 1:
            raise ValueError("L9_SINGLE_REQUIRES_ONE_DEVICE")
        if self.strategy != "single" and n < 2:
            raise ValueError(
                "L9_PARALLEL_REQUIRES_AT_LEAST_TWO_DEVICES_OR_CPU_DIAGNOSTIC_RANKS"
            )
        if self.strategy == "prefill_decode" and n != 2:
            raise ValueError("L9_PREFILL_DECODE_REQUIRES_TWO_RANKS")
        if self.strategy == "layer_shard" and self.device_type != "cuda":
            raise ValueError("L9_LAYER_SHARD_REQUIRES_DISTINCT_CUDA_DEVICES")
        if self.strategy == "dp" and len(self.prompts) < n:
            raise ValueError("L9_DP_REQUIRES_AT_LEAST_ONE_REQUEST_PER_RANK")
        if self.device_type == "cpu" and self.precision != "float32":
            raise ValueError("L9_CPU_DIAGNOSTICS_REQUIRE_FLOAT32")
        if self.device_type == "cpu" and self.devices != list(range(n)):
            raise ValueError("L9_CPU_RANK_LABELS_MUST_START_AT_ZERO")
        if any(not p.strip() or len(p) > 4096 for p in self.prompts):
            raise ValueError("L9_PROMPTS_MUST_BE_NONEMPTY_AND_BOUNDED")
        if self.trace_collectives and len(self.prompts) * self.max_output_tokens > 64:
            raise ValueError(
                "L9_CPU_COMMUNICATION_TRACE_LIMIT_64_TOTAL_OUTPUT_TOKENS_USE_OFF_FOR_PERFORMANCE"
            )
        return self

    @property
    def ranks(self):
        return 1 if self.strategy in {"single", "layer_shard"} else len(self.devices)


def validate_hardware(plan, hardware):
    p = ParallelPlan.model_validate(plan)
    if p.device_type == "cuda":
        if any(i >= len(hardware["devices"]) for i in p.devices):
            raise ValueError(
                "L9_REQUESTED_GPU_NOT_VISIBLE: select a multi-GPU node on the machines page"
            )
        if p.strategy in {"tp", "dp"} and not hardware["nccl"]:
            raise ValueError("L9_NCCL_UNAVAILABLE")
    if p.ranks > 1 and not hardware["gloo"]:
        raise ValueError("L9_GLOO_REQUIRED_FOR_BOUNDED_CONTROL_AND_KV_HANDOFF")
    return p


def catalogue():
    return {
        "schema_version": "l9-v1",
        "default": ParallelPlan().model_dump(),
        "strategies": [
            {"id": "single", "label": "单设备基线"},
            {"id": "dp", "label": "DP 独立副本 / 请求分片"},
            {"id": "tp", "label": "TP / Transformers 原生权重分片"},
            {"id": "layer_shard", "label": "按层跨卡放置 / 顺序执行"},
            {"id": "prefill_decode", "label": "Prefill / Decode 分离及实际 KV 传输"},
        ],
        "not_implemented": [
            "PP 多微批流水与通信计算重叠",
            "CP / SP 上下文与序列并行",
            "MoE EP 与专家负载均衡",
            "TP×PP×DP 混合并行",
            "跨主机 TP/NCCL 自动组网",
            "跨主机 KV 传输与异版本 KV 转换",
        ],
        "limits": {
            "requests": 8,
            "prompt_tokens": 512,
            "output_tokens": 128,
            "devices": 4,
        },
    }
