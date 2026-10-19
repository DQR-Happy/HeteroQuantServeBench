"""L6 request contracts. No tensor/runtime imports on the API process."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class MemoryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["model_default", "dynamic", "static", "offloaded_dynamic", "offloaded_static", "recompute"] = "model_default"
    max_cache_tokens: int | None = Field(None, ge=1, le=131072)
    prefix_reuse: StrictBool = False
    prefix_budget_bytes: int = Field(64 * 1024**2, ge=0, le=512 * 1024**2)
    prefix_max_entries: int = Field(8, ge=1, le=64)
    prefix_max_tokens: int = Field(1024, ge=1, le=8192)
    prefix_namespace: str = Field("default", pattern=r"^[a-zA-Z0-9_.-]{1,64}$")

    @model_validator(mode="after")
    def compatible(self):
        if self.prefix_reuse and (self.mode != "dynamic" or not self.prefix_budget_bytes):
            raise ValueError("PREFIX_REQUIRES_DYNAMIC_CACHE_AND_POSITIVE_BUDGET")
        if self.max_cache_tokens is not None and self.mode not in {"static", "offloaded_static"}:
            raise ValueError("EXPLICIT_CAPACITY_REQUIRES_STATIC_CACHE")
        return self


def validate_memory(plan, *, native=True, kv_bits=None, compiled=False, diagnostic=False, quality=False):
    resolved = MemoryPlan.model_validate(plan).model_dump()
    if not native:
        raise ValueError("L6_REQUIRES_LOCAL_NATIVE_PYTORCH_WORKER")
    if compiled or diagnostic or quality:
        raise ValueError("L6_COMPILE_GRAPH_CAPTURE_OR_TEACHER_FORCING_COMPOSITION_NOT_VERIFIED")
    if kv_bits and resolved["mode"] != "model_default":
        raise ValueError("L6_CUSTOM_LAYOUT_WITH_QUANTIZED_KV_NOT_VERIFIED_USE_MODEL_DEFAULT")
    return resolved


def catalogue():
    return {"schema_version": "l6-v1", "default": MemoryPlan().model_dump(), "modes": [
        {"id": "model_default", "label": "模型默认 KV（兼容现有 L4）"},
        {"id": "dynamic", "label": "动态 KV"},
        {"id": "static", "label": "预分配静态 KV"},
        {"id": "offloaded_dynamic", "label": "动态 KV / CPU 卸载与逐层预取"},
        {"id": "offloaded_static", "label": "静态 KV / CPU 卸载与逐层预取"},
        {"id": "recompute", "label": "关闭 KV / 完整历史重计算"}],
        "limitations": ["KV block pools are tensor storage pages, not OS/CUDA virtual pages or a PagedAttention kernel.",
                        "Process page faults are worker-scoped; GPU L2 hit rate and physical page migration are not inferred.",
                        "Prefix reuse currently supports full-attention dynamic KV only; entries belong to one resident model worker.",
                        "Independent teacher-forcing PPL does not measure autoregressive KV changes; compare actual generated tokens separately."]}
