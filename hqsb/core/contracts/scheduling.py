"""L8 v1: device-independent bounded workload and execution contracts."""
from dataclasses import dataclass, field
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    prompt: str = Field(min_length=1, max_length=8192)
    arrival_ms: int = Field(0, ge=0, le=30000)
    output_tokens: int = Field(32, ge=1, le=256)
    priority: int = Field(0, ge=-10, le=10)
    tenant: str = Field("default", min_length=1, max_length=64)
    cancel_after_ms: int | None = Field(None, ge=0, le=60000)
    deadline_ms: int = Field(60000, ge=1, le=120000)
    fixed_output_length: bool = True


class SchedulingLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_active_sequences: int = Field(4, ge=1, le=16)
    max_batch_sequences: int = Field(4, ge=1, le=16)
    max_batched_tokens: int = Field(256, ge=1, le=4096)
    max_padded_positions: int = Field(16384, ge=1, le=131072)
    max_temporary_bytes: int = Field(512 * 1024**2, ge=1024**2, le=2 * 1024**3)
    kv_capacity_tokens: int = Field(8192, ge=1, le=131072)
    prefill_chunk_tokens: int = Field(64, ge=1, le=4096)
    chunked_prefill: bool = True
    policy: Literal["fcfs", "priority_aging", "tenant_fair"] = "fcfs"
    tenant_weights: dict[str, float] = Field(default_factory=dict, max_length=32)
    phase_policy: Literal["balanced", "decode_first"] = "balanced"
    prefill_quota_tokens: int = Field(16, ge=1, le=4096)
    aging_ms: int = Field(100, ge=1, le=10000)
    preemption: Literal["off", "recompute"] = "off"
    max_preemptions: int = Field(4, ge=0, le=64)

    @model_validator(mode="after")
    def valid_weights(self):
        if any(not k or len(k) > 64 or not math.isfinite(v) or not .1 <= v <= 100 for k, v in self.tenant_weights.items()):
            raise ValueError("TENANT_WEIGHT_INVALID")
        if self.phase_policy == "decode_first" and self.prefill_quota_tokens > self.max_batched_tokens:
            raise ValueError("PREFILL_QUOTA_EXCEEDS_BUDGET")
        if self.preemption == "recompute" and not self.chunked_prefill:
            raise ValueError("RECOMPUTE_REQUIRES_CHUNKED_PREFILL")
        return self


class SchedulingExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deployment_id: str
    expected_epoch: str
    requests: list[WorkItem] = Field(min_length=1, max_length=128)
    modes: list[Literal["serial", "static", "continuous"]] = Field(
        default_factory=lambda: ["serial", "continuous"], min_length=1, max_length=3)
    limits: SchedulingLimits = Field(default_factory=SchedulingLimits)
    repeats: int = Field(1, ge=1, le=5)
    arrival_mode: Literal["open_loop", "closed_loop"] = "open_loop"
    concurrency: int = Field(4, ge=1, le=16)
    save_input: bool = False
    session_deadline_s: int = Field(120, ge=1, le=600)
    sender_lag_limit_ms: int = Field(20, ge=1, le=1000)
    ttft_slo_ms: int = Field(2000, ge=1, le=60000)
    itl_slo_ms: int = Field(200, ge=1, le=60000)

    @model_validator(mode="after")
    def unique(self):
        if len({r.id for r in self.requests}) != len(self.requests):
            raise ValueError("DUPLICATE_WORKLOAD_REQUEST_ID")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("DUPLICATE_MODE")
        if "serial" in self.modes and self.modes[0] != "serial":
            raise ValueError("SERIAL_REFERENCE_MUST_RUN_FIRST")
        if self.limits.chunked_prefill and self.limits.prefill_chunk_tokens > self.limits.max_batched_tokens:
            raise ValueError("CHUNK_EXCEEDS_TOKEN_BUDGET")
        return self


@dataclass
class Sequence:
    """Only the live engine mutates progress; executor owns device tensors."""
    id: str
    input_ids: list[int]
    limit: int
    received: float
    deadline: float
    priority: int = 0
    tenant: str = "default"
    cancel_at: float | None = None
    fixed: bool = True
    output: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    margins: list[float] = field(default_factory=list)
    consumed: int = 0
    state: str = "waiting"
    admitted: float | None = None
    ended: float | None = None
    reason: str | None = None
    error: str | None = None
    preemptions: int = 0
    recomputed: int = 0
    last_scheduled: float | None = None
    max_unscheduled_ms: float = 0

    @property
    def history(self):
        return self.input_ids + self.output

    @property
    def target(self):
        # Last emitted token remains pending until the next decode forward.
        return len(self.input_ids) + max(0, len(self.output) - 1)


@dataclass(frozen=True)
class Step:
    request_id: str
    start: int
    token_ids: tuple[int, ...]
    emit: bool
    phase: str


@dataclass(frozen=True)
class StepResult:
    tokens: tuple[int, ...]
    receipt: dict
