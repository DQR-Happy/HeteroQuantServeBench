"""L10 request, routing and managed-replica experiment contracts; no runtime imports."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApplicationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=8192)


class ApplicationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,64}$")
    tenant: str = Field("default", pattern=r"^[a-zA-Z0-9_.-]{1,64}$")
    prompt: str | None = Field(None, min_length=1, max_length=8192)
    messages: list[ApplicationMessage] = Field(default_factory=list, max_length=16)
    arrival_ms: int = Field(0, ge=0, le=30000)
    max_output_tokens: int = Field(32, ge=1, le=512)

    @model_validator(mode="after")
    def valid(self):
        if (self.prompt is not None) == bool(self.messages):
            raise ValueError("L10_EXACTLY_ONE_PROMPT_OR_MESSAGES_REQUIRED")
        return self


class ApplicationTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,64}$")
    machine_id: str = Field("local", pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    deployment_id: str = Field(min_length=1, max_length=128)
    lifecycle: Literal["existing", "managed"] = "existing"
    role: Literal["default", "small", "large"] = "default"


class ApplicationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    routing: Literal[
        "round_robin", "least_pending", "cache_affinity", "length_model"
    ] = "round_robin"
    short_input_chars: int = Field(128, ge=1, le=8192)
    result_cache: bool = False
    cache_ttl_s: int = Field(60, ge=1, le=3600)
    cache_max_entries: int = Field(32, ge=1, le=256)
    cache_max_bytes: int = Field(4 * 1024**2, ge=1024, le=16 * 1024**2)
    prefix_reuse: bool = False
    input_transform: Literal["none", "head_tail_tokens", "recent_messages"] = "none"
    input_token_cap: int = Field(256, ge=16, le=4096)
    recent_messages: int = Field(4, ge=1, le=16)
    output_cap: int | None = Field(None, ge=1, le=512)
    autoscale: bool = False
    min_replicas: int = Field(0, ge=0, le=4)
    max_replicas: int = Field(1, ge=1, le=4)
    idle_unload_ms: int = Field(1000, ge=100, le=10000)

    @model_validator(mode="after")
    def valid(self):
        if self.min_replicas > self.max_replicas:
            raise ValueError("L10_INVALID_REPLICA_RANGE")
        return self


def default_workload():
    return [
        ApplicationInput(id=f"request-{i}", prompt=p, arrival_ms=i * 150)
        for i, p in enumerate(
            [
                "Once upon a time, there was a little fox.",
                "Once upon a time, there was a little fox.",
                "Explain why the sky is blue in one sentence.",
                "Explain why the sky is blue in one sentence.",
            ]
        )
    ]


class ApplicationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    targets: list[ApplicationTarget] = Field(min_length=1, max_length=4)
    requests: list[ApplicationInput] = Field(
        default_factory=default_workload, min_length=1, max_length=32
    )
    baseline: ApplicationPolicy = Field(default_factory=ApplicationPolicy)
    candidate: ApplicationPolicy = Field(
        default_factory=lambda: ApplicationPolicy(
            routing="cache_affinity", result_cache=True
        )
    )
    concurrency: int = Field(2, ge=1, le=8)
    request_deadline_s: int = Field(90, ge=5, le=300)
    timeout_s: int = Field(300, ge=30, le=900)
    fixed_output_length: bool = False

    @model_validator(mode="after")
    def valid(self):
        for rows in (self.targets, self.requests):
            if len({r.id for r in rows}) != len(rows):
                raise ValueError("L10_IDS_MUST_BE_UNIQUE")
        if (
            sum(
                len(r.prompt or "") + sum(len(m.content) for m in r.messages)
                for r in self.requests
            )
            > 65536
        ):
            raise ValueError("L10_TOTAL_INPUT_CHARACTER_BUDGET_65536")
        if len({t.machine_id for t in self.targets}) != len(self.targets):
            raise ValueError("L10_ONE_TARGET_PER_NODE_REQUIRED")
        for p in (self.baseline, self.candidate):
            if p.autoscale and (
                p.min_replicas > len(self.targets)
                or sum(t.lifecycle == "existing" for t in self.targets) > p.max_replicas
            ):
                raise ValueError(
                    "L10_REPLICA_RANGE_INCOMPATIBLE_WITH_REGISTERED_TARGETS"
                )
            if p.routing == "length_model" and not {"small", "large"} <= {
                t.role for t in self.targets
            }:
                raise ValueError("L10_LENGTH_ROUTE_REQUIRES_SMALL_AND_LARGE_TARGETS")
            if p.input_transform == "head_tail_tokens" and any(
                r.messages for r in self.requests
            ):
                raise ValueError("L10_TOKEN_HEAD_TAIL_ONLY_SUPPORTS_RAW_PROMPT")
            if p.input_transform == "recent_messages" and any(
                r.prompt is not None for r in self.requests
            ):
                raise ValueError("L10_RECENT_MESSAGES_REQUIRES_CHAT_INPUT")
        return self
