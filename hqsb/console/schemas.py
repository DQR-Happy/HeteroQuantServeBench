"""Versioned HTTP response objects; flexible measurement fields preserve provenance."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Metrics(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_tokens: int | None = None
    output_tokens: int | None = None
    runtime_first_token_ms: float | None = None
    console_first_content_ms: float | None = None
    runtime_e2e_ms: float | None = None
    console_e2e_ms: float | None = None
    decode_tail_tokens_per_s: float | None = None
    measurement_profile: str | None = None


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    kind: Literal["load", "unload", "generate"]
    state: Literal[
        "queued",
        "running",
        "cancel_requested",
        "completed",
        "cancelled",
        "failed",
        "timed_out",
        "interrupted",
    ]
    created_at: float
    updated_at: float
    config: dict[str, Any]
    output: str
    metrics: Metrics
    error: str | None
    cleanup: str
    quality: str
    seq: int


class RunPage(BaseModel):
    items: list[RunRecord]
    total: int
    next_offset: int | None
