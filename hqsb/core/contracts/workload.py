"""C2 — WorkloadSpec contract.

Defines a fully-specified inference workload so that a benchmark pass is
reproducible: fixed batch/ISL/OSL, token provenance or dataset version,
sampling, warmup, repetitions, and concurrency/arrival/timeout parameters
(top-level architecture §5 C2).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field, field_validator

from hqsb.core.contracts.base import VersionedModel
from hqsb.core.errors import SchemaError


class WorkloadSpec(VersionedModel):
    """A versioned, fully-specified inference workload."""

    SCHEMA_VERSION = "1.0.0"

    name: str = Field(..., description="Workload case name, e.g. 'short'.")
    batch_size: int = Field(1, ge=1, description="Inference batch size.")
    input_tokens: int = Field(..., ge=1, description="Input sequence length (ISL).")
    output_tokens: int = Field(..., ge=1, description="Output sequence length (OSL).")
    seed: int = Field(0, description="Random seed for reproducible sampling.")
    sampling: str = Field(
        "greedy",
        description="Decoding strategy: 'greedy' or a sampling scheme name.",
    )
    warmup: int = Field(1, ge=0, description="Number of warmup passes.")
    repetitions: int = Field(1, ge=1, description="Number of timed passes.")
    token_ids: Optional[List[int]] = Field(
        default=None,
        description="Explicit input token IDs; supersedes tokenizer synthesis.",
    )
    dataset_version: Optional[str] = Field(
        default=None,
        description="Dataset/version used to derive the token IDs.",
    )
    concurrency: int = Field(
        1, ge=1, description="Number of concurrent requests (serving layer)."
    )
    arrival_process: str = Field(
        "fixed", description="Arrival process: 'fixed', 'poisson', 'burst', ..."
    )
    timeout_s: Optional[float] = Field(
        default=None, gt=0, description="Per-request timeout in seconds."
    )
    stop_condition: str = Field(
        "output_tokens", description="Stop criterion: 'output_tokens', 'eos', ..."
    )

    @field_validator("input_tokens", "output_tokens")
    @classmethod
    def _validate_positive(cls, value: int) -> int:
        if value < 1:
            raise SchemaError(f"token count must be >= 1, got {value}")
        return value


__all__ = ["WorkloadSpec"]
