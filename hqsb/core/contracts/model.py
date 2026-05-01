"""C1 — ModelArtifact contract.

Describes a fully-materialized, hash-pinned model snapshot. A benchmark
result references a :class:`ModelArtifact` so that its correctness and
performance can always be traced back to the exact weights, tokenizer, and
configuration that produced it (top-level architecture §5 C1).
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import Field

from hqsb.core.contracts.base import VersionedModel


class ModelArtifact(VersionedModel):
    """A versioned, hash-pinned description of a model snapshot."""

    SCHEMA_VERSION = "1.0.0"

    model_id: str = Field(..., description="Model identifier, e.g. 'Qwen/Qwen3-1.7B'.")
    source: str = Field(
        ...,
        description="Model source: 'modelscope', 'huggingface', or 'local'.",
    )
    revision: Optional[str] = Field(
        default=None,
        description="Upstream revision/tag/commit, when applicable.",
    )
    file_hashes: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from relative file path to lowercase SHA256 hex.",
    )
    architecture: str = Field(
        ...,
        description="Model architecture family, e.g. 'Qwen3ForCausalLM'.",
    )
    dtype: str = Field(
        ...,
        description="Weight precision, e.g. 'float16', 'bfloat16', 'float32'.",
    )
    quantization: Optional[str] = Field(
        default=None,
        description="Quantization scheme, if any (e.g. 'int4-gptq').",
    )
    layout: str = Field(
        default="dense",
        description="Parameter layout ('dense', 'moe', ...).",
    )
    context_length: Optional[int] = Field(
        default=None,
        description="Maximum supported context length in tokens.",
    )
    license: Optional[str] = Field(
        default=None,
        description="Model license identifier or URL.",
    )
    tool_version: Optional[str] = Field(
        default=None,
        description="Version of the tool/loader that produced this artifact.",
    )


__all__ = ["ModelArtifact"]
