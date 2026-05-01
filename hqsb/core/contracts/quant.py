"""C5 — QuantArtifact contract.

Describes a quantization recipe and its resulting artifact, from algorithm
and granularity to packing/layout, calibration provenance, and measured
accuracy (top-level architecture §5 C5).
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from hqsb.core.contracts.base import VersionedModel


class QuantArtifact(VersionedModel):
    """A versioned quantization recipe and artifact descriptor."""

    SCHEMA_VERSION = "1.0.0"

    algorithm: str = Field(
        ...,
        description="Quantization method: 'rtn', 'gptq', 'awq', 'smoothquant', ...",
    )
    bits: int = Field(..., ge=1, le=8, description="Weight bit width.")
    granularity: str = Field(
        ...,
        description="Scale granularity: 'per-tensor', 'per-channel', 'per-group'.",
    )
    symmetric: bool = Field(True, description="Whether quantization is symmetric.")
    group_size: Optional[int] = Field(
        default=None, gt=0, description="Group size for per-group granularity."
    )
    scale: Optional[str] = Field(
        default=None, description="Scale tensor identifier or hash."
    )
    zero_point: Optional[str] = Field(
        default=None, description="Zero-point tensor identifier or hash."
    )
    calibration: Optional[str] = Field(
        default=None, description="Calibration dataset/sample version."
    )
    packing: str = Field(
        "native", description="Weight packing layout, e.g. 'native', 'int4-packed'."
    )
    kernel_compatibility: str = Field(
        "unknown", description="Kernel/GEMM compatibility descriptor."
    )
    accuracy: Optional[dict] = Field(
        default=None,
        description="Measured accuracy summary (e.g. perplexity, token match).",
    )


__all__ = ["QuantArtifact"]
