"""C3 — OperatorSpec contract.

Describes an operator's semantic signature and numerical contract so that
multiple implementations (CUDA/Triton/CUTLASS/Ascend C) can be compared and
dispatched under one definition (top-level architecture §5 C3).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from hqsb.core.contracts.base import VersionedModel


class TensorSpec(VersionedModel):
    """A tensor's dtype, shape, and layout contract."""

    SCHEMA_VERSION = "1.0.0"

    name: str = Field(..., description="Tensor role, e.g. 'input', 'weight'.")
    dtype: str = Field(..., description="Element dtype, e.g. 'float32', 'float16'.")
    shape: List[int] = Field(..., description="Logical shape (dynamic dims as -1).")
    layout: str = Field(
        "contiguous",
        description="Memory layout: 'contiguous', 'channels_last', or stride map.",
    )


class OperatorSpec(VersionedModel):
    """A versioned, implementation-independent operator specification."""

    SCHEMA_VERSION = "1.0.0"

    name: str = Field(..., description="Operator name, e.g. 'rmsnorm'.")
    semantic_version: str = Field(
        ...,
        description="Semantic version of the operator's math contract.",
    )
    inputs: List[TensorSpec] = Field(..., description="Input tensors.")
    outputs: List[TensorSpec] = Field(..., description="Output tensors.")
    device: str = Field("cuda", description="Target device family.")
    stream: Optional[str] = Field(
        default=None, description="CUDA/Ascend stream semantics if stream-aware."
    )
    workspace_bytes: int = Field(
        0, ge=0, description="Optional workspace size in bytes."
    )
    deterministic: bool = Field(
        True, description="Whether the operator is bitwise-deterministic."
    )
    tolerance: float = Field(
        0.0, ge=0, description="Numerical tolerance vs reference (e.g. RMSE)."
    )
    implementation: str = Field(
        ...,
        description="Implementation variant identifier, e.g. 'v0_shared'.",
    )
    fallback: Optional[str] = Field(
        default=None,
        description="Fallback implementation name when unsupported.",
    )


__all__ = ["OperatorSpec", "TensorSpec"]
