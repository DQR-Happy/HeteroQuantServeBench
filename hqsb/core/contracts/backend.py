"""C4 — Backend contract.

Defines the abstract :class:`Backend` interface every runtime adapter must
implement, together with :class:`BackendCapability` (declared capabilities
used for dispatch/routing) and :class:`GenerationOutput` (the raw output a
backend returns without any statistical summarization).

By construction the backend *never* computes statistics or writes results;
it only reports capabilities and produces raw generation output. The
benchmark engine is responsible for turning that output into a
:class:`~hqsb.core.contracts.result.BenchmarkResult`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from pydantic import Field

from hqsb.core.contracts.base import VersionedModel
from hqsb.core.contracts.trace import TraceEvent


class BackendCapability(VersionedModel):
    """Declared capabilities of a backend, used for dispatch and routing."""

    SCHEMA_VERSION = "1.0.0"

    name: str = Field(..., description="Backend name, e.g. 'dummy', 'pytorch'.")
    supported_dtypes: List[str] = Field(
        default_factory=list, description="Weight dtypes supported."
    )
    max_batch: int = Field(1, ge=1, description="Maximum supported batch size.")
    max_context: Optional[int] = Field(
        default=None, description="Maximum context length in tokens."
    )
    streaming: bool = Field(False, description="Supports token streaming.")
    quantization: List[str] = Field(
        default_factory=list, description="Supported quantization schemes."
    )
    distributed: bool = Field(False, description="Supports multi-device parallelism.")

    def supports_dtype(self, dtype: str) -> bool:
        """Return True if ``dtype`` is within the declared supported set."""
        return dtype in self.supported_dtypes


class GenerationSample(VersionedModel):
    """A single raw generation pass, without any aggregation."""

    SCHEMA_VERSION = "1.0.0"

    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    generated_token_ids: List[int] = Field(default_factory=list)
    prefill_forward_ms: float = Field(0.0, ge=0)
    first_token_selection_ms: float = Field(0.0, ge=0)
    itl_ms: List[float] = Field(default_factory=list)
    peak_cuda_allocated_mb: float = Field(0.0, ge=0)
    peak_cuda_reserved_mb: float = Field(0.0, ge=0)


class GenerationOutput(VersionedModel):
    """Raw output of a backend's ``generate`` call."""

    SCHEMA_VERSION = "1.0.0"

    samples: List[GenerationSample] = Field(default_factory=list)
    trace_events: List[TraceEvent] = Field(default_factory=list)


class Backend(ABC):
    """Abstract runtime backend contract (C4).

    A backend exposes a uniform lifecycle (``load``/``warmup``/``generate``/
    ``stream``/``health``/``capabilities``/``metrics``/``close``) and must
    NOT compute statistics or write result files — those concerns belong to
    the benchmark engine.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The canonical backend name."""

    @abstractmethod
    def capabilities(self) -> BackendCapability:
        """Return the backend's declared capabilities."""

    @abstractmethod
    def load(self, artifact: object) -> None:
        """Load the model artifact (validated by the caller)."""

    @abstractmethod
    def warmup(self, workload: object) -> None:
        """Perform warmup passes for the given workload."""

    @abstractmethod
    def generate(self, workload: object, inputs: object) -> GenerationOutput:
        """Run generation and return raw samples + trace events."""

    @abstractmethod
    def health(self) -> bool:
        """Return True if the backend is healthy and ready."""

    @abstractmethod
    def metrics(self) -> Dict[str, object]:
        """Return backend-internal runtime metrics (not result statistics)."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources held by the backend."""

    # Optional streaming hook; concrete backends may override.
    def stream(self, workload: object, inputs: object):
        """Yield tokens as they are generated (optional)."""
        raise NotImplementedError(
            f"backend {self.name!r} does not implement streaming"
        )


__all__ = [
    "Backend",
    "BackendCapability",
    "GenerationOutput",
    "GenerationSample",
]
