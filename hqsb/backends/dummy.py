"""DummyBackend — a reference implementation of the Backend contract (C4).

This backend exists so that a third-party developer (or a coding agent) can
see, in one file, exactly what a minimal, correct backend looks like: how to
declare capabilities, implement the lifecycle, and return raw
:class:`GenerationOutput` without doing any statistics.

It produces *deterministic* token sequences seeded by the workload's
``seed`` field, making it usable in CPU-only CI and as a contract example
without any model weights or GPU.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from hqsb.core.contracts.backend import (
    Backend,
    BackendCapability,
    GenerationOutput,
    GenerationSample,
)
from hqsb.core.contracts.model import ModelArtifact
from hqsb.core.contracts.workload import WorkloadSpec


class DummyBackend(Backend):
    """A deterministic, dependency-free backend for contract validation.

    The generated token IDs are drawn from a seeded PRNG so identical
    workloads yield identical outputs. Per-step latency is a fixed constant,
    which keeps the example trivially verifiable.
    """

    def __init__(
        self,
        *,
        name: str = "dummy",
        per_token_latency_ms: float = 10.0,
        prefill_latency_ms: float = 20.0,
        token_vocab: int = 50_000,
    ) -> None:
        self._name = name
        self._per_token_latency_ms = per_token_latency_ms
        self._prefill_latency_ms = prefill_latency_ms
        self._token_vocab = token_vocab
        self._loaded_artifact: Optional[ModelArtifact] = None
        self._closed = False

    # ── Backend contract ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    def capabilities(self) -> BackendCapability:
        return BackendCapability(
            name=self._name,
            supported_dtypes=["float16", "float32"],
            max_batch=1,
            max_context=4096,
            streaming=False,
            quantization=[],
            distributed=False,
        )

    def load(self, artifact: object) -> None:
        """Accept and record a model artifact (no actual loading).

        Raises:
            TypeError: If ``artifact`` is not a :class:`ModelArtifact`.
        """
        if not isinstance(artifact, ModelArtifact):
            raise TypeError(
                f"DummyBackend.load expects ModelArtifact, got "
                f"{type(artifact).__name__}"
            )
        self._loaded_artifact = artifact
        self._closed = False

    def warmup(self, workload: object) -> None:
        """No-op warmup (dummy backend has nothing to warm up)."""
        if not isinstance(workload, WorkloadSpec):
            raise TypeError("warmup expects WorkloadSpec")

    def generate(self, workload: object, inputs: object) -> GenerationOutput:
        """Produce deterministic raw generation samples for ``workload``."""
        if not isinstance(workload, WorkloadSpec):
            raise TypeError("generate expects WorkloadSpec")

        samples: List[GenerationSample] = []

        for _ in range(workload.repetitions):
            # Reset the PRNG per repetition so repeated measurements of the
            # same workload are bitwise-identical (deterministic).
            rng = random.Random(workload.seed)
            token_ids = [
                rng.randrange(self._token_vocab)
                for _ in range(workload.output_tokens)
            ]
            itl_ms = [
                self._per_token_latency_ms
                for _ in range(max(workload.output_tokens - 1, 0))
            ]
            samples.append(
                GenerationSample(
                    input_tokens=workload.input_tokens,
                    output_tokens=workload.output_tokens,
                    generated_token_ids=token_ids,
                    prefill_forward_ms=self._prefill_latency_ms,
                    first_token_selection_ms=0.1,
                    itl_ms=itl_ms,
                    peak_cuda_allocated_mb=0.0,
                    peak_cuda_reserved_mb=0.0,
                )
            )

        return GenerationOutput(samples=samples, trace_events=[])

    def health(self) -> bool:
        return not self._closed

    def metrics(self) -> Dict[str, object]:
        return {
            "loaded": self._loaded_artifact is not None,
            "closed": self._closed,
        }

    def close(self) -> None:
        self._closed = True
        self._loaded_artifact = None


def make_dummy_backend(**kwargs) -> DummyBackend:
    """Factory for :class:`DummyBackend` (convenient for registry use)."""
    return DummyBackend(**kwargs)


__all__ = ["DummyBackend", "make_dummy_backend"]
