"""DummyBackend — a reference implementation of the Backend contract (C4).

This backend exists so that a third-party developer (or a coding agent) can
see, in one file, exactly what a minimal, correct backend looks like: how to
declare capabilities, implement the lifecycle, and return raw
:class:`GenerationOutput` without doing any statistics.

It produces *deterministic* token sequences seeded by the workload's
``seed`` field, making it usable in CPU-only CI and as a contract example
without any model weights or GPU.

Besides deterministic tokens, the backend also:

* emits :class:`~hqsb.core.contracts.trace.TraceEvent` (C7) across its
  lifecycle — ``load``/``warmup``/``generate``/``close`` — so a benchmark
  pass can be correlated afterwards via ``trace_id``/``span_id``/
  ``parent_span_id``;
* records per-method call counts (``load``/``warmup``/``generate``/``close``)
  so the observation matrix can assert exact invocation order and counts;
* can inject a failure at a chosen lifecycle stage (``fail_at``) so negative
  paths (load failure, generation failure, close failure) can be exercised
  deterministically without any real hardware or model.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

from hqsb.core.contracts.backend import (
    Backend,
    BackendCapability,
    GenerationOutput,
    GenerationSample,
)
from hqsb.core.contracts.model import ModelArtifact
from hqsb.core.contracts.trace import TraceEvent, TraceEventType
from hqsb.core.contracts.workload import WorkloadSpec
from hqsb.core.errors import BackendError
from hqsb.core.ids import new_span_id, new_trace_id


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
        fail_at: Optional[str] = None,
        fail_message: str = "injected dummy failure",
    ) -> None:
        self._name = name
        self._per_token_latency_ms = per_token_latency_ms
        self._prefill_latency_ms = prefill_latency_ms
        self._token_vocab = token_vocab

        # Fault injection: one of None | "load" | "warmup" | "generate" |
        # "close". When set, the matching method raises BackendError.
        self._fail_at = fail_at
        self._fail_message = fail_message

        self._loaded_artifact: Optional[ModelArtifact] = None
        self._closed = False

        # Trace state: one trace_id spans load → close.
        self._trace_id: Optional[str] = None
        self._root_span_id: Optional[str] = None
        self._trace_events: List[TraceEvent] = []

        # Call ledger for the observation matrix.
        self._load_count = 0
        self._warmup_count = 0
        self._generate_count = 0
        self._close_count = 0

    # ── Observable state (used by the E01-03 runner) ───────────────────

    @property
    def trace_events(self) -> List[TraceEvent]:
        """Trace events emitted during the current (or last) lifecycle."""
        return list(self._trace_events)

    @property
    def call_counts(self) -> Dict[str, int]:
        """Per-method invocation counts for the observation matrix."""
        return {
            "load": self._load_count,
            "warmup": self._warmup_count,
            "generate": self._generate_count,
            "close": self._close_count,
        }

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
            BackendError: If ``fail_at == "load"``.
        """
        if not isinstance(artifact, ModelArtifact):
            raise TypeError(
                f"DummyBackend.load expects ModelArtifact, got "
                f"{type(artifact).__name__}"
            )
        self._maybe_fail("load")
        self._emit(TraceEventType.MODEL_LOAD, f"{self._name}.load")
        self._loaded_artifact = artifact
        self._closed = False
        self._load_count += 1

    def warmup(self, workload: object) -> None:
        """No-op warmup (dummy backend has nothing to warm up)."""
        if not isinstance(workload, WorkloadSpec):
            raise TypeError("warmup expects WorkloadSpec")
        self._maybe_fail("warmup")
        self._emit(
            TraceEventType.PREFILL,
            f"{self._name}.warmup",
            parent_span_id=self._root_span_id,
        )
        self._warmup_count += 1

    def generate(self, workload: object, inputs: object) -> GenerationOutput:
        """Produce deterministic raw generation samples for ``workload``."""
        if not isinstance(workload, WorkloadSpec):
            raise TypeError("generate expects WorkloadSpec")
        self._maybe_fail("generate")
        self._begin_trace()

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
            self._emit(
                TraceEventType.DECODE,
                f"{self._name}.generate.sample",
                parent_span_id=self._root_span_id,
                attributes={
                    "output_tokens": workload.output_tokens,
                    "seed": workload.seed,
                },
            )
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

        self._emit(
            TraceEventType.OUTPUT,
            f"{self._name}.generate.done",
            parent_span_id=self._root_span_id,
            attributes={"samples": len(samples), "status": "completed"},
        )
        self._generate_count += 1

        return GenerationOutput(
            samples=samples,
            trace_events=list(self._trace_events),
        )

    def health(self) -> bool:
        return not self._closed

    def metrics(self) -> Dict[str, object]:
        return {
            "loaded": self._loaded_artifact is not None,
            "closed": self._closed,
            "call_counts": self.call_counts,
        }

    def close(self) -> None:
        """Release all resources held by the backend."""
        self._maybe_fail("close")
        if self._trace_id is not None:
            self._emit(
                TraceEventType.OUTPUT,
                f"{self._name}.close",
                parent_span_id=self._root_span_id,
                attributes={"status": "closed"},
            )
            self._trace_id = None
            self._root_span_id = None
        self._closed = True
        self._loaded_artifact = None
        self._close_count += 1

    # ── Trace / fault helpers ─────────────────────────────────────────

    def _begin_trace(self) -> None:
        if self._trace_id is None:
            self._trace_id = new_trace_id()

    def _emit(
        self,
        event_type: TraceEventType,
        name: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, object]] = None,
    ) -> TraceEvent:
        self._begin_trace()
        event = TraceEvent(
            event_type=event_type,
            timestamp_ns=time.monotonic_ns(),
            trace_id=self._trace_id,
            span_id=new_span_id(),
            parent_span_id=parent_span_id,
            name=name,
            attributes=dict(attributes or {}),
        )
        self._trace_events.append(event)
        if self._root_span_id is None:
            self._root_span_id = event.span_id
        return event

    def _maybe_fail(self, stage: str) -> None:
        if self._fail_at == stage:
            if self._trace_id is not None:
                self._emit(
                    TraceEventType.OUTPUT,
                    f"{self._name}.{stage}.failed",
                    parent_span_id=self._root_span_id,
                    attributes={"status": "failed", "error": self._fail_message},
                )
            raise BackendError(self._fail_message)


def make_dummy_backend(**kwargs) -> DummyBackend:
    """Factory for :class:`DummyBackend` (convenient for registry use)."""
    return DummyBackend(**kwargs)


__all__ = ["DummyBackend", "make_dummy_backend"]
