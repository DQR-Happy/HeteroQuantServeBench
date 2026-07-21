"""Backend-interface benchmark engine.

The engine orchestrates a benchmark pass against any :class:`Backend`
implementation and produces a versioned :class:`BenchmarkResult`. It is the
single place where raw backend output is summarized, correctness is gated,
and results are bound to environment/commit/config metadata.

Dependency rule (top-level architecture §6): this module depends only on
``hqsb.core`` contracts and the ``Backend`` interface. It does **not** import
any concrete backend, model loader, or serving module — the backend is
injected by the caller or resolved from the registry.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from hqsb.core.contracts.backend import Backend, GenerationSample
from hqsb.core.contracts.model import ModelArtifact
from hqsb.core.contracts.result import (
    BenchmarkResult,
    CorrectnessReport,
    EnvironmentInfo,
    ResourceUsage,
)
from hqsb.core.contracts.workload import WorkloadSpec
from hqsb.core.errors import BackendError, CapabilityError
from hqsb.core.ids import new_run_id
from hqsb.benchmark.metrics import latency_summary, model_core_timings


def _model_artifact_hash(artifact: Optional[ModelArtifact]) -> Optional[str]:
    """Compute a stable hash over an artifact's identity + file hashes.

    Delegates to :meth:`ModelArtifact.identity_hash` so the benchmark engine
    and every other consumer agree on one canonical artifact digest.
    """
    if artifact is None:
        return None
    return artifact.identity_hash()


def _workload_hash(workload: WorkloadSpec) -> str:
    canonical = json.dumps(
        workload.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _summarize(samples: List[GenerationSample]) -> Dict[str, Any]:
    """Aggregate raw samples into a statistical summary.

    Uses the existing latency-summary helper for ITL and computes simple
    means for the phase timings and throughput, matching the model-core
    metric definitions.
    """
    if not samples:
        return {}

    prefill_ms = [s.prefill_forward_ms for s in samples]
    timings = [
        model_core_timings(
            s.prefill_forward_ms, s.first_token_selection_ms, s.itl_ms
        )
        for s in samples
    ]
    ttft_ms = [t["model_core_ttft_ms"] for t in timings]
    decode_total_ms = [t["decode_total_ms"] for t in timings]
    e2e_ms = [t["model_core_e2e_ms"] for t in timings]
    itl_all = [lat for s in samples for lat in s.itl_ms]

    total_output_tokens = sum(s.output_tokens for s in samples)
    total_e2e_ms = sum(e2e_ms)
    total_decode_tokens = sum(max(s.output_tokens - 1, 0) for s in samples)
    total_decode_ms = sum(decode_total_ms)

    return {
        "repetitions": len(samples),
        "prefill_forward_ms_mean": _mean(prefill_ms),
        "model_core_ttft_ms_mean": _mean(ttft_ms),
        "decode_total_ms_mean": _mean(decode_total_ms),
        "model_core_e2e_ms_mean": _mean(e2e_ms),
        "decode_tokens_per_s": (
            total_decode_tokens / (total_decode_ms / 1000.0)
            if total_decode_ms > 0
            else 0.0
        ),
        "output_tokens_per_s": (
            total_output_tokens / (total_e2e_ms / 1000.0)
            if total_e2e_ms > 0
            else 0.0
        ),
        "itl": latency_summary(itl_all),
        "throughput_semantics": "decode_tail_and_output_v2",
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _determinism_correctness(samples: List[GenerationSample]) -> CorrectnessReport:
    """Default correctness gate: token sequences must be deterministic."""
    if not samples:
        return CorrectnessReport(passed=False, method="determinism")
    malformed = [index for index, sample in enumerate(samples)
                 if len(sample.generated_token_ids) != sample.output_tokens
                 or len(sample.itl_ms) != max(sample.output_tokens - 1, 0)]
    hashes = {
        hashlib.sha256(
            json.dumps(s.generated_token_ids).encode("utf-8")
        ).hexdigest()
        for s in samples
    }
    deterministic = len(hashes) == 1 and not malformed
    return CorrectnessReport(
        passed=deterministic,
        method="determinism",
        details={"distinct_sequence_hashes": len(hashes), "malformed_samples": malformed},
    )


class BenchmarkEngine:
    """Runs a workload against a backend and emits a BenchmarkResult."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def run(
        self,
        workload: WorkloadSpec,
        *,
        artifact: Optional[ModelArtifact] = None,
        environment: Optional[EnvironmentInfo] = None,
        git_commit: Optional[str] = None,
        git_dirty: Optional[bool] = None,
        check_capabilities: bool = True,
        load_artifact: bool = True,
    ) -> BenchmarkResult:
        """Execute ``workload`` against the configured backend.

        Args:
            workload: The workload to run.
            artifact: Optional model artifact to load into the backend.
            environment: Optional environment snapshot (else empty).
            git_commit: Optional git commit SHA.
            git_dirty: Optional working-tree dirty flag.
            check_capabilities: When True, verify the backend's declared
                capabilities cover the workload's dtype/batch before running
                (raises :class:`CapabilityError` otherwise).
            load_artifact: When True (default), call ``backend.load(artifact)``
                before running. Set False when the caller has already loaded
                the artifact (e.g. reuse across many workloads) to avoid
                redundant loads.

        Returns:
            A fully-populated :class:`BenchmarkResult`.

        Raises:
            CapabilityError: If a capability check fails.
            BackendError: If the backend fails during load/generate.
        """
        if check_capabilities:
            self._assert_capabilities(workload, artifact)

        try:
            if artifact is not None and load_artifact:
                self.backend.load(artifact)
            self.backend.warmup(workload)
            output = self.backend.generate(workload, None)
        except (CapabilityError, BackendError):
            raise
        except Exception as exc:
            raise BackendError(
                f"backend {self.backend.name!r} failed: {exc}"
            ) from exc

        samples = output.samples
        run_id = new_run_id()
        summary = _summarize(samples)
        summary["trace"] = {
            "run_id": run_id,
            "trace_ids": sorted({event.trace_id for event in output.trace_events}),
            "events": [event.model_dump(mode="json") for event in output.trace_events],
        }
        if output.backend_metrics:
            summary["backend_metrics"] = output.backend_metrics
        correctness = _determinism_correctness(samples)
        resource = self._aggregate_resource(samples)

        return BenchmarkResult(
            run_id=run_id,
            timestamp=time.time(),
            environment=environment or EnvironmentInfo(),
            git_commit=git_commit,
            git_dirty=git_dirty,
            model_artifact_hash=_model_artifact_hash(artifact),
            config_hash=_workload_hash(workload),
            workload=workload,
            raw_samples=[s.model_dump(mode="json") for s in samples],
            summary=summary,
            correctness=correctness,
            resource=resource,
            artifact_links=self._artifact_links(artifact),
        )

    def _assert_capabilities(
        self,
        workload: WorkloadSpec,
        artifact: Optional[ModelArtifact],
    ) -> None:
        capability = self.backend.capabilities()
        context = workload.input_tokens + workload.output_tokens
        if capability.max_context is not None and context > capability.max_context:
            raise CapabilityError(
                f"requested context {context} exceeds backend "
                f"{self.backend.name!r} max_context {capability.max_context}",
                details={"requested_context": context, "max_context": capability.max_context},
            )
        if workload.batch_size > capability.max_batch:
            raise CapabilityError(
                f"workload batch {workload.batch_size} exceeds backend "
                f"{self.backend.name!r} max_batch {capability.max_batch}",
                details={
                    "requested_batch": workload.batch_size,
                    "max_batch": capability.max_batch,
                },
            )
        if artifact is not None and not capability.supports_dtype(artifact.dtype):
            raise CapabilityError(
                f"backend {self.backend.name!r} does not support dtype "
                f"{artifact.dtype!r}",
                details={
                    "requested_dtype": artifact.dtype,
                    "supported_dtypes": capability.supported_dtypes,
                },
            )

    @staticmethod
    def _aggregate_resource(samples: List[GenerationSample]) -> ResourceUsage:
        if not samples:
            return ResourceUsage()
        return ResourceUsage(
            peak_cuda_allocated_mb=max(s.peak_cuda_allocated_mb for s in samples),
            peak_cuda_reserved_mb=max(s.peak_cuda_reserved_mb for s in samples),
            process_rss_mb=0.0,
        )

    @staticmethod
    def _artifact_links(artifact: Optional[ModelArtifact]) -> Dict[str, str]:
        if artifact is None:
            return {}
        return {"model_id": artifact.model_id, "source": artifact.source}


def run_backend(
    backend: Backend,
    workload: WorkloadSpec,
    **kwargs,
) -> BenchmarkResult:
    """Convenience wrapper: run ``workload`` on ``backend`` and return result."""
    return BenchmarkEngine(backend).run(workload, **kwargs)


__all__ = ["BenchmarkEngine", "run_backend"]
