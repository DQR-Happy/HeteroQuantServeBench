"""PyTorch reference backend implementing the C4 Backend contract.

This is the canonical FP16 Qwen3-1.7B reference runtime for S02. It wraps
:func:`hqsb.models.loader.load_qwen3` and
:func:`hqsb.benchmark.model_core.benchmark_model_core` behind the abstract
:class:`~hqsb.core.contracts.backend.Backend` interface, so the benchmark
engine can run it without knowing anything about ModelScope/Transformers.

Reference semantics (S02 execution step 2):

* greedy decoding;
* fixed token IDs (seeded by the workload);
* batch size 1;
* **no** HTTP, queue, or tokenizer timing — tokenizer time is excluded
  from all latencies (model-core only).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch

from hqsb.core.contracts.backend import (
    Backend,
    BackendCapability,
    GenerationOutput,
    GenerationSample,
)
from hqsb.core.contracts.model import ModelArtifact
from hqsb.core.contracts.workload import WorkloadSpec
from hqsb.core.errors import BackendError
from hqsb.benchmark.model_core import benchmark_model_core
from hqsb.benchmark.memory import cuda_memory_snapshot
from hqsb.benchmark.workload import make_fixed_token_input

logger = logging.getLogger(__name__)


class PyTorchBackend(Backend):
    """FP16 reference backend backed by ``load_qwen3`` + model-core engine.

    Args:
        model_path: Local model directory (defaults to ``~/models/hqsb/Qwen3-1.7B``).
        dtype: Weight precision (default FP16).
        attention_backend: Attention implementation (default eager).
        verify_manifest: Optional SHA256 manifest to verify before loading.
    """

    def __init__(
        self,
        *,
        model_path: str = "~/models/hqsb/Qwen3-1.7B",
        dtype: torch.dtype = torch.float16,
        attention_backend: str = "eager",
        verify_manifest: Optional[str] = None,
    ) -> None:
        self._model_path = model_path
        self._dtype = dtype
        self._attention_backend = attention_backend
        self._verify_manifest = verify_manifest

        self._tokenizer: Any = None
        self._model: Any = None
        self._load_time_s: Optional[float] = None
        self._artifact: Optional[ModelArtifact] = None

    # ── Backend contract ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "pytorch"

    def capabilities(self) -> BackendCapability:
        return BackendCapability(
            name=self.name,
            supported_dtypes=["float16", "float32", "bfloat16"],
            max_batch=1,
            max_context=32768,
            streaming=False,
            quantization=[],
            distributed=False,
        )

    def load(self, artifact: object) -> None:
        """Load the model artifact via ``load_qwen3``.

        Raises:
            TypeError: If ``artifact`` is not a :class:`ModelArtifact`.
            BackendError: If loading fails.
        """
        if not isinstance(artifact, ModelArtifact):
            raise TypeError(
                f"PyTorchBackend.load expects ModelArtifact, got "
                f"{type(artifact).__name__}"
            )

        # Idempotent load: re-loading the same artifact is a no-op. This
        # makes `engine.run` and caller-managed loads safe to compose.
        if self._model is not None and self._artifact is not None:
            if self._artifact.model_id == artifact.model_id:
                logger.debug("Model %s already loaded; skipping.", artifact.model_id)
                return

        try:
            from hqsb.models.loader import load_qwen3

            self._tokenizer, self._model, load_time_s = load_qwen3(
                self._model_path,
                dtype=self._dtype,
                attention_backend=self._attention_backend,
                verify_manifest=self._verify_manifest,
            )
        except Exception as exc:
            raise BackendError(
                f"failed to load model {artifact.model_id!r}: {exc}"
            ) from exc

        self._load_time_s = load_time_s
        self._artifact = artifact

    def warmup(self, workload: object) -> None:
        """Run a single short generation pass to warm caches and allocator."""
        if not isinstance(workload, WorkloadSpec):
            raise TypeError("warmup expects WorkloadSpec")
        self._require_loaded()

        inputs = make_fixed_token_input(
            self._tokenizer,
            workload.input_tokens,
            device=self._device(),
        )
        # Warmup with a tiny output; not timed or recorded.
        benchmark_model_core(self._model, inputs, output_tokens=2)

    def generate(self, workload: object, inputs: object) -> GenerationOutput:
        """Run ``repetitions`` model-core passes and return raw samples.

        Raises:
            TypeError: If ``workload`` is not a :class:`WorkloadSpec`.
            BackendError: If the model is not loaded or generation fails.
        """
        if not isinstance(workload, WorkloadSpec):
            raise TypeError("generate expects WorkloadSpec")
        self._require_loaded()

        samples: List[GenerationSample] = []
        first_pass_metrics: Dict[str, Any] = {}

        try:
            for _ in range(workload.repetitions):
                inputs = make_fixed_token_input(
                    self._tokenizer,
                    workload.input_tokens,
                    device=self._device(),
                )
                result = benchmark_model_core(
                    self._model, inputs, workload.output_tokens
                )
                samples.append(self._to_sample(result))
                if not first_pass_metrics:
                    first_pass_metrics = self._backend_metrics(result)
        except Exception as exc:
            raise BackendError(
                f"generation failed for workload {workload.name!r}: {exc}"
            ) from exc

        return GenerationOutput(
            samples=samples,
            trace_events=[],
            backend_metrics=first_pass_metrics,
        )

    def health(self) -> bool:
        return self._model is not None

    def metrics(self) -> Dict[str, object]:
        snapshot = cuda_memory_snapshot()
        return {
            "loaded": self._model is not None,
            "load_time_s": self._load_time_s,
            "model_id": self._artifact.model_id if self._artifact else None,
            "cuda_memory": snapshot,
        }

    def close(self) -> None:
        """Release the model and free CUDA memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Helpers ──────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if self._model is None:
            raise BackendError(
                "backend is not loaded; call load(artifact) before generate()"
            )

    def _device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _to_sample(result: Dict[str, Any]) -> GenerationSample:
        return GenerationSample(
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            generated_token_ids=list(result["generated_token_ids"]),
            prefill_forward_ms=float(result["prefill_forward_ms"]),
            first_token_selection_ms=float(result["first_token_selection_ms"]),
            itl_ms=[float(x) for x in result["raw_itl_ms"]],
            peak_cuda_allocated_mb=float(result["peak_cuda_allocated_mb"]),
            peak_cuda_reserved_mb=float(result["peak_cuda_reserved_mb"]),
        )

    @staticmethod
    def _backend_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "load_time_s": result.get("load_time_s"),
            "kv_cache": result.get("kv_cache", {}),
            "model_weight_bytes": result.get("model_weight_bytes", 0),
            "process_rss_bytes": result.get("process_rss_bytes", 0),
            "process_swap_bytes": result.get("process_swap_bytes", 0),
            "cuda_memory": cuda_memory_snapshot(),
        }


def make_pytorch_backend(**kwargs) -> PyTorchBackend:
    """Factory for :class:`PyTorchBackend` (convenient for registry use)."""
    return PyTorchBackend(**kwargs)


__all__ = ["PyTorchBackend", "make_pytorch_backend"]
