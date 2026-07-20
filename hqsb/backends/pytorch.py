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
from hqsb.core.errors import BackendError, CapabilityError, ConfigError
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
        manifest_allow_extra: Explicit metadata exceptions to the strict manifest gate.
        cpu_staging: Load through swappable CPU tensors before GPU consolidation.
    """

    def __init__(
        self,
        *,
        model_path: str = "~/models/hqsb/Qwen3-1.7B",
        dtype: torch.dtype = torch.float16,
        attention_backend: str = "eager",
        verify_manifest: Optional[str] = None,
        manifest_allow_extra: tuple[str, ...] = (),
        cpu_staging: bool = False,
    ) -> None:
        self._model_path = model_path
        self._dtype = dtype
        self._attention_backend = attention_backend
        self._verify_manifest = verify_manifest
        self._manifest_allow_extra = tuple(manifest_allow_extra)
        self._cpu_staging = cpu_staging

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
            if self._artifact.identity_hash() == artifact.identity_hash():
                logger.debug("Model %s already loaded; skipping.", artifact.model_id)
                return
            raise BackendError(
                "a different artifact is already loaded; close the backend before "
                "loading another revision, dtype or file manifest"
            )

        if artifact.dtype != str(self._dtype).removeprefix("torch."):
            raise CapabilityError(
                "artifact dtype differs from the configured PyTorch dtype",
                details={"requested_dtype": artifact.dtype, "actual_dtype": str(self._dtype)},
            )

        try:
            from hqsb.models.loader import load_qwen3

            self._tokenizer, self._model, load_time_s = load_qwen3(
                self._model_path,
                dtype=self._dtype,
                attention_backend=self._attention_backend,
                verify_manifest=self._verify_manifest,
                allow_extra=self._manifest_allow_extra,
                cpu_staging=self._cpu_staging,
            )
        except Exception as exc:
            raise BackendError(
                f"failed to load model {artifact.model_id!r}: {exc}"
            ) from exc

        self._load_time_s = load_time_s
        self._artifact = artifact

    def warmup(self, workload: object) -> None:
        """Run the requested number of short passes to warm caches and allocator."""
        if not isinstance(workload, WorkloadSpec):
            raise TypeError("warmup expects WorkloadSpec")
        self._require_loaded()

        self._validate_workload(workload)
        if workload.warmup == 0:
            return
        inputs = self._inputs(workload)
        # Honor the C2 warmup count without recording warmup as a sample.
        for _ in range(workload.warmup):
            benchmark_model_core(
                self._model, inputs, output_tokens=min(2, workload.output_tokens)
            )

    def generate(self, workload: object, inputs: object) -> GenerationOutput:
        """Run ``repetitions`` model-core passes and return raw samples.

        Raises:
            TypeError: If ``workload`` is not a :class:`WorkloadSpec`.
            BackendError: If the model is not loaded or generation fails.
        """
        if not isinstance(workload, WorkloadSpec):
            raise TypeError("generate expects WorkloadSpec")
        self._require_loaded()
        self._validate_workload(workload)
        model_inputs = self._inputs(workload, inputs)

        samples: List[GenerationSample] = []
        first_pass_metrics: Dict[str, Any] = {}

        try:
            for _ in range(workload.repetitions):
                result = benchmark_model_core(
                    self._model, model_inputs, workload.output_tokens
                )
                samples.append(self._to_sample(result))
                if not first_pass_metrics:
                    first_pass_metrics = self._backend_metrics(result)
                    first_pass_metrics["artifact_verification"] = {
                        "manifest": self._verify_manifest,
                        "enabled": self._verify_manifest is not None,
                        "strict_extra": True,
                        "allow_extra": list(self._manifest_allow_extra),
                    }
                    first_pass_metrics["cpu_staging"] = self._cpu_staging
        except Exception as exc:
            raise BackendError(
                f"generation failed for workload {workload.name!r}: {exc}"
            ) from exc

        return GenerationOutput(
            samples=samples,
            trace_events=[],
            backend_metrics=first_pass_metrics,
        )

    def _validate_workload(self, workload: WorkloadSpec) -> None:
        """Refuse semantics the fixed-length greedy reference cannot honor."""
        unsupported = (
            workload.batch_size != 1 or workload.sampling != "greedy"
            or workload.stop_condition != "output_tokens"
            or workload.concurrency != 1 or workload.timeout_s is not None
        )
        if unsupported:
            raise CapabilityError(
                "PyTorch reference supports batch=1, greedy, fixed output_tokens, "
                "concurrency=1 and no execution timeout"
            )
        if workload.input_tokens + workload.output_tokens > self.capabilities().max_context:
            raise CapabilityError("workload exceeds the reference context limit")

    def _inputs(self, workload: WorkloadSpec, inputs: object = None) -> Dict[str, Any]:
        """Preserve explicit C2/C4 token IDs; synthesize only when absent."""
        tokens = inputs if inputs is not None else workload.token_ids
        if tokens is None:
            return make_fixed_token_input(
                self._tokenizer, workload.input_tokens, device=self._device()
            )
        if not isinstance(tokens, (list, tuple)) or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in tokens
        ):
            raise ConfigError("inputs must be a sequence of non-negative integer token IDs")
        if len(tokens) != workload.input_tokens:
            raise ConfigError("explicit token count differs from WorkloadSpec.input_tokens")
        if workload.token_ids is not None and list(tokens) != workload.token_ids:
            raise ConfigError("C4 inputs differ from frozen WorkloadSpec.token_ids")
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self._device())
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

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
