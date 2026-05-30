"""Low-bit executors and their registry (E05-06 §6/§12).

Three executors exist and every experiment must carry all of them as controls:

``fp16_reference``
    plain FP16 GEMM on FP16 weights (the baseline);
``storage_only``
    the packed artifact is loaded and materialized to FP16 before the GEMM
    (storage/load effects only — *never* a low-bit execution claim);
``fused_dequant``
    the Triton kernel reads the packed bytes and dequantizes inside the GEMM
    (a real ``fused_dequant_weight_only`` path, requiring an observed symbol).

Each executor reports its observed kernel symbol, provider and layout, so a
result record can be checked against what actually ran rather than what was
requested (``requested/actual/reason`` for every path).
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

from hqsb.core.errors import CapabilityError, ConfigError
from hqsb.quant.artifact import PackedVariantRecord, QuantArtifactDocument
from hqsb.quant.execution import (
    FP16_REFERENCE,
    FUSED_DEQUANT_WEIGHT_ONLY,
    STORAGE_ONLY,
    ExecutionRecord,
)

EXECUTOR_FP16 = "fp16_reference"
EXECUTOR_STORAGE = "storage_only"
EXECUTOR_FUSED = "fused_dequant"


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - CPU-minimal CI
        raise CapabilityError(
            "executors need torch; install the 'benchmark' extra",
            details={"capability": "torch", "reason": str(exc)},
        ) from exc
    return torch


class BaseExecutor:
    """Common shape/metadata handling for the three executors."""

    name = "base"
    provider = "torch"
    execution_label = FP16_REFERENCE
    low_bit = False

    def __init__(self, *, device: str = "cuda") -> None:
        self.device = device

    def is_available(self) -> bool:
        try:
            torch = _require_torch()
        except CapabilityError:
            return False
        if self.device.startswith("cuda"):
            return bool(torch.cuda.is_available())
        return True

    @property
    def observed_kernel_symbol(self) -> str:
        return ""

    def describe(self) -> Dict[str, Any]:
        return {
            "executor": self.name,
            "provider": self.provider,
            "execution_label": self.execution_label,
            "low_bit": self.low_bit,
            "available": self.is_available(),
            "observed_kernel_symbol": self.observed_kernel_symbol,
            "device": self.device,
        }

    def record(
        self,
        *,
        module: str = "",
        phase: str = "",
        workload_id: str = "",
        m: int = 0,
        n: int = 0,
        k: int = 0,
        duration_ms: float = 0.0,
        expected_kernel: str = "",
        capabilities: str = "",
        fallback_reason: str = "",
    ) -> ExecutionRecord:
        return ExecutionRecord(
            module=module,
            phase=phase,
            workload_id=workload_id,
            m=m,
            n=n,
            k=k,
            expected_kernel=expected_kernel or self.observed_kernel_symbol,
            observed_kernel_symbol=self.observed_kernel_symbol,
            kernel_provider=self.provider,
            packed_layout_id="",
            capability_check=capabilities,
            fallback_reason=fallback_reason,
            weights_materialized_to_fp16=self.execution_label == STORAGE_ONLY,
            duration_ms=duration_ms,
            declared_label=self.execution_label,
        )


class Fp16ReferenceExecutor(BaseExecutor):
    """Plain FP16 GEMM (``x @ W^T``) — the baseline every other path needs."""

    name = EXECUTOR_FP16
    provider = "cublas_via_torch"
    execution_label = FP16_REFERENCE
    low_bit = False

    @property
    def observed_kernel_symbol(self) -> str:
        return "torch_aten_gemm_fp16"

    def gemm(self, x, weight):
        _require_torch()
        return x @ weight.t()

    def dequantize_weight(self, document: QuantArtifactDocument, variant: Optional[PackedVariantRecord] = None):
        raise CapabilityError(
            "the FP16 reference executor does not consume packed artifacts; "
            "use storage_only or fused_dequant",
            details={"capability": "packed_input", "executor": self.name},
        )


class StorageOnlyExecutor(BaseExecutor):
    """Materialize the packed artifact to FP16, then run a normal GEMM."""

    name = EXECUTOR_STORAGE
    provider = "torch"
    execution_label = STORAGE_ONLY
    low_bit = False

    @property
    def observed_kernel_symbol(self) -> str:
        return "dequantize_to_fp16_then_gemm"

    def dequantize_weight(self, document: QuantArtifactDocument, variant: Optional[PackedVariantRecord] = None):
        from ops.quant.w4a16_triton import dequantize_weight_reference, prepare_weights

        prepared = prepare_weights(document, variant, device=self.device)
        return dequantize_weight_reference(prepared)

    def gemm(self, x, weight):
        _require_torch()
        return x @ weight.t()

    def gemm_low_bit(self, x, document: QuantArtifactDocument, variant: Optional[PackedVariantRecord] = None):
        weight = self.dequantize_weight(document, variant)
        # The dequantized weight is float32; the GEMM must match the activation
        # dtype (storage-only is an FP16 GEMM over materialized weights).
        if hasattr(x, "dtype"):
            weight = weight.to(x.dtype)
        return self.gemm(x, weight)


class FusedDequantExecutor(BaseExecutor):
    """Triton kernel that dequantizes inside the GEMM (low-bit path)."""

    name = EXECUTOR_FUSED
    provider = "triton"
    execution_label = FUSED_DEQUANT_WEIGHT_ONLY
    low_bit = True

    def __init__(self, *, device: str = "cuda", block_m: int = 64, block_n: int = 64, block_k: int = 32) -> None:
        super().__init__(device=device)
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k

    @property
    def observed_kernel_symbol(self) -> str:
        from ops.quant.w4a16_triton import KERNEL_SYMBOL

        return KERNEL_SYMBOL

    @property
    def packed_layout_id(self) -> str:
        return ""

    def dequantize_weight(self, document: QuantArtifactDocument, variant: Optional[PackedVariantRecord] = None):
        from ops.quant.w4a16_triton import dequantize_weight_reference, prepare_weights

        prepared = prepare_weights(document, variant, device=self.device)
        return dequantize_weight_reference(prepared)

    def gemm_low_bit(self, x, document: QuantArtifactDocument, variant: Optional[PackedVariantRecord] = None):
        from ops.quant.w4a16_triton import gemm_low_bit, prepare_weights

        prepared = prepare_weights(document, variant, device=self.device)
        return gemm_low_bit(
            x,
            prepared,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
        )

    def record(self, **kwargs) -> ExecutionRecord:
        record = super().record(**kwargs)
        record.packed_layout_id = kwargs.get("packed_layout_id", "")
        return record


EXECUTOR_REGISTRY: Dict[str, Any] = {
    EXECUTOR_FP16: Fp16ReferenceExecutor,
    EXECUTOR_STORAGE: StorageOnlyExecutor,
    EXECUTOR_FUSED: FusedDequantExecutor,
}


def get_executor(name: str, **kwargs: Any) -> BaseExecutor:
    """Resolve an executor by name; unknown names are refused."""
    if name not in EXECUTOR_REGISTRY:
        raise ConfigError(
            f"unknown executor {name!r}; known: {sorted(EXECUTOR_REGISTRY)}"
        )
    return EXECUTOR_REGISTRY[name](**kwargs)


def executor_matrix() -> List[Dict[str, Any]]:
    """Availability matrix for all executors (with reasons, never silent)."""
    rows: List[Dict[str, Any]] = []
    for name, factory in sorted(EXECUTOR_REGISTRY.items()):
        try:
            executor = factory()
            descriptor = executor.describe()
        except Exception as exc:  # noqa: BLE001 - construction failure is a state
            descriptor = {
                "executor": name,
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(descriptor)
    return rows


def third_party_executor_report() -> Dict[str, Any]:
    """Probe optional third-party low-bit runtimes (documented capability only).

    Reported for completeness; the project's own claim must still rely on an
    observed kernel from an executor above, not on "the library exists".
    """
    report = {}
    for module_name in ("torchao", "tensorrt_llm", "vllm", "autoawq", "gptqmodel"):
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            report[module_name] = {"available": False, "reason": "not installed"}
            continue
        try:
            module = importlib.import_module(module_name)
            report[module_name] = {
                "available": True,
                "version": str(getattr(module, "__version__", "")),
            }
        except Exception as exc:  # noqa: BLE001
            report[module_name] = {
                "available": False,
                "reason": f"import failed: {type(exc).__name__}: {exc}",
            }
    return report


__all__ = [
    "EXECUTOR_FP16",
    "EXECUTOR_FUSED",
    "EXECUTOR_REGISTRY",
    "EXECUTOR_STORAGE",
    "BaseExecutor",
    "Fp16ReferenceExecutor",
    "FusedDequantExecutor",
    "StorageOnlyExecutor",
    "executor_matrix",
    "get_executor",
    "third_party_executor_report",
]
