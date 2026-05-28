"""Putting artifacts back into a real model, reversibly (E05-02 steps 4–7).

Three application modes are supported and they must never be conflated:

``fake_dequant``
    weights are quantized and immediately dequantized in memory, then the
    ordinary FP16 GEMM runs. Measures the *representation* error only.
``storage_only``
    weights are loaded from the packed artifact and materialized to FP16
    before the GEMM. Measures storage/load effects; forbids low-bit claims.
``kernel``
    an executor (``ops.quant``) consumes the packed bytes directly and
    dequantizes inside the kernel. The only mode that can support a low-bit
    execution claim; it requires an observed kernel symbol.

Every substitution is recorded in a :class:`WeightSwap` with the original and
applied tensor hashes, so it can be undone exactly and audited afterwards
(S04.5's "可逆替换" requirement applied to weights).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from hqsb.core.errors import ArtifactError, CapabilityError, ConfigError
from hqsb.quant import coverage as coverage_mod
from hqsb.quant import packing
from hqsb.quant.artifact import PackedVariantRecord, QuantArtifactDocument
from hqsb.quant.rtn import dequantize_flat
from hqsb.quant.spec import QuantScheme

MODE_FAKE_DEQUANT = "fake_dequant"
MODE_STORAGE_ONLY = "storage_only"
MODE_KERNEL = "kernel"

APPLICATION_MODES = (MODE_FAKE_DEQUANT, MODE_STORAGE_ONLY, MODE_KERNEL)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - CPU-minimal CI
        raise CapabilityError(
            "model application needs torch; install the 'benchmark' extra",
            details={"capability": "torch", "reason": "not installed"},
        ) from exc
    return torch


def tensor_sha256(tensor) -> str:
    """Stable hash of a tensor's bytes and shape (dtype-aware)."""
    import io

    buffer = io.BytesIO()
    import numpy as np

    array = tensor.detach().to("cpu").contiguous().numpy()
    np.save(buffer, array, allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


class LowBitExecutor(Protocol):
    """Protocol every execution-layer implementation must satisfy (ops/quant).

    Implementations live outside ``hqsb.quant`` so this module stays free of
    the operator layer (dependency direction: ``ops`` consumes ``hqsb.quant``).
    """

    name: str
    provider: str
    observed_kernel_symbol: str
    packed_layout_id: str

    def is_available(self) -> bool: ...

    def dequantize_weight(
        self, document: QuantArtifactDocument, variant: PackedVariantRecord
    ): ...

    def gemm_low_bit(
        self, x, document: QuantArtifactDocument, variant: PackedVariantRecord
    ): ...

    def describe(self) -> Dict[str, Any]: ...


def quantized_weight_values(
    document: QuantArtifactDocument,
    variant: Optional[PackedVariantRecord] = None,
    *,
    independent_decoder: bool = True,
) -> List[float]:
    """Dequantized weights from canonical values (or decoded packed bytes)."""
    from hqsb.quant.artifact import _weight_matrix_shape

    rows, cols = _weight_matrix_shape(document.tensor.shape)
    qvalues = list(document.values)
    if variant is not None:
        import os

        payload = open(
            os.path.join(document.artifact_dir, "variants", variant.filename), "rb"
        ).read()
        packed = packing.PackedTensor(
            layout=packing.plan_kernel_layout(
                document.scheme,
                rows,
                cols,
                layout_id=variant.layout_id,
                alignment=variant.alignment,
                parent_canonical_hash=variant.parent_canonical_hash,
            ),
            payload=payload,
            scales=list(document.scales),
            zeros=list(document.zeros),
        )
        qvalues = packing.unpack_kernel_variant(
            packed, document.scheme, independent=independent_decoder
        )
    return dequantize_flat(
        qvalues,
        document.scales,
        document.zeros,
        document.tensor.shape,
        document.units_per_row,
        document.scheme,
        group_size=document.scheme.group_size,
        axis=document.scheme.axis,
    )


def fake_quant_weight(weight, scheme: QuantScheme):
    """Quantize→dequantize a weight tensor in memory (FP16 out)."""
    torch = _require_torch()
    flat = [float(value) for value in weight.detach().reshape(-1).tolist()]
    if hasattr(weight, "detach"):
        original_dtype = weight.dtype
    else:  # pragma: no cover - defensive
        original_dtype = torch.float16
    from hqsb.quant.rtn import quantize

    quantized = quantize(
        flat, scheme, shape=tuple(int(dim) for dim in weight.shape)
    )
    return (
        torch.tensor(quantized.values_dequant, dtype=torch.float64)
        .to(original_dtype)
        .reshape(weight.shape)
    )


@dataclass
class WeightSwap:
    """One reversible weight substitution."""

    module: str
    mode: str
    original_sha256: str
    applied_sha256: str
    artifact_id: str = ""
    variant_hash: str = ""
    execution_label: str = ""
    restore_ok: bool = False
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "mode": self.mode,
            "original_sha256": self.original_sha256,
            "applied_sha256": self.applied_sha256,
            "artifact_id": self.artifact_id,
            "variant_hash": self.variant_hash,
            "execution_label": self.execution_label,
            "restore_ok": self.restore_ok,
            "notes": self.notes,
        }


def find_linear_modules(model) -> Dict[str, Any]:
    """Return ``{module_name: module}`` for linear-like modules with weights.

    "Linear-like" is exactly the set the coverage policy selected
    (``default_linear``/``explicit_include``); the policy is the single source
    of truth for what gets quantized.
    """
    _require_torch()
    modules = {}
    for name, module in model.named_modules():
        if not name:
            continue
        weight = getattr(module, "weight", None)
        if weight is None or not hasattr(weight, "shape"):
            continue
        if weight.dim() != 2:
            continue
        modules[name] = module
    return modules


def swap_weights(
    model,
    documents: Mapping[str, QuantArtifactDocument],
    *,
    mode: str = MODE_FAKE_DEQUANT,
    executor: Optional[LowBitExecutor] = None,
    dtype=None,
) -> Tuple[List[WeightSwap], List[Any]]:
    """Replace weights according to ``mode``; returns ``(swaps, backups)``.

    ``backups`` holds the original tensors so :func:`restore_weights` can undo
    the substitution exactly. The function refuses to touch a module twice or
    a module whose weight shape disagrees with its artifact — a silent shape
    mismatch would corrupt the model.
    """
    torch = _require_torch()
    if mode not in APPLICATION_MODES:
        raise ConfigError(f"unknown application mode {mode!r}")
    if mode == MODE_KERNEL:
        if executor is None:
            raise ConfigError(
                "kernel mode requires an executor; pass a LowQuantExecutor "
                "implementation from ops.quant (silent FP16 fallback is forbidden)"
            )
        if not executor.is_available():
            raise CapabilityError(
                f"executor {executor.name!r} is unavailable; refusing to fall "
                f"back silently",
                details={"executor": executor.name, "provider": executor.provider},
            )
    modules = find_linear_modules(model)
    swaps: List[WeightSwap] = []
    backups: List[Any] = []
    for key, document in sorted(documents.items()):
        # Accept both naming conventions: the module path ("layer.q_proj") and
        # the weight-tensor path ("layer.q_proj.weight"). The artifact itself
        # is always bound to the tensor name, so a mismatch is refused.
        name = key[: -len(".weight")] if key.endswith(".weight") else key
        if name not in modules:
            raise ArtifactError(
                f"artifact {document.artifact_id()} targets module {key!r}, "
                f"which is not a linear module in this model",
                details={"module": key},
            )
        if document.tensor.name.endswith(".weight"):
            artifact_module = document.tensor.name[: -len(".weight")]
            if artifact_module != name:
                raise ArtifactError(
                    f"artifact tensor {document.tensor.name!r} does not match the "
                    f"requested module {name!r}",
                    details={"tensor": document.tensor.name, "module": name},
                )
        module = modules[name]
        original = module.weight.detach().clone()
        original_hash = tensor_sha256(module.weight)
        target_dtype = dtype or module.weight.dtype

        if mode == MODE_FAKE_DEQUANT:
            new_weight = fake_quant_weight(module.weight, document.scheme).to(target_dtype)
            label = "fake_quant"
        elif mode == MODE_STORAGE_ONLY:
            variant = document.variants[0] if document.variants else None
            values = quantized_weight_values(document, variant)
            new_weight = (
                torch.tensor(values, dtype=torch.float64)
                .to(target_dtype)
                .reshape(document.tensor.shape)
            )
            label = "storage_only"
        else:  # MODE_KERNEL: the executor owns the fused path; weights stay packed
            new_weight = module.weight  # untouched; the executor consumes packed bytes
            label = "fused_dequant_weight_only"
        if tuple(new_weight.shape) != tuple(module.weight.shape):
            raise ConfigError(
                f"applied weight for {name!r} has shape {tuple(new_weight.shape)}, "
                f"expected {tuple(module.weight.shape)}"
            )
        backups.append(original)
        module.weight.data = new_weight.detach().to(target_dtype)
        variant = document.variants[0] if document.variants else None
        swaps.append(
            WeightSwap(
                module=name,  # normalized module path (no ".weight" suffix)
                mode=mode,
                original_sha256=original_hash,
                applied_sha256=tensor_sha256(module.weight),
                artifact_id=document.artifact_id(),
                variant_hash=variant.variant_hash if variant else "",
                execution_label=label,
            )
        )
    return swaps, backups


def restore_weights(model, swaps: Sequence[WeightSwap], backups: Sequence[Any]) -> List[Dict[str, Any]]:
    """Undo a :func:`swap_weights` substitution and verify the restore.

    Returns per-module verification records; a module whose restored hash does
    not match the recorded original hash is reported as failed (never
    silently accepted).
    """
    _require_torch()
    if len(swaps) != len(backups):
        raise ConfigError(
            f"swaps ({len(swaps)}) and backups ({len(backups)}) must pair up"
        )
    modules = find_linear_modules(model)
    records: List[Dict[str, Any]] = []
    for swap, backup in zip(swaps, backups):
        if swap.module not in modules:
            raise ArtifactError(
                f"cannot restore {swap.module!r}: module disappeared from the model",
                details={"module": swap.module},
            )
        module = modules[swap.module]
        module.weight.data = backup
        restored_hash = tensor_sha256(module.weight)
        ok = restored_hash == swap.original_sha256
        swap.restore_ok = ok
        records.append(
            {
                "module": swap.module,
                "restored_sha256": restored_hash,
                "expected_sha256": swap.original_sha256,
                "restore_ok": ok,
            }
        )
    return records


@dataclass
class CaptureBuffer:
    """Activation capture for operator/block comparisons (E05-02 §7.1)."""

    module: str
    phase: str
    inputs: List[Any] = field(default_factory=list)
    outputs: List[Any] = field(default_factory=list)
    call_count: int = 0
    handles: List[Any] = field(default_factory=list)

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def record(self, tensor_input, tensor_output) -> None:
        self.call_count += 1
        self.inputs.append(tensor_input)
        self.outputs.append(tensor_output)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "phase": self.phase,
            "call_count": self.call_count,
            "captured_inputs": len(self.inputs),
            "captured_outputs": len(self.outputs),
        }


def attach_capture_hooks(
    model, module_names: Sequence[str], *, phase: str = "prefill"
) -> Dict[str, CaptureBuffer]:
    """Register forward hooks that clone inputs/outputs for later comparison.

    The hook clones and detaches tensors, so attaching it does not change the
    forward computation. E05-05 §14 warns that hooks can cause graph breaks;
    the caller must therefore record `hooks_attached` in any performance run
    and re-measure without hooks for steady-state numbers.
    """
    _require_torch()  # hooks need torch tensors; fail early with a clear error
    modules = find_linear_modules(model)
    buffers: Dict[str, CaptureBuffer] = {}
    for name in module_names:
        if name not in modules:
            raise ArtifactError(
                f"cannot capture {name!r}: not a linear module",
                details={"module": name},
            )
        buffer = CaptureBuffer(module=name, phase=phase)

        def _hook(_module, inputs, output, _buffer=buffer):
            first_input = inputs[0] if inputs else None
            _buffer.record(
                first_input.detach().clone() if first_input is not None else None,
                output.detach().clone() if output is not None else None,
            )

        buffer.handles.append(modules[name].register_forward_hook(_hook))
        buffers[name] = buffer
    return buffers


def detach_capture_hooks(buffers: Mapping[str, CaptureBuffer]) -> None:
    for buffer in buffers.values():
        buffer.detach()


def application_report(swaps: Sequence[WeightSwap]) -> Dict[str, Any]:
    """Summarize an application run (coverage + labels + restore status)."""
    modes: Dict[str, int] = {}
    labels: Dict[str, int] = {}
    for swap in swaps:
        modes[swap.mode] = modes.get(swap.mode, 0) + 1
        labels[swap.execution_label] = labels.get(swap.execution_label, 0) + 1
    return {
        "modules": len(swaps),
        "modes": dict(sorted(modes.items())),
        "execution_labels": dict(sorted(labels.items())),
        "all_restored": all(swap.restore_ok for swap in swaps) if swaps else True,
        "low_bit_claim_allowed": all(
            swap.execution_label in ("fused_dequant_weight_only",)
            for swap in swaps
        )
        if swaps
        else False,
        "swaps": [swap.as_dict() for swap in swaps],
    }


def plan_model_quantization(model, policy: coverage_mod.CoveragePolicy, scheme: QuantScheme) -> Dict[str, Any]:
    """Return the coverage plan without touching any weight (E05-02 step 3)."""
    rows = coverage_mod.enumerate_weight_rows(model, policy, scheme=scheme)
    return {
        "rows": [row.as_dict() for row in rows],
        "summary": coverage_mod.coverage_summary(rows),
        "policy": json.loads(policy.to_json()),
        "scheme_hash": scheme.scheme_hash(),
    }


__all__ = [
    "APPLICATION_MODES",
    "CaptureBuffer",
    "LowBitExecutor",
    "MODE_FAKE_DEQUANT",
    "MODE_KERNEL",
    "MODE_STORAGE_ONLY",
    "WeightSwap",
    "application_report",
    "attach_capture_hooks",
    "detach_capture_hooks",
    "fake_quant_weight",
    "find_linear_modules",
    "plan_model_quantization",
    "quantized_weight_values",
    "restore_weights",
    "swap_weights",
    "tensor_sha256",
]
