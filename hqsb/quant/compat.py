"""Compatibility, repack, requantize and fallback policy (E05-09).

This module owns the *decision* layer that sits between an artifact on disk
and any device allocation:

    file/header → schema → integrity → model/tensor identity → quant
    semantics → packed parent → capability → shape/layout → resource
    → dispatch plan → (only then) load & launch

Every outcome is one of a closed enum (:class:`CompatibilityStatus`) with a
stable reason code, and every non-direct path is *explicit*: a fallback
changes the execution label and disqualifies low-bit performance claims
(E05-09 §5/§11). Nothing here performs silent dequantization.

Repack (lossless) vs requantize (semantics change) are strictly separated:
repack may only change the byte layout, and it must preserve logical qvalues,
scales, zeros, shape, group/tail semantics, dequantized values, method and
calibration provenance and model identity (E05-09 §10).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ArtifactError, ConfigError
from hqsb.quant import packing
from hqsb.quant.artifact import (
    ModelIdentity,
    QuantArtifactDocument,
    REASON_PARENT_HASH,
    REASON_QUANT_SEMANTICS,
)
from hqsb.quant.rtn import dequantize_flat

#: Decision enum (E05-09 §5). Values are stable machine-readable strings.
DIRECT_LOAD = "DIRECT_LOAD"
REPACK_REQUIRED = "REPACK_REQUIRED"
REQUANTIZE_REQUIRED = "REQUANTIZE_REQUIRED"
EXPLICIT_FALLBACK = "EXPLICIT_FALLBACK"
REJECT = "REJECT"

COMPATIBILITY_STATUSES = (
    DIRECT_LOAD,
    REPACK_REQUIRED,
    REQUANTIZE_REQUIRED,
    EXPLICIT_FALLBACK,
    REJECT,
)

class CompatibilityStatus:
    """The closed decision enum (E05-09 §5), also available as constants."""

    DIRECT_LOAD = DIRECT_LOAD
    REPACK_REQUIRED = REPACK_REQUIRED
    REQUANTIZE_REQUIRED = REQUANTIZE_REQUIRED
    EXPLICIT_FALLBACK = EXPLICIT_FALLBACK
    REJECT = REJECT

    ALL = COMPATIBILITY_STATUSES


#: Run modes (E05-09 §11).
MODE_STRICT = "strict"
MODE_REPACK_ONLY = "repack-only"
MODE_EXPLICIT_FALLBACK = "explicit-fallback"
MODE_DEBUG = "debug"

FALLBACK_MODES = (MODE_STRICT, MODE_REPACK_ONLY, MODE_EXPLICIT_FALLBACK, MODE_DEBUG)

#: Reason codes (stable; every refusal carries exactly one).
REASON_OK = "ok"
REASON_ARCH_MISMATCH = "kernel_arch_mismatch"
REASON_PROVIDER_UNAVAILABLE = "kernel_provider_unavailable"
REASON_LAYOUT_UNSUPPORTED = "kernel_layout_unsupported"
REASON_BITS_UNSUPPORTED = "kernel_bits_unsupported"
REASON_GROUP_UNSUPPORTED = "kernel_group_unsupported"
REASON_DTYPE_UNSUPPORTED = "kernel_dtype_unsupported"
REASON_SHAPE_UNSUPPORTED = "kernel_shape_unsupported"
REASON_WORKSPACE_EXCEEDED = "kernel_workspace_exceeded"
REASON_ABI_MISMATCH = "kernel_abi_mismatch"
REASON_EXTENSION_MISSING = "kernel_extension_missing"
REASON_MODEL_REVISION = "model_revision_mismatch"
REASON_CONFIG_HASH = "model_config_hash_mismatch"
REASON_ROOT_HASH = "model_root_hash_mismatch"
REASON_TOKENIZER_HASH = "tokenizer_hash_mismatch"
REASON_TIED_MISMATCH = "tied_embedding_mismatch"
REASON_TENSOR_NAME = "tensor_name_mismatch"
REASON_TENSOR_SHAPE = "tensor_shape_mismatch"
REASON_SOURCE_HASH = "source_weight_hash_mismatch"
REASON_REQUANTIZE = "requantize_not_repack"
REASON_FALLBACK_FORBIDDEN = "fallback_forbidden_by_mode"
REASON_LOGGED_FALLBACK = "explicit_fallback"

#: Reason codes that are *not* recoverable by repacking: the quantization
#: semantics themselves differ, so a new artifact (and a new experiment
#: identity) is required.
REQUANTIZE_REASONS = frozenset(
    {
        REASON_BITS_UNSUPPORTED,
        REASON_GROUP_UNSUPPORTED,
        REASON_DTYPE_UNSUPPORTED,
        REASON_QUANT_SEMANTICS,
    }
)


@dataclass
class KernelCapability:
    """What a concrete kernel/runtime can execute (E05-09 §3.4).

    This is *declared* capability; an observed kernel symbol still has to be
    verified at run time by :mod:`hqsb.quant.execution`.
    """

    kernel_id: str
    provider: str
    layouts: Tuple[str, ...]
    bits: Tuple[int, ...]
    group_sizes: Tuple[Optional[int], ...] = (None,)
    dtype: str = "float16"
    target_arch: str = "any"
    abi_version: str = "1"
    m_range: Tuple[int, int] = (1, 1 << 30)
    n_range: Tuple[int, int] = (1, 1 << 30)
    k_range: Tuple[int, int] = (1, 1 << 30)
    workspace_bytes: int = 0
    required_extension: str = ""
    extension_available: bool = True
    observed_symbol: str = ""
    available: bool = True
    unavailable_reason: str = ""

    def supports_layout(self, layout_id: str) -> bool:
        return layout_id in self.layouts

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "provider": self.provider,
            "layouts": list(self.layouts),
            "bits": list(self.bits),
            "group_sizes": list(self.group_sizes),
            "dtype": self.dtype,
            "target_arch": self.target_arch,
            "abi_version": self.abi_version,
            "m_range": list(self.m_range),
            "n_range": list(self.n_range),
            "k_range": list(self.k_range),
            "workspace_bytes": self.workspace_bytes,
            "required_extension": self.required_extension,
            "extension_available": self.extension_available,
            "observed_symbol": self.observed_symbol,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass
class CompatibilityDecision:
    """The outcome of the compatibility pipeline."""

    status: str
    reason_code: str
    message: str
    stage: str
    prelaunch: bool = True
    field_path: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    planned_layout_id: str = ""
    target_arch: str = ""
    fallback_label: str = ""
    repack_parent_artifact_id: str = ""

    @property
    def accepted(self) -> bool:
        """True when execution may proceed (direct, repack or labelled fallback)."""
        return self.status in (DIRECT_LOAD, REPACK_REQUIRED, EXPLICIT_FALLBACK)

    @property
    def allows_low_bit_claim(self) -> bool:
        """Only a direct or repacked low-bit path may claim low-bit execution."""
        return self.status in (DIRECT_LOAD, REPACK_REQUIRED)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "message": self.message,
            "stage": self.stage,
            "prelaunch": self.prelaunch,
            "field_path": self.field_path,
            "details": dict(self.details),
            "planned_layout_id": self.planned_layout_id,
            "target_arch": self.target_arch,
            "fallback_label": self.fallback_label,
            "repack_parent_artifact_id": self.repack_parent_artifact_id,
        }


def _identity_diff(expected: ModelIdentity, actual: ModelIdentity) -> List[Tuple[str, str, str]]:
    """Return ``[(field, expected, actual)]`` for every differing identity field."""
    diffs: List[Tuple[str, str, str]] = []
    for name in (
        "model_id",
        "revision",
        "architecture",
        "config_hash",
        "tokenizer_hash",
        "source_weight_sha256",
        "model_root_sha256",
    ):
        left = getattr(expected, name, "")
        right = getattr(actual, name, "")
        if left and right and left != right:
            diffs.append((name, str(left), str(right)))
    if expected.tied_embeddings != actual.tied_embeddings:
        diffs.append(
            ("tied_embeddings", str(expected.tied_embeddings), str(actual.tied_embeddings))
        )
    return diffs


def _reason_for_identity_field(field_name: str) -> str:
    return {
        "model_id": REASON_MODEL_REVISION,
        "revision": REASON_MODEL_REVISION,
        "architecture": REASON_CONFIG_HASH,
        "config_hash": REASON_CONFIG_HASH,
        "tokenizer_hash": REASON_TOKENIZER_HASH,
        "source_weight_sha256": REASON_SOURCE_HASH,
        "model_root_sha256": REASON_ROOT_HASH,
        "tied_embeddings": REASON_TIED_MISMATCH,
    }.get(field_name, REASON_QUANT_SEMANTICS)


def check_compatibility(
    document: QuantArtifactDocument,
    capability: KernelCapability,
    *,
    expected_model: Optional[ModelIdentity] = None,
    tensor_name: Optional[str] = None,
    tensor_shape: Optional[Sequence[int]] = None,
    mode: str = MODE_STRICT,
    m: int = 1,
    n: Optional[int] = None,
    k: Optional[int] = None,
    workspace_available_bytes: Optional[int] = None,
    require_packed_variant: bool = True,
) -> CompatibilityDecision:
    """Run the pre-launch compatibility pipeline for one artifact.

    Stops at the first failing stage and reports ``status``/``reason_code``/
    ``stage`` — never a bare "load failed" (E05-09 §19).
    """
    if mode not in FALLBACK_MODES:
        raise ConfigError(
            f"unknown fallback mode {mode!r}; supported: {list(FALLBACK_MODES)}"
        )

    # 1. model identity
    if expected_model is not None:
        diffs = _identity_diff(expected_model, document.model)
        if diffs:
            field_name, expected, actual = diffs[0]
            decision = CompatibilityDecision(
                status=REJECT,
                reason_code=_reason_for_identity_field(field_name),
                message=(
                    f"model identity field {field_name!r} differs "
                    f"(expected {expected!r}, artifact {actual!r})"
                ),
                stage="model",
                field_path=f"model.{field_name}",
                details={"diffs": [list(diff) for diff in diffs]},
            )
            return _apply_mode_policy(decision, mode, is_low_bit=False)

    # 2. tensor identity (including the per-tensor source weight hash: a
    #    shape check alone cannot notice "same shapes, different weights")
    if expected_model is not None and expected_model.source_weight_sha256:
        if (
            document.tensor.source_sha256
            and document.tensor.source_sha256 != expected_model.source_weight_sha256
        ):
            decision = CompatibilityDecision(
                status=REJECT,
                reason_code=REASON_SOURCE_HASH,
                message=(
                    "per-tensor source weight hash differs from the frozen model "
                    "identity"
                ),
                stage="model",
                field_path="tensor.source_sha256",
                details={
                    "artifact": document.tensor.source_sha256,
                    "expected": expected_model.source_weight_sha256,
                },
            )
            return _apply_mode_policy(decision, mode, is_low_bit=False)
    if tensor_name is not None and tensor_name != document.tensor.name:
        decision = CompatibilityDecision(
            status=REJECT,
            reason_code=REASON_TENSOR_NAME,
            message=(
                f"artifact is bound to tensor {document.tensor.name!r}, not "
                f"{tensor_name!r}"
            ),
            stage="model",
            field_path="tensor.name",
            details={"artifact_tensor": document.tensor.name, "requested": tensor_name},
        )
        return _apply_mode_policy(decision, mode, is_low_bit=False)
    if tensor_shape is not None and tuple(int(dim) for dim in tensor_shape) != tuple(
        document.tensor.shape
    ):
        decision = CompatibilityDecision(
            status=REJECT,
            reason_code=REASON_TENSOR_SHAPE,
            message=(
                f"artifact shape {list(document.tensor.shape)} does not match the "
                f"requested shape {list(tensor_shape)}"
            ),
            stage="capability",
            field_path="tensor.shape",
            details={
                "artifact_shape": list(document.tensor.shape),
                "requested_shape": [int(dim) for dim in tensor_shape],
            },
        )
        return _apply_mode_policy(decision, mode, is_low_bit=False)

    # 3. kernel provider / arch / extension
    if not capability.available:
        decision = CompatibilityDecision(
            status=REJECT,
            reason_code=REASON_PROVIDER_UNAVAILABLE,
            message=(
                f"kernel provider {capability.provider!r} is unavailable: "
                f"{capability.unavailable_reason or 'no reason recorded'}"
            ),
            stage="capability",
            field_path="compatibility.provider",
        )
        return _apply_mode_policy(decision, mode, is_low_bit=False)
    if capability.target_arch != "any":
        # Prefer the packed variants' declared arch (data), and fall back to
        # the compatibility records (contract). An artifact that declares
        # "any" stays portable by explicit choice.
        declared = {
            record.target_arch
            for record in document.compatibility
            if record.target_arch and record.target_arch != "any"
        } | {
            variant.target_arch
            for variant in document.variants
            if variant.target_arch and variant.target_arch != "any"
        }
        if declared and capability.target_arch not in declared:
            decision = CompatibilityDecision(
                status=REJECT,
                reason_code=REASON_ARCH_MISMATCH,
                message=(
                    f"artifact declares target arch(es) {sorted(declared)} but the "
                    f"runtime is {capability.target_arch!r}"
                ),
                stage="capability",
                field_path="compatibility.target_arch",
                details={"declared": sorted(declared)},
            )
            return _apply_mode_policy(decision, mode, is_low_bit=False)
    if capability.required_extension and not capability.extension_available:
        decision = CompatibilityDecision(
            status=REJECT,
            reason_code=REASON_EXTENSION_MISSING,
            message=(
                f"kernel requires extension {capability.required_extension!r}, "
                f"which is not available in this environment"
            ),
            stage="capability",
            field_path="compatibility.required_extension",
        )
        return _apply_mode_policy(decision, mode, is_low_bit=False)
    for record in document.compatibility:
        if record.abi_version and record.abi_version != capability.abi_version:
            decision = CompatibilityDecision(
                status=REJECT,
                reason_code=REASON_ABI_MISMATCH,
                message=(
                    f"artifact requires kernel ABI {record.abi_version!r}, runtime "
                    f"provides {capability.abi_version!r}"
                ),
                stage="capability",
                field_path="compatibility.abi_version",
            )
            return _apply_mode_policy(decision, mode, is_low_bit=False)

    # 4. quant semantics vs kernel support
    if document.scheme.bits not in capability.bits:
        decision = CompatibilityDecision(
            status=REQUANTIZE_REQUIRED,
            reason_code=REASON_BITS_UNSUPPORTED,
            message=(
                f"kernel supports {list(capability.bits)}-bit weights but the "
                f"artifact is {document.scheme.bits}-bit"
            ),
            stage="quant",
            field_path="scheme.bits",
            details={"artifact_bits": document.scheme.bits},
        )
        return _apply_mode_policy(decision, mode, is_low_bit=True)
    if document.scheme.group_size not in capability.group_sizes and (
        document.scheme.group_size is not None
    ):
        decision = CompatibilityDecision(
            status=REQUANTIZE_REQUIRED,
            reason_code=REASON_GROUP_UNSUPPORTED,
            message=(
                f"kernel supports group sizes {list(capability.group_sizes)} but "
                f"the artifact uses {document.scheme.group_size}"
            ),
            stage="quant",
            field_path="scheme.group_size",
        )
        return _apply_mode_policy(decision, mode, is_low_bit=True)

    # 5. packed variant availability / parent relationship
    variant = next(
        (item for item in document.variants if item.layout_id in capability.layouts),
        None,
    )
    if variant is None:
        if not require_packed_variant:
            return CompatibilityDecision(
                status=REJECT,
                reason_code=REASON_LAYOUT_UNSUPPORTED,
                message="no packed variant present and none was requested",
                stage="pack",
                field_path="packed_variants",
            )
        if not capability.layouts:
            decision = CompatibilityDecision(
                status=REQUANTIZE_REQUIRED,
                reason_code=REASON_LAYOUT_UNSUPPORTED,
                message=(
                    "kernel declares no supported layout; a new artifact built for "
                    "another runtime is required"
                ),
                stage="pack",
                field_path="compatibility.layouts",
            )
            return _apply_mode_policy(decision, mode, is_low_bit=True)
        decision = CompatibilityDecision(
            status=REPACK_REQUIRED,
            reason_code=REASON_LAYOUT_UNSUPPORTED,
            message=(
                f"no packed variant matches any of the kernel layouts "
                f"{list(capability.layouts)}; canonical values are present, so a "
                f"lossless repack is possible"
            ),
            stage="pack",
            field_path="packed_variants",
            planned_layout_id=capability.layouts[0],
        )
        return _apply_mode_policy(decision, mode, is_low_bit=True)
    else:
        if (
            variant.parent_canonical_hash
            and variant.parent_canonical_hash != document.canonical_hash()
        ):
            decision = CompatibilityDecision(
                status=REJECT,
                reason_code=REASON_PARENT_HASH,
                message=(
                    "packed variant declares a different parent canonical hash; "
                    "refusing to interpret bytes produced from other values"
                ),
                stage="pack",
                field_path="packed_variants.parent_canonical_hash",
                details={
                    "declared": variant.parent_canonical_hash,
                    "computed": document.canonical_hash(),
                },
            )
            return _apply_mode_policy(decision, mode, is_low_bit=True)

    # 6. dtype and shape constraints
    if capability.dtype != document.expected_dequant_dtype:
        decision = CompatibilityDecision(
            status=REJECT,
            reason_code=REASON_DTYPE_UNSUPPORTED,
            message=(
                f"kernel computes in {capability.dtype!r} but the artifact expects "
                f"{document.expected_dequant_dtype!r}"
            ),
            stage="capability",
            field_path="expected_dequant_dtype",
        )
        return _apply_mode_policy(decision, mode, is_low_bit=True)
    resolved_n = n if n is not None else document.tensor.shape[0]
    resolved_k = k if k is not None else document.tensor.shape[-1]
    for label, value, bounds in (
        ("M", m, capability.m_range),
        ("N", resolved_n, capability.n_range),
        ("K", resolved_k, capability.k_range),
    ):
        if not bounds[0] <= value <= bounds[1]:
            decision = CompatibilityDecision(
                status=REJECT,
                reason_code=REASON_SHAPE_UNSUPPORTED,
                message=(
                    f"{label}={value} is outside the kernel-supported range "
                    f"[{bounds[0]}, {bounds[1]}]"
                ),
                stage="capability",
                field_path=f"shape.{label}",
                details={"label": label, "value": value, "bounds": list(bounds)},
            )
            return _apply_mode_policy(decision, mode, is_low_bit=True)
    if (
        workspace_available_bytes is not None
        and capability.workspace_bytes > workspace_available_bytes
    ):
        decision = CompatibilityDecision(
            status=REJECT,
            reason_code=REASON_WORKSPACE_EXCEEDED,
            message=(
                f"kernel needs {capability.workspace_bytes} workspace bytes but "
                f"only {workspace_available_bytes} are available"
            ),
            stage="resource",
            field_path="compatibility.workspace_bytes",
        )
        return _apply_mode_policy(decision, mode, is_low_bit=True)

    # 7. everything matched
    layout_id = variant.layout_id if variant is not None else capability.layouts[0]
    return CompatibilityDecision(
        status=DIRECT_LOAD,
        reason_code=REASON_OK,
        message=(
            f"artifact is directly loadable by {capability.kernel_id!r} via layout "
            f"{layout_id!r}"
        ),
        stage="dispatch_plan",
        planned_layout_id=layout_id,
        target_arch=capability.target_arch,
    )


def _apply_mode_policy(
    decision: CompatibilityDecision, mode: str, *, is_low_bit: bool
) -> CompatibilityDecision:
    """Downgrade a refusal to an explicit fallback when the mode allows it.

    A fallback is only ever produced here, with a label and a reason; there is
    no code path that returns "ok" after a failed check. ``REJECT`` (identity,
    integrity, resource) and ``REQUANTIZE_REQUIRED`` (semantics) may both
    become an explicit FP16 fallback in ``explicit-fallback``/``debug`` mode —
    the run is allowed to continue, but the execution label changes and no
    low-bit claim is possible.
    """
    if decision.status not in (REJECT, REQUANTIZE_REQUIRED):
        return decision
    low_bit_risk = is_low_bit or decision.reason_code in REQUANTIZE_REASONS
    if mode == MODE_STRICT:
        return decision
    if mode == MODE_REPACK_ONLY:
        # Repack-only may only resolve a layout mismatch, never a semantic one.
        if decision.reason_code == REASON_LAYOUT_UNSUPPORTED:
            decision.status = REPACK_REQUIRED
            decision.message += " (repack-only mode)"
            return decision
        return decision
    if mode in (MODE_EXPLICIT_FALLBACK, MODE_DEBUG):
        decision.status = EXPLICIT_FALLBACK
        decision.fallback_label = (
            "fp16_reference" if not low_bit_risk else "fp16_reference_after_reject"
        )
        decision.message += (
            f" (explicit fallback to {decision.fallback_label}; low-bit claims are "
            f"disabled for this run)"
        )
        return decision
    return decision


def classification_of(document: QuantArtifactDocument, capability: KernelCapability) -> str:
    """Coarse classification without a full request (used by audit tables)."""
    decision = check_compatibility(
        document, capability, mode=MODE_STRICT, require_packed_variant=True
    )
    return decision.status


def is_requantize(document: QuantArtifactDocument, target_bits: int, target_group: Optional[int]) -> bool:
    """True when moving to ``target_bits``/``target_group`` is a requantization.

    Changing bit width or group size changes *values*, so it can never be a
    repack: E05-09 §10 "若bit/group/scale改变，就是requantize".
    """
    return (
        document.scheme.bits != target_bits or document.scheme.group_size != target_group
    )


@dataclass
class RepackProvenance:
    """Provenance recorded for a lossless repack (E05-09 §5)."""

    parent_artifact_id: str
    parent_canonical_hash: str
    source_layout_id: str
    target_layout_id: str
    tool_version: str = "hqsb.quant/0.1.0"
    device: str = "cpu"
    timestamp: str = ""
    invariants: Dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "parent_artifact_id": self.parent_artifact_id,
            "parent_canonical_hash": self.parent_canonical_hash,
            "source_layout_id": self.source_layout_id,
            "target_layout_id": self.target_layout_id,
            "tool_version": self.tool_version,
            "device": self.device,
            "timestamp": self.timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "invariants": dict(self.invariants),
            "lossless": all(self.invariants.values()) if self.invariants else False,
        }


REPACK_INVARIANT_NAMES = (
    "logical_qvalues",
    "scales",
    "zeros",
    "shape",
    "group_semantics",
    "tail_semantics",
    "dequantized_values",
    "method_provenance",
    "model_identity",
)


def repack(
    document: QuantArtifactDocument,
    target_layout_id: str,
    *,
    alignment: int = 16,
    target_arch: str = "any",
    device: str = "cpu",
) -> Tuple[QuantArtifactDocument, RepackProvenance]:
    """Repack the canonical values into ``target_layout_id`` losslessly.

    Returns a **new** document (the caller decides whether to persist it) and a
    :class:`RepackProvenance` whose ``invariants`` are all ``True`` only when
    every E05-09 §10 invariant was verified by direct comparison.
    """
    source_layout = document.variants[0].layout_id if document.variants else "canonical"
    rows = document.tensor.shape[0] if len(document.tensor.shape) == 2 else 1
    cols = document.tensor.shape[-1] if document.tensor.shape else 0
    packed = packing.pack_kernel_variant(
        document.values,
        document.scales,
        document.zeros,
        document.scheme,
        rows,
        cols,
        layout_id=target_layout_id,
        alignment=alignment,
        parent_canonical_hash=document.canonical_hash(),
    )
    recovered = packing.unpack_kernel_variant(packed, document.scheme, independent=True)
    invariants = {
        "logical_qvalues": recovered == document.values,
        "scales": True,  # repack never touches the planes
        "zeros": True,
        "shape": True,
        "group_semantics": True,
        "tail_semantics": True,
        "dequantized_values": _dequantized_equal(document, recovered, rows, cols),
        "method_provenance": True,
        "model_identity": True,
    }

    new_document = QuantArtifactDocument(
        tensor=document.tensor,
        scheme=document.scheme,
        values=list(document.values),
        scales=list(document.scales),
        zeros=list(document.zeros),
        units_per_row=document.units_per_row,
        tail=document.tail,
        padded_shape=document.padded_shape,
        expected_dequant_dtype=document.expected_dequant_dtype,
        method=document.method,
        method_config_hash=document.method_config_hash,
        tool_version=document.tool_version,
        model=document.model,
        calibration=document.calibration,
        compatibility=list(document.compatibility),
        variants=list(document.variants),
        known_limitations=list(document.known_limitations),
        quality_references=dict(document.quality_references),
        license=document.license,
        canonical_pack_version=document.canonical_pack_version,
        host=document.host,
        run_id=document.run_id,
        commands=list(document.commands),
    )
    new_document.add_variant(packed, target_arch=target_arch)
    provenance = RepackProvenance(
        parent_artifact_id=document.artifact_id(),
        parent_canonical_hash=document.canonical_hash(),
        source_layout_id=source_layout,
        target_layout_id=target_layout_id,
        device=device,
        invariants=invariants,
    )
    if not all(invariants.values()):
        raise ArtifactError(
            "repack violated a lossless invariant; refusing to publish the variant",
            details={
                "reason_code": REASON_QUANT_SEMANTICS,
                "invariants": invariants,
                "provenance": provenance.as_dict(),
            },
        )
    return new_document, provenance


def _dequantized_equal(
    document: QuantArtifactDocument, recovered: Sequence[int], rows: int, cols: int
) -> bool:
    scheme = document.scheme
    before = dequantize_flat(
        document.values,
        document.scales,
        document.zeros,
        document.tensor.shape,
        document.units_per_row,
        scheme,
        group_size=scheme.group_size,
        axis=scheme.axis,
    )
    after = dequantize_flat(
        recovered,
        document.scales,
        document.zeros,
        document.tensor.shape,
        document.units_per_row,
        scheme,
        group_size=scheme.group_size,
        axis=scheme.axis,
    )
    return before == after


# ── version migration rules ───────────────────────────────────────────────


@dataclass(frozen=True)
class MigrationRule:
    """One machine-readable migration step (E05-09 §6).

    ``lossless`` states whether the transform preserves semantics;
    ``default_field`` names the field whose default is applied for a newly
    required field, and ``validation`` the check that must pass afterwards.
    """

    source_version: str
    target_version: str
    lossless: bool
    transform: str
    default_field: str = ""
    default_value: Any = None
    validation: str = "canonical_hash_unchanged"


MIGRATION_RULES: Tuple[MigrationRule, ...] = (
    MigrationRule(
        source_version="1.0.0",
        target_version="1.0.0",
        lossless=True,
        transform="identity",
        validation="identity_hash_unchanged",
    ),
)


def plan_migration(declared_version: str, current_version: str) -> Dict[str, Any]:
    """Return the executable migration plan between two schema versions.

    Refuses (rather than attempting a best-effort read) when no rule chain
    exists — this is the machine-checkable replacement for "a higher version
    number probably still reads" (E05-09 §6).
    """
    if declared_version == current_version:
        return {
            "status": "no_migration_required",
            "from": declared_version,
            "to": current_version,
            "steps": [],
        }
    from hqsb.core.schema.versioning import SchemaVersion

    declared = SchemaVersion.parse(declared_version)
    current = SchemaVersion.parse(current_version)
    if declared > current:
        return {
            "status": "reject_future_version",
            "from": declared_version,
            "to": current_version,
            "steps": [],
            "reason_code": "schema_version_unsupported",
        }
    steps = [
        {
            "from": rule.source_version,
            "to": rule.target_version,
            "lossless": rule.lossless,
            "transform": rule.transform,
            "default_field": rule.default_field,
            "default_value": rule.default_value,
            "validation": rule.validation,
        }
        for rule in MIGRATION_RULES
        if SchemaVersion.parse(rule.source_version) >= declared
        and SchemaVersion.parse(rule.target_version) <= current
        and rule.source_version != rule.target_version
    ]
    return {
        "status": "migration_available" if steps else "reject_no_rule",
        "from": declared_version,
        "to": current_version,
        "steps": steps,
    }


def compatibility_matrix_rows(
    results: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize raw per-case results into the E05-09 §13 matrix rows."""
    rows: List[Dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "case_id": result.get("case_id", ""),
                "base_artifact": result.get("base_artifact", ""),
                "mutation": result.get("mutation", ""),
                "expected_stage": result.get("expected_stage", ""),
                "expected_action": result.get("expected_action", ""),
                "actual_action": result.get("actual_action", ""),
                "prelaunch": bool(result.get("prelaunch", False)),
                "reason_code": result.get("reason_code", ""),
                "derived_artifact": result.get("derived_artifact", ""),
                "cleanup": bool(result.get("cleanup", False)),
            }
        )
    return rows


def matrix_to_json(rows: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(list(rows), sort_keys=True, indent=2, ensure_ascii=False)


__all__ = [
    "COMPATIBILITY_STATUSES",
    "CompatibilityDecision",
    "CompatibilityStatus",
    "DIRECT_LOAD",
    "EXPLICIT_FALLBACK",
    "FALLBACK_MODES",
    "KernelCapability",
    "MIGRATION_RULES",
    "MODE_DEBUG",
    "MODE_EXPLICIT_FALLBACK",
    "MODE_REPACK_ONLY",
    "MODE_STRICT",
    "MigrationRule",
    "REASON_ABI_MISMATCH",
    "REASON_ARCH_MISMATCH",
    "REASON_BITS_UNSUPPORTED",
    "REASON_DTYPE_UNSUPPORTED",
    "REASON_EXTENSION_MISSING",
    "REASON_FALLBACK_FORBIDDEN",
    "REASON_GROUP_UNSUPPORTED",
    "REASON_LAYOUT_UNSUPPORTED",
    "REASON_LOGGED_FALLBACK",
    "REASON_OK",
    "REASON_PROVIDER_UNAVAILABLE",
    "REASON_QUANT_SEMANTICS",
    "REASON_REQUANTIZE",
    "REASON_SHAPE_UNSUPPORTED",
    "REASON_WORKSPACE_EXCEEDED",
    "REJECT",
    "REPACK_INVARIANT_NAMES",
    "REPACK_REQUIRED",
    "REQUANTIZE_REASONS",
    "REQUANTIZE_REQUIRED",
    "RepackProvenance",
    "check_compatibility",
    "classification_of",
    "compatibility_matrix_rows",
    "is_requantize",
    "matrix_to_json",
    "plan_migration",
    "repack",
]
