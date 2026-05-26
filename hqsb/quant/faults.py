"""Fault-injection matrix for ``QuantArtifact`` compatibility (E05-09 §8).

The matrix is *declarative*: every case names the field/file/byte it mutates,
the stage where it must be caught, the expected action and the expected reason
code. The runner copies the golden artifact into a scratch directory, applies
one mutation to the copy (never to the golden), runs the pre-launch validation
chain, and records what actually happened.

A case is reported as ``caught`` only when

* the refusal happened in the expected stage,
* before any device allocation/launch (``prelaunch=True`` by construction:
  the runner never touches a device), and
* with the expected reason code.

Anything else is a finding, not a pass — the experiment driver must treat a
mismatch as a failure of the artifact gate.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from hqsb.quant import compat
from hqsb.quant.artifact import (
    MANIFEST_FILENAME,
    REASON_MANIFEST_FIELD,
    REASON_SCHEMA_VERSION,
    REASON_FILE_HASH,
    REASON_QUANT_SEMANTICS,
    ModelIdentity,
    load_document,
    validate_artifact_dir,
)

#: Mutation categories (E05-09 §8.1–§8.5).
CATEGORY_SCHEMA = "schema_integrity"
CATEGORY_MODEL = "model_mismatch"
CATEGORY_QUANT = "quant_mismatch"
CATEGORY_PACKING = "packing_mismatch"
CATEGORY_CAPABILITY = "kernel_capability"

CATEGORIES = (
    CATEGORY_SCHEMA,
    CATEGORY_MODEL,
    CATEGORY_QUANT,
    CATEGORY_PACKING,
    CATEGORY_CAPABILITY,
)


def _read_manifest(artifact_dir: str) -> Dict[str, Any]:
    with open(os.path.join(artifact_dir, MANIFEST_FILENAME), encoding="utf-8") as handle:
        return json.load(handle)


def _write_manifest(artifact_dir: str, payload: Mapping[str, Any]) -> None:
    with open(os.path.join(artifact_dir, MANIFEST_FILENAME), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)


def _rewrite_canonical_hash(artifact_dir: str) -> None:
    """Refresh ``canonical_hash`` after a semantics mutation.

    Needed by the cases whose point is *declared semantics vs executable
    kernel* rather than internal consistency: with the hash refreshed, only
    the capability layer can notice that the artifact declares 8-bit weights
    while the kernel executes 4-bit (→ ``REQUANTIZE_REQUIRED``).
    """
    from hqsb.quant.artifact import load_document

    document = load_document(
        artifact_dir,
        verify_payloads=True,
        verify_variants=False,
        enforce_canonical_hash=False,
    )
    manifest = _read_manifest(artifact_dir)
    manifest["canonical_hash"] = document.canonical_hash()
    # The packed bytes belong to the *old* semantics, so they are dropped: the
    # case models "a canonical artifact whose declared scheme differs from the
    # kernel", which must be answered with REQUANTIZE_REQUIRED, not with a
    # silent reinterpretation of the old bytes.
    manifest["packed_variants"] = []
    _write_manifest(artifact_dir, manifest)
    variants_dir = os.path.join(artifact_dir, "variants")
    if os.path.isdir(variants_dir):
        shutil.rmtree(variants_dir)


def _rewrite_payload_hash(artifact_dir: str, relative: str, key: str) -> None:
    """Update the manifest hash of a payload the mutation changed.

    Used by the cases that target *semantic* validation rather than integrity:
    with the hash refreshed, the artifact is internally consistent and only a
    semantic check (codes, scales, parent hash) can catch the mutation.
    """
    import hashlib

    path = os.path.join(artifact_dir, relative)
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    manifest = _read_manifest(artifact_dir)
    if key == "canonical":
        manifest["canonical_file"]["sha256"] = digest
    elif key == "scales":
        manifest["scales_file"]["sha256"] = digest
    elif key == "variant":
        for record in manifest.get("packed_variants", []):
            if record["filename"] == os.path.basename(relative):
                record["payload_sha256"] = digest
    else:  # pragma: no cover - defensive
        raise KeyError(key)
    _write_manifest(artifact_dir, manifest)


def _variant_path(artifact_dir: str) -> Optional[str]:
    variants_dir = os.path.join(artifact_dir, "variants")
    if not os.path.isdir(variants_dir):
        return None
    names = sorted(os.listdir(variants_dir))
    return os.path.join("variants", names[0]) if names else None


@dataclass
class MutationCase:
    """One pre-registered fault-injection case."""

    case_id: str
    category: str
    mutation: str
    expected_stage: str
    expected_action: str
    expected_reason_code: str
    apply: Callable[[str], None]
    capability_variant: Optional[Dict[str, Any]] = field(default=None)
    notes: str = ""
    #: When set, the case starts from a synthetic fixture built with this
    #: layout variant instead of copying the passed golden artifact (used by
    #: the cross-layout repack case, which needs a *known* but unsupported
    #: layout rather than a corrupted one).
    fixture_variant: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "mutation": self.mutation,
            "expected_stage": self.expected_stage,
            "expected_action": self.expected_action,
            "expected_reason_code": self.expected_reason_code,
            "capability_variant": dict(self.capability_variant or {}),
            "fixture_variant": self.fixture_variant,
            "notes": self.notes,
        }


# ── mutation appliers ─────────────────────────────────────────────────────


def _mutate_schema_version_future(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    manifest["schema_version"] = "2.0.0"
    _write_manifest(artifact_dir, manifest)


def _mutate_kind(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    manifest["kind"] = "hqsb.not.a.quant.artifact"
    _write_manifest(artifact_dir, manifest)


def _mutate_missing_required_field(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    del manifest["canonical_count"]
    _write_manifest(artifact_dir, manifest)


def _mutate_wrong_type(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    manifest["canonical_count"] = "many"
    _write_manifest(artifact_dir, manifest)


def _mutate_bitflip_canonical(artifact_dir: str) -> None:
    path = os.path.join(artifact_dir, "canonical.bin")
    data = bytearray(open(path, "rb").read())
    data[0] ^= 0x01
    open(path, "wb").write(bytes(data))


def _mutate_truncate_canonical(artifact_dir: str) -> None:
    path = os.path.join(artifact_dir, "canonical.bin")
    data = open(path, "rb").read()
    open(path, "wb").write(data[:-1])


def _mutate_extra_payload(artifact_dir: str) -> None:
    """Add an undeclared payload file (must be refused, not ignored)."""
    path = os.path.join(artifact_dir, "canonical.bin")
    data = open(path, "rb").read()
    open(path, "wb").write(data + b"\x00")


def _mutate_bitflip_with_hash_fix(artifact_dir: str) -> None:
    path = os.path.join(artifact_dir, "canonical.bin")
    data = bytearray(open(path, "rb").read())
    data[0] = 0x88  # low nibble 8 -> code -8, illegal for a symmetric scheme
    open(path, "wb").write(bytes(data))
    _rewrite_payload_hash(artifact_dir, "canonical.bin", "canonical")


def _mutate_nan_scale(artifact_dir: str) -> None:
    path = os.path.join(artifact_dir, "scales.bin")
    data = open(path, "rb").read()
    open(path, "wb").write(struct.pack("<f", float("nan")) + data[4:])
    _rewrite_payload_hash(artifact_dir, "scales.bin", "scales")


def _mutate_model_revision(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    manifest["model"]["revision"] = "not-the-frozen-revision"
    _write_manifest(artifact_dir, manifest)


def _mutate_config_hash(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    manifest["model"]["config_hash"] = "a" * 64
    _write_manifest(artifact_dir, manifest)


def _mutate_tokenizer_hash(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    manifest["model"]["tokenizer_hash"] = "f" * 64
    _write_manifest(artifact_dir, manifest)


def _mutate_source_weight_hash(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    manifest["tensor"]["source_sha256"] = "1" * 64
    _write_manifest(artifact_dir, manifest)


def _mutate_tensor_name(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    manifest["tensor"]["name"] = manifest["tensor"]["name"] + ".renamed"
    _write_manifest(artifact_dir, manifest)


def _mutate_tensor_shape(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    shape = list(manifest["tensor"]["shape"])
    shape[-1] = shape[-1] + 2
    manifest["tensor"]["shape"] = shape
    _write_manifest(artifact_dir, manifest)


def _mutate_tied_flag(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    manifest["model"]["tied_embeddings"] = not manifest["model"].get(
        "tied_embeddings", False
    )
    _write_manifest(artifact_dir, manifest)


def _scheme_hash(scheme_payload: Mapping[str, Any]) -> str:
    """Recompute the scheme hash the way the artifact does (for mutations)."""
    from hqsb.quant.spec import QuantScheme

    return QuantScheme.from_mapping(dict(scheme_payload)).scheme_hash()


def _mutate_bits(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    bits = manifest["scheme"]["bits"]
    manifest["scheme"]["bits"] = 8 if bits == 4 else 4
    manifest["scheme_hash"] = _scheme_hash(manifest["scheme"])
    _write_manifest(artifact_dir, manifest)
    _rewrite_canonical_hash(artifact_dir)


def _mutate_group_size(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    group = manifest["scheme"].get("group_size")
    manifest["scheme"]["group_size"] = (group or 128) * 2
    manifest["scheme_hash"] = _scheme_hash(manifest["scheme"])
    _write_manifest(artifact_dir, manifest)
    _rewrite_canonical_hash(artifact_dir)


def _mutate_symmetric_flag(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    manifest["scheme"]["symmetric"] = not manifest["scheme"]["symmetric"]
    _write_manifest(artifact_dir, manifest)


def _mutate_scheme_hash(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    manifest["scheme_hash"] = "0" * 64
    _write_manifest(artifact_dir, manifest)


def _mutate_packing_layout(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    for record in manifest.get("packed_variants", []):
        record["layout_id"] = "hqsb.unknown.layout.v1"
    _write_manifest(artifact_dir, manifest)


def _mutate_packing_layout_known_but_wrong(artifact_dir: str) -> None:
    """Relabel a rowmajor payload as the high-first layout (both are known).

    The bytes are intact and the manifest is self-consistent; only a semantic
    decode comparison can notice that the declared nibble order is wrong.
    """
    from hqsb.quant.packing import LAYOUT_W4A16_HIFIRST_NK_V1

    manifest = _read_manifest(artifact_dir)
    for record in manifest.get("packed_variants", []):
        record["layout_id"] = LAYOUT_W4A16_HIFIRST_NK_V1
    _write_manifest(artifact_dir, manifest)


def _mutate_parent_hash(artifact_dir: str) -> None:
    manifest = _read_manifest(artifact_dir)
    for record in manifest.get("packed_variants", []):
        record["parent_canonical_hash"] = "0" * 64
    _write_manifest(artifact_dir, manifest)


def _mutate_variant_payload(artifact_dir: str) -> None:
    relative = _variant_path(artifact_dir)
    if relative is None:
        return
    path = os.path.join(artifact_dir, relative)
    data = bytearray(open(path, "rb").read())
    data[1] ^= 0xFF
    open(path, "wb").write(bytes(data))


def _mutate_variant_payload_with_hash_fix(artifact_dir: str) -> None:
    relative = _variant_path(artifact_dir)
    if relative is None:
        return
    path = os.path.join(artifact_dir, relative)
    data = bytearray(open(path, "rb").read())
    data[1] = 0xFF  # invalid two's-complement padding/codes for some schemes
    open(path, "wb").write(bytes(data))
    _rewrite_payload_hash(artifact_dir, relative, "variant")


def build_fault_matrix() -> List[MutationCase]:
    """Return the pre-registered E05-09 §8 fault matrix.

    The set is intentionally larger than the minimum: each stage (schema,
    model, quant, pack, capability, resource) and each decision enum value is
    exercised by at least one case whose expected reason code is unique to
    that stage, otherwise "it fails somewhere" would pass an incomplete gate.
    """
    cases: List[MutationCase] = [
        # ── §8.1 schema & integrity ─────────────────────────────────────
        MutationCase(
            "schema_future_version",
            CATEGORY_SCHEMA,
            "schema_version := 2.0.0",
            "schema",
            compat.REJECT,
            REASON_SCHEMA_VERSION,
            _mutate_schema_version_future,
            notes="future major must be refused, not guessed",
        ),
        MutationCase(
            "schema_wrong_kind",
            CATEGORY_SCHEMA,
            "kind := hqsb.not.a.quant.artifact",
            "schema",
            compat.REJECT,
            "artifact_kind_mismatch",
            _mutate_kind,
        ),
        MutationCase(
            "schema_missing_required_field",
            CATEGORY_SCHEMA,
            "delete canonical_count",
            "schema",
            compat.REJECT,
            REASON_MANIFEST_FIELD,
            _mutate_missing_required_field,
        ),
        MutationCase(
            "schema_wrong_type",
            CATEGORY_SCHEMA,
            "canonical_count := 'many'",
            "schema",
            compat.REJECT,
            REASON_MANIFEST_FIELD,
            _mutate_wrong_type,
        ),
        MutationCase(
            "integrity_bitflip",
            CATEGORY_SCHEMA,
            "flip one bit in canonical.bin",
            "schema",
            compat.REJECT,
            REASON_FILE_HASH,
            _mutate_bitflip_canonical,
        ),
        MutationCase(
            "integrity_truncated",
            CATEGORY_SCHEMA,
            "truncate canonical.bin by one byte",
            "schema",
            compat.REJECT,
            REASON_FILE_HASH,
            _mutate_truncate_canonical,
        ),
        MutationCase(
            "integrity_extra_payload",
            CATEGORY_SCHEMA,
            "append one byte to canonical.bin",
            "schema",
            compat.REJECT,
            REASON_FILE_HASH,
            _mutate_extra_payload,
        ),
        MutationCase(
            "quant_illegal_code_with_hash_fix",
            CATEGORY_QUANT,
            "write illegal code -8 into a symmetric artifact and refresh its hash",
            "quant",
            compat.REJECT,
            "code_out_of_range",
            _mutate_bitflip_with_hash_fix,
            notes="with a refreshed hash only the semantic code-range check can catch it",
        ),
        MutationCase(
            "quant_nan_scale_with_hash_fix",
            CATEGORY_QUANT,
            "set scales[0] := NaN and refresh scales hash",
            "quant",
            compat.REJECT,
            REASON_QUANT_SEMANTICS,
            _mutate_nan_scale,
        ),
        # ── §8.2 model mismatch ─────────────────────────────────────────
        MutationCase(
            "model_revision_mismatch",
            CATEGORY_MODEL,
            "model.revision := 'not-the-frozen-revision'",
            "model",
            compat.REJECT,
            compat.REASON_MODEL_REVISION,
            _mutate_model_revision,
        ),
        MutationCase(
            "model_config_hash",
            CATEGORY_MODEL,
            "model.config_hash := 0^64",
            "model",
            compat.REJECT,
            compat.REASON_CONFIG_HASH,
            _mutate_config_hash,
        ),
        MutationCase(
            "model_tokenizer_hash",
            CATEGORY_MODEL,
            "model.tokenizer_hash := f^64",
            "model",
            compat.REJECT,
            compat.REASON_TOKENIZER_HASH,
            _mutate_tokenizer_hash,
        ),
        MutationCase(
            "model_source_weight_hash",
            CATEGORY_MODEL,
            "tensor.source_sha256 := 1^64",
            "model",
            compat.REJECT,
            compat.REASON_SOURCE_HASH,
            _mutate_source_weight_hash,
            notes="same shapes, different weights: only the source hash catches it",
        ),
        MutationCase(
            "model_tensor_name",
            CATEGORY_MODEL,
            "rename tensor",
            "model",
            compat.REJECT,
            compat.REASON_TENSOR_NAME,
            _mutate_tensor_name,
        ),
        MutationCase(
            "model_tensor_shape",
            CATEGORY_MODEL,
            "widen the last tensor dimension by 2",
            "model",
            compat.REJECT,
            "canonical_shape_mismatch",
            _mutate_tensor_shape,
            notes="shape inconsistency is detected while loading the canonical layer",
        ),
        MutationCase(
            "model_tied_flag",
            CATEGORY_MODEL,
            "flip tied_embeddings",
            "model",
            compat.REJECT,
            compat.REASON_TIED_MISMATCH,
            _mutate_tied_flag,
        ),
        # ── §8.3 quant mismatch ─────────────────────────────────────────
        MutationCase(
            "quant_bits_declared_w8",
            CATEGORY_QUANT,
            "scheme.bits := 8 and refresh scheme hash",
            "quant",
            compat.REQUANTIZE_REQUIRED,
            compat.REASON_BITS_UNSUPPORTED,
            _mutate_bits,
            notes="declared semantics differ from the executable kernel: requantize, "
            "never a silent reinterpretation",
        ),
        MutationCase(
            "quant_group_declared_256",
            CATEGORY_QUANT,
            "scheme.group_size *= 2 and refresh scheme hash",
            "quant",
            compat.REQUANTIZE_REQUIRED,
            compat.REASON_GROUP_UNSUPPORTED,
            _mutate_group_size,
        ),
        MutationCase(
            "quant_symmetric_flag",
            CATEGORY_QUANT,
            "flip scheme.symmetric without touching the hash",
            "quant",
            compat.REJECT,
            REASON_QUANT_SEMANTICS,
            _mutate_symmetric_flag,
        ),
        MutationCase(
            "quant_scheme_hash",
            CATEGORY_QUANT,
            "scheme_hash := 0^64",
            "quant",
            compat.REJECT,
            REASON_QUANT_SEMANTICS,
            _mutate_scheme_hash,
        ),
        # ── §8.4 packing mismatch ───────────────────────────────────────
        MutationCase(
            "packing_unknown_layout",
            CATEGORY_PACKING,
            "variant.layout_id := hqsb.unknown.layout.v1",
            "schema",
            compat.REJECT,
            "unknown_layout_id",
            _mutate_packing_layout,
            notes="an id no reader knows is a format error, not a repack request",
        ),
        MutationCase(
            "packing_mislabeled_known_layout",
            CATEGORY_PACKING,
            "relabel a rowmajor payload as the high-first layout",
            "pack",
            compat.REJECT,
            "packed_bytes_value_mismatch",
            _mutate_packing_layout_known_but_wrong,
            notes="both layouts are known; only decoding proves the bytes are wrong",
        ),
        MutationCase(
            "packing_parent_hash",
            CATEGORY_PACKING,
            "variant.parent_canonical_hash := 0^64",
            "pack",
            compat.REJECT,
            "packed_variant_parent_hash_mismatch",
            _mutate_parent_hash,
        ),
        MutationCase(
            "packing_payload_bitflip",
            CATEGORY_PACKING,
            "flip a byte in a packed variant",
            "pack",
            compat.REJECT,
            REASON_FILE_HASH,
            _mutate_variant_payload,
        ),
        MutationCase(
            "packing_payload_with_hash_fix",
            CATEGORY_PACKING,
            "corrupt variant payload and refresh its hash",
            "pack",
            compat.REJECT,
            "packed_bytes_value_mismatch",
            _mutate_variant_payload_with_hash_fix,
            notes="with a refreshed hash only a semantic decode comparison catches it",
        ),
    ]

    cases.append(
        MutationCase(
            "packing_known_layout_unsupported",
            CATEGORY_PACKING,
            "artifact uses the hifirst layout while the kernel supports rowmajor",
            "pack",
            compat.REPACK_REQUIRED,
            compat.REASON_LAYOUT_UNSUPPORTED,
            _noop_mutation,
            fixture_variant="w4_hifirst",
            notes="cross-backend case: canonical values are compatible, the bytes "
            "are not; a lossless repack is required",
        )
    )

    # ── §8.5 kernel capability mismatch (artifact intact) ───────────────
    capability_cases = [
        ("capability_arch", {"target_arch": "sm_99"}, "capability", compat.REASON_ARCH_MISMATCH),
        ("capability_provider", {"available": False}, "capability", compat.REASON_PROVIDER_UNAVAILABLE),
        (
            "capability_extension",
            {"extension_available": False, "required_extension": "fp4"},
            "capability",
            compat.REASON_EXTENSION_MISSING,
        ),
        ("capability_abi", {"abi_version": "2"}, "capability", compat.REASON_ABI_MISMATCH),
        ("capability_dtype", {"dtype": "bfloat16"}, "capability", compat.REASON_DTYPE_UNSUPPORTED),
        ("capability_shape_m", {"m_range": (8, 4096)}, "capability", compat.REASON_SHAPE_UNSUPPORTED),
        (
            "capability_bits_requantize",
            {"bits": (8,)},
            "quant",
            compat.REASON_BITS_UNSUPPORTED,
        ),
        (
            "capability_group_requantize",
            {"group_sizes": (32,)},
            "quant",
            compat.REASON_GROUP_UNSUPPORTED,
        ),
        (
            "capability_workspace",
            {"workspace_bytes": 1 << 40},
            "resource",
            compat.REASON_WORKSPACE_EXCEEDED,
        ),
    ]
    for case_id, variant, stage, reason in capability_cases:
        action = (
            compat.REQUANTIZE_REQUIRED
            if reason in (compat.REASON_BITS_UNSUPPORTED, compat.REASON_GROUP_UNSUPPORTED)
            else compat.REJECT
        )
        cases.append(
            MutationCase(
                case_id,
                CATEGORY_CAPABILITY,
                f"capability override {variant}",
                stage,
                action,
                reason,
                _noop_mutation,
                capability_variant=variant,
                notes="artifact is intact; the runtime cannot execute its semantics",
            )
        )
    return cases


def _noop_mutation(artifact_dir: str) -> None:  # noqa: ARG001 - uniform signature
    """Capability cases mutate the *capability*, not the artifact."""
    return None


# ── runner ────────────────────────────────────────────────────────────────


@dataclass
class FaultCaseResult:
    case_id: str
    category: str
    mutation: str
    expected_stage: str
    expected_action: str
    expected_reason_code: str
    actual_action: str
    actual_stage: str
    reason_code: str
    prelaunch: bool
    caught: bool
    cleanup: bool
    detail: str = ""

    def as_row(self, base_artifact: str, derived_artifact: str = "") -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "base_artifact": base_artifact,
            "mutation": self.mutation,
            "expected_stage": self.expected_stage,
            "expected_action": self.expected_action,
            "actual_action": self.actual_action,
            "actual_stage": self.actual_stage,
            "prelaunch": self.prelaunch,
            "reason_code": self.reason_code,
            "derived_artifact": derived_artifact,
            "cleanup": self.cleanup,
            "caught": self.caught,
            "detail": self.detail,
        }


def run_fault_matrix(
    golden_dir: str,
    scratch_dir: str,
    capability: compat.KernelCapability,
    *,
    expected_model: Optional[ModelIdentity] = None,
    tensor_name: Optional[str] = None,
    tensor_shape: Optional[Sequence[int]] = None,
    workspace_available_bytes: int = 1 << 30,
    cases: Optional[Sequence[MutationCase]] = None,
) -> List[FaultCaseResult]:
    """Run the fault matrix against a golden artifact.

    Only *copies* are mutated: the golden directory is read once and never
    written (E05-09 §15 "故障注入直接破坏 golden 原件" is forbidden). The
    temporary copy is removed unless ``cleanup`` fails, which is recorded.
    """
    os.makedirs(scratch_dir, exist_ok=True)
    matrix = list(cases or build_fault_matrix())
    golden_id = os.path.basename(os.path.normpath(golden_dir))
    if expected_model is None or tensor_name is None or tensor_shape is None:
        # Default the expectation to the *golden* identity: reading it before
        # any mutation is what makes "the artifact was swapped" detectable.
        golden = load_document(golden_dir)
        expected_model = expected_model or golden.model
        tensor_name = tensor_name or golden.tensor.name
        tensor_shape = tensor_shape or golden.tensor.shape
    results: List[FaultCaseResult] = []
    for case in matrix:
        work = os.path.join(scratch_dir, case.case_id)
        if os.path.exists(work):
            shutil.rmtree(work)
        if case.fixture_variant:
            _build_fixture_variant(work, case.fixture_variant)
        else:
            shutil.copytree(golden_dir, work)
        detail = ""
        try:
            case.apply(work)
            action, stage, reason = _evaluate_case(
                work,
                capability,
                case,
                expected_model=expected_model,
                tensor_name=tensor_name,
                tensor_shape=tensor_shape,
                workspace_available_bytes=workspace_available_bytes,
            )
        except Exception as exc:  # noqa: BLE001 - a crash is itself a finding
            action, stage, reason = compat.REJECT, "unhandled_exception", type(exc).__name__
            detail = str(exc)
        cleanup_ok = True
        try:
            shutil.rmtree(work)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            cleanup_ok = False
            detail = f"{detail} cleanup_error={exc}".strip()
        caught = (
            action == case.expected_action
            and stage == case.expected_stage
            and reason == case.expected_reason_code
        )
        if case.category == CATEGORY_PACKING and case.expected_action == compat.REPACK_REQUIRED:
            # A layout mismatch is recoverable, not a refusal: "caught" means
            # the loader refused to run the mismatched bytes directly.
            caught = action == compat.REPACK_REQUIRED and reason == case.expected_reason_code
        results.append(
            FaultCaseResult(
                case_id=case.case_id,
                category=case.category,
                mutation=case.mutation,
                expected_stage=case.expected_stage,
                expected_action=case.expected_action,
                expected_reason_code=case.expected_reason_code,
                actual_action=action,
                actual_stage=stage,
                reason_code=reason,
                prelaunch=True,
                caught=bool(caught),
                cleanup=cleanup_ok,
                detail=detail or golden_id,
            )
        )
    return results


def _evaluate_case(
    work: str,
    capability: compat.KernelCapability,
    case: MutationCase,
    *,
    expected_model: Optional[ModelIdentity],
    tensor_name: Optional[str],
    tensor_shape: Optional[Sequence[int]],
    workspace_available_bytes: int = 1 << 30,
) -> tuple:
    """Return ``(action, stage, reason_code)`` for one mutated artifact."""
    report = validate_artifact_dir(work)
    if not report["accepted"]:
        reason = str(report.get("reason_code") or "")
        stage = str(report.get("stage") or "") or _stage_for_reason(reason)
        return compat.REJECT, stage, reason
    document = load_document(work)
    effective = copy.deepcopy(capability)
    for key, value in (case.capability_variant or {}).items():
        setattr(effective, key, value)
    decision = compat.check_compatibility(
        document,
        effective,
        expected_model=expected_model,
        tensor_name=tensor_name,
        tensor_shape=tensor_shape,
        workspace_available_bytes=workspace_available_bytes,
        mode=compat.MODE_STRICT,
    )
    return decision.status, decision.stage, decision.reason_code


def _build_fixture_variant(work: str, variant: str) -> None:
    """Materialize a synthetic fixture with a specific layout variant."""
    from hqsb.quant.fixtures import build_tiny_artifact

    if variant == "w4_hifirst":
        document = build_tiny_artifact(bits=4, layout_variant="hifirst")
    else:  # pragma: no cover - defensive
        raise KeyError(variant)
    document.save(work)


def _stage_for_reason(reason_code: str) -> str:
    """Map a structural/integrity reason code back to its pipeline stage."""
    if reason_code in (
        REASON_SCHEMA_VERSION,
        "artifact_kind_mismatch",
        REASON_MANIFEST_FIELD,
        REASON_FILE_HASH,
    ):
        return "schema"
    if reason_code in (REASON_QUANT_SEMANTICS, "code_out_of_range"):
        return "quant"
    if reason_code in ("packed_bytes_value_mismatch", "packed_variant_parent_hash_mismatch"):
        return "pack"
    return "schema"


def summarize_fault_matrix(results: Sequence[FaultCaseResult]) -> Dict[str, Any]:
    """Aggregate the matrix into the coverage/verdict summary used by E05-09."""
    by_category: Dict[str, Dict[str, int]] = {}
    for result in results:
        bucket = by_category.setdefault(result.category, {"total": 0, "caught": 0})
        bucket["total"] += 1
        bucket["caught"] += int(result.caught)
    return {
        "total_cases": len(results),
        "caught": sum(int(r.caught) for r in results),
        "uncaught": [r.case_id for r in results if not r.caught],
        "all_prelaunch": all(r.prelaunch for r in results),
        "cleanup_ok": all(r.cleanup for r in results),
        "by_category": by_category,
    }


def matrix_plan_json() -> str:
    """Deterministic JSON of the pre-registered matrix (no results)."""
    return json.dumps(
        [case.as_dict() for case in build_fault_matrix()],
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    )


__all__ = [
    "CATEGORIES",
    "CATEGORY_CAPABILITY",
    "CATEGORY_MODEL",
    "CATEGORY_PACKING",
    "CATEGORY_QUANT",
    "CATEGORY_SCHEMA",
    "FaultCaseResult",
    "MutationCase",
    "build_fault_matrix",
    "matrix_plan_json",
    "run_fault_matrix",
    "summarize_fault_matrix",
]
