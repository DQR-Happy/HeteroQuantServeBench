"""Canonical, versioned and fail-closed RTN QuantArtifact persistence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hqsb.quant.legacy_e05_01.rtn import (
    QuantizedTensor,
    RtnSpec,
    dequantize,
    pack_int4,
    pack_int8,
    unpack_int4,
    unpack_int8,
)


ARTIFACT_SCHEMA = "hqsb.quant_artifact"
ARTIFACT_VERSION = "1.0.0"
CANONICAL_PACKING_VERSION = "1.0.0"
MAX_LOGICAL_ELEMENTS = 2**40
_MANIFEST_KEYS = {
    "schema",
    "version",
    "artifact_id",
    "canonical_metadata_sha256",
    "identity",
    "provenance",
}
_IDENTITY_KEYS = {
    "source",
    "tensor",
    "quantization",
    "parameters",
    "canonical_packing",
    "calibration",
    "code",
    "expected_dequant_dtype",
    "tolerance_references",
    "license",
    "files",
    "packed_variants",
    "compatibility",
    "known_limitations",
}


class ArtifactValidationError(ValueError):
    """A stable, machine-readable artifact rejection."""

    def __init__(self, reason_code: str, message: str):
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {message}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _file_descriptor(path: Path, role: str, dtype: str, count: int) -> dict[str, Any]:
    return {
        "role": role,
        "path": path.name,
        "dtype": dtype,
        "count": count,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _packing_metadata(tensor: QuantizedTensor) -> dict[str, Any]:
    rows, cols = tensor.original_shape
    padded_cols = cols + (cols % 2) if tensor.spec.bits == 4 else cols
    return {
        "name": "canonical-row-major-signed",
        "version": CANONICAL_PACKING_VERSION,
        "encoding": "twos-complement",
        "byte_order": "little",
        "nibble_order": "even-index-low-odd-index-high"
        if tensor.spec.bits == 4
        else None,
        "row_byte_aligned": True,
        "padding_code": 0,
        "logical_shape": [rows, cols],
        "padded_shape": [rows, padded_cols],
        "valid_tail_per_parameter_group": list(tensor.group_valid_sizes),
    }


def save_quant_artifact(
    directory: str | Path,
    tensor: QuantizedTensor,
    *,
    source_model_hash: str,
    tensor_name: str,
    source_artifact_id: str | None = None,
    created_at_utc: str | None = None,
    code_version: str = "hqsb-rtn-v1",
    license_name: str = "Apache-2.0",
) -> dict[str, Any]:
    """Save canonical data and return its manifest.

    ``artifact_id`` hashes only the canonical identity subtree.  Provenance
    time and the output path therefore cannot perturb identity.
    """

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    rows, cols = tensor.original_shape
    if tensor.spec.bits == 4:
        qbytes = pack_int4(tensor.qvalues, tensor.original_shape)
    else:
        qbytes = pack_int8(tensor.qvalues, tensor.original_shape)
    scale_bytes = struct.pack(f"<{len(tensor.scales)}f", *tensor.scales)
    zero_bytes = (
        b""
        if tensor.spec.symmetric
        else struct.pack(f"<{len(tensor.zero_points)}i", *tensor.zero_points)
    )
    qpath = output / "qvalues.bin"
    scale_path = output / "scales.bin"
    _write_atomic(qpath, qbytes)
    _write_atomic(scale_path, scale_bytes)
    files = {
        "qvalues": _file_descriptor(
            qpath, "canonical_qvalues", f"int{tensor.spec.bits}-packed", rows * cols
        ),
        "scales": _file_descriptor(
            scale_path, "dequant_scales", "float32-le", len(tensor.scales)
        ),
    }
    zero_storage: dict[str, Any]
    zero_path = output / "zero_points.bin"
    if tensor.spec.symmetric:
        if zero_path.exists():
            zero_path.unlink()
        zero_storage = {
            "storage": "implicit-zero",
            "dtype": "int32-le",
            "count": len(tensor.zero_points),
        }
    else:
        _write_atomic(zero_path, zero_bytes)
        files["zero_points"] = _file_descriptor(
            zero_path, "dequant_zero_points", "int32-le", len(tensor.zero_points)
        )
        zero_storage = {
            "storage": "file",
            "dtype": "int32-le",
            "count": len(tensor.zero_points),
        }
    identity = {
        "source": {
            "model_artifact_id": source_artifact_id,
            "model_sha256": source_model_hash,
        },
        "tensor": {
            "name": tensor_name,
            "original_shape": [rows, cols],
            "original_dtype": "float32",
            "original_layout": "row-major-contiguous",
        },
        "quantization": tensor.spec.as_dict(),
        "parameters": {
            "shape": list(tensor.parameter_shape),
            "scale": {
                "storage": "file",
                "dtype": "float32-le",
                "count": len(tensor.scales),
            },
            "zero_point": zero_storage,
        },
        "canonical_packing": _packing_metadata(tensor),
        "calibration": {
            "mode": "NONE_WEIGHT_STATISTICS_ONLY",
            "dataset": None,
        },
        "code": {"implementation": "hqsb.quant.rtn", "version": code_version},
        "expected_dequant_dtype": "float32",
        "tolerance_references": [],
        "license": {"name": license_name, "provenance": "self-implemented RTN"},
        "files": files,
        "packed_variants": [],
        "compatibility": {
            "canonical_reader": f"{ARTIFACT_SCHEMA}/{ARTIFACT_VERSION}",
            "kernel_specific": False,
            "architectures": ["portable-cpu-reference"],
        },
        "known_limitations": [
            "reference artifact; not evidence of native low-bit GEMM execution",
            "two-dimensional weight tensors only",
        ],
    }
    identity_hash = sha256_bytes(canonical_json_bytes(identity))
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "version": ARTIFACT_VERSION,
        "artifact_id": f"sha256:{identity_hash}",
        "canonical_metadata_sha256": identity_hash,
        "identity": identity,
        "provenance": {
            "created_at_utc": created_at_utc or _utc_now(),
            "identity_excludes": ["provenance.created_at_utc", "output_path"],
        },
    }
    _write_atomic(output / "manifest.json", json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n")
    return manifest


def _reject(condition: bool, reason: str, message: str) -> None:
    if condition:
        raise ArtifactValidationError(reason, message)


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    _reject(actual != expected, "SCHEMA_FIELDS_INVALID", f"{name} fields are {actual}")


def _safe_file(root: Path, descriptor: Mapping[str, Any], role: str) -> bytes:
    relative = descriptor.get("path")
    _reject(
        not isinstance(relative, str)
        or Path(relative).name != relative
        or relative in ("", ".", ".."),
        "FILE_PATH_INVALID",
        f"{role} path must be a plain basename",
    )
    path = root / relative
    _reject(not path.is_file(), "FILE_MISSING", f"missing {role} file {relative!r}")
    payload = path.read_bytes()
    _reject(
        len(payload) != descriptor.get("bytes"),
        "DATA_SIZE_MISMATCH",
        f"{role} byte count differs from manifest",
    )
    _reject(
        sha256_bytes(payload) != descriptor.get("sha256"),
        "CHECKSUM_MISMATCH",
        f"{role} checksum differs from manifest",
    )
    return payload


def _int_pair(value: Any, name: str) -> tuple[int, int]:
    _reject(
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) for item in value),
        "SHAPE_INVALID",
        f"{name} must contain two integers",
    )
    first, second = value
    _reject(first <= 0 or second <= 0, "SHAPE_INVALID", f"{name} must be positive")
    _reject(
        first > MAX_LOGICAL_ELEMENTS // second,
        "SIZE_OVERFLOW",
        f"{name} product exceeds the artifact safety limit",
    )
    return first, second


def load_quant_artifact(
    directory: str | Path, *, expected_kernel_id: str | None = None
) -> tuple[QuantizedTensor, dict[str, Any]]:
    """Validate before data use, then load canonical qvalues and parameters."""

    root = Path(directory)
    manifest_path = root / "manifest.json"
    _reject(not manifest_path.is_file(), "MANIFEST_MISSING", "manifest.json is absent")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("MANIFEST_PARSE_ERROR", str(exc)) from exc
    _reject(not isinstance(manifest, dict), "MANIFEST_TYPE_INVALID", "manifest must be an object")
    _exact_keys(manifest, _MANIFEST_KEYS, "manifest")
    _reject(manifest["schema"] != ARTIFACT_SCHEMA, "SCHEMA_UNSUPPORTED", "unknown schema")
    _reject(
        manifest["version"] != ARTIFACT_VERSION,
        "VERSION_UNSUPPORTED",
        f"supported version is {ARTIFACT_VERSION}",
    )
    identity = manifest["identity"]
    _reject(not isinstance(identity, dict), "IDENTITY_INVALID", "identity must be an object")
    _exact_keys(identity, _IDENTITY_KEYS, "identity")
    tensor_meta = identity["tensor"]
    _reject(not isinstance(tensor_meta, dict), "TENSOR_METADATA_INVALID", "tensor metadata")
    _reject(
        tensor_meta.get("original_layout") != "row-major-contiguous",
        "LAYOUT_UNSUPPORTED",
        "only row-major-contiguous canonical tensors are supported",
    )
    rows, cols = _int_pair(tensor_meta.get("original_shape"), "original_shape")
    quant = identity["quantization"]
    _reject(not isinstance(quant, dict), "QUANTIZATION_INVALID", "quantization metadata")
    try:
        spec = RtnSpec(
            bits=quant.get("bits"),
            symmetric=quant.get("symmetric"),
            granularity=quant.get("granularity"),
            group_size=quant.get("group_size"),
            axis=quant.get("axis"),
            range_policy=quant.get("range_policy"),
            round_mode=quant.get("round_mode"),
            zero_group_policy=quant.get("zero_group_policy"),
            constant_group_policy=quant.get("constant_group_policy"),
            scale_dtype=quant.get("scale_dtype"),
            zero_point_dtype=quant.get("zero_point_dtype"),
            nan_inf_policy=quant.get("nan_inf_policy"),
        )
    except (TypeError, ValueError) as exc:
        message = str(exc)
        reason = "GROUP_AXIS_INVALID" if "axis" in message or "group" in message else "QUANTIZATION_INVALID"
        raise ArtifactValidationError(reason, message) from exc
    _reject(
        quant.get("algorithm") != "rtn"
        or quant.get("qmin") != spec.qmin
        or quant.get("qmax") != spec.qmax,
        "QUANTIZATION_INVALID",
        "algorithm or integer range differs from resolved RTN semantics",
    )
    packing = identity["canonical_packing"]
    expected_packing = {
        "name": "canonical-row-major-signed",
        "version": CANONICAL_PACKING_VERSION,
        "encoding": "twos-complement",
        "byte_order": "little",
        "nibble_order": "even-index-low-odd-index-high" if spec.bits == 4 else None,
        "row_byte_aligned": True,
        "padding_code": 0,
    }
    _reject(
        not isinstance(packing, dict)
        or any(packing.get(key) != value for key, value in expected_packing.items()),
        "PACKING_LAYOUT_UNSUPPORTED",
        "canonical packing semantics differ from v1",
    )
    _reject(
        packing.get("logical_shape") != [rows, cols],
        "SHAPE_MISMATCH",
        "packing logical shape differs from tensor shape",
    )
    padded_cols = cols + (cols % 2) if spec.bits == 4 else cols
    _reject(
        packing.get("padded_shape") != [rows, padded_cols],
        "SHAPE_MISMATCH",
        "padded shape differs from canonical row alignment",
    )
    if spec.granularity == "per-tensor":
        parameter_shape = (1, 1)
        expected_valid = [rows * cols]
    elif spec.granularity == "per-channel":
        parameter_shape = (rows, 1)
        expected_valid = [cols] * rows
    else:
        assert spec.group_size is not None
        groups_per_row = (cols + spec.group_size - 1) // spec.group_size
        parameter_shape = (rows, groups_per_row)
        expected_valid = []
        for _row in range(rows):
            expected_valid.extend(
                min(spec.group_size, cols - group * spec.group_size)
                for group in range(groups_per_row)
            )
    _reject(
        packing.get("valid_tail_per_parameter_group") != expected_valid,
        "GROUP_TAIL_INVALID",
        "group valid lengths do not cover the logical tensor exactly",
    )
    parameters = identity["parameters"]
    _reject(
        not isinstance(parameters, dict)
        or parameters.get("shape") != list(parameter_shape),
        "PARAMETER_SHAPE_MISMATCH",
        "scale/zero-point shape differs from grouping",
    )
    files = identity["files"]
    _reject(not isinstance(files, dict), "FILES_INVALID", "files must be an object")
    expected_file_keys = {"qvalues", "scales"} | (set() if spec.symmetric else {"zero_points"})
    _reject(set(files) != expected_file_keys, "FILES_INVALID", "unexpected or missing data role")
    qbytes = _safe_file(root, files["qvalues"], "qvalues")
    scale_bytes = _safe_file(root, files["scales"], "scales")
    parameter_count = parameter_shape[0] * parameter_shape[1]
    _reject(
        len(scale_bytes) != parameter_count * 4
        or files["scales"].get("count") != parameter_count,
        "SCALE_SIZE_MISMATCH",
        "scale payload does not match grouping",
    )
    scales = tuple(struct.unpack(f"<{parameter_count}f", scale_bytes))
    _reject(
        any(not math.isfinite(value) or value <= 0 for value in scales),
        "SCALE_INVALID",
        "scale must be positive and finite",
    )
    if spec.symmetric:
        zeros = (0,) * parameter_count
        zero_meta = parameters.get("zero_point", {})
        _reject(
            zero_meta.get("storage") != "implicit-zero"
            or zero_meta.get("count") != parameter_count,
            "ZERO_POINT_INVALID",
            "symmetric zero-point metadata must be implicit zero",
        )
    else:
        zero_bytes = _safe_file(root, files["zero_points"], "zero_points")
        _reject(
            len(zero_bytes) != parameter_count * 4
            or files["zero_points"].get("count") != parameter_count,
            "ZERO_POINT_SIZE_MISMATCH",
            "zero-point payload does not match grouping",
        )
        zeros = tuple(struct.unpack(f"<{parameter_count}i", zero_bytes))
        _reject(
            any(value < spec.qmin or value > spec.qmax for value in zeros),
            "ZERO_POINT_INVALID",
            "zero-point is outside the resolved integer range",
        )
    try:
        qvalues = (
            unpack_int4(qbytes, (rows, cols))
            if spec.bits == 4
            else unpack_int8(qbytes, (rows, cols))
        )
    except ValueError as exc:
        reason = "PACKING_PADDING_INVALID" if "padding" in str(exc) else "DATA_SIZE_MISMATCH"
        raise ArtifactValidationError(reason, str(exc)) from exc
    _reject(
        any(value < spec.qmin or value > spec.qmax for value in qvalues),
        "QVALUE_OUT_OF_RANGE",
        "canonical qvalue is outside the quantization scheme range",
    )
    identity_hash = sha256_bytes(canonical_json_bytes(identity))
    _reject(
        manifest["canonical_metadata_sha256"] != identity_hash
        or manifest["artifact_id"] != f"sha256:{identity_hash}",
        "IDENTITY_HASH_MISMATCH",
        "canonical identity hash does not match metadata",
    )
    if expected_kernel_id is not None:
        variants = identity["packed_variants"]
        _reject(
            not any(item.get("kernel_id") == expected_kernel_id for item in variants),
            "KERNEL_INCOMPATIBLE",
            f"no validated packed variant for kernel {expected_kernel_id!r}",
        )
    tensor = QuantizedTensor(
        spec=spec,
        original_shape=(rows, cols),
        qvalues=qvalues,
        scales=scales,
        zero_points=zeros,
        parameter_shape=parameter_shape,
        group_valid_sizes=tuple(expected_valid),
        clamped=(False,) * (rows * cols),
    )
    dequantize(tensor)  # Exercise the full validated metadata mapping before return.
    return tensor, manifest


__all__ = [
    "ARTIFACT_SCHEMA",
    "ARTIFACT_VERSION",
    "ArtifactValidationError",
    "canonical_json_bytes",
    "load_quant_artifact",
    "save_quant_artifact",
    "sha256_bytes",
    "sha256_file",
]
