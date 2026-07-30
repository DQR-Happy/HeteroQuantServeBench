"""Streaming model-level RTN artifacts for the E05-02 baseline.

The artifact stores canonical signed row-major INT8 or row-byte-aligned INT4
payloads plus FP32 scales.  Loading this artifact reconstructs selected Linear
weights into their original compute dtype.  It is intentionally a
``fake-dequant`` execution path: it measures representation error and portable
storage, and does not claim a native low-bit GEMM.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch


MODEL_ARTIFACT_SCHEMA = "hqsb.model_quant_artifact"
MODEL_ARTIFACT_VERSION = "1.0.0"
PACKING_VERSION = "hqsb-canonical-signed-row-major/1.0.0"


class QuantizationCancelled(RuntimeError):
    """Cooperative cancellation; an incomplete artifact has no manifest."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _module_map(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    return dict(model.named_modules())


def iter_quantized_linears(
    model: torch.nn.Module,
    *,
    excluded_suffixes: Iterable[str] = ("lm_head",),
) -> list[tuple[str, torch.nn.Linear]]:
    """Return the frozen set of two-dimensional Linear weights to quantize."""
    exclusions = tuple(excluded_suffixes)
    selected: list[tuple[str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if module.weight.ndim != 2:
            continue
        if any(name == suffix or name.endswith(f".{suffix}") for suffix in exclusions):
            continue
        selected.append((name, module))
    return selected


def quantization_coverage(
    model: torch.nn.Module,
    selected: list[tuple[str, torch.nn.Linear]],
) -> dict[str, Any]:
    selected_ids = {id(module.weight) for _, module in selected}
    selected_numel = sum(module.weight.numel() for _, module in selected)
    selected_bytes = sum(
        module.weight.numel() * module.weight.element_size() for _, module in selected
    )
    unique: dict[int, torch.nn.Parameter] = {}
    for parameter in model.parameters():
        unique.setdefault(id(parameter), parameter)
    total_numel = sum(parameter.numel() for parameter in unique.values())
    total_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in unique.values()
    )
    return {
        "selected_tensor_count": len(selected),
        "selected_parameter_count": selected_numel,
        "selected_source_bytes": selected_bytes,
        "total_unique_parameter_count": total_numel,
        "total_unique_parameter_bytes": total_bytes,
        "parameter_coverage": selected_numel / total_numel,
        "byte_coverage": selected_bytes / total_bytes,
        "selected_parameter_ids_unique": len(selected_ids) == len(selected),
    }


def _file_record(path: Path, role: str, dtype: str, count: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "role": role,
        "dtype": dtype,
        "count": count,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _safe_name(index: int, module_name: str) -> str:
    digest = hashlib.sha256(module_name.encode("utf-8")).hexdigest()[:12]
    return f"{index:04d}_{digest}"


@torch.inference_mode()
def save_model_quant_artifact(
    model: torch.nn.Module,
    directory: str | Path,
    *,
    bits: int,
    group_size: int | None,
    source_model_hash: str,
    source_revision: str,
    excluded_suffixes: Iterable[str] = ("lm_head",),
    row_chunk: int = 32,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Stream quantized weights; optional callbacks run between bounded chunks.

    Cooperative jobs require a fresh directory so cancellation cannot leave an
    older manifest claiming that partially replaced payloads are complete.
    """
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    if bits == 4 and (group_size is None or group_size <= 0):
        raise ValueError("INT4 requires a positive group_size")
    if bits == 8 and group_size is not None:
        raise ValueError("INT8 baseline is per-channel and has no group_size")
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")

    root = Path(directory)
    if (progress is not None or cancel_check is not None) and (
        root / "manifest.json"
    ).exists():
        raise FileExistsError(
            "Cooperative quantization requires a fresh artifact directory"
        )

    def check_cancelled():
        if cancel_check is not None and cancel_check():
            raise QuantizationCancelled("Quantization cancelled before artifact commit")

    check_cancelled()
    root.mkdir(parents=True, exist_ok=True)
    tensors_root = root / "tensors"
    tensors_root.mkdir(parents=True, exist_ok=True)
    selected = iter_quantized_linears(model, excluded_suffixes=excluded_suffixes)
    coverage = quantization_coverage(model, selected)
    started = time.perf_counter()
    entries: list[dict[str, Any]] = []
    total_sq_error = 0.0
    total_sq_source = 0.0
    total_abs_error = 0.0
    max_abs_error = 0.0
    total_values = 0

    for index, (name, module) in enumerate(selected):
        check_cancelled()
        weight = module.weight.detach()
        rows, cols = (int(weight.shape[0]), int(weight.shape[1]))
        stem = _safe_name(index, name)
        tensor_dir = tensors_root / stem
        tensor_dir.mkdir(parents=True, exist_ok=True)
        q_path = tensor_dir / "qvalues.bin"
        scale_path = tensor_dir / "scales.bin"
        q_count = rows * cols
        scale_count = rows if bits == 8 else rows * math.ceil(cols / int(group_size))
        tensor_sq_error = 0.0
        tensor_sq_source = 0.0
        tensor_abs_error = 0.0
        tensor_max_error = 0.0
        saturated = 0
        t0 = time.perf_counter()

        with q_path.open("wb") as q_handle, scale_path.open("wb") as scale_handle:
            for begin in range(0, rows, row_chunk):
                check_cancelled()
                end = min(rows, begin + row_chunk)
                source = weight[begin:end].float()
                if not bool(torch.isfinite(source).all().item()):
                    raise ValueError(f"non-finite source weight in {name}")

                if bits == 8:
                    amax = source.abs().amax(dim=1, keepdim=True)
                    scales = torch.where(amax == 0, torch.ones_like(amax), amax / 127.0)
                    q = torch.round(source / scales).clamp(-127, 127).to(torch.int8)
                    reconstructed = q.float() * scales
                    q_payload = q.cpu().contiguous().numpy().tobytes()
                    scale_payload = (
                        scales.squeeze(1).cpu().numpy().astype("<f4").tobytes()
                    )
                    saturated += int((q.abs() == 127).sum().item())
                else:
                    assert group_size is not None
                    groups = math.ceil(cols / group_size)
                    padded_cols = groups * group_size
                    if padded_cols != cols:
                        source_padded = torch.nn.functional.pad(
                            source, (0, padded_cols - cols), value=0.0
                        )
                    else:
                        source_padded = source
                    grouped = source_padded.reshape(end - begin, groups, group_size)
                    amax = grouped.abs().amax(dim=2, keepdim=True)
                    scales = torch.where(amax == 0, torch.ones_like(amax), amax / 7.0)
                    q_grouped = (
                        torch.round(grouped / scales).clamp(-7, 7).to(torch.int8)
                    )
                    reconstructed = (q_grouped.float() * scales).reshape(
                        end - begin, padded_cols
                    )[:, :cols]
                    q = q_grouped.reshape(end - begin, padded_cols)[:, :cols]
                    if cols % 2:
                        q_for_pack = torch.nn.functional.pad(q, (0, 1), value=0)
                    else:
                        q_for_pack = q
                    low = torch.bitwise_and(q_for_pack[:, 0::2].to(torch.int16), 0xF)
                    high = torch.bitwise_left_shift(
                        torch.bitwise_and(q_for_pack[:, 1::2].to(torch.int16), 0xF), 4
                    )
                    packed = torch.bitwise_or(low, high).to(torch.uint8)
                    q_payload = packed.cpu().contiguous().numpy().tobytes()
                    scale_payload = (
                        scales.squeeze(2).cpu().numpy().astype("<f4").tobytes()
                    )
                    saturated += int((q.abs() == 7).sum().item())

                q_handle.write(q_payload)
                scale_handle.write(scale_payload)
                delta = reconstructed - source
                tensor_sq_error += float(torch.sum(delta * delta).item())
                tensor_sq_source += float(torch.sum(source * source).item())
                tensor_abs_error += float(torch.sum(delta.abs()).item())
                tensor_max_error = max(
                    tensor_max_error, float(delta.abs().max().item())
                )
                if progress is not None:
                    progress(
                        {
                            "stage": "quantizing",
                            "tensor": name,
                            "completed_tensors": index,
                            "total_tensors": len(selected),
                            "rows_processed": end,
                            "total_rows": rows,
                        }
                    )

        check_cancelled()
        q_record = _file_record(
            q_path, "canonical_qvalues", f"int{bits}-packed", q_count
        )
        scale_record = _file_record(
            scale_path, "dequant_scales", "float32-le", scale_count
        )
        expected_q_bytes = rows * cols if bits == 8 else rows * math.ceil(cols / 2)
        expected_scale_bytes = scale_count * 4
        if q_record["bytes"] != expected_q_bytes:
            raise RuntimeError(f"q payload size mismatch for {name}")
        if scale_record["bytes"] != expected_scale_bytes:
            raise RuntimeError(f"scale payload size mismatch for {name}")

        entry = {
            "name": name,
            "source_shape": [rows, cols],
            "source_dtype": str(weight.dtype).replace("torch.", ""),
            "scheme": "symmetric-narrow-signed",
            "granularity": "per-channel" if bits == 8 else "per-group",
            "axis": 1,
            "group_size": group_size,
            "qmin": -127 if bits == 8 else -7,
            "qmax": 127 if bits == 8 else 7,
            "packing": {
                "version": PACKING_VERSION,
                "encoding": "twos-complement",
                "byte_order": "little",
                "nibble_order": "even-index-low-odd-index-high" if bits == 4 else None,
                "row_byte_aligned": True,
                "tail_valid": cols % group_size
                if bits == 4 and cols % int(group_size)
                else (group_size if bits == 4 else cols),
            },
            "files": {"qvalues": q_record, "scales": scale_record},
            "metrics": {
                "max_abs_error": tensor_max_error,
                "mean_abs_error": tensor_abs_error / q_count,
                "rmse": math.sqrt(tensor_sq_error / q_count),
                "nrmse": math.sqrt(tensor_sq_error / max(tensor_sq_source, 1e-30)),
                "saturated_count": saturated,
            },
            "offline_wall_time_s": time.perf_counter() - t0,
        }
        entries.append(entry)
        total_sq_error += tensor_sq_error
        total_sq_source += tensor_sq_source
        total_abs_error += tensor_abs_error
        max_abs_error = max(max_abs_error, tensor_max_error)
        total_values += q_count
        if progress is not None:
            progress(
                {
                    "stage": "quantizing",
                    "tensor": name,
                    "completed_tensors": index + 1,
                    "total_tensors": len(selected),
                    "rows_processed": rows,
                    "total_rows": rows,
                }
            )

    identity = {
        "schema": MODEL_ARTIFACT_SCHEMA,
        "version": MODEL_ARTIFACT_VERSION,
        "source": {
            "model_sha256": source_model_hash,
            "revision": source_revision,
        },
        "quantization": {
            "algorithm": "rtn",
            "bits": bits,
            "scheme": "symmetric-narrow-signed",
            "granularity": "per-channel" if bits == 8 else "per-group",
            "axis": 1,
            "group_size": group_size,
            "round_mode": "nearest-even",
            "scale_dtype": "float32",
            "activation_dtype": "float16",
        },
        "module_policy": {
            "include": "all torch.nn.Linear with 2-D weights",
            "excluded_suffixes": list(excluded_suffixes),
            "embedding": "float16",
            "norm": "float16",
            "bias": "float16",
            "lm_head": "float16",
        },
        "packing_version": PACKING_VERSION,
        "coverage": coverage,
        "tensors": [
            {
                key: value
                for key, value in entry.items()
                if key not in {"offline_wall_time_s", "metrics"}
            }
            for entry in entries
        ],
        "runtime_contract": {
            "canonical_storage": True,
            "native_low_bit_kernel": False,
            "load_behavior": "validate then dequantize whole tensor to source dtype",
            "performance_claim_allowed": False,
        },
    }
    identity_hash = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    manifest = {
        **identity,
        "artifact_id": f"sha256:{identity_hash}",
        "canonical_metadata_sha256": identity_hash,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "offline": {
            "wall_time_s": time.perf_counter() - started,
            "per_tensor_wall_time_s": {
                entry["name"]: entry["offline_wall_time_s"] for entry in entries
            },
            "per_tensor_metrics": {
                entry["name"]: entry["metrics"] for entry in entries
            },
            "values": total_values,
            "max_abs_error": max_abs_error,
            "mean_abs_error": total_abs_error / max(total_values, 1),
            "rmse": math.sqrt(total_sq_error / max(total_values, 1)),
            "nrmse": math.sqrt(total_sq_error / max(total_sq_source, 1e-30)),
        },
    }
    manifest_path = root / "manifest.json"
    check_cancelled()
    temporary = root / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest


def load_manifest(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != MODEL_ARTIFACT_SCHEMA:
        raise ValueError("unsupported model artifact schema")
    if manifest.get("version") != MODEL_ARTIFACT_VERSION:
        raise ValueError("unsupported model artifact version")
    identity = {
        key: manifest[key]
        for key in (
            "schema",
            "version",
            "source",
            "quantization",
            "module_policy",
            "packing_version",
            "coverage",
            "tensors",
            "runtime_contract",
        )
    }
    identity_hash = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    if manifest.get("canonical_metadata_sha256") != identity_hash:
        raise ValueError("artifact metadata hash mismatch")
    if manifest.get("artifact_id") != f"sha256:{identity_hash}":
        raise ValueError("artifact identity mismatch")
    return manifest


def _validate_payload(root: Path, tensor_dir: Path, record: dict[str, Any]) -> Path:
    name = record.get("path")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("artifact payload path must be a basename")
    path = tensor_dir / name
    if not path.is_file():
        raise ValueError(f"missing artifact payload {path.relative_to(root)}")
    if path.stat().st_size != record.get("bytes"):
        raise ValueError(f"artifact payload size mismatch: {path.relative_to(root)}")
    if _sha256_file(path) != record.get("sha256"):
        raise ValueError(
            f"artifact payload checksum mismatch: {path.relative_to(root)}"
        )
    return path


@torch.inference_mode()
def apply_model_quant_artifact(
    model: torch.nn.Module,
    directory: str | Path,
    *,
    expected_source_model_hash: str,
    row_chunk: int = 32,
) -> dict[str, Any]:
    """Validate and reconstruct an artifact into a freshly loaded FP16 model."""
    root = Path(directory)
    manifest = load_manifest(root)
    if manifest["source"]["model_sha256"] != expected_source_model_hash:
        raise ValueError("source model hash does not match artifact")
    modules = _module_map(model)
    bits = int(manifest["quantization"]["bits"])
    group_size = manifest["quantization"]["group_size"]
    started = time.perf_counter()
    reconstructed = 0

    for index, entry in enumerate(manifest["tensors"]):
        name = entry["name"]
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"artifact target is not a Linear module: {name}")
        rows, cols = map(int, entry["source_shape"])
        if tuple(module.weight.shape) != (rows, cols):
            raise ValueError(f"artifact shape mismatch for {name}")
        tensor_dir = root / "tensors" / _safe_name(index, name)
        q_path = _validate_payload(root, tensor_dir, entry["files"]["qvalues"])
        scale_path = _validate_payload(root, tensor_dir, entry["files"]["scales"])
        scale_array = np.memmap(scale_path, dtype="<f4", mode="r")
        weight = module.weight

        if bits == 8:
            if scale_array.size != rows:
                raise ValueError(f"scale count mismatch for {name}")
            q_array = np.memmap(q_path, dtype=np.int8, mode="r", shape=(rows, cols))
            for begin in range(0, rows, row_chunk):
                end = min(rows, begin + row_chunk)
                q = torch.from_numpy(np.asarray(q_array[begin:end]).copy()).to(
                    weight.device
                )
                scale = torch.from_numpy(np.asarray(scale_array[begin:end]).copy()).to(
                    weight.device
                )
                dequantized = q.float() * scale[:, None]
                weight[begin:end].copy_(dequantized.to(weight.dtype))
        elif bits == 4:
            if not isinstance(group_size, int) or group_size <= 0:
                raise ValueError("invalid INT4 group_size")
            groups = math.ceil(cols / group_size)
            row_bytes = math.ceil(cols / 2)
            if scale_array.size != rows * groups:
                raise ValueError(f"scale count mismatch for {name}")
            scales = scale_array.reshape(rows, groups)
            packed = np.memmap(
                q_path, dtype=np.uint8, mode="r", shape=(rows, row_bytes)
            )
            for begin in range(0, rows, row_chunk):
                end = min(rows, begin + row_chunk)
                payload = torch.from_numpy(np.asarray(packed[begin:end]).copy()).to(
                    weight.device
                )
                low = torch.bitwise_and(payload, 0xF).to(torch.int8)
                high = torch.bitwise_right_shift(payload, 4).to(torch.int8)
                q_unsigned = torch.stack((low, high), dim=2).reshape(end - begin, -1)[
                    :, :cols
                ]
                q = torch.where(q_unsigned >= 8, q_unsigned - 16, q_unsigned).to(
                    torch.int8
                )
                scale = torch.from_numpy(np.asarray(scales[begin:end]).copy()).to(
                    weight.device
                )
                padded_cols = groups * group_size
                if padded_cols != cols:
                    q_padded = torch.nn.functional.pad(
                        q, (0, padded_cols - cols), value=0
                    )
                else:
                    q_padded = q
                dequantized = (
                    q_padded.reshape(end - begin, groups, group_size).float()
                    * scale[:, :, None]
                ).reshape(end - begin, padded_cols)[:, :cols]
                weight[begin:end].copy_(dequantized.to(weight.dtype))
        else:
            raise ValueError("unsupported artifact bit width")
        reconstructed += rows * cols

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "artifact_id": manifest["artifact_id"],
        "bits": bits,
        "reconstructed_values": reconstructed,
        "load_validate_dequant_s": time.perf_counter() - started,
        "execution_path": "canonical-low-bit-storage_then_full-fp16-dequant",
        "native_low_bit_kernel": False,
    }


def artifact_disk_usage(directory: str | Path) -> dict[str, int]:
    root = Path(directory)
    by_role = {"qvalues": 0, "scales": 0, "manifest": 0, "total": 0}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        size = path.stat().st_size
        by_role["total"] += size
        if path.name == "qvalues.bin":
            by_role["qvalues"] += size
        elif path.name == "scales.bin":
            by_role["scales"] += size
        elif path.name == "manifest.json":
            by_role["manifest"] += size
    return by_role


__all__ = [
    "MODEL_ARTIFACT_SCHEMA",
    "MODEL_ARTIFACT_VERSION",
    "PACKING_VERSION",
    "QuantizationCancelled",
    "apply_model_quant_artifact",
    "artifact_disk_usage",
    "iter_quantized_linears",
    "load_manifest",
    "quantization_coverage",
    "save_model_quant_artifact",
]
