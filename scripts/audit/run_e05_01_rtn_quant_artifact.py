#!/usr/bin/env python3
"""E05-01 RTN mathematics, packing and QuantArtifact evidence collector."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hqsb.core.contracts.quant import QuantArtifact as C5QuantArtifact
from hqsb.quant.legacy_e05_01.artifact import (
    ARTIFACT_SCHEMA,
    ARTIFACT_VERSION,
    ArtifactValidationError,
    canonical_json_bytes,
    load_quant_artifact,
    save_quant_artifact,
    sha256_bytes,
    sha256_file,
)
from hqsb.quant.legacy_e05_01.rtn import (
    QuantizedTensor,
    RtnSpec,
    dequantize,
    pack_int4,
    pack_int8,
    quantize,
    round_nearest_even,
    unpack_int4,
    unpack_int8,
)


EXPERIMENT = "E05-01"
SCHEMA = "hqsb.s05.rtn_quant_math_and_artifact/v1"
SEED = 20260919
SOURCE_HASH = hashlib.sha256(b"hqsb-e05-01-synthetic-source-v1").hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def run(argv: Sequence[str], timeout: int = 120) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(argv),
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        return {
            "argv": list(argv),
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_s": time.perf_counter() - started,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": list(argv),
            "exit_code": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "duration_s": time.perf_counter() - started,
            "timed_out": True,
        }


def flatten(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [value for row in matrix for value in row]


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def matrix_hash(matrix: Sequence[Sequence[float]]) -> str:
    values = flatten(matrix)
    return sha256_bytes(struct.pack(f"<{len(values)}d", *values))


def reconstruction_hash(matrix: Sequence[Sequence[float]]) -> str:
    values = flatten(matrix)
    return sha256_bytes(struct.pack(f"<{len(values)}d", *values))


def q_hash(values: Iterable[int]) -> str:
    payload = json.dumps(list(values), separators=(",", ":")).encode()
    return sha256_bytes(payload)


def metrics(source: Sequence[Sequence[float]], reconstructed: Sequence[Sequence[float]]) -> dict[str, Any]:
    left, right = flatten(source), flatten(reconstructed)
    errors = [estimate - value for value, estimate in zip(left, right)]
    squared = [error * error for error in errors]
    dot = sum(value * estimate for value, estimate in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    mse = sum(squared) / len(squared)
    signal = sum(value * value for value in left) / len(left)
    return {
        "max_abs": max(abs(error) for error in errors),
        "mean_abs": sum(abs(error) for error in errors) / len(errors),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "cosine": dot / (left_norm * right_norm) if left_norm and right_norm else 1.0,
        "sqnr_db": 10.0 * math.log10(signal / mse) if signal > 0 and mse > 0 else None,
    }


def environment() -> dict[str, Any]:
    git_commit = run(["git", "rev-parse", "HEAD"], 30)
    git_status = run(["git", "status", "--short"], 30)
    torch_info: dict[str, Any] = {"available": False}
    try:
        import torch

        torch_info = {
            "available": True,
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "compute_capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None,
        }
    except Exception as exc:
        torch_info = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    source_paths = [
        REPO / "hqsb/quant/__init__.py",
        REPO / "hqsb/quant/legacy_e05_01/rtn.py",
        REPO / "hqsb/quant/legacy_e05_01/artifact.py",
        REPO / "scripts/audit/e05_01_run.sh",
        REPO / "scripts/audit/run_e05_01_rtn_quant_artifact.py",
        REPO / "tests/unit/quant/test_rtn.py",
        REPO / "tests/unit/quant/test_artifact.py",
    ]
    return {
        "schema": f"{SCHEMA}/environment",
        "experiment_id": EXPERIMENT,
        "collected_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "byteorder": sys.byteorder,
        "git_commit": git_commit["stdout"].strip(),
        "git_worktree_dirty": bool(git_status["stdout"].strip()),
        "git_status_sha256": sha256_bytes(git_status["stdout"].encode()),
        "source_identity": {
            relative(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in source_paths
        },
        "torch": torch_info,
    }


def protocol() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}/protocol",
        "experiment_id": EXPERIMENT,
        "state": "FROZEN_BEFORE_FORMAL_MATRIX",
        "frozen_at_utc": utc_now(),
        "seed": SEED,
        "pass_rule": "all 10 E05-01 clauses true; PASS_NEGATIVE is forbidden",
        "integer_ranges": {
            "symmetric": {"int4": [-7, 7], "int8": [-127, 127]},
            "asymmetric": {"int4": [-8, 7], "int8": [-128, 127]},
        },
        "round_mode": "round-to-nearest-even implemented explicitly",
        "parameter_compute": "Python binary64 statistics; scale rounded once to IEEE FP32 before q calculation",
        "zero_group_policy": "scale=1, zero_point=0, q=0",
        "constant_group_policy": "scale=abs(c), zero_point=0, q=sign(c), exact for finite nonzero c",
        "granularity": {
            "per-tensor": "one group for all O*K values",
            "per-channel": "one group per output row",
            "per-group": "each output row split along K/axis=1; ceil(K/G) groups",
        },
        "parameter_storage": {"scale": "float32 little-endian", "zero_point": "int32 little-endian"},
        "int4_packing": {
            "encoding": "signed two's-complement",
            "even_index": "low nibble",
            "odd_index": "high nibble",
            "row_alignment": "each row starts on a byte boundary",
            "odd_tail": "zero high padding nibble; nonzero is rejected",
        },
        "nonfinite_policy": "reject before quantization or load",
        "artifact_identity": "SHA-256 of canonical identity metadata and data checksums; excludes time and paths",
        "canonical_vs_kernel": "canonical row-major artifact has no implicit kernel-specific packed variant",
        "timing_scope": "offline reference quantize/pack/save/load/unpack/dequant only; not inference latency",
    }


def schema_document() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{ARTIFACT_SCHEMA}/{ARTIFACT_VERSION}",
        "title": "HQSB canonical QuantArtifact",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema",
            "version",
            "artifact_id",
            "canonical_metadata_sha256",
            "identity",
            "provenance",
        ],
        "properties": {
            "schema": {"const": ARTIFACT_SCHEMA},
            "version": {"const": ARTIFACT_VERSION},
            "artifact_id": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            "canonical_metadata_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "identity": {
                "type": "object",
                "description": "Strict identity subtree validated by hqsb.quant.artifact before data use",
            },
            "provenance": {
                "type": "object",
                "description": "Non-identity creation metadata",
            },
        },
    }


def collect_golden(out: Path) -> dict[str, Any]:
    every_code = tuple(range(-8, 8))
    expected_every_code_hex = "98badcfe10325476"
    actual_every_code = pack_int4(every_code, (1, 16))
    halfway_values = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    expected_halfway = [-4, -2, -2, 0, 0, 2, 2, 4]
    actual_halfway = [round_nearest_even(value) for value in halfway_values]
    tail_source = [[-7.0, -0.5, 0.0, 0.5, 7.0]]
    tail_tensor = quantize(
        tail_source, RtnSpec(bits=4, symmetric=True, granularity="per-tensor")
    )
    expected_tail = {
        "scale": [1.0],
        "zero_point": [0],
        "qvalues": [-7, 0, 0, 0, 7],
        "packed_hex": "090007",
        "reconstruction": [[-7.0, 0.0, 0.0, 0.0, 7.0]],
    }
    actual_tail = {
        "scale": list(tail_tensor.scales),
        "zero_point": list(tail_tensor.zero_points),
        "qvalues": list(tail_tensor.qvalues),
        "packed_hex": pack_int4(tail_tensor.qvalues, tail_tensor.original_shape).hex(),
        "reconstruction": dequantize(tail_tensor),
    }
    constant = quantize(
        [[-0.25, -0.25, -0.25]],
        RtnSpec(bits=4, symmetric=False, granularity="per-tensor"),
    )
    checks = {
        "every_int4_code_bytes": actual_every_code.hex() == expected_every_code_hex,
        "every_int4_code_round_trip": unpack_int4(actual_every_code, (1, 16)) == every_code,
        "halfway_both_signs": actual_halfway == expected_halfway,
        "tail_manual_quantization": actual_tail == expected_tail,
        "asymmetric_constant_exact": dequantize(constant)
        == [[-0.25, -0.25, -0.25]],
    }
    value = {
        "schema": f"{SCHEMA}/golden_vectors",
        "manual_oracle": {
            "every_int4_code": list(every_code),
            "expected_packed_hex": expected_every_code_hex,
            "halfway_values": halfway_values,
            "expected_halfway": expected_halfway,
            "tail_source": tail_source,
            "tail_expected": expected_tail,
        },
        "candidate": {
            "every_code_packed_hex": actual_every_code.hex(),
            "halfway": actual_halfway,
            "tail": actual_tail,
            "constant_group": {
                "qvalues": list(constant.qvalues),
                "scales": list(constant.scales),
                "zero_points": list(constant.zero_points),
                "reconstruction": dequantize(constant),
            },
        },
        "checks": checks,
        "all_pass": all(checks.values()),
    }
    write_json(out / "golden_vectors.json", value)
    return value


def collect_input_boundaries(out: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def expect_reject(name: str, operation: Callable[[], Any]) -> None:
        try:
            operation()
            rows.append({"name": name, "expected": "REJECT", "observed": "ACCEPT", "pass": False})
        except (TypeError, ValueError) as exc:
            rows.append(
                {
                    "name": name,
                    "expected": "REJECT",
                    "observed": type(exc).__name__,
                    "message": str(exc),
                    "pass": True,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "name": name,
                    "expected": "REJECT",
                    "observed": type(exc).__name__,
                    "message": str(exc),
                    "pass": False,
                }
            )

    def expect_accept(name: str, operation: Callable[[], bool]) -> None:
        try:
            passed = bool(operation())
            rows.append({"name": name, "expected": "ACCEPT", "observed": "ACCEPT", "pass": passed})
        except Exception as exc:
            rows.append(
                {
                    "name": name,
                    "expected": "ACCEPT",
                    "observed": type(exc).__name__,
                    "message": str(exc),
                    "pass": False,
                }
            )

    symmetric = RtnSpec(4, True, "per-tensor")
    expect_reject("empty_matrix", lambda: quantize([], symmetric))
    expect_reject("empty_row", lambda: quantize([[]], symmetric))
    expect_reject("ragged_matrix", lambda: quantize([[1.0], [1.0, 2.0]], symmetric))
    expect_reject("nan_input", lambda: quantize([[float("nan")]], symmetric))
    expect_reject("positive_inf_input", lambda: quantize([[float("inf")]], symmetric))
    expect_reject("negative_inf_input", lambda: quantize([[float("-inf")]], symmetric))
    expect_reject("unsupported_bits", lambda: RtnSpec(3, True, "per-tensor"))
    expect_reject("wrong_axis", lambda: RtnSpec(4, True, "per-tensor", axis=0))
    expect_reject("missing_group_size", lambda: RtnSpec(4, True, "per-group"))
    expect_reject("zero_group_size", lambda: RtnSpec(4, True, "per-group", 0))
    expect_reject("group_size_on_tensor", lambda: RtnSpec(4, True, "per-tensor", 4))
    expect_reject("float32_scale_underflow", lambda: quantize([[1e-50]], symmetric))
    expect_reject("float32_scale_overflow", lambda: quantize([[1e300]], symmetric))
    expect_accept(
        "zero_group",
        lambda: dequantize(quantize([[0.0, 0.0]], RtnSpec(4, False, "per-tensor")))
        == [[0.0, 0.0]],
    )
    expect_accept(
        "negative_constant_group",
        lambda: dequantize(
            quantize([[-0.25, -0.25]], RtnSpec(4, False, "per-tensor"))
        )
        == [[-0.25, -0.25]],
    )
    expect_accept(
        "positive_constant_group",
        lambda: dequantize(quantize([[0.25, 0.25]], RtnSpec(4, False, "per-tensor")))
        == [[0.25, 0.25]],
    )
    expect_accept(
        "representable_float32_subnormal_scale",
        lambda: all(
            math.isfinite(value)
            for value in flatten(dequantize(quantize([[1e-40]], symmetric)))
        ),
    )
    try:
        import torch

        non_contiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        expect_reject("non_contiguous_framework_tensor", lambda: quantize(non_contiguous, symmetric))
    except Exception as exc:
        rows.append(
            {
                "name": "non_contiguous_framework_tensor",
                "expected": "REJECT",
                "observed": "SETUP_ERROR",
                "message": f"{type(exc).__name__}: {exc}",
                "pass": False,
            }
        )
    value = {
        "schema": f"{SCHEMA}/input_boundaries",
        "rows": rows,
        "case_count": len(rows),
        "all_pass": all(row["pass"] for row in rows),
        "non_contiguous_policy": "portable reference accepts only explicit list/tuple rows; framework tensors require an adapter that materializes canonical row-major data",
    }
    write_json(out / "input_boundary_cases.json", value)
    return value


def synthetic_matrix(rows: int, cols: int, salt: int) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in range(rows):
        values = []
        for column in range(cols):
            token = (row * 131 + column * 17 + salt * 19) % 97
            value = (token - 48) / 5.0
            if column % 29 == 0:
                value = 0.0
            elif column % 31 == 0:
                value = (column % 8) + 0.5
            elif column % 37 == 0:
                value = -((column % 8) + 0.5)
            values.append(value)
        matrix.append(values)
    if cols > 1:
        matrix[-1][-1] = 11.0
    return matrix


def independent_quantize(
    matrix: Sequence[Sequence[float]], spec: RtnSpec
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[int, ...]]:
    """Independent oracle using Python's specified bankers ``round``."""

    rows, cols = len(matrix), len(matrix[0])
    flat = flatten(matrix)
    groups: list[list[int]] = []
    if spec.granularity == "per-tensor":
        groups.append(list(range(rows * cols)))
    elif spec.granularity == "per-channel":
        groups.extend(list(range(row * cols, (row + 1) * cols)) for row in range(rows))
    else:
        assert spec.group_size is not None
        for row in range(rows):
            for start_column in range(0, cols, spec.group_size):
                groups.append(
                    list(
                        range(
                            row * cols + start_column,
                            row * cols + min(cols, start_column + spec.group_size),
                        )
                    )
                )
    qvalues = [0] * len(flat)
    scales: list[float] = []
    zeros: list[int] = []
    for indices in groups:
        values = [flat[index] for index in indices]
        if spec.symmetric:
            amax = max(abs(value) for value in values)
            scale = 1.0 if amax == 0 else float32(amax / spec.qmax)
            zero = 0
            codes = [round(value / scale) for value in values] if amax else [0] * len(values)
        else:
            xmin, xmax = min(values), max(values)
            if xmin == xmax:
                if xmin == 0:
                    scale, zero, codes = 1.0, 0, [0] * len(values)
                else:
                    scale, zero = float32(abs(xmin)), 0
                    codes = ([1] if xmin > 0 else [-1]) * len(values)
            else:
                scale = float32((xmax - xmin) / (spec.qmax - spec.qmin))
                zero = max(spec.qmin, min(spec.qmax, round(spec.qmin - xmin / scale)))
                codes = [round(value / scale) + zero for value in values]
        codes = [max(spec.qmin, min(spec.qmax, code)) for code in codes]
        for index, code in zip(indices, codes):
            qvalues[index] = code
        scales.append(scale)
        zeros.append(zero)
    return tuple(qvalues), tuple(scales), tuple(zeros)


def group_indices(tensor: QuantizedTensor) -> list[list[int]]:
    rows, cols = tensor.original_shape
    spec = tensor.spec
    if spec.granularity == "per-tensor":
        return [list(range(rows * cols))]
    if spec.granularity == "per-channel":
        return [list(range(row * cols, (row + 1) * cols)) for row in range(rows)]
    assert spec.group_size is not None
    return [
        list(range(row * cols + start, row * cols + min(cols, start + spec.group_size)))
        for row in range(rows)
        for start in range(0, cols, spec.group_size)
    ]


def monotonic_within_groups(source: Sequence[Sequence[float]], tensor: QuantizedTensor) -> bool:
    values = flatten(source)
    for indices in group_indices(tensor):
        ordered = sorted((values[index], tensor.qvalues[index]) for index in indices)
        if any(left[1] > right[1] for left, right in zip(ordered, ordered[1:])):
            return False
    return True


def bounded_unclamped_error(source: Sequence[Sequence[float]], tensor: QuantizedTensor) -> bool:
    values = flatten(source)
    reconstructed = flatten(dequantize(tensor))
    for parameter_index, indices in enumerate(group_indices(tensor)):
        scale = tensor.scales[parameter_index]
        for index in indices:
            if not tensor.clamped[index]:
                tolerance = 0.50001 * scale + max(1e-7, abs(values[index]) * 1e-7)
                if abs(values[index] - reconstructed[index]) > tolerance:
                    return False
    return True


def fixed_parameter_requantize(tensor: QuantizedTensor) -> tuple[int, ...]:
    """Requantize reconstructed values without re-estimating scale or zero."""

    reconstructed = flatten(dequantize(tensor))
    result = [0] * len(reconstructed)
    for parameter_index, indices in enumerate(group_indices(tensor)):
        scale = tensor.scales[parameter_index]
        zero = tensor.zero_points[parameter_index]
        for index in indices:
            raw = round_nearest_even(reconstructed[index] / scale) + zero
            result[index] = max(tensor.spec.qmin, min(tensor.spec.qmax, raw))
    return tuple(result)


def matrix_specs() -> list[tuple[RtnSpec, int, str]]:
    result: list[tuple[RtnSpec, int, str]] = []
    common_k = [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2048, 2051]
    for bits in (8, 4):
        for symmetric in (True, False):
            for granularity in ("per-tensor", "per-channel"):
                for cols in common_k:
                    result.append((RtnSpec(bits, symmetric, granularity), cols, "fixed-K"))
            seen: set[tuple[int, int]] = set()
            for group_size in (1, 16, 32, 64, 128):
                for cols in (1, max(1, group_size - 1), group_size, group_size + 1, 2048, 2051):
                    seen.add((group_size, cols))
            for cols in (1, 17, 2048, 2051):
                seen.add((cols, cols))
                seen.add((cols + 1, cols))
            for group_size, cols in sorted(seen):
                result.append(
                    (
                        RtnSpec(bits, symmetric, "per-group", group_size=group_size),
                        cols,
                        "G/K-boundary",
                    )
                )
    return result


def collect_factorial(out: Path) -> dict[str, Any]:
    raw_path = out / "factorial_cases.jsonl.gz"
    property_names = [
        "q_in_range",
        "scale_positive_finite",
        "dequant_shape_finite",
        "input_unchanged",
        "independent_reference_equal",
        "pack_round_trip",
        "fixed_parameter_idempotent_q",
        "recomputed_parameter_idempotence_if_stable",
        "monotonic_within_group",
        "unclamped_error_bound",
    ]
    totals = {name: 0 for name in property_names}
    failures: list[dict[str, Any]] = []
    coverage = {
        "bits": set(),
        "symmetric": set(),
        "granularity": set(),
        "group_size": set(),
        "K": set(),
        "tail_case_count": 0,
        "odd_K_count": 0,
    }
    count = 0
    recomputed_parameter_stable_count = 0
    recomputed_q_equal_count = 0
    worst = {"max_abs": -1.0, "case_id": None}
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        for case_id, (spec, cols, case_class) in enumerate(matrix_specs()):
            source = synthetic_matrix(2, cols, case_id + SEED)
            original_copy = copy.deepcopy(source)
            tensor = quantize(source, spec)
            reconstructed = dequantize(tensor)
            reference_q, reference_scale, reference_zero = independent_quantize(source, spec)
            packed = (
                pack_int4(tensor.qvalues, tensor.original_shape)
                if spec.bits == 4
                else pack_int8(tensor.qvalues, tensor.original_shape)
            )
            unpacked = (
                unpack_int4(packed, tensor.original_shape)
                if spec.bits == 4
                else unpack_int8(packed, tensor.original_shape)
            )
            requantized = quantize(reconstructed, spec)
            parameters_stable = (
                requantized.scales == tensor.scales
                and requantized.zero_points == tensor.zero_points
            )
            requantized_q_equal = requantized.qvalues == tensor.qvalues
            recomputed_parameter_stable_count += int(parameters_stable)
            recomputed_q_equal_count += int(requantized_q_equal)
            checks = {
                "q_in_range": all(spec.qmin <= value <= spec.qmax for value in tensor.qvalues),
                "scale_positive_finite": all(
                    value > 0 and math.isfinite(value) for value in tensor.scales
                ),
                "dequant_shape_finite": len(reconstructed) == 2
                and all(len(row) == cols for row in reconstructed)
                and all(math.isfinite(value) for value in flatten(reconstructed)),
                "input_unchanged": source == original_copy,
                "independent_reference_equal": tensor.qvalues == reference_q
                and tensor.scales == reference_scale
                and tensor.zero_points == reference_zero,
                "pack_round_trip": unpacked == tensor.qvalues,
                "fixed_parameter_idempotent_q": fixed_parameter_requantize(tensor)
                == tensor.qvalues,
                "recomputed_parameter_idempotence_if_stable": (
                    not parameters_stable or requantized_q_equal
                ),
                "monotonic_within_group": monotonic_within_groups(source, tensor),
                "unclamped_error_bound": bounded_unclamped_error(source, tensor),
            }
            for name, passed in checks.items():
                totals[name] += int(passed)
            if not all(checks.values()):
                failures.append(
                    {"case_id": case_id, "failed": [name for name, value in checks.items() if not value]}
                )
            case_metrics = metrics(source, reconstructed)
            if case_metrics["max_abs"] > worst["max_abs"]:
                worst = {"max_abs": case_metrics["max_abs"], "case_id": case_id}
            group_size = spec.group_size
            is_tail = bool(group_size and cols % group_size)
            coverage["bits"].add(spec.bits)
            coverage["symmetric"].add(spec.symmetric)
            coverage["granularity"].add(spec.granularity)
            if group_size is not None:
                coverage["group_size"].add(group_size)
            coverage["K"].add(cols)
            coverage["tail_case_count"] += int(is_tail)
            coverage["odd_K_count"] += int(cols % 2 == 1)
            record = {
                "schema": f"{SCHEMA}/factorial_case",
                "case_id": case_id,
                "case_class": case_class,
                "spec": spec.as_dict(),
                "shape": [2, cols],
                "is_tail": is_tail,
                "input": source,
                "input_sha256": matrix_hash(source),
                "parameter_shape": list(tensor.parameter_shape),
                "group_valid_sizes": list(tensor.group_valid_sizes),
                "scales": list(tensor.scales),
                "zero_points": list(tensor.zero_points),
                "qvalues": list(tensor.qvalues),
                "qvalues_sha256": q_hash(tensor.qvalues),
                "clamp_count": tensor.clamp_count,
                "recomputed_parameter_stable": parameters_stable,
                "recomputed_q_equal": requantized_q_equal,
                "requantized_scales": list(requantized.scales),
                "requantized_zero_points": list(requantized.zero_points),
                "packed_hex": packed.hex(),
                "packed_sha256": sha256_bytes(packed),
                "reconstruction": reconstructed,
                "reconstruction_sha256": reconstruction_hash(reconstructed),
                "metrics": case_metrics,
                "checks": checks,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            count += 1
    summary = {
        "schema": f"{SCHEMA}/factorial_summary",
        "case_count": count,
        "all_pass": not failures,
        "property_pass_counts": totals,
        "property_expected_count": count,
        "idempotence_diagnostics": {
            "fixed_parameter_expected_count": count,
            "recomputed_parameter_stable_count": recomputed_parameter_stable_count,
            "recomputed_q_equal_count": recomputed_q_equal_count,
            "interpretation": "q equality after parameter re-estimation is required only when scale/zp remain stable",
        },
        "failures": failures,
        "coverage": {
            "bits": sorted(coverage["bits"]),
            "symmetric": sorted(coverage["symmetric"]),
            "granularity": sorted(coverage["granularity"]),
            "group_sizes": sorted(coverage["group_size"]),
            "K_values": sorted(coverage["K"]),
            "tail_case_count": coverage["tail_case_count"],
            "odd_K_count": coverage["odd_K_count"],
        },
        "worst_tensor_error": worst,
        "raw_file": raw_path.name,
        "raw_file_bytes": raw_path.stat().st_size,
        "raw_file_sha256": sha256_file(raw_path),
    }
    write_json(out / "factorial_summary.json", summary)
    return summary


def create_example_artifacts(out: Path) -> dict[str, Any]:
    package = out / "golden_test_vector_package"
    source = [
        [-7.0, -3.5, -0.5, 0.0, 0.5, 3.5, 7.0, 2.0, -2.0],
        [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.25],
    ]
    configs = [
        ("rtn_w4_symmetric_group4_tail", RtnSpec(4, True, "per-group", 4)),
        ("rtn_w8_asymmetric_channel", RtnSpec(8, False, "per-channel")),
    ]
    rows = []
    c5_rows = []
    for name, spec in configs:
        tensor = quantize(source, spec)
        artifact_dir = package / name
        manifest = save_quant_artifact(
            artifact_dir,
            tensor,
            source_model_hash=SOURCE_HASH,
            source_artifact_id="synthetic:E05-01-v1",
            tensor_name="golden.weight",
        )
        loaded, loaded_manifest = load_quant_artifact(artifact_dir)
        c5 = C5QuantArtifact(
            algorithm="rtn",
            bits=spec.bits,
            granularity=spec.granularity,
            symmetric=spec.symmetric,
            group_size=spec.group_size,
            scale=manifest["identity"]["files"]["scales"]["sha256"],
            zero_point=(
                manifest["identity"]["files"].get("zero_points", {}).get("sha256")
                or "implicit-zero"
            ),
            calibration="NONE_WEIGHT_STATISTICS_ONLY",
            packing="canonical-row-major-signed/v1.0.0",
            kernel_compatibility="canonical-only; explicit repack required",
            accuracy=metrics(source, dequantize(tensor)),
        )
        c5_rows.append({"name": name, "descriptor": c5.model_dump(mode="json")})
        rows.append(
            {
                "name": name,
                "path": relative(artifact_dir),
                "artifact_id": manifest["artifact_id"],
                "canonical_metadata_sha256": manifest["canonical_metadata_sha256"],
                "q_equal": loaded.qvalues == tensor.qvalues,
                "scale_equal": loaded.scales == tensor.scales,
                "zero_equal": loaded.zero_points == tensor.zero_points,
                "dequant_max_diff": max(
                    abs(left - right)
                    for left, right in zip(flatten(dequantize(loaded)), flatten(dequantize(tensor)))
                ),
                "loaded_id_equal": loaded_manifest["artifact_id"] == manifest["artifact_id"],
            }
        )
    write_json(out / "c5_descriptors.json", {"schema": "hqsb.c5/examples", "rows": c5_rows})
    return {"source": source, "rows": rows, "package": package}


def collect_roundtrip(out: Path, examples: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for source_row in examples["rows"]:
        artifact_dir = Path(source_row["path"])
        if not artifact_dir.is_absolute():
            artifact_dir = REPO / artifact_dir
        loaded, manifest = load_quant_artifact(artifact_dir)
        with tempfile.TemporaryDirectory(prefix="hqsb-e05-01-resave-") as temporary:
            second = save_quant_artifact(
                Path(temporary) / "artifact",
                loaded,
                source_model_hash=SOURCE_HASH,
                source_artifact_id="synthetic:E05-01-v1",
                tensor_name="golden.weight",
                created_at_utc="2099-01-01T00:00:00Z",
            )
        code = (
            "import json,sys; from hqsb.quant.artifact import load_quant_artifact; "
            "from hqsb.quant.rtn import dequantize; "
            "t,m=load_quant_artifact(sys.argv[1]); "
            "print(json.dumps({'artifact_id':m['artifact_id'],'q':list(t.qvalues),"
            "'scales':list(t.scales),'zeros':list(t.zero_points),'dequant':dequantize(t)}))"
        )
        child = run([sys.executable, "-c", code, str(artifact_dir)], 60)
        child_payload = json.loads(child["stdout"]) if child["exit_code"] == 0 else {}
        checks = {
            "same_process_load": source_row["q_equal"]
            and source_row["scale_equal"]
            and source_row["zero_equal"]
            and source_row["dequant_max_diff"] == 0.0,
            "identity_stable_across_time_and_path": second["artifact_id"] == manifest["artifact_id"],
            "new_process_exit_zero": child["exit_code"] == 0,
            "new_process_identity_equal": child_payload.get("artifact_id") == manifest["artifact_id"],
            "new_process_q_equal": child_payload.get("q") == list(loaded.qvalues),
            "new_process_scale_equal": child_payload.get("scales") == list(loaded.scales),
            "new_process_zero_equal": child_payload.get("zeros") == list(loaded.zero_points),
            "new_process_dequant_equal": child_payload.get("dequant") == dequantize(loaded),
        }
        rows.append(
            {
                "name": source_row["name"],
                "artifact_id": manifest["artifact_id"],
                "resaved_artifact_id": second["artifact_id"],
                "child_stderr": child["stderr"],
                "checks": checks,
                "all_pass": all(checks.values()),
            }
        )
    result = {
        "schema": f"{SCHEMA}/roundtrip",
        "rows": rows,
        "all_pass": all(row["all_pass"] for row in rows),
    }
    write_json(out / "roundtrip.json", result)
    return result


def refresh_identity(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for descriptor in manifest["identity"]["files"].values():
        path = root / descriptor["path"]
        descriptor["bytes"] = path.stat().st_size
        descriptor["sha256"] = sha256_file(path)
    digest = sha256_bytes(canonical_json_bytes(manifest["identity"]))
    manifest["canonical_metadata_sha256"] = digest
    manifest["artifact_id"] = f"sha256:{digest}"
    write_json(manifest_path, manifest)


def mutate_manifest(root: Path, callback: Callable[[dict[str, Any]], None]) -> None:
    path = root / "manifest.json"
    value = json.loads(path.read_text())
    callback(value)
    write_json(path, value)


def collect_invalid_cases(out: Path, examples: dict[str, Any]) -> dict[str, Any]:
    symmetric_source = examples["package"] / "rtn_w4_symmetric_group4_tail"
    asymmetric_source = examples["package"] / "rtn_w8_asymmetric_channel"
    cases: list[dict[str, Any]] = []

    def execute(
        name: str,
        expected: str,
        mutation: Callable[[Path], None] | None = None,
        *,
        base: Path = symmetric_source,
        kernel: str | None = None,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix=f"hqsb-e05-01-invalid-{name}-") as temporary:
            root = Path(temporary) / "artifact"
            shutil.copytree(base, root)
            if mutation:
                mutation(root)
            try:
                load_quant_artifact(root, expected_kernel_id=kernel)
                observed, message, rejected = "ACCEPTED", "", False
            except ArtifactValidationError as exc:
                observed, message, rejected = exc.reason_code, str(exc), True
            except Exception as exc:
                observed, message, rejected = type(exc).__name__, str(exc), True
            cases.append(
                {
                    "name": name,
                    "expected_reason": expected,
                    "observed_reason": observed,
                    "message": message,
                    "rejected_before_return": rejected,
                    "pass": rejected and observed == expected,
                }
            )

    execute(
        "unknown_version",
        "VERSION_UNSUPPORTED",
        lambda root: mutate_manifest(root, lambda value: value.update(version="99.0.0")),
    )
    execute(
        "unknown_layout",
        "LAYOUT_UNSUPPORTED",
        lambda root: mutate_manifest(
            root,
            lambda value: value["identity"]["tensor"].update(original_layout="column-major"),
        ),
    )
    execute(
        "wrong_axis",
        "GROUP_AXIS_INVALID",
        lambda root: mutate_manifest(
            root, lambda value: value["identity"]["quantization"].update(axis=0)
        ),
    )
    execute(
        "zero_group_size",
        "GROUP_AXIS_INVALID",
        lambda root: mutate_manifest(
            root, lambda value: value["identity"]["quantization"].update(group_size=0)
        ),
    )
    execute(
        "shape_mismatch",
        "SHAPE_MISMATCH",
        lambda root: mutate_manifest(
            root, lambda value: value["identity"]["tensor"].update(original_shape=[2, 8])
        ),
    )
    execute(
        "size_overflow",
        "SIZE_OVERFLOW",
        lambda root: mutate_manifest(
            root,
            lambda value: value["identity"]["tensor"].update(original_shape=[2**39, 4]),
        ),
    )
    execute(
        "short_q_data",
        "DATA_SIZE_MISMATCH",
        lambda root: (root / "qvalues.bin").write_bytes((root / "qvalues.bin").read_bytes()[:-1]),
    )
    execute(
        "extra_q_data",
        "DATA_SIZE_MISMATCH",
        lambda root: (root / "qvalues.bin").write_bytes((root / "qvalues.bin").read_bytes() + b"\0"),
    )

    def flip_checksum(root: Path) -> None:
        path = root / "qvalues.bin"
        payload = bytearray(path.read_bytes())
        payload[0] ^= 1
        path.write_bytes(payload)

    execute("bad_checksum", "CHECKSUM_MISMATCH", flip_checksum)

    def illegal_padding(root: Path) -> None:
        path = root / "qvalues.bin"
        payload = bytearray(path.read_bytes())
        payload[-1] |= 0xF0
        path.write_bytes(payload)
        refresh_identity(root)

    execute("illegal_odd_padding", "PACKING_PADDING_INVALID", illegal_padding)

    def nan_scale(root: Path) -> None:
        path = root / "scales.bin"
        payload = bytearray(path.read_bytes())
        payload[:4] = struct.pack("<f", float("nan"))
        path.write_bytes(payload)
        refresh_identity(root)

    execute("nan_scale", "SCALE_INVALID", nan_scale)

    def bad_zero(root: Path) -> None:
        path = root / "zero_points.bin"
        payload = bytearray(path.read_bytes())
        payload[:4] = struct.pack("<i", 10_000)
        path.write_bytes(payload)
        refresh_identity(root)

    execute("zero_point_out_of_range", "ZERO_POINT_INVALID", bad_zero, base=asymmetric_source)
    execute(
        "unknown_packing_version",
        "PACKING_LAYOUT_UNSUPPORTED",
        lambda root: mutate_manifest(
            root,
            lambda value: value["identity"]["canonical_packing"].update(version="2.0.0"),
        ),
    )
    execute(
        "wrong_endian",
        "PACKING_LAYOUT_UNSUPPORTED",
        lambda root: mutate_manifest(
            root,
            lambda value: value["identity"]["canonical_packing"].update(byte_order="big"),
        ),
    )
    execute("missing_scale_file", "FILE_MISSING", lambda root: (root / "scales.bin").unlink())
    execute("kernel_variant_mismatch", "KERNEL_INCOMPATIBLE", kernel="cutlass-sm87-w4")
    result = {
        "schema": f"{SCHEMA}/invalid_cases",
        "case_count": len(cases),
        "rows": cases,
        "all_pass": all(row["pass"] for row in cases),
    }
    write_json(out / "invalid_cases.json", result)
    return result


def collect_pack_properties(out: Path) -> dict[str, Any]:
    deterministic_values = [((index * 13 + 5) % 16) - 8 for index in range(4097)]
    rows = []
    for shape in ((1, 1), (1, 16), (3, 5), (17, 17), (1, 4097)):
        count = shape[0] * shape[1]
        values = deterministic_values[:count]
        packed = pack_int4(values, shape)
        unpacked = unpack_int4(packed, shape)
        repacked = pack_int4(unpacked, shape)
        rows.append(
            {
                "shape": list(shape),
                "logical_count": count,
                "packed_bytes": len(packed),
                "packed_sha256": sha256_bytes(packed),
                "unpack_pack_equal": tuple(values) == unpacked,
                "pack_unpack_bytes_stable": repacked == packed,
            }
        )
    result = {
        "schema": f"{SCHEMA}/pack_properties",
        "rows": rows,
        "all_pass": all(
            row["unpack_pack_equal"] and row["pack_unpack_bytes_stable"] for row in rows
        ),
    }
    write_json(out / "pack_properties.json", result)
    return result


def collect_cross_implementation(out: Path) -> dict[str, Any]:
    values = [-7.0, -6.5, -5.5, -0.5, 0.0, 0.5, 1.5, 6.5, 7.0]
    expected = [-7, -6, -6, 0, 0, 0, 2, 6, 7]
    result: dict[str, Any] = {
        "schema": f"{SCHEMA}/cross_implementation",
        "values": values,
        "expected": expected,
        "python_builtin_round": [round(value) for value in values],
        "explicit_spec_round": [round_nearest_even(value) for value in values],
        "torch": {"available": False},
        "cuda_pack_unpack": {
            "status": "NOT_IMPLEMENTED_OPTIONAL",
            "reason": "canonical portable CPU packing is the E05-01 oracle; kernel packing belongs to E05-06",
        },
    }
    try:
        import torch

        cpu = torch.round(torch.tensor(values, dtype=torch.float64)).to(torch.int64).tolist()
        torch_result: dict[str, Any] = {"available": True, "cpu": cpu, "cpu_equal": cpu == expected}
        if torch.cuda.is_available():
            cuda = (
                torch.round(torch.tensor(values, dtype=torch.float32, device="cuda"))
                .to(torch.int32)
                .cpu()
                .tolist()
            )
            torch_result.update({"cuda": cuda, "cuda_equal": cuda == expected})
        result["torch"] = torch_result
    except Exception as exc:
        result["torch"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    required_checks = {
        "python_builtin": result["python_builtin_round"] == expected,
        "explicit_spec": result["explicit_spec_round"] == expected,
        "torch_cpu": result["torch"].get("cpu_equal", False),
    }
    if "cuda_equal" in result["torch"]:
        required_checks["torch_cuda"] = result["torch"]["cuda_equal"]
    result["required_checks"] = required_checks
    result["all_pass"] = all(required_checks.values())
    write_json(out / "cross_implementation.json", result)
    return result


def size_case(out: Path, name: str, spec: RtnSpec, rows: int, cols: int) -> dict[str, Any]:
    source = synthetic_matrix(rows, cols, len(name))
    tensor = quantize(source, spec)
    artifact_dir = out / "size_artifacts" / name
    manifest = save_quant_artifact(
        artifact_dir,
        tensor,
        source_model_hash=SOURCE_HASH,
        tensor_name=f"size.{name}",
    )
    groups = len(tensor.scales)
    predicted_q = rows * (((cols + 1) // 2) if spec.bits == 4 else cols)
    predicted_scale = groups * 4
    predicted_zero = 0 if spec.symmetric else groups * 4
    predicted_payload = predicted_q + predicted_scale + predicted_zero
    actual_files = manifest["identity"]["files"]
    actual_payload = sum(item["bytes"] for item in actual_files.values())
    manifest_bytes = (artifact_dir / "manifest.json").stat().st_size
    return {
        "name": name,
        "shape": [rows, cols],
        "spec": spec.as_dict(),
        "fp16_raw_bytes": rows * cols * 2,
        "groups": groups,
        "prediction": {
            "q_bytes": predicted_q,
            "scale_bytes": predicted_scale,
            "zero_bytes": predicted_zero,
            "payload_bytes": predicted_payload,
        },
        "actual": {
            "q_bytes": actual_files["qvalues"]["bytes"],
            "scale_bytes": actual_files["scales"]["bytes"],
            "zero_bytes": actual_files.get("zero_points", {}).get("bytes", 0),
            "payload_bytes": actual_payload,
            "manifest_bytes": manifest_bytes,
            "on_disk_total_bytes": actual_payload + manifest_bytes,
            "memory_mapped_payload_bytes": actual_payload,
            "loaded_logical_host_payload_bytes": rows * cols + groups * 4 + predicted_zero,
        },
        "residual_bytes": actual_payload - predicted_payload,
        "payload_compression_vs_fp16": (rows * cols * 2) / actual_payload,
        "on_disk_compression_vs_fp16": (rows * cols * 2) / (actual_payload + manifest_bytes),
        "artifact_id": manifest["artifact_id"],
    }


def collect_size_model(out: Path) -> dict[str, Any]:
    cases = [
        size_case(out, "w4_sym_o3_k17_g16", RtnSpec(4, True, "per-group", 16), 3, 17),
        size_case(out, "w4_asym_o3_k17_g16", RtnSpec(4, False, "per-group", 16), 3, 17),
        size_case(out, "w8_sym_o3_k17_g16", RtnSpec(8, True, "per-group", 16), 3, 17),
        size_case(out, "w8_asym_o3_k17_g16", RtnSpec(8, False, "per-group", 16), 3, 17),
        size_case(out, "w4_sym_o64_k2051_g32", RtnSpec(4, True, "per-group", 32), 64, 2051),
    ]
    device: dict[str, Any] = {"available": False}
    try:
        import torch

        if torch.cuda.is_available():
            artifact_dir = out / "size_artifacts" / "w4_sym_o64_k2051_g32"
            tensor, _ = load_quant_artifact(artifact_dir)
            qbytes = (artifact_dir / "qvalues.bin").read_bytes()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            q_device = torch.tensor(list(qbytes), dtype=torch.uint8, device="cuda")
            scale_device = torch.tensor(list(tensor.scales), dtype=torch.float32, device="cuda")
            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()
            device = {
                "available": True,
                "case": "w4_sym_o64_k2051_g32",
                "tensor_payload_bytes": q_device.numel() * q_device.element_size()
                + scale_device.numel() * scale_device.element_size(),
                "allocator_live_delta_bytes": after - before,
                "allocator_peak_absolute_bytes": peak,
                "note": "allocator delta is reported separately from exact tensor payload",
            }
            del q_device, scale_device
            torch.cuda.synchronize()
        else:
            device = {"available": False, "reason": "torch.cuda unavailable"}
    except Exception as exc:
        device = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    result = {
        "schema": f"{SCHEMA}/size_model",
        "formula": "q + FP32 scales + optional INT32 zero-points; W4 rows are byte-aligned",
        "rows": cases,
        "device_allocation": device,
        "all_payload_residuals_zero": all(row["residual_bytes"] == 0 for row in cases),
    }
    write_json(out / "size_model.json", result)
    return result


def measure(operation: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    tracemalloc.start()
    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    value = operation()
    wall_ns = time.perf_counter_ns() - wall_start
    cpu_ns = time.process_time_ns() - cpu_start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, {"wall_ms": wall_ns / 1e6, "cpu_ms": cpu_ns / 1e6, "python_peak_bytes": peak}


def collect_offline_cost(out: Path) -> dict[str, Any]:
    source = synthetic_matrix(64, 2051, SEED)
    spec = RtnSpec(4, True, "per-group", 32)
    cost_dir = out / "offline_cost_artifact"
    samples = []
    last_tensor: QuantizedTensor | None = None
    for repeat in range(3):
        tensor, quant_cost = measure(lambda: quantize(source, spec))
        packed, pack_cost = measure(lambda: pack_int4(tensor.qvalues, tensor.original_shape))
        _manifest, save_cost = measure(
            lambda: save_quant_artifact(
                cost_dir,
                tensor,
                source_model_hash=SOURCE_HASH,
                tensor_name="offline_cost.weight",
            )
        )
        loaded_pair, load_cost = measure(lambda: load_quant_artifact(cost_dir))
        _reconstruction, dequant_cost = measure(lambda: dequantize(loaded_pair[0]))
        samples.append(
            {
                "repeat": repeat,
                "quantize": quant_cost,
                "pack": pack_cost,
                "save": save_cost,
                "load_validate_unpack": load_cost,
                "dequantize": dequant_cost,
                "packed_bytes": len(packed),
            }
        )
        last_tensor = tensor
    operations = ["quantize", "pack", "save", "load_validate_unpack", "dequantize"]
    summary = {}
    for operation in operations:
        summary[operation] = {
            metric: sorted(sample[operation][metric] for sample in samples)[1]
            for metric in ("wall_ms", "cpu_ms", "python_peak_bytes")
        }
    result = {
        "schema": f"{SCHEMA}/offline_cost",
        "shape": [64, 2051],
        "spec": spec.as_dict(),
        "samples": samples,
        "median": summary,
        "q_sha256": q_hash(last_tensor.qvalues if last_tensor else []),
        "scope": "offline Python reference cost only; excludes model inference and kernel execution",
    }
    write_json(out / "offline_cost.json", result)
    return result


def evidence_manifest(out: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in out.rglob("*") if item.is_file()):
        if path.name == "EVIDENCE_MANIFEST.json":
            continue
        rows.append(
            {
                "path": str(path.relative_to(out)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    value = {
        "schema": f"{SCHEMA}/evidence_manifest",
        "experiment_id": EXPERIMENT,
        "generated_at_utc": utc_now(),
        "file_count": len(rows),
        "files": rows,
    }
    write_json(out / "EVIDENCE_MANIFEST.json", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "protocol.json", protocol())
    write_json(out / "environment.json", environment())
    write_json(out / "quant_artifact_schema.json", schema_document())

    golden = collect_golden(out)
    boundaries = collect_input_boundaries(out)
    factorial = collect_factorial(out)
    pack = collect_pack_properties(out)
    examples = create_example_artifacts(out)
    roundtrip = collect_roundtrip(out, examples)
    invalid = collect_invalid_cases(out, examples)
    cross = collect_cross_implementation(out)
    size = collect_size_model(out)
    offline = collect_offline_cost(out)

    conditions = {
        "1_int8_int4_symmetric_asymmetric_math_complete": factorial["all_pass"],
        "2_range_round_constant_policies_unique": golden["all_pass"],
        "3_tensor_channel_group_axis_correct": factorial["all_pass"]
        and factorial["coverage"]["granularity"]
        == ["per-channel", "per-group", "per-tensor"],
        "4_all_tail_groups_safe": factorial["all_pass"]
        and factorial["coverage"]["tail_case_count"] > 0,
        "5_all_int4_codes_and_odd_pack_roundtrip": golden["all_pass"] and pack["all_pass"],
        "6_new_process_save_load_dequant_consistent": roundtrip["all_pass"],
        "7_canonical_hash_stable": roundtrip["all_pass"],
        "8_size_formula_closes": size["all_payload_residuals_zero"],
        "9_invalid_artifacts_rejected_before_return": invalid["all_pass"],
        "10_golden_property_raw_complete": factorial["case_count"] > 0
        and (out / "factorial_cases.jsonl.gz").is_file()
        and boundaries["all_pass"]
        and cross["all_pass"]
        and bool(offline["samples"]),
    }
    verdict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT,
        "generated_at_utc": utc_now(),
        "expected_effect": "freeze quantization mathematics and artifact semantics",
        "single_item_standard": "tail/boundary groups correct; same artifact reloads identically; illegal layout/version rejected",
        "conditions": conditions,
        "passed_conditions": [name for name, value in conditions.items() if value],
        "failed_conditions": [name for name, value in conditions.items() if not value],
        "expected_effect_met": all(conditions.values()),
        "single_item_standard_met": all(conditions.values()),
        "overall": "PASS" if all(conditions.values()) else "FAIL",
        "no_pass_negative": True,
        "scope_boundary": "proves RTN math and canonical artifact correctness; does not prove model quality or native low-bit speedup",
    }
    write_json(out / "verdict.json", verdict)
    evidence_manifest(out)
    print(json.dumps(verdict, ensure_ascii=False, sort_keys=True))
    return 0 if verdict["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
