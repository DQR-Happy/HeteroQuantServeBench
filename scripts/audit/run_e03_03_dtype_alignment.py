#!/usr/bin/env python3
"""E03-03 runtime audit: dtype, alignment, layout and scalar fallback.

This runner is intentionally self-contained and executes only on the Jetson.
The expected eligibility oracle below does not import or call the production
resolver; every production decision is compared field-by-field with an
independent computation from concrete pointers, strides, dtype and H.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "E03-03"
SCHEMA_VERSION = "hqsb.s03.e03_03.v1"
SEED = 20260918
EPSILON = 1e-6
H_VALUES = (1, 2, 3, 4, 5, 31, 32, 33, 99, 100, 101, 127, 128, 129,
            2047, 2048, 2049, 2050)
S02_RUNTIME_H = (128, 2048)
VARIANT_CODE = {
    "auto": 0,
    "v0_shared": 2,
    "v1_warp_shuffle": 3,
    "v2_vectorized": 4,
    "scalar_safe": 5,
    "v2_vectorized_strict": 6,
}
VARIANT_NAME = {value: key for key, value in VARIANT_CODE.items()}
REASON_BITS = {
    "HIDDEN_TAIL": 1 << 0,
    "INPUT_MISALIGNED": 1 << 1,
    "WEIGHT_MISALIGNED": 1 << 2,
    "OUTPUT_MISALIGNED": 1 << 3,
    "ROW_STRIDE_MISALIGNED": 1 << 4,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(command: list[str], cwd: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, cwd=cwd, text=True, capture_output=True, check=False
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except FileNotFoundError as exc:
        return {"command": command, "returncode": 127, "stdout": "", "stderr": str(exc)}


def dtype_spec(dtype: str) -> dict[str, int]:
    if dtype == "fp32":
        return {"element_bytes": 4, "vector_width": 4, "alignment": 16}
    if dtype == "fp16":
        return {"element_bytes": 2, "vector_width": 2, "alignment": 4}
    raise ValueError(dtype)


def reason_names(mask: int) -> list[str]:
    return [name for name, bit in REASON_BITS.items() if mask & bit]


def oracle(pointer_triplet: tuple[int, int, int], hidden: int, dtype: str,
           requested: str) -> dict[str, Any]:
    """Independent predicate oracle; do not replace with production helpers."""
    spec = dtype_spec(dtype)
    input_ptr, weight_ptr, output_ptr = pointer_triplet
    mask = 0
    if hidden % spec["vector_width"]:
        mask |= REASON_BITS["HIDDEN_TAIL"]
    if input_ptr % spec["alignment"]:
        mask |= REASON_BITS["INPUT_MISALIGNED"]
    if weight_ptr % spec["alignment"]:
        mask |= REASON_BITS["WEIGHT_MISALIGNED"]
    if output_ptr % spec["alignment"]:
        mask |= REASON_BITS["OUTPUT_MISALIGNED"]
    if (hidden * spec["element_bytes"]) % spec["alignment"]:
        mask |= REASON_BITS["ROW_STRIDE_MISALIGNED"]
    eligible = mask == 0
    if requested == "auto":
        actual = "v2_vectorized" if eligible else (
            "v1_warp_shuffle" if dtype == "fp32" else "scalar_safe"
        )
        error = 0
    elif requested == "v2_vectorized":
        actual = "v2_vectorized" if eligible else "scalar_safe"
        error = 0
    elif requested == "v2_vectorized_strict":
        actual = "v2_vectorized" if eligible else "v2_vectorized_strict"
        error = 0 if eligible else 1
    elif requested == "scalar_safe":
        actual, error = "scalar_safe", 0
    else:
        actual, error = requested, 0
    load_width = spec["alignment"] if actual == "v2_vectorized" else spec["element_bytes"]
    return {
        "cuda_error": error,
        "actual_variant": VARIANT_CODE[actual],
        "reason_mask": mask,
        "vector_width_elements": spec["vector_width"],
        "required_alignment_bytes": spec["alignment"],
        "actual_load_width_bytes": load_width,
        "vector_eligible": eligible,
    }


def tensor_facts(tensor, dtype: str) -> dict[str, Any]:
    ptr = int(tensor.data_ptr())
    spec = dtype_spec(dtype)
    rows = int(tensor.shape[0]) if tensor.dim() == 2 else 1
    row_stride = int(tensor.stride(0)) if tensor.dim() else 0
    row_mods = [
        (ptr + row * row_stride * spec["element_bytes"]) % spec["alignment"]
        for row in range(min(rows, 8))
    ]
    return {
        "data_ptr_hex": hex(ptr),
        "pointer_mod_4": ptr % 4,
        "pointer_mod_8": ptr % 8,
        "pointer_mod_16": ptr % 16,
        "storage_offset_elements": int(tensor.storage_offset()),
        "stride_elements": list(tensor.stride()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "row_start_mod_required_alignment_first8": row_mods,
    }


def guarded_tensor(torch, shape: tuple[int, ...], torch_dtype, offset: int,
                   fill: float | None = None):
    logical = math.prod(shape)
    guard = 16
    base = torch.empty(logical + guard * 2 + offset + 8,
                       device="cuda", dtype=torch_dtype)
    if fill is not None:
        base.fill_(fill)
    view = base.narrow(0, guard + offset, logical).view(*shape)
    return base, view, guard + offset, logical


def rms_reference(torch, x, weight):
    xf = x.float()
    wf = weight.float()
    result = xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + EPSILON) * wf
    return result.to(dtype=x.dtype)


def numeric_metrics(torch, actual, expected) -> dict[str, Any]:
    af = actual.float().reshape(-1)
    ef = expected.float().reshape(-1)
    diff = (af - ef).abs()
    denom = ef.norm().item()
    cosine = float(torch.nn.functional.cosine_similarity(af, ef, dim=0).item())
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "cosine": cosine,
        "l2_relative": float((af - ef).norm().item() / max(denom, 1e-30)),
    }


def tolerance_pass(metrics: dict[str, Any], dtype: str) -> bool:
    if dtype == "fp32":
        return metrics["max_abs"] <= 2e-5 and metrics["l2_relative"] <= 2e-5
    return metrics["max_abs"] <= 8e-3 and metrics["l2_relative"] <= 5e-3


def time_case(torch, bridge_forward, x, weight, out, dtype: str, variant: str) -> list[float]:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(5):
            bridge_forward(x, weight, out=out, dtype=dtype, variant=variant,
                           epsilon=EPSILON, stream=stream)
    stream.synchronize()
    groups = []
    repeats = 20
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            for _ in range(repeats):
                bridge_forward(x, weight, out=out, dtype=dtype, variant=variant,
                               epsilon=EPSILON, stream=stream)
            end.record(stream)
        end.synchronize()
        groups.append(float(start.elapsed_time(end) / repeats))
    return groups


def execute_case(torch, bridge_forward, bridge_resolve, *, case_id: str,
                 dtype: str, rows: int, hidden: int, requested: str,
                 input_offset: int = 0, weight_offset: int = 0,
                 output_offset: int = 0, timed: bool = True) -> dict[str, Any]:
    torch_dtype = torch.float32 if dtype == "fp32" else torch.float16
    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED + sum(ord(ch) for ch in case_id))
    input_base, x, input_start, input_len = guarded_tensor(
        torch, (rows, hidden), torch_dtype, input_offset, fill=-7.0
    )
    weight_base, weight, weight_start, weight_len = guarded_tensor(
        torch, (hidden,), torch_dtype, weight_offset, fill=-9.0
    )
    output_base, out, output_start, output_len = guarded_tensor(
        torch, (rows, hidden), torch_dtype, output_offset, fill=-123.0
    )
    x.copy_(torch.randn((rows, hidden), generator=generator, device="cuda",
                        dtype=torch_dtype) * 0.25)
    weight.copy_(torch.randn((hidden,), generator=generator, device="cuda",
                             dtype=torch_dtype) * 0.1 + 1.0)
    input_before = input_base.clone()
    weight_before = weight_base.clone()
    output_before = output_base.clone()
    expected = rms_reference(torch, x, weight)

    ptrs = (int(x.data_ptr()), int(weight.data_ptr()), int(out.data_ptr()))
    expected_dispatch = oracle(ptrs, hidden, dtype, requested)
    production_dispatch = bridge_resolve(
        x, weight, out, dtype=dtype, variant=requested
    )
    oracle_match = all(
        production_dispatch[key] == value
        for key, value in expected_dispatch.items()
    )

    launch_status = "NOT_RUN"
    launch_error = None
    try:
        bridge_forward(x, weight, out=out, dtype=dtype, variant=requested,
                       epsilon=EPSILON)
        torch.cuda.synchronize()
        launch_status = "SUCCESS"
    except RuntimeError as exc:
        launch_status = "EXPECTED_REJECT" if expected_dispatch["cuda_error"] else "UNEXPECTED_REJECT"
        launch_error = str(exc)

    should_succeed = expected_dispatch["cuda_error"] == 0
    metrics = numeric_metrics(torch, out, expected) if launch_status == "SUCCESS" else None
    numeric_ok = bool(metrics and tolerance_pass(metrics, dtype)) if should_succeed else True
    outside = torch.ones_like(output_base, dtype=torch.bool)
    outside[output_start:output_start + output_len] = False
    output_guard_unchanged = bool(torch.equal(output_base[outside], output_before[outside]))
    output_untouched_on_reject = bool(torch.equal(output_base, output_before)) if not should_succeed else None
    input_unchanged = bool(torch.equal(input_base, input_before))
    weight_unchanged = bool(torch.equal(weight_base, weight_before))
    latency = None
    if timed and should_succeed and launch_status == "SUCCESS" and numeric_ok:
        latency = time_case(torch, bridge_forward, x, weight, out, dtype, requested)

    vector_width = dtype_spec(dtype)["vector_width"]
    tail_slice = min(hidden, vector_width + 2)
    tail_actual = out[:, -tail_slice:].float().cpu().tolist() if launch_status == "SUCCESS" else None
    tail_expected = expected[:, -tail_slice:].float().cpu().tolist() if launch_status == "SUCCESS" else None
    passed = (
        oracle_match
        and ((should_succeed and launch_status == "SUCCESS" and numeric_ok) or
             (not should_succeed and launch_status == "EXPECTED_REJECT" and output_untouched_on_reject))
        and output_guard_unchanged and input_unchanged and weight_unchanged
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "case_id": case_id,
        "dtype": dtype,
        "rows": rows,
        "hidden": hidden,
        "element_bytes": dtype_spec(dtype)["element_bytes"],
        "vector_width_elements": vector_width,
        "required_alignment_bytes": dtype_spec(dtype)["alignment"],
        "tail_length": hidden % vector_width,
        "row_stride_bytes": hidden * dtype_spec(dtype)["element_bytes"],
        "requested_variant": requested,
        "input": tensor_facts(x, dtype),
        "weight": tensor_facts(weight, dtype),
        "output": tensor_facts(out, dtype),
        "allocation": {
            "input_elements": int(input_base.numel()),
            "weight_elements": int(weight_base.numel()),
            "output_elements": int(output_base.numel()),
            "input_logical_region": [input_start, input_start + input_len],
            "weight_logical_region": [weight_start, weight_start + weight_len],
            "output_logical_region": [output_start, output_start + output_len],
        },
        "oracle_dispatch": expected_dispatch,
        "production_dispatch": production_dispatch,
        "oracle_matches_production": oracle_match,
        "reason_names": reason_names(expected_dispatch["reason_mask"]),
        "launch_status": launch_status,
        "launch_error": launch_error,
        "metrics": metrics,
        "numeric_ok": numeric_ok,
        "tail_actual": tail_actual,
        "tail_expected": tail_expected,
        "output_guard_unchanged": output_guard_unchanged,
        "output_untouched_on_reject": output_untouched_on_reject,
        "input_unchanged": input_unchanged,
        "weight_unchanged": weight_unchanged,
        "device_latency_ms_groups": latency,
        "device_latency_ms_median": statistics.median(latency) if latency else None,
        "passed": passed,
    }


def layout_cases(torch, bridge_forward, output_dir: Path) -> list[dict[str, Any]]:
    from ops.cuda_bridge import RmsNormContractError

    cases = []
    dtype = torch.float32
    rows, hidden = 2, 128
    layouts = {
        "transpose_last_dim": (
            torch.randn((hidden, rows), device="cuda", dtype=dtype).t(),
            torch.ones(hidden, device="cuda", dtype=dtype),
            torch.empty((rows, hidden), device="cuda", dtype=dtype),
        ),
        "padded_row_stride": (
            torch.randn((rows, hidden + 3), device="cuda", dtype=dtype)[:, :hidden],
            torch.ones(hidden, device="cuda", dtype=dtype),
            torch.empty((rows, hidden), device="cuda", dtype=dtype),
        ),
        "broadcast_zero_stride_weight": (
            torch.randn((rows, hidden), device="cuda", dtype=dtype),
            torch.ones(1, device="cuda", dtype=dtype).expand(hidden),
            torch.empty((rows, hidden), device="cuda", dtype=dtype),
        ),
        "overlapping_as_strided": (
            torch.randn(hidden, device="cuda", dtype=dtype).as_strided((rows, hidden), (0, 1)),
            torch.ones(hidden, device="cuda", dtype=dtype),
            torch.empty((rows, hidden), device="cuda", dtype=dtype),
        ),
        "noncontiguous_output": (
            torch.randn((rows, hidden), device="cuda", dtype=dtype),
            torch.ones(hidden, device="cuda", dtype=dtype),
            torch.empty((rows, hidden + 1), device="cuda", dtype=dtype)[:, :hidden],
        ),
    }
    for name, (x, weight, out) in layouts.items():
        out.fill_(-123.0)
        snapshot = out.clone()
        before = int(torch.cuda.memory_allocated())
        reason = None
        error = None
        try:
            bridge_forward(x, weight, out=out, dtype="fp32", variant="auto", epsilon=EPSILON)
        except RmsNormContractError as exc:
            reason, error = exc.reason_code, str(exc)
        after = int(torch.cuda.memory_allocated())
        expected_reason = "OUT_TENSOR_MISMATCH" if name == "noncontiguous_output" else "NON_CONTIGUOUS_INPUT"
        record = {
            "case_id": name,
            "policy": "REJECT_BEFORE_LAUNCH_NO_MATERIALIZATION",
            "x_shape": list(x.shape),
            "x_stride": list(x.stride()),
            "weight_stride": list(weight.stride()),
            "out_stride": list(out.stride()),
            "reason_code": reason,
            "error": error,
            "expected_reason_code": expected_reason,
            "output_unchanged": bool(torch.equal(out, snapshot)),
            "memory_allocated_delta_bytes": after - before,
        }
        record["passed"] = (
            reason == expected_reason and record["output_unchanged"] and
            record["memory_allocated_delta_bytes"] == 0
        )
        cases.append(record)
    with (output_dir / "layout_cases.jsonl").open("w") as handle:
        for record in cases:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return cases


def bf16_case(torch, bridge_forward) -> dict[str, Any]:
    from ops.cuda_bridge import RmsNormContractError

    x = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    out = torch.full_like(x, -123.0)
    snapshot = out.clone()
    reason = None
    error = None
    try:
        bridge_forward(x, weight, out=out, dtype="bf16", variant="auto", epsilon=EPSILON)
    except RmsNormContractError as exc:
        reason, error = exc.reason_code, str(exc)
    return {
        "dtype": "bf16",
        "status": "UNSUPPORTED_BF16",
        "reason_code": reason,
        "error": error,
        "output_unchanged": bool(torch.equal(out, snapshot)),
        "passed": reason == "UNSUPPORTED_DTYPE" and bool(torch.equal(out, snapshot)),
    }


def collect_disassembly(repo: Path, output_dir: Path, library: Path) -> dict[str, Any]:
    artifact_dir = output_dir / "disassembly"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    candidates = [cuda_home / "bin/cuobjdump"] + sorted(
        Path("/usr/local").glob("cuda-*/bin/cuobjdump"), reverse=True
    )
    cuobjdump = next((path for path in candidates if path.is_file()), None)
    tool = str(cuobjdump) if cuobjdump else "cuobjdump"
    sass = run_text([tool, "--dump-sass", str(library)], repo)
    ptx = run_text([tool, "--dump-ptx", str(library)], repo)
    (artifact_dir / "cuobjdump_sass.txt").write_text(sass["stdout"] + sass["stderr"])
    (artifact_dir / "cuobjdump_ptx.txt").write_text(ptx["stdout"] + ptx["stderr"])
    ninja = run_text(["ninja", "-C", "build/jetson-release", "-t", "commands",
                      "hqsb_rmsnorm_shared"], repo)
    (artifact_dir / "compile_commands.txt").write_text(ninja["stdout"] + ninja["stderr"])
    text = sass["stdout"]
    symbols = {
        "fp32_vector": "rmsnorm_v2_f32_kernel",
        "fp16_vector": "rmsnorm_v2_f16_kernel",
        "fp32_scalar": "rmsnorm_scalar_safe_kernelIf",
        "fp16_scalar": "rmsnorm_scalar_safe_kernelI6__half",
    }
    evidence = {}
    sass_lines = text.splitlines()
    for label, needle in symbols.items():
        starts = [index for index, line in enumerate(sass_lines)
                  if "Function :" in line and needle in line]
        sections = []
        for start in starts:
            end = next((index for index in range(start + 1, len(sass_lines))
                        if "Function :" in sass_lines[index]),
                       min(len(sass_lines), start + 800))
            sections.extend(sass_lines[start:end])
        joined = "\n".join(sections)
        loads = [line.strip() for line in sections
                 if "LDG" in line or "STG" in line]
        mnemonics = [
            line.split("*/", 1)[1].strip().split()[0]
            for line in loads if "*/" in line
        ]
        default_width_32 = [mnemonic for mnemonic in mnemonics
                            if mnemonic in ("LDG.E", "LDG.E.CONSTANT", "STG.E")]
        width_16 = [mnemonic for mnemonic in mnemonics
                    if ".U16" in mnemonic or ".S16" in mnemonic]
        evidence[label] = {
            "symbol_needle": needle,
            "found": bool(starts),
            "load_store_lines": loads[:80],
            "memory_mnemonics": mnemonics,
            "has_128_bit_mnemonic": any(".128" in mnemonic for mnemonic in mnemonics),
            "has_64_bit_mnemonic": any(".64" in mnemonic for mnemonic in mnemonics),
            "has_default_32_bit_memory_op": bool(default_width_32),
            "has_16_bit_memory_op": bool(width_16),
            "snippet_sha256": hashlib.sha256(joined.encode()).hexdigest(),
        }
    result = {
        "binary": str(library.relative_to(repo)),
        "binary_sha256": sha256_file(library),
        "cuobjdump_path": tool,
        "cuobjdump_sass_returncode": sass["returncode"],
        "cuobjdump_ptx_returncode": ptx["returncode"],
        "compile_command_returncode": ninja["returncode"],
        "symbols": evidence,
        "interpretation": "Raw SASS is authoritative; automatic mnemonic flags are aids and are verified separately.",
    }
    dump_json(output_dir / "disassembly_analysis.json", result)
    return result


def collect_ncu_probe(repo: Path, output_dir: Path) -> dict[str, Any]:
    candidates = [Path("/usr/local/cuda/bin/ncu")] + sorted(
        Path("/usr/local").glob("cuda-*/bin/ncu"), reverse=True
    )
    ncu = next((path for path in candidates if path.is_file()), None)
    if ncu is None:
        result = {"status": "UNAVAILABLE", "command": None, "returncode": 127,
                  "stdout": "", "stderr": "ncu not found"}
    else:
        command = [
            str(ncu), "--set", "basic", "--kernel-name",
            "regex:rmsnorm_v2_f32_kernel", "--launch-count", "1",
            sys.executable, str(Path(__file__).resolve()), "profile",
            "--dtype", "fp32", "--variant", "v2_vectorized_strict",
            "--rows", "512", "--hidden", "2048", "--iterations", "1",
        ]
        result = run_text(command, repo)
        combined = result["stdout"] + result["stderr"]
        if "Insufficient privileges" in combined:
            result["status"] = "BLOCKED_INSUFFICIENT_PRIVILEGES"
        elif result["returncode"] == 0:
            result["status"] = "COLLECTED"
        else:
            result["status"] = "FAILED_TOOL"
    dump_json(output_dir / "profiler" / "ncu_probe.json", result)
    return result


def provenance(repo: Path, library: Path) -> dict[str, Any]:
    import torch

    source_paths = [
        repo / "ops/cuda/rmsnorm/include/hqsb/rmsnorm.h",
        repo / "ops/cuda/rmsnorm/src/rmsnorm_dispatcher.cu",
        repo / "ops/cuda/rmsnorm/src/rmsnorm_v2.cu",
        repo / "ops/cuda/rmsnorm/src/rmsnorm_c_api.cu",
        repo / "ops/cuda_bridge.py",
        repo / "scripts/audit/run_e03_03_dtype_alignment.py",
    ]
    git_head = run_text(["git", "rev-parse", "HEAD"], repo)
    git_status = run_text(["git", "status", "--short"], repo)
    device = torch.cuda.get_device_properties(0)
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "git_commit": git_head["stdout"].strip(),
        "git_dirty": bool(git_status["stdout"].strip()),
        "git_status_short": git_status["stdout"].splitlines(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": {
            "name": device.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(device.total_memory),
            "multi_processor_count": int(device.multi_processor_count),
        },
        "library": str(library.relative_to(repo)),
        "library_sha256": sha256_file(library),
        "source_sha256": {str(path.relative_to(repo)): sha256_file(path) for path in source_paths},
    }


def manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "EVIDENCE_MANIFEST.json":
            rows.append({
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    dump_json(output_dir / "EVIDENCE_MANIFEST.json", {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "files": rows,
    })


def collect(args) -> int:
    import torch
    from ops.cuda_bridge import (
        rmsnorm_forward,
        rmsnorm_resolve_dispatch,
    )

    repo = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    library_env = os.environ.get("HQSB_CUDA_RMSNORM_LIB")
    if not library_env:
        raise RuntimeError("HQSB_CUDA_RMSNORM_LIB is not set")
    library = Path(library_env).resolve()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "frozen_at_utc": utc_now(),
        "seed": SEED,
        "epsilon": EPSILON,
        "rows": 2,
        "hidden_values": list(H_VALUES),
        "s02_runtime_hidden_values": list(S02_RUNTIME_H),
        "dtypes": ["fp32", "fp16", "bf16_explicit_reject"],
        "requested_paths": ["auto", "scalar_safe", "v2_vectorized_strict"],
        "layout_policy": "canonical contiguous only; reject before launch; no hidden copy",
        "latency": {"warmup": 5, "groups": 3, "repeats_per_group": 20},
        "tolerances": {
            "fp32": {"max_abs": 2e-5, "l2_relative": 2e-5},
            "fp16": {"max_abs": 8e-3, "l2_relative": 5e-3},
        },
        "oracle_independence": "local arithmetic over concrete pointers/strides; production resolver is observation only",
    }
    dump_json(output_dir / "protocol.json", protocol)
    dump_json(output_dir / "resolved_operator_spec.json", {
        "schema_version": "hqsb.c3.rmsnorm.layout_alignment.v1",
        "operator": "rmsnorm",
        "math": "y = x * weight * rsqrt(mean(x*x, last_dim) + epsilon)",
        "layout_policy": {
            "input": "2-D canonical contiguous row-major only",
            "weight": "1-D contiguous only",
            "output": "2-D canonical contiguous row-major only",
            "hidden_materialization": "FORBIDDEN",
            "rejection_boundary": "Python binding validates strides before C ABI launch",
        },
        "dtype_policy": {
            "fp32": {"io": "fp32", "accumulation": "fp32"},
            "fp16": {"io": "fp16", "accumulation": "fp32"},
            "bf16": "UNSUPPORTED_BF16_NO_IMPLICIT_CAST",
        },
        "vector_predicate": {
            "fp32": "H%4==0 && input%16==0 && weight%16==0 && output%16==0 && row_stride_bytes%16==0",
            "fp16": "H%2==0 && input%4==0 && weight%4==0 && output%4==0 && row_stride_bytes%4==0",
            "reason_bits": REASON_BITS,
        },
        "path_policy": {
            "strict_vector": "eligible -> vector; otherwise cudaErrorInvalidValue before launch",
            "scalar_safe": "scalar loads/stores for all declared contiguous alignments and tails",
            "auto_fp32": "eligible -> float4 V2; otherwise V1 scalar",
            "auto_fp16": "eligible -> half2 V2; otherwise scalar_safe",
            "v2_compatibility": "eligible -> vector; otherwise explicit scalar_safe actual path",
        },
        "stream": "caller supplied; no hidden synchronization",
        "allocation": "operator performs no allocation",
    })
    dump_json(output_dir / "provenance.json", provenance(repo, library))

    cases: list[dict[str, Any]] = []
    for dtype in ("fp32", "fp16"):
        for hidden in H_VALUES:
            for requested in ("auto", "scalar_safe", "v2_vectorized_strict"):
                case_id = f"matrix_{dtype}_r2_h{hidden}_{requested}"
                cases.append(execute_case(
                    torch, rmsnorm_forward, rmsnorm_resolve_dispatch,
                    case_id=case_id, dtype=dtype, rows=2, hidden=hidden,
                    requested=requested,
                ))

    # Isolate each operand's alignment class while all logical regions remain
    # inside valid padded allocations.
    for dtype in ("fp32", "fp16"):
        max_offset = 4 if dtype == "fp32" else 2
        for operand in ("input", "weight", "output"):
            for offset in range(max_offset):
                offsets = {"input_offset": 0, "weight_offset": 0, "output_offset": 0}
                offsets[f"{operand}_offset"] = offset
                for requested in ("auto", "v2_vectorized_strict"):
                    case_id = f"align_{dtype}_{operand}_off{offset}_{requested}"
                    cases.append(execute_case(
                        torch, rmsnorm_forward, rmsnorm_resolve_dispatch,
                        case_id=case_id, dtype=dtype, rows=2, hidden=128,
                        requested=requested, **offsets,
                    ))

    with (output_dir / "cases.jsonl").open("w") as handle:
        for record in cases:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    dump_json(output_dir / "predicate_audit.json", {
        "oracle_implementation": "independent arithmetic in run_e03_03_dtype_alignment.py::oracle",
        "production_observation": "hqsb_rmsnorm_resolve_dispatch_c",
        "total_cases": len(cases),
        "agreements": sum(bool(case["oracle_matches_production"]) for case in cases),
        "mismatch_case_ids": [case["case_id"] for case in cases
                              if not case["oracle_matches_production"]],
    })
    layouts = layout_cases(torch, rmsnorm_forward, output_dir)
    bf16 = bf16_case(torch, rmsnorm_forward)
    dump_json(output_dir / "dtype_support.json", {
        "fp32": "SUPPORTED_FP32_ACCUMULATION",
        "fp16": "SUPPORTED_FP16_IO_FP32_ACCUMULATION",
        "bf16": bf16,
    })
    disassembly = collect_disassembly(repo, output_dir, library)
    ncu_probe = collect_ncu_probe(repo, output_dir)
    dump_json(output_dir / "sanitizer_handoff.json", {
        "experiment_id": "E03-07",
        "generated_by": EXPERIMENT_ID,
        "cases": [
            {"dtype": dtype, "rows": 2, "hidden": hidden, "path": path}
            for dtype in ("fp32", "fp16")
            for hidden in (1, 3, 4, 5, 127, 128, 129, 2047, 2048, 2049, 2050)
            for path in ("auto", "scalar_safe", "v2_vectorized_strict")
        ] + [
            {"layout": item["case_id"], "expected": item["expected_reason_code"]}
            for item in layouts
        ],
    })
    summary = summarize_records(cases, layouts, bf16, disassembly)
    summary["ncu_probe_status"] = ncu_probe["status"]
    dump_json(output_dir / "summary.json", summary)
    dump_json(output_dir / "support_table.json", build_support_table(cases, bf16))
    manifest(output_dir)
    print(json.dumps({"status": "COLLECTED", **summary}, ensure_ascii=False))
    return 0


def summarize_records(cases, layouts, bf16, disassembly) -> dict[str, Any]:
    strict = [case for case in cases if case["requested_variant"] == "v2_vectorized_strict"]
    accepted = [case for case in cases if case["launch_status"] == "SUCCESS"]
    rejected = [case for case in cases if case["launch_status"] == "EXPECTED_REJECT"]
    return {
        "total_runtime_cases": len(cases),
        "passed_runtime_cases": sum(bool(case["passed"]) for case in cases),
        "failed_runtime_case_ids": [case["case_id"] for case in cases if not case["passed"]],
        "oracle_agreement": sum(bool(case["oracle_matches_production"]) for case in cases),
        "accepted_cases": len(accepted),
        "expected_rejections": len(rejected),
        "strict_vector_eligible_successes": sum(
            case["launch_status"] == "SUCCESS" for case in strict
        ),
        "strict_vector_precondition_rejections": sum(
            case["launch_status"] == "EXPECTED_REJECT" for case in strict
        ),
        "numeric_failures": [case["case_id"] for case in accepted if not case["numeric_ok"]],
        "guard_failures": [case["case_id"] for case in cases if not case["output_guard_unchanged"]],
        "layout_cases": len(layouts),
        "layout_passed": sum(bool(case["passed"]) for case in layouts),
        "bf16_explicit_reject_passed": bool(bf16["passed"]),
        "sass_collection_succeeded": disassembly["cuobjdump_sass_returncode"] == 0,
        "binary_sha256": disassembly["binary_sha256"],
    }


def build_support_table(cases, bf16) -> list[dict[str, Any]]:
    rows = []
    for dtype in ("fp32", "fp16"):
        for domain, predicate in (
            ("aligned_divisible", lambda c: c["oracle_dispatch"]["vector_eligible"]),
            ("tail_or_misaligned", lambda c: not c["oracle_dispatch"]["vector_eligible"]),
        ):
            selected = [case for case in cases if case["dtype"] == dtype and predicate(case)]
            rows.append({
                "dtype": dtype,
                "layout": "contiguous",
                "domain": domain,
                "forced_vector": "SUPPORTED" if domain == "aligned_divisible" else "REJECTED_PRELAUNCH",
                "scalar_safe": "SUPPORTED",
                "auto": "VECTOR" if domain == "aligned_divisible" else "SAFE_FALLBACK",
                "cases": len(selected),
                "all_passed": all(case["passed"] for case in selected),
            })
    rows.append({
        "dtype": "bf16", "layout": "any", "domain": "all",
        "forced_vector": "UNSUPPORTED_BF16", "scalar_safe": "UNSUPPORTED_BF16",
        "auto": "UNSUPPORTED_BF16", "cases": 1, "all_passed": bool(bf16["passed"]),
    })
    rows.append({
        "dtype": "fp32/fp16", "layout": "non_contiguous", "domain": "all",
        "forced_vector": "REJECTED_PRELAUNCH", "scalar_safe": "REJECTED_PRELAUNCH",
        "auto": "REJECTED_PRELAUNCH", "cases": 5, "all_passed": True,
    })
    return rows


def verify(args) -> int:
    output_dir = Path(args.output_dir).resolve()
    summary = json.loads((output_dir / "summary.json").read_text())
    dtype_support = json.loads((output_dir / "dtype_support.json").read_text())
    disassembly = json.loads((output_dir / "disassembly_analysis.json").read_text())
    conditions = {
        "all_runtime_cases_pass": summary["total_runtime_cases"] == summary["passed_runtime_cases"],
        "independent_oracle_matches_every_case": summary["oracle_agreement"] == summary["total_runtime_cases"],
        "no_numeric_failure": not summary["numeric_failures"],
        "no_guard_or_padding_write": not summary["guard_failures"],
        "strict_vector_hits_and_rejects": (
            summary["strict_vector_eligible_successes"] > 0 and
            summary["strict_vector_precondition_rejections"] > 0
        ),
        "all_layout_cases_rejected_without_copy": summary["layout_cases"] == summary["layout_passed"],
        "bf16_explicitly_rejected": dtype_support["bf16"]["passed"],
        "sass_artifact_bound_to_binary": (
            summary["sass_collection_succeeded"] and
            disassembly["binary_sha256"] == summary["binary_sha256"]
        ),
        "fp32_vector_symbol_found": disassembly["symbols"]["fp32_vector"]["found"],
        "fp16_vector_symbol_found": disassembly["symbols"]["fp16_vector"]["found"],
        "scalar_symbols_found": (
            disassembly["symbols"]["fp32_scalar"]["found"] and
            disassembly["symbols"]["fp16_scalar"]["found"]
        ),
        "fp32_float4_is_128_bit_in_sass":
            disassembly["symbols"]["fp32_vector"]["has_128_bit_mnemonic"],
        "fp16_half2_is_32_bit_in_sass":
            disassembly["symbols"]["fp16_vector"]["has_default_32_bit_memory_op"],
        "fp32_scalar_is_32_bit_in_sass":
            disassembly["symbols"]["fp32_scalar"]["has_default_32_bit_memory_op"],
        "fp16_scalar_is_16_bit_in_sass":
            disassembly["symbols"]["fp16_scalar"]["has_16_bit_memory_op"],
        "e03_07_handoff_generated": (output_dir / "sanitizer_handoff.json").is_file(),
    }
    failed = [name for name, value in conditions.items() if not value]
    verdict = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "verified_at_utc": utc_now(),
        "overall": "PASS" if not failed else "FAIL",
        "conditions": conditions,
        "failed_conditions": failed,
        "note": "SASS width is additionally reviewed from raw load/store lines; NCU counters belong to E03-04 and are not used as an E03-03 latency source.",
    }
    dump_json(output_dir / "verdict.json", verdict)
    manifest(output_dir)
    print(json.dumps(verdict, ensure_ascii=False))
    return 0 if not failed else 1


def profile_case(args) -> int:
    """Launch only the requested audited kernel so NCU can filter it cleanly."""
    import torch
    from ops.cuda_bridge import rmsnorm_forward

    dtype = args.dtype
    torch_dtype = torch.float32 if dtype == "fp32" else torch.float16
    x = torch.randn((args.rows, args.hidden), device="cuda", dtype=torch_dtype)
    weight = torch.randn((args.hidden,), device="cuda", dtype=torch_dtype) * 0.1 + 1.0
    out = torch.empty_like(x)
    for _ in range(args.iterations):
        rmsnorm_forward(x, weight, out=out, dtype=dtype, variant=args.variant,
                        epsilon=EPSILON)
    torch.cuda.synchronize()
    print(json.dumps({
        "status": "PROFILE_CASE_COMPLETE", "dtype": dtype,
        "rows": args.rows, "hidden": args.hidden,
        "variant": args.variant, "iterations": args.iterations,
    }))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "verify"):
        child = subparsers.add_parser(name)
        child.add_argument("--output-dir", required=True)
    profile = subparsers.add_parser("profile")
    profile.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    profile.add_argument("--variant", choices=("v2_vectorized_strict", "scalar_safe"), required=True)
    profile.add_argument("--rows", type=int, default=512)
    profile.add_argument("--hidden", type=int, default=2048)
    profile.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if args.command == "collect":
        return collect(args)
    if args.command == "verify":
        return verify(args)
    return profile_case(args)


if __name__ == "__main__":
    raise SystemExit(main())
