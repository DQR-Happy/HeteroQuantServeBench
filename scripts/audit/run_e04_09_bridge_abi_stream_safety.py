#!/usr/bin/env python3
"""E04-09 Python/C ABI validation and CUDA current-stream safety audit.

The audit deliberately observes the production bridge as it exists.  Missing
ABI negotiation, current-stream propagation, validation or error fields are
recorded as failures instead of being emulated by this harness.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
EXPERIMENT = "E04-09"
SCHEMA = "hqsb.s04.bridge_abi_stream_safety/v1"
EPSILON = 1e-5
SEED = 20260919
GATE_CYCLES = 100_000_000
KERNEL_TOKENS = ("rmsnorm_v0_kernel", "rmsnorm_v1_kernel", "rmsnorm_v2_",
                 "rmsnorm_scalar_safe_kernel")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(argv: Sequence[str], timeout: int = 120) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(list(argv), cwd=REPO, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False, timeout=timeout)
        return {"argv": list(argv), "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
                "duration_s": time.perf_counter() - started, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        return {"argv": list(argv), "exit_code": 124,
                "stdout": exc.stdout or "", "stderr": exc.stderr or "",
                "duration_s": time.perf_counter() - started, "timed_out": True}


def git(*args: str) -> str:
    return run(["git", *args], 30)["stdout"].strip()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def source_identity() -> dict[str, Any]:
    paths = [
        REPO / "ops/cuda_bridge.py",
        REPO / "ops/dispatcher.py",
        REPO / "ops/cuda/rmsnorm/include/hqsb/rmsnorm.h",
        REPO / "ops/cuda/rmsnorm/src/rmsnorm_c_api.cu",
        REPO / "ops/cuda/rmsnorm/src/rmsnorm_dispatcher.cu",
        REPO / "ops/cuda/rmsnorm/src/rmsnorm_v0.cu",
        REPO / "ops/cuda/rmsnorm/src/rmsnorm_v1.cu",
        REPO / "ops/cuda/rmsnorm/src/rmsnorm_v2.cu",
    ]
    return {relative(path): {"bytes": path.stat().st_size,
                             "sha256": sha256_file(path)} for path in paths}


def inherited_evidence() -> dict[str, Any]:
    paths = [
        REPO / "docs/stage_experiments/S03/E03-06/raw/verdict.json",
        REPO / "docs/stage_experiments/S03/E03-07/raw/verdict.json",
        REPO / "docs/stage_experiments/S04/E04-08/raw/verdict.json",
    ]
    rows = []
    for path in paths:
        value = read_json(path, {})
        rows.append({"path": relative(path), "exists": path.is_file(),
                     "sha256": sha256_file(path) if path.is_file() else None,
                     "experiment_id": value.get("experiment_id"),
                     "overall": value.get("overall"),
                     "failed_conditions": value.get("failed_conditions", [])})
    return {"rows": rows, "all_pass": all(row["overall"] == "PASS" for row in rows)}


def static_contract_audit(library: Path) -> dict[str, Any]:
    bridge_path = REPO / "ops/cuda_bridge.py"
    dispatcher_path = REPO / "ops/dispatcher.py"
    header_path = REPO / "ops/cuda/rmsnorm/include/hqsb/rmsnorm.h"
    c_api_path = REPO / "ops/cuda/rmsnorm/src/rmsnorm_c_api.cu"
    dispatcher_cuda_path = REPO / "ops/cuda/rmsnorm/src/rmsnorm_dispatcher.cu"
    bridge = bridge_path.read_text()
    dispatcher = dispatcher_path.read_text()
    header = header_path.read_text()
    c_api = c_api_path.read_text()
    native = "\n".join(path.read_text() for path in (
        c_api_path, dispatcher_cuda_path,
        REPO / "ops/cuda/rmsnorm/src/rmsnorm_v0.cu",
        REPO / "ops/cuda/rmsnorm/src/rmsnorm_v1.cu",
        REPO / "ops/cuda/rmsnorm/src/rmsnorm_v2.cu",
    ))
    symbols = run(["nm", "-D", "--defined-only", str(library)])
    symbol_names = sorted(set(re.findall(r"\b(hqsb_[A-Za-z0-9_]+)$", symbols["stdout"], re.M)))
    required_negotiation = {
        "abi_version_query": any(token in "\n".join(symbol_names).lower()
                                 for token in ("abi_version", "get_version")),
        "build_identity_query": any(token in "\n".join(symbol_names).lower()
                                    for token in ("build_identity", "build_id", "library_hash")),
        "operator_version_query": any("operator_version" in item.lower()
                                      for item in symbol_names),
        "versioned_request_struct": bool(re.search(r"struct\s+.*(?:Request|Args)", header))
                                    and "struct_size" in header and "abi_major" in header,
        "versioned_error_struct": "struct_size" in header and "error" in header.lower(),
    }
    argtype_fragment = "fn_ex.argtypes = fn.argtypes + [ctypes.c_void_p]"
    hidden_calls = re.findall(r"cuda(?:Device|Stream|Event)Synchronize\s*\(|cudaMemcpy\s*\(", native)
    allocation_calls = re.findall(r"cuda(?:Malloc|Free)\s*\(", native)
    launch_lines = [line.strip() for line in native.splitlines() if "<<<" in line]
    return {
        "source_identity": source_identity(),
        "library": {"path": relative(library), "bytes": library.stat().st_size,
                    "sha256": sha256_file(library)},
        "nm": {"exit_code": symbols["exit_code"], "symbols": symbol_names,
               "stderr": symbols["stderr"]},
        "abi_negotiation": required_negotiation,
        "abi_frozen": all(required_negotiation.values()),
        "ctypes": {
            "argtypes_present": "fn.argtypes = [" in bridge and argtype_fragment in bridge,
            "restype_present": "fn.restype = ctypes.c_int" in bridge
                                and "fn_ex.restype = ctypes.c_int" in bridge,
            "pointer_width": ctypes.sizeof(ctypes.c_void_p) * 8,
            "longlong_width": ctypes.sizeof(ctypes.c_longlong) * 8,
            "int_width": ctypes.sizeof(ctypes.c_int) * 8,
        },
        "python_stream_contract": {
            "stream_optional": "stream=None" in bridge,
            "none_maps_to_null_default": "if stream is None:\n            stream_ptr = None" in bridge,
            "current_stream_lookup": "current_stream" in bridge,
            "dispatcher_passes_stream": "stream=" in dispatcher,
            "legacy_no_stream_entry": "hqsb_rmsnorm_forward_c" in c_api,
        },
        "validation_tokens": {
            name: token in bridge for name, token in {
                "tensor_type": "isinstance(x, torch.Tensor)", "cuda_device": "is_cuda",
                "dtype": "DTYPE_MISMATCH", "shape": "SHAPE_INVALID",
                "contiguous": "is_contiguous", "epsilon": "math.isfinite",
                "storage_capacity": "untyped_storage", "storage_offset": "storage_offset",
                "negative_or_zero_stride": "stride()", "alias_or_overlap": "overlap",
                "alignment": "alignment", "workspace": "workspace",
                "device_index_equality_out": "out.device != x.device",
            }.items()
        },
        "native_boundary": {
            "validates_null_shape_dtype_variant_epsilon": all(token in native for token in
                ("input == nullptr", "rows < 1", "decode_dtype", "decode_variant",
                 "epsilon_is_valid")),
            "can_validate_pointer_device": "cudaPointerGetAttributes" in native,
            "can_validate_allocation_range": "cudaMemGetAddressRange" in native,
            "receives_strides": "stride" in c_api.lower(),
            "validates_alias": "overlap" in native.lower() or "alias" in native.lower(),
            "receives_workspace": "workspace" in c_api.lower(),
            "integer_rows_guard": "rows >" in native or "rows <=" in native,
        },
        "error_contract": {
            "stable_project_status_enum": "Status" in header,
            "structured_error_detail": all(token in header.lower() for token in
                                           ("category", "stage", "retryable", "cause")),
            "python_preserves_native_cause": "raise RuntimeError" not in bridge,
            "raw_cuda_int_only": "fn_ex.restype = ctypes.c_int" in bridge,
        },
        "lifetime_contract": {
            "python_record_stream": "record_stream" in bridge,
            "python_retains_input_weight_until_event": "record_event" in bridge
                or "_pending" in bridge,
            "native_hidden_allocation": allocation_calls,
            "caller_owns_buffers_comment": "caller owns" in header,
        },
        "sync_audit": {"hidden_blocking_calls": hidden_calls,
                       "no_hidden_blocking_calls": not hidden_calls,
                       "launch_sites": launch_lines,
                       "all_launch_sites_name_stream": bool(launch_lines)
                           and all("stream" in line for line in launch_lines)},
    }


def initialize(args: argparse.Namespace) -> int:
    out = Path(args.output_dir).resolve()
    library = Path(args.rms_library).resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": f"{SCHEMA}/protocol", "experiment_id": EXPERIMENT,
        "state": "FROZEN_BEFORE_FORMAL_RUNTIME", "frozen_at_utc": utc_now(),
        "operator": "RMSNorm", "epsilon": EPSILON, "seed": SEED,
        "gate_cycles": GATE_CYCLES,
        "pass_rule": "all 12 E04-09 clauses true; no PASS_NEGATIVE",
        "dangerous_case_policy": "arbitrary host pointer, destroyed stream and overlap cases run only in subprocess/sanitizer",
        "stream_policy": "stream=None must resolve torch.cuda.current_stream(x.device), never legacy stream 0",
        "error_policy": "validation, ABI, capability, library, immediate launch and async completion are distinct",
        "timing_policy": "warmup; CUDA events for device completion; perf_counter_ns for host; three fresh processes",
        "workspace_policy": "RMSNorm declares zero workspace; non-applicable is explicit, never inferred",
        "lifetime_policy": "all raw-pointer operands must remain alive through completion on the actual launch stream",
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "inherited_evidence.json", inherited_evidence())
    write_json(out / "abi_and_source_audit.json", static_contract_audit(library))
    return 0


class Api:
    def __init__(self, path: Path):
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        self.fn = self.lib.hqsb_rmsnorm_forward_ex_c
        self.fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_longlong, ctypes.c_longlong, ctypes.c_float,
                            ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.fn.restype = ctypes.c_int

    def call(self, x: int | None, w: int | None, y: int | None, rows: int,
             hidden: int, epsilon: float, dtype: int, variant: int,
             stream: int | None) -> int:
        return int(self.fn(ctypes.c_void_p(x) if x else None,
                           ctypes.c_void_p(w) if w else None,
                           ctypes.c_void_p(y) if y else None,
                           rows, hidden, ctypes.c_float(epsilon), dtype, variant,
                           ctypes.c_void_p(stream) if stream else None))


class StreamIds:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.fn = self.lib.cuStreamGetId
        self.fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
        self.fn.restype = ctypes.c_int

    def get(self, stream: Any) -> dict[str, Any]:
        handle = int(stream.cuda_stream)
        value = ctypes.c_ulonglong()
        rc = int(self.fn(ctypes.c_void_p(handle), ctypes.byref(value)))
        return {"handle": handle, "handle_hex": hex(handle), "driver_query_rc": rc,
                "driver_stream_id": int(value.value) if rc == 0 else None}


def reference(torch: Any, x: Any, weight: Any) -> Any:
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + EPSILON)
            * weight.float()).to(x.dtype)


def check(torch: Any, actual: Any, expected: Any) -> dict[str, Any]:
    diff = (actual.float() - expected.float()).abs()
    passed = bool(torch.allclose(actual.float(), expected.float(), atol=4e-3, rtol=4e-3))
    return {"passed": passed, "max_abs": float(diff.max().item()),
            "finite": bool(torch.isfinite(actual).all().item())}


def python_validation_matrix(torch: Any, bridge: Any) -> dict[str, Any]:
    torch.manual_seed(SEED)
    base_x = torch.randn(2, 128, device="cuda", dtype=torch.float32)
    base_w = torch.randn(128, device="cuda", dtype=torch.float32)
    cases: list[tuple[str, str, Callable[[], Any]]] = [
        ("cpu_tensor", "DEVICE_INVALID", lambda: bridge(base_x.cpu(), base_w.cpu(), dtype="fp32")),
        ("unknown_dtype", "UNSUPPORTED_DTYPE", lambda: bridge(base_x, base_w, dtype="bf16")),
        ("declared_dtype_mismatch", "DTYPE_MISMATCH", lambda: bridge(base_x, base_w, dtype="fp16")),
        ("rank_invalid", "SHAPE_INVALID", lambda: bridge(base_x.reshape(1, 2, 128), base_w, dtype="fp32")),
        ("weight_rank", "WEIGHT_SHAPE_MISMATCH", lambda: bridge(base_x, base_w.reshape(1, 128), dtype="fp32")),
        ("weight_length", "WEIGHT_SHAPE_MISMATCH", lambda: bridge(base_x, base_w[:127], dtype="fp32")),
        ("noncontiguous", "NON_CONTIGUOUS_INPUT", lambda: bridge(base_x[:, ::2], base_w[:64], dtype="fp32")),
        ("epsilon_zero", "EPSILON_INVALID", lambda: bridge(base_x, base_w, dtype="fp32", epsilon=0.0)),
        ("epsilon_nan", "EPSILON_INVALID", lambda: bridge(base_x, base_w, dtype="fp32", epsilon=float("nan"))),
        ("variant_unknown", "UNSUPPORTED_VARIANT", lambda: bridge(base_x, base_w, dtype="fp32", variant=99)),
        ("out_shape", "OUT_TENSOR_MISMATCH", lambda: bridge(base_x, base_w, dtype="fp32", out=base_x[:1].clone())),
        ("out_dtype", "OUT_TENSOR_MISMATCH", lambda: bridge(base_x, base_w, dtype="fp32", out=torch.empty_like(base_x, dtype=torch.float16))),
        ("out_cpu", "OUT_TENSOR_MISMATCH", lambda: bridge(base_x, base_w, dtype="fp32", out=torch.empty_like(base_x, device="cpu"))),
    ]
    rows = []
    for case_id, expected_reason, fn in cases:
        torch.cuda.synchronize()
        try:
            fn()
            torch.cuda.synchronize()
            rows.append({"case_id": case_id, "expected_reason": expected_reason,
                         "rejected": False, "passed": False, "exception": None})
        except Exception as exc:
            reason = getattr(exc, "reason_code", None)
            rows.append({"case_id": case_id, "expected_reason": expected_reason,
                         "rejected": True, "passed": reason == expected_reason,
                         "exception_type": type(exc).__name__, "reason_code": reason,
                         "message": str(exc), "details": getattr(exc, "details", None)})
    return {
        "cases": rows, "all_executed_cases_pass": all(row["passed"] for row in rows),
        "launch_counter_available": False,
        "unexecutable_on_single_gpu": ["wrong CUDA device index"],
        "unimplemented_or_unvalidated_fields": ["storage capacity", "storage offset",
            "negative/zero/overlapping stride", "partial/output-weight alias",
            "pointer alignment policy at Python layer", "workspace metadata"],
    }


def c_abi_matrix(torch: Any, api: Api, stream: Any) -> dict[str, Any]:
    torch.manual_seed(SEED + 1)
    x = torch.randn(2, 128, device="cuda", dtype=torch.float32)
    w = torch.randn(128, device="cuda", dtype=torch.float32)
    y = torch.full_like(x, 23.0)
    sp = int(stream.cuda_stream)
    invalid = [
        ("null_input", None, w.data_ptr(), y.data_ptr(), 2, 128, EPSILON, 0, 3),
        ("null_weight", x.data_ptr(), None, y.data_ptr(), 2, 128, EPSILON, 0, 3),
        ("null_output", x.data_ptr(), w.data_ptr(), None, 2, 128, EPSILON, 0, 3),
        ("rows_zero", x.data_ptr(), w.data_ptr(), y.data_ptr(), 0, 128, EPSILON, 0, 3),
        ("hidden_zero", x.data_ptr(), w.data_ptr(), y.data_ptr(), 2, 0, EPSILON, 0, 3),
        ("hidden_int_overflow", x.data_ptr(), w.data_ptr(), y.data_ptr(), 1, 2**31, EPSILON, 0, 3),
        ("epsilon_zero", x.data_ptr(), w.data_ptr(), y.data_ptr(), 2, 128, 0.0, 0, 3),
        ("epsilon_nan", x.data_ptr(), w.data_ptr(), y.data_ptr(), 2, 128, float("nan"), 0, 3),
        ("dtype_unknown", x.data_ptr(), w.data_ptr(), y.data_ptr(), 2, 128, EPSILON, 99, 3),
        ("variant_unknown", x.data_ptr(), w.data_ptr(), y.data_ptr(), 2, 128, EPSILON, 0, 99),
        ("reference_variant", x.data_ptr(), w.data_ptr(), y.data_ptr(), 2, 128, EPSILON, 0, 1),
    ]
    rows = []
    for values in invalid:
        case_id, *params = values
        before = y.clone()
        rc = api.call(*params, sp)
        stream.synchronize()
        unchanged = bool(torch.equal(before, y))
        rows.append({"case_id": case_id, "expected": "prelaunch reject",
                     "return_code": rc, "rejected": rc != 0,
                     "output_unchanged": unchanged, "passed": rc != 0 and unchanged})

    # These are allocation-safe metadata spoofs: the raw ABI has enough
    # storage, but no Tensor/device/stride metadata with which to reject them.
    spoof_y = torch.full_like(x, 31.0)
    torch.cuda.synchronize()
    rc_dtype = api.call(x.data_ptr(), w.data_ptr(), spoof_y.data_ptr(), 2, 128,
                        EPSILON, 1, 4, sp)
    stream.synchronize()
    rows.append({"case_id": "actual_fp32_pointer_declared_fp16",
                 "expected": "reject actual dtype mismatch", "return_code": rc_dtype,
                 "rejected": rc_dtype != 0, "output_changed": not bool(torch.all(spoof_y == 31.0)),
                 "passed": rc_dtype != 0})
    spoof_y.fill_(31.0)
    torch.cuda.synchronize()
    rc_shape = api.call(x.data_ptr(), w.data_ptr(), spoof_y.data_ptr(), 2, 64,
                        EPSILON, 0, 3, sp)
    stream.synchronize()
    rows.append({"case_id": "actual_h128_declared_h64",
                 "expected": "reject allocation/shape metadata mismatch", "return_code": rc_shape,
                 "rejected": rc_shape != 0, "output_changed": not bool(torch.all(spoof_y == 31.0)),
                 "passed": rc_shape != 0})
    return {
        "cuda_error_invalid_value": 1, "cases": rows,
        "strict_prelaunch_cases_pass": all(row["passed"] for row in rows[:len(invalid)]),
        "metadata_spoof_cases_pass": all(row["passed"] for row in rows[len(invalid):]),
        "dangerous_cases": ["host pointer/wrong device", "destroyed stream",
                            "partial alias", "output-weight alias"],
        "dangerous_cases_location": "sanitizer subprocess plus inherited E03-07 evidence",
    }


def profile_one(torch: Any, output: Path, case_id: str, invoke: Callable[[], Any],
                requested: dict[str, Any], expected_backend: str) -> dict[str, Any]:
    torch.cuda.synchronize()
    label = f"E04_09_{case_id}"
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function(label):
            value = invoke()
        torch.cuda.synchronize()
    trace = output / "timeline" / f"{case_id}.json"
    trace.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace))
    data = read_json(trace, {})
    kernels = []
    for event in data.get("traceEvents", []):
        if event.get("cat") != "kernel":
            continue
        name = str(event.get("name", ""))
        if any(token in name for token in KERNEL_TOKENS) or expected_backend in name.lower():
            kernels.append({"name": name, "stream_id": event.get("args", {}).get("stream"),
                            "duration_us": event.get("dur"), "start_us": event.get("ts")})
    annotations = [event for event in data.get("traceEvents", [])
                   if event.get("cat") == "user_annotation" and event.get("name") == label]
    hidden = []
    if annotations:
        lo = annotations[0].get("ts", -1)
        hi = lo + annotations[0].get("dur", 0)
        for event in data.get("traceEvents", []):
            if event.get("cat") != "cuda_runtime" or not (lo <= event.get("ts", -2) <= hi):
                continue
            name = str(event.get("name", ""))
            if any(token in name for token in ("DeviceSynchronize", "StreamSynchronize",
                                               "EventSynchronize", "cudaMemcpy")) \
                    and "MemcpyAsync" not in name:
                hidden.append(name)
    result = value[1].as_dict() if isinstance(value, tuple) and hasattr(value[1], "as_dict") else None
    return {"case_id": case_id, "trace": relative(trace), "requested_stream": requested,
            "target_kernels": kernels, "actual_stream_ids": sorted(set(
                item["stream_id"] for item in kernels if item["stream_id"] is not None)),
            "actual_stream_equals_requested": bool(kernels) and all(
                item["stream_id"] == requested["driver_stream_id"] for item in kernels),
            "hidden_blocking_api_inside_annotation": hidden,
            "no_hidden_blocking_api": not hidden, "dispatch_decision": result}


def stream_and_timeline(torch: Any, output: Path, bridge: Any) -> dict[str, Any]:
    from ops.capability import BackendCapabilities, detect_capabilities
    from ops.dispatcher import OperatorDispatcher

    ids = StreamIds()
    x = torch.randn(7, 128, device="cuda", dtype=torch.float32)
    w = torch.randn(128, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    current = torch.cuda.Stream()
    explicit = torch.cuda.Stream()
    default = torch.cuda.default_stream()
    cases = []
    with torch.cuda.stream(current):
        cases.append(profile_one(torch, output, "python_omitted_in_current_context",
            lambda: bridge(x, w, dtype="fp32", variant=3, out=y, stream=None),
            ids.get(current), "rmsnorm"))
    with torch.cuda.stream(explicit):
        cases.append(profile_one(torch, output, "python_explicit_nondefault",
            lambda: bridge(x, w, dtype="fp32", variant=3, out=y, stream=explicit),
            ids.get(explicit), "rmsnorm"))
    with torch.cuda.stream(default):
        cases.append(profile_one(torch, output, "python_explicit_default",
            lambda: bridge(x, w, dtype="fp32", variant=3, out=y, stream=default),
            ids.get(default), "rmsnorm"))
    dispatcher = OperatorDispatcher()
    with torch.cuda.stream(current):
        cases.append(profile_one(torch, output, "dispatcher_auto_current",
            lambda: dispatcher.run_rmsnorm(x, w), ids.get(current), "rmsnorm"))

    cap = detect_capabilities()
    fallback_cap = BackendCapabilities(
        cuda_available=cap.cuda_available, device_capability=cap.device_capability,
        triton_available=cap.triton_available, triton_version=cap.triton_version,
        cutlass_available=cap.cutlass_available, cutlass_include_dir=cap.cutlass_include_dir,
        tilelang_available=cap.tilelang_available, tilelang_version=cap.tilelang_version,
        cublas_available=cap.cublas_available, cuda_rmsnorm_available=False,
        cuda_rmsnorm_lib=None, notes=("E04-09 forced CUDA capability unavailable",))
    fallback = OperatorDispatcher(fallback_cap)
    # E04-06 owns cold compile/autotune accounting.  Warm once outside the
    # profiled annotation so this experiment asks whether a steady fallback
    # preserves the original current stream without per-call global sync.
    with torch.cuda.stream(current):
        fallback.run_rmsnorm(x, w)
    torch.cuda.synchronize()
    with torch.cuda.stream(current):
        cases.append(profile_one(torch, output, "dispatcher_capability_fallback_current",
            lambda: fallback.run_rmsnorm(x, w), ids.get(current), "rmsnorm"))
    return {"cases": cases, "fallback_timeline_state": "steady_after_one_warmup",
            "current_omitted_pass": next(row for row in cases if row["case_id"] ==
                "python_omitted_in_current_context")["actual_stream_equals_requested"],
            "dispatcher_auto_current_pass": next(row for row in cases if row["case_id"] ==
                "dispatcher_auto_current")["actual_stream_equals_requested"],
            "explicit_default_nondefault_pass": all(row["actual_stream_equals_requested"]
                for row in cases if row["case_id"].startswith("python_explicit")),
            "fallback_current_pass": next(row for row in cases if row["case_id"] ==
                "dispatcher_capability_fallback_current")["actual_stream_equals_requested"]}


def gate_case(torch: Any, bridge: Any, explicit: bool) -> dict[str, Any]:
    ids = StreamIds()
    rows, hidden = 128, 2048
    target, gate, independent = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float16)
    x = torch.full_like(source, 7.0)
    y = torch.full_like(source, -13.0)
    consumer = torch.full_like(source, 17.0)
    expected = reference(torch, source, weight) * 1.125
    release = torch.cuda.Event(True); done = torch.cuda.Event(True)
    marker = torch.cuda.Event(True)
    torch.cuda.synchronize()
    with torch.cuda.stream(gate):
        torch.cuda._sleep(GATE_CYCLES)
        release.record(gate)
    target.wait_event(release)
    with torch.cuda.stream(target):
        x.copy_(source)
        start = time.perf_counter_ns()
        bridge(x, weight, dtype="fp16", variant=4, out=y,
               stream=target if explicit else None)
        stop = time.perf_counter_ns()
        torch.mul(y, 1.125, out=consumer)
        done.record(target)
    with torch.cuda.stream(independent):
        torch.ones(4096, device="cuda").mul_(2.0)
        marker.record(independent)
    return_complete = bool(done.query())
    marker.synchronize()
    target_after_marker = bool(done.query())
    done.synchronize()
    torch.cuda.synchronize()
    correctness = check(torch, consumer, expected)
    return {"case_id": "explicit_gate" if explicit else "omitted_current_gate",
            "requested_stream": ids.get(target), "gate_stream": ids.get(gate),
            "independent_stream": ids.get(independent), "gate_cycles": GATE_CYCLES,
            "host_submit_us": (stop - start) / 1000.0,
            "target_complete_at_api_return": return_complete,
            "independent_completed_before_target": not target_after_marker,
            "correctness": correctness,
            "passed": correctness["passed"] if explicit else not correctness["passed"],
            "interpretation": "negative control must expose omitted-current stream bug"
                if not explicit else "explicit stream must preserve producer-op-consumer order"}


def dual_stream_case(torch: Any, bridge: Any) -> dict[str, Any]:
    ids = StreamIds()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    bundles = []
    for index, shape in enumerate(((127, 2048), (193, 129))):
        x = torch.randn(*shape, device="cuda", dtype=torch.float16)
        w = torch.randn(shape[1], device="cuda", dtype=torch.float16)
        y = torch.empty_like(x)
        bundles.append((x, w, y, reference(torch, x, w)))
    torch.cuda.synchronize()
    events = []
    for stream, (x, w, y, _) in zip(streams, bundles):
        with torch.cuda.stream(stream):
            bridge(x, w, dtype="fp16", variant=0, out=y, stream=stream)
            event = torch.cuda.Event(); event.record(stream); events.append(event)
    for event in events:
        event.synchronize()
    checks = [check(torch, y, expected) for (_, _, y, expected) in bundles]
    return {"case_id": "two_explicit_streams", "streams": [ids.get(s) for s in streams],
            "independent_workspace": True, "workspace_bytes_each": 0,
            "checks": checks, "passed": all(item["passed"] for item in checks)}


def lifetime_audit(torch: Any, bridge: Any) -> dict[str, Any]:
    stream = torch.cuda.Stream()
    x = torch.randn(64, 2048, device="cuda", dtype=torch.float16)
    w = torch.randn(2048, device="cuda", dtype=torch.float16)
    expected = reference(torch, x, w)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        y = bridge(x, w, dtype="fp16", variant=4, stream=stream)
        done = torch.cuda.Event(); done.record(stream)
    done.synchronize()
    positive = check(torch, y, expected)
    bridge_source = (REPO / "ops/cuda_bridge.py").read_text()
    return {"positive_storage_held_to_event": positive,
            "temporary_copy_created": False, "workspace_bytes": 0,
            "wrapper_records_raw_pointer_use_on_stream": "record_stream" in bridge_source,
            "wrapper_retains_x_weight_until_completion": "_pending" in bridge_source
                or "record_event" in bridge_source,
            "caller_must_retain_inputs": True,
            "public_wrapper_safe_for_temporary_inputs": False,
            "finding": "ctypes use is invisible to the PyTorch caching allocator; wrapper neither record_streams operands nor retains them"}


def time_callable(torch: Any, fn: Callable[[], Any], iterations: int) -> dict[str, Any]:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(True); stop_event = torch.cuda.Event(True)
    host_start = time.perf_counter_ns(); start_event.record()
    for _ in range(iterations):
        fn()
    stop_event.record(); stop_event.synchronize(); host_stop = time.perf_counter_ns()
    return {"iterations": iterations,
            "device_us_per_call": float(start_event.elapsed_time(stop_event)) * 1000.0 / iterations,
            "host_completion_us_per_call": (host_stop - host_start) / 1000.0 / iterations}


def overhead(torch: Any, api: Api, bridge: Any, process_index: int) -> dict[str, Any]:
    from ops.dispatcher import OperatorDispatcher
    torch.manual_seed(SEED + 100 + process_index)
    x = torch.randn(7, 128, device="cuda", dtype=torch.float32)
    w = torch.randn(128, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x); stream = torch.cuda.Stream(); sp = int(stream.cuda_stream)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        raw = time_callable(torch, lambda: api.call(x.data_ptr(), w.data_ptr(), y.data_ptr(),
            7, 128, EPSILON, 0, 3, sp), 200)
        forced = time_callable(torch, lambda: bridge(x, w, dtype="fp32", variant=3,
            out=y, stream=stream), 200)
    dispatcher = OperatorDispatcher()
    with torch.cuda.stream(torch.cuda.default_stream()):
        auto = time_callable(torch, lambda: dispatcher.run_rmsnorm(x, w), 200)
    cpu_x, cpu_w = x.cpu(), w.cpu()
    error_start = time.perf_counter_ns()
    count = 1000
    for _ in range(count):
        try:
            bridge(cpu_x, cpu_w, dtype="fp32")
        except Exception:
            pass
    error_us = (time.perf_counter_ns() - error_start) / 1000.0 / count
    return {"process_index": process_index, "raw_c_abi": raw,
            "python_forced_explicit": forced, "python_auto_dispatcher": auto,
            "validation_reject_host_us_per_call": error_us,
            "bridge_device_overhead_us": forced["device_us_per_call"] - raw["device_us_per_call"],
            "auto_device_overhead_us": auto["device_us_per_call"] - raw["device_us_per_call"]}


def runtime(args: argparse.Namespace) -> int:
    import torch
    from ops.cuda_bridge import rmsnorm_forward

    out = Path(args.output_dir).resolve()
    library = Path(args.rms_library).resolve()
    api = Api(library)
    index = int(args.process_index)
    if not torch.cuda.is_available():
        write_json(out / f"runtime_proc{index}.json", {"status": "BLOCKED", "reason": "CUDA unavailable"})
        return 3
    payload: dict[str, Any] = {"schema": f"{SCHEMA}/runtime", "process_index": index,
                               "generated_at_utc": utc_now(), "status": "PASS"}
    try:
        payload["overhead"] = overhead(torch, api, rmsnorm_forward, index)
        if index == 0:
            payload["python_validation"] = python_validation_matrix(torch, rmsnorm_forward)
            stream = torch.cuda.Stream()
            payload["c_abi_validation"] = c_abi_matrix(torch, api, stream)
            payload["stream_timeline"] = stream_and_timeline(torch, out, rmsnorm_forward)
            payload["gate"] = [gate_case(torch, rmsnorm_forward, True),
                               gate_case(torch, rmsnorm_forward, False)]
            payload["dual_stream"] = dual_stream_case(torch, rmsnorm_forward)
            payload["lifetime"] = lifetime_audit(torch, rmsnorm_forward)
            props = torch.cuda.get_device_properties(0)
            payload["provenance"] = {
                "hostname": platform.node(), "platform": platform.platform(),
                "python": platform.python_version(), "torch": torch.__version__,
                "torch_cuda": torch.version.cuda, "device": props.name,
                "compute_capability": [props.major, props.minor],
                "total_memory": props.total_memory,
                "git_commit": git("rev-parse", "HEAD"),
                "git_dirty": bool(git("status", "--porcelain")),
                "library": {"path": relative(library), "sha256": sha256_file(library)},
            }
    except Exception as exc:
        payload["status"] = "ERROR"
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc()
    write_json(out / f"runtime_proc{index}.json", payload)
    if index == 0:
        for key, filename in (("python_validation", "python_validation.json"),
                              ("c_abi_validation", "c_abi_validation.json"),
                              ("stream_timeline", "stream_timeline.json"),
                              ("gate", "gate_results.json"),
                              ("dual_stream", "dual_stream.json"),
                              ("lifetime", "lifetime.json"),
                              ("provenance", "provenance.json")):
            if key in payload:
                write_json(out / filename, payload[key])
    write_json(out / f"overhead_proc{index}.json", payload.get("overhead", {}))
    return 0 if payload["status"] == "PASS" else 2


CASE_RE = re.compile(r"HQSB_CASE id=(\S+) status=(\S+) launch=(-?\d+) completion=(-?\d+) "
                     r"guards=(\S+) detail=(\S+)")
ERROR_RE = re.compile(r"ERROR SUMMARY:\s*(\d+)\s+error", re.I)


def sanitizer_summary(out: Path) -> dict[str, Any]:
    index_path = out / "sanitizer/index.tsv"
    rows = []
    if not index_path.is_file():
        return {"available": False, "runs": [], "positive_clean": False,
                "negative_boundary_rejected_cleanly": False}
    with index_path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            log = out / row["log"]
            text = log.read_text(errors="replace") if log.is_file() else ""
            cases = [{"id": match.group(1), "status": match.group(2),
                      "launch_status": int(match.group(3)),
                      "completion_status": int(match.group(4)),
                      "guards": match.group(5), "detail": match.group(6)}
                     for match in CASE_RE.finditer(text)]
            summaries = [int(value) for value in ERROR_RE.findall(text)]
            rows.append({**row, "exit_code": int(row["exit_code"]),
                         "elapsed_s": int(row["elapsed_s"]), "cases": cases,
                         "error_summaries": summaries,
                         "issue_count": max(summaries) if summaries else None,
                         "log_sha256": sha256_file(log) if log.is_file() else None})
    by_id = {row["run_id"]: row for row in rows}
    clean = by_id.get("rms_matrix_clean", {})
    negative = by_id.get("api_negative_exposes_boundary", {})
    negative_cases = negative.get("cases", [])
    return {"available": bool(rows), "runs": rows,
            "positive_clean": clean.get("exit_code") == 0 and clean.get("issue_count") == 0
                and bool(clean.get("cases")),
            "negative_boundary_rejected_cleanly": negative.get("exit_code") == 0
                and negative.get("issue_count") == 0
                and bool(negative_cases) and all(case["status"] == "PASS" for case in negative_cases),
            "negative_failures": [case for case in negative_cases if case["status"] != "PASS"]}


def analyze(args: argparse.Namespace) -> int:
    out = Path(args.output_dir).resolve()
    audit = read_json(out / "abi_and_source_audit.json", {})
    py = read_json(out / "python_validation.json", {})
    c_abi = read_json(out / "c_abi_validation.json", {})
    stream = read_json(out / "stream_timeline.json", {})
    gates = read_json(out / "gate_results.json", [])
    dual = read_json(out / "dual_stream.json", {})
    lifetime = read_json(out / "lifetime.json", {})
    inherited = read_json(out / "inherited_evidence.json", {})
    sanitizer = sanitizer_summary(out)
    write_json(out / "sanitizer.json", sanitizer)
    overhead_rows = [read_json(out / f"overhead_proc{i}.json", {}) for i in range(3)]
    complete_overhead = [row for row in overhead_rows if row.get("raw_c_abi")]
    overhead_summary = {"process_count": len(complete_overhead), "rows": complete_overhead}
    for key in ("bridge_device_overhead_us", "auto_device_overhead_us",
                "validation_reject_host_us_per_call"):
        overhead_summary[f"{key}_median"] = statistics.median(
            row[key] for row in complete_overhead) if complete_overhead else None
    write_json(out / "overhead_summary.json", overhead_summary)

    explicit_gate = next((row for row in gates if row.get("case_id") == "explicit_gate"), {})
    omitted_gate = next((row for row in gates if row.get("case_id") == "omitted_current_gate"), {})
    abi = audit.get("abi_negotiation", {})
    validation_tokens = audit.get("validation_tokens", {})
    native = audit.get("native_boundary", {})
    errors = audit.get("error_contract", {})
    conditions = {
        "1_abi_version_size_symbol_build_identity_frozen": bool(audit.get("abi_frozen")),
        "2_python_and_c_abi_independent_validation_effective": bool(
            py.get("all_executed_cases_pass") and c_abi.get("strict_prelaunch_cases_pass")
            and c_abi.get("metadata_spoof_cases_pass")),
        "3_device_dtype_shape_layout_alignment_alias_workspace_stable": bool(
            py.get("all_executed_cases_pass") and all(validation_tokens.get(name) for name in
                ("cuda_device", "dtype", "shape", "contiguous", "storage_capacity",
                 "storage_offset", "negative_or_zero_stride", "alias_or_overlap",
                 "alignment", "workspace", "device_index_equality_out"))
            and native.get("can_validate_pointer_device")
            and native.get("can_validate_allocation_range") and native.get("receives_strides")
            and native.get("validates_alias") and native.get("receives_workspace")),
        "4_ctypes_calling_convention_and_integer_width_correct": bool(
            audit.get("ctypes", {}).get("argtypes_present")
            and audit.get("ctypes", {}).get("restype_present")
            and audit.get("ctypes", {}).get("pointer_width") == 64
            and audit.get("ctypes", {}).get("longlong_width") == 64),
        "5_current_nondefault_stream_end_to_end": bool(
            stream.get("current_omitted_pass") and stream.get("dispatcher_auto_current_pass")),
        "6_default_gate_dual_fallback_no_race_or_global_sync": bool(
            stream.get("explicit_default_nondefault_pass") and stream.get("fallback_current_pass")
            and explicit_gate.get("passed") and omitted_gate.get("passed")
            and dual.get("passed") and audit.get("sync_audit", {}).get("no_hidden_blocking_calls")
            and all(case.get("no_hidden_blocking_api") for case in stream.get("cases", []))),
        "7_async_lifetime_is_safe": bool(lifetime.get("positive_storage_held_to_event", {}).get("passed")
            and lifetime.get("wrapper_records_raw_pointer_use_on_stream")
            and lifetime.get("wrapper_retains_x_weight_until_completion")),
        "8_immediate_async_library_abi_errors_distinguishable": bool(
            errors.get("stable_project_status_enum") and errors.get("structured_error_detail")
            and errors.get("python_preserves_native_cause") and all(abi.values())),
        "9_sanitizer_target_hit_without_unexplained_error": bool(
            sanitizer.get("positive_clean") and sanitizer.get("negative_boundary_rejected_cleanly")),
        "10_bridge_overhead_has_raw_baseline": len(complete_overhead) == 3,
        "11_dangerous_cases_isolated": bool(sanitizer.get("available")
            and "sanitizer" in c_abi.get("dangerous_cases_location", "")),
        "12_actual_backend_and_reason_written": all(
            case.get("dispatch_decision") is not None for case in stream.get("cases", [])
            if case.get("case_id", "").startswith("dispatcher_")),
    }
    failed = [name for name, value in conditions.items() if not value]
    overall = "PASS" if not failed else "FAIL"
    verdict = {"schema": SCHEMA, "experiment_id": EXPERIMENT,
               "generated_at_utc": utc_now(), "overall": overall,
               "conditions": conditions, "failed_conditions": failed,
               "expected_effect": "bridge rejects unsafe boundaries and follows current stream",
               "single_item_standard": "two-layer validation; current stream; multi-stream no race/global sync",
               "expected_effect_met": overall == "PASS", "single_item_standard_met": overall == "PASS",
               "inherited_stage_gates_all_pass": inherited.get("all_pass", False),
               "critical_findings": [
                   "ABI/version/struct-size/build identity negotiation is absent" if not conditions[
                       "1_abi_version_size_symbol_build_identity_frozen"] else None,
                   "raw C ABI accepts metadata-spoofed dtype/shape because it receives bare pointers only"
                       if not c_abi.get("metadata_spoof_cases_pass") else None,
                   "Python stream=None and production dispatcher launch on legacy default stream"
                       if not conditions["5_current_nondefault_stream_end_to_end"] else None,
                   "dangerous alias/host-pointer/destroyed-stream boundaries are not cleanly rejected"
                       if not conditions["9_sanitizer_target_hit_without_unexplained_error"] else None,
                   "ctypes raw-pointer lifetime is not recorded with the PyTorch allocator"
                       if not conditions["7_async_lifetime_is_safe"] else None,
               ]}
    verdict["critical_findings"] = [item for item in verdict["critical_findings"] if item]
    write_json(out / "verdict.json", verdict)

    manifest = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "EVIDENCE_MANIFEST.json":
            manifest.append({"path": str(path.relative_to(out)), "bytes": path.stat().st_size,
                             "sha256": sha256_file(path)})
    write_json(out / "EVIDENCE_MANIFEST.json",
               {"schema": "hqsb.evidence-manifest/v1", "files": manifest})
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if overall == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("initialize", "runtime"):
        item = sub.add_parser(name)
        item.add_argument("--output-dir", required=True)
        item.add_argument("--rms-library", required=True)
        if name == "runtime":
            item.add_argument("--process-index", type=int, required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    return {"initialize": initialize, "runtime": runtime, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
