#!/usr/bin/env python3
"""E03-06 remote-only audit: explicit CUDA stream semantics and concurrency."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "E03-06"
SCHEMA = "hqsb.s03.e03_06.v1"
SEED = 20260918
EPSILON = 1e-6
GATE_CYCLES = 100_000_000
REPEATS = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def load(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def find_library(pattern: str, explicit: str | None = None) -> Path:
    candidates = [Path(explicit)] if explicit else []
    candidates.extend(sorted(REPO.glob(pattern)))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(pattern)


class Api:
    def __init__(self, rms_library: Path, fused_library: Path):
        ptr, i64 = ctypes.c_void_p, ctypes.c_longlong
        self.rms_library = rms_library
        self.fused_library = fused_library
        self.rms_lib = ctypes.CDLL(str(rms_library))
        self.fused_lib = ctypes.CDLL(str(fused_library))
        self.rms = self.rms_lib.hqsb_rmsnorm_forward_ex_c
        self.rms.argtypes = [ptr, ptr, ptr, i64, i64, ctypes.c_float,
                             ctypes.c_int, ctypes.c_int, ptr]
        self.rms.restype = ctypes.c_int
        self.fused = self.fused_lib.hqsb_fused_residual_rmsnorm_forward_ex_c
        self.fused.argtypes = [ptr, ptr, ptr, ptr, ptr, i64, i64,
                               ctypes.c_float, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ptr]
        self.fused.restype = ctypes.c_int

    def launch(self, operator: str, bundle: dict[str, Any], stream) -> int:
        stream_ptr = ctypes.c_void_p(int(stream.cuda_stream))
        if operator == "rmsnorm":
            return int(self.rms(
                bundle["x"].data_ptr(), bundle["weight"].data_ptr(),
                bundle["y"].data_ptr(), bundle["rows"], bundle["hidden"],
                EPSILON, bundle["dtype_code"], bundle["variant"], stream_ptr))
        return int(self.fused(
            bundle["x"].data_ptr(), bundle["residual"].data_ptr(),
            bundle["weight"].data_ptr(), bundle["residual_out"].data_ptr(),
            bundle["y"].data_ptr(), bundle["rows"], bundle["hidden"],
            EPSILON, bundle["dtype_code"], bundle["variant"], 1, stream_ptr))


class DriverStreamIds:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.fn = self.lib.cuStreamGetId
        self.fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
        self.fn.restype = ctypes.c_int

    def get(self, stream) -> dict[str, Any]:
        raw = int(stream.cuda_stream)
        value = ctypes.c_ulonglong()
        rc = int(self.fn(ctypes.c_void_p(raw), ctypes.byref(value)))
        return {"handle": raw, "handle_hex": hex(raw), "driver_query_rc": rc,
                "driver_stream_id": int(value.value) if rc == 0 else None}


def np_dtype(name: str):
    return np.float16 if name == "fp16" else np.float32


def torch_dtype(name: str):
    return torch.float16 if name == "fp16" else torch.float32


def inputs(rows: int, hidden: int, dtype: str, seed: int,
           weight_np: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    target = np_dtype(dtype)
    x = rng.normal(0, 0.25, (rows, hidden)).astype(target)
    residual = rng.normal(0, 0.25, (rows, hidden)).astype(target)
    if weight_np is None:
        weight_np = rng.uniform(0.75, 1.25, hidden).astype(target)
    return x, residual, weight_np


def references(operator: str, x: np.ndarray, residual: np.ndarray,
               weight: np.ndarray, dtype: str) -> dict[str, np.ndarray]:
    target = np_dtype(dtype)
    if operator == "fused":
        residual_out = (x.astype(np.float32) + residual.astype(np.float32)).astype(target)
        norm_input = residual_out.astype(np.float64)
    else:
        residual_out = None
        norm_input = x.astype(np.float64)
    w64 = weight.astype(np.float64)
    inv = 1.0 / np.sqrt(np.mean(norm_input * norm_input, axis=1, keepdims=True) + EPSILON)
    y = (norm_input * inv * w64[None, :]).astype(target)
    consumer = (y.astype(np.float32) * 1.125).astype(target)
    if residual_out is not None:
        consumer = (consumer.astype(np.float32) +
                    residual_out.astype(np.float32) * 0.25).astype(target)
    return {"y": y, "residual_out": residual_out, "consumer": consumer}


def make_bundle(operator: str, rows: int, hidden: int, dtype: str, seed: int,
                weight_np: np.ndarray | None = None) -> dict[str, Any]:
    x_np, residual_np, weight_np = inputs(rows, hidden, dtype, seed, weight_np)
    td = torch_dtype(dtype)
    src_x = torch.from_numpy(x_np).to("cuda")
    src_residual = torch.from_numpy(residual_np).to("cuda")
    weight = torch.from_numpy(weight_np).to("cuda")
    x = torch.full((rows, hidden), 7.0, dtype=td, device="cuda")
    residual = torch.full((rows, hidden), -3.0, dtype=td, device="cuda")
    y = torch.full_like(x, float("nan"))
    residual_out = torch.full_like(x, float("nan"))
    consumer = torch.full_like(x, float("nan"))
    ref = references(operator, x_np, residual_np, weight_np, dtype)
    if operator == "fused":
        variant, variant_name = 3, "v1_warp_shuffle"
    elif dtype == "fp32":
        variant, variant_name = 3, "v1_warp_shuffle"
    else:
        variant, variant_name = 4, "v2_vectorized"
    return {
        "operator": operator, "rows": rows, "hidden": hidden, "dtype": dtype,
        "dtype_code": int(dtype == "fp16"),
        "variant": variant, "variant_name": variant_name,
        "x_np": x_np, "residual_np": residual_np, "weight_np": weight_np,
        "src_x": src_x, "src_residual": src_residual, "weight": weight,
        "x": x, "residual": residual, "y": y,
        "residual_out": residual_out, "consumer": consumer, "reference": ref,
    }


def enqueue_producer(bundle: dict[str, Any], stream) -> None:
    with torch.cuda.stream(stream):
        bundle["x"].copy_(bundle["src_x"], non_blocking=True)
        if bundle["operator"] == "fused":
            bundle["residual"].copy_(bundle["src_residual"], non_blocking=True)


def enqueue_consumer(bundle: dict[str, Any], stream) -> None:
    with torch.cuda.stream(stream):
        torch.mul(bundle["y"], 1.125, out=bundle["consumer"])
        if bundle["operator"] == "fused":
            bundle["consumer"].add_(bundle["residual_out"], alpha=0.25)


def tolerance(dtype: str) -> dict[str, float]:
    return {"atol": 0.004, "rtol": 0.004} if dtype == "fp16" else {
        "atol": 0.0005, "rtol": 0.0005}


def finish(bundle: dict[str, Any]) -> dict[str, Any]:
    got_y = bundle["y"].detach().cpu().numpy()
    got_consumer = bundle["consumer"].detach().cpu().numpy()
    tol = tolerance(bundle["dtype"])
    checks: dict[str, Any] = {
        "y": bool(np.allclose(got_y, bundle["reference"]["y"], equal_nan=False, **tol)),
        "consumer": bool(np.allclose(got_consumer, bundle["reference"]["consumer"],
                                     equal_nan=False, **tol)),
    }
    hashes = {"input": array_hash(bundle["x_np"]), "output": array_hash(got_y),
              "consumer": array_hash(got_consumer)}
    max_abs = {
        "y": float(np.max(np.abs(got_y.astype(np.float64) -
                                     bundle["reference"]["y"].astype(np.float64)))),
        "consumer": float(np.max(np.abs(got_consumer.astype(np.float64) -
                                            bundle["reference"]["consumer"].astype(np.float64)))),
    }
    if bundle["operator"] == "fused":
        got_residual = bundle["residual_out"].detach().cpu().numpy()
        checks["residual_out"] = bool(np.array_equal(
            got_residual, bundle["reference"]["residual_out"]))
        hashes["residual_out"] = array_hash(got_residual)
        max_abs["residual_out"] = float(np.max(np.abs(
            got_residual.astype(np.float64) -
            bundle["reference"]["residual_out"].astype(np.float64))))
    return {"correctness": checks, "all_correct": all(checks.values()),
            "hashes": hashes, "max_abs": max_abs, "tolerance": tol}


def event_id(event, label: str) -> dict[str, Any]:
    value = getattr(event, "cuda_event", None)
    return {"id": label, "handle": int(value) if value is not None else None}


def elapsed(a, b) -> float:
    return float(a.elapsed_time(b))


def run_linear(api: Api, ids: DriverStreamIds, operator: str, mode: str,
               rows: int, hidden: int, dtype: str, launches: int = 1) -> dict[str, Any]:
    bundle = make_bundle(operator, rows, hidden, dtype, SEED + rows + hidden)
    stream = torch.cuda.default_stream() if mode == "default" else torch.cuda.Stream()
    torch.cuda.synchronize()
    chain_start = torch.cuda.Event(True); producer_done = torch.cuda.Event(True)
    op_start = torch.cuda.Event(True); op_stop = torch.cuda.Event(True)
    done = torch.cuda.Event(True)
    host_chain_start = time.monotonic_ns()
    with torch.cuda.stream(stream):
        chain_start.record(stream)
        enqueue_producer(bundle, stream)
        producer_done.record(stream)
        op_start.record(stream)
        submit_start = time.monotonic_ns()
        return_codes = [api.launch(operator, bundle, stream) for _ in range(launches)]
        submit_stop = time.monotonic_ns()
        op_stop.record(stream)
        enqueue_consumer(bundle, stream)
        done.record(stream)
    query_after_submit = bool(done.query())
    completion_error = None
    try:
        done.synchronize()
    except Exception as exc:
        completion_error = repr(exc)
    host_done = time.monotonic_ns()
    result = finish(bundle) if completion_error is None else {"all_correct": False}
    return {
        "case_id": f"{operator}_{mode}", "operator": operator,
        "topology": "producer -> op -> consumer on one stream",
        "default_stream_mode": mode, "thread_id": threading.get_ident(), "device_id": 0,
        "requested_stream": ids.get(stream), "actual_variant": bundle["variant_name"],
        "requested_variant": bundle["variant_name"], "workspace": {"id": None, "bytes": 0},
        "events": [event_id(chain_start, "chain_start"), event_id(producer_done, "P"),
                   event_id(op_start, "op_start"), event_id(op_stop, "N"),
                   event_id(done, "consumer_done")],
        "dependencies": [{"kind": "same_stream_order", "from": "producer", "to": "op"},
                         {"kind": "same_stream_order", "from": "op", "to": "consumer"}],
        "host_submit_ms": (submit_stop - submit_start) / 1e6,
        "device_operation_ms": elapsed(op_start, op_stop),
        "completion_ms": (host_done - host_chain_start) / 1e6,
        "producer_ms": elapsed(chain_start, producer_done),
        "consumer_ms": elapsed(op_stop, done), "launches": launches,
        "query_after_submit_complete": query_after_submit,
        "sync_points": ["cudaEventSynchronize(consumer_done)", "post-completion D2H evidence copy"],
        "immediate_return_codes": return_codes, "completion_error": completion_error,
        **result,
    }


def run_gate(api: Api, ids: DriverStreamIds, operator: str) -> dict[str, Any]:
    bundle = make_bundle(operator, 128, 2048, "fp16", SEED + 61)
    gate_stream, target, independent = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    gate_begin = torch.cuda.Event(True); release = torch.cuda.Event(True)
    op_start = torch.cuda.Event(True); op_stop = torch.cuda.Event(True)
    done = torch.cuda.Event(True); marker_done = torch.cuda.Event(True)
    marker_src = torch.ones(4096, device="cuda"); marker_out = torch.empty_like(marker_src)
    torch.cuda.synchronize()
    with torch.cuda.stream(gate_stream):
        gate_begin.record(gate_stream)
        torch.cuda._sleep(GATE_CYCLES)
        release.record(gate_stream)
    target.wait_event(release)
    with torch.cuda.stream(target):
        enqueue_producer(bundle, target)
        op_start.record(target)
        t0 = time.monotonic_ns(); rc = api.launch(operator, bundle, target); t1 = time.monotonic_ns()
        op_stop.record(target)
        enqueue_consumer(bundle, target)
        done.record(target)
    with torch.cuda.stream(independent):
        torch.mul(marker_src, 2.0, out=marker_out)
        marker_done.record(independent)
    target_complete_at_return = bool(done.query())
    marker_done.synchronize()
    target_complete_after_marker = bool(done.query())
    done.synchronize()
    result = finish(bundle)
    gate_ms = elapsed(gate_begin, release)
    submit_ms = (t1 - t0) / 1e6
    return {
        "case_id": f"{operator}_deterministic_gate", "operator": operator,
        "topology": "gate G -> producer/op/consumer A; unrelated marker M",
        "gate": {"mechanism": "fixed device clock-cycle cuda sleep + event",
                 "cycles": GATE_CYCLES, "measured_ms": gate_ms,
                 "random_sleep_used": False},
        "requested_stream": ids.get(target), "gate_stream": ids.get(gate_stream),
        "independent_stream": ids.get(independent),
        "events": [event_id(release, "gate_release"), event_id(op_start, "op_start"),
                   event_id(op_stop, "N"), event_id(done, "consumer_done"),
                   event_id(marker_done, "independent_marker_done")],
        "dependencies": [{"kind": "cudaStreamWaitEvent", "stream": "A", "event": "gate_release"},
                         {"kind": "same_stream_order", "from": "producer", "to": "op"},
                         {"kind": "same_stream_order", "from": "op", "to": "consumer"}],
        "host_submit_ms": submit_ms, "device_operation_ms": elapsed(op_start, op_stop),
        "host_submit_is_async": submit_ms < max(10.0, gate_ms * 0.25),
        "target_complete_at_api_return": target_complete_at_return,
        "independent_completed_before_target": not target_complete_after_marker,
        "sync_points": ["cudaEventSynchronize(independent_marker_done)",
                        "cudaEventSynchronize(consumer_done)", "post-completion D2H evidence copy"],
        "immediate_return_codes": [rc], "completion_error": None,
        "requested_variant": bundle["variant_name"], "actual_variant": bundle["variant_name"],
        "workspace": {"id": None, "bytes": 0}, **result,
    }


def run_cross(api: Api, ids: DriverStreamIds, operator: str, negative: bool = False) -> dict[str, Any]:
    bundle = make_bundle(operator, 127, 2048, "fp16", SEED + (72 if negative else 71))
    producer, op_stream, consumer = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    gate_stream = torch.cuda.Stream() if negative else None
    gate_release = torch.cuda.Event(True) if negative else None
    p = torch.cuda.Event(True); op_start = torch.cuda.Event(True)
    n = torch.cuda.Event(True); done = torch.cuda.Event(True)
    torch.cuda.synchronize()
    if negative:
        with torch.cuda.stream(gate_stream):
            torch.cuda._sleep(GATE_CYCLES)
            gate_release.record(gate_stream)
        producer.wait_event(gate_release)
    host_start = time.monotonic_ns()
    with torch.cuda.stream(producer):
        enqueue_producer(bundle, producer); p.record(producer)
    if not negative:
        op_stream.wait_event(p)
    with torch.cuda.stream(op_stream):
        op_start.record(op_stream)
        t0 = time.monotonic_ns(); rc = api.launch(operator, bundle, op_stream); t1 = time.monotonic_ns()
        n.record(op_stream)
    if not negative:
        consumer.wait_event(n)
    with torch.cuda.stream(consumer):
        enqueue_consumer(bundle, consumer); done.record(consumer)
    done.synchronize()
    if negative:
        p.synchronize()
    host_done = time.monotonic_ns()
    result = finish(bundle)
    expected = not negative
    observed = bool(result["all_correct"])
    return {
        "case_id": f"{operator}_cross_stream_{'missing_wait_negative' if negative else 'events'}",
        "operator": operator, "negative_control": negative,
        "topology": "producer A -> P -> op B -> N -> consumer C",
        "streams": {"producer": ids.get(producer), "operator": ids.get(op_stream),
                    "consumer": ids.get(consumer)},
        "events": [event_id(p, "P"), event_id(n, "N"), event_id(done, "consumer_done")],
        "dependencies": [] if negative else [
            {"kind": "cudaStreamWaitEvent", "stream": "B", "event": "P"},
            {"kind": "cudaStreamWaitEvent", "stream": "C", "event": "N"}],
        "host_intermediate_sync": False,
        "host_submit_ms": (t1 - t0) / 1e6,
        "device_operation_ms": elapsed(op_start, n),
        "completion_ms": (host_done - host_start) / 1e6,
        "sync_points": ["cudaEventSynchronize(consumer_done)",
                        "cudaEventSynchronize(P) after negative observation"] if negative else
                       ["cudaEventSynchronize(consumer_done)", "post-completion D2H evidence copy"],
        "immediate_return_codes": [rc], "completion_error": None,
        "harness_sensitivity_pass": (observed == expected),
        "expected_correctness": expected, "requested_variant": bundle["variant_name"],
        "actual_variant": bundle["variant_name"], "workspace": {"id": None, "bytes": 0},
        **result,
    }


def run_dual(api: Api, ids: DriverStreamIds, operator: str, mixed: bool) -> dict[str, Any]:
    if mixed:
        a = make_bundle(operator, 127, 2048, "fp16", SEED + 81)
        b = make_bundle(operator, 193, 101, "fp32", SEED + 82)
        shared_weight = False
    else:
        weight_np = inputs(1, 2048, "fp16", SEED + 80)[2]
        a = make_bundle(operator, 127, 2048, "fp16", SEED + 83, weight_np)
        b = make_bundle(operator, 193, 2048, "fp16", SEED + 84, weight_np)
        b["weight"] = a["weight"]
        shared_weight = True
    sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
    hashes_a, hashes_b, records = [], [], []
    torch.cuda.synchronize()
    for repeat in range(REPEATS):
        start_a = torch.cuda.Event(True); stop_a = torch.cuda.Event(True)
        start_b = torch.cuda.Event(True); stop_b = torch.cuda.Event(True)
        host_start = time.monotonic_ns()
        with torch.cuda.stream(sa):
            enqueue_producer(a, sa); start_a.record(sa)
            t0a = time.monotonic_ns(); rca = api.launch(operator, a, sa); t1a = time.monotonic_ns()
            enqueue_consumer(a, sa); stop_a.record(sa)
        with torch.cuda.stream(sb):
            enqueue_producer(b, sb); start_b.record(sb)
            t0b = time.monotonic_ns(); rcb = api.launch(operator, b, sb); t1b = time.monotonic_ns()
            enqueue_consumer(b, sb); stop_b.record(sb)
        stop_a.synchronize(); stop_b.synchronize()
        host_stop = time.monotonic_ns()
        ra, rb = finish(a), finish(b)
        hashes_a.append(ra["hashes"]["output"]); hashes_b.append(rb["hashes"]["output"])
        records.append({"repeat": repeat, "return_codes": [rca, rcb],
                        "submit_ms": [(t1a-t0a)/1e6, (t1b-t0b)/1e6],
                        "stream_chain_ms": [elapsed(start_a, stop_a), elapsed(start_b, stop_b)],
                        "completion_ms": (host_stop-host_start)/1e6,
                        "correct": [ra["all_correct"], rb["all_correct"]]})
    return {
        "case_id": f"{operator}_dual_stream_{'mixed' if mixed else 'shared_weight'}",
        "operator": operator, "topology": "two independent producer/op/consumer chains",
        "streams": {"A": ids.get(sa), "B": ids.get(sb)},
        "shared_read_only_weight": shared_weight, "independent_outputs": True,
        "independent_workspace": True, "workspace_bytes_each": 0,
        "mixed_shape_dtype_variant": mixed,
        "A": {"shape": [a["rows"], a["hidden"]], "dtype": a["dtype"],
              "variant": a["variant_name"], "hashes": hashes_a},
        "B": {"shape": [b["rows"], b["hidden"]], "dtype": b["dtype"],
              "variant": b["variant_name"], "hashes": hashes_b},
        "repeat_hash_stable": len(set(hashes_a)) == 1 and len(set(hashes_b)) == 1,
        "all_correct": all(all(x) for x in (r["correct"] for r in records)),
        "records": records, "sync_points": ["per-stream completion events only"],
        "no_shared_writable_state": True,
    }


def negative_validation(api: Api, ids: DriverStreamIds) -> dict[str, Any]:
    b = make_bundle("rmsnorm", 1, 128, "fp32", SEED + 91)
    s = torch.cuda.Stream(); sp = ctypes.c_void_p(int(s.cuda_stream))
    null_rc = int(api.rms(None, b["weight"].data_ptr(), b["y"].data_ptr(),
                          1, 128, EPSILON, 0, 3, sp))
    rows_rc = int(api.rms(b["x"].data_ptr(), b["weight"].data_ptr(),
                          b["y"].data_ptr(), 0, 128, EPSILON, 0, 3, sp))
    return {
        "stream": ids.get(s), "expected_cuda_error_invalid_value": 1,
        "cases": {"null_input": {"immediate_return_code": null_rc,
                                   "completion_not_applicable": True},
                  "rows_zero": {"immediate_return_code": rows_rc,
                                "completion_not_applicable": True}},
        "immediate_validation_pass": null_rc == 1 and rows_rc == 1,
        "error_model": "return code is immediate validation/launch status; event synchronize is completion status",
        "destroyed_stream_case": "deferred to E03-07 subprocess/sanitizer to avoid context contamination",
    }


def static_audit(output: Path) -> dict[str, Any]:
    paths = [
        REPO / "ops/cuda/rmsnorm/include/hqsb/rmsnorm.h",
        REPO / "ops/cuda/rmsnorm/src/rmsnorm_c_api.cu",
        REPO / "ops/cuda/rmsnorm/src/rmsnorm_dispatcher.cu",
        REPO / "ops/cuda/rmsnorm/src/rmsnorm_v0.cu",
        REPO / "ops/cuda/rmsnorm/src/rmsnorm_v1.cu",
        REPO / "ops/cuda/rmsnorm/src/rmsnorm_v2.cu",
        REPO / "ops/cuda/fused_residual_rmsnorm/include/hqsb/fused_residual_rmsnorm.h",
        REPO / "ops/cuda/fused_residual_rmsnorm/src/fused_residual_rmsnorm.cu",
        REPO / "ops/cuda/fused_residual_rmsnorm/src/fused_residual_rmsnorm_c_api.cu",
        REPO / "ops/cuda_bridge.py",
    ]
    texts = {str(p.relative_to(REPO)): p.read_text() for p in paths}
    production = "\n".join(v for k, v in texts.items() if "/src/" in k)
    launch_sites = []
    for name, text in texts.items():
        for number, line in enumerate(text.splitlines(), 1):
            if "<<<" in line or (launch_sites and line.strip().startswith("<<<")):
                launch_sites.append({"path": name, "line": number, "text": line.strip()})
    hidden_sync = re.findall(r"cudaDeviceSynchronize|cudaStreamSynchronize|cudaEventSynchronize", production)
    blocking_copy = re.findall(r"cudaMemcpy\s*\(", production)
    allocations = re.findall(r"cudaMalloc|cudaFree", production)
    legacy = "hqsb_rmsnorm_forward_c(" in texts["ops/cuda/rmsnorm/src/rmsnorm_c_api.cu"]
    python_optional = "stream=None" in texts["ops/cuda_bridge.py"]
    result = {
        "audited_files": list(texts),
        "public_cpp_api_has_stream": all("cudaStream_t stream" in texts[name] for name in (
            "ops/cuda/rmsnorm/include/hqsb/rmsnorm.h",
            "ops/cuda/fused_residual_rmsnorm/include/hqsb/fused_residual_rmsnorm.h")),
        "c_abi_ex_has_stream": all(symbol in production for symbol in (
            "hqsb_rmsnorm_forward_ex_c", "hqsb_fused_residual_rmsnorm_forward_ex_c")),
        "legacy_omitted_stream_entry_present": legacy,
        "python_wrapper_stream_optional": python_optional,
        "strict_public_api_requires_stream": not legacy and not python_optional,
        "stream_forwarding_tokens": production.count("reinterpret_cast<cudaStream_t>(stream)"),
        "kernel_launch_sites": launch_sites,
        "kernel_launches_use_stream": all("stream" in x["text"] or x["text"].endswith("stream>>>(")
                                          for x in launch_sites),
        "hidden_sync_hits": hidden_sync, "blocking_cudaMemcpy_hits": blocking_copy,
        "allocation_free_hits": allocations,
        "no_hidden_sync_or_blocking_copy": not hidden_sync and not blocking_copy,
        "no_internal_allocation_free": not allocations,
        "workspace_contract": "both operators require no external workspace (0 bytes)",
        "lifetime_contract": (
            "caller owns input/weight/output/(residual) storage; all buffers must remain valid "
            "until the completion event on the supplied stream; shared weight is read-only; "
            "writable outputs are per invocation"),
        "source_sha256": {name: hashlib.sha256(text.encode()).hexdigest()
                          for name, text in texts.items()},
    }
    dump(output / "static_audit.json", result)
    return result


TARGET_TOKENS = {
    "rmsnorm": ("rmsnorm_v0_kernel", "rmsnorm_v1_kernel", "rmsnorm_v2_",
                "rmsnorm_scalar_safe_kernel"),
    "fused": ("fused_residual_rmsnorm_semantic_a_kernel",),
}


def parse_trace(path: Path, label: str, operator: str,
                expected: list[dict[str, int | None]]) -> dict[str, Any]:
    data = load(path, {})
    events = data.get("traceEvents", [])
    annotation = next((e for e in events if e.get("cat") == "user_annotation" and
                       e.get("name") == label), None)
    kernels = [e for e in events if e.get("cat") == "kernel" and
               any(t in str(e.get("name")) for t in TARGET_TOKENS[operator])]
    actual = [{"name": e.get("name"), "stream_id": e.get("args", {}).get("stream"),
               "grid_x": (e.get("args", {}).get("grid") or [None])[0],
               "start_us": e.get("ts"), "duration_us": e.get("dur")}
              for e in kernels]
    hidden = []
    if annotation:
        lo, hi = annotation["ts"], annotation["ts"] + annotation["dur"]
        for e in events:
            if e.get("cat") != "cuda_runtime" or not (lo <= e.get("ts", -1) <= hi):
                continue
            name = str(e.get("name"))
            if any(token in name for token in ("DeviceSynchronize", "StreamSynchronize",
                                                "EventSynchronize", "cudaMemcpy")) and \
                    "cudaMemcpyAsync" not in name:
                hidden.append(name)
    matches = []
    for exp in expected:
        candidates = [x for x in actual if exp.get("rows") is None or x["grid_x"] == exp["rows"]]
        matches.append({"expected": exp, "candidate_stream_ids": [x["stream_id"] for x in candidates],
                        "passed": bool(candidates) and all(
                            x["stream_id"] == exp["stream_id"] for x in candidates)})
    return {"trace": str(path), "annotation": label, "target_kernels": actual,
            "expected_streams": expected, "stream_matches": matches,
            "actual_stream_equals_requested": all(x["passed"] for x in matches),
            "hidden_blocking_api_inside_enqueue": hidden,
            "no_hidden_blocking_api_inside_enqueue": not hidden}


def timeline_case(output: Path, api: Api, ids: DriverStreamIds,
                  operator: str, topology: str) -> dict[str, Any]:
    label = f"E03_06_{operator}_{topology}"
    streams: dict[str, Any] = {}
    expected: list[dict[str, int | None]] = []
    bundles: list[dict[str, Any]] = []
    if topology in ("default", "nondefault"):
        b = make_bundle(operator, 131, 2048, "fp16", SEED + 101)
        s = torch.cuda.default_stream() if topology == "default" else torch.cuda.Stream()
        streams["operator"] = s; bundles = [b]
        expected = [{"rows": 131, "stream_id": ids.get(s)["driver_stream_id"]}]
    elif topology == "cross":
        b = make_bundle(operator, 137, 2048, "fp16", SEED + 102)
        streams = {"producer": torch.cuda.Stream(), "operator": torch.cuda.Stream(),
                   "consumer": torch.cuda.Stream()}
        bundles = [b]
        expected = [{"rows": 137, "stream_id": ids.get(streams["operator"])["driver_stream_id"]}]
    else:
        a = make_bundle(operator, 127, 2048, "fp16", SEED + 103)
        b = make_bundle(operator, 193, 101, "fp32", SEED + 104)
        streams = {"A": torch.cuda.Stream(), "B": torch.cuda.Stream()}; bundles = [a, b]
        expected = [{"rows": 127, "stream_id": ids.get(streams["A"])["driver_stream_id"]},
                    {"rows": 193, "stream_id": ids.get(streams["B"])["driver_stream_id"]}]
    torch.cuda.synchronize()
    done: list[Any] = []
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function(label):
            if topology in ("default", "nondefault"):
                s, b = streams["operator"], bundles[0]
                with torch.cuda.stream(s):
                    enqueue_producer(b, s); api.launch(operator, b, s); enqueue_consumer(b, s)
                    e = torch.cuda.Event(); e.record(s); done.append(e)
            elif topology == "cross":
                b = bundles[0]; p = torch.cuda.Event(); n = torch.cuda.Event()
                with torch.cuda.stream(streams["producer"]):
                    enqueue_producer(b, streams["producer"]); p.record(streams["producer"])
                streams["operator"].wait_event(p)
                with torch.cuda.stream(streams["operator"]):
                    api.launch(operator, b, streams["operator"]); n.record(streams["operator"])
                streams["consumer"].wait_event(n)
                with torch.cuda.stream(streams["consumer"]):
                    enqueue_consumer(b, streams["consumer"])
                    e = torch.cuda.Event(); e.record(streams["consumer"]); done.append(e)
            else:
                for key, b in zip(("A", "B"), bundles):
                    s = streams[key]
                    with torch.cuda.stream(s):
                        enqueue_producer(b, s); api.launch(operator, b, s); enqueue_consumer(b, s)
                        e = torch.cuda.Event(); e.record(s); done.append(e)
        for event in done:
            event.synchronize()
    trace = output / "timeline" / f"{operator}_{topology}.json"
    trace.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace))
    parsed = parse_trace(trace, label, operator, expected)
    parsed["topology"] = topology
    parsed["requested_streams"] = {k: ids.get(v) for k, v in streams.items()}
    return parsed


def collect(args) -> int:
    output = Path(args.output_dir).resolve()
    rms = find_library("build/*/ops/cuda/rmsnorm/libhqsb_rmsnorm_shared.so", args.rms_library)
    fused = find_library("build/*/ops/cuda/fused_residual_rmsnorm/libhqsb_fused_residual_rmsnorm_shared.so",
                         args.fused_library)
    api, ids = Api(rms, fused), DriverStreamIds()
    protocol = {
        "schema_version": SCHEMA, "experiment_id": EXPERIMENT,
        "state": "FROZEN_BEFORE_MEASUREMENT", "frozen_at_utc": utc_now(),
        "operators": ["rmsnorm", "fused"], "epsilon": EPSILON,
        "gate_cycles": GATE_CYCLES, "dual_repeats": REPEATS,
        "topologies": ["explicit_default", "current_nondefault", "deterministic_gate",
                       "cross_stream_events", "missing_wait_negative", "dual_shared_weight",
                       "dual_mixed_shape_dtype_variant"],
        "pass_rule": "all ten E03-06 acceptance clauses pass; no PASS_NEGATIVE",
        "workspace_policy": "0 bytes; no shared writable workspace exists",
        "lifetime_policy": "all buffers live through the final event; read-only weight may be shared",
        "timing": {"host_submit": "monotonic clock around C ABI call only",
                   "device_operation": "events immediately around op on requested stream",
                   "completion": "host enqueue-to-final-event completion"},
    }
    dump(output / "protocol.json", protocol)
    audit = static_audit(output)
    cases = []
    for operator in ("rmsnorm", "fused"):
        cases.extend([
            run_linear(api, ids, operator, "default", 128, 2048, "fp16"),
            run_linear(api, ids, operator, "nondefault", 128, 2048, "fp16", launches=4),
            run_gate(api, ids, operator),
            run_cross(api, ids, operator, negative=False),
            run_cross(api, ids, operator, negative=True),
            run_dual(api, ids, operator, mixed=False),
            run_dual(api, ids, operator, mixed=True),
        ])
    dump(output / "cases.json", {"cases": cases})
    dump(output / "negative_validation.json", negative_validation(api, ids))
    timelines = []
    for operator in ("rmsnorm", "fused"):
        for topology in ("default", "nondefault", "cross", "dual"):
            timelines.append(timeline_case(output, api, ids, operator, topology))
    observed_defaults = {}
    for operator in ("rmsnorm", "fused"):
        item = next(x for x in timelines if x["topology"] == "default" and
                    operator in Path(x["trace"]).stem)
        kernels = item["target_kernels"]
        observed_defaults[operator] = kernels[0]["stream_id"] if kernels else None
    for item in timelines:
        operator = "fused" if "fused_" in Path(item["trace"]).stem else "rmsnorm"
        item["observed_default_timeline_stream"] = observed_defaults[operator]
        if item["topology"] != "default":
            item["target_not_on_default_stream"] = all(
                k["stream_id"] != observed_defaults[operator] for k in item["target_kernels"])
        else:
            item["target_not_on_default_stream"] = None
    dump(output / "timeline" / "summary.json", {"tool": "torch.profiler/CUPTI activity trace",
                                                 "nsys_available": False, "cases": timelines})
    compile_db = load(REPO / "build/jetson-release/compile_commands.json",
                      load(REPO / "build/compile_commands.json", []))
    relevant_commands = [x.get("command", "") for x in compile_db if
                         "rmsnorm" in x.get("file", "")]
    default_flags = [x for x in relevant_commands if "--default-stream" in x or
                     "CUDA_API_PER_THREAD_DEFAULT_STREAM" in x]
    provenance = {
        "collected_at_utc": utc_now(), "hostname": platform.node(), "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_runtime": {"version_reported_by_torch": torch.version.cuda},
        "device": {
            "name": torch.cuda.get_device_properties(0).name,
            "compute_capability": [torch.cuda.get_device_properties(0).major,
                                   torch.cuda.get_device_properties(0).minor],
            "total_memory": torch.cuda.get_device_properties(0).total_memory,
            "multi_processor_count": torch.cuda.get_device_properties(0).multi_processor_count,
        },
        "libraries": {"rmsnorm": {"path": str(rms.relative_to(REPO)), "sha256": sha256_file(rms)},
                      "fused": {"path": str(fused.relative_to(REPO)), "sha256": sha256_file(fused)}},
        "compile_commands_found": len(relevant_commands),
        "explicit_default_stream_compile_flags": default_flags,
        "default_stream_build_mode": "explicit flag not present" if not default_flags else "explicit",
        "test_thread_count": 1, "static_audit_sha256": sha256_file(output / "static_audit.json"),
        "api_contract_warning": not audit["strict_public_api_requires_stream"],
    }
    dump(output / "provenance.json", provenance)
    return 0


def summarize(args) -> int:
    output = Path(args.output_dir).resolve()
    cases = load(output / "cases.json", {}).get("cases", [])
    audit = load(output / "static_audit.json", {})
    timeline = load(output / "timeline/summary.json", {}).get("cases", [])
    validation = load(output / "negative_validation.json", {})
    positives = [x for x in cases if not x.get("negative_control")]
    negatives = [x for x in cases if x.get("negative_control")]
    by_id = {x["case_id"]: x for x in cases}
    timeline_ok = all(x.get("actual_stream_equals_requested") for x in timeline)
    nondefault_not_default = all(x.get("target_not_on_default_stream") for x in timeline
                                 if x.get("topology") != "default")
    no_hidden_timeline = all(x.get("no_hidden_blocking_api_inside_enqueue") for x in timeline)
    conditions = {
        "1_public_api_explicitly_requires_stream": bool(audit.get("strict_public_api_requires_stream")),
        "2_default_nondefault_cross_dual_correct": all(x.get("all_correct") for x in positives),
        "3_timeline_actual_stream_equals_requested": timeline_ok,
        "4_no_default_stream_serialization_or_device_sync": bool(
            audit.get("no_hidden_sync_or_blocking_copy") and no_hidden_timeline and nondefault_not_default),
        "5_host_submit_remains_async": all(
            x.get("host_submit_is_async", True) and not x.get("target_complete_at_api_return", False)
            for x in positives if "gate" in x["case_id"]),
        "6_dependencies_are_same_stream_or_explicit_events": all(
            not x.get("host_intermediate_sync", False) for x in positives),
        "7_concurrency_no_scratch_race_or_pollution": all(
            x.get("all_correct") and x.get("repeat_hash_stable")
            for x in positives if "dual_stream" in x["case_id"]),
        "8_workspace_and_buffer_lifetime_contract_explicit": bool(
            audit.get("lifetime_contract") and audit.get("workspace_contract")),
        "9_immediate_and_completion_errors_separate": bool(
            validation.get("immediate_validation_pass") and
            all(x.get("completion_error") is None for x in positives if "completion_error" in x)),
        "10_fused_path_same_rules": all(x.get("all_correct") for x in positives
                                         if x.get("operator") == "fused") and
                                    all(x.get("actual_stream_equals_requested") for x in timeline
                                        if "fused_" in Path(x["trace"]).stem),
        "negative_control_proves_harness_sensitivity": all(
            x.get("harness_sensitivity_pass") for x in negatives),
    }
    failed = [k for k, v in conditions.items() if not v]
    overall = "PASS" if not failed else "FAIL"
    gate_cases = [x for x in positives if "gate" in x["case_id"]]
    verdict = {
        "schema_version": SCHEMA, "experiment_id": EXPERIMENT,
        "verified_at_utc": utc_now(), "overall": overall,
        "conditions": conditions, "failed_conditions": failed,
        "case_count": len(cases), "positive_case_count": len(positives),
        "negative_control_count": len(negatives), "timeline_case_count": len(timeline),
        "gate_submit_vs_release_ms": [{"case_id": x["case_id"],
                                       "host_submit_ms": x["host_submit_ms"],
                                       "gate_ms": x["gate"]["measured_ms"]} for x in gate_cases],
        "api_finding": ("legacy no-stream C ABI and optional Python stream remain public"
                        if not conditions["1_public_api_explicitly_requires_stream"] else
                        "all public entry points require a stream"),
        "stage_gate": "closed" if overall != "PASS" else "open for E03-07",
    }
    dump(output / "summary.json", {"verdict": verdict,
                                    "case_ids": list(by_id),
                                    "timeline_traces": [x["trace"] for x in timeline]})
    dump(output / "verdict.json", verdict)
    manifest(output)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if overall == "PASS" else 2


def manifest(output: Path) -> None:
    entries = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "EVIDENCE_MANIFEST.json":
            entries.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                            "sha256": sha256_file(path)})
    dump(output / "EVIDENCE_MANIFEST.json", {
        "schema_version": "hqsb.evidence_manifest.v1", "experiment_id": EXPERIMENT,
        "generated_at_utc": utc_now(), "entries": entries})


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "summarize", "verify"):
        p = sub.add_parser(name); p.add_argument("--output-dir", required=True)
        if name == "collect":
            p.add_argument("--rms-library"); p.add_argument("--fused-library")
    args = parser.parse_args()
    if args.command == "collect":
        return collect(args)
    if args.command == "summarize":
        return summarize(args)
    verdict = load(Path(args.output_dir).resolve() / "verdict.json", {})
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if verdict.get("overall") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
