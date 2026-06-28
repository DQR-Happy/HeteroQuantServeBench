#!/usr/bin/env python3
"""E03-05 remote-only audit: residual add + RMSNorm fusion."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import inspect
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "E03-05"
SCHEMA = "hqsb.s03.e03_05.v1"
SEED = 20260918
EPSILON = 1e-6
PROCESSES = 3
GROUPS = 4
WARMUP = 12
GUARD_BAND = 0.05
VARIANT = 3
SEMANTIC = 1
SHAPES = (
    {"shape_id": "decode_h2048", "rows": 1, "hidden": 2048,
     "role": "real_decode_hidden", "source": "S02 Qwen3 hidden-width decode"},
    {"shape_id": "prefill128_h2048", "rows": 128, "hidden": 2048,
     "role": "real_prefill_hidden", "source": "S02 Qwen3 prefill ISL=128"},
    {"shape_id": "prefill1024_h2048", "rows": 1024, "hidden": 2048,
     "role": "real_prefill_hidden", "source": "S02/E03-04 real prefill regime"},
    {"shape_id": "decode_h128", "rows": 1, "hidden": 128,
     "role": "small_decode_boundary", "source": "S02 real Q/K width; synthetic for fusion"},
    {"shape_id": "odd_h101", "rows": 17, "hidden": 101,
     "role": "odd_tail_boundary", "source": "E03-02/E03-03 synthetic boundary"},
    {"shape_id": "max_h8192", "rows": 2, "hidden": 8192,
     "role": "declared_max_hidden", "source": "E03-01 required boundary"},
)
DTYPES = ("fp32", "fp16")
NCU_SECTIONS = ("LaunchStats", "Occupancy", "MemoryWorkloadAnalysis")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def load(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(command: list[str], timeout: int = 300) -> dict[str, Any]:
    try:
        p = subprocess.run(command, cwd=REPO, text=True, capture_output=True,
                           timeout=timeout, check=False)
        return {"command": command, "returncode": p.returncode,
                "stdout": p.stdout, "stderr": p.stderr}
    except Exception as exc:
        return {"command": command, "returncode": 127, "stdout": "",
                "stderr": repr(exc)}


def find_library(explicit: str | None) -> Path:
    candidates = [Path(explicit)] if explicit else []
    env = os.environ.get("HQSB_FUSED_RMSNORM_LIB")
    if env:
        candidates.append(Path(env))
    candidates.extend(sorted(REPO.glob(
        "build/*/ops/cuda/fused_residual_rmsnorm/"
        "libhqsb_fused_residual_rmsnorm_shared.so")))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("libhqsb_fused_residual_rmsnorm_shared.so not found")


class Api:
    def __init__(self, library: Path):
        self.library = library
        self.lib = ctypes.CDLL(str(library))
        ptr = ctypes.c_void_p
        i64 = ctypes.c_longlong
        self.fused = self.lib.hqsb_fused_residual_rmsnorm_forward_ex_c
        self.fused.argtypes = [ptr, ptr, ptr, ptr, ptr, i64, i64,
                               ctypes.c_float, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ptr]
        self.fused.restype = ctypes.c_int
        self.separate = self.lib.hqsb_separate_residual_rmsnorm_forward_ex_c
        self.separate.argtypes = [ptr, ptr, ptr, ptr, ptr, i64, i64,
                                  ctypes.c_float, ctypes.c_int, ctypes.c_int, ptr]
        self.separate.restype = ctypes.c_int
        self.add = self.lib.hqsb_residual_add_forward_ex_c
        self.add.argtypes = [ptr, ptr, ptr, i64, i64, ctypes.c_int, ptr]
        self.add.restype = ctypes.c_int
        self.norm = self.lib.hqsb_fused_baseline_rmsnorm_forward_ex_c
        self.norm.argtypes = [ptr, ptr, ptr, i64, i64, ctypes.c_float,
                              ctypes.c_int, ptr]
        self.norm.restype = ctypes.c_int
        self.shared = self.lib.hqsb_fused_residual_dynamic_shared_bytes_c
        self.shared.argtypes = [i64, ctypes.c_int, ctypes.c_int]
        self.shared.restype = i64
        self.occupancy = self.lib.hqsb_fused_residual_occupancy_c
        self.occupancy.argtypes = [i64, ctypes.c_int, ctypes.c_int]
        self.occupancy.restype = ctypes.c_int

    def bind(self, mode: str, x, residual, weight, residual_out, y,
             rows: int, hidden: int, dtype_code: int, stream):
        stream_ptr = int(stream.cuda_stream)
        if mode == "fused":
            return lambda: int(self.fused(
                x.data_ptr(), residual.data_ptr(), weight.data_ptr(),
                residual_out.data_ptr(), y.data_ptr(), rows, hidden, EPSILON,
                dtype_code, VARIANT, SEMANTIC, stream_ptr))
        if mode == "separate":
            return lambda: int(self.separate(
                x.data_ptr(), residual.data_ptr(), weight.data_ptr(),
                residual_out.data_ptr(), y.data_ptr(), rows, hidden, EPSILON,
                dtype_code, SEMANTIC, stream_ptr))
        if mode == "add":
            return lambda: int(self.add(x.data_ptr(), residual.data_ptr(),
                                        residual_out.data_ptr(), rows, hidden,
                                        dtype_code, stream_ptr))
        if mode == "rmsnorm":
            return lambda: int(self.norm(residual_out.data_ptr(),
                                         weight.data_ptr(), y.data_ptr(), rows,
                                         hidden, EPSILON, dtype_code, stream_ptr))
        raise ValueError(mode)


def np_dtype(name: str):
    return np.float16 if name == "fp16" else np.float32


def torch_dtype(name: str):
    return torch.float16 if name == "fp16" else torch.float32


def make_inputs(rows: int, hidden: int, dtype: str, seed: int,
                mode: str = "random"):
    rng = np.random.default_rng(seed)
    if mode == "zero":
        x = np.zeros((rows, hidden), dtype=np.float32)
        residual = np.zeros_like(x)
    elif mode == "tiny":
        x = rng.uniform(-1e-4, 1e-4, (rows, hidden)).astype(np.float32)
        residual = rng.uniform(-1e-4, 1e-4, (rows, hidden)).astype(np.float32)
    elif mode == "large":
        x = rng.uniform(-1e3, 1e3, (rows, hidden)).astype(np.float32)
        residual = rng.uniform(-1e3, 1e3, (rows, hidden)).astype(np.float32)
    else:
        x = rng.normal(0, 0.25, (rows, hidden)).astype(np.float32)
        residual = rng.normal(0, 0.25, (rows, hidden)).astype(np.float32)
    weight = rng.uniform(0.75, 1.25, hidden).astype(np.float32)
    target = np_dtype(dtype)
    return x.astype(target), residual.astype(target), weight.astype(target)


def references(x: np.ndarray, residual: np.ndarray, weight: np.ndarray,
               dtype: str) -> dict[str, np.ndarray]:
    target = np_dtype(dtype)
    s_acc = x.astype(np.float32) + residual.astype(np.float32)
    residual_a = s_acc.astype(target)
    a64 = residual_a.astype(np.float64)
    w64 = weight.astype(np.float64)
    inv_a = 1.0 / np.sqrt(np.mean(a64 * a64, axis=1, keepdims=True) + EPSILON)
    y_a = (a64 * inv_a * w64[None, :]).astype(target)
    b64 = s_acc.astype(np.float64)
    inv_b = 1.0 / np.sqrt(np.mean(b64 * b64, axis=1, keepdims=True) + EPSILON)
    y_b = (b64 * inv_b * w64[None, :]).astype(target)
    return {"residual_a": residual_a, "y_a": y_a, "y_b": y_b,
            "s_acc_fp32": s_acc}


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    actual64, reference64 = actual.astype(np.float64), reference.astype(np.float64)
    same_class = np.array_equal(np.isnan(actual64), np.isnan(reference64)) and \
        np.array_equal(np.isposinf(actual64), np.isposinf(reference64)) and \
        np.array_equal(np.isneginf(actual64), np.isneginf(reference64))
    finite = np.isfinite(actual64) & np.isfinite(reference64)
    if not finite.any():
        return {"finite_count": 0, "ieee_class_match": same_class,
                "max_abs": None, "mean_abs": None, "rmse": None,
                "l2_relative": None, "cosine": None}
    a, r = actual64[finite], reference64[finite]
    diff = a - r
    rnorm = float(np.linalg.norm(r))
    anorm = float(np.linalg.norm(a))
    return {
        "finite_count": int(finite.sum()), "ieee_class_match": same_class,
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "l2_relative": float(np.linalg.norm(diff) / max(rnorm, 1e-6)),
        "cosine": float(np.dot(a, r) / (anorm * rnorm))
        if anorm > 0 and rnorm > 0 else None,
    }


def tolerance(dtype: str, hidden: int) -> dict[str, float]:
    if dtype == "fp16":
        return {"atol": 0.00390625, "l2rel": 0.001}
    return {"atol": 0.0005 if hidden <= 2048 else 0.002,
            "l2rel": 0.0005}


def passes(metric: dict[str, Any], tol: dict[str, float]) -> bool:
    return bool(metric.get("ieee_class_match") and
                metric.get("max_abs") is not None and
                metric["max_abs"] <= tol["atol"] and
                metric["l2_relative"] <= tol["l2rel"])


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values), q))


def time_launch(launch, stream, launches: int) -> dict[str, Any]:
    for _ in range(WARMUP):
        if launch() != 0:
            return {"error": "warmup launch failed"}
    stream.synchronize()
    device, submit, completion = [], [], []
    allocated_before = int(torch.cuda.memory_allocated())
    reserved_before = int(torch.cuda.memory_reserved())
    for _ in range(GROUPS):
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record(stream)
        for _ in range(launches):
            rc = launch()
            if rc:
                return {"error": f"device timing launch returned {rc}"}
        stop.record(stream)
        stop.synchronize()
        device.append(float(start.elapsed_time(stop) / launches))
        t0 = time.monotonic_ns()
        for _ in range(launches):
            rc = launch()
            if rc:
                return {"error": f"host timing launch returned {rc}"}
        t1 = time.monotonic_ns()
        stream.synchronize()
        t2 = time.monotonic_ns()
        submit.append((t1 - t0) / 1e6 / launches)
        completion.append((t2 - t0) / 1e6 / launches)
    return {
        "device_ms_samples": device,
        "host_submit_ms_samples": submit,
        "host_submit_completion_ms_samples": completion,
        "allocator": {
            "memory_allocated_before": allocated_before,
            "memory_allocated_after": int(torch.cuda.memory_allocated()),
            "memory_reserved_before": reserved_before,
            "memory_reserved_after": int(torch.cuda.memory_reserved()),
        },
    }


def make_plan(output: Path) -> int:
    upstream = {}
    for exp in ("E03-01", "E03-02", "E03-03", "E03-04"):
        upstream[exp] = load(REPO / f"docs/stage_experiments/S03/{exp}/raw/verdict.json", {})
    plan = {
        "schema_version": SCHEMA, "experiment_id": EXPERIMENT,
        "state": "FROZEN_BEFORE_MEASUREMENT", "frozen_at_utc": utc_now(),
        "semantic": {
            "id": "A", "version": 1,
            "residual_out": "cast_dtype(float32(x)+float32(residual))",
            "normalization_input": "the rounded residual_out",
            "outputs": ["residual_out", "y"],
        },
        "alias_policy": "all output/input overlap and output/output overlap rejected",
        "workspace_bytes": 0, "hidden_max": 8192,
        "dtypes": list(DTYPES), "bf16": "unsupported/rejected",
        "shapes": list(SHAPES), "processes": PROCESSES, "groups": GROUPS,
        "warmup": WARMUP, "guard_band": GUARD_BAND, "epsilon": EPSILON,
        "ordinary_modes": ["add", "rmsnorm", "separate", "fused"],
        "timing": "same explicit non-default stream; separate has no intermediate sync",
        "logical_bytes": {"separate_elements_per_row": "6*H",
                          "fused_elements_per_row": "5*H",
                          "saved": "one residual_out global read (H elements)"},
        "upstream_verdicts": upstream,
    }
    dump(output / "protocol.json", plan)
    print(json.dumps({"status": "FROZEN", "cases_per_process": len(SHAPES)*2}))
    return 0


def model_dataflow(output: Path) -> dict[str, Any]:
    model_path = Path("/home/jetson/models/hqsb/Qwen3-1.7B")
    config_path = model_path / "config.json"
    config = load(config_path, {})
    evidence: dict[str, Any] = {
        "model_path": str(model_path), "config_exists": config_path.is_file(),
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "hidden_size": config.get("hidden_size"),
        "rms_norm_eps": config.get("rms_norm_eps"),
        "torch_dtype": config.get("torch_dtype"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "conclusion": "residual is updated by low-precision tensor addition and is consumed by the next residual path; fused API must return both residual_out and normalized y",
        "model_claim_scope": "source/config evidence only; model-level replacement is deferred to S04.5",
    }
    try:
        from transformers.models.qwen3 import modeling_qwen3
        layer_src = inspect.getsource(modeling_qwen3.Qwen3DecoderLayer.forward)
        norm_src = inspect.getsource(modeling_qwen3.Qwen3RMSNorm.forward)
        evidence.update({
            "transformers_version": __import__("transformers").__version__,
            "decoder_forward_sha256": hashlib.sha256(layer_src.encode()).hexdigest(),
            "rmsnorm_forward_sha256": hashlib.sha256(norm_src.encode()).hexdigest(),
            "decoder_residual_lines": [line.strip() for line in layer_src.splitlines()
                                       if "residual" in line or "layernorm" in line],
            "rmsnorm_lines": [line.strip() for line in norm_src.splitlines()
                              if "variance" in line or "float" in line or "weight" in line],
        })
    except Exception as exc:
        evidence["source_inspection_error"] = repr(exc)
    dump(output / "model_dataflow.json", evidence)
    return evidence


def collect(args) -> int:
    output = Path(args.output_dir).resolve()
    if load(output / "protocol.json", {}).get("state") != "FROZEN_BEFORE_MEASUREMENT":
        raise RuntimeError("run plan first")
    library = find_library(args.library)
    api = Api(library)
    stream = torch.cuda.Stream()
    records = []
    for shape_index, shape in enumerate(SHAPES):
        for dtype in DTYPES:
            rows, hidden = shape["rows"], shape["hidden"]
            seed = SEED + args.process_index * 1000 + shape_index * 10 + (dtype == "fp16")
            nx, nr, nw = make_inputs(rows, hidden, dtype, seed)
            x = torch.from_numpy(nx).to("cuda")
            residual = torch.from_numpy(nr).to("cuda")
            weight = torch.from_numpy(nw).to("cuda")
            torch.cuda.synchronize()
            refs = references(nx, nr, nw, dtype)
            outputs = {}
            correctness = {}
            for mode in ("separate", "fused"):
                ro, y = torch.empty_like(x), torch.empty_like(x)
                launch = api.bind(mode, x, residual, weight, ro, y, rows,
                                  hidden, int(dtype == "fp16"), stream)
                rc = launch(); stream.synchronize()
                got_ro, got_y = ro.cpu().numpy(), y.cpu().numpy()
                ro_m = metrics(got_ro, refs["residual_a"])
                y_m = metrics(got_y, refs["y_a"])
                tol = tolerance(dtype, hidden)
                correctness[mode] = {
                    "return_code": rc, "residual_vs_semantic_a": ro_m,
                    "y_vs_semantic_a": y_m,
                    "residual_exact": bool(np.array_equal(got_ro, refs["residual_a"])),
                    "passed": rc == 0 and np.array_equal(got_ro, refs["residual_a"]) and
                              passes(y_m, tol),
                }
                outputs[mode] = (got_ro, got_y)
            propagation = {
                "fused_y_vs_separate_y": metrics(outputs["fused"][1], outputs["separate"][1]),
                "separate_y_vs_semantic_b": metrics(outputs["separate"][1], refs["y_b"]),
                "fused_y_vs_semantic_b": metrics(outputs["fused"][1], refs["y_b"]),
                "semantic_a_vs_b": metrics(refs["y_a"], refs["y_b"]),
            }
            launches = 30 if rows * hidden >= 1024 * 2048 else 120 if rows > 1 else 240
            timings = {}
            allocation = {}
            ro, y = torch.empty_like(x), torch.empty_like(x)
            # Seed the standalone RMSNorm input once; timed launches allocate nothing.
            ro.copy_(torch.from_numpy(refs["residual_a"]).to("cuda"))
            torch.cuda.synchronize()
            for mode in ("add", "rmsnorm", "separate", "fused"):
                launch = api.bind(mode, x, residual, weight, ro, y, rows,
                                  hidden, int(dtype == "fp16"), stream)
                timed = time_launch(launch, stream, launches)
                allocation[mode] = timed.pop("allocator", {})
                timings[mode] = timed
            record = {
                "schema_version": SCHEMA, "experiment_id": EXPERIMENT,
                "process_index": args.process_index,
                "case_id": f"{shape['shape_id']}|{dtype}", **shape,
                "dtype": dtype, "epsilon": EPSILON, "semantic": "A/v1",
                "input_sha256": hashlib.sha256(nx.tobytes()+nr.tobytes()+nw.tobytes()).hexdigest(),
                "correctness": correctness, "error_propagation": propagation,
                "tolerance": tolerance(dtype, hidden), "timings": timings,
                "allocation": allocation, "launches_per_group": launches,
                "fused_dynamic_shared_bytes": int(api.shared(hidden, int(dtype == "fp16"), VARIANT)),
                "fused_max_active_blocks_per_sm": int(api.occupancy(hidden, int(dtype == "fp16"), VARIANT)),
                "library_sha256": sha256_file(library),
            }
            record["status"] = "MEASURED" if all(v["passed"] for v in correctness.values()) and \
                all("error" not in v for v in timings.values()) else "FAIL"
            records.append(record)
            print(record["case_id"], record["status"], flush=True)
    path = output / f"cases_proc{args.process_index}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    if args.process_index == 0:
        model_dataflow(output)
        git = run(["git", "status", "--short"])
        dump(output / "provenance.json", {
            "schema_version": SCHEMA, "timestamp_utc": utc_now(),
            "git_commit": run(["git", "rev-parse", "HEAD"])["stdout"].strip(),
            "git_dirty": bool(git["stdout"].strip()),
            "git_status_short": git["stdout"].splitlines(),
            "platform": platform.platform(), "python": sys.version,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "library": str(library.relative_to(REPO)),
            "library_sha256": sha256_file(library),
            "source_sha256": {str(p.relative_to(REPO)): sha256_file(p) for p in (
                REPO / "ops/cuda/fused_residual_rmsnorm/src/fused_residual_rmsnorm.cu",
                REPO / "ops/cuda/fused_residual_rmsnorm/src/fused_residual_rmsnorm_c_api.cu",
                REPO / "ops/cuda/fused_residual_rmsnorm/include/hqsb/fused_residual_rmsnorm.h",
                Path(__file__).resolve())},
        })
        dump(output / "binary_resources.json", {
            "cuobjdump": run(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(library)]),
            "compile_contract": {"target": "hqsb_fused_residual_rmsnorm_shared",
                                 "flags": ["-O3", "-lineinfo"], "fast_math": False},
        })
    return 0


def profile_timeline(args) -> int:
    output = Path(args.output_dir).resolve()
    library = find_library(args.library)
    api = Api(library)
    rows, hidden, dtype = 128, 2048, "fp16"
    nx, nr, nw = make_inputs(rows, hidden, dtype, SEED)
    x, residual, weight = [torch.from_numpy(a).to("cuda") for a in (nx, nr, nw)]
    ro, y = torch.empty_like(x), torch.empty_like(x)
    stream = torch.cuda.Stream(); torch.cuda.synchronize()
    summary = {"tool": "torch.profiler CUDA activity trace (NSys-equivalent timeline)",
               "cases": {}}
    for mode in ("separate", "fused"):
        launch = api.bind(mode, x, residual, weight, ro, y, rows, hidden, 1, stream)
        for _ in range(5): launch()
        stream.synchronize()
        before = {"allocated": int(torch.cuda.memory_allocated()),
                  "reserved": int(torch.cuda.memory_reserved())}
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                profile_memory=True, record_shapes=True) as prof:
            with torch.profiler.record_function(f"E03_05_{mode}"):
                rc = launch()
                stream.synchronize()
        trace = output / "timeline" / f"{mode}.json"
        trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace))
        cuda_events = []
        allocator_events = []
        for event in prof.events():
            name = str(event.name)
            if "cudaMalloc" in name or "cudaFree" in name or "cudaMemcpy" in name:
                allocator_events.append(name)
            if any(token in name for token in ("residual_add_kernel", "rmsnorm_",
                                                "fused_residual_rmsnorm")):
                cuda_events.append(name)
        after = {"allocated": int(torch.cuda.memory_allocated()),
                 "reserved": int(torch.cuda.memory_reserved())}
        summary["cases"][mode] = {
            "return_code": rc, "target_kernel_events": cuda_events,
            "target_kernel_count": len(cuda_events),
            "malloc_free_memcpy_events": allocator_events,
            "allocator_before": before, "allocator_after": after,
            "trace": str(trace.relative_to(output)),
        }
    dump(output / "timeline" / "summary.json", summary)
    return 0


def ncu_profile_target(args) -> int:
    library = find_library(args.library)
    api = Api(library)
    nx, nr, nw = make_inputs(args.rows, args.hidden, "fp16", SEED)
    x, residual, weight = [torch.from_numpy(a).to("cuda") for a in (nx, nr, nw)]
    ro, y = torch.empty_like(x), torch.empty_like(x)
    stream = torch.cuda.Stream(); torch.cuda.synchronize()
    launch = api.bind(args.mode, x, residual, weight, ro, y, args.rows,
                      args.hidden, 1, stream)
    for _ in range(4):
        if launch() != 0: return 2
    stream.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    rc = launch(); stream.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    return 0 if rc == 0 else 2


def parse_raw_csv(text: str) -> dict[str, Any]:
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3:
        return {"error": "raw CSV unavailable", "counters": {}}
    header, units, values = rows[0], rows[1], rows[2]
    counters = {}
    tokens = ("dram__", "lts__t_sector", "launch__registers",
              "launch__shared_mem", "launch__occupancy", "warps_active",
              "warps_issue_stalled")
    for i, name in enumerate(header):
        if not any(t in name for t in tokens): continue
        raw = values[i] if i < len(values) else ""
        try: value: Any = float(raw.replace(",", ""))
        except ValueError: value = None if raw.lower() in ("", "n/a", "nan", "-") else raw
        counters[name] = {"value": value, "unit": units[i] if i < len(units) else "", "raw": raw}
    return {"counters": counters, "counter_count": len(counters),
            "dram_counter_count": sum(k.startswith("dram__") for k in counters),
            "l2_sector_counter_count": sum(k.startswith("lts__t_sector") for k in counters)}


def ncu_capture(args) -> int:
    output = Path(args.output_dir).resolve()
    library = find_library(args.library)
    ncu = "/usr/local/cuda/bin/ncu"
    dump(output / "ncu" / "tool_query.json", {
        "version": run([ncu, "--version"]),
        "sections": run([ncu, "--list-sections"], timeout=180),
        "requested_sections": list(NCU_SECTIONS),
    })
    outcomes = []
    for rows, label in ((1, "decode"), (1024, "prefill")):
        for mode in ("separate", "fused"):
            profile_id = f"{label}_{mode}"
            stem = output / "ncu" / profile_id
            report = stem.with_suffix(".ncu-rep")
            command = ["sudo", "-n", "-E", ncu, "--profile-from-start", "off"]
            for section in NCU_SECTIONS:
                command += ["--section", section]
            command += ["--export", str(report), "--force-overwrite",
                        sys.executable, str(Path(__file__).resolve()), "profile",
                        "--library", str(library), "--mode", mode,
                        "--rows", str(rows), "--hidden", "2048"]
            captured = run(command, timeout=900)
            stem.with_suffix(".stdout.txt").write_text(captured["stdout"])
            stem.with_suffix(".stderr.txt").write_text(captured["stderr"])
            details = run(["sudo", "-n", ncu, "--import", str(report), "--csv",
                           "--page", "details", "--print-units", "base"], 300) \
                if report.is_file() else {"returncode": 1, "stdout": "", "stderr": "no report"}
            raw = run(["sudo", "-n", ncu, "--import", str(report), "--csv",
                       "--page", "raw", "--print-units", "base"], 300) \
                if report.is_file() else {"returncode": 1, "stdout": "", "stderr": "no report"}
            stem.with_suffix(".csv").write_text(details["stdout"])
            stem.with_suffix(".raw.csv").write_text(raw["stdout"])
            if report.is_file():
                run(["sudo", "-n", "chown", f"{os.getuid()}:{os.getgid()}", str(report)])
            kernel_names = []
            for line in details["stdout"].splitlines():
                if "kernel" in line.lower() and ("residual" in line.lower() or "rmsnorm" in line.lower()):
                    kernel_names.append(line[:500])
            outcome = {
                "profile_id": profile_id, "mode": mode, "rows": rows, "hidden": 2048,
                "command": command, "capture_returncode": captured["returncode"],
                "export_returncode": details["returncode"], "report_exists": report.is_file(),
                "report_sha256": sha256_file(report) if report.is_file() else None,
                "details_kernel_lines": kernel_names, "raw": parse_raw_csv(raw["stdout"]),
            }
            dump(stem.with_suffix(".json"), outcome); outcomes.append(outcome)
            print(profile_id, captured["returncode"], flush=True)
    dump(output / "ncu" / "capture_summary.json", {"outcomes": outcomes})
    return 0 if all(x["capture_returncode"] == 0 and x["report_exists"] for x in outcomes) else 2


def read_records(output: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(output.glob("cases_proc*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return records


def summarize(args) -> int:
    output = Path(args.output_dir).resolve()
    records = read_records(output)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["case_id"], []).append(record)
    table = []
    for case_id, group in sorted(grouped.items()):
        base = group[0]
        item = {k: base[k] for k in ("case_id", "shape_id", "role", "rows", "hidden", "dtype")}
        item["processes"] = len(group)
        item["all_correct"] = all(r["status"] == "MEASURED" for r in group)
        item["timing"] = {}
        for mode in ("add", "rmsnorm", "separate", "fused"):
            device = [v for r in group for v in r["timings"][mode].get("device_ms_samples", [])]
            host = [v for r in group for v in r["timings"][mode].get("host_submit_completion_ms_samples", [])]
            item["timing"][mode] = {
                "device_ms_median": statistics.median(device),
                "device_ms_p95": percentile(device, .95),
                "host_ms_median": statistics.median(host),
                "samples": len(device),
            }
        sep = item["timing"]["separate"]; fused = item["timing"]["fused"]
        item["speedup_device"] = sep["device_ms_median"] / fused["device_ms_median"]
        item["speedup_host"] = sep["host_ms_median"] / fused["host_ms_median"]
        elem = 2 if item["dtype"] == "fp16" else 4
        n = item["rows"] * item["hidden"]
        item["logical_bytes"] = {"separate": 6*n*elem, "fused": 5*n*elem,
                                 "saved": n*elem, "saved_fraction": 1/6}
        item["guard_band_pass_device"] = item["speedup_device"] >= 1 + GUARD_BAND
        item["allocator_stable"] = all(
            r["allocation"][m]["memory_allocated_before"] == r["allocation"][m]["memory_allocated_after"]
            and r["allocation"][m]["memory_reserved_before"] == r["allocation"][m]["memory_reserved_after"]
            for r in group for m in ("separate", "fused"))
        item["shared_bytes"] = base["fused_dynamic_shared_bytes"]
        item["max_active_blocks_per_sm"] = base["fused_max_active_blocks_per_sm"]
        table.append(item)
    dump(output / "ordinary_summary.json", {"cases": table})
    timeline = load(output / "timeline/summary.json", {})
    ncu = load(output / "ncu/capture_summary.json", {})
    fused_tl = timeline.get("cases", {}).get("fused", {})
    separate_tl = timeline.get("cases", {}).get("separate", {})
    conditions = {
        "matrix_complete": len(table) == len(SHAPES) * len(DTYPES),
        "three_independent_processes": all(x["processes"] == 3 for x in table),
        "all_correct": all(x["all_correct"] for x in table),
        "residual_and_y_checked": all(
            all(r["correctness"][m]["residual_exact"] and r["correctness"][m]["passed"]
                for m in ("separate", "fused")) for r in records),
        "ordinary_device_host_complete": all(
            all(x["timing"][m]["samples"] == PROCESSES*GROUPS
                for m in ("add", "rmsnorm", "separate", "fused")) for x in table),
        "logical_bytes_saved": all(x["logical_bytes"]["saved"] > 0 for x in table),
        "allocation_stable": all(x["allocator_stable"] for x in table),
        "timeline_launch_reduction": separate_tl.get("target_kernel_count") == 2 and
                                     fused_tl.get("target_kernel_count") == 1,
        "timeline_no_hidden_copy_alloc": not separate_tl.get("malloc_free_memcpy_events") and
                                         not fused_tl.get("malloc_free_memcpy_events"),
        "ncu_captures_present": len(ncu.get("outcomes", [])) == 4 and
                                all(x.get("report_exists") for x in ncu.get("outcomes", [])),
        "model_dataflow_frozen": load(output / "model_dataflow.json", {}).get("hidden_size") == 2048,
    }
    performance_positive = any(x["guard_band_pass_device"] for x in table)
    overall = "PASS" if all(conditions.values()) else "FAIL"
    verdict = {
        "schema_version": SCHEMA, "experiment_id": EXPERIMENT,
        "verified_at_utc": utc_now(), "overall": overall,
        "performance_classification": "PASS" if performance_positive else "PASS_NEGATIVE",
        "conditions": conditions,
        "failed_conditions": [k for k, v in conditions.items() if not v],
        "guard_band": GUARD_BAND,
        "performance_positive_cases": [x["case_id"] for x in table if x["guard_band_pass_device"]],
        "dispatch_recommendation": "eligible candidates only; E03-06/E03-07 stream and sanitizer gates remain before auto routing",
    }
    dump(output / "summary.json", {
        "case_count": len(table), "all_speedups": [{"case_id": x["case_id"],
        "speedup_device": x["speedup_device"], "speedup_host": x["speedup_host"],
        "guard_band": x["guard_band_pass_device"]} for x in table],
        "ncu_note": "measured DRAM counters may be unavailable on Tegra; raw availability is explicit",
        "verdict": verdict,
    })
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
        "generated_at_utc": utc_now(), "entries": entries,
    })


def verify(args) -> int:
    verdict = load(Path(args.output_dir).resolve() / "verdict.json", {})
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if verdict.get("overall") == "PASS" else 2


def negative(args) -> int:
    output = Path(args.output_dir).resolve(); api = Api(find_library(args.library))
    x = torch.zeros((1, 128), device="cuda"); r = torch.zeros_like(x)
    w = torch.ones(128, device="cuda"); ro = torch.empty_like(x); y = torch.empty_like(x)
    stream = torch.cuda.Stream(); s = int(stream.cuda_stream)
    cases = {
        "null_input": int(api.fused(None, r.data_ptr(), w.data_ptr(), ro.data_ptr(), y.data_ptr(), 1, 128, EPSILON, 0, 3, 1, s)),
        "rows_zero": int(api.fused(x.data_ptr(), r.data_ptr(), w.data_ptr(), ro.data_ptr(), y.data_ptr(), 0, 128, EPSILON, 0, 3, 1, s)),
        "bf16_code": int(api.fused(x.data_ptr(), r.data_ptr(), w.data_ptr(), ro.data_ptr(), y.data_ptr(), 1, 128, EPSILON, 2, 3, 1, s)),
        "semantic_b": int(api.fused(x.data_ptr(), r.data_ptr(), w.data_ptr(), ro.data_ptr(), y.data_ptr(), 1, 128, EPSILON, 0, 3, 2, s)),
        "output_alias_input": int(api.fused(x.data_ptr(), r.data_ptr(), w.data_ptr(), x.data_ptr(), y.data_ptr(), 1, 128, EPSILON, 0, 3, 1, s)),
        "outputs_alias": int(api.fused(x.data_ptr(), r.data_ptr(), w.data_ptr(), ro.data_ptr(), ro.data_ptr(), 1, 128, EPSILON, 0, 3, 1, s)),
        "hidden_too_large": int(api.fused(x.data_ptr(), r.data_ptr(), w.data_ptr(), ro.data_ptr(), y.data_ptr(), 1, 8193, EPSILON, 0, 3, 1, s)),
    }
    payload = {"expected_cuda_error_invalid_value": 1,
               "cases": {k: {"return_code": v, "passed": v == 1} for k, v in cases.items()}}
    dump(output / "negative_cases.json", payload)
    return 0 if all(v == 1 for v in cases.values()) else 2


def main() -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "timeline", "ncu", "summarize", "verify", "negative"):
        p = sub.add_parser(name); p.add_argument("--output-dir", required=True)
        if name in ("timeline", "ncu", "negative"): p.add_argument("--library")
    p = sub.add_parser("collect"); p.add_argument("--output-dir", required=True)
    p.add_argument("--process-index", type=int, choices=range(PROCESSES), required=True)
    p.add_argument("--library")
    p = sub.add_parser("profile"); p.add_argument("--library", required=True)
    p.add_argument("--mode", choices=("separate", "fused"), required=True)
    p.add_argument("--rows", type=int, required=True); p.add_argument("--hidden", type=int, required=True)
    args = parser.parse_args()
    return {"plan": lambda: make_plan(Path(args.output_dir).resolve()),
            "collect": lambda: collect(args), "timeline": lambda: profile_timeline(args),
            "ncu": lambda: ncu_capture(args), "profile": lambda: ncu_profile_target(args),
            "negative": lambda: negative(args), "summarize": lambda: summarize(args),
            "verify": lambda: verify(args)}[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
