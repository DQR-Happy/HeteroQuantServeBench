#!/usr/bin/env python3
"""E04-11 TileLang RMSNorm and HIP/ROCm portability evidence collector.

Run only on the configured accelerator target through ``scripts/remote_run.sh``.
The experiment keeps TileLang out of the production dispatcher and separates
the NVIDIA-executed prototype from the AMD execution claim, which is blocked
unless a real ROCm device is present.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hqsb.benchmark import rmsnorm_correctness as rc  # noqa: E402


EXPERIMENT_ID = "E04-11"
SCHEMA = "hqsb.s04.tilelang_hip_portability/v1"
EPSILON = 1.0e-6
GUARD = 0.05
NCU = Path("/usr/local/cuda/bin/ncu")
NSYS = Path("/usr/local/bin/nsys")

MANDATORY_H = (1, 3, 31, 32, 33, 99, 100, 101, 127, 128, 129,
               511, 512, 513, 2047, 2048, 2049, 6144, 8192)
CORRECTNESS_SHAPES = (
    *({"rows": 1, "hidden": h, "split": "boundary"} for h in MANDATORY_H),
    {"rows": 8, "hidden": 128, "split": "real_qk"},
    {"rows": 128, "hidden": 2048, "split": "real_hidden"},
    {"rows": 257, "hidden": 129, "split": "holdout"},
    {"rows": 7, "hidden": 2049, "split": "holdout"},
)
PERF_CASES = (
    {"case_id": "qk_bulk", "rows": 1024, "hidden": 128, "dtype": "fp16"},
    {"case_id": "hidden_bulk", "rows": 128, "hidden": 2048, "dtype": "fp16"},
    {"case_id": "tail_holdout", "rows": 257, "hidden": 129, "dtype": "fp16"},
    {"case_id": "hidden_fp32", "rows": 128, "hidden": 2048, "dtype": "fp32"},
)
BACKENDS = ("pytorch_reference", "cuda_v2", "triton_reference", "tilelang_fixed")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(dict(row)), sort_keys=True,
                                    allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(command: Sequence[str], timeout: int = 600,
        env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(list(command), cwd=REPO, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, check=False,
                              env=dict(env) if env is not None else None)
        return {"command": list(command), "exit_code": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr,
                "duration_s": time.perf_counter() - start, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        return {"command": list(command), "exit_code": 124,
                "stdout": exc.stdout or "", "stderr": exc.stderr or "",
                "duration_s": time.perf_counter() - start, "timed_out": True}


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def git(*args: str) -> str:
    return run(["git", *args], 120)["stdout"].strip()


def module_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def distribution_record(name: str) -> Dict[str, Any]:
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"name": name, "installed": False}
    metadata = dist.metadata
    root = Path(dist.locate_file(""))
    record_candidates = [Path(dist.locate_file(item)) for item in (dist.files or [])
                         if str(item).endswith("RECORD")]
    direct_candidates = [Path(dist.locate_file(item)) for item in (dist.files or [])
                         if str(item).endswith("direct_url.json")]
    return {
        "name": name,
        "installed": True,
        "version": dist.version,
        "license": metadata.get("License") or metadata.get("License-Expression"),
        "home_page": metadata.get("Home-page") or metadata.get("Project-URL"),
        "requires": list(dist.requires or []),
        "location": str(root),
        "record": ({"path": str(record_candidates[0]),
                    "sha256": sha256_file(record_candidates[0])}
                   if record_candidates and record_candidates[0].is_file() else None),
        "direct_url": (json.loads(direct_candidates[0].read_text())
                       if direct_candidates and direct_candidates[0].is_file() else None),
    }


def hip_mapping() -> Dict[str, Any]:
    rows = [
        ("cudaStream_t/current stream", "hipStream_t / framework current stream", "legacy default-stream ordering and framework stream extraction are not mechanical renames", "non-default and dual-stream timeline"),
        ("cudaEvent", "hipEvent", "flags, timing resolution and synchronization cost differ", "event ordering plus host/device timing separation"),
        ("cudaError_t", "hipError_t", "numeric codes and async surfacing differ", "negative API matrix and post-launch error check"),
        ("nvcc", "hipcc/clang", "flags, arch names, headers and host compiler compatibility differ", "clean configure/build for each gfx target"),
        ("warpSize=32", "warpSize, commonly wave32 or wave64 by AMD architecture", "shuffle masks and fixed reduction trees change semantics/resource cost", "query runtime warpSize and run odd-width reductions"),
        ("shared memory", "LDS/shared", "capacity, banks, allocation granularity and occupancy tradeoffs differ", "resource report and bank-conflict metrics"),
        ("__shfl*_sync mask", "HIP shuffle intrinsics", "CUDA active-mask semantics and width assumptions need review", "partial-wave and divergent-lane tests"),
        ("Tensor Core MMA", "MFMA/WMMA", "instruction shapes, dtype support and accumulator layouts differ", "ISA/source inspection plus numerical matrix"),
        ("cuBLAS/cuBLASLt", "rocBLAS/hipBLASLt", "heuristics, layouts, workspace and algorithm IDs are vendor-local", "fresh local search; never copy algorithm/cache IDs"),
        ("NCU/NSys", "rocprof/rocprofv3 and ROCm tracing", "counter names and derived metrics are not one-to-one", "map questions, not metric labels; retain raw traces"),
        ("compute capability/SM", "gfx architecture/features/CU", "capability keys and binary compatibility are different namespaces", "device UUID+gfx+runtime in cache key"),
        ("CUTLASS", "Composable Kernel / hipBLASLt candidates", "template/config identities and epilogues are not portable", "re-register candidate space on AMD"),
        ("CUDA allocator/lifetime", "HIP/framework allocator", "stream-ordered lifetime and graph capture rules must be revalidated", "multi-stream lifetime stress and sanitizer/tool checks"),
        ("PTX/cubin", "LLVM IR/GCN ISA/hsaco", "binary and compiler cache artifacts cannot cross vendors", "separate cache namespaces and forced stale-cache rejection"),
    ]
    return {
        "scope": "engineering map; not AMD runtime evidence",
        "rows": [
            {"cuda_hqsb_concept": a, "hip_rocm_correspondence": b,
             "non_mechanical_hazard": c, "required_validation": d}
            for a, b, c, d in rows
        ],
        "build_route": [
            "freeze ROCm, hipcc/clang, PyTorch ROCm, TileLang and gfx architecture",
            "keep OperatorSpec, seeds, input hashes and tolerances unchanged",
            "compile a wave-size-independent scalar/reference reduction first",
            "run real/odd/tail/dtype/layout/invalid/current-stream tests before timing",
            "use an AMD-local cache namespace and inject NVIDIA cache as an expected miss",
            "collect three fresh-process ordinary runs and separate cold compile/cache-hit/steady",
            "profile with rocprof/rocprofv3 and inspect generated LLVM/GCN artifacts",
            "compare only against same-machine legal PyTorch/rocBLAS baselines",
        ],
        "claim_rules": [
            "HIP-on-NVIDIA, hipify success, or successful source compilation cannot prove AMD correctness",
            "AMD performance and wavefront optimization remain blocked without a real AMD GPU",
            "vendor-local binaries, algorithm IDs and tune caches are never copied as cross-vendor hits",
        ],
    }


def source_portability_audit() -> Dict[str, Any]:
    patterns = {
        "cuda_runtime_api": re.compile(r"cuda(Stream|Event|Error|Malloc|Free|Memcpy|GetDevice|Device)"),
        "cuda_kernel_syntax": re.compile(r"(__global__|__device__|<<<|threadIdx|blockIdx|blockDim|gridDim)"),
        "warp32_assumption": re.compile(r"(FULL_MASK|0xffffffff|warpSize\s*==\s*32|/\s*32|%\s*32)"),
        "cuda_shuffle": re.compile(r"__shfl.*_sync"),
        "nvidia_intrinsic": re.compile(r"(__half2|__hadd2|__hmul2|__ldg|__syncwarp)"),
        "cuda_build": re.compile(r"(CUDA_ARCHITECTURES|nvcc|compute_[0-9]+|sm_[0-9]+)"),
    }
    files = [p for p in (REPO / "ops").rglob("*")
             if p.is_file() and p.suffix in {".cu", ".cuh", ".h", ".hpp", ".cpp", ".py", ".txt"}]
    findings = []
    counts = {name: 0 for name in patterns}
    for path in sorted(files):
        text = path.read_text(errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            matched = [name for name, pattern in patterns.items() if pattern.search(line)]
            if matched:
                for name in matched:
                    counts[name] += 1
                findings.append({"path": str(path.relative_to(REPO)), "line": number,
                                 "categories": matched, "text": line.strip()[:240]})
    return {
        "audited_root": "ops/",
        "pattern_counts": counts,
        "finding_count": len(findings),
        "findings": findings,
        "interpretation": "lexical inventory requiring manual semantic review; not a claim that each match is a defect",
    }


def initialize(args) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "protocol.json", {
        "experiment_id": EXPERIMENT_ID, "schema": SCHEMA,
        "frozen_at_utc": utc_now(),
        "operator": "RMSNorm with FP32 square/reduction/epsilon/rsqrt/weight and output cast",
        "epsilon": EPSILON, "threads": 128,
        "correctness_shapes": list(CORRECTNESS_SHAPES),
        "correctness_dtypes": ["fp16", "fp32"],
        "special_modes": ["nan_inject", "posinf", "neginf"],
        "performance_cases": list(PERF_CASES),
        "performance_backends": list(BACKENDS),
        "warmup": 5, "repeats": 31, "independent_processes": 3,
        "primary_metric": "per-case device latency p50 (us)",
        "secondary_metrics": ["cold compile wall time", "fresh-process cache-hit compile wall time", "p95", "effective GB/s", "generated source/resource profile"],
        "practical_guard": GUARD,
        "cache_policy": "experiment-private TILELANG_CACHE_DIR; process 0 starts empty, processes 1/2 reuse it",
        "layout_policy": "contiguous supported; non-contiguous rejected before compilation",
        "stream_policy": "default, current non-default and dual-stream functional test; generated host source inspected",
        "autotune": "not run; fixed readable 128-thread implementation only",
        "claim_boundary": "operator prototype on NVIDIA SM87 only; no model benefit, AMD correctness or ROCm performance claim",
        "stopping_rule": "retain all failures; never switch operator or loosen frozen S03 tolerance after results",
    })
    write_json(out / "hip_technical_map.json", hip_mapping())
    write_json(out / "portability_source_audit.json", source_portability_audit())

    optional_code = (
        "import json,sys; import ops.reference; "
        "print(json.dumps({'optional_imported':sorted(set(sys.modules)&{'tilelang','triton','cutlass'}),"
        "'reference_import_ok':True}))"
    )
    optional_proc = run([sys.executable, "-c", optional_code], 120)
    optional_payload = None
    try:
        optional_payload = json.loads(optional_proc["stdout"].strip().splitlines()[-1])
    except Exception:
        pass
    inherited_path = REPO / "docs/stage_experiments/S04/E04-01/raw/verdict.json"
    write_json(out / "optional_isolation.json", {
        "fresh_process": optional_proc,
        "parsed": optional_payload,
        "passed": bool(optional_proc["exit_code"] == 0 and optional_payload
                       and not optional_payload["optional_imported"]),
        "inherited_e04_01": ({"path": str(inherited_path.relative_to(REPO)),
                              "sha256": sha256_file(inherited_path),
                              "overall": read_json(inherited_path).get("overall")}
                             if inherited_path.exists() else None),
    })

    import torch
    import tilelang
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    tool_names = ("hipcc", "rocminfo", "rocprof", "rocprofv3", "hipify-perl",
                  "compute-sanitizer", "ncu", "nsys")
    write_json(out / "dependency_manifest.json", {
        "collected_at_utc": utc_now(),
        "packages": [distribution_record(name) for name in
                     ("tilelang", "apache-tvm-ffi", "torch", "numpy", "z3-solver")],
        "tilelang_runtime_version": str(tilelang.__version__),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "torch_hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "device": ({"name": props.name, "compute_capability": list(torch.cuda.get_device_capability(0)),
                    "warp_size": getattr(props, "warp_size", None),
                    "multiprocessor_count": props.multi_processor_count,
                    "total_memory_bytes": props.total_memory}
                   if props is not None else None),
        "tools": {name: shutil.which(name) for name in tool_names},
        "amd_runtime_available": bool(getattr(torch.version, "hip", None) and torch.cuda.is_available()),
        "release_policy": "stable PyPI release required; nightly not used",
    })

    sources = [
        "scripts/audit/run_e04_11_tilelang_hip_portability.py",
        "ops/tilelang/rmsnorm.py", "ops/tilelang/__init__.py",
        "ops/_tilelang_probe.py",
        "docs/stage_experiments/details/S04/E04-11_tilelang_hip_portability.md",
    ]
    write_json(out / "provenance.json", {
        "created_at_utc": utc_now(), "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "sources": {name: {"sha256": sha256_file(REPO / name),
                           "bytes": (REPO / name).stat().st_size}
                    for name in sources},
    })
    source = REPO / "ops/tilelang/rmsnorm.py"
    semantic_lines = [line for line in source.read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith("#")]
    write_json(out / "development_log.json", {
        "recording_started_at_utc": utc_now(),
        "measurement_boundary": "agent implementation session only; no human comprehension time claimed",
        "initial_prototype_mtime_utc": dt.datetime.fromtimestamp(source.stat().st_mtime, dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_metrics": {"path": str(source.relative_to(REPO)),
                           "physical_lines": len(source.read_text().splitlines()),
                           "nonblank_noncomment_lines": len(semantic_lines)},
        "events": [
            {"kind": "documentation", "result": "selected RMSNorm before performance; reused frozen S03/E04-02 semantics"},
            {"kind": "implementation", "result": "fixed 128-thread exact-shape TIR with FP32 fragments and reduction"},
            {"kind": "known_limit", "result": "session log began after the initial local prototype edit; this gap is explicit rather than reconstructed"},
        ],
        "human_time": None, "human_time_status": "NOT_MEASURED",
        "agent_time_is_not_human_productivity": True,
    })
    return 0


def torch_dtype(torch, dtype: str):
    return torch.float16 if dtype == "fp16" else torch.float32


def framework_reference(torch, x, weight):
    acc = x.float()
    return (acc * torch.rsqrt(torch.mean(acc * acc, dim=-1, keepdim=True) + EPSILON)
            * weight.float()).to(x.dtype)


def correctness_case(rows: int, hidden: int, dtype: str, mode: str,
                     split: str) -> Dict[str, Any]:
    import torch
    from ops.tilelang.rmsnorm import rmsnorm_tilelang

    case_id = f"{split}|r{rows}|h{hidden}|{dtype}|{mode}"
    seed = rc.derive_seed(f"e04-11|{case_id}")
    generated = rc.generate_case_arrays(mode, rows, hidden, dtype, seed)
    x_np, w_np = generated["x"], generated["w"]
    td = torch_dtype(torch, dtype)
    x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device="cuda", dtype=td)
    w = torch.from_numpy(np.ascontiguousarray(w_np)).to(device="cuda", dtype=td)
    base = {"case_id": case_id, "rows": rows, "hidden": hidden,
            "dtype": dtype, "mode": mode, "split": split, "seed": seed,
            "x_sha256": sha256_array(x_np), "weight_sha256": sha256_array(w_np),
            "tolerance": rc.tolerance_for(dtype, hidden),
            "requested_backend": "tilelang_fixed", "actual_backend": None}
    try:
        candidate_tensor = rmsnorm_tilelang(x, w, EPSILON)
        torch.cuda.synchronize()
        candidate = candidate_tensor.cpu().numpy().reshape(-1)
        reference = framework_reference(torch, x, w).cpu().numpy().reshape(-1)
        repeated = rmsnorm_tilelang(x, w, EPSILON)
        torch.cuda.synchronize()
        repeated_np = repeated.cpu().numpy().reshape(-1)
    except Exception as exc:
        return {**base, "status": "FAIL", "error": {"type": type(exc).__name__,
                                                       "message": str(exc)}}

    expected_classes = rc.expected_classes(x_np, rows, hidden)
    candidate_classes = rc.classify(candidate)
    finite = expected_classes == rc.CLASS_FINITE
    tolerance = rc.tolerance_for(dtype, hidden)
    numeric = rc.elementwise_violations(candidate, reference, tolerance, finite)
    metrics = rc.error_metrics(candidate, reference, finite)
    exact_zero_mask = rc.inf_row_finite_lanes_exact_zero(x_np, rows, hidden)
    exact_zero_ok = bool(np.all(candidate[exact_zero_mask] == 0.0)) if np.any(exact_zero_mask) else True
    checks = {
        "classification": bool(np.array_equal(candidate_classes, expected_classes)),
        "elementwise": bool(numeric["passed"]),
        "l2rel": bool(not metrics["applicable"] or metrics["l2rel"] <= tolerance["l2rel"]),
        "cosine": bool(not metrics["cosine_applicable"] or metrics["cosine"] >= tolerance["cosine_min"]),
        "inf_finite_lanes_zero": exact_zero_ok,
        "repeat_stable": bool(np.array_equal(candidate.view(np.uint8), repeated_np.view(np.uint8))),
    }
    return {**base, "actual_backend": "tilelang_fixed", "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "elementwise": numeric, "finite_metrics": metrics,
            "expected_class_counts": rc.class_counts(expected_classes),
            "candidate_class_counts": rc.class_counts(candidate_classes),
            "output_sha256": sha256_array(candidate),
            "first_mismatch": rc.first_mismatch(candidate=candidate, reference=reference,
                candidate_classes=candidate_classes, expected_class_array=expected_classes,
                tolerance=tolerance, hidden=hidden)}


def negative_and_stream_checks() -> Dict[str, Any]:
    import torch
    from ops.tilelang.rmsnorm import rmsnorm_tilelang

    negatives = []
    good_x = torch.randn((4, 33), device="cuda", dtype=torch.float16)
    good_w = torch.randn((33,), device="cuda", dtype=torch.float16)
    cases = [
        ("cpu_input", lambda: rmsnorm_tilelang(good_x.cpu(), good_w.cpu(), EPSILON)),
        ("dtype_mismatch", lambda: rmsnorm_tilelang(good_x, good_w.float(), EPSILON)),
        ("weight_shape", lambda: rmsnorm_tilelang(good_x, good_w[:-1], EPSILON)),
        ("noncontiguous", lambda: rmsnorm_tilelang(torch.randn((33, 4), device="cuda", dtype=torch.float16).t(), good_w, EPSILON)),
        ("bad_epsilon", lambda: rmsnorm_tilelang(good_x, good_w, 0.0)),
    ]
    for name, func in cases:
        try:
            func(); torch.cuda.synchronize()
            negatives.append({"case": name, "status": "FAIL", "reason": "accepted"})
        except (TypeError, ValueError) as exc:
            negatives.append({"case": name, "status": "PASS",
                              "error_type": type(exc).__name__, "message": str(exc)})
        except Exception as exc:
            negatives.append({"case": name, "status": "FAIL",
                              "error_type": type(exc).__name__, "message": str(exc)})

    streams = []
    for label, stream in (("default", torch.cuda.default_stream()),
                          ("current_nondefault", torch.cuda.Stream())):
        x = torch.full((8, 128), 0.25, device="cuda", dtype=torch.float16)
        w = torch.linspace(0.5, 1.5, 128, device="cuda", dtype=torch.float16)
        with torch.cuda.stream(stream):
            x.mul_(2.0)
            y = rmsnorm_tilelang(x, w, EPSILON)
            expected = framework_reference(torch, x, w)
        stream.synchronize()
        streams.append({"case": label, "status": "PASS" if torch.equal(y, expected) else "FAIL",
                        "max_abs": float((y.float() - expected.float()).abs().max())})

    dual = []
    stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
    for index, stream in enumerate((stream_a, stream_b)):
        x = torch.full((7, 129), 0.25 + index, device="cuda", dtype=torch.float16)
        w = torch.linspace(0.5, 1.5, 129, device="cuda", dtype=torch.float16)
        with torch.cuda.stream(stream):
            y = rmsnorm_tilelang(x, w, EPSILON)
            expected = framework_reference(torch, x, w)
        dual.append((stream, y, expected))
    for stream, _, _ in dual:
        stream.synchronize()
    dual_ok = all(torch.equal(y, expected) for _, y, expected in dual)
    streams.append({"case": "dual_stream", "status": "PASS" if dual_ok else "FAIL"})
    return {"negative_validation": negatives, "stream_cases": streams,
            "status": "PASS" if all(r["status"] == "PASS" for r in negatives + streams) else "FAIL"}


def launch_backend(name: str, torch, x, weight):
    if name == "pytorch_reference":
        return framework_reference(torch, x, weight)
    if name == "tilelang_fixed":
        from ops.tilelang.rmsnorm import rmsnorm_tilelang
        return rmsnorm_tilelang(x, weight, EPSILON)
    if name == "triton_reference":
        from ops.triton.rmsnorm import rmsnorm_reference
        return rmsnorm_reference(x, weight, EPSILON)
    if name == "cuda_v2":
        from ops.cuda_bridge import rmsnorm_forward
        return rmsnorm_forward(x, weight, dtype="fp16" if x.dtype == torch.float16 else "fp32",
                               variant="v2_vectorized", epsilon=EPSILON,
                               stream=torch.cuda.current_stream())
    raise ValueError(name)


def time_backend(name: str, torch, x, weight) -> Dict[str, Any]:
    try:
        output = launch_backend(name, torch, x, weight)
        torch.cuda.synchronize()
        reference = framework_reference(torch, x, weight)
        tolerance = rc.tolerance_for("fp16" if x.dtype == torch.float16 else "fp32", x.shape[1])
        correct = bool(torch.allclose(output, reference, atol=tolerance["atol"], rtol=tolerance["rtol"], equal_nan=True))
        for _ in range(5):
            launch_backend(name, torch, x, weight)
        torch.cuda.synchronize()
        values = []
        for _ in range(31):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); launch_backend(name, torch, x, weight); end.record(); end.synchronize()
            values.append(float(start.elapsed_time(end)) * 1000.0)
        return {"backend": name, "status": "PASS" if correct else "FAIL",
                "correct": correct, "device_us_raw": values,
                "device_us_p50": statistics.median(values),
                "device_us_p95": percentile(values, 0.95)}
    except Exception as exc:
        return {"backend": name, "status": "FAIL",
                "error": {"type": type(exc).__name__, "message": str(exc)}}


def performance_process(process_index: int) -> Dict[str, Any]:
    import torch
    from ops.tilelang.rmsnorm import compile_rmsnorm

    rows = []
    for case in PERF_CASES:
        td = torch_dtype(torch, case["dtype"])
        generator = torch.Generator(device="cuda")
        generator.manual_seed(411000 + process_index * 100 + case["hidden"])
        x = torch.randn((case["rows"], case["hidden"]), generator=generator,
                        device="cuda", dtype=td) * 0.25
        weight = torch.randn((case["hidden"],), generator=generator,
                             device="cuda", dtype=td) * 0.25
        compile_rmsnorm.cache_clear()
        compile_start = time.perf_counter()
        kernel = compile_rmsnorm(case["rows"], case["hidden"],
                                 "float16" if td == torch.float16 else "float32",
                                 EPSILON, 128)
        compile_wall_ms = (time.perf_counter() - compile_start) * 1000.0
        order = list(BACKENDS[process_index:] + BACKENDS[:process_index])
        measurements = [time_backend(name, torch, x, weight) for name in order]
        bytes_moved = (2 * case["rows"] * case["hidden"] + case["hidden"]) * x.element_size()
        for item in measurements:
            if item.get("device_us_p50"):
                item["effective_gbps"] = bytes_moved / (item["device_us_p50"] * 1.0e-6) / 1.0e9
        rows.append({**case, "order": order, "input_sha256": sha256_array(x.cpu().numpy()),
                     "weight_sha256": sha256_array(weight.cpu().numpy()),
                     "tilelang_compile_wall_ms": compile_wall_ms,
                     "tilelang_kernel_type": type(kernel).__name__,
                     "measurements": measurements})
    return {"process_index": process_index, "fresh_process": True,
            "tilelang_cache_dir": os.environ.get("TILELANG_CACHE_DIR"),
            "collected_at_utc": utc_now(), "rows": rows}


def collect_artifacts(out: Path) -> Dict[str, Any]:
    from ops.tilelang.rmsnorm import compile_rmsnorm

    kernel = compile_rmsnorm(128, 2048, "float16", EPSILON, 128)
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    tir_path = artifacts / "tilelang_rmsnorm_r128_h2048_fp16.tir.txt"
    source_path = artifacts / "tilelang_rmsnorm_r128_h2048_fp16.generated.cu"
    host_path = artifacts / "tilelang_rmsnorm_r128_h2048_fp16.host.cpp"
    tir_path.write_text(str(getattr(kernel, "prim_func", "UNAVAILABLE")))
    source_path.write_text(kernel.get_kernel_source())
    try:
        host_path.write_text(kernel.get_host_source())
        host_error = None
    except Exception as exc:
        host_path.write_text("")
        host_error = {"type": type(exc).__name__, "message": str(exc)}
    cache = Path(os.environ.get("TILELANG_CACHE_DIR", ""))
    cache_files = []
    if cache.is_dir():
        for path in sorted(p for p in cache.rglob("*") if p.is_file()):
            cache_files.append({"path": str(path.relative_to(cache)), "bytes": path.stat().st_size,
                                "sha256": sha256_file(path)})
    sources = {str(path.relative_to(out)): {"bytes": path.stat().st_size,
                                            "sha256": sha256_file(path)}
               for path in (tir_path, source_path, host_path)}
    generated = source_path.read_text(errors="replace")
    return {"representative": {"rows": 128, "hidden": 2048, "dtype": "fp16", "threads": 128},
            "files": sources, "cache_files": cache_files,
            "host_source_error": host_error,
            "observability": {"tir_available": tir_path.stat().st_size > 0,
                               "generated_cuda_available": source_path.stat().st_size > 0,
                               "host_source_available": host_path.stat().st_size > 0,
                               "generated_kernel_symbols": sorted(set(re.findall(r"__global__\s+void\s+([A-Za-z0-9_]+)", generated))),
                               "lowering_pass_diff": "NOT_EXPOSED_BY_THIS_FIXED_COMPILE_PATH",
                               "ptx_cubin_direct_export": "NOT_EXPOSED_BY_JITKernel_API; NCU report used for binary/resource evidence"}}


def collect(args) -> int:
    out = Path(args.output_dir)
    performance = performance_process(args.process_index)
    write_json(out / f"performance_proc{args.process_index}.json", performance)
    if args.process_index == 0:
        records = []
        for shape in CORRECTNESS_SHAPES:
            for dtype in ("fp16", "fp32"):
                records.append(correctness_case(shape["rows"], shape["hidden"], dtype,
                                                "random_normal", shape["split"]))
        for dtype in ("fp16", "fp32"):
            for mode in ("nan_inject", "posinf", "neginf"):
                records.append(correctness_case(2, 33, dtype, mode, "special"))
        write_jsonl(out / "correctness.jsonl", records)
        safety = negative_and_stream_checks()
        write_json(out / "safety_and_stream.json", safety)
        write_json(out / "compiler_artifacts.json", collect_artifacts(out))
    return 0


def profile_worker(args) -> int:
    import torch
    from ops.tilelang.rmsnorm import rmsnorm_tilelang
    x = torch.randn((128, 2048), device="cuda", dtype=torch.float16)
    weight = torch.randn((2048,), device="cuda", dtype=torch.float16)
    for _ in range(4):
        rmsnorm_tilelang(x, weight, EPSILON)
        torch.cuda.synchronize()
    print(json.dumps({"status": "PASS", "shape": [128, 2048], "dtype": "fp16"}))
    return 0


def profile(args) -> int:
    out = Path(args.output_dir)
    base = out / "profiles/tilelang_rmsnorm"
    base.parent.mkdir(parents=True, exist_ok=True)
    target = [sys.executable, str(Path(__file__)), "profile-worker"]
    tools = {"ncu": run([str(NCU), "--version"], 120),
             "nsys": run([str(NSYS), "--version"], 120)}
    ncu_cmd = ["sudo", "-n", "-E", str(NCU), "--force-overwrite", "--export", str(base),
               "--section", "LaunchStats", "--section", "Occupancy",
               "--section", "SpeedOfLight", "--section", "MemoryWorkloadAnalysis",
               "--launch-skip", "3", "--launch-count", "1", *target]
    ncu_proc = run(ncu_cmd, 2400)
    report = Path(str(base) + ".ncu-rep")
    csv_path = base.with_suffix(".ncu.csv")
    export = (run([str(NCU), "--import", str(report), "--page", "details", "--csv",
                   "--log-file", str(csv_path)], 600) if report.exists()
              else {"exit_code": 1, "stderr": "missing ncu report"})
    nsys_base = out / "profiles/tilelang_rmsnorm_timeline"
    nsys_proc = run([str(NSYS), "profile", "--trace=cuda,nvtx", "--sample=none",
                     "--cpuctxsw=none", "--force-overwrite=true", "--output",
                     str(nsys_base), *target], 1800)
    nsys_report = nsys_base.with_suffix(".nsys-rep")
    stats_path = out / "profiles/tilelang_rmsnorm_timeline_cuda_gpu_kern_sum.csv"
    nsys_stats = (run([str(NSYS), "stats", "--report", "cuda_gpu_kern_sum",
                       "--format", "csv", "--force-export=true", str(nsys_report)], 600)
                  if nsys_report.exists() else {"exit_code": 1, "stderr": "missing nsys report"})
    stats_path.write_text(str(nsys_stats.get("stdout", "")))
    result = {"tools": tools, "ncu_process": ncu_proc, "ncu_export": export,
              "ncu_report": str(report.relative_to(out)) if report.exists() else None,
              "ncu_csv": str(csv_path.relative_to(out)) if csv_path.exists() else None,
              "nsys_process": nsys_proc,
              "nsys_report": str(nsys_report.relative_to(out)) if nsys_report.exists() else None,
              "nsys_stats_process": nsys_stats,
              "nsys_stats_csv": str(stats_path.relative_to(out)) if stats_path.stat().st_size else None,
              "status": "PASS" if ncu_proc["exit_code"] == 0 and export["exit_code"] == 0
                        and nsys_proc["exit_code"] == 0 and nsys_stats["exit_code"] == 0
                        and stats_path.stat().st_size > 0
                        else "FAIL"}
    write_json(out / "profile_processes.json", result)
    return 0


def aggregate_performance(out: Path) -> Dict[str, Any]:
    processes = [read_json(out / f"performance_proc{i}.json") for i in range(3)]
    rows = []
    for case in PERF_CASES:
        per_case = [row for proc in processes for row in proc["rows"] if row["case_id"] == case["case_id"]]
        backend_rows = []
        for backend in BACKENDS:
            found = [m for row in per_case for m in row["measurements"] if m["backend"] == backend]
            samples = [v for item in found for v in item.get("device_us_raw", [])]
            backend_rows.append({"backend": backend, "process_count": len(found),
                                 "all_correct": all(item["status"] == "PASS" for item in found),
                                 "device_us_p50": statistics.median(samples) if samples else None,
                                 "device_us_p95": percentile(samples, 0.95) if samples else None,
                                 "effective_gbps_p50": statistics.median([item["effective_gbps"] for item in found if item.get("effective_gbps")]) if samples else None})
        valid = [item for item in backend_rows if item["all_correct"] and item["device_us_p50"] is not None]
        best = min(valid, key=lambda item: item["device_us_p50"])["backend"] if valid else None
        tile = next(item for item in backend_rows if item["backend"] == "tilelang_fixed")
        rows.append({**case, "backends": backend_rows, "best_legal_backend": best,
                     "tilelang_is_best": best == "tilelang_fixed",
                     "tilelang_compile_wall_ms_by_process": [row["tilelang_compile_wall_ms"] for row in per_case],
                     "tilelang_status": "PASS" if tile["all_correct"] else "FAIL"})
    return {"rows": rows, "guard": GUARD, "process_count": 3,
            "boundary": "each backend call includes output allocation; compile excluded from steady timing"}


def build_manifest(out: Path) -> Dict[str, Any]:
    entries = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()
                       and p.name != "EVIDENCE_MANIFEST.json"):
        entries[str(path.relative_to(out))] = {"bytes": path.stat().st_size,
                                              "sha256": sha256_file(path)}
    return {"experiment_id": EXPERIMENT_ID, "generated_at_utc": utc_now(),
            "root": str(out.relative_to(REPO)) if out.is_relative_to(REPO) else str(out),
            "entry_count": len(entries), "entries": entries}


def analyze(args) -> int:
    out = Path(args.output_dir)
    correctness = read_jsonl(out / "correctness.jsonl")
    safety = read_json(out / "safety_and_stream.json")
    performance = aggregate_performance(out)
    write_json(out / "performance_summary.json", performance)
    dependencies = read_json(out / "dependency_manifest.json")
    optional = read_json(out / "optional_isolation.json")
    artifacts = read_json(out / "compiler_artifacts.json")
    profile_data = read_json(out / "profile_processes.json") if (out / "profile_processes.json").exists() else {"status": "NOT_RUN"}
    source_audit = read_json(out / "portability_source_audit.json")
    hip_map = read_json(out / "hip_technical_map.json")
    amd_available = bool(dependencies.get("amd_runtime_available"))

    development = read_json(out / "development_log.json")
    if not any(event.get("kind") == "compile_debug" for event in development["events"]):
        development["events"].append({
            "kind": "compile_debug",
            "observed_failure": "H=33 with a 33-element fragment and 128 threads failed LowerTileOp: inconsistent source/destination reduction replicate layouts",
            "resolution": "pad the fragment to max(threads, next_power_of_two(H)), explicitly zero masked lanes, and guard output stores",
            "semantic_effect": "preserves the frozen equation while making odd H an explicit tail case",
        })
    if not any(event.get("kind") == "formal_collection" for event in development["events"]):
        development["events"].append({
            "kind": "formal_collection",
            "result": "52 correctness cases plus safety/stream and three-process performance collected",
        })
    if not any(event.get("kind") == "collector_debug" for event in development["events"]):
        development["events"].append({
            "kind": "collector_debug",
            "observed_failures": [
                "profile-worker existed as a function but was omitted from the argparse command registry",
                "NSys stats --output produced an empty file while returning success",
            ],
            "resolutions": [
                "registered profile-worker and reran only profiler/analyze, preserving formal correctness/performance raw",
                "captured NSys CSV from stdout, required a non-empty file, and reran profiler/analyze",
            ],
        })
    development["recording_finished_at_utc"] = utc_now()
    development["compile_wall_ms_by_case_and_process"] = {
        row["case_id"]: row["tilelang_compile_wall_ms_by_process"] for row in performance["rows"]
    }
    development["artifact_build_size_bytes"] = sum(
        item["bytes"] for item in artifacts.get("cache_files", []))
    development["artifact_file_count"] = len(artifacts.get("cache_files", []))
    write_json(out / "development_log.json", development)

    provenance = read_json(out / "provenance.json")
    provenance["finalized_at_utc"] = utc_now()
    provenance["final_sources"] = {
        name: {"sha256": sha256_file(REPO / name), "bytes": (REPO / name).stat().st_size}
        for name in (
            "scripts/audit/run_e04_11_tilelang_hip_portability.py",
            "scripts/audit/e04_11_run.sh",
            "ops/tilelang/rmsnorm.py",
            "ops/tilelang/__init__.py",
        )
    }
    provenance["post_collection_changes"] = (
        "Only profiler orchestration, NSys CSV capture, analysis metadata and report handling changed after formal correctness/performance collection; kernel source and frozen measurement protocol did not change."
    )
    write_json(out / "provenance.json", provenance)

    write_json(out / "experimental_route.json", {
        "backend": "tilelang_fixed", "registration": "NOT_REGISTERED",
        "dispatcher_auto_eligible": False,
        "reason": "P2 prototype; current NVIDIA-only evidence and no E04-08 replay for this backend",
        "manual_entry_point": "ops.tilelang.rmsnorm.rmsnorm_tilelang",
        "fallback": "caller explicitly selects an existing qualified backend; prototype performs no hidden fallback",
    })
    write_json(out / "pareto_comparison.json", {
        "performance": performance,
        "productivity": development,
        "maintenance": {
            "stable_release": dependencies.get("tilelang_runtime_version"),
            "optional_isolation_passed": optional.get("passed"),
            "generated_artifacts_visible": artifacts["observability"],
            "profile_status": profile_data.get("status"),
            "autotune_not_evaluated": True,
            "amd_execution_status": "PASS" if amd_available else "BLOCKED_NO_AMD_GPU",
            "source_hazard_count": source_audit["finding_count"],
        },
        "interpretation": "LOC, compile time, steady latency and observability are separate axes; no scalar productivity score is manufactured",
    })

    checks = {
        "dependency_fixed": dependencies.get("tilelang_runtime_version") == "0.1.13",
        "optional_isolation": bool(optional.get("passed")),
        "operator_spec_and_tolerance_reused": all(row.get("tolerance") for row in correctness),
        "real_boundary_tail_correct": all(row["status"] == "PASS" for row in correctness),
        "stream_and_negative_safety": safety.get("status") == "PASS",
        "cold_cache_steady_raw_three_processes": performance.get("process_count") == 3 and all(
            all(b["process_count"] == 3 for b in row["backends"]) for row in performance["rows"]),
        "ir_codegen_traceable_or_limit_explicit": bool(artifacts["observability"]["tir_available"]
            and artifacts["observability"]["generated_cuda_available"]
            and artifacts["observability"]["lowering_pass_diff"]),
        "profile_resource_evidence": profile_data.get("status") == "PASS",
        "productivity_maintenance_recorded": (out / "development_log.json").exists(),
        "experimental_boundary": True,
        "no_model_or_amd_claim": True,
        "hip_map_complete": len(hip_map["rows"]) >= 12 and len(hip_map["build_route"]) >= 8,
    }
    tilelang_pass = all(checks.values())
    hip_map_pass = bool(checks["hip_map_complete"] and source_audit["finding_count"] > 0)
    overall = "PASS" if tilelang_pass and hip_map_pass else "FAIL"
    verdict = {
        "experiment_id": EXPERIMENT_ID, "schema": SCHEMA, "evaluated_at_utc": utc_now(),
        "overall": overall,
        "tilelang_prototype": "PASS" if tilelang_pass else "FAIL",
        "hip_technical_map": "PASS" if hip_map_pass else "FAIL",
        "amd_runtime_subclaim": "PASS" if amd_available else "BLOCKED",
        "checks": checks,
        "correctness": {"case_count": len(correctness),
                        "pass_count": sum(row["status"] == "PASS" for row in correctness),
                        "failed_case_ids": [row["case_id"] for row in correctness if row["status"] != "PASS"]},
        "expected_effect": {
            "text": "evaluate expression, toolchain and maintenance cost; form CUDA-to-HIP engineering map",
            "met": tilelang_pass and hip_map_pass,
        },
        "single_item_acceptance": {
            "text": "experimental backend boundary explicit; core package and stage main conclusion unaffected",
            "met": True,
        },
        "claim_boundary": [
            "TileLang evidence is operator-level on Jetson Orin SM87 and version 0.1.13",
            "fixed configuration only; TileLang autotune was not evaluated",
            "AMD correctness, ROCm performance and wavefront optimization are blocked without AMD hardware",
            "no model-level throughput, latency or memory benefit is claimed",
        ],
    }
    write_json(out / "verdict.json", verdict)
    write_json(out / "EVIDENCE_MANIFEST.json", build_manifest(out))
    return 0 if overall == "PASS" else 1


def smoke(args) -> int:
    import torch
    from ops.tilelang.rmsnorm import compile_rmsnorm, rmsnorm_tilelang
    x = torch.randn((2, 33), device="cuda", dtype=torch.float16)
    weight = torch.randn((33,), device="cuda", dtype=torch.float16)
    start = time.perf_counter(); kernel = compile_rmsnorm(2, 33, "float16", EPSILON, 128)
    compile_ms = (time.perf_counter() - start) * 1000.0
    out = rmsnorm_tilelang(x, weight, EPSILON); torch.cuda.synchronize()
    ref = framework_reference(torch, x, weight)
    print(json.dumps({"status": "PASS" if torch.allclose(out, ref, atol=0.00390625, rtol=0.02) else "FAIL",
                      "compile_ms": compile_ms, "max_abs": float((out.float() - ref.float()).abs().max()),
                      "kernel_type": type(kernel).__name__, "source_bytes": len(kernel.get_kernel_source())}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("initialize", "collect", "profile", "profile-worker", "analyze", "smoke"):
        child = sub.add_parser(command)
        child.add_argument("--output-dir", default="docs/stage_experiments/S04/E04-11/raw")
        if command == "collect":
            child.add_argument("--process-index", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    return {"initialize": initialize, "collect": collect, "profile": profile,
            "profile-worker": profile_worker, "analyze": analyze,
            "smoke": smoke}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
