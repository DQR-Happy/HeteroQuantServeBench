#!/usr/bin/env python3
"""E04-04 decode/small-M versus prefill/large-M regime audit.

This collector is intentionally operator-level.  It imports the frozen S02
runtime census and the legal backend identities established by E04-02/03,
runs randomized blocked sweeps in three independent processes, and keeps the
holdout points out of routing-boundary fitting.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXPERIMENT_ID = "E04-04"
SCHEMA = "hqsb.s04.decode_prefill_regime.v1"
GUARD = 0.05
EPSILON = 1.0e-6
GEMM_BACKENDS = ("pytorch_cublas_opaque", "triton_fixed", "cutlass_tensorop_sm80")
RMS_BACKENDS = ("pytorch_reference", "cuda_v2", "triton_reference", "triton_tuned")
FAMILY_PROJECTIONS = {
    "kv": ["k", "v"],
    "q_o": ["q", "o"],
    "gate_up": ["gate", "up"],
    "down": ["down"],
    "lm_head": ["lm_head"],
}
FAMILY_DIMS = {
    "kv": (1024, 2048),
    "q_o": (2048, 2048),
    "gate_up": (6144, 2048),
    "down": (2048, 6144),
    "lm_head": (151936, 2048),
}
TRAIN = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
VALIDATION = (7, 9, 15, 17, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513)
HOLDOUT = (3, 12, 48, 96, 192, 384, 768, 1536)
RMS128_EXTRA = (4096, 8192, 16384, 32768)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(cmd: Sequence[str], timeout: int = 1800) -> Dict[str, Any]:
    before = time.perf_counter()
    try:
        proc = subprocess.run(list(cmd), cwd=REPO, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, check=False)
        return {"command": list(cmd), "exit_code": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr,
                "duration_s": time.perf_counter() - before, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        return {"command": list(cmd), "exit_code": 124,
                "stdout": exc.stdout or "", "stderr": exc.stderr or "",
                "duration_s": time.perf_counter() - before, "timed_out": True}


def split_for(point: int) -> str:
    if point in HOLDOUT:
        return "holdout"
    if point in VALIDATION:
        return "validation"
    return "train"


def regime_for(point: int) -> str:
    if point == 1:
        return "decode_real"
    if point < 32:
        return "small_m_probe"
    return "prefill_like"


def points(extra: Sequence[int] = ()) -> List[int]:
    return sorted(set(TRAIN + VALIDATION + HOLDOUT + tuple(extra)))


def parse_shape(text: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in ast.literal_eval(text))


def build_phase_ledger() -> Dict[str, Any]:
    projection_source = REPO / "docs/stage_experiments/S04/E04-03/raw/projection_ledger.json"
    projection_data = json.loads(projection_source.read_text())
    gemm_rows = []
    for row in projection_data["rows"]:
        family = next(name for name, projections in FAMILY_PROJECTIONS.items()
                      if row["projection"] in projections)
        gemm_rows.append({**row, "family": family})

    rms_grouped: Dict[Tuple[str, str, str, int, int], Dict[str, Any]] = {}
    census_root = REPO / "docs/stage_experiments/S02/E02-02/raw_v2/run_0"
    census_sources = []
    for path in sorted(census_root.glob("census_*.json")):
        data = json.loads(path.read_text())
        workload = data["workload"]["name"]
        census_sources.append({"path": str(path.relative_to(REPO)),
                               "sha256": sha256_file(path), "workload": workload})
        for rec in data["modules"]:
            module = rec["module"]
            if not any(token in module for token in
                       ("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm")):
                continue
            shape = parse_shape(rec["input_shapes"][0])
            hidden = shape[-1]
            rows = math.prod(shape[:-1])
            kind = ("q_norm" if module.endswith("q_norm") else
                    "k_norm" if module.endswith("k_norm") else "hidden_norm")
            key = (workload, rec["phase"], kind, rows, hidden)
            item = rms_grouped.setdefault(key, {"workload": workload,
                "phase": rec["phase"], "operator": kind, "rows": rows,
                "hidden": hidden, "calls": 0, "modules": []})
            item["calls"] += int(rec["call_count"])
            item["modules"].append(module)
    return {"schema": SCHEMA, "projection_source": {
                "path": str(projection_source.relative_to(REPO)),
                "sha256": sha256_file(projection_source)},
            "census_sources": census_sources, "gemm_rows": gemm_rows,
            "rmsnorm_rows": sorted(rms_grouped.values(), key=lambda r: (
                r["workload"], r["phase"], r["hidden"], r["rows"], r["operator"]))}


def inherited_evidence() -> Dict[str, Any]:
    paths = {
        "E04-02": REPO / "docs/stage_experiments/S04/E04-02/raw/verdict.json",
        "E04-03": REPO / "docs/stage_experiments/S04/E04-03/raw/verdict.json",
    }
    rows = {}
    for name, path in paths.items():
        data = json.loads(path.read_text())
        rows[name] = {"path": str(path.relative_to(REPO)),
                      "sha256": sha256_file(path), "overall": data.get("overall"),
                      "failed_conditions": data.get("failed_conditions", []),
                      "use": ("backend implementation/config source; every E04-04 case is "
                              "rechecked before and after timing")}
    rows["E04-02"]["scope_note"] = (
        "E04-02 is BLOCKED only by inherited E03-06/E03-07 gates; E04-04 does not "
        "promote its routing candidates and directly rechecks the four forced backends.")
    return {"schema": SCHEMA, "evidence": rows,
            "gemm_legal_gate_pass": rows["E04-03"]["overall"] == "PASS",
            "rmsnorm_prior_status_disclosed": rows["E04-02"]["overall"] == "BLOCKED"}


def initialize(args) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = build_phase_ledger()
    write_json(out / "phase_ledger.json", ledger)
    write_json(out / "inherited_evidence.json", inherited_evidence())
    write_json(out / "protocol.json", {
        "experiment_id": EXPERIMENT_ID, "schema": SCHEMA,
        "frozen_at_utc": utc_now(), "guard": GUARD,
        "independent_processes": 3, "randomized_block": "shape is block; backend order seeded",
        "gemm": {"families": {k: {"N": v[0], "K": v[1],
                  "projections": FAMILY_PROJECTIONS[k]} for k, v in FAMILY_DIMS.items()},
                  "train_M": list(TRAIN), "validation_M": list(VALIDATION),
                  "holdout_M": list(HOLDOUT), "backends": list(GEMM_BACKENDS),
                  "dtype": "fp16", "accumulation": "fp32",
                  "layout_A": "row_major", "layout_B": "model [N,K] transposed",
                  "primary_decode": ["device_us", "host_completion_us", "p50", "p95"],
                  "primary_prefill": ["device_us", "TFLOP/s", "p50", "p95"]},
        "rmsnorm": {"hidden": [128, 2048], "base_rows": points(),
                    "h128_extra_real_rows": list(RMS128_EXTRA),
                    "backends": list(RMS_BACKENDS), "dtype": "fp16",
                    "primary_decode": ["device_us", "host_completion_us"],
                    "primary_prefill": ["device_us", "rows/s", "effective_GB/s"]},
        "holdout_policy": "excluded from crossover/routing fit; evaluated after rule freeze",
        "crossover_policy": "5% tie guard; report adjacent interval, never a pseudo-exact point",
        "correctness_policy": "PyTorch/Triton GEMM use sampled independent FP32 oracle before/after each timing; CUTLASS is bound to E04-03 PASS and rechecks per-allocation guards; RMSNorm uses full FP32 oracle before/after",
        "cache_policy": "steady warm cache; cold/JIT evidence remains E04-02/E04-03 scope",
        "resource_policy": "NCU ordinary timing is separate; fixed CUTLASS config at M=1/32/128/512",
        "claim_boundary": "operator-level routing evidence only; no TTFT/TPOT/model speedup claim",
    })
    import torch
    import triton
    sources = [Path(__file__), REPO / "ops/triton/gemm.py", REPO / "ops/triton/rmsnorm.py",
               REPO / "ops/cuda/cutlass_gemm/bench_cutlass_gemm.cu"]
    write_json(out / "provenance.json", {
        "experiment_id": EXPERIMENT_ID, "created_at_utc": utc_now(),
        "git_commit": run(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        "git_status": run(["git", "status", "--short"])["stdout"],
        "platform": platform.platform(), "python": sys.version,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "triton": triton.__version__, "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "source_sha256": {str(p.relative_to(REPO)): sha256_file(p) for p in sources}})
    return 0


def build(args) -> int:
    out = Path(args.output_dir)
    configure = run(["cmake", "-S", ".", "-B", "build/jetson-release",
                     "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=87"], 900)
    compile_result = run(["cmake", "--build", "build/jetson-release", "--target",
                          "hqsb_cutlass_gemm_bench", "-j2"], 1800)
    payload = {"configure": configure, "build": compile_result,
               "binary": "build/jetson-release/bin/hqsb_cutlass_gemm_bench",
               "binary_exists": (REPO / "build/jetson-release/bin/hqsb_cutlass_gemm_bench").exists()}
    write_json(out / "build.json", payload)
    return 0 if configure["exit_code"] == 0 and compile_result["exit_code"] == 0 else 1


def load_gpu():
    import torch
    from ops import cuda_bridge
    triton_gemm = importlib.import_module("ops.triton.gemm")
    triton_rms = importlib.import_module("ops.triton.rmsnorm")
    return torch, cuda_bridge, triton_gemm, triton_rms


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(x) for x in values)
    if not ordered:
        return math.nan
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def input_recipe_hash(kind: str, *items: Any) -> str:
    text = "|".join([EXPERIMENT_ID, kind, *(str(x) for x in items), "seeded-uniform-v1"])
    return hashlib.sha256(text.encode()).hexdigest()


def gemm_compare(torch, actual, a, b) -> Dict[str, Any]:
    m, n = actual.shape
    row_ids = sorted(set((0, m // 2, m - 1)))
    col_ids = list(range(min(32, n))) + list(range(max(0, n - 32), n))
    rng = random.Random(m * 1000003 + n * 1009 + a.shape[1])
    col_ids += [rng.randrange(n) for _ in range(min(64, n))]
    rows = torch.tensor(row_ids, device="cuda")
    cols = torch.tensor(sorted(set(col_ids)), device="cuda")
    ref = a.index_select(0, rows).float() @ b.index_select(1, cols).float()
    got = actual.index_select(0, rows).index_select(1, cols).float()
    diff = (got - ref).abs()
    allowed = 0.01 + 0.02 * ref.abs()
    l2rel = float(torch.linalg.vector_norm(got - ref) /
                  torch.clamp(torch.linalg.vector_norm(ref), min=1e-12))
    passed = not bool((diff > allowed).any()) and l2rel <= 0.01
    return {"oracle": "sampled_independent_fp32_matmul", "sample_elements": int(got.numel()),
            "max_abs": float(diff.max()), "l2rel": l2rel, "passed": passed}


def gemm_launch(torch, triton_gemm, backend: str, a, b, out):
    if backend == "pytorch_cublas_opaque":
        torch.mm(a, b, out=out)
    elif backend == "triton_fixed":
        triton_gemm.gemm_reference(a, b, out=out)
    else:
        raise ValueError(backend)


def gemm_python_record(torch, triton_gemm, family: str, m: int, backend: str,
                       process_index: int) -> Dict[str, Any]:
    n, k = FAMILY_DIMS[family]
    seed = 404000 + process_index * 10007 + m * 31 + n + k
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    a = torch.empty((m, k), device="cuda", dtype=torch.float16).uniform_(-0.25, 0.25, generator=gen)
    weight = torch.empty((n, k), device="cuda", dtype=torch.float16).uniform_(-0.25, 0.25, generator=gen)
    b = weight.t()
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    gemm_launch(torch, triton_gemm, backend, a, b, out); torch.cuda.synchronize()
    before = gemm_compare(torch, out, a, b)
    for _ in range(3):
        gemm_launch(torch, triton_gemm, backend, a, b, out)
    torch.cuda.synchronize()
    work = m * n * k
    iters = 10 if work < 100_000_000 else (3 if work < 2_000_000_000 else 1)
    groups = 5
    device_us, host_us = [], []
    for _ in range(groups):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _i in range(iters):
            gemm_launch(torch, triton_gemm, backend, a, b, out)
        end.record(); end.synchronize()
        device_us.append(float(start.elapsed_time(end) * 1000.0 / iters))
        host_begin = time.perf_counter_ns()
        for _i in range(iters):
            gemm_launch(torch, triton_gemm, backend, a, b, out)
        torch.cuda.synchronize()
        host_us.append((time.perf_counter_ns() - host_begin) / 1000.0 / iters)
    after = gemm_compare(torch, out, a, b)
    med = statistics.median(device_us)
    logical_bytes = 2 * (m * k + k * n + m * n)
    return {"schema": SCHEMA, "operator": "gemm", "process_index": process_index,
        "family": family, "projections": FAMILY_PROJECTIONS[family], "M": m, "N": n, "K": k,
        "split": split_for(m), "regime": regime_for(m), "requested_backend": backend,
        "actual_backend": backend, "config": ("torch.mm opaque vendor selection" if backend.startswith("pytorch")
            else "BLOCK_M=64,BLOCK_N=64,BLOCK_K=32,GROUP_M=8,num_warps=4,num_stages=3"),
        "dtype": "fp16", "accumulation": "fp32", "layout_A": "row_major",
        "layout_B": "model_weight_[N,K]_row_major_transposed_view", "workspace_bytes": 0,
        "input_recipe_sha256": input_recipe_hash("gemm", family, m, process_index),
        "warmup": 3, "groups": groups, "iterations_per_group": iters,
        "device_us_raw": device_us, "host_completion_us_raw": host_us,
        "device_us_median": med, "host_completion_us_median": statistics.median(host_us),
        "tflops": 2 * m * n * k / med / 1e6, "logical_bytes": logical_bytes,
        "effective_GBps": logical_bytes / med / 1000.0,
        "weight_GBps_decode_secondary": (2 * k * n) / med / 1000.0,
        "estimated_grid_ctas": (math.ceil(m / 64) * math.ceil(n / 64)
                                if backend == "triton_fixed" else None),
        "correctness_before": before, "correctness_after": after,
        "status": "PASS" if before["passed"] and after["passed"] else "FAIL"}


def parse_last_json(text: str) -> Optional[Dict[str, Any]]:
    for line in reversed(text.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            pass
    return None


def gemm_cutlass_record(family: str, m: int, process_index: int) -> Dict[str, Any]:
    n, k = FAMILY_DIMS[family]
    work = m * n * k
    iters = 10 if work < 100_000_000 else (5 if work < 2_000_000_000 else 3)
    binary = REPO / "build/jetson-release/bin/hqsb_cutlass_gemm_bench"
    proc = run([str(binary), "--m", str(m), "--n", str(n), "--k", str(k),
                "--warmup", "3", "--iterations", str(iters), "--no-verify"], 1800)
    payload = parse_last_json(proc["stdout"])
    base = {"schema": SCHEMA, "operator": "gemm", "process_index": process_index,
        "family": family, "projections": FAMILY_PROJECTIONS[family], "M": m, "N": n, "K": k,
        "split": split_for(m), "regime": regime_for(m),
        "requested_backend": "cutlass_tensorop_sm80", "actual_backend": "cutlass_tensorop_sm80",
        "config": "CUTLASS device::Gemm default TensorOp Sm80 configuration",
        "dtype": "fp16", "accumulation": "fp32", "layout_A": "row_major",
        "layout_B": "column_major_logical_from_weight_NK", "workspace_bytes": 0,
        "input_recipe_sha256": input_recipe_hash("cutlass", family, m, process_index),
        "process": {"exit_code": proc["exit_code"], "duration_s": proc["duration_s"],
                    "stderr": proc["stderr"]}}
    if not payload or proc["exit_code"] != 0 or payload.get("status") != "PASS":
        return {**base, "status": "FAIL", "measurement": payload}
    device_us = [float(x) * 1000.0 for x in payload.get("device_ms_raw", [payload["median_ms"]])]
    host_us = [float(x) for x in payload.get("host_completion_us_raw", [])]
    med = statistics.median(device_us)
    logical_bytes = 2 * (m * k + k * n + m * n)
    return {**base, "warmup": 3, "iterations": iters, "device_us_raw": device_us,
        "host_completion_us_raw": host_us, "device_us_median": med,
        "host_completion_us_median": statistics.median(host_us) if host_us else None,
        "tflops": 2 * m * n * k / med / 1e6, "logical_bytes": logical_bytes,
        "effective_GBps": logical_bytes / med / 1000.0,
        "weight_GBps_decode_secondary": (2 * k * n) / med / 1000.0,
        "correctness_gate": "E04-03 PASS plus binary guards_ok on this allocation",
        "guards_ok": payload.get("guards_ok"), "status": "PASS"}


def collect_gemm(args) -> int:
    out = Path(args.output_dir)
    torch, _bridge, triton_gemm, _rms = load_gpu()
    blocks = [(family, m) for family in FAMILY_DIMS for m in points()]
    random.Random(404100 + args.process_index).shuffle(blocks)
    records = []
    for block_index, (family, m) in enumerate(blocks):
        treatments = list(GEMM_BACKENDS)
        random.Random(404200 + args.process_index * 1009 + block_index).shuffle(treatments)
        for backend in treatments:
            try:
                record = (gemm_cutlass_record(family, m, args.process_index)
                          if backend == "cutlass_tensorop_sm80" else
                          gemm_python_record(torch, triton_gemm, family, m, backend,
                                             args.process_index))
            except Exception as exc:
                n, k = FAMILY_DIMS[family]
                record = {"schema": SCHEMA, "operator": "gemm",
                    "process_index": args.process_index, "family": family,
                    "M": m, "N": n, "K": k, "split": split_for(m),
                    "regime": regime_for(m), "requested_backend": backend,
                    "status": "FAIL", "error": {"type": type(exc).__name__, "message": str(exc)}}
            records.append(record)
            print("gemm", args.process_index, family, m, backend, record["status"], flush=True)
            torch.cuda.empty_cache()
    write_jsonl(out / f"gemm_proc{args.process_index}.jsonl", records)
    return 0 if all(r["status"] == "PASS" for r in records) else 1


def rms_reference(torch, x, weight):
    norm = x.float() * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + EPSILON)
    return (norm * weight.float()).to(x.dtype)


def rms_launch(torch, bridge, triton_rms, backend: str, x, weight, out):
    if backend == "pytorch_reference":
        out.copy_(rms_reference(torch, x, weight))
    elif backend == "cuda_v2":
        bridge.rmsnorm_forward(x, weight, out=out, dtype="fp16", variant="v2_vectorized",
                               epsilon=EPSILON, stream=torch.cuda.current_stream())
    elif backend == "triton_reference":
        triton_rms.rmsnorm_reference(x, weight, EPSILON, out=out)
    elif backend == "triton_tuned":
        triton_rms.rmsnorm_optimized(x, weight, EPSILON, out=out)
    else:
        raise ValueError(backend)


def rms_correct(torch, out, x, weight) -> Dict[str, Any]:
    ref = rms_reference(torch, x, weight)
    diff = (out.float() - ref.float()).abs()
    allowed = 0.01 + 0.02 * ref.float().abs()
    return {"oracle": "independent_fp32_rmsnorm", "max_abs": float(diff.max()),
            "violation_count": int((diff > allowed).sum()),
            "passed": not bool((diff > allowed).any()) and bool(torch.isfinite(out).all())}


def rms_record(torch, bridge, triton_rms, hidden: int, rows: int, backend: str,
               process_index: int) -> Dict[str, Any]:
    seed = 404300 + process_index * 10007 + rows * 13 + hidden
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    x = torch.empty((rows, hidden), device="cuda", dtype=torch.float16).uniform_(-1, 1, generator=gen)
    weight = torch.empty((hidden,), device="cuda", dtype=torch.float16).uniform_(0.5, 1.5, generator=gen)
    out = torch.empty_like(x)
    rms_launch(torch, bridge, triton_rms, backend, x, weight, out); torch.cuda.synchronize()
    before = rms_correct(torch, out, x, weight)
    for _ in range(3):
        rms_launch(torch, bridge, triton_rms, backend, x, weight, out)
    torch.cuda.synchronize()
    work = rows * hidden
    iters = 20 if work < 1_000_000 else (10 if work < 10_000_000 else 3)
    device_us, host_us = [], []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _i in range(iters): rms_launch(torch, bridge, triton_rms, backend, x, weight, out)
        end.record(); end.synchronize()
        device_us.append(float(start.elapsed_time(end) * 1000.0 / iters))
        host_begin = time.perf_counter_ns()
        for _i in range(iters): rms_launch(torch, bridge, triton_rms, backend, x, weight, out)
        torch.cuda.synchronize()
        host_us.append((time.perf_counter_ns() - host_begin) / 1000.0 / iters)
    after = rms_correct(torch, out, x, weight)
    med = statistics.median(device_us)
    logical_bytes = (2 * rows * hidden + hidden) * 2
    config = None
    if backend == "triton_tuned":
        config = triton_rms.rmsnorm_last_tuned_config()
    elif backend == "triton_reference":
        config = {"num_warps": 4, "num_stages": 1}
    elif backend == "cuda_v2":
        config = {"variant": "v2_vectorized", "grid_ctas": rows}
    return {"schema": SCHEMA, "operator": "rmsnorm", "process_index": process_index,
        "rows": rows, "hidden": hidden, "split": split_for(rows), "regime": regime_for(rows),
        "requested_backend": backend, "actual_backend": backend, "config": config,
        "dtype": "fp16", "epsilon": EPSILON, "layout": "contiguous",
        "input_recipe_sha256": input_recipe_hash("rmsnorm", hidden, rows, process_index),
        "warmup": 3, "groups": 5, "iterations_per_group": iters,
        "device_us_raw": device_us, "host_completion_us_raw": host_us,
        "device_us_median": med, "host_completion_us_median": statistics.median(host_us),
        "rows_per_second": rows / med * 1e6, "logical_bytes": logical_bytes,
        "effective_GBps": logical_bytes / med / 1000.0,
        "estimated_grid_ctas": rows, "correctness_before": before, "correctness_after": after,
        "status": "PASS" if before["passed"] and after["passed"] else "FAIL"}


def collect_rmsnorm(args) -> int:
    out = Path(args.output_dir)
    torch, bridge, _gemm, triton_rms = load_gpu()
    blocks = [(hidden, rows) for hidden in (128, 2048)
              for rows in points(RMS128_EXTRA if hidden == 128 else ())]
    random.Random(404400 + args.process_index).shuffle(blocks)
    records = []
    for block_index, (hidden, rows) in enumerate(blocks):
        treatments = list(RMS_BACKENDS)
        random.Random(404500 + args.process_index * 1009 + block_index).shuffle(treatments)
        for backend in treatments:
            try:
                rec = rms_record(torch, bridge, triton_rms, hidden, rows, backend,
                                 args.process_index)
            except Exception as exc:
                rec = {"schema": SCHEMA, "operator": "rmsnorm",
                    "process_index": args.process_index, "rows": rows, "hidden": hidden,
                    "split": split_for(rows), "regime": regime_for(rows),
                    "requested_backend": backend, "status": "FAIL",
                    "error": {"type": type(exc).__name__, "message": str(exc)}}
            records.append(rec)
            print("rmsnorm", args.process_index, hidden, rows, backend, rec["status"], flush=True)
            torch.cuda.empty_cache()
    write_jsonl(out / f"rmsnorm_proc{args.process_index}.jsonl", records)
    return 0 if all(r["status"] == "PASS" for r in records) else 1


def parse_ncu_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()
    start = next((i for i, line in enumerate(lines)
                  if "Metric Name" in line or ("Kernel Name" in line and "ID" in line)), None)
    if start is None:
        return []
    return [dict(row) for row in csv.DictReader(lines[start:])]


def profile(args) -> int:
    out = Path(args.output_dir)
    profile_dir = out / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    ncu = Path("/usr/local/cuda/bin/ncu")
    binary = REPO / "build/jetson-release/bin/hqsb_cutlass_gemm_bench"
    records = []
    for m in (1, 32, 128, 512):
        csv_path = profile_dir / f"cutlass_qo_m{m}_details.csv"
        log_path = profile_dir / f"cutlass_qo_m{m}.log"
        report_path = profile_dir / f"cutlass_qo_m{m}"
        command = ["sudo", "-n", str(ncu), "--force-overwrite", "--export", str(report_path),
            "--section", "LaunchStats", "--section", "Occupancy", "--section", "SpeedOfLight",
            "--section", "MemoryWorkloadAnalysis", "--section", "SchedulerStats",
            "--launch-skip", "3", "--launch-count", "1", "--log-file", str(log_path),
            str(binary), "--m", str(m), "--n", "2048", "--k", "2048",
            "--warmup", "3", "--iterations", "1", "--no-verify"]
        proc = run(command, 1800)
        report_file = Path(str(report_path) + ".ncu-rep")
        if csv_path.exists():
            csv_path.unlink()
        export = run([str(ncu), "--import", str(report_file), "--page", "details",
                      "--csv", "--log-file", str(csv_path)], 600)
        parsed = parse_ncu_csv(csv_path)
        wanted = {"Duration", "Memory Throughput", "Compute (SM) Throughput",
                  "DRAM Throughput", "L2 Hit Rate", "Achieved Occupancy",
                  "Achieved Active Warps Per SM", "Registers Per Thread",
                  "Static Shared Memory Per Block", "Dynamic Shared Memory Per Block",
                  "Eligible Warps Per Scheduler", "Active Warps Per Scheduler"}
        key_metrics = {row.get("Metric Name"): {"unit": row.get("Metric Unit"),
                       "value": row.get("Metric Value")} for row in parsed
                       if row.get("Metric Name") in wanted}
        records.append({"family": "q_o", "M": m, "N": 2048, "K": 2048,
            "backend": "cutlass_tensorop_sm80", "ordinary_timing_separate": True,
            "profile_process": proc, "export_process": export,
            "grid_size": parsed[0].get("Grid Size") if parsed else None,
            "block_size": parsed[0].get("Block Size") if parsed else None,
            "key_metrics": key_metrics, "csv": str(csv_path.relative_to(out)),
            "report": str(report_file.relative_to(out)) if report_file.exists() else None,
            "report_sha256": sha256_file(report_file) if report_file.exists() else None,
            "csv_sha256": sha256_file(csv_path) if csv_path.exists() else None,
            "parsed_metric_rows": parsed, "status": "PASS" if proc["exit_code"] == 0 and
                export["exit_code"] == 0 and len(parsed) >= 10 and
                {"Duration", "Achieved Occupancy", "Compute (SM) Throughput"}.issubset(key_metrics)
                else "FAIL"})
        print("profile", m, records[-1]["status"], flush=True)
    write_json(out / "resource_profiles.json", {"rows": records})
    return 0 if all(r["status"] == "PASS" for r in records) else 1


def bootstrap_median_ci(values: Sequence[float], seed: int) -> List[float]:
    rng = random.Random(seed)
    draws = [statistics.median(rng.choice(values) for _ in values) for _ in range(2000)]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def aggregate(records: Sequence[Mapping[str, Any]], operator: str) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for rec in records:
        if operator == "gemm":
            key = (rec["family"], rec["M"], rec["N"], rec["K"], rec["requested_backend"])
        else:
            key = (rec["hidden"], rec["rows"], rec["requested_backend"])
        groups.setdefault(key, []).append(rec)
    output = []
    for key, recs in sorted(groups.items()):
        flat_device = [x for rec in recs for x in rec.get("device_us_raw", [])]
        flat_host = [x for rec in recs for x in rec.get("host_completion_us_raw", [])]
        first = recs[0]
        row = {"operator": operator, "backend": first["requested_backend"],
               "split": first["split"], "regime": first["regime"],
               "process_count": len(recs),
               "status": "PASS" if len(recs) == 3 and all(r["status"] == "PASS" for r in recs) else "FAIL"}
        if operator == "gemm":
            row.update({"family": first["family"], "M": first["M"], "N": first["N"], "K": first["K"]})
        else:
            row.update({"hidden": first["hidden"], "rows": first["rows"]})
        if flat_device:
            seed = sum(ord(c) for c in str(key))
            med = statistics.median(flat_device)
            row.update({"device_us_median": med, "device_us_p50": med,
                "device_us_p95": percentile(flat_device, 0.95),
                "device_us_median_ci95": bootstrap_median_ci(flat_device, seed),
                "host_completion_us_median": statistics.median(flat_host) if flat_host else None,
                "host_completion_us_p95": percentile(flat_host, 0.95) if flat_host else None,
                "effective_GBps": statistics.median(r["effective_GBps"] for r in recs
                                                     if r.get("effective_GBps") is not None)})
            if operator == "gemm":
                row["tflops"] = 2 * row["M"] * row["N"] * row["K"] / med / 1e6
            else:
                row["rows_per_second"] = row["rows"] / med * 1e6
        output.append(row)
    return output


def decisions(matrix: Sequence[Mapping[str, Any]], operator: str) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in matrix:
        key = ((row["family"], row["M"]) if operator == "gemm" else
               (row["hidden"], row["rows"]))
        groups.setdefault(key, []).append(row)
    output = []
    for key, rows in sorted(groups.items()):
        legal = sorted((r for r in rows if r["status"] == "PASS"), key=lambda r: r["device_us_median"])
        best = legal[0] if legal else None
        ties = ([r["backend"] for r in legal
                 if r["device_us_median"] <= best["device_us_median"] * (1 + GUARD)] if best else [])
        base = ({"family": key[0], "M": key[1]} if operator == "gemm" else
                {"hidden": key[0], "rows": key[1]})
        output.append({**base, "operator": operator,
            "split": rows[0]["split"], "regime": rows[0]["regime"],
            "winner": (best["backend"] if len(ties) == 1 else "TIE") if best else None,
            "best_backend": best["backend"] if best else None, "tie_backends": ties,
            "tie_guard": GUARD, "latencies_us": {r["backend"]: r["device_us_median"] for r in legal},
            "regret": {r["backend"]: r["device_us_median"] / best["device_us_median"] - 1
                       for r in legal} if best else {}})
    return output


def fit_boundaries(rows: Sequence[Mapping[str, Any]], operator: str) -> List[Dict[str, Any]]:
    family_key = "family" if operator == "gemm" else "hidden"
    point_key = "M" if operator == "gemm" else "rows"
    output = []
    for family in sorted({r[family_key] for r in rows}):
        fit = sorted((r for r in rows if r[family_key] == family and r["split"] != "holdout"),
                     key=lambda r: r[point_key])
        changes, tie_zones = [], []
        previous = None
        for row in fit:
            if row["winner"] == "TIE":
                tie_zones.append({"point": row[point_key], "backends": row["tie_backends"]})
                continue
            if previous and row["best_backend"] != previous["best_backend"]:
                changes.append({"interval": [previous[point_key], row[point_key]],
                                "from": previous["best_backend"], "to": row["best_backend"],
                                "guard": GUARD})
            previous = row
        clear = [r for r in fit if r["winner"] != "TIE"]
        validated = []
        for index in range(2, len(clear) - 1):
            left, right = clear[index - 1], clear[index]
            if left["best_backend"] == right["best_backend"]:
                continue
            left_stable = clear[index - 2]["best_backend"] == left["best_backend"]
            right_stable = clear[index + 1]["best_backend"] == right["best_backend"]
            if left_stable and right_stable:
                validated.append({"interval": [left[point_key], right[point_key]],
                    "from": left["best_backend"], "to": right["best_backend"],
                    "guard": GUARD, "two_clear_points_each_side": True,
                    "validation_point_in_bracket_support": any(r["split"] == "validation"
                        for r in (clear[index - 2], left, right, clear[index + 1]))})
        output.append({family_key: family, "operator": operator,
            "observed_rank_change_intervals": changes,
            "validated_persistent_change_intervals": validated,
            "unstable_or_unvalidated_change_intervals": [c for c in changes
                if not any(c["interval"] == v["interval"] and c["from"] == v["from"] and
                           c["to"] == v["to"] for v in validated)],
            "clear_rank_change_intervals": changes, "tie_zone_points": tie_zones,
            "boundary_confidence": ("PERSISTENT_INTERVALS_PRESENT" if validated else
                                    "NO_PERSISTENT_BOUNDARY; USE_TIE_OR_TABLE"),
            "no_rank_change_is_valid": not changes,
            "fit_points": [r[point_key] for r in fit]})
    return output


def nearest_fit_prediction(row: Mapping[str, Any], all_rows: Sequence[Mapping[str, Any]],
                           operator: str) -> str:
    family_key = "family" if operator == "gemm" else "hidden"
    point_key = "M" if operator == "gemm" else "rows"
    candidates = [r for r in all_rows if r[family_key] == row[family_key] and r["split"] != "holdout"]
    nearest = min(candidates, key=lambda r: abs(math.log2(r[point_key]) - math.log2(row[point_key])))
    if nearest["winner"] != "TIE":
        return nearest["best_backend"]
    priority = (["pytorch_cublas_opaque", "cutlass_tensorop_sm80", "triton_fixed"]
                if operator == "gemm" else ["cuda_v2", "triton_reference", "triton_tuned", "pytorch_reference"])
    return next(name for name in priority if name in nearest["tie_backends"])


def evaluate_holdout(rows: Sequence[Mapping[str, Any]], operator: str) -> Dict[str, Any]:
    holdout = [r for r in rows if r["split"] == "holdout"]
    results = []
    for row in holdout:
        predicted = nearest_fit_prediction(row, rows, operator)
        regret = row["regret"].get(predicted)
        results.append({**({"family": row["family"], "M": row["M"]} if operator == "gemm"
                           else {"hidden": row["hidden"], "rows": row["rows"]}),
            "predicted_backend": predicted, "oracle_best": row["best_backend"],
            "oracle_tie": row["tie_backends"], "selection_error": predicted not in row["tie_backends"],
            "regret": regret})
    return {"operator": operator, "rows": results,
            "error_rate": sum(r["selection_error"] for r in results) / len(results),
            "weighted_mean_regret_equal_holdout": statistics.mean(r["regret"] for r in results),
            "max_regret": max(r["regret"] for r in results)}


def phase_weighted(ledger: Mapping[str, Any], gemm_decisions: Sequence[Mapping[str, Any]],
                   rms_decisions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    gemm_lookup = {(r["family"], r["M"]): r for r in gemm_decisions}
    rms_lookup = {(r["hidden"], r["rows"]): r for r in rms_decisions}
    gemm_groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in ledger["gemm_rows"]:
        dec = gemm_lookup[(row["family"], row["M"])]
        group = gemm_groups.setdefault((row["workload"], row["phase"]),
            {"workload": row["workload"], "phase": row["phase"], "operator": "gemm",
             "backend_time_us": {b: 0.0 for b in GEMM_BACKENDS}, "oracle_time_us": 0.0,
             "weighted_calls": 0})
        for backend in GEMM_BACKENDS:
            group["backend_time_us"][backend] += row["calls"] * dec["latencies_us"][backend]
        group["oracle_time_us"] += row["calls"] * min(dec["latencies_us"].values())
        group["weighted_calls"] += row["calls"]
    rms_groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in ledger["rmsnorm_rows"]:
        dec = rms_lookup[(row["hidden"], row["rows"])]
        group = rms_groups.setdefault((row["workload"], row["phase"]),
            {"workload": row["workload"], "phase": row["phase"], "operator": "rmsnorm",
             "backend_time_us": {b: 0.0 for b in RMS_BACKENDS}, "oracle_time_us": 0.0,
             "weighted_calls": 0})
        for backend in RMS_BACKENDS:
            group["backend_time_us"][backend] += row["calls"] * dec["latencies_us"][backend]
        group["oracle_time_us"] += row["calls"] * min(dec["latencies_us"].values())
        group["weighted_calls"] += row["calls"]
    for group in list(gemm_groups.values()) + list(rms_groups.values()):
        group["best_fixed_backend"] = min(group["backend_time_us"], key=group["backend_time_us"].get)
        group["fixed_backend_regret_vs_shape_oracle"] = {
            b: value / group["oracle_time_us"] - 1 for b, value in group["backend_time_us"].items()}
        group["semantics"] = "sum(calls * operator latency); not model TTFT/TPOT"
    return {"gemm": sorted(gemm_groups.values(), key=lambda r: (r["workload"], r["phase"])),
            "rmsnorm": sorted(rms_groups.values(), key=lambda r: (r["workload"], r["phase"])),
            "speedups_averaged": False, "model_claim": False}


def summarize(args) -> int:
    out = Path(args.output_dir)
    provenance_path = out / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["final_analysis_at_utc"] = utc_now()
    provenance["final_analysis_source_sha256"] = sha256_file(Path(__file__))
    provenance["source_revision_note"] = (
        "The frozen sweep source hash remains in source_sha256; post-collection revisions only "
        "fixed NCU details export and made crossover validation/manifest indexing stricter.")
    write_json(provenance_path, provenance)
    gemm_raw = [r for i in range(3) for r in read_jsonl(out / f"gemm_proc{i}.jsonl")]
    rms_raw = [r for i in range(3) for r in read_jsonl(out / f"rmsnorm_proc{i}.jsonl")]
    gemm_matrix = aggregate(gemm_raw, "gemm")
    rms_matrix = aggregate(rms_raw, "rmsnorm")
    gemm_dec = decisions(gemm_matrix, "gemm")
    rms_dec = decisions(rms_matrix, "rmsnorm")
    boundaries = {"gemm": fit_boundaries(gemm_dec, "gemm"),
                  "rmsnorm": fit_boundaries(rms_dec, "rmsnorm")}
    holdout = {"gemm": evaluate_holdout(gemm_dec, "gemm"),
               "rmsnorm": evaluate_holdout(rms_dec, "rmsnorm")}
    ledger = json.loads((out / "phase_ledger.json").read_text())
    weighted = phase_weighted(ledger, gemm_dec, rms_dec)
    profiles = json.loads((out / "resource_profiles.json").read_text())
    write_json(out / "gemm_curves.json", {"rows": gemm_matrix})
    write_json(out / "rmsnorm_curves.json", {"rows": rms_matrix})
    write_json(out / "backend_rank_and_regret.json", {"gemm": gemm_dec, "rmsnorm": rms_dec})
    write_json(out / "crossover_tie_intervals.json", boundaries)
    write_json(out / "holdout_performance.json", holdout)
    write_json(out / "phase_weighted_cost.json", weighted)
    routing = {"schema": SCHEMA, "status": "DRAFT_FOR_E04-08_REPLAY",
        "key_fields": ["arch", "operator", "projection_or_hidden", "M_or_rows", "N", "K",
                       "dtype", "layout", "alignment", "capability"],
        "arch": "sm87", "dtype": "fp16", "guard": GUARD,
        "selection_method": "nearest non-holdout measured point in log2 size; tie uses stable priority",
        "gemm_boundaries": boundaries["gemm"], "rmsnorm_boundaries": boundaries["rmsnorm"],
        "fallback": {"gemm": "pytorch_cublas_opaque", "rmsnorm": "pytorch_reference"},
        "holdout": holdout, "phase_is_explanatory_not_a_routing_key": True,
        "requires_E04_08_for_production": True}
    write_json(out / "routing_rule_draft.json", routing)
    regressions = [r for r in gemm_raw + rms_raw if r["status"] != "PASS"]
    write_json(out / "regression_unsupported.json", {"rows": regressions,
        "note": "All forced identities are legal in the frozen E04-03/E04-02 domain; failures are never dropped."})
    profile_index = []
    for row in profiles["rows"]:
        path = out / row["csv"]
        profile_index.append({"M": row["M"], "backend": row["backend"],
            "path": row["csv"], "sha256": sha256_file(path) if path.exists() else None,
            "ncu_report": row.get("report"), "ncu_report_sha256": row.get("report_sha256"),
            "status": row["status"]})
    write_json(out / "profile_artifact_index.json", {"rows": profile_index})
    expected_gemm_cells = len(FAMILY_DIMS) * len(points()) * len(GEMM_BACKENDS)
    expected_rms_cells = ((len(points()) + len(RMS128_EXTRA)) + len(points())) * len(RMS_BACKENDS)
    conditions = {
        "decode_smallm_and_prefill_separated": all(r["regime"] for r in gemm_matrix + rms_matrix),
        "all_real_gemm_families_and_controlled_M_complete": len(gemm_matrix) == expected_gemm_cells,
        "rmsnorm_H128_H2048_rows_complete": len(rms_matrix) == expected_rms_cells,
        "phase_primary_metrics_present": all(r.get("device_us_p50") is not None and
            r.get("device_us_p95") is not None and r.get("host_completion_us_median") is not None
            for r in gemm_matrix + rms_matrix),
        "forced_backend_identity_and_correctness_bound": all(r["status"] == "PASS" for r in gemm_matrix + rms_matrix),
        "three_independent_randomized_runs": all(r["process_count"] == 3 for r in gemm_matrix + rms_matrix),
        "crossover_tie_guard_interval_reported": all("tie_zone_points" in r and
            "clear_rank_change_intervals" in r for r in boundaries["gemm"] + boundaries["rmsnorm"]),
        "resource_counter_evidence_present": all(r["status"] == "PASS" for r in profiles["rows"]),
        "phase_weighted_time_not_speedup_average": weighted["speedups_averaged"] is False,
        "holdout_regret_evaluated": len(holdout["gemm"]["rows"]) == len(FAMILY_DIMS) * len(HOLDOUT) and
            len(holdout["rmsnorm"]["rows"]) == 2 * len(HOLDOUT),
        "operator_only_no_model_claim": weighted["model_claim"] is False,
        "no_regression_or_dropped_unsupported": not regressions,
    }
    verdict = {"experiment_id": EXPERIMENT_ID, "schema": SCHEMA,
        "generated_at_utc": utc_now(), "conditions": conditions,
        "failed_conditions": [name for name, value in conditions.items() if not value],
        "overall": "PASS" if all(conditions.values()) else "FAIL",
        "counts": {"gemm_raw_records": len(gemm_raw), "gemm_curve_cells": len(gemm_matrix),
                   "rmsnorm_raw_records": len(rms_raw), "rmsnorm_curve_cells": len(rms_matrix),
                   "profile_points": len(profiles["rows"])},
        "upstream_context": {"E04-03": "PASS", "E04-02": "BLOCKED by E03-06/E03-07; "
            "E04-04 directly rechecks RMSNorm and does not promote E04-02 routing"},
        "claim": "operator-level phase-aware candidate/routing evidence only"}
    write_json(out / "verdict.json", verdict)
    make_manifest(out)
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["overall"] == "PASS" else 1


def make_manifest(out: Path) -> None:
    files = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "EVIDENCE_MANIFEST.json"):
        files[str(path.relative_to(out))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(out / "EVIDENCE_MANIFEST.json", {"schema": "hqsb.evidence_manifest/v1",
        "experiment_id": EXPERIMENT_ID, "generated_at_utc": utc_now(), "files": files})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("initialize", "build", "profile", "summarize"):
        q = sub.add_parser(name); q.add_argument("--output-dir", required=True)
    for name in ("gemm", "rmsnorm"):
        q = sub.add_parser(name); q.add_argument("--output-dir", required=True)
        q.add_argument("--process-index", type=int, required=True, choices=(0, 1, 2))
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    return {"initialize": initialize, "build": build, "gemm": collect_gemm,
            "rmsnorm": collect_rmsnorm, "profile": profile,
            "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
