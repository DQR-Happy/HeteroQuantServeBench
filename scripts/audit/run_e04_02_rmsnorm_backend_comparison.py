#!/usr/bin/env python3
"""E04-02: PyTorch/CUDA/Triton RMSNorm correctness and performance matrix.

Run only on the Jetson through ``scripts/audit/e04_02_run.sh``.  The collector
uses the frozen E03-01 semantic/tolerance helpers and writes machine-readable
raw evidence; it does not edit any S03 evidence.
"""

from __future__ import annotations

import argparse
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
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hqsb.benchmark import rmsnorm_correctness as rc  # noqa: E402


EXPERIMENT_ID = "E04-02"
EPSILON = 1.0e-6
GUARD = 0.05
SCHEMA = "hqsb.s04.rmsnorm_backend_comparison.v1"
BACKENDS = (
    "pytorch_reference",
    "cuda_v0",
    "cuda_v1",
    "cuda_v2",
    "triton_reference",
    "triton_tuned",
)
DTYPES = ("fp16", "fp32")
MANDATORY_H = (1, 3, 31, 32, 33, 99, 100, 101, 127, 128, 129,
               511, 512, 513, 2047, 2048, 2049, 6144, 8192)
SPECIAL_SHAPES = ((2, 33), (1, 2049))


def utc_now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_safe(value: Any) -> Any:
    """Convert diagnostic NaN/Inf scalars to JSON null; IEEE classes stay explicit."""
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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(json_safe(dict(row)), sort_keys=True,
                               allow_nan=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def run_text(cmd: Sequence[str]) -> str:
    return subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False).stdout.strip()


def shape_plan() -> List[Dict[str, Any]]:
    items: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def add(rows: int, hidden: int, tag: str, split: str = "validation") -> None:
        key = (int(rows), int(hidden))
        rec = items.setdefault(key, {"rows": key[0], "hidden": key[1],
                                    "tags": [], "split": split})
        if tag not in rec["tags"]:
            rec["tags"].append(tag)
        if rec["split"] != "holdout" and split == "holdout":
            rec["split"] = split
        elif rec["split"] == "validation" and split == "tune":
            rec["split"] = split

    for hidden in MANDATORY_H:
        add(1, hidden, "MANDATORY_H_BOUNDARY")
    for rows in (3, 7, 15):
        add(rows, 2048, "MANDATORY_SMALL_IRREGULAR_BATCH")

    # Exact S02/Qwen3 shape families.  H=128 is Q/K norm; H=2048 is hidden norm.
    for rows in (1, 32, 128, 512, 2048):
        add(rows, 2048, "S02_REAL_HIDDEN", "tune" if rows == 128 else "validation")
    for rows in (8, 16, 256, 512, 1024, 2048, 8192, 16384, 32768):
        add(rows, 128, "S02_REAL_QK", "tune" if rows == 1024 else "validation")

    # Frozen before collection.  These unseen row/H combinations are always run last.
    for rows, hidden in ((7, 2049), (257, 129), (2048, 128), (512, 2048)):
        add(rows, hidden, "HOLDOUT", "holdout")

    order = {"tune": 0, "validation": 1, "holdout": 2}
    return sorted(items.values(), key=lambda x: (order[x["split"]], x["hidden"], x["rows"]))


def backend_supported(backend: str, dtype: str) -> Tuple[bool, Optional[str]]:
    if dtype == "fp16" and backend in ("cuda_v0", "cuda_v1"):
        return False, "S03 frozen contract: CUDA V0/V1 are FP32-only"
    return True, None


def torch_dtype(torch, dtype: str):
    return torch.float16 if dtype == "fp16" else torch.float32


def framework_reference(torch, x, w, epsilon: float):
    y = x.float() * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + epsilon)
    return (y * w.float()).to(x.dtype)


def load_backends():
    import torch
    from ops import cuda_bridge
    triton_rms = importlib.import_module("ops.triton.rmsnorm")
    return torch, cuda_bridge, triton_rms


def launch(backend: str, torch, cuda_bridge, triton_rms, x, w, out, stream):
    if backend == "pytorch_reference":
        out.copy_(framework_reference(torch, x, w, EPSILON))
    elif backend.startswith("cuda_"):
        variant = {"cuda_v0": "v0_shared", "cuda_v1": "v1_warp_shuffle",
                   "cuda_v2": "v2_vectorized"}[backend]
        cuda_bridge.rmsnorm_forward(x, w, out=out,
                                    dtype="fp16" if x.dtype == torch.float16 else "fp32",
                                    variant=variant, epsilon=EPSILON, stream=stream)
    elif backend == "triton_reference":
        triton_rms.rmsnorm_reference(x, w, EPSILON, out=out)
    elif backend == "triton_tuned":
        triton_rms.rmsnorm_optimized(x, w, EPSILON, out=out)
    else:
        raise ValueError(backend)
    return out


def make_arrays(rows: int, hidden: int, dtype: str, mode: str, seed_key: str):
    seed = rc.derive_seed(seed_key)
    generated = rc.generate_case_arrays(mode, rows, hidden, dtype, seed)
    return generated, seed


def correctness_record(*, torch, cuda_bridge, triton_rms, backend: str, dtype: str,
                       rows: int, hidden: int, mode: str, process_index: int,
                       split: str, tags: Sequence[str]) -> Dict[str, Any]:
    case_id = f"p{process_index}|{split}|r{rows}|h{hidden}|{dtype}|{mode}|{backend}"
    supported, reason = backend_supported(backend, dtype)
    base = {"schema": SCHEMA, "case_id": case_id, "process_index": process_index,
            "split": split, "tags": list(tags), "rows": rows, "hidden": hidden,
            "dtype": dtype, "layout": "contiguous", "mode": mode,
            "requested_backend": backend,
            "actual_backend": backend if supported else None,
            "epsilon": EPSILON, "tolerance": rc.tolerance_for(dtype, hidden)}
    if not supported:
        return {**base, "status": "EXPECTED_UNSUPPORTED", "reason": reason}

    generated, seed = make_arrays(rows, hidden, dtype, mode, case_id.rsplit("|", 1)[0])
    x_np, w_np = generated["x"], generated["w"]
    td = torch_dtype(torch, dtype)
    x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device="cuda", dtype=td)
    w = torch.from_numpy(np.ascontiguousarray(w_np)).to(device="cuda", dtype=td)
    out = torch.empty_like(x)
    x_hash_before, w_hash_before = sha256_array(x_np), sha256_array(w_np)
    stream = torch.cuda.Stream()
    error = None
    try:
        with torch.cuda.stream(stream):
            launch(backend, torch, cuda_bridge, triton_rms, x, w, out, stream)
        stream.synchronize()
        candidate = out.cpu().numpy().reshape(-1)
    except Exception as exc:  # evidence must retain compile/launch failures
        error = {"type": type(exc).__name__, "message": str(exc)}
        candidate = None
    if candidate is None:
        return {**base, "seed": seed, "x_sha256": x_hash_before,
                "weight_sha256": w_hash_before, "status": "FAIL", "error": error}

    fw = framework_reference(torch, x, w, EPSILON)
    torch.cuda.synchronize()
    fw_np = fw.cpu().numpy().reshape(-1)
    fp64 = rc.fp64_oracle(x_np, w_np, rows, hidden, EPSILON)
    expected_cls = rc.expected_classes(x_np, rows, hidden)
    candidate_cls = rc.classify(candidate)
    finite_mask = (expected_cls == rc.CLASS_FINITE) & np.isfinite(fw_np)
    tolerance = rc.tolerance_for(dtype, hidden)
    metrics = rc.error_metrics(candidate, fw_np, finite_mask)
    violations = rc.elementwise_violations(candidate, fw_np, tolerance, finite_mask)
    cosine_ok = metrics["cosine"] is None or metrics["cosine"] >= tolerance["cosine_min"]
    l2_ok = (not metrics["applicable"]) or metrics["l2rel"] <= tolerance["l2rel"]
    class_ok = bool(np.array_equal(candidate_cls, expected_cls))
    fp64_mask = np.isfinite(fp64) & np.isfinite(fw_np)
    fp64_metrics = rc.error_metrics(fw_np, fp64, fp64_mask)
    exact_zero_mask = rc.inf_row_finite_lanes_exact_zero(x_np, rows, hidden)
    exact_zero_ok = bool(np.all(candidate[exact_zero_mask] == 0)) if np.any(exact_zero_mask) else True
    repeat = torch.empty_like(x)
    with torch.cuda.stream(stream):
        launch(backend, torch, cuda_bridge, triton_rms, x, w, repeat, stream)
    stream.synchronize()
    repeat_np = repeat.cpu().numpy()
    stable = bool(np.array_equal(candidate.reshape(rows, hidden), repeat_np, equal_nan=True))
    passed = class_ok and violations["passed"] and cosine_ok and l2_ok and exact_zero_ok and stable
    config = triton_rms.rmsnorm_last_tuned_config() if backend == "triton_tuned" else None
    return {**base, "seed": seed, "x_sha256": x_hash_before,
            "weight_sha256": w_hash_before, "output_sha256": sha256_array(repeat_np),
            "input_unchanged": sha256_array(x.cpu().numpy()) == x_hash_before,
            "weight_unchanged": sha256_array(w.cpu().numpy()) == w_hash_before,
            "classification_ok": class_ok,
            "candidate_class_counts": rc.class_counts(candidate_cls),
            "expected_class_counts": rc.class_counts(expected_cls),
            "finite_metrics_vs_fp32_framework": metrics,
            "elementwise": violations, "fp32_framework_vs_fp64": fp64_metrics,
            "fp64_declared_divergence": [dtype, mode] in [list(x) for x in rc.DECLARED_DIVERGENCE],
            "inf_row_finite_lanes_exact_zero": exact_zero_ok,
            "repeat_bitwise_stable": stable, "tuned_config": config,
            "status": "PASS" if passed else "FAIL",
            "first_mismatch": None if passed else rc.first_mismatch(
                candidate=candidate, reference=fw_np, candidate_classes=candidate_cls,
                expected_class_array=expected_cls, tolerance=tolerance, hidden=hidden)}


def timing_record(*, torch, cuda_bridge, triton_rms, backend: str, dtype: str,
                  rows: int, hidden: int, process_index: int, split: str,
                  tags: Sequence[str]) -> Dict[str, Any]:
    case_id = f"p{process_index}|{split}|r{rows}|h{hidden}|{dtype}|{backend}"
    supported, reason = backend_supported(backend, dtype)
    base = {"case_id": case_id, "process_index": process_index, "split": split,
            "tags": list(tags), "rows": rows, "hidden": hidden, "dtype": dtype,
            "layout": "contiguous", "requested_backend": backend,
            "actual_backend": backend if supported else None}
    if not supported:
        return {**base, "status": "EXPECTED_UNSUPPORTED", "reason": reason}

    correctness = correctness_record(torch=torch, cuda_bridge=cuda_bridge,
        triton_rms=triton_rms, backend=backend, dtype=dtype, rows=rows, hidden=hidden,
        mode="random_normal", process_index=process_index, split=split, tags=tags)
    if correctness["status"] != "PASS":
        return {**base, "status": "FAIL", "correctness": correctness}

    generated, _ = make_arrays(rows, hidden, dtype, "random_normal", case_id.rsplit("|", 1)[0])
    td = torch_dtype(torch, dtype)
    x = torch.from_numpy(np.ascontiguousarray(generated["x"])).to("cuda", dtype=td)
    w = torch.from_numpy(np.ascontiguousarray(generated["w"])).to("cuda", dtype=td)
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(5):
            launch(backend, torch, cuda_bridge, triton_rms, x, w, out, stream)
    stream.synchronize()
    elements = rows * hidden
    iters = 30 if elements <= 8192 else (10 if elements <= 262144 else 3)
    groups = 5
    device_us, host_us = [], []
    for _ in range(groups):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            for _i in range(iters):
                launch(backend, torch, cuda_bridge, triton_rms, x, w, out, stream)
            end.record(stream)
        end.synchronize()
        device_us.append(float(start.elapsed_time(end) * 1000.0 / iters))
        t0 = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            for _i in range(iters):
                launch(backend, torch, cuda_bridge, triton_rms, x, w, out, stream)
        stream.synchronize()
        host_us.append(float((time.perf_counter_ns() - t0) / 1000.0 / iters))
    logical_bytes = rows * hidden * (2 if dtype == "fp16" else 4) * 3
    med = statistics.median(device_us)
    return {**base, "status": "PASS", "correctness_case_id": correctness["case_id"],
            "x_sha256": correctness["x_sha256"], "weight_sha256": correctness["weight_sha256"],
            "warmup": 5, "groups": groups, "iterations_per_group": iters,
            "device_us_raw": device_us, "host_completion_us_raw": host_us,
            "logical_bytes": logical_bytes,
            "effective_GBps_median": logical_bytes / med / 1000.0,
            "byte_convention": "rows*H*(read_x+read_w+write_y); logical, not DRAM",
            "tuned_config": triton_rms.rmsnorm_last_tuned_config()
                if backend == "triton_tuned" else None}


def collect_process(args) -> int:
    out_dir = Path(args.output_dir)
    torch, cuda_bridge, triton_rms = load_backends()
    torch.cuda.init()
    records: List[Dict[str, Any]] = []
    rng = random.Random(40200 + args.process_index)
    for shape in shape_plan():
        backend_order = list(BACKENDS)
        rng.shuffle(backend_order)
        for dtype in DTYPES:
            for backend in backend_order:
                rec = timing_record(torch=torch, cuda_bridge=cuda_bridge,
                    triton_rms=triton_rms, backend=backend, dtype=dtype,
                    rows=shape["rows"], hidden=shape["hidden"],
                    process_index=args.process_index, split=shape["split"], tags=shape["tags"])
                records.append(rec)
                print(f"[{args.process_index}] {rec['case_id']} {rec['status']}", flush=True)
    write_jsonl(out_dir / f"cases_proc{args.process_index}.jsonl", records)

    if args.process_index == 0:
        collect_special_evidence(out_dir, torch, cuda_bridge, triton_rms)
    return 0


def collect_special_evidence(out_dir: Path, torch, cuda_bridge, triton_rms) -> None:
    special: List[Dict[str, Any]] = []
    for rows, hidden in SPECIAL_SHAPES:
        for dtype in DTYPES:
            for mode in rc.INPUT_MODES:
                for backend in BACKENDS:
                    rec = correctness_record(torch=torch, cuda_bridge=cuda_bridge,
                        triton_rms=triton_rms, backend=backend, dtype=dtype,
                        rows=rows, hidden=hidden, mode=mode, process_index=0,
                        split="adversarial", tags=["SPECIAL_VALUE", "TAIL"])
                    special.append(rec)
    write_jsonl(out_dir / "special_correctness.jsonl", special)


def collect_special(args) -> int:
    torch, cuda_bridge, triton_rms = load_backends()
    collect_special_evidence(Path(args.output_dir), torch, cuda_bridge, triton_rms)
    return 0


def collect_correctness_matrix(args) -> int:
    out_dir = Path(args.output_dir)
    torch, cuda_bridge, triton_rms = load_backends()
    records: List[Dict[str, Any]] = []
    for shape in shape_plan():
        for dtype in DTYPES:
            for backend in BACKENDS:
                rec = correctness_record(torch=torch, cuda_bridge=cuda_bridge,
                    triton_rms=triton_rms, backend=backend, dtype=dtype,
                    rows=shape["rows"], hidden=shape["hidden"], mode="random_normal",
                    process_index=99, split=shape["split"], tags=shape["tags"])
                records.append(rec)
    write_jsonl(out_dir / "correctness_matrix.jsonl", records)
    return 0 if all(r["status"] in ("PASS", "EXPECTED_UNSUPPORTED") for r in records) else 1


def cache_inventory(root: Path) -> Dict[str, Any]:
    files = []
    if root.exists():
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    return {"file_count": len(files), "total_bytes": sum(x["bytes"] for x in files),
            "files": files}


def cold_worker(args) -> int:
    t0 = time.perf_counter_ns()
    torch, _cuda_bridge, triton_rms = load_backends()
    import_ms = (time.perf_counter_ns() - t0) / 1e6
    td = torch_dtype(torch, args.dtype)
    x = torch.randn((args.rows, args.hidden), device="cuda", dtype=td)
    w = torch.randn((args.hidden,), device="cuda", dtype=td)
    out = torch.empty_like(x)
    fn = triton_rms.rmsnorm_reference if args.backend == "triton_reference" else triton_rms.rmsnorm_optimized
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    fn(x, w, EPSILON, out=out)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter_ns() - t1) / 1e6
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t2 = time.perf_counter_ns()
    start.record()
    fn(x, w, EPSILON, out=out)
    end.record()
    end.synchronize()
    second_host_ms = (time.perf_counter_ns() - t2) / 1e6
    second_device_ms = start.elapsed_time(end)
    ref = framework_reference(torch, x, w, EPSILON)
    correct = bool(torch.allclose(out, ref, atol=rc.tolerance_for(args.dtype, args.hidden)["atol"],
                                  rtol=rc.tolerance_for(args.dtype, args.hidden)["rtol"],
                                  equal_nan=False))
    payload = {"backend": args.backend, "dtype": args.dtype, "rows": args.rows,
               "hidden": args.hidden, "cache_dir": os.environ.get("TRITON_CACHE_DIR"),
               "import_probe_ms": import_ms,
               "first_jit_or_autotune_plus_execution_ms": first_ms,
               "first_execution_after_compile_host_ms": second_host_ms,
               "first_execution_after_compile_device_ms": second_device_ms,
               "correctness": correct,
               "selected_config": triton_rms.rmsnorm_last_tuned_config()
                    if args.backend == "triton_tuned" else {"num_warps": 4, "num_stages": 1},
               "config_hash": triton_rms.rmsnorm_config_hash(args.rows, args.hidden, args.dtype, EPSILON)}
    write_json(Path(args.worker_output), payload)
    return 0


def collect_cold(args) -> int:
    out_dir = Path(args.output_dir)
    root = out_dir / "triton_cold_caches"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    script = str(Path(__file__).resolve())
    for backend in ("triton_reference", "triton_tuned"):
        for dtype in DTYPES:
            for hidden in (128, 2048, 2049):
                rows = 128 if hidden in (128, 2048) else 7
                cache_dir = Path(tempfile.mkdtemp(prefix=f"{backend}_{dtype}_h{hidden}_", dir=root))
                pair = {"backend": backend, "dtype": dtype, "rows": rows, "hidden": hidden,
                        "cache_kind": "experiment-private; no global cache modified", "runs": []}
                for state in ("empty_cache", "cache_hit_fresh_process"):
                    worker_out = cache_dir / f"{state}.json"
                    env = dict(os.environ)
                    env["TRITON_CACHE_DIR"] = str(cache_dir)
                    env["TRITON_PRINT_AUTOTUNING"] = "1"
                    proc = subprocess.run([sys.executable, script, "cold-worker",
                        "--backend", backend, "--dtype", dtype, "--rows", str(rows),
                        "--hidden", str(hidden), "--worker-output", str(worker_out)],
                        cwd=REPO, env=env, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, check=False)
                    payload = json.loads(worker_out.read_text()) if worker_out.exists() else None
                    pair["runs"].append({"state": state, "exit_code": proc.returncode,
                        "stdout": proc.stdout, "stderr": proc.stderr, "measurement": payload,
                        "cache_after": cache_inventory(cache_dir)})
                records.append(pair)
    write_json(out_dir / "cold_states.json", {"schema": SCHEMA, "records": records})
    return 0 if all(r["exit_code"] == 0 for p in records for r in p["runs"]) else 1


def collect_layout_stream(args) -> int:
    out_dir = Path(args.output_dir)
    torch, cuda_bridge, triton_rms = load_backends()
    layout_rows = []
    for dtype in DTYPES:
        td = torch_dtype(torch, dtype)
        base = torch.randn((4, 66), device="cuda", dtype=td)
        x = base[:, ::2]
        w = torch.randn((33,), device="cuda", dtype=td)
        for backend in BACKENDS:
            error = None
            supported, support_reason = backend_supported(backend, dtype)
            if not supported:
                layout_rows.append({"dtype": dtype, "backend": backend,
                    "layout": "stride_1=2", "status": "EXPECTED_UNSUPPORTED",
                    "reason": support_reason})
                continue
            out = torch.empty_like(x.contiguous())
            try:
                launch(backend, torch, cuda_bridge, triton_rms, x, w, out,
                       torch.cuda.current_stream())
                torch.cuda.synchronize()
                actual = "SUPPORTED" if backend == "pytorch_reference" else "UNEXPECTED_ACCEPT"
            except Exception as exc:
                actual = "EXPECTED_REJECT" if backend != "pytorch_reference" else "UNEXPECTED_REJECT"
                error = {"type": type(exc).__name__, "message": str(exc)}
            layout_rows.append({"dtype": dtype, "backend": backend, "layout": "stride_1=2",
                "expected": "SUPPORTED" if backend == "pytorch_reference" else "REJECT",
                "actual": actual, "error": error,
                "status": "PASS" if actual in ("SUPPORTED", "EXPECTED_REJECT") else "FAIL"})

    stream_rows = []
    for backend in ("triton_reference", "triton_tuned"):
        for dtype in DTYPES:
            td = torch_dtype(torch, dtype)
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            outs, refs = [], []
            for index, stream in enumerate(streams):
                x = torch.full((17, 129), float(index + 1), device="cuda", dtype=td)
                w = torch.linspace(0.5, 1.5, 129, device="cuda", dtype=td)
                out = torch.empty_like(x)
                with torch.cuda.stream(stream):
                    launch(backend, torch, cuda_bridge, triton_rms, x, w, out, stream)
                    refs.append(framework_reference(torch, x, w, EPSILON))
                outs.append(out)
            for stream in streams:
                stream.synchronize()
            tol = rc.tolerance_for(dtype, 129)
            ok = all(torch.allclose(y, ref, atol=tol["atol"], rtol=tol["rtol"])
                     for y, ref in zip(outs, refs))
            stream_rows.append({"backend": backend, "dtype": dtype,
                "default_stream": False, "two_independent_streams": True,
                "current_stream_correct": bool(ok), "status": "PASS" if ok else "FAIL"})
    write_json(out_dir / "layout_stream_safety.json",
               {"layout_cases": layout_rows, "stream_cases": stream_rows,
                "scope_note": "E04-02 spot check; full bridge/current-stream gate belongs to E04-09"})
    return 0


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(args) -> int:
    out_dir = Path(args.output_dir)
    all_records = []
    for index in range(3):
        all_records.extend(read_jsonl(out_dir / f"cases_proc{index}.jsonl"))
    groups: Dict[Tuple[int, int, str, str, str], List[Dict[str, Any]]] = {}
    for rec in all_records:
        key = (rec["rows"], rec["hidden"], rec["dtype"], rec["requested_backend"], rec["split"])
        groups.setdefault(key, []).append(rec)
    matrix = []
    for key, recs in sorted(groups.items()):
        rows, hidden, dtype, backend, split = key
        status = "PASS" if all(r["status"] == "PASS" for r in recs) else recs[0]["status"]
        row = {"rows": rows, "hidden": hidden, "dtype": dtype, "backend": backend,
               "split": split, "status": status, "process_count": len(recs),
               "actual_backends": sorted({str(r.get("actual_backend")) for r in recs})}
        if status == "PASS":
            device = [v for r in recs for v in r["device_us_raw"]]
            host = [v for r in recs for v in r["host_completion_us_raw"]]
            logical_bytes = recs[0]["logical_bytes"]
            row.update({"device_us_median": statistics.median(device),
                        "device_us_p95": percentile(device, 95),
                        "host_completion_us_median": statistics.median(host),
                        "host_completion_us_p95": percentile(host, 95),
                        "logical_bytes": logical_bytes,
                        "effective_GBps": logical_bytes / statistics.median(device) / 1000.0,
                        "configs": [r.get("tuned_config") for r in recs if r.get("tuned_config")]})
        else:
            row["reason"] = recs[0].get("reason") or recs[0].get("correctness", {}).get("error")
        matrix.append(row)

    by_shape: Dict[Tuple[int, int, str, str], List[Dict[str, Any]]] = {}
    for row in matrix:
        by_shape.setdefault((row["rows"], row["hidden"], row["dtype"], row["split"]), []).append(row)
    winners, routing = [], []
    for key, rows_ in sorted(by_shape.items()):
        eligible = sorted([r for r in rows_ if r["status"] == "PASS"], key=lambda r: r["device_us_median"])
        if not eligible:
            winners.append({"rows": key[0], "hidden": key[1], "dtype": key[2],
                            "split": key[3], "status": "NO_ELIGIBLE"})
            continue
        best = eligible[0]
        second = eligible[1] if len(eligible) > 1 else None
        advantage = (second["device_us_median"] / best["device_us_median"] - 1.0) if second else math.inf
        ties = [r["backend"] for r in eligible
                if r["device_us_median"] <= best["device_us_median"] * (1.0 + GUARD)]
        win = {"rows": key[0], "hidden": key[1], "dtype": key[2], "split": key[3],
               "winner": best["backend"] if len(ties) == 1 else "TIE",
               "tie_backends": ties, "best_backend": best["backend"],
               "best_device_us": best["device_us_median"], "guard": GUARD,
               "advantage_over_second": advantage}
        winners.append(win)
        if (key[3] == "holdout" and len(ties) == 1 and advantage >= GUARD
                and best["backend"] != "pytorch_reference"):
            routing.append({**win, "safety_gate": "all three process correctness PASS",
                            "candidate": best["backend"]})

    special = read_jsonl(out_dir / "special_correctness.jsonl")
    correctness_matrix = read_jsonl(out_dir / "correctness_matrix.jsonl")
    cold = json.loads((out_dir / "cold_states.json").read_text())
    safety = json.loads((out_dir / "layout_stream_safety.json").read_text())
    inherited = inherited_evidence()
    conditions = {
        "six_backend_identities_forced_actual": all(any(r["status"] == "PASS" and
            r["requested_backend"] == b and r["actual_backend"] == b for r in all_records) for b in BACKENDS),
        "inherits_s03_pass_spec_tolerance": inherited["all_required_pass"],
        "mandatory_real_odd_tail_special_complete": all(r["status"] in ("PASS", "EXPECTED_UNSUPPORTED") for r in special)
            and all(any(m["hidden"] == h for m in matrix) for h in MANDATORY_H),
        "supported_correct_no_silent_fallback": all(r["status"] == "PASS" and
            r["actual_backend"] == r["requested_backend"] for r in correctness_matrix
            if backend_supported(r["requested_backend"], r["dtype"])[0]),
        "cold_jit_tune_cache_first_separated": all(run["exit_code"] == 0 and run["measurement"]
            for pair in cold["records"] for run in pair["runs"]),
        "device_host_raw_three_processes": all(len(v) == 3 for v in groups.values()),
        "full_matrix_preserves_all_statuses": len(matrix) == len(shape_plan()) * len(DTYPES) * len(BACKENDS),
        "holdout_generalization_executed": any(w["split"] == "holdout" for w in winners),
        "routing_only_safe_guarded_domains": all(r["split"] == "holdout" and
            r["advantage_over_second"] >= GUARD for r in routing),
        "layout_and_stream_spot_checks_pass": all(x["status"] in ("PASS", "EXPECTED_UNSUPPORTED")
            for x in safety["layout_cases"] + safety["stream_cases"]),
    }
    experiment_failures = [k for k, v in conditions.items()
                           if not v and k != "inherits_s03_pass_spec_tolerance"]
    blocked = not conditions["inherits_s03_pass_spec_tolerance"] and not experiment_failures
    verdict = {"experiment_id": EXPERIMENT_ID, "schema": SCHEMA,
               "generated_at_utc": utc_now(), "conditions": conditions,
               "failed_conditions": [k for k, v in conditions.items() if not v],
               "blocked_by": (["E03-06", "E03-07"] if blocked else []),
               "overall": "PASS" if all(conditions.values()) else ("BLOCKED" if blocked else "FAIL"),
               "matrix_cells": len(matrix), "winner_cells": len(winners),
               "routing_candidate_cells": len(routing),
               "ordinary_records": len(all_records),
               "correctness_matrix_records": len(correctness_matrix),
               "special_records": len(special)}
    write_json(out_dir / "backend_matrix.json", {"rows": matrix})
    write_json(out_dir / "winner_matrix.json", {"guard": GUARD, "rows": winners})
    write_json(out_dir / "routing_candidates.json", {"rows": routing})
    write_json(out_dir / "inherited_evidence.json", inherited)
    write_json(out_dir / "verdict.json", verdict)
    make_manifest(out_dir)
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["overall"] in ("PASS", "BLOCKED") else 1


def inherited_evidence() -> Dict[str, Any]:
    paths = {
        "e03_01_verdict": REPO / "docs/stage_experiments/S03/E03-01/raw/verdict.json",
        "e03_01_spec": REPO / "docs/stage_experiments/S03/E03-01/raw/resolved_spec.json",
        "e03_02_verdict": REPO / "docs/stage_experiments/S03/E03-02/raw/verdict.json",
        "e03_03_verdict": REPO / "docs/stage_experiments/S03/E03-03/raw/verdict.json",
        "e03_06_verdict": REPO / "docs/stage_experiments/S03/E03-06/raw/verdict.json",
        "e03_07_verdict": REPO / "docs/stage_experiments/S03/E03-07/raw/verdict.json",
        "e04_01_verdict": REPO / "docs/stage_experiments/S04/E04-01/raw/verdict.json",
    }
    evidence, required_pass = {}, []
    for name, path in paths.items():
        if path.exists():
            data = json.loads(path.read_text())
            evidence[name] = {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path),
                              "overall": data.get("overall"),
                              "canonical_sha256": data.get("canonical_sha256")}
            if name != "e03_01_spec":
                required_pass.append(data.get("overall") == "PASS")
        else:
            evidence[name] = {"path": str(path.relative_to(REPO)), "missing": True}
            required_pass.append(False)
    return {"resolved_spec_expected_sha256":
            "84728d65d928bae64ad26c55a38843f7cb2517bacf5a6fe128c205397231a718",
            "resolved_spec_actual_sha256": evidence.get("e03_01_spec", {}).get("canonical_sha256"),
            "evidence": evidence,
            "all_required_pass": all(required_pass) and
                evidence.get("e03_01_spec", {}).get("canonical_sha256") ==
                "84728d65d928bae64ad26c55a38843f7cb2517bacf5a6fe128c205397231a718"}


def make_manifest(out_dir: Path) -> None:
    files = {}
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()
                       and p.name != "EVIDENCE_MANIFEST.json"):
        files[str(path.relative_to(out_dir))] = {"bytes": path.stat().st_size,
                                                "sha256": sha256_file(path)}
    write_json(out_dir / "EVIDENCE_MANIFEST.json",
               {"schema": "hqsb.evidence_manifest/v1", "experiment_id": EXPERIMENT_ID,
                "generated_at_utc": utc_now(), "files": files})


def initialize(args) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = shape_plan()
    protocol = {"experiment_id": EXPERIMENT_ID, "schema": SCHEMA,
        "frozen_at_utc": utc_now(), "epsilon": EPSILON, "guard": GUARD,
        "backends": list(BACKENDS), "dtypes": list(DTYPES), "mandatory_h": list(MANDATORY_H),
        "shape_plan": plan, "special_modes": list(rc.INPUT_MODES),
        "layout_policy": "contiguous is supported; non-contiguous must be explicit support or reject",
        "cold_policy": "experiment-private empty cache then same-cache fresh process",
        "holdout_policy": "tagged before collection and ordered after tune/validation",
        "claim_boundary": "operator microbenchmark only; no model-level speedup claim"}
    write_json(out_dir / "protocol.json", protocol)
    source_paths = [Path(__file__), REPO / "ops/triton/rmsnorm.py",
        REPO / "ops/cuda_bridge.py", REPO / "hqsb/benchmark/rmsnorm_correctness.py"]
    provenance = {"experiment_id": EXPERIMENT_ID, "created_at_utc": utc_now(),
        "platform": platform.platform(), "python": sys.version, "git_commit": run_text(["git", "rev-parse", "HEAD"]),
        "git_status": run_text(["git", "status", "--short"]),
        "source_sha256": {str(p.relative_to(REPO)): sha256_file(p) for p in source_paths},
        "cuda_visible": None}
    try:
        import torch
        provenance.update({"torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
            "cuda_visible": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None})
        import triton
        provenance["triton_version"] = triton.__version__
    except Exception as exc:
        provenance["probe_error"] = {"type": type(exc).__name__, "message": str(exc)}
    write_json(out_dir / "provenance.json", provenance)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("initialize", "cold", "layout-stream", "collect-special",
                 "correctness-matrix", "summarize"):
        q = sub.add_parser(name)
        q.add_argument("--output-dir", required=True)
    q = sub.add_parser("collect")
    q.add_argument("--output-dir", required=True)
    q.add_argument("--process-index", type=int, required=True, choices=(0, 1, 2))
    q = sub.add_parser("cold-worker")
    q.add_argument("--backend", required=True, choices=("triton_reference", "triton_tuned"))
    q.add_argument("--dtype", required=True, choices=DTYPES)
    q.add_argument("--rows", type=int, required=True)
    q.add_argument("--hidden", type=int, required=True)
    q.add_argument("--worker-output", required=True)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    return {"initialize": initialize, "collect": collect_process,
            "collect-special": collect_special, "cold": collect_cold,
            "correctness-matrix": collect_correctness_matrix,
            "cold-worker": cold_worker, "layout-stream": collect_layout_stream,
            "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
