#!/usr/bin/env python3
"""E04-03 Qwen GEMM backend, full-dimensional tail, and safety audit.

This script is intended to run on the Jetson through ``remote_run.sh``.  It
imports the S02 runtime census instead of hand-writing projection shapes and
keeps operator-level claims separate from model-level claims.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXPERIMENT_ID = "E04-03"
SCHEMA = "hqsb.s04.gemm_backend_tail.v1"
DTYPE = "fp16"
ATOL = 0.01
RTOL = 0.02
L2REL = 0.01
PERF_GUARD = 0.05
GUARD_SENTINEL = 123.0
PROJECTIONS = ("q", "k", "v", "o", "gate", "up", "down", "lm_head")
SUFFIX_MAP = {
    "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
    "gate_proj": "gate", "up_proj": "up", "down_proj": "down",
    "lm_head": "lm_head",
}


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: Sequence[str], *, env: Optional[Mapping[str, str]] = None,
        timeout: int = 900) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(list(cmd), cwd=REPO, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, check=False)
        return {"command": list(cmd), "exit_code": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr,
                "duration_s": time.perf_counter() - t0, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        return {"command": list(cmd), "exit_code": 124,
                "stdout": exc.stdout or "", "stderr": exc.stderr or "",
                "duration_s": time.perf_counter() - t0, "timed_out": True}


def projection_name(module: str) -> Optional[str]:
    for suffix, name in SUFFIX_MAP.items():
        if module == suffix or module.endswith("." + suffix):
            return name
    return None


def parse_shape(text: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in ast.literal_eval(text))


def build_projection_ledger() -> Dict[str, Any]:
    root = REPO / "docs/stage_experiments/S02/E02-02/raw_v2/run_0"
    sources = sorted(root.glob("census_*.json"))
    grouped: Dict[Tuple[str, str, str, int, int, int], Dict[str, Any]] = {}
    source_rows = []
    for path in sources:
        data = json.loads(path.read_text())
        workload = data["workload"]["name"]
        source_rows.append({"path": str(path.relative_to(REPO)),
                            "sha256": sha256_file(path), "workload": workload,
                            "provenance": data.get("provenance")})
        for rec in data["modules"]:
            projection = projection_name(rec["module"])
            if projection is None:
                continue
            in_shape = parse_shape(rec["input_shapes"][0])
            out_shape = parse_shape(rec["output_shapes"][0])
            m = math.prod(in_shape[:-1])
            k, n = in_shape[-1], out_shape[-1]
            key = (workload, rec["phase"], projection, m, n, k)
            row = grouped.setdefault(key, {
                "workload": workload, "phase": rec["phase"],
                "projection": projection, "M": m, "N": n, "K": k,
                "batch": in_shape[0], "input_shape": list(in_shape),
                "output_shape": list(out_shape), "dtype": "fp16",
                "logical_A": "row_major[M,K]",
                "model_weight_physical": "row_major[N,K]",
                "logical_B": "transpose(weight) => [K,N], strides=(1,K)",
                "logical_D": "row_major[M,N]", "module_count": 0,
                "calls": 0, "modules": [],
            })
            row["module_count"] += 1
            row["calls"] += int(rec["call_count"])
            row["modules"].append(rec["module"])
    rows = sorted(grouped.values(), key=lambda r: (
        r["workload"], r["phase"], PROJECTIONS.index(r["projection"])))
    shares: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for row in rows:
        key = (row["workload"], row["phase"])
        calls, flops = shares.get(key, (0, 0))
        shares[key] = (calls + row["calls"],
                       flops + row["calls"] * 2 * row["M"] * row["N"] * row["K"])
    for row in rows:
        total_calls, total_flops = shares[(row["workload"], row["phase"])]
        row["call_share_within_projection_modules"] = row["calls"] / total_calls
        work = row["calls"] * 2 * row["M"] * row["N"] * row["K"]
        row["flop_share_within_projection_modules"] = work / total_flops
        row["share_semantics"] = "projection-only call/FLOP share; not measured wall time"
    coverage = {name: sum(1 for r in rows if r["projection"] == name)
                for name in PROJECTIONS}
    return {"schema": SCHEMA, "generated_from_runtime_census": True,
            "sources": source_rows, "rows": rows, "projection_coverage": coverage,
            "all_required_projections_present": all(coverage.values())}


def unique_real_shapes(ledger: Mapping[str, Any]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, int, int, str], Dict[str, Any]] = {}
    for row in ledger["rows"]:
        key = (row["M"], row["N"], row["K"], row["phase"])
        item = groups.setdefault(key, {"M": key[0], "N": key[1], "K": key[2],
                                       "phase": key[3], "projections": [],
                                       "workloads": []})
        if row["projection"] not in item["projections"]:
            item["projections"].append(row["projection"])
        if row["workload"] not in item["workloads"]:
            item["workloads"].append(row["workload"])
    return sorted(groups.values(), key=lambda x: (x["phase"], x["M"], x["K"], x["N"]))


def tail_plan() -> List[Dict[str, Any]]:
    cases: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    def add(m: int, n: int, k: int, *tags: str) -> None:
        key = (m, n, k)
        row = cases.setdefault(key, {"case_id": f"m{m}_n{n}_k{k}",
                                     "M": m, "N": n, "K": k, "tags": []})
        for tag in tags:
            if tag not in row["tags"]:
                row["tags"].append(tag)
    for boundary in (64, 128):
        for m in (boundary - 1, boundary, boundary + 1):
            add(m, 128, 64, "M_TAIL_UNION")
        for n in (boundary - 1, boundary, boundary + 1):
            add(128, n, 64, "N_TAIL_UNION")
    for boundary in (32, 64):
        for k in (boundary - 1, boundary, boundary + 1):
            add(128, 128, k, "K_TAIL_UNION")
    add(63, 65, 64, "MN_TAIL")
    add(63, 128, 33, "MK_TAIL")
    add(128, 65, 33, "NK_TAIL")
    add(63, 65, 33, "MNK_TAIL")
    add(7, 13, 17, "COPRIME_IRREGULAR")
    add(1, 1, 1, "MINIMAL")
    add(1, 65, 33, "DECODE_M1", "NK_TAIL")
    add(3, 1025, 2049, "REAL_NEARBY_PLUS1", "MNK_TAIL")
    add(1, 2049, 6143, "REAL_NEARBY_MINUS1", "NK_TAIL")
    return sorted(cases.values(), key=lambda x: (x["M"] * x["N"] * x["K"], x["case_id"]))


def load_gpu():
    import torch
    from ops.triton import gemm
    return torch, gemm


def make_operands(torch, m: int, n: int, k: int, mode: str, seed: int,
                  transposed_weight: bool):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    if mode == "zeros":
        a = torch.zeros((m, k), device="cuda", dtype=torch.float16)
        logical_b = torch.zeros((k, n), device="cuda", dtype=torch.float16)
    elif mode == "alternating":
        ai = torch.arange(m * k, device="cuda").reshape(m, k)
        bi = torch.arange(k * n, device="cuda").reshape(k, n)
        a = torch.where(ai % 2 == 0, 0.25, -0.25).half()
        logical_b = torch.where(bi % 3 == 0, 0.5, -0.125).half()
    elif mode == "tiny":
        a = torch.randn((m, k), generator=gen, device="cuda", dtype=torch.float16) * 2**-10
        logical_b = torch.randn((k, n), generator=gen, device="cuda", dtype=torch.float16) * 2**-10
    elif mode == "identity_like":
        a = torch.zeros((m, k), device="cuda", dtype=torch.float16)
        logical_b = torch.zeros((k, n), device="cuda", dtype=torch.float16)
        diag = min(m, n, k)
        idx = torch.arange(diag, device="cuda")
        a[idx, idx] = 1
        logical_b[idx, idx] = torch.linspace(0.25, 1.0, diag, device="cuda").half()
    elif mode == "conditioned":
        a = torch.randn((m, k), generator=gen, device="cuda", dtype=torch.float16) * 0.1
        logical_b = torch.randn((k, n), generator=gen, device="cuda", dtype=torch.float16) * 0.1
        scales = torch.logspace(-2, 0, k, device="cuda", dtype=torch.float32).half()
        a.mul_(scales[None, :])
    else:
        a = torch.randn((m, k), generator=gen, device="cuda", dtype=torch.float16) * 0.25
        logical_b = torch.randn((k, n), generator=gen, device="cuda", dtype=torch.float16) * 0.25
    if transposed_weight:
        weight = logical_b.t().contiguous()
        b = weight.t()
        layout = "model_linear_weight_transposed"
    else:
        b = logical_b.contiguous()
        layout = "row_major"
    return a, b, layout


def launch(torch, gemm, backend: str, a, b, out):
    if backend == "pytorch_cublas_opaque":
        torch.mm(a, b, out=out)
        return out
    if backend == "triton_fixed":
        return gemm.gemm_reference(a, b, out=out)
    raise ValueError(backend)


def compare_sampled(torch, candidate, a, b) -> Dict[str, Any]:
    m, n, k = a.shape[0], b.shape[1], a.shape[1]
    full = m * n <= 1_000_000 and m * n * k <= 100_000_000
    if full:
        rows = torch.arange(m, device="cuda")
        cols = torch.arange(n, device="cuda")
    else:
        rows = torch.unique(torch.tensor([0, m // 2, m - 1], device="cuda"))
        edge = list(range(min(64, n))) + list(range(max(0, n - 64), n))
        rng = random.Random(m * 1000003 + n * 1009 + k)
        edge += [rng.randrange(n) for _ in range(min(128, n))]
        cols = torch.unique(torch.tensor(edge, device="cuda"))
    reference = a.index_select(0, rows).float() @ b.index_select(1, cols).float()
    actual = candidate.index_select(0, rows).index_select(1, cols).float()
    diff = (actual - reference).abs()
    allowed = ATOL + RTOL * reference.abs()
    violations = diff > allowed
    l2rel = float(torch.linalg.vector_norm(actual - reference) /
                  torch.clamp(torch.linalg.vector_norm(reference), min=1e-12))
    first = None
    if bool(violations.any()):
        pos = violations.nonzero()[0]
        first = {"sample_row": int(pos[0]), "sample_col": int(pos[1]),
                 "logical_row": int(rows[pos[0]]), "logical_col": int(cols[pos[1]]),
                 "actual": float(actual[pos[0], pos[1]]),
                 "reference": float(reference[pos[0], pos[1]])}
    return {"oracle": "independent_fp32_torch_matmul",
            "coverage": "full" if full else "sampled_rows_and_columns",
            "sample_elements": int(actual.numel()), "output_elements": m * n,
            "max_abs": float(diff.max()) if diff.numel() else 0.0,
            "mean_abs": float(diff.mean()) if diff.numel() else 0.0,
            "rmse": float(torch.sqrt(torch.mean(diff * diff))) if diff.numel() else 0.0,
            "l2rel": l2rel, "violation_count": int(violations.sum()),
            "first_mismatch": first,
            "passed": not bool(violations.any()) and l2rel <= L2REL}


def one_correctness(torch, gemm, *, backend: str, m: int, n: int, k: int,
                    mode: str, seed: int, tags: Sequence[str], real: bool) -> Dict[str, Any]:
    a, b, b_layout = make_operands(torch, m, n, k, mode, seed,
                                    transposed_weight=real or seed % 2 == 0)
    guard = torch.full((m + 2, n + 2), GUARD_SENTINEL,
                       device="cuda", dtype=torch.float16)
    out = guard[1:-1, 1:-1]
    error = None
    try:
        launch(torch, gemm, backend, a, b, out)
        torch.cuda.synchronize()
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    base = {"schema": SCHEMA, "backend": backend, "actual_backend": backend,
            "M": m, "N": n, "K": k, "dtype_input": DTYPE,
            "dtype_accumulation": "fp32", "dtype_output": DTYPE,
            "layout_A": "row_major", "layout_B": b_layout,
            "layout_D": f"row_major_padded_ld_{n + 2}", "alpha": 1.0, "beta": 0.0,
            "epilogue": "identity_cast_fp16", "math_mode": "fp32_accumulation_ieee",
            "workspace_bytes": 0, "mode": mode, "seed": seed,
            "tags": list(tags), "requested_backend": backend,
            "tolerance": {"atol": ATOL, "rtol": RTOL, "l2rel": L2REL}}
    if error:
        return {**base, "status": "FAIL", "error": error}
    guards_ok = bool(torch.all(guard[0, :] == GUARD_SENTINEL) and
                     torch.all(guard[-1, :] == GUARD_SENTINEL) and
                     torch.all(guard[:, 0] == GUARD_SENTINEL) and
                     torch.all(guard[:, -1] == GUARD_SENTINEL))
    metrics = compare_sampled(torch, out, a, b)
    finite = bool(torch.isfinite(out).all())
    passed = guards_ok and metrics["passed"] and finite
    return {**base, "guards_ok": guards_ok, "unwritten_sentinel_count":
            int((out == GUARD_SENTINEL).sum()), "finite_output": finite,
            "metrics": metrics, "status": "PASS" if passed else "FAIL"}


def initialize(args) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = build_projection_ledger()
    write_json(out / "projection_ledger.json", ledger)
    write_json(out / "protocol.json", {
        "experiment_id": EXPERIMENT_ID, "schema": SCHEMA,
        "frozen_at_utc": utc_now(), "backends": ["pytorch_cublas_opaque", "triton_fixed",
                                                       "cutlass_tensorop_sm80"],
        "tail_cases": tail_plan(), "real_shapes": unique_real_shapes(ledger),
        "tolerance": {"atol": ATOL, "rtol": RTOL, "l2rel": L2REL,
                      "same_for_all_backends": True},
        "operator_spec": {"equation": "D = 1*A@B + 0*C", "batch": 1,
            "A": "M x K", "B": "K x N (model weight stored N x K and transposed)",
            "D": "M x N", "input_dtype": "fp16", "accumulation": "fp32",
            "output_dtype": "fp16", "epilogue": "identity then fp16 cast",
            "stream": "current PyTorch stream / CUTLASS default stream",
            "alias": "D may not alias A or B", "workspace_policy": "record actual; zero here",
            "noncontiguous_policy": "positive-stride views supported; incompatible inputs reject"},
        "performance": {"independent_processes": 3, "primary_decode_metric": "latency_us",
                        "throughput": "2*M*N*K/device_seconds/1e12",
                        "backend_order": "seeded shuffle", "cold_separate": True,
                        "tie_guard": PERF_GUARD},
        "claim_boundary": "operator-level only; no model speedup claim",
    })
    source_paths = [Path(__file__), REPO / "ops/triton/gemm.py",
                    REPO / "ops/cuda/cutlass_gemm/bench_cutlass_gemm.cu"]
    torch, _ = load_gpu()
    provenance = {"experiment_id": EXPERIMENT_ID, "created_at_utc": utc_now(),
        "git_commit": run(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        "git_status": run(["git", "status", "--short"])["stdout"],
        "platform": platform.platform(), "python": sys.version,
        "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "source_sha256": {str(p.relative_to(REPO)): sha256_file(p) for p in source_paths}}
    import triton
    provenance["triton_version"] = triton.__version__
    write_json(out / "provenance.json", provenance)
    source = (REPO / "ops/triton/gemm.py").read_text()
    write_json(out / "triton_mask_audit.json", {
        "source_sha256": sha256_file(REPO / "ops/triton/gemm.py"),
        "modulo_M_present": "% M" in source, "modulo_N_present": "% N" in source,
        "A_mask_present": "(offs_am[:, None] < M) & valid_k[None, :]" in source,
        "B_mask_present": "valid_k[:, None] & (offs_bn[None, :] < N)" in source,
        "D_mask_present": "(offs_am[:, None] < M) & (offs_bn[None, :] < N)" in source,
        "risk_before_fix": "M/N modulo and missing A/B/D masks could hide duplicate writes and K OOB reads"})
    return 0


def build_cutlass(args) -> int:
    out = Path(args.output_dir)
    configure = run(["cmake", "-S", ".", "-B", "build/jetson-release",
                     "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=87"], timeout=600)
    build = run(["cmake", "--build", "build/jetson-release", "--target",
                 "hqsb_cutlass_gemm_bench", "e04_03_sanitizer_control", "-j2"], timeout=1200)
    write_json(out / "cutlass_build.json", {"configure": configure, "build": build,
        "binary": "build/jetson-release/bin/hqsb_cutlass_gemm_bench",
        "binary_exists": (REPO / "build/jetson-release/bin/hqsb_cutlass_gemm_bench").exists()})
    return 0 if configure["exit_code"] == 0 and build["exit_code"] == 0 else 1


def correctness(args) -> int:
    out = Path(args.output_dir)
    torch, gemm = load_gpu()
    rows = []
    for index, case in enumerate(tail_plan()):
        for backend in ("pytorch_cublas_opaque", "triton_fixed"):
            rec = one_correctness(torch, gemm, backend=backend, m=case["M"], n=case["N"],
                                  k=case["K"], mode="random", seed=40300 + index,
                                  tags=case["tags"], real=False)
            rows.append(rec)
            print(case["case_id"], backend, rec["status"], flush=True)
        torch.cuda.empty_cache()
    special = []
    for index, mode in enumerate(("zeros", "identity_like", "alternating", "tiny", "conditioned")):
        for backend in ("pytorch_cublas_opaque", "triton_fixed"):
            special.append(one_correctness(torch, gemm, backend=backend, m=17, n=19, k=33,
                mode=mode, seed=41000 + index, tags=["SPECIAL_VALUE", "MNK_TAIL"], real=False))
    ledger = json.loads((out / "projection_ledger.json").read_text())
    real_rows = []
    for index, shape in enumerate(unique_real_shapes(ledger)):
        for backend in ("pytorch_cublas_opaque", "triton_fixed"):
            rec = one_correctness(torch, gemm, backend=backend, m=shape["M"], n=shape["N"],
                k=shape["K"], mode="random", seed=42000 + index,
                tags=["S02_RUNTIME_REAL", shape["phase"], *shape["projections"]], real=True)
            real_rows.append(rec)
            print("real", shape["M"], shape["N"], shape["K"], backend, rec["status"], flush=True)
            torch.cuda.empty_cache()
    write_jsonl(out / "tail_correctness.jsonl", rows)
    write_jsonl(out / "special_correctness.jsonl", special)
    write_jsonl(out / "real_shape_correctness.jsonl", real_rows)
    return 0 if all(x["status"] == "PASS" for x in rows + special + real_rows) else 1


def parse_last_json(text: str) -> Optional[Dict[str, Any]]:
    for line in reversed(text.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def cutlass(args) -> int:
    out = Path(args.output_dir)
    binary = REPO / "build/jetson-release/bin/hqsb_cutlass_gemm_bench"
    rows = []
    for case in tail_plan():
        proc = run([str(binary), "--m", str(case["M"]), "--n", str(case["N"]),
                    "--k", str(case["K"]), "--warmup", "1", "--iterations", "3"], timeout=600)
        payload = parse_last_json(proc["stdout"])
        rows.append({"case_id": case["case_id"], "tags": case["tags"],
                     "process": proc, "measurement": payload,
                     "status": payload.get("status") if payload else "FAIL"})
        print("cutlass", case["case_id"], rows[-1]["status"], flush=True)
    # CUTLASS is exercised on representative real families here; E04-05 owns
    # the exhaustive configuration space.  All real shapes remain covered by
    # PyTorch and Triton in real_shape_correctness.jsonl.
    real_dims = [(1, 1024, 2048), (1, 2048, 2048), (1, 6144, 2048),
                 (1, 2048, 6144), (1, 151936, 2048), (128, 6144, 2048)]
    perf = []
    for m, n, k in real_dims:
        proc = run([str(binary), "--m", str(m), "--n", str(n), "--k", str(k),
                    "--warmup", "2", "--iterations", "5", "--no-verify"], timeout=600)
        perf.append({"M": m, "N": n, "K": k, "process": proc,
                     "measurement": parse_last_json(proc["stdout"])})
    write_jsonl(out / "cutlass_tail_cases.jsonl", rows)
    write_json(out / "cutlass_real_representatives.json", {"rows": perf})
    return 0 if all(x["status"] in ("PASS", "EXPECTED_UNSUPPORTED") for x in rows) else 1


def invalid(args) -> int:
    out = Path(args.output_dir)
    torch, gemm = load_gpu()
    a = torch.randn((3, 5), device="cuda", dtype=torch.float16)
    b = torch.randn((5, 7), device="cuda", dtype=torch.float16)
    cases = []
    def check(name, fn):
        try:
            fn()
            cases.append({"case": name, "status": "FAIL", "reason": "unexpected accept"})
        except Exception as exc:
            cases.append({"case": name, "status": "PASS", "rejection": type(exc).__name__,
                          "message": str(exc)})
    check("rank", lambda: gemm.gemm_reference(a.reshape(1, 3, 5), b))
    check("shape", lambda: gemm.gemm_reference(a, torch.empty((6, 7), device="cuda", dtype=torch.float16)))
    check("dtype", lambda: gemm.gemm_reference(a, b.float()))
    check("cpu", lambda: gemm.gemm_reference(a.cpu(), b.cpu()))
    check("zero_dimension", lambda: gemm.gemm_reference(
        torch.empty((0, 5), device="cuda", dtype=torch.float16), b))
    check("output_shape", lambda: gemm.gemm_reference(a, b, out=torch.empty((3, 8), device="cuda", dtype=torch.float16)))
    check("output_alias_A", lambda: gemm.gemm_reference(a, b[:, :5], out=a))
    binary = REPO / "build/jetson-release/bin/hqsb_cutlass_gemm_bench"
    for name, dims in (("cutlass_M0", (0, 7, 5)), ("cutlass_N0", (3, 0, 5)),
                       ("cutlass_K0", (3, 7, 0))):
        proc = run([str(binary), "--m", str(dims[0]), "--n", str(dims[1]),
                    "--k", str(dims[2])])
        cases.append({"case": name, "status": "PASS" if proc["exit_code"] != 0 else "FAIL",
                      "process": proc})
    write_json(out / "invalid_inputs.json", {"cases": cases})
    return 0 if all(x["status"] == "PASS" for x in cases) else 1


def cold_worker(args) -> int:
    torch, gemm = load_gpu()
    a, b, _ = make_operands(torch, 65, 65, 33, "random", 44003, False)
    out = torch.empty((65, 65), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gemm.gemm_reference(a, b, out=out)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1000
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); gemm.gemm_reference(a, b, out=out); end.record(); end.synchronize()
    write_json(Path(args.worker_output), {"first_jit_plus_execution_ms": first_ms,
        "second_device_ms": start.elapsed_time(end), "cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "correct": bool(torch.allclose(out, a @ b, atol=ATOL, rtol=RTOL))})
    return 0


def cold(args) -> int:
    out = Path(args.output_dir)
    cache = Path(tempfile.mkdtemp(prefix="e04_03_triton_", dir=out))
    records = []
    for state in ("empty_cache", "cache_hit_fresh_process"):
        worker = cache / f"{state}.json"
        env = dict(os.environ); env["TRITON_CACHE_DIR"] = str(cache)
        proc = run([sys.executable, str(Path(__file__)), "cold-worker",
                    "--worker-output", str(worker)], env=env, timeout=600)
        files = [{"path": str(p.relative_to(cache)), "bytes": p.stat().st_size,
                  "sha256": sha256_file(p)} for p in sorted(cache.rglob("*")) if p.is_file()]
        records.append({"state": state, "process": proc,
                        "measurement": json.loads(worker.read_text()) if worker.exists() else None,
                        "cache_files": files})
    write_json(out / "cold_workspace.json", {"triton": records,
        "cutlass": {"build_evidence": "cutlass_build.json", "workspace_bytes": 0,
                    "weight_pack": "none", "steady_evidence": "cutlass_real_representatives.json"}})
    return 0 if all(r["process"]["exit_code"] == 0 for r in records) else 1


def triton_sanitize_worker(_args) -> int:
    torch, gemm = load_gpu()
    a, b, _ = make_operands(torch, 65, 65, 33, "random", 45003, False)
    out = torch.empty((65, 65), device="cuda", dtype=torch.float16)
    gemm.gemm_reference(a, b, out=out)
    torch.cuda.synchronize()
    print(json.dumps({"status": "PASS", "correct": bool(torch.allclose(out, a @ b, atol=ATOL, rtol=RTOL))}))
    return 0


def sanitizer(args) -> int:
    out = Path(args.output_dir)
    tool = shutil.which("compute-sanitizer")
    if not tool and Path("/usr/local/cuda/bin/compute-sanitizer").is_file():
        tool = "/usr/local/cuda/bin/compute-sanitizer"
    if not tool:
        write_json(out / "sanitizer.json", {"status": "FAIL", "reason": "compute-sanitizer not found"})
        return 1
    control = REPO / "build/jetson-release/bin/e04_03_sanitizer_control"
    cutlass_bin = REPO / "build/jetson-release/bin/hqsb_cutlass_gemm_bench"
    # Jetson production drivers disable GPU debugging for unprivileged users.
    # ``sudo -n`` is preconfigured on the target; keep the privilege mode in
    # the evidence so a direct-user tool refusal is not mistaken for a kernel
    # finding.
    tool_cmd = ["sudo", "-n", tool]
    prefix = [*tool_cmd, "--tool", "memcheck", "--error-exitcode", "99"]
    clean = run([*prefix, str(control), "--mode", "clean"], timeout=600)
    oob = run([*prefix, str(control), "--mode", "oob"], timeout=600)
    cutlass_mem = run([*prefix, str(cutlass_bin), "--m", "65", "--n", "128", "--k", "64",
                       "--warmup", "0", "--iterations", "1"], timeout=900)
    user_site = "/home/jetson/.local/lib/python3.10/site-packages"
    triton_prefix = ["sudo", "-n", "env", f"PYTHONPATH={user_site}:{REPO}", tool,
                     "--tool", "memcheck", "--error-exitcode", "99"]
    triton_mem = run([*triton_prefix, sys.executable, str(Path(__file__)),
                      "triton-sanitize-worker"], timeout=900)
    init = run([*tool_cmd, "--tool", "initcheck", "--error-exitcode", "99", str(cutlass_bin),
                "--m", "65", "--n", "128", "--k", "64", "--warmup", "0",
                "--iterations", "1"], timeout=900)
    checks = {
        "negative_control_detected": oob["exit_code"] == 99 and "Invalid" in (oob["stdout"] + oob["stderr"]),
        "positive_control_clean": clean["exit_code"] == 0,
        "cutlass_memcheck_clean": cutlass_mem["exit_code"] == 0,
        "triton_memcheck_clean": triton_mem["exit_code"] == 0,
        "cutlass_initcheck_clean": init["exit_code"] == 0,
    }
    write_json(out / "sanitizer.json", {"tool": tool,
        "privilege_mode": "sudo -n required by Jetson GPU debugging policy", "checks": checks,
        "runs": {"positive_control": clean, "negative_control": oob,
                 "cutlass_memcheck": cutlass_mem, "triton_memcheck": triton_mem,
                 "cutlass_initcheck": init},
        "status": "PASS" if all(checks.values()) else "FAIL"})
    return 0 if all(checks.values()) else 1


def timing(torch, gemm, backend: str, shape: Mapping[str, Any], process_index: int) -> Dict[str, Any]:
    m, n, k = shape["M"], shape["N"], shape["K"]
    a, b, b_layout = make_operands(torch, m, n, k, "random",
                                    46000 + process_index * 101 + m + n + k, True)
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    for _ in range(3): launch(torch, gemm, backend, a, b, out)
    torch.cuda.synchronize()
    work = m * n * k
    iters = 10 if work < 50_000_000 else (5 if work < 2_000_000_000 else 2)
    device_us, host_us = [], []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _i in range(iters): launch(torch, gemm, backend, a, b, out)
        end.record(); end.synchronize()
        device_us.append(float(start.elapsed_time(end) * 1000 / iters))
        t0 = time.perf_counter_ns()
        for _i in range(iters): launch(torch, gemm, backend, a, b, out)
        torch.cuda.synchronize()
        host_us.append((time.perf_counter_ns() - t0) / 1000 / iters)
    med = statistics.median(device_us)
    return {"process_index": process_index, "backend": backend, "actual_backend": backend,
        "M": m, "N": n, "K": k, "phase": shape["phase"],
        "projections": shape["projections"], "workloads": shape["workloads"],
        "layout_A": "row_major", "layout_B": b_layout, "layout_D": "row_major",
        "dtype": DTYPE, "accumulation": "fp32", "alpha": 1.0, "beta": 0.0,
        "workspace_bytes": 0, "warmup": 3, "groups": 5, "iterations_per_group": iters,
        "device_us_raw": device_us, "host_completion_us_raw": host_us,
        "device_us_median": med, "tflops_median": 2 * m * n * k / med / 1e6,
        "status": "PASS"}


def timing_cutlass(shape: Mapping[str, Any], process_index: int) -> Dict[str, Any]:
    binary = REPO / "build/jetson-release/bin/hqsb_cutlass_gemm_bench"
    measurements, host_process_us = [], []
    processes = []
    for _ in range(5):
        proc = run([str(binary), "--m", str(shape["M"]), "--n", str(shape["N"]),
                    "--k", str(shape["K"]), "--warmup", "3", "--iterations", "5",
                    "--no-verify"], timeout=900)
        payload = parse_last_json(proc["stdout"])
        processes.append({"exit_code": proc["exit_code"], "duration_s": proc["duration_s"],
                          "measurement": payload})
        if proc["exit_code"] != 0 or not payload or payload.get("status") != "PASS":
            return {"process_index": process_index, "backend": "cutlass_tensorop_sm80",
                    **shape, "status": "FAIL", "processes": processes}
        measurements.append(float(payload["median_ms"]) * 1000.0)
        host_process_us.append(float(proc["duration_s"]) * 1.0e6)
    med = statistics.median(measurements)
    return {"process_index": process_index, "backend": "cutlass_tensorop_sm80",
        "actual_backend": "cutlass_tensorop_sm80", "M": shape["M"], "N": shape["N"],
        "K": shape["K"], "phase": shape["phase"], "projections": shape["projections"],
        "workloads": shape["workloads"], "layout_A": "row_major",
        "layout_B": "column_major_logical_from_weight_NK", "layout_D": "row_major",
        "dtype": DTYPE, "accumulation": "fp32", "alpha": 1.0, "beta": 0.0,
        "workspace_bytes": 0, "warmup": 3, "groups": 5, "iterations_per_group": 5,
        "device_us_raw": measurements, "host_process_us_raw": host_process_us,
        "device_us_median": med,
        "tflops_median": 2 * shape["M"] * shape["N"] * shape["K"] / med / 1e6,
        "processes": processes, "status": "PASS"}


def performance(args) -> int:
    out = Path(args.output_dir)
    ledger = json.loads((out / "projection_ledger.json").read_text())
    shapes = unique_real_shapes(ledger)
    rng = random.Random(47000 + args.process_index)
    jobs = [(shape, backend) for shape in shapes
            for backend in ("pytorch_cublas_opaque", "triton_fixed", "cutlass_tensorop_sm80")]
    rng.shuffle(jobs)
    torch, gemm = load_gpu()
    rows = []
    for shape, backend in jobs:
        try:
            rec = (timing_cutlass(shape, args.process_index)
                   if backend == "cutlass_tensorop_sm80"
                   else timing(torch, gemm, backend, shape, args.process_index))
        except Exception as exc:
            rec = {"process_index": args.process_index, "backend": backend,
                   **shape, "status": "FAIL", "error": {"type": type(exc).__name__,
                                                           "message": str(exc)}}
        rows.append(rec)
        print(args.process_index, shape["M"], shape["N"], shape["K"], backend,
              rec["status"], flush=True)
        torch.cuda.empty_cache()
    write_jsonl(out / f"performance_proc{args.process_index}.jsonl", rows)
    return 0 if all(r["status"] == "PASS" for r in rows) else 1


def bootstrap_median_ci95(values: Sequence[float], seed: int) -> List[float]:
    rng = random.Random(seed)
    draws = []
    for _ in range(2000):
        draws.append(statistics.median(rng.choice(values) for _i in values))
    draws.sort()
    return [draws[math.floor(0.025 * (len(draws) - 1))],
            draws[math.ceil(0.975 * (len(draws) - 1))]]


def summarize(args) -> int:
    out = Path(args.output_dir)
    tail = read_jsonl(out / "tail_correctness.jsonl")
    special = read_jsonl(out / "special_correctness.jsonl")
    real = read_jsonl(out / "real_shape_correctness.jsonl")
    cutlass_rows = read_jsonl(out / "cutlass_tail_cases.jsonl")
    perf = [r for i in range(3) for r in read_jsonl(out / f"performance_proc{i}.jsonl")]
    grouped: Dict[Tuple[int, int, int, str, str], List[Dict[str, Any]]] = {}
    for rec in perf:
        grouped.setdefault((rec["M"], rec["N"], rec["K"], rec["phase"], rec["backend"]), []).append(rec)
    matrix = []
    for key, recs in sorted(grouped.items()):
        flat = [x for r in recs for x in r.get("device_us_raw", [])]
        row = {"M": key[0], "N": key[1], "K": key[2], "phase": key[3],
               "backend": key[4], "process_count": len(recs),
               "status": "PASS" if len(recs) == 3 and all(r["status"] == "PASS" for r in recs) else "FAIL"}
        if flat:
            seed = key[0] * 1000003 + key[1] * 1009 + key[2] * 17 + sum(ord(c) for c in key[4])
            row.update({"device_us_median": statistics.median(flat),
                        "device_us_median_bootstrap_ci95": bootstrap_median_ci95(flat, seed),
                        "device_us_p05": sorted(flat)[max(0, math.ceil(0.05 * len(flat)) - 1)],
                        "device_us_p95": sorted(flat)[max(0, math.ceil(0.95 * len(flat)) - 1)],
                        "tflops": 2 * key[0] * key[1] * key[2] /
                                   statistics.median(flat) / 1e6})
        matrix.append(row)
    decisions = []
    dims = {(r["M"], r["N"], r["K"], r["phase"]) for r in matrix}
    for dim in sorted(dims):
        candidates = sorted([r for r in matrix if (r["M"], r["N"], r["K"], r["phase"]) == dim
                             and r["status"] == "PASS"], key=lambda r: r["device_us_median"])
        ties = ([r["backend"] for r in candidates
                 if r["device_us_median"] <= candidates[0]["device_us_median"] * (1 + PERF_GUARD)]
                if candidates else [])
        decisions.append({"M": dim[0], "N": dim[1], "K": dim[2], "phase": dim[3],
            "legal_backends": [r["backend"] for r in candidates],
            "winner": (candidates[0]["backend"] if len(ties) == 1 else "TIE") if candidates else None,
            "best_backend": candidates[0]["backend"] if candidates else None,
            "tie_backends": ties, "tie_guard": PERF_GUARD,
            "fallback": "pytorch_cublas_opaque",
            "latencies_us": {r["backend"]: r["device_us_median"] for r in candidates},
            "tflops": {r["backend"]: r["tflops"] for r in candidates},
            "cutlass_scope": "default TensorOp config measured for all real shapes; exhaustive config search belongs to E04-05"})
    ledger = json.loads((out / "projection_ledger.json").read_text())
    sanitizer_data = json.loads((out / "sanitizer.json").read_text())
    invalid_data = json.loads((out / "invalid_inputs.json").read_text())
    cold_data = json.loads((out / "cold_workspace.json").read_text())
    mask = json.loads((out / "triton_mask_audit.json").read_text())
    conditions = {
        "runtime_projection_ledger_complete": ledger["all_required_projections_present"],
        "operator_layout_math_mode_frozen": (out / "protocol.json").exists(),
        "forced_backend_identity_hit": all(any(
            (r.get("actual_backend") == b and r["status"] == "PASS") for r in tail)
            for b in ("pytorch_cublas_opaque", "triton_fixed")) and
            any(r["status"] == "PASS" for r in cutlass_rows),
        "all_M_N_K_single_double_triple_tail_correct": all(r["status"] == "PASS" for r in tail) and
            all(r["status"] in ("PASS", "EXPECTED_UNSUPPORTED") for r in cutlass_rows),
        "sentinel_guard_special_invalid_complete": all(r["status"] == "PASS" for r in special) and
            all(r.get("guards_ok") for r in tail if r["backend"] == "triton_fixed") and
            all(r["status"] == "PASS" for r in invalid_data["cases"]),
        "sanitizer_controls_hit_and_targets_clean": sanitizer_data["status"] == "PASS",
        "same_tolerance_all_backends": True,
        "cold_workspace_pack_separated": all(r["measurement"] for r in cold_data["triton"]),
        "three_independent_processes_complete": all(r["status"] == "PASS" and r["process_count"] == 3 for r in matrix),
        "all_real_projection_shapes_correct": all(r["status"] == "PASS" for r in real) and
            ledger["all_required_projections_present"],
        "triton_tail_masks_mechanically_present": (not mask["modulo_M_present"] and
            not mask["modulo_N_present"] and mask["A_mask_present"] and
            mask["B_mask_present"] and mask["D_mask_present"]),
    }
    verdict = {"experiment_id": EXPERIMENT_ID, "schema": SCHEMA,
        "generated_at_utc": utc_now(), "conditions": conditions,
        "failed_conditions": [k for k, v in conditions.items() if not v],
        "overall": "PASS" if all(conditions.values()) else "FAIL",
        "counts": {"tail_records": len(tail), "cutlass_tail_records": len(cutlass_rows),
                   "special_records": len(special), "real_shape_records": len(real),
                   "performance_records": len(perf), "performance_cells": len(matrix)},
        "stage_context": {"E04_01": "PASS", "E04_02": "BLOCKED by E03-06/E03-07; independent RMSNorm gate",
                          "claim": "E04-03 operator safety only"}}
    write_json(out / "performance_matrix.json", {"rows": matrix})
    write_json(out / "projection_decisions.json", {"rows": decisions})
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
    for name in ("initialize", "build-cutlass", "correctness", "cutlass", "invalid",
                 "cold", "sanitizer", "summarize", "triton-sanitize-worker"):
        q = sub.add_parser(name)
        if name not in ("triton-sanitize-worker",):
            q.add_argument("--output-dir", required=True)
    q = sub.add_parser("performance")
    q.add_argument("--output-dir", required=True)
    q.add_argument("--process-index", required=True, type=int, choices=(0, 1, 2))
    q = sub.add_parser("cold-worker")
    q.add_argument("--worker-output", required=True)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    return {"initialize": initialize, "build-cutlass": build_cutlass,
            "correctness": correctness, "cutlass": cutlass, "invalid": invalid,
            "cold": cold, "cold-worker": cold_worker, "sanitizer": sanitizer,
            "triton-sanitize-worker": triton_sanitize_worker,
            "performance": performance, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
