#!/usr/bin/env python3
"""E04-08 production dispatcher auto/forced replay and stage-gate audit.

This script audits the dispatcher that is actually imported from ``ops``.  It
does not implement a shadow dispatcher in the test harness: missing APIs and
missing route metadata are evidence, not features to emulate.  Expected
decisions are frozen independently from the production helper and runtime
results are wrapped in C6 BenchmarkResult envelopes for offline replay.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
EXPERIMENT_ID = "E04-08"
SCHEMA = "hqsb.s04.dispatcher_replay/v1"
POLICY_VERSION = "e04-08-audit-current-production-v1"
REGRET_GUARD = 0.05


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")


def run(argv: Sequence[str], timeout: int = 300) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(list(argv), cwd=REPO, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout, check=False)
        return {"argv": list(argv), "exit_code": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr,
                "duration_s": time.perf_counter() - start, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        return {"argv": list(argv), "exit_code": 124, "stdout": exc.stdout or "",
                "stderr": exc.stderr or "", "duration_s": time.perf_counter() - start,
                "timed_out": True}


def git(*args: str) -> str:
    return run(["git", *args], 30)["stdout"].strip()


def relative(path: Path) -> str:
    return str(path.relative_to(REPO))


def source_inventory() -> Dict[str, Any]:
    paths = {
        "dispatcher": REPO / "ops/dispatcher.py",
        "capability": REPO / "ops/capability.py",
        "cuda_bridge": REPO / "ops/cuda_bridge.py",
        "triton_rmsnorm": REPO / "ops/triton/rmsnorm.py",
        "triton_gemm": REPO / "ops/triton/gemm.py",
        "c6_contract": REPO / "hqsb/core/contracts/result.py",
    }
    return {name: {"path": relative(path), "bytes": path.stat().st_size,
                   "sha256": sha256_file(path)} for name, path in paths.items()}


def inherited_evidence() -> Dict[str, Any]:
    rows = []
    for number in range(1, 8):
        experiment = f"E04-{number:02d}"
        path = REPO / f"docs/stage_experiments/S04/{experiment}/raw/verdict.json"
        if path.exists():
            value = read_json(path)
            rows.append({"experiment_id": experiment, "path": relative(path),
                         "sha256": sha256_file(path), "overall": value.get("overall"),
                         "failed_conditions": value.get("failed_conditions", [])})
        else:
            rows.append({"experiment_id": experiment, "path": relative(path),
                         "sha256": None, "overall": "MISSING",
                         "failed_conditions": ["verdict missing"]})
    return {"rows": rows, "all_pass": all(row["overall"] == "PASS" for row in rows),
            "blocked_or_failed": [row["experiment_id"] for row in rows
                                  if row["overall"] != "PASS"]}


def independent_expected_table() -> Dict[str, Any]:
    common = {
        "device": {"id": 0, "arch": "sm87", "uuid": "runtime-captured"},
        "dtype": "case-specific", "layout": "contiguous_row_major",
        "alignment": "case-specific", "stream": "explicit_current_stream",
        "workspace_budget_bytes": "case-specific", "cold_policy": "steady",
        "policy_version": POLICY_VERSION, "spec_version": "E03-01-frozen",
    }
    cases = [
        ("rms_auto_aligned", "rmsnorm", "AUTO", {"rows": 7, "H": 128}, "cuda.v2"),
        ("rms_auto_odd_tail", "rmsnorm", "AUTO", {"rows": 7, "H": 129}, "cuda.v1_or_safe"),
        ("rms_force_cuda_v0", "rmsnorm", "FORCE_CUDA_V0", {"rows": 7, "H": 128}, "cuda.v0"),
        ("rms_force_cuda_v1", "rmsnorm", "FORCE_CUDA_V1", {"rows": 7, "H": 128}, "cuda.v1"),
        ("rms_force_cuda_v2", "rmsnorm", "FORCE_CUDA_V2", {"rows": 7, "H": 128}, "cuda.v2"),
        ("rms_force_triton_ref", "rmsnorm", "FORCE_TRITON_REF", {"rows": 7, "H": 128}, "triton.reference"),
        ("rms_force_triton_tuned", "rmsnorm", "FORCE_TRITON_TUNED", {"rows": 7, "H": 128}, "triton.tuned"),
        ("gemm_auto_vendor", "gemm", "AUTO", {"M": 16, "N": 1024, "K": 2048}, "cublas"),
        ("gemm_force_cublas", "gemm", "FORCE_CUBLAS", {"M": 17, "N": 130, "K": 258}, "cublas"),
        ("gemm_force_triton", "gemm", "FORCE_TRITON_FIXED", {"M": 17, "N": 130, "K": 258}, "triton.fixed"),
        ("gemm_force_cutlass", "gemm", "FORCE_CUTLASS", {"M": 17, "N": 130, "K": 258}, "cutlass"),
        ("unknown_dtype", "rmsnorm", "AUTO", {"rows": 7, "H": 128, "dtype": "bf16"}, "REJECT_OR_SAFE_REFERENCE"),
        ("unknown_shape", "gemm", "AUTO", {"M": 5, "N": 37, "K": 73}, "SAFE_FALLBACK"),
        ("workspace_reject", "gemm", "AUTO", {"M": 1024, "N": 6144, "K": 2048,
                                                 "workspace_budget_bytes": 0}, "NON_WORKSPACE_BACKEND"),
        ("runtime_async_failure", "rmsnorm", "AUTO", {"rows": 7, "H": 128}, "ERROR_NO_FALLBACK"),
    ]
    output = []
    for case_id, operator, requested, shape, actual in cases:
        key = dict(common)
        key.update({"operator": operator, "shape": shape})
        output.append({"case_id": case_id, "dispatch_key": key, "requested": requested,
                       "expected_actual": actual, "expected_forced_fallback": False,
                       "oracle_source": "frozen E04-01..07 evidence plus E04-08 protocol"})
    payload = {"schema": f"{SCHEMA}/expected-table", "independent_from_production": True,
               "cases": output}
    payload["table_sha256"] = digest(payload)
    return payload


def initialize(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dispatcher = (REPO / "ops/dispatcher.py").read_text()
    policy = {
        "schema": f"{SCHEMA}/policy", "policy_version": POLICY_VERSION,
        "frozen_at_utc": utc_now(), "online_benchmark": False, "regret_guard": REGRET_GUARD,
        "tie_policy": "prefer fewer dependencies, lower cold cost, then stable vendor/reference",
        "unknown_policy": "reject unsupported semantics; otherwise safe registered reference",
        "forced_policy": "forced requests never fallback",
        "async_failure_policy": "report error; never retry on a possibly polluted output/context",
        "source_evidence": ["E04-01 capability", "E04-02 RMSNorm", "E04-03 GEMM",
                            "E04-04 regime/holdout", "E04-05 CUTLASS",
                            "E04-06 Triton cache", "E04-07 causal analysis"],
    }
    registry = {
        "schema": f"{SCHEMA}/registry", "registry_version": POLICY_VERSION,
        "declared_or_evidenced_backends": [
            {"id": "reference.rmsnorm", "operator": "rmsnorm", "forced_required": True},
            {"id": "cuda.v0", "operator": "rmsnorm", "forced_required": True},
            {"id": "cuda.v1", "operator": "rmsnorm", "forced_required": True},
            {"id": "cuda.v2", "operator": "rmsnorm", "forced_required": True},
            {"id": "triton.reference", "operator": "rmsnorm", "forced_required": True},
            {"id": "triton.tuned", "operator": "rmsnorm", "forced_required": True},
            {"id": "cublas", "operator": "gemm", "forced_required": True},
            {"id": "triton.fixed", "operator": "gemm", "forced_required": True},
            {"id": "cutlass", "operator": "gemm", "forced_required": True},
        ],
        "production_api_observation": {
            "cutlass_token_present": "cutlass" in dispatcher.lower(),
            "requested_token_present": "requested" in dispatcher,
            "candidate_evaluations_present": "candidate_evaluations" in dispatcher,
            "fallback_chain_present": "fallback_chain" in dispatcher,
        },
    }
    required_key_tokens = ("rows", "hidden", "M", "N", "K", "dtype", "layout", "stride",
                           "alignment", "stream", "workspace", "device", "build", "spec",
                           "policy", "cache")
    source_audit = {
        "schema": f"{SCHEMA}/source-audit", "source_identity": source_inventory(),
        "required_key_fields": {token: token in dispatcher for token in required_key_tokens},
        "capability_policy_execution_separated": False,
        "dispatch_result_fields": {name: name in dispatcher for name in
                                   ("requested", "actual", "candidate_evaluations",
                                    "fallback_chain", "score_source", "cache_identity",
                                    "workspace", "stream", "status", "evidence_refs")},
        "forced_api_in_signature": False,
        "versioned_policy_in_production": "policy_version" in dispatcher,
        "route_cache_in_production": "cache_identity" in dispatcher,
        "c6_emission_in_production": "BenchmarkResult" in dispatcher,
    }
    write_json(out / "protocol.json", policy)
    write_json(out / "inherited_evidence.json", inherited_evidence())
    write_json(out / "backend_registry.json", registry)
    write_json(out / "expected_decision_table.json", independent_expected_table())
    write_json(out / "source_contract_audit.json", source_audit)
    write_json(out / "provenance.json", {
        "schema": f"{SCHEMA}/provenance", "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": utc_now(), "hostname": platform.node(),
        "platform": platform.platform(), "python": platform.python_version(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    })
    return 0


def _record_error(case_id: str, exc: BaseException) -> Dict[str, Any]:
    return {"case_id": case_id, "status": "ERROR", "error_type": type(exc).__name__,
            "error": str(exc), "traceback_tail": traceback.format_exc().splitlines()[-8:]}


def _allclose(torch, actual, expected, dtype: str) -> Dict[str, Any]:
    diff = (actual.float() - expected.float()).abs()
    atol, rtol = ((3e-2, 3e-2) if dtype == "fp16" else (3e-5, 3e-5))
    passed = bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol))
    return {"passed": passed, "atol": atol, "rtol": rtol,
            "max_abs": float(diff.max().item()),
            "output_hash": hashlib.sha256(actual.detach().cpu().numpy().tobytes()).hexdigest()}


def _cuda_timing(torch, fn, iterations: int = 30) -> Dict[str, Any]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    host_start = time.perf_counter_ns()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    host_us = (time.perf_counter_ns() - host_start) / 1000.0 / iterations
    return {"iterations": iterations, "device_us": float(start.elapsed_time(end) * 1000 / iterations),
            "host_completion_us": host_us}


def _direct_backend_rows(torch, process_index: int) -> List[Dict[str, Any]]:
    from ops.reference import rmsnorm_torch
    from ops.cuda_bridge import rmsnorm_forward as cuda_forward
    from ops.triton import gemm as triton_gemm
    from ops.triton import rmsnorm as triton_rms

    torch.manual_seed(40800 + process_index)
    rows: List[Dict[str, Any]] = []
    x = torch.randn((7, 128), device="cuda", dtype=torch.float32)
    weight = torch.randn((128,), device="cuda", dtype=torch.float32)
    reference = rmsnorm_torch(x, weight)
    backends = {
        "reference.rmsnorm": lambda: rmsnorm_torch(x, weight),
        "cuda.v0": lambda: cuda_forward(x, weight, dtype="fp32", variant="v0_shared"),
        "cuda.v1": lambda: cuda_forward(x, weight, dtype="fp32", variant="v1_warp_shuffle"),
        "cuda.v2": lambda: cuda_forward(x, weight, dtype="fp32", variant="v2_vectorized_strict"),
        "triton.reference": lambda: triton_rms.rmsnorm_reference(x, weight),
        "triton.tuned": lambda: triton_rms.rmsnorm_optimized(x, weight),
    }
    for backend, fn in backends.items():
        try:
            actual = fn()
            torch.cuda.synchronize()
            check = _allclose(torch, actual, reference, "fp32")
            rows.append({"case_id": f"direct_{backend}", "operator": "rmsnorm",
                         "requested_backend": backend, "actual_backend": backend,
                         "via_production_dispatcher": False, "status": "PASS" if check["passed"] else "FAIL",
                         "correctness": check})
        except Exception as exc:
            rows.append({**_record_error(f"direct_{backend}", exc), "operator": "rmsnorm",
                         "requested_backend": backend, "actual_backend": None,
                         "via_production_dispatcher": False})

    a = torch.randn((17, 258), device="cuda", dtype=torch.float16)
    b = torch.randn((258, 130), device="cuda", dtype=torch.float16)
    gemm_reference = a @ b
    for backend, fn in {
        "cublas": lambda: a @ b,
        "triton.fixed": lambda: triton_gemm.gemm_reference(a, b),
    }.items():
        try:
            actual = fn()
            torch.cuda.synchronize()
            check = _allclose(torch, actual, gemm_reference, "fp16")
            rows.append({"case_id": f"direct_{backend}", "operator": "gemm",
                         "requested_backend": backend, "actual_backend": backend,
                         "via_production_dispatcher": False, "status": "PASS" if check["passed"] else "FAIL",
                         "correctness": check})
        except Exception as exc:
            rows.append({**_record_error(f"direct_{backend}", exc), "operator": "gemm",
                         "requested_backend": backend, "actual_backend": None,
                         "via_production_dispatcher": False})
    return rows


def _cutlass_direct(process_index: int) -> Dict[str, Any]:
    if process_index != 0:
        return {"case_id": "direct_cutlass", "status": "SKIP_DUPLICATE_PROCESS",
                "via_production_dispatcher": False}
    binary = REPO / "build/e04-05-cost-v2/bin/e04_05_cutlass_space"
    config = "small_m64n64k32_w32n32_s2_a1_linear"
    if not binary.exists():
        return {"case_id": "direct_cutlass", "status": "BINARY_MISSING",
                "binary": relative(binary), "via_production_dispatcher": False}
    proc = run([str(binary), "--config", config, "--m", "17", "--n", "130",
                "--k", "258", "--warmup", "2", "--iterations", "3", "--verify"], 300)
    parsed = None
    for line in reversed(proc["stdout"].splitlines()):
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                pass
            break
    passed = proc["exit_code"] == 0 and parsed is not None and parsed.get("status") == "PASS"
    return {"case_id": "direct_cutlass", "operator": "gemm", "requested_backend": "cutlass",
            "actual_backend": "cutlass" if passed else None, "config": config,
            "binary": relative(binary), "binary_sha256": sha256_file(binary),
            "via_production_dispatcher": False, "status": "PASS" if passed else "FAIL",
            "measurement": parsed, "process": proc}


def _mock_capability_rows() -> List[Dict[str, Any]]:
    from ops.capability import BackendCapabilities
    from ops.dispatcher import OperatorDispatcher

    base = dict(cuda_available=True, device_capability=(8, 7), triton_available=True,
                triton_version="runtime", cutlass_available=True,
                cutlass_include_dir="third_party/cutlass/include", tilelang_available=False,
                tilelang_version=None, cublas_available=True, cuda_rmsnorm_available=True,
                cuda_rmsnorm_lib="runtime", notes=())
    cases = [
        ("all_available", {}, ("cuda", "cublas")),
        ("cuda_rms_missing", {"cuda_rmsnorm_available": False, "cuda_rmsnorm_lib": None},
         ("triton", "cublas")),
        ("arch_mismatch", {"device_capability": (8, 0)}, ("triton", "cublas")),
        ("triton_missing", {"triton_available": False}, ("cuda", "cublas")),
        ("cublas_missing", {"cublas_available": False}, ("cuda", "triton")),
        ("accelerators_missing", {"cuda_rmsnorm_available": False, "cuda_rmsnorm_lib": None,
                                  "triton_available": False, "cublas_available": False},
         ("torch", "torch")),
    ]
    rows = []
    for case_id, changes, expected in cases:
        values = dict(base); values.update(changes)
        dispatcher = OperatorDispatcher(BackendCapabilities(**values))
        rms = dispatcher.select_rmsnorm("fp32", 128)
        gemm = dispatcher.select_gemm("fp16")
        actual = (rms.backend, gemm.backend)
        rows.append({"case_id": case_id, "expected": list(expected), "actual": list(actual),
                     "status": "PASS" if actual == expected else "FAIL",
                     "rmsnorm": rms.as_dict(), "gemm": gemm.as_dict(),
                     "selection_only": True})
    return rows


def _thread_case(index: int) -> Dict[str, Any]:
    import torch
    from ops.dispatcher import OperatorDispatcher
    from ops.reference import rmsnorm_torch

    dispatcher = OperatorDispatcher()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.manual_seed(40900 + index)
        x = torch.randn((7 + index, 128), device="cuda", dtype=torch.float32)
        w = torch.randn((128,), device="cuda", dtype=torch.float32)
        expected = rmsnorm_torch(x, w)
        actual, decision = dispatcher.run_rmsnorm(x, w)
    stream.synchronize()
    check = _allclose(torch, actual, expected, "fp32")
    return {"thread": index, "stream": int(stream.cuda_stream), "decision": decision.as_dict(),
            "status": "PASS" if check["passed"] else "FAIL", "correctness": check}


def runtime(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    import torch
    from hqsb.core.contracts.result import BenchmarkResult, CorrectnessReport, EnvironmentInfo
    from ops.capability import detect_capabilities
    from ops.dispatcher import OperatorDispatcher
    from ops.reference import rmsnorm_torch

    process_index = args.process_index
    torch.manual_seed(40800 + process_index)
    capabilities = detect_capabilities()
    dispatcher = OperatorDispatcher(capabilities)
    auto_rows: List[Dict[str, Any]] = []
    x = torch.randn((7, 128), device="cuda", dtype=torch.float32)
    w = torch.randn((128,), device="cuda", dtype=torch.float32)
    try:
        expected = rmsnorm_torch(x, w)
        actual, decision = dispatcher.run_rmsnorm(x, w)
        torch.cuda.synchronize()
        check = _allclose(torch, actual, expected, "fp32")
        auto_rows.append({"case_id": "rms_auto_actual", "requested": "AUTO",
                          "actual": decision.backend, "decision": decision.as_dict(),
                          "dispatch_key_observed": {"dtype": "fp32", "hidden": 128},
                          "missing_key_fields": ["rows", "layout", "alignment", "stream",
                                                 "device_identity", "binary", "policy", "cache"],
                          "status": "PASS" if check["passed"] else "FAIL", "correctness": check})
    except Exception as exc:
        auto_rows.append(_record_error("rms_auto_actual", exc))

    a = torch.randn((17, 258), device="cuda", dtype=torch.float16)
    b = torch.randn((258, 130), device="cuda", dtype=torch.float16)
    try:
        expected = a @ b
        actual, decision = dispatcher.run_gemm(a, b)
        torch.cuda.synchronize()
        check = _allclose(torch, actual, expected, "fp16")
        auto_rows.append({"case_id": "gemm_auto_actual", "requested": "AUTO",
                          "actual": decision.backend, "decision": decision.as_dict(),
                          "dispatch_key_observed": {"dtype": "fp16"},
                          "missing_key_fields": ["M", "N", "K", "layout", "alignment", "stream",
                                                 "workspace", "device_identity", "binary", "policy", "cache"],
                          "status": "PASS" if check["passed"] else "FAIL", "correctness": check})
    except Exception as exc:
        auto_rows.append(_record_error("gemm_auto_actual", exc))

    try:
        xb = x.to(torch.bfloat16); wb = w.to(torch.bfloat16)
        dispatcher.run_rmsnorm(xb, wb)
        torch.cuda.synchronize()
        auto_rows.append({"case_id": "unknown_bf16", "requested": "AUTO",
                          "status": "UNEXPECTED_ACCEPT"})
    except Exception as exc:
        auto_rows.append({"case_id": "unknown_bf16", "requested": "AUTO",
                          "status": "EXPECTED_REJECT", "error_type": type(exc).__name__,
                          "error": str(exc), "selection_preclassified_dtype": "fp32"})

    forced = []
    rms_signature = str(inspect.signature(dispatcher.run_rmsnorm))
    gemm_signature = str(inspect.signature(dispatcher.run_gemm))
    for backend in ("reference.rmsnorm", "cuda.v0", "cuda.v1", "cuda.v2",
                    "triton.reference", "triton.tuned", "cublas", "triton.fixed", "cutlass"):
        forced.append({"backend": backend, "dispatcher_api_available": False,
                       "status": "FORCED_API_MISSING", "fallback_observed": None,
                       "rmsnorm_signature": rms_signature, "gemm_signature": gemm_signature})

    direct = _direct_backend_rows(torch, process_index)
    direct.append(_cutlass_direct(process_index))
    mocks = _mock_capability_rows()

    overhead = []
    from ops.cuda_bridge import rmsnorm_forward as cuda_forward
    pure_rms = lambda: cuda_forward(x, w, dtype="fp32", variant="v2_vectorized_strict")
    auto_rms = lambda: dispatcher.run_rmsnorm(x, w)[0]
    pure_gemm = lambda: a @ b
    auto_gemm = lambda: dispatcher.run_gemm(a, b)[0]
    for operator, pure, wrapped in (("rmsnorm", pure_rms, auto_rms), ("gemm", pure_gemm, auto_gemm)):
        try:
            pure_time = _cuda_timing(torch, pure)
            auto_time = _cuda_timing(torch, wrapped)
            overhead.append({"operator": operator, "process_index": process_index,
                             "pure": pure_time, "auto": auto_time,
                             "device_overhead_us": auto_time["device_us"] - pure_time["device_us"],
                             "host_overhead_us": auto_time["host_completion_us"] - pure_time["host_completion_us"],
                             "device_overhead_ratio_to_pure": auto_time["device_us"] / pure_time["device_us"] - 1,
                             "status": "PASS"})
        except Exception as exc:
            overhead.append({**_record_error(f"overhead_{operator}", exc), "operator": operator,
                             "process_index": process_index})

    concurrency_rows = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            concurrency_rows = list(executor.map(_thread_case, range(4)))
    except Exception as exc:
        concurrency_rows = [_record_error("thread_stream_concurrency", exc)]

    # A deterministic pre-launch exception must propagate; the current AUTO API
    # has no fallback chain and therefore must not claim that it recovered.
    recovery = []
    import ops.cuda_bridge as bridge
    original = bridge.rmsnorm_forward
    try:
        def injected(*_args, **_kwargs):
            raise RuntimeError("E04-08 injected pre-launch failure")
        bridge.rmsnorm_forward = injected
        try:
            dispatcher.run_rmsnorm(x, w)
            recovery.append({"case": "prelaunch_injected", "status": "UNEXPECTED_SUCCESS"})
        except Exception as exc:
            recovery.append({"case": "prelaunch_injected", "status": "ERROR_PROPAGATED_NO_FALLBACK",
                             "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        bridge.rmsnorm_forward = original
    try:
        actual, decision = dispatcher.run_rmsnorm(x, w)
        torch.cuda.synchronize()
        recovery.append({"case": "next_call_after_failure", "status": "PASS",
                         "actual": decision.backend, "correctness": _allclose(torch, actual,
                                                                              rmsnorm_torch(x, w), "fp32")})
    except Exception as exc:
        recovery.append(_record_error("next_call_after_failure", exc))

    env = EnvironmentInfo(platform=platform.platform(), device=torch.cuda.get_device_name(0),
                          compute_capability=list(torch.cuda.get_device_capability(0)),
                          python_version=platform.python_version(), torch_version=str(torch.__version__),
                          cuda_version=str(torch.version.cuda),
                          framework_versions={"triton": str(capabilities.triton_version or "unavailable")})
    c6 = BenchmarkResult(
        run_id=f"E04-08-proc{process_index}-{int(time.time())}", timestamp=time.time(),
        environment=env, git_commit=git("rev-parse", "HEAD"),
        git_dirty=bool(git("status", "--porcelain")),
        config_hash=read_json(out / "expected_decision_table.json")["table_sha256"],
        raw_samples=auto_rows,
        summary={"experiment_id": EXPERIMENT_ID, "requested_modes": ["AUTO"],
                 "forced_api_available": False, "capabilities": capabilities.as_dict(),
                 "overhead": overhead, "concurrency": concurrency_rows, "recovery": recovery},
        correctness=CorrectnessReport(
            passed=all(row.get("status") == "PASS" for row in auto_rows
                       if row["case_id"] in ("rms_auto_actual", "gemm_auto_actual")),
            method="operator reference comparison with frozen E03 tolerance",
            details={"auto_rows": len(auto_rows)}),
        artifact_links={"expected_table": "expected_decision_table.json",
                        "production_dispatcher": "ops/dispatcher.py"},
    )
    write_json(out / f"c6_proc{process_index}.json", c6.model_dump(mode="json"))
    write_jsonl(out / f"auto_proc{process_index}.jsonl", auto_rows)
    write_jsonl(out / f"direct_proc{process_index}.jsonl", direct)
    write_json(out / f"forced_proc{process_index}.json", {"rows": forced})
    write_json(out / f"capability_replay_proc{process_index}.json", {"rows": mocks})
    write_json(out / f"overhead_proc{process_index}.json", {"rows": overhead})
    write_json(out / f"concurrency_proc{process_index}.json", {"rows": concurrency_rows})
    write_json(out / f"recovery_proc{process_index}.json", {"rows": recovery})
    return 0


def holdout_current_policy() -> Dict[str, Any]:
    path = REPO / "docs/stage_experiments/S04/E04-04/raw/backend_rank_and_regret.json"
    data = read_json(path)
    mapping = {"gemm": "pytorch_cublas_opaque", "rmsnorm": "cuda_v2"}
    result: Dict[str, Any] = {"source": relative(path), "source_sha256": sha256_file(path),
                              "guard": REGRET_GUARD, "operators": {}}
    all_regrets = []
    for operator in ("gemm", "rmsnorm"):
        rows = []
        for item in data[operator]:
            if item.get("split") != "holdout":
                continue
            selected = mapping[operator]
            regret = item.get("regret", {}).get(selected)
            row = {key: item[key] for key in (("family", "M") if operator == "gemm"
                                              else ("hidden", "rows"))}
            row.update({"selected_by_current_production": selected,
                        "oracle_best": item.get("best_backend"),
                        "oracle_tie": item.get("tie_backends"), "regret": regret,
                        "over_guard": regret is None or regret > REGRET_GUARD})
            rows.append(row)
            if regret is not None:
                all_regrets.append(regret)
        regrets = [row["regret"] for row in rows if row["regret"] is not None]
        result["operators"][operator] = {
            "rows": rows, "count": len(rows),
            "weighted_mean_regret_equal_holdout": statistics.mean(regrets) if regrets else None,
            "max_regret": max(regrets) if regrets else None,
            "over_guard_count": sum(bool(row["over_guard"]) for row in rows),
        }
    result["combined_weighted_mean_regret"] = statistics.mean(all_regrets) if all_regrets else None
    result["pass"] = bool(all_regrets) and all(value <= REGRET_GUARD for value in all_regrets)
    return result


def make_manifest(out: Path) -> None:
    files = {}
    for path in sorted(item for item in out.rglob("*")
                       if item.is_file() and item.name != "EVIDENCE_MANIFEST.json"):
        files[str(path.relative_to(out))] = {"bytes": path.stat().st_size,
                                             "sha256": sha256_file(path)}
    write_json(out / "EVIDENCE_MANIFEST.json", {
        "schema": "hqsb.evidence_manifest/v1", "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": utc_now(), "files": files})


def analyze(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    autos = [row for index in range(3) for row in read_jsonl(out / f"auto_proc{index}.jsonl")]
    directs = [row for index in range(3) for row in read_jsonl(out / f"direct_proc{index}.jsonl")]
    forced = [row for index in range(3)
              for row in read_json(out / f"forced_proc{index}.json")["rows"]]
    overhead = [row for index in range(3)
                for row in read_json(out / f"overhead_proc{index}.json")["rows"]]
    concurrency = [row for index in range(3)
                   for row in read_json(out / f"concurrency_proc{index}.json")["rows"]]
    recovery = [row for index in range(3)
                for row in read_json(out / f"recovery_proc{index}.json")["rows"]]
    source = read_json(out / "source_contract_audit.json")
    inherited = read_json(out / "inherited_evidence.json")
    holdout = holdout_current_policy()
    write_json(out / "holdout_regret_current_policy.json", holdout)

    overhead_summary = []
    for operator in ("rmsnorm", "gemm"):
        rows = [row for row in overhead if row.get("operator") == operator and row.get("status") == "PASS"]
        overhead_summary.append({"operator": operator, "process_count": len(rows),
                                 "device_overhead_us_median": statistics.median(
                                     row["device_overhead_us"] for row in rows) if rows else None,
                                 "host_overhead_us_median": statistics.median(
                                     row["host_overhead_us"] for row in rows) if rows else None,
                                 "device_overhead_ratio_median": statistics.median(
                                     row["device_overhead_ratio_to_pure"] for row in rows) if rows else None})
    write_json(out / "overhead_summary.json", {"rows": overhead_summary})

    auto_actual = sorted({row.get("actual") for row in autos if row.get("actual")})
    direct_hits = sorted({row.get("actual_backend") for row in directs
                          if row.get("status") == "PASS" and row.get("actual_backend")})
    route_coverage = {
        "declared_backend_count": 9,
        "production_forced_api_backend_hits": [],
        "direct_backend_hits_not_dispatcher_evidence": direct_hits,
        "production_auto_actual_backends": auto_actual,
        "forced_api_missing_records": sum(row["status"] == "FORCED_API_MISSING" for row in forced),
        "rejection_reasons_structured": False,
        "fallback_edges_structured": False,
        "boundary_key_cases_executable": False,
        "actual_symbol_cross_check": False,
        "unknown_bf16_rejected_after_wrong_fp32_preclassification": all(
            row.get("status") == "EXPECTED_REJECT" for row in autos if row.get("case_id") == "unknown_bf16"),
    }
    write_json(out / "route_coverage.json", route_coverage)

    complete_key = all(source["required_key_fields"].values())
    production_fields = all(source["dispatch_result_fields"].values())
    conditions = {
        "complete_key_registry_versioned_policy": complete_key and
            source["versioned_policy_in_production"],
        "capability_policy_execution_separated": source["capability_policy_execution_separated"],
        "every_backend_forced_hit": len(route_coverage["production_forced_api_backend_hits"]) == 9,
        "every_declared_auto_backend_actual_symbol_hit": route_coverage["actual_symbol_cross_check"],
        "route_rejection_fallback_edges_covered": route_coverage["rejection_reasons_structured"] and
            route_coverage["fallback_edges_structured"] and route_coverage["boundary_key_cases_executable"],
        "requested_actual_candidates_reason_cache_written_to_c6": production_fields and
            source["c6_emission_in_production"],
        "unknown_unsupported_safe_predictable": False,
        "async_failure_not_silently_fallback": all(
            row.get("status") != "UNEXPECTED_SUCCESS" for row in recovery
            if row.get("case") == "prelaunch_injected"),
        "holdout_weighted_regret_within_guard": holdout["pass"],
        "auto_overhead_quantified_three_runs": all(row["process_count"] == 3
                                                    for row in overhead_summary),
        "cache_identity_invalidation_correct": source["route_cache_in_production"],
        "multithread_multistream_no_race": len(concurrency) == 12 and
            all(row.get("status") == "PASS" for row in concurrency),
    }
    verdict = {
        "schema": f"{SCHEMA}/verdict", "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": utc_now(), "conditions": conditions,
        "failed_conditions": [name for name, passed in conditions.items() if not passed],
        "overall": "PASS" if all(conditions.values()) else "FAIL",
        "upstream_all_pass": inherited["all_pass"],
        "upstream_blocked_or_failed": inherited["blocked_or_failed"],
        "counts": {"auto_records": len(autos), "direct_backend_records": len(directs),
                   "forced_api_records": len(forced), "concurrency_records": len(concurrency)},
        "claim": "audit of current production dispatcher; direct backend execution is not counted as forced dispatcher evidence",
        "stage_gate_rule": "E04-08 cannot PASS_NEGATIVE; every listed condition must be true",
    }
    write_json(out / "verdict.json", verdict)
    make_manifest(out)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    # A completed experiment may legitimately produce FAIL.  Keep orchestration
    # successful so the report and manifest are always synchronized back.
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    sub = value.add_subparsers(dest="command", required=True)
    for name in ("initialize", "analyze"):
        item = sub.add_parser(name); item.add_argument("--output-dir", required=True)
    item = sub.add_parser("runtime"); item.add_argument("--output-dir", required=True)
    item.add_argument("--process-index", required=True, type=int, choices=(0, 1, 2))
    return value


def main() -> int:
    args = parser().parse_args()
    return {"initialize": initialize, "runtime": runtime, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
