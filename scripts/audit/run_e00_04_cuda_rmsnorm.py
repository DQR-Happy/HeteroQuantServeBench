#!/usr/bin/env python3
"""E00-04 — CUDA device query + minimal RMSNorm correctness smoke + CUDA-unavailable path.

Question
--------
Can the target machine discover a CUDA device, is the CUDA/compiler toolchain
usable, is the minimal RMSNorm CUDA path numerically correct, and does the
program fail *clearly* when CUDA is unavailable?

Hypothesis (falsifiable, pre-registered)
----------------------------------------
H1  (normal path) On the recorded target device the CUDA device is discovered
    (``cuda_available`` true, ``device_count`` > 0, arch/runtime explainable),
    the RMSNorm CUDA shared library builds and loads, the dispatcher selects
    the CUDA backend (no silent fallback), and every pre-registered minimal
    correctness case matches an independent PyTorch reference within the
    pre-registered per-dtype tolerance (FP32 ``5e-4``, FP16 ``2e-2``) with no
    NaN/Inf, correct output shape/dtype/device, and no input mutation.
H0  Either the device path is not usable, or some correctness case fails, or
    the CUDA-unavailable path silently falls back / returns success.

Design
------
* **Part 1 — device query**: run the C++ ``hqsb_device_query`` binary plus the
  torch-layer checks, and merge into ``device_query.json``.
* **Part 2 — build + load**: rebuild the CMake shared library, record compiler
  version / arch / build log / binary sha256, then verify the ctypes bridge
  loads and the dispatcher selects CUDA.
* **Part 3 — correctness**: run a frozen matrix of cases against the CUDA
  bridge (variant 0 = auto), plus explicit per-variant checks, comparing
  against ``ops.triton.rmsnorm.rmsnorm_torch`` (independent PyTorch reference).
* **Part 4 — negative path**: spawn a subprocess with ``CUDA_VISIBLE_DEVICES=""``
  and require a non-zero exit with a locatable error (no silent CPU fallback).

Raw output
----------
``device_query.json``, ``environment.json``, ``build.log``,
``binary-sha256.txt``, ``rmsnorm_correctness_raw.jsonl``,
``rmsnorm_error_cases.jsonl``, ``cuda_unavailable.log``, ``command.txt``,
``file_manifest.json``, plus a run record ``e00_04_run_<run_id>.json``.

Usage
-----
    python3 scripts/audit/run_e00_04_cuda_rmsnorm.py \
        --output-dir docs/stage_experiments/S00/E00-04/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hqsb.core.fingerprint import (  # noqa: E402
    collect_commit,
    collect_device_basic,
    collect_os,
    collect_packages,
    collect_power,
    collect_python,
    collect_volatile,
    sha256_hex,
)
from hqsb.core.ids import new_run_id  # noqa: E402
from hqsb.hardware.probe import cuda_device_probe  # noqa: E402

EXPERIMENT_ID = "E00-04"
STAGE = "S00"

# ── Frozen correctness matrix ─────────────────────────────────────────────
# shape comes from the S00 baseline / Qwen3-1.7B config (hidden=2048) / the
# S03 operator test protocol; it is NOT tuned to the candidate kernel.
#
# variant encoding mirrors ops/cuda/rmsnorm/src/rmsnorm_c_api.cu:
#   0 = auto, 2 = v0_shared, 3 = v1_warp_shuffle, 4 = v2_vectorized.
#
# (case_id, rows, hidden, dtype, input_mode, seed, epsilon, variant)
_CORRECTNESS_CASES: List[Tuple[str, int, int, str, str, int, float, int]] = [
    # minimal legal shape
    ("c01_minimal_fp32", 1, 1, "fp32", "normal", 1234, 1e-5, 0),
    # actual model hidden size (Qwen3-1.7B hidden=2048)
    ("c02_model_hidden_fp32", 16, 2048, "fp32", "normal", 1234, 1e-5, 0),
    # non-multiple-of-4 boundary (routes to V1)
    ("c03_non_aligned_fp32", 8, 101, "fp32", "normal", 1234, 1e-5, 0),
    # larger shape, no OOM (S00 baseline 512 x 1024)
    ("c04_baseline_fp32", 512, 1024, "fp32", "normal", 1234, 1e-5, 0),
    # FP16 model hidden
    ("c05_model_hidden_fp16", 32, 2048, "fp16", "normal", 1234, 1e-5, 0),
    # FP16 odd tail (V2 half2 scalar-tail fallback)
    ("c06_non_aligned_fp16", 8, 3, "fp16", "normal", 1234, 1e-5, 0),
    # zero input
    ("c07_zeros_fp32", 128, 256, "fp32", "zeros", 1234, 1e-5, 0),
    # near-zero (tiny) input
    ("c08_near_zero_fp32", 128, 256, "fp32", "tiny", 1234, 1e-5, 0),
    # large-but-finite input
    ("c09_large_fp32", 128, 256, "fp32", "large", 1234, 1e-5, 0),
    # FP16 zero input
    ("c10_zeros_fp16", 32, 256, "fp16", "zeros", 1234, 1e-5, 0),
    # epsilon semantics: a different epsilon must be honored
    ("c11_epsilon_1e2_fp32", 128, 256, "fp32", "normal", 1234, 1e-2, 0),
]

# Explicit per-variant checks (prove every variant loads and is correct).
_VARIANT_CHECKS: List[Tuple[str, int, int, str, int]] = [
    ("v0_shared_fp32", 512, 1024, "fp32", 2),
    ("v1_warp_fp32", 512, 1024, "fp32", 3),
    ("v2_vectorized_fp32", 512, 1024, "fp32", 4),
    ("v1_warp_non_aligned_fp32", 8, 101, "fp32", 3),
    ("v2_vectorized_fp16", 32, 2048, "fp16", 4),
]

# Pre-registered tolerances (S03 test standard: per-dtype thresholds).
_TOLERANCE = {
    "fp32": {"max_abs": 5e-4, "max_rel": 5e-4},
    "fp16": {"max_abs": 2e-2, "max_rel": 2e-2},
}

# Relative-error guard: elements with |ref| below this use |ref| = guard, so
# near-zero outputs (where RMSNorm output ~ 0) don't produce meaningless
# relative errors.
_REL_GUARD = 1e-3

_DEFAULT_CUDA_LIB = "build/jetson-release/ops/cuda/rmsnorm/libhqsb_rmsnorm_shared.so"
_DEVICE_QUERY_BIN = "build/jetson-release/bin/hqsb_device_query"
_RMSNORM_TEST_BIN = "build/jetson-release/bin/hqsb_rmsnorm_test"


# ── small helpers ─────────────────────────────────────────────────────────


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT), **kwargs)


def _git(*args: str) -> str:
    try:
        proc = _run(["git", *args])
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except OSError:
        return ""


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── Part 1: device query ──────────────────────────────────────────────────


def collect_environment_block() -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_commit_short": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "started_at_utc": _now_utc(),
    }


def collect_environment_json() -> Dict[str, Any]:
    """Full environment section (reuses the E00-03 fingerprint collectors)."""
    device = collect_device_basic(cuda_device_probe())
    power = collect_power()
    return {
        "os": collect_os().model_dump(mode="json"),
        "device": device.model_dump(mode="json"),
        "python": collect_python().model_dump(mode="json"),
        "packages": collect_packages(),
        "power": power.model_dump(mode="json"),
        "volatile": collect_volatile().model_dump(mode="json"),
        "commit": collect_commit(str(_REPO_ROOT)).model_dump(mode="json"),
    }


def run_device_query() -> Tuple[Dict[str, Any], str]:
    """Run the C++ device-query binary and the torch checks; merge into JSON."""
    cpp_proc = _run([_DEVICE_QUERY_BIN], timeout=60)
    cpp_stdout = cpp_proc.stdout or ""
    cpp_stderr = cpp_proc.stderr or ""

    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0

    device_name = ""
    total_memory = 0
    capability = None
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_memory = int(props.total_memory)
            cap = torch.cuda.get_device_capability(0)
            capability = [int(cap[0]), int(cap[1])]
        except Exception:
            pass

    device = collect_device_basic(cuda_device_probe())
    power = collect_power()

    query = {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_name": device_name,
        "total_memory_bytes": total_memory,
        "total_memory_gib": round(total_memory / (1024.0**3), 2) if total_memory else 0.0,
        "capability_or_arch": capability,
        "driver_version": device.cuda_driver_version,
        "runtime_version": device.cuda_runtime_version or (torch.version.cuda or ""),
        "compiler_version": device.nvcc_version,
        "board_compatible": device.board_compatible,
        "l4t_release": device.l4t_release,
        "power_mode": power.nvpmodel_mode,
        "power_mode_name": power.nvpmodel_name,
        "jetson_clocks_active": power.jetson_clocks_active,
        "cpu_governor": power.cpu_governor,
        "gpu_governor": power.gpu_governor,
        "query_exit_code": int(cpp_proc.returncode),
        "cpp_query_stdout": cpp_stdout,
        "cpp_query_stderr": cpp_stderr,
    }
    return query, cpp_stdout + cpp_stderr


# ── Part 2: build + load ──────────────────────────────────────────────────


def run_build(out_dir: Path, clean_first: bool = False) -> Dict[str, Any]:
    """Rebuild the shared library; record build info and binary sha256."""
    build_cmd = ["cmake", "--build", "build/jetson-release"]
    if clean_first:
        build_cmd.append("--clean-first")
    configure_cmd = [
        "cmake",
        "-S",
        ".",
        "-B",
        "build/jetson-release",
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=87",
    ]

    proc = _run(build_cmd, timeout=1800)
    (out_dir / "build.log").write_text(
        f"$ {' '.join(build_cmd)}\n\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}\n",
        encoding="utf-8",
    )

    nvcc_version = ""
    nvcc_proc = _run(["nvcc", "--version"], timeout=30)
    nvcc_version = (nvcc_proc.stdout or "") + (nvcc_proc.stderr or "")

    lib_path = _REPO_ROOT / _DEFAULT_CUDA_LIB
    binary_sha256 = _file_sha256(lib_path) if lib_path.is_file() else ""
    (out_dir / "binary-sha256.txt").write_text(
        f"{binary_sha256}  {_DEFAULT_CUDA_LIB}\n", encoding="utf-8"
    )

    return {
        "configure_command": " ".join(configure_cmd),
        "build_command": " ".join(build_cmd),
        "build_exit_code": int(proc.returncode),
        "nvcc_version_full": nvcc_version.strip(),
        "cuda_arch": "87",
        "library_path": _DEFAULT_CUDA_LIB,
        "library_sha256": binary_sha256,
    }


def run_load_check() -> Dict[str, Any]:
    """Verify the bridge loads and the dispatcher selects CUDA (no fallback)."""
    from ops.capability import detect_capabilities
    from ops.dispatcher import OperatorDispatcher

    cap = detect_capabilities()
    dispatcher = OperatorDispatcher(cap)

    decisions = {
        "fp32_2048": dispatcher.select_rmsnorm("fp32", 2048).as_dict(),
        "fp32_101": dispatcher.select_rmsnorm("fp32", 101).as_dict(),
        "fp16_2048": dispatcher.select_rmsnorm("fp16", 2048).as_dict(),
    }

    # Prove the ctypes bridge actually loads the shared library.
    load_ok = False
    load_error = ""
    try:
        from ops.cuda_bridge import rmsnorm_forward as _rf

        _rf  # noqa: B018 (touch symbol)
        import torch

        x = torch.randn(2, 8, device="cuda", dtype=torch.float32)
        w = torch.ones(8, device="cuda", dtype=torch.float32)
        _rf(x, w, dtype="fp32", variant=0, epsilon=1e-5)
        torch.cuda.synchronize()
        load_ok = True
    except Exception as exc:  # pragma: no cover - depends on environment
        load_error = f"{type(exc).__name__}: {exc}"

    return {
        "capabilities": cap.as_dict(),
        "dispatcher_decisions": decisions,
        "bridge_load_ok": load_ok,
        "bridge_load_error": load_error,
        "cuda_selected": all(
            d["backend"] == "cuda" for d in decisions.values()
        ),
    }


# ── Part 3: correctness ───────────────────────────────────────────────────


def _make_input(rows: int, hidden: int, dtype: str, mode: str, seed: int):
    import torch

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16}
    tdtype = dtype_map[dtype]

    g = torch.Generator(device="cuda")
    g.manual_seed(seed)

    if mode == "zeros":
        x = torch.zeros(rows, hidden, dtype=tdtype, device="cuda")
    elif mode == "tiny":
        x = torch.empty(rows, hidden, dtype=tdtype, device="cuda").uniform_(
            1e-6, 1e-5, generator=g
        )
    elif mode == "large":
        x = torch.empty(rows, hidden, dtype=tdtype, device="cuda").uniform_(
            1e6, 1e7, generator=g
        )
    else:  # normal
        x = torch.randn(rows, hidden, dtype=tdtype, device="cuda", generator=g)

    # Weight in [0.5, 1.5] (matches the C++ test's generate_weight).
    w = torch.empty(hidden, dtype=tdtype, device="cuda").uniform_(
        0.5, 1.5, generator=g
    )
    return x, w


def _metrics(candidate: Any, reference: Any) -> Dict[str, Any]:
    import torch

    cand = candidate.float()
    ref = reference.float()
    diff = (cand - ref).abs()

    abs_err = diff
    # relative error with guard
    rel_err = diff / ref.abs().clamp_min(_REL_GUARD)

    nan_or_inf = bool(torch.isnan(candidate).any() or torch.isinf(candidate).any())

    return {
        "max_abs_error": float(abs_err.max().item()),
        "mean_abs_error": float(abs_err.mean().item()),
        "rmse": float(torch.sqrt((diff ** 2).mean()).item()),
        "max_rel_error": float(rel_err.max().item()),
        "nan_or_inf": nan_or_inf,
    }


def run_correctness(out_dir: Path) -> Dict[str, Any]:
    import torch

    from ops.cuda_bridge import rmsnorm_forward as cuda_forward
    from ops.triton.rmsnorm import rmsnorm_torch

    records: List[Dict[str, Any]] = []
    failures: List[str] = []

    cases = list(_CORRECTNESS_CASES)
    for case_id, rows, hidden, dtype, variant in _VARIANT_CHECKS:
        cases.append((case_id, rows, hidden, dtype, "normal", 1234, 1e-5, variant))

    for (case_id, rows, hidden, dtype, mode, seed, eps, variant) in cases:
        x, w = _make_input(rows, hidden, dtype, mode, seed)

        reference = rmsnorm_torch(x, w, eps)
        x_before = x.clone()

        error = ""
        exit_code = 0
        status = "PASS"
        candidate = None
        try:
            candidate = cuda_forward(x, w, dtype=dtype, variant=variant, epsilon=eps)
            torch.cuda.synchronize()
        except Exception as exc:  # pragma: no cover - failure path
            error = f"{type(exc).__name__}: {exc}"
            exit_code = 1
            status = "FAIL"

        if candidate is not None:
            m = _metrics(candidate, reference)
            tol = _TOLERANCE[dtype]
            input_mutated = not torch.equal(x, x_before)
            shape_ok = tuple(candidate.shape) == (rows, hidden)
            expected_dtype = {"fp32": "float32", "fp16": "float16"}[dtype]
            dtype_ok = str(candidate.dtype).replace("torch.", "") == expected_dtype
            device_ok = str(candidate.device.type) == "cuda"

            metric_ok = (
                m["max_abs_error"] <= tol["max_abs"]
                and m["max_rel_error"] <= tol["max_rel"]
                and not m["nan_or_inf"]
            )

            if not (metric_ok and shape_ok and dtype_ok and device_ok and not input_mutated):
                status = "FAIL"
                failures.append(case_id)

            record = {
                "case_id": case_id,
                "shape": [rows, hidden],
                "dtype": dtype,
                "input_mode": mode,
                "seed": seed,
                "epsilon": eps,
                "variant": variant,
                "reference_implementation": "ops.triton.rmsnorm.rmsnorm_torch (PyTorch FP32-accum)",
                "candidate_implementation": "ops.cuda_bridge.rmsnorm_forward (CUDA shared lib)",
                "status": status,
                "exit_code": exit_code,
                **m,
                "output_shape": list(candidate.shape),
                "output_dtype": str(candidate.dtype),
                "output_device": str(candidate.device),
                "input_mutated": input_mutated,
                "error": error,
            }
        else:
            record = {
                "case_id": case_id,
                "shape": [rows, hidden],
                "dtype": dtype,
                "input_mode": mode,
                "seed": seed,
                "epsilon": eps,
                "variant": variant,
                "reference_implementation": "ops.triton.rmsnorm.rmsnorm_torch",
                "candidate_implementation": "ops.cuda_bridge.rmsnorm_forward",
                "status": status,
                "exit_code": exit_code,
                "error": error,
            }

        records.append(record)

    # Write JSONL (one record per line).
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    (out_dir / "rmsnorm_correctness_raw.jsonl").write_text(lines, encoding="utf-8")

    summary = {
        "total": len(records),
        "passed": sum(1 for r in records if r["status"] == "PASS"),
        "failed": sum(1 for r in records if r["status"] == "FAIL"),
        "failed_cases": failures,
    }
    return summary


# ── Part 4: negative path ─────────────────────────────────────────────────

_NEGATIVE_PROBE = r'''
import json, sys
sys.path.insert(0, ".")
import torch
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.cuda.device_count() =", torch.cuda.device_count())

from ops.capability import detect_capabilities
cap = detect_capabilities()
print("detect_capabilities.cuda_available =", cap.cuda_available)
print("detect_capabilities.cuda_rmsnorm_available =", cap.cuda_rmsnorm_available)

from ops.dispatcher import OperatorDispatcher
d = OperatorDispatcher(cap)
decision = d.select_rmsnorm("fp32", 2048)
print("dispatcher.select_rmsnorm(fp32,2048) =", json.dumps(decision.as_dict()))

# Attempt the CUDA RMSNorm entry: it must fail clearly (no silent fallback).
from ops.cuda_bridge import rmsnorm_forward, CudaRmsnormUnavailable
x = torch.randn(2, 8)
w = torch.rand(8)
try:
    rmsnorm_forward(x, w, dtype="fp32", variant=0, epsilon=1e-5)
    print("UNEXPECTED: rmsnorm_forward returned without error (silent fallback?)")
    sys.exit(0)
except CudaRmsnormUnavailable as exc:
    print("CudaRmsnormUnavailable raised:", exc)
    sys.exit(3)
except Exception as exc:
    print("raised:", type(exc).__name__, exc)
    sys.exit(3)
'''


def run_negative_path(out_dir: Path) -> Dict[str, Any]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    log_lines: List[str] = []
    log_lines.append("# E00-04 negative path: CUDA_VISIBLE_DEVICES=\"\"")
    log_lines.append(f"# command: CUDA_VISIBLE_DEVICES=\"\" python3 -c <probe>")
    log_lines.append("")

    probe_proc = subprocess.run(
        [sys.executable, "-c", _NEGATIVE_PROBE],
        capture_output=True, text=True, cwd=str(_REPO_ROOT), env=env, timeout=300,
    )
    log_lines.append("## Python probe (RMSNorm entry under hidden CUDA)")
    log_lines.append(f"$ CUDA_VISIBLE_DEVICES=\"\" {sys.executable} -c <probe>")
    log_lines.append(f"exit_code = {probe_proc.returncode}")
    log_lines.append("--- stdout ---")
    log_lines.append(probe_proc.stdout)
    log_lines.append("--- stderr ---")
    log_lines.append(probe_proc.stderr)
    log_lines.append("")

    cpp_proc = subprocess.run(
        [_DEVICE_QUERY_BIN], capture_output=True, text=True,
        cwd=str(_REPO_ROOT), env=env, timeout=60,
    )
    log_lines.append("## C++ device query under hidden CUDA")
    log_lines.append(f"$ CUDA_VISIBLE_DEVICES=\"\" {_DEVICE_QUERY_BIN}")
    log_lines.append(f"exit_code = {cpp_proc.returncode}")
    log_lines.append("--- stdout ---")
    log_lines.append(cpp_proc.stdout)
    log_lines.append("--- stderr ---")
    log_lines.append(cpp_proc.stderr)

    (out_dir / "cuda_unavailable.log").write_text(
        "\n".join(log_lines), encoding="utf-8"
    )

    # Structured error-case records.
    error_cases = [
        {
            "case_id": "neg_python_rmsnorm_entry",
            "requested_capability": "cuda",
            "environment": "CUDA_VISIBLE_DEVICES=''",
            "exit_code": int(probe_proc.returncode),
            "non_zero_exit": probe_proc.returncode != 0,
            "cuda_available_reported": "cuda_available = False" in probe_proc.stdout,
            "error_mentions_cuda": "CudaRmsnormUnavailable" in probe_proc.stdout,
            "silent_fallback": "UNEXPECTED" in probe_proc.stdout,
            "stdout": probe_proc.stdout,
            "stderr": probe_proc.stderr,
        },
        {
            "case_id": "neg_cpp_device_query",
            "requested_capability": "cuda",
            "environment": "CUDA_VISIBLE_DEVICES=''",
            "exit_code": int(cpp_proc.returncode),
            "non_zero_exit": cpp_proc.returncode != 0,
            "mentions_no_device": "No CUDA device" in (cpp_proc.stdout + cpp_proc.stderr),
            "stdout": cpp_proc.stdout,
            "stderr": cpp_proc.stderr,
        },
    ]
    lines = "\n".join(json.dumps(c, ensure_ascii=False) for c in error_cases) + "\n"
    (out_dir / "rmsnorm_error_cases.jsonl").write_text(lines, encoding="utf-8")

    return {
        "python_probe_exit_code": int(probe_proc.returncode),
        "cpp_query_exit_code": int(cpp_proc.returncode),
        "python_non_zero": probe_proc.returncode != 0,
        "cpp_non_zero": cpp_proc.returncode != 0,
        "python_error_locatable": "CudaRmsnormUnavailable" in probe_proc.stdout,
        "no_silent_fallback": "UNEXPECTED" not in probe_proc.stdout,
    }


# ── drivers ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E00-04 CUDA device + RMSNorm smoke.")
    parser.add_argument(
        "--output-dir",
        default="docs/stage_experiments/S00/E00-04/raw",
        help="Directory for raw artifacts.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-build", action="store_true", help="Skip the rebuild.")
    parser.add_argument(
        "--clean-first",
        action="store_true",
        help="Clean all targets before building (proves the toolchain end-to-end).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or new_run_id()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    started_at = _now_utc()

    # Part 1 — device query
    device_query, device_query_text = run_device_query()
    (out_dir / "device_query.json").write_text(
        json.dumps(device_query, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # environment.json
    environment = collect_environment_json()
    environment["environment_block"] = collect_environment_block()
    (out_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Part 2 — build + load
    if args.skip_build:
        build_info: Dict[str, Any] = {
            "build_command": "(skipped)",
            "build_exit_code": 0,
            "library_sha256": _file_sha256(_REPO_ROOT / _DEFAULT_CUDA_LIB)
            if (_REPO_ROOT / _DEFAULT_CUDA_LIB).is_file()
            else "",
        }
    else:
        build_info = run_build(out_dir, clean_first=args.clean_first)

    load_info = run_load_check()

    # Part 3 — correctness
    correctness_summary = run_correctness(out_dir)

    # Part 4 — negative path
    negative_summary = run_negative_path(out_dir)

    # command.txt
    build_cmd_line = "cmake --build build/jetson-release"
    if args.clean_first:
        build_cmd_line += " --clean-first"
    commands = [
        f"cmake -S . -B build/jetson-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87",
        build_cmd_line,
        f"{_DEVICE_QUERY_BIN}",
        f"{_RMSNORM_TEST_BIN}",
        f"CUDA_VISIBLE_DEVICES=\"\" python3 -c <negative probe>",
        f"CUDA_VISIBLE_DEVICES=\"\" {_DEVICE_QUERY_BIN}",
    ]
    (out_dir / "command.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    ended_at = _now_utc()

    # Aggregate run record.
    record: Dict[str, Any] = {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "environment": collect_environment_block(),
        "device_query": device_query,
        "build": build_info,
        "load": load_info,
        "correctness_summary": correctness_summary,
        "negative_path": negative_summary,
        "decision": None,  # filled below
    }

    # Determine overall decision.
    normal_path_pass = (
        device_query["cuda_available"]
        and device_query["device_count"] > 0
        and device_query["query_exit_code"] == 0
        and load_info["cuda_selected"]
        and load_info["bridge_load_ok"]
        and correctness_summary["failed"] == 0
    )
    negative_path_pass = (
        negative_summary["python_non_zero"]
        and negative_summary["cpp_non_zero"]
        and negative_summary["python_error_locatable"]
        and negative_summary["no_silent_fallback"]
    )
    record["decision"] = {
        "normal_path_pass": normal_path_pass,
        "negative_path_pass": negative_path_pass,
        "overall": "PASS" if (normal_path_pass and negative_path_pass) else "FAIL",
    }

    json_path = out_dir / f"e00_04_run_{run_id}.json"
    json_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # file_manifest.json
    manifest = {
        name: _file_sha256(out_dir / name)
        for name in sorted(p.name for p in out_dir.iterdir() if p.is_file())
    }
    (out_dir / "file_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[{EXPERIMENT_ID}] run_id={run_id}")
    print(
        f"[{EXPERIMENT_ID}] device: cuda_available={device_query['cuda_available']} "
        f"count={device_query['device_count']} name={device_query['device_name']} "
        f"arch={device_query['capability_or_arch']} exit={device_query['query_exit_code']}"
    )
    print(
        f"[{EXPERIMENT_ID}] build exit={build_info.get('build_exit_code')} "
        f"lib_sha256={build_info.get('library_sha256', '')[:16]}"
    )
    print(
        f"[{EXPERIMENT_ID}] load: bridge_ok={load_info['bridge_load_ok']} "
        f"cuda_selected={load_info['cuda_selected']}"
    )
    print(
        f"[{EXPERIMENT_ID}] correctness {correctness_summary['passed']}/"
        f"{correctness_summary['total']} pass "
        f"(failed={correctness_summary['failed']})"
    )
    print(
        f"[{EXPERIMENT_ID}] negative: python_exit={negative_summary['python_probe_exit_code']} "
        f"cpp_exit={negative_summary['cpp_query_exit_code']} "
        f"locatable={negative_summary['python_error_locatable']} "
        f"no_fallback={negative_summary['no_silent_fallback']}"
    )
    print(f"[{EXPERIMENT_ID}] decision={record['decision']}")
    return 0 if record["decision"]["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
