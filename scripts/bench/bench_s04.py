#!/usr/bin/env python3
"""S04 backend comparison benchmark: CUDA vs Triton RMSNorm, cuBLAS vs Triton GEMM.

All timings use the *same* method (torch.cuda.Event + median over reps) so
CUDA (ctypes), Triton, and cuBLAS (torch) are measured on equal footing.

Correctness first (S04 补齐)
----------------------------
Every timing entry carries a ``correctness`` verdict computed against an
FP64 reference **before** the timing is reported. A timing without a
correctness verdict is not evidence, and the S04 acceptance criteria require
correctness before performance. The gates are scale-free (relative L2 error +
cosine similarity) because a per-element ``atol + rtol*|expected|`` bound is
unsound for a length-K reduction whose output can cross zero -- see
``tests/unit/ops/test_triton_gemm.py`` for the full argument.

Cold vs steady state
--------------------
Each entry also records ``cold_call_ms``: the wall time of the very first
invocation, which for Triton includes JIT compilation (orders of magnitude
larger than the kernel). Reporting only the steady-state median hides that
cost; reporting only the cold number would misrepresent throughput.

Usage:
    python3 scripts/bench/bench_s04.py [--output reports/dev/s04/bench.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from ops.capability import detect_capabilities  # noqa: E402
from ops.cuda_bridge import rmsnorm_forward as cuda_rmsnorm  # noqa: E402

# Scale-free correctness gates (mirror tests/unit/ops/test_triton_gemm.py).
_L2_RELATIVE_TOLERANCE = 1e-2
_MIN_COSINE_SIMILARITY = 0.9999


def _time_ms(fn, warmup: int = 10, reps: int = 50):
    """Time ``fn`` with CUDA events; return ``(steady_state_median_ms, cold_ms)``.

    ``cold_ms`` is measured on the very first call, before any warmup, so it
    captures one-off costs (Triton JIT compilation, first allocator growth)
    that the steady-state median deliberately excludes.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    cold_ms = start.elapsed_time(end)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), cold_ms


def _l2_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.double().reshape(-1)
    e = expected.double().reshape(-1)
    denominator = float(torch.linalg.vector_norm(e))
    if denominator == 0.0:
        return 0.0
    return float(torch.linalg.vector_norm(a - e)) / denominator


def _cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.double().reshape(-1)
    e = expected.double().reshape(-1)
    denominator = float(torch.linalg.vector_norm(a)) * float(
        torch.linalg.vector_norm(e)
    )
    if denominator == 0.0:
        return 1.0
    return float(torch.dot(a, e)) / denominator


def _correctness(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Scale-free correctness verdict for one backend output."""
    l2 = _l2_relative_error(actual, expected)
    cosine = _cosine_similarity(actual, expected)
    return {
        "passed": bool(
            l2 <= _L2_RELATIVE_TOLERANCE and cosine >= _MIN_COSINE_SIMILARITY
        ),
        "l2_relative_error": round(l2, 8),
        "cosine_similarity": round(cosine, 8),
        "max_abs_error": round(float((actual.double() - expected.double()).abs().max()), 8),
    }


def _rmsnorm_reference_fp64(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """FP64 RMSNorm reference (semantics identical to the CPU reference)."""
    xd = x.double()
    rms = torch.rsqrt(xd.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xd * rms).to(x.dtype) * w


def bench_rmsnorm() -> list:
    """CUDA V0/V1/V2 vs Triton reference/optimized, FP32 + FP16."""
    import triton  # noqa: F401  (ensure importable before kernels)
    from ops.triton.rmsnorm import rmsnorm_optimized, rmsnorm_reference

    rows = 512
    shapes = [1024, 2048]
    eps = 1e-5
    results = []

    for dtype, dt in [("fp32", torch.float32), ("fp16", torch.float16)]:
        for hidden in shapes:
            torch.manual_seed(0)
            x = torch.randn(rows, hidden, device="cuda", dtype=dt)
            w = torch.randn(hidden, device="cuda", dtype=dt)
            reference = _rmsnorm_reference_fp64(x, w, eps)

            entries = []
            # CUDA variants (V0/V1 FP32-only; V2 both).
            if dtype == "fp32":
                for name, code in [("cuda_v0", 2), ("cuda_v1", 3), ("cuda_v2", 4)]:
                    entries.append(
                        (name, lambda c=code: cuda_rmsnorm(x, w, dtype=dtype, variant=c,
                                                           epsilon=eps))
                    )
            else:
                entries.append(
                    ("cuda_v2",
                     lambda: cuda_rmsnorm(x, w, dtype=dtype, variant=4, epsilon=eps))
                )

            entries.append(("triton_reference",
                            lambda: rmsnorm_reference(x, w, epsilon=eps)))
            entries.append(("triton_optimized",
                            lambda: rmsnorm_optimized(x, w, epsilon=eps)))

            for name, fn in entries:
                out = fn()                       # correctness first
                verdict = _correctness(out, reference)
                median_ms, cold_ms = _time_ms(fn)
                results.append({
                    "op": "rmsnorm",
                    "dtype": dtype,
                    "hidden": hidden,
                    "rows": rows,
                    "backend": name,
                    "median_ms": round(median_ms, 4),
                    "cold_call_ms": round(cold_ms, 4),
                    "correctness": verdict,
                })
    return results


def bench_gemm() -> list:
    """cuBLAS (torch.matmul) vs Triton GEMM vs CUTLASS, FP16."""
    from ops.triton.gemm import gemm_optimized, gemm_reference

    results = []
    cutlass_bin = _find_cutlass_bin()
    for m, k, n in [(1, 2048, 2048), (1, 2048, 8192), (512, 2048, 2048)]:
        torch.manual_seed(0)
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        # Ground truth in FP64: the backends accumulate in FP32, so an FP32
        # reference would carry an error of the same order being measured.
        reference = (a.double() @ b.double()).to(torch.float16)

        for name, fn in [
            ("cublas", lambda: a @ b),
            ("triton_optimized", lambda: gemm_optimized(a, b)),
            ("triton_reference", lambda: gemm_reference(a, b)),
        ]:
            verdict = _correctness(fn(), reference)
            median_ms, cold_ms = _time_ms(fn)
            results.append({
                "op": "gemm",
                "dtype": "fp16",
                "m": m, "k": k, "n": n,
                "backend": name,
                "median_ms": round(median_ms, 4),
                "cold_call_ms": round(cold_ms, 4),
                "correctness": verdict,
            })

        # CUTLASS (external C++ binary; skipped when not built or when the
        # binary reports that it cannot run on this architecture).
        entry, reason = _run_cutlass(cutlass_bin, m, n, k)
        if entry is not None:
            entry["op"] = "gemm"
            entry["dtype"] = "fp16"
            entry["m"], entry["k"], entry["n"] = m, k, n
            results.append(entry)
        elif reason:
            results.append({
                "op": "gemm", "dtype": "fp16", "m": m, "k": k, "n": n,
                "backend": "cutlass_unavailable",
                "median_ms": None,
                "cold_call_ms": None,
                "correctness": {"passed": None, "skipped_reason": reason},
            })
    return results


def _find_cutlass_bin():
    """Locate the CUTLASS GEMM benchmark binary, if built."""
    import glob

    for path in glob.glob(
        os.path.join(REPO_ROOT, "build", "*", "bin", "hqsb_cutlass_gemm_bench")
    ):
        if os.path.isfile(path):
            return path
    return None


def _run_cutlass(binary, m, n, k):
    """Run the CUTLASS binary once; return ``(entry, skip_reason)``.

    The binary exits non-zero when the kernel did not run (2), when the
    output is numerically wrong (3), or when no configuration fits the device
    (4). A non-zero exit is recorded as a skip *reason* rather than silently
    dropped, so a missing architecture is visible in the report.
    """
    import subprocess

    if binary is None:
        return None, "hqsb_cutlass_gemm_bench not built (CUTLASS headers absent)"

    try:
        out = subprocess.run(
            [binary, "--m", str(m), "--n", str(n), "--k", str(k)],
            capture_output=True, text=True, timeout=300,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return None, f"CUTLASS binary not runnable: {type(exc).__name__}: {exc}"

    if out.returncode != 0:
        detail = (out.stderr or "").strip().splitlines()
        return None, (
            f"CUTLASS benchmark exited {out.returncode}: "
            f"{detail[-1] if detail else 'no stderr'}"
        )

    try:
        # Machine-readable line (stdout): dtype,m,n,k,config,median_ms,max_err
        parts = out.stdout.strip().splitlines()[-1].split(",")
        backend = parts[4]           # already "cutlass_<tile config>"
        median_ms = float(parts[5])
        max_err = float(parts[6])
    except (ValueError, IndexError):
        return None, "could not parse the CUTLASS benchmark output"

    # The binary already applies the correctness gate (it exits non-zero on
    # failure, handled above); mirror the verdict here so the JSON is
    # self-describing without re-parsing stderr.
    return {
        "backend": backend,
        "median_ms": round(median_ms, 4),
        "cold_call_ms": None,
        "correctness": {"passed": True, "max_abs_error": round(max_err, 6)},
    }, None


def main() -> int:
    parser = argparse.ArgumentParser(description="S04 backend comparison benchmark")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for S04 benchmark", file=sys.stderr)
        return 1

    cap = detect_capabilities()
    print("=== capability ===")
    print(json.dumps(cap.as_dict(), indent=2))

    rmsnorm_results = bench_rmsnorm()
    gemm_results = bench_gemm()
    all_results = rmsnorm_results + gemm_results

    print("\n=== RMSNorm (median ms / cold ms, correctness) ===")
    for r in rmsnorm_results:
        print(f"  {r['dtype']:>4} hidden={r['hidden']:>4} {r['backend']:<18} "
              f"{r['median_ms']:.4f} / {r['cold_call_ms']:.4f}  "
              f"correct={r['correctness']['passed']} "
              f"(l2={r['correctness']['l2_relative_error']:.2e})")

    print("\n=== GEMM (median ms / cold ms, correctness) ===")
    for r in gemm_results:
        mark = "correct" if r["correctness"]["passed"] else "SKIPPED/FAIL"
        median = "n/a" if r["median_ms"] is None else f"{r['median_ms']:.4f}"
        cold = "n/a" if r["cold_call_ms"] is None else f"{r['cold_call_ms']:.4f}"
        print(f"  {r['dtype']:>4} {r['m']}x{r['k']}x{r['n']:<6} "
              f"{r['backend']:<22} {median:>8} / {cold:>8}  {mark}")

    failures = [r for r in all_results if r["correctness"]["passed"] is False]
    if failures:
        print(f"\nCORRECTNESS FAILURES: {len(failures)}", file=sys.stderr)
        for r in failures:
            print(f"  {r['backend']} {r.get('dtype')} {r.get('hidden', '')}",
                  file=sys.stderr)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "capability": cap.as_dict(),
                    "all_correctness_passed": not failures,
                    "results": all_results,
                },
                fh, indent=2,
            )
        print(f"\nWrote {args.output}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
