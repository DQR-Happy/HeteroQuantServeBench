#!/usr/bin/env python3
"""S04 backend comparison benchmark: CUDA vs Triton RMSNorm, cuBLAS vs Triton GEMM.

All timings use the *same* method (torch.cuda.Event + median over reps) so
CUDA (ctypes), Triton, and cuBLAS (torch) are measured on equal footing.
Results are written as JSON for the comparison report.

Usage:
    python3 scripts/bench/bench_s04.py [--output reports/dev/s04/bench.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

from ops.capability import detect_capabilities
from ops.cuda_bridge import rmsnorm_forward as cuda_rmsnorm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)


def _median_time_ms(fn, warmup: int = 10, reps: int = 50) -> float:
    """Time ``fn`` (a callable performing one full kernel launch) via events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def bench_rmsnorm() -> list:
    """CUDA V0/V1/V2 vs Triton reference/optimized, FP32 + FP16."""
    import triton  # noqa: F401  (ensure importable before kernels)
    from ops.triton.rmsnorm import rmsnorm_optimized, rmsnorm_reference

    rows = 512
    shapes = [1024, 2048]
    results = []

    for dtype, dt in [("fp32", torch.float32), ("fp16", torch.float16)]:
        for hidden in shapes:
            torch.manual_seed(0)
            x = torch.randn(rows, hidden, device="cuda", dtype=dt)
            w = torch.randn(hidden, device="cuda", dtype=dt)

            entries = {}

            # CUDA variants (V0/V1 FP32-only; V2 both).
            if dtype == "fp32":
                for name, code in [("v0", 2), ("v1", 3), ("v2", 4)]:
                    entries[f"cuda_{name}"] = _median_time_ms(
                        lambda c=code: cuda_rmsnorm(x, w, dtype=dtype, variant=c)
                    )
            else:
                entries["cuda_v2"] = _median_time_ms(
                    lambda: cuda_rmsnorm(x, w, dtype=dtype, variant=4)
                )

            # Triton
            entries["triton_reference"] = _median_time_ms(
                lambda: rmsnorm_reference(x, w)
            )
            entries["triton_optimized"] = _median_time_ms(
                lambda: rmsnorm_optimized(x, w)
            )

            for name, ms in entries.items():
                results.append({
                    "op": "rmsnorm",
                    "dtype": dtype,
                    "hidden": hidden,
                    "rows": rows,
                    "backend": name,
                    "median_ms": round(ms, 4),
                })
    return results


def bench_gemm() -> list:
    """cuBLAS (torch.matmul) vs Triton GEMM, FP16."""
    from ops.triton.gemm import gemm_optimized, gemm_reference

    results = []
    cutlass_bin = _find_cutlass_bin()
    # Qwen3 decode-relevant GEMM shapes (M=1 batch, K=hidden, N=hidden).
    for m, k, n in [(1, 2048, 2048), (1, 2048, 8192), (512, 2048, 2048)]:
        torch.manual_seed(0)
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)

        cublas_ms = _median_time_ms(lambda: a @ b)
        triton_ms = _median_time_ms(lambda: gemm_optimized(a, b))
        # reference uses a fixed tile, measure for completeness
        triton_ref_ms = _median_time_ms(lambda: gemm_reference(a, b))

        results.append({
            "op": "gemm",
            "dtype": "fp16",
            "m": m, "k": k, "n": n,
            "backend": "cublas",
            "median_ms": round(cublas_ms, 4),
        })
        results.append({
            "op": "gemm",
            "dtype": "fp16",
            "m": m, "k": k, "n": n,
            "backend": "triton_optimized",
            "median_ms": round(triton_ms, 4),
        })
        results.append({
            "op": "gemm",
            "dtype": "fp16",
            "m": m, "k": k, "n": n,
            "backend": "triton_reference",
            "median_ms": round(triton_ref_ms, 4),
        })

        # CUTLASS (external C++ binary; skipped when not built).
        if cutlass_bin is not None:
            cutlass_ms, cutlass_err = _run_cutlass(cutlass_bin, m, n, k)
            if cutlass_ms is not None:
                results.append({
                    "op": "gemm",
                    "dtype": "fp16",
                    "m": m, "k": k, "n": n,
                    "backend": "cutlass_default",
                    "median_ms": round(cutlass_ms, 4),
                    "max_err": cutlass_err,
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
    """Run the CUTLASS binary once; return (median_ms, max_err) or (None, None)."""
    import subprocess

    try:
        out = subprocess.run(
            [binary, "--m", str(m), "--n", str(n), "--k", str(k)],
            capture_output=True, text=True, timeout=120,
        )
        if out.returncode != 0:
            return None, None
        # Output line: "fp16,m,n,k,cutlass_default,median_ms,max_err"
        line = out.stdout.strip().splitlines()[-1]
        parts = line.split(",")
        return float(parts[5]), float(parts[6])
    except (subprocess.TimeoutExpired, ValueError, IndexError, OSError):
        return None, None


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

    print("\n=== RMSNorm (median ms) ===")
    for r in rmsnorm_results:
        print(f"  {r['dtype']:>4} hidden={r['hidden']:>4} {r['backend']:<18} {r['median_ms']:.4f}")

    print("\n=== GEMM (median ms) ===")
    for r in gemm_results:
        print(f"  {r['dtype']:>4} {r['m']}x{r['k']}x{r['n']:<6} {r['backend']:<18} {r['median_ms']:.4f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({"capability": cap.as_dict(), "results": all_results}, fh, indent=2)
        print(f"\nWrote {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
