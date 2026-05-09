#!/usr/bin/env python3
"""Dump Triton compilation metadata (TTGIR/LLIR/PTX) for RMSNorm and GEMM.

S04 step 4: save generated-kernel artifacts so the report can analyze the
memory access, register usage, and launch differences between the Triton
kernels and the S03 CUDA kernels.

Usage:
    python3 scripts/bench/dump_triton_ir.py --output-dir reports/dev/s04/ir
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from ops.capability import detect_capabilities


def _compiled_kernels(kernel):
    """Yield (key, CompiledKernel) from the Triton 3.x device cache."""
    dc = kernel.device_caches
    for device, tup in dc.items():
        # Triton 3.7: device_caches[device] == (dict, dict, GPUTarget, backend, fn)
        cache_dict = tup[0]
        for key, compiled in cache_dict.items():
            yield key, compiled


def _dump(kernel, name: str, out_dir: str) -> list:
    """Save each compiled variant's IR/PTX and collect register metadata."""
    entries = []
    for idx, (key, compiled) in enumerate(_compiled_kernels(kernel)):
        asm = compiled.asm
        stages = {}
        for stage in ("ttgir", "llir", "ptx"):
            if stage in asm:
                path = os.path.join(out_dir, f"{name}_{idx}.{stage}")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(asm[stage])
                stages[stage] = path

        entries.append({
            "name": name,
            "variant_index": idx,
            "n_regs": getattr(compiled, "n_regs", None),
            "n_spills": getattr(compiled, "n_spills", None),
            "stages": stages,
            "ptx_line_count": asm.get("ptx", "").count("\n"),
        })
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump Triton IR/PTX metadata")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cap = detect_capabilities()
    if not cap.triton_available:
        print("Triton unavailable; nothing to dump.", file=sys.stderr)
        return 1

    out_dir = args.output_dir or os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        "reports", "dev", "s04", "ir",
    )
    os.makedirs(out_dir, exist_ok=True)

    from ops.triton.rmsnorm import _rmsnorm_kernel
    from ops.triton.gemm import _gemm_kernel

    # Trigger compilation.
    x = torch.randn(512, 2048, device="cuda", dtype=torch.float32)
    w = torch.randn(2048, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    _rmsnorm_kernel[(512,)](x, w, out, 2048, 1e-5, BLOCK=1024, num_warps=4)
    torch.cuda.synchronize()

    a = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    b = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    c = torch.empty(64, 64, device="cuda", dtype=torch.float32)
    _gemm_kernel[(1,)](a, b, c, 64, 64, 64, 64, 1, 1, 64, 64, 64,
                       BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4)
    torch.cuda.synchronize()

    metadata = (
        _dump(_rmsnorm_kernel, "rmsnorm_reference", out_dir)
        + _dump(_gemm_kernel, "gemm_reference", out_dir)
    )

    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    for m in metadata:
        print(f"{m['name']}[{m['variant_index']}]: n_regs={m['n_regs']} "
              f"n_spills={m['n_spills']} ptx_lines={m['ptx_line_count']}")
    print(f"\nArtifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
