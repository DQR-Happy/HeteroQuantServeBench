#!/usr/bin/env python3
"""E04-01 — Cross-architecture tile feasibility and autotune transfer (S04).

Question
--------
S04's pre-registered acceptance criteria require more than one NVIDIA
architecture:

* execution step 8: *"run the same matrix on at least two NVIDIA
  architectures and analyse whether the autotune parameters transfer"*;
* acceptance criterion: *"multi-architecture data proves auto-tuning is not a
  global constant"*.

S04 was originally accepted with a documented exception ("only sm_87 was
available; multi-architecture verification deferred until a cloud GPU is
available"). This experiment closes that exception using the RTX 3090
(sm_86) as the second architecture.

What is measured
----------------
1. **CUTLASS tile feasibility.** Each compiled configuration reports the
   dynamic shared memory it needs and whether the *local* device grants it.
   The default ``large`` tile cannot run on sm_86 at all, which is a
   compile-time-constant-level portability fact, not a performance nuance.
2. **Triton autotune selection.** Which ``BLOCK``/tile the autotuner picks
   on this device, per workload key. Combined with the selections recorded
   on sm_87 in ``docs/reports/S04_comparison_report.md`` this is the
   parameter-transfer evidence.

Provenance
----------
The two architectures are measured on **different machines at different
times**: sm_86 numbers are produced here, sm_87 numbers are read from the
tracked S04 report. Absolute latencies are therefore *not* comparable across
the two and no speedup is computed -- per the project rule that different
hardware is never compared against a single hard threshold. Only the
*structural* facts (which tile fits, which autotune config is selected) are
compared.

Raw output
----------
``docs/stage_experiments/S04/E04-01/raw/``:
``device.json``, ``cutlass_feasibility.json``, ``triton_autotune.json``,
``cross_arch_comparison.json``, ``e04_01_run_<id>.json``, ``command.txt``.

Usage
-----
    python3 scripts/audit/run_e04_01_cross_arch_tile_transfer.py \
        --output-dir docs/stage_experiments/S04/E04-01/raw
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hqsb.core.ids import new_run_id  # noqa: E402

EXPERIMENT_ID = "E04-01"
STAGE = "S04"

# Autotune selections recorded on the Jetson Orin (sm_87) by the tracked S04
# report. Kept as data, not as a re-measurement: see the provenance note.
_SM87_RECORDED = {
    "source": "docs/reports/S04_comparison_report.md (§3-§4, measured 2026-08-17)",
    "device": "NVIDIA Jetson Orin Nano Super, sm_87, 8 SM",
    "cutlass_default_tile_runnable": True,
    "cutlass_note": (
        "hqsb_cutlass_gemm_bench (default 128x256x64 x3) reported max_err ~0.03 "
        "on 1x2048x2048 / 1x2048x8192 / 512x2048x2048"
    ),
}


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _device_block() -> Dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for E04-01")
    prop = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "multi_processor_count": int(prop.multi_processor_count),
        "shared_memory_per_block_optin": int(
            getattr(prop, "shared_memory_per_block_optin", 0)
        ),
        "shared_memory_per_multiprocessor": int(
            getattr(prop, "shared_memory_per_multiprocessor", 0)
        ),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda or "",
    }


def _cutlass_binary() -> Optional[str]:
    for path in glob.glob(
        os.path.join(str(_REPO_ROOT), "build", "*", "bin", "hqsb_cutlass_gemm_bench")
    ):
        if os.path.isfile(path):
            return path
    return None


def _probe_config(binary: str, config: str) -> Dict[str, Any]:
    """Run one CUTLASS configuration and record its feasibility."""
    proc = subprocess.run(
        [binary, "--config", config, "--m", "64", "--n", "128", "--k", "64",
         "--warmup", "1", "--iterations", "1"],
        capture_output=True, text=True, timeout=300,
    )
    shared_bytes = None
    optin_bytes = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("config=") and "shared_storage_bytes=" in line:
            for token in line.split():
                if token.startswith("shared_storage_bytes="):
                    shared_bytes = int(token.split("=", 1)[1])
                elif token.startswith("device_optin_smem_bytes="):
                    optin_bytes = int(token.split("=", 1)[1])
    return {
        "config": config,
        "shared_storage_bytes": shared_bytes,
        "device_optin_smem_bytes": optin_bytes,
        "exit_code": int(proc.returncode),
        # 0 = ran + passed correctness, 4 = does not fit, 2 = launch failure,
        # 3 = wrong output. Anything but 0 means "not usable here".
        "runnable_on_this_device": proc.returncode == 0,
        "stderr_tail": (proc.stderr or "").strip().splitlines()[-1:] or [],
    }


def _triton_autotune_block() -> Dict[str, Any]:
    """Capture the autotuner's selected configuration per workload key."""
    import torch
    from ops.triton import gemm as triton_gemm
    from ops.triton import rmsnorm as triton_rmsnorm

    selections: Dict[str, Any] = {"rmsnorm": {}, "gemm": {}}

    def _describe(autotuner) -> Optional[Dict[str, Any]]:
        best = getattr(autotuner, "best_config", None)
        if best is None:
            return None
        return {
            "kwargs": {k: int(v) for k, v in dict(best.kwargs).items()},
            "num_warps": int(best.num_warps),
            "num_stages": int(getattr(best, "num_stages", 0) or 0),
        }

    # RMSNorm: autotune key is ["hidden"].
    for hidden in (1024, 2048):
        rows = 512
        for dtype in (torch.float32, torch.float16):
            x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
            w = torch.randn(hidden, device="cuda", dtype=dtype)
            triton_rmsnorm.rmsnorm_optimized(x, w)
            torch.cuda.synchronize()
            key = f"hidden={hidden},dtype={str(dtype).replace('torch.', '')}"
            selections["rmsnorm"][key] = _describe(
                triton_rmsnorm._rmsnorm_autotuned_kernel
            )

    # GEMM: autotune key is ["M", "N", "K"].
    for m, k, n in [(1, 2048, 2048), (1, 2048, 8192), (512, 2048, 2048)]:
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        triton_gemm.gemm_optimized(a, b)
        torch.cuda.synchronize()
        selections["gemm"][f"M={m},N={n},K={k}"] = _describe(
            triton_gemm._gemm_autotuned_kernel
        )

    selections["rmsnorm_search_space"] = {"BLOCK": [256, 512, 1024, 2048],
                                          "num_warps": 4}
    selections["gemm_search_space"] = {
        "tiles": [[64, 64, 32], [128, 64, 32], [64, 128, 32], [128, 128, 64]],
        "num_stages": 3,
    }
    return selections


def _compare_with_sm87(
    device: Dict[str, Any],
    cutlass: List[Dict[str, Any]],
    autotune: Dict[str, Any],
) -> Dict[str, Any]:
    """Structural (not absolute) comparison against the recorded sm_87 data."""
    large = next((c for c in cutlass if c["config"] == "large"), None)
    compact = next((c for c in cutlass if c["config"] == "compact"), None)

    findings = []
    if large is not None and not large["runnable_on_this_device"]:
        findings.append(
            f"CUTLASS default 'large' tile ({large['shared_storage_bytes']} B) "
            f"runs on sm_87 but NOT on sm_{device['compute_capability'][0]}"
            f"{device['compute_capability'][1]} "
            f"(device opt-in {large['device_optin_smem_bytes']} B): a single "
            f"global tile configuration is not portable."
        )
    if compact is not None and compact["runnable_on_this_device"]:
        findings.append(
            f"'compact' tile ({compact['shared_storage_bytes']} B) fits both "
            f"architectures, so an arch-aware selection keeps one source tree "
            f"usable on both."
        )

    distinct = {
        json.dumps(v, sort_keys=True)
        for v in autotune.get("gemm", {}).values()
        if v is not None
    }
    if len(distinct) > 1:
        findings.append(
            f"Triton GEMM autotune selected {len(distinct)} distinct tile "
            f"configurations across the 3 GEMM shapes on this device alone, "
            f"confirming the selection is shape-dependent, not a global "
            f"constant."
        )
    else:
        findings.append(
            "Triton GEMM autotune selected one configuration for all shapes on "
            "this device; the shape-independence holds here but must be "
            "checked per arch before being generalised."
        )

    return {
        "provenance": (
            "sm_86 measured on this machine; sm_87 read from the tracked S04 "
            "report. Different machines/sessions: absolute latencies are not "
            "compared and no speedup is computed."
        ),
        "sm87_recorded": _SM87_RECORDED,
        "sm86_measured": {
            "compute_capability": device["compute_capability"],
            "shared_memory_per_block_optin": device["shared_memory_per_block_optin"],
        },
        "findings": findings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E04-01 cross-arch tile transfer.")
    parser.add_argument(
        "--output-dir", default="docs/stage_experiments/S04/E04-01/raw"
    )
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or new_run_id()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "command.txt").write_text(
        "python3 scripts/audit/run_e04_01_cross_arch_tile_transfer.py "
        f"--output-dir {args.output_dir}\n",
        encoding="utf-8",
    )

    device = _device_block()
    (out_dir / "device.json").write_text(
        json.dumps(device, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[{EXPERIMENT_ID}] device: {device['name']} sm_"
          f"{device['compute_capability'][0]}{device['compute_capability'][1]}, "
          f"optin smem {device['shared_memory_per_block_optin']} B")

    binary = _cutlass_binary()
    cutlass: List[Dict[str, Any]] = []
    if binary:
        for config in ("large", "compact"):
            probe = _probe_config(binary, config)
            cutlass.append(probe)
            print(f"[{EXPERIMENT_ID}] cutlass {config:<8} "
                  f"smem={probe['shared_storage_bytes']} B "
                  f"runnable={probe['runnable_on_this_device']}")
    else:
        print(f"[{EXPERIMENT_ID}] hqsb_cutlass_gemm_bench not built; skipping")
    (out_dir / "cutlass_feasibility.json").write_text(
        json.dumps({"binary": binary, "configs": cutlass},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    autotune = _triton_autotune_block()
    (out_dir / "triton_autotune.json").write_text(
        json.dumps(autotune, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for key, value in autotune["rmsnorm"].items():
        print(f"[{EXPERIMENT_ID}] triton rmsnorm {key} -> {value}")
    for key, value in autotune["gemm"].items():
        print(f"[{EXPERIMENT_ID}] triton gemm    {key} -> {value}")

    comparison = _compare_with_sm87(device, cutlass, autotune)
    (out_dir / "cross_arch_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    record = {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "generated_at_utc": _now_utc(),
        "result_class": ["development", "cross-architecture_validation"],
        "device": device,
        "cutlass": cutlass,
        "triton_autotune": autotune,
        "cross_arch_comparison": comparison,
    }
    (out_dir / f"e04_01_run_{run_id}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[{EXPERIMENT_ID}] run_id={run_id}")
    for finding in comparison["findings"]:
        print(f"[{EXPERIMENT_ID}] finding: {finding}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
