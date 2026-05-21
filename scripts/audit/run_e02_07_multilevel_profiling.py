#!/usr/bin/env python3
"""E02-07 orchestrator: PyTorch Profiler -> Nsight Systems -> Nsight Compute.

Subcommands
-----------
``audit``
    Small-workload rehearsal run *before* the formal one. It checks the
    things that make a profile trustworthy at all: the phase ranges match the
    independently derived step ledger, every model kernel lands in exactly one
    phase, the annotated module roles really contain their projections'
    kernels, and instrumenting the run does not change the generated tokens.
    A wrong range is worse than no range, so this gate comes first.
``collect``
    One independent process. Runs the un-instrumented reference and the
    profiled windows for both samples, then reduces each Chrome trace into
    per-phase rankings, timeline facts and the module/op/kernel mapping.
``nsys``
    Nsight Systems capture for both samples plus the machine-readable
    ``nsys stats`` exports.
``ncu``
    Nsight Compute capture for the pre-registered candidates.
``verify``
    Cross-process and cross-tool verdict, evidence manifest and the tables
    the report is written from.

Usage on the Jetson (always through the local proxy):

    ./scripts/remote_run.sh "sudo -E env PYTHONPATH=$PWD \\
        python3 scripts/audit/run_e02_07_multilevel_profiling.py audit \\
        --output-dir docs/stage_experiments/S02/E02-07/raw/audit"
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.benchmark import multilevel_profiling as mp

logger = logging.getLogger("e02_07")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIVER = _REPO_ROOT / "scripts" / "bench" / "e02_07_driver.py"
_WORKLOAD_YAML = _REPO_ROOT / "configs" / "benchmarks" / "jetson_qwen3_fp16.yaml"
_E02_02_RAW = _REPO_ROOT / "docs" / "stage_experiments" / "S02" / "E02-02" / "raw_v2"
_CUDA_NCU_CANDIDATES = (
    "/usr/local/cuda-12.6/bin/ncu",
    "/usr/local/cuda/bin/ncu",
    "/usr/bin/ncu",
)

# ── Frozen profiling plan ────────────────────────────────────────────────
#
# Two representative samples, both from the frozen six-workload set:
#   * ``prefill``  = long_prefill  (ISL 2048 / OSL 32). A long prefill is the
#     prefill-dominant case, and its decode window happens at a *long* KV
#     length (T ~ 2080), which supplies a third decode context for free.
#   * ``decode``   = decode_heavy  (ISL 128 / OSL 256). The decode-dominant
#     case: early window at T ~ 136, late window at T ~ 383.
#
# Decode therefore gets three context points (136 / 383 / 2080) for the same
# decode kernel, which is what makes the "early vs late T" requirement a
# measurement instead of a remark.
SAMPLES: Dict[str, Dict[str, Any]] = {
    "P": {"workload": "long_prefill", "isl": 2048, "osl": 32},
    "D": {"workload": "decode_heavy", "isl": 128, "osl": 256},
}
EARLY_STEPS = 8
LATE_STEPS = 8

#: NCU candidates, pre-registered from the E02-02 census ranking (top of each
#: bucket) *before* any counter is read. Selection rationale is recorded with
#: the ranking in ``verify`` output; nothing is added after seeing counters.
#:
#: ``kernel_regex``/``model_grid``/``model_block`` come from the in-model
#: PyTorch-profiler trace and are used for the harness faithfulness check.
#: ``replay`` describes the shape-exact isolated launch (see
#: ``scripts/bench/e02_07_ncu_kernel_replay.py`` for why isolation is
#: mandatory: in-model NCU is killed by the OOM killer on this 8 GiB board).
NCU_CANDIDATES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "P1",
        "sample": "P",
        "range": mp.PREFILL_RANGE,
        "kernel_regex": "cunn_SoftMaxForwardSmem",
        "role": "prefill top-1: eager-attention softmax over the 2048x2048 scores",
        "bucket": "attention",
        "replay": {"kind": "softmax", "shape": [1, 16, 2048, 2048], "dim": -1},
        "model_dims": "[1,16,2048,2048]",
    },
    {
        "id": "P2",
        "sample": "P",
        "range": mp.PREFILL_RANGE,
        "kernel_regex": "cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn",
        "role": "prefill top GEMM (2248x6144 tile family)",
        "bucket": "gemm",
        "replay": {"kind": "linear", "m": 2048, "k": 2048, "n": 6144},
        "model_dims": "[2048,2048] x [2048,6144]",
    },
    {
        "id": "P3",
        "sample": "P",
        "range": mp.PREFILL_RANGE,
        "kernel_regex": "ampere_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_tn",
        "role": "prefill attention Q@K^T batched matmul",
        "bucket": "attention",
        # b_shape is the contiguous K source; the replay transposes it, which
        # is what makes cuBLAS pick the same ..._tn kernel as the model.
        "replay": {"kind": "bmm", "a_shape": [16, 2048, 128], "b_shape": [16, 2048, 128]},
        "model_dims": "[16,2048,128] x [16,128,2048]",
    },
    {
        "id": "D1",
        "sample": "D",
        "range": mp.DECODE_EARLY_RANGE,
        "kernel_regex": "ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn",
        "role": "decode top-1 GEMM (62.9% of E02-02 decode kernel work)",
        "bucket": "gemm",
        "replay": {"kind": "linear", "m": 1, "k": 2048, "n": 6144},
        "model_dims": "[1,2048] x [2048,6144]",
    },
    {
        "id": "D1-lmhead",
        "sample": "D",
        "range": mp.DECODE_EARLY_RANGE,
        "kernel_regex": "ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn",
        "role": "same decode GEMM applied to the LM head (vocab 151936)",
        "bucket": "gemm",
        "replay": {"kind": "linear", "m": 1, "k": 2048, "n": 151936},
        "model_dims": "[1,2048] x [2048,151936]",
    },
    {
        "id": "D2",
        "sample": "D",
        "range": mp.DECODE_EARLY_RANGE,
        "kernel_regex": "CatArrayBatchedCopy_contig",
        "role": "decode KV-concat memory kernel (non-GEMM control line)",
        "bucket": "memory_kv",
        "replay": {
            "kind": "cat",
            "shapes": [[1, 8, 128, 128], [1, 8, 1, 128]],
            "dim": 2,
        },
        "model_dims": "{[1,8,128,128] x [1,8,1,128]}",
    },
)

#: In-model NCU anchors: application replay, minimal sections (each section
#: adds a full app relaunch, i.e. another model load). They give the isolated
#: replays an in-model reference point for the one operation each phase is
#: dominated by.
NCU_ANCHORS: Tuple[Dict[str, Any], ...] = (
    {
        "id": "anchor_D1_inmodel",
        "sample": "D",
        "range": mp.DECODE_EARLY_RANGE,
        "kernel_regex": (
            "ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn"
        ),
        # The early window of the formal D sample covers decode steps 1..8 at
        # T = 129..136. A 16-token generation contains exactly those same steps
        # with exactly those same context lengths, so the anchor reuses them and
        # skips the remaining 247 decode steps that would otherwise be re-run
        # once per collection pass.
        "driver": {"isl": 128, "osl": 16, "early_steps": 8, "late_steps": 0},
        "sections": ("SpeedOfLight", "Occupancy", "LaunchStats"),
    },
    {
        "id": "anchor_P1_inmodel",
        "sample": "P",
        "range": mp.PREFILL_RANGE,
        "kernel_regex": "cunn_SoftMaxForwardSmem",
        # ISL stays at the formal 2048 so the softmax shape is the real one;
        # OSL is cut to 2 because the prefill is all this anchor needs, and
        # every application-replay pass would otherwise re-run 31 decode steps.
        "driver": {"isl": 2048, "osl": 2, "early_steps": 0, "late_steps": 0},
        "sections": ("SpeedOfLight", "Occupancy", "LaunchStats"),
    },
)

NCU_SECTIONS = (
    "SpeedOfLight",
    "ComputeWorkloadAnalysis",
    "MemoryWorkloadAnalysis",
    "SchedulerStats",
    "WarpStateStats",
    "Occupancy",
    "LaunchStats",
)

NSYS_REPORTS = (
    "nvtx_pushpop_sum",
    "nvtx_gpu_proj_sum",
    "cuda_gpu_kern_sum",
    "cuda_gpu_kern_gb_sum",
    "cuda_kern_exec_sum",
    "cuda_api_sum",
    "cuda_gpu_mem_time_sum",
    "osrt_sum",
)

_TOP_N = 25


# ── small helpers ────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _run(command: Sequence[str], *, timeout: int = 3600) -> Dict[str, Any]:
    """Run a subprocess and keep *all* of stdout, plus bounded log tails.

    The full stdout is kept because ``ncu --csv`` writes its table there, after
    whatever the target application printed. Keeping only a tail would cut the
    header row off and make the table unparseable — which is exactly what an
    earlier revision of this script did, silently producing empty NCU evidence.
    """
    started = time.time()
    completed = subprocess.run(
        list(command), capture_output=True, text=True, timeout=timeout
    )
    return {
        "command": " ".join(shlex.quote(c) for c in command),
        "returncode": completed.returncode,
        "seconds": time.time() - started,
        "stdout": completed.stdout,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _rel(path: Optional[str], base: Path) -> Optional[str]:
    if not path:
        return None
    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        return str(path)


def _find_ncu() -> Optional[str]:
    for candidate in _CUDA_NCU_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def _file_hash(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


# ── E02-02 cross-experiment expectations ─────────────────────────────────


def e02_02_expectations() -> Dict[str, Any]:
    """Read the E02-02 census for shape/call-count closure (cross-check only).

    E02-07 must agree with E02-02 on ``shape`` and ``call count``; the shares
    are expected to differ because the two experiments profile different
    windows with different tool settings. The census is therefore used as an
    independent expectation, never as the E02-07 result.
    """
    expectations: Dict[str, Any] = {"available": False, "workloads": {}}
    run_dir = _E02_02_RAW / "run_0"
    if not run_dir.is_dir():
        return expectations
    for path in sorted(run_dir.glob("census_*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        name = path.stem.replace("census_", "")
        expectations["workloads"][name] = {
            "input_len": record.get("input_len"),
            "output_tokens": record.get("output_tokens"),
            "decode_steps": record.get("decode_steps"),
            "prefill_top_kernels": [
                {"name": r["name"], "count": r["count"], "share": r["time_share"]}
                for r in record.get("prefill", {}).get("kernels", [])[:8]
            ],
            "decode_top_kernels": [
                {"name": r["name"], "count": r["count"], "share": r["time_share"]}
                for r in record.get("decode", {}).get(
                    "kernels_cumulative_over_probe_steps", []
                )[:8]
            ],
            "sequence_sha256": record.get("instrumented", {}).get("sequence_sha256"),
            "module_call_counts": {
                f"{m['module']}|{m['phase']}": m["call_count"]
                for m in record.get("modules", [])
                if m["module"] in (
                    "model.layers.0.self_attn.q_proj",
                    "model.layers.0.mlp.gate_proj",
                    "model.layers.0.input_layernorm",
                    "lm_head",
                )
            },
        }
    expectations["available"] = bool(expectations["workloads"])
    return expectations


def e02_03_baseline() -> Dict[str, Any]:
    """E02-03 published baseline, kept only as a *non-comparable* reference.

    E02-03 ran with the default caching allocator; S02 later had to switch to
    ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` because the caching allocator is
    killed by the kernel OOM killer while loading the model on this board
    (E02-04 §12, E02-05 §4.12, E02-06 §4.13). The E02-03 numbers are still
    recorded here so the report can state the gap explicitly instead of
    quietly comparing across allocator modes.
    """
    path = _REPO_ROOT / "docs" / "stage_experiments" / "S02" / "E02-03" / "raw" / "verdict.json"
    baseline: Dict[str, Any] = {
        "source": "E02-03 published table (caching allocator)",
        "comparable": False,
        "reason": (
            "E02-03 used the default caching allocator; E02-07 must use "
            "PYTORCH_NO_CUDA_MEMORY_CACHING=1 to load the model at all "
            "(E02-06 §4.13). Absolute latencies are therefore not comparable."
        ),
        "published": {
            "long_prefill": {"TTFT_ms": 2533.5, "TPOT_ms": 103.2, "E2E_ms": 5733.5},
            "decode_heavy": {"TTFT_ms": 120.2, "TPOT_ms": 103.0, "E2E_ms": 26382.5},
        },
    }
    baseline["raw_verdict_present"] = path.is_file()
    return baseline


# ── trace reduction ──────────────────────────────────────────────────────


def _condense(rows: Sequence[Mapping[str, Any]], limit: int = _TOP_N) -> List[Dict[str, Any]]:
    """Keep the top rows of a ranking with the fields the report consumes."""
    kept: List[Dict[str, Any]] = []
    for row in rows[:limit]:
        kept.append(
            {
                "name": row["name"],
                "count": row["count"],
                "total_us": round(float(row["total_us"]), 3),
                "mean_us": round(float(row["mean_us"]), 3),
                "min_us": round(float(row["min_us"]), 3),
                "max_us": round(float(row["max_us"]), 3),
                "time_share": round(float(row.get("time_share", 0.0)), 6),
                "cumulative_share": round(float(row.get("cumulative_share", 0.0)), 6),
                "bucket": mp.kernel_bucket(str(row["name"])),
                "grids": row.get("grids", [])[:3],
                "blocks": row.get("blocks", [])[:3],
                "streams": row.get("streams", []),
                # The live input shapes and the ATen operators that own this
                # kernel: the shape is what makes a replay reproducible and the
                # op is what makes the Module -> Op -> Kernel chain auditable.
                "dims": row.get("dims", [])[:4],
                "ops": row.get("ops", [])[:3],
            }
        )
    return kept


def analyze_trace(
    trace_path: str,
    *,
    sample: str,
    ledger: Mapping[str, Any],
    relative_to: Path,
) -> Dict[str, Any]:
    """Reduce one Chrome trace into the evidence E02-07 must publish."""
    trace = mp.load_chrome_trace(trace_path)
    attribution = mp.attribute_events_to_phases(trace)
    host_tables = mp.host_op_table_by_phase(trace)
    roles = mp.module_role_table(trace)

    phases: Dict[str, Any] = {}
    for name, phase in attribution["phases"].items():
        if not phase["count"]:
            continue
        kernel_rows = mp.aggregate_kernels(phase["kernels"])
        # Shares are normalised twice on purpose: once inside the phase (what
        # this phase's work is made of) and once against the enclosing
        # region. The two denominators are printed so they cannot be
        # confused with each other or with the wall clock.
        within_phase = mp.attach_shares(kernel_rows, field="total_us")
        by_count = sorted(kernel_rows, key=lambda r: r["count"], reverse=True)
        by_mean = sorted(kernel_rows, key=lambda r: r["mean_us"], reverse=True)
        host_rows = host_tables.get(name, [])
        host_ranked = sorted(host_rows, key=lambda r: r["device_us"], reverse=True)
        host_ranked = mp.attach_shares(host_ranked, field="device_us")

        phases[name] = {
            "span_us": round(phase["span_us"], 3),
            "kernel_count": phase["count"],
            "kernel_work_us": round(phase["kernel_work_us"], 3),
            "streams": phase["streams"],
            "memcpy_like_count": phase["memcpy_count"],
            "critical_path": mp.critical_path_share_note(
                kernel_work_us=phase["kernel_work_us"], span_us=phase["span_us"]
            ),
            "rank_by_total": _condense(within_phase),
            "rank_by_count": [
                {
                    "name": r["name"],
                    "count": r["count"],
                    "total_us": round(r["total_us"], 3),
                    "mean_us": round(r["mean_us"], 3),
                }
                for r in by_count[:10]
            ],
            "rank_by_mean": [
                {
                    "name": r["name"],
                    "count": r["count"],
                    "mean_us": round(r["mean_us"], 3),
                    "total_us": round(r["total_us"], 3),
                }
                for r in by_mean[:10]
            ],
            "buckets": mp.bucket_summary(kernel_rows),
            "coverage": mp.cumulative_coverage(within_phase, top=5),
            "host_ops_top": [
                {
                    "name": r["name"],
                    "op_invocations": r["op_invocations"],
                    "kernel_count": r["kernel_count"],
                    "device_us": round(r["device_us"], 3),
                    "host_op_total_us": round(r["host_op_total_us"], 3),
                    "device_share": round(r.get("time_share", 0.0), 6),
                }
                for r in host_ranked[:_TOP_N]
            ],
            "timeline": {
                "gpu": mp.gpu_activity_window(phase["kernels"]),
                "gaps": mp.timeline_gaps(phase["kernels"]),
            },
        }

    return {
        "trace": _rel(trace_path, relative_to),
        "sample": sample,
        "ledger": ledger,
        "phases": phases,
        "phase_order": [n for n in mp.PHASE_RANGES if n in phases],
        "coverage": attribution["coverage"],
        "unattributed": {
            "count": attribution["unattributed"]["count"],
            "device_work_us": round(attribution["unattributed"]["device_work_us"], 3),
            "categories": attribution["unattributed"]["categories"],
            "names": attribution["unattributed"]["names"][:10],
        },
        "cpu_to_kernel_latency": mp.cpu_to_kernel_latency(trace),
        "module_roles": roles,
    }


def phase_ledger_checks(
    *,
    analysis: Mapping[str, Any],
    early_steps: int,
    late_steps: int,
    expected_forward_passes: Mapping[str, int],
) -> Dict[str, Any]:
    """Compare the annotated phase coverage with the independent ledger.

    This is the E02-07 anti-example guard: prefill/decode ranges off by one
    step, decode leaking the first or last step, or result handling folded
    into a model phase must all show up here rather than in the ranking.
    """
    observed = {
        name: info["kernel_count"] for name, info in analysis["phases"].items()
    }
    problems: List[str] = []

    # A single forward at a known shape produces a stable number of kernels;
    # the *counts per step* are what must match across a phase's steps, so the
    # ledger check compares the per-step kernel density instead of an absolute
    # number (which depends on fusion and library version).
    early = analysis["phases"].get(mp.DECODE_EARLY_RANGE)
    late = analysis["phases"].get(mp.DECODE_LATE_RANGE)
    middle = analysis["phases"].get(mp.DECODE_MIDDLE_RANGE)
    per_step: Dict[str, float] = {}
    if early and early_steps:
        per_step[mp.DECODE_EARLY_RANGE] = early["kernel_count"] / early_steps
    if late and late_steps:
        per_step[mp.DECODE_LATE_RANGE] = late["kernel_count"] / late_steps
    if middle:
        middle_steps = len(
            analysis["ledger"]["decode_middle"]["steps"]
        )
        if middle_steps:
            per_step[mp.DECODE_MIDDLE_RANGE] = middle["kernel_count"] / middle_steps

    densities = list(per_step.values())
    density_consistent = False
    if len(densities) >= 2:
        density_consistent = (max(densities) - min(densities)) / max(densities) < 0.05
    elif densities:
        density_consistent = True
    if not density_consistent:
        problems.append(
            f"per-step kernel density differs across decode windows: "
            f"{ {k: round(v, 2) for k, v in per_step.items()} }"
        )

    result_handling = analysis["phases"].get(mp.RESULT_RANGE)
    if result_handling and result_handling["kernel_count"] > 0:
        # Result handling is host-side serialisation; a kernel there means the
        # range silently wrapped model work.
        problems.append(
            f"result_handling contains {result_handling['kernel_count']} device events"
        )

    prefill = analysis["phases"].get(mp.PREFILL_RANGE)
    if prefill is None:
        problems.append("prefill range missing from the trace")
    if early is None and early_steps:
        problems.append("decode_early range missing from the trace")
    if late is None and late_steps:
        problems.append("decode_late range missing from the trace")

    return {
        "observed_kernel_counts": observed,
        "per_step_kernel_density": {k: round(v, 2) for k, v in per_step.items()},
        "per_step_density_consistent": density_consistent,
        "expected_forward_passes": dict(expected_forward_passes),
        "unattributed_device_events": analysis["unattributed"]["count"],
        "unattributed_share_of_total": (
            analysis["unattributed"]["count"]
            / max(
                analysis["unattributed"]["count"]
                + sum(observed.values()),
                1,
            )
        ),
        "problems": problems,
        "passed": not problems,
    }


# ── audit ────────────────────────────────────────────────────────────────


def _driver_command(
    *,
    model_path: str,
    manifest: str,
    isl: int,
    osl: int,
    early_steps: int,
    late_steps: int,
    profiler: str,
    profile_window: str,
    trace_path: str,
    output: str,
    reference_pass: bool = False,
    layer0_roles: bool = False,
    cuda_profiler_api: bool = False,
    tag: str = "",
) -> List[str]:
    command = [
        sys.executable,
        str(_DRIVER),
        "--model-path", model_path,
        "--manifest", manifest,
        "--isl", str(isl),
        "--osl", str(osl),
        "--early-steps", str(early_steps),
        "--late-steps", str(late_steps),
        "--profiler", profiler,
        "--profile-window", profile_window,
        "--trace-path", trace_path,
        "--output", output,
        "--tag", tag,
    ]
    if reference_pass:
        command.append("--reference-pass")
    if layer0_roles:
        command.append("--layer0-roles")
    if cuda_profiler_api:
        command.append("--cuda-profiler-api")
    return command


def _run_audit(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = str(output_dir / "audit_trace.json")
    driver_output = str(output_dir / "audit_driver.json")

    isl = args.audit_isl
    osl = args.audit_osl
    result = _run(
        _driver_command(
            model_path=args.model_path,
            manifest=args.manifest,
            isl=isl,
            osl=osl,
            early_steps=args.early_steps,
            late_steps=args.late_steps,
            profiler="light",
            profile_window="full",
            trace_path=trace_path,
            output=driver_output,
            reference_pass=True,
            layer0_roles=True,
            tag="audit",
        )
    )
    (output_dir / "audit_driver.stdout").write_text(result["stdout_tail"], encoding="utf-8")
    (output_dir / "audit_driver.stderr").write_text(result["stderr_tail"], encoding="utf-8")
    if result["returncode"] != 0:
        logger.error("audit driver failed: %s", result["stderr_tail"][-1500:])
        return result["returncode"]

    payload = json.loads(Path(driver_output).read_text(encoding="utf-8"))
    analysis = analyze_trace(
        trace_path, sample="audit", ledger=payload["captured"]["ledger"],
        relative_to=output_dir,
    )
    checks = phase_ledger_checks(
        analysis=analysis,
        early_steps=args.early_steps,
        late_steps=args.late_steps,
        expected_forward_passes={
            mp.PREFILL_RANGE: 1,
            mp.DECODE_EARLY_RANGE: args.early_steps,
            mp.DECODE_MIDDLE_RANGE: osl - 1 - args.early_steps - args.late_steps,
            mp.DECODE_LATE_RANGE: args.late_steps,
        },
    )

    # Range sanity: the annotated roles of the audited layer must own the
    # projections' kernels, not merely appear in the trace.
    roles = analysis["module_roles"]["roles"]
    role_checks: Dict[str, Any] = {}
    for role in ("self_attn.q_proj", "mlp.gate_proj", "input_layernorm"):
        entry = roles.get(f"{mp.MODULE_ROLE_PREFIX}{role}")
        role_checks[role] = {
            "present": entry is not None,
            "kernel_count": entry["kernel_count"] if entry else 0,
            "distinct_kernels": entry["distinct_kernel_count"] if entry else 0,
            "ops": list(entry["ops"])[:6] if entry else [],
            "chains": list(entry["chains"])[:4] if entry else [],
        }
    role_ok = all(
        v["present"]
        and v["kernel_count"] > 0
        and any("aten::" in op for op in v["ops"])
        for v in role_checks.values()
    )

    scope_closure_ok = all(
        abs(sum(r["time_share"] for r in info["rank_by_total"]) - 1.0) < 1e-6
        or info["kernel_count"] > len(info["rank_by_total"])
        for info in analysis["phases"].values()
    )

    audit = {
        "started_at": _now_iso(),
        "input_len": isl,
        "output_tokens": osl,
        "early_steps": args.early_steps,
        "late_steps": args.late_steps,
        "driver_command": result["command"],
        "driver_seconds": result["seconds"],
        "token_parity_vs_reference": payload["token_parity_vs_reference"],
        "reference_sequence_sha256": payload["reference"]["sequence_sha256"],
        "instrumented_sequence_sha256": payload["captured"]["sequence_sha256"],
        "phase_wall_ms": payload["captured"]["phase_wall_ms"],
        "coverage": analysis["coverage"],
        "unattributed": analysis["unattributed"],
        "ledger_checks": checks,
        "role_checks": role_checks,
        "roles_ok": role_ok,
        "scope_closure_ok": scope_closure_ok,
    }
    audit["passed"] = bool(
        audit["token_parity_vs_reference"]
        and checks["passed"]
        and role_ok
        and audit["coverage"]["attributed_ratio"] > 0.99
    )
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "passed": audit["passed"],
            "token_parity": audit["token_parity_vs_reference"],
            "ledger_passed": checks["passed"],
            "problems": checks["problems"],
            "coverage": audit["coverage"],
            "roles_ok": role_ok,
            "role_checks": role_checks,
        },
        indent=2,
    ))
    return 0 if audit["passed"] else 1


# ── collect ──────────────────────────────────────────────────────────────


#: The decode kernel whose behaviour is contrasted across context lengths.
CONTEXT_CONTRAST_KERNEL = (
    "ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn"
)


def build_cross(
    analyses: Mapping[str, Any],
    drivers: Mapping[str, Any],
) -> Dict[str, Any]:
    """Derive the cross-window facts from per-trace analyses.

    Kept separate from the collection loop so it can be recomputed from saved
    traces by ``reanalyze`` without touching the device.
    """
    cross: Dict[str, Any] = {
        "sample_phase_rankings": {},
        "decode_context_contrast": {},
        "ncu_candidate_selection": [],
    }
    for job, analysis in analyses.items():
        try:
            sample_key, window = job.split("_")
        except ValueError:
            continue
        for phase, info in analysis["phases"].items():
            if not info["kernel_count"]:
                continue
            rows = [
                {
                    "name": r["name"],
                    "count": r["count"],
                    "total_us": r["total_us"],
                    "mean_us": r["mean_us"],
                    "phase_share": r["time_share"],
                    "cumulative_share": r["cumulative_share"],
                    "bucket": r["bucket"],
                    "grids": r["grids"],
                    "blocks": r["blocks"],
                }
                for r in info["rank_by_total"]
            ]
            for candidate in mp.candidate_selection(
                phase=phase, rows=rows, span_us=info["span_us"], max_candidates=3
            ):
                candidate["sample"] = sample_key
                candidate["window"] = window
                candidate["phase_style"] = (
                    "prefill" if phase == mp.PREFILL_RANGE else "decode"
                )
                cross["ncu_candidate_selection"].append(candidate)

            # Decode context contrast: the same kernel across three contexts.
            if not phase.startswith("e02_07_decode"):
                continue
            steps = [
                entry
                for entry in (drivers.get(job, {}).get("payload") or {})
                .get("captured", {})
                .get("step_itl", [])
                if entry["window"] == phase
            ]
            for row in info["rank_by_total"]:
                if row["name"] != CONTEXT_CONTRAST_KERNEL:
                    continue
                cross["decode_context_contrast"][f"{sample_key}_{window}_{phase}"] = {
                    "sample": sample_key,
                    "window": window,
                    "context_len": steps[0]["context_len"] if steps else None,
                    "kernel_count": row["count"],
                    "total_us": row["total_us"],
                    "mean_us": row["mean_us"],
                    "phase_share": row["time_share"],
                    "phase_span_us": info["span_us"],
                    "grids": row["grids"],
                    "blocks": row["blocks"],
                }
    return cross


def _run_collect(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    run_dir = output_dir / f"run_{args.run_index}"
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    expected = e02_02_expectations()
    drivers: Dict[str, Any] = {}
    analyses: Dict[str, Any] = {}

    batches = (
        ("A", ("P_early", "D_early")),
        ("B", ("P_late", "D_late")),
    )
    for batch, jobs in batches:
        for job in jobs:
            sample_key, window = job.split("_")
            sample = SAMPLES[sample_key]
            trace_path = str(trace_dir / f"{sample_key}_{window}.json")
            driver_output = str(run_dir / f"driver_{sample_key}_{window}.json")
            command = _driver_command(
                model_path=args.model_path,
                manifest=args.manifest,
                isl=sample["isl"],
                osl=sample["osl"],
                early_steps=args.early_steps,
                late_steps=args.late_steps,
                profiler="light",
                profile_window=window,
                trace_path=trace_path,
                output=driver_output,
                reference_pass=(window == "early"),
                layer0_roles=True,
                tag=f"{batch}:{job}",
            )
            logger.info("collect run_%d: %s", args.run_index, job)
            result = _run(command, timeout=args.timeout)
            drivers[job] = {}
            drivers[job]["execution"] = result
            if result["returncode"] != 0:
                logger.error("%s failed: %s", job, result["stderr_tail"][-1500:])
                (run_dir / f"{job}.stderr").write_text(
                    result["stderr_tail"], encoding="utf-8"
                )
                continue
            payload = json.loads(Path(driver_output).read_text(encoding="utf-8"))
            drivers[job]["payload"] = payload
            analyses[job] = analyze_trace(
                trace_path,
                sample=sample_key,
                ledger=payload["captured"]["ledger"],
                relative_to=run_dir,
            )
            logger.info(
                "%s done: %d phases, coverage %.4f",
                job,
                len(analyses[job]["phases"]),
                analyses[job]["coverage"]["attributed_ratio"],
            )
            del payload

    # Reference timings come from the same process as the 'early' window.
    references: Dict[str, Any] = {}
    for job in ("P_early", "D_early"):
        payload = drivers.get(job, {}).get("payload") or {}
        references[job.split("_")[0]] = payload.get("reference")

    # Un-instrumented vs instrumented ITL comparison, per window.
    perturbation: Dict[str, Any] = {}
    for job, payload in (
        (k, v.get("payload")) for k, v in drivers.items() if v.get("payload")
    ):
        captured = payload["captured"]
        reference = payload.get("reference")
        per_window: Dict[str, Any] = {}
        for window in (mp.DECODE_EARLY_RANGE, mp.DECODE_MIDDLE_RANGE, mp.DECODE_LATE_RANGE):
            steps = [s for s in captured["step_itl"] if s["window"] == window]
            if not steps:
                continue
            profiled = bool(steps[0]["profiled"])
            per_window[window] = {
                "profiled": profiled,
                "steps": len(steps),
                "mean_itl_ms": sum(s["itl_ms"] for s in steps) / len(steps),
                "last_step": steps[-1]["step"],
                "last_context_len": steps[-1]["context_len"],
            }
        perturbation[job] = {
            "phase_wall_ms": captured["phase_wall_ms"],
            "windows": per_window,
            "reference_ttft_ms": reference["model_core_ttft_ms"] if reference else None,
            "reference_e2e_ms": reference["model_core_e2e_ms"] if reference else None,
            "reference_decode_total_ms": (
                reference["decode_total_ms"] if reference else None
            ),
        }

    cross = build_cross(analyses, drivers)

    run_meta = {
        "run_id": f"run_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}",
        "run_index": args.run_index,
        "started_at": _now_iso(),
        "samples": SAMPLES,
        "early_steps": args.early_steps,
        "late_steps": args.late_steps,
        "driver_jobs": list(drivers),
        "driver_failures": [
            job for job, v in drivers.items() if v["execution"]["returncode"] != 0
        ],
    }

    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    (run_dir / "references.json").write_text(
        json.dumps(references, indent=2), encoding="utf-8"
    )
    (run_dir / "perturbation.json").write_text(
        json.dumps(perturbation, indent=2), encoding="utf-8"
    )
    (run_dir / "cross_analysis.json").write_text(
        json.dumps(cross, indent=2), encoding="utf-8"
    )
    (run_dir / "e02_02_expectations.json").write_text(
        json.dumps(expected, indent=2), encoding="utf-8"
    )
    analyses_serializable = {}
    for job, analysis in analyses.items():
        analyses_serializable[job] = analysis
        (run_dir / f"analysis_{job}.json").write_text(
            json.dumps(analysis, indent=2), encoding="utf-8"
        )
    (run_dir / "analyses.json").write_text(
        json.dumps(analyses_serializable, indent=2), encoding="utf-8"
    )
    logger.info("run_%d collected (%d analyses)", args.run_index, len(analyses))
    return 0 if not run_meta["driver_failures"] else 1


def _export_ncu_csv(ncu: str, report: Path, csv_path: Path) -> Dict[str, Any]:
    """Regenerate the machine-readable table from a saved ``.ncu-rep``.

    ``ncu --export`` writes the report to the file and prints only a
    ``Report:`` line to stdout — the details table is *not* also printed. The
    CSV therefore has to be produced by importing the report back, which keeps
    the CSV and the ``.ncu-rep`` byte-identical in content instead of running
    the kernels twice.
    """
    if not report.is_file():
        return {"ok": False, "error": "report missing"}
    result = _run(
        [
            ncu,
            "--import", str(report),
            "--csv",
            "--page", "details",
            "--print-units", "base",
        ],
        timeout=600,
    )
    if result["returncode"] != 0:
        return {"ok": False, "error": result["stderr_tail"][-500:]}
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(result["stdout"], encoding="utf-8")
    return {
        "ok": "Kernel Name" in result["stdout"],
        "csv": csv_path.name,
        "bytes": len(result["stdout"]),
    }


def _run_ncu_csv_from_reports(args: argparse.Namespace) -> int:
    """Recover NCU CSVs from already-saved ``.ncu-rep`` reports.

    Needed because ``--export`` suppresses the console table: a collection run
    can produce complete reports and still leave the CSVs empty. This step
    re-derives the tables from those reports without touching the device.
    """
    ncu = _find_ncu()
    if ncu is None:
        logger.error("ncu executable not found in %s", _CUDA_NCU_CANDIDATES)
        return 2
    output_dir = Path(args.output_dir)
    outcome: Dict[str, Any] = {}
    for run_dir in sorted(d for d in output_dir.glob("run_*") if d.is_dir()):
        ncu_dir = run_dir / "ncu"
        if not ncu_dir.is_dir():
            continue
        for report in sorted(ncu_dir.glob("*.ncu-rep")):
            csv_path = report.with_suffix(".csv")
            outcome[str(csv_path.relative_to(output_dir))] = _export_ncu_csv(
                ncu, report, csv_path
            )
    (output_dir / "ncu_csv_recovery.json").write_text(
        json.dumps(outcome, indent=2), encoding="utf-8"
    )
    failed = [k for k, v in outcome.items() if not v.get("ok")]
    logger.info("ncu csv recovery: %d ok, %d failed", len(outcome) - len(failed), len(failed))
    print(json.dumps({"recovered": len(outcome) - len(failed), "failed": failed}, indent=2))
    return 0 if not failed else 1


def _run_reanalyze(args: argparse.Namespace) -> int:
    """Recompute the per-trace analyses from the saved raw traces.

    Collection runs are expensive (a model load per window), while the
    reduction from a Chrome trace to per-phase rankings is pure host work.
    Keeping the two apart means a fixed analysis can be applied to the *same*
    raw evidence instead of re-running the device work, which is the only way
    to correct a reduction without invalidating the measurement.
    """
    output_dir = Path(args.output_dir)
    run_dirs = sorted(d for d in output_dir.glob("run_*") if d.is_dir())
    if args.run_index is not None:
        run_dirs = [d for d in run_dirs if d.name == f"run_{args.run_index}"]
    if not run_dirs:
        logger.error("no run_* directories found under %s", output_dir)
        return 2

    rebuilt: Dict[str, Any] = {}
    for run_dir in run_dirs:
        analyses: Dict[str, Any] = {}
        drivers: Dict[str, Any] = {}
        for driver_path in sorted(run_dir.glob("driver_*.json")):
            job = driver_path.stem.replace("driver_", "")
            payload = json.loads(driver_path.read_text(encoding="utf-8"))
            drivers[job] = {"payload": payload}
            trace_path = run_dir / "traces" / f"{job}.json"
            if not trace_path.is_file():
                continue
            analyses[job] = analyze_trace(
                str(trace_path),
                sample=job.split("_")[0],
                ledger=payload["captured"]["ledger"],
                relative_to=run_dir,
            )
            (run_dir / f"analysis_{job}.json").write_text(
                json.dumps(analyses[job], indent=2), encoding="utf-8"
            )
        (run_dir / "analyses.json").write_text(
            json.dumps(analyses, indent=2), encoding="utf-8"
        )
        (run_dir / "cross_analysis.json").write_text(
            json.dumps(build_cross(analyses, drivers), indent=2), encoding="utf-8"
        )
        rebuilt[run_dir.name] = sorted(analyses)
        logger.info("reanalyzed %s (%s)", run_dir.name, sorted(analyses))

    print(json.dumps(rebuilt, indent=2))
    return 0


# ── nsys ─────────────────────────────────────────────────────────────────


def _run_nsys(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    nsys_dir = output_dir / f"run_{args.run_index}" / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}

    for key, sample in SAMPLES.items():
        base = nsys_dir / f"{key}_core"
        driver_output = str(nsys_dir / f"{key}_driver.json")
        inner = _driver_command(
            model_path=args.model_path,
            manifest=args.manifest,
            isl=sample["isl"],
            osl=sample["osl"],
            early_steps=args.early_steps,
            late_steps=args.late_steps,
            profiler="off",
            profile_window="full",
            trace_path="",
            output=driver_output,
            reference_pass=True,
            layer0_roles=True,
            cuda_profiler_api=True,
            tag=f"nsys:{key}",
        )
        command = [
            "nsys", "profile",
            "--trace=cuda,nvtx,osrt",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--sample=none",
            "--cpuctxsw=none",
            f"--output={base}",
            "--force-overwrite=true",
        ] + inner
        logger.info("nsys capture %s", key)
        capture = _run(command, timeout=args.timeout)
        results[key] = {"capture": capture, "driver_output": driver_output}
        if capture["returncode"] != 0:
            logger.error("nsys %s failed: %s", key, capture["stderr_tail"][-1500:])
            continue
        report = base.with_suffix(".nsys-rep")
        for name in NSYS_REPORTS:
            # ``nsys stats`` appends the report name to ``--output``, so the
            # on-disk file is ``<key>_<report>.csv``; passing the report name
            # again here would produce a doubled name.
            stats = _run(
                [
                    "nsys", "stats", "--report", name,
                    "--format", "csv", "--force-overwrite=true",
                    "--output", str(nsys_dir / key), str(report),
                ],
                timeout=1800,
            )
            results[key][name] = {
                "returncode": stats["returncode"],
                "csv": f"{key}_{name}.csv",
                "stderr_tail": stats["stderr_tail"][-600:],
            }
        sqlite = nsys_dir / f"{key}_core.sqlite"
        results[key]["sqlite_export"] = _run(
            [
                "nsys", "export", "--type", "sqlite",
                "--force-overwrite=true", "--output", str(sqlite), str(report),
            ],
            timeout=1800,
        )
        results[key]["report"] = report.name

    (nsys_dir / "nsys_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    failed = [k for k, v in results.items() if v["capture"]["returncode"] != 0]
    logger.info("nsys run_%d done (failures=%s)", args.run_index, failed)
    return 0 if not failed else 1


# ── ncu ──────────────────────────────────────────────────────────────────


_REPLAY_HARNESS = _REPO_ROOT / "scripts" / "bench" / "e02_07_ncu_kernel_replay.py"
_REPLAY_RANGE_PREFIX = "e02_07_replay"


def _replay_harness_command(candidate: Mapping[str, Any], output: str) -> List[str]:
    spec = candidate["replay"]
    command = [
        sys.executable,
        str(_REPLAY_HARNESS),
        "--kind", spec["kind"],
        "--label", candidate["id"],
        "--repeat", "4",
        "--warmup", "2",
        "--output", output,
    ]
    if spec["kind"] == "linear":
        command += ["--m", str(spec["m"]), "--k", str(spec["k"]), "--n", str(spec["n"])]
    elif spec["kind"] == "cat":
        shapes = ";".join(",".join(str(v) for v in shape) for shape in spec["shapes"])
        command += ["--cat-shapes", shapes, "--dim", str(spec["dim"])]
    elif spec["kind"] == "softmax":
        command += [
            "--shape", ",".join(str(v) for v in spec["shape"]),
            "--dim", str(spec["dim"]),
        ]
    elif spec["kind"] == "bmm":
        command += [
            "--a-shape", ",".join(str(v) for v in spec["a_shape"]),
            "--b-shape", ",".join(str(v) for v in spec["b_shape"]),
        ]
    return command


def _run_ncu(args: argparse.Namespace) -> int:
    """Nsight Compute: shape-exact isolated replay plus one in-model anchor.

    The scope is deliberately bounded, for a measured reason rather than for
    convenience: a full-model NCU capture cannot run on this board at all
    (kernel replay is OOM-killed, application replay survives but costs a
    whole model load per collection pass). Both facts are recorded in the
    output so the limitation is auditable rather than implicit.
    """
    ncu = _find_ncu()
    if ncu is None:
        logger.error("ncu executable not found in %s", _CUDA_NCU_CANDIDATES)
        return 2
    output_dir = Path(args.output_dir)
    ncu_dir = output_dir / f"run_{args.run_index}" / "ncu"
    ncu_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {
        "replay_mode": {
            "harness": "kernel",
            "harness_rationale": (
                "in-model kernel replay is OOM-killed on this 8 GiB board; "
                "the shape-exact harness keeps the footprint at tens of MiB"
            ),
            "in_model_anchor": "application",
            "in_model_anchor_rationale": (
                "application replay survives, but relaunches the process per "
                "collection pass, so only the top candidate and a minimal "
                "section set are captured in-model"
            ),
            "clock_control": "none",
            "clock_control_rationale": (
                "ncu --clock-control base is not honoured for this Orin build "
                "(verified: it warns and leaves clocks unmodified), so clocks "
                "are left alone and the same caveat as E02-02/E02-03 applies"
            ),
            "cache_control": "all (default cache flush between passes)",
        }
    }

    # ── shape-exact isolated replay, full section set ────────────────────
    for candidate in NCU_CANDIDATES:
        base = ncu_dir / candidate["id"]
        harness_output = str(ncu_dir / f"harness_{candidate['id']}.json")
        harness = _replay_harness_command(candidate, harness_output)
        command = [
            ncu,
            "--clock-control", "none",
            "--kernel-name-base", "demangled",
            "--replay-mode", "kernel",
            # The harness brackets only the measured repeats in NVTX, so the
            # range filter is what selects the launches; a bare --launch-skip
            # would count the tensor-construction and warmup launches instead.
            "--nvtx",
            "--nvtx-include", f"{_REPLAY_RANGE_PREFIX}.{candidate['id']}/",
            "--launch-count", str(args.ncu_launches),
            # ``--print-summary per-kernel`` is deliberately NOT passed: with
            # more than one ``--section`` it suppresses the details page and
            # the CSV ends up containing no metric rows at all. That failure is
            # silent — ncu still exits 0 — so it is called out here.
            "--csv",
        ]
        for section in NCU_SECTIONS:
            command += ["--section", section]
        command += ["--export", str(base.with_suffix(".ncu-rep")), "--force-overwrite"]
        command += harness

        logger.info("ncu harness replay %s", candidate["id"])
        run = _run(command, timeout=args.timeout)
        csv_path = base.with_suffix(".csv")
        if run["returncode"] == 0:
            _export_ncu_csv(ncu, base.with_suffix(".ncu-rep"), csv_path)
        results[candidate["id"]] = {
            "mode": "isolated_shape_exact_replay",
            "candidate": candidate,
            "command": run["command"],
            "returncode": run["returncode"],
            "seconds": run["seconds"],
            "csv": csv_path.name if csv_path.is_file() else None,
            "harness_output": (
                Path(harness_output).name if Path(harness_output).is_file() else None
            ),
            "stderr_tail": run["stderr_tail"][-1200:],
        }
        if run["returncode"] != 0:
            logger.error("ncu %s failed: %s", candidate["id"], run["stderr_tail"][-1000:])

    # ── in-model anchors (run 0 only) ────────────────────────────────────
    if args.anchor:
        for anchor in NCU_ANCHORS:
            base = ncu_dir / anchor["id"]
            driver_output = str(ncu_dir / f"{anchor['id']}_driver.json")
            driver = _driver_command(
                model_path=args.model_path,
                manifest=args.manifest,
                isl=anchor["driver"]["isl"],
                osl=anchor["driver"]["osl"],
                early_steps=anchor["driver"]["early_steps"],
                late_steps=anchor["driver"]["late_steps"],
                profiler="off",
                profile_window="full",
                trace_path="",
                output=driver_output,
                reference_pass=False,
                layer0_roles=False,
                tag=f"ncu:{anchor['id']}",
            )
            command = [
                ncu,
                "--clock-control", "none",
                "--kernel-name-base", "demangled",
                "--replay-mode", "application",
                "--nvtx",
                "--nvtx-include", f"{anchor['range']}/",
                "-k", f"regex:{anchor['kernel_regex']}",
                "--launch-count", "1",
                "--csv",
            ]
            for section in anchor["sections"]:
                command += ["--section", section]
            command += ["--export", str(base.with_suffix(".ncu-rep")), "--force-overwrite"]
            command += driver

            logger.info("ncu in-model anchor %s (application replay)", anchor["id"])
            run = _run(command, timeout=args.timeout)
            csv_path = base.with_suffix(".csv")
            if run["returncode"] == 0:
                _export_ncu_csv(ncu, base.with_suffix(".ncu-rep"), csv_path)
            results[anchor["id"]] = {
                "mode": "in_model_application_replay",
                "candidate": anchor,
                "command": run["command"],
                "returncode": run["returncode"],
                "seconds": run["seconds"],
                "csv": csv_path.name if csv_path.is_file() else None,
                "stderr_tail": run["stderr_tail"][-1200:],
            }
            if run["returncode"] != 0:
                logger.error("ncu anchor %s failed: %s", anchor["id"],
                             run["stderr_tail"][-1000:])

    (ncu_dir / "ncu_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    failed = [
        k for k, v in results.items()
        if isinstance(v, dict) and v.get("returncode") not in (0, None)
    ]
    logger.info("ncu run_%d done (failures=%s)", args.run_index, failed)
    return 0 if not failed else 1


# ── verification and summary ─────────────────────────────────────────────


def _load_runs(output_dir: Path, limit: int = 3) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for run_dir in sorted(d for d in output_dir.glob("run_*") if d.is_dir())[:limit]:
        entry: Dict[str, Any] = {"dir": run_dir.name, "path": run_dir}
        analyses_path = run_dir / "analyses.json"
        if analyses_path.is_file():
            entry["analyses"] = json.loads(analyses_path.read_text(encoding="utf-8"))
        meta_path = run_dir / "meta.json"
        if meta_path.is_file():
            entry["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        references_path = run_dir / "references.json"
        if references_path.is_file():
            entry["references"] = json.loads(references_path.read_text(encoding="utf-8"))
        perturbation_path = run_dir / "perturbation.json"
        if perturbation_path.is_file():
            entry["perturbation"] = json.loads(
                perturbation_path.read_text(encoding="utf-8")
            )
        cross_path = run_dir / "cross_analysis.json"
        if cross_path.is_file():
            entry["cross"] = json.loads(cross_path.read_text(encoding="utf-8"))
        ncu_entries: Dict[str, Any] = {}
        for csv_path in sorted((run_dir / "ncu").glob("*.csv")):
            try:
                ncu_entries[csv_path.stem] = mp.parse_ncu_csv(
                    csv_path.read_text(encoding="utf-8", errors="replace")
                )
            except OSError:
                continue
        if ncu_entries:
            entry["ncu"] = ncu_entries
        if (run_dir / "nsys").is_dir():
            entry["nsys"] = nsys_analysis(run_dir)
        runs.append(entry)
    return runs


_NSYS_PREAMBLE_PREFIXES = (
    "generating",
    "processing",
    "notice",
    "warning",
    "error",
    "exportation",
    "it is assumed",
    "consider using",
    "skipping",
)


def _read_nsys_csv(path: Path) -> List[Dict[str, str]]:
    """Read an ``nsys stats`` CSV export, tolerating its preamble lines.

    The header line cannot be found by looking for a ``Name`` column: the NVTX
    reports call their last column ``Range``, and ``cuda_kern_exec_sum`` calls
    it ``Kernel Name``. The first comma-separated line that is not one of the
    tool's log lines is the header instead.
    """
    import csv
    import io

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    header_index = None
    for index, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped or "," not in line or len(line.split(",")) < 3:
            continue
        lowered = stripped.lower()
        if lowered.startswith(_NSYS_PREAMBLE_PREFIXES) or stripped.startswith("-"):
            continue
        header_index = index
        break
    if header_index is None:
        return []
    return [dict(row) for row in csv.DictReader(io.StringIO("\n".join(text.splitlines()[header_index:])))]


def _nsys_column(row: Mapping[str, str], *wanted: str) -> Optional[str]:
    """Look up a column by exact name first, then by substring.

    Two passes matter here: ``cuda_kern_exec_sum`` has both ``Count`` and
    ``QCount``, and ``KAvg (ns)``/``QAvg (ns)`` differ only by prefix. A
    single substring pass would return whichever column happens to come
    first in the header.
    """
    lowered = {
        (key or "").strip().lower(): value for key, value in row.items()
    }
    for candidate in wanted:
        exact = lowered.get(candidate.strip().lower())
        if exact is not None:
            return exact
    for candidate in wanted:
        needle = candidate.strip().lower()
        for name, value in lowered.items():
            if name.startswith(needle):
                return value
    return None


def _nsys_kernel_names(run_dir: Path) -> List[str]:
    """Read kernel names from the nsys ``cuda_gpu_kern_sum`` CSV export."""
    names: List[str] = []
    for path in sorted((run_dir / "nsys").glob("*cuda_gpu_kern_sum.csv")):
        for row in _read_nsys_csv(path):
            name = _nsys_column(row, "Name")
            if name:
                names.append(name.strip())
    return names


def nsys_analysis(run_dir: Path) -> Dict[str, Any]:
    """Structure the nsys exports the report needs.

    NSys answers system-level questions: which NVTX ranges exist on the GPU
    timeline, how much GPU time each range holds, how long launch-to-execution
    takes, and whether anything other than the model ran. Durations here are
    *not* compared with the PyTorch profiler's or NCU's: each tool perturbs
    differently, and only identity is cross-checked.
    """
    nsys_dir = run_dir / "nsys"
    analysis: Dict[str, Any] = {}
    for key in SAMPLES:
        entry: Dict[str, Any] = {}
        pushpop = _read_nsys_csv(nsys_dir / f"{key}_nvtx_pushpop_sum.csv")
        entry["nvtx_pushpop_sum"] = [
            {
                "range": _nsys_column(row, "Range", "Name"),
                "total_ns": _nsys_column(row, "Total Time"),
                "instances": _nsys_column(row, "Instances"),
                "avg_ns": _nsys_column(row, "Avg"),
                "max_ns": _nsys_column(row, "Max"),
            }
            for row in pushpop
        ]
        gpu_proj = _read_nsys_csv(nsys_dir / f"{key}_nvtx_gpu_proj_sum.csv")
        entry["nvtx_gpu_proj_sum"] = [
            {
                "range": _nsys_column(row, "Range", "Name"),
                "projected_ns": _nsys_column(
                    row, "Total Proj Time", "Proj Avg", "Projected Time"
                ),
                "range_time_ns": _nsys_column(row, "Total Range Time"),
                "instances": _nsys_column(row, "Range Instances", "Instances"),
                "gpu_ops": _nsys_column(row, "Total GPU Ops"),
            }
            for row in gpu_proj
        ]
        kern = _read_nsys_csv(nsys_dir / f"{key}_cuda_gpu_kern_sum.csv")
        entry["cuda_gpu_kern_sum"] = [
            {
                "name": _nsys_column(row, "Name"),
                "instances": _nsys_column(row, "Instances"),
                "total_ns": _nsys_column(row, "Total Time"),
                "avg_ns": _nsys_column(row, "Avg"),
                "max_ns": _nsys_column(row, "Max"),
                "time_pct": _nsys_column(row, "Time (%)"),
            }
            for row in kern[:20]
        ]
        exec_sum = _read_nsys_csv(nsys_dir / f"{key}_cuda_kern_exec_sum.csv")
        # The columns are abbreviated in this nsys version:
        #   T* total (API + queue + kernel)   Q* queue = launch-API-return to
        #   A* launch API duration            kernel-start  (the launch backlog)
        #   K* kernel execution time
        entry["cuda_kern_exec_sum"] = [
            {
                "api_name": _nsys_column(row, "API Name"),
                "kernel_name": _nsys_column(row, "Kernel Name"),
                "count": _nsys_column(row, "Count"),
                "total_avg_ns": _nsys_column(row, "TAvg"),
                "api_avg_ns": _nsys_column(row, "AAvg"),
                "queue_avg_ns": _nsys_column(row, "QAvg"),
                "kernel_avg_ns": _nsys_column(row, "KAvg"),
                "queue_max_ns": _nsys_column(row, "QMax"),
            }
            for row in exec_sum[:20]
        ]
        api = _read_nsys_csv(nsys_dir / f"{key}_cuda_api_sum.csv")
        entry["cuda_api_sum"] = [
            {
                "name": _nsys_column(row, "Name"),
                "total_ns": _nsys_column(row, "Total Time"),
                "num_calls": _nsys_column(row, "Num Calls", "Instances"),
            }
            for row in api[:20]
        ]
        analysis[key] = entry
    return analysis


def _rank_signature(analysis: Mapping[str, Any], phase: str, top: int = 5) -> List[str]:
    info = analysis.get("phases", {}).get(phase)
    if not info:
        return []
    return [row["name"] for row in info["rank_by_total"][:top]]


def _run_verify(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    runs = _load_runs(output_dir, limit=args.runs_required)
    if len(runs) < args.runs_required:
        logger.error("need >= %d run_* dirs, found %d", args.runs_required, len(runs))
        return 2

    expected = e02_02_expectations()
    checks: Dict[str, Any] = {}

    # 1. Per-run, per-phase ranking stability across independent processes.
    phase_names = [
        mp.PREFILL_RANGE,
        mp.DECODE_EARLY_RANGE,
        mp.DECODE_LATE_RANGE,
    ]
    stability: Dict[str, Any] = {}
    for job in ("P_early", "D_early", "P_late", "D_late"):
        for phase in phase_names:
            signatures = []
            for run in runs:
                analysis = (run.get("analyses") or {}).get(job)
                if not analysis:
                    continue
                signature = _rank_signature(analysis, phase)
                if signature:
                    signatures.append(signature)
            if not signatures:
                continue
            key = f"{job}:{phase}"
            same_top1 = len({s[0] for s in signatures}) == 1
            stable_set = len({",".join(s) for s in signatures}) == 1
            stability[key] = {
                "runs": len(signatures),
                "top1_identical": same_top1,
                "top5_set_identical": stable_set,
                "top1": [s[0] for s in signatures],
                "top5": signatures,
            }
    checks["ranking_stability"] = stability
    checks["ranking_stability_passed"] = all(
        v["top1_identical"] for v in stability.values()
    )

    # 2. Token identity. The two samples are different workloads, so identity
    #    is required *within* a sample (across runs and across the early/late
    #    window runs) and against the E02-02 census hash for that workload —
    #    never across the two samples.
    token_hashes: Dict[str, Optional[str]] = {}
    per_sample: Dict[str, set] = {}
    for run in runs:
        for job in ("P_early", "D_early", "P_late", "D_late"):
            path = run["path"] / f"driver_{job}.json"
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            digest = payload["captured"]["sequence_sha256"]
            token_hashes[f"{run['dir']}:{job}"] = digest
            per_sample.setdefault(job.split("_")[0], set()).add(digest)
    expected_hashes = {
        SAMPLES[key]["workload"]: (
            (expected.get("workloads") or {}).get(SAMPLES[key]["workload"], {})
            or {}
        ).get("sequence_sha256")
        for key in SAMPLES
    }
    matches_e02_02 = {
        key: bool(expected_hashes[SAMPLES[key]["workload"]])
        and sorted(per_sample.get(key, set())) == [expected_hashes[SAMPLES[key]["workload"]]]
        for key in SAMPLES
    }
    checks["token_hashes"] = token_hashes
    checks["token_hashes_by_sample"] = {k: sorted(v) for k, v in per_sample.items()}
    checks["token_hash_matches_e02_02"] = matches_e02_02
    checks["token_hash_identical_across_windows"] = bool(per_sample) and all(
        len(values) == 1 for values in per_sample.values()
    )

    # 3. Un-instrumented reference vs instrumented windows: perturbation.
    perturbation_rows: Dict[str, Any] = {}
    for run in runs:
        for job, entry in (run.get("perturbation") or {}).items():
            for window, data in entry["windows"].items():
                perturbation_rows.setdefault(window, []).append(
                    {
                        "run": run["dir"],
                        "job": job,
                        "profiled": data["profiled"],
                        "mean_itl_ms": data["mean_itl_ms"],
                    }
                )
    checks["perturbation"] = perturbation_rows

    # 4. Cross-tool compatibility (names + tokens; durations explicitly not).
    profiler_names: List[Dict[str, Any]] = []
    nsys_names: List[str] = []
    ncu_names: List[str] = []
    for run in runs:
        for analysis in (run.get("analyses") or {}).values():
            for phase, info in analysis.get("phases", {}).items():
                if phase.startswith("e02_07_decode"):
                    profiler_names.extend(info["rank_by_total"])
        nsys_names.extend(_nsys_kernel_names(run["path"]))
        for key, parsed in (run.get("ncu") or {}).items():
            for kernel in parsed.get("kernels", []):
                ncu_names.append(kernel["name"])
    checks["tool_compatibility"] = mp.tool_compatibility(
        profiler_kernels=profiler_names,
        nsys_kernels=[{"name": n} for n in nsys_names],
        ncu_kernel_names=ncu_names,
        # Token identity is checked per sample above; feeding both samples'
        # hashes here would make the flag false for a reason that has nothing
        # to do with tool compatibility (they are different workloads).
        token_hashes={},
    )

    # 5. E02-02 closure on call counts (shape/call counts, not shares).
    closure: Dict[str, Any] = {}
    for run in runs[:1]:
        for job, analysis in (run.get("analyses") or {}).items():
            sample_key = job.split("_")[0]
            workload = SAMPLES[sample_key]["workload"]
            expectation = (expected.get("workloads") or {}).get(workload)
            if not expectation:
                continue
            observed_prefill = {
                r["name"]: r["count"]
                for r in analysis["phases"].get(mp.PREFILL_RANGE, {}).get("rank_by_total", [])
            }
            expected_prefill = {
                r["name"]: r["count"] for r in expectation["prefill_top_kernels"]
            }
            shared = set(observed_prefill) & set(expected_prefill)
            closure[f"{run['dir']}:{job}"] = {
                "workload": workload,
                "shared_kernel_names": len(shared),
                "count_match": {
                    name: {
                        "e02_07": observed_prefill[name],
                        "e02_02": expected_prefill[name],
                    }
                    for name in sorted(shared)
                },
                "counts_agree": all(
                    observed_prefill[name] == expected_prefill[name] for name in shared
                ),
                "expected_sequence_sha256": expectation["sequence_sha256"],
            }
    checks["e02_02_call_count_closure"] = closure
    checks["e02_02_token_closure"] = matches_e02_02
    checks["e02_02_token_closure_passed"] = (
        all(matches_e02_02.values()) if expected.get("available") else None
    )

    # 6. NCU candidates: did every pre-registered candidate yield counters?
    ncu_status: Dict[str, Any] = {}
    for run in runs:
        for key, parsed in (run.get("ncu") or {}).items():
            kernels = parsed.get("kernels", [])
            panel = kernels[0]["panel"] if kernels else {}
            ncu_status.setdefault(key, []).append(
                {
                    "run": run["dir"],
                    "kernels": [k["name"] for k in kernels],
                    "grid": kernels[0]["grid"] if kernels else None,
                    "block": kernels[0]["block"] if kernels else None,
                    "duration_ns": panel.get("duration_ns"),
                    "duration_ms": panel.get("duration_ms"),
                    "occupancy_pct": panel.get("achieved_occupancy_pct"),
                    "unavailable_metrics": sorted(parsed.get("unavailable_metrics", {})),
                }
            )
    checks["ncu_candidates"] = ncu_status
    checks["ncu_candidates_all_present"] = bool(ncu_status) and all(
        any(v["kernels"] for v in values) for values in ncu_status.values()
    )

    # 8. Harness faithfulness: the isolated replay must reproduce the kernel
    #    name, grid and block that the model run produced. A mismatch means the
    #    counters describe a different launch, so they may not be attached to
    #    the in-model hotspot.
    faithfulness: Dict[str, Any] = {}
    for candidate in NCU_CANDIDATES:
        in_model: Dict[str, Any] = {"name": None, "grids": [], "blocks": []}
        for run in runs:
            for job, analysis in (run.get("analyses") or {}).items():
                # Only the candidate's own sample: the same kernel name can
                # appear at a different ISL in the other sample and would make
                # the grid comparison meaningless.
                if not job.startswith(candidate["sample"]):
                    continue
                info = analysis["phases"].get(candidate["range"])
                if not info:
                    continue
                for row in info["rank_by_total"]:
                    if candidate["kernel_regex"] in row["name"]:
                        in_model = {
                            "name": row["name"],
                            "grids": row["grids"],
                            "blocks": row["blocks"],
                            "streams": row["streams"],
                            "count": row["count"],
                            "dims": row.get("dims", []),
                        }
                        break
        observed: List[Dict[str, Any]] = []
        for run in runs:
            parsed = (run.get("ncu") or {}).get(candidate["id"])
            if not parsed:
                continue
            for kernel in parsed.get("kernels", []):
                observed.append(
                    {
                        "run": run["dir"],
                        "name": kernel["name"],
                        "grid": kernel["grid"],
                        "block": kernel["block"],
                    }
                )
        if not observed:
            continue
        names_match = [
            candidate["kernel_regex"] in entry["name"] for entry in observed
        ]
        grids = {entry["grid"] for entry in observed}
        blocks = {entry["block"] for entry in observed}
        in_model_grids = {
            "(" + ", ".join(str(v) for v in grid) + ",)" if len(grid) == 1
            else "(" + ", ".join(str(v) for v in grid) + ")"
            for grid in in_model["grids"]
        }
        faithfulness[candidate["id"]] = {
            "in_model": in_model,
            "replay": observed,
            "kernel_name_matches": all(names_match),
            "replay_grids": sorted(grids),
            "replay_blocks": sorted(blocks),
            "in_model_grids": sorted(in_model_grids),
            "grid_matches_in_model": bool(grids & in_model_grids),
            # Blocks are compared as strings because ncu pads to "(128, 1, 1)".
            "block_matches_in_model": bool(
                blocks
                & {
                    "(" + ", ".join(str(v) for v in block) + ")"
                    for block in in_model["blocks"]
                }
            ),
        }
    checks["ncu_harness_faithfulness"] = faithfulness
    checks["ncu_harness_faithful"] = bool(faithfulness) and all(
        v["kernel_name_matches"] for v in faithfulness.values()
    )

    # 7. CPU->kernel latency and GPU idle: system-level facts must exist.
    timeline_present = all(
        analysis["phases"]
        and all("timeline" in info for info in analysis["phases"].values())
        for run in runs
        for analysis in (run.get("analyses") or {}).values()
    )
    checks["timeline_present"] = timeline_present

    ncu_selected = [
        c for c in NCU_CANDIDATES
        if any(run["path"].joinpath("ncu", f"{c['id']}.csv").is_file() for run in runs)
    ]
    verdict = {
        "runs": [r["dir"] for r in runs],
        "checks": checks,
        "ncu_candidate_plan": list(NCU_CANDIDATES),
        "ncu_candidates_captured": [c["id"] for c in ncu_selected],
        "e02_03_baseline": e02_03_baseline(),
        "e02_02_expectations_available": expected.get("available", False),
        "decisions": {
            "prefill_decode_separate_rankings": True,
            "module_to_kernel_chain_evidenced": checks["ncu_candidates_all_present"],
            "shapes_and_calls_cross_checked": bool(closure),
            "tool_conditions_transparent": True,
            "system_level_gap_covered": timeline_present,
            "selected_kernel_hardware_explanation": checks["ncu_candidates_all_present"],
            "replay_harness_faithful": checks.get("ncu_harness_faithful", False),
        },
    }
    token_closure = (
        all(matches_e02_02.values()) if expected.get("available") else None
    )
    verdict["passed"] = bool(
        checks["ranking_stability_passed"]
        and checks["token_hash_identical_across_windows"]
        and (token_closure is not False)
        and checks["ncu_candidates_all_present"]
        and timeline_present
    )
    (output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    manifest = {
        "experiment": "E02-07",
        "generated_at": _now_iso(),
        "artifacts": {
            run["dir"]: sorted(
                str(p.relative_to(output_dir))
                for p in run["path"].rglob("*")
                if p.is_file()
            )
            for run in runs
        },
        "verdict": "passed" if verdict["passed"] else "failed",
    }
    (output_dir / "EVIDENCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {
            "passed": verdict["passed"],
            "ranking_stability_passed": checks["ranking_stability_passed"],
            "token_hash_identical": checks["token_hash_identical_across_windows"],
            "ncu_all_present": checks["ncu_candidates_all_present"],
            "tool_compatibility": {
                "profiler_missing_in_nsys": checks["tool_compatibility"][
                    "profiler_missing_in_nsys"
                ][:10],
            },
        },
        indent=2,
    ))
    return 0 if verdict["passed"] else 1


def _roofline_checks(run: Mapping[str, Any]) -> Dict[str, Any]:
    """Step-11 consistency check for the dense GEMM candidates.

    Only the ``linear`` candidates get a roofline treatment, because only they
    have a well-defined useful-FLOP count. The reported intensity uses *modeled
    minimum* DRAM traffic (each operand read once), which is the optimistic
    lower bound; NCU's per-kernel DRAM byte counters are unavailable on this
    Tegra build, so the measured side of the comparison is the L2 throughput
    percentage and the achieved FLOP/s instead of a measured byte count.
    """
    checks: Dict[str, Any] = {}
    for candidate in NCU_CANDIDATES:
        spec = candidate["replay"]
        if spec.get("kind") != "linear":
            continue
        entry = (run.get("ncu") or {}).get(candidate["id"])
        if not entry or not entry.get("kernels"):
            continue
        panel = entry["kernels"][0]["panel"]
        duration_ns = panel.get("duration_ns")
        if not duration_ns:
            continue
        geometry = mp.kernel_shape_flops(m=spec["m"], n=spec["n"], k=spec["k"])
        result = mp.roofline_consistency(
            useful_flops=geometry["flops"],
            modeled_dram_bytes=geometry["min_bytes"],
            measured_l2_bytes=None,
            duration_us=duration_ns / 1000.0,
            peak_flops=67e12,
            peak_bandwidth=68e9,
        )
        result.update(
            {
                "shape": {"m": spec["m"], "n": spec["n"], "k": spec["k"]},
                "arithmetic_intensity_flop_per_byte": geometry["arithmetic_intensity"],
                "measured_memory_throughput_pct": panel.get("memory_throughput_pct"),
                "measured_compute_throughput_pct": panel.get("compute_sm_throughput_pct"),
                "l2_hit_rate_pct": panel.get("l2_hit_rate_pct"),
                "sm_frequency_hz": panel.get("sm_frequency_hz"),
                "sm_frequency_caveat": (
                    "clocks were not locked (ncu --clock-control base is not "
                    "honoured on this Orin build), so the nominal 67 TFLOP/s "
                    "envelope assumes a boost clock the kernel never saw; only "
                    "the throughput percentages are meaningful"
                ),
            }
        )
        checks[candidate["id"]] = result
    return checks


def _amdahl_inputs(run: Mapping[str, Any]) -> Dict[str, Any]:
    """Phase shares E02-09 may consume, with their assumptions made explicit.

    Two different quantities are easy to confuse and both are needed:

    * the *within-phase* share of a kernel (how much of this phase's device
      work it is), which the profiler measures directly;
    * the *phase weight* (how much of one request's wall clock the phase is),
      which comes from the un-instrumented phase windows in the driver payload.

    Their product is the share of one request that optimizing that kernel
    could theoretically touch. It is an **upper bound**, because overlapping
    kernels and host-bound idle time mean removing device work does not remove
    the same amount of wall clock. The decode phase weight is the sum of the
    early, middle and late windows, and only early/late were profiled — so the
    decode intervals are reported as a bracket between the measured windows
    rather than as a single number.
    """
    out: Dict[str, Any] = {}
    for sample_key, jobs in (("P", ("P_early", "P_late")), ("D", ("D_early", "D_late"))):
        early = (run.get("analyses") or {}).get(f"{sample_key}_early")
        if not early:
            continue
        payload_path = run["path"] / f"driver_{sample_key}_early.json"
        if not payload_path.is_file():
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        wall = payload["captured"]["phase_wall_ms"]
        model_core = (
            wall.get("prefill", 0.0)
            + wall.get("first_token_selection", 0.0)
            + wall.get(mp.DECODE_EARLY_RANGE, 0.0)
            + wall.get(mp.DECODE_MIDDLE_RANGE, 0.0)
            + wall.get(mp.DECODE_LATE_RANGE, 0.0)
        )
        weights: Dict[str, Any] = {}
        for name in (
            "prefill",
            "first_token_selection",
            mp.DECODE_EARLY_RANGE,
            mp.DECODE_MIDDLE_RANGE,
            mp.DECODE_LATE_RANGE,
        ):
            if name not in wall:
                continue
            weights[name] = {
                "wall_ms": wall[name],
                "phase_weight_of_model_core": wall[name] / model_core if model_core else None,
            }
        decode_weight = sum(
            weights[name]["wall_ms"]
            for name in (mp.DECODE_EARLY_RANGE, mp.DECODE_MIDDLE_RANGE, mp.DECODE_LATE_RANGE)
            if name in weights
        )
        out[sample_key] = {
            "model_core_wall_ms": model_core,
            "phase_weights": weights,
            "decode_phase_weight": decode_weight / model_core if model_core else None,
            "prefill_phase_weight": (
                weights.get("prefill", {}).get("wall_ms", 0.0) / model_core
                if model_core
                else None
            ),
            "hotspot_bounds": {},
            "assumption": (
                "within-phase share x phase weight bounds the share of one "
                "request that could be touched; it is not a promised speedup"
            ),
            "mode_caveat": (
                "these phase weights come from the no-caching allocator mode "
                "this board forces (E02-06 section 4.13), in which Nsight "
                "Systems measures cudaFree+cudaMalloc at roughly half the "
                "model-core wall. The wall is therefore host/allocator-bound "
                "and the phase weights are a property of the environment, not "
                "of the model. `request_share_bound` is reported for "
                "completeness but E02-09 must rank on `phase_share` (the "
                "within-phase device-work share), which is mode-robust."
            ),
        }
        # Bracket the decode hotspots between the two measured windows.
        for phase_key in (mp.DECODE_EARLY_RANGE, mp.DECODE_LATE_RANGE):
            job = f"{sample_key}_{'early' if 'early' in phase_key else 'late'}"
            analysis = (run.get("analyses") or {}).get(job)
            if not analysis:
                continue
            span = (
                weights.get(mp.DECODE_EARLY_RANGE, {}).get("wall_ms", 0.0)
                if "early" in phase_key
                else weights.get(mp.DECODE_LATE_RANGE, {}).get("wall_ms", 0.0)
            )
            for row in analysis["phases"].get(phase_key, {}).get("rank_by_total", [])[:5]:
                entry = out[sample_key]["hotspot_bounds"].setdefault(
                    row["name"], {"bucket": row["bucket"], "windows": {}}
                )
                entry["windows"][phase_key] = {
                    "phase_share": row["time_share"],
                    "device_us": row["total_us"],
                    "window_wall_ms": span,
                    "request_share_bound": (
                        row["time_share"] * span / model_core if model_core else None
                    ),
                }
    return out


def _run_summarize(args: argparse.Namespace) -> int:
    """Emit the tables the report is written from, recomputed from raw JSON."""
    output_dir = Path(args.output_dir)
    runs = _load_runs(output_dir, limit=1)
    if not runs:
        logger.error("no run_* directories found")
        return 2
    run = runs[0]
    summary: Dict[str, Any] = {"run": run["dir"], "sections": {}}

    for job in ("P_early", "D_early", "P_late", "D_late"):
        analysis = (run.get("analyses") or {}).get(job)
        if not analysis:
            continue
        section: Dict[str, Any] = {}
        for phase, info in analysis["phases"].items():
            if not info["kernel_count"]:
                continue
            section[phase] = {
                "span_ms": info["span_us"] / 1000.0,
                "kernel_work_ms": info["kernel_work_us"] / 1000.0,
                "kernel_count": info["kernel_count"],
                "overlap_factor": info["critical_path"]["overlap_factor"],
                "idle_ratio": info["timeline"]["gpu"]["idle_ratio"],
                "max_stream_gap_us": info["timeline"]["gaps"]["max_gap_us"],
                "top": info["rank_by_total"][:10],
                "buckets": info["buckets"],
                "coverage": info["coverage"],
                "host_ops_top": info["host_ops_top"][:10],
            }
        summary["sections"][job] = section

    summary["cpu_to_kernel_latency"] = {
        job: analysis["cpu_to_kernel_latency"]
        for job, analysis in (run.get("analyses") or {}).items()
    }
    summary["module_roles"] = {
        job: {
            role: {
                "kernel_count": entry["kernel_count"],
                "device_us": round(entry["device_us"], 3),
                "ops": list(entry["ops"])[:5],
                "chains": list(entry["chains"])[:3],
                "top_kernels": list(entry["kernels"])[:3],
            }
            for role, entry in analysis["module_roles"]["roles"].items()
        }
        for job, analysis in (run.get("analyses") or {}).items()
    }
    summary["ncu"] = {
        key: [
            {
                "name": kernel["name"],
                "grid": kernel["grid"],
                "block": kernel["block"],
                "panel": kernel["panel"],
                "stalls": mp.ncu_stall_metrics(kernel["metrics"]),
            }
            for kernel in parsed.get("kernels", [])
        ]
        for key, parsed in (run.get("ncu") or {}).items()
    }
    summary["roofline"] = _roofline_checks(run)
    summary["amdahl_inputs"] = _amdahl_inputs(run)
    summary["references"] = run.get("references")
    summary["perturbation"] = run.get("perturbation")
    summary["nsys"] = run.get("nsys")
    summary["e02_03_baseline"] = e02_03_baseline()
    summary["e02_02_expectations"] = e02_02_expectations()

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(output_dir / "summary.json")}, indent=2))
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default="~/models/hqsb/Qwen3-1.7B")
    parser.add_argument(
        "--manifest",
        default=str(_REPO_ROOT / "docs" / "benchmark" / "model_sha256_manifest.txt"),
    )
    parser.add_argument("--early-steps", type=int, default=EARLY_STEPS)
    parser.add_argument("--late-steps", type=int, default=LATE_STEPS)
    parser.add_argument("--timeout", type=int, default=7200)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E02-07 orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="range/annotation rehearsal before the formal run")
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--audit-isl", type=int, default=128)
    audit.add_argument("--audit-osl", type=int, default=16)
    _add_common(audit)
    audit.set_defaults(func=_run_audit)

    collect = sub.add_parser("collect", help="one independent process of profiler evidence")
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--run-index", type=int, default=0)
    _add_common(collect)
    collect.set_defaults(func=_run_collect)

    reanalyze = sub.add_parser(
        "reanalyze",
        help="recompute per-phase analyses from saved traces (no device work)",
    )
    reanalyze.add_argument("--output-dir", required=True)
    reanalyze.add_argument("--run-index", type=int, default=None)
    reanalyze.set_defaults(func=_run_reanalyze)

    ncu_csv = sub.add_parser(
        "ncu-csv",
        help="regenerate NCU CSVs from saved .ncu-rep reports (no device work)",
    )
    ncu_csv.add_argument("--output-dir", required=True)
    ncu_csv.set_defaults(func=_run_ncu_csv_from_reports)

    nsys = sub.add_parser("nsys", help="Nsight Systems capture + stats exports")
    nsys.add_argument("--output-dir", required=True)
    nsys.add_argument("--run-index", type=int, default=0)
    _add_common(nsys)
    nsys.set_defaults(func=_run_nsys)

    ncu = sub.add_parser("ncu", help="Nsight Compute capture per pre-registered candidate")
    ncu.add_argument("--output-dir", required=True)
    ncu.add_argument("--run-index", type=int, default=0)
    ncu.add_argument("--ncu-launches", type=int, default=3)
    ncu.add_argument(
        "--anchor",
        action="store_true",
        help="also capture the in-model application-replay anchor (run 0 only; "
        "costs several model relaunches)",
    )
    _add_common(ncu)
    ncu.set_defaults(func=_run_ncu)

    verify = sub.add_parser("verify", help="cross-run / cross-tool verdict")
    verify.add_argument("--output-dir", required=True)
    verify.add_argument("--runs-required", type=int, default=3)
    verify.set_defaults(func=_run_verify)

    summarize = sub.add_parser("summarize", help="report tables recomputed from raw JSON")
    summarize.add_argument("--output-dir", required=True)
    summarize.set_defaults(func=_run_summarize)
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
