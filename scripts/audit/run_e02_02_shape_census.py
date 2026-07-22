#!/usr/bin/env python3
"""E02-02 runner (reworked): Qwen3 runtime shape census with auditable evidence.

This is the REWORKED E02-02 runner. It supersedes the first version whose
``time_share`` summed the profiler's host-op and device-kernel layers together
(double-counting every kernel 2x). See
``docs/stage_experiments/details/S02/E02-02_runtime_shape_census.md`` §8 step 8
and the E02-02 report's correction record.

Subcommands
-----------
``audit``
    Small profiler audit on one short workload: proves the collector is correct
    *before* the expensive formal run. Checks that (a) instrumenting the run
    does not change the generated tokens vs the un-instrumented reference,
    (b) the two scopes are split and each closes to 1.0, (c) the host-op and
    device-kernel scopes describe the same device work (structural 2x), and
    (d) the raw Chrome trace really contains ``cat="kernel"`` events whose
    names match the kernel-scope table.
``collect``
    One independent process. Runs all six workloads: un-instrumented reference
    timing, then the census (module hooks + per-phase op/kernel tables + raw
    traces). Writes ``run_<index>/``.
``verify``
    Cross-process verification over ``run_0..run_2``: coverage, key-shape /
    call-count consistency, token parity, scope separation, and (if present)
    the NSYS audit. Writes ``verdict.json`` and ``EVIDENCE_MANIFEST.json``.

Usage (on the Jetson, via the local proxy):

    python3 scripts/audit/run_e02_02_shape_census.py audit \
        --output-dir docs/stage_experiments/S02/E02-02/raw_v2/audit
    for i in 0 1 2; do
      python3 scripts/audit/run_e02_02_shape_census.py collect \
          --output-dir docs/stage_experiments/S02/E02-02/raw_v2 --run-index $i
    done
    python3 scripts/audit/run_e02_02_shape_census.py verify \
        --output-dir docs/stage_experiments/S02/E02-02/raw_v2
"""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import logging
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

from hqsb.benchmark.correctness import hash_token_sequence
from hqsb.benchmark.model_core import benchmark_model_core
from hqsb.benchmark.profiling import cumulative_kernel_time_us, scope_totals_us
from hqsb.benchmark.shape_census import collect_shape_census
from hqsb.benchmark.workload import make_fixed_token_input
from hqsb.benchmark.workload_config import load_workload_dicts
from hqsb.models.loader import load_qwen3

logger = logging.getLogger("e02_02")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKLOAD_YAML = _REPO_ROOT / "configs" / "benchmarks" / "jetson_qwen3_fp16.yaml"

_REQUIRED_LAYER_SUBMODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "input_layernorm",
    "post_attention_layernorm",
]

# Qwen3-1.7B KV cache tensor shape (batch=1, num_kv_heads=8, seq, head_dim=128).
_KV_SHAPE_RE = re.compile(r"^\[1, 8, \d+, 128\]$")


# ── identity / environment ───────────────────────────────────────────────

def _git_commit() -> Optional[str]:
    return _shell(["git", "rev-parse", "HEAD"])


def _git_dirty() -> Optional[bool]:
    out = _shell(["git", "status", "--porcelain"])
    return None if out is None else bool(out.strip())


def _shell(cmd: List[str], timeout: int = 30) -> Optional[str]:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _file_hash(path: str) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _thermal_c() -> Optional[List[float]]:
    temps: List[float] = []
    for zone in sorted(Path("/sys/devices/virtual/thermal").glob("thermal_zone*")):
        try:
            temps.append(int((zone / "temp").read_text().strip()) / 1000.0)
        except Exception:
            # Some sysfs thermal zones do not support a normal read() and can
            # raise beyond OSError; telemetry is best-effort, so skip them.
            continue
    return temps or None


def _gpu_cur_freq_hz() -> Optional[int]:
    for path in Path("/sys").glob("devices/gpu.0/devfreq/*/cur_freq"):
        try:
            return int(path.read_text().strip())
        except Exception:
            continue
    return None


def _environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
    }
    try:
        import transformers

        env["transformers_version"] = transformers.__version__
    except Exception:  # pragma: no cover - transformers always present in runs
        env["transformers_version"] = None
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        env["device"] = torch.cuda.get_device_name(0)
        env["compute_capability"] = [int(cc[0]), int(cc[1])]
    else:
        env["device"] = "cpu"
        env["compute_capability"] = None
    env["nvpmodel"] = _shell(["nvpmodel", "-q"])
    env["jetson_clocks"] = _shell(["jetson_clocks", "--show"])
    env["gpu_cur_freq_hz"] = _gpu_cur_freq_hz()
    env["thermal_c"] = _thermal_c()
    return env


def _identity(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "environment": _environment(),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "model_manifest_sha256": _file_hash(args.manifest),
        "workload_yaml": str(args.workload_yaml),
        "workload_yaml_sha256": _file_hash(str(args.workload_yaml)),
        "declared": {
            "dtype": "float16",
            "attention_backend": "eager",
            "cache_impl": "transformers DynamicCache (use_cache=True)",
            "batch_size": 1,
            "sampling": "greedy",
        },
    }


# ── coverage (module-level, unchanged semantics) ─────────────────────────

def _layer_indices(module_paths: Set[str]) -> List[int]:
    indices: Set[int] = set()
    for path in module_paths:
        m = re.match(r"^model\.layers\.(\d+)(?:\.|$)", path)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def _coverage_checks(record: Dict[str, Any]) -> Dict[str, Any]:
    paths = {r["module"] for r in record["modules"]}
    layer_indices = _layer_indices(paths)

    missing_per_layer: Dict[int, List[str]] = {}
    for idx in layer_indices:
        missing = [
            sub
            for sub in _REQUIRED_LAYER_SUBMODULES
            if f"model.layers.{idx}.{sub}" not in paths
        ]
        if missing:
            missing_per_layer[idx] = missing

    structural = {
        "lm_head": "lm_head" in paths,
        "final_norm": "model.norm" in paths,
        "embed_tokens": "model.embed_tokens" in paths,
    }

    kv_shapes: List[str] = []
    kv_phases: Set[str] = set()
    for row in record["modules"]:
        for shape in row.get("output_shapes", []):
            if _KV_SHAPE_RE.match(shape):
                kv_shapes.append(shape)
                kv_phases.add(row["phase"])

    return {
        "layer_count": len(layer_indices),
        "complete_decoder_block": not missing_per_layer and bool(layer_indices),
        "missing_layer_submodules": missing_per_layer,
        "structural": structural,
        "kv_cache_covered": bool(kv_shapes) and kv_phases == {"prefill", "decode"},
        "kv_phases": sorted(kv_phases),
    }


def _scope_check(record: Dict[str, Any]) -> Dict[str, Any]:
    """Per-phase scope closure: each scope's shares sum to 1.0."""
    out: Dict[str, Any] = {}
    for phase, key in (("prefill", "prefill"), ("decode", "decode")):
        block = record[key]
        if phase == "prefill":
            ops, kernels = block["aten_ops"], block["kernels"]
        else:
            ops = block["aten_ops_cumulative_over_probe_steps"]
            kernels = block["kernels_cumulative_over_probe_steps"]
        out[phase] = {
            "cpu_rows": len(ops),
            "kernel_rows": len(kernels),
            "cpu_share_sum": sum(r["time_share"] for r in ops),
            "kernel_share_sum": sum(r["time_share"] for r in kernels),
            "cpu_cumulative_us": scope_totals_us(ops).get("cpu", 0.0),
            "kernel_cumulative_us": scope_totals_us(kernels).get("kernel", 0.0),
        }
    return out


# ── trace audit ──────────────────────────────────────────────────────────

# Chrome-trace categories that represent work executed on the device. Kineto
# reports compute kernels as ``kernel`` and device memcpy/memset as
# ``gpu_memcpy``/``gpu_memset``; all three are "device scope" work.
_DEVICE_TRACE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")


def _audit_trace(path: str) -> Dict[str, Any]:
    """Summarise a Chrome trace by event category and device-event names."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    events = data.get("traceEvents", []) if isinstance(data, dict) else []
    by_cat: Counter = Counter()
    device_names: Set[str] = set()
    for event in events:
        cat = str(event.get("cat", ""))
        by_cat[cat] += 1
        if cat in _DEVICE_TRACE_CATEGORIES:
            device_names.add(str(event.get("name", "")))
    return {
        "ok": True,
        "event_count": len(events),
        "by_category": dict(by_cat),
        "device_event_count": sum(
            by_cat.get(cat, 0) for cat in _DEVICE_TRACE_CATEGORIES
        ),
        "device_event_names": sorted(device_names),
    }


# ── audit command ────────────────────────────────────────────────────────

def _run_audit(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    trace_dir = output_dir / "traces"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, load_time_s = load_qwen3(
        args.model_path,
        dtype=torch.float16,
        attention_backend="eager",
        verify_manifest=args.manifest,
        allow_extra=("model_sha256_manifest.txt",),
        # Host staging avoids the pinned-memory 2x load peak that OOMs this
        # 8 GiB unified-memory device (see loader.load_qwen3 docstring).
        cpu_staging=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = make_fixed_token_input(tokenizer, args.input_tokens, device=device)
    benchmark_model_core(model, inputs, 2)  # warmup

    reference = benchmark_model_core(model, inputs, args.output_tokens)
    ref_tokens = reference["generated_token_ids"]

    t0 = time.perf_counter()
    census = collect_shape_census(
        model, inputs, args.output_tokens, trace_dir=str(trace_dir)
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0

    token_parity = hash_token_sequence(census["generated_token_ids"]) == hash_token_sequence(
        ref_tokens
    )

    # Trace audit: the kernel-scope table names must appear as device events
    # (``kernel`` / ``gpu_memcpy`` / ``gpu_memset`` categories) in the trace.
    trace_audit = _audit_trace(census["prefill"]["trace"]) if census["prefill"]["trace"] else {}
    kernel_names = set(trace_audit.get("device_event_names", []))
    kernel_rows = census["prefill"]["kernels"]
    matched = [r["name"] for r in kernel_rows if r["name"] in kernel_names]
    unmatched = [r["name"] for r in kernel_rows if r["name"] not in kernel_names]

    scopes = _scope_check(
        {
            "prefill": census["prefill"],
            "decode": census["decode"],
        }
    )

    audit = {
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "decode_probe_steps": census["decode_probe_steps"],
        "load_time_s": load_time_s,
        "reference_e2e_ms": reference["model_core_e2e_ms"],
        "instrumented_wall_ms": wall_ms,
        "token_parity": token_parity,
        "reference_sequence_sha256": hash_token_sequence(ref_tokens),
        "instrumented_sequence_sha256": census["sequence_sha256"],
        "trace_audit": trace_audit,
        "kernel_rows_matched_in_trace": len(matched),
        "kernel_rows_unmatched_in_trace": unmatched,
        "scope_check": scopes,
        "prefill_trace": census["prefill"]["trace"],
        "traces": [
            census["prefill"]["trace"],
            *[v["trace"] for v in census["decode"]["per_step"].values()],
        ],
    }

    scopes_closed = all(
        abs(v["cpu_share_sum"] - 1.0) < 1e-6 and abs(v["kernel_share_sum"] - 1.0) < 1e-6
        and v["cpu_rows"] > 0 and v["kernel_rows"] > 0
        for v in scopes.values()
    )
    audit["scopes_closed"] = scopes_closed
    audit["passed"] = bool(
        token_parity
        and scopes_closed
        and trace_audit.get("ok")
        and not unmatched
    )

    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    logger.info("audit passed=%s (token_parity=%s, unmatched_kernels=%d)",
                audit["passed"], token_parity, len(unmatched))
    print(json.dumps({k: audit[k] for k in (
        "passed", "token_parity", "scopes_closed",
        "kernel_rows_matched_in_trace", "kernel_rows_unmatched_in_trace",
    )}, indent=2))
    return 0 if audit["passed"] else 1


# ── collect command ──────────────────────────────────────────────────────

def _run_collect(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    run_dir = output_dir / f"run_{args.run_index}"
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)

    identity = _identity(args)
    tokenizer, model, load_time_s = load_qwen3(
        args.model_path,
        dtype=torch.float16,
        attention_backend="eager",
        verify_manifest=args.manifest,
        allow_extra=("model_sha256_manifest.txt",),
        # Host staging avoids the pinned-memory 2x load peak that OOMs this
        # 8 GiB unified-memory device (see loader.load_qwen3 docstring).
        cpu_staging=True,
    )
    workloads = load_workload_dicts(str(args.workload_yaml))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    warm_inputs = make_fixed_token_input(tokenizer, 32, device=device)
    _ = model(
        input_ids=warm_inputs["input_ids"],
        attention_mask=warm_inputs["attention_mask"],
        use_cache=True,
    )
    if device == "cuda":
        torch.cuda.synchronize()

    records: List[Dict[str, Any]] = []
    only = set(args.only.split(",")) if getattr(args, "only", None) else None
    for spec in workloads:
        name = spec["name"]
        if only and name not in only:
            continue
        isl = int(spec["input_tokens"])
        osl = int(spec["output_tokens"])
        logger.info("Census %s (ISL=%d OSL=%d) run_%d", name, isl, osl, args.run_index)

        inputs = make_fixed_token_input(tokenizer, isl, device=device)
        reference = benchmark_model_core(model, inputs, osl)
        ref_tokens = reference["generated_token_ids"]

        t0 = time.perf_counter()
        census = collect_shape_census(
            model,
            inputs,
            osl,
            # Namespace traces per workload: a single shared directory would
            # let same-named files (prefill_trace.json, decode_step1_trace.json)
            # from different workloads overwrite each other.
            trace_dir=str(run_dir / "traces" / name),
            prefill_trace_max_isl=args.trace_max_isl,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0

        token_parity = hash_token_sequence(census["generated_token_ids"]) == (
            hash_token_sequence(ref_tokens)
        )

        record = {
            "workload": {"name": name, "input_tokens": isl, "output_tokens": osl},
            "provenance": {
                "modules": "runtime_forward_hook",
                "aten_ops": "torch_profiler_events (scope=cpu)",
                "kernels": "torch_profiler_events (scope=kernel)",
                "timing_metric": "cumulative_gpu_kernel_work_us (not wall clock)",
                "shapes": "read_from_live_tensors",
            },
            "input_len": census["input_len"],
            "output_tokens": census["output_tokens"],
            "decode_steps": census["decode_steps"],
            "decode_probe_steps": census["decode_probe_steps"],
            "reference": {
                "generated_token_ids": ref_tokens,
                "sequence_sha256": hash_token_sequence(ref_tokens),
                "model_core_e2e_ms": reference["model_core_e2e_ms"],
            },
            "instrumented": {
                "generated_token_ids": census["generated_token_ids"],
                "sequence_sha256": census["sequence_sha256"],
                "wall_ms": wall_ms,
            },
            "token_parity": token_parity,
            "modules": census["modules"],
            "prefill": {
                "probe_scope": census["prefill"]["probe_scope"],
                "aten_ops": census["prefill"]["aten_ops"],
                "kernels": census["prefill"]["kernels"],
                "trace": _rel(census["prefill"]["trace"], run_dir),
                "kernel_cumulative_us": cumulative_kernel_time_us(
                    census["prefill"]["kernels"]
                ),
            },
            "decode": {
                "probe_scope": census["decode"]["probe_scope"],
                "probe_steps": census["decode"]["probe_steps"],
                "per_step": {
                    step: {
                        "step": entry["step"],
                        "context_len": entry["context_len"],
                        "aten_ops": entry["aten_ops"],
                        "kernels": entry["kernels"],
                        "trace": _rel(entry["trace"], run_dir),
                    }
                    for step, entry in census["decode"]["per_step"].items()
                },
                "aten_ops_cumulative_over_probe_steps": census["decode"][
                    "aten_ops_cumulative_over_probe_steps"
                ],
                "kernels_cumulative_over_probe_steps": census["decode"][
                    "kernels_cumulative_over_probe_steps"
                ],
            },
        }
        record["coverage"] = _coverage_checks(record)
        record["scope_check"] = _scope_check(
            {"prefill": census["prefill"], "decode": census["decode"]}
        )
        records.append(record)

        (run_dir / f"census_{name}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        logger.info(
            "  wrote census_%s.json (token_parity=%s, %d module records)",
            name, token_parity, len(census["modules"]),
        )

        del record, census, reference
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    run_meta = {
        "run_id": f"run_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}",
        "run_index": args.run_index,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "identity": identity,
        "load_time_s": load_time_s,
        "workloads": [r["workload"]["name"] for r in records],
        "token_parity_all": all(r["token_parity"] for r in records),
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )
    logger.info("run_%d complete (token_parity_all=%s)",
                args.run_index, run_meta["token_parity_all"])
    return 0


def _rel(path: Optional[str], base: Path) -> Optional[str]:
    if not path:
        return None
    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        return str(path)


# ── verify command ───────────────────────────────────────────────────────

def _module_signature(record: Dict[str, Any]) -> str:
    """Stable hash over module shapes + call counts (cross-run determinism)."""
    canonical = []
    for row in sorted(record["modules"], key=lambda r: (r["module"], r["phase"])):
        canonical.append(
            [row["module"], row["phase"], row["call_count"],
             row.get("input_shapes"), row.get("output_shapes"),
             row.get("input_dtypes"), row.get("output_dtypes")]
        )
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _run_verify(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    run_dirs = sorted(
        d for d in output_dir.glob("run_*") if d.is_dir()
    )
    if len(run_dirs) < 3:
        logger.error("need >= 3 run_* dirs, found %d", len(run_dirs))
        return 2

    runs = []
    for run_dir in run_dirs[:3]:
        workloads = {
            p.stem.replace("census_", ""): json.loads(p.read_text(encoding="utf-8"))
            for p in run_dir.glob("census_*.json")
        }
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        runs.append({"dir": run_dir.name, "meta": meta, "workloads": workloads})

    names = sorted(runs[0]["workloads"])
    per_workload: Dict[str, Any] = {}
    for name in names:
        records = [r["workloads"].get(name) for r in runs]
        if any(rec is None for rec in records):
            per_workload[name] = {"complete": False}
            continue
        sigs = {_module_signature(rec) for rec in records}
        hashes = {rec["instrumented"]["sequence_sha256"] for rec in records}
        call_counts = {
            json.dumps(
                sorted(
                    (m["module"], m["phase"], m["call_count"])
                    for m in rec["modules"]
                ),
                separators=(",", ":"),
            )
            for rec in records
        }
        coverage = [
            _coverage_checks(rec) for rec in records
        ]
        token_parity = [rec["token_parity"] for rec in records]
        scopes = [rec["scope_check"] for rec in records]
        scopes_closed = all(
            abs(s[phase][key] - 1.0) < 1e-6
            for s in scopes
            for phase in ("prefill", "decode")
            for key in ("cpu_share_sum", "kernel_share_sum")
        )
        per_workload[name] = {
            "complete": True,
            "runs": len(records),
            "module_signature_identical": len(sigs) == 1,
            "token_hash_identical": len(hashes) == 1,
            "call_counts_identical": len(call_counts) == 1,
            "coverage_ok": all(
                c["complete_decoder_block"]
                and c["kv_cache_covered"]
                and all(c["structural"].values())
                for c in coverage
            ),
            "layer_count": coverage[0]["layer_count"],
            "token_parity_all": all(token_parity),
            "scopes_closed": scopes_closed,
        }

    all_ok = all(v.get("complete") and v["module_signature_identical"]
                 and v["token_hash_identical"] and v["call_counts_identical"]
                 and v["coverage_ok"] and v["token_parity_all"]
                 and v["scopes_closed"] for v in per_workload.values())

    nsys_dir = output_dir / "nsys"
    nsys_present = nsys_dir.is_dir() and any(nsys_dir.iterdir()) if nsys_dir.is_dir() else False

    verdict = {
        "runs": [r["dir"] for r in runs],
        "identity": runs[0]["meta"].get("identity"),
        "workloads": per_workload,
        "run_level_token_parity": all(
            r["meta"].get("token_parity_all") for r in runs
        ),
        "nsys_audit_present": nsys_present,
        "passed": all_ok,
    }
    (output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )

    manifest = {
        "experiment": "E02-02",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "identity": runs[0]["meta"].get("identity"),
        "artifacts": {
            run_dir.name: sorted(
                str(p.relative_to(output_dir))
                for p in run_dir.rglob("*")
                if p.is_file()
            )
            for run_dir in run_dirs[:3]
        },
        "verdict": "passed" if all_ok else "failed",
    }
    (output_dir / "EVIDENCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.info("verify passed=%s", all_ok)
    print(json.dumps(verdict, indent=2))
    return 0 if all_ok else 1


# ── CLI ──────────────────────────────────────────────────────────────────

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default="~/models/hqsb/Qwen3-1.7B")
    parser.add_argument(
        "--manifest",
        default=str(_REPO_ROOT / "docs" / "benchmark" / "model_sha256_manifest.txt"),
    )
    parser.add_argument("--workload-yaml", default=str(_WORKLOAD_YAML))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E02-02 shape census (reworked)")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="small profiler audit before the full run")
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--input-tokens", type=int, default=128)
    audit.add_argument("--output-tokens", type=int, default=16)
    _add_common(audit)
    audit.set_defaults(func=_run_audit)

    collect = sub.add_parser("collect", help="one independent process, six workloads")
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--run-index", type=int, default=0)
    collect.add_argument(
        "--only", default=None, help="comma-separated workload names (debug)"
    )
    collect.add_argument(
        "--trace-max-isl",
        type=int,
        default=1024,
        help="export the prefill Chrome trace only up to this ISL (decode "
        "traces are always exported); large prefill traces can OOM the device",
    )
    _add_common(collect)
    collect.set_defaults(func=_run_collect)

    verify = sub.add_parser("verify", help="cross-process verification")
    verify.add_argument("--output-dir", required=True)
    verify.set_defaults(func=_run_verify)
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
