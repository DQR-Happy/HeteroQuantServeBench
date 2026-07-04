#!/usr/bin/env python3
"""E04-10 cross-architecture prerequisite freezer and availability audit.

This collector deliberately refuses to manufacture a second architecture.  It
freezes architecture-A evidence before any architecture-B results are seen,
captures the current CUDA target, and emits an explicit BLOCKED matrix when no
real, different NVIDIA architecture evidence bundle is supplied.

Run only on a CUDA target through ``scripts/remote_run.sh``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "E04-10"
SCHEMA = "hqsb.s04.cross_architecture_transfer/v1"
SECOND_ARCH_STATE = "NOT_RUN_SECOND_ARCH_UNAVAILABLE"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str, timeout: int = 60) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            list(args), cwd=REPO, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout, check=False,
        )
        return {
            "command": list(args), "exit_code": proc.returncode,
            "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(args), "exit_code": 124,
            "stdout": "", "stderr": f"{type(exc).__name__}: {exc}",
        }


def optional_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(errors="replace").strip("\x00\n ")


def git_output(*args: str) -> str:
    return command("git", *args, timeout=120)["stdout"]


def device_environment() -> Dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("E04-10 collection requires a real CUDA target")
    props = torch.cuda.get_device_properties(0)
    capability = list(torch.cuda.get_device_capability(0))
    uuid = getattr(props, "uuid", None)
    driver_text = optional_text(Path("/proc/driver/nvidia/version"))
    serial_path = Path("/proc/device-tree/serial-number")
    serial_hash = (
        hashlib.sha256(serial_path.read_bytes().rstrip(b"\x00")).hexdigest()
        if serial_path.exists() else None
    )
    return {
        "role": "architecture_A",
        "hostname": platform.node(),
        "device_index": 0,
        "device_name": props.name,
        "device_uuid": str(uuid) if uuid is not None else None,
        "serial_sha256": serial_hash,
        "compute_capability": capability,
        "arch": f"sm{capability[0]}{capability[1]}",
        "multiprocessor_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "warp_size": getattr(props, "warp_size", None),
        "max_threads_per_block": getattr(props, "max_threads_per_block", None),
        "max_threads_per_multiprocessor": getattr(
            props, "max_threads_per_multi_processor", None
        ),
        "shared_memory_per_block_bytes": getattr(
            props, "shared_memory_per_block", None
        ),
        "shared_memory_per_multiprocessor_bytes": getattr(
            props, "shared_memory_per_multiprocessor", None
        ),
        "registers_per_block": getattr(props, "regs_per_block", None),
        "nominal_roof": {
            "fp16_dense_flops_per_second": 67e12,
            "dram_bandwidth_bytes_per_second": 68e9,
            "source": "hqsb/benchmark/roofline.py ORIN_NANO_SUPER_FP16",
            "kind": "nominal_model_not_measured",
        },
        "software": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "driver": driver_text,
            "triton": _module_version("triton"),
        },
        "power_clock": {
            "nvpmodel_query": command("nvpmodel", "-q"),
            "jetson_clocks_show": command("sudo", "-n", "jetson_clocks", "--show"),
            "tegrastats_bounded_sample": command(
                "timeout", "3s", "tegrastats", "--interval", "1000", timeout=10
            ),
        },
    }


def _module_version(name: str) -> Optional[str]:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:  # capability evidence, never hide failure
        return f"UNAVAILABLE:{type(exc).__name__}:{exc}"


def source_record(relative: str) -> Dict[str, Any]:
    path = REPO / relative
    return {
        "path": relative,
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def load(relative: str) -> Any:
    return read_json(REPO / relative)


def freeze_upstream() -> Dict[str, Any]:
    verdicts = []
    for number in range(1, 10):
        exp = f"E04-{number:02d}"
        relative = f"docs/stage_experiments/S04/{exp}/raw/verdict.json"
        path = REPO / relative
        if path.exists():
            data = read_json(path)
            verdicts.append({
                "experiment_id": exp,
                "overall": data.get("overall"),
                "path": relative,
                "sha256": sha256_file(path),
            })
        else:
            verdicts.append({
                "experiment_id": exp, "overall": "MISSING",
                "path": relative, "sha256": None,
            })

    rms_policy = load("docs/stage_experiments/S04/E04-06/raw/frozen_policy.json")
    cutlass_topk = load("docs/stage_experiments/S04/E04-05/raw/frozen_topk.json")
    build_cost = load("docs/stage_experiments/S04/E04-05/raw/build_cost.json")
    holdout = load("docs/stage_experiments/S04/E04-06/raw/holdout_generalization.json")
    crossovers = load("docs/stage_experiments/S04/E04-04/raw/crossover_tie_intervals.json")
    route_table = load("docs/stage_experiments/S04/E04-08/raw/expected_decision_table.json")

    cutlass_rows = []
    for row in cutlass_topk.get("rows", []):
        cutlass_rows.append({
            "family": row.get("family"), "regime": row.get("regime"),
            "top_k": row.get("top_k"), "train_points": row.get("train_points"),
        })
    crossover_summary = {}
    for operator, rows in crossovers.items():
        crossover_summary[operator] = [
            {
                "family": row.get("family"),
                "fit_points": row.get("fit_points"),
                "clear_rank_change_intervals": row.get("clear_rank_change_intervals"),
                "tie_zone_points": row.get("tie_zone_points"),
            }
            for row in rows
        ]

    return {
        "experiment_id": EXPERIMENT_ID,
        "frozen_at_utc": utc_now(),
        "state": "FROZEN_BEFORE_ANY_ARCHITECTURE_B_RESULT",
        "upstream_verdicts": verdicts,
        "architecture_A": {
            "rmsnorm_triton_policy": rms_policy.get("policies", []),
            "rmsnorm_holdout": holdout.get("rows", []),
            "cutlass_top_k": cutlass_rows,
            "cutlass_build_cost": {
                "configure_duration_s": build_cost.get("configure", {}).get("duration_s"),
                "clean_build_duration_s": build_cost.get("clean_directory_build", {}).get("duration_s"),
                "incremental_build_duration_s": build_cost.get("incremental_build", {}).get("duration_s"),
                "binary_bytes": build_cost.get("binary_bytes"),
                "instantiated_configs": build_cost.get("instantiated_configs"),
            },
            "crossover_summary": crossover_summary,
            "frozen_route_cases": route_table.get("cases", []),
        },
        "limitations_already_known": [
            "E04-08 route table is an expected/audit table, not a qualified production cache",
            "E04-09 public bridge/current-stream gate failed",
            "Architecture-A winners are evidence inputs only and must not be copied as B cache hits",
        ],
    }


def target_access_audit(env_a: Mapping[str, Any], out: Path) -> Dict[str, Any]:
    remote_script = optional_text(REPO / "scripts/remote_run.sh") or ""
    configured_hosts = sorted(set(re.findall(r"[A-Za-z0-9_.-]+@(?:\d{1,3}\.){3}\d{1,3}", remote_script)))
    external_b_bundle = os.environ.get("HQSB_E04_10_ARCH_B_BUNDLE")
    b_bundle_path = Path(external_b_bundle).resolve() if external_b_bundle else None
    bundle_present = bool(b_bundle_path and b_bundle_path.is_file())
    return {
        "audited_at_utc": utc_now(),
        "configured_execution_hosts": configured_hosts,
        "current_target": {
            "hostname": env_a["hostname"], "device_name": env_a["device_name"],
            "arch": env_a["arch"], "compute_capability": env_a["compute_capability"],
        },
        "architecture_B_bundle_env": external_b_bundle,
        "architecture_B_bundle_present": bundle_present,
        "architecture_B_runtime_contacted": False,
        "distinct_real_nvidia_architecture_available": False,
        "reason": (
            "No architecture-B evidence bundle was supplied and the repository has only one configured remote target"
            if not bundle_present else
            "Bundle ingestion is intentionally not automatic; B must be collected on the real target with the frozen protocol"
        ),
        "stage_rule": "same GPU, changed power mode, or fake compute capability cannot satisfy E04-10",
    }


def required_matrix() -> Dict[str, Any]:
    rows = []
    required = [
        ("environment", "model/UUID/CC/SM/resources/memory/driver/runtime/toolkit/power/clock"),
        ("functional_smoke", "reference plus minimal RMSNorm/GEMM"),
        ("cache_invalidation", "A Triton/route/capability cache rejected on B"),
        ("correctness", "real, tail, dtype, layout, stream, bridge, sanitizer"),
        ("zero_shot_A_to_B", "A high-level configs recompiled for B"),
        ("B_local_search", "Triton and CUTLASS search on train/validation/holdout"),
        ("ordinary_timing", "at least three independent B processes"),
        ("rank_regret", "Spearman/Kendall/top-k/winner/regret/unsupported"),
        ("crossover_shift", "decode/prefill crossover comparison"),
        ("profile_pairs", "one portable and one needs-retune pair"),
        ("routing", "common rule plus arch-local leaf and B holdout"),
        ("tune_build_cost", "compile success/time/trials/cache/binary/break-even"),
    ]
    for item, contract in required:
        rows.append({
            "item": item, "contract": contract,
            "architecture_A_evidence": "FROZEN_OR_INHERITED",
            "architecture_B_evidence": SECOND_ARCH_STATE,
            "status": "BLOCKED",
        })
    return {
        "experiment_id": EXPERIMENT_ID,
        "rows": rows,
        "completed_items": 0,
        "required_items": len(rows),
        "coverage_fraction": 0.0,
        "denominator_policy": "unavailable/unsupported B cases remain in the denominator",
    }


def transfer_metrics() -> Dict[str, Any]:
    unavailable = {
        "value": None, "status": SECOND_ARCH_STATE,
        "reason": "requires measured architecture-B data",
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "architecture_pair": {"A": "sm87", "B": None},
        "functional_portability_coverage": dict(unavailable),
        "zero_shot_transfer_regret": dict(unavailable),
        "spearman_rank": dict(unavailable),
        "kendall_tau": dict(unavailable),
        "top_k_overlap": dict(unavailable),
        "winner_retention": dict(unavailable),
        "crossover_shift": dict(unavailable),
        "performance_portability_harmonic_metric": {
            **unavailable,
            "policy": "not computed; unsupported/unavailable platform cannot be silently dropped",
        },
        "shape_weighted_B_regret": dict(unavailable),
        "unsupported_transfers": {
            "count": None, "status": SECOND_ARCH_STATE,
            "reason": "B compilation was not attempted without B hardware",
        },
    }


def cache_guard() -> Dict[str, Any]:
    invalidation = load("docs/stage_experiments/S04/E04-06/raw/invalidation_matrix.json")
    raw_text = json.dumps(invalidation, sort_keys=True)
    return {
        "experiment_id": EXPERIMENT_ID,
        "architecture_A_static_evidence": {
            "source": "E04-06 invalidation_matrix.json",
            "source_sha256": sha256_file(REPO / "docs/stage_experiments/S04/E04-06/raw/invalidation_matrix.json"),
            "device_or_arch_identity_present": any(
                token in raw_text for token in ("compute_capability", "device", "multiprocessor_count")
            ),
        },
        "A_cache_presented_to_B": False,
        "B_rejection_observed": False,
        "B_recompile_observed": False,
        "status": SECOND_ARCH_STATE,
        "safety_rule": "A binary/cache is never relabelled as a B hit; B must recompile and retune in its own namespace",
    }


def verdict(targets: Mapping[str, Any]) -> Dict[str, Any]:
    conditions = {
        "1_second_real_nvidia_architecture": False,
        "2_A_results_frozen_before_B": True,
        "3_common_operator_spec_input_tolerance_protocol": False,
        "4_B_correctness_stream_safety": False,
        "5_A_cache_binary_rejected_on_B": False,
        "6_zero_shot_and_B_local_tune_complete": False,
        "7_rank_regret_winner_crossover_unsupported_complete": False,
        "8_portable_and_retune_profile_explanations": False,
        "9_common_and_arch_local_routing_rules": False,
        "10_three_independent_runs_each_architecture": False,
        "11_tune_and_compile_cost_comparison": False,
        "12_no_model_or_hardware_global_extrapolation": True,
    }
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": utc_now(),
        "overall": "BLOCKED",
        "blocker": targets["reason"],
        "conditions": conditions,
        "passed_conditions": [key for key, value in conditions.items() if value],
        "blocked_conditions": [key for key, value in conditions.items() if not value],
        "expected_effect": "prove tuning parameters are not global constants using measured cross-architecture evidence",
        "expected_effect_met": False,
        "single_item_standard": "same protocol reproduced; portable and retune rules; A cache not copied",
        "single_item_standard_met": False,
        "stage_gate": "S04 remains BLOCKED and must not enter formal S04.5",
        "classification": "EVIDENCE_INSUFFICIENT",
    }


def evidence_manifest(out: Path) -> Dict[str, Any]:
    files = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name == "EVIDENCE_MANIFEST.json":
            continue
        files.append({
            "path": str(path.relative_to(out)), "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "schema": f"{SCHEMA}/manifest",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": utc_now(),
        "files": files,
        "file_count": len(files),
    }


def collect(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    environment = device_environment()
    protocol = {
        "schema": f"{SCHEMA}/protocol",
        "experiment_id": EXPERIMENT_ID,
        "frozen_at_utc": utc_now(),
        "state": "FROZEN_BEFORE_ANY_ARCHITECTURE_B_RESULT",
        "architecture_requirement": "real NVIDIA GPU with compute capability different from architecture A",
        "common_contract": [
            "same OperatorSpec/reference/tolerance", "same shape ledger/input seed/raw hash",
            "same train/validation/holdout principle", "same timing and failure taxonomy",
            "three independent processes per architecture",
        ],
        "transfer_rule": "reuse high-level A config only after recompiling for B; never copy cubin or mark A cache as B hit",
        "missing_B_rule": "overall BLOCKED; fake architecture and same-card power modes forbidden",
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "architecture_A_environment.json", environment)
    frozen = freeze_upstream()
    write_json(out / "architecture_A_frozen_evidence.json", frozen)
    targets = target_access_audit(environment, out)
    write_json(out / "target_access_audit.json", targets)
    write_json(out / "common_matrix_status.json", required_matrix())
    write_json(out / "transfer_metrics.json", transfer_metrics())
    write_json(out / "cache_isolation_status.json", cache_guard())
    write_json(out / "verdict.json", verdict(targets))
    provenance = {
        "schema": f"{SCHEMA}/provenance",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": utc_now(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output("status", "--short")),
        "collector": source_record("scripts/audit/run_e04_10_cross_architecture_transfer.py"),
        "wrapper": source_record("scripts/audit/e04_10_run.sh"),
        "detail_spec": source_record("docs/stage_experiments/details/S04/E04-10_cross_architecture_transfer.md"),
    }
    write_json(out / "provenance.json", provenance)
    write_json(out / "EVIDENCE_MANIFEST.json", evidence_manifest(out))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["collect"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "collect":
        collect(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
