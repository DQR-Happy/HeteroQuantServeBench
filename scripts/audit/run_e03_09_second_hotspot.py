#!/usr/bin/env python3
"""E03-09 second-hotspot dependency gate and resume-evidence freezer.

The S03 protocol orders E03-08 before E03-09.  This runner therefore refuses
to launch, benchmark, or profile a second-hotspot implementation unless the
E02-09 hotspot decision and E03-08 dispatcher gate are both PASS.  While the
gate is closed it still produces a complete, hash-bound record of the selected
hotspot, proposed OperatorSpec, independent strategies, planned cases, missing
measurements, and the exact conditions required to resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "hqsb.s03.e03_09"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def command(argv: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {
            "argv": argv,
            "exit_code": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"argv": argv, "exit_code": None, "error": repr(exc)}


def identity(repo: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(repo)),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path) if path.exists() else None,
    }


def load_verdict(repo: Path, experiment_id: str, path: Path) -> dict[str, Any]:
    item = {"experiment_id": experiment_id, **identity(repo, path)}
    if not path.exists():
        return {**item, "overall": "MISSING", "stage_gate": "closed"}
    verdict = read_json(path)
    return {
        **item,
        "overall": verdict.get("overall", verdict.get("status", "UNKNOWN")),
        "stage_gate": verdict.get("stage_gate"),
        "failed_conditions": verdict.get(
            "failed_conditions", verdict.get("failed_or_unverified_conditions", [])
        ),
        "blocking_dependencies": verdict.get("blocking_dependencies", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "e02_09_decision": repo / "docs/stage_experiments/S02/E02-09/raw/decision.json",
        "e02_09_shares": repo / "docs/stage_experiments/S02/E02-09/raw/shares.json",
        "e02_09_shape_weighting": repo / "docs/stage_experiments/S02/E02-09/raw/shape_weighting.json",
        "e02_09_verdict": repo / "docs/stage_experiments/S02/E02-09/raw/verdict.json",
        "e03_06_verdict": repo / "docs/stage_experiments/S03/E03-06/raw/verdict.json",
        "e03_07_verdict": repo / "docs/stage_experiments/S03/E03-07/raw/verdict.json",
        "e03_08_verdict": repo / "docs/stage_experiments/S03/E03-08/raw/verdict.json",
        "e03_08_dependencies": repo / "docs/stage_experiments/S03/E03-08/raw/dependency_gates.json",
        "s03_protocol": repo / "docs/stage_experiments/details/S03/README.md",
        "e03_09_detail": repo / "docs/stage_experiments/details/S03/E03-09_second_hotspot_transfer.md",
    }
    missing_inputs = [name for name, path in input_paths.items() if not path.exists()]
    input_identity = {name: identity(repo, path) for name, path in input_paths.items()}

    e02 = load_verdict(repo, "E02-09", input_paths["e02_09_verdict"])
    e03_06 = load_verdict(repo, "E03-06", input_paths["e03_06_verdict"])
    e03_07 = load_verdict(repo, "E03-07", input_paths["e03_07_verdict"])
    e03_08 = load_verdict(repo, "E03-08", input_paths["e03_08_verdict"])
    direct_prerequisites = {"E02-09": e02["overall"], "E03-08": e03_08["overall"]}
    failed_direct = [name for name, status in direct_prerequisites.items() if status != "PASS"]
    transitive_blockers = [
        name for name, item in (("E03-06", e03_06), ("E03-07", e03_07))
        if item["overall"] != "PASS"
    ]
    gate_open = not missing_inputs and not failed_direct
    dependency_gates = {
        "schema_version": f"{SCHEMA}.dependencies/v1",
        "ordering_source": "details/S03/README.md section 8: E03-08 dispatcher -> E03-09 second hotspot",
        "gate_rule": "E02-09 and E03-08 must both be PASS before E03-09 dynamic work",
        "inputs_missing": missing_inputs,
        "dependencies": [e02, e03_06, e03_07, e03_08],
        "direct_prerequisites": direct_prerequisites,
        "failed_direct_prerequisites": failed_direct,
        "transitive_safety_blockers": transitive_blockers,
        "gate_open": gate_open,
        "decision": "PROCEED" if gate_open else "BLOCK",
    }
    write_json(out / "dependency_gates.json", dependency_gates)

    decision = read_json(input_paths["e02_09_decision"]) if input_paths["e02_09_decision"].exists() else {}
    selected = decision.get("record", {}).get("selected", [])
    hotspot = next((item for item in selected if item.get("operator_id") == "prefill_softmax"), None)
    hotspot_import = {
        "schema_version": f"{SCHEMA}.hotspot-import/v1",
        "source": input_identity["e02_09_decision"],
        "source_verdict": e02["overall"],
        "selected_second_hotspot": hotspot,
        "rank_one_handling": decision.get("record", {}).get("rank_one_handling"),
        "why_this_is_not_a_gemm_experiment": (
            "E02-09 selected prefill softmax as the S03 self-developed second line; "
            "decode GEMM remains on the mature-library route and may not replace that decision."
        ),
        "import_status": "VALID" if hotspot and e02["overall"] == "PASS" else "INVALID",
        "dynamic_revalidation_status": "NOT_RUN_BLOCKED" if not gate_open else "REQUIRED_BEFORE_RUN",
    }
    write_json(out / "hotspot_decision_import.json", hotspot_import)

    operator_spec = {
        "schema_version": f"{SCHEMA}.operator-spec-draft/v1",
        "status": "DRAFT_NOT_RUNTIME_VALIDATED",
        "operator_id": "prefill_attention_softmax",
        "version": "e03-09-draft-v1-blocked",
        "boundary": "softmax over the last dimension after attention scaling and additive mask",
        "equation": "m=max_j(x_j); y_i=exp(x_i-m)/sum_j(exp(x_j-m))",
        "inputs": [{"name": "scores_after_mask", "shape": "[B,Hq,Q,K]", "runtime_real": [[1,16,128,128],[1,16,2048,2048]]}],
        "outputs": [{"name": "probabilities", "shape": "same as input"}],
        "axis": -1,
        "dtype": {"input": "fp16", "accumulation": "fp32", "output": "fp16"},
        "layout": "contiguous row-major required by draft fast paths; other strides reject or framework-fallback",
        "mask": "additive causal/padding mask is already applied; finite, -Inf, and all-masked policies require explicit tests",
        "special_values": "NaN/Inf and all-masked behavior must match the independent oracle/framework contract",
        "alias": "input/output alias forbidden until explicitly proven safe",
        "stream": "explicit caller CUDA stream; no default-stream substitution",
        "workspace": "strategy-specific and caller-owned; no hidden allocation",
        "supported_domain": "B=1,Hq=16,Q=K in registered prefill length classes; odd/boundary lengths explicit support-or-reject",
        "fallback": "aten::_softmax with identical dtype, axis, input tensor, and current stream",
        "tolerance_ceiling": decision.get("s03_protocol", {}).get("correctness", {}).get("second_hotspot", {}).get("tolerance_ceiling"),
        "reference": "independent CPU FP64 stable-softmax oracle plus FP32 oracle; framework result is cross-check only",
        "unresolved_before_execution": [
            "exact all-masked-row policy of the current framework/model path",
            "runtime stride/alignment census and current kernel symbol",
            "whether output dtype is preserved for every selected model path",
        ],
    }
    write_json(out / "operator_spec_draft.json", operator_spec)

    strategies = {
        "schema_version": f"{SCHEMA}.strategy-plan/v1",
        "status": "FROZEN_NOT_IMPLEMENTED_OR_EXECUTED",
        "baseline": {
            "id": "framework_aten_softmax",
            "meaningful_implementation": True,
            "expected_domain": "all supported cases and fallback",
            "actual_kernel_from_s02": {"T128": "softmax_warp_forward", "T2048": "cunn_SoftMaxForwardSmem"},
        },
        "candidates": [
            {
                "id": "cta_row_vectorized",
                "hypothesis": "one row per CTA with vectorized loads can reduce instruction overhead for K=2048",
                "expected_win_domain": "long prefill, aligned contiguous K=2048",
                "expected_loss_domain": "short or odd K where launch/tail overhead dominates",
                "risk": "shared-memory and block synchronization cost; tail/alignment correctness",
                "stop": "reject on correctness/stream/sanitizer failure or if 95% CI does not clear 5% guard",
            },
            {
                "id": "multirow_warp_shuffle",
                "hypothesis": "several rows per CTA with warp reductions can improve small/medium-K row throughput",
                "expected_win_domain": "short prefill and routing boundary lengths",
                "expected_loss_domain": "K=2048 if cross-warp staging or occupancy dominates",
                "risk": "active-lane masks, cross-warp reduction, exp staging, all-masked behavior",
                "stop": "reject on correctness/stream/sanitizer failure or no registered domain clears guard",
            },
        ],
        "distinctness_requirement": "timeline/symbol must prove the two candidates do not resolve to the same kernel",
        "execution_status": "NOT_RUN_BLOCKED",
    }
    write_json(out / "strategy_plan.json", strategies)

    planned_cases = {
        "schema_version": f"{SCHEMA}.planned-cases/v1",
        "status": "NOT_RUN_BLOCKED",
        "real_shapes": [
            {"case": "short", "shape": [1,16,128,128], "calls_per_prefill": 28},
            {"case": "long", "shape": [1,16,2048,2048], "calls_per_prefill": 28},
        ],
        "boundary_shapes": [1,2,31,32,33,127,128,129,255,256,257,2047,2048,2049],
        "value_classes": ["zero", "tiny", "large_signed", "nan", "pos_inf", "neg_inf", "masked", "all_masked"],
        "layout_classes": ["contiguous_aligned", "misaligned", "odd_tail", "non_contiguous_reject_or_fallback"],
        "required_modes": ["framework_baseline", "forced_cta_row", "forced_multirow_warp", "auto", "fallback"],
        "processes_min": 3,
        "ordinary_and_profiler_separate": True,
        "profile_representatives": ["candidate_win", "no_gain_or_regression", "short", "long", "route_boundary"],
        "safety": ["non_default_stream", "concurrent_streams", "workspace_lifetime", "invalid_input", "compute_sanitizer"],
        "guard_band_relative": 0.05,
    }
    write_json(out / "planned_case_matrix.json", planned_cases)

    blocked_reason = [*failed_direct, *[x for x in transitive_blockers if x not in failed_direct]]
    collection_status = {
        "schema_version": f"{SCHEMA}.collection/v1",
        "required_by_checklist": {
            "hotspot_shapes_and_share": {"status": "IMPORTED_HISTORICAL_NOT_REVALIDATED", "artifact": "hotspot_decision_import.json"},
            "strategy_hypotheses": {"status": "FROZEN_NOT_EXECUTED", "artifact": "strategy_plan.json"},
            "correctness": {"status": "NOT_COLLECTED_BLOCKED", "reason": blocked_reason},
            "microbenchmark": {"status": "NOT_COLLECTED_BLOCKED", "reason": blocked_reason},
            "profile": {"status": "NOT_COLLECTED_BLOCKED", "reason": blocked_reason},
            "selection_or_abandonment_reason": {"status": "NOT_AVAILABLE_BEFORE_VALID_RUN", "reason": blocked_reason},
        },
        "dynamic_trace_count": 0,
        "correctness_case_count": 0,
        "ordinary_benchmark_process_count": 0,
        "profiler_case_count": 0,
        "dispatcher_case_count": 0,
        "invented_or_reused_as_e03_09_measurement": False,
    }
    write_json(out / "collection_status.json", collection_status)

    protocol = {
        "schema_version": f"{SCHEMA}.protocol/v1",
        "experiment_id": "E03-09",
        "frozen_before_dynamic_execution": True,
        "gate_rule": dependency_gates["gate_rule"],
        "pass_rule": "all eight E03-09 acceptance conditions true and at least one usable real-domain candidate",
        "pass_negative_rule": "two legal strategies complete correctness/stream/sanitizer and sufficient sampling, but no gain clears guard",
        "blocked_rule": "hotspot input missing or an ordered correctness/safety/dispatcher prerequisite is not PASS",
        "performance_guard": decision.get("s03_protocol", {}).get("performance", {}).get("guard_band"),
        "measurement_rule": "ordinary benchmark, timeline, NCU, and sanitizer are separate runs",
        "no_claims_while_blocked": ["correctness", "speedup", "profile mechanism", "dispatch", "Amdahl update", "PASS_NEGATIVE"],
    }
    write_json(out / "protocol.json", protocol)

    provenance = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "commands": {
            "git_head": command(["git", "rev-parse", "HEAD"]),
            "git_status": command(["git", "status", "--short"]),
            "nvidia_smi": command(["nvidia-smi"]),
            "nvcc": command(["/usr/local/cuda/bin/nvcc", "--version"]),
        },
        "input_identity": input_identity,
        "dynamic_cuda_work_executed": False,
    }
    write_json(out / "provenance.json", provenance)

    conditions = {
        "1_second_hotspot_selected_from_s02_evidence": bool(hotspot) and e02["overall"] == "PASS",
        "2_module_op_kernel_shape_share_current_baseline_revalidated": False,
        "3_independent_c3_reference_and_tolerance_executed": False,
        "4_two_meaningful_strategies_forced_hit": False,
        "5_real_boundary_correct_safe_stream_clean": False,
        "6_ordinary_raw_profile_and_regression_complete": False,
        "7_shape_dispatch_and_fallback_executed": False,
        "8_usable_real_domain_candidate_formed": False,
    }
    overall = "BLOCKED" if not gate_open else ("PASS" if all(conditions.values()) else "FAIL")
    verdict = {
        "schema_version": f"{SCHEMA}.verdict/v1",
        "experiment_id": "E03-09",
        "verified_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "overall": overall,
        "dependency_gate_open": gate_open,
        "blocking_dependencies": blocked_reason,
        "conditions": conditions,
        "failed_or_unverified_conditions": [name for name, value in conditions.items() if not value],
        "expected_effect_achieved": False,
        "single_item_pass_standard_met": False,
        "pass_negative_eligible": False,
        "reason": "E03-08 is not PASS; dynamic E03-09 work would violate the frozen S03 safety order",
        "stage_gate": "closed" if not gate_open else "evaluated",
    }
    write_json(out / "verdict.json", verdict)

    manifest = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "EVIDENCE_MANIFEST.json":
            manifest.append({
                "path": str(path.relative_to(out)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    write_json(out / "EVIDENCE_MANIFEST.json", {"schema_version": "hqsb.evidence-manifest/v1", "files": manifest})
    print(json.dumps(verdict, ensure_ascii=False))
    return 0 if overall == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
