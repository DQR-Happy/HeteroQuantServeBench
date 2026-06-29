#!/usr/bin/env python3
"""E03-08 dependency gate and dispatcher evidence freezer.

This audit deliberately does not execute an auto-routed kernel while the
E03-06/E03-07 safety gates are closed.  It freezes the independent registry,
routing expectations, missing evidence and source/binary identities needed to
resume the experiment without silently turning a blocked run into a pass.
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


SCHEMA = "hqsb.s03.e03_08"


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


def source_check(text: str, needle: str) -> bool:
    return needle in text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    dependency_paths = {
        "E03-01": repo / "docs/stage_experiments/S03/E03-01/raw/verdict.json",
        "E03-02": repo / "docs/stage_experiments/S03/E03-02/raw/verdict.json",
        "E03-03": repo / "docs/stage_experiments/S03/E03-03/raw/verdict.json",
        "E03-04": repo / "docs/stage_experiments/S03/E03-04/raw/verdict.json",
        "E03-05": repo / "docs/stage_experiments/S03/E03-05/raw/verdict.json",
        "E03-06": repo / "docs/stage_experiments/S03/E03-06/raw/verdict.json",
        "E03-07": repo / "docs/stage_experiments/S03/E03-07/raw/verdict.json",
    }
    dependencies: list[dict[str, Any]] = []
    for experiment_id, path in dependency_paths.items():
        if path.exists():
            verdict = read_json(path)
            dependencies.append(
                {
                    "experiment_id": experiment_id,
                    "path": str(path.relative_to(repo)),
                    "sha256": sha256(path),
                    "overall": verdict.get("overall"),
                    "failed_conditions": verdict.get("failed_conditions", []),
                    "stage_gate": verdict.get("stage_gate"),
                }
            )
        else:
            dependencies.append(
                {
                    "experiment_id": experiment_id,
                    "path": str(path.relative_to(repo)),
                    "sha256": None,
                    "overall": "MISSING",
                    "failed_conditions": ["verdict artifact missing"],
                    "stage_gate": "closed",
                }
            )

    by_id = {item["experiment_id"]: item for item in dependencies}
    hard_prerequisites = ("E03-01", "E03-03", "E03-05", "E03-06", "E03-07")
    failed_prerequisites = [
        experiment_id
        for experiment_id in hard_prerequisites
        if by_id[experiment_id]["overall"] != "PASS"
    ]
    gate_open = not failed_prerequisites
    dependency_evidence = {
        "schema_version": f"{SCHEMA}.dependencies/v1",
        "hard_prerequisites": list(hard_prerequisites),
        "gate_rule": "all hard prerequisites must be PASS before any auto route executes",
        "dependencies": dependencies,
        "failed_prerequisites": failed_prerequisites,
        "gate_open": gate_open,
        "decision": "PROCEED" if gate_open else "BLOCK",
    }
    write_json(out / "dependency_gates.json", dependency_evidence)

    dispatcher_path = repo / "ops/dispatcher.py"
    header_path = repo / "ops/cuda/rmsnorm/include/hqsb/rmsnorm.h"
    cuda_dispatcher_path = repo / "ops/cuda/rmsnorm/src/rmsnorm_dispatcher.cu"
    c_api_path = repo / "ops/cuda/rmsnorm/src/rmsnorm_c_api.cu"
    fused_header_path = repo / "ops/cuda/fused_residual_rmsnorm/include/hqsb/fused_residual_rmsnorm.h"
    sources = {
        "python_dispatcher": dispatcher_path,
        "rmsnorm_header": header_path,
        "cuda_dispatcher": cuda_dispatcher_path,
        "c_api": c_api_path,
        "fused_header": fused_header_path,
    }
    source_identity = {
        name: {
            "path": str(path.relative_to(repo)),
            "sha256": sha256(path) if path.exists() else None,
            "bytes": path.stat().st_size if path.exists() else None,
        }
        for name, path in sources.items()
    }

    rms_binary = repo / "build/jetson-release/ops/cuda/rmsnorm/libhqsb_rmsnorm_shared.so"
    fused_binary = repo / "build/jetson-release/ops/cuda/fused_residual_rmsnorm/libhqsb_fused_residual_rmsnorm_shared.so"
    binary_identity = {}
    for name, path in {"rmsnorm": rms_binary, "fused_residual_rmsnorm": fused_binary}.items():
        binary_identity[name] = {
            "path": str(path.relative_to(repo)),
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else None,
            "bytes": path.stat().st_size if path.exists() else None,
        }

    quarantine = [
        "E03-06 public API stream contract failed",
        "E03-07 sanitizer/API safety gate failed",
    ]
    registry = {
        "schema_version": f"{SCHEMA}.registry/v1",
        "registry_version": "e03-08-draft-v1-blocked",
        "operator": "rmsnorm",
        "semantic_spec": "E03-01 resolved OperatorSpec",
        "source_identity": source_identity,
        "binary_identity": binary_identity,
        "entries": [
            {
                "implementation_id": "cuda.v0_shared.b256",
                "requested": "forced_v0",
                "dtype": ["fp32"],
                "layout": "contiguous row-major (raw ABI cannot inspect stride)",
                "workspace_bytes": 0,
                "stream": "explicit C++/_ex C ABI; legacy no-stream C ABI still public",
                "status": "QUARANTINED",
                "reasons": quarantine,
                "fallback_target": None,
            },
            {
                "implementation_id": "cuda.v1_warp_shuffle.b256",
                "requested": "forced_v1",
                "dtype": ["fp32"],
                "layout": "contiguous row-major (raw ABI cannot inspect stride)",
                "workspace_bytes": 0,
                "stream": "explicit C++/_ex C ABI; legacy no-stream C ABI still public",
                "status": "QUARANTINED",
                "reasons": quarantine,
                "fallback_target": None,
            },
            {
                "implementation_id": "cuda.v2_vectorized.b256",
                "requested": "forced_v2_strict",
                "dtype": ["fp32", "fp16"],
                "layout": "contiguous row-major",
                "alignment": {"fp32_bytes": 16, "fp16_bytes": 4},
                "tail": {"fp32_h_mod": 4, "fp16_h_mod": 2},
                "workspace_bytes": 0,
                "status": "QUARANTINED",
                "reasons": quarantine,
                "fallback_target": None,
            },
            {
                "implementation_id": "cuda.scalar_safe.b256",
                "requested": "forced_scalar_safe",
                "dtype": ["fp32", "fp16"],
                "layout": "contiguous row-major",
                "workspace_bytes": 0,
                "status": "QUARANTINED",
                "reasons": quarantine,
                "fallback_target": None,
            },
            {
                "implementation_id": "cuda.fused_residual_rmsnorm.v1.b256",
                "requested": "forced_fused",
                "dtype": ["fp32", "fp16"],
                "semantic_version": "residual_out=x+residual; y=RMSNorm(residual_out)",
                "workspace_bytes": 0,
                "status": "QUARANTINED",
                "reasons": quarantine,
                "fallback_target": "separate residual_add + rmsnorm (not registered as an E03-08 executable reference)",
            },
            {
                "implementation_id": "registered_reference",
                "requested": "reference",
                "dtype": ["fp32", "fp16"],
                "status": "BINARY_MISSING",
                "reasons": [
                    "RmsNormVariant::kReference is documented CPU-only and rmsnorm_forward rejects it",
                    "no explicit same-device production reference is registered",
                ],
                "fallback_target": None,
            },
        ],
    }
    write_json(out / "candidate_registry.json", registry)

    candidates_path = repo / "docs/stage_experiments/S03/E03-02/raw/routing_candidates.json"
    candidates = read_json(candidates_path) if candidates_path.exists() else {"domains": []}
    routing_rules = []
    for index, domain in enumerate(candidates.get("domains", []), start=1):
        routing_rules.append(
            {
                "rule_id": f"rms-fp32-domain-{index:02d}",
                "source": str(candidates_path.relative_to(repo)),
                "source_sha256": sha256(candidates_path),
                "predicate": {"dtype": domain.get("dtype"), **domain.get("domain", {})},
                "candidate": domain.get("proposed"),
                "proposal_kind": domain.get("proposal_kind"),
                "guard_band": domain.get("guard_band"),
                "status": "DISABLED_BY_SAFETY_GATE",
                "reason": "E03-06/E03-07 hard prerequisites are not PASS",
            }
        )
    fused_summary_path = repo / "docs/stage_experiments/S03/E03-05/raw/summary.json"
    fused_summary = read_json(fused_summary_path) if fused_summary_path.exists() else {}
    routing_draft = {
        "schema_version": f"{SCHEMA}.routing-draft/v1",
        "table_version": "e03-08-draft-v1-blocked",
        "frozen_static_table": True,
        "online_autotune": False,
        "rules": routing_rules,
        "fused_candidate_cases": fused_summary.get("verdict", {}).get(
            "performance_positive_cases", []
        ),
        "fused_status": "DISABLED_BY_SAFETY_GATE",
        "unknown_shape_policy": "registered_reference",
        "unknown_shape_policy_executable": False,
        "unknown_shape_failure": "NO_IMPLEMENTATION",
    }
    write_json(out / "routing_rules_draft.json", routing_draft)

    required_key = {
        "operator": "rmsnorm",
        "spec_version": "E03-01",
        "device_id": 0,
        "device_identity": "must be captured at execution",
        "compute_capability": [8, 7],
        "driver_runtime": "must be captured at execution",
        "binary_hash": binary_identity["rmsnorm"]["sha256"],
        "dtype_in_weight_out": "case-specific",
        "accumulation_dtype": "fp32",
        "rows": "case-specific",
        "hidden": "case-specific",
        "layout_stride": "case-specific",
        "alignment_class": "case-specific",
        "tail_class": "case-specific",
        "alias_mode": "case-specific",
        "workspace_capability": 0,
        "requested_mode": "case-specific",
        "routing_table_version": "e03-08-draft-v1-blocked",
    }
    cases = [
        ("forced_v0_eligible", "forced_v0", "v0_shared"),
        ("forced_v1_eligible", "forced_v1", "v1_warp_shuffle"),
        ("forced_v2_aligned", "forced_v2_strict", "v2_vectorized"),
        ("auto_v2_aligned", "auto", "v2_vectorized"),
        ("auto_odd_tail", "auto", "scalar_safe_or_v1"),
        ("auto_fused_domain", "auto_fused", "fused_v1"),
        ("auto_unknown_shape", "auto", "registered_reference"),
        ("auto_bf16", "auto", "reject_or_registered_reference"),
        ("auto_noncontiguous", "auto", "reject_or_registered_reference"),
        ("capability_v2_missing", "auto", "v1"),
        ("capability_v1_missing", "auto", "v0"),
        ("capability_v0_missing", "auto", "registered_reference"),
        ("capability_all_missing", "auto", "NO_IMPLEMENTATION"),
        ("workspace_unavailable", "auto", "fallback_or_reject"),
        ("unknown_variant", "unknown", "reject"),
        ("runtime_prelaunch_failure", "auto", "policy-controlled fallback"),
        ("runtime_immediate_failure", "auto", "return_error_no_retry"),
        ("runtime_async_failure", "auto", "return_error_no_retry"),
    ]
    expected_table = []
    for case_id, requested, expected in cases:
        expected_table.append(
            {
                "case_id": case_id,
                "dispatch_key_template": required_key,
                "requested": requested,
                "expected_actual": expected,
                "expected_reason": "case-specific structured reason",
                "expected_fallback_chain": [],
                "execution_status": "NOT_RUN_BLOCKED",
                "block_reason": failed_prerequisites,
            }
        )
    write_json(
        out / "expected_decision_table.json",
        {
            "schema_version": f"{SCHEMA}.expected-table/v1",
            "independent_from_production_dispatcher": True,
            "boundary_policy": "numeric/modulo predicates require boundary-1,boundary,boundary+1",
            "cases": expected_table,
        },
    )

    py_text = dispatcher_path.read_text()
    header_text = header_path.read_text()
    cuda_text = cuda_dispatcher_path.read_text()
    gaps = {
        "capability_policy_execution_separated": False,
        "dispatch_key_has_rows": source_check(py_text, "rows"),
        "dispatch_key_has_layout_or_stride": source_check(py_text, "stride") or source_check(py_text, "layout"),
        "dispatch_key_has_alignment": source_check(py_text, "alignment"),
        "dispatch_key_has_device_build_spec_identity": all(
            source_check(py_text, token) for token in ("device", "build", "spec")
        ),
        "decision_reports_requested": source_check(py_text, "requested"),
        "decision_reports_actual": source_check(py_text, "actual"),
        "decision_reports_fallback_chain": source_check(py_text, "fallback_chain"),
        "decision_reports_reason": source_check(py_text, "reason"),
        "auto_actual_observable_at_c_abi": source_check(header_text, "RmsNormDispatchInfo")
        and source_check(cuda_text, "rmsnorm_resolve_dispatch"),
        "forced_v2_never_fallback": not source_check(
            cuda_text,
            "actual = vector_eligible ? RmsNormVariant::kV2Vectorized\n                             : RmsNormVariant::kScalarSafe",
        ),
        "registered_same_device_reference": not source_check(
            header_text, "CPU FP64 reference (test/benchmark only; never on device)"
        ),
        "cache_key_and_invalidation_implemented": source_check(py_text, "cache")
        and source_check(py_text, "binary"),
        "legacy_no_stream_api_absent": "hqsb_rmsnorm_forward_c" not in c_api_path.read_text(),
    }
    static_audit = {
        "schema_version": f"{SCHEMA}.static-audit/v1",
        "checks": gaps,
        "failed_checks": [name for name, value in gaps.items() if not value],
        "notes": {
            "auto_observability_scope": "C++ resolve info exists, but Python production decision is not a complete replay trace",
            "forced_v2_compatibility": "requested kV2Vectorized may resolve to kScalarSafe; strict forced mode is separate",
            "reference": "the only named C++ reference is CPU/test-only and production forward rejects it",
        },
    }
    write_json(out / "static_gap_audit.json", static_audit)

    protocol = {
        "schema_version": f"{SCHEMA}.protocol/v1",
        "experiment_id": "E03-08",
        "frozen_before_execution": True,
        "prerequisite_rule": dependency_evidence["gate_rule"],
        "pass_rule": "all ten detail-spec acceptance conditions true; no PASS_NEGATIVE",
        "blocked_rule": "any prerequisite correctness/stream/sanitizer gate not PASS",
        "runtime_failure_policy": {
            "selection_prelaunch": "fallback only before output/workspace mutation",
            "immediate_launch": "return error; no automatic retry",
            "asynchronous": "return error and isolate process; never rerun in same output",
        },
        "cache_invalidation_dimensions": [
            "device id/UUID",
            "binary hash",
            "driver/runtime",
            "capability provider generation",
            "routing table version",
            "OperatorSpec version",
        ],
        "hidden_actions_forbidden": [
            "dtype cast",
            "contiguous copy",
            "stream switch",
            "CPU transfer",
            "internal allocation",
            "synchronization",
            "online autotune",
        ],
    }
    write_json(out / "protocol.json", protocol)

    collection_status = {
        "schema_version": f"{SCHEMA}.collection/v1",
        "required_by_checklist": {
            "dispatch_key": {"status": "FROZEN_NOT_EXECUTED", "artifact": "expected_decision_table.json"},
            "requested": {"status": "FROZEN_NOT_EXECUTED", "artifact": "expected_decision_table.json"},
            "actual": {"status": "NOT_COLLECTED_BLOCKED", "reason": failed_prerequisites},
            "reason": {"status": "EXPECTED_ONLY", "artifact": "expected_decision_table.json"},
            "fallback": {"status": "EXPECTED_ONLY", "artifact": "expected_decision_table.json"},
            "correctness": {"status": "NOT_COLLECTED_BLOCKED", "reason": failed_prerequisites},
            "latency": {"status": "NOT_COLLECTED_BLOCKED", "reason": failed_prerequisites},
        },
        "executed_dispatch_case_count": 0,
        "invented_or_reused_as_e03_08_measurement": False,
    }
    write_json(out / "collection_status.json", collection_status)

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
        "source_identity": source_identity,
        "binary_identity": binary_identity,
    }
    write_json(out / "provenance.json", provenance)

    conditions = {
        "1_registry_capability_policy_execution_separated": gaps["capability_policy_execution_separated"],
        "2_complete_dispatch_key": all(
            gaps[name]
            for name in (
                "dispatch_key_has_rows",
                "dispatch_key_has_layout_or_stride",
                "dispatch_key_has_alignment",
                "dispatch_key_has_device_build_spec_identity",
            )
        ),
        "3_each_declared_path_auto_and_forced_hit": False,
        "4_forced_never_fallback": gaps["forced_v2_never_fallback"],
        "5_unknown_unfavorable_capability_missing_safe": False,
        "6_requested_actual_reason_fallback_complete": all(
            gaps[name]
            for name in (
                "decision_reports_requested",
                "decision_reports_actual",
                "decision_reports_reason",
                "decision_reports_fallback_chain",
            )
        ),
        "7_runtime_failure_policy_dynamically_verified": False,
        "8_cache_invalidation_and_concurrency_safe": gaps["cache_key_and_invalidation_implemented"],
        "9_stream_preserved_no_hidden_alloc_copy_sync": False,
        "10_performance_sanity_no_unexplained_regression": False,
    }
    overall = "BLOCKED" if not gate_open else ("PASS" if all(conditions.values()) else "FAIL")
    verdict = {
        "schema_version": f"{SCHEMA}.verdict/v1",
        "experiment_id": "E03-08",
        "verified_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "overall": overall,
        "dependency_gate_open": gate_open,
        "blocking_dependencies": failed_prerequisites,
        "conditions": conditions,
        "failed_or_unverified_conditions": [name for name, value in conditions.items() if not value],
        "executed_dispatch_case_count": 0,
        "reason": "hard prerequisites failed; executing auto routing would violate the frozen S03 safety order",
        "stage_gate": "closed",
    }
    write_json(out / "verdict.json", verdict)

    manifest = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "EVIDENCE_MANIFEST.json":
            manifest.append(
                {
                    "path": str(path.relative_to(out)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_json(
        out / "EVIDENCE_MANIFEST.json",
        {"schema_version": "hqsb.evidence-manifest/v1", "files": manifest},
    )
    print(json.dumps(verdict, ensure_ascii=False))
    return 0 if overall == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
