#!/usr/bin/env python3
"""Regenerate ``configs/infra/*.yaml`` from the code constants (E13 scaffolding).

The thirteen S13 documents are *frozen vocabularies*.  Authoring them by hand
invites silent drift from the implementation, so this generator writes them from
the same constants the audit test compares against:

    python3 scripts/infra/gen_infra_specs.py            # write the documents
    python3 scripts/infra/gen_infra_specs.py --check     # fail if a document would change

``--check`` is the mode used by the test suite: if a constant changes without the
document being regenerated, the audit fails instead of the report quietly citing a
stale vocabulary.

The documents deliberately contain **no measured values**: thresholds, replicas,
traffic fractions, prices and durations are campaign inputs.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml  # noqa: E402

from hqsb.infra import (  # noqa: E402
    artifacts,
    autoscaling,
    canary,
    capacity,
    deployment,
    faults,
    identity,
    lifecycle,
    observability,
    records,
    scheduling,
    security,
    supply_chain,
    experiment,
)

SPEC_DIR = os.path.join(REPO_ROOT, "configs", "infra")

COMMON: Dict[str, Any] = {
    "schema_version": "1.0.0",
    "status": "FROZEN_VOCABULARY_ONLY",
    "notes": "只冻结结构与词汇，不含任何测量值；阈值/副本数/流量比例属于 campaign 输入",
}


def _spec(kind: str, name: str, description: str, **kwargs: Any) -> Dict[str, Any]:
    document = {"kind": kind, "name": name, "description": description, **COMMON}
    document.update(kwargs)
    return document


def documents() -> Dict[str, Dict[str, Any]]:
    """The thirteen S13 documents, keyed by file name."""
    return {
        "release_identity_spec.yaml": _spec(
            "hqsb.infra.release_identity_spec",
            "S13 release/OCI/source identity",
            "部署身份单元：ReleaseBundle 字段、digest 规则与可重建等级（E13-01 §6–§7）",
            release_statuses=list(identity.RELEASE_STATUSES),
            reproducibility_levels=list(identity.REPRODUCIBILITY_LEVELS),
            required_identity_fields=list(identity.REQUIRED_IDENTITY_FIELDS),
            nondeterminism_sources=list(identity.NONDETERMINISM_SOURCES),
            bundle_fields=list(identity.RELEASE_BUNDLE_FIELDS),
        ),
        "supply_chain_spec.yaml": _spec(
            "hqsb.infra.supply_chain_spec",
            "S13 supply-chain gate vocabulary",
            "镜像分层、SBOM/漏洞/秘密/许可证扫描与 release gate 决策（E13-01 §8–§10）",
            image_stages=list(supply_chain.IMAGE_STAGES),
            gate_decisions=list(supply_chain.GATE_DECISIONS),
            vuln_statuses=list(supply_chain.VULN_STATUSES),
            severities=list(supply_chain.SEVERITIES),
            secret_surfaces=list(records.SECRET_SCAN_SURFACES),
            model_file_patterns=list(records.MODEL_FILE_PATTERNS),
            negative_cases=list(supply_chain.NEGATIVE_CASES),
            required_evidence_parts=list(supply_chain.REQUIRED_EVIDENCE_PARTS),
            blocking_severity=supply_chain.DEFAULT_BLOCKING_SEVERITY,
        ),
        "deployment_spec.yaml": _spec(
            "hqsb.infra.deployment_spec",
            "S13 clean-bootstrap vocabulary",
            "clean level、部署状态机、阶段与失败清理（E13-02 §2–§3）",
            clean_levels=list(deployment.CLEAN_LEVELS),
            deployment_states=list(deployment.DEPLOYMENT_STATES),
            stage_names=list(deployment.STAGE_NAMES),
            residual_kinds=list(deployment.RESIDUAL_KINDS),
            negative_cases=list(deployment.NEGATIVE_CASES),
            forbidden_implicit_state=list(deployment.FORBIDDEN_IMPLICIT_STATE),
        ),
        "scheduling_spec.yaml": _spec(
            "hqsb.infra.scheduling_spec",
            "S13 placement/isolation vocabulary",
            "设备共享模式、拓扑策略、label 信任与隔离层（E13-03 §3–§4）",
            sharing_modes=list(scheduling.SHARING_MODES),
            partition_profiles=list(scheduling.PARTITION_PROFILES),
            topology_policies=list(scheduling.TOPOLOGY_POLICIES),
            link_domains=list(scheduling.LINK_DOMAINS),
            label_classes=list(scheduling.LABEL_CLASSES),
            protected_label_classes=list(scheduling.PROTECTED_LABEL_CLASSES),
            isolation_layers=list(scheduling.ISOLATION_LAYERS),
            negative_cases=list(scheduling.NEGATIVE_CASES),
            locality_metrics=list(scheduling.LOCALITY_METRICS),
        ),
        "artifact_spec.yaml": _spec(
            "hqsb.infra.artifact_spec",
            "S13 model-artifact lifecycle vocabulary",
            "content-addressed cache key、验证/兼容检查、pin/GC 与切换回滚（E13-04 §3–§5）",
            lifecycle_states=list(records.ARTIFACT_LIFECYCLE_STATES),
            cache_key_fields=list(artifacts.CACHE_KEY_FIELDS),
            cache_areas=list(artifacts.CACHE_AREAS),
            verification_checks=list(artifacts.VERIFICATION_CHECKS),
            compatibility_checks=list(artifacts.COMPATIBILITY_CHECKS),
            pin_kinds=list(artifacts.PIN_KINDS),
            gc_reasons=list(artifacts.GC_REASONS),
            fault_cases=list(artifacts.FAULT_CASES),
        ),
        "lifecycle_spec.yaml": _spec(
            "hqsb.infra.lifecycle_spec",
            "S13 pod/request lifecycle vocabulary",
            "三类 probe 语义、终止预算、请求完成语义与资源释放（E13-05 §2–§4）",
            probe_kinds=list(records.PROBE_KINDS),
            probe_semantics=dict(lifecycle.PROBE_SEMANTICS),
            termination_stages=list(lifecycle.TERMINATION_STAGES),
            completion_semantics=list(records.REQUEST_COMPLETION_SEMANTICS),
            readiness_only_reasons=list(lifecycle.READINESS_ONLY_REASONS),
            release_targets=list(lifecycle.RELEASE_TARGETS),
            negative_cases=list(lifecycle.NEGATIVE_CASES),
            coverage_paths=list(lifecycle.COVERAGE_PATHS),
        ),
        "capacity_admission_spec.yaml": _spec(
            "hqsb.infra.capacity_admission_spec",
            "S13 capacity/admission vocabulary",
            "内存账本组件、admission 策略与 reason code、扫参与功能轴（E13-06 §2–§5）",
            memory_components=list(records.RESOURCE_MEMORY_COMPONENTS),
            admission_policies=list(capacity.ADMISSION_POLICIES),
            admission_decisions=list(records.ADMISSION_DECISIONS),
            reason_codes=list(capacity.REASON_CODES),
            sweep_stages=list(capacity.SWEEP_STAGES),
            budget_features=list(capacity.BUDGET_FEATURES),
            fault_cases=list(capacity.FAULT_CASES),
        ),
        "autoscaling_spec.yaml": _spec(
            "hqsb.infra.autoscaling_spec",
            "S13 autoscaling vocabulary",
            "策略集合、指标单位、episode 形状与降载保护（E13-07 §3–§5）",
            policies=list(autoscaling.POLICIES),
            metric_units=dict(autoscaling.METRIC_UNITS),
            leading_signals=list(autoscaling.LEADING_SIGNALS),
            episodes=list(records.AUTOSCALING_EPISODES),
            actions=list(records.AUTOSCALING_ACTIONS),
            failure_cases=list(autoscaling.FAILURE_CASES),
            scale_down_protections=list(autoscaling.SCALE_DOWN_PROTECTIONS),
        ),
        "observability_spec.yaml": _spec(
            "hqsb.infra.observability_spec",
            "S13 observability vocabulary",
            "层级 taxonomy、基数与脱敏策略、采样、单位与告警严重度（E13-08 §2–§6）",
            layers=list(records.OBSERVABILITY_LAYERS),
            signal_classes=list(records.SIGNAL_CLASSES),
            forbidden_metric_labels=list(records.FORBIDDEN_METRIC_LABELS),
            trace_only_attributes=list(records.TRACE_ONLY_ATTRIBUTES),
            sampling_kinds=list(observability.SAMPLING_KINDS),
            always_keep_cases=list(observability.ALWAYS_KEEP_CASES),
            telemetry_components=list(observability.TELEMETRY_COMPONENTS),
            base_units=list(observability.BASE_UNITS),
            latency_boundaries=list(observability.LATENCY_BOUNDARIES),
            alert_severities=list(records.ALERT_SEVERITIES),
            policy_default_marker=records.POLICY_DEFAULT_MARKER,
        ),
        "fault_spec.yaml": _spec(
            "hqsb.infra.fault_spec",
            "S13 fault-injection vocabulary",
            "故障层级/机制、降级策略、裁决与不变量（E13-09 §2–§5）",
            layers=list(records.FAULT_LAYERS),
            mechanisms=list(records.FAULT_MECHANISMS),
            degradation_strategies=list(records.DEGRADATION_STRATEGIES),
            verdicts=list(records.FAULT_VERDICTS),
            pre_injection_gates=list(faults.PRE_INJECTION_GATES),
            service_invariants=list(faults.SERVICE_INVARIANTS),
            resource_invariants=list(faults.RESOURCE_INVARIANTS),
            fault_contract_fields=list(faults.FAULT_CONTRACT_FIELDS),
        ),
        "canary_spec.yaml": _spec(
            "hqsb.infra.canary_spec",
            "S13 canary/release-governance vocabulary",
            "状态机、G0–G8 gate 层级、流量分配单位与 override 规则（E13-10 §2–§6）",
            states=list(records.CANARY_STATES),
            decisions=list(records.CANARY_DECISIONS),
            gates=list(records.CANARY_GATES),
            hard_gates=list(records.CANARY_HARD_GATES),
            assignment_units=list(records.ASSIGNMENT_UNITS),
            candidate_kinds=list(canary.CANDIDATE_KINDS),
            metric_directions=dict(canary.METRIC_DIRECTIONS),
            override_actions=list(canary.OVERRIDE_ACTIONS),
            forbidden_override_actions=list(canary.FORBIDDEN_OVERRIDE_ACTIONS),
            promotion_path=list(canary.PROMOTION_PATH),
        ),
        "security_spec.yaml": _spec(
            "hqsb.infra.security_spec",
            "S13 multi-tenant security vocabulary",
            "不变量 MT-I01..I12、安全用例类型、配额与滥用矩阵（E13-11 §4–§7）",
            invariants=list(records.TENANT_INVARIANTS),
            invariant_text=dict(records.TENANT_INVARIANT_TEXT),
            case_kinds=list(records.SECURITY_CASE_KINDS),
            verdicts=list(records.SECURITY_VERDICTS),
            subjects=list(records.TENANT_SUBJECTS),
            k8s_verbs=list(security.K8S_VERBS),
            rbac_risks=list(security.RBAC_RISKS),
            static_quota_resources=list(security.STATIC_QUOTA_RESOURCES),
            dynamic_budgets=list(security.DYNAMIC_BUDGETS),
            abuse_kinds=list(security.ABUSE_KINDS),
            audit_event_kinds=list(security.AUDIT_EVENT_KINDS),
            deny_case_kinds=list(security.DENY_CASE_KINDS),
            negative_result_wording=security.NEGATIVE_RESULT_WORDING,
        ),
        "experiment_spec.yaml": _spec(
            "hqsb.infra.experiment_spec",
            "S13 experiment governance vocabulary",
            "实验清单、协议状态、证据阶梯与缺失语义；并声明 destructive 故障需要安全策略（§20.2）",
            experiments=[f"E13-{index:02d}" for index in range(1, 12)],
            statuses=list(records.PROTOCOL_STATUSES),
            evidence_levels=list(records.EVIDENCE_LEVELS),
            missingness_codes=list(records.MISSINGNESS_CODES),
            prerequisite_states=list(records.PREREQUISITE_STATES),
            tables=list(records.TABLE_SCHEMAS),
            run_root="experiment_results/S13",
            claim_levels=list(experiment.CLAIM_LEVELS),
            safety_policy_required=True,
            removed_validation=[
                "未删除任何既有测试",
                "未放宽任何 correctness/quality gate",
                "实验层判定保持 BLOCKED，未以负结果替代",
            ],
            rules=[
                "驱动默认拒绝产出结论（PASS/FAIL/PASS_NEGATIVE 需要 --execute + 前置满足 + raw 样本）",
                "docs/stage_experiments/** 为只读协议树，任何实验产物不得写入该目录",
            ],
        ),
    }


def _serialize(document: Dict[str, Any]) -> str:
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100)


def write_documents(target_dir: str = SPEC_DIR) -> List[str]:
    os.makedirs(target_dir, exist_ok=True)
    written: List[str] = []
    for filename, document in documents().items():
        path = os.path.join(target_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(_serialize(document))
        written.append(path)
    return written


def check_documents(target_dir: str = SPEC_DIR) -> Tuple[bool, List[str]]:
    """Report documents that differ from the generated content (no silent drift)."""
    drifted: List[str] = []
    for filename, document in documents().items():
        path = os.path.join(target_dir, filename)
        expected = _serialize(document)
        if not os.path.isfile(path):
            drifted.append(f"{filename}: missing")
            continue
        with open(path, encoding="utf-8") as handle:
            actual = handle.read()
        if actual != expected:
            drifted.append(f"{filename}: differs from the generated vocabulary")
    return not drifted, drifted


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the S13 frozen vocabularies.")
    parser.add_argument("--check", action="store_true", help="fail if a document would change")
    parser.add_argument("--dir", default=SPEC_DIR, help="target directory")
    args = parser.parse_args(argv)
    if args.check:
        ok, drifted = check_documents(args.dir)
        print(f"[infra_specs] documents={len(documents())} drifted={len(drifted)} status={'PASS' if ok else 'FAIL'}")
        for item in drifted:
            print(f"  {item}")
        return 0 if ok else 1
    written = write_documents(args.dir)
    print(f"[infra_specs] wrote {len(written)} documents to {args.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
