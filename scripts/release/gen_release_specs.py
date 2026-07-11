#!/usr/bin/env python3
"""Generate and drift-check the frozen S15 vocabularies under ``configs/release/``.

The documents are **vocabularies, not results**: they freeze claim taxonomy,
evidence levels, channel limits, detector patterns, the §22 evidence package,
documentation code-block classes, release policy, figure sampling strata, the
demo/narrative ladders and the reproduction/upstream ladders.  They contain no
measured number (``UNMEASURED`` stands in for anything not yet measured) and no
machine-specific path.

Usage:
    scripts/release/gen_release_specs.py            # write configs/release/*.yaml
    scripts/release/gen_release_specs.py --check    # drift check (exit 1 on drift)
"""

from __future__ import annotations

import os
import sys
from typing import Dict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml  # noqa: E402

SPEC_DIR = os.path.join(REPO_ROOT, "configs", "release")

UNMEASURED = "UNMEASURED"

#: The eight frozen vocabularies, in a stable order.
DOCUMENTS = [
    {
        "kind": "claim-taxonomy",
        "version": "v1",
        "sources": [
            "docs/stage_experiments/details/S15/README.md",
            "docs/stage_experiments/details/S15/E15-01_claim_ledger_evidence_gate.md",
        ],
        "note": "claim 类型、证据等级、渠道最低等级与严重度；不含任何测量值",
        "levels": {
            "PLANNED": "只存在计划/设计，未实现或未运行",
            "SOURCE": "代码存在",
            "TEST": "当前版本的自动测试通过",
            "RUNTIME": "目标设备实测",
            "MODEL": "真实模型闭环",
            "SERVICE": "并发/SLO 口径",
            "PORTABLE": "统一协议跨硬件",
        },
        "channels": {
            "roadmap": "PLANNED",
            "design_document": "SOURCE",
            "test_status": "TEST",
            "readme_results": "RUNTIME",
            "service_dashboard": "SERVICE",
            "cross_hardware_guide": "PORTABLE",
            "demo": "mirror_claim",
            "resume_bullet": "mirror_claim",
        },
        "severities": {
            "P0": "错误性能/质量数字、越级证据、错误模型/硬件、质量失败仍宣传、伪造归属",
            "P1": "范围或限制缺失、证据入口不可达、当前版本 stale",
            "P2": "不改变事实的措辞/导航/格式问题",
            "INFO": "历史 claim、论文 context 或预期趋势，已明确标非 HQSB 实测",
        },
    },
    {
        "kind": "detector-patterns",
        "version": "v1",
        "sources": [
            "docs/stage_experiments/details/S15/E15-01_claim_ledger_evidence_gate.md",
        ],
        "note": "可检测语言模式与非 claim 排除规则（中英文变体）",
        "patterns": {
            "numeric": "数字/百分比/倍数/时延/吞吐/内存/功耗/成本/误差/质量/置信区间",
            "superlative": "最快/最优/最高/首个/fastest/best-in-class/SOTA",
            "capability": "支持/兼容/production-ready/生产级/tail-safe/确定性/可扩展/跨硬件/端到端",
            "comparison": "更快/更慢/优于/接近/不退化/持平/faster/no regression/on par with",
        },
        "exclusions": {
            "code_block": "代码示例不是 claim",
            "formula": "公式推导不是 claim",
            "citation": "论文/引用转述不是 HQSB claim",
            "plan": "计划/roadmap/未来式不是当前能力",
            "problem_statement": "问题陈述/假设不是结论",
            "issue_quote": "checklist/标题不是 claim",
        },
    },
    {
        "kind": "evidence-package",
        "version": "v1",
        "sources": [
            "docs/stage_experiments/details/S15/README.md",
        ],
        "note": "三个冻结对象与每项实验的统一最低数据包（§7、§22）",
        "sections": {
            "identity": "source、tag、manifests、checksums、environment",
            "protocols": "frozen workload/operator/quality/statistics contracts",
            "raw": "immutable per-sample outputs and profiler originals",
            "normalized": "schema-validated tables derived from raw",
            "analysis": "scripts、queries、notebooks without hidden state",
            "figures": "generated figures plus metadata and source point IDs",
            "reports": "correctness、performance、limitation、acceptance",
            "release": "artifact inventory、SBOM、provenance、license reports",
            "reproduction": "CPU、accelerator and independent reviewer records",
            "communication": "demo/narrative/FAQ/claim exports",
        },
        "uniform_package": [
            "protocol.yaml",
            "release_candidate_ref.json",
            "environment.json",
            "participants_or_agents.json",
            "command_or_action_log.jsonl",
            "raw",
            "derived",
            "findings.json",
            "limitations.md",
            "acceptance.json",
            "evidence_manifest.yaml",
        ],
    },
    {
        "kind": "documentation-policy",
        "version": "v1",
        "sources": [
            "docs/stage_experiments/details/S15/E15-04_executable_docs_bilingual_consistency.md",
        ],
        "note": "code block 分类、命令验收状态与缺陷严重度/所有权矩阵；不含测量值",
        "classes": {
            "EXECUTE_SAFE": "CPU、只读、快速，CI 每次真实运行",
            "EXECUTE_ACCELERATOR": "需 GPU/NPU，在设备 CI/nightly 运行",
            "DRY_RUN": "会构建/部署/花费资源，验证参数和计划",
            "MANUAL_EXTERNAL": "需要账户、上游提交或云资源，校验结构并记录人工证据",
            "DISPLAY_ONLY": "伪代码/输出示例，必须明确标注",
        },
        "command_verdicts": [
            "PASS",
            "FAIL",
            "BLOCKED_DEVICE",
            "DRY_RUN_PASS",
            "MANUAL_VERIFIED",
            "DISPLAY_ONLY_VALID",
            "INVALID_UNCLASSIFIED",
        ],
        "matrix": {
            "SKIP": "不是最终状态；必须带责任原因、运行环境和失效日期",
            "support_cell": "declared ∩ source adapter ∩ test evidence ∩ runtime freshness ∩ release availability ∩ limitation policy",
        },
    },
    {
        "kind": "release-policy",
        "version": "v1",
        "sources": [
            "docs/stage_experiments/details/S15/E15-05_release_supply_chain_provenance.md",
        ],
        "note": "发布版本语义、资产种类、供应链 finding 状态与模型/数据边界；无测量值",
        "policy": {
            "versioning": "semver",
            "immutable_assets": True,
            "retraction_requires_new_version": True,
            "model_weights_distributed": False,
            "automated_publish": False,
            "download_bytes": UNMEASURED,
        },
        "asset_kinds": ["wheel", "sdist", "oci", "sample", "evidence", "docs"],
        "finding_states": [
            "OPEN",
            "FIXED_IN_NEW_CANDIDATE",
            "NOT_AFFECTED_WITH_EVIDENCE",
            "ACCEPTED_RISK_WITH_OWNER_AND_EXPIRY",
            "ASSET_WITHHELD",
            "RELEASE_BLOCKED",
        ],
    },
    {
        "kind": "figure-audit",
        "version": "v1",
        "sources": [
            "docs/stage_experiments/details/S15/E15-06_plot_raw_lineage_regeneration.md",
        ],
        "note": "抽样 strata、强制样本、重建差异等级与 lineage 预算；无测量值",
        "strata": [
            "stage",
            "evidence_level",
            "layer_micro_model_service",
            "hardware",
            "figure_type",
            "positive_negative_result",
            "error_bars",
            "public_channel",
        ],
        "mandatory_samples": [
            "hero_primary_claim",
            "maximum_value",
            "minimum_value",
            "direction_reversal",
            "cross_hardware_comparison",
            "pareto_point",
            "failure_or_missing_marker",
            "table_cell",
            "negative_result",
        ],
        "rebuild_classes": {
            "D0": "byte identical",
            "D1": "仅允许 metadata/压缩/抗锯齿差异",
            "D2": "浮点差在预注册 tolerance，事实/排序/CI 不变",
            "D3": "点值相同但 interval/n/label/axis 不同（FAIL）",
            "D4": "query/filter/member set 不同（FAIL）",
            "D5": "raw/identity/actual path 断链（FAIL）",
        },
        "access_budget": {
            "max_parse_seconds": 30.0,
            "max_clicks": 4.0,
            "max_download_mb": 50.0,
        },
    },
    {
        "kind": "demo-narrative",
        "version": "v1",
        "sources": [
            "docs/stage_experiments/details/S15/E15-07_demo_fault_fallback_rehearsal.md",
            "docs/stage_experiments/details/S15/E15-08_interview_narrative_adversarial_qa.md",
            "docs/stage_experiments/details/S15/E15-11_recruiter_first_impression_usability.md",
        ],
        "note": "demo 状态、故障矩阵、SLO 字段、三种时长内容门与评分锚点；无测量值",
        "states": [
            "LIVE_MEASUREMENT",
            "LIVE_REGENERATION",
            "CACHED_VERIFIED_RESULT",
            "PRERECORDED_VERIFIED_RUN",
            "SIMULATED_UI",
        ],
        "fault_scenarios": [
            "normal",
            "no_device",
            "busy_device",
            "cache_miss",
            "network_down",
            "model_missing",
            "timeout",
            "bad_asset",
            "web_unavailable",
            "evidence_link_down",
            "correctness_fail",
        ],
        "slo_fields": [
            "t_preflight",
            "t_live_action",
            "t_fault_detect",
            "t_fallback_complete",
            "t_normal_total",
            "t_fault_total",
            "t_evidence_lookup",
            "manual_actions_allowed",
            "cleanup_criterion",
        ],
        "duration_gates": {"3m": 180, "10m": 600, "30m": 1800},
        "hard_failures": [
            "fabricated_number",
            "speedup_despite_correctness_fail",
            "micro_as_model_service",
            "third_party_as_own_implementation",
            "hidden_counter_evidence",
            "contradictory_role_versions",
        ],
        "first_impression_participants": 3,
        "first_impression_minutes": 5,
    },
    {
        "kind": "reproduction-upstream",
        "version": "v1",
        "sources": [
            "docs/stage_experiments/details/S15/E15-09_independent_clean_room_reproduction.md",
            "docs/stage_experiments/details/S15/E15-10_upstream_contribution_quality.md",
        ],
        "note": "复现层级、help 分级、上游贡献质量维度与 disposition 表述；无测量值",
        "levels": {
            "R0": "获取、digest、provenance/SBOM",
            "R1": "clean install + CPU sample",
            "R2": "raw→normalized→figure",
            "R3": "accelerator correctness + actual path",
            "R4": "confirmatory effect + mechanism",
            "R5": "service/portable claim",
        },
        "help_levels": {
            "L0": "no author contact",
            "L1": "clarification already present in public docs",
            "L2": "author answers conceptual question without new command",
            "L3": "author supplies missing command/config/file",
            "L4": "author remotely operates or patches reviewer environment",
        },
        "quality_dimensions": [
            "authenticity",
            "boundary",
            "novelty",
            "reproducer",
            "diagnosis",
            "patch",
            "tests",
            "performance",
            "communication",
            "attribution",
            "downstream",
        ],
        "upstream_dispositions": [
            "OPEN",
            "UNDER_REVIEW",
            "ACCEPTED",
            "MERGED",
            "RELEASED",
            "DUPLICATE",
            "REJECTED",
            "WONTFIX",
            "STALLED",
        ],
    },
]


def render_documents() -> str:
    """Render all documents to their on-disk YAML text."""
    rendered: Dict[str, str] = {}
    for document in DOCUMENTS:
        filename = f"{document['kind']}.yaml"
        header = (
            "# GENERATED by scripts/release/gen_release_specs.py — do not edit by hand.\n"
            "# 冻结词汇表：不含任何测量值；实测结果一律落在 artifacts/S15/ 下。\n"
        )
        rendered[filename] = header + yaml.safe_dump(document, sort_keys=True, allow_unicode=True, width=100)
    return {name: rendered[name] for name in sorted(rendered)}


def write_documents() -> None:
    os.makedirs(SPEC_DIR, exist_ok=True)
    rendered = render_documents()
    for filename, text in rendered.items():
        with open(os.path.join(SPEC_DIR, filename), "w", encoding="utf-8") as handle:
            handle.write(text)


def check_documents(directory: str) -> tuple[bool, list[str]]:
    """Return ``(ok, drifted)``: re-render and compare with what is on disk."""
    rendered = render_documents()
    drifted: list[str] = []
    for filename, expected in sorted(rendered.items()):
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            drifted.append(f"missing {filename}")
            continue
        with open(path, encoding="utf-8") as handle:
            actual = handle.read()
        if actual != expected:
            drifted.append(f"drifted {filename}")
    return (not drifted), drifted


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--check" in args:
        ok, drifted = check_documents(SPEC_DIR)
        for item in drifted:
            print(f"[drift] {item}")
        print(f"spec check: ok={ok} drifted={len(drifted)}")
        return 0 if ok else 1
    write_documents()
    print(f"wrote {len(DOCUMENTS)} frozen vocabularies to {SPEC_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
