"""E15-09 — third-party clean-room reproduction and fix/retest.

Protocol: ``docs/stage_experiments/details/S15/E15-09_independent_clean_room_reproduction.md``
(45 steps).

Independence is a *variable*, not a badge.  The module makes that executable:

* :class:`ReproductionContract` — candidate, target level, hero claim,
  comparability, budget, help policy, guard band, frozen *before* the run (step 1);
* :class:`ReviewerIndependenceStatement` — excludes anyone who worked on
  S00–S15 and records the prior exposure / conflict (steps 2–3);
* :class:`ReceivedMaterials` — the only inputs the reviewer gets (release URL,
  digest, docs, manifest, model acquisition, issue channel), with a digest so a
  late extra script cannot be smuggled in (steps 5–6);
* :data:`HELP_LEVELS` and :data:`HELP_PASS_EFFECT` — L0–L4, where L3/L4
  invalidate the current candidate (§3.2, §9.1);
* :data:`REPRODUCTION_LEVELS` and :data:`LEVEL_CLAIM_MATRIX` — R0–R5 with the
  *most a project may say* per level (§9);
* :class:`HelpLog`, :class:`Finding`, :class:`IssueRecord` — the help trail, the
  immutable finding and the structured issue the reviewer must file before
  contacting the author (steps 27–31);
* :func:`classify_finding`, :func:`severity_of`, :func:`protect_original_failure`,
  :func:`minimal_fix_check`, :func:`new_candidate_required`,
  :func:`retest_scope`, :func:`regression_scope` — the fix/retest loop
  (steps 30–37);
* :func:`reproduction_metrics`, :func:`reviewer_variation`,
  :func:`author_fact_response`, :func:`independence_audit` — steps 40–44;
* :func:`freeze_reproduction` — step 45.

Nothing here runs a clean-room session or contacts a reviewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release import records as rec
from hqsb.release.identity import SCHEMA_PREFIX, canonical_digest, is_digest

#: Help levels (§3.2).
HELP_LEVELS: Tuple[str, ...] = rec.HELP_LEVELS

#: Reproduction levels R0–R5 (§3.3).
REPRODUCTION_LEVELS: Tuple[str, ...] = rec.REPRODUCTION_LEVELS

#: Help level → effect on the current candidate (§9.1).
HELP_PASS_EFFECT: Tuple[Mapping[str, str], ...] = (
    {"level": "L0", "current_candidate": "可直接判定", "fix_required": "no"},
    {"level": "L1", "current_candidate": "可判定，记录 docs 路径", "fix_required": "no"},
    {"level": "L2", "current_candidate": "条件判定，检查文档歧义", "fix_required": "if ambiguity"},
    {"level": "L3", "current_candidate": "FAIL", "fix_required": "yes"},
    {"level": "L4", "current_candidate": "不能作独立 PASS", "fix_required": "new clean session"},
)

#: Level → the most the project may say, and what it may never say (§9).
LEVEL_CLAIM_MATRIX: Tuple[Mapping[str, str], ...] = (
    {"level": "R0", "may_say": "发布资产可独立验证", "may_not_say": "结果已复现"},
    {"level": "R1", "may_say": "最低入口由第三方跑通", "may_not_say": "GPU/NPU 性能复现"},
    {"level": "R2", "may_say": "选定图表由第三方重建", "may_not_say": "测量本身在新硬件重做"},
    {"level": "R3", "may_say": "实现路径/语义在该设备复现", "may_not_say": "性能收益复现"},
    {"level": "R4", "may_say": "hero 主要结果在条件 C 下复现", "may_not_say": "未覆盖 service/portable"},
    {"level": "R5", "may_say": "服务级/第二硬件结果在该场景复现", "may_not_say": "所有设备可迁移"},
)

#: Finding categories (step 30).
FINDING_CATEGORIES: Tuple[str, ...] = (
    "docs",
    "packaging",
    "schema",
    "artifact",
    "identity",
    "correctness",
    "performance",
    "environment",
    "license_security",
    "infrastructure",
)

#: Finding severities (step 31).
FINDING_SEVERITIES: Tuple[str, ...] = ("P0", "P1", "P2")

#: Received-materials set the reviewer may receive (step 5).
RECEIVED_MATERIAL_KINDS: Tuple[str, ...] = (
    "release_url",
    "release_digest",
    "readme",
    "docs",
    "artifact_manifest",
    "model_acquisition",
    "issue_channel",
)

#: What must NOT be handed to the reviewer (step 6).
FORBIDDEN_MATERIALS: Tuple[str, ...] = (
    "internal_s15_design",
    "author_troubleshooting_notes",
    "author_shell_history",
    "unpublished_fix",
    "author_local_image",
    "author_cache",
)


@dataclass
class ReproductionContract:
    """The frozen contract (step 1)."""

    campaign_id: str
    candidate_id: str
    target_level: str
    hero_claim_id: str = ""
    comparability: str = ""
    time_budget_s: float = 0.0
    resource_budget: str = ""
    help_policy: str = ""
    guard_band: Optional[float] = None
    pass_rule: str = ""

    schema_version = f"{SCHEMA_PREFIX}.reproduction-contract.v1"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.campaign_id or not self.candidate_id:
            findings.append("ReproductionContract: campaign_id and candidate_id are required")
        if self.target_level not in REPRODUCTION_LEVELS:
            findings.append(f"ReproductionContract: unknown target level {self.target_level!r}")
        if not self.hero_claim_id:
            findings.append("ReproductionContract: the hero claim must be frozen before the reviewer starts")
        if self.comparability not in rec.COMPARABILITY_CLASSES:
            findings.append(f"ReproductionContract: unknown comparability {self.comparability!r}")
        if self.time_budget_s <= 0:
            findings.append("ReproductionContract: the time budget must be declared")
        if not self.help_policy:
            findings.append(
                "ReproductionContract: the help policy (L0–L4 handling) must be frozen before any help is given"
            )
        if self.guard_band is None or self.guard_band <= 0:
            findings.append("ReproductionContract: the guard band must be pre-registered")
        if not self.pass_rule:
            findings.append("ReproductionContract: the PASS/FAIL rule must be written down")
        return findings


@dataclass
class ReviewerIndependenceStatement:
    """Independence criteria (steps 2–3)."""

    reviewer_id: str
    worked_on_project: bool
    prior_exposure: str = ""
    conflict_of_interest: str = ""
    baseline_skill: str = ""
    uses_independent_agent: bool = False
    agent_context_isolation: bool = False
    signed_privacy_agreement: bool = False

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.reviewer_id:
            findings.append("ReviewerIndependenceStatement: reviewer_id is required")
        if self.worked_on_project:
            findings.append(
                "ReviewerIndependenceStatement: someone who worked on S00–S15 is not independent "
                "(E15-09 step 2)"
            )
        if not self.prior_exposure:
            findings.append("ReviewerIndependenceStatement: prior exposure must be recorded (a blank is a finding)")
        if not self.baseline_skill:
            findings.append("ReviewerIndependenceStatement: the reviewer's baseline skill must be recorded")
        if self.uses_independent_agent and not self.agent_context_isolation:
            findings.append(
                "ReviewerIndependenceStatement: an agent used as reviewer needs an isolated context that only "
                "receives the public bundle"
            )
        if not self.signed_privacy_agreement:
            findings.append("ReviewerIndependenceStatement: the scope/privacy agreement must be signed (step 4)")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "worked_on_project": self.worked_on_project,
            "prior_exposure": self.prior_exposure,
            "conflict_of_interest": self.conflict_of_interest,
            "baseline_skill": self.baseline_skill,
            "uses_independent_agent": self.uses_independent_agent,
            "agent_context_isolation": self.agent_context_isolation,
            "signed_privacy_agreement": self.signed_privacy_agreement,
        }


@dataclass
class ReceivedMaterials:
    """The only inputs the reviewer receives (steps 5–6)."""

    materials: Mapping[str, str] = field(default_factory=dict)
    excluded: Tuple[str, ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        for kind in RECEIVED_MATERIAL_KINDS:
            if kind not in self.materials:
                findings.append(f"ReceivedMaterials: {kind!r} is not provided (the reviewer should not have to ask for the entry)")
        for kind, value in sorted(self.materials.items()):
            if kind == "release_digest" and value and not is_digest(value):
                findings.append("ReceivedMaterials: release_digest must be sha256:<hex>")
            if kind == "release_url" and ("localhost" in value or "file://" in value):
                findings.append("ReceivedMaterials: the release URL may not point at the author's machine")
        for kind in FORBIDDEN_MATERIALS:
            if kind in self.materials:
                findings.append(
                    f"ReceivedMaterials: {kind!r} must not be handed to the reviewer (information wall, step 6)"
                )
        return findings

    @property
    def digest(self) -> str:
        return canonical_digest({key: self.materials[key] for key in sorted(self.materials)})


@dataclass
class HelpEvent:
    """One help request and response, all recorded (step 27)."""

    level: str
    question: str
    response: str = ""
    recorded_at_utc: str = ""
    via_public_channel: bool = False

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.level not in HELP_LEVELS:
            findings.append(f"HelpEvent: unknown help level {self.level!r}")
        if not self.question:
            findings.append("HelpEvent: the question must be recorded (a verbal hint is still a hint)")
        if self.level in ("L3", "L4") and not self.response:
            findings.append("HelpEvent: an L3/L4 response must be logged so its effect is auditable")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "question": self.question,
            "response": self.response,
            "recorded_at_utc": self.recorded_at_utc,
            "via_public_channel": self.via_public_channel,
        }


class HelpLog:
    """The full help trail with the pass effect (steps 27–29)."""

    def __init__(self) -> None:
        self.events: List[HelpEvent] = []

    def add(self, event: HelpEvent) -> HelpEvent:
        problems = event.problems()
        if problems:
            raise ConfigError("invalid help event: " + "; ".join(problems))
        self.events.append(event)
        return event

    def highest_level(self) -> str:
        order = {level: index for index, level in enumerate(HELP_LEVELS)}
        return max((event.level for event in self.events), key=lambda level: order[level], default="L0")

    def blocks_pass(self) -> bool:
        return self.highest_level() in ("L3", "L4")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "events": [event.as_dict() for event in self.events],
            "highest_level": self.highest_level(),
            "blocks_pass": self.blocks_pass(),
        }


@dataclass
class Finding:
    """An immutable reproduction finding (steps 28–31)."""

    finding_id: str
    category: str = ""
    severity: str = ""
    affected_claims: Tuple[str, ...] = ()
    affected_levels: Tuple[str, ...] = ()
    summary: str = ""
    issue_url: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.finding_id:
            findings.append("Finding: finding_id is required")
        if self.category not in FINDING_CATEGORIES:
            findings.append(f"Finding: unknown category {self.category!r}")
        if self.severity not in FINDING_SEVERITIES:
            findings.append(f"Finding: unknown severity {self.severity!r}")
        if not self.summary:
            findings.append("Finding: a finding needs a summary")
        if self.severity == "P0" and not self.affected_claims:
            findings.append("Finding: a P0 finding must name the affected claim(s)")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "category": self.category,
            "severity": self.severity,
            "affected_claims": list(self.affected_claims),
            "affected_levels": list(self.affected_levels),
            "summary": self.summary,
            "issue_url": self.issue_url,
        }


def classify_finding(*, category: str) -> str:
    """Map a raw finding to a category; 'environment' is not a catch-all (step 30)."""
    if category not in FINDING_CATEGORIES:
        raise ConfigError(f"unknown finding category {category!r}")
    return category


def severity_of(*, blocks_conclusion: bool, major_friction: bool) -> str:
    """P0 = wrong conclusion/security; P1 = major friction; P2 = minor usability (step 31)."""
    if blocks_conclusion:
        return "P0"
    if major_friction:
        return "P1"
    return "P2"


def protect_original_failure(*, snapshot_digest: str, overwritten: bool) -> List[str]:
    """The original failure must survive the fix (step 32)."""
    findings: List[str] = []
    if not snapshot_digest or not is_digest(snapshot_digest):
        findings.append("the original failure snapshot has no digest (it cannot be proven unchanged)")
    if overwritten:
        findings.append("the original failure was overwritten; the fix erased the proof that the problem existed")
    return findings


def minimal_fix_check(*, layer: str, changes: Sequence[str], regression_tests_added: bool) -> List[str]:
    """A fix goes to the owning layer, with a regression test (step 33)."""
    findings: List[str] = []
    if layer not in FINDING_CATEGORIES:
        findings.append(f"fix was not routed to a known owning layer, got {layer!r}")
    if not changes:
        findings.append("the fix is empty")
    if not regression_tests_added:
        findings.append("the fix has no regression test (a fix without a test only fixes this session)")
    return findings


def new_candidate_required(*, previous_candidate_id: str, new_candidate_id: str, gates_rerun: Sequence[str]) -> List[str]:
    """A P0 fix produces a new candidate with gates rerun (step 34)."""
    findings: List[str] = []
    if previous_candidate_id == new_candidate_id:
        findings.append("the fix reused the candidate id; a frozen candidate must not be overwritten (step 34)")
    required_gates = ("E15-01", "E15-04", "E15-05")
    missing = [gate for gate in required_gates if gate not in gates_rerun]
    if missing:
        findings.append(f"the prerequisite gates were not rerun for the new candidate: {', '.join(missing)}")
    return findings


def retest_scope(*, original_reviewer: str, retest_reviewer: str, from_failure_point: bool) -> List[str]:
    """The *same* reviewer retests the *same* problem (step 35)."""
    findings: List[str] = []
    if original_reviewer != retest_reviewer:
        findings.append("the fix must be retested by the original reviewer, not a new one who may bypass the original blocker")
    if not from_failure_point:
        findings.append("the retest must start from the original failure point")
    return findings


def regression_scope(*, changed_claims: Sequence[str], other_claims: Sequence[str], rerun_full_hero: bool) -> List[str]:
    """A P0 fix forces a full hero confirmation, not only the unit test (steps 36–37)."""
    findings: List[str] = []
    overlap = sorted(set(changed_claims) & set(other_claims))
    if overlap:
        findings.append(f"claims changed and unchanged at the same time: {', '.join(overlap)}")
    if changed_claims and not rerun_full_hero:
        findings.append("a P0 fix requires at least one full clean hero confirmation (step 37)")
    return findings


@dataclass
class ReproductionMetrics:
    """R0–R5 level completion, times, help and consistency (step 40)."""

    achieved_level: Optional[str] = None
    stage_times_s: Mapping[str, float] = field(default_factory=dict)
    help_levels: Mapping[str, int] = field(default_factory=dict)
    effect_consistency: str = rec.STATUS_NOT_RUN
    mechanism_consistency: str = rec.STATUS_NOT_RUN
    correctness_status: str = rec.STATUS_NOT_RUN
    actual_path_status: str = rec.STATUS_NOT_RUN

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.achieved_level and self.achieved_level not in REPRODUCTION_LEVELS:
            findings.append(f"ReproductionMetrics: unknown achieved level {self.achieved_level!r}")
        for level in self.help_levels:
            if level not in HELP_LEVELS:
                findings.append(f"ReproductionMetrics: unknown help level {level!r}")
        if self.achieved_level in ("R3", "R4", "R5"):
            if self.correctness_status != rec.STATUS_PASS or self.actual_path_status != rec.STATUS_PASS:
                findings.append("ReproductionMetrics: R3+ cannot be claimed without correctness and actual path")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "achieved_level": self.achieved_level,
            "stage_times_s": {key: self.stage_times_s[key] for key in sorted(self.stage_times_s)},
            "help_levels": {key: int(self.help_levels[key]) for key in sorted(self.help_levels)},
            "effect_consistency": self.effect_consistency,
            "mechanism_consistency": self.mechanism_consistency,
            "correctness_status": self.correctness_status,
            "actual_path_status": self.actual_path_status,
        }


def reproduction_metrics(record: rec.CleanRoomReproductionRecord) -> Dict[str, Any]:
    """Summarise the achieved level and consistency (step 40)."""
    problems = record.validate()
    return {
        "achieved_level": record.achieved_level,
        "target_level": record.target_level,
        "help_levels": {key: record.help_events_by_level[key] for key in sorted(record.help_events_by_level)},
        "effect_consistency": record.effect_consistency,
        "mechanism_consistency": record.mechanism_consistency,
        "problems": problems,
        "note": "一个 campaign 可能达到 R4 但 R2 某个非主图失败；层级与 finding 必须同时列出",
    }


def reviewer_variation(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Two reviewers are two sessions, not a distribution (step 41)."""
    if len(records) < 2:
        raise ConfigError("reviewer_variation needs at least two reviewer sessions")
    return {
        "reviewer_sessions": len(records),
        "per_reviewer": [{key: item[key] for key in ("reviewer_id", "achieved_level") if key in item} for item in records],
        "note": "两个 reviewer session 不能当大量样本；分别报告，不合并成总体比例",
    }


def author_fact_response(*, original_judgement: str, revised_judgement: str, changed_by_author: bool) -> List[str]:
    """The author may correct facts but never rewrite the reviewer's judgement (step 39)."""
    findings: List[str] = []
    if changed_by_author and revised_judgement != original_judgement:
        findings.append("the author changed the reviewer's judgement; only provable factual errors may be corrected")
    if not original_judgement or not revised_judgement:
        findings.append("both the original and the revised judgement must be kept")
    return findings


def independence_audit(
    *,
    received_digest: str,
    help_log: HelpLog,
    reviewer: ReviewerIndependenceStatement,
    report_written_by_reviewer: bool,
) -> Dict[str, Any]:
    """A second pair of eyes checks whether 'independent' is true (step 44)."""
    findings = list(reviewer.problems())
    if reviewer.worked_on_project:
        findings.append("independence audit: the reviewer worked on the project")
    if help_log.blocks_pass():
        findings.append(f"independence audit: highest help level {help_log.highest_level()} invalidates independence")
    if not report_written_by_reviewer:
        findings.append("independence audit: the report was not written by the reviewer")
    return {
        "received_materials_digest": received_digest,
        "reviewer": reviewer.as_dict(),
        "help_log": help_log.as_dict(),
        "findings": findings,
        "independent": not findings,
    }


def freeze_reproduction(
    *, contract: ReproductionContract, record: rec.CleanRoomReproductionRecord, help_log: HelpLog
) -> Dict[str, Any]:
    """Freeze the reproduction record (step 45)."""
    problems = list(contract.problems()) + list(record.validate())
    if record.status == rec.STATUS_PASS and help_log.blocks_pass():
        problems.append("a PASS with L3/L4 help is not independent (§9.1)")
    if problems:
        raise ConfigError("refusing to freeze the reproduction: " + "; ".join(problems[:5]))
    return {
        "schema_version": f"{SCHEMA_PREFIX}.clean-room-reproduction.v1",
        "contract": contract.campaign_id,
        "record": record.as_dict(),
        "help_log": help_log.as_dict(),
        "record_digest": canonical_digest(record.as_dict()),
    }


# ── PROTOCOL_STEPS (45) ─────────────────────────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 ReproductionContract", ("clean_room.ReproductionContract",)),
    (2, "定义 reviewer 独立性标准", ("clean_room.ReviewerIndependenceStatement",)),
    (3, "选择 reviewer", ("clean_room.ReviewerIndependenceStatement", "clean_room.reviewer_variation")),
    (4, "签署范围和隐私约定", ("clean_room.ReviewerIndependenceStatement", "records.CleanRoomReproductionRecord")),
    (5, "冻结公开材料集合", ("clean_room.ReceivedMaterials", "clean_room.RECEIVED_MATERIAL_KINDS")),
    (6, "建立信息隔离墙", ("clean_room.ReceivedMaterials.problems", "clean_room.FORBIDDEN_MATERIALS")),
    (7, "选择 clean 环境", ("quickstart.SessionIdentity", "quickstart.prove_no_hqsb_residue")),
    (8, "采集 reviewer 环境指纹", ("hero_replay.EnvironmentFingerprint", "experiment.environment_fingerprint")),
    (9, "设置独立记录机制", ("telemetry.ActionLog", "clean_room.HelpLog")),
    (10, "开始盲时钟", ("quickstart.TotalClock", "telemetry.TimeToEventSummary")),
    (11, "让 reviewer 复述项目和 claim", ("clean_room.ReceivedMaterials", "records.CleanRoomReproductionRecord")),
    (12, "获取并验证 release", ("supply_chain.verify_provenance", "supply_chain.ArtifactManifest")),
    (13, "执行 CPU quickstart", ("quickstart.QuickstartContract", "quickstart.freeze_session")),
    (14, "执行 evidence bundle inventory", ("contracts.PublicEvidenceBundle", "identity.content_address_aggregate")),
    (15, "执行抽定图表重建", ("figures.draw_sample", "figures.rebuild_chain")),
    (16, "获取模型和运行依赖", ("hero_replay.ArtifactIdentity", "supply_chain.RightsDecision")),
    (17, "执行 accelerator preflight", ("hero_replay.CapabilityPreflight", "hero_replay.DeviceHealth")),
    (18, "验证 ModelArtifact/WorkloadSpec", ("hero_replay.ArtifactIdentity", "hero_replay.check_workload_tokens")),
    (19, "运行 reference correctness", ("hero_replay.evaluate_correctness_matrix",)),
    (20, "运行 candidate correctness", ("hero_replay.evaluate_correctness_matrix",)),
    (21, "验证 actual path", ("hero_replay.ActualPathEvidence",)),
    (22, "执行冻结性能 protocol", ("hero_replay.ConfirmatoryPlan", "hero_replay.BlockedEstimate")),
    (23, "采集同轮 profile", ("hero_replay.profile_reconciliation_rows", "figures.LineageLayer")),
    (24, "计算独立统计结果", ("hero_replay.BlockedEstimate.effect", "hero_replay.AmdahlModel")),
    (25, "比较 reproduction guard band", ("hero_replay.compare_with_release", "hero_replay.replay_verdict")),
    (26, "审计 Claim Ledger 样本", ("claims.ClaimLedger", "claims.detect_orphans")),
    (27, "记录所有帮助请求", ("clean_room.HelpLog", "clean_room.HELP_LEVELS")),
    (28, "要求先提交结构化 issue", ("clean_room.Finding", "clean_room.classify_finding")),
    (29, "作者按公开信息响应", ("clean_room.HelpLog", "clean_room.HELP_PASS_EFFECT")),
    (30, "给 finding 分类", ("clean_room.classify_finding", "clean_room.FINDING_CATEGORIES")),
    (31, "给 finding 定严重度", ("clean_room.severity_of", "clean_room.FINDING_SEVERITIES")),
    (32, "保护原失败证据", ("clean_room.protect_original_failure",)),
    (33, "作者实施最小修复", ("clean_room.minimal_fix_check", "docs_gate.classify_defect")),
    (34, "生成新 Release Candidate", ("clean_room.new_candidate_required", "contracts.new_candidate")),
    (35, "由原 reviewer 复测原问题", ("clean_room.retest_scope",)),
    (36, "验证无回归", ("clean_room.regression_scope",)),
    (37, "重复完整 hero confirmation", ("clean_room.regression_scope", "hero_replay.freeze_replay")),
    (38, "进行 reviewer 独立结论撰写", ("clean_room.independence_audit", "records.CleanRoomReproductionRecord.report_uri")),
    (39, "作者事实核对但不改判断", ("clean_room.author_fact_response",)),
    (40, "计算复现层级和指标", ("clean_room.reproduction_metrics", "clean_room.LEVEL_CLAIM_MATRIX")),
    (41, "评估 reviewer 间变异", ("clean_room.reviewer_variation",)),
    (42, "更新文档和 Claim Ledger", ("clean_room.LEVEL_CLAIM_MATRIX", "claims.revise_claim")),
    (43, "发布复现报告和 issue 链", ("clean_room.Finding", "docs_gate.check_release_links")),
    (44, "执行独立审计复核", ("clean_room.independence_audit",)),
    (45, "冻结 CleanRoomReproductionRecord", ("clean_room.freeze_reproduction", "records.CleanRoomReproductionRecord")),
)

TITLE = "第三方 Clean-room Reproduction 与修复复测"
LEVEL = "P0"
CLAIM_BOUNDARY = "E15-09 代表至少一次受控、独立的主要结果复现；不代表所有用户、所有设备或未来版本都能自动复现"


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the clean-room interfaces (labelled smoke)."""
    problems: List[str] = []
    contract = ReproductionContract(
        campaign_id="c-1",
        candidate_id="cand-1",
        target_level="R4",
        hero_claim_id="CLM-hero",
        comparability="compatible",
        time_budget_s=7200.0,
        resource_budget="one reviewer + one clean host",
        help_policy="L0–L2 only for a PASS",
        guard_band=0.10,
        pass_rule="no L3/L4 help + correctness/actual-path PASS + effect direction consistent",
    )
    problems.extend(contract.problems())
    reviewer = ReviewerIndependenceStatement(
        reviewer_id="rev-1",
        worked_on_project=False,
        prior_exposure="none",
        conflict_of_interest="none",
        baseline_skill="python + linux + accelerator",
        signed_privacy_agreement=True,
    )
    problems.extend(reviewer.problems())
    materials = ReceivedMaterials(
        materials={
            "release_url": "https://github.com/example/hqsb/releases/tag/v0.1.0",
            "release_digest": "sha256:" + "0" * 64,
            "readme": "README.md",
            "docs": "docs/",
            "artifact_manifest": "manifest.json",
            "model_acquisition": "per-license fetch",
            "issue_channel": "issues",
        }
    )
    problems.extend(materials.problems())
    if "author_troubleshooting_notes" in materials.materials:
        problems.append("forbidden material leaked into the received set")
    log = HelpLog()
    log.add(HelpEvent(level="L3", question="how do I install", response="run this private command"))
    if not log.blocks_pass():
        problems.append("L3 help did not block the pass")
    if new_candidate_required(previous_candidate_id="cand-1", new_candidate_id="cand-1", gates_rerun=("E15-01",)):
        pass
    else:
        problems.append("reusing a candidate id was accepted")
    metrics = ReproductionMetrics(achieved_level="R4", correctness_status=rec.STATUS_PASS, actual_path_status=rec.STATUS_PASS)
    if metrics.problems():
        problems.append("a valid R4 metrics record was rejected")
    bad = ReproductionMetrics(achieved_level="R4", correctness_status=rec.STATUS_FAIL, actual_path_status=rec.STATUS_PASS)
    if not bad.problems():
        problems.append("R4 was claimed without correctness")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "help_levels": len(HELP_LEVELS),
        "reproduction_levels": len(REPRODUCTION_LEVELS),
        "finding_categories": len(FINDING_CATEGORIES),
        "problems": problems,
    }
