"""E15-10 — real upstream issue/PR/documentation contribution.

Protocol: ``docs/stage_experiments/details/S15/E15-10_upstream_contribution_quality.md``
(45 steps).

A contribution is not a link on a resume.  It is a compressed, reviewable object
that a maintainer can act on, with a real HQSB finding behind it.  The module
implements the whole pipeline as data and checks:

* :class:`ContributionStudyContract` — the frozen quality rubric and the rule
  that "merged" is not the only pass criterion (step 1);
* :class:`CandidateFinding` + :func:`boundary_triage` — upstream / hqsb / usage /
  known_limitation, because a local bug pushed upstream is a failure (§2, step 4);
* :class:`MinimalReproducer` — the causal ablation that keeps the trigger while
  dropping everything else, with negative controls and a recorded failure rate
  (steps 12–16);
* :class:`IssueDraft` — version, environment, minimal code, expected/actual,
  impact, workaround, evidence (step 19);
* :class:`PatchDesign`, :func:`check_patch`, :func:`check_regression_test`,
  :func:`check_performance_pr` — root cause, invariant, tests, and the "no
  fastest-single-value" rule (steps 26–31);
* :class:`ReviewRound` — the design history a collaborator can later narrate
  (steps 34–36);
* :data:`UPSTREAM_DISPOSITIONS` and :data:`DISPOSITION_CLAIMS` — what each final
  state lets the project say (§9.1);
* :func:`quality_score` — the §9 eleven-dimension table, with the hard failures
  that no score can rescue;
* :func:`attribution_audit`, :func:`downstream_actions`,
  :func:`freeze_contribution` — steps 40–45.

Nothing here submits an issue, opens a PR or contacts a maintainer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release import records as rec
from hqsb.release.identity import SCHEMA_PREFIX, canonical_digest

#: Boundary triage outcomes (step 4).
BOUNDARY_TRIAGE: Tuple[str, ...] = rec.BOUNDARY_TRIAGE

#: Contribution types (step 18).
CONTRIBUTION_TYPES: Tuple[str, ...] = rec.UpstreamContributionRecord.CONTRIBUTION_TYPES

#: Upstream dispositions (§9.1).
UPSTREAM_DISPOSITIONS: Tuple[str, ...] = rec.UPSTREAM_STATUSES

#: Disposition → what may be said and what may not (§9.1).
DISPOSITION_CLAIMS: Tuple[Mapping[str, str], ...] = (
    {"status": "OPEN", "may_say": "已提交并在审查", "may_not_say": "上游已接受"},
    {"status": "UNDER_REVIEW", "may_say": "已提交并在审查", "may_not_say": "上游已接受"},
    {"status": "ACCEPTED", "may_say": "设计方向获认可", "may_not_say": "未合并时说已进入代码"},
    {"status": "MERGED", "may_say": "已合并 commit X", "may_not_say": "未发布时说用户版本已支持"},
    {"status": "RELEASED", "may_say": "已在版本 V 发布", "may_not_say": "外推其他版本"},
    {"status": "DUPLICATE", "may_say": "补充/关联已有问题", "may_not_say": "我独立发现首个 bug"},
    {"status": "REJECTED", "may_say": "提交并依据理由调整方案", "may_not_say": "隐藏拒绝或说合并"},
    {"status": "WONTFIX", "may_say": "提交并依据理由调整方案", "may_not_say": "说合并"},
    {"status": "STALLED", "may_say": "当前无维护者结论", "may_not_say": "暗示接受或拒绝"},
)

#: The §9 quality dimensions.
QUALITY_DIMENSIONS: Tuple[str, ...] = (
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
)

#: Hard failures of §9 — no dimension score can rescue these.
QUALITY_HARD_FAILURES: Tuple[str, ...] = (
    "fabricated_problem",
    "local_misuse_pushed_upstream",
    "obvious_duplicate_no_new_info",
    "private_model_or_credentials_in_reproducer",
    "regression_test_missing_for_code_pr",
    "fastest_single_value_as_performance",
    "unexplained_agent_patch",
    "attribution_misrepresented",
)

#: What a code/performance PR must additionally satisfy (§9).
PR_REQUIRED_DIMENSIONS: Tuple[str, ...] = ("patch", "tests", "communication", "attribution")
PERFORMANCE_REQUIRED_DIMENSIONS: Tuple[str, ...] = ("reproducer", "diagnosis", "performance", "downstream")


@dataclass
class ContributionStudyContract:
    """The frozen contract (step 1)."""

    contribution_id: str
    candidate_id: str = ""
    target_upstreams: Tuple[str, ...] = ()
    quality_rubric: Mapping[str, str] = field(default_factory=dict)
    budget: str = ""
    success_statuses: Tuple[str, ...] = ()
    prohibit_metrics: Tuple[str, ...] = ()

    schema_version = f"{SCHEMA_PREFIX}.contribution-contract.v1"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.contribution_id or not self.candidate_id:
            findings.append("ContributionStudyContract: contribution_id and candidate_id are required")
        if not self.target_upstreams:
            findings.append("ContributionStudyContract: the candidate target upstreams must be declared")
        for upstream in self.target_upstreams:
            if upstream not in ("pytorch", "triton", "vllm", "sglang", "cann", "transformers", "other"):
                findings.append(f"ContributionStudyContract: unknown target upstream {upstream!r}")
        if not self.budget:
            findings.append("ContributionStudyContract: the budget must be declared")
        if not self.success_statuses:
            findings.append("ContributionStudyContract: the success statuses must be frozen (merged is not the only one)")
        if "metric_gaming" not in self.prohibit_metrics and "刷量" not in self.prohibit_metrics:
            findings.append("ContributionStudyContract: the no-metric-gaming rule must be declared")
        return findings


@dataclass
class CandidateFinding:
    """A real HQSB finding that could become an upstream contribution (step 2–3)."""

    finding_id: str
    run_id: str = ""
    environment: str = ""
    summary: str = ""
    category: str = ""
    observed_frequency: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("finding_id", "run_id", "summary", "category"):
            if not getattr(self, name):
                findings.append(f"CandidateFinding: {name} is required (a fabricated problem has no run id)")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "run_id": self.run_id,
            "environment": self.environment,
            "summary": self.summary,
            "category": self.category,
            "observed_frequency": self.observed_frequency,
        }


def boundary_triage(
    *, error_source: str, evidence: Sequence[str] = ()
) -> str:
    """upstream / hqsb / usage / known_limitation (step 4)."""
    if error_source not in BOUNDARY_TRIAGE:
        raise ConfigError(f"unknown boundary triage {error_source!r}")
    if error_source == "upstream" and not evidence:
        raise ConfigError(
            "a finding attributed to upstream needs evidence; without it the maintainer is being asked "
            "to debug the project (§9: 本地误用推上游是硬失败)"
        )
    return error_source


@dataclass
class MinimalReproducer:
    """The minimal causal ablation (steps 12–16)."""

    reproducer_id: str
    steps: Tuple[str, ...] = ()
    dependencies: Tuple[str, ...] = ()
    negative_controls: Tuple[str, ...] = ()
    failure_rate: Optional[float] = None
    seed_concurrency_timing: str = ""
    contains_private_state: bool = False
    command_count: int = 0

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.steps:
            findings.append("MinimalReproducer: no steps (a verbal reproducer is not a reproducer)")
        if self.command_count == 0 and self.command_count > 20:
            findings.append("MinimalReproducer: the command count must be declared")
        if not self.negative_controls:
            findings.append("MinimalReproducer: a reproducer without a negative control cannot locate the boundary")
        if self.failure_rate is not None:
            if not 0.0 <= self.failure_rate <= 1.0:
                findings.append("MinimalReproducer: failure_rate must be in [0, 1]")
            if self.failure_rate < 1.0 and not self.seed_concurrency_timing:
                findings.append("MinimalReproducer: a non-deterministic reproducer needs seed/concurrency/timing")
        if self.contains_private_state:
            findings.append(
                "MinimalReproducer: the reproducer depends on HQSB private state/models/credentials "
                "(a maintainer cannot run it)"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reproducer_id": self.reproducer_id,
            "steps": list(self.steps),
            "dependencies": list(self.dependencies),
            "negative_controls": list(self.negative_controls),
            "failure_rate": self.failure_rate,
            "seed_concurrency_timing": self.seed_concurrency_timing,
            "contains_private_state": self.contains_private_state,
            "command_count": self.command_count,
        }


@dataclass
class IssueDraft:
    """An actionable issue text (step 19)."""

    title: str = ""
    component: str = ""
    version: str = ""
    environment: str = ""
    minimal_code: str = ""
    expected: str = ""
    actual: str = ""
    impact: str = ""
    workaround: str = ""
    evidence: Tuple[str, ...] = ()
    performance_claim: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("title", "version", "minimal_code", "expected", "actual", "impact"):
            if not getattr(self, name):
                findings.append(f"IssueDraft: {name} is required (an issue with no expected/actual is noise)")
        if not self.component or not self.title:
            findings.append("IssueDraft: a title naming the component is required")
        if self.performance_claim and not self.evidence:
            findings.append("IssueDraft: a performance claim without evidence is a marketing link, not an issue")
        return findings


def check_performance_wording(draft: IssueDraft) -> List[str]:
    """Performance phrasing must carry scope/method/n/CI (§9, step 20)."""
    findings: List[str] = []
    if not draft.performance_claim:
        return findings
    required_tokens = ("baseline", "hardware", "shape", "dtype", "method", "n", "ci")
    missing = [token for token in required_tokens if token not in draft.performance_claim.lower()]
    if missing:
        findings.append(f"performance wording is missing: {', '.join(missing)}")
    if re.search(r"single (?:best|run)", draft.performance_claim.lower()):
        findings.append("a single best value may not stand in for a performance claim")
    return findings


@dataclass
class PatchDesign:
    """Root cause, invariant, compatibility, alternatives (step 26)."""

    root_cause: str = ""
    invariant: str = ""
    compatibility: str = ""
    alternatives: Tuple[str, ...] = ()
    focused: bool = True
    includes_unrelated_refactor: bool = False

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.root_cause or not self.invariant:
            findings.append("PatchDesign: root cause and invariant are required (no special-case patch)")
        if not self.focused:
            findings.append("PatchDesign: the patch must be focused; a special case for one shape breaks the design")
        if self.includes_unrelated_refactor:
            findings.append("PatchDesign: an unrelated refactor must not be bundled into a bug-fix PR")
        if not self.alternatives:
            findings.append("PatchDesign: the rejected alternatives should be recorded")
        return findings


@dataclass
class ReviewRound:
    """One review round with the design change it drove (steps 34–36)."""

    round_id: str
    concern: str = ""
    adopted: str = ""
    code_or_test_change: str = ""
    new_evidence: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.concern:
            findings.append("ReviewRound: the reviewer concern must be recorded")
        if not self.adopted:
            findings.append("ReviewRound: the adopt/reject decision must be recorded")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "concern": self.concern,
            "adopted": self.adopted,
            "code_or_test_change": self.code_or_test_change,
            "new_evidence": self.new_evidence,
        }


def check_patch(design: PatchDesign, *, regression_test: bool, compatibility: bool) -> List[str]:
    """The patch gate: root cause, regression test, compatibility (steps 26–29)."""
    findings = list(design.problems())
    if not regression_test:
        findings.append("check_patch: a code PR needs a regression test that fails before and passes after")
    if not compatibility:
        findings.append("check_patch: API/ABI/backward compatibility was not analysed (step 31)")
    return findings


def quality_score(
    *,
    dimensions: Mapping[str, bool],
    hard_failures: Sequence[str],
    contribution_type: str,
) -> Dict[str, Any]:
    """The §9 quality gate: dimensions plus hard failures (step 43)."""
    if contribution_type not in CONTRIBUTION_TYPES:
        raise ConfigError(f"unknown contribution type {contribution_type!r}")
    unknown = [name for name in dimensions if name not in QUALITY_DIMENSIONS]
    if unknown:
        raise ConfigError(f"unknown quality dimensions {', '.join(sorted(unknown))}")
    unknown_hard = [name for name in hard_failures if name not in QUALITY_HARD_FAILURES]
    if unknown_hard:
        raise ConfigError(f"unknown hard failures {', '.join(sorted(unknown_hard))}")
    passed = [name for name in QUALITY_DIMENSIONS if dimensions.get(name, False)]
    if contribution_type in ("code", "performance"):
        missing = [name for name in PR_REQUIRED_DIMENSIONS if name not in passed]
        if missing:
            hard_failures = tuple(list(hard_failures) + [f"missing_{name}_for_{contribution_type}" for name in missing])
    if contribution_type == "performance":
        missing = [name for name in PERFORMANCE_REQUIRED_DIMENSIONS if name not in passed]
        if missing:
            hard_failures = tuple(list(hard_failures) + [f"missing_{name}_for_performance" for name in missing])
    return {
        "contribution_type": contribution_type,
        "passed_dimensions": passed,
        "missing_dimensions": [name for name in QUALITY_DIMENSIONS if name not in passed],
        "hard_failures": list(hard_failures),
        "ok": not hard_failures,
        "note": "高质量 issue 可以没有 patch，但 authenticity/boundary/reproducer/diagnosis/communication/attribution/downstream 必须合格",
    }


def attribution_audit(*, personal: Sequence[str], agent: Sequence[str], maintainer: Sequence[str]) -> List[str]:
    """Human/agent/maintainer attribution is explicit (§9, step 42)."""
    findings: List[str] = []
    if not personal:
        findings.append("attribution audit: no personal contribution is recorded (a maintainer rewrote everything?)")
    for role, items in (("agent", agent), ("maintainer", maintainer)):
        for item in items:
            if item in personal:
                findings.append(f"attribution audit: {item!r} appears as both personal and {role}")
    return findings


def downstream_actions(*, upstream_status: str, released: bool) -> Tuple[str, ...]:
    """What HQSB must do after the upstream outcome (step 40)."""
    if upstream_status not in UPSTREAM_DISPOSITIONS:
        raise ConfigError(f"unknown upstream status {upstream_status!r}")
    if upstream_status in ("MERGED", "RELEASED") and not released:
        return ("pin_commit_or_keep_workaround", "replay_in_hqsb", "update_claim_and_support_matrix")
    if upstream_status in ("MERGED", "RELEASED"):
        return ("update_dependency_pin", "replay_in_hqsb", "update_claim_and_support_matrix")
    if upstream_status in ("REJECTED", "WONTFIX", "STALLED"):
        return ("keep_workaround", "record_limitation", "update_support_matrix")
    if upstream_status == "DUPLICATE":
        return ("link_related_issue", "replay_in_hqsb_if_version_relevant")
    return ("track_status", "replay_in_hqsb")


def disposition_claim(status: str) -> Mapping[str, str]:
    """The permitted wording for a final disposition (§9.1)."""
    for row in DISPOSITION_CLAIMS:
        if row["status"] == status:
            return dict(row)
    raise ConfigError(f"unknown disposition {status!r}")


def freeze_contribution(
    *, contract: ContributionStudyContract, record: rec.UpstreamContributionRecord, score: Mapping[str, Any]
) -> Dict[str, Any]:
    """Freeze the contribution record (step 45)."""
    problems = list(contract.problems()) + list(record.validate())
    if record.status == rec.STATUS_PASS and not score.get("ok"):
        problems.append("the contribution record claims PASS but the quality score did not pass")
    if problems:
        raise ConfigError("refusing to freeze the contribution: " + "; ".join(problems[:5]))
    return {
        "schema_version": f"{SCHEMA_PREFIX}.upstream-contribution.v1",
        "contribution_id": contract.contribution_id,
        "record": record.as_dict(),
        "quality_score": dict(score),
        "record_digest": canonical_digest(record.as_dict()),
    }


# ── PROTOCOL_STEPS (45) ─────────────────────────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 ContributionStudyContract", ("upstream.ContributionStudyContract",)),
    (2, "建立真实问题候选池", ("upstream.CandidateFinding",)),
    (3, "为候选保存原始证据", ("upstream.CandidateFinding.problems", "identity.EvidenceRef")),
    (4, "先排查 HQSB 自身错误", ("upstream.boundary_triage", "upstream.BOUNDARY_TRIAGE")),
    (5, "复现于上游最小环境", ("upstream.MinimalReproducer",)),
    (6, "选择候选目标上游", ("upstream.ContributionStudyContract.target_upstreams",)),
    (7, "读取当前贡献指南", ("upstream.ContributionStudyContract", "telemetry.ActionLog")),
    (8, "确认安全披露边界", ("upstream.boundary_triage", "supply_chain.retraction_plan")),
    (9, "搜索重复 issue/PR", ("upstream.ContributionStudyContract", "telemetry.ActionLog")),
    (10, "检查当前主干和最新 release", ("upstream.MinimalReproducer", "docs_gate.check_version_consistency")),
    (11, "执行版本二分或边界定位", ("upstream.MinimalReproducer.negative_controls",)),
    (12, "最小化输入", ("upstream.MinimalReproducer",)),
    (13, "最小化依赖", ("upstream.MinimalReproducer.dependencies",)),
    (14, "最小化执行步骤", ("upstream.MinimalReproducer.command_count",)),
    (15, "验证 reproducer 稳定性", ("upstream.MinimalReproducer.failure_rate", "upstream.MinimalReproducer.problems")),
    (16, "加入负对照", ("upstream.MinimalReproducer.negative_controls",)),
    (17, "采集必要诊断", ("upstream.IssueDraft.evidence", "telemetry.ActionLog")),
    (18, "判断贡献类型", ("upstream.CONTRIBUTION_TYPES", "records.UpstreamContributionRecord.CONTRIBUTION_TYPES")),
    (19, "写问题陈述", ("upstream.IssueDraft",)),
    (20, "限制性能措辞", ("upstream.check_performance_wording",)),
    (21, "执行公开前隐私扫描", ("upstream.IssueDraft", "supply_chain.verify_pii_scan")),
    (22, "执行许可和可提交性检查", ("supply_chain.LicenseInventory", "upstream.IssueDraft")),
    (23, "记录 Agent 使用边界", ("upstream.attribution_audit", "contracts.ContributionRecord")),
    (24, "让独立技术 reviewer 预审", ("upstream.quality_score", "claims.ContextAdjudication")),
    (25, "提交 issue/RFC", ("upstream.IssueDraft", "upstream.disposition_claim")),
    (26, "若适合，设计最小 patch", ("upstream.PatchDesign",)),
    (27, "补 regression test", ("upstream.check_patch",)),
    (28, "运行上游规定测试", ("upstream.check_patch", "telemetry.ActionLog")),
    (29, "运行 correctness 扩展矩阵", ("upstream.check_patch", "hero_replay.evaluate_correctness_matrix")),
    (30, "运行性能 benchmark", ("upstream.check_performance_wording", "hero_replay.BlockedEstimate")),
    (31, "审查 API/ABI 和 backward compatibility", ("upstream.PatchDesign.compatibility",)),
    (32, "更新文档和 release-note fragment", ("upstream.downstream_actions",)),
    (33, "提交 PR 并关联 issue", ("upstream.ReviewRound", "upstream.disposition_claim")),
    (34, "响应自动 CI 和 reviewer", ("upstream.ReviewRound",)),
    (35, "记录每轮设计变化", ("upstream.ReviewRound",)),
    (36, "处理替代方案", ("upstream.PatchDesign.alternatives", "upstream.ReviewRound")),
    (37, "处理 duplicate/won't-fix", ("upstream.disposition_claim", "upstream.downstream_actions")),
    (38, "处理 stalled/no-response", ("upstream.disposition_claim", "upstream.UPSTREAM_DISPOSITIONS")),
    (39, "验证最终上游状态", ("upstream.disposition_claim", "records.UpstreamContributionRecord")),
    (40, "将上游结果回流 HQSB", ("upstream.downstream_actions",)),
    (41, "在 HQSB 重放原问题", ("hero_replay.freeze_replay", "upstream.downstream_actions")),
    (42, "执行个人贡献审计", ("upstream.attribution_audit", "contracts.ContributionRecord")),
    (43, "让第三方评审贡献质量", ("upstream.quality_score", "upstream.QUALITY_DIMENSIONS")),
    (44, "生成可面试贡献故事", ("upstream.ReviewRound", "narrative.generate_faq")),
    (45, "冻结 UpstreamContributionRecord", ("upstream.freeze_contribution", "records.UpstreamContributionRecord")),
)

TITLE = "真实上游 Issue/PR/文档贡献"
LEVEL = "P0"
CLAIM_BOUNDARY = "E15-10 证明真实、可核验的上游协作；merged 不是唯一通过标准"


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the upstream-contribution interfaces (labelled smoke)."""
    problems: List[str] = []
    contract = ContributionStudyContract(
        contribution_id="contrib-1",
        candidate_id="cand-1",
        target_upstreams=("triton",),
        budget="one issue + patch",
        success_statuses=("OPEN", "UNDER_REVIEW", "ACCEPTED", "MERGED"),
        prohibit_metrics=("metric_gaming",),
    )
    problems.extend(contract.problems())
    if boundary_triage(error_source="upstream", evidence=("run id r1",)) != "upstream":
        problems.append("an evidenced upstream attribution did not triage as upstream")
    upstream_without_evidence = False
    try:
        boundary_triage(error_source="upstream", evidence=())
    except ConfigError:
        upstream_without_evidence = True
    if not upstream_without_evidence:
        problems.append("an evidence-less upstream attribution was accepted")
    reproducer = MinimalReproducer(
        reproducer_id="r-1",
        steps=("install", "run", "observe"),
        negative_controls=("change dtype → issue disappears",),
        failure_rate=0.5,
        seed_concurrency_timing="seed=0",
        command_count=3,
    )
    problems.extend(reproducer.problems())
    score = quality_score(
        dimensions={name: True for name in QUALITY_DIMENSIONS},
        hard_failures=(),
        contribution_type="code",
    )
    if not score["ok"]:
        problems.append("a fully-passing code contribution was rejected")
    fabricated = quality_score(
        dimensions={name: True for name in QUALITY_DIMENSIONS},
        hard_failures=("fabricated_problem",),
        contribution_type="issue",
    )
    if fabricated["ok"]:
        problems.append("a fabricated problem passed the quality gate")
    if not downstream_actions(upstream_status="REJECTED", released=False):
        problems.append("a rejected PR did not produce downstream actions")
    claim = disposition_claim("MERGED")
    if "已进入代码" in claim["may_say"] and claim["may_not_say"] == "":
        problems.append("disposition claim is wrong")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "contribution_types": len(CONTRIBUTION_TYPES),
        "quality_dimensions": len(QUALITY_DIMENSIONS),
        "dispositions": len(UPSTREAM_DISPOSITIONS),
        "problems": problems,
    }
