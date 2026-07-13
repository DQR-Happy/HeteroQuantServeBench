"""E15-08 — 3/10/30 minute narratives and adversarial Q&A.

Protocol: ``docs/stage_experiments/details/S15/E15-08_interview_narrative_adversarial_qa.md``
(45 steps).

The stage converts technical depth into *defensible* communication, with one hard
rule above the others: the three durations and the two role variants share one
fact spine; only the emphasis changes, never the facts.

* :class:`NarrativeContract` and :class:`FactCard` — the allowed/forbidden claims
  and the per-claim fact object every sentence renders from (steps 1, 5);
* :func:`role_matrix` — the two target-role perspectives (telecom/grid platform vs
  AI-infra/kernel), each with its required emphasis and one retained technical
  depth item (steps 2, 12–13);
* :class:`NarrativeOutline` — the 3/10/30 nested spine with the §9.1 content-node
  table, and :func:`check_two_role_fact_diff` which refuses a differing number
  between the two role versions (steps 9–11, 14);
* :data:`QUESTION_CATEGORIES` and :func:`question_bank` — the adversarial question
  taxonomy (correctness, benchmark, CUDA/operator, Amdahl, runtime/serving,
  heterogeneous/distributed, failure, contribution) with mandatory categories
  (steps 18–26, 28);
* :class:`AnswerRubric` — the six-dimension 0–3 scoring plus the hard-fail list
  of §9 (step 27);
* :class:`InterviewSession` — durations, nodes covered, fact errors, overclaims,
  answer scores, evidence lookup and attribution mismatches (steps 29–40);
* :func:`generate_faq`, :func:`generate_resume_bullets` — FAQ and resume
  candidates that only render verified claims, never a placeholder (steps 43–44);
* :func:`freeze_narrative` — step 45.

Nothing here conducts an interview or produces a claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release import records as rec
from hqsb.release.identity import SCHEMA_PREFIX, canonical_digest

#: The three nested durations.
DURATION_VARIANTS: Tuple[str, ...] = rec.InterviewSessionResult.DURATION_VARIANTS

#: The two role variants.
ROLE_VARIANTS: Tuple[str, ...] = rec.InterviewSessionResult.ROLE_VARIANTS

#: The content nodes every variant must cover (§9.1).
REQUIRED_NODES: Tuple[str, ...] = rec.InterviewSessionResult.REQUIRED_NODES

#: Adversarial question categories (step 18).
QUESTION_CATEGORIES: Tuple[str, ...] = (
    "project_value",
    "model",
    "kernel",
    "numerical_correctness",
    "benchmark",
    "profile",
    "amdahl",
    "quantization",
    "runtime",
    "serving",
    "heterogeneous_distributed",
    "production",
    "security",
    "failure",
    "contribution",
)

#: Categories that must be in every sampling round (step 28).
MANDATORY_QUESTION_CATEGORIES: Tuple[str, ...] = (
    "numerical_correctness",
    "benchmark",
    "amdahl",
    "failure",
    "contribution",
)

#: Role emphasis: what each audience cares about, and the one depth item retained.
ROLE_MATRIX: Tuple[Mapping[str, Any], ...] = (
    {
        "role": "telecom_grid",
        "emphasis": ("稳定契约", "异构适配", "可观测", "SLO", "供应链", "故障回退", "交付纪律"),
        "retained_depth": "底层算子正确性",
        "forbidden": "只讲项目管理不讲技术",
    },
    {
        "role": "ai_infra_kernel",
        "emphasis": ("模型语义", "shape", "SIMT/访存", "数值", "dispatcher", "micro→model"),
        "retained_depth": "生产/证据治理",
        "forbidden": "只讲 kernel 峰值不讲闭环",
    },
)

#: The §9.1 minimum content per node per duration.
CONTENT_GATES: Mapping[str, Mapping[str, str]] = {
    "3m": {
        "Problem": "真实瓶颈与为何跨层",
        "Architecture": "一张主链图",
        "Hero": "一条 verified 结论",
        "Correctness": "明确先于性能",
        "Performance": "scope + effect",
        "Engineering": "fallback/证据",
        "Contribution": "一句准确边界",
        "Limitation": "至少一项",
    },
    "10m": {
        "Problem": "加 workload/约束",
        "Architecture": "C1–C7 和模块边界",
        "Hero": "profile→kernel→model",
        "Correctness": "oracle/threshold",
        "Performance": "方法/n/CI/Amdahl",
        "Engineering": "Runtime/production",
        "Contribution": "关键个人决策",
        "Limitation": "失败故事",
    },
    "30m": {
        "Problem": "加替代方案和选题依据",
        "Architecture": "数据/控制/证据流及权衡",
        "Hero": "公式、code path、消融、反例",
        "Correctness": "误差传播与失败定位",
        "Performance": "counters、偏差、service/energy",
        "Engineering": "SRE/异构/供应链/维护成本",
        "Contribution": "Agent/框架/review 分工证据",
        "Limitation": "adoption/no-go 与下一实验",
    },
}

#: Hard failures of §9 — no score can rescue these.
HARD_FAILURES: Tuple[str, ...] = (
    "fabricated_number",
    "speedup_despite_correctness_fail",
    "micro_as_model_service",
    "third_party_as_own_implementation",
    "hidden_counter_evidence",
    "contradictory_role_versions",
)

#: Answer-scoring dimensions and their 0–3 anchors (abbreviated) (§9).
SCORING_DIMENSIONS: Tuple[str, ...] = ("fact", "mechanism", "evidence", "boundary", "decision", "expression")


@dataclass
class NarrativeContract:
    """The frozen contract: allowed/forbidden claims, audience, thresholds (step 1)."""

    candidate_id: str
    allowed_claims: Tuple[str, ...] = ()
    forbidden_claims: Tuple[str, ...] = ()
    audiences: Tuple[str, ...] = ROLE_VARIANTS
    pass_line: float = 2.0
    evidence_lookup_limit_s: float = 60.0

    schema_version = f"{SCHEMA_PREFIX}.narrative-contract.v1"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.candidate_id:
            findings.append("NarrativeContract: candidate_id is required")
        if not self.allowed_claims:
            findings.append("NarrativeContract: the allowed claims must be declared (a blank cheque drifts)")
        overlap = sorted(set(self.allowed_claims) & set(self.forbidden_claims))
        if overlap:
            findings.append(f"NarrativeContract: claims both allowed and forbidden: {', '.join(overlap)}")
        if not 0.0 <= self.pass_line <= 3.0:
            findings.append("NarrativeContract: pass_line must be within the 0–3 rubric")
        if self.evidence_lookup_limit_s <= 0:
            findings.append("NarrativeContract: the evidence-lookup limit must be positive")
        return findings


@dataclass
class FactCard:
    """One claim reduced to a sayable, evidence-bound card (step 5)."""

    claim_id: str
    text: str
    unit: str = ""
    scope: str = ""
    baseline: str = ""
    interval: str = ""
    artifact_ref: str = ""
    limitation: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.claim_id.startswith("CLM-"):
            findings.append(f"FactCard: claim_id must be CLM-*, got {self.claim_id!r}")
        if not self.text:
            findings.append("FactCard: the one-line sayable text is required")
        if not self.limitation:
            findings.append(f"FactCard({self.claim_id}): a fact card without its boundary is half a fact")
        if not self.artifact_ref:
            findings.append(f"FactCard({self.claim_id}): the evidence entry is required (数字必须能定位)")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "unit": self.unit,
            "scope": self.scope,
            "baseline": self.baseline,
            "interval": self.interval,
            "artifact_ref": self.artifact_ref,
            "limitation": self.limitation,
        }


def role_matrix() -> Tuple[Mapping[str, Any], ...]:
    """The two role emphases, as data (steps 2, 12–13)."""
    return ROLE_MATRIX


@dataclass
class NarrativeOutline:
    """One duration variant with its node coverage (steps 9–11)."""

    duration_variant: str
    nodes: Mapping[str, str] = field(default_factory=dict)

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.duration_variant not in DURATION_VARIANTS:
            findings.append(f"NarrativeOutline: unknown duration {self.duration_variant!r}")
        missing = [node for node in REQUIRED_NODES if node not in self.nodes]
        if missing:
            findings.append(f"NarrativeOutline({self.duration_variant}): nodes missing: {', '.join(missing)}")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "duration_variant": self.duration_variant,
            "nodes": {key: self.nodes[key] for key in sorted(self.nodes)},
        }


def check_two_role_fact_diff(
    telecom: Mapping[str, Any], kernel: Mapping[str, Any]
) -> List[str]:
    """The two role versions must differ in emphasis, never in facts (step 14)."""
    from hqsb.release.contracts import check_bilingual_fact_consistency

    fact_fields = ("claim_id", "point", "unit", "interval", "baseline", "evidence_level", "status", "limitation")
    findings: List[str] = []
    for claim_id in sorted(set(telecom) | set(kernel)):
        left = telecom.get(claim_id, {})
        right = kernel.get(claim_id, {})
        if left.get("claim_id") != right.get("claim_id"):
            findings.append(f"{claim_id}: claim appears in one role version only")
            continue
        diff = check_bilingual_fact_consistency(
            {name: left.get(name) for name in fact_fields}, {name: right.get(name) for name in fact_fields}
        )
        findings.extend(diff)
    return findings


def question_bank() -> Dict[str, List[Mapping[str, Any]]]:
    """The frozen adversarial question bank (steps 18–26)."""
    return {
        "numerical_correctness": [
            {"question": "RMSNorm 误差可接受的标准是什么，greedy token 相同是否足够？", "category": "numerical_correctness"},
            {"question": "fused 计算的 rounding 与首错位如何定位？", "category": "numerical_correctness"},
        ],
        "benchmark": [
            {"question": "你的实验单位是独立 run 还是 token？同步与噪声如何处理？", "category": "benchmark"},
            {"question": "baseline 是否公平？cache 与 thermal 是否控制？", "category": "benchmark"},
        ],
        "kernel": [
            {"question": "reduction 的访存合并、alignment/tail 如何处理？", "category": "kernel"},
        ],
        "amdahl": [
            {"question": "给出热点占比、micro speedup、额外开销与总收益上限。", "category": "amdahl"},
        ],
        "runtime": [
            {"question": "paged KV 与 continuous batching 的区别，何时用 prefix cache？", "category": "runtime"},
        ],
        "serving": [
            {"question": "goodput 与 P99 的定义，过载时如何 admission？", "category": "serving"},
        ],
        "heterogeneous_distributed": [
            {"question": "CUDA/Ascend 结果如何比较才公平？TP 通信量如何算？", "category": "heterogeneous_distributed"},
        ],
        "failure": [
            {"question": "如果 profile 证明 RMSNorm 不是热点怎么办？", "category": "failure"},
            {"question": "新 GPU 上方向反转怎么办？", "category": "failure"},
        ],
        "contribution": [
            {"question": "哪些是你做的，哪些是框架/Agent 做的？", "category": "contribution"},
        ],
    }


def sample_questions(categories: Sequence[str], *, seed: int) -> List[Mapping[str, Any]]:
    """Sampled questions with the mandatory categories forced in (step 28)."""
    if seed < 0:
        raise ConfigError("sample_questions: the seed must be frozen")
    bank = question_bank()
    chosen: List[Mapping[str, Any]] = []
    for category in MANDATORY_QUESTION_CATEGORIES:
        items = bank.get(category, [])
        if not items:
            raise ConfigError(f"question bank is missing mandatory category {category!r}")
        chosen.append(items[seed % len(items)])
    for category in categories:
        if category in MANDATORY_QUESTION_CATEGORIES:
            continue
        items = bank.get(category)
        if items:
            chosen.append(items[(seed + len(chosen)) % len(items)])
    return chosen


@dataclass
class AnswerRubric:
    """The six-dimension scoring rubric with hard failures (step 27)."""

    scores: Mapping[str, float] = field(default_factory=dict)
    hard_failures: Tuple[str, ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        for dimension in SCORING_DIMENSIONS:
            if dimension not in self.scores:
                findings.append(f"AnswerRubric: dimension {dimension!r} was not scored")
                continue
            score = self.scores[dimension]
            if not 0.0 <= score <= 3.0:
                findings.append(f"AnswerRubric: score {score} for {dimension!r} is outside 0–3")
        unknown = [name for name in self.hard_failures if name not in HARD_FAILURES]
        if unknown:
            findings.append(f"AnswerRubric: unknown hard failures {', '.join(sorted(unknown))}")
        return findings

    def total(self) -> float:
        if not self.scores:
            return 0.0
        return float(sum(self.scores.values()) / len(self.scores))

    def passed(self, *, pass_line: float) -> bool:
        return not self.hard_failures and self.total() >= pass_line

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scores": {key: self.scores[key] for key in sorted(self.scores)},
            "hard_failures": list(self.hard_failures),
            "total": self.total(),
        }


@dataclass
class InterviewSession:
    """One interview/narrative session (steps 29–40)."""

    session_id: str
    duration_variant: str = ""
    role_variant: str = ""
    actual_duration_s: float = 0.0
    nodes_covered: Tuple[str, ...] = ()
    fact_errors: Tuple[str, ...] = ()
    overclaims: Tuple[str, ...] = ()
    evidence_refs_used: Tuple[str, ...] = ()
    question_categories: Tuple[str, ...] = ()
    answer_scores: Mapping[str, float] = field(default_factory=dict)
    unknown_handling_scores: Tuple[float, ...] = ()
    evidence_lookup_times_s: Tuple[float, ...] = ()
    attribution_mismatches: Tuple[str, ...] = ()
    reviewer_ids: Tuple[str, ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.session_id:
            findings.append("InterviewSession: session_id is required")
        if self.duration_variant not in DURATION_VARIANTS:
            findings.append(f"InterviewSession: unknown duration {self.duration_variant!r}")
        if self.role_variant not in ROLE_VARIANTS:
            findings.append(f"InterviewSession: unknown role {self.role_variant!r}")
        if not self.reviewer_ids:
            findings.append("InterviewSession: at least one reviewer is required (a self-review is not a review)")
        if self.fact_errors or self.overclaims or self.attribution_mismatches:
            findings.append("InterviewSession: fact errors/overclaims/attribution mismatches are hard failures (§9)")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "duration_variant": self.duration_variant,
            "role_variant": self.role_variant,
            "actual_duration_s": self.actual_duration_s,
            "nodes_covered": list(self.nodes_covered),
            "fact_errors": list(self.fact_errors),
            "overclaims": list(self.overclaims),
            "evidence_refs_used": list(self.evidence_refs_used),
            "question_categories": list(self.question_categories),
            "answer_scores": {key: self.answer_scores[key] for key in sorted(self.answer_scores)},
            "unknown_handling_scores": list(self.unknown_handling_scores),
            "evidence_lookup_times_s": list(self.evidence_lookup_times_s),
            "attribution_mismatches": list(self.attribution_mismatches),
            "reviewer_ids": list(self.reviewer_ids),
        }


def generate_faq(questions: Sequence[Mapping[str, Any]], *, cards: Mapping[str, FactCard]) -> Dict[str, Any]:
    """FAQ with a short/deep answer and an evidence index (step 43)."""
    faq: List[Dict[str, Any]] = []
    for question in questions:
        category = str(question.get("category", ""))
        faq.append(
            {
                "question": question.get("question", ""),
                "category": category,
                "claim_refs": [],
                "short_answer_template": "结论先行 + 机制 + 边界",
                "forbidden_wording": "空洞术语、伪造数字、越级 claim",
            }
        )
    return {"faq": faq, "evidence_index": [card.as_dict() for card in cards.values()]}


def generate_resume_bullets(cards: Sequence[FactCard]) -> Tuple[Dict[str, Any], ...]:
    """Resume bullets render *only* verified claims; placeholders are refused (step 44)."""
    bullets: List[Dict[str, Any]] = []
    for card in cards:
        problems = card.problems()
        if problems:
            raise ConfigError(f"refusing to render a resume bullet from an invalid card: {'; '.join(problems)}")
        bullets.append(
            {
                "claim_id": card.claim_id,
                "bullet": f"{card.text}（{card.scope}；基线 {card.baseline}；限制 {card.limitation}）",
                "evidence_ref": card.artifact_ref,
                "note": "一句话仍保留硬件/对象/边界；无法保留范围就宁可不写该数字",
            }
        )
    return tuple(bullets)


def freeze_narrative(
    *, contract: NarrativeContract, record: rec.InterviewSessionResult, sessions: Sequence[InterviewSession]
) -> Dict[str, Any]:
    """Freeze the interview record (step 45)."""
    problems = list(contract.problems()) + list(record.validate())
    for session in sessions:
        problems.extend(session.problems())
    if problems:
        raise ConfigError("refusing to freeze the narrative: " + "; ".join(problems[:5]))
    return {
        "schema_version": f"{SCHEMA_PREFIX}.interview-narrative.v1",
        "contract": contract.candidate_id,
        "record": record.as_dict(),
        "sessions": [session.as_dict() for session in sessions],
        "record_digest": canonical_digest(record.as_dict()),
    }


# ── PROTOCOL_STEPS (45) ─────────────────────────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 NarrativeContract", ("narrative.NarrativeContract",)),
    (2, "建立岗位能力矩阵", ("narrative.role_matrix", "narrative.ROLE_MATRIX")),
    (3, "选择唯一 Hero Story", ("hero_replay.HeroStoryContract", "narrative.FactCard")),
    (4, "选择一个失败故事", ("narrative.FactCard", "claims.adjudicate_release_eligibility")),
    (5, "冻结事实卡片", ("narrative.FactCard",)),
    (6, "冻结 ContributionRecord", ("contracts.ContributionRecord",)),
    (7, "写一句项目定义", ("narrative.NarrativeContract.allowed_claims",)),
    (8, "写 30 秒价值段", ("narrative.FactCard", "demo.DemoObjective")),
    (9, "设计 3 分钟骨架", ("narrative.NarrativeOutline", "narrative.CONTENT_GATES")),
    (10, "设计 10 分钟扩展", ("narrative.NarrativeOutline",)),
    (11, "设计 30 分钟深挖", ("narrative.NarrativeOutline",)),
    (12, "设计运营商/电网强调层", ("narrative.ROLE_MATRIX", "narrative.role_matrix")),
    (13, "设计 AI Infra/算子强调层", ("narrative.ROLE_MATRIX",)),
    (14, "执行两版本事实 diff", ("narrative.check_two_role_fact_diff", "contracts.check_bilingual_fact_consistency")),
    (15, "准备最小架构图", ("figures.check_min_context", "figures.FigureSpec")),
    (16, "准备 hero 图和反例图", ("figures.FigureSpec", "figures.check_min_context")),
    (17, "准备 evidence drill-down 路径", ("telemetry.EvidenceLookupTrial", "figures.check_access_friction")),
    (18, "建立问题分类法", ("narrative.QUESTION_CATEGORIES",)),
    (19, "编写 correctness 对抗题", ("narrative.question_bank",)),
    (20, "编写 benchmark 对抗题", ("narrative.question_bank",)),
    (21, "编写 CUDA/算子对抗题", ("narrative.question_bank",)),
    (22, "编写 Amdahl 对抗题", ("narrative.question_bank", "hero_replay.AmdahlModel")),
    (23, "编写 Runtime/Serving 对抗题", ("narrative.question_bank",)),
    (24, "编写异构/分布式对抗题", ("narrative.question_bank",)),
    (25, "编写失败/反事实问题", ("narrative.question_bank",)),
    (26, "编写个人贡献追问", ("narrative.question_bank", "contracts.ContributionRecord")),
    (27, "建立回答 rubric", ("narrative.AnswerRubric", "narrative.SCORING_DIMENSIONS")),
    (28, "冻结问题抽样规则", ("narrative.sample_questions", "narrative.MANDATORY_QUESTION_CATEGORIES")),
    (29, "录制 3 分钟无打断讲述", ("narrative.InterviewSession", "records.InterviewSessionResult")),
    (30, "录制 10 分钟讲述", ("narrative.InterviewSession",)),
    (31, "录制 30 分钟讲述", ("narrative.InterviewSession",)),
    (32, "执行运营商/电网模拟面试", ("narrative.InterviewSession", "narrative.ROLE_VARIANTS")),
    (33, "执行 AI Infra/算子模拟面试", ("narrative.InterviewSession",)),
    (34, "执行 correctness 追问链", ("narrative.AnswerRubric",)),
    (35, "执行 benchmark 追问链", ("narrative.AnswerRubric",)),
    (36, "执行 Amdahl 现场题", ("hero_replay.AmdahlModel.ideal_upper_bound", "narrative.AnswerRubric")),
    (37, "执行 evidence 随机定位", ("telemetry.EvidenceLookupTrial", "narrative.NarrativeContract.evidence_lookup_limit_s")),
    (38, "执行未知问题处理测试", ("narrative.AnswerRubric", "narrative.HARD_FAILURES")),
    (39, "执行矛盾证据测试", ("hero_replay.replay_verdict", "narrative.AnswerRubric")),
    (40, "执行 attribution 审计", ("contracts.ContributionRecord.problems", "narrative.InterviewSession")),
    (41, "汇总盲评和分歧", ("narrative.InterviewSession.reviewer_ids", "telemetry.TimeToEventSummary")),
    (42, "修订表达而非事实", ("narrative.check_two_role_fact_diff", "claims.revise_claim")),
    (43, "生成 FAQ 和证据索引", ("narrative.generate_faq",)),
    (44, "生成简历 bullet 候选", ("narrative.generate_resume_bullets",)),
    (45, "冻结 InterviewNarrativeRecord", ("narrative.freeze_narrative", "records.InterviewSessionResult")),
)

TITLE = "3/10/30 分钟讲述与对抗技术问答"
LEVEL = "P0"
CLAIM_BOUNDARY = "E15-08 证明表达可验证且技术上可防守；不代表真实招聘结果，也不替代 E15-09 的独立复现"


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the narrative interfaces (labelled smoke)."""
    problems: List[str] = []
    contract = NarrativeContract(
        candidate_id="cand-1",
        allowed_claims=("CLM-1",),
        forbidden_claims=(),
        pass_line=2.0,
        evidence_lookup_limit_s=60.0,
    )
    problems.extend(contract.problems())
    card = FactCard(
        claim_id="CLM-1",
        text="在目标 shape 上，自定义 RMSNorm kernel 相对 reference 的时延",
        unit="ms",
        scope="Qwen3 / 目标设备 / 预注册 shape",
        baseline="reference RMSNorm",
        interval="[x, y]",
        artifact_ref="raw::sha256:...",
        limitation="只在预注册 shape 上成立",
    )
    problems.extend(card.problems())
    outline = NarrativeOutline(duration_variant="10m", nodes={node: CONTENT_GATES["10m"][node] for node in REQUIRED_NODES})
    problems.extend(outline.problems())
    sampled = sample_questions(("kernel",), seed=3)
    if not any(item["category"] in MANDATORY_QUESTION_CATEGORIES for item in sampled):
        problems.append("mandatory question categories were not forced into the sample")
    rubric = AnswerRubric(scores={dimension: 2.0 for dimension in SCORING_DIMENSIONS}, hard_failures=())
    if not rubric.passed(pass_line=2.0):
        problems.append("an average-2.0 answer did not pass the 2.0 line")
    fabricated = AnswerRubric(scores={dimension: 3.0 for dimension in SCORING_DIMENSIONS}, hard_failures=("fabricated_number",))
    if fabricated.passed(pass_line=2.0):
        problems.append("a fabricated number passed despite the hard failure")
    diff = check_two_role_fact_diff(
        {"CLM-1": {"claim_id": "CLM-1", "point": 1.0, "unit": "ms"}},
        {"CLM-1": {"claim_id": "CLM-1", "point": 1.5, "unit": "ms"}},
    )
    if not diff:
        problems.append("a role-variant number drift was not detected")
    bullets = generate_resume_bullets((card,))
    if not bullets or "CLM-1" not in bullets[0]["claim_id"]:
        problems.append("resume bullet generation failed")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "question_categories": len(QUESTION_CATEGORIES),
        "mandatory_categories": len(MANDATORY_QUESTION_CATEGORIES),
        "scoring_dimensions": len(SCORING_DIMENSIONS),
        "problems": problems,
    }
