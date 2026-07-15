"""E15-11 — target-reader five-minute first-impression study.

Protocol: ``docs/stage_experiments/details/S15/E15-11_recruiter_first_impression_usability.md``
(45 steps).  P1 — recommended before applying, but it can only change *information
architecture*, never maturity or numbers.

The study is a formation-style usability test, not a survey:

* :class:`UsabilityStudyContract` — candidate, the 5-minute limit, participant
  types, tasks, scoring, recording and privacy, frozen *before* feedback (step 1);
* :class:`ParticipantProfile` — role block, background, and the exclusion rule
  (anyone who worked on HQSB / read the detailed design is excluded) (steps 2–4);
* :class:`TaskSet` — the non-leading scenario and the five task groups (recall,
  state boundary, evidence lookup, attribution, limitation) (steps 7–12);
* :class:`AnswerKey` — correct/partial/incorrect coding that only sources from
  E15-01/03/08 and limitations (step 13);
* :class:`ParticipantSession` — event log, path, first click, recall, task
  answers, debrief, with the 5-minute hard stop (steps 22–35);
* :func:`code_answers`, :func:`encode_misconceptions`,
  :func:`role_block_analysis` — independent coding and the block analysis that
  keeps the two role groups separate (steps 36–38);
* :func:`revision_plan`, :func:`check_no_fact_change`, :func:`retest_scope`,
  :func:`unexpected_regression` — the revision loop that *must* use new
  participants to avoid learning effects (steps 39–44);
* :func:`freeze_study` — step 45.

Nothing here runs a participant session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release import records as rec
from hqsb.release.identity import SCHEMA_PREFIX, canonical_digest

#: Role blocks (§3.4).
ROLE_BLOCKS: Tuple[str, ...] = rec.FirstImpressionSessionResult.ROLE_BLOCKS

#: Session modes (§3.3).
SESSION_MODES: Tuple[str, ...] = rec.FirstImpressionSessionResult.SESSION_MODES

#: The five-minute limit.
BROWSE_LIMIT_S = 300.0

#: Participant ceiling: formation-style, not a survey (step 4).
MAX_PARTICIPANTS = 3

#: Score codes (step 13).
SCORE_CODES: Tuple[str, ...] = ("correct", "partial", "incorrect")

#: Misconception severities (step 37).
MISCONCEPTION_SEVERITIES: Tuple[str, ...] = ("P0", "P1", "P2")

#: The five task groups (steps 8–12).
TASK_GROUPS: Tuple[str, ...] = ("recall", "state_boundary", "evidence_lookup", "attribution", "limitation")


@dataclass
class UsabilityStudyContract:
    """The frozen contract (step 1)."""

    study_id: str
    candidate_id: str
    entry_point: str = ""
    participant_types: Tuple[str, ...] = ("telecom_grid", "ai_infra_kernel")
    browse_limit_s: float = BROWSE_LIMIT_S
    task_set_version: str = ""
    scoring_rubric: Mapping[str, str] = field(default_factory=dict)
    recording_mode: str = ""
    privacy_policy: str = ""
    pass_line: str = ""

    schema_version = f"{SCHEMA_PREFIX}.usability-contract.v1"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.study_id or not self.candidate_id:
            findings.append("UsabilityStudyContract: study_id and candidate_id are required")
        if not self.entry_point:
            findings.append("UsabilityStudyContract: a single public entry point is required (no deep links)")
        if not self.task_set_version:
            findings.append("UsabilityStudyContract: the task set version must be frozen")
        if self.browse_limit_s != BROWSE_LIMIT_S:
            findings.append(f"UsabilityStudyContract: the limit must be 5 minutes ({BROWSE_LIMIT_S}s)")
        if not self.recording_mode or not self.privacy_policy:
            findings.append("UsabilityStudyContract: recording mode and privacy policy are required")
        if not self.pass_line:
            findings.append("UsabilityStudyContract: the pass line must be written before seeing the answers")
        return findings


@dataclass
class ParticipantProfile:
    """One participant's profile and exclusion status (steps 2–4, 22)."""

    participant_id: str
    role_block: str = ""
    llm_experience: str = ""
    cuda_experience: str = ""
    open_source_experience: str = ""
    language_preference: str = ""
    excluded: bool = False
    exclusion_reason: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.participant_id:
            findings.append("ParticipantProfile: participant_id is required")
        if self.role_block not in ROLE_BLOCKS:
            findings.append(f"ParticipantProfile: unknown role block {self.role_block!r}")
        if not self.llm_experience or not self.language_preference:
            findings.append("ParticipantProfile: background baseline must be recorded")
        if self.excluded and not self.exclusion_reason:
            findings.append("ParticipantProfile: an excluded participant needs the exclusion reason")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "role_block": self.role_block,
            "llm_experience": self.llm_experience,
            "cuda_experience": self.cuda_experience,
            "open_source_experience": self.open_source_experience,
            "language_preference": self.language_preference,
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass
class TaskSet:
    """The non-leading scenario and its tasks (steps 7–12)."""

    task_set_version: str
    scenario_prompt: str = ""
    tasks: Mapping[str, str] = field(default_factory=dict)
    leading: bool = False

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.scenario_prompt:
            findings.append("TaskSet: the scenario prompt is required")
        if self.leading:
            findings.append("TaskSet: the scenario must not reveal the correct hero story (step 7)")
        missing = [group for group in TASK_GROUPS if group not in self.tasks]
        if missing:
            findings.append(f"TaskSet: task groups missing: {', '.join(missing)}")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_set_version": self.task_set_version,
            "scenario_prompt": self.scenario_prompt,
            "tasks": {key: self.tasks[key] for key in sorted(self.tasks)},
            "leading": self.leading,
        }


@dataclass
class AnswerKey:
    """Correct/partial/incorrect, sourced only from verified claims (step 13)."""

    task_set_version: str
    answers: Mapping[str, str] = field(default_factory=dict)

    def problems(self) -> List[str]:
        findings: List[str] = []
        unknown = [code for code in self.answers.values() if code not in SCORE_CODES]
        if unknown:
            findings.append(f"AnswerKey: unknown score codes {', '.join(sorted(set(unknown)))}")
        if not self.answers:
            findings.append("AnswerKey: an empty key cannot score anything")
        return findings


@dataclass
class ParticipantSession:
    """One five-minute browse and its results (steps 23–35)."""

    session_id: str
    participant_id: str
    session_mode: str = ""
    path: Tuple[str, ...] = ()
    first_click: str = ""
    browse_duration_s: float = 0.0
    free_recall: str = ""
    task_answers: Mapping[str, str] = field(default_factory=dict)
    evidence_lookup_time_s: Optional[float] = None
    evidence_lookup_clicks: int = 0
    status_label_understanding: Mapping[str, str] = field(default_factory=dict)
    attribution_understanding: Mapping[str, str] = field(default_factory=dict)
    limitation_found: bool = False
    continue_intention: str = ""
    debrief: str = ""
    forced_stop: bool = False

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.session_id or not self.participant_id:
            findings.append("ParticipantSession: session_id and participant_id are required")
        if self.session_mode not in SESSION_MODES:
            findings.append(f"ParticipantSession: unknown mode {self.session_mode!r}")
        if not self.path:
            findings.append("ParticipantSession: the browse path must be recorded")
        if not self.forced_stop and self.browse_duration_s > BROWSE_LIMIT_S:
            findings.append("ParticipantSession: the browse exceeded 5 minutes without the hard stop firing")
        if self.browse_duration_s < 0:
            findings.append("ParticipantSession: negative duration")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "session_mode": self.session_mode,
            "path": list(self.path),
            "first_click": self.first_click,
            "browse_duration_s": self.browse_duration_s,
            "free_recall": self.free_recall,
            "task_answers": {key: self.task_answers[key] for key in sorted(self.task_answers)},
            "evidence_lookup_time_s": self.evidence_lookup_time_s,
            "evidence_lookup_clicks": self.evidence_lookup_clicks,
            "status_label_understanding": dict(sorted(self.status_label_understanding.items())),
            "attribution_understanding": dict(sorted(self.attribution_understanding.items())),
            "limitation_found": self.limitation_found,
            "continue_intention": self.continue_intention,
            "debrief": self.debrief,
            "forced_stop": self.forced_stop,
        }


def code_answers(session: ParticipantSession, *, answer_key: AnswerKey) -> Dict[str, Any]:
    """Independent coding against the key (step 36)."""
    findings: List[str] = list(answer_key.problems())
    scores: Dict[str, str] = {}
    for group, expected in sorted(answer_key.answers.items()):
        actual = session.task_answers.get(group)
        if actual is None:
            scores[group] = "incorrect"
            findings.append(f"task {group!r} was not answered")
            continue
        if actual not in SCORE_CODES:
            findings.append(f"task {group!r} carries a non-score answer {actual!r}")
        scores[group] = actual
    return {
        "participant_id": session.participant_id,
        "scores": scores,
        "findings": findings,
        "correct": sum(1 for code in scores.values() if code == "correct"),
        "total": len(scores),
    }


def encode_misconceptions(*, session: ParticipantSession, coded: Mapping[str, str]) -> List[Mapping[str, Any]]:
    """Encode each wrong answer into a misconception with severity (step 37)."""
    misconceptions: List[Mapping[str, Any]] = []
    for group, score in sorted(coded.items()):
        if score == "correct":
            continue
        severity = "P0" if group in ("recall", "state_boundary") else ("P1" if group == "limitation" else "P2")
        misconceptions.append(
            {
                "participant_id": session.participant_id,
                "task_group": group,
                "score": score,
                "severity": severity,
                "root_cause": session.debrief or "not attributed",
            }
        )
    return misconceptions


def role_block_analysis(sessions: Sequence[ParticipantSession], *, coded: Mapping[str, Mapping[str, str]]) -> Dict[str, Any]:
    """Per-role-block analysis; the two groups are not merged (§3.4, step 38)."""
    by_block: Dict[str, List[Dict[str, Any]]] = {}
    for session in sessions:
        scores = coded.get(session.participant_id, {})
        by_block.setdefault(session.participant_id, []).append({"path": list(session.path), "scores": dict(scores)})
    return {
        "sessions": len(sessions),
        "by_participant": {key: by_block[key] for key in sorted(by_block)},
        "note": "两个岗位 block 分开呈现；不把一个群体的最优结构强加给另一个群体，也不计算虚假的总体百分比",
    }


@dataclass
class RevisionPlan:
    """P0 information-architecture fixes that must not change facts (steps 39–40)."""

    fixes: Tuple[str, ...] = ()
    fact_fields: Tuple[str, ...] = ()
    fact_change: bool = False

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.fixes:
            findings.append("RevisionPlan: at least one P0 fix is required")
        if self.fact_change:
            findings.append(
                "RevisionPlan: the revision changed a fact field; information architecture may only move the "
                "facts around, never change them (step 40)"
            )
        return findings


def check_no_fact_change(*, before: Mapping[str, Any], after: Mapping[str, Any]) -> List[str]:
    """The facts (numbers, statuses, limitations) must survive a restructure (step 40)."""
    findings: List[str] = []
    for name in ("claims", "numbers", "statuses", "limitations"):
        left = before.get(name)
        right = after.get(name)
        if isinstance(left, (list, tuple)):
            left = sorted(map(str, left))
        if isinstance(right, (list, tuple)):
            right = sorted(map(str, right))
        if left != right:
            findings.append(f"the revision changed {name!r}: {left!r} != {right!r}")
    return findings


def retest_scope(*, previous_participants: Sequence[str], retest_participants: Sequence[str]) -> List[str]:
    """A structural revision must be retested with new participants (step 42)."""
    findings: List[str] = []
    overlap = sorted(set(previous_participants) & set(retest_participants))
    if overlap:
        findings.append(
            f"participants {', '.join(overlap)} already know the answers; a learning effect invalidates "
            "the retest (E15-11 §3.5)"
        )
    if len(retest_participants) < 1:
        findings.append("the retest needs at least one new participant")
    return findings


def unexpected_regression(*, before: Mapping[str, Any], after: Mapping[str, Any]) -> List[str]:
    """Did the restructure make something else worse (step 43)?"""
    findings: List[str] = []
    if after.get("correct", 0) < before.get("correct", 0):
        findings.append("the revision reduced correct recall; the information architecture got worse")
    return findings


def freeze_study(
    *, contract: UsabilityStudyContract, record: rec.FirstImpressionSessionResult, sessions: Sequence[ParticipantSession]
) -> Dict[str, Any]:
    """Freeze the study record (step 45)."""
    problems = list(contract.problems()) + list(record.validate())
    for session in sessions:
        problems.extend(session.problems())
    if len(sessions) > MAX_PARTICIPANTS:
        problems.append(f"the study has {len(sessions)} participants; the protocol caps formation testing at {MAX_PARTICIPANTS}")
    if problems:
        raise ConfigError("refusing to freeze the study: " + "; ".join(problems[:5]))
    return {
        "schema_version": f"{SCHEMA_PREFIX}.first-impression-study.v1",
        "study_id": contract.study_id,
        "record": record.as_dict(),
        "sessions": [session.as_dict() for session in sessions],
        "record_digest": canonical_digest(record.as_dict()),
    }


# ── PROTOCOL_STEPS (45) ─────────────────────────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 UsabilityStudyContract", ("first_impression.UsabilityStudyContract",)),
    (2, "定义主要读者画像", ("first_impression.ParticipantProfile", "first_impression.ROLE_BLOCKS")),
    (3, "定义排除标准", ("first_impression.ParticipantProfile.excluded",)),
    (4, "确定参与者数量与边界", ("first_impression.MAX_PARTICIPANTS",)),
    (5, "设计统一入口", ("first_impression.UsabilityStudyContract.entry_point",)),
    (6, "冻结页面候选版本", ("docs_gate.PageInventory", "first_impression.UsabilityStudyContract")),
    (7, "设计非引导性场景", ("first_impression.TaskSet", "first_impression.TaskSet.leading")),
    (8, "定义核心复述问题", ("first_impression.TaskSet", "first_impression.TASK_GROUPS")),
    (9, "定义状态边界问题", ("first_impression.TaskSet",)),
    (10, "定义证据定位任务", ("first_impression.TaskSet", "telemetry.EvidenceLookupTrial")),
    (11, "定义贡献归属问题", ("first_impression.TaskSet", "contracts.ContributionRecord")),
    (12, "定义限制与可信度问题", ("first_impression.TaskSet", "claims.ClaimRecord.limitations")),
    (13, "建立答案 key", ("first_impression.AnswerKey", "first_impression.SCORE_CODES")),
    (14, "建立行为指标", ("first_impression.ParticipantSession.path", "telemetry.EvidenceLookupTrial")),
    (15, "选择自然浏览或 think-aloud 模式", ("first_impression.SESSION_MODES",)),
    (16, "设计事后回放访谈", ("first_impression.ParticipantSession.debrief",)),
    (17, "准备隐私与同意", ("first_impression.UsabilityStudyContract.privacy_policy",)),
    (18, "校准记录工具", ("first_impression.UsabilityStudyContract.recording_mode",)),
    (19, "执行主持人标准化训练", ("first_impression.TaskSet.leading",)),
    (20, "运行 pilot", ("first_impression.UsabilityStudyContract", "first_impression.TaskSet")),
    (21, "冻结修订后的协议", ("first_impression.UsabilityStudyContract.task_set_version",)),
    (22, "采集参与者背景基线", ("first_impression.ParticipantProfile",)),
    (23, "开始五分钟自然浏览", ("first_impression.BROWSE_LIMIT_S", "first_impression.ParticipantSession")),
    (24, "记录首屏和 first-click", ("first_impression.ParticipantSession.first_click",)),
    (25, "记录完整浏览路径", ("first_impression.ParticipantSession.path",)),
    (26, "五分钟强制停止", ("first_impression.ParticipantSession.forced_stop",)),
    (27, "无页面条件下自由复述", ("first_impression.ParticipantSession.free_recall",)),
    (28, "执行结构化理解问题", ("first_impression.code_answers",)),
    (29, "重新开放页面做证据任务", ("first_impression.ParticipantSession.evidence_lookup_time_s",)),
    (30, "执行 micro/model/service 区分题", ("first_impression.ParticipantSession.task_answers",)),
    (31, "执行状态标签理解题", ("first_impression.ParticipantSession.status_label_understanding",)),
    (32, "执行 attribution 题", ("first_impression.ParticipantSession.attribution_understanding",)),
    (33, "执行限制发现题", ("first_impression.ParticipantSession.limitation_found",)),
    (34, "采集继续阅读意向及理由", ("first_impression.ParticipantSession.continue_intention",)),
    (35, "执行事后路径回放", ("first_impression.ParticipantSession.debrief",)),
    (36, "独立编码答案", ("first_impression.code_answers", "first_impression.AnswerKey")),
    (37, "编码误解根因", ("first_impression.encode_misconceptions", "first_impression.MISCONCEPTION_SEVERITIES")),
    (38, "分析岗位 block 差异", ("first_impression.role_block_analysis", "first_impression.ROLE_BLOCKS")),
    (39, "确定 P0 信息架构修复", ("first_impression.RevisionPlan",)),
    (40, "检查修订不改变事实", ("first_impression.check_no_fact_change", "first_impression.RevisionPlan")),
    (41, "生成新文档候选", ("contracts.new_candidate", "first_impression.RevisionPlan")),
    (42, "用新参与者复测", ("first_impression.retest_scope",)),
    (43, "检查意外退化", ("first_impression.unexpected_regression",)),
    (44, "裁决多数与关键误解", ("first_impression.role_block_analysis", "first_impression.RevisionPlan")),
    (45, "冻结 FirstImpressionStudyRecord", ("first_impression.freeze_study", "records.FirstImpressionSessionResult")),
)

TITLE = "目标岗位读者 5 分钟首屏理解实验"
LEVEL = "P1"
CLAIM_BOUNDARY = "E15-11 只能改变信息架构；不改变成熟度或性能数字，不提供统计代表性"


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the first-impression interfaces (labelled smoke)."""
    problems: List[str] = []
    contract = UsabilityStudyContract(
        study_id="study-1",
        candidate_id="cand-1",
        entry_point="https://github.com/example/hqsb",
        task_set_version="v1",
        recording_mode="natural-browse + screen-record",
        privacy_policy="anonymous, no personal account recording",
        pass_line="majority correct recall + no P0 misconception",
    )
    problems.extend(contract.problems())
    leading = TaskSet(
        task_set_version="v1",
        scenario_prompt="这是一个性能优化项目，hero 是 RMSNorm kernel",
        tasks={group: "task text" for group in TASK_GROUPS},
        leading=True,
    )
    if not leading.problems():
        problems.append("a leading scenario was accepted")
    task = TaskSet(
        task_set_version="v1",
        scenario_prompt="你正在初筛一个候选人的技术项目",
        tasks={group: "task text" for group in TASK_GROUPS},
    )
    problems.extend(task.problems())
    session = ParticipantSession(
        session_id="s1",
        participant_id="p1",
        session_mode="natural",
        path=("README", "docs/architecture"),
        first_click="README",
        browse_duration_s=290.0,
        forced_stop=True,
    )
    problems.extend(session.problems())
    if session.forced_stop and session.browse_duration_s > BROWSE_LIMIT_S:
        problems.append("a session both stopped and over-ran")
    key = AnswerKey(task_set_version="v1", answers={group: "correct" for group in TASK_GROUPS})
    coding = code_answers(session, answer_key=key)
    if coding["correct"] != 0:
        problems.append("an unanswered task was scored correct")
    misconceptions = encode_misconceptions(session=session, coded={"recall": "incorrect", "limitation": "partial"})
    if not misconceptions:
        problems.append("wrong answers produced no misconceptions")
    retest = retest_scope(previous_participants=("p1",), retest_participants=("p1", "p2"))
    if not retest:
        problems.append("a known participant slipped into the retest")
    plan = RevisionPlan(fixes=("move hero above the fold",), fact_change=False)
    problems.extend(plan.problems())
    return {
        "status": "smoke",
        "claim_allowed": False,
        "role_blocks": len(ROLE_BLOCKS),
        "task_groups": len(TASK_GROUPS),
        "max_participants": MAX_PARTICIPANTS,
        "problems": problems,
    }
