"""E15-07 — 3–5 minute demo, fault injection and honest degradation.

Protocol: ``docs/stage_experiments/details/S15/E15-07_demo_fault_fallback_rehearsal.md``
(45 steps).

A demo is a small service with an SLO, a fault model and a recovery path — not a
screencast of one lucky run.  The module implements the vocabulary and the checks:

* :class:`DemoObjective` and :class:`SegmentBudget` — what the audience must
  remember, with the 3–5 minute budget split by segment (steps 1, 4);
* :data:`MEASUREMENT_STATES` — live / regenerated / cached / prerecorded /
  simulated, each with a visual and verbal label, because "prerecorded" that is
  not labelled is exactly "fake live" (step 5, §3.2);
* :class:`StateMachine` — the preflight/normal/fault/fallback/evidence/complete
  states with maximum dwell times (step 6);
* :class:`DemoEnvironment`, :func:`clean_start_check`,
  :func:`artifact_integrity_check`, :func:`privacy_check`,
  :func:`preflight_report` — steps 12–16;
* :class:`FallbackThresholds` — frozen *before* the run, so a live command over
  budget is cancelled instead of "waited out" (step 17, §9.1);
* :class:`DemoScript` + :func:`check_script_provenance` — every action names its
  claim id and its tested command; a demo may not invent a second, untested
  command set (steps 18–21);
* :data:`FAULT_ACTION_MATRIX` — the §9 matrix: each fault → detection signal →
  legal action → permitted wording → forbidden wording (steps 25–32, 33);
* :class:`RehearsalSession` — timing, faults, observer state-accuracy and the
  evidence drill-down trial (steps 22–24, 34–35, 39–40);
* :class:`DemoRecordingManifest` — the recording's run identity, cuts, speed-ups,
  subtitles and frame-level live-state (step 42, §9.2);
* :func:`check_no_external_writes`, :func:`check_observer_state_accuracy`,
  :func:`check_recording_supply_chain` — steps 37–38, 43.

Nothing here runs a demo, plays a video or injects a fault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release import records as rec
from hqsb.release.identity import SCHEMA_PREFIX, canonical_digest, is_digest

#: Demo measurement states (§3.2).
MEASUREMENT_STATES: Tuple[str, ...] = rec.DEMO_MEASUREMENT_STATES

#: Fault scenarios (§9 matrix left column).
FAULT_SCENARIOS: Tuple[str, ...] = rec.DEMO_FAULT_SCENARIOS

#: State-machine states (step 6).
STATES: Tuple[str, ...] = (
    "preflight",
    "normal",
    "fault_detected",
    "fallback_selected",
    "evidence_shown",
    "complete",
    "abort",
)

#: Normal-path segments the budget must cover (step 7).
NORMAL_PATH_SEGMENTS: Tuple[str, ...] = (
    "identity",
    "reference",
    "profile",
    "candidate",
    "result",
    "evidence",
)

#: SLO fields that must be frozen before a rehearsal (§9.1).
SLO_FIELDS: Tuple[str, ...] = (
    "t_preflight",
    "t_live_action",
    "t_fault_detect",
    "t_fallback_complete",
    "t_normal_total",
    "t_fault_total",
    "t_evidence_lookup",
    "manual_actions_allowed",
    "cleanup_criterion",
)

#: The §9 fault → action → wording matrix.
FAULT_ACTION_MATRIX: Tuple[Mapping[str, str], ...] = (
    {
        "fault": "no_device",
        "detection": "capability unavailable",
        "legal_action": "CPU/report + historical evidence",
        "allowed_wording": "当前无设备，下面重建已验证结果",
        "forbidden_wording": "现在运行了 GPU",
    },
    {
        "fault": "busy_device",
        "detection": "health/memory preflight",
        "legal_action": "stop live, show capacity evidence",
        "allowed_wording": "现场资源不满足协议",
        "forbidden_wording": "杀他人进程/缩工作量偷跑",
    },
    {
        "fault": "cache_miss",
        "detection": "cache event/ETA",
        "legal_action": "达阈值切预录",
        "allowed_wording": "当前在编译，播放同版本预录",
        "forbidden_wording": "把预录完成当刚才命令完成",
    },
    {
        "fault": "network_down",
        "detection": "fetch/remote health",
        "legal_action": "本地只读 evidence",
        "allowed_wording": "离线查看冻结 bundle",
        "forbidden_wording": "使用旧网页截图无版本",
    },
    {
        "fault": "model_missing",
        "detection": "manifest/source error",
        "legal_action": "不换模型，切历史 run",
        "allowed_wording": "模型未就绪，身份门阻断",
        "forbidden_wording": "临时用 tiny 模型冒充",
    },
    {
        "fault": "bad_asset",
        "detection": "integrity gate",
        "legal_action": "隔离坏资产，切验证副本",
        "allowed_wording": "完整性失败，未消费",
        "forbidden_wording": "忽略警告继续",
    },
    {
        "fault": "timeout",
        "detection": "watchdog",
        "legal_action": "cancel + cleanup + prerecorded",
        "allowed_wording": "现场 smoke 超时",
        "forbidden_wording": "继续等到面试结束",
    },
    {
        "fault": "web_unavailable",
        "detection": "display health",
        "legal_action": "terminal/local PDF",
        "allowed_wording": "切本地同 digest 视图",
        "forbidden_wording": "口头报无法展示的数字",
    },
    {
        "fault": "evidence_link_down",
        "detection": "URI check",
        "legal_action": "本地 manifest/claim cache",
        "allowed_wording": "远端不可用，使用冻结副本",
        "forbidden_wording": "声称链接正常",
    },
    {
        "fault": "correctness_fail",
        "detection": "gate result",
        "legal_action": "停止性能叙事",
        "allowed_wording": "本轮无性能资格",
        "forbidden_wording": "仍展示更快数字",
    },
)


@dataclass
class DemoObjective:
    """What the audience must remember in five minutes (step 1)."""

    objective_id: str
    hero_claim_id: str = ""
    one_sentence_value: str = ""
    key_judgement: str = ""
    limitation: str = ""
    negative_story_claim_id: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("objective_id", "hero_claim_id", "one_sentence_value", "key_judgement", "limitation"):
            if not getattr(self, name):
                findings.append(f"DemoObjective: {name} is required")
        if self.hero_claim_id == self.negative_story_claim_id:
            findings.append("DemoObjective: the hero claim and the failure story must be different claims")
        return findings


@dataclass
class SegmentBudget:
    """The frozen 3–5 minute budget (step 4)."""

    segments: Mapping[str, float] = field(default_factory=dict)
    fault_switch_margin_s: float = 30.0

    MIN_TOTAL_S = 180.0
    MAX_TOTAL_S = 300.0

    def problems(self) -> List[str]:
        findings: List[str] = []
        for segment in NORMAL_PATH_SEGMENTS:
            if segment not in self.segments:
                findings.append(f"SegmentBudget: segment {segment!r} has no budget")
        total = float(sum(self.segments.values()))
        if total + self.fault_switch_margin_s > self.MAX_TOTAL_S:
            findings.append(
                f"SegmentBudget: {total + self.fault_switch_margin_s:.0f}s > {self.MAX_TOTAL_S:.0f}s; "
                "a normal script that only just fits has no room for any fault"
            )
        if total < self.MIN_TOTAL_S:
            findings.append("SegmentBudget: the normal path under-fills the minimum 3 minutes")
        if self.fault_switch_margin_s <= 0:
            findings.append("SegmentBudget: the fault-switch margin must be positive")
        for segment, seconds in self.segments.items():
            if seconds <= 0:
                findings.append(f"SegmentBudget: segment {segment!r} has a non-positive budget")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "segments": {key: self.segments[key] for key in sorted(self.segments)},
            "total_s": float(sum(self.segments.values())),
            "fault_switch_margin_s": self.fault_switch_margin_s,
        }


@dataclass
class StateMachine:
    """The demo state machine with maximum dwell times (step 6)."""

    states: Tuple[str, ...] = STATES
    max_dwell_s: Mapping[str, float] = field(default_factory=dict)

    def problems(self) -> List[str]:
        findings: List[str] = []
        for state in STATES:
            if state not in self.states:
                findings.append(f"StateMachine: state {state!r} is missing")
            if state not in self.max_dwell_s:
                findings.append(f"StateMachine: state {state!r} has no maximum dwell time")
        unknown = [state for state in self.max_dwell_s if state not in STATES]
        if unknown:
            findings.append(f"StateMachine: unknown dwell states {', '.join(sorted(unknown))}")
        return findings

    def next_state(self, current: str, event: str) -> str:
        if current not in STATES:
            raise ConfigError(f"unknown state {current!r}")
        transitions: Mapping[Tuple[str, str], str] = {
            ("preflight", "ok"): "normal",
            ("preflight", "fail"): "fallback_selected",
            ("normal", "fault"): "fault_detected",
            ("fault_detected", "threshold"): "fallback_selected",
            ("normal", "evidence"): "evidence_shown",
            ("fallback_selected", "evidence"): "evidence_shown",
            ("evidence_shown", "done"): "complete",
            ("fault_detected", "abort"): "abort",
        }
        next_state = transitions.get((current, event))
        if next_state is None:
            raise ConfigError(f"no transition from {current!r} on event {event!r}")
        return next_state


@dataclass
class DemoEnvironment:
    """The frozen demonstration environment (step 12)."""

    environment_id: str
    host_kind: str = ""
    accelerator_remote: str = ""
    network: str = ""
    terminal: str = ""
    browser: str = ""
    resolution: str = ""
    shell: str = ""
    package_digest: str = ""
    account_permissions: str = ""
    personal_notifications_off: bool = False

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("environment_id", "host_kind", "terminal", "shell"):
            if not getattr(self, name):
                findings.append(f"DemoEnvironment: {name} is required (a rehearsal in an incomparable environment proves nothing)")
        if not self.personal_notifications_off:
            findings.append("DemoEnvironment: personal notifications must be off before sharing the screen (step 15)")
        return findings


def clean_start_check(*, processes_closed: bool, terminal_cleared: bool, no_author_server: bool, no_stale_output: bool) -> List[str]:
    """The clean-start check (step 13)."""
    findings: List[str] = []
    for name, ok in (
        ("processes_closed", processes_closed),
        ("terminal_cleared", terminal_cleared),
        ("no_author_server", no_author_server),
        ("no_stale_output", no_stale_output),
    ):
        if not ok:
            findings.append(f"clean start: {name} failed; the script depends on pre-run state")
    return findings


def artifact_integrity_check(*, assets: Sequence[Mapping[str, Any]], verify: Callable[[str], bool]) -> List[str]:
    """Verify package/model/bundle/video/figure hashes before the demo (step 14)."""
    findings: List[str] = []
    for asset in assets:
        name = str(asset.get("name", ""))
        if not is_digest(str(asset.get("sha256", ""))):
            findings.append(f"{name}: no sha256 recorded")
        if not verify(name):
            findings.append(f"{name}: integrity check failed (a corrupted file would only surface mid-demo)")
    return findings


def privacy_check(*, shell_history: Sequence[str], desktop: Sequence[str], bookmarks: Sequence[str]) -> List[str]:
    """The pre-share privacy check (step 15)."""
    findings: List[str] = []
    if shell_history:
        findings.append("shell history must be cleared before sharing the screen")
    for name, items in (("desktop", desktop), ("bookmarks", bookmarks)):
        for item in items:
            if any(token in item for token in ("token", "密钥", "secret", "pass")):
                findings.append(f"{name} contains a credential-looking item: {item!r}")
    return findings


def preflight_report(*, capability: str, model_reachable: bool, cache_state: str, disk_ok: bool, profiler_ok: bool, remote_session: bool) -> Dict[str, Any]:
    """Structured preflight output (step 16)."""
    checks = {
        "capability": capability,
        "model_reachable": model_reachable,
        "cache_state": cache_state,
        "disk_ok": disk_ok,
        "profiler_ok": profiler_ok,
        "remote_session": remote_session,
    }
    return {
        "checks": checks,
        "verdict": "ready" if all(value not in (False, "", "unavailable") for value in checks.values()) else "not_ready",
        "note": "preflight 输出结构化结论；主流程不能等 preflight 失败后才失败",
    }


@dataclass
class FallbackThresholds:
    """Frozen fallback thresholds (§9.1, step 17)."""

    thresholds: Mapping[str, float] = field(default_factory=dict)
    fields: Tuple[str, ...] = SLO_FIELDS

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in SLO_FIELDS:
            if name not in self.thresholds:
                findings.append(f"FallbackThresholds: SLO field {name!r} is not frozen")
            elif self.thresholds[name] < 0:
                findings.append(f"FallbackThresholds: {name} may not be negative")
        return findings

    def live_command_exceeded(self, elapsed_s: float) -> bool:
        limit = self.thresholds.get("t_live_action")
        return limit is not None and elapsed_s > limit


@dataclass
class ScriptAction:
    """One demo-script action (step 18)."""

    purpose: str
    visible_output: str
    claim_id: str = ""
    estimated_seconds: float = 0.0
    next_state: str = ""
    command_ref: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.purpose or not self.visible_output:
            findings.append("ScriptAction: purpose and visible output are required")
        if not self.claim_id:
            findings.append("ScriptAction: every displayed number must bind to a claim id (§18)")
        if self.estimated_seconds <= 0:
            findings.append("ScriptAction: a positive estimated duration is required for the budget")
        if self.next_state not in STATES:
            findings.append(f"ScriptAction: unknown next state {self.next_state!r}")
        return findings


@dataclass
class DemoScript:
    """The per-sentence script with claim binding (steps 18–21)."""

    actions: Tuple[ScriptAction, ...] = ()
    tested_command_refs: Tuple[str, ...] = ()
    figure_ids: Tuple[str, ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.actions:
            findings.append("DemoScript: an empty script is not a demo")
        for action in self.actions:
            findings.extend(action.problems())
        refs = {action.command_ref for action in self.actions if action.command_ref}
        untested = sorted(refs - set(self.tested_command_refs))
        if untested:
            findings.append(
                f"DemoScript: commands not tested by E15-04: {', '.join(untested)} "
                "(a demo may not carry a second, untested command set — step 19)"
            )
        return findings


def check_script_provenance(script: DemoScript, *, verified_claims: Sequence[str], verified_figures: Sequence[str]) -> List[str]:
    """Numbers from E15-01, figures from E15-06, commands from E15-04 (steps 19–21)."""
    findings = list(script.problems())
    for action in script.actions:
        if action.claim_id not in verified_claims:
            findings.append(f"action claim {action.claim_id!r} is not a verified claim (step 20)")
    for figure_id in script.figure_ids:
        if figure_id not in verified_figures:
            findings.append(f"figure {figure_id!r} was not audited by E15-06 (step 21)")
    return findings


@dataclass
class RehearsalSession:
    """One rehearsal with timing, faults and observer outcome (steps 22–24, 39–40)."""

    session_id: str
    scenario: str
    duration_s: float = 0.0
    segments: Tuple[Mapping[str, Any], ...] = ()
    faults: Tuple[str, ...] = ()
    measurement_states_shown: Tuple[str, ...] = ()
    observer_state_accuracy: Optional[float] = None
    evidence_lookup: Optional[Mapping[str, Any]] = None
    undocumented_actions: Tuple[str, ...] = ()
    fallback_selected: str = ""
    fallback_time_s: Optional[float] = None
    external_writes: Tuple[str, ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.session_id or not self.scenario:
            findings.append("RehearsalSession: session_id and scenario are required")
        if self.scenario not in FAULT_SCENARIOS:
            findings.append(f"RehearsalSession: unknown scenario {self.scenario!r}")
        if self.duration_s < 0:
            findings.append("RehearsalSession: negative duration")
        unknown = [state for state in self.measurement_states_shown if state not in MEASUREMENT_STATES]
        if unknown:
            findings.append(f"RehearsalSession: unknown measurement states {', '.join(sorted(unknown))}")
        if self.observer_state_accuracy is not None and not 0.0 <= self.observer_state_accuracy <= 1.0:
            findings.append("RehearsalSession: observer accuracy must be in [0, 1]")
        if self.external_writes:
            findings.append(f"RehearsalSession: the demo changed external state: {', '.join(self.external_writes)}")
        return findings

    def passed(self, *, budget: SegmentBudget) -> bool:
        if self.scenario == "normal":
            return budget.MIN_TOTAL_S <= self.duration_s <= budget.MAX_TOTAL_S and not self.external_writes
        return bool(self.fallback_selected) and not self.external_writes and self.fallback_time_s is not None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "scenario": self.scenario,
            "duration_s": self.duration_s,
            "segments": [dict(item) for item in self.segments],
            "faults": list(self.faults),
            "measurement_states_shown": list(self.measurement_states_shown),
            "observer_state_accuracy": self.observer_state_accuracy,
            "evidence_lookup": dict(self.evidence_lookup) if self.evidence_lookup else None,
            "undocumented_actions": list(self.undocumented_actions),
            "fallback_selected": self.fallback_selected,
            "fallback_time_s": self.fallback_time_s,
            "external_writes": list(self.external_writes),
        }


def check_no_external_writes(*, writes: Sequence[str], allowed: Sequence[str] = ()) -> List[str]:
    """The demo must be read-only against the outside world (step 38)."""
    findings: List[str] = []
    for action in writes:
        if action in allowed:
            continue
        if any(verb in action for verb in ("create cloud", "publish release", "submit issue", "清理共享", "kill")):
            findings.append(f"external side effect not allowed during a demo: {action!r}")
    return findings


def check_observer_state_accuracy(judgements: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Whether observers can tell live from prerecorded (step 34)."""
    if not judgements:
        raise ConfigError("check_observer_state_accuracy: no judgements (one observer is not a measurement)")
    correct = sum(1 for item in judgements if item.get("actual_state") == item.get("judged_state"))
    return {
        "judgements": len(judgements),
        "correct": correct,
        "accuracy": correct / len(judgements),
        "note": "作者自己判断“已经很明显”不算数；必须由观察者判断",
    }


def analyse_reliability(sessions: Sequence[RehearsalSession]) -> Dict[str, Any]:
    """Completion rate and timing distribution across sessions (step 40)."""
    if not sessions:
        raise ConfigError("analyse_reliability: no sessions (one rehearsal is not a reliability estimate)")
    durations = sorted(session.duration_s for session in sessions)
    completed = sum(1 for session in sessions if session.duration_s > 0)
    from hqsb.release.telemetry import _percentile

    payload: Dict[str, Any] = {
        "sessions": len(sessions),
        "completed": completed,
        "completion_rate": completed / len(sessions),
        "min_s": durations[0],
        "max_s": durations[-1],
        "overran_segments": [
            session.session_id
            for session in sessions
            if any(seg.get("overrun_s", 0) > 0 for seg in session.segments)
        ],
        "note": "只报最快一遍等于伪造；报告 session 数、完成率与时间分布，小样本明确限制",
    }
    if len(durations) >= 5:
        payload["median_s"] = _percentile(durations, 0.5)
        payload["p95_s"] = _percentile(durations, 0.95)
    return payload


@dataclass
class DemoRecordingManifest:
    """A recording is a release artifact, not a clip (step 42, §9.2)."""

    recording_id: str
    source_run: str = ""
    candidate_id: str = ""
    cuts: Tuple[Tuple[float, float], ...] = ()
    speed_ups: Tuple[str, ...] = ()
    subtitle_version: str = ""
    frame_level_live_state: bool = False
    sha256: str = ""
    license: str = ""
    hardware_label: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("recording_id", "source_run", "candidate_id", "sha256", "hardware_label"):
            if not getattr(self, name):
                findings.append(f"DemoRecordingManifest: {name} is required")
        if self.sha256 and not is_digest(self.sha256):
            findings.append("DemoRecordingManifest: sha256 must be sha256:<hex>")
        if not self.frame_level_live_state:
            findings.append(
                "DemoRecordingManifest: frame-level live-state is required; a video without it is a "
                "concept video, not run evidence (§9.2)"
            )
        if self.speed_ups and not self.subtitle_version:
            findings.append("DemoRecordingManifest: speed-ups need the subtitle version to stay honest")
        if self.cuts and not self.subtitle_version:
            findings.append("DemoRecordingManifest: edits must be visible (subtitle/time-jump marker)")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "source_run": self.source_run,
            "candidate_id": self.candidate_id,
            "cuts": [list(cut) for cut in self.cuts],
            "speed_ups": list(self.speed_ups),
            "subtitle_version": self.subtitle_version,
            "frame_level_live_state": self.frame_level_live_state,
            "sha256": self.sha256,
            "license": self.license,
            "hardware_label": self.hardware_label,
        }


def check_recording_supply_chain(recording: DemoRecordingManifest, *, candidate_digest: str) -> List[str]:
    """The recording must belong to the current candidate and have a rights decision (step 43)."""
    findings = list(recording.problems())
    if recording.candidate_id and recording.candidate_id != candidate_digest[:12]:
        findings.append("the recording was made on a different release candidate than the one being presented")
    if not recording.license:
        findings.append("the recording needs a license decision (media rights are part of E15-05)")
    return findings


def fault_matrix_for(scenario: str) -> Mapping[str, str]:
    """Look up one row of the fault-action matrix."""
    for row in FAULT_ACTION_MATRIX:
        if row["fault"] == scenario:
            return dict(row)
    raise ConfigError(f"unknown fault scenario {scenario!r}")


def freeze_rehearsal(
    *, objective: DemoObjective, budget: SegmentBudget, record: rec.DemoSessionResult, sessions: Sequence[RehearsalSession]
) -> Dict[str, Any]:
    """Freeze the rehearsal record, refusing a normal PASS outside 3–5 minutes (step 45)."""
    problems = list(objective.problems()) + list(budget.problems()) + list(record.validate())
    for session in sessions:
        problems.extend(session.problems())
    if problems:
        raise ConfigError("refusing to freeze the rehearsal: " + "; ".join(problems[:5]))
    return {
        "schema_version": f"{SCHEMA_PREFIX}.demo-rehearsal.v1",
        "objective": objective.one_sentence_value,
        "budget": budget.as_dict(),
        "record": record.as_dict(),
        "sessions": [session.as_dict() for session in sessions],
        "record_digest": canonical_digest(record.as_dict()),
    }


# ── PROTOCOL_STEPS (45) ─────────────────────────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 DemoObjective", ("demo.DemoObjective",)),
    (2, "选择单一 Hero Claim", ("demo.DemoObjective", "claims.adjudicate_release_eligibility")),
    (3, "定义目标观众", ("demo.DemoObjective", "narrative.role_matrix")),
    (4, "冻结 3–5 分钟时间预算", ("demo.SegmentBudget", "demo.NORMAL_PATH_SEGMENTS")),
    (5, "建立 Demo 状态标签", ("demo.MEASUREMENT_STATES",)),
    (6, "建立场景状态机", ("demo.StateMachine", "demo.STATES")),
    (7, "选择正常路径", ("demo.NORMAL_PATH_SEGMENTS", "demo.DemoScript")),
    (8, "选择最小 live 工作量", ("demo.SegmentBudget", "hero_replay.ConfirmatoryPlan")),
    (9, "准备离线 evidence 路径", ("demo.artifact_integrity_check", "contracts.PublicEvidenceBundle")),
    (10, "准备预录路径", ("demo.DemoRecordingManifest",)),
    (11, "准备 CPU fallback", ("quickstart.SCENARIO_MATRIX", "demo.fault_matrix_for")),
    (12, "冻结演示环境", ("demo.DemoEnvironment",)),
    (13, "执行干净启动检查", ("demo.clean_start_check",)),
    (14, "执行 artifact 完整性检查", ("demo.artifact_integrity_check",)),
    (15, "执行隐私和通知检查", ("demo.privacy_check", "demo.DemoEnvironment")),
    (16, "执行设备和网络 preflight", ("demo.preflight_report",)),
    (17, "冻结 fallback 决策阈值", ("demo.FallbackThresholds", "demo.SLO_FIELDS")),
    (18, "编写逐句 Demo 脚本", ("demo.DemoScript", "demo.ScriptAction")),
    (19, "验证所有命令来自 E15-04", ("demo.check_script_provenance", "docs_gate.run_code_blocks")),
    (20, "验证所有数字来自 E15-01", ("demo.check_script_provenance", "contracts.ClaimRecord")),
    (21, "验证所有图来自 E15-06", ("demo.check_script_provenance", "figures.FigureSpec")),
    (22, "执行正常路径冷启动演练", ("demo.RehearsalSession", "demo.SegmentBudget")),
    (23, "执行正常路径热启动演练", ("demo.RehearsalSession", "demo.RehearsalSession.passed")),
    (24, "重复正常演练", ("demo.analyse_reliability",)),
    (25, "注入无 GPU/NPU", ("demo.fault_matrix_for", "demo.FAULT_ACTION_MATRIX")),
    (26, "注入设备忙或显存不足", ("demo.fault_matrix_for", "demo.check_no_external_writes")),
    (27, "注入 cache miss", ("demo.fault_matrix_for",)),
    (28, "注入网络不可用", ("demo.fault_matrix_for",)),
    (29, "注入模型缺失或 hash 错", ("demo.fault_matrix_for", "hero_replay.ArtifactIdentity")),
    (30, "注入 live command 超时", ("demo.FallbackThresholds.live_command_exceeded",)),
    (31, "注入坏 release/evidence asset", ("demo.artifact_integrity_check",)),
    (32, "注入网页/投屏不可用", ("demo.fault_matrix_for",)),
    (33, "验证 fallback 口头诚实性", ("demo.FAULT_ACTION_MATRIX", "demo.check_observer_state_accuracy")),
    (34, "验证状态识别", ("demo.check_observer_state_accuracy",)),
    (35, "执行证据 drill-down", ("telemetry.EvidenceLookupTrial", "figures.check_access_friction")),
    (36, "展示一个失败或无收益案例", ("demo.DemoObjective.negative_story_claim_id",)),
    (37, "验证现场输出可读性", ("docs_gate.check_accessibility",)),
    (38, "验证演示不改变外部状态", ("demo.check_no_external_writes",)),
    (39, "记录每次动作与偏差", ("telemetry.ActionLog", "demo.RehearsalSession")),
    (40, "分析时长和可靠性", ("demo.analyse_reliability", "telemetry.TimeToEventSummary")),
    (41, "修订脚本和故障树", ("demo.fault_matrix_for", "demo.DemoScript.problems")),
    (42, "录制候选正式视频", ("demo.DemoRecordingManifest",)),
    (43, "执行录屏供应链检查", ("demo.check_recording_supply_chain", "supply_chain.RightsDecision")),
    (44, "让非作者按脚本演示", ("demo.RehearsalSession", "quickstart.InterventionLog")),
    (45, "冻结 DemoRehearsalRecord", ("demo.freeze_rehearsal", "records.DemoSessionResult")),
)

TITLE = "3–5 分钟 Demo、故障注入与诚实降级"
LEVEL = "P0"
CLAIM_BOUNDARY = "E15-07 证明演示在有限时间和故障下可靠、诚实；不证明观众已经理解技术（E15-08/E15-11）"


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the demo interfaces (labelled smoke)."""
    problems: List[str] = []
    objective = DemoObjective(
        objective_id="demo-1",
        hero_claim_id="CLM-hero",
        one_sentence_value="profile→kernel→model 的因果闭环",
        key_judgement="micro 收益不能冒充 model/service 收益",
        limitation="本故事只在目标设备成立",
        negative_story_claim_id="CLM-neg",
    )
    problems.extend(objective.problems())
    budget = SegmentBudget(
        segments={segment: 30.0 for segment in NORMAL_PATH_SEGMENTS},
        fault_switch_margin_s=30.0,
    )
    problems.extend(budget.problems())
    over = SegmentBudget(segments={segment: 60.0 for segment in NORMAL_PATH_SEGMENTS}, fault_switch_margin_s=10.0)
    if not over.problems():
        problems.append("an over-budget normal script was accepted")
    machine = StateMachine(max_dwell_s={state: 10.0 for state in STATES})
    if machine.next_state("preflight", "ok") != "normal":
        problems.append("the state machine did not transition preflight→normal")
    script = DemoScript(
        actions=(
            ScriptAction(
                purpose="show identity",
                visible_output="candidate digest",
                claim_id="CLM-hero",
                estimated_seconds=20.0,
                next_state="normal",
                command_ref="cmd-1",
            ),
        ),
        tested_command_refs=("cmd-1",),
    )
    if script.problems():
        problems.append("a well-formed script was rejected")
    untested = DemoScript(
        actions=(ScriptAction(purpose="x", visible_output="y", claim_id="CLM-hero", estimated_seconds=1.0, next_state="normal", command_ref="cmd-2"),),
        tested_command_refs=("cmd-1",),
    )
    if not untested.problems():
        problems.append("an untested command slipped into the script")
    session = RehearsalSession(session_id="s1", scenario="normal", duration_s=200.0)
    if not session.passed(budget=budget):
        problems.append("an in-budget normal rehearsal did not pass")
    quick = RehearsalSession(session_id="s2", scenario="normal", duration_s=60.0)
    if quick.passed(budget=budget):
        problems.append("a 60s normal rehearsal passed (the demo under-fills 3 minutes)")
    recording = DemoRecordingManifest(recording_id="r1", source_run="run-1", candidate_id="cand-1", sha256="sha256:" + "0" * 64, hardware_label="RTX 3090")
    if not recording.problems():
        problems.append("a recording without frame-level live-state was accepted")
    accuracy = check_observer_state_accuracy(
        ({"actual_state": "live", "judged_state": "live"}, {"actual_state": "prerecorded", "judged_state": "live"})
    )
    if accuracy["accuracy"] != 0.5:
        problems.append("observer state accuracy is wrong")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "measurement_states": len(MEASUREMENT_STATES),
        "fault_scenarios": len(FAULT_SCENARIOS),
        "problems": problems,
    }
