"""E12-09: software maturity rubric, engineering cost and tool coverage.

"Feels mature" is not data.  This module turns a maturity evaluation into
reviewable evidence:

* rubric levels 0–4 are anchored per dimension, and ``NOT_EVALUATED`` is a
  separate state from ``0`` — an unmeasured task is neither worst nor best;
* wall / active / blocked / rework time are distinct: a slow build is not a
  manual effort, and manual debugging is not a download wait;
* a silent fallback is high risk even when the task "succeeded";
* agreement between reviewers is reported as exact/adjacent with the disputed
  items — never averaged away into one number;
* the primary report shows the raw dimension×task vector; any aggregate must
  expose its weights, its coverage and its sensitivity, and may never fold
  maturity into a performance score.

Nothing here installs, builds or runs software.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.evaluation.identity import stable_id
from hqsb.evaluation.records import TABLE_SCHEMAS

EXPERIMENT_ID = "E12-09"
TITLE = "软件成熟度、开发/维护成本与可复核 Rubric"
CLAIM_BOUNDARY = (
    "本实验通过证明工程成熟度被可复核地评价，"
    "不证明任何软件栈已达到生产级可靠性；后者需要 S13 的部署与故障实验。"
)

MATURITY_DIMENSIONS: Tuple[str, ...] = (
    "installability",
    "build_compile_lifecycle",
    "framework_model_compatibility",
    "debug_profiling_observability",
    "reliability_and_recovery",
    "documentation_support",
    "upgrade_maintenance",
    "portability_lockin",
)

RUBRIC_LEVELS: Tuple[int, ...] = (0, 1, 2, 3, 4)
NOT_EVALUATED = "NOT_EVALUATED"

#: Anchored rubric (dimension x level -> observable anchor).
RUBRIC_ANCHORS: Mapping[Tuple[str, int], str] = {
    ("installability", 0): "task cannot be completed with a feasible recovery path (or evidence missing)",
    ("installability", 1): "completed only with author/vendor help or an undocumented, high-risk workaround",
    ("installability", 2): "completed with several failures, manual steps, weak diagnostics or version fragility",
    ("installability", 3): "reproducible from documented/automated steps with a few explicit workarounds",
    ("installability", 4): "stable, automated, error-diagnosable across operators/sessions",
    ("build_compile_lifecycle", 0): "no reproducible build, or no evidence",
    ("build_compile_lifecycle", 1): "build only with hand-carried patches",
    ("build_compile_lifecycle", 2): "build succeeds after multiple failures with manual steps",
    ("build_compile_lifecycle", 3): "AOT/JIT and cache are repeatable with clear diagnostics",
    ("build_compile_lifecycle", 4): "cold rebuild reproduces the same artifact identity automatically",
    ("debug_profiling_observability", 0): "a request cannot be mapped to op/kernel/device",
    ("debug_profiling_observability", 1): "mapping requires the author's manual digging",
    ("debug_profiling_observability", 2): "mapping possible but partly manual",
    ("debug_profiling_observability", 3): "request → op → kernel/device with a saved evidence chain",
    ("debug_profiling_observability", 4): "automated, exportable, correlated to counters and ranks",
    ("documentation_support", 0): "no usable documentation or no evidence of following it",
    ("documentation_support", 1): "only author knowledge, undocumented",
    ("documentation_support", 2): "documentation exists but is ambiguous/stale in places",
    ("documentation_support", 3): "a second session reproduces from the documentation",
    ("documentation_support", 4): "documentation is versioned, tested, and complete",
}

_LEVEL_ANCHOR_DEFAULT: Mapping[int, str] = {
    0: "failed or unevaluated (a NOT_EVALUATED is never a 0)",
    1: "author/vendor help or a risky workaround required",
    2: "completed with repeated failures and manual steps",
    3: "reproducible from documentation with a few explicit workarounds",
    4: "stable, automated and diagnosable across sessions",
}


def anchor_for(dimension: str, level: int) -> str:
    if dimension not in MATURITY_DIMENSIONS:
        raise ConfigError(f"unknown maturity dimension {dimension!r}")
    if level not in RUBRIC_LEVELS:
        raise ConfigError(f"unknown rubric level {level}")
    return RUBRIC_ANCHORS.get((dimension, level), _LEVEL_ANCHOR_DEFAULT[level])


# ── tasks ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    dimension: str
    description: str
    inputs: Tuple[str, ...]
    expected_output: str
    success_criteria: str
    time_budget_s: int
    allowed_help: Tuple[str, ...]
    termination_condition: str
    applicability: str = "required"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.dimension not in MATURITY_DIMENSIONS:
            problems.append(f"unknown dimension {self.dimension!r}")
        for name in ("description", "expected_output", "success_criteria", "termination_condition"):
            if not getattr(self, name):
                problems.append(f"task {self.task_id} is missing {name!r}")
        if self.time_budget_s <= 0:
            problems.append(f"task {self.task_id} needs a positive time budget")
        if self.applicability not in ("required", "optional", "not_applicable"):
            problems.append(f"unknown applicability {self.applicability!r}")
        return problems


STANDARD_TASKS: Tuple[TaskSpec, ...] = (
    TaskSpec("read_compat_matrix", "installability", "read the compatibility matrix and prepare a clean environment", (), "compatibility checklist", "a matching base environment is prepared", 900, (), "environment prepared"),
    TaskSpec("install_stack", "installability", "install driver/runtime/framework", (), "versioned stack", "the stack loads", 3600, ("vendor docs",), "stack loads"),
    TaskSpec("device_smoke", "build_compile_lifecycle", "verify the device with a minimal op", (), "verified smoke", "correctness passes", 600, (), "smoke passes"),
    TaskSpec("load_model", "framework_model_compatibility", "load the frozen ModelArtifact", (), "hash-verified load", "hash and memory verified", 1800, (), "model loads"),
    TaskSpec("reference_model_core", "framework_model_compatibility", "run the reference model-core", (), "reference tokens", "tokens match the reference", 1800, (), "tokens match"),
    TaskSpec("enable_quant", "framework_model_compatibility", "enable the target precision/quant", (), "quant engine", "quality gate passes", 1800, (), "quality passes"),
    TaskSpec("custom_kernel", "build_compile_lifecycle", "build and load a custom kernel", (), "dispatched kernel", "actual backend recorded", 2400, (), "kernel dispatches"),
    TaskSpec("graph_compiler", "build_compile_lifecycle", "run the graph/compiler path", (), "captured graph", "graph and lowering visible", 2400, (), "graph visible"),
    TaskSpec("start_service", "framework_model_compatibility", "start the service and run a fixed trace", (), "served tokens", "streaming tokens match", 1800, (), "tokens match"),
    TaskSpec("profile_slow_request", "debug_profiling_observability", "profile one slow request to op/kernel/device", (), "evidence chain", "request → op → kernel mapped", 1800, (), "mapping produced"),
    TaskSpec("multi_device", "framework_model_compatibility", "multi-device collective/model", (), "cross-rank match", "collective correctness", 3600, (), "collective passes", applicability="optional"),
    TaskSpec("inject_failure", "reliability_and_recovery", "inject a common failure and recover", (), "recovery record", "recovered with a recorded path", 1800, (), "recovered"),
    TaskSpec("cold_rebuild", "build_compile_lifecycle", "rebuild from a cold cache", (), "same artifact identity", "artifact identity matches", 2400, (), "identity matches"),
    TaskSpec("minor_upgrade", "upgrade_maintenance", "upgrade one component and re-run correctness", (), "regression result", "correctness still passes", 2400, (), "regression passes"),
    TaskSpec("document_reproduce", "documentation_support", "a second session reproduces from the docs", (), "independent run", "the session completes from docs alone", 3600, ("docs",), "second session succeeds"),
)


def validate_applicability(task: TaskSpec, *, capability_evidence: Sequence[Mapping[str, Any]], profile_requires: bool) -> Dict[str, Any]:
    """``NOT_APPLICABLE`` needs capability/profile justification, never hides a failure."""
    if task.applicability != "not_applicable":
        return {"task_id": task.task_id, "applicable": True, "reason": ""}
    supported = any(
        row.get("capability") == task.task_id and row.get("status") == "VERIFIED"
        for row in capability_evidence
    )
    if profile_requires:
        return {
            "task_id": task.task_id,
            "applicable": True,
            "reason": "the business profile requires this capability: NOT_APPLICABLE is not allowed",
        }
    return {
        "task_id": task.task_id,
        "applicable": supported,
        "reason": "" if supported else "capability genuinely absent; recorded as NOT_APPLICABLE with evidence",
    }


# ── timing and sessions ───────────────────────────────────────────────────


@dataclass
class TimingBreakdown:
    wall_s: float
    active_s: float
    blocked_s: float
    rework_s: float

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("wall_s", "active_s", "blocked_s", "rework_s"):
            if getattr(self, name) < 0:
                problems.append(f"{name} must be non-negative")
        if self.active_s + self.blocked_s + self.rework_s > self.wall_s + 1e-9:
            problems.append("active+blocked+rework exceeds wall time: the breakdown is inconsistent")
        return problems


def split_times(events: Sequence[Mapping[str, Any]]) -> TimingBreakdown:
    """Classify session events into the four time buckets (no double counting)."""
    active = blocked = rework = 0.0
    for event in events:
        duration = float(event.get("duration_s", 0.0))
        kind = str(event.get("kind", ""))
        if kind == "active":
            active += duration
        elif kind == "blocked":
            blocked += duration
        elif kind == "rework":
            rework += duration
        else:
            raise ConfigError(f"unknown session-event kind {kind!r}: wall/active/blocked/rework must be explicit")
    wall = active + blocked + rework
    return TimingBreakdown(wall_s=wall, active_s=active, blocked_s=blocked, rework_s=rework)


FAILURE_TAXONOMY: Tuple[str, ...] = (
    "DOC_INCORRECT_OR_MISSING",
    "DEPENDENCY_RESOLUTION",
    "VERSION_ABI_INCOMPATIBLE",
    "INSTALL_OR_PERMISSION",
    "BUILD_OR_COMPILE",
    "ARTIFACT_OR_CACHE",
    "RUNTIME_EXECUTION",
    "NUMERICAL_CORRECTNESS",
    "SILENT_FALLBACK",
    "DYNAMIC_SHAPE_OR_GRAPH",
    "QUANT_OR_LAYOUT",
    "SERVICE_OR_PROTOCOL",
    "DISTRIBUTED_OR_TOPOLOGY",
    "PROFILER_OR_OBSERVABILITY",
    "RESOURCE_OOM_OR_LEAK",
    "UPGRADE_REGRESSION",
    "UNKNOWN",
)


def classify_failure(*, category: str, severity: str, root_cause_confidence: str, evidence_refs: Sequence[str]) -> Dict[str, Any]:
    if category not in FAILURE_TAXONOMY:
        raise ConfigError(f"unknown failure category {category!r}")
    if severity not in ("low", "medium", "high", "critical"):
        raise ConfigError(f"unknown severity {severity!r}")
    if root_cause_confidence not in ("low", "medium", "high"):
        raise ConfigError(f"unknown root-cause confidence {root_cause_confidence!r}")
    if category != "UNKNOWN" and not evidence_refs:
        raise ConfigError(f"category {category!r} without evidence: keep UNKNOWN with low confidence instead")
    if category == "UNKNOWN" and root_cause_confidence != "low":
        raise ConfigError("UNKNOWN is only honest with low confidence")
    return {
        "incident_id": stable_id("incident", {"category": category, "evidence": sorted(evidence_refs)}),
        "category": category,
        "severity": severity,
        "root_cause_confidence": root_cause_confidence,
        "evidence_refs": list(evidence_refs),
    }


@dataclass
class SessionRecord:
    session_id: str
    candidate_id: str
    operator_or_session_id: str
    experience_band: str
    clean_or_warm: str
    task_id: str
    start: str
    end: str
    wall_s: float
    active_s: float
    blocked_s: float
    rework_s: float
    success_status: str
    correctness_status: str
    actual_backend: str
    failure_ids: Tuple[str, ...] = ()
    retry_count: int = 0
    manual_step_count: int = 0
    documentation_source_ids: Tuple[str, ...] = ()
    workaround_ids: Tuple[str, ...] = ()
    artifact_refs: Tuple[str, ...] = ()
    rubric_dimension_levels: Mapping[str, int] = field(default_factory=dict)
    reviewer_ids: Tuple[str, ...] = ()
    limitations: Tuple[str, ...] = ()
    privacy_redaction_status: str = "redacted"

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("operator_or_session_id", "task_id", "actual_backend"):
            if not getattr(self, name):
                problems.append(f"a session record is missing {name!r}")
        if self.clean_or_warm not in ("clean", "warm"):
            problems.append(f"unknown clean_or_warm {self.clean_or_warm!r}")
        if self.success_status not in ("success", "partial", "fail", "not_applicable"):
            problems.append(f"unknown success status {self.success_status!r}")
        if self.correctness_status not in ("pass", "fail", "not_run"):
            problems.append(f"unknown correctness status {self.correctness_status!r}")
        if self.experience_band not in ("novice", "intermediate", "expert"):
            problems.append(f"unknown experience band {self.experience_band!r}")
        for name in ("wall_s", "active_s", "blocked_s", "rework_s"):
            if getattr(self, name) < 0:
                problems.append(f"{name} must be non-negative")
        if self.privacy_redaction_status not in ("redacted", "none_required"):
            problems.append(f"unknown redaction status {self.privacy_redaction_status!r}")
        if self.success_status == "success" and self.correctness_status == "fail":
            problems.append("a task cannot succeed while correctness fails")
        return problems


def task_metrics(session: SessionRecord) -> Dict[str, Any]:
    return {
        "session_id": session.session_id,
        "task_id": session.task_id,
        "success_status": session.success_status,
        "wall_s": session.wall_s,
        "active_s": session.active_s,
        "blocked_s": session.blocked_s,
        "rework_s": session.rework_s,
        "manual_step_count": session.manual_step_count,
        "retry_count": session.retry_count,
        "failure_count": len(session.failure_ids),
        "first_correct_inference_s": session.active_s if session.success_status == "success" else None,
    }


class WorkaroundRegistry:
    def __init__(self) -> None:
        self._rows: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        *,
        workaround_id: str,
        kind: str,
        owner: str,
        upstream_issue: str,
        first_version: str,
        last_verified_version: str,
        risk: str,
        automation_status: str,
        maintenance_hours: float,
        removal_condition: str,
    ) -> Dict[str, Any]:
        if risk not in ("low", "medium", "high"):
            raise ConfigError(f"unknown workaround risk {risk!r}")
        for name, value in (
            ("owner", owner),
            ("upstream_issue", upstream_issue),
            ("first_version", first_version),
            ("last_verified_version", last_verified_version),
            ("removal_condition", removal_condition),
        ):
            if not value:
                raise ConfigError(f"a workaround needs {name!r} (version/owner/risk are not optional)")
        row = {
            "workaround_id": workaround_id,
            "kind": kind,
            "owner": owner,
            "upstream_issue": upstream_issue,
            "first_version": first_version,
            "last_verified_version": last_verified_version,
            "risk": risk,
            "automation_status": automation_status,
            "maintenance_hours": maintenance_hours,
            "removal_condition": removal_condition,
        }
        self._rows[workaround_id] = row
        return row

    def rows(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(self._rows[name] for name in sorted(self._rows))

    def at_risk(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(row for row in self.rows() if row["risk"] == "high")


TOOL_COVERAGE_AREAS: Tuple[str, ...] = (
    "install",
    "build",
    "runtime",
    "model",
    "service",
    "distributed",
    "power",
)


def tool_coverage_row(
    *, candidate_id: str, area: str, tool: str, can_observe: bool, can_export: bool, can_correlate: bool, can_automate: bool, limitation: str
) -> Dict[str, Any]:
    if area not in TOOL_COVERAGE_AREAS:
        raise ConfigError(f"unknown tool coverage area {area!r}")
    return {
        "candidate_id": candidate_id,
        "area": area,
        "tool": tool,
        "can_observe": can_observe,
        "can_export": can_export,
        "can_correlate": can_correlate,
        "can_automate": can_automate,
        "limitation": limitation,
    }


def coverage_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    per_area: Dict[str, List[bool]] = {}
    for row in rows:
        per_area.setdefault(str(row["area"]), []).append(bool(row["can_observe"]))
    return {
        "areas_covered": sorted(per_area),
        "coverage": {name: sum(values) / len(values) for name, values in sorted(per_area.items())},
        "missing_areas": sorted(set(TOOL_COVERAGE_AREAS) - set(per_area)),
    }


def silent_fallback_check(*, requested_backend: str, actual_backend: str, fallback_reason: str) -> Dict[str, Any]:
    if not requested_backend or not actual_backend:
        raise ConfigError("silent-fallback check needs both backends")
    silent = requested_backend != actual_backend and not fallback_reason
    return {
        "requested": requested_backend,
        "actual": actual_backend,
        "silent": silent,
        "risk": "high" if silent else "none",
        "note": "a silent fallback is high risk even when the task 'succeeded'",
    }


# ── rubric ratings and agreement ──────────────────────────────────────────


@dataclass(frozen=True)
class RubricRating:
    session_id: str
    task_id: str
    dimension: str
    level: int
    not_evaluated: bool
    rationale: str
    reviewer: str
    evidence_refs: Tuple[str, ...]

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.dimension not in MATURITY_DIMENSIONS:
            problems.append(f"unknown dimension {self.dimension!r}")
        if not self.not_evaluated and self.level not in RUBRIC_LEVELS:
            problems.append(f"unknown rubric level {self.level}")
        if self.not_evaluated and self.level != -1:
            problems.append("NOT_EVALUATED must be a separate state, not a numeric 0")
        if not self.rationale:
            problems.append("a rating needs its anchored rationale")
        if not self.reviewer:
            problems.append("a rating needs its reviewer")
        return problems


def rate(
    *,
    session_id: str,
    task_id: str,
    dimension: str,
    level: Optional[int],
    rationale: str,
    reviewer: str,
    evidence_refs: Sequence[str],
    not_evaluated: bool = False,
) -> RubricRating:
    if not_evaluated and level is not None:
        raise ConfigError(
            "NOT_EVALUATED is a separate state, not a numeric 0: do not pass a level together "
            "with not_evaluated=True"
        )
    if not not_evaluated and level is None:
        raise ConfigError("a rating needs a numeric level unless it is NOT_EVALUATED")
    rating = RubricRating(
        session_id=session_id,
        task_id=task_id,
        dimension=dimension,
        level=-1 if not_evaluated else int(level),  # type: ignore[arg-type]
        not_evaluated=not_evaluated,
        rationale=rationale,
        reviewer=reviewer,
        evidence_refs=tuple(evidence_refs),
    )
    problems = rating.validate()
    if problems:
        raise ConfigError("invalid rubric rating: " + "; ".join(problems))
    return rating


def assert_performance_blind(ratings: Sequence[RubricRating], performance_rows: Sequence[Mapping[str, Any]]) -> None:
    """The rater must not see the performance ranking when scoring."""
    reviewed = {rating.session_id for rating in ratings}
    ranked = {str(row.get("session_id", "")) for row in performance_rows}
    # The invariant we can enforce structurally: a rating keyed to a session that
    # also has a performance ranking is a red flag for non-blind scoring.
    if reviewed & ranked:
        raise ConfigError(
            "rubric sessions that also carry performance rankings were scored non-blind"
        )


def aggregate_scores(ratings: Sequence[RubricRating]) -> Dict[str, Any]:
    """Raw dimension x task view first; an aggregate is a *secondary* view."""
    evaluated = [rating for rating in ratings if not rating.not_evaluated]
    coverage = len(evaluated) / len(ratings) if ratings else 0.0
    by_dimension: Dict[str, List[int]] = {}
    for rating in evaluated:
        by_dimension.setdefault(rating.dimension, []).append(rating.level)
    raw = {dimension: sorted(levels) for dimension, levels in sorted(by_dimension.items())}
    means = {
        dimension: sum(levels) / len(levels) for dimension, levels in sorted(by_dimension.items())
    }
    return {
        "raw_vector": raw,
        "not_evaluated_count": sum(1 for rating in ratings if rating.not_evaluated),
        "coverage": coverage,
        "mean_by_dimension": means,
        "weights": {},
        "note": "the mean is a secondary view; the primary report is the raw dimension x task vector",
    }


def coverage_penalty_check(ratings: Sequence[RubricRating], sessions: Sequence[SessionRecord]) -> Dict[str, Any]:
    """A low-completion platform must not score only its successful tasks."""
    failed = [session.session_id for session in sessions if session.success_status in ("fail", "partial")]
    rated = {rating.session_id for rating in ratings}
    unrated_failures = sorted(set(failed) - rated)
    return {
        "failed_sessions": sorted(failed),
        "unrated_failed_sessions": unrated_failures,
        "penalty_applies": bool(unrated_failures),
        "rule": "only scoring successful tasks hides the failure rate",
    }


def agreement(ratings: Sequence[RubricRating]) -> Dict[str, Any]:
    """Exact/adjacent agreement with the disputed items kept, never averaged away."""
    by_cell: Dict[Tuple[str, str, str], List[RubricRating]] = {}
    for rating in ratings:
        key = (rating.session_id, rating.task_id, rating.dimension)
        by_cell.setdefault(key, []).append(rating)
    exact = adjacent = 0
    disputes: List[Dict[str, Any]] = []
    for key, cells in sorted(by_cell.items()):
        levels = {cell.level for cell in cells if not cell.not_evaluated}
        if len(levels) <= 1:
            exact += 1
        elif max(levels) - min(levels) <= 1:
            adjacent += 1
        else:
            disputes.append(
                {
                    "session_id": key[0],
                    "task_id": key[1],
                    "dimension": key[2],
                    "levels": sorted(levels),
                    "resolution": "split the dimension or refine the anchors",
                }
            )
    return {
        "cells": len(by_cell),
        "exact_agreement": exact,
        "adjacent_agreement": adjacent,
        "disputed_items": disputes,
        "reviewer_count": len({rating.reviewer for rating in ratings}),
    }


def learning_curve(sessions: Sequence[SessionRecord], *, task_id: str) -> Dict[str, Any]:
    ordered = sorted(
        [session for session in sessions if session.task_id == task_id],
        key=lambda session: session.start,
    )
    if not ordered:
        raise ConfigError(f"no sessions recorded for task {task_id!r}")
    return {
        "task_id": task_id,
        "first_session_active_s": ordered[0].active_s,
        "repeat_session_active_s": ordered[-1].active_s if len(ordered) > 1 else None,
        "sessions": len(ordered),
        "novice_and_expert_both_reported": len({session.experience_band for session in ordered}) > 1,
    }


def maintenance_estimate(
    *, workarounds: Sequence[Mapping[str, Any]], incidents: Sequence[Mapping[str, Any]], upgrades: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """A range of maintenance hours, never an annual constant from one incident."""
    hours = [float(row.get("maintenance_hours", 0.0)) for row in workarounds]
    incident_hours = [float(row.get("recovery_hours", 0.0)) for row in incidents]
    upgrade_hours = [float(row.get("migration_hours", 0.0)) for row in upgrades]
    values = hours + incident_hours + upgrade_hours
    return {
        "active_hours_low": min(values) if values else 0.0,
        "active_hours_high": max(values) if values else 0.0,
        "basis": "per-event hours; a frequency model needs the incident/upgrade rate",
        "single_incident_exaggeration_guard": len(values) < 3,
    }


def upgrade_record(
    *, component: str, from_version: str, to_version: str, correctness_smoke: str, breaking_changes: Sequence[str], migration_hours: float
) -> Dict[str, Any]:
    if correctness_smoke not in ("pass", "fail", "not_run"):
        raise ConfigError(f"unknown correctness smoke {correctness_smoke!r}")
    if correctness_smoke == "not_run":
        raise ConfigError("an upgrade must re-run correctness smoke: import success is not correctness")
    return {
        "upgrade_id": stable_id("upgrade", {"component": component, "to": to_version}),
        "component": component,
        "from_version": from_version,
        "to_version": to_version,
        "correctness_smoke": correctness_smoke,
        "breaking_changes": list(breaking_changes),
        "migration_hours": migration_hours,
    }


def rollback_verification(*, artifact_identity_restored: bool, cache_restored: bool, environment_restored: bool) -> Dict[str, Any]:
    ok = artifact_identity_restored and cache_restored and environment_restored
    return {"status": "OK" if ok else "FAIL", "problems": [name for name, value in (("artifact", artifact_identity_restored), ("cache", cache_restored), ("environment", environment_restored)) if not value]}


def documentation_gaps(gaps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "gaps": [dict(row) for row in gaps],
        "count": len(gaps),
        "severity": "high" if any(row.get("severity") == "high" for row in gaps) else "low",
    }


def deployable_constraints(*, silent_fallbacks: Sequence[str], undiagnosable_failures: Sequence[str], missing_capabilities: Sequence[str], maintenance_risk: str) -> Dict[str, Any]:
    return {
        "silent_fallbacks": list(silent_fallbacks),
        "undiagnosable_failures": list(undiagnosable_failures),
        "missing_capabilities": list(missing_capabilities),
        "maintenance_risk": maintenance_risk,
        "status": "deployable_with_risk" if not silent_fallbacks else "not_deployable",
    }


def maturity_verdict(
    *,
    candidate_id: str,
    ratings: Sequence[RubricRating],
    sessions: Sequence[SessionRecord],
    coverage: Mapping[str, Any],
) -> Dict[str, Any]:
    aggregate = aggregate_scores(ratings)
    completion = len([session for session in sessions if session.success_status == "success"]) / len(sessions) if sessions else 0.0
    strengths = [dimension for dimension, levels in aggregate["raw_vector"].items() if levels and min(levels) >= 3]
    gaps = [dimension for dimension, levels in aggregate["raw_vector"].items() if levels and max(levels) <= 1]
    return {
        "candidate_id": candidate_id,
        "task_completion_rate": completion,
        "rubric_vector": aggregate["raw_vector"],
        "coverage": aggregate["coverage"],
        "strengths": strengths,
        "gaps": gaps,
        "forbidden_claims": [
            "production reliability (that is S13's deployment and fault experiments)",
            "an aggregate maturity score folded into a performance ranking",
        ],
    }


def table_schemas() -> Mapping[str, Tuple[str, ...]]:
    return {name: TABLE_SCHEMAS[name] for name in sorted(TABLE_SCHEMAS) if name.startswith("e12_09.")}


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only callability check; never an experiment."""
    session = SessionRecord(
        session_id="s1",
        candidate_id="cand_demo",
        operator_or_session_id="op1",
        experience_band="novice",
        clean_or_warm="clean",
        task_id="load_model",
        start="2026-09-19T00:00:00Z",
        end="2026-09-19T00:30:00Z",
        wall_s=1800.0,
        active_s=600.0,
        blocked_s=1000.0,
        rework_s=200.0,
        success_status="success",
        correctness_status="pass",
        actual_backend="cuda",
    )
    _rating = rate(
        session_id="s1",
        task_id="load_model",
        dimension="installability",
        level=3,
        rationale="documented and reproduced",
        reviewer="r1",
        evidence_refs=("hqsb://S12/c/session/1",),
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "session_valid": session.validate() == [],
        "not_evaluated_is_not_zero": _expect_config_error(
            lambda: rate(session_id="s2", task_id="t", dimension="installability", level=0, rationale="x", reviewer="r", evidence_refs=(), not_evaluated=True)
        ),
        "silent_fallback_is_high_risk": silent_fallback_check(requested_backend="cuda", actual_backend="eager", fallback_reason="")["risk"]
        == "high",
        "unknown_failure_needs_low_confidence": _expect_config_error(
            lambda: classify_failure(category="UNKNOWN", severity="high", root_cause_confidence="high", evidence_refs=())
        ),
        "workaround_needs_owner": _expect_config_error(
            lambda: WorkaroundRegistry().register(
                workaround_id="w", kind="patch", owner="", upstream_issue="", first_version="", last_verified_version="", risk="high", automation_status="manual", maintenance_hours=1.0, removal_condition=""
            )
        ),
        "rubric_anchored": anchor_for("installability", 4).startswith("stable"),
    }


def _expect_config_error(callable_: Any) -> bool:
    try:
        callable_()
    except ConfigError:
        return True
    return False


PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结评价范围", ("maturity:MATURITY_DIMENSIONS", "maturity:SessionRecord.candidate_id")),
    (2, "冻结标准任务与成功条件", ("maturity:STANDARD_TASKS", "maturity:TaskSpec")),
    (3, "定义操作者/会话设计", ("maturity:SessionRecord.operator_or_session_id", "maturity:SessionRecord.experience_band")),
    (4, "定义 clean 与 warm 条件", ("maturity:SessionRecord.clean_or_warm", "maturity:SessionRecord")),
    (5, "冻结计时方法", ("maturity:split_times", "maturity:TimingBreakdown")),
    (6, "冻结支持渠道", ("maturity:TaskSpec.allowed_help", "maturity:SessionRecord.documentation_source_ids")),
    (7, "冻结 rubric 锚点", ("maturity:RUBRIC_ANCHORS", "maturity:anchor_for")),
    (8, "建立 session recorder", ("maturity:SessionRecord", "maturity:task_metrics")),
    (9, "执行环境准备任务", ("maturity:TaskSpec", "maturity:SessionRecord")),
    (10, "执行安装任务", ("maturity:TaskSpec", "maturity:SessionRecord")),
    (11, "执行设备 smoke", ("maturity:SessionRecord.correctness_status", "capability:CapabilityEvidence")),
    (12, "执行模型加载/reference", ("maturity:SessionRecord", "maturity:task_metrics")),
    (13, "执行 precision/quant 任务", ("maturity:SessionRecord", "maturity:FAILURE_TAXONOMY")),
    (14, "执行 custom kernel build/load", ("maturity:SessionRecord.actual_backend", "maturity:silent_fallback_check")),
    (15, "执行 graph/compiler 任务", ("maturity:SessionRecord", "maturity:WorkaroundRegistry")),
    (16, "执行服务任务", ("maturity:SessionRecord", "maturity:task_metrics")),
    (17, "执行 profiler 定位任务", ("maturity:tool_coverage_row", "maturity:TOOL_COVERAGE_AREAS")),
    (18, "执行 memory/OOM 调试任务", ("maturity:classify_failure", "maturity:FAILURE_TAXONOMY")),
    (19, "执行 distributed 任务", ("maturity:SessionRecord", "maturity:validate_applicability")),
    (20, "执行 artifact/cache 恢复", ("maturity:rollback_verification", "maturity:upgrade_record")),
    (21, "执行版本不匹配诊断", ("maturity:classify_failure", "maturity:FAILURE_TAXONOMY")),
    (22, "执行文档复现", ("maturity:documentation_gaps", "maturity:TaskSpec")),
    (23, "执行 cold rebuild", ("maturity:upgrade_record", "maturity:rollback_verification")),
    (24, "执行小版本升级", ("maturity:upgrade_record", "maturity:rollback_verification")),
    (25, "执行回退", ("maturity:rollback_verification", "maturity:WorkaroundRegistry")),
    (26, "分类所有事件", ("maturity:classify_failure", "maturity:FAILURE_TAXONOMY")),
    (27, "验证 silent fallback", ("maturity:silent_fallback_check", "maturity:SessionRecord.actual_backend")),
    (28, "计算任务完成指标", ("maturity:task_metrics", "maturity:TimingBreakdown")),
    (29, "计算工具覆盖", ("maturity:coverage_summary", "maturity:tool_coverage_row")),
    (30, "建立 workaround registry", ("maturity:WorkaroundRegistry", "maturity:WorkaroundRegistry.at_risk")),
    (31, "独立 rubric 评分", ("maturity:rate", "maturity:assert_performance_blind")),
    (32, "分析一致性和分歧", ("maturity:agreement", "maturity:RubricRating")),
    (33, "计算学习曲线", ("maturity:learning_curve", "maturity:SessionRecord")),
    (34, "计算维护成本区间", ("maturity:maintenance_estimate", "maturity:WorkaroundRegistry")),
    (35, "映射 deployable constraints", ("maturity:deployable_constraints", "maturity:coverage_penalty_check")),
    (36, "形成成熟度 verdict", ("maturity:maturity_verdict", "campaign:AcceptanceDecision")),
)
