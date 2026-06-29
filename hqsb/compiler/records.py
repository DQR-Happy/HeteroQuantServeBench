"""Unified S11 data contracts, case state machine and failure classification.

Protocol anchors: ``details/S11/README.md`` §14 (compiler cost accounting),
§20 (state machine + failure taxonomy), §21 (CompileRun / Pattern Decision /
Guard Event / Autotune Trial / Cost-model Decision records).

These records are the *only* accepted shapes for the S11 raw tables.  Every
record validates its own required fields and refuses unknown keys, so a report
cannot silently reinterpret a field.  No function here runs anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import canonical_json, sha256_text

# ── state machine (protocol §20) ───────────────────────────────────────────

STATUS_PLANNED = "PLANNED"
STATUS_READY = "READY"
STATUS_RUNNING = "RUNNING"
STATUS_PASS = "PASS"
STATUS_FAIL_CORRECTNESS = "FAIL_CORRECTNESS"
STATUS_FAIL_CAPTURE = "FAIL_CAPTURE"
STATUS_FAIL_REWRITE_LEGALITY = "FAIL_REWRITE_LEGALITY"
STATUS_FAIL_LOWERING = "FAIL_LOWERING"
STATUS_FAIL_COMPILE = "FAIL_COMPILE"
STATUS_FAIL_RUNTIME = "FAIL_RUNTIME"
STATUS_FAIL_PERFORMANCE = "FAIL_PERFORMANCE"
STATUS_FAIL_CACHE_SAFETY = "FAIL_CACHE_SAFETY"
STATUS_BLOCKED_PREREQUISITE = "BLOCKED_PREREQUISITE"
STATUS_NOT_APPLICABLE_CAPABILITY = "NOT_APPLICABLE_CAPABILITY"
STATUS_NOT_RUN_RESOURCE_UNAVAILABLE = "NOT_RUN_RESOURCE_UNAVAILABLE"
STATUS_NOT_RUN_TOOL_UNAVAILABLE = "NOT_RUN_TOOL_UNAVAILABLE"

#: Handbook statuses (``docs/stage_experiments/README.md`` §3) plus the
#: experiment-level BLOCKED state used by the run scaffolding.
STATUS_NOT_STARTED = "NOT_STARTED"
STATUS_PASS_NEGATIVE = "PASS_NEGATIVE"
STATUS_FAIL = "FAIL"
STATUS_BLOCKED = "BLOCKED"
STATUS_N_A_BY_ADR = "N/A_BY_ADR"

CASE_STATUSES: Tuple[str, ...] = (
    STATUS_PLANNED,
    STATUS_READY,
    STATUS_RUNNING,
    STATUS_PASS,
    STATUS_FAIL_CORRECTNESS,
    STATUS_FAIL_CAPTURE,
    STATUS_FAIL_REWRITE_LEGALITY,
    STATUS_FAIL_LOWERING,
    STATUS_FAIL_COMPILE,
    STATUS_FAIL_RUNTIME,
    STATUS_FAIL_PERFORMANCE,
    STATUS_FAIL_CACHE_SAFETY,
    STATUS_BLOCKED_PREREQUISITE,
    STATUS_NOT_APPLICABLE_CAPABILITY,
    STATUS_NOT_RUN_RESOURCE_UNAVAILABLE,
    STATUS_NOT_RUN_TOOL_UNAVAILABLE,
)

CONCLUSION_STATUSES: Tuple[str, ...] = (
    STATUS_PASS,
    STATUS_PASS_NEGATIVE,
    STATUS_FAIL,
    STATUS_BLOCKED,
    STATUS_N_A_BY_ADR,
)

CASE_TERMINAL_STATUSES: Tuple[str, ...] = (
    STATUS_PASS,
    STATUS_FAIL_CORRECTNESS,
    STATUS_FAIL_CAPTURE,
    STATUS_FAIL_REWRITE_LEGALITY,
    STATUS_FAIL_LOWERING,
    STATUS_FAIL_COMPILE,
    STATUS_FAIL_RUNTIME,
    STATUS_FAIL_PERFORMANCE,
    STATUS_FAIL_CACHE_SAFETY,
    STATUS_BLOCKED_PREREQUISITE,
    STATUS_NOT_APPLICABLE_CAPABILITY,
    STATUS_NOT_RUN_RESOURCE_UNAVAILABLE,
    STATUS_NOT_RUN_TOOL_UNAVAILABLE,
)

#: Allowed transitions.  A case may never jump from a failure back to a
#: "running" state without an explicit new run id.
ALLOWED_CASE_TRANSITIONS: Mapping[str, Tuple[str, ...]] = {
    STATUS_PLANNED: (STATUS_READY, STATUS_BLOCKED_PREREQUISITE, STATUS_NOT_APPLICABLE_CAPABILITY),
    STATUS_READY: (STATUS_RUNNING, STATUS_BLOCKED_PREREQUISITE, STATUS_NOT_RUN_RESOURCE_UNAVAILABLE),
    STATUS_RUNNING: tuple(
        status for status in CASE_TERMINAL_STATUSES if status != STATUS_PASS
    ) + (STATUS_PASS,),
}


@dataclass
class CaseState:
    """One experimental case with an auditable state history."""

    case_id: str
    status: str = STATUS_PLANNED
    history: List[Dict[str, str]] = field(default_factory=list)
    reason: str = ""

    def transition(self, new_status: str, reason: str = "") -> None:
        if new_status not in CASE_STATUSES:
            raise ConfigError(f"unknown case status {new_status!r}")
        allowed = ALLOWED_CASE_TRANSITIONS.get(self.status, ())
        if self.status == new_status:
            raise ConfigError(f"case {self.case_id}: no-op transition to {new_status!r}")
        if new_status not in allowed and self.status in CASE_TERMINAL_STATUSES:
            raise ConfigError(
                f"case {self.case_id}: terminal status {self.status!r} cannot move to {new_status!r}"
            )
        if self.status in ALLOWED_CASE_TRANSITIONS and new_status not in allowed:
            raise ConfigError(
                f"case {self.case_id}: illegal transition {self.status!r} -> {new_status!r}"
            )
        self.history.append({"from": self.status, "to": new_status, "reason": reason})
        self.status = new_status
        self.reason = reason

    @property
    def terminal(self) -> bool:
        return self.status in CASE_TERMINAL_STATUSES

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "reason": self.reason,
            "terminal": self.terminal,
            "history": list(self.history),
        }


# ── failure taxonomy (protocol §20) ────────────────────────────────────────

ERROR_CATALOG: Mapping[str, str] = {
    "CAPTURE_GRAPH_BREAK": "a Python frame region was split by a graph break",
    "CAPTURE_UNSUPPORTED_PYTHON": "unsupported Python construct prevented capture",
    "EXPORT_CONSTRAINT": "export could not prove the input satisfies its constraints",
    "PATTERN_NEAR_MISS": "structurally similar but semantically different site",
    "ALIAS_UNSAFE": "alias graph cannot be preserved by the replacement",
    "MUTATION_UNSAFE": "mutation order/visibility cannot be preserved",
    "STATE_UNSAFE": "KV/RNG/state effects cannot be preserved",
    "NO_LEGAL_LOWERING": "no legal lowering candidate",
    "TARGET_UNSUPPORTED": "target capability does not cover the candidate",
    "GUARD_FALSE": "runtime guard failed for a variant",
    "COMPILE_ERROR": "compiler/codegen failed",
    "ABI_MISMATCH": "binary/ABI incompatibility",
    "BINARY_INCOMPATIBLE": "binary cannot be loaded on this target",
    "CACHE_MISS_EXPECTED": "cache miss that the policy expected (e.g. version change)",
    "CACHE_CORRUPT": "cache entry failed integrity validation",
    "CACHE_KEY_COLLISION": "two distinct builds produced the same key",
    "AUTOTUNE_INVALID": "candidate rejected by static legality/resources",
    "AUTOTUNE_CORRECTNESS_FAIL": "candidate compiled but failed correctness",
    "COST_MODEL_LOW_CONFIDENCE": "selection fell back because confidence was low",
    "FALLBACK_EAGER": "executed the eager/reference path",
    "FALLBACK_REFERENCE_KERNEL": "executed the reference lowering of the semantic op",
    "ERROR_NO_SAFE_FALLBACK": "no safe fallback exists; the request must fail explicitly",
}

ERROR_CLASSES: Mapping[str, Tuple[str, ...]] = {
    "capture": ("CAPTURE_GRAPH_BREAK", "CAPTURE_UNSUPPORTED_PYTHON", "EXPORT_CONSTRAINT"),
    "rewrite": ("PATTERN_NEAR_MISS", "ALIAS_UNSAFE", "MUTATION_UNSAFE", "STATE_UNSAFE"),
    "lowering": ("NO_LEGAL_LOWERING", "TARGET_UNSUPPORTED", "GUARD_FALSE"),
    "compile": ("COMPILE_ERROR", "ABI_MISMATCH", "BINARY_INCOMPATIBLE"),
    "cache": ("CACHE_MISS_EXPECTED", "CACHE_CORRUPT", "CACHE_KEY_COLLISION"),
    "autotune": ("AUTOTUNE_INVALID", "AUTOTUNE_CORRECTNESS_FAIL", "COST_MODEL_LOW_CONFIDENCE"),
    "fallback": ("FALLBACK_EAGER", "FALLBACK_REFERENCE_KERNEL", "ERROR_NO_SAFE_FALLBACK"),
}

#: Status each failure code must map to when it is the *reason* for a case.
FAILURE_STATUS_BY_CODE: Mapping[str, str] = {
    "CAPTURE_GRAPH_BREAK": STATUS_FAIL_CAPTURE,
    "CAPTURE_UNSUPPORTED_PYTHON": STATUS_FAIL_CAPTURE,
    "EXPORT_CONSTRAINT": STATUS_FAIL_CAPTURE,
    "PATTERN_NEAR_MISS": STATUS_FAIL_REWRITE_LEGALITY,
    "ALIAS_UNSAFE": STATUS_FAIL_REWRITE_LEGALITY,
    "MUTATION_UNSAFE": STATUS_FAIL_REWRITE_LEGALITY,
    "STATE_UNSAFE": STATUS_FAIL_REWRITE_LEGALITY,
    "NO_LEGAL_LOWERING": STATUS_FAIL_LOWERING,
    "TARGET_UNSUPPORTED": STATUS_FAIL_LOWERING,
    "GUARD_FALSE": STATUS_FAIL_LOWERING,
    "COMPILE_ERROR": STATUS_FAIL_COMPILE,
    "ABI_MISMATCH": STATUS_FAIL_COMPILE,
    "BINARY_INCOMPATIBLE": STATUS_FAIL_COMPILE,
    "CACHE_MISS_EXPECTED": STATUS_PASS,  # an expected miss is not a failure
    "CACHE_CORRUPT": STATUS_FAIL_CACHE_SAFETY,
    "CACHE_KEY_COLLISION": STATUS_FAIL_CACHE_SAFETY,
    "AUTOTUNE_INVALID": STATUS_PASS,  # rejection before running a candidate
    "AUTOTUNE_CORRECTNESS_FAIL": STATUS_FAIL_CORRECTNESS,
    "COST_MODEL_LOW_CONFIDENCE": STATUS_PASS,  # safe fallback, not a failure
    "FALLBACK_EAGER": STATUS_PASS,
    "FALLBACK_REFERENCE_KERNEL": STATUS_PASS,
    "ERROR_NO_SAFE_FALLBACK": STATUS_FAIL_RUNTIME,
}


def classify_error_code(code: str) -> Dict[str, Any]:
    """Map a code to its class and the case status it implies."""
    if code not in ERROR_CATALOG:
        return {
            "code": code,
            "known": False,
            "class": "UNKNOWN",
            "status": STATUS_FAIL_RUNTIME,
            "reason": "unregistered error code: register it before use (no silent UNKNOWN)",
        }
    for group, codes in ERROR_CLASSES.items():
        if code in codes:
            return {
                "code": code,
                "known": True,
                "class": group,
                "status": FAILURE_STATUS_BY_CODE[code],
                "reason": ERROR_CATALOG[code],
            }
    raise ConfigError(f"catalog entry {code!r} is in no class")  # pragma: no cover


# ── compiler cost accounting (protocol §14) ────────────────────────────────

COMPILER_COST_KEYS: Tuple[str, ...] = (
    "capture_time",
    "graph_transform_time",
    "lowering_selection_time",
    "autotune_search_time",
    "codegen_time",
    "native_compile_link_time",
    "artifact_write_time",
    "cache_lookup_and_load_time",
    "first_execution_lazy_init_time",
    "warm_steady_runtime",
)

#: Keys that may never appear inside a *steady-state* latency distribution.
NON_STEADY_COST_KEYS: Tuple[str, ...] = tuple(
    key for key in COMPILER_COST_KEYS if key != "warm_steady_runtime"
)


#: Keys that must be present in a *compile* breakdown.  ``warm_steady_runtime``
#: deliberately does not appear here: steady state is measured in the runtime
#: samples table and must never be merged into compile cost (protocol §14).
COMPILE_BREAKDOWN_REQUIRED: Tuple[str, ...] = ("capture_time", "codegen_time")

#: Backend step → compiler cost key mapping used by ``BackendPlan.as_compile_run``.
STEP_TO_COST_KEY: Mapping[str, str] = {
    "capture_adapter": "capture_time",
    "canonicalize_and_verify": "graph_transform_time",
    "run_semantic_passes": "graph_transform_time",
    "analyze_target": "lowering_selection_time",
    "enumerate_lowerings": "lowering_selection_time",
    "build_guards": "lowering_selection_time",
    "select_candidate": "lowering_selection_time",
    "materialize_callable": "codegen_time",
    "emit_artifact_manifest": "artifact_write_time",
}


def validate_timing_breakdown(breakdown: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    for key in breakdown:
        if key not in COMPILER_COST_KEYS:
            problems.append(f"unknown timing key {key!r}")
    for key in COMPILE_BREAKDOWN_REQUIRED:
        if key not in breakdown:
            problems.append(f"timing breakdown missing {key!r}")
    for key, value in breakdown.items():
        if not isinstance(value, (int, float)) or value < 0:
            problems.append(f"timing {key!r} must be a non-negative number")
    return problems


def merge_step_timings(step_times: Mapping[str, float]) -> Dict[str, float]:
    """Aggregate backend step timings onto the compiler cost keys."""
    merged: Dict[str, float] = {}
    for step, elapsed in step_times.items():
        key = STEP_TO_COST_KEY.get(step)
        if key is None:
            continue
        merged[key] = merged.get(key, 0.0) + float(elapsed)
    return merged


def effective_compile_cost(cold_total_s: float, warm_cache_load_cost_s: float) -> float:
    """``effective_compile_cost = cold_total - warm_cache_load_cost``."""
    if cold_total_s < 0 or warm_cache_load_cost_s < 0:
        raise ConfigError("costs must be non-negative")
    return max(0.0, cold_total_s - warm_cache_load_cost_s)


def break_even_calls(effective_cost_s: float, per_call_saving_s: float) -> Optional[float]:
    """Calls needed to amortise compilation, or ``None`` (``NEVER``).

    A non-positive per-call saving means there is no positive break-even; the
    protocol forbids explaining a regression as "it pays off after enough
    calls".
    """
    if per_call_saving_s <= 0:
        return None
    return effective_cost_s / per_call_saving_s


def amortization_row(
    *, phase: str, cold_total_s: float, warm_load_s: float, baseline_steady_s: float,
    compiled_steady_s: float, call_count: float,
) -> Dict[str, Any]:
    """Break-even accounting for one phase (prefill/decode computed separately)."""
    saving = baseline_steady_s - compiled_steady_s
    effective = effective_compile_cost(cold_total_s, warm_load_s)
    calls = break_even_calls(effective, saving)
    return {
        "phase": phase,
        "baseline_steady_ms": baseline_steady_s * 1000.0,
        "compiled_steady_ms": compiled_steady_s * 1000.0,
        "per_call_saving_ms": saving * 1000.0,
        "effective_compile_cost_ms": effective * 1000.0,
        "break_even_calls": None if calls is None else round(calls, 3),
        "break_even": "NEVER" if calls is None else "FINITE",
        "amortized_latency_ms": (
            None if saving <= 0 else (cold_total_s + compiled_steady_s * call_count) / call_count * 1000.0
        ),
        "note": (
            "per-call saving <= 0: no positive break-even exists"
            if calls is None
            else "break-even is per phase; prefill and decode must not be averaged"
        ),
    }


# ── §21.1 CompileRun ───────────────────────────────────────────────────────

COMPILE_RUN_FIELDS: Tuple[str, ...] = (
    "run_id",
    "compile_id",
    "case_id",
    "model_artifact_id",
    "workload_id",
    "operator_spec_id",
    "capture_mode",
    "compiler_stack_versions",
    "source_graph_id",
    "canonical_ir_id",
    "targeted_ir_id",
    "pass_pipeline_id",
    "target_id",
    "capability_snapshot_id",
    "guard_set_id",
    "variant_id",
    "autotune_session_id",
    "cost_model_id",
    "selected_lowering_id",
    "fallback_id",
    "compile_status",
    "runtime_status",
    "correctness_status",
    "timing_breakdown",
    "peak_memory",
    "artifact_manifest_id",
    "trace_id",
    "result_id",
)


@dataclass
class CompileRun:
    """One compile/run record (protocol §21.1)."""

    run_id: str
    compile_id: str
    case_id: str
    model_artifact_id: str = ""
    workload_id: str = ""
    operator_spec_id: str = ""
    capture_mode: str = ""
    compiler_stack_versions: Mapping[str, str] = field(default_factory=dict)
    source_graph_id: str = ""
    canonical_ir_id: str = ""
    targeted_ir_id: str = ""
    pass_pipeline_id: str = ""
    target_id: str = ""
    capability_snapshot_id: str = ""
    guard_set_id: str = ""
    variant_id: str = ""
    autotune_session_id: str = ""
    cost_model_id: str = ""
    selected_lowering_id: str = ""
    fallback_id: str = ""
    compile_status: str = STATUS_PLANNED
    runtime_status: str = STATUS_PLANNED
    correctness_status: str = "not_run"
    timing_breakdown: Mapping[str, float] = field(default_factory=dict)
    peak_memory: Optional[int] = None
    artifact_manifest_id: str = ""
    trace_id: str = ""
    result_id: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("run_id", "compile_id", "case_id"):
            if not getattr(self, name):
                problems.append(f"missing required field {name!r}")
        if self.fallback_id and not self.selected_lowering_id:
            problems.append("fallback recorded without a selected lowering")
        if self.selected_lowering_id and not self.fallback_id:
            problems.append(
                "every selected lowering must name a fallback (fail-closed requirement)"
            )
        if self.correctness_status not in ("pass", "fail", "not_run"):
            problems.append(f"unknown correctness_status {self.correctness_status!r}")
        problems.extend(validate_timing_breakdown(self.timing_breakdown) if self.timing_breakdown else [])
        if self.compile_status == STATUS_PASS and not self.artifact_manifest_id:
            problems.append("a successful compile must emit an artifact manifest id")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "compile_id": self.compile_id,
            "case_id": self.case_id,
            "model_artifact_id": self.model_artifact_id,
            "workload_id": self.workload_id,
            "operator_spec_id": self.operator_spec_id,
            "capture_mode": self.capture_mode,
            "compiler_stack_versions": dict(sorted(self.compiler_stack_versions.items())),
            "source_graph_id": self.source_graph_id,
            "canonical_ir_id": self.canonical_ir_id,
            "targeted_ir_id": self.targeted_ir_id,
            "pass_pipeline_id": self.pass_pipeline_id,
            "target_id": self.target_id,
            "capability_snapshot_id": self.capability_snapshot_id,
            "guard_set_id": self.guard_set_id,
            "variant_id": self.variant_id,
            "autotune_session_id": self.autotune_session_id,
            "cost_model_id": self.cost_model_id,
            "selected_lowering_id": self.selected_lowering_id,
            "fallback_id": self.fallback_id,
            "compile_status": self.compile_status,
            "runtime_status": self.runtime_status,
            "correctness_status": self.correctness_status,
            "timing_breakdown": dict(sorted(self.timing_breakdown.items())),
            "peak_memory": self.peak_memory,
            "artifact_manifest_id": self.artifact_manifest_id,
            "trace_id": self.trace_id,
            "result_id": self.result_id,
        }
        return payload

    def record_hash(self) -> str:
        return sha256_text(canonical_json(self.as_dict()))


# ── §21.2 Pattern decision ─────────────────────────────────────────────────

PATTERN_DECISION_FIELDS: Tuple[str, ...] = (
    "decision_id",
    "source_node_ids",
    "source_locations",
    "pattern_id",
    "pattern_version",
    "structural_match",
    "semantic_predicates",
    "alias_effect_summary",
    "target_candidates",
    "selected_id",
    "fallback_id",
    "before_ir_id",
    "after_ir_id",
)

SEMANTIC_PREDICATE_FIELDS: Tuple[str, ...] = (
    "name",
    "expected",
    "actual",
    "outcome",
)

TARGET_CANDIDATE_FIELDS: Tuple[str, ...] = (
    "id",
    "capability",
    "guard",
    "evidence",
    "cost",
    "outcome",
    "reason",
)


@dataclass
class PatternDecisionRecord:
    """One matcher/legality decision (protocol §21.2)."""

    decision_id: str
    source_node_ids: Tuple[str, ...]
    source_locations: Tuple[str, ...]
    pattern_id: str
    pattern_version: str
    structural_match: bool
    semantic_predicates: Tuple[Mapping[str, Any], ...]
    alias_effect_summary: Mapping[str, Any] = field(default_factory=dict)
    target_candidates: Tuple[Mapping[str, Any], ...] = ()
    selected_id: str = ""
    fallback_id: str = ""
    before_ir_id: str = ""
    after_ir_id: str = ""
    reject_code: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("decision_id", "pattern_id", "pattern_version", "before_ir_id"):
            if not getattr(self, name):
                problems.append(f"missing required field {name!r}")
        if not self.structural_match and not self.reject_code:
            problems.append("a non-match must carry a reject code")
        if self.reject_code and self.reject_code not in ERROR_CATALOG:
            problems.append(f"unknown reject code {self.reject_code!r}")
        if self.structural_match and self.selected_id and not self.after_ir_id:
            problems.append("a committed rewrite must record the after-IR id")
        for predicate in self.semantic_predicates:
            missing = [key for key in ("name", "outcome") if key not in predicate]
            if missing:
                problems.append(f"predicate record missing {missing}")
        for candidate in self.target_candidates:
            missing = [key for key in ("id", "outcome") if key not in candidate]
            if missing:
                problems.append(f"target candidate record missing {missing}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "source_node_ids": list(self.source_node_ids),
            "source_locations": list(self.source_locations),
            "pattern_id": self.pattern_id,
            "pattern_version": self.pattern_version,
            "structural_match": self.structural_match,
            "semantic_predicates": [dict(row) for row in self.semantic_predicates],
            "alias_effect_summary": dict(self.alias_effect_summary),
            "target_candidates": [dict(row) for row in self.target_candidates],
            "selected_id": self.selected_id,
            "fallback_id": self.fallback_id,
            "before_ir_id": self.before_ir_id,
            "after_ir_id": self.after_ir_id,
            "reject_code": self.reject_code,
        }


# ── §21.3 Guard event ──────────────────────────────────────────────────────

GUARD_EVENT_FIELDS: Tuple[str, ...] = (
    "timestamp",
    "run_id",
    "frame_id",
    "variant_id",
    "guard_id",
    "expression",
    "source",
    "category",
    "actual_values",
    "outcome",
    "action",
    "new_variant_id",
    "cache_event_id",
    "latency",
)

GUARD_ACTIONS: Tuple[str, ...] = ("reuse", "recompile", "fallback", "error")


@dataclass
class GuardEventRecord:
    """One guard evaluation (protocol §21.3)."""

    run_id: str
    frame_id: str
    variant_id: str
    guard_id: str
    expression: str
    source: str
    category: str
    actual_values: Mapping[str, Any]
    outcome: bool
    action: str
    timestamp: str = ""
    new_variant_id: str = ""
    cache_event_id: str = ""
    latency_us: Optional[float] = None
    sequence_index: int = 0

    def validate(self, guard_categories: Sequence[str] = ()) -> List[str]:
        problems: List[str] = []
        if self.action not in GUARD_ACTIONS:
            problems.append(f"unknown guard action {self.action!r}")
        if self.outcome is False and self.action == "reuse":
            problems.append(
                "guard false with action='reuse' would execute an incompatible binary "
                "(wrong reuse is an automatic FAIL)"
            )
        if self.action == "recompile" and not self.new_variant_id:
            problems.append("action='recompile' must record the new variant id")
        if guard_categories and self.category not in guard_categories:
            problems.append(f"unknown guard category {self.category!r}")
        if not self.expression:
            problems.append("guard expression must be recorded (not just an id)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "sequence_index": self.sequence_index,
            "frame_id": self.frame_id,
            "variant_id": self.variant_id,
            "guard_id": self.guard_id,
            "expression": self.expression,
            "source": self.source,
            "category": self.category,
            "actual_values": dict(self.actual_values),
            "outcome": self.outcome,
            "action": self.action,
            "new_variant_id": self.new_variant_id,
            "cache_event_id": self.cache_event_id,
            "latency_us": self.latency_us,
        }


# ── §21.4 Autotune trial ───────────────────────────────────────────────────

AUTOTUNE_TRIAL_FIELDS: Tuple[str, ...] = (
    "session_id",
    "task_id",
    "shape_group",
    "split",
    "candidate_config",
    "legality_status",
    "reject_reason",
    "compile_status",
    "compile_time",
    "correctness_status",
    "error_metrics",
    "raw_timing_samples",
    "thermal_state",
    "resource_estimate",
    "resource_actual",
    "selected",
    "confirmation_status",
)

AUTOTUNE_SPLITS: Tuple[str, ...] = ("tuning", "validation", "holdout", "confirmation")


@dataclass
class AutotuneTrialRecord:
    """One candidate trial (protocol §21.4) — winners alone are not enough."""

    session_id: str
    task_id: str
    shape_group: str
    split: str
    candidate_config: Mapping[str, Any]
    legality_status: str
    compile_status: str = ""
    correctness_status: str = ""
    reject_reason: str = ""
    compile_time: Optional[float] = None
    error_metrics: Mapping[str, Any] = field(default_factory=dict)
    raw_timing_samples: Tuple[float, ...] = ()
    thermal_state: Mapping[str, Any] = field(default_factory=dict)
    resource_estimate: Mapping[str, Any] = field(default_factory=dict)
    resource_actual: Mapping[str, Any] = field(default_factory=dict)
    selected: bool = False
    confirmation_status: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.split not in AUTOTUNE_SPLITS:
            problems.append(f"unknown split {self.split!r}")
        if self.legality_status not in ("legal", "invalid"):
            problems.append(f"unknown legality status {self.legality_status!r}")
        if self.legality_status == "invalid" and not self.reject_reason:
            problems.append("an invalid candidate must record a structured reject reason")
        if self.legality_status == "invalid" and self.raw_timing_samples:
            problems.append(
                "an invalid candidate must not be timed: latency=inf is not an observation"
            )
        if self.compile_status == "failed" and self.correctness_status == "pass":
            problems.append("a candidate cannot pass correctness without compiling")
        if self.correctness_status == "pass" and self.compile_status != "ok":
            problems.append("correctness measured before compile success")
        if self.selected and self.correctness_status != "pass":
            problems.append("a selected candidate must pass correctness first")
        if self.selected and not self.confirmation_status:
            problems.append("a selected winner requires an independent confirmation status")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "shape_group": self.shape_group,
            "split": self.split,
            "candidate_config": dict(sorted(self.candidate_config.items())),
            "legality_status": self.legality_status,
            "reject_reason": self.reject_reason,
            "compile_status": self.compile_status,
            "compile_time": self.compile_time,
            "correctness_status": self.correctness_status,
            "error_metrics": dict(self.error_metrics),
            "raw_timing_samples": list(self.raw_timing_samples),
            "thermal_state": dict(self.thermal_state),
            "resource_estimate": dict(self.resource_estimate),
            "resource_actual": dict(self.resource_actual),
            "selected": self.selected,
            "confirmation_status": self.confirmation_status,
        }


# ── §21.5 Cost-model decision ──────────────────────────────────────────────

COST_MODEL_DECISION_FIELDS: Tuple[str, ...] = (
    "model_id",
    "dataset_version",
    "split_group",
    "feature_schema_version",
    "feature_values",
    "candidate_predictions",
    "confidence",
    "chosen",
    "oracle",
    "baseline",
    "chosen_latency",
    "oracle_latency",
    "regret",
    "fallback_triggered",
    "fallback_result",
)


@dataclass
class CostModelDecisionRecord:
    """One selection decision (protocol §21.5)."""

    model_id: str
    dataset_version: str
    split_group: str
    feature_schema_version: str
    feature_values: Mapping[str, Any]
    candidate_predictions: Mapping[str, float]
    confidence: float
    chosen: str = ""
    oracle: str = ""
    baseline: str = ""
    chosen_latency: Optional[float] = None
    oracle_latency: Optional[float] = None
    regret: Optional[float] = None
    fallback_triggered: bool = False
    fallback_result: str = ""

    def validate(self, *, eligible: Sequence[str] = ()) -> List[str]:
        problems: List[str] = []
        for name in ("model_id", "dataset_version", "feature_schema_version"):
            if not getattr(self, name):
                problems.append(f"missing required field {name!r}")
        if not 0.0 <= self.confidence <= 1.0:
            problems.append("confidence must be in [0, 1]")
        if self.chosen and eligible and self.chosen not in eligible:
            problems.append(
                f"chosen candidate {self.chosen!r} is not in the eligible set — a cost model "
                "may never make an illegal candidate legal"
            )
        if self.fallback_triggered and not self.fallback_result:
            problems.append("a fallback must record its result")
        if not self.fallback_triggered and self.confidence < 0.0:  # pragma: no cover
            problems.append("impossible confidence")
        if self.regret is not None and self.oracle_latency is None:
            problems.append("regret requires the oracle latency")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "dataset_version": self.dataset_version,
            "split_group": self.split_group,
            "feature_schema_version": self.feature_schema_version,
            "feature_values": dict(sorted(self.feature_values.items())),
            "candidate_predictions": dict(sorted(self.candidate_predictions.items())),
            "confidence": self.confidence,
            "chosen": self.chosen,
            "oracle": self.oracle,
            "baseline": self.baseline,
            "chosen_latency": self.chosen_latency,
            "oracle_latency": self.oracle_latency,
            "regret": self.regret,
            "fallback_triggered": self.fallback_triggered,
            "fallback_result": self.fallback_result,
        }


# ── table schemas ──────────────────────────────────────────────────────────

TABLE_SCHEMAS: Mapping[str, Tuple[str, ...]] = {
    "compile_run": COMPILE_RUN_FIELDS,
    "pattern_decision": PATTERN_DECISION_FIELDS,
    "guard_event": GUARD_EVENT_FIELDS,
    "autotune_trial": AUTOTUNE_TRIAL_FIELDS,
    "cost_model_decision": COST_MODEL_DECISION_FIELDS,
}


def validate_table_row(table: str, row: Mapping[str, Any]) -> List[str]:
    """Validate a raw-table row against its schema (unknown keys rejected)."""
    if table not in TABLE_SCHEMAS:
        raise ConfigError(f"unknown table {table!r}; known: {sorted(TABLE_SCHEMAS)}")
    required = set(TABLE_SCHEMAS[table])
    unknown = sorted(set(row) - required)
    problems = [f"{table}: unknown key {key!r}" for key in unknown]
    return problems


def record_required_fields(name: str) -> Tuple[str, ...]:
    """Expose the protocol field list for a record kind."""
    try:
        return TABLE_SCHEMAS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"unknown record kind {name!r}") from exc
