"""E14-07 — Agent tool/memory/asynchronous workflow: waits, failures and traces.

Optional P2: ``N/A_BY_ADR`` when not selected, and its existence must never be
used to claim "Agent Infra is done" (``E14-07`` §13).

An agent workflow's end-to-end time is usually *not* dominated by one decode:

```text
E2E = model queue/inference + tool queue/execution/network + orchestration wait
      + memory/state IO + retry/backoff (+ external/human wait)
```

so reporting "total task duration" cannot tell whether an inference optimisation
helped, and reporting only model tokens/s ignores the tool and orchestration
bottlenecks.  This module supplies the observability and safety primitives that
make the decomposition possible, under two hard boundaries (§2):

* tools run in a **sandbox or read-only** environment, no real third-party
  messages and no irreversible actions;
* the task must have a **determinate oracle**, and "general intelligence" is not
  evaluated — only the system behaviour of a given workflow.

Interfaces provided:

* :class:`TaskFamily` (step 2), :class:`ToolContract` (steps 3–4) with schema and
  policy validation *before* execution;
* :class:`WorkflowStateMachine` — the 13 states plus failure branches of §4.1,
  with an idempotency key per attempt (step 5);
* :class:`SuccessOracle` (step 6), :class:`MemoryContract` (step 8),
  :class:`RetryPolicy` (step 9);
* :class:`TraceSchema` — span kinds, links and privacy rules (step 11);
* :func:`validate_tool_call` / :func:`validate_tool_result` (steps 14–15, 29);
* :func:`reconstruct_critical_path` — the critical path, not the sum of spans
  (step 18);
* :func:`instrumentation_overhead` (step 19), :func:`phase_baseline` (step 20);
* :func:`latency_sweep` (step 21), :func:`fanout_sweep` (step 22),
  :func:`concurrency_sweep` (step 23), :func:`mixed_workflow_fairness` (step 24);
* :func:`tool_wait_decoupling` (step 25), :func:`memory_consistency` (step 26);
* :func:`fault_cases` + :func:`check_fault_accounting` (steps 27–31);
* :func:`cancel_propagation` (step 32), :func:`retry_storm_protection` (step 33);
* :class:`AgentWorkloadDecision` (step 40).

Nothing here calls a model or executes a tool; the arithmetic is over supplied
spans, attempts and payload sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.experimental import records as rec
from hqsb.experimental.contracts import AdoptionDecision
from hqsb.experimental.identity import digest_text

EXPERIMENT_ID = "E14-07"
TITLE = "Agent Tool/Memory/异步 Workflow 的等待、失败与 Trace"
LEVEL = "P2"

CLAIM_BOUNDARY = (
    "只作为 Serving/Trace 压测扩展，不取代 kernel/Runtime 主线，也不升级为通用 Agent 平台 claim；"
    "不执行时标 N/A_BY_ADR（E14-07 §13）。"
)

#: Span kinds that must be separable (§4.3).
SPAN_KINDS: Tuple[str, ...] = (
    "model",
    "tool",
    "orchestration",
    "memory",
    "queue",
    "retry_wait",
)

#: Tool safety levels (step 3).
TOOL_SAFETY_LEVELS: Tuple[str, ...] = ("read_only", "simulated", "compensable", "forbidden")

#: JSON-schema types a tool parameter may declare (step 4).
PARAMETER_TYPES: Tuple[str, ...] = ("string", "integer", "number", "boolean", "array", "object")

#: Error classes and whether a retry is permitted (step 9).
ERROR_CLASSES: Mapping[str, bool] = {
    "transient_network": True,
    "rate_limited": True,
    "tool_timeout": True,
    "model_timeout": True,
    "invalid_arguments": False,
    "permission_denied": False,
    "unknown_tool": False,
    "malformed_result": False,
    "budget_exhausted": False,
}

#: Fault families an E14-07 run must inject (steps 27–33).
FAULT_FAMILIES: Tuple[str, ...] = (
    "tool_timeout",
    "rate_limit",
    "malformed_result",
    "oversized_result",
    "transient_error",
    "worker_crash_before_persist",
    "worker_crash_after_persist",
    "cancel",
    "retry_storm",
)

#: Fields that must never become metric labels (step 11).
FORBIDDEN_LABEL_FIELDS: Tuple[str, ...] = (
    "prompt",
    "tool_payload",
    "tool_result_body",
    "user_message",
    "api_key",
    "authorization",
    "session_token",
)

#: Dimensions the run budget must bound (step 12).
BUDGET_DIMENSIONS: Tuple[str, ...] = (
    "max_model_tokens",
    "max_tool_calls",
    "max_workflow_wall_time_s",
    "max_queue_depth",
    "max_cost_units",
    "max_error_rate",
)


@dataclass(frozen=True)
class TaskFamily:
    """Step 2: an input, a possible tool sequence and a verifiable terminal state."""

    task_family_id: str
    description: str
    inputs: Tuple[str, ...] = ()
    allowed_tools: Tuple[str, ...] = ()
    max_steps: int = 0
    deterministic_termination: bool = False
    external_side_effects: bool = False

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("task_family_id", "description"):
            if not getattr(self, name):
                findings.append(f"TaskFamily: {name} is required")
        if not self.inputs:
            findings.append("TaskFamily: the input space must be declared")
        if not self.allowed_tools:
            findings.append("TaskFamily: the allowed tool set must be declared")
        if self.max_steps <= 0:
            findings.append("TaskFamily: max_steps must be bounded (开放式聊天无法定义成功)")
        if not self.deterministic_termination:
            findings.append("TaskFamily: a determinate terminal state is required for an oracle")
        if self.external_side_effects:
            findings.append(
                "TaskFamily: real external side effects are forbidden (E14-07 §2 沙箱/只读)"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_family_id": self.task_family_id,
            "description": self.description,
            "inputs": list(self.inputs),
            "allowed_tools": list(self.allowed_tools),
            "max_steps": self.max_steps,
            "deterministic_termination": self.deterministic_termination,
            "external_side_effects": self.external_side_effects,
        }


@dataclass
class ToolContract:
    """Steps 3–4: name, schema, safety level, timeout, idempotency and so on."""

    tool_id: str
    name: str
    safety_level: str
    parameters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    required: Tuple[str, ...] = ()
    result_schema: Mapping[str, Any] = field(default_factory=dict)
    timeout_s: float = 0.0
    idempotent: bool = False
    version: str = ""
    max_payload_bytes: int = 0
    permission_scope: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("tool_id", "name", "version", "permission_scope"):
            if not getattr(self, name):
                findings.append(f"ToolContract {self.tool_id or '<unnamed>'}: {name} is required")
        if self.safety_level not in TOOL_SAFETY_LEVELS:
            findings.append(
                f"ToolContract {self.name}: safety_level {self.safety_level!r} must be one of "
                f"{', '.join(TOOL_SAFETY_LEVELS)}"
            )
        if self.safety_level == "forbidden":
            findings.append(f"ToolContract {self.name}: a forbidden tool must not be in the list at all")
        if self.timeout_s <= 0:
            findings.append(f"ToolContract {self.name}: a timeout is required")
        if not self.idempotent and self.safety_level != "read_only":
            findings.append(
                f"ToolContract {self.name}: a non-idempotent tool needs compensation semantics, "
                "otherwise a retry duplicates its side effect"
            )
        if self.max_payload_bytes <= 0:
            findings.append(f"ToolContract {self.name}: max_payload_bytes must be bounded")
        for parameter, spec in sorted(self.parameters.items()):
            if spec.get("type") not in PARAMETER_TYPES:
                findings.append(
                    f"ToolContract {self.name}: parameter {parameter!r} has unknown type {spec.get('type')!r}"
                )
        unknown_required = [name for name in self.required if name not in self.parameters]
        if unknown_required:
            findings.append(
                f"ToolContract {self.name}: required parameters are not declared: {', '.join(unknown_required)}"
            )
        if not self.result_schema:
            findings.append(f"ToolContract {self.name}: a result schema is required (结果必须被校验)")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "safety_level": self.safety_level,
            "parameters": {key: dict(value) for key, value in sorted(self.parameters.items())},
            "required": list(self.required),
            "result_schema": dict(self.result_schema),
            "timeout_s": self.timeout_s,
            "idempotent": self.idempotent,
            "version": self.version,
            "max_payload_bytes": self.max_payload_bytes,
            "permission_scope": self.permission_scope,
        }


#: Argument names that indicate a path-traversal or injection attempt (step 15).
DANGEROUS_ARGUMENT_TOKENS: Tuple[str, ...] = ("../", "..\\", "/etc/", "~/.ssh", "curl ", "rm -rf", ";", "|", "$(")


def validate_tool_call(call: Mapping[str, Any], contracts: Mapping[str, ToolContract]) -> Dict[str, Any]:
    """Steps 14–15: schema, required fields, ranges, enum and permission — *before* execution.

    ``E14-07`` §10: 模型生成合法 JSON 不等于工具调用安全; the validator therefore
    refuses unknown fields and dangerous arguments rather than filling defaults.
    """
    problems: List[str] = []
    name = str(call.get("name", ""))
    contract = contracts.get(name)
    if contract is None:
        return {
            "name": name,
            "runnable": False,
            "problems": [f"unknown tool {name!r}"],
            "permission_checked": False,
        }
    arguments = dict(call.get("arguments") or {})
    missing = [key for key in contract.required if key not in arguments]
    if missing:
        problems.append(f"missing required arguments: {', '.join(missing)}")
    unknown = sorted(set(arguments) - set(contract.parameters))
    if unknown:
        problems.append(f"unknown arguments must be rejected, not ignored: {', '.join(unknown)}")
    for key, spec in contract.parameters.items():
        if key not in arguments:
            continue
        value = arguments[key]
        expected = spec.get("type")
        if expected == "string" and not isinstance(value, str):
            problems.append(f"argument {key!r} must be a string")
        elif expected in ("integer", "number") and not isinstance(value, (int, float)):
            problems.append(f"argument {key!r} must be numeric")
        elif expected == "boolean" and not isinstance(value, bool):
            problems.append(f"argument {key!r} must be a boolean")
        elif expected == "array" and not isinstance(value, list):
            problems.append(f"argument {key!r} must be an array")
        limits = spec.get("enum")
        if limits and value not in limits:
            problems.append(f"argument {key!r}={value!r} is outside the declared enum")
        low, high = spec.get("minimum"), spec.get("maximum")
        if isinstance(value, (int, float)):
            if low is not None and value < low:
                problems.append(f"argument {key!r}={value!r} is below the minimum {low}")
            if high is not None and value > high:
                problems.append(f"argument {key!r}={value!r} is above the maximum {high}")
        if isinstance(value, str):
            haystack = value.lower()
            hit = next((token for token in DANGEROUS_ARGUMENT_TOKENS if token in haystack), None)
            if hit:
                problems.append(f"argument {key!r} contains a forbidden token {hit!r}")
            if len(value) > spec.get("maxLength", 10_000):
                problems.append(f"argument {key!r} exceeds its maxLength")
    if contract.safety_level == "forbidden":
        problems.append("the tool is marked forbidden")
    permission = str(call.get("permission_scope", contract.permission_scope))
    permission_checked = "permission_scope" in call
    if not permission_checked:
        problems.append("the caller did not declare a permission scope: 越权调用必须被拒绝")
    elif permission != contract.permission_scope:
        problems.append(
            f"permission scope {permission!r} does not cover {contract.permission_scope!r}"
        )
    return {
        "name": name,
        "runnable": not problems,
        "normalised_arguments": {key: arguments[key] for key in sorted(arguments)} if not problems else {},
        "problems": problems,
        "permission_checked": permission_checked,
        "contract_version": contract.version,
    }


def validate_tool_result(result: Mapping[str, Any], contract: ToolContract) -> Dict[str, Any]:
    """Step 29: a malformed/oversized result must not enter the next prompt.

    ``E14-07`` §10: 未验证结果直接进入下一 prompt is a FAIL — the result is a
    data-plane payload, not trusted text.
    """
    problems: List[str] = []
    if "payload" not in result:
        problems.append("result has no payload")
    size = int(result.get("bytes", len(str(result.get("payload", "")))))
    if size > contract.max_payload_bytes:
        problems.append(f"result is {size} bytes, above the {contract.max_payload_bytes} cap")
    for field_name in ("contains_injection",):
        if result.get(field_name):
            problems.append("the result is flagged as containing prompt-injection content")
    declared = contract.result_schema.get("required", ())
    payload = result.get("payload")
    if isinstance(payload, Mapping):
        missing = [key for key in declared if key not in payload]
        if missing:
            problems.append(f"result payload is missing {', '.join(missing)}")
    elif declared:
        problems.append("result payload is not an object although the schema requires fields")
    return {
        "bytes": size,
        "accepted": not problems,
        "problems": problems,
        "truncated": size > contract.max_payload_bytes,
    }


@dataclass
class WorkflowStateMachine:
    """Step 5: the persistent state machine with per-attempt idempotency keys."""

    workflow_id: str
    tenant_id: str = ""
    model_artifact_id: str = ""
    trace_id: str = ""
    events: List[Mapping[str, Any]] = field(default_factory=list)

    def transition(
        self, old_state: str, new_state: str, *, attempt: int = 1,
        idempotency_key: str = "", tool_contract_id: str = "", timestamp_ns: int = 0,
        cost: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        event = {
            "workflow_id": self.workflow_id,
            "tenant_id": self.tenant_id,
            "old_state": old_state,
            "new_state": new_state,
            "attempt": attempt,
            "idempotency_key_hash": digest_text(idempotency_key) if idempotency_key else "",
            "model_artifact_id": self.model_artifact_id,
            "tool_contract_id": tool_contract_id,
            "trace_id": self.trace_id,
            "timestamp_ns": timestamp_ns,
            "cost_accumulated": dict(cost or {}),
        }
        self.events.append(event)
        return event

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.workflow_id:
            findings.append("WorkflowStateMachine: workflow_id is required")
        previous = ""
        for index, event in enumerate(self.events):
            new_state = str(event.get("new_state", ""))
            old_state = str(event.get("old_state", ""))
            if new_state not in rec.WORKFLOW_STATES:
                findings.append(f"event {index}: unknown state {new_state!r}")
                continue
            if index and old_state != previous:
                findings.append(f"event {index}: old_state {old_state!r} != previous {previous!r}")
            if index and not rec.is_valid_transition("agent_workflow", previous, new_state):
                findings.append(f"illegal workflow transition {previous} -> {new_state}")
            if int(event.get("attempt", 1)) > 1 and not event.get("idempotency_key_hash"):
                findings.append(
                    f"event {index}: attempt > 1 without an idempotency key (重试可能产生重复副作用)"
                )
            if new_state in ("TOOL_DISPATCHED", "TOOL_RUNNING") and not event.get("tool_contract_id"):
                findings.append(f"event {index}: a tool state without a tool contract id")
            previous = new_state
        if not self.events:
            findings.append("no state events recorded: crash 后不能只靠日志重建状态")
        return findings

    def terminal_state(self) -> str:
        return str(self.events[-1].get("new_state", "")) if self.events else ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "tenant_id": self.tenant_id,
            "events": [dict(event) for event in self.events],
            "terminal_state": self.terminal_state(),
            "problems": self.problems(),
        }


@dataclass(frozen=True)
class SuccessOracle:
    """Step 6: correct result, allowed paths, forbidden actions, partial credit."""

    oracle_id: str
    expected_terminal: str
    allowed_tool_sequences: Tuple[Tuple[str, ...], ...] = ()
    forbidden_actions: Tuple[str, ...] = ()
    evaluator_version: str = ""
    partial_credit: Mapping[str, float] = field(default_factory=dict)

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("oracle_id", "expected_terminal", "evaluator_version"):
            if not getattr(self, name):
                findings.append(f"SuccessOracle: {name} is required")
        if not self.allowed_tool_sequences:
            findings.append("SuccessOracle: allowed tool sequences must be declared, not inferred")
        if not self.forbidden_actions:
            findings.append(
                "SuccessOracle: forbidden actions must be declared (否则性能可能来自跳过必要工具)"
            )
        for key, value in sorted(self.partial_credit.items()):
            if not 0.0 <= float(value) <= 1.0:
                findings.append(f"SuccessOracle: partial credit for {key!r} is outside [0, 1]")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "oracle_id": self.oracle_id,
            "expected_terminal": self.expected_terminal,
            "allowed_tool_sequences": [list(item) for item in self.allowed_tool_sequences],
            "forbidden_actions": list(self.forbidden_actions),
            "evaluator_version": self.evaluator_version,
            "partial_credit": {key: self.partial_credit[key] for key in sorted(self.partial_credit)},
        }


def score_task(
    *, oracle: SuccessOracle, terminal_state: str, tool_sequence: Sequence[str], actions: Sequence[str]
) -> Dict[str, Any]:
    """Step 16/35: score against the oracle, including forbidden actions."""
    problems: List[str] = []
    forbidden_hit = [action for action in actions if action in set(oracle.forbidden_actions)]
    if forbidden_hit:
        problems.append(f"forbidden actions were performed: {', '.join(forbidden_hit)}")
    sequence = tuple(tool_sequence)
    allowed = any(sequence[: len(candidate)] == candidate for candidate in oracle.allowed_tool_sequences)
    if not allowed:
        problems.append(f"tool sequence {list(sequence)} is not among the allowed sequences")
    success = terminal_state == oracle.expected_terminal and not problems
    return {
        "terminal_state": terminal_state,
        "expected_terminal": oracle.expected_terminal,
        "tool_sequence": list(sequence),
        "forbidden_actions_hit": forbidden_hit,
        "success": success,
        "partial": None if success else oracle.partial_credit.get(terminal_state),
        "problems": problems,
    }


@dataclass(frozen=True)
class MemoryContract:
    """Step 8: session/state/vector/cache read-write with version, TTL and tenant."""

    scope: str
    versioned: bool
    ttl_s: float = 0.0
    tenant_isolated: bool = False
    sensitive_fields: Tuple[str, ...] = ()
    conflict_policy: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.scope not in ("session", "state", "vector", "cache"):
            findings.append(f"MemoryContract: unknown scope {self.scope!r}")
        if not self.versioned:
            findings.append(
                "MemoryContract: unversioned memory cannot detect conflicting concurrent updates"
            )
        if self.scope in ("session", "state") and self.ttl_s <= 0:
            findings.append(f"MemoryContract: {self.scope} memory needs a TTL")
        if not self.tenant_isolated:
            findings.append("MemoryContract: tenant isolation is required (memory 串租户 = FAIL)")
        if not self.conflict_policy:
            findings.append("MemoryContract: a conflict policy is required (last-write-wins 静默丢状态)")
        if not self.sensitive_fields:
            findings.append(
                "MemoryContract: sensitive fields must be listed so they can be redacted even when empty"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "versioned": self.versioned,
            "ttl_s": self.ttl_s,
            "tenant_isolated": self.tenant_isolated,
            "sensitive_fields": list(self.sensitive_fields),
            "conflict_policy": self.conflict_policy,
        }


def memory_consistency(cases: Sequence[Mapping[str, Any]]) -> List[str]:
    """Step 26: concurrent writes, expiry and cross-tenant reads must be consistent."""
    problems: List[str] = []
    for case in cases:
        kind = str(case.get("kind", ""))
        if kind not in ("concurrent_write", "expired_read", "cross_tenant_read", "version_regression"):
            problems.append(f"unknown memory case {kind!r}")
            continue
        if kind == "concurrent_write" and case.get("silent_loss"):
            problems.append("a concurrent write was silently lost (last-write-wins)")
        if kind == "expired_read" and case.get("read_succeeded"):
            problems.append("an expired entry was still readable")
        if kind == "cross_tenant_read" and case.get("read_succeeded"):
            problems.append("a cross-tenant read succeeded (memory 串租户 = FAIL)")
        if kind == "version_regression" and case.get("accepted"):
            problems.append("an older version overwrote a newer one")
    if not cases:
        problems.append("no memory consistency cases recorded")
    return problems


@dataclass(frozen=True)
class RetryPolicy:
    """Step 9: which errors are retryable, plus attempts, backoff, jitter and dedup."""

    policy_id: str
    error_classes: Mapping[str, bool] = field(default_factory=dict)
    max_attempts: int = 0
    base_backoff_s: float = 0.0
    max_backoff_s: float = 0.0
    jitter: bool = False
    idempotency_key_required: bool = True
    circuit_breaker_threshold: int = 0

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.policy_id:
            findings.append("RetryPolicy: policy_id is required")
        if self.max_attempts <= 0:
            findings.append("RetryPolicy: max_attempts must be bounded")
        if self.base_backoff_s <= 0:
            findings.append("RetryPolicy: a base backoff is required (所有失败立即无限重试)")
        if self.max_backoff_s < self.base_backoff_s:
            findings.append("RetryPolicy: max_backoff_s is below base_backoff_s")
        if not self.jitter:
            findings.append("RetryPolicy: jitter is required to avoid synchronised retries")
        if self.circuit_breaker_threshold <= 0:
            findings.append("RetryPolicy: a circuit breaker threshold is required")
        for name, retryable in sorted(self.error_classes.items()):
            if name not in ERROR_CLASSES:
                findings.append(f"RetryPolicy: unknown error class {name!r}")
            elif retryable is not ERROR_CLASSES[name]:
                findings.append(
                    f"RetryPolicy: error class {name!r} is marked retryable={retryable} but the "
                    f"classification says {ERROR_CLASSES[name]}"
                )
        missing = [name for name in ERROR_CLASSES if name not in self.error_classes]
        if missing:
            findings.append(f"RetryPolicy: error classes not classified: {', '.join(missing)}")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "error_classes": {key: self.error_classes[key] for key in sorted(self.error_classes)},
            "max_attempts": self.max_attempts,
            "base_backoff_s": self.base_backoff_s,
            "max_backoff_s": self.max_backoff_s,
            "jitter": self.jitter,
            "idempotency_key_required": self.idempotency_key_required,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
        }


@dataclass
class TraceSchema:
    """Step 11: span names, links, resources and the privacy rules."""

    span_names: Tuple[str, ...] = ()
    attribute_keys: Tuple[str, ...] = ()
    sampled: bool = False
    cardinality_bound: int = 0

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.span_names:
            findings.append("TraceSchema: span names are required")
        for kind in SPAN_KINDS:
            if kind not in self.span_names:
                findings.append(f"TraceSchema: no span for {kind!r}; the wait cannot be separated from execution")
        leaked = sorted(set(self.attribute_keys) & set(FORBIDDEN_LABEL_FIELDS))
        if leaked:
            findings.append(
                f"TraceSchema: payload fields must not become labels: {', '.join(leaked)}"
            )
        if self.cardinality_bound <= 0:
            findings.append("TraceSchema: a cardinality bound is required")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "span_names": list(self.span_names),
            "attribute_keys": list(self.attribute_keys),
            "sampled": self.sampled,
            "cardinality_bound": self.cardinality_bound,
        }


def run_budget(budgets: Mapping[str, Any]) -> Dict[str, Any]:
    """Step 12: an unbounded agent loop consumes without limit."""
    problems: List[str] = []
    for dimension in BUDGET_DIMENSIONS:
        if dimension not in budgets:
            problems.append(f"run budget is missing {dimension!r}")
        elif budgets[dimension] is None:
            problems.append(f"run budget {dimension!r} is unmeasured; unknown is not zero")
        elif isinstance(budgets[dimension], (int, float)) and budgets[dimension] <= 0:
            problems.append(f"run budget {dimension!r} must be positive")
    return {"budgets": {key: budgets[key] for key in sorted(budgets)}, "problems": problems,
            "ok": not problems}


# ── trace structure and critical path (steps 17–19) ────────────────────────


def validate_trace_structure(spans: Sequence[Mapping[str, Any]], *, root_span_id: str = "") -> Dict[str, Any]:
    """Step 17: parent/link structure with no orphan span.

    ``E14-07`` §10: 仅靠时间接近猜关联 fails the moment two tool calls overlap, so
    every non-root span must name an existing parent and every fan-out must use a
    link rather than a fabricated parent.
    """
    problems: List[str] = []
    ids = {str(span.get("span_id")) for span in spans if span.get("span_id")}
    roots = [span for span in spans if not span.get("parent_span_id")]
    if len(roots) != 1:
        problems.append(f"{len(roots)} root spans found; exactly one workflow root is expected")
    if root_span_id and roots and str(roots[0].get("span_id")) != root_span_id:
        problems.append("the declared root span id does not match the span without a parent")
    for span in spans:
        span_id = str(span.get("span_id", ""))
        parent = span.get("parent_span_id")
        if parent and str(parent) not in ids:
            problems.append(f"span {span_id}: parent {parent!r} does not exist (orphan)")
        for link in span.get("links", ()) or ():
            if str(link) not in ids:
                problems.append(f"span {span_id}: link {link!r} does not exist")
        kind = str(span.get("kind", ""))
        if kind not in SPAN_KINDS:
            problems.append(f"span {span_id}: unknown kind {kind!r}")
        if span.get("end_ns") and span.get("start_ns") and span["end_ns"] < span["start_ns"]:
            problems.append(f"span {span_id}: end_ns precedes start_ns")
    return {"spans": len(spans), "roots": len(roots), "problems": problems, "ok": not problems}


def reconstruct_critical_path(
    spans: Sequence[Mapping[str, Any]], *, root_span_id: str
) -> Dict[str, Any]:
    """Step 18: the critical path, because the sum of spans exceeds the E2E time.

    ``E14-07`` §10: span 求和当 E2E is wrong whenever work overlaps; the critical
    path and the parallel slack are the two numbers that explain the difference.
    """
    problems: List[str] = []
    by_parent: Dict[str, List[Mapping[str, Any]]] = {}
    for span in spans:
        if span.get("start_ns") is None or span.get("end_ns") is None:
            problems.append(f"span {span.get('span_id')!r} has no [start, end) interval")
            continue
        by_parent.setdefault(str(span.get("parent_span_id") or ""), []).append(span)
    if not spans:
        return {"error": "no spans supplied"}

    def longest(node_id: str) -> Tuple[int, List[str], int]:
        children = by_parent.get(node_id, [])
        if not children:
            return 0, [], 0
        best_total = -1
        best_path: List[str] = []
        parallel_total = 0
        for child in children:
            duration = int(child["end_ns"]) - int(child["start_ns"])
            sub_total, sub_path, sub_parallel = longest(str(child.get("span_id", "")))
            parallel_total += duration
            total = duration + sub_total
            if total > best_total:
                best_total = total
                best_path = [str(child.get("span_id", ""))] + sub_path
        return best_total, best_path, max(parallel_total, best_total)

    root = root_span_id or next((str(span.get("span_id")) for span in spans if not span.get("parent_span_id")), "")
    duration, path, covered = longest(root)
    # ``is not None`` rather than truthiness: a span that legitimately starts at
    # monotonic time 0 would otherwise be dropped from the sum and the slack
    # would come out negative.
    span_sum = sum(
        int(span["end_ns"]) - int(span["start_ns"])
        for span in spans
        if span.get("end_ns") is not None and span.get("start_ns") is not None
    )
    starts = [int(span["start_ns"]) for span in spans if span.get("start_ns") is not None]
    ends = [int(span["end_ns"]) for span in spans if span.get("end_ns") is not None]
    wall = (max(ends) - min(starts)) if starts and ends else 0
    return {
        "root_span_id": root,
        "critical_path_ns": duration,
        "critical_path": path,
        "span_sum_ns": span_sum,
        "wall_clock_ns": wall,
        "parallel_slack_ns": span_sum - wall,
        "idle_gap_ns": wall - duration,
        "problems": problems,
        "note": "span 求和不等于 E2E：必须给出 critical path 与 idle gap（step 18）",
    }


def instrumentation_overhead(
    *, off: Mapping[str, float], on: Mapping[str, float], sampled: Mapping[str, float]
) -> Dict[str, Any]:
    """Step 19: tracing must not become the bottleneck.

    ``E14-07`` §10 forbids letting full-payload tracing dominate; the comparison is
    over the same workload with tracing off, on and sampled.
    """
    problems: List[str] = []
    for metric in ("latency_ms", "cpu_ms", "network_bytes", "storage_bytes"):
        for label, dataset in (("off", off), ("on", on), ("sampled", sampled)):
            if metric not in dataset:
                problems.append(f"overhead dataset {label!r} is missing {metric!r}")
    overhead_ratio = None
    if off.get("latency_ms") and on.get("latency_ms"):
        overhead_ratio = float(on["latency_ms"]) / float(off["latency_ms"])
        if overhead_ratio > 1.1:
            problems.append(f"tracing adds {(overhead_ratio - 1) * 100:.1f}% latency; reduce sampling or payloads")
    return {
        "latency_ratio_on": overhead_ratio,
        "latency_ratio_sampled": (
            float(sampled["latency_ms"]) / float(off["latency_ms"])
            if off.get("latency_ms") and sampled.get("latency_ms")
            else None
        ),
        "problems": problems,
        "ok": not problems,
    }


def phase_baseline(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 20: model queue/TTFT/tokens vs tool queue/execute/network vs orchestration."""
    problems: List[str] = []
    required = (
        "model_queue_ms", "model_ttft_ms", "model_tokens", "tool_queue_ms",
        "tool_execute_ms", "tool_network_ms", "orchestration_ms", "memory_io_ms",
    )
    totals = {key: 0.0 for key in required}
    for row in rows:
        missing = [key for key in required if key not in row]
        if missing:
            problems.append(f"phase row is missing {', '.join(missing)}")
            continue
        for key in required:
            totals[key] += float(row[key])
    grand = sum(totals.values())
    shares = {key: (value / grand if grand else 0.0) for key, value in totals.items()}
    return {
        "totals": totals,
        "shares": shares,
        "model_share": (totals["model_queue_ms"] + totals["model_ttft_ms"]) / grand if grand else 0.0,
        "tool_share": (totals["tool_queue_ms"] + totals["tool_execute_ms"] + totals["tool_network_ms"]) / grand if grand else 0.0,
        "problems": problems,
        "note": "只报总时延无法知道推理优化是否改善了体验（step 20）",
    }


# ── sweeps (steps 21–25) ───────────────────────────────────────────────────


def latency_sweep(rows: Sequence[Mapping[str, Any]], *, expected_slope: float, tolerance: float) -> Dict[str, Any]:
    """Step 21: injected tool latency must propagate to E2E as the model predicts."""
    problems: List[str] = []
    deltas: List[Dict[str, Any]] = []
    for row in rows:
        for key in ("injected_tool_ms", "measured_e2e_ms", "baseline_e2e_ms"):
            if key not in row:
                problems.append(f"latency row is missing {key!r}")
                break
        else:
            observed = float(row["measured_e2e_ms"]) - float(row["baseline_e2e_ms"])
            deltas.append({"injected_tool_ms": row["injected_tool_ms"], "observed_e2e_delta_ms": observed})
    return {
        "rows": deltas,
        "expected_slope": expected_slope,
        "tolerance": tolerance,
        "problems": problems,
        "note": "用受控 proxy 注入，不用不稳定的外部 API 作为唯一实验源（step 21）",
    }


def fanout_sweep(
    *, fanouts: Sequence[int], e2e_p50_ms: Sequence[float], e2e_p99_ms: Sequence[float], slowest_tool_ms: Sequence[float]
) -> Dict[str, Any]:
    """Step 22: more fan-out is not automatically faster — the slowest tool dominates."""
    if not (len(fanouts) == len(e2e_p50_ms) == len(e2e_p99_ms) == len(slowest_tool_ms)):
        raise ConfigError("fanout sweeps must supply p50, p99 and slowest-tool for every point")
    problems: List[str] = []
    rows = [
        {"fanout": fanout, "p50_ms": p50, "p99_ms": p99, "slowest_tool_ms": slowest}
        for fanout, p50, p99, slowest in zip(fanouts, e2e_p50_ms, e2e_p99_ms, slowest_tool_ms)
    ]
    for index in range(1, len(rows)):
        if rows[index]["p50_ms"] < rows[index - 1]["p50_ms"] and rows[index]["p99_ms"] > rows[index - 1]["p99_ms"]:
            problems.append(
                f"fan-out {rows[index]['fanout']}: the mean improved while P99 worsened "
                "(最慢工具/连接池/限流主导)"
            )
    return {"rows": rows, "problems": problems, "note": "fan-out 数越多不一定越快（step 22）"}


def concurrency_sweep(rows: Sequence[Mapping[str, Any]], *, required: Sequence[str]) -> Dict[str, Any]:
    """Step 23: from low load to saturation, with rejection and utilisation."""
    problems: List[str] = []
    for row in rows:
        missing = [key for key in required if key not in row]
        if missing:
            problems.append(f"concurrency row {row.get('concurrency', '<unnamed>')} is missing {', '.join(missing)}")
    return {
        "rows": len(rows),
        "fields": list(required),
        "problems": problems,
        "note": "只测最大任务/s 会掩盖饱和点与拒绝行为（step 23）",
    }


def mixed_workflow_fairness(
    *, short_rows: Sequence[Mapping[str, Any]], long_rows: Sequence[Mapping[str, Any]], slo_ratio: float
) -> Dict[str, Any]:
    """Step 24: a long tool chain must not starve short tasks."""
    problems: List[str] = []
    worst = 0.0
    for row in short_rows:
        ratio = row.get("p99_ratio_to_slo")
        if ratio is None:
            problems.append("a short-workflow row has no p99_ratio_to_slo")
            continue
        worst = max(worst, float(ratio))
    if not short_rows:
        problems.append("no short-workflow rows supplied")
    if worst > slo_ratio:
        problems.append(f"short workflows reached {worst:.3f}× the SLO ratio (limit {slo_ratio})")
    return {
        "short": len(short_rows),
        "long": len(long_rows),
        "worst_short_slo_ratio": worst,
        "problems": problems,
        "ok": not problems,
    }


def tool_wait_decoupling(
    *, waiting_rows: Sequence[Mapping[str, Any]], kv_used_while_waiting: bool, requeued_on_resume: bool
) -> Dict[str, Any]:
    """Step 25: a workflow waiting on a tool must not hold a model slot or KV.

    ``E14-07`` §10: tool wait 占 GPU/KV converts an external wait into a device
    resource leak, which no latency number reveals by itself.
    """
    problems: List[str] = []
    if kv_used_while_waiting:
        problems.append("a workflow waiting on a tool still holds KV: the external wait becomes a GPU leak")
    if not requeued_on_resume:
        problems.append("the workflow was not re-queued when its tool returned")
    for row in waiting_rows:
        if row.get("model_slot_held"):
            problems.append(
                f"workflow {row.get('workflow_id', '<unnamed>')} held a model execution slot while waiting"
            )
        if row.get("resume_latency_ms") is None:
            problems.append(
                f"workflow {row.get('workflow_id', '<unnamed>')} does not record its resume latency"
            )
    if not waiting_rows:
        problems.append("no waiting workflows recorded")
    return {"rows": len(waiting_rows), "problems": problems, "ok": not problems}


# ── faults (steps 27–33) ───────────────────────────────────────────────────


def fault_cases() -> Tuple[Dict[str, Any], ...]:
    """Steps 27–33: the injected faults and the outcome each must produce."""
    mapping = {
        "tool_timeout": "重试或终止，且原调用不得继续产生副作用",
        "rate_limit": "退避 + 全局/租户 budget + admission，避免同步重试风暴",
        "malformed_result": "结果校验拒绝，原始 payload 与 contract id 存档",
        "oversized_result": "按 max_payload_bytes 拒绝或截断，禁止未验证内容进入下一 prompt",
        "transient_error": "可重试且 attempts/cost 被记录",
        "worker_crash_before_persist": "lease 到期重派，无重复副作用",
        "worker_crash_after_persist": "重放时靠 idempotency key 去重",
        "cancel": "传播到排队模型与工具，无法取消的外部动作被标记",
        "retry_storm": "jitter/circuit breaker 生效，队列与错误率有界",
    }
    return tuple(
        {"fault": name, "required_outcome": mapping[name], "isolation_required": "沙箱/proxy 注入"}
        for name in FAULT_FAMILIES
    )


def check_fault_accounting(records: Sequence[Mapping[str, Any]]) -> List[str]:
    """Steps 27–33: each injected fault states its state and resource consequence."""
    problems: List[str] = []
    known = set(FAULT_FAMILIES)
    for record in records:
        fault = str(record.get("fault", ""))
        if fault not in known:
            problems.append(f"unknown fault {fault!r}")
            continue
        for key in ("terminal_state", "resource_released", "duplicate_side_effect", "attempts", "cost_units"):
            if key not in record:
                problems.append(f"{fault}: fault record is missing {key!r}")
        if record.get("duplicate_side_effect"):
            problems.append(f"{fault}: a retry produced a duplicate side effect")
        if record.get("resource_released") is False:
            problems.append(f"{fault}: resources were not released")
        if record.get("terminal_state") not in rec.WORKFLOW_STATES:
            problems.append(f"{fault}: terminal state {record.get('terminal_state')!r} is not a workflow state")
        if int(record.get("attempts", 0)) < 1:
            problems.append(f"{fault}: attempts must be >= 1")
    return problems


def cancel_propagation(
    *, events: Sequence[Mapping[str, Any]], uncancellable_actions: Sequence[str]
) -> List[str]:
    """Step 32: cancel must reach queued model calls, tools and sub-tasks."""
    problems: List[str] = []
    for event in events:
        target = str(event.get("target", ""))
        if target not in ("model_queue", "model_running", "tool_running", "child_workflow", "memory_write"):
            problems.append(f"cancel target {target!r} is not documented")
        if event.get("cancelled") is False and target not in {"tool_running"}:
            problems.append(f"cancel did not propagate to {target!r}")
        if target == "tool_running" and event.get("cancelled") is False and not event.get("marked_uncancellable"):
            problems.append("an uncancellable external action was not marked as such")
    for action in uncancellable_actions:
        if not isinstance(action, str) or not action:
            problems.append(f"uncancellable action {action!r} must be a description")
    if not events:
        problems.append("no cancellation events recorded")
    return problems


def retry_storm_protection(
    *, concurrent_failures: int, jittered: bool, circuit_open: bool, queue_depth_max: int, error_rate_max: float
) -> Dict[str, Any]:
    """Step 33: simultaneous failures must not become a thundering herd."""
    problems: List[str] = []
    if not jittered:
        problems.append("retries are not jittered: they will synchronise")
    if not circuit_open:
        problems.append("the circuit breaker did not open under simultaneous failures")
    if queue_depth_max <= 0:
        problems.append("queue depth is unbounded during the storm")
    if error_rate_max > 0.5:
        problems.append(f"error rate reached {error_rate_max:.2f} during the storm")
    return {
        "concurrent_failures": concurrent_failures,
        "queue_depth_max": queue_depth_max,
        "error_rate_max": error_rate_max,
        "problems": problems,
        "ok": not problems,
    }


def reconcile_phase_model(
    *, predicted_e2e_ms: float, measured_e2e_ms: float, critical_path_ns: int, retries: int
) -> Dict[str, Any]:
    """Step 37: rebuild E2E from the critical path, queues and retries."""
    if predicted_e2e_ms <= 0:
        return {"error": "predicted_e2e_ms must be positive"}
    residual = measured_e2e_ms - predicted_e2e_ms
    return {
        "predicted_e2e_ms": predicted_e2e_ms,
        "measured_e2e_ms": measured_e2e_ms,
        "residual_ms": residual,
        "residual_ratio": residual / predicted_e2e_ms,
        "critical_path_ms": critical_path_ns / 1e6,
        "retries": retries,
        "explained": abs(residual) <= 0.15 * predicted_e2e_ms,
        "note": "把所有收益归给 LLM inference 是不合格的；必须用 critical path 与 retry 解释（step 37）",
    }


def agent_workload_decision(
    *,
    decision_id: str,
    adoption_requested: bool,
    oracle: Mapping[str, Any],
    trace: Mapping[str, Any],
    fault_problems: Sequence[str],
    privacy_ok: bool,
    evidence_refs: Sequence[str],
) -> AdoptionDecision:
    """Step 40: keep it as a Serving/Trace workload, or state what was not proven."""
    problems: List[str] = []
    if not oracle.get("success"):
        problems.append("the task oracle did not pass")
    if trace.get("problems"):
        problems.append("the trace cannot decompose the waits")
    problems.extend(fault_problems)
    if not privacy_ok:
        problems.append("trace privacy/cardinality did not pass")
    if problems or not adoption_requested:
        decision = rec.RESEARCH_ONLY if not problems else rec.BLOCKED_EVIDENCE
        allowed: Tuple[str, ...] = ()
    else:
        decision = rec.ADOPT_EXPERIMENTAL
        allowed = ("保留为 Serving/Trace 压测 workload，不替代 kernel 主线",)
    return AdoptionDecision(
        decision_id=decision_id,
        experiment_id=EXPERIMENT_ID,
        decision=decision,
        allowed_claims=allowed,
        forbidden_claims=(
            "通用 Agent 平台能力",
            "prompt/tool payload 进入指标 label",
            "跳过工具换取性能",
            "用 Agent 工作替代 kernel 主线",
        ),
        quality_status=rec.STATUS_PASS if oracle.get("success") else rec.STATUS_FAIL_CORRECTNESS,
        performance_status=rec.STATUS_NOT_RUN,
        maturity=rec.MATURITY_SOURCE_INTEGRATED,
        evidence_refs=tuple(evidence_refs),
        limitations=tuple(problems) + ("结论限定为给定 workflow 的系统行为，不评价通用智能",),
        reopened_if=("有确定性 oracle 与受控工具沙箱的新任务族时可重开",),
    )


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the E14-07 interfaces (labelled smoke, not an experiment)."""
    contract = ToolContract(
        tool_id="t1", name="search", safety_level="read_only",
        parameters={"query": {"type": "string", "maxLength": 64}},
        required=("query",), result_schema={"required": ("hits",)}, timeout_s=5.0,
        idempotent=True, version="1", max_payload_bytes=4096, permission_scope="read:docs",
    )
    ok_call = validate_tool_call(
        {"name": "search", "arguments": {"query": "kv cache"}, "permission_scope": "read:docs"}, {"search": contract}
    )
    bad_call = validate_tool_call(
        {"name": "search", "arguments": {"query": "../../etc/passwd", "extra": 1}}, {"search": contract}
    )
    machine = WorkflowStateMachine(workflow_id="w1", model_artifact_id="m1", trace_id="tr1")
    machine.transition("CREATED", "MODEL_DECISION")
    machine.transition("MODEL_DECISION", "COMPLETED")
    spans = [
        {"span_id": "root", "parent_span_id": "", "kind": "orchestration", "start_ns": 0, "end_ns": 100},
        {"span_id": "m", "parent_span_id": "root", "kind": "model", "start_ns": 0, "end_ns": 40},
        {"span_id": "t", "parent_span_id": "root", "kind": "tool", "start_ns": 40, "end_ns": 90},
    ]
    path = reconstruct_critical_path(spans, root_span_id="root")
    oracle = SuccessOracle(
        oracle_id="o1", expected_terminal="COMPLETED", allowed_tool_sequences=(("search",),),
        forbidden_actions=("write_file",), evaluator_version="v1",
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "valid_call_runnable": ok_call["runnable"],
        "bad_call_rejected": not bad_call["runnable"],
        "bad_call_problems": len(bad_call["problems"]),
        "machine_problems": machine.problems(),
        "critical_path_ns": path["critical_path_ns"],
        "parallel_slack_ns": path["parallel_slack_ns"],
        "oracle_problems": len(oracle.problems()),
        "faults": len(fault_cases()),
        "span_kinds": list(SPAN_KINDS),
    }


# ── result accessors ───────────────────────────────────────────────────────

def model_share(result: Mapping[str, Any]) -> float:
    """Step 20: the fraction of E2E that inference actually owns."""
    return float(result.get("model_share", 0.0))


def tool_share(result: Mapping[str, Any]) -> float:
    """Step 20: the fraction owned by tool queue/execute/network (the wait)."""
    return float(result.get("tool_share", 0.0))


# ── protocol step table (40 steps of details/S14/E14-07) ───────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "用 ADR 决定是否执行", ("frontier:issue_contract", "agent:CLAIM_BOUNDARY")),
    (2, "冻结一个最小任务族", ("agent:TaskFamily", "agent:TaskFamily.deterministic_termination")),
    (3, "冻结工具清单与安全级别", ("agent:ToolContract", "agent:TOOL_SAFETY_LEVELS")),
    (4, "冻结 tool schema", ("agent:ToolContract.parameters", "agent:PARAMETER_TYPES")),
    (5, "冻结 workflow 状态机", ("agent:WorkflowStateMachine", "records:WORKFLOW_TRANSITIONS")),
    (6, "冻结 task success oracle", ("agent:SuccessOracle", "agent:score_task")),
    (7, "冻结模型与 prompt artifact", ("contracts:PolicySnapshot", "identity:digest_text")),
    (8, "冻结 memory 语义", ("agent:MemoryContract", "agent:memory_consistency")),
    (9, "冻结 retry/idempotency 语义", ("agent:RetryPolicy", "agent:ERROR_CLASSES")),
    (10, "冻结并发/负载矩阵", ("frontier:WorkloadStrata", "agent:concurrency_sweep")),
    (11, "冻结 trace schema 与隐私", ("agent:TraceSchema", "agent:FORBIDDEN_LABEL_FIELDS")),
    (12, "冻结资源与中止预算", ("agent:run_budget", "agent:BUDGET_DIMENSIONS")),
    (13, "建立确定性 fake-tool baseline", ("agent:ToolContract.safety_level", "agent:phase_baseline")),
    (14, "验证合法 tool-call parse/validation", ("agent:validate_tool_call", "agent:ToolContract.required")),
    (15, "验证非法/越权 tool-call", ("agent:validate_tool_call", "agent:DANGEROUS_ARGUMENT_TOKENS")),
    (16, "运行单任务正确性 baseline", ("agent:score_task", "agent:WorkflowStateMachine")),
    (17, "验证 trace parent/link 结构", ("agent:validate_trace_structure", "agent:SPAN_KINDS")),
    (18, "验证时间轴和 critical path", ("agent:reconstruct_critical_path", "agent:TraceSchema.span_names")),
    (19, "测 instrumentation overhead", ("agent:instrumentation_overhead", "agent:TraceSchema.cardinality_bound")),
    (20, "运行 model/tool 分阶段 baseline", ("agent:phase_baseline", "records:PROFILE_LAYER_FIELDS")),
    (21, "运行工具延迟扫描", ("agent:latency_sweep", "agent:tool_share", "agent:model_share")),
    (22, "运行 fan-out/fan-in 扫描", ("agent:fanout_sweep", "agent:SPAN_KINDS")),
    (23, "运行 Agent 并发扫描", ("agent:concurrency_sweep", "frontier:StatisticsPlan")),
    (24, "运行长短 workflow 混合", ("agent:mixed_workflow_fairness", "frontier:StopRules")),
    (25, "验证模型 batching 与 tool wait 解耦", ("agent:tool_wait_decoupling",
                                                 "posttraining:check_kv_reuse_identity")),
    (26, "验证 memory 读写一致性", ("agent:memory_consistency", "agent:MemoryContract.conflict_policy")),
    (27, "注入 tool timeout", ("agent:fault_cases", "agent:check_fault_accounting")),
    (28, "注入 rate limit/过载", ("agent:RetryPolicy.circuit_breaker_threshold", "agent:check_fault_accounting")),
    (29, "注入畸形/超大 tool result", ("agent:validate_tool_result", "agent:ToolContract.max_payload_bytes")),
    (30, "注入模型/工具暂时性错误", ("agent:ERROR_CLASSES", "agent:check_fault_accounting")),
    (31, "注入 worker crash", ("agent:check_fault_accounting", "agent:WorkflowStateMachine.transition")),
    (32, "执行 cancel/用户超时", ("agent:cancel_propagation", "records:WORKFLOW_STATES")),
    (33, "执行 retry storm 保护", ("agent:retry_storm_protection", "agent:RetryPolicy.jitter")),
    (34, "选择一个编排策略候选", ("frontier:BaselinePair", "frontier:BaselinePair.intended_difference")),
    (35, "运行 candidate correctness/quality", ("agent:score_task", "contracts:check_quality_before_performance")),
    (36, "运行 candidate 服务 A/B", ("agent:concurrency_sweep", "frontier:StatisticsPlan")),
    (37, "对账阶段模型与实测", ("agent:reconcile_phase_model", "agent:phase_baseline")),
    (38, "在 holdout workflow/故障确认", ("frontier:WorkloadStrata.holdout_id", "agent:fault_cases")),
    (39, "跨 run/time block 重复", ("records:EXPERIMENT_UNITS", "frontier:StatisticsPlan.minimum_repeats")),
    (40, "形成 AgentWorkloadDecision", ("agent:agent_workload_decision", "contracts:AdoptionDecision.validate")),
)
