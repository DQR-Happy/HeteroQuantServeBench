"""Pattern specs, semantic predicates, rewrite and coverage (E06-04).

A pattern is three things, not one (E06-04 §2):

1. **structure** — ops, edges, constants and data flow (candidate generation);
2. **semantic predicates** — dtype, shape, eps, alias, user count, mutation,
   quant policy (safety);
3. **lowering capability** — whether any registered target can actually run it.

Structure matching alone never rewrites anything.  The same shape of graph with
a different eps, an extra user, an in-place residual write or a different quant
layout must be rejected **with a field-level reason**, and
:func:`mutation_report` exists to prove the predicates still reject injected
near-misses (E06-04 §11 steps 4/15).

Pattern hits, lowering selection and observed kernels are three separate states
(E06-04 §11 step 12); this module only ever produces the first.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError, SchemaError
from hqsb.integration.graph import Graph, GraphDiff, GraphNode, IRLevel, Ref

# ── decisions and reject taxonomy ─────────────────────────────────────────


class DecisionStatus:
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE_REASON"
    HIT = "HIT"
    REJECT = "REJECT"
    UNKNOWN_NEEDS_REVIEW = "UNKNOWN_NEEDS_REVIEW"

    ALL = (ELIGIBLE, INELIGIBLE, HIT, REJECT, UNKNOWN_NEEDS_REVIEW)


class RejectReason:
    """Reason taxonomy (E06-04 §8); codes carry the failing field."""

    STRUCTURE_MISMATCH = "STRUCTURE_MISMATCH"
    SEMANTIC_EPS = "SEMANTIC_EPS"
    SEMANTIC_NORM = "SEMANTIC_NORM"
    SEMANTIC_ROUND = "SEMANTIC_ROUND"
    EXTRA_USER = "EXTRA_USER"
    ALIAS_OR_MUTATION = "ALIAS_OR_MUTATION"
    SIDE_EFFECT = "SIDE_EFFECT"
    DTYPE = "DTYPE"
    SHAPE = "SHAPE"
    STRIDE_LAYOUT = "STRIDE_LAYOUT"
    QUANT_POLICY = "QUANT_POLICY"
    BACKEND_CAPABILITY = "BACKEND_CAPABILITY"
    DYNAMIC_CONSTRAINT = "DYNAMIC_CONSTRAINT"
    VERSION_UNSUPPORTED = "VERSION_UNSUPPORTED"
    UNKNOWN = "UNKNOWN"

    ALL = (
        STRUCTURE_MISMATCH,
        SEMANTIC_EPS,
        SEMANTIC_NORM,
        SEMANTIC_ROUND,
        EXTRA_USER,
        ALIAS_OR_MUTATION,
        SIDE_EFFECT,
        DTYPE,
        SHAPE,
        STRIDE_LAYOUT,
        QUANT_POLICY,
        BACKEND_CAPABILITY,
        DYNAMIC_CONSTRAINT,
        VERSION_UNSUPPORTED,
        UNKNOWN,
    )

    @classmethod
    def require(cls, code: str) -> str:
        if code not in cls.ALL:
            raise SchemaError(
                f"unknown reject reason {code!r}; taxonomy is frozen",
                details={"field": "reason", "allowed": list(cls.ALL)},
            )
        return code


# ── pattern structure ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class PatternNode:
    """One node of a structural pattern.

    ``inputs`` binds each positional argument: ``"$key"`` references another
    pattern node, ``"?"`` matches anything (including a graph input).
    """

    key: str
    op: str
    inputs: Tuple[str, ...] = ()
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "op": self.op, "inputs": list(self.inputs)}


@dataclass(frozen=True)
class Structure:
    """A concrete structure the pattern accepts (same semantics, same IR level)."""

    name: str
    nodes: Tuple[PatternNode, ...]

    @property
    def root(self) -> str:
        return self.nodes[-1].key

    @property
    def structural_signature(self) -> str:
        payload = "|".join(f"{node.op}({','.join(node.inputs)})" for node in self.nodes)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "signature": self.structural_signature,
            "nodes": [node.as_dict() for node in self.nodes],
        }


@dataclass(frozen=True)
class Match:
    """A structural candidate: pattern key → graph node."""

    case_id: str
    structure: str
    bindings: Mapping[str, str]  # pattern key -> graph node name
    module_path: str = ""
    source_stack: str = ""

    @property
    def nodes(self) -> Tuple[str, ...]:
        return tuple(self.bindings.values())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "structure": self.structure,
            "bindings": dict(self.bindings),
            "module_path": self.module_path,
            "source_stack": self.source_stack,
        }


def find_matches(
    graph: Graph,
    structure: Structure,
    *,
    case_prefix: str = "",
    module_path: str = "",
) -> Tuple[Match, ...]:
    """Enumerate structural candidates (no safety judgement yet)."""
    matches: List[Match] = []

    def bind(index: int, bindings: Dict[str, str]) -> None:
        if index == len(structure.nodes):
            matches.append(
                Match(
                    case_id=(
                        f"{case_prefix or graph.structural_hash()[:8]}:"
                        f"{structure.name}:{len(matches)}"
                    ),
                    structure=structure.name,
                    bindings=dict(bindings),
                    module_path=module_path or graph.module_path,
                )
            )
            return
        pattern_node = structure.nodes[index]
        for graph_node in graph.nodes:
            if graph_node.op != pattern_node.op:
                continue
            if graph_node.name in bindings.values():
                continue
            ok = True
            for position, expected in enumerate(pattern_node.inputs):
                if expected == "?":
                    continue
                # ``"$key"`` references another pattern node by its key.
                key = expected[1:] if expected.startswith("$") else expected
                bound = bindings.get(key)
                if bound is None or position >= len(graph_node.args):
                    ok = False
                    break
                arg = graph_node.args[position]
                if not isinstance(arg, Ref) or arg.node != bound:
                    ok = False
                    break
            if not ok:
                continue
            bindings[pattern_node.key] = graph_node.name
            bind(index + 1, bindings)
            bindings.pop(pattern_node.key, None)

    bind(0, {})
    return tuple(matches)


# ── semantic context and predicates ───────────────────────────────────────


@dataclass
class QuantDescriptor:
    """Duck-typed projection of an S05 QuantArtifact (no import required)."""

    scheme: str = ""
    group_size: Optional[int] = None
    scale_axis: int = 0
    layout_id: str = ""
    artifact_hash: str = ""
    tail_policy: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scheme": self.scheme,
            "group_size": self.group_size,
            "scale_axis": self.scale_axis,
            "layout_id": self.layout_id,
            "artifact_hash": self.artifact_hash,
            "tail_policy": self.tail_policy,
        }


@dataclass
class PatternContext:
    """Everything the predicates may look at, extracted from graph + model.

    The context is deliberately *data*: an audit run can store it and re-run the
    predicates later, which is what makes "reject has a field-level reason"
    checkable instead of impressionistic.
    """

    graph: Graph
    declared_eps: Optional[float] = None
    pattern_eps: Optional[float] = None
    norm_axis: int = -1
    hidden_axis: int = -1
    hidden_size: Optional[int] = None
    weight_shapes: Mapping[str, Tuple[int, ...]] = field(default_factory=dict)
    weight_dtype: str = "float16"
    add_alpha: Optional[float] = None
    residual_mutated: bool = False
    extra_users: Mapping[str, int] = field(default_factory=dict)
    quant: Optional[QuantDescriptor] = None
    request: Optional[Any] = None  # hqsb.integration.dispatch.CapabilityRequest
    capability: Optional[Any] = None  # dispatch.OperatorCapability
    supported_versions: Tuple[str, ...] = ()
    version: str = ""
    dynamic_bounds: Optional[Tuple[int, int]] = None
    rewrite_expected: bool = True

    def copy(self) -> "PatternContext":
        return replace(
            self,
            weight_shapes=dict(self.weight_shapes),
            extra_users=dict(self.extra_users),
        )


@dataclass(frozen=True)
class PredicateResult:
    """One predicate evaluation, including the field it looked at."""

    name: str
    passed: bool
    reason: str
    field_name: str
    expected: Any = None
    actual: Any = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "reason": self.reason,
            "field": self.field_name,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True)
class PredicateSpec:
    """Serializable description of one semantic predicate."""

    name: str
    kind: str  # "semantic" | "capability" | "dynamic" | "version"
    reason: str
    field_name: str
    expected: str
    note: str = ""

    def __post_init__(self) -> None:
        RejectReason.require(self.reason)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "reason": self.reason,
            "field": self.field_name,
            "expected": self.expected,
            "note": self.note,
        }


#: Predicate implementations.  Each returns ``(passed, actual)``; ``expected``
#: lives in the :class:`PredicateSpec` so the report can print both.
_PREDICATE_IMPLS: Dict[str, Callable[[Match, PatternContext], Tuple[bool, Any]]] = {}


def _predicate(name: str) -> Callable[[Callable[[Match, PatternContext], Tuple[bool, Any]]], Callable[[Match, PatternContext], Tuple[bool, Any]]]:
    def decorate(func: Callable[[Match, PatternContext], Tuple[bool, Any]]) -> Callable[[Match, PatternContext], Tuple[bool, Any]]:
        _PREDICATE_IMPLS[name] = func
        return func

    return decorate


@_predicate("eps_matches_model")
def _eps_matches(match: Match, ctx: PatternContext) -> Tuple[bool, Any]:
    if ctx.pattern_eps is None or ctx.declared_eps is None:
        return False, {"pattern_eps": ctx.pattern_eps, "declared_eps": ctx.declared_eps}
    return abs(float(ctx.pattern_eps) - float(ctx.declared_eps)) <= 1e-12, ctx.pattern_eps


@_predicate("norm_axis_is_hidden")
def _norm_axis(match: Match, ctx: PatternContext) -> Tuple[bool, Any]:
    return ctx.norm_axis in (-1, ctx.hidden_axis), ctx.norm_axis


@_predicate("add_has_no_alpha")
def _add_alpha(match: Match, ctx: PatternContext) -> Tuple[bool, Any]:
    return ctx.add_alpha in (None, 1.0), ctx.add_alpha


@_predicate("no_extra_user")
def _no_extra_user(match: Match, ctx: PatternContext) -> Tuple[bool, Any]:
    observed: Dict[str, Any] = {}
    ok = True
    for key, node_name in match.bindings.items():
        count = ctx.graph.user_count(node_name)
        limit = ctx.extra_users.get(node_name, 1)
        observed[key] = {"users": count, "declared_limit": limit}
        if count > limit:
            ok = False
    return ok, observed


@_predicate("residual_not_mutated")
def _residual_not_mutated(match: Match, ctx: PatternContext) -> Tuple[bool, Any]:
    return (not ctx.residual_mutated), ctx.residual_mutated


@_predicate("no_impure_node_between")
def _no_impure_between(match: Match, ctx: PatternContext) -> Tuple[bool, Any]:
    involved = set(match.bindings.values())
    impure = [node.name for node in ctx.graph.impure_nodes() if node.name in involved]
    return not impure, impure


@_predicate("dtype_supported")
def _dtype_supported(match: Match, ctx: PatternContext) -> Tuple[bool, Any]:
    supported = ("float16", "bfloat16", "float32")
    return ctx.weight_dtype in supported, ctx.weight_dtype


@_predicate("weight_shape_matches_hidden")
def _weight_shape(match: Match, ctx: PatternContext) -> Tuple[bool, Any]:
    if ctx.hidden_size is None:
        return True, None
    shapes = {name: list(shape) for name, shape in ctx.weight_shapes.items()}
    ok = all(len(shape) == 1 and shape[0] == ctx.hidden_size for shape in ctx.weight_shapes.values())
    return ok, shapes


@_predicate("quant_policy_matches_artifact")
def _quant_policy(match: Match, ctx: PatternContext) -> Tuple[bool, Any]:
    if ctx.quant is None:
        return True, None
    declared = ctx.quant
    if not declared.layout_id or not declared.artifact_hash:
        return False, declared.as_dict()
    return True, declared.as_dict()


@_predicate("quant_group_tail_allowed")
def _quant_tail(match: Match, ctx: PatternContext) -> Tuple[bool, Any]:
    if ctx.quant is None or ctx.quant.group_size is None:
        return True, None
    policy = (ctx.quant.tail_policy or "").upper()
    return policy in ("ALLOW", "PAD", "MASK"), policy


@_predicate("layout_supported")
def _layout_supported(match: Match, ctx: PatternContext) -> Tuple[bool, Any]:
    if ctx.request is None or ctx.capability is None:
        return True, "unchecked"
    decision = ctx.capability.supports(ctx.request)
    return decision.ok, decision.as_dict()


@_predicate("version_supported")
def _version_supported(match: Match, ctx: PatternContext) -> Tuple[bool, Any]:
    if not ctx.supported_versions:
        return False, {"version": ctx.version, "supported": []}
    return ctx.version in ctx.supported_versions, ctx.version


@_predicate("dynamic_bounds_known")
def _dynamic_bounds(match: Match, ctx: PatternContext) -> Tuple[bool, Any]:
    if ctx.dynamic_bounds is None:
        return True, None
    low, high = ctx.dynamic_bounds
    ok = 0 < low <= high
    return ok, list(ctx.dynamic_bounds)


# ── pattern spec ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PatternSpec:
    """A versioned pattern: IR level + structures + predicates + targets."""

    pattern_id: str
    version: str
    ir_level: str
    capture_modes: Tuple[str, ...]
    decomposition_set: str
    semantics: str
    structures: Tuple[Structure, ...]
    predicates: Tuple[PredicateSpec, ...]
    lowering_targets: Tuple[str, ...] = ()
    supported_versions: Tuple[str, ...] = ("1.0.0",)
    expected_fusion_semantics: str = ""

    def __post_init__(self) -> None:
        if self.ir_level not in IRLevel.ALL:
            raise ConfigError(
                f"{self.pattern_id}: unknown IR level {self.ir_level!r}",
                details={"field": "ir_level"},
            )
        for predicate in self.predicates:
            if predicate.name not in _PREDICATE_IMPLS:
                raise SchemaError(
                    f"{self.pattern_id}: predicate {predicate.name!r} has no implementation",
                    details={"field": "predicates"},
                )
        if not self.structures:
            raise SchemaError(
                f"{self.pattern_id}: at least one structure is required",
                details={"field": "structures"},
            )

    @property
    def signature(self) -> str:
        payload = json.dumps(
            {
                "id": self.pattern_id,
                "version": self.version,
                "ir_level": self.ir_level,
                "decomposition_set": self.decomposition_set,
                "capture_modes": list(self.capture_modes),
                "structures": [structure.as_dict() for structure in self.structures],
                "predicates": [predicate.as_dict() for predicate in self.predicates],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "version": self.version,
            "signature": self.signature,
            "ir_level": self.ir_level,
            "capture_modes": list(self.capture_modes),
            "decomposition_set": self.decomposition_set,
            "semantics": self.semantics,
            "structures": [structure.as_dict() for structure in self.structures],
            "predicates": [predicate.as_dict() for predicate in self.predicates],
            "lowering_targets": list(self.lowering_targets),
            "supported_versions": list(self.supported_versions),
            "expected_fusion_semantics": self.expected_fusion_semantics,
        }


# ── evaluation ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Decision:
    """Outcome of evaluating one candidate, with every predicate recorded."""

    candidate: Match
    status: str
    reason: str
    predicate_results: Tuple[PredicateResult, ...] = ()
    pattern_id: str = ""
    pattern_version: str = ""
    ir_level: str = ""
    capture_mode: str = ""
    lowering_targets: Tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        return self.status in (DecisionStatus.ELIGIBLE, DecisionStatus.HIT)

    @property
    def hit(self) -> bool:
        return self.status == DecisionStatus.HIT

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.as_dict(),
            "pattern_id": self.pattern_id,
            "pattern_version": self.pattern_version,
            "ir_level": self.ir_level,
            "capture_mode": self.capture_mode,
            "status": self.status,
            "reason": self.reason,
            "predicates": [item.as_dict() for item in self.predicate_results],
            "lowering_targets": list(self.lowering_targets),
        }


def evaluate_candidate(
    spec: PatternSpec,
    match: Match,
    ctx: PatternContext,
    *,
    short_circuit: bool = True,
) -> Decision:
    """Evaluate every predicate; a failure yields a field-level reject.

    ``short_circuit=False`` is the audit mode: all predicates are evaluated and
    recorded so a reviewer can see the full factor table (E06-04 §11 step 6).
    Runtime paths may short-circuit, audit runs must not.
    """
    results: List[PredicateResult] = []
    first_failure: Optional[PredicateResult] = None
    for predicate in spec.predicates:
        impl = _PREDICATE_IMPLS[predicate.name]
        passed, actual = impl(match, ctx)
        result = PredicateResult(
            name=predicate.name,
            passed=bool(passed),
            reason=predicate.reason,
            field_name=predicate.field_name,
            expected=predicate.expected,
            actual=actual,
        )
        results.append(result)
        if not passed and first_failure is None:
            first_failure = result
            if short_circuit:
                break
    if first_failure is not None:
        return Decision(
            candidate=match,
            status=DecisionStatus.REJECT,
            reason=first_failure.reason,
            predicate_results=tuple(results),
            pattern_id=spec.pattern_id,
            pattern_version=spec.version,
            ir_level=spec.ir_level,
            capture_mode=ctx.graph.capture_mode,
            lowering_targets=spec.lowering_targets,
        )
    status = DecisionStatus.HIT if spec.lowering_targets else DecisionStatus.ELIGIBLE
    return Decision(
        candidate=match,
        status=status,
        reason="PREDICATES_PASSED",
        predicate_results=tuple(results),
        pattern_id=spec.pattern_id,
        pattern_version=spec.version,
        ir_level=spec.ir_level,
        capture_mode=ctx.graph.capture_mode,
        lowering_targets=spec.lowering_targets,
    )


def scan_graph(
    spec: PatternSpec,
    graph: Graph,
    make_context: Callable[[Graph, Match], PatternContext],
    *,
    short_circuit: bool = True,
) -> Tuple[Decision, ...]:
    """Enumerate candidates and evaluate them, in one auditable pass."""
    if graph.ir_level != spec.ir_level:
        return ()
    decisions: List[Decision] = []
    for structure in spec.structures:
        for match in find_matches(graph, structure, module_path=graph.module_path):
            ctx = make_context(graph, match)
            decisions.append(evaluate_candidate(spec, match, ctx, short_circuit=short_circuit))
    return tuple(decisions)


# ── rewrite (never in place) ──────────────────────────────────────────────


@dataclass(frozen=True)
class RewriteRecord:
    """Everything a rewrite must persist (E06-04 §9)."""

    pattern_id: str
    pattern_version: str
    case_id: str
    matched_nodes: Tuple[str, ...]
    new_op: str
    new_schema_hash: str
    preserved_metadata: Mapping[str, Any] = field(default_factory=dict)
    guard_changes: Tuple[str, ...] = ()
    lowering_candidate: str = ""
    state_refs_changed: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_version": self.pattern_version,
            "case_id": self.case_id,
            "matched_nodes": list(self.matched_nodes),
            "new_op": self.new_op,
            "new_schema_hash": self.new_schema_hash,
            "preserved_metadata": dict(self.preserved_metadata),
            "guard_changes": list(self.guard_changes),
            "lowering_candidate": self.lowering_candidate,
            "state_refs_changed": list(self.state_refs_changed),
        }


@dataclass(frozen=True)
class RewriteResult:
    graph: Graph
    diff: GraphDiff
    record: RewriteRecord

    def as_dict(self) -> Dict[str, Any]:
        return {"diff": self.diff.as_dict(), "record": self.record.as_dict()}


ReplacementFactory = Callable[[Decision, PatternContext], GraphNode]


def apply_rewrite(
    graph: Graph,
    decision: Decision,
    ctx: PatternContext,
    replacement: ReplacementFactory,
    *,
    new_schema_hash: str = "",
) -> RewriteResult:
    """Apply a hit to a **copy** of the graph, returning diff + audit record."""
    if not decision.hit:
        raise ConfigError(
            f"refusing to rewrite on status {decision.status!r} (reason={decision.reason})",
            details={"pattern_id": decision.pattern_id},
        )
    working = graph.copy()
    node = replacement(decision, ctx)
    if node.name in decision.candidate.nodes:
        raise ConfigError(
            f"replacement node {node.name!r} reuses a matched node name; the rewrite "
            "would delete its own output",
            details={"field": "replacement.name"},
        )
    replaced = working.with_node(node)
    for matched in decision.candidate.nodes:
        replaced = replaced.without_node(matched)
    diff = graph.diff(replaced)
    preserved = {
        "module_path": graph.node(decision.candidate.nodes[0]).module_path,
        "source_stack": graph.node(decision.candidate.nodes[0]).source_stack,
        "capture_mode": graph.capture_mode,
        "ir_level": graph.ir_level,
    }
    record = RewriteRecord(
        pattern_id=decision.pattern_id,
        pattern_version=decision.pattern_version,
        case_id=decision.candidate.case_id,
        matched_nodes=decision.candidate.nodes,
        new_op=node.op,
        new_schema_hash=new_schema_hash,
        preserved_metadata=preserved,
        guard_changes=(),
        lowering_candidate=(decision.lowering_targets[0] if decision.lowering_targets else ""),
        state_refs_changed=(),
    )
    return RewriteResult(graph=replaced, diff=diff, record=record)


# ── coverage ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RuntimeUse:
    """Dynamic facts for one candidate (from C7 counter/timing data)."""

    case_id: str
    calls: int = 0
    total_time_ms: float = 0.0
    workload: str = ""
    phase: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "calls": self.calls,
            "total_time_ms": self.total_time_ms,
            "workload": self.workload,
            "phase": self.phase,
        }


@dataclass(frozen=True)
class CoverageReport:
    """Node/call/time/model coverage plus precision and recall (E06-04 §7)."""

    eligible_nodes: int
    rewritten_nodes: int
    eligible_calls: int
    rewritten_calls: int
    eligible_time_ms: float
    rewritten_time_ms: float
    workloads_with_hit: int
    workloads_total: int
    true_hits: int
    all_hits: int
    false_positives: int
    rejects: int

    @property
    def node_coverage(self) -> float:
        return _ratio(self.rewritten_nodes, self.eligible_nodes)

    @property
    def call_coverage(self) -> float:
        return _ratio(self.rewritten_calls, self.eligible_calls)

    @property
    def time_coverage(self) -> float:
        return _ratio(self.rewritten_time_ms, self.eligible_time_ms)

    @property
    def model_coverage(self) -> float:
        return _ratio(self.workloads_with_hit, self.workloads_total)

    @property
    def precision(self) -> float:
        return _ratio(self.true_hits, self.all_hits)

    @property
    def recall(self) -> float:
        return _ratio(self.true_hits, self.eligible_nodes)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "node_coverage": self.node_coverage,
            "call_coverage": self.call_coverage,
            "time_coverage": self.time_coverage,
            "model_coverage": self.model_coverage,
            "precision": self.precision,
            "recall": self.recall,
            "false_positives": self.false_positives,
            "rejects": self.rejects,
            "counts": {
                "eligible_nodes": self.eligible_nodes,
                "rewritten_nodes": self.rewritten_nodes,
                "eligible_calls": self.eligible_calls,
                "rewritten_calls": self.rewritten_calls,
                "workloads_with_hit": self.workloads_with_hit,
                "workloads_total": self.workloads_total,
                "true_hits": self.true_hits,
                "all_hits": self.all_hits,
            },
        }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def coverage_report(
    decisions: Sequence[Decision],
    runtime: Sequence[RuntimeUse] = (),
    *,
    workloads_total: int = 0,
    ground_truth: Optional[Mapping[str, bool]] = None,
) -> CoverageReport:
    """Aggregate coverage from decisions + dynamic call data.

    ``ground_truth`` maps ``case_id`` → whether the candidate really was
    semantically eligible (from the E06-03 differential oracle).  Only cases
    with a positive label count as true hits; everything else is a false
    positive, which the protocol requires to be zero or blocked before rewrite.
    """
    truth = dict(ground_truth or {})
    eligible = [item for item in decisions if item.eligible]
    hits = [item for item in decisions if item.hit]
    rejects = [item for item in decisions if item.status == DecisionStatus.REJECT]
    by_case = {item.case_id: item for item in runtime}
    eligible_call_total = sum(by_case.get(item.candidate.case_id, RuntimeUse(item.candidate.case_id)).calls for item in eligible)
    hit_call_total = sum(by_case.get(item.candidate.case_id, RuntimeUse(item.candidate.case_id)).calls for item in hits)
    eligible_time = sum(by_case.get(item.candidate.case_id, RuntimeUse(item.candidate.case_id)).total_time_ms for item in eligible)
    hit_time = sum(by_case.get(item.candidate.case_id, RuntimeUse(item.candidate.case_id)).total_time_ms for item in hits)
    workloads = {by_case.get(item.candidate.case_id, RuntimeUse(item.candidate.case_id)).workload for item in hits}
    workloads.discard("")
    true_hits = sum(1 for item in hits if truth.get(item.candidate.case_id, True))
    false_positives = sum(1 for item in hits if not truth.get(item.candidate.case_id, True))
    return CoverageReport(
        eligible_nodes=len(eligible),
        rewritten_nodes=len(hits),
        eligible_calls=eligible_call_total,
        rewritten_calls=hit_call_total,
        eligible_time_ms=eligible_time,
        rewritten_time_ms=hit_time,
        workloads_with_hit=len(workloads),
        workloads_total=workloads_total or len(workloads),
        true_hits=true_hits,
        all_hits=len(hits),
        false_positives=false_positives,
        rejects=len(rejects),
    )


# ── false-positive mutation testing ───────────────────────────────────────


@dataclass(frozen=True)
class Mutation:
    """One near-miss that the predicates must reject."""

    name: str
    expected_reason: str
    apply: Callable[[PatternContext], PatternContext]
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "expected_reason": self.expected_reason,
            "description": self.description,
        }


@dataclass(frozen=True)
class MutationOutcome:
    mutation: str
    expected_reason: str
    actual_status: str
    actual_reason: str

    @property
    def ok(self) -> bool:
        return (
            self.actual_status == DecisionStatus.REJECT
            and self.actual_reason == self.expected_reason
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mutation": self.mutation,
            "expected_reason": self.expected_reason,
            "actual_status": self.actual_status,
            "actual_reason": self.actual_reason,
            "ok": self.ok,
        }


def mutation_report(
    spec: PatternSpec,
    match: Match,
    ctx: PatternContext,
    mutations: Sequence[Mutation],
) -> Tuple[MutationOutcome, ...]:
    """Run every mutation and check the predicates still reject it."""
    outcomes: List[MutationOutcome] = []
    for mutation in mutations:
        mutated = mutation.apply(ctx.copy())
        decision = evaluate_candidate(spec, match, mutated, short_circuit=False)
        outcomes.append(
            MutationOutcome(
                mutation=mutation.name,
                expected_reason=mutation.expected_reason,
                actual_status=decision.status,
                actual_reason=decision.reason,
            )
        )
    return tuple(outcomes)


def standard_mutations() -> Tuple[Mutation, ...]:
    """A small default corpus so the negative path is exercised by tests.

    Every mutation changes exactly one semantic factor (E06-04 §11 step 3).
    ``SEMANTIC_EPS`` is used for scalar numeric semantics (eps, add alpha):
    both change the computed value rather than the graph structure.
    """

    def set_eps(value: Optional[float]) -> Callable[[PatternContext], PatternContext]:
        def apply(ctx: PatternContext) -> PatternContext:
            ctx.pattern_eps = value
            return ctx

        return apply

    def set_alpha(value: Optional[float]) -> Callable[[PatternContext], PatternContext]:
        def apply(ctx: PatternContext) -> PatternContext:
            ctx.add_alpha = value
            return ctx

        return apply

    def mutate_residual(ctx: PatternContext) -> PatternContext:
        ctx.residual_mutated = True
        return ctx

    def drop_version(ctx: PatternContext) -> PatternContext:
        ctx.supported_versions = ()
        return ctx

    return (
        Mutation(
            name="eps_mismatch",
            expected_reason=RejectReason.SEMANTIC_EPS,
            apply=set_eps(1e-5),
            description="different eps changes the math, not just the rounding",
        ),
        Mutation(
            name="add_alpha",
            expected_reason=RejectReason.SEMANTIC_EPS,
            apply=set_alpha(0.5),
            description="residual scaling is a different computation (numeric semantics)",
        ),
        Mutation(
            name="residual_in_place",
            expected_reason=RejectReason.ALIAS_OR_MUTATION,
            apply=mutate_residual,
            description="in-place residual write changes downstream users",
        ),
        Mutation(
            name="version_unknown",
            expected_reason=RejectReason.VERSION_UNSUPPORTED,
            apply=drop_version,
            description="unknown capture/decomposition version must default to reject",
        ),
    )


# ── the frozen pattern specs ──────────────────────────────────────────────


def residual_add_rmsnorm_pattern() -> PatternSpec:
    """residual add → RMSNorm (E06-04 §4.1), functional fused contract."""
    return PatternSpec(
        pattern_id="hqsb.pattern.residual_add_rmsnorm",
        version="1.0.0",
        ir_level=IRLevel.DYNAMO_FX,
        capture_modes=("dynamo", "export_non_strict"),
        decomposition_set="aten_default",
        semantics="r_new = residual + x; y = rms_norm(r_new, weight, eps); both fresh",
        structures=(
            Structure(
                name="aten_add_then_rmsnorm",
                nodes=(
                    PatternNode(key="add", op="aten.add.Tensor", inputs=("?", "?")),
                    PatternNode(key="norm", op="hqsb.rms_norm", inputs=("$add", "?", "?")),
                ),
            ),
        ),
        predicates=(
            PredicateSpec("eps_matches_model", "semantic", RejectReason.SEMANTIC_EPS, "eps", "equal to the model's declared eps"),
            PredicateSpec("add_has_no_alpha", "semantic", RejectReason.SEMANTIC_EPS, "alpha", "None or 1.0 (numeric semantics)"),
            PredicateSpec("norm_axis_is_hidden", "semantic", RejectReason.SEMANTIC_NORM, "norm_axis", "-1 or the hidden axis"),
            PredicateSpec("no_extra_user", "semantic", RejectReason.EXTRA_USER, "users", "1 (the consumer) unless declared"),
            PredicateSpec("residual_not_mutated", "semantic", RejectReason.ALIAS_OR_MUTATION, "residual.mutation", "no in-place write"),
            PredicateSpec("no_impure_node_between", "semantic", RejectReason.SIDE_EFFECT, "impure_nodes", "none inside the match"),
            PredicateSpec("weight_shape_matches_hidden", "semantic", RejectReason.SHAPE, "weight.shape", "[H]"),
            PredicateSpec("dtype_supported", "semantic", RejectReason.DTYPE, "weight.dtype", "fp16/bf16/fp32"),
            PredicateSpec("layout_supported", "capability", RejectReason.BACKEND_CAPABILITY, "capability", "kernel supports shape/dtype/layout"),
            PredicateSpec("version_supported", "version", RejectReason.VERSION_UNSUPPORTED, "version", "in the frozen supported set"),
        ),
        lowering_targets=("hqsb.cuda.fused_add_rms_norm", "hqsb.triton.fused_add_rms_norm"),
        supported_versions=("1.0.0",),
        expected_fusion_semantics="hqsb.integration.differential.FROZEN_ADD_RMSNORM_SEMANTICS",
    )


def rmsnorm_quantize_pattern() -> PatternSpec:
    """RMSNorm → quantize (second pattern; S05 route, E06-04 §4.2)."""
    return PatternSpec(
        pattern_id="hqsb.pattern.rmsnorm_quantize",
        version="1.0.0",
        ir_level=IRLevel.DYNAMO_FX,
        capture_modes=("dynamo",),
        decomposition_set="aten_default",
        semantics="y = rms_norm(x, weight, eps); q = quantize(y, artifact)",
        structures=(
            Structure(
                name="rmsnorm_then_quantize",
                nodes=(
                    PatternNode(key="norm", op="hqsb.rms_norm", inputs=("?", "?", "?")),
                    PatternNode(key="quant", op="hqsb.quantize", inputs=("$norm", "?")),
                ),
            ),
        ),
        predicates=(
            PredicateSpec("quant_policy_matches_artifact", "semantic", RejectReason.QUANT_POLICY, "quant.artifact", "scheme/group/layout bound"),
            PredicateSpec("quant_group_tail_allowed", "semantic", RejectReason.QUANT_POLICY, "quant.tail_policy", "ALLOW/PAD/MASK"),
            PredicateSpec("no_extra_user", "semantic", RejectReason.EXTRA_USER, "users", "1 (the consumer) unless declared"),
            PredicateSpec("no_impure_node_between", "semantic", RejectReason.SIDE_EFFECT, "impure_nodes", "none"),
            PredicateSpec("dtype_supported", "semantic", RejectReason.DTYPE, "dtype", "quant-supported dtype"),
            PredicateSpec("layout_supported", "capability", RejectReason.BACKEND_CAPABILITY, "capability", "low-bit kernel supports the layout"),
            PredicateSpec("version_supported", "version", RejectReason.VERSION_UNSUPPORTED, "version", "in the frozen supported set"),
        ),
        lowering_targets=("hqsb.triton.dequant_linear",),
        supported_versions=("1.0.0",),
        expected_fusion_semantics="S05 quant/dequant math unchanged; layout id part of the identity",
    )


def frozen_patterns() -> Tuple[PatternSpec, ...]:
    return (residual_add_rmsnorm_pattern(), rmsnorm_quantize_pattern())


class PatternRegistry:
    """Versioned registry: one owner per (id, version), never silent overwrite."""

    def __init__(self) -> None:
        self._specs: Dict[Tuple[str, str], PatternSpec] = {}

    def register(self, spec: PatternSpec, *, replace_existing: bool = False) -> None:
        key = (spec.pattern_id, spec.version)
        if key in self._specs and not replace_existing:
            raise SchemaError(
                f"pattern {spec.pattern_id} version {spec.version} already registered; "
                "bump the version instead of overwriting it",
                details={"field": "pattern_id"},
            )
        if key in self._specs and replace_existing:
            existing = self._specs[key]
            if existing.signature == spec.signature:
                return
            raise SchemaError(
                f"refusing to replace {spec.pattern_id}@{spec.version} with different "
                "content; bump the version",
                details={"field": "signature"},
            )
        self._specs[key] = spec

    def get(self, pattern_id: str, version: str) -> PatternSpec:
        try:
            return self._specs[(pattern_id, version)]
        except KeyError as exc:
            raise SchemaError(
                f"pattern {pattern_id}@{version} is not registered",
                details={"field": "pattern_id"},
            ) from exc

    def all(self) -> Tuple[PatternSpec, ...]:
        return tuple(self._specs[key] for key in sorted(self._specs))

    @property
    def ir_levels(self) -> Tuple[str, ...]:
        return tuple(sorted({spec.ir_level for spec in self._specs.values()}))

    def as_dict(self) -> Dict[str, Any]:
        return {"patterns": [spec.as_dict() for spec in self.all()]}


def frozen_registry() -> PatternRegistry:
    registry = PatternRegistry()
    for spec in frozen_patterns():
        registry.register(spec)
    return registry


__all__ = [
    "Decision",
    "DecisionStatus",
    "Match",
    "Mutation",
    "MutationOutcome",
    "PatternContext",
    "PatternNode",
    "PatternRegistry",
    "PatternSpec",
    "PredicateResult",
    "PredicateSpec",
    "QuantDescriptor",
    "RejectReason",
    "RewriteRecord",
    "RewriteResult",
    "RuntimeUse",
    "CoverageReport",
    "Structure",
    "apply_rewrite",
    "coverage_report",
    "evaluate_candidate",
    "find_matches",
    "frozen_patterns",
    "frozen_registry",
    "mutation_report",
    "residual_add_rmsnorm_pattern",
    "rmsnorm_quantize_pattern",
    "scan_graph",
    "standard_mutations",
]
