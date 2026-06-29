"""Semantic pattern matching, legality proofs, atomic rewrites, pass discipline.

Protocol anchor: ``details/S11/E11-02_semantic_pattern_rewrite_idempotence.md``.
The engine separates the four responsibilities the protocol demands (§8):

1. ``structural matcher`` — candidate discovery only (roles, edges, users);
2. ``semantic predicates`` — dtype/cast, shape/reduction, epsilon, users,
   alias/mutation/effect, layout, return contract, each with an explicit proof
   source and a fail-closed ``UNKNOWN`` outcome;
3. ``replacement builder`` — one versioned semantic op, copied metadata and a
   referenced fallback subgraph;
4. ``proof record`` — every predicate's expected/actual/outcome/reason, plus
   the pass version, so a rejection can be explained without re-running.

Rewrites are atomic by construction: the input graph is never mutated; a new
graph is returned only after the replacement passes the IR verifier.  The
pipeline adds determinism, fixed-point iteration and idempotence accounting.

This module never executes a model and never decides profitability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import canonical_json, sha256_text
from hqsb.compiler.ir import (
    EffectSet,
    IRGraph,
    IRValue,
    IROp,
    IRVerifier,
    import_op_sequence,
)
from hqsb.compiler.pattern_library import (
    ANCHOR_ROLE,
    MATCH,
    NO_CHANGE,
    OUTCOME_PASS,
    OUTCOME_REJECT,
    OUTCOME_UNKNOWN,
    PATTERN_RESIDUAL_RMSNORM,
    PATTERN_VERSION,
    PREDICATE_CATALOG,
    PREDICATE_ORDER,
    REJECT,
    ROLE_CHAIN_RESIDUAL_RMSNORM,
)
from hqsb.compiler.records import PatternDecisionRecord

#: Epsilon values the v0 pattern is frozen against.  A different value is a
#: different mathematical function, not a tuning knob.
ALLOWED_EPS = (1e-5, 1e-6, 1e-4)

#: Cast operations that may appear inside the pattern chain (declared
#: decomposition variants).  Anything else is a structural mismatch.
DECOMPOSITION_CAST_OPS: Tuple[str, ...] = ("aten.to",)

#: Failure-injection points (E11-02 step 30).
INJECTION_POINTS: Tuple[str, ...] = ("matcher", "replacement", "verifier", "commit")

#: Upper bound on committed rewrites per pass (termination guarantee).
MAX_SITES_PER_PASS = 64


# ── graph construction from declarative specs ──────────────────────────────


def build_graph(spec: Mapping[str, Any], *, level: str = "HQSB_CANONICAL") -> IRGraph:
    """Build a verifiable IR graph from a fixture/ingest spec."""
    constraints = [
        _constraint(item) for item in spec.get("constraints", ())
    ]
    return import_op_sequence(
        graph_id=spec["graph_id"],
        ops=spec["ops"],
        inputs=spec.get("inputs", ()),
        outputs=spec.get("outputs", ()),
        constraints=tuple(constraints),
        level=level,
        source_graph_id=spec.get("source_graph_id", ""),
        target_id=spec.get("target_id", ""),
    )


def _constraint(item: Mapping[str, Any]):
    from hqsb.compiler.ir import ShapeConstraint

    return ShapeConstraint(
        constraint_id=item["constraint_id"],
        expression=item["expression"],
        category=item.get("category", "semantic"),
        origin=item.get("origin", "user"),
        lower=item.get("lower"),
        upper=item.get("upper"),
        divisibility=item.get("divisibility"),
        relation=item.get("relation", ""),
        semantic_required=item.get("semantic_required", True),
        verifier_status=item.get("verifier_status", "VERIFIED"),
    )


# ── structural matcher (step 5, 19) ────────────────────────────────────────


@dataclass
class StructuralCandidate:
    """A candidate site with its role binding and the closest miss level."""

    site_id: str
    role_ops: Mapping[str, str] = field(default_factory=dict)
    value_bindings: Mapping[str, str] = field(default_factory=dict)
    cast_ops: Tuple[str, ...] = ()
    matched_roles: int = 0
    missing_role: str = ""
    complete: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "site_id": self.site_id,
            "role_ops": dict(self.role_ops),
            "value_bindings": dict(self.value_bindings),
            "cast_ops": list(self.cast_ops),
            "matched_roles": self.matched_roles,
            "missing_role": self.missing_role,
            "complete": self.complete,
        }


def _producer_map(graph: IRGraph) -> Dict[str, str]:
    return {value_id: op.op_id for op in graph.ops for value_id in op.results}


def _consumers_map(graph: IRGraph) -> Dict[str, List[str]]:
    users: Dict[str, List[str]] = {}
    for op in graph.ops:
        for operand in op.operands:
            users.setdefault(operand, []).append(op.op_id)
    return users


def _consumers_index(graph: IRGraph) -> Dict[str, List[Tuple[str, int]]]:
    """``value_id -> [(op_id, operand_index), ...]`` (positions matter)."""
    index: Dict[str, List[Tuple[str, int]]] = {}
    for op in graph.ops:
        for position, operand in enumerate(op.operands):
            index.setdefault(operand, []).append((op.op_id, position))
    return index


def _consumers_through_casts(
    graph: IRGraph, value_id: str, consumers: Mapping[str, Sequence[Tuple[str, int]]]
) -> List[Tuple[IROp, int, Tuple[str, ...]]]:
    """Consumers reachable from ``value_id`` through declared cast decompositions."""
    results: List[Tuple[IROp, int, Tuple[str, ...]]] = []
    frontier: List[Tuple[str, Tuple[str, ...]]] = [(value_id, ())]
    seen = {value_id}
    while frontier:
        current, chain = frontier.pop(0)
        for op_id, position in consumers.get(current, ()):
            op = graph.op(op_id)
            if (
                op.semantic_op in DECOMPOSITION_CAST_OPS
                and len(op.operands) == 1
                and op.results
            ):
                result = op.results[0]
                if result not in seen:
                    seen.add(result)
                    frontier.append((result, chain + (op.op_id,)))
                continue
            results.append((op, position, chain))
    return results


def _resolve_through_casts(
    graph: IRGraph, value_id: str, producers: Mapping[str, str]
) -> Tuple[str, Tuple[str, ...]]:
    """Skip declared decomposition casts; return ``(base_value, cast_ops)``."""
    casts: List[str] = []
    current = value_id
    for _ in range(8):
        producer = producers.get(current, "")
        if not producer:
            return current, tuple(casts)
        op = graph.op(producer)
        if op.semantic_op in DECOMPOSITION_CAST_OPS and len(op.operands) == 1:
            casts.append(op.op_id)
            current = op.operands[0]
            continue
        return current, tuple(casts)
    raise ConfigError("cast chain too deep while resolving pattern roles")


def find_candidates(
    graph: IRGraph,
    *,
    chain: Sequence[
        Tuple[str, Tuple[str, ...], Tuple[Tuple[str, int], ...], Optional[Tuple[str, int]]]
    ] = ROLE_CHAIN_RESIDUAL_RMSNORM,
) -> List[StructuralCandidate]:
    """Discover candidate sites; never decide semantics or profitability.

    The walk starts at the anchor role (whose operands must be graph inputs)
    and follows the declared consumer links forward through declared cast
    decompositions.  A partially matched anchor is still reported with the
    role where the chain broke, so the proof record can say *how close* the
    site was instead of dropping it.
    """
    producers = _producer_map(graph)
    consumers = _consumers_index(graph)
    input_ids = set(graph.inputs)
    chain_by_role = {role: (names, operands, link) for role, names, operands, link in chain}
    order = [role for role, _, _, _ in chain]
    anchor_names, anchor_operands, _ = chain_by_role[ANCHOR_ROLE]
    candidates: List[StructuralCandidate] = []
    for anchor in graph.ops:
        if anchor.semantic_op not in anchor_names or len(anchor.operands) < 2:
            continue
        site = StructuralCandidate(site_id=f"site_{anchor.op_id}")
        roles: Dict[str, str] = {ANCHOR_ROLE: anchor.op_id}
        values: Dict[str, str] = {}
        casts: List[str] = []
        missing = ""
        matched = 1
        input_operands = 0
        for value_role, index in anchor_operands:
            if index >= len(anchor.operands):
                missing = f"{ANCHOR_ROLE}:{value_role}"
                break
            value_id = anchor.operands[index]
            if value_id in input_ids:
                input_operands += 1
            values[value_role] = value_id
        if not missing and input_operands < 2:
            missing = f"{ANCHOR_ROLE}:inputs"
        previous_value = anchor.results[0] if anchor.results else ""
        for role in order[1:]:
            if missing:
                break
            names, operand_roles, link = chain_by_role[role]
            expected_index = chain_by_role[order[order.index(role) - 1]][2]
            if expected_index is None:
                missing = f"{role}:link"
                break
            _, consumer_index = expected_index
            found: Optional[Tuple[IROp, Tuple[str, ...]]] = None
            for op, position, cast_chain in _consumers_through_casts(
                graph, previous_value, consumers
            ):
                if op.semantic_op in names and position == consumer_index:
                    found = (op, cast_chain)
                    break
            if found is None:
                missing = role
                break
            op, cast_chain = found
            roles[role] = op.op_id
            matched += 1
            casts.extend(cast_chain)
            for value_role, index in operand_roles:
                if index >= len(op.operands):
                    missing = f"{role}:{value_role}"
                    break
                base, cast_ops = _resolve_through_casts(graph, op.operands[index], producers)
                casts.extend(cast_ops)
                if value_role in values and values[value_role] != base:
                    missing = f"{role}:{value_role}_mismatch"
                    break
                values[value_role] = base
            previous_value = op.results[0] if op.results else ""
        if not missing:
            bad = [
                role
                for role in ("x", "residual", "weight", "eps_const")
                if values.get(role) not in input_ids
            ]
            if bad:
                missing = ",".join(f"{role}:not_input" for role in bad)
        site.role_ops = roles
        site.value_bindings = values
        site.cast_ops = tuple(sorted(set(casts)))
        site.matched_roles = matched
        site.missing_role = missing
        site.complete = not missing and matched == len(order)
        candidates.append(site)
    return candidates


# ── semantic predicates (steps 6–13, 21) ───────────────────────────────────


@dataclass
class PredicateResult:
    """One predicate's expected/actual/outcome with its proof source."""

    name: str
    expected: Any
    actual: Any
    outcome: str
    proof_source: str
    reject_code: str = ""
    detail: str = ""
    source_nodes: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "expected": self.expected,
            "actual": self.actual,
            "outcome": self.outcome,
            "proof_source": self.proof_source,
            "reject_code": self.reject_code,
            "detail": self.detail,
            "source_nodes": list(self.source_nodes),
        }


def _reject(name: str, expected: Any, actual: Any, code: str, detail: str, nodes: Sequence[str]) -> PredicateResult:
    return PredicateResult(
        name=name,
        expected=expected,
        actual=actual,
        outcome=OUTCOME_REJECT,
        proof_source=PREDICATE_CATALOG[name]["purpose"],
        reject_code=code,
        detail=detail,
        source_nodes=tuple(nodes),
    )


def _pass(name: str, expected: Any, actual: Any, detail: str = "", nodes: Sequence[str] = ()) -> PredicateResult:
    return PredicateResult(
        name=name,
        expected=expected,
        actual=actual,
        outcome=OUTCOME_PASS,
        proof_source=PREDICATE_CATALOG[name]["purpose"],
        detail=detail,
        source_nodes=tuple(nodes),
    )


def _unknown(name: str, expected: Any, actual: Any, code: str, detail: str, nodes: Sequence[str]) -> PredicateResult:
    return PredicateResult(
        name=name,
        expected=expected,
        actual=actual,
        outcome=OUTCOME_UNKNOWN,
        proof_source=PREDICATE_CATALOG[name]["purpose"],
        reject_code=code,
        detail=detail,
        source_nodes=tuple(nodes),
    )


def evaluate_predicates(
    graph: IRGraph, candidate: StructuralCandidate
) -> List[PredicateResult]:
    """Run every predicate in the frozen order; fail closed on unknowns."""
    results: List[PredicateResult] = []
    if not candidate.complete:
        return [
            _reject(
                "structural_match",
                "complete role binding",
                {"matched_roles": candidate.matched_roles, "missing": candidate.missing_role},
                "PATTERN_NEAR_MISS",
                "role chain incomplete: the site is a near-miss, not a candidate",
                tuple(candidate.role_ops.values()),
            )
        ]
    role_ops = candidate.role_ops
    ops = [graph.op(role_ops[role]) for role in role_ops]
    add_op = graph.op(role_ops["add_residual"])
    mean_op = graph.op(role_ops["mean"])
    eps_op = graph.op(role_ops["eps_add"])
    pow_op = graph.op(role_ops["pow"])
    scale_op = graph.op(role_ops["scale"])
    results.append(
        _pass(
            "structural_match",
            "roles bound in order",
            list(role_ops),
            "matcher found a complete role assignment",
            tuple(op.op_id for op in ops),
        )
    )

    # 2) dtype / cast
    cast_dtypes = sorted(
        {str(graph.op(cast_id).attrs.get("dtype", "unknown")) for cast_id in candidate.cast_ops}
    )
    bad_casts = [dtype for dtype in cast_dtypes if dtype != "fp32"]
    if bad_casts:
        results.append(
            _reject(
                "dtype_cast",
                "casts inside the pattern only to the declared accumulation dtype (fp32)",
                {"casts": candidate.cast_ops, "targets": bad_casts},
                "PATTERN_NEAR_MISS",
                "the reduction input is rounded to storage precision: accumulation differs",
                candidate.cast_ops,
            )
        )
    else:
        results.append(
            _pass(
                "dtype_cast",
                "consistent operand/accumulator dtypes",
                {"casts": candidate.cast_ops, "targets": cast_dtypes},
                "only declared decomposition casts present",
                candidate.cast_ops,
            )
        )

    # 3) shape / reduction
    dim = mean_op.attrs.get("dim")
    keepdim = mean_op.attrs.get("keepdim")
    exponent = pow_op.attrs.get("exponent")
    weight_value = graph.value(candidate.value_bindings["weight"])
    weight_rank = len(weight_value.type.dims) if weight_value.type else 0
    hidden_relation = [
        item.expression
        for item in graph.constraints
        if "H" in item.expression or "hidden" in item.expression
    ]
    reduction_ok = (
        mean_op.semantic_op == "aten.mean"
        and dim in (-1, weight_rank and -1)
        and keepdim is True
        and exponent in (2, 2.0)
        and weight_rank == 1
    )
    if not reduction_ok:
        results.append(
            _reject(
                "shape_reduction",
                "mean(dim=-1, keepdim=True), exponent 2, rank-1 weight",
                {
                    "op": mean_op.semantic_op,
                    "dim": dim,
                    "keepdim": keepdim,
                    "exponent": exponent,
                    "weight_rank": weight_rank,
                },
                "PATTERN_NEAR_MISS",
                "reduction/broadcast semantics differ from the contract",
                (mean_op.op_id, pow_op.op_id),
            )
        )
    elif not hidden_relation:
        results.append(
            _unknown(
                "shape_reduction",
                "an explicit hidden-size relation",
                [],
                "PATTERN_NEAR_MISS",
                "no hidden divisibility/tail constraint recorded: the variant domain is unproven",
                (mean_op.op_id,),
            )
        )
    else:
        results.append(
            _pass(
                "shape_reduction",
                "mean(dim=-1, keepdim=True), exponent 2, rank-1 weight",
                {"relations": hidden_relation},
                "",
                (mean_op.op_id,),
            )
        )

    # 4) epsilon
    eps_value = eps_op.attrs.get("value")
    eps_input = candidate.value_bindings.get("eps_const", "")
    eps_position_ok = eps_op.operands and eps_op.operands[0] == candidate.value_bindings.get("variance")
    if not eps_position_ok:
        results.append(
            _reject(
                "epsilon",
                "eps added to the variance before rsqrt",
                list(eps_op.operands),
                "PATTERN_NEAR_MISS",
                "eps participates at a different point of the expression",
                (eps_op.op_id,),
            )
        )
    elif eps_value is None:
        results.append(
            _unknown(
                "epsilon",
                f"literal in {ALLOWED_EPS} or a declared runtime scalar",
                {"eps_input": eps_input, "value": None},
                "PATTERN_NEAR_MISS",
                "runtime-dependent epsilon without a declared domain: fail closed",
                (eps_op.op_id,),
            )
        )
    elif float(eps_value) not in ALLOWED_EPS:
        results.append(
            _reject(
                "epsilon",
                list(ALLOWED_EPS),
                float(eps_value),
                "PATTERN_NEAR_MISS",
                "epsilon outside the frozen allowed set is a different function",
                (eps_op.op_id,),
            )
        )
    else:
        results.append(
            _pass(
                "epsilon",
                list(ALLOWED_EPS),
                float(eps_value),
                "",
                (eps_op.op_id,),
            )
        )

    # 5) users / liveness
    consumers = _consumers_map(graph)
    pattern_op_ids = set(role_ops.values()) | set(candidate.cast_ops)
    residual_value = candidate.value_bindings["residual_new"]
    external_users = [
        user for user in consumers.get(residual_value, []) if user not in pattern_op_ids
    ]
    returns_residual = residual_value in graph.outputs
    if external_users and not returns_residual:
        results.append(
            _reject(
                "users_liveness",
                "the replacement returns residual_new when it has extra users",
                {"external_users": external_users, "outputs": list(graph.outputs)},
                "PATTERN_NEAR_MISS",
                "residual_new is consumed outside the pattern but the replacement drops it",
                tuple(external_users),
            )
        )
    else:
        results.append(
            _pass(
                "users_liveness",
                "all extra users preserved",
                {"external_users": external_users, "returns_residual": returns_residual},
                "" if not external_users else "replacement must return residual_new exactly",
                tuple(external_users),
            )
        )

    # 6) alias / mutation / effect
    add_effects = add_op.effects
    if add_effects.mutation == "inplace" or "ph_r" in add_effects.writes:
        results.append(
            _reject(
                "alias_mutation_effect",
                "out-of-place residual add",
                add_effects.as_dict(),
                "MUTATION_UNSAFE",
                "the original graph writes back to residual; a functional replacement "
                "would change the observable state",
                (add_op.op_id,),
            )
        )
    elif add_effects.risk_state() == "UNKNOWN_UNSAFE":
        results.append(
            _reject(
                "alias_mutation_effect",
                "PROVED_SAFE alias/mutation",
                add_effects.as_dict(),
                "ALIAS_UNSAFE",
                "effect unknown: fail closed instead of assuming purity",
                (add_op.op_id,),
            )
        )
    else:
        results.append(
            _pass(
                "alias_mutation_effect",
                "PROVED_SAFE",
                add_effects.risk_state(),
                "",
                (add_op.op_id,),
            )
        )

    # 7) layout
    layout_changes_alias = bool(add_op.attrs.get("layout_changes_alias", False))
    if layout_changes_alias:
        results.append(
            _reject(
                "layout",
                "layout does not change alias semantics",
                {"layout_changes_alias": True},
                "ALIAS_UNSAFE",
                "the layout/view changes alias identity; the semantic pass cannot authorise it",
                (scale_op.op_id,),
            )
        )
    else:
        results.append(
            _pass(
                "layout",
                "unsupported strides may be deferred to the lowering guard",
                {"layout_changes_alias": False},
                "candidate-capability conditions stay with E11-03 guards/fallback",
                (scale_op.op_id,),
            )
        )

    # 8) return contract
    if residual_value not in graph.outputs and external_users:
        results.append(
            _reject(
                "return_contract",
                "outputs include every externally consumed value",
                {"outputs": list(graph.outputs), "external_users": external_users},
                "PATTERN_NEAR_MISS",
                "replacement output arity does not match the original subgraph",
                (graph.op(role_ops["weight_mul"]).op_id,),
            )
        )
    else:
        results.append(
            _pass(
                "return_contract",
                "normalized (+ residual_new when consumed)",
                list(graph.outputs),
                "",
                (graph.op(role_ops["weight_mul"]).op_id,),
            )
        )
    return results


# ── decisions and atomic rewrite (steps 12–14, 20, 30) ─────────────────────


@dataclass
class Decision:
    """The full decision record for one candidate site."""

    decision_id: str
    site_id: str
    pattern_id: str
    pattern_version: str
    structural_match: bool
    predicates: Tuple[PredicateResult, ...]
    before_ir_id: str
    after_ir_id: str = ""
    status: str = MATCH
    reject_code: str = ""
    rewrite_count: int = 0
    preserved_outputs: Tuple[str, ...] = ()
    source_nodes: Tuple[str, ...] = ()
    source_locations: Tuple[str, ...] = ()
    alias_effect_summary: Mapping[str, Any] = field(default_factory=dict)
    target_candidates: Tuple[Mapping[str, Any], ...] = ()
    selected_id: str = ""
    fallback_id: str = ""

    def reject_reason(self) -> Dict[str, Any]:
        for predicate in self.predicates:
            if predicate.outcome in (OUTCOME_REJECT, OUTCOME_UNKNOWN):
                return {
                    "predicate": predicate.name,
                    "code": predicate.reject_code,
                    "detail": predicate.detail,
                    "expected": predicate.expected,
                    "actual": predicate.actual,
                }
        return {}

    def to_record(self) -> PatternDecisionRecord:
        return PatternDecisionRecord(
            decision_id=self.decision_id,
            source_node_ids=self.source_nodes,
            source_locations=self.source_locations,
            pattern_id=self.pattern_id,
            pattern_version=self.pattern_version,
            structural_match=self.structural_match,
            semantic_predicates=tuple(item.as_dict() for item in self.predicates),
            alias_effect_summary=dict(self.alias_effect_summary),
            target_candidates=self.target_candidates,
            selected_id=self.selected_id,
            fallback_id=self.fallback_id,
            before_ir_id=self.before_ir_id,
            after_ir_id=self.after_ir_id,
            reject_code=self.reject_code,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "site_id": self.site_id,
            "pattern_id": self.pattern_id,
            "pattern_version": self.pattern_version,
            "structural_match": self.structural_match,
            "status": self.status,
            "reject_code": self.reject_code,
            "reject_reason": self.reject_reason(),
            "predicates": [item.as_dict() for item in self.predicates],
            "before_ir_id": self.before_ir_id,
            "after_ir_id": self.after_ir_id,
            "rewrite_count": self.rewrite_count,
            "preserved_outputs": list(self.preserved_outputs),
            "source_nodes": list(self.source_nodes),
            "source_locations": list(self.source_locations),
            "fallback_id": self.fallback_id,
        }


def decide(
    graph: IRGraph, candidate: StructuralCandidate, *, decision_id: str = ""
) -> Decision:
    """Evaluate a candidate into a MATCH/REJECT decision with proof records."""
    predicates = tuple(evaluate_predicates(graph, candidate))
    failure = next(
        (item for item in predicates if item.outcome in (OUTCOME_REJECT, OUTCOME_UNKNOWN)), None
    )
    source_nodes = tuple(candidate.role_ops.values())
    source_locations = tuple(
        str(graph.op(op_id).source.get("module_path", "")) for op_id in source_nodes
    )
    status = REJECT if failure else MATCH
    return Decision(
        decision_id=decision_id or f"dec_{candidate.site_id}",
        site_id=candidate.site_id,
        pattern_id=PATTERN_RESIDUAL_RMSNORM,
        pattern_version=PATTERN_VERSION,
        structural_match=candidate.complete,
        predicates=predicates,
        before_ir_id=graph.graph_id,
        status=status,
        reject_code=failure.reject_code if failure else "",
        preserved_outputs=tuple(
            graph.outputs
        ) if not failure else (),
        source_nodes=source_nodes,
        source_locations=source_locations,
        alias_effect_summary={
            row.name: {"outcome": row.outcome, "actual": row.actual}
            for row in predicates
            if row.name == "alias_mutation_effect"
        },
    )


def decide_already_fused(graph: IRGraph) -> Optional[Decision]:
    """``already fused`` is NO_CHANGE, not a successful rewrite (E11-02 §12)."""
    fused = [
        op
        for op in graph.ops
        if op.semantic_op == "hqsb::fused_add_rms_norm" and op.pattern_id == PATTERN_RESIDUAL_RMSNORM
    ]
    if not fused:
        return None
    return Decision(
        decision_id=f"nochange_{fused[0].op_id}",
        site_id=f"site_{fused[0].op_id}",
        pattern_id=PATTERN_RESIDUAL_RMSNORM,
        pattern_version=PATTERN_VERSION,
        structural_match=True,
        predicates=(
            PredicateResult(
                name="structural_match",
                expected="decomposed pattern",
                actual={"semantic_op": fused[0].semantic_op},
                outcome=OUTCOME_PASS,
                proof_source=PREDICATE_CATALOG["structural_match"]["purpose"],
                detail="graph already contains the semantic op: no change, not a rewrite",
            ),
        ),
        before_ir_id=graph.graph_id,
        after_ir_id=graph.graph_id,
        status=NO_CHANGE,
        source_nodes=(fused[0].op_id,),
    )


@dataclass
class RewriteOutcome:
    graph: IRGraph
    decision: Decision
    changed: bool
    committed: bool
    error: str = ""
    verifier_ok: bool = False
    verifier_issues: Tuple[Mapping[str, Any], ...] = ()
    node_mapping: Mapping[str, str] = field(default_factory=dict)
    fallback_subgraph: Mapping[str, Any] = field(default_factory=dict)
    injection_point: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.as_dict(),
            "changed": self.changed,
            "committed": self.committed,
            "error": self.error,
            "verifier_ok": self.verifier_ok,
            "verifier_issues": [dict(item) for item in self.verifier_issues],
            "node_mapping": dict(sorted(self.node_mapping.items())),
            "fallback_subgraph_graph_id": self.fallback_subgraph.get("graph_id", ""),
            "injection_point": self.injection_point,
        }


def apply_rewrite(
    graph: IRGraph,
    candidate: StructuralCandidate,
    *,
    decision_id: str = "",
    inject_at: str = "",
    verifier: Optional[IRVerifier] = None,
) -> RewriteOutcome:
    """Replace the candidate with the fused semantic op — atomically.

    The original ``graph`` object is never modified.  On any injected or real
    failure the caller receives ``changed=False`` plus the untouched graph and
    a structured error, so a half-rewritten graph cannot leak out.
    """
    if inject_at and inject_at not in INJECTION_POINTS:
        raise ConfigError(f"unknown injection point {inject_at!r}; known: {list(INJECTION_POINTS)}")
    decision = decide(graph, candidate, decision_id=decision_id)
    if inject_at == "matcher":
        return _failed_rewrite(
            graph, decision, "INJECTED_MATCHER_FAILURE", inject_at, "matcher failure injected"
        )
    if decision.status != MATCH:
        return _failed_rewrite(
            graph,
            decision,
            "REJECTED_NOT_LEGAL",
            inject_at,
            f"candidate rejected by predicate {decision.reject_reason().get('predicate', '')}",
        )
    if inject_at == "replacement":
        return _failed_rewrite(
            graph, decision, "INJECTED_REPLACEMENT_FAILURE", inject_at, "replacement failure injected"
        )

    role_ops = set(candidate.role_ops.values()) | set(candidate.cast_ops)
    residual_value = candidate.value_bindings["residual_new"]
    output_value = graph.op(candidate.role_ops["weight_mul"]).results[0]
    x_value = candidate.value_bindings["x"]
    residual_input = candidate.value_bindings["residual"]
    weight_value = candidate.value_bindings["weight"]
    eps_value = candidate.value_bindings["eps_const"]
    removed_ops = [op for op in graph.ops if op.op_id in role_ops]
    kept_ops = [op for op in graph.ops if op.op_id not in role_ops]
    fused_source = dict(graph.op(candidate.role_ops["add_residual"]).source)
    fused_source["pattern_id"] = PATTERN_RESIDUAL_RMSNORM
    fused_source["pattern_version"] = PATTERN_VERSION
    fused_source["replaced_ops"] = sorted(role_ops)
    fused = IROp(
        op_id=f"fused_{candidate.role_ops['add_residual']}",
        semantic_op="hqsb::fused_add_rms_norm",
        operands=(x_value, residual_input, weight_value, eps_value),
        results=(output_value, residual_value),
        schema_version="1.0.0",
        attrs={
            "eps": float(graph.op(candidate.role_ops["eps_add"]).attrs.get("value", 0.0)),
            "accumulate_dtype": "fp32",
            "multi_output": True,
        },
        effects=EffectSet(evidence="schema", mutation="none"),
        source=fused_source,
        target_legal={},
        candidates=(),
        fallback=f"graph:{graph.graph_id}",
        pattern_id=PATTERN_RESIDUAL_RMSNORM,
    )
    reproducer = {output_value: fused.op_id, residual_value: fused.op_id}
    values: List[IRValue] = []
    for value in graph.values:
        if value.value_id in reproducer:
            values.append(
                IRValue(
                    value_id=value.value_id,
                    type=value.type,
                    producer=reproducer[value.value_id],
                    users=value.users,
                    source_lineage=value.source_lineage,
                    is_input=value.is_input,
                    is_output=value.is_output,
                )
            )
        elif value.producer in role_ops:
            # intermediate value of the replaced subgraph: no longer produced
            continue
        else:
            values.append(value)
    if all(value.value_id != output_value for value in values):
        raise ConfigError(f"rewrite lost the output value {output_value!r}")
    rebuilt = IRGraph(
        graph_id=graph.graph_id,
        level=graph.level,
        ops=tuple([*kept_ops, fused]),
        values=tuple(values),
        inputs=graph.inputs,
        outputs=graph.outputs,
        constraints=graph.constraints,
        schema_version=graph.schema_version,
        source_graph_id=graph.source_graph_id,
        target_id=graph.target_id,
        guards=graph.guards,
        fallback={
            "graph_id": graph.graph_id,
            "reason": "original decomposed subgraph retained for rollback/fallback",
            "removed_ops": [op.as_dict() for op in removed_ops],
        },
    )
    if inject_at == "verifier":
        return _failed_rewrite(
            graph, decision, "INJECTED_VERIFIER_FAILURE", inject_at, "verifier failure injected"
        )
    report = (verifier or IRVerifier()).verify(rebuilt)
    if not report.ok:
        return RewriteOutcome(
            graph=graph,
            decision=decision,
            changed=False,
            committed=False,
            error="rewritten graph failed the IR verifier",
            verifier_ok=False,
            verifier_issues=tuple(item.as_dict() for item in report.issues),
            fallback_subgraph=rebuilt.fallback,
            injection_point=inject_at,
        )
    if inject_at == "commit":
        return _failed_rewrite(
            graph, decision, "INJECTED_COMMIT_FAILURE", inject_at, "commit failure injected"
        )
    committed_decision = Decision(
        **{
            **decision.__dict__,
            "after_ir_id": rebuilt.graph_id,
            "rewrite_count": 1,
        }
    )
    return RewriteOutcome(
        graph=rebuilt,
        decision=committed_decision,
        changed=True,
        committed=True,
        verifier_ok=True,
        node_mapping={op.op_id: fused.op_id for op in removed_ops},
        fallback_subgraph=rebuilt.fallback,
        injection_point=inject_at,
    )


def _failed_rewrite(
    graph: IRGraph, decision: Decision, error: str, inject_at: str, detail: str
) -> RewriteOutcome:
    return RewriteOutcome(
        graph=graph,
        decision=decision,
        changed=False,
        committed=False,
        error=f"{error}: {detail}",
        verifier_ok=False,
        injection_point=inject_at,
    )


# ── pass pipeline: determinism, fixed point, idempotence (steps 28–29) ─────


@dataclass
class PassOutcome:
    graph: IRGraph
    decisions: Tuple[Decision, ...]
    rewrite_count: int
    reject_count: int
    no_change_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph.graph_id,
            "rewrite_count": self.rewrite_count,
            "reject_count": self.reject_count,
            "no_change_count": self.no_change_count,
            "decisions": [item.as_dict() for item in self.decisions],
        }


def run_fused_add_rmsnorm_pass(graph: IRGraph) -> PassOutcome:
    """One deterministic pass over every candidate site.

    Sites are processed in ascending ``site_id`` order and the graph is
    re-scanned after each commit, so a rewrite can never be applied against a
    stale op id; repeated and parallel runs produce the same canonical output.
    """
    decisions: List[Decision] = []
    current = graph
    rewrite_count = 0
    reject_count = 0
    already = decide_already_fused(current)
    if already is not None:
        decisions.append(already)
        return PassOutcome(
            graph=current,
            decisions=tuple(decisions),
            rewrite_count=0,
            reject_count=0,
            no_change_count=1,
        )
    seen_incomplete: set = set()
    for _ in range(MAX_SITES_PER_PASS):
        candidates = find_candidates(current)
        complete = sorted(
            (item for item in candidates if item.complete), key=lambda item: item.site_id
        )
        for candidate in candidates:
            if candidate.complete or candidate.site_id in seen_incomplete:
                continue
            seen_incomplete.add(candidate.site_id)
            decisions.append(decide(current, candidate))
            reject_count += 1
        if not complete:
            break
        outcome = apply_rewrite(current, complete[0])
        decisions.append(outcome.decision)
        if outcome.committed:
            current = outcome.graph
            rewrite_count += 1
        else:
            reject_count += 1
            break
    return PassOutcome(
        graph=current,
        decisions=tuple(decisions),
        rewrite_count=rewrite_count,
        reject_count=reject_count,
        no_change_count=0,
    )


@dataclass
class PassSpec:
    name: str
    version: str
    fn: Callable[[IRGraph], PassOutcome]
    kind: str = "rewrite"

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "kind": self.kind}


DEFAULT_PIPELINE: Tuple[PassSpec, ...] = (
    PassSpec(name="fuse_add_rmsnorm", version=PATTERN_VERSION, fn=run_fused_add_rmsnorm_pass),
)


@dataclass
class PassIteration:
    iteration: int
    input_canonical_hash: str
    candidate_count: int
    rewrite_count: int
    reject_count: int
    no_change_count: int
    output_canonical_hash: str
    verifier_ok: bool
    dead_nodes: int
    elapsed_us: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "input_canonical_hash": self.input_canonical_hash,
            "candidate_count": self.candidate_count,
            "rewrite_count": self.rewrite_count,
            "reject_count": self.reject_count,
            "no_change_count": self.no_change_count,
            "output_canonical_hash": self.output_canonical_hash,
            "verifier_ok": self.verifier_ok,
            "dead_nodes": self.dead_nodes,
            "elapsed_us": self.elapsed_us,
        }


@dataclass
class PipelineRun:
    pipeline: Tuple[Mapping[str, str], ...]
    iterations: Tuple[PassIteration, ...]
    final_graph: IRGraph
    converged: bool
    oscillation: bool
    metadata_proliferation: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pipeline": [dict(item) for item in self.pipeline],
            "iterations": [item.as_dict() for item in self.iterations],
            "converged": self.converged,
            "oscillation": self.oscillation,
            "metadata_proliferation": self.metadata_proliferation,
            "final_canonical_hash": self.final_graph.canonical_hash()[0],
        }


def run_pipeline(
    graph: IRGraph,
    *,
    pipeline: Sequence[PassSpec] = DEFAULT_PIPELINE,
    max_iterations: int = 8,
) -> PipelineRun:
    """Iterate the pipeline to a fixed point with oscillation detection."""
    if max_iterations <= 0:
        raise ConfigError("max_iterations must be positive")
    seen: Dict[str, int] = {}
    iterations: List[PassIteration] = []
    current = graph
    converged = False
    oscillation = False
    proliferation = False
    passthrough: List[PassOutcome] = []
    for index in range(1, max_iterations + 1):
        before_hash, _ = current.canonical_hash()
        rewrite_count = 0
        reject_count = 0
        no_change = 0
        candidate_count = 0
        verifier_ok = True
        for spec in pipeline:
            candidate_count += len(find_candidates(current))
            outcome = spec.fn(current)
            passthrough.append(outcome)
            rewrite_count += outcome.rewrite_count
            reject_count += outcome.reject_count
            no_change += outcome.no_change_count
            current = outcome.graph
            report = IRVerifier().verify(current)
            verifier_ok = verifier_ok and report.ok
        after_hash, _ = current.canonical_hash()
        dead_nodes = _dead_node_count(current)
        iterations.append(
            PassIteration(
                iteration=index,
                input_canonical_hash=before_hash,
                candidate_count=candidate_count,
                rewrite_count=rewrite_count,
                reject_count=reject_count,
                no_change_count=no_change,
                output_canonical_hash=after_hash,
                verifier_ok=verifier_ok,
                dead_nodes=dead_nodes,
            )
        )
        if before_hash == after_hash and rewrite_count == 0:
            converged = True
            break
        if after_hash in seen:
            oscillation = True
            break
        seen[after_hash] = index
    if not converged and not oscillation:
        converged = False
    proliferation = _metadata_proliferates(graph, current, passthrough)
    return PipelineRun(
        pipeline=tuple(spec.as_dict() for spec in pipeline),
        iterations=tuple(iterations),
        final_graph=current,
        converged=converged,
        oscillation=oscillation,
        metadata_proliferation=proliferation,
    )


def idempotence_report(
    graph: IRGraph, *, pipeline: Sequence[PassSpec] = DEFAULT_PIPELINE
) -> Dict[str, Any]:
    """``canonical(P(P(G))) == canonical(P(G))`` and second-pass rewrites = 0."""
    first = run_pipeline(graph, pipeline=pipeline)
    second = run_pipeline(first.final_graph, pipeline=pipeline)
    first_hash = first.final_graph.canonical_hash()[0]
    second_hash = second.final_graph.canonical_hash()[0]
    second_rewrites = sum(item.rewrite_count for item in second.iterations)
    return {
        "first_hash": first_hash,
        "second_hash": second_hash,
        "second_pass_rewrites": second_rewrites,
        "hash_stable": first_hash == second_hash,
        "idempotent": first_hash == second_hash and second_rewrites == 0,
        "metadata_proliferation": second.metadata_proliferation,
        "first_run": first.as_dict(),
        "second_run": second.as_dict(),
    }


def determinism_report(
    graph: IRGraph, *, runs: int = 2, pipeline: Sequence[PassSpec] = DEFAULT_PIPELINE
) -> Dict[str, Any]:
    """Same input + same config ⇒ same canonical output (step 31)."""
    hashes: List[str] = []
    rewrites: List[int] = []
    for _ in range(runs):
        outcome = run_pipeline(graph, pipeline=pipeline)
        hashes.append(outcome.final_graph.canonical_hash()[0])
        rewrites.append(sum(item.rewrite_count for item in outcome.iterations))
    return {
        "runs": runs,
        "hashes": hashes,
        "rewrite_counts": rewrites,
        "deterministic": len(set(hashes)) == 1 and len(set(rewrites)) == 1,
        "note": "cross-process determinism additionally requires a cold-process replay",
    }


def _dead_node_count(graph: IRGraph) -> int:
    consumers = _consumers_map(graph)
    return sum(
        1
        for value in graph.values
        if not value.is_output and not consumers.get(value.value_id) and not value.is_input
    )


def _metadata_proliferates(
    before: IRGraph, after: IRGraph, outcomes: Sequence[PassOutcome]
) -> bool:
    """Provenance metadata must not grow on repeated no-op passes."""
    total_rewrites = sum(item.rewrite_count for item in outcomes)
    if total_rewrites == 0 and before.canonical_hash()[0] != after.canonical_hash()[0]:
        return True
    return False


# ── corpus evaluation (steps 15–21) ────────────────────────────────────────


@dataclass
class CorpusRow:
    name: str
    expected: str
    actual: str
    reject_code: str
    decision: Decision

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "expected": self.expected,
            "actual": self.actual,
            "reject_code": self.reject_code,
            "expectation_met": self.expectation_met,
            "decision": self.decision.as_dict(),
        }

    @property
    def expectation_met(self) -> bool:
        if self.expected == MATCH:
            return self.actual == MATCH
        if self.expected == NO_CHANGE:
            return self.actual == NO_CHANGE
        return self.actual == REJECT


def evaluate_corpus(rows: Sequence[CorpusRow]) -> Dict[str, Any]:
    """TP/FP/TN/FN with the hard rule ``false positive == 0``."""
    tp = sum(1 for row in rows if row.expected == MATCH and row.actual == MATCH)
    fn = sum(1 for row in rows if row.expected == MATCH and row.actual != MATCH)
    tn = sum(1 for row in rows if row.expected != MATCH and row.actual == REJECT)
    fp = sum(1 for row in rows if row.expected == REJECT and row.actual == MATCH)
    no_change_mismatch = sum(
        1 for row in rows if row.expected == NO_CHANGE and row.actual != NO_CHANGE
    )
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "rows": len(rows),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "no_change_mismatch": no_change_mismatch,
        "precision": precision,
        "recall": recall,
        "false_positive_zero": fp == 0,
        "rule": (
            "a false positive (near-miss rewritten and justified only by allclose) fails the "
            "experiment; low recall with FP=0 is the safe failure mode"
        ),
        "unmet_expectations": [row.name for row in rows if not row.expectation_met],
    }


def build_corpus_rows(
    plan: Sequence[Mapping[str, Any]], *, graph_builder: Callable[..., Dict[str, Any]]
) -> List[CorpusRow]:
    """Run the frozen pipeline over a pre-registered corpus plan."""
    rows: List[CorpusRow] = []
    for entry in plan:
        spec = graph_builder(graph_id=f"fixture_{entry['class']}", variant=entry["variant"])
        graph = build_graph(spec)
        outcome = run_fused_add_rmsnorm_pass(graph)
        if outcome.decisions:
            # prefer the decision for a *structurally matched* site: incomplete
            # candidates are recorded for diagnostics but do not decide the class
            decision = next(
                (item for item in outcome.decisions if item.structural_match),
                outcome.decisions[0],
            )
        else:
            decision = Decision(
                decision_id=f"none_{entry['class']}",
                site_id="",
                pattern_id=PATTERN_RESIDUAL_RMSNORM,
                pattern_version=PATTERN_VERSION,
                structural_match=False,
                predicates=(),
                before_ir_id=graph.graph_id,
                status=REJECT,
                reject_code="PATTERN_NEAR_MISS",
            )
        if decision.status == MATCH and not decision.after_ir_id:
            # a legal site that produced no committed rewrite is not a MATCH
            decision = Decision(
                **{**decision.__dict__, "status": REJECT, "reject_code": "PATTERN_NEAR_MISS"}
            )
        rows.append(
            CorpusRow(
                name=entry["class"],
                expected=entry["expected"],
                actual=decision.status,
                reject_code=decision.reject_code,
                decision=decision,
            )
        )
    return rows


# ── provenance / atomicity audits (steps 22–23, 30) ────────────────────────


def provenance_audit(before: IRGraph, after: IRGraph, outcome: RewriteOutcome) -> Dict[str, Any]:
    """Source/shape/effect provenance must survive the rewrite."""
    fused_ops = [op for op in after.ops if op.semantic_op == "hqsb::fused_add_rms_norm"]
    problems: List[str] = []
    if not fused_ops:
        problems.append("no fused op after a committed rewrite")
    for op in fused_ops:
        if "source_nodes" not in op.source or "module_path" not in op.source:
            problems.append(f"{op.op_id}: source lineage lost")
        if op.pattern_id != PATTERN_RESIDUAL_RMSNORM:
            problems.append(f"{op.op_id}: pattern provenance lost")
        if op.effects.evidence == "unknown":
            problems.append(f"{op.op_id}: effect evidence downgraded to unknown")
    if not after.fallback:
        problems.append("fallback subgraph reference missing")
    if after.outputs != before.outputs:
        problems.append("output arity changed by the rewrite")
    return {
        "ok": not problems,
        "problems": problems,
        "node_mapping": dict(sorted(outcome.node_mapping.items())),
        "fallback": after.fallback.get("graph_id", ""),
    }


def metadata_diff(before: IRGraph, after: IRGraph) -> Dict[str, Any]:
    """Check that metadata did not proliferate (second pass stability)."""
    return {
        "ops_before": len(before.ops),
        "ops_after": len(after.ops),
        "values_before": len(before.values),
        "values_after": len(after.values),
        "constraints_unchanged": before.constraints == after.constraints,
        "guards_unchanged": before.guards == after.guards,
        "digest_before": before.canonical_hash()[0],
        "digest_after": after.canonical_hash()[0],
    }


def atomicity_report(
    graph: IRGraph, points: Sequence[str] = INJECTION_POINTS
) -> Dict[str, Any]:
    """Inject a failure at every point; the original graph must survive."""
    rows: List[Dict[str, Any]] = []
    candidates = [item for item in find_candidates(graph) if item.complete]
    for point in points:
        if not candidates:
            rows.append(
                {
                    "point": point,
                    "candidate": "",
                    "committed": False,
                    "error": "no legal candidate to inject into",
                    "graph_unchanged": True,
                }
            )
            continue
        candidate = candidates[0]
        before_hash = graph.canonical_hash()[0]
        outcome = apply_rewrite(graph, candidate, inject_at=point)
        rows.append(
            {
                "point": point,
                "candidate": candidate.site_id,
                "committed": outcome.committed,
                "error": outcome.error,
                "graph_unchanged": outcome.graph is graph
                and graph.canonical_hash()[0] == before_hash,
                "fallback_registered": False,
            }
        )
    return {
        "rows": rows,
        "all_atomic": all(row["graph_unchanged"] and not row["committed"] for row in rows),
        "rule": (
            "a failed pass must not leave a half-rewritten graph, a half-registered node or a "
            "polluted cache"
        ),
    }


def pass_identity(pipeline: Sequence[PassSpec] = DEFAULT_PIPELINE) -> Dict[str, Any]:
    """Pass pipeline identity: rules and config enter the compile identity."""
    payload = {
        "passes": [spec.as_dict() for spec in pipeline],
        "predicate_order": list(PREDICATE_ORDER),
    }
    return {"digest": sha256_text(canonical_json(payload)), "payload": payload}


def decision_rows(decisions: Iterable[Decision]) -> List[Dict[str, Any]]:
    return [item.as_dict() for item in decisions]
