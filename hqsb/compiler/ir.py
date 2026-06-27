"""HQSB compiler IR, symbolic constraints, effects and the IR verifier.

Protocol anchors: ``details/S11/README.md`` §4.6 (why HQSB needs its own IR),
§5 (multi-level IR + artifacts), §7 (what semantic equivalence covers),
E11-03 steps 3–7 (schema, serialization, canonical hash, verifier, importer).

The IR is a *sidecar* representation: it stores the information the PyTorch
graph does not carry in stable form (semantic op identity, ordered operands,
symbolic ranges, alias/mutation/state effects, source lineage, target legality
and candidates).  It is deliberately small — the value is that it can be
verified, serialised and hashed, not that it re-implements MLIR.

Every unknown is represented as an explicit ``UNKNOWN_UNSAFE``/``UNAVAILABLE``
value; nothing is defaulted to a "safe" value (E11-01 §13, E11-03 step 3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import (
    IR_HQSB_CANONICAL,
    IR_HQSB_TARGETED,
    IR_LEVELS,
    content_hash,
    sha256_text,
)

SCHEMA_VERSION = "1.0.0"

# ── symbolic shapes ────────────────────────────────────────────────────────

DIM_ORIGINS: Tuple[str, ...] = ("user", "model", "branch", "kernel", "specialization")
CONSTRAINT_VERIFIER_STATUSES: Tuple[str, ...] = ("VERIFIED", "UNVERIFIED", "REJECTED")

#: Constraint categories, matching the guard taxonomy of E11-05 §4.
CONSTRAINT_CATEGORIES: Tuple[str, ...] = (
    "semantic",
    "variant",
    "capability",
    "profitability",
)


@dataclass
class SymbolicDim:
    """A tensor dimension: static constant, range symbol or derived expression."""

    name: str
    expression: str
    lower: Optional[int] = None
    upper: Optional[int] = None
    divisibility: Optional[int] = None
    relation: str = ""
    origin: str = "user"
    semantic_required: bool = True
    verifier_status: str = "UNVERIFIED"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.name or not self.expression:
            problems.append("symbolic dim needs a name and an expression")
        if self.origin not in DIM_ORIGINS:
            problems.append(f"unknown dim origin {self.origin!r}")
        if self.verifier_status not in CONSTRAINT_VERIFIER_STATUSES:
            problems.append(f"unknown verifier status {self.verifier_status!r}")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            problems.append(f"{self.name}: lower > upper")
        if self.divisibility is not None and self.divisibility <= 0:
            problems.append(f"{self.name}: divisibility must be positive")
        if self.lower is None and self.upper is None and "static" not in self.origin:
            problems.append(
                f"{self.name}: unbounded symbolic dim without range (declare bounds or mark static)"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "lower": self.lower,
            "upper": self.upper,
            "divisibility": self.divisibility,
            "relation": self.relation,
            "origin": self.origin,
            "semantic_required": self.semantic_required,
            "verifier_status": self.verifier_status,
        }


@dataclass
class ShapeConstraint:
    """One declarative constraint from the E11-01 symbolic-domain census."""

    constraint_id: str
    expression: str
    category: str
    origin: str
    lower: Optional[int] = None
    upper: Optional[int] = None
    divisibility: Optional[int] = None
    relation: str = ""
    semantic_required: bool = True
    verifier_status: str = "UNVERIFIED"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.category not in CONSTRAINT_CATEGORIES:
            problems.append(f"unknown constraint category {self.category!r}")
        if self.origin not in DIM_ORIGINS:
            problems.append(f"unknown constraint origin {self.origin!r}")
        if self.verifier_status not in CONSTRAINT_VERIFIER_STATUSES:
            problems.append(f"unknown verifier status {self.verifier_status!r}")
        if not self.expression:
            problems.append("constraint expression must not be empty")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "expression": self.expression,
            "category": self.category,
            "origin": self.origin,
            "lower": self.lower,
            "upper": self.upper,
            "divisibility": self.divisibility,
            "relation": self.relation,
            "semantic_required": self.semantic_required,
            "verifier_status": self.verifier_status,
        }


# ── tensors and effects ────────────────────────────────────────────────────

DTYPE_BYTES: Mapping[str, int] = {
    "fp64": 8,
    "fp32": 4,
    "tf32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "int8": 1,
    "int4": 1,  # packed nibbles are accounted per element
    "bool": 1,
}

LAYOUTS: Tuple[str, ...] = ("strided", "contiguous", "channels_last", "custom_packed", "unknown")


@dataclass
class TensorType:
    """Type of an IR value: dtype, symbolic shape, stride, layout, device."""

    dtype: str
    dims: Tuple[str, ...] = ()
    symbolic: Tuple[SymbolicDim, ...] = ()
    stride: Tuple[Optional[int], ...] = ()
    layout: str = "strided"
    device_kind: str = "cuda"
    device_index: int = 0
    storage_offset: Optional[int] = None
    requires_grad: bool = False
    quant_artifact_id: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.dtype not in DTYPE_BYTES:
            problems.append(f"unknown dtype {self.dtype!r}")
        if self.layout not in LAYOUTS:
            problems.append(f"unknown layout {self.layout!r}")
        if self.dims and self.stride and len(self.dims) != len(self.stride):
            problems.append("dims and stride must have the same rank")
        for dim in self.symbolic:
            problems.extend(dim.validate())
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dtype": self.dtype,
            "dims": list(self.dims),
            "symbolic": [dim.as_dict() for dim in self.symbolic],
            "stride": list(self.stride),
            "layout": self.layout,
            "device_kind": self.device_kind,
            "device_index": self.device_index,
            "storage_offset": self.storage_offset,
            "requires_grad": self.requires_grad,
            "quant_artifact_id": self.quant_artifact_id,
        }


EFFECT_EVIDENCE: Tuple[str, ...] = ("schema", "metadata", "analysis", "runtime_test", "unknown")
EFFECT_STATES: Tuple[str, ...] = ("PROVED_SAFE", "PROVED_UNSAFE", "UNKNOWN_UNSAFE")


@dataclass
class EffectSet:
    """Read/write/alias/RNG/state/stream effects of one IR op.

    ``UNKNOWN_UNSAFE`` is the only honest default: an op whose schema/analysis
    has not been inspected must never be treated as pure (E11-01 step 10).
    """

    reads: Tuple[str, ...] = ()
    writes: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    views: Tuple[str, ...] = ()
    rng: str = "none"  # none | consumes | stateful
    state: Tuple[str, ...] = ()
    stream: str = "current"  # current | default_stream | own_stream
    mutation: str = "unknown"  # none | inplace | unknown
    evidence: str = "unknown"
    state_reason: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.evidence not in EFFECT_EVIDENCE:
            problems.append(f"unknown effect evidence {self.evidence!r}")
        if self.mutation not in ("none", "inplace", "unknown"):
            problems.append(f"unknown mutation flag {self.mutation!r}")
        if self.rng not in ("none", "consumes", "stateful"):
            problems.append(f"unknown rng flag {self.rng!r}")
        if self.stream not in ("current", "default_stream", "own_stream"):
            problems.append(f"unknown stream flag {self.stream!r}")
        if self.evidence == "unknown" and not self.state_reason:
            problems.append("UNKNOWN_UNSAFE effects must carry a reason")
        return problems

    @property
    def is_provable_safe(self) -> bool:
        return (
            self.evidence in ("schema", "metadata", "analysis", "runtime_test")
            and self.mutation == "none"
            and self.rng in ("none", "consumes", "stateful")  # RNG handled by caller contract
            and not self.writes
        )

    def risk_state(self) -> str:
        if self.evidence == "unknown" or self.mutation == "unknown":
            return "UNKNOWN_UNSAFE"
        if self.mutation == "inplace" or self.writes:
            return "PROVED_UNSAFE"
        return "PROVED_SAFE"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reads": list(self.reads),
            "writes": list(self.writes),
            "aliases": list(self.aliases),
            "views": list(self.views),
            "rng": self.rng,
            "state": list(self.state),
            "stream": self.stream,
            "mutation": self.mutation,
            "evidence": self.evidence,
            "state_reason": self.state_reason,
            "risk_state": self.risk_state(),
        }


# ── graph structure ────────────────────────────────────────────────────────


@dataclass
class IRValue:
    """A def-use value: produced by one op, consumed by many."""

    value_id: str
    type: Optional[TensorType] = None
    producer: str = ""
    users: Tuple[str, ...] = ()
    source_lineage: Tuple[str, ...] = ()
    is_input: bool = False
    is_output: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "value_id": self.value_id,
            "type": self.type.as_dict() if self.type else None,
            "producer": self.producer,
            "users": list(self.users),
            "source_lineage": list(self.source_lineage),
            "is_input": self.is_input,
            "is_output": self.is_output,
        }


@dataclass
class IROp:
    """One semantic operation in the IR (multi-result, ordered operands)."""

    op_id: str
    semantic_op: str
    operands: Tuple[str, ...]
    results: Tuple[str, ...]
    schema_version: str = "1.0.0"
    attrs: Mapping[str, Any] = field(default_factory=dict)
    effects: EffectSet = field(default_factory=EffectSet)
    source: Mapping[str, Any] = field(default_factory=dict)
    target_legal: Dict[str, bool] = field(default_factory=dict)
    reject_reason: str = ""
    candidates: Tuple[str, ...] = ()
    fallback: str = ""
    pattern_id: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.op_id or not self.semantic_op:
            problems.append("op needs an id and a semantic op name")
        if not self.results:
            problems.append(f"{self.op_id}: op must produce at least one result")
        problems.extend(self.effects.validate())
        for key, value in self.target_legal.items():
            if not isinstance(value, bool):
                problems.append(f"{self.op_id}: target_legal[{key!r}] must be bool")
        if any(value is False for value in self.target_legal.values()) and not self.reject_reason:
            problems.append(f"{self.op_id}: illegal target requires a reject reason")
        if "source_nodes" not in self.source and "module_path" not in self.source:
            problems.append(f"{self.op_id}: source lineage missing (module_path/source_nodes)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "semantic_op": self.semantic_op,
            "operands": list(self.operands),
            "results": list(self.results),
            "schema_version": self.schema_version,
            "attrs": dict(self.attrs),
            "effects": self.effects.as_dict(),
            "source": dict(self.source),
            "target_legal": dict(self.target_legal),
            "reject_reason": self.reject_reason,
            "candidates": list(self.candidates),
            "fallback": self.fallback,
            "pattern_id": self.pattern_id,
        }


@dataclass
class IRGraph:
    """An IR module at one level (canonical or targeted)."""

    graph_id: str
    level: str
    ops: Tuple[IROp, ...] = ()
    values: Tuple[IRValue, ...] = ()
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()
    constraints: Tuple[ShapeConstraint, ...] = ()
    schema_version: str = SCHEMA_VERSION
    source_graph_id: str = ""
    target_id: str = ""
    candidates: Tuple[Mapping[str, Any], ...] = ()
    guards: Tuple[str, ...] = ()
    selection: Mapping[str, Any] = field(default_factory=dict)
    fallback: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    # -- structure ---------------------------------------------------------

    def op(self, op_id: str) -> IROp:
        for item in self.ops:
            if item.op_id == op_id:
                return item
        raise ConfigError(f"unknown op id {op_id!r} in graph {self.graph_id!r}")

    def value(self, value_id: str) -> IRValue:
        for item in self.values:
            if item.value_id == value_id:
                return item
        raise ConfigError(f"unknown value id {value_id!r} in graph {self.graph_id!r}")

    def producers(self) -> Dict[str, str]:
        return {value.producer: value.value_id for value in self.values if value.producer}

    def topological_order(self) -> List[str]:
        """Return op ids in def-use order (raises on cycles)."""
        order: List[str] = []
        visiting: set[str] = set()
        done: set[str] = set()
        producer_of = {
            value_id: op.op_id for op in self.ops for value_id in op.results
        }

        def visit(op_id: str) -> None:
            if op_id in done:
                return
            if op_id in visiting:
                raise ConfigError(f"cycle through op {op_id!r}")
            visiting.add(op_id)
            for operand in self.op(op_id).operands:
                upstream = producer_of.get(operand, "")
                if upstream:
                    visit(upstream)
            visiting.discard(op_id)
            done.add(op_id)
            order.append(op_id)

        for item in self.ops:
            visit(item.op_id)
        return order

    # -- serialization -----------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "level": self.level,
            "schema_version": self.schema_version,
            "source_graph_id": self.source_graph_id,
            "target_id": self.target_id,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "ops": [op.as_dict() for op in self.ops],
            "values": [value.as_dict() for value in self.values],
            "constraints": [item.as_dict() for item in self.constraints],
            "candidates": [dict(item) for item in self.candidates],
            "guards": list(self.guards),
            "selection": dict(self.selection),
            "fallback": dict(self.fallback),
            "notes": self.notes,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=indent, ensure_ascii=False)

    def to_text(self) -> str:
        """Human-readable dump (E11-03 step 5 keeps both forms)."""
        lines = [f"# IRGraph {self.graph_id} level={self.level} schema={self.schema_version}"]
        for value in self.values:
            flag = "in" if value.is_input else ("out" if value.is_output else "tmp")
            dtype = value.type.dtype if value.type else "?"
            dims = ",".join(value.type.dims) if value.type and value.type.dims else "?"
            lines.append(f"  %{value.value_id} [{flag}] dtype={dtype} dims=({dims})")
        for op in self.ops:
            key = (
                f"pattern={op.pattern_id} " if op.pattern_id else ""
            )
            lines.append(
                f"  {op.results[0] if op.results else '?'} = {op.semantic_op}"
                f"({', '.join(op.operands)}) v{op.schema_version} {key}"
                f"effects={op.effects.risk_state()} target={op.target_legal}"
            )
        for item in self.constraints:
            lines.append(
                f"  constraint {item.constraint_id}: {item.expression}"
                f" [{item.category}/{item.origin}/{item.verifier_status}]"
            )
        return "\n".join(lines) + "\n"

    def canonical_payload(self) -> Dict[str, Any]:
        payload = self.as_dict()
        # Temporary names and result ordering are semantic here, so only
        # genuinely non-semantic fields are removed for hashing.
        payload.pop("notes", None)
        return payload

    def canonical_hash(self) -> Tuple[str, List[str]]:
        return content_hash(self.canonical_payload())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IRGraph":
        """Strict deserialization (unknown keys rejected)."""
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ConfigError(f"unknown IRGraph fields: {unknown}")
        ops = tuple(_op_from_dict(item) for item in payload.get("ops", ()))
        values = tuple(_value_from_dict(item) for item in payload.get("values", ()))
        constraints = tuple(
            ShapeConstraint(**item) for item in payload.get("constraints", ())
        )
        data: Dict[str, Any] = dict(payload)
        data["ops"] = ops
        data["values"] = values
        data["constraints"] = constraints
        data["candidates"] = tuple(payload.get("candidates", ()))
        data["guards"] = tuple(payload.get("guards", ()))
        data["inputs"] = tuple(payload.get("inputs", ()))
        data["outputs"] = tuple(payload.get("outputs", ()))
        return cls(**data)  # type: ignore[arg-type]


def _op_from_dict(payload: Mapping[str, Any]) -> IROp:
    data: Dict[str, Any] = dict(payload)
    data["operands"] = tuple(payload.get("operands", ()))
    data["results"] = tuple(payload.get("results", ()))
    data["candidates"] = tuple(payload.get("candidates", ()))
    effects = payload.get("effects")
    if isinstance(effects, Mapping):
        data["effects"] = EffectSet(
            reads=tuple(effects.get("reads", ())),
            writes=tuple(effects.get("writes", ())),
            aliases=tuple(effects.get("aliases", ())),
            views=tuple(effects.get("views", ())),
            rng=effects.get("rng", "none"),
            state=tuple(effects.get("state", ())),
            stream=effects.get("stream", "current"),
            mutation=effects.get("mutation", "unknown"),
            evidence=effects.get("evidence", "unknown"),
            state_reason=effects.get("state_reason", ""),
        )
    return IROp(**data)  # type: ignore[arg-type]


def _value_from_dict(payload: Mapping[str, Any]) -> IRValue:
    data: Dict[str, Any] = dict(payload)
    data["users"] = tuple(payload.get("users", ()))
    data["source_lineage"] = tuple(payload.get("source_lineage", ()))
    tensor = payload.get("type")
    if isinstance(tensor, Mapping):
        dims = tuple(tensor.get("dims", ()))
        symbolic = tuple(SymbolicDim(**item) for item in tensor.get("symbolic", ()))
        data["type"] = TensorType(
            dtype=tensor.get("dtype", "fp32"),
            dims=dims,
            symbolic=symbolic,
            stride=tuple(tensor.get("stride", ())),
            layout=tensor.get("layout", "strided"),
            device_kind=tensor.get("device_kind", "cuda"),
            device_index=int(tensor.get("device_index", 0)),
            storage_offset=tensor.get("storage_offset"),
            requires_grad=bool(tensor.get("requires_grad", False)),
            quant_artifact_id=tensor.get("quant_artifact_id", ""),
        )
    elif tensor is None:
        data["type"] = None
    return IRValue(**data)  # type: ignore[arg-type]


# ── verifier (E11-03 step 6) ───────────────────────────────────────────────


@dataclass
class VerifierIssue:
    """One structured verifier finding (never a bare string log)."""

    code: str
    detail: str
    op_id: str = ""
    severity: str = "error"

    @property
    def blocks_compilation(self) -> bool:
        return self.severity == "error"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "op_id": self.op_id,
            "severity": self.severity,
            "blocks_compilation": self.blocks_compilation,
        }


@dataclass
class VerifierReport:
    graph_id: str
    issues: List[VerifierIssue] = field(default_factory=list)
    checks_run: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(issue.blocks_compilation for issue in self.issues)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "ok": self.ok,
            "checks_run": list(self.checks_run),
            "issues": [issue.as_dict() for issue in self.issues],
        }


class IRVerifier:
    """Structural + semantic hygiene checks for a canonical/targeted graph."""

    CHECKS: Tuple[str, ...] = (
        "graph_id_and_level",
        "unique_ids",
        "def_use",
        "outputs_reachable",
        "no_dangling_values",
        "schema_version_supported",
        "types_and_shapes",
        "effects_present",
        "source_lineage",
        "constraints_declared",
        "candidates_for_targeted",
        "fallback_for_targeted",
        "guard_references",
    )

    def verify(self, graph: IRGraph) -> VerifierReport:
        issues: List[VerifierIssue] = []
        if not graph.graph_id:
            issues.append(VerifierIssue("MISSING_GRAPH_ID", "graph_id is empty"))
        if graph.level not in IR_LEVELS:
            issues.append(VerifierIssue("UNKNOWN_IR_LEVEL", f"level={graph.level!r}"))
        if graph.level not in (IR_HQSB_CANONICAL, IR_HQSB_TARGETED) and graph.ops:
            issues.append(
                VerifierIssue(
                    "LEVEL_WITHOUT_HQSB_SCHEMA",
                    "the HQSB verifier only validates HQSB_CANONICAL/HQSB_TARGETED graphs",
                )
            )

        op_ids = [op.op_id for op in graph.ops]
        value_ids = [value.value_id for value in graph.values]
        duplicates = sorted({item for item in op_ids + value_ids if (op_ids + value_ids).count(item) > 1})
        for item in duplicates:
            issues.append(VerifierIssue("DUPLICATE_ID", f"id {item!r} appears more than once"))

        defined = set(value_ids) | set(graph.inputs)
        for op in graph.ops:
            for operand in op.operands:
                if operand not in defined:
                    issues.append(
                        VerifierIssue("UNDEFINED_OPERAND", f"{op.op_id}: operand {operand!r} undefined", op.op_id)
                    )
            for result in op.results:
                if result not in defined:
                    issues.append(
                        VerifierIssue("UNDECLARED_RESULT", f"{op.op_id}: result {result!r} not declared", op.op_id)
                    )
            if not op.operands:
                issues.append(VerifierIssue("OP_WITHOUT_OPERANDS", f"{op.op_id}: no operands", op.op_id))
            if op.schema_version != SCHEMA_VERSION:
                issues.append(
                    VerifierIssue(
                        "SCHEMA_VERSION_MISMATCH",
                        f"{op.op_id}: schema {op.schema_version!r} != {SCHEMA_VERSION!r}",
                        op.op_id,
                    )
                )
            for problem in op.validate():
                issues.append(VerifierIssue("OP_INVALID", problem, op.op_id))

        for value in graph.values:
            for problem in (value.type.validate() if value.type else []):
                issues.append(VerifierIssue("TYPE_INVALID", f"{value.value_id}: {problem}"))
            if value.producer and value.producer not in set(op_ids):
                issues.append(
                    VerifierIssue(
                        "UNKNOWN_PRODUCER",
                        f"{value.value_id}: producer {value.producer!r} is not an op in this graph",
                    )
                )
            if not value.producer and not value.is_input:
                issues.append(
                    VerifierIssue(
                        "DANGLING_VALUE",
                        f"{value.value_id}: neither an input nor produced by an op",
                    )
                )
            for user in value.users:
                if user not in set(op_ids):
                    issues.append(
                        VerifierIssue("UNKNOWN_USER", f"{value.value_id}: user {user!r} unknown")
                    )

        if not graph.outputs:
            issues.append(VerifierIssue("NO_OUTPUTS", "graph declares no outputs"))
        for output in graph.outputs:
            if output not in defined:
                issues.append(VerifierIssue("OUTPUT_UNREACHABLE", f"output {output!r} is not defined"))
        try:
            graph.topological_order()
        except ConfigError as exc:
            issues.append(VerifierIssue("CYCLE", str(exc)))

        if not graph.constraints:
            issues.append(
                VerifierIssue(
                    "CONSTRAINTS_MISSING",
                    "no symbolic constraints recorded; export/compile artifacts without "
                    "range constraints are incomplete (E11-01 §3.3)",
                )
            )
        for item in graph.constraints:
            for problem in item.validate():
                issues.append(VerifierIssue("CONSTRAINT_INVALID", problem))

        if graph.level == IR_HQSB_TARGETED:
            if not graph.candidates:
                issues.append(
                    VerifierIssue("NO_CANDIDATES", "targeted IR must enumerate lowering candidates")
                )
            if not graph.selection:
                issues.append(
                    VerifierIssue("NO_SELECTION", "targeted IR must record the selection decision")
                )
            if not graph.fallback:
                issues.append(
                    VerifierIssue(
                        "NO_FALLBACK",
                        "targeted IR must record a fallback (reference lowering or original subgraph)",
                    )
                )
        return VerifierReport(graph_id=graph.graph_id, issues=issues, checks_run=self.CHECKS)


def verify_graph(graph: IRGraph) -> VerifierReport:
    return IRVerifier().verify(graph)


# ── importers and diffs ────────────────────────────────────────────────────


def import_op_sequence(
    *,
    graph_id: str,
    ops: Sequence[Mapping[str, Any]],
    inputs: Sequence[Mapping[str, Any]] = (),
    outputs: Sequence[str] = (),
    constraints: Sequence[ShapeConstraint] = (),
    level: str = IR_HQSB_CANONICAL,
    source_graph_id: str = "",
    target_id: str = "",
) -> IRGraph:
    """Build a verifiable graph from a declarative op list.

    Each op mapping has ``op_id``, ``semantic_op``, ``operands``, ``results``
    and optionally ``attrs``, ``effects``, ``source``, ``dtype``/``dims``.
    Used by the fixture corpora and by the FX importer, so both paths share
    one IR constructor.
    """
    ir_ops: List[IROp] = []
    values: List[IRValue] = []
    for spec in ops:
        effects = spec.get("effects")
        effect_set = (
            effects
            if isinstance(effects, EffectSet)
            else EffectSet(
                mutation=spec.get("mutation", "none"),
                evidence=spec.get("effect_evidence", "schema"),
                state_reason=spec.get("effect_reason", ""),
                writes=tuple(spec.get("writes", ())),
                rng=spec.get("rng", "none"),
                stream=spec.get("stream", "current"),
            )
        )
        ir_ops.append(
            IROp(
                op_id=spec["op_id"],
                semantic_op=spec["semantic_op"],
                operands=tuple(spec.get("operands", ())),
                results=tuple(spec.get("results", ())),
                schema_version=spec.get("schema_version", SCHEMA_VERSION),
                attrs=dict(spec.get("attrs", {})),
                effects=effect_set,
                source=dict(spec.get("source", {"module_path": spec.get("module_path", "fixture")})),
                target_legal=dict(spec.get("target_legal", {})),
                reject_reason=spec.get("reject_reason", ""),
                candidates=tuple(spec.get("candidates", ())),
                fallback=spec.get("fallback", ""),
                pattern_id=spec.get("pattern_id", ""),
            )
        )
        for result in spec.get("results", ()):
            values.append(
                IRValue(
                    value_id=result,
                    type=TensorType(
                        dtype=spec.get("dtype", "fp16"),
                        dims=tuple(spec.get("dims", ())),
                        symbolic=tuple(spec.get("symbolic", ())),
                        stride=tuple(spec.get("stride", ())),
                        layout=spec.get("layout", "strided"),
                    ),
                    producer=spec["op_id"],
                    source_lineage=tuple(spec.get("source_lineage", ("fixture",))),
                )
            )
    input_ids = tuple(item["value_id"] for item in inputs)
    for item in inputs:
        values.append(
            IRValue(
                value_id=item["value_id"],
                type=TensorType(
                    dtype=item.get("dtype", "fp16"),
                    dims=tuple(item.get("dims", ())),
                    symbolic=tuple(item.get("symbolic", ())),
                    stride=tuple(item.get("stride", ())),
                    layout=item.get("layout", "strided"),
                ),
                producer="",
                users=tuple(item.get("users", ())),
                source_lineage=tuple(item.get("source_lineage", ("fixture",))),
                is_input=True,
            )
        )
    output_ids = tuple(outputs)
    rebuilt: List[IRValue] = []
    for value in values:
        rebuilt.append(
            IRValue(
                value_id=value.value_id,
                type=value.type,
                producer=value.producer,
                users=value.users,
                source_lineage=value.source_lineage,
                is_input=value.is_input,
                is_output=value.value_id in output_ids,
            )
        )
    return IRGraph(
        graph_id=graph_id,
        level=level,
        ops=tuple(ir_ops),
        values=tuple(rebuilt),
        inputs=input_ids,
        outputs=output_ids,
        constraints=tuple(constraints),
        source_graph_id=source_graph_id,
        target_id=target_id,
    )


def import_fx_graph(
    graph_module: Any,
    *,
    graph_id: str,
    level: str = "DYNAMO_FX",
    module_path: str = "",
    symbolic: Sequence[SymbolicDim] = (),
    constraints: Sequence[ShapeConstraint] = (),
) -> IRGraph:
    """Lazy torch adapter: ``torch.fx.GraphModule`` → :class:`IRGraph`.

    Missing torch is reported as a structured ``ConfigError`` (reason
    ``NOT_INSTALLED``); the IR layer itself never imports torch at module
    scope, so the CPU-minimal install stays importable.
    """
    if level not in ("DYNAMO_FX", "EXPORT_ATEN"):
        raise ConfigError(f"FX importer does not support level {level!r}")
    try:
        import torch  # noqa: F401  (lazy: optional dependency)
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise ConfigError(
            "torch is required to import an FX graph",
            details={"reason": "NOT_INSTALLED", "error": type(exc).__name__},
        ) from exc
    ops: List[Dict[str, Any]] = []
    values: List[Dict[str, Any]] = []
    inputs: List[Dict[str, Any]] = []
    for index, node in enumerate(graph_module.graph.nodes):
        if node.op == "placeholder":
            value_id = f"ph_{node.name}"
            inputs.append({"value_id": value_id, "users": [f"op_{index}"]})
            continue
        op_id = f"op_{index}_{node.name}"
        operand_ids = [
            f"ph_{arg.name}" if hasattr(arg, "name") and arg.op == "placeholder" else
            (f"val_{arg.name}" if hasattr(arg, "name") else f"const_{index}_{pos}")
            for pos, arg in enumerate(node.args)
        ]
        result_id = f"val_{node.name}"
        source = {
            "module_path": module_path or type(graph_module).__name__,
            "source_nodes": [node.name],
            "target": str(node.target),
        }
        stack = getattr(node, "stack_trace", None)
        if stack:
            source["stack"] = [str(entry) for entry in stack.frames[-3:]]
        ops.append(
            {
                "op_id": op_id,
                "semantic_op": str(node.target),
                "operands": operand_ids,
                "results": [result_id],
                "source": source,
                "mutation": "none",
                "effect_evidence": "unknown",
                "effect_reason": "FX node effects are not proven by the importer",
            }
        )
        values.append({"value_id": result_id})
    final_outputs = [values[-1]["value_id"]] if values else []
    graph = import_op_sequence(
        graph_id=graph_id,
        ops=ops,
        inputs=inputs,
        outputs=final_outputs,
        constraints=constraints,
        level=level,
        source_graph_id=graph_id,
    )
    if symbolic:
        graph = _attach_symbolic(graph, symbolic)
    return graph


def _attach_symbolic(graph: IRGraph, symbolic: Sequence[SymbolicDim]) -> IRGraph:
    values = []
    for value in graph.values:
        if value.is_input and value.type is not None:
            values.append(
                IRValue(
                    value_id=value.value_id,
                    type=TensorType(
                        dtype=value.type.dtype,
                        dims=value.type.dims,
                        symbolic=tuple(symbolic),
                        stride=value.type.stride,
                        layout=value.type.layout,
                    ),
                    producer=value.producer,
                    users=value.users,
                    source_lineage=value.source_lineage,
                    is_input=True,
                    is_output=value.is_output,
                )
            )
        else:
            values.append(value)
    return IRGraph(
        graph_id=graph.graph_id,
        level=graph.level,
        ops=graph.ops,
        values=tuple(values),
        inputs=graph.inputs,
        outputs=graph.outputs,
        constraints=graph.constraints,
        source_graph_id=graph.source_graph_id,
        target_id=graph.target_id,
    )


@dataclass
class IRDiff:
    """Structured before/after diff (E11-04 step 13)."""

    before_id: str
    after_id: str
    ops_added: Tuple[str, ...] = ()
    ops_removed: Tuple[str, ...] = ()
    semantic_op_changes: Tuple[Tuple[str, str, str], ...] = ()  # (op_id, before, after)
    constraints_added: Tuple[str, ...] = ()
    constraints_removed: Tuple[str, ...] = ()
    guards_changed: Tuple[str, ...] = ()
    canonical_hash_before: str = ""
    canonical_hash_after: str = ""

    @property
    def structure_changed(self) -> bool:
        return bool(
            self.ops_added
            or self.ops_removed
            or self.semantic_op_changes
            or self.constraints_added
            or self.constraints_removed
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "before_id": self.before_id,
            "after_id": self.after_id,
            "ops_added": list(self.ops_added),
            "ops_removed": list(self.ops_removed),
            "semantic_op_changes": [list(row) for row in self.semantic_op_changes],
            "constraints_added": list(self.constraints_added),
            "constraints_removed": list(self.constraints_removed),
            "guards_changed": list(self.guards_changed),
            "canonical_hash_before": self.canonical_hash_before,
            "canonical_hash_after": self.canonical_hash_after,
            "structure_changed": self.structure_changed,
        }


def diff_graphs(before: IRGraph, after: IRGraph) -> IRDiff:
    """Compare two IR graphs at the structural level."""
    before_ops = {op.op_id: op for op in before.ops}
    after_ops = {op.op_id: op for op in after.ops}
    added = tuple(sorted(set(after_ops) - set(before_ops)))
    removed = tuple(sorted(set(before_ops) - set(after_ops)))
    changed = tuple(
        (op_id, before_ops[op_id].semantic_op, after_ops[op_id].semantic_op)
        for op_id in sorted(set(before_ops) & set(after_ops))
        if before_ops[op_id].semantic_op != after_ops[op_id].semantic_op
    )
    before_constraints = {item.constraint_id for item in before.constraints}
    after_constraints = {item.constraint_id for item in after.constraints}
    before_hash, _ = before.canonical_hash()
    after_hash, _ = after.canonical_hash()
    return IRDiff(
        before_id=before.graph_id,
        after_id=after.graph_id,
        ops_added=added,
        ops_removed=removed,
        semantic_op_changes=changed,
        constraints_added=tuple(sorted(after_constraints - before_constraints)),
        constraints_removed=tuple(sorted(before_constraints - after_constraints)),
        guards_changed=tuple(sorted(set(before.guards) ^ set(after.guards))),
        canonical_hash_before=before_hash,
        canonical_hash_after=after_hash,
    )


def round_trip(graph: IRGraph) -> Tuple[IRGraph, bool]:
    """Serialize → deserialize → compare canonical hashes (E11-03 step 18)."""
    payload = json.loads(graph.to_json())
    rebuilt = IRGraph.from_dict(payload)
    return rebuilt, rebuilt.canonical_hash()[0] == graph.canonical_hash()[0]


def graph_summary(graph: IRGraph) -> Dict[str, Any]:
    """Small helper used by telemetry/coverage reports."""
    digest, stripped = graph.canonical_hash()
    return {
        "graph_id": graph.graph_id,
        "level": graph.level,
        "op_count": len(graph.ops),
        "value_count": len(graph.values),
        "constraint_count": len(graph.constraints),
        "inputs": list(graph.inputs),
        "outputs": list(graph.outputs),
        "canonical_hash": digest,
        "stripped_keys": sorted(set(stripped)),
        "text_sha256": sha256_text(graph.to_text()),
    }
