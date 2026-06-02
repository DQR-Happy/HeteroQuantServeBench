"""Capture-mode-aware graph IR, structural hash and graph diff (E06-02/04/07).

The same RMSNorm can appear as one node at one IR level and as
``pow``/``mean``/``rsqrt``/``mul`` at another (E06-04 §3).  A pattern signature
that does not bind the IR level, capture mode, decomposition set and version is
therefore meaningless.  This module provides:

* :class:`GraphNode` / :class:`Graph` — a small, torch-free IR;
* :class:`CaptureMode` / :class:`IRLevel` — the labels every graph record must
  carry (the protocol forbids calling everything "an FX graph");
* structural hashing and normalisation (node *names* never enter the hash);
* :meth:`Graph.diff` — removed/added nodes, dead code before/after, guard
  changes and parameter-reference changes;
* adapters that convert a real ``torch.fx`` graph / exported program into this
  IR **lazily**, so the module imports on a CPU-minimal install.

Rewriting always produces a new :class:`Graph`; the caller's graph object is
never mutated in place (E06-04 §11 step 9: the original graph must not be
irreversibly polluted).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

from hqsb.core.errors import ConfigError


class CaptureMode:
    """How the graph was obtained (E06-02 §9 capture matrix)."""

    DIRECT_META = "direct_meta"
    FAKE_TENSOR_MODE = "fake_tensor_mode"
    OPCHECK = "opcheck"
    FX_SHAPE_PROP = "fx_shape_prop"
    DYNAMO = "dynamo"
    EXPORT_STRICT = "export_strict"
    EXPORT_NON_STRICT = "export_non_strict"
    INDUCTOR = "inductor"

    ALL = (
        DIRECT_META,
        FAKE_TENSOR_MODE,
        OPCHECK,
        FX_SHAPE_PROP,
        DYNAMO,
        EXPORT_STRICT,
        EXPORT_NON_STRICT,
        INDUCTOR,
    )


class IRLevel:
    """IR level a graph lives at (E06-04 §3)."""

    MODULE_FX = "module_fx"
    DYNAMO_FX = "dynamo_fx"
    EXPORT_ATEN = "export_aten"
    PRE_DECOMPOSITION = "pre_decomposition"
    POST_DECOMPOSITION = "post_decomposition"
    POST_FUNCTIONALIZATION = "post_functionalization"
    INDUCTOR_PRE_FUSION = "inductor_pre_fusion"
    INDUCTOR_POST_FUSION = "inductor_post_fusion"

    ALL = (
        MODULE_FX,
        DYNAMO_FX,
        EXPORT_ATEN,
        PRE_DECOMPOSITION,
        POST_DECOMPOSITION,
        POST_FUNCTIONALIZATION,
        INDUCTOR_PRE_FUSION,
        INDUCTOR_POST_FUSION,
    )


# ── nodes and edges ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Ref:
    """A reference to another node's output."""

    node: str
    index: int = 0

    def as_json(self) -> Dict[str, Any]:
        return {"ref": self.node, "index": self.index}


Value = Union[Ref, int, float, bool, str, None, Tuple[Any, ...]]


def _value_key(value: Any) -> Any:
    if isinstance(value, Ref):
        return ("ref", value.index)
    if isinstance(value, tuple):
        return ("tuple", tuple(_value_key(item) for item in value))
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return repr(value)


@dataclass(frozen=True)
class GraphNode:
    """One operation in the graph.

    ``is_impure`` marks nodes with side effects (``aten._print``, in-place
    writes, RNG).  ``alias_inputs`` records which positional inputs the op
    writes; both feed the pattern predicates (E06-04 §4.1).
    """

    name: str
    op: str
    args: Tuple[Value, ...] = ()
    kwargs: Tuple[Tuple[str, Value], ...] = ()
    meta: Tuple[Tuple[str, Any], ...] = ()
    module_path: str = ""
    source_stack: str = ""
    is_impure: bool = False
    alias_inputs: Tuple[int, ...] = ()

    @property
    def refs(self) -> Tuple[Ref, ...]:
        return tuple(arg for arg in self.args if isinstance(arg, Ref))

    def kwargs_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.kwargs}

    def meta_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.meta}

    def with_meta(self, **entries: Any) -> "GraphNode":
        merged = dict(self.meta)
        merged.update(entries)
        return replace(self, meta=tuple(sorted(merged.items())))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "op": self.op,
            "args": [arg.as_json() if isinstance(arg, Ref) else _value_key(arg) for arg in self.args],
            "kwargs": {key: (value.as_json() if isinstance(value, Ref) else _value_key(value)) for key, value in self.kwargs},
            "module_path": self.module_path,
            "is_impure": self.is_impure,
            "alias_inputs": list(self.alias_inputs),
        }


@dataclass
class Graph:
    """A capture-mode-labelled graph with structural identity."""

    nodes: Tuple[GraphNode, ...] = ()
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[Ref, ...] = ()
    capture_mode: str = CaptureMode.DYNAMO
    ir_level: str = IRLevel.DYNAMO_FX
    decomposition_set: str = "none"
    module_path: str = ""
    model_id: str = ""
    version: str = ""

    def __post_init__(self) -> None:
        if self.capture_mode not in CaptureMode.ALL:
            raise ConfigError(
                f"unknown capture mode {self.capture_mode!r}; a graph must record how "
                f"it was captured (allowed={list(CaptureMode.ALL)})",
                details={"field": "capture_mode"},
            )
        if self.ir_level not in IRLevel.ALL:
            raise ConfigError(
                f"unknown IR level {self.ir_level!r} (allowed={list(IRLevel.ALL)})",
                details={"field": "ir_level"},
            )

    # ── lookup ────────────────────────────────────────────────────────

    def node(self, name: str) -> GraphNode:
        for node in self.nodes:
            if node.name == name:
                return node
        raise ConfigError(
            f"graph has no node {name!r}",
            details={"field": "node", "available": [node.name for node in self.nodes]},
        )

    def has(self, name: str) -> bool:
        return any(node.name == name for node in self.nodes)

    def user_count(self, name: str) -> int:
        count = 0
        for node in self.nodes:
            count += sum(1 for ref in node.refs if ref.node == name)
        count += sum(1 for ref in self.outputs if ref.node == name)
        return count

    def users_of(self, name: str) -> Tuple[GraphNode, ...]:
        return tuple(node for node in self.nodes if any(ref.node == name for ref in node.refs))

    def op_multiset(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for node in self.nodes:
            counts[node.op] = counts.get(node.op, 0) + 1
        return counts

    def impure_nodes(self) -> Tuple[GraphNode, ...]:
        return tuple(node for node in self.nodes if node.is_impure)

    # ── structure / identity ──────────────────────────────────────────

    def canonical_form(self) -> Tuple[Tuple[Any, ...], ...]:
        """Canonical shape of the graph: node names replaced by topo indices."""
        index = {node.name: position for position, node in enumerate(self.nodes)}
        form: List[Tuple[Any, ...]] = []
        for node in self.nodes:
            args = tuple(
                ("in", index[arg.node], arg.index) if isinstance(arg, Ref) else _value_key(arg)
                for arg in node.args
            )
            kwargs = tuple(sorted((key, _value_key(value)) for key, value in node.kwargs))
            form.append((node.op, args, kwargs, node.is_impure, node.alias_inputs))
        outputs = tuple((index[ref.node], ref.index) for ref in self.outputs)
        return (tuple(form), outputs)

    def structural_hash(self) -> str:
        payload = json.dumps(self.canonical_form(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def normalized_text(self) -> str:
        index = {node.name: position for position, node in enumerate(self.nodes)}
        lines = []
        for position, node in enumerate(self.nodes):
            args = ", ".join(
                f"%{index[arg.node]}" if isinstance(arg, Ref) else repr(_value_key(arg))
                for arg in node.args
            )
            flags = " [impure]" if node.is_impure else ""
            lines.append(f"%{position} = {node.op}({args}){flags}")
        outs = ", ".join(f"%{index[ref.node]}" for ref in self.outputs)
        return "\n".join(lines) + (f"\nreturn ({outs})" if outs else "")

    def topology_signature(self) -> Tuple[str, ...]:
        """Op names in topological order (used for coarse drift detection)."""
        return tuple(node.op for node in self.nodes)

    # ── comparison ────────────────────────────────────────────────────

    def diff(self, other: "Graph") -> "GraphDiff":
        before_ops = _op_records(self)
        after_ops = _op_records(other)
        removed = _multiset_difference(before_ops, after_ops)
        added = _multiset_difference(after_ops, before_ops)
        dead_before = _dead_nodes(self)
        dead_after = _dead_nodes(other)
        before_meta = {
            node.name: node.meta_dict() for node in self.nodes if node.meta
        }
        after_meta = {
            node.name: node.meta_dict() for node in other.nodes if node.meta
        }
        return GraphDiff(
            before_hash=self.structural_hash(),
            after_hash=other.structural_hash(),
            removed=removed,
            added=added,
            meta_changed={
                name: {"before": before_meta[name], "after": after_meta.get(name, {})}
                for name in sorted(set(before_meta) & set(after_meta))
                if before_meta[name] != after_meta[name]
            },
            dead_code_before=dead_before,
            dead_code_after=dead_after,
            impure_before=tuple(node.name for node in self.impure_nodes()),
            impure_after=tuple(node.name for node in other.impure_nodes()),
            ir_level_before=self.ir_level,
            ir_level_after=other.ir_level,
        )

    # ── copying / editing (never in place) ────────────────────────────

    def copy(self) -> "Graph":
        return replace(self, nodes=tuple(self.nodes))

    def with_node(self, node: GraphNode) -> "Graph":
        return replace(self, nodes=(*self.nodes, node))

    def without_node(self, name: str) -> "Graph":
        return replace(
            self,
            nodes=tuple(node for node in self.nodes if node.name != name),
            outputs=tuple(ref for ref in self.outputs if ref.node != name),
        )

    def replaced(self, name: str, node: GraphNode) -> "Graph":
        return replace(
            self,
            nodes=tuple(node if existing.name == name else existing for existing in self.nodes),
        )

    def dead_nodes(self) -> Tuple[str, ...]:
        return _dead_nodes(self)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "capture_mode": self.capture_mode,
            "ir_level": self.ir_level,
            "decomposition_set": self.decomposition_set,
            "module_path": self.module_path,
            "model_id": self.model_id,
            "structural_hash": self.structural_hash(),
            "node_count": len(self.nodes),
            "op_counts": self.op_multiset(),
            "nodes": [node.as_dict() for node in self.nodes],
            "outputs": [ref.as_json() for ref in self.outputs],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=2, ensure_ascii=False)


def _op_records(graph: Graph) -> List[Tuple[str, str]]:
    return [
        (node.op, node.name if not node.module_path else node.module_path)
        for node in graph.nodes
    ]


def _multiset_difference(left: Sequence[Any], right: Sequence[Any]) -> Tuple[Any, ...]:
    remaining = list(right)
    result: List[Any] = []
    for item in left:
        if item in remaining:
            remaining.remove(item)
        else:
            result.append(item)
    return tuple(result)


def _dead_nodes(graph: Graph) -> Tuple[str, ...]:
    keep = {ref.node for ref in graph.outputs}
    changed = True
    while changed:
        changed = False
        for node in graph.nodes:
            if node.name in keep:
                for ref in node.refs:
                    if ref.node not in keep:
                        keep.add(ref.node)
                        changed = True
    return tuple(node.name for node in graph.nodes if node.name not in keep)


# ── graph diff ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GraphDiff:
    """Before/after graph facts a rewrite must persist (E06-04 §9)."""

    before_hash: str
    after_hash: str
    removed: Tuple[Any, ...] = ()
    added: Tuple[Any, ...] = ()
    meta_changed: Mapping[str, Any] = field(default_factory=dict)
    dead_code_before: Tuple[str, ...] = ()
    dead_code_after: Tuple[str, ...] = ()
    impure_before: Tuple[str, ...] = ()
    impure_after: Tuple[str, ...] = ()
    ir_level_before: str = ""
    ir_level_after: str = ""

    @property
    def node_delta(self) -> int:
        return len(self.added) - len(self.removed)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "removed": list(self.removed),
            "added": list(self.added),
            "node_delta": self.node_delta,
            "meta_changed": {k: dict(v) if isinstance(v, Mapping) else v for k, v in self.meta_changed.items()},
            "dead_code_before": list(self.dead_code_before),
            "dead_code_after": list(self.dead_code_after),
            "impure_before": list(self.impure_before),
            "impure_after": list(self.impure_after),
            "ir_level_before": self.ir_level_before,
            "ir_level_after": self.ir_level_after,
        }


# ── torch adapters (lazy) ─────────────────────────────────────────────────


def from_fx_graph_module(
    graph_module: Any,
    *,
    capture_mode: str = CaptureMode.DYNAMO,
    ir_level: str = IRLevel.DYNAMO_FX,
    decomposition_set: str = "none",
    module_path: str = "",
    model_id: str = "",
) -> Graph:
    """Convert a ``torch.fx.GraphModule`` into the HQSB IR.

    Imports nothing at module scope; the caller must already have torch loaded
    (the runner owns that decision).
    """
    try:  # pragma: no cover - exercised only with torch installed
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - CPU-minimal
        raise ConfigError(
            "converting an FX graph module requires torch; install the "
            "'benchmark' extra",
            details={"field": "torch", "reason": "NOT_INSTALLED", "error": str(exc)},
        ) from exc

    graph = graph_module.graph
    name_map: Dict[Any, str] = {}
    nodes: List[GraphNode] = []
    for position, node in enumerate(graph.nodes):
        hqsb_name = f"n{position}_{node.name}"
        name_map[node] = hqsb_name

    def convert(value: Any) -> Value:
        if value in name_map:
            return Ref(node=name_map[value])
        if isinstance(value, (list, tuple)):
            return tuple(convert(item) for item in value)
        if isinstance(value, (int, float, bool, str)) or value is None:
            return value
        return repr(value)

    for position, node in enumerate(graph.nodes):
        op = str(node.target) if not isinstance(node.target, str) else node.target
        is_impure = op.endswith("_") or op in ("aten._print", "aten._assert_async") or (
            isinstance(node.target, str) and node.target.endswith("_")
        )
        alias_inputs = tuple(
            index
            for index, arg in enumerate(node.args)
            if isinstance(arg, str) and arg.endswith("_")
        )
        nodes.append(
            GraphNode(
                name=name_map[node],
                op=op,
                args=tuple(convert(arg) for arg in node.args),
                kwargs=tuple(sorted((key, convert(value)) for key, value in node.kwargs.items())),
                meta=tuple(sorted((key, repr(value)) for key, value in (node.meta or {}).items())),
                module_path=module_path,
                source_stack="",
                is_impure=bool(is_impure),
                alias_inputs=alias_inputs,
            )
        )
    outputs = tuple(
        Ref(node=name_map[node]) for node in graph.nodes if node.op == "output" and node in name_map
    )
    return Graph(
        nodes=tuple(nodes),
        outputs=outputs,
        capture_mode=capture_mode,
        ir_level=ir_level,
        decomposition_set=decomposition_set,
        module_path=module_path,
        model_id=model_id,
    )


def from_node_sequence(
    nodes: Sequence[Mapping[str, Any]],
    *,
    capture_mode: str = CaptureMode.FX_SHAPE_PROP,
    ir_level: str = IRLevel.MODULE_FX,
    decomposition_set: str = "none",
    module_path: str = "",
    model_id: str = "",
) -> Graph:
    """Build a graph from a declarative node list (fixtures and negative corpora).

    Each entry is ``{"name": str, "op": str, "args": [str|value], "kwargs": {},
    "is_impure": bool, "alias_inputs": [int]}``.  A string argument that matches
    another node's name is a reference; ``"$name"`` is an *explicit* reference
    and is refused when the node does not exist (so a typo in a fixture cannot
    silently become a constant).
    """
    names = {entry["name"] for entry in nodes}
    built: List[GraphNode] = []
    for entry in nodes:
        args: List[Value] = []
        for arg in entry.get("args", ()):
            if isinstance(arg, str) and arg.startswith("$"):
                target = arg[1:]
                if target not in names:
                    raise ConfigError(
                        f"node {entry['name']!r} references unknown node {target!r}",
                        details={"field": "args"},
                    )
                args.append(Ref(node=target))
                continue
            if isinstance(arg, str) and arg in names:
                args.append(Ref(node=arg))
                continue
            args.append(arg)
        built.append(
            GraphNode(
                name=str(entry["name"]),
                op=str(entry["op"]),
                args=tuple(args),
                kwargs=tuple(sorted((entry.get("kwargs") or {}).items())),
                meta=tuple(sorted((entry.get("meta") or {}).items())),
                module_path=str(entry.get("module_path", module_path)),
                source_stack=str(entry.get("source_stack", "")),
                is_impure=bool(entry.get("is_impure", False)),
                alias_inputs=tuple(entry.get("alias_inputs", ())),
            )
        )
    return Graph(
        nodes=tuple(built),
        outputs=tuple(Ref(node=str(entry["name"])) for entry in nodes if entry.get("output")),
        capture_mode=capture_mode,
        ir_level=ir_level,
        decomposition_set=decomposition_set,
        module_path=module_path,
        model_id=model_id,
    )


def graph_summary(graphs: Iterable[Graph]) -> Dict[str, Any]:
    """Aggregate op counts and hashes across a graph corpus (coverage input)."""
    per_graph = []
    op_totals: Dict[str, int] = {}
    for graph in graphs:
        counts = graph.op_multiset()
        for op, count in counts.items():
            op_totals[op] = op_totals.get(op, 0) + count
        per_graph.append(
            {
                "capture_mode": graph.capture_mode,
                "ir_level": graph.ir_level,
                "structural_hash": graph.structural_hash(),
                "node_count": len(graph.nodes),
            }
        )
    return {"graphs": len(per_graph), "op_totals": op_totals, "per_graph": per_graph}


__all__ = [
    "CaptureMode",
    "Graph",
    "GraphDiff",
    "GraphNode",
    "IRLevel",
    "Ref",
    "Value",
    "from_fx_graph_module",
    "from_node_sequence",
    "graph_summary",
]
