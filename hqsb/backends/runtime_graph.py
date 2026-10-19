"""Bounded Dynamo FX metadata and analysis; never reads tensor values.

Module paths are the join key shared with architecture_capture.module_inventory.
This is the pre-lowering graph, not a claim about Inductor's physical schedule.
"""
from collections import Counter
import re

from hqsb.core.optimization import fingerprint


def module_path(raw, known):
    path = re.sub(r"^L\[['\"]self['\"]\]\.?", "", str(raw))
    # Dynamo emits nested ModuleList/ModuleDict access as _modules['key'].
    # Canonicalize only that syntax, then require an exact inventory match.
    path = re.sub(r"(?:^|\.)_modules\[['\"]([^'\"]+)['\"]\]", lambda m: "." + m[1], path).lstrip(".")
    path = re.sub(r"\[(\d+)\]", r".\1", path)
    path = path.removeprefix("_orig_mod.")
    return path if path in known else None


def tensor_meta(value):
    # FakeTensor metadata only. Symbolic dimensions remain strings, never int().
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return None
    return {"shape": [x if type(x) is int else str(x)[:128] for x in value.shape],
            "dtype": str(value.dtype), "device": str(getattr(value, "device", "unknown")),
            "stride": [x if type(x) is int else str(x)[:128] for x in value.stride()]}


def target_name(target):
    if isinstance(target, str) or getattr(target, "_schema", None) is not None:
        return str(target)[:256]
    name = getattr(target, "__qualname__", getattr(target, "__name__", None))
    if name:
        return (str(getattr(target, "__module__", "")) + "." + name)[:256]
    return re.sub(r"0x[0-9a-fA-F]+", "<address>", str(target))[:256]


def describe_graph(gm, known, *, graph_id, phase, max_nodes=4096, max_edges=20000):
    all_nodes = list(gm.graph.nodes)
    kept = all_nodes[:max_nodes]
    ids = {node: f"{graph_id}:{node.name}" for node in kept}
    nodes, edges = [], []
    for node in kept:
        stack = node.meta.get("nn_module_stack", {})
        paths = list(dict.fromkeys(p for entry in stack.values()
                                   if isinstance(entry, (tuple, list)) and entry
                                   if (p := module_path(entry[0], known)) is not None))
        value = node.meta.get("example_value", node.meta.get("val"))
        values = value if isinstance(value, (tuple, list)) else [value]
        schema = getattr(node.target, "_schema", None)
        nodes.append({"id": ids[node], "name": node.name, "kind": node.op,
                      "target": target_name(node.target), "module_paths": paths,
                      "module": paths[-1] if paths else None,
                      "source_stack": str(node.meta.get("stack_trace", ""))[-1024:],
                      "outputs": [m for v in values[:16] if (m := tensor_meta(v))],
                      "mutable": bool(schema.is_mutable) if schema is not None else None})
        for source in node.all_input_nodes:
            if source in ids and len(edges) < max_edges:
                edges.append({"source": ids[source], "target": ids[node], "kind": "data"})
    reason = getattr(gm, "compile_subgraph_reason", None)
    breaks = []
    if reason is not None and getattr(reason, "graph_break", False):
        frames = getattr(reason, "user_stack", [])
        breaks.append({"reason": str(getattr(reason, "reason", ""))[:2000],
                       "source": [{"file": f.filename, "line": f.lineno, "function": f.name}
                                  for f in frames[-5:]]})
    return {"id": graph_id, "phase": phase, "nodes": nodes, "edges": edges,
            "graph_breaks": breaks, "total_nodes": len(all_nodes),
            "truncated": len(all_nodes) > len(kept) or sum(len(n.all_input_nodes) for n in kept) > len(edges),
            "graph_sha256": fingerprint({"nodes": [{k: v for k, v in n.items() if k != "source_stack"} for n in nodes], "edges": edges})}


def analyze_graphs(graphs):
    nodes = [n for g in graphs for n in g["nodes"] if n["kind"].startswith("call_")]
    ops = Counter(n["target"] for n in nodes)
    modules = Counter(n["module"] for n in nodes if n["module"] is not None)
    return {"recorded_call_nodes": len(nodes), "mapped_call_nodes": sum(modules.values()),
            "module_counts": dict(modules), "operator_counts": dict(ops),
            "mutable_schema_nodes": sum(n["mutable"] is True for n in nodes),
            "unknown_effect_nodes": sum(n["mutable"] is None for n in nodes),
            "graph_breaks": [b for g in graphs for b in g["graph_breaks"]],
            "interpretation": "counts are over recorded specializations, not executed kernel counts",
            "recommendations": (["Inspect graph-break source boundaries before forcing fullgraph."]
                                if any(g["graph_breaks"] for g in graphs) else []) +
                               ["Compare new graph counts across identical and changed shapes; use dynamic policy if specialization churn is observed.",
                                "Use an independent profiler run to verify physical fusion and launch reduction."]}
