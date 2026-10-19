"""Bounded graph adapters. IR parsing records only explicit buffer dependencies.

IR is a versioned compiler diagnostic, not ATen and not device kernel execution.
Unrecognized formats remain readable as text with an explicit parser status.
"""
import re


def ir_graph(text, *, max_nodes=2048, max_edges=12000):
    # Torch 2.8 prints fused children at column zero, just like top-level
    # nodes. The explicit .snodes ownership record, not indentation, decides
    # hierarchy. Counting child bodies as new scheduled units is incorrect.
    child_ids = set(re.findall(r"(?m)^op\d+(?:_op\d+)*\.snodes\[\d+\] =\s*\n(op\d+(?:_op\d+)*):", text))
    headers = [m for m in re.finditer(r"(?m)^(op\d+(?:_op\d+)*): (.*)", text) if m[1] not in child_ids]
    chunks = [text[m.start():headers[i+1].start() if i+1 < len(headers) else len(text)]
              for i, m in enumerate(headers)]
    nodes, producers, reads = [], {}, {}
    for chunk in chunks:
        match = re.match(r"(op\d+(?:_op\d+)*): (.*)", chunk)
        if not match:
            continue
        name, target = match.groups()
        nodes.append({"id": name, "name": name, "target": target[:256], "kind": "scheduler",
                      "detail": chunk[:20000], "module": None, "outputs": []})
        # Only the top-level scheduler node's writes/deps, not fused child nodes.
        head = re.split(r"(?m)^op\d+(?:_op\d+)*\.snodes\[|^\s+op\d+(?:_op\d+)*:", chunk)[0]
        field = lambda key: re.search(rf"(?ms)^{re.escape(name)}\.{key}\s*=[ \t]*(.*?)(?=^{re.escape(name)}\.\w+\s*=|\Z)", head)
        outputs = field("writes")
        deps = field("unmet_dependencies")
        buffers = lambda value: re.findall(r"(?:MemoryDep|StarDep)\((?:name=)?'([^']+)'", value)
        if outputs:
            for buf in buffers(outputs[1]):
                producers[buf] = name
        reads[name] = buffers(deps[1]) if deps else []
    total = len(nodes)
    nodes = nodes[:max_nodes]
    kept = {n["id"] for n in nodes}
    edges = [{"source": producers[b], "target": target, "kind": "data", "buffer": b}
             for target, buffers_read in reads.items() if target in kept
             for b in sorted(set(buffers_read)) if producers.get(b) in kept and producers[b] != target]
    return {"nodes": nodes, "edges": edges[:max_edges], "total_nodes": total,
            "truncated": total > len(nodes) or len(edges) > max_edges,
            "parser_status": "parsed" if nodes else "unrecognized_or_empty",
            "edge_scope": "explicit top-level writes and unmet_dependencies; aliases/control not inferred"}


def onnx_graph(model, *, max_nodes=4096, max_edges=20000):
    """ModelProto -> UI graph. Never serialize initializer/Constant values."""
    graph, nodes, producers = model.graph, [], {}
    metadata = {}
    for value in [*graph.input, *graph.value_info, *graph.output]:
        t = value.type.tensor_type
        metadata[value.name] = {"shape": [d.dim_param or d.dim_value for d in t.shape.dim],
                                "dtype": str(t.elem_type)}
    for i, value in enumerate(graph.input):
        name = f"input{i}"
        nodes.append({"id": name, "name": value.name, "target": value.name,
                      "kind": "placeholder", "outputs": [metadata[value.name]], "module": None})
        producers[value.name] = name
    operations = list(graph.node)
    for i, node in enumerate(operations):
        for output in node.output:
            producers[output] = f"n{i}"
    edges = []
    for i, node in enumerate(operations):
        name = f"n{i}"
        nodes.append({"id": name, "name": node.name, "target": f"{node.domain or 'ai.onnx'}::{node.op_type}",
                      "kind": "onnx", "module": None,
                      "outputs": [metadata[v] for v in node.output if v in metadata],
                      "metadata": {p.key: p.value[:1000] for p in node.metadata_props},
                      "inputs": list(node.input), "output_names": list(node.output)})
        edges.extend({"source": producers[v], "target": name, "kind": "data"}
                     for v in node.input if v in producers)
    for i, value in enumerate(graph.output):
        name = f"output{i}"
        nodes.append({"id":name, "name":value.name, "target":value.name, "kind":"output",
                      "module":None, "outputs":[metadata[value.name]]})
        if value.name in producers:
            edges.append({"source":producers[value.name], "target":name, "kind":"data"})
    total = len(nodes)
    nodes = nodes[:max_nodes]
    kept = {n["id"] for n in nodes}
    edges = [e for e in edges if e["source"] in kept and e["target"] in kept]
    return {"nodes": nodes, "edges": edges[:max_edges], "total_nodes": total,
            "truncated": total > len(nodes) or len(edges) > max_edges,
            "opsets": [{"domain": v.domain, "version": v.version} for v in model.opset_import],
            "module_mapping": "exporter metadata only; no inferred exact module join"}
