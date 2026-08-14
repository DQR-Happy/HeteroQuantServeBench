"""Only tool-provided IDs create op/kernel links; time proximity proves nothing."""


def trace_links(events, *, limit=200):
    rows = list(events)
    operators, launches = {}, {}
    for row in rows:
        if row["category"].lower() not in {"cpu_op", "cuda_runtime", "cuda_driver"}:
            continue
        args = row.get("args", {})
        identity = args.get("External id", args.get("External Id"))
        if identity is not None:
            owners = operators if row["category"].lower() == "cpu_op" else launches
            owners.setdefault(str(identity), []).append(row["id"])
    edges, ambiguous, unmatched, total = [], 0, 0, 0
    for row in rows:
        if row["category"].lower() not in {"kernel", "gpu_kernel"}:
            continue
        args = row.get("args", {})
        identity = args.get("External id", args.get("External Id"))
        # Native launch rows repeat the owning op ID; they are not rival ops.
        candidates = (
            (operators.get(str(identity)) or launches.get(str(identity), []))
            if identity is not None
            else []
        )
        if len(candidates) != 1:
            ambiguous += len(candidates) > 1
            unmatched += not candidates
            continue
        total += 1
        if len(edges) < limit:
            edges.append(
                {
                    "from_event": candidates[0],
                    "to_event": row["id"],
                    "method": "tool_external_id",
                    "confidence": "tool_correlated",
                }
            )
    return {
        "edges": edges,
        "ambiguous": ambiguous,
        "unmatched": unmatched,
        "total_correlated": total,
        "truncated": total > len(edges),
        "scope": "single_capture_only",
        "limit": limit,
        "limitations": ["No temporal-only links; no cross-run causal chain."],
    }
