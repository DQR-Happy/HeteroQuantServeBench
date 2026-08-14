"""Pure evidence projections; no device imports, inferred physical addresses or verdicts.

This parser lives with evidence, not the core-only evaluation policy layer.
"""

from hqsb.observation.contracts import metric
from pathlib import Path

from hqsb.observation.storage import EvidenceReader, sha256
from hqsb.observation.allocator import lifetimes, snapshot_attribution


def analyze(directory):
    reader = EvidenceReader(directory)
    manifest = reader.manifest()
    metrics, boundaries, kv, layouts, os_rows, pages, allocator, problems = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    attribution = []
    lifetime_report = None
    kinds = {}
    for row in reader.records():
        kind, body = row["kind"], row["payload"]
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind == "boundary":
            boundaries.append(body)
        elif kind == "framework_profile" and body.get("trace_sha256"):
            path = reader.directory / "trace.json"
            if (
                path.is_symlink()
                or path.stat().st_size > 128 * 1024**2
                or sha256(path) != body["trace_sha256"]
            ):
                raise ValueError("FRAMEWORK_TRACE_HASH_MISMATCH")
        elif kind == "tensor_inventory":
            layouts.append(body)
        elif kind == "kv_inventory":
            inv = body["inventory"]
            kv.append(
                {
                    "label": body["label"],
                    "step": body.get("step"),
                    "logical_bytes": inv["logical_total_bytes"],
                    "unique_storage_bytes": inv["unique_storage_bytes"],
                    "availability": inv["availability"],
                    "entries": inv["entries"],
                }
            )
        elif kind == "os_sample":
            os_rows.append({**body, "evidence_seq": row["seq"]})
        elif kind == "page_sample":
            sample = body["pages"]
            pages.append(
                {
                    "label": body["label"],
                    "sampled_pages": len(sample),
                    "present_pages": sum(p["present"] for p in sample),
                    "swapped_pages": sum(p["swapped"] for p in sample),
                    "scope": body["scope"],
                    "selection": body["selection"],
                }
            )
        elif kind == "memory_sample":
            for name in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes"):
                metrics.append(
                    {
                        "label": body["label"],
                        "source_record_seqs": [row["seq"]],
                        **metric(
                            name,
                            body.get(name),
                            "byte",
                            source="torch_cuda_allocator",
                            scope="worker_allocator",
                        ),
                    }
                )
        elif kind == "source_status":
            problems.append(body)
        elif kind == "allocator_snapshot":
            snapshot = body["snapshot"]
            segments = snapshot.get("segments", [])
            blocks = [
                block for segment in segments for block in segment.get("blocks", [])
            ]
            counts = {}
            traces = snapshot.get("device_traces", [])
            for trace in traces:
                for event in trace:
                    counts[event["action"]] = counts.get(event["action"], 0) + 1
            allocator.append(
                {
                    "label": body["label"],
                    "time_source": body["time_source"],
                    "segments": len(segments),
                    "blocks": len(blocks),
                    "event_counts": counts,
                    "active_bytes": sum(
                        b["size"] for b in blocks if b["state"] == "active_allocated"
                    ),
                    "awaiting_free_bytes": sum(
                        b["size"]
                        for b in blocks
                        if b["state"] == "active_awaiting_free"
                    ),
                    "inactive_bytes": sum(
                        b["size"] for b in blocks if b["state"] == "inactive"
                    ),
                    "history_prefix_may_be_overwritten": body[
                        "history_prefix_may_be_overwritten"
                    ],
                    "scope": body["scope"],
                }
            )
            if body["label"] == "before_cleanup":
                lifetime_report = lifetimes(
                    snapshot,
                    prefix_may_be_overwritten=body["history_prefix_may_be_overwritten"],
                )
                lifetime_report["anchor"] = body["label"]
                tensors = [
                    entry for inv in layouts for entry in inv["inventory"]["entries"]
                ]
                tensors += [
                    entry
                    for inv in kv
                    if inv["label"] == "after_generation"
                    for entry in inv["entries"]
                ]
                attribution = snapshot_attribution(tensors, snapshot, lifetime_report)
            elif body["label"] == "after_cleanup":
                lifetime_report = lifetimes(
                    snapshot,
                    prefix_may_be_overwritten=body["history_prefix_may_be_overwritten"],
                )
                lifetime_report["anchor"] = body["label"]
    if os_rows:
        first = next(
            (row for row in os_rows if row.get("label") == "before_request"), os_rows[0]
        )
        last = next(
            (row for row in reversed(os_rows) if row.get("label") == "after_cleanup"),
            os_rows[-1],
        )
        for field in ("minor", "major"):
            delta = last["faults"][field] - first["faults"][field]
            metrics.append(
                {
                    "source_record_seqs": [first["evidence_seq"], last["evidence_seq"]],
                    **metric(
                        f"cpu_{field}_fault_delta",
                        delta if delta >= 0 else None,
                        "fault",
                        source="proc_stat",
                        scope="worker_cpu_process",
                        kind="derived",
                        reason="COUNTER_RESET",
                    ),
                }
            )
        for field in (
            "pgfault",
            "pgmajfault",
            "pswpin",
            "pswpout",
            "pgscan_kswapd",
            "pgsteal_kswapd",
        ):
            before = first["system_vmstat"]["values"].get(field)
            after = last["system_vmstat"]["values"].get(field)
            delta = (
                after - before
                if before is not None and after is not None and after >= before
                else None
            )
            metrics.append(
                {
                    "source_record_seqs": [first["evidence_seq"], last["evidence_seq"]],
                    **metric(
                        f"system_{field}_delta",
                        delta,
                        "event" if field in {"pgfault", "pgmajfault"} else "page",
                        source="proc_vmstat",
                        scope="system_not_attributable_to_model",
                        kind="derived",
                    ),
                }
            )
        for label, row in (("before", first), ("after", last)):
            for group, name, scope in (
                ("status", "VmRSS", "worker_process"),
                ("status", "VmSwap", "worker_process"),
                ("smaps_rollup", "Pss", "worker_process"),
                ("system_memory", "MemAvailable", "system"),
                ("system_memory", "Cached", "system"),
            ):
                metrics.append(
                    {
                        "label": label,
                        "source_record_seqs": [row["evidence_seq"]],
                        **metric(
                            name,
                            row[group]["values"].get(name),
                            "byte",
                            source=f"procfs:{group}",
                            scope=scope,
                        ),
                    }
                )
    return {
        "schema_version": "1.0.0",
        "provenance": {
            "manifest_sha256": sha256(reader.raw / "manifest.json"),
            "analysis_version": "1.0.0",
            "analysis_module_sha256": sha256(Path(__file__)),
        },
        "context": manifest["context"],
        "execution_status": manifest["execution_status"],
        "cleanup_status": manifest["cleanup_status"],
        "supervisor_failure": manifest.get("supervisor_failure"),
        "capture_status": manifest["capture_status"],
        "publication": manifest["publication"],
        "record_counts": kinds,
        "metrics": metrics,
        "boundaries": boundaries,
        "kv": kv,
        "tensor_inventories": layouts,
        "allocator": allocator,
        "allocation_attribution": attribution,
        "allocation_lifetimes": lifetime_report,
        "page_samples": pages,
        "source_problems": problems,
        "os_source_availability": {
            name: last[name]["availability"]
            for name in (
                "status",
                "smaps_rollup",
                "system_memory",
                "system_vmstat",
                "psi",
            )
        }
        if os_rows
        else {},
        "limitations": [
            "Process OS counters include the observer threads and metadata allocations.",
            "OS and allocator scopes overlap and must not be added.",
            "Snapshot allocation traces use sequence, not calibrated per-event timestamps.",
            "No complete activation/workspace or physical GPU page attribution.",
            "Hardware cache counters require a separate Nsight capture.",
        ],
    }


def compare(left, right, *, intervention="attention_requested"):
    """Report mechanism differences; require explicit quality/performance contracts for adoption."""
    reasons = []
    if intervention not in {"attention_requested", "cleanup_policy"}:
        raise ValueError("COMPARISON_INTERVENTION_UNSUPPORTED")
    for item in (left, right):
        identity = item["context"].get("model_identity")
        if (
            isinstance(identity, dict)
            and identity.get("scope") != "verified_manifest_at_load"
        ):
            reasons.append("WEIGHT_IDENTITY_UNVERIFIED")
            break
    if left["context"].get("output_token_hash") != right["context"].get(
        "output_token_hash"
    ):
        reasons.append("GENERATED_CONTEXT_DIVERGED")
    for name in ("packages", "kernel", "machine", "allocator_environment"):
        if left["context"].get("source", {}).get(name) != right["context"].get(
            "source", {}
        ).get(name):
            reasons.append(f"ENVIRONMENT_MISMATCH:{name}")
    for field in (
        "source",
        "model_identity",
        "dtype",
        "cleanup_policy",
        "attention_requested",
    ):
        if field == intervention:
            continue
        if left["context"].get(field) != right["context"].get(field):
            # Process and source file paths are deliberately excluded from source equality below.
            if field == "source" and left["context"].get(field, {}).get(
                "source_digest"
            ) == right["context"].get(field, {}).get("source_digest"):
                continue
            reasons.append(f"CONTROL_MISMATCH:{field}")
    if left["context"].get("plan") != right["context"].get("plan"):
        reasons.append("OBSERVATION_PLAN_MISMATCH")
    if left["context"].get("workload_hash") != right["context"].get(
        "workload_hash"
    ) or not left["context"].get("workload_hash"):
        reasons.append("WORKLOAD_IDENTITY_MISSING_OR_MISMATCH")
    if any(item["capture_status"] != "complete" for item in (left, right)):
        reasons.append("PARTIAL_CAPTURE")
    a = {(r["metric_id"], r.get("label")): r for r in left["metrics"]}
    b = {(r["metric_id"], r.get("label")): r for r in right["metrics"]}
    diffs = []
    for key in sorted(a.keys() | b.keys(), key=str):
        x, y = a.get(key), b.get(key)
        valid = (
            x
            and y
            and x["availability"] == y["availability"] == "observed"
            and x["unit"] == y["unit"]
            and x["scope"] == y["scope"]
        )
        diffs.append(
            {
                "metric_id": key[0],
                "label": key[1],
                "left": x["value"] if x else None,
                "right": y["value"] if y else None,
                "delta": y["value"] - x["value"] if valid else None,
                "scope": x["scope"] if x else y["scope"],
            }
        )
    return {
        "schema_version": "1.0.0",
        "comparable": not reasons,
        "intervention": {
            "field": intervention,
            "left": left["context"].get(intervention),
            "right": right["context"].get(intervention),
        },
        "reasons": reasons,
        "differences": diffs,
        "quality_status": "not_evaluated",
        "decision": "hold",
        "notice": "Descriptive mechanisms only; no adoption or speedup claim without quality and paired uninstrumented measurements.",
    }
