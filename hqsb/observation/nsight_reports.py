"""Version-aware native report exports; units and collection scope stay explicit."""
import csv
import io
import math
from pathlib import Path
import sqlite3


def ncu_metrics(text):
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if '"Metric Name"' in line and '"Metric Value"' in line), None)
    if start is None:
        return {"availability": "empty", "kernels": [], "metric_rows": 0}
    kernels, count = {}, 0
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        if not row.get("Kernel Name") or not row.get("Metric Name"):
            continue
        key = (row.get("Process ID"), row.get("ID"), row.get("Kernel Name"))
        kernel = kernels.setdefault(key, {"process_id": row.get("Process ID"), "launch_id": row.get("ID"),
            "name": row["Kernel Name"], "grid": row.get("Grid Size"), "block": row.get("Block Size"),
            "device": row.get("Device"), "context": row.get("Context"), "stream": row.get("Stream"), "metrics": []})
        raw = row.get("Metric Value", "")
        try:
            value = float(raw.replace(",", ""))
            if not math.isfinite(value):
                value = None
        except ValueError:
            value = None
        kernel["metrics"].append({"name": row["Metric Name"], "unit": row.get("Metric Unit"),
                                  "value": value, "raw": raw, "section": row.get("Section Name")})
        count += 1
    return {"availability": "observed" if kernels else "empty", "kernels": list(kernels.values()),
            "metric_rows": count, "scope": "selected_kernel_launches_under_application_replay"}


def nsys_timeline(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        names = dict(db.execute("SELECT id,value FROM StringIds")) if "StringIds" in tables else {}
        events, counts, devices, kernels, copies, apis = [], {}, {}, {}, {}, {}
        total = 0
        for table, kind in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                            ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
            if table not in tables:
                continue
            kept = 0
            for item in db.execute(f'SELECT * FROM "{table}" ORDER BY start'):
                row = dict(item)
                if row.get("end", 0) < row.get("start", 0):
                    continue
                begin, end = row["start"], row["end"]
                device = str(row.get("deviceId", "unknown"))
                intervals = devices.setdefault(device, [])
                intervals.append((begin, end))
                name_id = row.get("demangledName", row.get("shortName"))
                name = names.get(name_id, str(name_id)) if kind == "kernel" else kind
                event = {"kind": kind, "name": name, "start_ns": begin, "end_ns": end,
                         "duration_ns": end - begin, "device": device, "stream": row.get("streamId"),
                         "context": row.get("contextId"), "correlation_id": row.get("correlationId"),
                         "global_pid": row.get("globalPid")}
                for key in ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ", "registersPerThread", "staticSharedMemory", "dynamicSharedMemory"):
                    if key in row:
                        event[key] = row[key]
                total += 1
                counts[kind] = counts.get(kind, 0) + 1
                if kept < 5000:
                    events.append(event)
                    kept += 1
                if kind == "kernel":
                    key = (device, name)
                    group = kernels.setdefault(key, {"name": name, "device": device, "launches": 0, "summed_duration_ns": 0})
                    group["launches"] += 1
                    group["summed_duration_ns"] += end - begin
                elif kind == "memcpy":
                    direction = str(row.get("copyKind", "unknown"))
                    group = copies.setdefault(direction, {"copy_kind": direction, "operations": 0, "bytes": 0, "summed_duration_ns": 0})
                    group["operations"] += 1
                    group["bytes"] += row.get("bytes", 0)
                    group["summed_duration_ns"] += end - begin
        activity = []
        for device, intervals in devices.items():
            intervals.sort()
            first, right = intervals[0]
            busy = 0
            for begin, end in intervals[1:]:
                if begin > right:
                    busy += right - first
                    first, right = begin, end
                else:
                    right = max(right, end)
            busy += right - first
            span = max(end for _, end in intervals) - intervals[0][0]
            activity.append({"device": device, "activity_union_ns": busy, "first_to_last_activity_ns": span,
                             "gap_within_activity_span_ns": span - busy,
                             "scope": "union_of_captured_kernel_memcpy_memset_intervals"})
        if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
            for item in db.execute("SELECT nameId,COUNT(*),SUM(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME GROUP BY nameId"):
                name = names.get(item[0], str(item[0]))
                apis[name] = {"name": name, "calls": item[1], "summed_host_duration_ns": item[2]}
        return {"availability": "observed" if counts.get("kernel") else "empty", "counts": counts,
                "top_kernels": sorted(kernels.values(), key=lambda k: k["summed_duration_ns"], reverse=True)[:50],
                "device_activity": activity, "copies": list(copies.values()),
                "cuda_apis": sorted(apis.values(), key=lambda r: r["summed_host_duration_ns"], reverse=True)[:50],
                "timeline": sorted(events, key=lambda r: r["start_ns"])[:5000], "timeline_total_events": total,
                "timeline_truncated": total > 5000, "time_domain": "nsys_export_relative_nanoseconds",
                "kernel_duration_scope": "sum_of_launch_durations_may_overlap", "tables": sorted(tables)}
    finally:
        db.close()
