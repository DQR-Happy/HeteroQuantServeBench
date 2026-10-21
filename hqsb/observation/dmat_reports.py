"""Exact address-event reader and bounded DMAT aggregation; no GPU imports.

Addresses and absolute GPU timestamps leave this module as decimal/hex strings.
Only relative ns and bounded counts are sent as JSON numbers to the browser.
"""
from collections import Counter
import hashlib
import json
from math import prod
from pathlib import Path
import struct
from xml.sax.saxutils import escape

HEADER = struct.Struct("<8I")
EVENT = struct.Struct("<QQIIII")
MAX_FRAME_BYTES = 256 * 1024**2


def read_frame(path):
    import numpy as np
    path = Path(path)
    with path.open("rb") as stream:
        magic = stream.read(8)
        if magic not in (b"HQDMAT1\n", b"HQDMAT2\n"):
            raise ValueError("DMAT_MAGIC")
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("DMAT_TRUNCATED_HEADER")
        *dims, shared, capacity = HEADER.unpack(header)
        threads = prod(dims)
        event_bytes = 48 if magic == b"HQDMAT2\n" else 32
        size = threads * (capacity * event_bytes + 8)
        if min(dims) < 1 or not 1 <= capacity <= 2048 or size > MAX_FRAME_BYTES:
            raise ValueError("DMAT_FRAME_BUDGET")
        if path.stat().st_size != 40 + size:
            raise ValueError("DMAT_TRUNCATED_OR_TRAILING_DATA")
        counts = np.frombuffer(stream.read(threads * 8), dtype="<u8")
        clocks = [("clock", "<u8"), ("entry", "<u8")] if event_bytes == 48 else []
        dtype = np.dtype([("time", "<u8"), *clocks, ("address", "<u8"), ("site", "<u4"),
                          ("thread", "<u4"), ("mask", "<u4"), ("width", "<u4")])
        records = np.frombuffer(stream.read(threads * capacity * event_bytes), dtype=dtype).reshape(threads, capacity)
    return {"grid": dims[:3], "block": dims[3:], "shared_bytes": shared, "capacity": capacity,
            "counts": counts, "records": records, "event_bytes": event_bytes}


def aggregate(frame, sites, *, page_bytes=65536, time_bins=128, clock="GPU_globaltimer", tensors=(), window=None, bin_width=None, color_scale=None, address_origin="virtual"):
    import numpy as np
    if page_bytes not in (4096, 16384, 65536) or not 16 <= time_bins <= 512:
        raise ValueError("DMAT_AGGREGATION_PARAMETERS")
    counts, raw, capacity = frame["counts"], frame["records"], frame["capacity"]
    indices = np.arange(capacity)[None, :] < np.minimum(counts, capacity)[:, None]
    rows = raw[indices]
    owners = np.broadcast_to(np.arange(len(counts))[:, None], indices.shape)[indices]
    block_threads = prod(frame["block"])
    if len(rows) and (np.any(rows["thread"] != owners) or np.any(rows["time"] == 0) or
                      np.any((rows["mask"] & (np.uint32(1) << ((owners % block_threads) % 32).astype("uint32"))) == 0)):
        raise ValueError("DMAT_EVENT_THREAD_OR_CLOCK")
    lookup = {s["site"]: s for s in sites}
    for site in np.unique(rows["site"]):
        selected = rows[rows["site"] == site]
        if int(site) not in lookup or np.any(selected["width"] == 0) or np.any(selected["width"] > lookup[int(site)]["width"]) or (not lookup[int(site)].get("variable_width") and np.any(selected["width"] != lookup[int(site)]["width"])):
            raise ValueError("DMAT_EVENT_SITE_OR_WIDTH")
    if frame.get("event_bytes", 32) == 48:
        from hqsb.observation.dmat_grid import grid
        return grid(frame, rows, sites, page_bytes=page_bytes, time_bins=time_bins, clock=clock,
                    tensors=tensors, window=window, bin_width=bin_width, color_scale=color_scale, address_origin=address_origin)
    if clock != "GPU_globaltimer":
        raise ValueError("DMAT_LEGACY_TRACE_REQUIRES_GLOBALTIMER: 此记录保存全局时间，重新采集可获得线程入口归一化时间。")
    if window is not None:
        from hqsb.observation.dmat_grid import grid
        return grid(frame, rows, sites, page_bytes=page_bytes, time_bins=time_bins, clock=clock,
                    tensors=tensors, window=window, bin_width=bin_width, color_scale=color_scale, address_origin=address_origin)
    first = int(rows["time"].min()) if len(rows) else 0
    span = int(rows["time"].max()) - first + 1 if len(rows) else 1
    requested_time_bins = time_bins
    # Bound the browser payload; contiguous display rows preserve gaps through
    # their explicit VA labels. Coarsen address bins instead of dropping events.
    address_bin = page_bytes
    while len(np.unique(rows["address"] // address_bin)) > 256:
        address_bin *= 2
    pages = sorted(int(x) * address_bin for x in np.unique(rows["address"] // address_bin))
    # Up to 32 launches share one API receipt. Bound each dense grid as well as
    # the raw frame; preserve every event by coarsening, never by truncating.
    time_bins = min(time_bins, max(16, 8192 // max(1, len(pages))))
    bin_ns = max(1, (span + time_bins - 1) // time_bins)
    page_index = {page: i for i, page in enumerate(pages)}
    cells, read_counts, write_counts, byte_counts = Counter(), Counter(), Counter(), Counter()
    site_counts = Counter(int(x) for x in rows["site"])
    for row in rows:
        key = (page_index[int(row["address"]) // address_bin * address_bin], (int(row["time"]) - first) // bin_ns)
        cells[key] += 1
        (read_counts if lookup[int(row["site"])]["operation"] == "read" else write_counts)[key] += 1
        byte_counts[key] += int(row["width"])
    attempted = sum(int(x) for x in counts)
    return {"events": len(rows), "attempted_events": attempted, "dropped_events": attempted - len(rows),
            "overflow_threads": int(np.count_nonzero(counts > capacity)), "complete": attempted == len(rows),
            "gpu_start_ns": str(first), "duration_ns": span, "time_bin_ns": bin_ns, "time_bins": time_bins,
            "requested_time_bins": requested_time_bins,
            "requested_address_bin_bytes": page_bytes, "address_bin_bytes": address_bin,
            "addresses": [hex(page) for page in pages], "address_axis": "sorted_observed_virtual_address_bins_gaps_collapsed",
            "density_semantics": "enabled_thread_memory_instruction_events_grouped_by_start_address",
            "cells": [{"row": y, "column": x, "events": n, "reads": read_counts[y, x],
                       "writes": write_counts[y, x], "instruction_bytes": byte_counts[y, x]} for (y, x), n in sorted(cells.items())],
            "sites": [{**site, "events": site_counts[site["site"]]} for site in sites]}


def svg_plot(plot, title):
    width, height = 1100, max(240, 80 + len(plot["addresses"]) * 7)
    rows = max(1, len(plot["addresses"]))
    maximum = max([c["events"] for c in plot["cells"]] or [1])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="12" y="20" font-size="14">{escape(title)}</text>']
    for cell in plot["cells"]:
        x, y = 170 + cell["column"] * 900 / plot["time_bins"], 35 + cell["row"] * (height - 75) / rows
        opacity = cell.get("level", 0) / 5 if "level" in cell else .15 + .85 * cell["events"] / maximum
        color = next((o["color"] for o in plot.get("objects", []) if o["id"] == cell.get("object")), "#326ea5")
        parts.append(f'<rect x="{x}" y="{y}" width="{900 / plot["time_bins"]}" height="{(height - 75) / rows}" fill="{color}" opacity="{opacity}"><title>{plot["addresses"][cell["row"]]}: {cell["events"]} events</title></rect>')
    step = max(1, rows // 12)
    for i in range(0, len(plot["addresses"]), step):
        label = plot.get("address_rows", [{}] * len(plot["addresses"]))[i].get("label", plot["addresses"][i])
        parts.append(f'<text x="5" y="{45 + i * (height - 75) / rows}" font-size="10">{escape(label)}</text>')
    clock = plot.get("clock_domain", "GPU globaltimer")
    unit = plot.get("time_unit", "ns")
    parts.append(f'<text x="170" y="{height - 12}" font-size="11">{clock}: {plot.get("time_start", 0)}–{plot.get("time_end", plot["time_bin_ns"] * plot["time_bins"])} {unit}; address bin {plot["address_bin_bytes"]} B; gaps collapsed</text></svg>')
    return "".join(parts)


def summarize_memory(root, *, page_bytes=65536, time_bins=128, clock=None):
    root = Path(root).resolve()
    launches, coverage, artifacts = [], [], []
    for sidecar in sorted(root.glob("**/hqsb-launches.jsonl")):
        coverage.append(json.loads((sidecar.parent / "coverage.json").read_text()))
        identities = [json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()]
        captured = [item for item in identities if item["captured"]]
        if len(list(sidecar.parent.glob("memory-*.dmat"))) != len(captured):
            raise ValueError("DMAT_FRAME_LINEAGE_COUNT")
        for index, identity in enumerate(captured):
            if len(launches) >= 32:
                raise ValueError("DMAT_FRAME_COUNT_BUDGET")
            path = sidecar.parent / f"memory-{index}.dmat"
            frame = read_frame(path)
            # Native compiler folders are siblings of result/ under this trace.
            # Resolve there first so sealed artifacts keep working after export
            # to another machine; preserve the original absolute path as evidence.
            original = Path(identity["memory_metadata"])
            portable = sidecar.parent / "kernel" / original.name / "memory-sites.json"
            metadata = (portable if portable.is_file() else original / "memory-sites.json").resolve()
            if not metadata.is_relative_to(root):
                raise ValueError("DMAT_METADATA_LINEAGE")
            source = json.loads(metadata.read_text())
            if frame["grid"] != identity["grid"] or frame["block"] != identity["block"] or source["kernel"] != identity["kernel"] or frame["capacity"] != source["capacity"]:
                raise ValueError("DMAT_LAUNCH_LINEAGE")
            selected_clock = clock or ("thread_entry_clock64" if frame["event_bytes"] == 48 else "GPU_globaltimer")
            context = identity.get("tensor_context", {})
            plot = aggregate(frame, source["sites"], page_bytes=page_bytes, time_bins=time_bins,
                             clock=selected_clock, tensors=context.get("tensors", []))
            relative = str(path.relative_to(root))
            plot_path = path.with_suffix(".svg")
            plot_path.write_text(svg_plot(plot, identity["kernel"]))
            launches.append({"index": index, "kernel": identity["kernel"], "grid": frame["grid"], "block": frame["block"],
                             "trace": relative, "svg": str(plot_path.relative_to(root)), "probe": source, "context": context, "plot": plot})
            artifacts.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if sum(c["captured_launches"] for c in coverage) != len(launches):
        raise ValueError("DMAT_COVERAGE_FRAME_COUNT")
    events = sum(x["plot"]["events"] for x in launches)
    partial = any(not x["plot"]["complete"] for x in launches) or any(c.get("probe_failed_launches", 0) or c.get("other_thread_launches", 0) for c in coverage)
    domains = {x["plot"].get("clock_domain", "GPU_globaltimer") for x in launches}
    primary_clock = next(iter(domains)) if len(domains) == 1 else "per_launch"
    return {"tool": "eretrace", "mode": "memory", "schema_version": "hqsb-dmat-2", "evidence_kind": "measured",
            "clock_domain": primary_clock, "time_unit": "cycles" if primary_clock == "thread_entry_clock64" else "ns", "clock_resolution": "device_dependent",
            "availability": "partial" if partial else "observed" if events else "empty", "coverage": coverage,
            "counts": {"launches": len(launches), "events": events, "dropped_events": sum(x["plot"]["dropped_events"] for x in launches)},
            "memory_launches": launches, "decoded_artifacts": artifacts,
            "measurement_role": "instrumented_diagnostic", "physical_page_mapping": "requires_driver_evidence",
            "cache_hit_rate": "requires_hardware_counters"}
