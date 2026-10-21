"""Bounded streaming reader for ERETrace reference/raw warp frames.

Integer fields remain exact. clock64 starts are SM-local cycle counters; this
reader aggregates elapsed cycles and never merges starts into a GPU timeline.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

HEADER = struct.Struct("<10IQ")
MAX_FRAME_BYTES = 16 * 1024**2


def exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("ERE_TRUNCATED_FRAME")
    return data


def decode_reference(data, n, warps):
    import numpy as np
    if len(data) < 2064:
        raise ValueError("ERE_REFERENCE_HEADER")
    flags, group, ew, version = struct.unpack_from("<4I", data)
    # NTRF022 uses CTA anchors, omitted anchor locals, elapsed16. Flat CTAs
    # use the artifact's version=1 signed delta24 representation.
    if flags or ew != 2:
        raise ValueError("ERE_UNACCEPTED_REFERENCE_PAYLOAD")
    out = np.empty(n, dtype=[("start", "<u8"), ("elapsed", "<u4"), ("sm", "<u4")])
    bases = np.frombuffer(data, dtype="<u8", count=256, offset=16)
    if warps <= 2:
        if version != 1 or group != 3 or len(data) != 2064 + n * 6:
            raise ValueError("ERE_FLAT_LAYOUT")
        cols = np.frombuffer(data, dtype="u1", count=n * 3, offset=2064).reshape(n, 3)
        bits = cols[:, 0].astype("int64") | (cols[:, 1].astype("int64") << 8) | (cols[:, 2].astype("int64") << 16)
        delta = (bits ^ (1 << 23)) - (1 << 23)
        elapsed = np.frombuffer(data, dtype="<u2", count=n, offset=2064 + n * 3)
        sm = np.frombuffer(data, dtype="u1", count=n, offset=2064 + n * 5)
        start = bases[sm] + delta.astype("uint64")
    else:
        groups = (n + warps - 1) // warps
        m = n - groups
        local_bytes = (m * 12 + 7) // 8
        if version != 5 or group != warps or len(data) != 2064 + groups * 4 + local_bytes + n * 2:
            raise ValueError("ERE_CTA_LAYOUT")
        anchors = np.frombuffer(data, dtype="u1", count=groups * 4, offset=2064).reshape(groups, 4)
        bits = anchors[:, 0].astype("int64") | (anchors[:, 1].astype("int64") << 8) | (anchors[:, 2].astype("int64") << 16)
        sm = np.repeat(anchors[:, 3], warps)[:n]
        base = bases[anchors[:, 3]] + ((bits ^ (1 << 23)) - (1 << 23)).astype("uint64")
        packed = np.frombuffer(data, dtype="u1", count=local_bytes, offset=2064 + groups * 4)
        # Pad for the last odd 12-bit entry. Padding is only a decode guard.
        packed = np.pad(packed, (0, 1))
        bit = np.arange(m, dtype="int64") * 12
        value = (packed[bit // 8].astype("uint32") | (packed[bit // 8 + 1].astype("uint32") << 8)) >> (bit % 8)
        local = np.zeros(n, dtype="int64")
        local[np.arange(n) % warps != 0] = ((value & 4095).astype("int64") ^ 2048) - 2048
        start = np.repeat(base, warps)[:n] + local.astype("uint64")
        elapsed = np.frombuffer(data, dtype="<u2", count=n, offset=2064 + groups * 4 + local_bytes)
    out["start"], out["elapsed"], out["sm"] = start, elapsed, sm
    return out


def frames(path):
    import numpy as np
    dtype = np.dtype([("start", "<u8"), ("elapsed", "<u4"), ("sm", "<u4")])
    with Path(path).open("rb") as stream:
        magic = exact(stream, 8)
        if magic not in {b"NTRF022\n", b"NTGPU13\n"}:
            raise ValueError("ERE_FORMAT_UNSUPPORTED:" + repr(magic))
        index = 0
        while True:
            header = stream.read(48)
            if not header:
                break
            if len(header) != 48:
                raise ValueError("ERE_TRUNCATED_HEADER")
            gx, gy, gz, bx, by, bz, shared, maps, size, divisor, offset = HEADER.unpack(header)
            if min(gx, gy, gz, bx, by, bz) < 1 or maps != 1 or size != 16 or divisor != 32 or offset != 48:
                raise ValueError("ERE_FRAME_SHAPE")
            warps = (bx * by * bz + 31) // 32
            n = gx * gy * gz * warps
            if n * 16 > MAX_FRAME_BYTES:
                raise ValueError("ERE_FRAME_BUDGET")
            word, = struct.unpack("<Q", exact(stream, 8))
            tag, length = word >> 62, word & ((1 << 62) - 1)
            if tag == 0 and length == 0:
                data = exact(stream, n * 16)
                rows, encoding = np.frombuffer(data, dtype=dtype), "raw"
            elif tag == 1 and length == n * 16:
                data = exact(stream, length)
                rows, encoding = np.frombuffer(data, dtype=dtype), "raw_recovery"
            elif tag == 3 and magic == b"NTRF022\n":
                expected = 2064 + n * 6 if warps <= 2 else 2064 + 4 * ((n + warps - 1) // warps) + ((n - (n + warps - 1) // warps) * 12 + 7) // 8 + n * 2
                if length != expected or length > MAX_FRAME_BYTES:
                    raise ValueError("ERE_PAYLOAD_SIZE")
                rows, encoding = decode_reference(exact(stream, length), n, warps), "reference"
            else:
                raise ValueError("ERE_FRAME_TAG")
            # Zero records represent unobserved/early-return lane-0 slots.
            if np.any(rows["sm"] > 255):
                raise ValueError("ERE_SM_DOMAIN")
            yield {"index": index, "grid": [gx, gy, gz], "block": [bx, by, bz], "encoding": encoding,
                   "header": header, "rows": rows, "raw_bytes": n * 16}
            index += 1


def summarize(root, *, max_frames=32):
    import numpy as np
    root = Path(root)
    sm_counts, sm_cycles, hist, encodings = Counter(), Counter(), Counter(), Counter()
    launches, coverage, collector, artifacts = [], [], [], []
    total_records = zeros = logical = stored = 0
    for stats in sorted(root.glob("**/arena-stats.json")):
        collector.append(json.loads(stats.read_text()))
    for item in sorted(root.glob("**/coverage.json")):
        coverage.append(json.loads(item.read_text()))
    for path in sorted(root.glob("**/arena-*.nts")):
        sidecar = path.parent / "hqsb-launches.jsonl"
        identities = [json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()] if sidecar.exists() else []
        identities = [x for x in identities if x["captured"]]
        digest = hashlib.sha256()
        count = 0
        for frame in frames(path):
            if len(launches) >= max_frames:
                raise ValueError("ERE_FRAME_COUNT_BUDGET")
            rows = frame.pop("rows")
            index = frame["index"]
            if index >= len(identities) or frame["grid"] != identities[index]["grid"] or frame["block"] != identities[index]["block"]:
                raise ValueError("ERE_LAUNCH_LINEAGE_MISMATCH")
            digest.update(frame.pop("header")); digest.update(rows.tobytes())
            valid = (rows["start"] != 0) | (rows["elapsed"] != 0) | (rows["sm"] != 0)
            active = rows[valid]
            n = len(rows)
            zeros += n - len(active); total_records += n; logical += frame["raw_bytes"] + 48
            for sm in np.unique(active["sm"]):
                times = active["elapsed"][active["sm"] == sm]
                sm_counts[int(sm)] += len(times); sm_cycles[int(sm)] += int(times.sum(dtype="uint64"))
            buckets, values = np.unique(np.where(active["elapsed"] > 0, np.floor(np.log2(np.maximum(active["elapsed"], 1))).astype("int64") + 1, 0), return_counts=True)
            hist.update({int(k): int(v) for k, v in zip(buckets, values)})
            encodings[frame["encoding"]] += 1
            times = active["elapsed"]
            launches.append({**frame, "kernel": identities[index]["kernel"], "trace": str(path.relative_to(root)),
                             "records": n, "observed_records": len(active),
                             "elapsed_cycles_mean": float(times.mean()) if len(times) else None,
                             "elapsed_cycles_p50": float(np.percentile(times, 50)) if len(times) else None,
                             "elapsed_cycles_p95": float(np.percentile(times, 95)) if len(times) else None,
                             "elapsed_cycles_max": int(times.max()) if len(times) else None})
            count += 1
        if count != len(identities):
            raise ValueError("ERE_FRAME_LINEAGE_COUNT")
        stored += path.stat().st_size
        artifacts.append({"path": str(path.relative_to(root)), "decoded_sha256": digest.hexdigest(), "frames": count})
    reported = sum(x["captured_launches"] for x in coverage)
    if reported != len(launches):
        raise ValueError("ERE_COVERAGE_FRAME_COUNT")
    ptx_sources = [{"artifact": str(p.relative_to(root)), **json.loads(p.read_text())} for p in sorted(root.glob("**/kernel/*/ptx-origin.json"))]
    incomplete = any(x.get("probe_failed_launches", 0) or x.get("other_thread_launches", 0) for x in coverage)
    availability = "empty" if not total_records - zeros else "partial" if incomplete else "observed"
    return {"availability": availability, "tool": "eretrace", "evidence_kind": "measured",
            "clock_domain": "clock64_per_SM", "elapsed_unit": "cycles", "sample_semantics": "lane0_entry_to_own_exit",
            "attribution": "engine_capture_window_kernel_launch_CTA_warp", "launches": launches,
            "counts": {"launches": len(launches), "warp_slots": total_records, "observed_records": total_records - zeros, "zero_slots": zeros},
            "sm_distribution": [{"sm": sm, "records": sm_counts[sm], "summed_elapsed_cycles": sm_cycles[sm]} for sm in sorted(sm_counts)],
            "elapsed_histogram": [{"log2_bucket": k, "records": hist[k]} for k in sorted(hist)],
            "encoding_counts": dict(encodings), "trace_bytes": stored, "logical_bytes": logical,
            "logical_to_stored_ratio": logical / stored if stored else None,
            "coverage": coverage, "collector": collector, "decoded_artifacts": artifacts,
            "ptx_sources": ptx_sources,
            "additional_metrics": {"cache_hit_rate": "use_ncu_hardware_counters", "GPU_physical_pages": "requires_driver_memory_events",
                                   "memory_addresses": "separate_variable_event_frontend", "global_timeline": "requires_cross_SM_clock_calibration"}}
