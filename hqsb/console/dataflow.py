"""Project profiler copy activities into a bounded logical data-flow view.

This is a read-only evidence projection: no device imports, address inference,
or correspondence inferred between tensor inventories and profiler activities.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping

MAX_COPY_DETAILS = 200
_MAX_SAFE_INTEGER = 2**53 - 1
_COPY_CATEGORIES = {"gpu_memcpy", "cuda_memcpy", "memcpy"}
_HOST_CATEGORIES = {
    "cpu_op",
    "cuda_runtime",
    "cuda_driver",
    "python_function",
    "user_annotation",
}
_KERNEL_CATEGORIES = {"kernel", "gpu_kernel", "cuda_kernel"}
_DIRECTIONS = {
    "host_to_device": ("host", "device", "主机 → 设备（逻辑拷贝）"),
    "device_to_host": ("device", "host", "设备 → 主机（逻辑拷贝）"),
    "device_to_device": ("device", "device", "设备 → 设备（逻辑拷贝）"),
    "peer_to_peer": ("device", "peer_device", "设备间 Peer 拷贝（端点未解析）"),
    "host_to_host": ("host", "host", "主机 → 主机（逻辑拷贝）"),
    "unknown": ("unknown", "unknown", "方向未知的拷贝"),
}
_COPY_CODES = {
    "htod": "host_to_device",
    "dtoh": "device_to_host",
    "dtod": "device_to_device",
    "ptop": "peer_to_peer",
    "htoh": "host_to_host",
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    try:
        return value if math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def _bytes(value):
    number = _number(value)
    if number is None or number > _MAX_SAFE_INTEGER or int(number) != number:
        return None
    return int(number)


def _identifier(value):
    if isinstance(value, str):
        return value[:120]
    if isinstance(value, int) and not isinstance(value, bool):
        return value if abs(value) <= _MAX_SAFE_INTEGER else str(value)[:120]
    return None


def _direction(name: str) -> str:
    code = re.search(r"\b([hdp]to[hdp])\b", name, re.IGNORECASE)
    if code:
        return _COPY_CODES.get(code[1].lower(), "unknown")
    # Alternate exporters spell out the same logical endpoints. Do not infer
    # directions from stream/device IDs or ambiguous physical memory names.
    endpoints = re.search(
        r"\b(host|device)\s*(?:->|→|to)\s*(host|device)\b",
        name,
        re.IGNORECASE,
    )
    if endpoints:
        return f"{endpoints[1].lower()}_to_{endpoints[2].lower()}"
    return "unknown"


def _copy_activity(category: str, name: str) -> bool:
    if category in _HOST_CATEGORIES:
        return False
    return category in _COPY_CATEGORIES or bool(
        re.match(r"^memcpy\b", name, re.IGNORECASE)
    )


def _aggregate():
    return {
        "count": 0,
        "bytes_known_events": 0,
        "known_bytes": 0,
        "duration_known_events": 0,
        "duration_ms_sum": 0.0,
    }


def _add(bucket, byte_count, duration):
    bucket["count"] += 1
    if byte_count is not None:
        bucket["bytes_known_events"] += 1
        bucket["known_bytes"] += byte_count
    if duration is not None:
        bucket["duration_known_events"] += 1
        bucket["duration_ms_sum"] += duration


def _finish(bucket):
    result = dict(bucket)
    known = bucket["known_bytes"] if bucket["bytes_known_events"] else None
    # Returning null is preferable to silently rounding large JSON integers.
    result["known_bytes"] = (
        known if known is None or known <= _MAX_SAFE_INTEGER else None
    )
    complete = bucket["count"] > 0 and bucket["bytes_known_events"] == bucket["count"]
    result["bytes"] = result["known_bytes"] if complete else None
    result["bytes_complete"] = complete and result["known_bytes"] is not None
    if not bucket["duration_known_events"] or not math.isfinite(
        bucket["duration_ms_sum"]
    ):
        result["duration_ms_sum"] = None
    return result


def build_dataflow(
    run: Mapping, events: Iterable[Mapping], trace_summary: Mapping
) -> dict:
    """Aggregate retained normalized events without copying the trace event list.

    ``bytes`` is a total only when every copy activity explicitly supplies a
    valid byte count. ``known_bytes`` is the reported subset, never a complete
    traffic estimate. Durations are activity sums, not elapsed time or bandwidth.
    """
    metrics = run.get("metrics") or {}
    observation = metrics.get("observation") or {}
    profile = observation.get("profile") or {}
    summary = _aggregate()
    buckets = {}
    details = []
    kernel_count = 0
    event_count = 0
    for event in events:
        if not isinstance(event, Mapping):
            continue
        event_count += 1
        category = str(event.get("category", "")).lower()
        name = str(event.get("name", ""))[:2000]
        if category in _KERNEL_CATEGORIES:
            kernel_count += 1
            # A kernel name containing "memcpy" is not a copy-engine activity.
            continue
        if not _copy_activity(category, name):
            continue
        args = event.get("args") or {}
        if not isinstance(args, Mapping):
            args = {}
        direction = _direction(name)
        byte_count = _bytes(args.get("bytes"))
        duration = _number(event.get("duration_ms"))
        bucket = buckets.setdefault(direction, _aggregate())
        _add(bucket, byte_count, duration)
        _add(summary, byte_count, duration)
        if len(details) < MAX_COPY_DETAILS:
            details.append(
                {
                    "id": _identifier(event.get("id")),
                    "name": name,
                    "direction": direction,
                    "start_ms": _number(event.get("start_ms")),
                    "duration_ms": duration,
                    "bytes": byte_count,
                    "correlation": _identifier(args.get("correlation")),
                    "stream": _identifier(args.get("stream")),
                    "device": _identifier(args.get("device")),
                    "source": "torch.profiler.ChromeTrace",
                }
            )

    edges = []
    for direction, (source, target, label) in _DIRECTIONS.items():
        if direction in buckets:
            edges.append(
                {
                    "direction": direction,
                    "source": source,
                    "target": target,
                    "label": label,
                    **_finish(buckets[direction]),
                }
            )
    totals = _finish(summary)
    totals["copy_events"] = totals.pop("count")
    totals["kernel_events"] = kernel_count
    status = trace_summary.get("status", "not_collected")
    if status == "available":
        if not summary["count"]:
            status = "no_copy_events"
        elif not totals["bytes_complete"]:
            status = "partial"
    limitations = [
        "仅投影 profiler 采集窗口中保留的拷贝活动；未观测到不代表未发生。",
        "箭头表示 Host/Device 逻辑地址空间方向，不是物理总线路径。Jetson 统一内存不等于独立 HBM。",
        "未解析 HBM/DRAM、L2、寄存器、物理地址或每字节流向；不根据张量名称猜测拷贝归属。",
        "bytes 仅在每条活动都明确提供字节数时给出；known_bytes 只累计已知子集，同一字节可能被重复拷贝。",
        "duration_ms_sum 是活动耗时之和，重叠流不可当作墙钟时延；这里不计算有效带宽。",
        "CPU 的 cudaMemcpy API 与 GPU memcpy 活动不重复累计；kernel 数不代表所有内存读写次数。",
        "参数/KV 张量存储、allocator、RSS 和系统可用内存属于不同口径，不能相加。",
        "最多返回 200 条拷贝明细；汇总覆盖所有传入的保留事件，完整原件可独立下载。",
    ]
    for limitation in trace_summary.get("limitations") or []:
        if isinstance(limitation, str) and limitation not in limitations:
            limitations.append(limitation[:2000])
    return {
        "schema_version": 1,
        "run_id": run.get("id"),
        "status": status,
        "source": "torch.profiler.ChromeTrace",
        "clock_domain": trace_summary.get("clock_domain", "profiler_trace_relative"),
        "coverage": {
            "profile_window": dict(profile.get("coverage") or {}),
            "activities": list(profile.get("activities") or []),
            "trace_status": trace_summary.get("status", "not_collected"),
            "retained_events": event_count,
            "excluded_or_truncated_events": trace_summary.get(
                "excluded_or_truncated_events"
            ),
            "copy_bytes_known_events": summary["bytes_known_events"],
            "copy_events": summary["count"],
            "bytes_complete": totals["bytes_complete"],
            "trace_sha256": trace_summary.get("sha256"),
            "scope": "retained_profiler_capture_window_not_entire_inference",
        },
        "nodes": [
            {"id": "host", "label": "CPU / 主机逻辑地址空间"},
            {"id": "device", "label": "GPU / 设备逻辑地址空间"},
            {"id": "peer_device", "label": "Peer 设备（端点未知）"},
            {"id": "unknown", "label": "未解析端点"},
        ],
        "edges": edges,
        "totals": totals,
        "copy_events": details,
        "copy_events_truncated": max(0, summary["count"] - len(details)),
        "limitations": limitations,
    }
