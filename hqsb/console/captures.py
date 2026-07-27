"""Bounded trace queries and artifact access. No profiler or device imports."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import codecs
from pathlib import Path

RUN_ID = re.compile(r"run_[0-9a-f]{32}\Z")
MAX_TRACE_BYTES = 64 * 1024 * 1024
MAX_EVENTS = 150_000
TRACE_CHUNK_BYTES = 64 * 1024
MAX_TRACE_VALUE_BYTES = 1024 * 1024
ARG_KEYS = (
    "External id",
    "External Id",
    "correlation",
    "stream",
    "device",
    "bytes",
    "grid",
    "block",
    "registers per thread",
    "shared memory",
    "Input Dims",
    "Input type",
    "Concrete Inputs",
    "Ev Idx",
)


def safe_directory(root: Path, key: str) -> Path:
    if not RUN_ID.fullmatch(key):
        raise KeyError(key)
    root = root.resolve()
    path = root / key
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise KeyError(key)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _safe_arg(value, depth=0):
    if depth > 6:
        return None
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return value if _finite(value) else None
    if isinstance(value, str):
        return value[:2000]
    if isinstance(value, list):
        return [_safe_arg(item, depth + 1) for item in value[:64]]
    if isinstance(value, dict):
        return {
            str(key)[:80]: _safe_arg(item, depth + 1)
            for key, item in list(value.items())[:64]
        }
    return value if value is None or isinstance(value, bool) else None


class _TraceJSONReader:
    """Incremental JSON values with bounded byte buffers, using stdlib syntax.

    The container grammar is parsed separately so the large traceEvents array
    is never passed to JSONDecoder. Other top-level values share the 1 MiB
    per-value budget. No regular expression searches through string content.
    """

    def __init__(self, stream):
        self.stream = stream
        self.decoder = json.JSONDecoder()
        self.utf8 = codecs.getincrementaldecoder("utf-8")()
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.bytes_read = 0

    def _fill(self):
        # Compact only at chunk boundaries, not once per small trace event.
        if self.position:
            self.buffer = self.buffer[self.position :]
            self.position = 0
        buffered_bytes = len(self.buffer.encode("utf-8")) + len(self.utf8.getstate()[0])
        if buffered_bytes > MAX_TRACE_VALUE_BYTES:
            raise ValueError("Trace value exceeds the 1 MiB budget")
        # One extra byte is sufficient to distinguish an exactly budget-sized
        # value from the delimiter/number continuation at the next boundary.
        remaining = MAX_TRACE_VALUE_BYTES - buffered_bytes
        chunk = self.stream.read(min(TRACE_CHUNK_BYTES, max(1, remaining + 1)))
        self.bytes_read += len(chunk)
        if self.bytes_read > MAX_TRACE_BYTES:
            raise ValueError("Trace grew beyond the original-file read budget")
        if chunk:
            self.buffer += self.utf8.decode(chunk)
        else:
            self.buffer += self.utf8.decode(b"", final=True)
            self.eof = True

    def peek(self):
        while True:
            while (
                self.position < len(self.buffer)
                and self.buffer[self.position] in " \t\r\n"
            ):
                self.position += 1
            if self.position < len(self.buffer) or self.eof:
                return self.buffer[self.position : self.position + 1]
            self._fill()

    def expect(self, character):
        if self.peek() != character:
            raise ValueError("Invalid JSON trace container")
        self.position += 1

    def value(self):
        if not self.peek():
            raise ValueError("Truncated JSON trace value")
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if self.eof:
                    raise ValueError("Invalid or truncated JSON trace value") from None
            else:
                # A number can end at a chunk boundary before further digits
                # or an exponent arrive. Require a complete lexical boundary.
                boundary = self.buffer[end : end + 1]
                if (boundary and boundary in " \t\r\n,]}:") or (
                    end == len(self.buffer) and self.eof
                ):
                    if (
                        len(self.buffer[self.position : end].encode("utf-8"))
                        > MAX_TRACE_VALUE_BYTES
                    ):
                        raise ValueError("Trace value exceeds the 1 MiB budget")
                    self.position = end
                    return value
                if self.eof:
                    raise ValueError("Invalid JSON value boundary")
            if (
                len(self.buffer[self.position :].encode("utf-8"))
                > MAX_TRACE_VALUE_BYTES
            ):
                raise ValueError("Trace value exceeds the 1 MiB budget")
            self._fill()


def _iter_trace_events(stream):
    """Yield only the top-level traceEvents items and validate the whole file."""
    reader = _TraceJSONReader(stream)
    reader.expect("{")
    seen_events = False
    if reader.peek() == "}":
        reader.expect("}")
    else:
        while True:
            key = reader.value()
            if not isinstance(key, str):
                raise ValueError("Trace object keys must be strings")
            reader.expect(":")
            if key == "traceEvents":
                if seen_events:
                    raise ValueError("Duplicate top-level traceEvents key")
                seen_events = True
                reader.expect("[")
                if reader.peek() == "]":
                    reader.expect("]")
                else:
                    while True:
                        yield reader.value()
                        if reader.peek() == "]":
                            reader.expect("]")
                            break
                        reader.expect(",")
            else:
                # Metadata is decoded and discarded one bounded value at a time.
                reader.value()
            if reader.peek() == "}":
                reader.expect("}")
                break
            reader.expect(",")
    if reader.peek():
        raise ValueError("Trailing content after the JSON trace object")


def normalize_trace(raw: dict) -> tuple[list[dict], dict]:
    """Preserve tool correlation IDs; never infer causal joins from time alone."""
    entries = raw.get("traceEvents", [])
    if not isinstance(entries, list):
        raise ValueError("traceEvents must be an array")
    return _normalize_events(entries)


def _normalize_events(entries) -> tuple[list[dict], dict]:
    """Consume one event at a time; only bounded normalized rows are retained."""
    events = []
    dropped = 0
    for ordinal, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("ph") != "X":
            continue
        if (
            not _finite(entry.get("ts"))
            or not _finite(entry.get("dur"))
            or entry["dur"] < 0
        ):
            dropped += 1
            continue
        if len(events) >= MAX_EVENTS:
            dropped += 1
            continue
        args = entry.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        events.append(
            {
                "id": ordinal,
                "name": str(entry.get("name", "unknown"))[:2000],
                "category": str(entry.get("cat", "unknown"))[:80],
                "pid": str(entry.get("pid", "")),
                "tid": str(entry.get("tid", "")),
                "start_ms": entry["ts"] / 1000,
                "duration_ms": entry["dur"] / 1000,
                "args": {
                    key: _safe_arg(args[key])
                    for key in ARG_KEYS
                    if key in args and key != "Concrete Inputs"
                },
                "source": "torch.profiler.ChromeTrace",
            }
        )
    origin = min((row["start_ms"] for row in events), default=0)
    for row in events:
        row["start_ms"] -= origin
    events.sort(key=lambda row: (row["start_ms"], row["id"]))
    counts = {}
    for row in events:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return events, {
        "status": "partial" if dropped else "available",
        "events": len(events),
        "excluded_or_truncated_events": dropped,
        "categories": counts,
        "origin_us": origin * 1000,
        "clock_domain": "profiler_trace_relative",
        "limitations": [
            "时间线为采集窗口，不代表未采集 token；与 worker 阶段图是不同起点。",
            "只对工具提供的关联 ID 做展示；框架算子与 kernel 可多对多，未验证的关联不补全。",
            "工具自身丢失事件数若未提供则未知；本计数只记录解析排除/截断。",
            "JSON 按顶层 traceEvents 逐项读取；最多保留 150000 个完整事件，单项 JSON 上限 1 MiB。",
        ],
    }


class CaptureRepository:
    """Single-entry bounded cache; captures remain isolated from run-event SQLite."""

    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()
        self._key = None
        self._events: list[dict] = []
        self._summary: dict = {}

    def trace_file(self, key: str) -> Path:
        directory = safe_directory(self.root, key)
        path = directory / "trace.json"
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
        ):
            raise KeyError(key)
        return path

    def _load(self, key: str):
        path = self.trace_file(key)
        stat = path.stat()
        identity = (key, stat.st_mtime_ns, stat.st_size)
        if identity == self._key:
            return
        self._key, self._events, self._summary = None, [], {}
        if stat.st_size > MAX_TRACE_BYTES:
            self._summary = {
                "status": "download_only",
                "events": None,
                "limitations": ["原件超过 64 MiB 解析预算，请下载到专用工具分析。"],
            }
        else:
            try:
                with path.open("rb") as stream:
                    self._events, self._summary = _normalize_events(
                        _iter_trace_events(stream)
                    )
            except (ValueError, TypeError, AttributeError, RecursionError):
                self._summary = {
                    "status": "invalid",
                    "events": None,
                    "limitations": [
                        "Trace 格式不完整、不支持、嵌套过深或单项超过 1 MiB 解析预算；保留原件供诊断。"
                    ],
                }
        self._summary.update(
            bytes=stat.st_size, sha256=sha256_file(path), capture_id=key
        )
        self._key = identity

    def summary(self, key: str) -> dict:
        with self.lock:
            try:
                self._load(key)
            except KeyError:
                return {
                    "status": "not_collected",
                    "capture_id": key,
                    "events": 0,
                    "limitations": [
                        "此请求没有可用 trace；历史请求不会自动补采或重放。"
                    ],
                }
            return dict(self._summary)

    def project(self, key: str, projector):
        """Build a bounded read-only view without copying the cached event list.

        Projectors consume the iterator synchronously under the repository lock;
        they must return summaries, never mutate or retain the source events.
        A missing trace is an explicit data state, not a fabricated empty run.
        """
        with self.lock:
            try:
                self._load(key)
            except KeyError:
                return projector(
                    iter(()),
                    {
                        "status": "not_collected",
                        "capture_id": key,
                        "events": None,
                        "limitations": ["此请求未保存 trace，无法恢复拷贝事件。"],
                    },
                )
            return projector(iter(self._events), dict(self._summary))

    def query(
        self,
        key: str,
        *,
        category: str | None = None,
        search: str = "",
        start_ms: float = 0,
        end_ms: float | None = None,
        offset: int = 0,
        limit: int = 200,
    ) -> dict:
        with self.lock:
            self._load(key)
            rows = [
                row
                for row in self._events
                if (category is None or row["category"] == category)
                and search.casefold() in row["name"].casefold()
                and row["start_ms"] + row["duration_ms"] >= start_ms
                and (end_ms is None or row["start_ms"] <= end_ms)
            ]
            return {
                "items": rows[offset : offset + limit],
                "total": len(rows),
                "next_offset": offset + limit if offset + limit < len(rows) else None,
                "summary": dict(self._summary),
            }
