"""Cold-load / first-request / warm-steady / KV-reset cost separation (E02-06).

E02-06 answers: *when an inference process is slow "the first time", which of
process launch, framework import, artifact verification, device init, weight
load, first-shape initialization, or KV state is responsible?* (see
``docs/stage_experiments/details/S02/E02-06_cold_warm_and_cache_states.md``.)

This module keeps the **pure, torch-free** half of that experiment so it is
unit-testable without a GPU or model weights:

* :class:`EventTimeline` — a host-monotonic event recorder with a single,
  documented clock domain (protocol §5). No event name may be recorded twice,
  so a silently overwritten timestamp cannot pass as evidence.
* :func:`read_proc_io` / :func:`read_uptime_seconds` /
  :func:`read_process_start_ticks` / :func:`process_start_monotonic_ns` — place
  the **process start** on the same monotonic clock as the in-process events,
  which is what lets "startup to model ready" include interpreter launch
  instead of quietly measuring "after import" (protocol §8 step 1).
* :func:`derive_windows` / :func:`startup_attribution` — the named cost windows
  and a **non-overlapping** attribution whose parts sum to the startup total,
  so sub-spans are never double-counted (protocol §5).
* :func:`validate_event_ordering` — the causal-order check that turns a broken
  timeline into an evidence error rather than a plausible number.
* :func:`compilation_evidence` — records *whether* a compiler path was enabled
  and, when it was not, states that explicitly instead of reporting a fake
  "compile cost = 0" (protocol §3.2 / §9 step 9).
* :func:`kv_reset_verdict` — the state-isolation verdict from measured cache
  lengths and an independent reference, including the bounded negative control
  (protocol §8 step 7).
* :func:`stability_reached` — the pre-registered steady-state rule (rolling
  window + minimum warmups + maximum wait), so "steady" is a criterion rather
  than "the samples we decided to keep".

Every duration is stored in **milliseconds** and every size in **bytes**;
presentation converts and always names the unit (protocol §5).
"""

from __future__ import annotations

import os
import statistics
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

MIB = 1024**2

__all__ = [
    "MIB",
    "EventTimeline",
    "bytes_to_mib",
    "compilation_evidence",
    "derive_windows",
    "kv_reset_verdict",
    "process_start_monotonic_ns",
    "read_meminfo_fields",
    "read_proc_io",
    "read_process_start_ticks",
    "read_uptime_seconds",
    "stability_reached",
    "startup_attribution",
    "summarize_series",
    "validate_event_ordering",
]

# ───────────────────────────── units ─────────────────────────────


def bytes_to_mib(nbytes: float) -> float:
    """Convert bytes to **MiB** (binary, 2^20); the unit is always binary here."""
    return float(nbytes) / MIB


# ───────────────────────────── event timeline ─────────────────────────────


class EventTimeline:
    """Host-monotonic event recorder for the E02-06 cost windows.

    All timestamps come from :func:`time.monotonic_ns`, i.e. one clock domain
    (protocol §5: "启动墙钟与 CUDA event 不混合相减"). Recording the same name
    twice raises, because a duplicate is almost always a copy/paste bug that
    would otherwise silently replace the real timestamp.
    """

    def __init__(self) -> None:
        self._events: Dict[str, int] = {}
        self._order: List[str] = []

    def record(self, name: str, ns: Optional[int] = None) -> int:
        """Record an event and return its timestamp (ns, monotonic domain)."""
        if name in self._events:
            raise ValueError(f"duplicate event name: {name!r}")
        value = int(time.monotonic_ns() if ns is None else ns)
        self._events[name] = value
        self._order.append(name)
        return value

    def get(self, name: str) -> Optional[int]:
        """Timestamp of ``name`` or ``None`` when it was never recorded."""
        return self._events.get(name)

    def span_ms(self, begin: str, end: str) -> Optional[float]:
        """Duration ``end - begin`` in ms, or ``None`` if either is missing."""
        return _span_ms(self._events, begin, end)

    @property
    def events(self) -> Dict[str, int]:
        """Copy of the recorded ``{name: monotonic_ns}`` mapping."""
        return dict(self._events)

    @property
    def names(self) -> List[str]:
        """Event names in record order."""
        return list(self._order)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-serializable view of the timeline."""
        return {"events": dict(self._events), "order": list(self._order)}


# ───────────────────────── process-clock helpers ─────────────────────────


def read_uptime_seconds(path: str = "/proc/uptime") -> Optional[float]:
    """Read the first field of ``/proc/uptime`` (seconds since boot)."""
    try:
        with open(path, encoding="utf-8") as fh:
            return float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def read_process_start_ticks(
    pid: Optional[int] = None, stat_path: Optional[str] = None
) -> Optional[int]:
    """Read field 22 (``starttime``) of ``/proc/<pid>/stat`` in clock ticks.

    The ``comm`` field may contain spaces and parentheses, so parsing starts
    after the **last** ``)``; index 19 of the remainder is ``starttime``
    (field 22 overall, field 3 is ``state``).
    """
    path = stat_path or f"/proc/{os.getpid() if pid is None else pid}/stat"
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 1 :].split()
    # fields[0] == state == field 3 -> starttime (field 22) is index 19.
    if len(fields) <= 19:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def process_start_monotonic_ns(
    now_mono_ns: int,
    uptime_s: Optional[float],
    starttime_ticks: Optional[int],
    clk_tck: Optional[float],
) -> Optional[int]:
    """Place process start on the ``monotonic`` clock.

    ``/proc`` reports process start as clock ticks since boot while
    :func:`time.monotonic_ns` is also boot-relative on Linux, so::

        start_ns = now_mono_ns - (uptime_s - starttime_ticks / clk_tck) * 1e9

    Returns ``None`` when any input is unavailable (e.g. non-Linux), so the
    caller can fall back to an explicit "after-import" measurement instead of
    fabricating a process-start time.
    """
    if uptime_s is None or starttime_ticks is None or not clk_tck:
        return None
    elapsed_s = uptime_s - (starttime_ticks / clk_tck)
    if elapsed_s < 0:
        return None
    return int(now_mono_ns - elapsed_s * 1e9)


def read_proc_io(path: str = "/proc/self/io") -> Dict[str, int]:
    """Read ``/proc/self/io`` counters (bytes), or ``{}`` when unavailable.

    ``rchar`` counts bytes returned by read syscalls (page-cache hits count),
    while ``read_bytes`` counts bytes actually fetched from the storage layer.
    Their difference is the page-cache-hit evidence E02-06 needs: after the
    artifact gate has read every weight file, the loader's ``read_bytes``
    delta should collapse towards zero (protocol §4).
    """
    counters: Dict[str, int] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                value = value.strip()
                if value:
                    try:
                        counters[key.strip()] = int(value)
                    except ValueError:
                        continue
    except OSError:
        return {}
    return counters


# ───────────────────────── cost windows ─────────────────────────


def read_meminfo_fields(
    fields: Sequence[str] = (
        "MemTotal",
        "MemFree",
        "MemAvailable",
        "Buffers",
        "Cached",
        "SwapTotal",
        "SwapFree",
        "Shmem",
    ),
    path: str = "/proc/meminfo",
) -> Dict[str, int]:
    """Read selected ``/proc/meminfo`` fields as **bytes**.

    Used as page-cache evidence when ``/proc/self/io`` is unavailable (this
    Jetson kernel has ``CONFIG_TASK_IO_ACCOUNTING`` off): ``Cached`` grows when
    the artifact gate streams every weight file, and a later loader that does
    not grow ``Cached`` again is reading from that page cache rather than from
    storage (protocol §4).
    """
    wanted = set(fields)
    out: Dict[str, int] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                key = key.strip()
                if key not in wanted:
                    continue
                parts = rest.split()
                if not parts:
                    continue
                try:
                    out[key] = int(parts[0]) * 1024  # meminfo values are kB
                except ValueError:
                    continue
    except OSError:
        return {}
    return out


def _span_ms(events: Mapping[str, int], begin: str, end: str) -> Optional[float]:
    start, stop = events.get(begin), events.get(end)
    if start is None or stop is None:
        return None
    return (stop - start) / 1e6


def derive_windows(events: Mapping[str, int]) -> Dict[str, Optional[float]]:
    """Derive the named E02-06 cost windows (ms) from a raw event mapping.

    The four *formal* cost classes are:

    * ``startup_to_model_ready_ms`` — process launch → weights + device ready;
    * ``first_request_latency_ms`` — first target-shape request (no shape warmup);
    * ``steady_request_latency_ms`` — a warm, per-request-reset request;
    * ``reset_latency_ms`` — clearing request state after a filled KV.

    Everything else is a sub-span used to attribute the startup total.
    ``None`` means the event pair was not measured — never 0.0.
    """
    return {
        # ── sub-spans ──
        "interpreter_startup_ms": _span_ms(events, "process_start", "imports_begin"),
        "framework_import_ms": _span_ms(events, "imports_begin", "imports_ready"),
        "pre_import_ms": _span_ms(events, "process_start", "imports_ready"),
        "monitor_start_ms": _span_ms(events, "monitor_start_begin", "monitor_start_end"),
        "artifact_verify_ms": _span_ms(
            events, "artifact_verify_begin", "artifact_verified"
        ),
        "device_init_ms": _span_ms(events, "device_init_begin", "device_init_end"),
        "load_ms": _span_ms(events, "load_begin", "model_ready"),
        "prep_overhead_ms": _span_ms(events, "imports_ready", "artifact_verify_begin"),
        # ── the four formal costs ──
        "startup_to_model_ready_ms": _span_ms(events, "process_start", "model_ready"),
        "first_request_latency_ms": _span_ms(
            events, "first_request_begin", "first_request_end"
        ),
        "startup_to_first_result_ms": _span_ms(
            events, "process_start", "first_request_end"
        ),
        "steady_request_latency_ms": _span_ms(
            events, "steady_request_begin", "steady_request_end"
        ),
        "reset_latency_ms": _span_ms(events, "cache_reset_begin", "cache_reset_end"),
        "close_ms": _span_ms(events, "close_begin", "close_complete"),
    }


def startup_attribution(
    windows: Mapping[str, Optional[float]]
) -> Dict[str, Any]:
    """Split ``startup_to_model_ready`` into non-overlapping, named parts.

    Parts (all disjoint by construction because each is bounded by consecutive
    events): interpreter launch, framework/import, artifact verification,
    device initialization, weight load. The remainder — monitoring start,
    snapshots, and other prep — is reported as ``unattributed_overhead_ms``
    rather than folded into the loader (protocol §5: sub-spans must not be
    summed and relabelled as the load).
    """
    total = windows.get("startup_to_model_ready_ms")
    named = {
        "interpreter_startup_ms": windows.get("interpreter_startup_ms"),
        "framework_import_ms": windows.get("framework_import_ms"),
        "artifact_verify_ms": windows.get("artifact_verify_ms"),
        "device_init_ms": windows.get("device_init_ms"),
        "load_ms": windows.get("load_ms"),
    }
    known = [v for v in named.values() if v is not None]
    named_total = sum(known) if known else None
    overhead = (
        total - named_total
        if (total is not None and named_total is not None)
        else None
    )
    return {
        "total_startup_to_model_ready_ms": total,
        "parts": {**named, "prep_overhead_ms": windows.get("prep_overhead_ms")},
        "named_parts_total_ms": named_total,
        "unattributed_overhead_ms": overhead,
        "complete": total is not None,
        "non_overlapping": overhead is None or overhead >= -1e-6,
    }


def validate_event_ordering(events: Mapping[str, int]) -> List[str]:
    """Return the causal-order violations of an event mapping.

    The order below is the pre-registered causal chain. A violation is an
    evidence fault (the clock is broken or an event was recorded twice), not a
    result to be smoothed over.
    """
    chain = [
        "process_start",
        "imports_begin",
        "imports_ready",
        "monitor_start_begin",
        "monitor_start_end",
        "artifact_verify_begin",
        "artifact_verified",
        "device_init_begin",
        "device_init_end",
        "load_begin",
        "model_ready",
        "first_request_begin",
        "first_request_end",
        "cache_reset_begin",
        "cache_reset_end",
        "close_begin",
        "close_complete",
    ]
    present = [(name, events[name]) for name in chain if name in events]
    violations: List[str] = []
    for (prev_name, prev_ns), (name, ns) in zip(present, present[1:]):
        if ns < prev_ns:
            violations.append(
                f"ordering violation: {name} ({ns}) < {prev_name} ({prev_ns})"
            )
    return violations


# ───────────────────────── compile evidence ─────────────────────────


def compilation_evidence(
    *,
    enabled: bool,
    backend: Optional[str] = None,
    cache_dir: Optional[str] = None,
    recompile_events: int = 0,
    dynamo_counters: Optional[Mapping[str, Any]] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Describe the compilation path without inventing a cost for it.

    The frozen S02 reference is eager, so ``enabled=False``. The protocol is
    explicit that "not measured" must never be reported as "compile cost = 0"
    (protocol §3.2): the ``statement`` field spells out which case applies.
    """
    if enabled:
        statement = (
            f"compiler path enabled (backend={backend}); "
            f"recompile_events={recompile_events}"
        )
    else:
        statement = (
            "compile not enabled: this is the eager FP16 reference; no compile "
            "cost is claimed (and none is reported as zero)"
        )
    return {
        "enabled": bool(enabled),
        "backend": backend if enabled else None,
        "cache_dir": cache_dir if enabled else None,
        "recompile_events": int(recompile_events) if enabled else 0,
        "dynamo_counters": dict(dynamo_counters) if dynamo_counters else None,
        "statement": statement,
        "notes": notes,
    }


# ───────────────────────── KV reset verdict ─────────────────────────


def kv_reset_verdict(
    *,
    cache_filled_before_reset: Optional[int],
    request_b_filled_after_prefill: Optional[int],
    expected_filled_after_prefill: int,
    request_b_matches_reference: bool,
    negative_control: Mapping[str, Any],
) -> Dict[str, Any]:
    """Verdict for "KV reset truly isolates an independent request".

    An independent request B must (a) start from an empty cache, so its own
    prefill fills exactly ``expected_filled_after_prefill`` positions, and
    (b) reproduce the independent reference. The negative control ("no reset",
    i.e. append B to A's live cache) must be *detected* — otherwise the check
    proves nothing (protocol §8 step 7).
    """
    fresh_start = (
        request_b_filled_after_prefill is not None
        and request_b_filled_after_prefill == expected_filled_after_prefill
    )
    reference_ok = bool(request_b_matches_reference)
    negative_detected = bool(negative_control.get("detected"))
    passed = bool(fresh_start and reference_ok and negative_detected)
    reasons: List[str] = []
    if not fresh_start:
        reasons.append(
            "request B did not start from an empty cache "
            f"(filled={request_b_filled_after_prefill}, "
            f"expected={expected_filled_after_prefill})"
        )
    if not reference_ok:
        reasons.append("request B output differs from the independent reference")
    if not negative_detected:
        reasons.append("negative control (no-reset continuation) was not detected")
    return {
        "cache_filled_before_reset": cache_filled_before_reset,
        "request_b_filled_after_prefill": request_b_filled_after_prefill,
        "expected_filled_after_prefill": expected_filled_after_prefill,
        "request_b_starts_empty": fresh_start,
        "request_b_matches_reference": reference_ok,
        "negative_control_detected": negative_detected,
        "negative_control": dict(negative_control),
        "passed": passed,
        "reasons": reasons,
    }


# ───────────────────────── steady-state rule ─────────────────────────


def stability_reached(
    latencies_ms: Sequence[float],
    *,
    window: int = 3,
    rel_tol: float = 0.10,
    min_samples: int = 3,
) -> Dict[str, Any]:
    """Pre-registered steady-state test over a rolling window of samples.

    Steady means "the last ``window`` samples agree within ``rel_tol``", with a
    minimum sample count; it is *not* "delete the first sample and average the
    rest" (protocol §3.3). ``max_wait`` enforcement lives in the caller.
    """
    values = [float(v) for v in latencies_ms]
    if len(values) < max(min_samples, window):
        return {
            "reached": False,
            "reason": "insufficient_samples",
            "samples": len(values),
            "required": max(min_samples, window),
            "window": [],
            "median_ms": None,
            "spread_ratio": None,
            "rel_tol": rel_tol,
        }
    tail = values[-window:]
    median = statistics.median(tail)
    spread = (max(tail) - min(tail)) / median if median > 0 else float("inf")
    return {
        "reached": spread <= rel_tol,
        "reason": "within_tolerance" if spread <= rel_tol else "still_drifting",
        "samples": len(values),
        "window": tail,
        "median_ms": median,
        "spread_ratio": spread,
        "rel_tol": rel_tol,
    }


def summarize_series(values: Sequence[float]) -> Dict[str, Any]:
    """Aggregate a scalar series into ``{count, min, max, mean, p50, p95}``.

    Empty input yields ``count=0`` with ``None`` statistics — never a fabricated
    peak. Percentiles use linear interpolation (the repo-wide convention).
    """
    from hqsb.benchmark.metrics import percentile

    clean = [float(v) for v in values]
    if not clean:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p95": None,
        }
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": statistics.mean(clean),
        "p50": percentile(clean, 0.50),
        "p95": percentile(clean, 0.95),
    }
