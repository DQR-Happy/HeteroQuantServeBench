"""Device-event timing harness for low-bit GEMM microbenchmarks (E05-06 §10.1).

Requirements the harness enforces:

* pre-allocated inputs/outputs/workspace (allocation is not part of a kernel
  measurement);
* warmup until compilation and clocks stabilise, reported separately from
  steady state;
* device-event timing with an explicit synchronisation boundary;
* the output is *consumed* (checksummed) so dead-code elimination cannot
  report a fast empty kernel;
* launch count and workspace bytes are recorded per shape;
* median, p5/p95 and a 95% confidence interval over the raw samples.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from hqsb.core.errors import ConfigError
from hqsb.benchmark.metrics import latency_summary, percentile


@dataclass
class BenchmarkSample:
    """One timed repetition."""

    index: int
    latency_ms: float
    warmup: bool
    checksum: float
    process_id: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "latency_ms": self.latency_ms,
            "warmup": self.warmup,
            "checksum": self.checksum,
            "process_id": self.process_id,
        }


@dataclass
class MicrobenchResult:
    """Raw samples plus the summary for one measured callable."""

    name: str
    samples: List[BenchmarkSample] = field(default_factory=list)
    warmup_samples: List[BenchmarkSample] = field(default_factory=list)
    launch_count: int = 1
    workspace_bytes: int = 0
    output_consumed: bool = False
    notes: str = ""

    def summary(self) -> Dict[str, Any]:
        latencies = [sample.latency_ms for sample in self.samples]
        if not latencies:
            return {"count": 0}
        ordered = sorted(latencies)
        n = len(ordered)
        mean = statistics.fmean(latencies)
        stdev = statistics.pstdev(latencies)
        half_width = 1.96 * stdev / math.sqrt(n) if n > 1 else float("nan")
        summary = latency_summary(latencies)
        summary.update(
            {
                "ci95_low_ms": mean - half_width if n > 1 else float("nan"),
                "ci95_high_ms": mean + half_width if n > 1 else float("nan"),
                "p5_ms": percentile(latencies, 0.05),
                "relative_stdev": stdev / mean if mean else float("nan"),
                "process_count": len({sample.process_id for sample in self.samples}),
            }
        )
        return summary

    def first_call_ms(self) -> Optional[float]:
        """Cold/first-call latency (JIT/autotune), reported separately."""
        if not self.warmup_samples:
            return None
        return self.warmup_samples[0].latency_ms

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary(),
            "first_call_ms": self.first_call_ms(),
            "warmup_samples": [sample.as_dict() for sample in self.warmup_samples],
            "raw": [sample.as_dict() for sample in self.samples],
            "launch_count": self.launch_count,
            "workspace_bytes": self.workspace_bytes,
            "output_consumed": self.output_consumed,
            "notes": self.notes,
        }


def _checksum(output) -> float:
    """Consume the output so a dead kernel cannot look fast.

    A callable may return a tensor, a scalar or ``None``; the checksum reduces
    each to a float so the harness can prove the output was actually produced
    (and would flag a kernel that elided the store).
    """
    if output is None:
        return float("nan")
    if hasattr(output, "detach"):
        flat = output.detach().reshape(-1)
        if flat.numel() == 0:
            return 0.0
        return float(flat[: min(1024, flat.numel())].float().sum().item())
    if isinstance(output, (int, float)):
        return float(output)
    try:
        return float(sum(output))
    except TypeError:  # pragma: no cover - a non-numeric callable output
        return float("nan")


def benchmark_callable(
    fn: Callable[[], Any],
    *,
    name: str,
    warmup: int = 5,
    repeats: int = 20,
    use_cuda_events: bool = True,
    workspace_bytes: int = 0,
    launch_count: int = 1,
) -> MicrobenchResult:
    """Time ``fn`` with warmup, repeats and a synchronised boundary.

    ``use_cuda_events`` requires CUDA; on a CPU-only host the host clock is
    used and the result notes that the measurement is not device time.
    """
    if warmup < 0 or repeats <= 0:
        raise ConfigError(
            f"warmup must be >= 0 and repeats > 0, got {warmup}/{repeats}"
        )
    import os

    pid = os.getpid()
    result = MicrobenchResult(name=name, workspace_bytes=workspace_bytes, launch_count=launch_count)
    torch = None
    events = False
    if use_cuda_events:
        try:
            import torch as _torch

            events = bool(_torch.cuda.is_available())
            torch = _torch
        except ImportError:
            events = False
    for index in range(warmup):
        if events:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = fn()
            end.record()
            torch.cuda.synchronize()
            latency_ms = float(start.elapsed_time(end))
        else:
            started = time.perf_counter()
            output = fn()
            latency_ms = (time.perf_counter() - started) * 1000.0
        result.warmup_samples.append(
            BenchmarkSample(
                index=index,
                latency_ms=latency_ms,
                warmup=True,
                checksum=_checksum(output),
                process_id=pid,
            )
        )
    for index in range(repeats):
        if events:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = fn()
            end.record()
            torch.cuda.synchronize()
            latency_ms = float(start.elapsed_time(end))
        else:
            if torch is not None:
                torch.cuda.synchronize()
            started = time.perf_counter()
            output = fn()
            latency_ms = (time.perf_counter() - started) * 1000.0
        result.samples.append(
            BenchmarkSample(
                index=index,
                latency_ms=latency_ms,
                warmup=False,
                checksum=_checksum(output),
                process_id=pid,
            )
        )
    result.output_consumed = all(
        not math.isnan(sample.checksum) for sample in result.samples
    )
    if not events:
        result.notes = (
            "measured with the host clock (no CUDA events available); this is "
            "not device time and must be labelled accordingly"
        )
    return result


def dequant_decomposition_plan() -> List[Dict[str, str]]:
    """The six required dequant decomposition paths (E05-06 §12).

    Returned as a plan (name → what it isolates) so the driver cannot silently
    skip a control: ``unpack_only`` may be unavailable without a separate
    kernel, and in that case the report must say so instead of inventing a
    precise split (E05-06 §12 last paragraph).
    """
    return [
        {"path": "unpack_only", "isolates": "bit unpack + sign extension"},
        {"path": "dequant_to_buffer", "isolates": "unpack + scale/zero apply + write"},
        {"path": "dequant_to_buffer_plus_fp16_gemm", "isolates": "materialization + launch + GEMM"},
        {"path": "fused_dequant_gemm", "isolates": "fused kernel (HQSB)"},
        {"path": "vendor_fused", "isolates": "vendor fused kernel (if available)"},
        {"path": "fp16_gemm", "isolates": "FP16 GEMM baseline"},
    ]


def workspace_report(prepared) -> Dict[str, Any]:
    """Report workspace/launch accounting for a prepared kernel run."""
    return {
        "packed_bytes": int(prepared.packed.numel()),
        "scale_bytes": int(prepared.scales.numel() * prepared.scales.element_size()),
        "zero_bytes": (
            int(prepared.zeros.numel() * prepared.zeros.element_size())
            if prepared.zeros is not None
            else 0
        ),
        "row_stride_bytes": prepared.row_stride_bytes,
        "groups_per_row": prepared.groups_per_row,
        "group_size": prepared.group_size,
        "bits": prepared.bits,
        "launch_count": 1,
        "note": "the kernel allocates only the output; no workspace buffer is used",
    }


__all__ = [
    "BenchmarkSample",
    "MicrobenchResult",
    "benchmark_callable",
    "dequant_decomposition_plan",
    "workspace_report",
]
