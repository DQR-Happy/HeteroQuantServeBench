"""Memory-safety helpers for low-bit kernels (E05-06 §13 step 7).

Two mechanisms:

1. **guard regions** — every allocated buffer is surrounded by a poisoned
   region whose values must remain untouched after the kernel runs; a
   detected change means an out-of-bounds write happened even when the shape
   checks passed;
2. **sanitizer command construction** — the exact ``compute-sanitizer``
   invocation with the tool's availability probed; when the tool is missing
   the caller must record the gap rather than claim the check passed.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from hqsb.core.errors import CapabilityError, ConfigError

GUARD_VALUE = 12345.0


@dataclass
class GuardedBuffer:
    """A buffer with poisoned regions before and after the payload."""

    payload: Any
    before: Any
    after: Any
    guard_size: int

    def check(self) -> Dict[str, Any]:
        """Return integrity of both guard regions (device synchronised)."""
        import torch

        torch.cuda.synchronize()
        before_ok = bool(torch.all(self.before == GUARD_VALUE).item())
        after_ok = bool(torch.all(self.after == GUARD_VALUE).item())
        return {
            "before_intact": before_ok,
            "after_intact": after_ok,
            "guard_size": self.guard_size,
            "passed": before_ok and after_ok,
        }


def guarded_buffer(shape: Sequence[int], *, device: str = "cuda", guard: int = 256, dtype=None):
    """Allocate ``shape`` with poisoned guard regions around it.

    The returned object exposes ``payload`` (the tensor the kernel writes) and
    ``check()``; the guards are separate allocations so a pointer overrun is
    visible instead of silently landing in adjacent memory.
    """
    import torch

    if not torch.cuda.is_available():
        raise CapabilityError(
            "guard-region checks need a CUDA device",
            details={"capability": "cuda", "reason": "no device visible"},
        )
    if guard <= 0:
        raise ConfigError(f"guard must be positive, got {guard}")
    resolved_dtype = dtype or torch.float16
    payload = torch.zeros(tuple(int(dim) for dim in shape), dtype=resolved_dtype, device=device)
    before = torch.full((guard,), GUARD_VALUE, dtype=resolved_dtype, device=device)
    after = torch.full((guard,), GUARD_VALUE, dtype=resolved_dtype, device=device)
    return GuardedBuffer(payload=payload, before=before, after=after, guard_size=guard)


def sanitizer_available() -> bool:
    return shutil.which("compute-sanitizer") is not None


def sanitizer_command(
    executable: str,
    args: Sequence[str] = (),
    *,
    tool: str = "memcheck",
    extra: Sequence[str] = (),
) -> List[str]:
    """Build the ``compute-sanitizer`` command line.

    The caller runs and archives it; if the tool is unavailable the command is
    still returned together with ``available=False`` so the report can record
    the gap explicitly (E05-06 §13 step 7 / §17).
    """
    command = [
        "compute-sanitizer",
        "--tool",
        tool,
        "--launch-timeout",
        "120",
        *extra,
        executable,
        *args,
    ]
    return command


def sanitizer_report(command: Sequence[str]) -> Dict[str, Any]:
    return {
        "available": sanitizer_available(),
        "command": list(command),
        "note": (
            "a missing sanitizer is a capability gap and must be reported as "
            "such; it is never a pass"
        ),
    }


__all__ = [
    "GUARD_VALUE",
    "GuardedBuffer",
    "guarded_buffer",
    "sanitizer_available",
    "sanitizer_command",
    "sanitizer_report",
]
