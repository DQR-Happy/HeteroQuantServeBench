"""Real PyTorch custom-operator bridge used by the S06 Jetson experiments.

This module is intentionally separate from the pure contract models in
``hqsb.integration.dispatch`` and ``hqsb.integration.meta``.  It owns the
actual ``torch.library`` schemas and calls the already-audited S03 CUDA C ABI
on the caller's current stream.  Importing the module does not register an
operator or load CUDA; callers must invoke :func:`register_torch_ops`.

The public operators are functional and inference-only::

    hqsb::rms_norm(Tensor x, Tensor weight, float eps) -> Tensor
    hqsb::fused_add_rms_norm(
        Tensor x, Tensor residual, Tensor weight, float eps
    ) -> (Tensor normalized, Tensor updated_residual)

Both CUDA implementations allocate their outputs through PyTorch and launch
the native kernel on ``torch.cuda.current_stream``.  The native libraries do
not allocate or synchronize internally.
"""

from __future__ import annotations

import ctypes
import glob
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_RMS_NORM = "rms_norm(Tensor x, Tensor weight, float eps) -> Tensor"
SCHEMA_FUSED_ADD_RMS_NORM = (
    "fused_add_rms_norm(Tensor x, Tensor residual, Tensor weight, float eps) "
    "-> (Tensor normalized, Tensor updated_residual)"
)


class TorchOperatorError(RuntimeError):
    """A stable pre-launch validation or native launch failure."""

    def __init__(self, reason: str, message: str, *, stage: str = "dispatch"):
        super().__init__(f"{reason}: {message}")
        self.reason = reason
        self.stage = stage


@dataclass
class TorchOpAudit:
    """Process-local audit counters; values are evidence, not pass criteria."""

    registered: bool = False
    registration_calls: int = 0
    schema_owner: str = ""
    rms_library: str = ""
    fused_library: str = ""
    calls: Dict[str, int] = field(
        default_factory=lambda: {
            "cpu_rms_norm": 0,
            "cuda_rms_norm": 0,
            "meta_rms_norm": 0,
            "cpu_fused_add_rms_norm": 0,
            "cuda_fused_add_rms_norm": 0,
            "meta_fused_add_rms_norm": 0,
        }
    )
    launches: List[Dict[str, Any]] = field(default_factory=list)
    launch_count: int = 0
    launches_dropped: int = 0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "registered": self.registered,
            "registration_calls": self.registration_calls,
            "schema_owner": self.schema_owner,
            "rms_library": self.rms_library,
            "fused_library": self.fused_library,
            "calls": dict(self.calls),
            "launches": list(self.launches),
            "launch_count": self.launch_count,
            "launches_dropped": self.launches_dropped,
        }


_AUDIT = TorchOpAudit()
_LOCK = threading.Lock()
_LIBRARIES: List[Any] = []
_NATIVE: Optional["NativeKernels"] = None


def audit_snapshot() -> Dict[str, Any]:
    """Return a detached snapshot of real registration/dispatch activity."""

    return _AUDIT.snapshot()


def reset_audit_counters() -> None:
    """Clear call/launch counters while preserving registration identity."""

    for key in _AUDIT.calls:
        _AUDIT.calls[key] = 0
    _AUDIT.launches.clear()
    _AUDIT.launch_count = 0
    _AUDIT.launches_dropped = 0


def _record_launch(record: Dict[str, Any]) -> None:
    """Keep exact totals and a bounded recent sample for long lifecycle runs."""

    _AUDIT.launch_count += 1
    if len(_AUDIT.launches) >= 1000:
        _AUDIT.launches.pop(0)
        _AUDIT.launches_dropped += 1
    _AUDIT.launches.append(record)


def _find_library(root: Path, explicit: Optional[str], pattern: str) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(Path(item) for item in sorted(glob.glob(str(root / pattern))))
    preferred = [path for path in candidates if "final-canonical" in str(path)]
    for path in preferred + candidates:
        if path.is_file():
            return path.resolve()
    raise TorchOperatorError(
        "LIBRARY_NOT_FOUND",
        f"no native library matched {pattern!r} under {root}",
        stage="load",
    )


class NativeKernels:
    """Strict ctypes binding for the S03 stream-aware CUDA C ABI."""

    def __init__(
        self,
        root: Path,
        *,
        rms_library: Optional[str] = None,
        fused_library: Optional[str] = None,
    ) -> None:
        self.rms_path = _find_library(
            root,
            rms_library,
            "build/*/ops/cuda/rmsnorm/libhqsb_rmsnorm_shared.so",
        )
        self.fused_path = _find_library(
            root,
            fused_library,
            "build/*/ops/cuda/fused_residual_rmsnorm/"
            "libhqsb_fused_residual_rmsnorm_shared.so",
        )
        ptr = ctypes.c_void_p
        i64 = ctypes.c_longlong
        self.rms_lib = ctypes.CDLL(str(self.rms_path))
        self.fused_lib = ctypes.CDLL(str(self.fused_path))
        self.rms = self.rms_lib.hqsb_rmsnorm_forward_ex_c
        self.rms.argtypes = [
            ptr,
            ptr,
            ptr,
            i64,
            i64,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ptr,
        ]
        self.rms.restype = ctypes.c_int
        self.fused = self.fused_lib.hqsb_fused_residual_rmsnorm_forward_ex_c
        self.fused.argtypes = [
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
            i64,
            i64,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ptr,
        ]
        self.fused.restype = ctypes.c_int

    @staticmethod
    def _dtype_code(dtype: Any) -> int:
        import torch

        if dtype == torch.float32:
            return 0
        if dtype == torch.float16:
            return 1
        raise TorchOperatorError("DTYPE_UNSUPPORTED", f"unsupported dtype {dtype}")

    def launch_rms_norm(self, x: Any, weight: Any, output: Any, eps: float) -> None:
        import torch

        rows = x.numel() // x.shape[-1]
        hidden = x.shape[-1]
        stream = torch.cuda.current_stream(x.device)
        code = self._dtype_code(x.dtype)
        rc = int(
            self.rms(
                x.data_ptr(),
                weight.data_ptr(),
                output.data_ptr(),
                rows,
                hidden,
                float(eps),
                code,
                0,
                ctypes.c_void_p(int(stream.cuda_stream)),
            )
        )
        _record_launch(
            {
                "op": "hqsb::rms_norm",
                "kernel_symbol": "hqsb_rmsnorm_forward_ex_c",
                "rows": int(rows),
                "hidden": int(hidden),
                "dtype": str(x.dtype),
                "device": str(x.device),
                "stream": int(stream.cuda_stream),
                "return_code": rc,
            }
        )
        if rc:
            raise TorchOperatorError("CUDA_LAUNCH_ERROR", f"native RMSNorm returned {rc}")

    def launch_fused(
        self,
        x: Any,
        residual: Any,
        weight: Any,
        normalized: Any,
        updated: Any,
        eps: float,
    ) -> None:
        import torch

        rows = x.numel() // x.shape[-1]
        hidden = x.shape[-1]
        stream = torch.cuda.current_stream(x.device)
        code = self._dtype_code(x.dtype)
        rc = int(
            self.fused(
                x.data_ptr(),
                residual.data_ptr(),
                weight.data_ptr(),
                updated.data_ptr(),
                normalized.data_ptr(),
                rows,
                hidden,
                float(eps),
                code,
                0,
                1,
                ctypes.c_void_p(int(stream.cuda_stream)),
            )
        )
        _record_launch(
            {
                "op": "hqsb::fused_add_rms_norm",
                "kernel_symbol": "hqsb_fused_residual_rmsnorm_forward_ex_c",
                "rows": int(rows),
                "hidden": int(hidden),
                "dtype": str(x.dtype),
                "device": str(x.device),
                "stream": int(stream.cuda_stream),
                "return_code": rc,
            }
        )
        if rc:
            raise TorchOperatorError(
                "CUDA_LAUNCH_ERROR", f"native fused add+RMSNorm returned {rc}"
            )


def _validate_common(x: Any, weight: Any, eps: float) -> None:
    import torch

    if x.dtype not in (torch.float16, torch.float32):
        raise TorchOperatorError("DTYPE_UNSUPPORTED", f"x.dtype={x.dtype}")
    if weight.dtype != x.dtype:
        raise TorchOperatorError(
            "DTYPE_MISMATCH", f"weight.dtype={weight.dtype}, x.dtype={x.dtype}"
        )
    if x.ndim < 2:
        raise TorchOperatorError("RANK_UNSUPPORTED", f"rank={x.ndim}")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise TorchOperatorError(
            "SHAPE_MISMATCH", f"x={tuple(x.shape)}, weight={tuple(weight.shape)}"
        )
    if not x.is_contiguous() or not weight.is_contiguous():
        raise TorchOperatorError(
            "LAYOUT_UNSUPPORTED",
            f"x_stride={tuple(x.stride())}, weight_stride={tuple(weight.stride())}",
        )
    if not eps > 0:
        raise TorchOperatorError("EPSILON_INVALID", f"eps={eps}")
    if x.shape[-1] > 8192:
        raise TorchOperatorError("HIDDEN_UNSUPPORTED", f"hidden={x.shape[-1]}")
    if x.requires_grad or weight.requires_grad:
        raise TorchOperatorError("AUTOGRAD_UNSUPPORTED", "S06 operators are inference-only")


def _rms_reference(x: Any, weight: Any, eps: float) -> Any:
    import torch

    _validate_common(x, weight, eps)
    value = x.float()
    inv = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + float(eps))
    return (value * inv * weight.float()).to(x.dtype)


def _fused_reference(x: Any, residual: Any, weight: Any, eps: float) -> Tuple[Any, Any]:
    _validate_common(x, weight, eps)
    if residual.device != x.device or residual.dtype != x.dtype:
        raise TorchOperatorError(
            "INPUT_MISMATCH",
            f"residual device/dtype={residual.device}/{residual.dtype}, "
            f"x={x.device}/{x.dtype}",
        )
    if residual.shape != x.shape:
        raise TorchOperatorError(
            "SHAPE_MISMATCH", f"x={tuple(x.shape)}, residual={tuple(residual.shape)}"
        )
    if not residual.is_contiguous():
        raise TorchOperatorError("LAYOUT_UNSUPPORTED", f"residual_stride={residual.stride()}")
    if residual.requires_grad:
        raise TorchOperatorError("AUTOGRAD_UNSUPPORTED", "S06 operators are inference-only")
    updated = (x.float() + residual.float()).to(x.dtype)
    normalized = _rms_reference(updated, weight, eps)
    return normalized, updated


def _cpu_rms_norm(x: Any, weight: Any, eps: float) -> Any:
    _AUDIT.calls["cpu_rms_norm"] += 1
    return _rms_reference(x, weight, eps)


def _cpu_fused(x: Any, residual: Any, weight: Any, eps: float) -> Tuple[Any, Any]:
    _AUDIT.calls["cpu_fused_add_rms_norm"] += 1
    return _fused_reference(x, residual, weight, eps)


def _cuda_rms_norm(x: Any, weight: Any, eps: float) -> Any:
    import torch

    _validate_common(x, weight, eps)
    if x.device.type != "cuda" or weight.device != x.device:
        raise TorchOperatorError("DEVICE_MISMATCH", f"x={x.device}, weight={weight.device}")
    assert _NATIVE is not None
    output = torch.empty_like(x)
    _NATIVE.launch_rms_norm(x, weight, output, eps)
    _AUDIT.calls["cuda_rms_norm"] += 1
    return output


def _cuda_fused(x: Any, residual: Any, weight: Any, eps: float) -> Tuple[Any, Any]:
    import torch

    _validate_common(x, weight, eps)
    if residual.device != x.device or weight.device != x.device:
        raise TorchOperatorError(
            "DEVICE_MISMATCH", f"x={x.device}, residual={residual.device}, weight={weight.device}"
        )
    if residual.dtype != x.dtype or residual.shape != x.shape or not residual.is_contiguous():
        raise TorchOperatorError(
            "INPUT_MISMATCH",
            f"x={tuple(x.shape)}/{x.dtype}/{x.stride()}, "
            f"residual={tuple(residual.shape)}/{residual.dtype}/{residual.stride()}",
        )
    if residual.requires_grad:
        raise TorchOperatorError("AUTOGRAD_UNSUPPORTED", "S06 operators are inference-only")
    assert _NATIVE is not None
    normalized = torch.empty_like(x)
    updated = torch.empty_like(x)
    _NATIVE.launch_fused(x, residual, weight, normalized, updated, eps)
    _AUDIT.calls["cuda_fused_add_rms_norm"] += 1
    return normalized, updated


def _meta_rms_norm(x: Any, weight: Any, eps: float) -> Any:
    import torch

    _validate_common(x, weight, eps)
    _AUDIT.calls["meta_rms_norm"] += 1
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _meta_fused(x: Any, residual: Any, weight: Any, eps: float) -> Tuple[Any, Any]:
    import torch

    _validate_common(x, weight, eps)
    if residual.shape != x.shape or residual.dtype != x.dtype or residual.device != x.device:
        raise TorchOperatorError("INPUT_MISMATCH", "residual metadata differs from x")
    if not residual.is_contiguous():
        raise TorchOperatorError("LAYOUT_UNSUPPORTED", f"residual_stride={residual.stride()}")
    _AUDIT.calls["meta_fused_add_rms_norm"] += 1
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty_like(x, memory_format=torch.contiguous_format),
    )


def register_torch_ops(
    repo_root: str | os.PathLike[str],
    *,
    rms_library: Optional[str] = None,
    fused_library: Optional[str] = None,
) -> TorchOpAudit:
    """Register the real S06 schemas and CPU/CUDA/Meta implementations.

    Repeated calls in the same process are idempotent.  A different schema
    owner is still rejected by PyTorch, which the E06-01 conflict probe tests
    in an isolated child process.
    """

    global _NATIVE
    import torch

    with _LOCK:
        _AUDIT.registration_calls += 1
        if _AUDIT.registered:
            return _AUDIT
        root = Path(repo_root).resolve()
        _NATIVE = NativeKernels(
            root, rms_library=rms_library, fused_library=fused_library
        )
        definition = torch.library.Library("hqsb", "DEF")
        definition.define(SCHEMA_RMS_NORM)
        definition.define(SCHEMA_FUSED_ADD_RMS_NORM)
        implementation = torch.library.Library("hqsb", "IMPL")
        implementation.impl("rms_norm", _cpu_rms_norm, "CPU")
        implementation.impl("rms_norm", _cuda_rms_norm, "CUDA")
        implementation.impl("rms_norm", _meta_rms_norm, "Meta")
        implementation.impl("fused_add_rms_norm", _cpu_fused, "CPU")
        implementation.impl("fused_add_rms_norm", _cuda_fused, "CUDA")
        implementation.impl("fused_add_rms_norm", _meta_fused, "Meta")
        _LIBRARIES.extend((definition, implementation))
        _AUDIT.registered = True
        _AUDIT.schema_owner = __file__
        _AUDIT.rms_library = str(_NATIVE.rms_path)
        _AUDIT.fused_library = str(_NATIVE.fused_path)
        return _AUDIT


def dispatch_tables() -> Dict[str, str]:
    """Return PyTorch's actual dispatcher table for both registered ops."""

    import torch

    return {
        "hqsb::rms_norm": torch._C._dispatch_dump_table("hqsb::rms_norm"),
        "hqsb::fused_add_rms_norm": torch._C._dispatch_dump_table(
            "hqsb::fused_add_rms_norm"
        ),
    }


__all__ = [
    "SCHEMA_RMS_NORM",
    "SCHEMA_FUSED_ADD_RMS_NORM",
    "TorchOperatorError",
    "TorchOpAudit",
    "NativeKernels",
    "audit_snapshot",
    "reset_audit_counters",
    "register_torch_ops",
    "dispatch_tables",
]
