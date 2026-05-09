"""ctypes bridge to the S03 CUDA RMSNorm shared library.

This is a *thin* binding: it exposes the S03 C++ kernels to Python without
pybind11 or a torch extension, so the S04 unified dispatcher can select the
hand-written CUDA implementation on an equal footing with Triton. The
contract mirrors the C ABI in ``ops/cuda/rmsnorm/src/rmsnorm_c_api.cu``.

The bridge never imports triton or torch at module scope; importing
``ops.cuda_bridge`` is always safe (even on CPU-only machines).
"""

from __future__ import annotations

import ctypes
from typing import Optional

from ops.capability import detect_capabilities


class CudaRmsnormUnavailable(RuntimeError):
    """Raised when the CUDA RMSNorm shared library cannot be loaded."""


class _CudaRmsnormBridge:
    """Loads and caches the shared library, exposing ``forward``."""

    def __init__(self) -> None:
        self._lib = None

    def _ensure_loaded(self):
        if self._lib is not None:
            return self._lib

        capabilities = detect_capabilities()
        if not capabilities.cuda_rmsnorm_available:
            raise CudaRmsnormUnavailable(
                "CUDA RMSNorm shared library not available; "
                "build it with `cmake --build build/jetson-release` "
                "(or set HQSB_CUDA_RMSNORM_LIB)"
            )

        lib = ctypes.CDLL(capabilities.cuda_rmsnorm_lib)
        fn = lib.hqsb_rmsnorm_forward_c
        fn.argtypes = [
            ctypes.c_void_p,   # input
            ctypes.c_void_p,   # weight
            ctypes.c_void_p,   # output
            ctypes.c_longlong,  # rows
            ctypes.c_longlong,  # hidden
            ctypes.c_float,    # epsilon
            ctypes.c_int,      # dtype: 0=fp32, 1=fp16
            ctypes.c_int,      # variant: 0=auto, 2=v0, 3=v1, 4=v2
        ]
        fn.restype = ctypes.c_int
        self._lib = lib
        return lib

    def forward(
        self,
        x,
        weight,
        *,
        dtype: str = "fp32",
        variant: int = 0,
        epsilon: float = 1e-5,
        out=None,
    ):
        """Run RMSNorm on CUDA tensors via the hand-written kernels.

        Args:
            x: ``(rows, hidden)`` contiguous CUDA tensor.
            weight: ``(hidden,)`` contiguous CUDA tensor.
            dtype: ``"fp32"`` or ``"fp16"``.
            variant: 0=auto, 2=v0, 3=v1, 4=v2 (see C ABI).
            epsilon: denominator stabilizer.
            out: Optional pre-allocated output tensor; else allocated.

        Returns:
            The output tensor ``(rows, hidden)``.
        """
        import torch

        lib = self._ensure_loaded()
        rows, hidden = x.shape[0], x.shape[1]

        if out is None:
            out = torch.empty_like(x)
        else:
            if out.shape != x.shape:
                raise ValueError(f"out shape {tuple(out.shape)} != x shape {tuple(x.shape)}")

        dtype_code = 1 if dtype == "fp16" else 0
        err = lib.hqsb_rmsnorm_forward_c(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            rows,
            hidden,
            ctypes.c_float(epsilon),
            dtype_code,
            variant,
        )
        if err != 0:
            raise RuntimeError(f"hqsb_rmsnorm_forward_c failed with cudaError={err}")
        return out


# Module-level singleton (lazy-loaded).
_bridge = _CudaRmsnormBridge()


def rmsnorm_forward(x, weight, **kwargs):
    """Convenience wrapper around :class:`_CudaRmsnormBridge.forward`."""
    return _bridge.forward(x, weight, **kwargs)


__all__ = [
    "CudaRmsnormUnavailable",
    "rmsnorm_forward",
]
