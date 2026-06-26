"""ctypes bridge to the S03 CUDA RMSNorm shared library.

This is a *thin* binding: it exposes the S03 C++ kernels to Python without
pybind11 or a torch extension, so the S04 unified dispatcher can select the
hand-written CUDA implementation on an equal footing with Triton. The
contract mirrors the C ABI in ``ops/cuda/rmsnorm/src/rmsnorm_c_api.cu``.

The bridge never imports triton or torch at module scope; importing
``ops.cuda_bridge`` is always safe (even on CPU-only machines).

Validation policy (E03-01 §8 step 8 / §12.6)
-------------------------------------------
Everything the operator's frozen contract requires is checked *here*, before
the kernel is launched, and a violation raises :class:`RmsNormContractError`
carrying a stable ``reason_code``. Nothing is silently coerced:

* an unknown ``dtype`` string is **not** mapped to ``fp32``;
* a non-contiguous tensor is **not** silently treated as contiguous;
* a mismatched ``weight`` length is **not** ignored;
* a NaN/Inf/non-positive ``epsilon`` is **not** passed through.

The lower C ABI enforces the same policy for dtype/variant/epsilon/shape, so a
caller that bypasses this bridge still cannot get a silent wrong result.
"""

from __future__ import annotations

import ctypes
import math
from typing import Optional

from ops.capability import detect_capabilities

#: dtype string -> C ABI dtype code (mirrors rmsnorm_c_api.cu).
_DTYPE_CODE = {"fp32": 0, "fp16": 1}

#: Stable reason codes for contract violations raised by this bridge.
RMSNORM_REASON_CODES = (
    "UNSUPPORTED_DTYPE",
    "UNSUPPORTED_VARIANT",
    "SHAPE_INVALID",
    "DEVICE_INVALID",
    "NON_CONTIGUOUS_INPUT",
    "DTYPE_MISMATCH",
    "WEIGHT_SHAPE_MISMATCH",
    "EPSILON_INVALID",
    "OUT_TENSOR_MISMATCH",
)

#: C ABI variant codes accepted by ``hqsb_rmsnorm_forward_ex_c``.
_VARIANT_CODE = {
    "auto": 0,
    "reference": 1,
    "v0_shared": 2,
    "v1_warp_shuffle": 3,
    "v2_vectorized": 4,
    "scalar_safe": 5,
    "v2_vectorized_strict": 6,
}


class CudaRmsnormUnavailable(RuntimeError):
    """Raised when the CUDA RMSNorm shared library cannot be loaded."""


class RmsNormContractError(ValueError):
    """Raised when an input violates the frozen RMSNorm operator contract.

    Attributes:
        reason_code: One of :data:`RMSNORM_REASON_CODES`.
        details: Machine-readable context (shape/dtype/stride/... as strings).
    """

    def __init__(self, reason_code: str, message: str, details: Optional[dict] = None):
        if reason_code not in RMSNORM_REASON_CODES:
            raise ValueError(f"unknown reason_code {reason_code!r}")
        super().__init__(f"[{reason_code}] {message}")
        self.reason_code = reason_code
        self.details = dict(details or {})


def _resolve_variant_code(variant) -> int:
    """Map a variant name or numeric code to the C ABI code."""
    if isinstance(variant, str):
        code = _VARIANT_CODE.get(variant)
        if code is None:
            raise RmsNormContractError(
                "UNSUPPORTED_VARIANT",
                f"unknown variant name {variant!r}; "
                f"expected one of {sorted(_VARIANT_CODE)}",
                {"variant": variant},
            )
        return code
    try:
        code = int(variant)
    except (TypeError, ValueError):
        raise RmsNormContractError(
            "UNSUPPORTED_VARIANT",
            f"variant must be an int code or a known name, got {variant!r}",
            {"variant": repr(variant)},
        ) from None
    if code not in _VARIANT_CODE.values():
        raise RmsNormContractError(
            "UNSUPPORTED_VARIANT",
            f"variant code {code} is not in the frozen set "
            f"{sorted(set(_VARIANT_CODE.values()))}",
            {"variant": code},
        )
    return code


class _CudaRmsnormBridge:
    """Loads and caches the shared library, exposing ``forward``."""

    def __init__(self) -> None:
        self._lib = None
        self._has_ex = None
        self._has_occupancy = False
        self._has_dispatch_resolver = False

    def _ensure_loaded(self):
        if self._lib is not None:
            return self._lib

        capabilities = detect_capabilities()
        if not capabilities.cuda_rmsnorm_available:
            raise CudaRmsnormUnavailable(
                "CUDA RMSNorm shared library not available; "
                "build it with `cmake --build build/<preset>` "
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
            ctypes.c_int,      # variant: 0=auto, 1=reference, 2=v0, 3=v1, 4=v2
        ]
        fn.restype = ctypes.c_int

        # Optional: the stream-aware entry point (added by E03-01). The symbol
        # may be absent on a stale build, in which case callers fall back to
        # the default-stream entry point.
        self._has_ex = hasattr(lib, "hqsb_rmsnorm_forward_ex_c")
        if self._has_ex:
            fn_ex = lib.hqsb_rmsnorm_forward_ex_c
            fn_ex.argtypes = fn.argtypes + [ctypes.c_void_p]  # + stream
            fn_ex.restype = ctypes.c_int

        # Optional: the occupancy query (added by E03-02). Purely a
        # launch-configuration property; absent on older builds.
        self._has_occupancy = hasattr(lib, "hqsb_rmsnorm_occupancy_c")
        if self._has_occupancy:
            fn_occ = lib.hqsb_rmsnorm_occupancy_c
            fn_occ.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
            fn_occ.restype = ctypes.c_int

        self._has_dispatch_resolver = hasattr(
            lib, "hqsb_rmsnorm_resolve_dispatch_c"
        )
        if self._has_dispatch_resolver:
            fn_resolve = lib.hqsb_rmsnorm_resolve_dispatch_c
            fn_resolve.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_longlong,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            fn_resolve.restype = ctypes.c_int

        self._lib = lib
        return lib

    # ── validation ────────────────────────────────────────────────────

    @staticmethod
    def _torch():
        import torch  # local import: keep module import torch-free

        return torch

    def _validate(self, x, weight, dtype, out, epsilon):
        """Raise :class:`RmsNormContractError` before any launch."""
        torch = self._torch()

        dtype_code = _DTYPE_CODE.get(str(dtype))
        if dtype_code is None:
            raise RmsNormContractError(
                "UNSUPPORTED_DTYPE",
                f"dtype {dtype!r} is outside the frozen RMSNorm contract "
                f"(supported: {sorted(_DTYPE_CODE)}); no implicit cast is applied",
                {"dtype": str(dtype)},
            )
        expected = torch.float16 if dtype_code == 1 else torch.float32

        if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
            raise RmsNormContractError(
                "SHAPE_INVALID",
                "x and weight must be torch.Tensor instances",
                {"x_type": type(x).__name__, "weight_type": type(weight).__name__},
            )

        if x.dim() != 2:
            raise RmsNormContractError(
                "SHAPE_INVALID",
                f"x must be 2-D (rows, hidden); got {tuple(x.shape)}",
                {"x_shape": list(x.shape)},
            )
        if weight.dim() != 1:
            raise RmsNormContractError(
                "WEIGHT_SHAPE_MISMATCH",
                f"weight must be 1-D (hidden,); got {tuple(weight.shape)}",
                {"weight_shape": list(weight.shape)},
            )

        rows, hidden = int(x.shape[0]), int(x.shape[1])
        if rows < 1 or hidden < 1:
            raise RmsNormContractError(
                "SHAPE_INVALID",
                f"rows and hidden must be >= 1; got rows={rows}, hidden={hidden}",
                {"rows": rows, "hidden": hidden},
            )
        if int(weight.numel()) != hidden:
            raise RmsNormContractError(
                "WEIGHT_SHAPE_MISMATCH",
                f"weight length {int(weight.numel())} != hidden {hidden}",
                {"weight_numel": int(weight.numel()), "hidden": hidden},
            )

        if not x.is_cuda or not weight.is_cuda:
            raise RmsNormContractError(
                "DEVICE_INVALID",
                "x and weight must both live on a CUDA device",
                {"x_device": str(x.device), "weight_device": str(weight.device)},
            )
        if x.device != weight.device:
            raise RmsNormContractError(
                "DEVICE_INVALID",
                f"x is on {x.device} but weight is on {weight.device}",
                {"x_device": str(x.device), "weight_device": str(weight.device)},
            )

        if x.dtype != expected or weight.dtype != expected:
            raise RmsNormContractError(
                "DTYPE_MISMATCH",
                f"declared dtype={dtype!r} requires both tensors to be "
                f"{expected}; got x={x.dtype}, weight={weight.dtype}",
                {
                    "declared_dtype": str(dtype),
                    "x_dtype": str(x.dtype),
                    "weight_dtype": str(weight.dtype),
                },
            )

        # The kernels are written against contiguous row-major storage and
        # ignore strides entirely. Accepting a strided view would silently
        # produce a wrong result, so it is rejected instead.
        if not x.is_contiguous() or not weight.is_contiguous():
            raise RmsNormContractError(
                "NON_CONTIGUOUS_INPUT",
                "x and weight must be contiguous (the kernels consume raw "
                "row-major pointers and do not interpret strides)",
                {
                    "x_stride": list(x.stride()),
                    "weight_stride": list(weight.stride()),
                },
            )

        if out is not None:
            if not isinstance(out, torch.Tensor):
                raise RmsNormContractError(
                    "OUT_TENSOR_MISMATCH",
                    "out must be a torch.Tensor",
                    {"out_type": type(out).__name__},
                )
            if tuple(out.shape) != (rows, hidden):
                raise RmsNormContractError(
                    "OUT_TENSOR_MISMATCH",
                    f"out shape {tuple(out.shape)} != x shape {(rows, hidden)}",
                    {"out_shape": list(out.shape)},
                )
            if out.dtype != expected:
                raise RmsNormContractError(
                    "OUT_TENSOR_MISMATCH",
                    f"out dtype {out.dtype} != expected {expected}",
                    {"out_dtype": str(out.dtype)},
                )
            if not out.is_cuda or not out.is_contiguous():
                raise RmsNormContractError(
                    "OUT_TENSOR_MISMATCH",
                    "out must be a contiguous CUDA tensor",
                    {"out_device": str(out.device)},
                )

        try:
            eps = float(epsilon)
        except (TypeError, ValueError):
            raise RmsNormContractError(
                "EPSILON_INVALID",
                f"epsilon must be a real number; got {epsilon!r}",
                {"epsilon": repr(epsilon)},
            ) from None
        if not (eps > 0.0) or not math.isfinite(eps):
            raise RmsNormContractError(
                "EPSILON_INVALID",
                f"epsilon must be finite and > 0; got {epsilon!r}",
                {"epsilon": repr(epsilon)},
            )

        return rows, hidden, dtype_code

    # ── execution ─────────────────────────────────────────────────────

    def forward(
        self,
        x,
        weight,
        *,
        dtype: str = "fp32",
        variant=0,
        epsilon: float = 1e-5,
        out=None,
        stream=None,
    ):
        """Run RMSNorm on CUDA tensors via the hand-written kernels.

        Args:
            x: ``(rows, hidden)`` contiguous CUDA tensor.
            weight: ``(hidden,)`` contiguous CUDA tensor (same dtype/device).
            dtype: ``"fp32"`` or ``"fp16"`` (anything else is rejected).
            variant: A frozen variant name, including ``scalar_safe`` and
                ``v2_vectorized_strict``, or its numeric C ABI code.
            epsilon: denominator stabilizer (finite, > 0).
            out: Optional pre-allocated output tensor; else allocated.
            stream: Optional explicit CUDA stream: a ``torch.cuda.Stream`` or a
                raw ``cudaStream_t`` integer. ``None`` keeps the historical
                default-stream behaviour.

        Returns:
            The output tensor ``(rows, hidden)``.

        Raises:
            CudaRmsnormUnavailable: If the shared library cannot be loaded.
            RmsNormContractError: If an input violates the operator contract
                (no kernel is launched in that case).
            RuntimeError: If the kernel itself reports a CUDA error.
        """
        import torch

        lib = self._ensure_loaded()
        rows, hidden, dtype_code = self._validate(x, weight, dtype, out, epsilon)
        variant_code = _resolve_variant_code(variant)

        if out is None:
            out = torch.empty_like(x)

        if stream is None:
            stream_ptr = None
        elif hasattr(stream, "cuda_stream"):
            stream_ptr = int(stream.cuda_stream)
        else:
            stream_ptr = int(stream)

        if self._has_ex and stream_ptr is not None:
            err = lib.hqsb_rmsnorm_forward_ex_c(
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                rows,
                hidden,
                ctypes.c_float(epsilon),
                dtype_code,
                variant_code,
                ctypes.c_void_p(stream_ptr),
            )
        else:
            # Stream is either unspecified (default stream) or the built
            # library predates the stream-aware entry point.
            err = lib.hqsb_rmsnorm_forward_c(
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                rows,
                hidden,
                ctypes.c_float(epsilon),
                dtype_code,
                variant_code,
            )
        if err != 0:
            raise RuntimeError(f"hqsb_rmsnorm_forward_c failed with cudaError={err}")
        return out

    def resolve_dispatch(self, x, weight, out, *, dtype: str, variant=0):
        """Return the production dispatch decision without launching.

        E03-03 compares this result with an independently implemented oracle.
        For a rejected strict-vector request, ``cuda_error`` is non-zero but
        all predicate fields remain populated so the rejection is diagnosable.
        """
        lib = self._ensure_loaded()
        if not self._has_dispatch_resolver:
            raise CudaRmsnormUnavailable(
                "the loaded library does not export the E03-03 resolver"
            )
        rows, hidden, dtype_code = self._validate(x, weight, dtype, out, 1e-5)
        del rows
        variant_code = _resolve_variant_code(variant)
        actual = ctypes.c_int(-1)
        reasons = ctypes.c_uint(0)
        vector_width = ctypes.c_int(0)
        alignment = ctypes.c_int(0)
        load_width = ctypes.c_int(0)
        err = lib.hqsb_rmsnorm_resolve_dispatch_c(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            hidden,
            dtype_code,
            variant_code,
            ctypes.byref(actual),
            ctypes.byref(reasons),
            ctypes.byref(vector_width),
            ctypes.byref(alignment),
            ctypes.byref(load_width),
        )
        return {
            "cuda_error": int(err),
            "requested_variant": variant_code,
            "actual_variant": int(actual.value),
            "reason_mask": int(reasons.value),
            "vector_width_elements": int(vector_width.value),
            "required_alignment_bytes": int(alignment.value),
            "actual_load_width_bytes": int(load_width.value),
            "vector_eligible": int(reasons.value) == 0,
        }

    # ── launch-configuration resources (E03-02) ───────────────────────

    def occupancy(self, *, variant, dtype: str, block_size: int) -> int:
        """Theoretical max active blocks per SM for one launch configuration.

        This is a *resource* query, not an execution: it returns 0 for an
        unsupported (variant, dtype) pair or an invalid block size, which is
        distinguishable from "not queried" (which raises).
        """
        lib = self._ensure_loaded()
        if not self._has_occupancy:
            raise CudaRmsnormUnavailable(
                "the loaded library does not export hqsb_rmsnorm_occupancy_c; "
                "rebuild with `cmake --build build/jetson-release`"
            )
        dtype_code = _DTYPE_CODE.get(str(dtype))
        if dtype_code is None:
            raise RmsNormContractError(
                "UNSUPPORTED_DTYPE",
                f"dtype {dtype!r} is outside the frozen RMSNorm contract",
                {"dtype": str(dtype)},
            )
        variant_code = _resolve_variant_code(variant)
        return int(
            lib.hqsb_rmsnorm_occupancy_c(variant_code, dtype_code, int(block_size))
        )


# Module-level singleton (lazy-loaded).
_bridge = _CudaRmsnormBridge()


def rmsnorm_forward(x, weight, **kwargs):
    """Convenience wrapper around :class:`_CudaRmsnormBridge.forward`."""
    return _bridge.forward(x, weight, **kwargs)


def rmsnorm_occupancy(*, variant, dtype: str, block_size: int) -> int:
    """Convenience wrapper around :class:`_CudaRmsnormBridge.occupancy`."""
    return _bridge.occupancy(variant=variant, dtype=dtype, block_size=block_size)


def rmsnorm_resolve_dispatch(x, weight, out, *, dtype: str, variant=0):
    """Resolve the concrete production path without launching a kernel."""
    return _bridge.resolve_dispatch(x, weight, out, dtype=dtype, variant=variant)


__all__ = [
    "CudaRmsnormUnavailable",
    "RMSNORM_REASON_CODES",
    "RmsNormContractError",
    "rmsnorm_forward",
    "rmsnorm_occupancy",
    "rmsnorm_resolve_dispatch",
]
