"""HQSB operator layer: CUDA/Triton/cuBLAS kernels and the unified dispatcher.

The package is split into:

* :mod:`ops.capability` — runtime backend detection (never raises);
* :mod:`ops.cuda_bridge` — ctypes bridge to the S03 CUDA shared library;
* :mod:`ops.triton` — Triton kernels (RMSNorm, GEMM) with autotune;
* :mod:`ops.dispatcher` — unified implementation selection.

Importing ``ops`` must never import triton or torch eagerly; those imports
happen lazily inside the capability probe and the Triton kernels.
"""

__all__ = [
    "capability",
    "cuda_bridge",
    "dispatcher",
    "triton",
]
