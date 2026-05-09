"""Triton kernels for HQSB (S04).

Importing this package does NOT import triton eagerly: ``triton`` is imported
lazily inside each kernel module so that the CPU/CUDA core remains importable
on machines without Triton installed. Use :func:`ops.capability.detect_capabilities`
to check ``triton_available`` before calling these kernels.
"""

__all__ = ["gemm", "rmsnorm"]
