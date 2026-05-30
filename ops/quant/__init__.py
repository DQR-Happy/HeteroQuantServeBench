"""Execution layer for low-bit weight-only GEMM (S05 / E05-06).

This package is the *only* place where a packed ``QuantArtifact`` variant is
turned into a kernel. It consumes the semantic layer (``hqsb.quant``) and
never defines quantization math itself:

* :mod:`ops.quant.capability` — what this environment can actually execute,
  with structured reasons;
* :mod:`ops.quant.executors` — the executor registry: FP16 reference,
  storage-only (materialize then FP16 GEMM) and fused-dequant (Triton), each
  reporting an observed kernel symbol and an execution label;
* :mod:`ops.quant.w4a16_triton` — the Triton W8/W4 fused-dequant GEMM that
  implements the ``hqsb.w4a16.rowmajor.nk.v1`` /
  ``hqsb.w8a16.rowmajor.nk.v1`` layouts;
* :mod:`ops.quant.microbench` — device-event timing with warmup/repeats and
  launch/workspace accounting;
* :mod:`ops.quant.safety` — guard regions and sanitizer command construction
  for the memory-safety check.

Importing this package must not import torch or triton: the modules import
them lazily inside functions so the CPU-minimal environment can import the
package and receive a structured capability error instead of an ImportError.
"""

from __future__ import annotations

__all__ = [
    "capability",
    "executors",
    "microbench",
    "safety",
    "w4a16_triton",
]
