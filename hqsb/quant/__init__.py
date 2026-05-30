"""QuantLab — S05 quantization semantics, artifacts, calibration and policies.

Module boundary (top-level architecture §3.3, ``module_ownership``):

* ``hqsb.quant`` owns *quantization semantics*: scheme/round/granularity,
  logical qvalues, packing layouts, ``QuantArtifact`` documents,
  calibration data identity/statistics, quality and sensitivity analysis,
  mixed-precision policies and deployment decisions.
* ``hqsb.quant`` must not import the concrete *execution* layer (``ops``) or
  any backend: it declares requirements, capability descriptors and
  protocols; kernels and model plumbing consume them from outside
  (``ops/quant`` and ``scripts/quant``).
* ``hqsb.core`` must never import this package.
* Heavy third-party imports (``torch``, ``numpy``) are lazy and optional.
  Every module here imports with the CPU-minimal installation
  (``pydantic`` + ``PyYAML``); a missing accelerator is reported as a
  structured :class:`~hqsb.core.errors.CapabilityError`, never a silent
  fallback.

The package intentionally performs no work at import time: no file IO, no
device probing, no model loading.
"""

from __future__ import annotations

__all__ = [
    "artifact",
    "compat",
    "packing",
    "rounding",
    "rtn",
    "spec",
]
