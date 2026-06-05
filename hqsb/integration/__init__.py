"""Framework integration — S06 custom-op, graph, compile and fallback contracts.

Module boundary (control plane §3.2/§3.3, ``docs/architecture/module_ownership.md``):

* ``hqsb.integration`` owns *framework integration*: stable operator schema and
  dispatch boundaries, Meta/FakeTensor metadata contracts, graph IR and
  pattern rewriting, guard/graph-break/recompile accounting, compile cache
  identity, lowering selection, CUDA Graph contracts, error/ABI taxonomy,
  resource lifecycle and model/backend adapters.
* ``hqsb.integration`` consumes ``hqsb.core`` contracts (C3/C4/C6/C7) and may
  lazily consult ``hqsb.quant`` artifact metadata, but defines no quantization
  mathematics and no kernel algorithm (``ops/*`` owns those).
* ``hqsb.core`` must never import this package; ``hqsb.benchmark`` and
  ``hqsb.models`` must not depend on it (enforced by
  ``scripts/audit/import_dependency_gate.py`` and
  ``tests/unit/integration/test_experiment_interface_map.py``).
* Heavy third-party imports (``torch``, ``triton``) are lazy and optional: every
  module imports with the CPU-minimal installation (``pydantic`` + ``PyYAML``).
  A missing accelerator/compiler is reported as a structured reason, never as a
  silent fallback.
* Nothing here executes an experiment. The S06 driver
  (``scripts/integration/run_e06.py``) refuses to emit a conclusion by default;
  see :mod:`hqsb.integration.experiment`.
"""

from __future__ import annotations

__all__ = [
    "adapter",
    "cache",
    "cuda_graph",
    "differential",
    "dispatch",
    "experiment",
    "graph",
    "guards",
    "interface_map",
    "lifecycle",
    "lowering",
    "meta",
    "patterns",
    "policies",
    "specs",
    "taxonomy",
    "telemetry",
]
