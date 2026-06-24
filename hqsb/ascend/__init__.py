"""HQSB Ascend C / CANN heterogeneous backend layer (S09).

The package owns the *control plane* of the Ascend backend: the compatibility
manifest and four-state capability table, the host/device Tiling ABI, the
operator oracles and shared test vectors, the framework binding contract
(current device/stream, two-phase workspace, fallback policy), the C4 Backend
adapter, the quantization/format compatibility layer, the model-core A/B
identity, the msprof profiling model, the CUDA<->Ascend fair-comparison
framework and the fault/recovery taxonomy.

Where it sits in the dependency graph
-------------------------------------
``hqsb.ascend`` is a **backend adapter region**: it consumes the stable
contracts and the shared numeric conventions, and is consumed through the C4
``Backend`` ABC plus the registry — never by a direct import::

    core ← models/benchmark ← backends/integration/quant/ascend ← runtime ← serving

Design rules (see ``docs/architecture/module_ownership.md`` rules R7-R9):

* no module-level ``torch``/``torch_npu``/``triton``/``numpy`` import — heavy
  engines are probed lazily and a missing engine is a *structured* reason
  (``UNAVAILABLE`` + reason code), never an ``ImportError``;
* ``ops`` is never imported: Ascend C kernels are addressed by provider name
  and source/binary hash, exactly as ``hqsb.runtime``/``hqsb.serving`` do;
* ``hqsb.runtime`` and ``hqsb.serving`` are never imported (they sit above);
* nothing else in the repository may import this package.

Maturity
--------
Every module here is **M1/M2**: the interfaces exist (M1) and are covered by
CPU tests in this tree (M2).  Nothing in this package has run on Ascend
silicon, so no module may produce an M3+ claim.  :mod:`hqsb.ascend.experiment`
enforces that: the prerequisite gate reports ``BLOCKED`` while no real device,
CANN toolchain or E09-01 verdict exists, and ``RunDirectory.write_verdict``
refuses PASS/FAIL/PASS_NEGATIVE without raw samples.

:mod:`hqsb.ascend.sim_device` is a **CPU test double** of the Ascend execution
model.  It exists to prove the Tiling ABI is self-consistent (host and device
parse the same bytes) and that tail/multi-core partitioning is correct; it is
*not* an Ascend result and its outputs must never be reported as one.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "STAGE",
    "__version__",
]

#: The stage this package implements (protocol statuses live in ``experiment``).
STAGE = "S09"

__version__ = "0.1.0"

#: Public API re-exported lazily so that ``import hqsb.ascend`` stays cheap and
#: free of optional dependencies (mirrors ``hqsb.runtime``/``hqsb.serving``).
_LAZY: dict[str, str] = {
    "backend": "hqsb.ascend.backend",
    "build": "hqsb.ascend.build",
    "capability": "hqsb.ascend.capability",
    "comparison": "hqsb.ascend.comparison",
    "compatibility": "hqsb.ascend.compatibility",
    "experiment": "hqsb.ascend.experiment",
    "faults": "hqsb.ascend.faults",
    "framework": "hqsb.ascend.framework",
    "interface_map": "hqsb.ascend.interface_map",
    "kernel_source": "hqsb.ascend.kernel_source",
    "mapping": "hqsb.ascend.mapping",
    "model_core": "hqsb.ascend.model_core",
    "operators": "hqsb.ascend.operators",
    "profiling": "hqsb.ascend.profiling",
    "probes": "hqsb.ascend.probes",
    "quant_format": "hqsb.ascend.quant_format",
    "runtime_binding": "hqsb.ascend.runtime_binding",
    "sim_device": "hqsb.ascend.sim_device",
    "specs": "hqsb.ascend.specs",
    "telemetry": "hqsb.ascend.telemetry",
    "test_vectors": "hqsb.ascend.test_vectors",
    "tiling": "hqsb.ascend.tiling",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy loader
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    globals()[name] = module
    return module


def __dir__() -> list[str]:  # pragma: no cover - introspection helper
    return sorted(set(__all__) | set(_LAZY))
