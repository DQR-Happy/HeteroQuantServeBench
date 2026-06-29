"""HQSB AI compiler and automatic optimisation layer (S11).

The package implements the *capability and evidence* layer of
``docs/stage_experiments/details/S11``:

* graph capture, graph breaks, guards and symbolic-domain census (E11-01);
* semantic pattern rewrites, near-miss rejection and pass idempotence (E11-02);
* graph → HQSB IR → lowering registry → real kernel closed loop (E11-03);
* IR / generated-code / PTX-SASS / hardware attribution (E11-04);
* dynamic shapes, recompile, variants and fallback safety (E11-05);
* autotune search space, holdout and budget (E11-06);
* cost model, oracle regret and low-confidence fallback (E11-07);
* compile-artifact cache, cross-process hits and invalidation (E11-08);
* TVM Relax/TensorIR or MLIR portable lowering (E11-09);
* zero-trust quality gates for AI/Agent kernel candidates (E11-10).

Design constraints (``docs/architecture/module_ownership.md``):

* the layer sits above ``integration``/``runtime``/``serving``/``distributed``;
  no lower region may import it;
* it never imports ``ops`` — kernels are addressed by capability/provider
  names plus artifact locators;
* it must import on the CPU-minimal installation, so no module-level
  ``torch``/``triton``/``numpy``;
* every probe is read-only and every unavailable value is recorded as
  ``UNAVAILABLE(reason)`` — never as ``0``;
* no interface in this package executes an experiment or emits a conclusion.

Heavy submodules are imported lazily: ``import hqsb.compiler`` must not pull in
the whole layer.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

#: Stage id used by the experiment scaffolding and evidence manifests.
STAGE = "S11"

#: Experiments delivered by this layer (details README §3 / 清单).
EXPERIMENTS: Tuple[str, ...] = tuple(f"E11-{index:02d}" for index in range(1, 11))

_LAZY: Dict[str, str] = {
    # L0 — identity / IR / records
    "identity": "identity",
    "ir": "ir",
    "records": "records",
    # L1 — front-end capture and guards
    "capture": "capture",
    "guards": "guards",
    # L2 — semantic rewrites
    "pattern_library": "pattern_library",
    "rewrite": "rewrite",
    # L3 — lowering back end
    "targets": "targets",
    "lowering": "lowering",
    "backend": "backend",
    "codegen": "codegen",
    # L4 — search: autotune and cost model
    "autotune": "autotune",
    "costmodel": "costmodel",
    # L5 — artifact lifecycle
    "cache": "cache",
    # L6 — second compiler stack
    "portable": "portable",
    # L7 — AI-assisted kernel gate
    "aigate": "aigate",
    # L8 — evidence and scaffolding
    "telemetry": "telemetry",
    "specs": "specs",
    "experiment": "experiment",
    "interface_map": "interface_map",
}

__all__ = ["EXPERIMENTS", "STAGE", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    """Import a submodule on first attribute access (PEP 562)."""
    if name in _LAZY:
        import importlib

        return importlib.import_module(f"{__name__}.{_LAZY[name]}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# NOTE: the submodules are intentionally NOT imported here (not even under
# ``TYPE_CHECKING``): a package-level import edge would create the cycle
# ``hqsb.compiler -> ...experiment -> ...specs -> hqsb.compiler`` that the
# import dependency gate (rule set >= 1.4.0) rejects.  ``_LAZY`` above is the
# only supported access path, and the gate's static scan stays acyclic.
