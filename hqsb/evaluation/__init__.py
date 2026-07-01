"""HQSB cross-hardware evaluation and unified benchmark layer (S12).

The package implements the *capability and evidence* layer of
``docs/stage_experiments/details/S12``:

* comparability audit, field classification, comparison contracts and the
  four-state verdict engine (E12-01);
* capability registry, probe evidence matrix and invalidation rules (E12-02);
* the four-layer unified replay: raw observations → normalized results
  (E12-03);
* repeatability, drift, noise control and exclusion ledgers (E12-04);
* hierarchical Roofline/Amdahl models, prediction errors and residuals (E12-05);
* power-window alignment, energy integration and efficiency metrics (E12-06);
* cloud/owned TCO, cost per token and sensitivity/break-even (E12-07);
* business-profile constrained Pareto frontiers and recommendations (E12-08);
* software maturity rubric, engineering cost and tool coverage (E12-09);
* end-to-end evidence lineage, validation and regeneration (E12-10).

Design constraints (``docs/architecture/module_ownership.md``):

* the layer sits above every other region; no lower region may import it;
* it never imports ``ops`` — kernels are addressed by capability/provider
  names plus artifact locators;
* it must import on the CPU-minimal installation, so no module-level
  ``torch``/``triton``/``numpy``;
* missing data is a *state* (``MEASUREMENT_UNAVAILABLE``, ``PRICE_UNAVAILABLE``,
  ``NOT_APPLICABLE_CAPABILITY``, …), never a numeric ``0``;
* nothing in this package executes a formal experiment or emits a conclusion:
  the driver refuses to write PASS/FAIL/PASS_NEGATIVE without satisfied
  prerequisites and raw samples.

Heavy submodules are imported lazily: ``import hqsb.evaluation`` must not pull
in the whole layer.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

#: Stage id used by the experiment scaffolding and evidence manifests.
STAGE = "S12"

#: Experiments delivered by this layer (details README §3 / 清单).
EXPERIMENTS: Tuple[str, ...] = tuple(f"E12-{index:02d}" for index in range(1, 11))

_LAZY: Dict[str, str] = {
    # L0 — identity / records / vocabulary
    "identity": "identity",
    "records": "records",
    "contracts": "contracts",
    "layers": "layers",
    "campaign": "campaign",
    # L1 — comparability and capability (E12-01, E12-02)
    "candidates": "candidates",
    "comparability": "comparability",
    "platform": "platform",
    "capability": "capability",
    # L2 — unified replay and repeatability (E12-03, E12-04)
    "benchmark": "benchmark",
    "repeatability": "repeatability",
    # L3 — explanation, energy, cost (E12-05, E12-06, E12-07)
    "roofline": "roofline",
    "energy": "energy",
    "cost": "cost",
    # L4 — decision and maturity (E12-08, E12-09)
    "pareto": "pareto",
    "maturity": "maturity",
    # L5 — lineage and regeneration (E12-10)
    "lineage": "lineage",
    # L6 — evidence and scaffolding
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
# ``hqsb.evaluation -> ...experiment -> ...specs -> hqsb.evaluation`` that the
# import dependency gate (rule set >= 1.6.0) rejects.  ``_LAZY`` above is the
# only supported access path, and the gate's static scan stays acyclic.
