"""HQSB distributed inference and communication layer (S10).

The package implements the *capability and evidence* layer of
``docs/stage_experiments/details/S10``: topology identity and rank placement
(E10-01), collective microbenchmark semantics and cost models (E10-02),
communicator safety (E10-03), tensor parallel plans and the communication
ledger (E10-04), scaling work units (E10-05), overlap schedules (E10-06),
PP/CP/SP boundaries (E10-07), MoE expert parallelism (E10-08), multi-rank
traces (E10-09) and distributed fault boundedness (E10-10).

Design constraints (``docs/architecture/module_ownership.md``):

* the layer sits above ``runtime``/``serving``; no lower region may import it;
* it never imports ``ops`` — kernels are addressed by capability/provider name;
* it must import on the CPU-minimal installation, so no module-level
  ``torch``/``triton``/``numpy``;
* every probe is read-only and every unavailable value is recorded as
  ``UNAVAILABLE(reason)`` — never as ``0``;
* no interface in this package executes an experiment or emits a conclusion.

Heavy submodules are imported lazily: ``import hqsb.distributed`` must not pull
in the whole layer.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

#: Stage id used by the experiment scaffolding and evidence manifests.
STAGE = "S10"

#: Experiments delivered by this layer (details README §3).
EXPERIMENTS: Tuple[str, ...] = tuple(f"E10-{index:02d}" for index in range(1, 11))

_LAZY: Dict[str, str] = {
    # core planes
    "topology": "topology",
    "ranks": "ranks",
    "placement": "placement",
    "probes": "probes",
    "backend": "backend",
    "collectives": "collectives",
    "faults": "faults",
    "sequence": "sequence",
    # model / performance planes
    "parallel_plan": "parallel_plan",
    "ledger": "ledger",
    "scaling": "scaling",
    "overlap": "overlap",
    "boundary": "boundary",
    "moe": "moe",
    "traces": "traces",
    # evidence / scaffolding
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
# ``hqsb.distributed -> ...experiment -> ...specs -> hqsb.distributed`` that the
# import dependency gate (rule set >= 1.3.0) rejects.  ``_LAZY`` above is the
# only supported access path, and the gate's static scan stays acyclic.
