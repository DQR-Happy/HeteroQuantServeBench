"""HQSB train-serve coordination and frontier-extension layer (S14).

The package implements the *capability and evidence* layer of
``docs/stage_experiments/details/S14``:

* canonical identity for every S14 artifact — ``TrainingRunArtifact``,
  ``CheckpointArtifact``, ``ServingModelArtifact``, ``PolicySnapshot``,
  ``TrajectoryRecord``, ``FrontierStudyContract``, ``AdoptionDecision``
  (``identity``, ``contracts``);
* the S14 vocabularies: the eleven experiment statuses of
  ``details/S14/README.md`` §21, the nine maturity levels of §6, the nine
  adoption decisions of §7.7, the six capability states of
  ``E14-01`` §3.2, the state machines and the per-table schemas (``records``);
* the cross-cutting invariants shared by all thirteen experiments
  (``contracts``);
* the experimental dependency boundary: extras registry, feature flags,
  capability probe, import purity, side-effect audit and the per-extra verdict
  (``dependencies``, E14-01);
* distributed training state, checkpoint completeness and trajectory
  continuity (``training``, E14-02);
* the conversion DAG and the layered train→serve parity gates
  (``parity``, E14-03);
* SFT/DPO/GRPO objective oracles, policy lineage, queue/staleness and
  sync/async state machines (``posttraining``, E14-04);
* literature registry, candidate estimands, preregistration and the single
  primary frontier branch (``frontier``, E14-05);
* the four conditional-P0 frontier branches (``speculative`` ``moe``
  ``long_context`` ``sparsity``, E14-F1…F4);
* the three optional transfers (``multimodal`` ``agent`` ``edge``,
  E14-06/07/08).

Design constraints (``docs/architecture/module_ownership.md`` rules R15/R16):

* the layer sits above every other region; no lower region may import it;
* it never imports ``ops`` — nothing here starts a trainer, an engine, a
  cluster or a kernel;
* it must import on the CPU-minimal installation, so no module-level
  ``torch``/``triton``/``numpy``/``transformers``/``ray``/``vllm``;
* a missing framework is a *state* (``UNAVAILABLE_DEPENDENCY``,
  ``ABI_MISMATCH``, ``DEVICE_UNAVAILABLE``, ``DISABLED_BY_CONFIG``), never a
  silent fallback and never a numeric ``0``;
* nothing in this package executes a formal S14 experiment or emits a
  conclusion: the driver refuses to write PASS/FAIL/PASS_NEGATIVE/
  FAIL_PERFORMANCE_HYPOTHESIS without satisfied prerequisites *and* raw
  samples.

Heavy submodules are imported lazily: ``import hqsb.experimental`` must not pull
in the whole layer.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

#: Stage id used by the experiment scaffolding and evidence manifests.
STAGE = "S14"

#: Experiments delivered by this layer (``details/S14/README.md`` §2).
EXPERIMENTS: Tuple[str, ...] = (
    "E14-01",
    "E14-02",
    "E14-03",
    "E14-04",
    "E14-05",
    "E14-F1",
    "E14-F2",
    "E14-F3",
    "E14-F4",
    "E14-06",
    "E14-07",
    "E14-08",
)

#: The public P0 chain (``details/S14/README.md`` §3, ``T``).
PRIMARY_CHAIN: Tuple[str, ...] = ("E14-02", "E14-03", "E14-04")

#: The four conditional-P0 frontier branches (``F``); exactly one is primary.
FRONTIER_BRANCHES: Tuple[str, ...] = ("E14-F1", "E14-F2", "E14-F3", "E14-F4")

#: The optional transfers (``X``); ``N/A_BY_ADR`` does not block S14.
OPTIONAL_TRANSFERS: Tuple[str, ...] = ("E14-06", "E14-07", "E14-08")

_LAZY: Dict[str, str] = {
    # L0 — identity / records / contracts / campaign
    "identity": "identity",
    "records": "records",
    "contracts": "contracts",
    "campaign": "campaign",
    # L1 — dependency boundary (E14-01)
    "dependencies": "dependencies",
    # L2 — train → serve chain (E14-02, E14-03, E14-04)
    "training": "training",
    "parity": "parity",
    "posttraining": "posttraining",
    # L3 — frontier selection and the four branches (E14-05, E14-F1…F4)
    "frontier": "frontier",
    "speculative": "speculative",
    "moe": "moe",
    "long_context": "long_context",
    "sparsity": "sparsity",
    # L4 — optional transfers (E14-06, E14-07, E14-08)
    "multimodal": "multimodal",
    "agent": "agent",
    "edge": "edge",
    # L5 — evidence and scaffolding
    "telemetry": "telemetry",
    "specs": "specs",
    "experiment": "experiment",
    "interface_map": "interface_map",
}

__all__ = [
    "EXPERIMENTS",
    "FRONTIER_BRANCHES",
    "OPTIONAL_TRANSFERS",
    "PRIMARY_CHAIN",
    "STAGE",
    *sorted(_LAZY),
]


def __getattr__(name: str) -> Any:
    """Import a submodule on first attribute access (PEP 562)."""
    if name in _LAZY:
        import importlib

        return importlib.import_module(f"{__name__}.{_LAZY[name]}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# NOTE: the submodules are intentionally NOT imported here (not even under
# ``TYPE_CHECKING``): a package-level import edge would create the cycle
# ``hqsb.experimental -> ...experiment -> ...specs -> hqsb.experimental`` that
# the import dependency gate (rule set >= 1.8.0) rejects.  ``_LAZY`` above is the
# only supported access path, and the gate's static scan stays acyclic.
