"""HQSB production / cloud-native / reliability layer (S13).

The package implements the *capability and evidence* layer of
``docs/stage_experiments/details/S13``:

* release, OCI and source identity plus the ``ReleaseBundle`` field set
  (``identity``);
* the S13 vocabularies, the three lifecycle state machines (pod/process, model
  artifact, request), the evidence ladder and the table schemas (``records``);
* the fourteen cross-cutting §23 validators (``contracts``);
* campaign layout, manifest and the execution-safety policy (``campaign``);
* supply-chain gates: digest DAG, reproducibility levels, SBOM/vulnerability/
  secret/license/provenance and the release decision (``supply_chain``, E13-01);
* clean-environment bootstrap and the deployment state machine (``deployment``,
  E13-02);
* accelerator placement, capability labels, NUMA/topology and isolation
  (``scheduling``, E13-03);
* model artifact download/verify/atomic activation/rollback/GC (``artifacts``,
  E13-04);
* probes, graceful drain, rolling restart and terminal request accounting
  (``lifecycle``, E13-05);
* memory ledger, KV budget, admission policies and OOM margin (``capacity``,
  E13-06);
* metric-age-aware autoscaling and control stability (``autoscaling``, E13-07);
* the request→hardware observability contract and RCA records (``observability``,
  E13-08);
* the fault contract, blast-radius safety and MTTD/MTTM/MTTR accounting
  (``faults``, E13-09);
* canary state machine, gate hierarchy, sequential decisions and rollback
  (``canary``, E13-10);
* multi-tenant identity, quota, network, redaction and audit evidence
  (``security``, E13-11).

Design constraints (``docs/architecture/module_ownership.md``):

* the layer sits above every other region; no lower region may import it;
* it never imports ``ops`` — nothing here starts a container, a cluster or a
  kernel;
* it must import on the CPU-minimal installation, so no module-level
  ``torch``/``triton``/``numpy``;
* missing data is a *state* (``NOT_RUN_CLUSTER_UNAVAILABLE``,
  ``SCANNER_UNAVAILABLE``, ``NOT_RUN_SAFETY_BOUNDARY``, …), never a numeric ``0``;
* nothing in this package executes a formal S13 experiment or emits a
  conclusion: the driver refuses to write PASS/FAIL/PASS_NEGATIVE without
  satisfied prerequisites *and* raw samples.

Heavy submodules are imported lazily: ``import hqsb.infra`` must not pull in the
whole layer.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

#: Stage id used by the experiment scaffolding and evidence manifests.
STAGE = "S13"

#: Experiments delivered by this layer (details/S13/README.md §2).
EXPERIMENTS: Tuple[str, ...] = tuple(f"E13-{index:02d}" for index in range(1, 12))

_LAZY: Dict[str, str] = {
    # L0 — identity / records / contracts / campaign
    "identity": "identity",
    "records": "records",
    "contracts": "contracts",
    "campaign": "campaign",
    # L1 — supply chain (E13-01)
    "supply_chain": "supply_chain",
    # L2 — deployment and placement (E13-02, E13-03)
    "deployment": "deployment",
    "scheduling": "scheduling",
    # L3 — artifact lifecycle and pod/request lifecycle (E13-04, E13-05)
    "artifacts": "artifacts",
    "lifecycle": "lifecycle",
    # L4 — capacity, admission, autoscaling (E13-06, E13-07)
    "capacity": "capacity",
    "autoscaling": "autoscaling",
    # L5 — observability and faults (E13-08, E13-09)
    "observability": "observability",
    "faults": "faults",
    # L6 — release governance and security (E13-10, E13-11)
    "canary": "canary",
    "security": "security",
    # L7 — evidence and scaffolding
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
# ``hqsb.infra -> ...experiment -> ...specs -> hqsb.infra`` that the import
# dependency gate (rule set >= 1.7.0) rejects.  ``_LAZY`` above is the only
# supported access path, and the gate's static scan stays acyclic.
