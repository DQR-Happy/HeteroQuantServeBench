"""HQSB release, open-source and job-evidence layer (S15).

The package implements the *capability and evidence* layer of
``docs/stage_experiments/details/S15``:

* canonical identity for every S15 artifact — release candidates, claims, claim
  ledgers, evidence bundles, figures, points, release artifacts, provenance,
  SBOMs, demo assets, narratives, contributions and reviewer sessions
  (``identity``);
* the three frozen objects of ``details/S15/README.md`` §7 —
  ``ReleaseCandidateSnapshot``, ``PublicEvidenceBundle`` and
  ``FinalAcceptanceDecision`` — plus ``ClaimRecord`` (§8) and
  ``ContributionRecord`` (§19) (``contracts``);
* the S15 vocabularies: the six experiment statuses of §13.5, the claim
  lifecycle, the finding severity/disposition ladders, the demo measurement
  states, the documentation command verdicts, the figure rebuild classes, the
  upstream dispositions, and one minimum record per experiment (``records``);
* the §22 uniform run data package and the execution-safety policy (``campaign``);
* the triple-gated experiment scaffolding (``experiment``): prerequisites,
  preregistration, evidence manifest and verdicts — a conclusion is impossible by
  default;
* action logs, time-to-event statistics, evidence-lookup trials and detector
  metrics (``telemetry``);
* the frozen vocabularies under ``configs/release/`` (``specs``).

Design constraints (``docs/architecture/module_ownership.md`` rules R17/R18):

* the layer sits above every other region; no lower region may import it;
* it never imports ``ops`` — nothing here builds a wheel, publishes a release,
  uploads an artifact or contacts an upstream repository by itself;
* it must import on the CPU-minimal installation, so no module-level
  ``torch``/``triton``/``numpy``/``requests``/``httpx``;
* a missing capability is a *state* (``BLOCKED``, ``INVALID``), never a silent
  fallback and never a fabricated ``0``;
* nothing in this package executes a formal S15 experiment or emits a
  conclusion: the driver refuses to write ``PASS``/``FAIL`` without satisfied
  prerequisites, ``--execute`` and raw samples.

Heavy submodules are imported lazily: ``import hqsb.release`` must not pull in the
whole layer.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

#: Stage id used by the experiment scaffolding and evidence manifests.
STAGE = "S15"

#: Experiments delivered by this layer (``S15_实验清单.md`` §必做实验).
EXPERIMENTS: Tuple[str, ...] = (
    "E15-01",
    "E15-02",
    "E15-03",
    "E15-04",
    "E15-05",
    "E15-06",
    "E15-07",
    "E15-08",
    "E15-09",
    "E15-10",
    "E15-11",
)

#: The P0 chain: every listed experiment must pass before S15 is complete.
P0_EXPERIMENTS: Tuple[str, ...] = tuple(experiment_id for experiment_id in EXPERIMENTS if experiment_id != "E15-11")

#: The one P1 experiment (heading-architecture study, recommended before applying).
P1_EXPERIMENTS: Tuple[str, ...] = ("E15-11",)

#: G0–G8 hierarchical gates (``details/S15/README.md`` §11).
STAGE_GATES: Tuple[Tuple[str, str], ...] = (
    ("G0", "upstream_freeze"),
    ("G1", "claim_truth"),
    ("G2", "executable_entry"),
    ("G3", "technical_reproduction"),
    ("G4", "release_integrity"),
    ("G5", "communication_robustness"),
    ("G6", "independent_validation"),
    ("G7", "public_collaboration"),
    ("G8", "audience_comprehension"),
)

_LAZY: Dict[str, str] = {
    # L0 — identity / records / contracts / campaign
    "identity": "identity",
    "records": "records",
    "contracts": "contracts",
    "campaign": "campaign",
    # L1 — claim truth (E15-01)
    "claims": "claims",
    # L2 — entry and documents (E15-02, E15-04)
    "quickstart": "quickstart",
    "docs_gate": "docs_gate",
    # L3 — reproduction and figures (E15-03, E15-06)
    "hero_replay": "hero_replay",
    "figures": "figures",
    # L4 — release and communication (E15-05, E15-07, E15-08)
    "supply_chain": "supply_chain",
    "demo": "demo",
    "narrative": "narrative",
    # L5 — external validation and public collaboration (E15-09, E15-10)
    "clean_room": "clean_room",
    "upstream": "upstream",
    "first_impression": "first_impression",
    # L6 — evidence and scaffolding
    "telemetry": "telemetry",
    "specs": "specs",
    "experiment": "experiment",
    "interface_map": "interface_map",
}

__all__ = [
    "EXPERIMENTS",
    "P0_EXPERIMENTS",
    "P1_EXPERIMENTS",
    "STAGE",
    "STAGE_GATES",
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
# ``hqsb.release -> ...experiment -> ...specs -> hqsb.release`` that the import
# dependency gate (rule set >= 1.9.0) rejects.  ``_LAZY`` above is the only
# supported access path, and the gate's static scan stays acyclic.
