"""「实验步骤 → 代码接口」对照表 for all eleven S15 experiments (495 steps).

Each experiment's 45 protocol steps are mapped to concrete interfaces of this
package; :func:`resolve_interfaces` **imports every referenced symbol**, so the
mapping cannot rot into documentation — a renamed function fails the audit
instead of silently pointing at nothing.

The per-experiment step tables live with the modules they describe
(``PROTOCOL_STEPS`` in ``claims.py`` … ``first_impression.py``) so the mapping and
the implementation are edited together.  Entries are data only: **no interface
here executes an experiment**, and none of them can write a conclusion.

Step counts (``docs/stage_experiments/details/S15/`` — 11 files × 45 steps):

===========  ======  ==============================
experiment   steps   module
===========  ======  ==============================
E15-01       45      ``claims``
E15-02       45      ``quickstart``
E15-03       45      ``hero_replay``
E15-04       45      ``docs_gate``
E15-05       45      ``supply_chain``
E15-06       45      ``figures``
E15-07       45      ``demo``
E15-08       45      ``narrative``
E15-09       45      ``clean_room``
E15-10       45      ``upstream``
E15-11       45      ``first_impression``
===========  ======  ==============================
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from hqsb.core.errors import ConfigError

MODULE_BY_PREFIX: Dict[str, str] = {
    # foundations
    "identity": "hqsb.release.identity",
    "records": "hqsb.release.records",
    "contracts": "hqsb.release.contracts",
    "campaign": "hqsb.release.campaign",
    # experiments
    "claims": "hqsb.release.claims",
    "quickstart": "hqsb.release.quickstart",
    "hero_replay": "hqsb.release.hero_replay",
    "docs_gate": "hqsb.release.docs_gate",
    "supply_chain": "hqsb.release.supply_chain",
    "figures": "hqsb.release.figures",
    "demo": "hqsb.release.demo",
    "narrative": "hqsb.release.narrative",
    "clean_room": "hqsb.release.clean_room",
    "upstream": "hqsb.release.upstream",
    "first_impression": "hqsb.release.first_impression",
    # evidence and scaffolding
    "telemetry": "hqsb.release.telemetry",
    "specs": "hqsb.release.specs",
    "experiment": "hqsb.release.experiment",
}

#: (module name, experiment id, level).
EXPERIMENT_MODULES: Tuple[Tuple[str, str, str], ...] = (
    ("claims", "E15-01", "P0"),
    ("quickstart", "E15-02", "P0"),
    ("hero_replay", "E15-03", "P0"),
    ("docs_gate", "E15-04", "P0"),
    ("supply_chain", "E15-05", "P0"),
    ("figures", "E15-06", "P0"),
    ("demo", "E15-07", "P0"),
    ("narrative", "E15-08", "P0"),
    ("clean_room", "E15-09", "P0"),
    ("upstream", "E15-10", "P0"),
    ("first_impression", "E15-11", "P1"),
)

EXPECTED_STEPS_PER_EXPERIMENT = 45

#: The driver entry every experiment shares.
DRIVER_TEMPLATE = "scripts/release/run_e15.py --experiment {experiment_id}"


@dataclass
class StepMapping:
    """One protocol step: its title and the interfaces that implement it."""

    index: int
    title: str
    interfaces: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "title": self.title, "interfaces": list(self.interfaces)}


@dataclass
class ExperimentMapping:
    """One experiment: 45 steps, a driver entry and the claim boundary."""

    experiment_id: str
    title: str
    module: str
    level: str = "P0"
    claim_boundary: str = ""
    steps: Tuple[StepMapping, ...] = ()
    load_error: str = ""

    @property
    def driver(self) -> str:
        return DRIVER_TEMPLATE.format(experiment_id=self.experiment_id)

    @property
    def complete(self) -> bool:
        return (
            not self.load_error
            and len(self.steps) == EXPECTED_STEPS_PER_EXPERIMENT
            and all(step.interfaces for step in self.steps)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "module": self.module,
            "level": self.level,
            "driver": self.driver,
            "claim_boundary": self.claim_boundary,
            "steps": [step.as_dict() for step in self.steps],
            "complete": self.complete,
            "load_error": self.load_error,
        }


def _load_mapping(module_name: str, experiment_id: str, level: str) -> ExperimentMapping:
    module_path = f"hqsb.release.{module_name}"
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # pragma: no cover - the report is the error path
        return ExperimentMapping(
            experiment_id=experiment_id,
            title="",
            module=module_path,
            level=level,
            load_error=f"cannot import module: {type(exc).__name__}: {exc}",
        )
    steps_raw = getattr(module, "PROTOCOL_STEPS", ())
    title = getattr(module, "TITLE", "")
    claim_boundary = getattr(module, "CLAIM_BOUNDARY", "")
    steps = tuple(
        StepMapping(index=int(row[0]), title=str(row[1]), interfaces=tuple(str(item) for item in row[2]))
        for row in steps_raw
    )
    return ExperimentMapping(
        experiment_id=experiment_id,
        title=title,
        module=module_path,
        level=level,
        claim_boundary=claim_boundary,
        steps=steps,
    )


EXPERIMENTS: Tuple[ExperimentMapping, ...] = tuple(
    _load_mapping(module_name, experiment_id, level) for module_name, experiment_id, level in EXPERIMENT_MODULES
)


def mapping_for(experiment_id: str) -> ExperimentMapping:
    for mapping in EXPERIMENTS:
        if mapping.experiment_id == experiment_id:
            return mapping
    raise ConfigError(f"unknown S15 experiment {experiment_id!r}")


def total_steps() -> int:
    return sum(len(mapping.steps) for mapping in EXPERIMENTS)


def _resolve_symbol(reference: str) -> Tuple[bool, str]:
    """Resolve ``module[.attr[.attr...]]`` to a live object.

    ``module`` is resolved through :data:`MODULE_BY_PREFIX`; the remaining path is
    walked with ``getattr`` so class attributes (``records.HeroReplayResult.quality_status``)
    and nested constants resolve too.
    """
    parts = reference.split(".")
    if not parts or not parts[0]:
        return False, f"empty interface reference {reference!r}"
    head = parts[0]
    module_path = MODULE_BY_PREFIX.get(head, f"hqsb.release.{head}")
    try:
        obj: Any = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001
        return False, f"cannot import {module_path!r}: {type(exc).__name__}: {exc}"
    for index, attr in enumerate(parts[1:], start=1):
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            # A dataclass instance field is not a class attribute (especially with
            # ``field(default_factory=...)``), but it is still part of the
            # interface: ``records.HeroReplayResult.primary_effect`` resolves to a
            # real field.  Accept it only when it is a declared dataclass field.
            if isinstance(obj, type) and attr in getattr(obj, "__dataclass_fields__", {}):
                continue
            return False, f"{reference!r}: {'.'.join(parts[: index + 1])!r} has no attribute {attr!r}"
    return True, ""


def resolve_interfaces() -> Dict[str, Any]:
    """Resolve every step interface; return the aggregate report."""
    experiments: List[Dict[str, Any]] = []
    steps_total = 0
    steps_without = 0
    interfaces_total = 0
    references_total = 0
    failures: List[str] = []
    ok = True
    for mapping in EXPERIMENTS:
        step_rows: List[Dict[str, Any]] = []
        for step in mapping.steps:
            steps_total += 1
            resolved: List[Dict[str, Any]] = []
            if not step.interfaces:
                steps_without += 1
                ok = False
            for reference in step.interfaces:
                references_total += 1
                good, error = _resolve_symbol(reference)
                resolved.append({"reference": reference, "ok": good, "error": error})
                if not good:
                    ok = False
                    failures.append(f"{mapping.experiment_id} step {step.index}: {error}")
            step_rows.append({"index": step.index, "title": step.title, "interfaces": resolved})
        experiments.append(
            {
                "experiment_id": mapping.experiment_id,
                "title": mapping.title,
                "level": mapping.level,
                "complete": mapping.complete,
                "load_error": mapping.load_error,
                "steps": step_rows,
            }
        )
    return {
        "experiments": experiments,
        "steps": steps_total,
        "expected_steps": len(EXPERIMENTS) * EXPECTED_STEPS_PER_EXPERIMENT,
        "steps_without_interfaces": steps_without,
        "interfaces": interfaces_total,
        "references": references_total,
        "ok": ok,
        "failures": failures,
    }


def steps_without_interfaces() -> List[Tuple[str, int]]:
    """Every experiment step that has no interface at all."""
    missing: List[Tuple[str, int]] = []
    for mapping in EXPERIMENTS:
        for step in mapping.steps:
            if not step.interfaces:
                missing.append((mapping.experiment_id, step.index))
    return missing


def coverage_summary() -> Dict[str, Any]:
    """Per-experiment step coverage (all 45 steps must map to ≥1 interface)."""
    rows: List[Dict[str, Any]] = []
    for mapping in EXPERIMENTS:
        rows.append(
            {
                "experiment_id": mapping.experiment_id,
                "steps": len(mapping.steps),
                "expected": EXPECTED_STEPS_PER_EXPERIMENT,
                "with_interfaces": sum(1 for step in mapping.steps if step.interfaces),
                "complete": mapping.complete,
                "load_error": mapping.load_error,
            }
        )
    return {
        "experiments": rows,
        "total_steps": total_steps(),
        "expected_steps": len(EXPERIMENTS) * EXPECTED_STEPS_PER_EXPERIMENT,
        "missing": [f"{experiment_id}:{index}" for experiment_id, index in steps_without_interfaces()],
    }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the interface map (labelled smoke)."""
    result = resolve_interfaces()
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiments": len(EXPERIMENTS),
        "steps": result["steps"],
        "expected_steps": result["expected_steps"],
        "references": result["references"],
        "ok": result["ok"],
        "failures": result["failures"][:5],
        "steps_without_interfaces": result["steps_without_interfaces"],
    }
