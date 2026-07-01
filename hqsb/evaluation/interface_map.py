"""「实验步骤 → 代码接口」对照表 for all ten S12 experiments (360 steps).

Each experiment's 36 protocol steps are mapped to concrete interfaces of this
package; :func:`resolve_interfaces` imports every referenced symbol so the
mapping cannot rot into documentation — a renamed function fails the audit
instead of silently pointing at nothing.

The per-experiment step tables live with the modules they describe
(``PROTOCOL_STEPS`` in ``comparability.py`` … ``lineage.py``) so the mapping and
the implementation are edited together.  Entries are data only: no interface
here executes an experiment.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

MODULE_BY_PREFIX: Dict[str, str] = {
    # foundations
    "identity": "hqsb.evaluation.identity",
    "records": "hqsb.evaluation.records",
    "contracts": "hqsb.evaluation.contracts",
    "layers": "hqsb.evaluation.layers",
    "campaign": "hqsb.evaluation.campaign",
    # experiments
    "candidates": "hqsb.evaluation.candidates",
    "comparability": "hqsb.evaluation.comparability",
    "platform": "hqsb.evaluation.platform",
    "capability": "hqsb.evaluation.capability",
    "benchmark": "hqsb.evaluation.benchmark",
    "repeatability": "hqsb.evaluation.repeatability",
    "roofline": "hqsb.evaluation.roofline",
    "energy": "hqsb.evaluation.energy",
    "cost": "hqsb.evaluation.cost",
    "pareto": "hqsb.evaluation.pareto",
    "maturity": "hqsb.evaluation.maturity",
    "lineage": "hqsb.evaluation.lineage",
    # evidence and scaffolding
    "telemetry": "hqsb.evaluation.telemetry",
    "specs": "hqsb.evaluation.specs",
    "experiment": "hqsb.evaluation.experiment",
}

#: (module name, experiment id, level).  Titles/claim boundaries come from the
#: experiment modules themselves so they cannot drift from the protocol files.
EXPERIMENT_MODULES: Tuple[Tuple[str, str, str], ...] = (
    ("comparability", "E12-01", "P0"),
    ("capability", "E12-02", "P0"),
    ("benchmark", "E12-03", "P0"),
    ("repeatability", "E12-04", "P0"),
    ("roofline", "E12-05", "P0"),
    ("energy", "E12-06", "P0"),
    ("cost", "E12-07", "P0"),
    ("pareto", "E12-08", "P0"),
    ("maturity", "E12-09", "P0"),
    ("lineage", "E12-10", "P0"),
)

EXPECTED_STEPS_PER_EXPERIMENT = 36


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
    """One experiment: 36 steps, a driver entry and the claim boundary."""

    experiment_id: str
    title: str
    module: str
    level: str = "P0"
    claim_boundary: str = ""
    steps: Tuple[StepMapping, ...] = ()
    load_error: str = ""

    @property
    def driver(self) -> str:
        return f"scripts/evaluation/run_e12.py --experiment {self.experiment_id}"

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
    try:
        module = importlib.import_module(f"hqsb.evaluation.{module_name}")
    except Exception as exc:  # pragma: no cover - the report is the error path
        return ExperimentMapping(
            experiment_id=experiment_id,
            title="",
            module=f"hqsb.evaluation.{module_name}",
            level=level,
            load_error=f"cannot import module: {type(exc).__name__}: {exc}",
        )
    steps = tuple(
        StepMapping(index=int(index), title=str(title), interfaces=tuple(interfaces))
        for index, title, interfaces in getattr(module, "PROTOCOL_STEPS", ())
    )
    return ExperimentMapping(
        experiment_id=str(getattr(module, "EXPERIMENT_ID", experiment_id)),
        title=str(getattr(module, "TITLE", "")),
        module=f"hqsb.evaluation.{module_name}",
        level=level,
        claim_boundary=str(getattr(module, "CLAIM_BOUNDARY", "")),
        steps=steps,
        load_error="" if steps else "module exposes no PROTOCOL_STEPS",
    )


def experiments() -> Tuple[ExperimentMapping, ...]:
    """Load every experiment mapping (import failures are reported, not raised)."""
    return tuple(
        _load_mapping(module_name, experiment_id, level)
        for module_name, experiment_id, level in EXPERIMENT_MODULES
    )


EXPERIMENTS: Tuple[ExperimentMapping, ...] = experiments()


def mapping_for(experiment_id: str) -> ExperimentMapping:
    for mapping in EXPERIMENTS:
        if mapping.experiment_id == experiment_id:
            return mapping
    raise ConfigError(f"unknown experiment id {experiment_id!r}")


def total_steps() -> int:
    return sum(len(mapping.steps) for mapping in EXPERIMENTS)


def _resolve(symbol: str) -> Optional[str]:
    if ":" not in symbol:
        return f"malformed interface reference {symbol!r} (expected 'module:Symbol')"
    prefix, attribute = symbol.split(":", 1)
    module_name = MODULE_BY_PREFIX.get(prefix)
    if module_name is None:
        return f"unknown module prefix {prefix!r}"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - import failure is the error report
        return f"cannot import {module_name}: {type(exc).__name__}: {exc}"
    target: Any = module
    for part in attribute.split("."):
        if hasattr(target, part):
            target = getattr(target, part)
            continue
        # Dataclass fields (required fields are not class attributes) are also
        # legitimate interface references: ``capability:Feature.success_criteria``
        # names the success_criteria field of the Feature record.
        fields = getattr(target, "__dataclass_fields__", {}) if isinstance(target, type) else {}
        if part in fields:
            target = fields[part]
            continue
        return f"{module_name} has no attribute {part!r} (from {symbol})"
    return None


def resolve_interfaces(*, experiment_id: Optional[str] = None) -> Dict[str, Any]:
    """Import-check every mapped interface; failures are reported, never hidden."""
    failures: List[Dict[str, str]] = []
    steps = 0
    references = 0
    unique: set = set()
    mappings = EXPERIMENTS if experiment_id is None else (mapping_for(experiment_id),)
    for mapping in mappings:
        if mapping.load_error:
            failures.append(
                {
                    "experiment_id": mapping.experiment_id,
                    "step": "0",
                    "symbol": mapping.module,
                    "problem": mapping.load_error,
                }
            )
        for step in mapping.steps:
            steps += 1
            for symbol in step.interfaces:
                references += 1
                unique.add(symbol)
                problem = _resolve(symbol)
                if problem:
                    failures.append(
                        {
                            "experiment_id": mapping.experiment_id,
                            "step": str(step.index),
                            "symbol": symbol,
                            "problem": problem,
                        }
                    )
    return {
        "experiments": len(mappings),
        "steps": steps,
        "expected_steps": len(mappings) * EXPECTED_STEPS_PER_EXPERIMENT,
        "interfaces": len(unique),
        "references": references,
        "failures": failures,
        "ok": not failures,
    }


def mapping_table_markdown() -> str:
    lines = [
        "| 实验 | 级别 | 步骤数 | 驱动入口 | 覆盖能力（示例） |",
        "|---|---|---|---|---|",
    ]
    for mapping in EXPERIMENTS:
        sample = ", ".join(mapping.steps[0].interfaces[:2]) if mapping.steps else ""
        lines.append(
            f"| {mapping.experiment_id} | {mapping.level} | {len(mapping.steps)} | "
            f"`{mapping.driver}` | {sample} … |"
        )
    return "\n".join(lines)


def step_table_for(experiment_id: str) -> str:
    mapping = mapping_for(experiment_id)
    lines = [
        f"### {mapping.experiment_id} {mapping.title}",
        "",
        f"claim boundary: {mapping.claim_boundary}",
        "",
        "| # | 步骤 | 代码接口 |",
        "|---|---|---|",
    ]
    for step in mapping.steps:
        lines.append(f"| {step.index} | {step.title} | {', '.join(step.interfaces)} |")
    return "\n".join(lines)


def step_rows(experiment_id: str) -> Sequence[Mapping[str, Any]]:
    """Flatten one experiment's steps into report-ready rows."""
    mapping = mapping_for(experiment_id)
    return [
        {
            "experiment_id": mapping.experiment_id,
            "step": step.index,
            "title": step.title,
            "interfaces": list(step.interfaces),
        }
        for step in mapping.steps
    ]


def interface_owner(symbol: str) -> str:
    """Return the module a symbol belongs to (used by the report tables)."""
    prefix = symbol.split(":", 1)[0]
    return MODULE_BY_PREFIX.get(prefix, "")
