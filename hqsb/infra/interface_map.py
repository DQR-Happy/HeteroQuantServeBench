"""「实验步骤 → 代码接口」对照表 for all eleven S13 experiments (418 steps).

Each experiment's 38 protocol steps are mapped to concrete interfaces of this
package; :func:`resolve_interfaces` imports every referenced symbol so the mapping
cannot rot into documentation — a renamed function fails the audit instead of
silently pointing at nothing.

The per-experiment step tables live with the modules they describe
(``PROTOCOL_STEPS`` in ``supply_chain.py`` … ``security.py``) so the mapping and
the implementation are edited together.  Entries are data only: no interface here
executes an experiment.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

MODULE_BY_PREFIX: Dict[str, str] = {
    # foundations
    "identity": "hqsb.infra.identity",
    "records": "hqsb.infra.records",
    "contracts": "hqsb.infra.contracts",
    "campaign": "hqsb.infra.campaign",
    # experiments
    "supply_chain": "hqsb.infra.supply_chain",
    "deployment": "hqsb.infra.deployment",
    "scheduling": "hqsb.infra.scheduling",
    "artifacts": "hqsb.infra.artifacts",
    "lifecycle": "hqsb.infra.lifecycle",
    "capacity": "hqsb.infra.capacity",
    "autoscaling": "hqsb.infra.autoscaling",
    "observability": "hqsb.infra.observability",
    "faults": "hqsb.infra.faults",
    "canary": "hqsb.infra.canary",
    "security": "hqsb.infra.security",
    # evidence and scaffolding
    "telemetry": "hqsb.infra.telemetry",
    "specs": "hqsb.infra.specs",
    "experiment": "hqsb.infra.experiment",
}

#: (module name, experiment id, level).  Titles/claim boundaries come from the
#: experiment modules themselves so they cannot drift from the protocol files.
EXPERIMENT_MODULES: Tuple[Tuple[str, str, str], ...] = (
    ("supply_chain", "E13-01", "P0"),
    ("deployment", "E13-02", "P0"),
    ("scheduling", "E13-03", "P0"),
    ("artifacts", "E13-04", "P0"),
    ("lifecycle", "E13-05", "P0"),
    ("capacity", "E13-06", "P0"),
    ("autoscaling", "E13-07", "P0"),
    ("observability", "E13-08", "P0"),
    ("faults", "E13-09", "P0"),
    ("canary", "E13-10", "P0"),
    ("security", "E13-11", "P1"),
)

EXPECTED_STEPS_PER_EXPERIMENT = 38

#: Cross-module symbols referenced by step tables that live in a *lower* region.
EXTERNAL_PREFIXES: Mapping[str, str] = {
    "serving": "hqsb.serving",
}


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
    """One experiment: 38 steps, a driver entry and the claim boundary."""

    experiment_id: str
    title: str
    module: str
    level: str = "P0"
    claim_boundary: str = ""
    steps: Tuple[StepMapping, ...] = ()
    load_error: str = ""

    @property
    def driver(self) -> str:
        return f"scripts/infra/run_e13.py --experiment {self.experiment_id}"

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
        module = importlib.import_module(f"hqsb.infra.{module_name}")
    except Exception as exc:  # pragma: no cover - the report is the error path
        return ExperimentMapping(
            experiment_id=experiment_id,
            title="",
            module=f"hqsb.infra.{module_name}",
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
        module=f"hqsb.infra.{module_name}",
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
    module_name = MODULE_BY_PREFIX.get(prefix) or EXTERNAL_PREFIXES.get(prefix)
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
        # Dataclass fields are legitimate interface references as well
        # (``capacity:AdmissionDecision.reason_code`` names the field).
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


def step_table_markdown(experiment_id: str, *, interfaces_per_row: int = 4) -> str:
    """Markdown step table for the development report (generated, not hand-copied)."""
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
        shown = ", ".join(f"`{symbol}`" for symbol in step.interfaces[:interfaces_per_row])
        if len(step.interfaces) > interfaces_per_row:
            shown += f" … (+{len(step.interfaces) - interfaces_per_row})"
        lines.append(f"| {step.index} | {step.title} | {shown} |")
    return "\n".join(lines)


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


def interface_owner(symbol: str) -> str:
    """Return the module a symbol belongs to (used by the report tables)."""
    prefix = symbol.split(":", 1)[0]
    return MODULE_BY_PREFIX.get(prefix) or EXTERNAL_PREFIXES.get(prefix, "")


def covered_prefixes(experiment_id: str) -> List[str]:
    mapping = mapping_for(experiment_id)
    return sorted({symbol.split(":", 1)[0] for step in mapping.steps for symbol in step.interfaces})


def steps_without_interfaces() -> List[Dict[str, Any]]:
    """Any step that lost its interfaces (the audit fails rather than ignoring it)."""
    missing: List[Dict[str, Any]] = []
    for mapping in EXPERIMENTS:
        if len(mapping.steps) != EXPECTED_STEPS_PER_EXPERIMENT:
            missing.append(
                {
                    "experiment_id": mapping.experiment_id,
                    "reason": f"expected {EXPECTED_STEPS_PER_EXPERIMENT} steps, found {len(mapping.steps)}",
                }
            )
        for step in mapping.steps:
            if not step.interfaces:
                missing.append(
                    {"experiment_id": mapping.experiment_id, "reason": f"step {step.index} has no interfaces"}
                )
    return missing
