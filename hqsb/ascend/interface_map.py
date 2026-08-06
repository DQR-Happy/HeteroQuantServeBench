"""Trace every S09 protocol step to its executable evidence surface.

This map is intentionally a *coverage map*, not a maturity claim.  On a host
without Ascend/CANN each step resolves to the campaign collector and receives a
``NOT_RUN``/``COLLECTED_PREFLIGHT_ONLY`` row.  It does not claim that a real
kernel, framework or model path executed.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict

from hqsb.ascend.experiment import EXPERIMENTS, PROTOCOLS, protocol_steps


STEP_INTERFACES = (
    "hqsb.ascend.experiment.collect_campaign",
    "hqsb.ascend.experiment.collect_capability_snapshot",
)


def _resolve(symbol: str) -> bool:
    module_name, _, attribute = symbol.rpartition(".")
    module = importlib.import_module(module_name)
    return hasattr(module, attribute)


def resolve_interfaces(root: str | Path) -> Dict[str, Any]:
    failures = []
    mappings = []
    total_steps = 0
    for experiment_id in EXPERIMENTS:
        steps = protocol_steps(root, experiment_id)
        total_steps += len(steps)
        mappings.append(
            {
                "experiment_id": experiment_id,
                "title": PROTOCOLS[experiment_id].title,
                "steps": len(steps),
                "interfaces": list(STEP_INTERFACES),
                "maturity": "M1_PROTOCOL_ORCHESTRATION",
                "hardware_maturity": "BLOCKED_UNTIL_REAL_ASCEND_RUN",
            }
        )
    for symbol in STEP_INTERFACES:
        try:
            if not _resolve(symbol):
                failures.append({"interface": symbol, "reason": "attribute missing"})
        except Exception as exc:  # noqa: BLE001 - audit must return a structured failure
            failures.append({"interface": symbol, "reason": f"{type(exc).__name__}: {exc}"})
    return {
        "ok": not failures and total_steps == 280,
        "stage": "S09",
        "experiments": len(mappings),
        "steps": total_steps,
        "interfaces": len(STEP_INTERFACES),
        "mappings": mappings,
        "failures": failures,
        "notice": (
            "Resolved means the collection/refusal path exists; it does not mean the "
            "Ascend data-plane step ran or passed."
        ),
    }


def mapping_table_markdown(root: str | Path) -> str:
    report = resolve_interfaces(root)
    lines = [
        "| 实验 | 标题 | 步骤 | 编排成熟度 | 硬件成熟度 |",
        "|---|---|---:|---|---|",
    ]
    for row in report["mappings"]:
        lines.append(
            f"| {row['experiment_id']} | {row['title']} | {row['steps']} | "
            f"{row['maturity']} | {row['hardware_maturity']} |"
        )
    lines.append(f"| 合计 |  | {report['steps']} |  |  |")
    return "\n".join(lines) + "\n"


__all__ = ["STEP_INTERFACES", "mapping_table_markdown", "resolve_interfaces"]
