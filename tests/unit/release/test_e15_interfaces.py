"""Interface-completeness tests for all eleven S15 experiments.

Three things are verified per experiment, all at the *interface* layer (no
execution, no conclusion):

1. the experiment module imports and its ``smoke_self_check`` returns no problems;
2. its ``PROTOCOL_STEPS`` table has exactly 45 steps and every step carries at
   least one interface reference;
3. every interface reference resolves to a live object (the mapping cannot rot).
"""

from __future__ import annotations

import importlib
from typing import Tuple

import pytest

from hqsb.release import interface_map as imap

EXPERIMENTS: Tuple[Tuple[str, str, str], ...] = imap.EXPERIMENT_MODULES

MODULE_BY_ID = {experiment_id: name for name, experiment_id, _level in EXPERIMENTS}


@pytest.mark.parametrize("experiment_id", sorted(MODULE_BY_ID))
def test_experiment_module_imports_and_smokes(experiment_id: str) -> None:
    module = importlib.import_module(f"hqsb.release.{MODULE_BY_ID[experiment_id]}")
    result = getattr(module, "smoke_self_check")()
    assert result.get("claim_allowed") is False
    assert result.get("problems") == []


@pytest.mark.parametrize("experiment_id", sorted(MODULE_BY_ID))
def test_step_table_is_complete(experiment_id: str) -> None:
    mapping = imap.mapping_for(experiment_id)
    assert mapping.load_error == "", mapping.load_error
    assert len(mapping.steps) == imap.EXPECTED_STEPS_PER_EXPERIMENT
    assert all(step.interfaces for step in mapping.steps), (
        f"{experiment_id} has steps without interfaces: "
        f"{[step.index for step in mapping.steps if not step.interfaces]}"
    )
    assert mapping.complete


def test_all_steps_resolve() -> None:
    result = imap.resolve_interfaces()
    assert result["steps"] == result["expected_steps"] == 495
    assert result["ok"], result["failures"][:5]
    assert result["steps_without_interfaces"] == 0


def test_driver_reports_no_conclusion_by_default() -> None:
    """The driver entry must never write a conclusion in its default state."""
    from hqsb.release import experiment as exp
    from hqsb.release import records as rec

    run = exp.RunDirectory("/tmp/hqsb-s15-e15-01-smoke", "E15-01", "interface_only")
    run.create()
    verdict = run.write_status(rec.STATUS_BLOCKED, "prerequisites unsatisfied", None)
    assert verdict["conclusion"] is False
    assert verdict["status"] == rec.STATUS_BLOCKED
