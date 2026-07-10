"""Every S14 experiment exposes a complete, callable, conclusion-free interface.

The tests here are the executable form of the task's central requirement: *a
reader who has only read one experiment file must be able to run its steps
without writing any capability code*.  Concretely, for each of the twelve
experiments we assert that:

* the module imports and exposes ``EXPERIMENT_ID``/``TITLE``/``LEVEL``/
  ``CLAIM_BOUNDARY`` and a ``PROTOCOL_STEPS`` table with **exactly 40** steps;
* every step references at least one interface that resolves to a real symbol
  (the resolver lives in ``hqsb.experimental.interface_map``, so a renamed
  function fails here rather than silently pointing at nothing);
* ``smoke_self_check()`` runs, is labelled ``smoke`` and carries
  ``claim_allowed=False`` — an interface self-check must not look like a result;
* the module declares the *negative* path that its protocol requires (a wrong
  artefact, an unknown flag, a truncated episode …).  A gate that has never been
  shown to reject something is not a gate.

Nothing here executes an experiment.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pytest

from hqsb.experimental import interface_map as imap

EXPERIMENT_IDS: List[str] = [mapping.experiment_id for mapping in imap.EXPERIMENTS]

#: Experiment → a callable that must *reject* a bad input.  The negative control
#: is named per experiment so a missing one is a failing test, not a paragraph.
NEGATIVE_CONTROLS: Dict[str, str] = {
    "E14-01": "unknown_flag_rejected",
    "E14-02": "single_device_status",
    "E14-03": "silent_overwrite_rejected",
    "E14-04": "sync_illegal_transition_detected",
    "E14-05": "blocked_prerequisite_status",
    "E14-F1": "guard_action",
    "E14-F2": "balanced_gini",
    "E14-F3": "fairness_violation_detected",
    "E14-F4": "noncompliant_rejected",
    "E14-06": "llm_metric_rejected",
    "E14-07": "bad_call_rejected",
    "E14-08": "map_only_record_problems",
}


def _module(experiment_id: str) -> Any:
    mapping = imap.mapping_for(experiment_id)
    return importlib.import_module(mapping.module)


@pytest.mark.unit
@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_module_declares_its_identity(experiment_id: str) -> None:
    module = _module(experiment_id)
    assert module.EXPERIMENT_ID == experiment_id
    assert module.TITLE, f"{experiment_id} has no TITLE"
    assert module.LEVEL, f"{experiment_id} has no LEVEL"
    assert module.CLAIM_BOUNDARY, f"{experiment_id} has no CLAIM_BOUNDARY"
    # The claim boundary must name what the experiment does *not* prove.
    assert any(
        token in module.CLAIM_BOUNDARY for token in ("不", "仅", "只")
    ), f"{experiment_id} CLAIM_BOUNDARY does not bound the claim: {module.CLAIM_BOUNDARY!r}"


@pytest.mark.unit
@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_step_table_is_complete(experiment_id: str) -> None:
    module = _module(experiment_id)
    steps = list(module.PROTOCOL_STEPS)
    assert len(steps) == imap.EXPECTED_STEPS_PER_EXPERIMENT, (
        f"{experiment_id} maps {len(steps)} steps, expected {imap.EXPECTED_STEPS_PER_EXPERIMENT}"
    )
    assert [step[0] for step in steps] == list(range(1, 41)), (
        f"{experiment_id} step indices are not 1..40"
    )
    for index, title, interfaces in steps:
        assert title, f"{experiment_id} step {index} has no title"
        assert interfaces, f"{experiment_id} step {index} has no interface"

    result = imap.resolve_interfaces(experiment_id=experiment_id)
    assert result["ok"], f"{experiment_id} has unresolvable interface references: {result['failures'][:3]}"
    assert result["steps"] == imap.EXPECTED_STEPS_PER_EXPERIMENT


@pytest.mark.unit
@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_smoke_self_check_is_labelled_smoke(experiment_id: str) -> None:
    module = _module(experiment_id)
    payload = module.smoke_self_check()
    assert payload.get("status") == "smoke", f"{experiment_id} smoke is not labelled smoke"
    assert payload.get("claim_allowed") is False, (
        f"{experiment_id} smoke self-check claims a result; it must be an interface check only"
    )
    assert payload.get("experiment_id") == experiment_id


@pytest.mark.unit
@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_experiment_has_a_negative_control(experiment_id: str) -> None:
    module = _module(experiment_id)
    key = NEGATIVE_CONTROLS[experiment_id]
    payload = module.smoke_self_check()
    assert key in payload, (
        f"{experiment_id}: the negative control {key!r} is missing from the smoke output "
        "(a gate that was never shown to reject anything is not a gate)"
    )
    observed = payload[key]
    # Either a bool that is True (a rejection was observed) or a non-empty
    # problem/status value; a silently empty list is not a rejection.
    if isinstance(observed, bool):
        assert observed, f"{experiment_id}: negative control {key!r} did not fire"
    elif isinstance(observed, (list, tuple)):
        assert observed, f"{experiment_id}: negative control {key!r} reported nothing"
    else:
        assert observed not in ("", None, []), f"{experiment_id}: negative control {key!r} is empty"


@pytest.mark.unit
def test_no_smoke_check_emits_a_conclusion_status() -> None:
    """A self-check may *contain* the status vocabulary but must not *emit* a conclusion.

    The step tables legitimately reference ``records:STATUS_PASS_NEGATIVE`` (the
    driver needs the vocabulary to write a negative result), so the invariant that
    matters is about the smoke *output*: an interface self-check that reports
    PASS/FAIL would look like an experiment result.
    """
    from hqsb.experimental import records as rec

    offending_values: List[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, str):
            if value in rec.CONCLUSION_STATUSES:
                offending_values.append(f"{path}={value}")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    for experiment_id in EXPERIMENT_IDS:
        walk(_module(experiment_id).smoke_self_check(), experiment_id)
    assert not offending_values, (
        f"a smoke self-check emitted a conclusion status: {offending_values[:5]}"
    )


@pytest.mark.unit
def test_interface_map_reports_full_coverage() -> None:
    result = imap.resolve_interfaces()
    assert result["experiments"] == 12
    assert result["steps"] == result["expected_steps"] == 480
    assert result["interfaces"] >= 400, "the mapping is thinner than expected"
    assert result["ok"], result["failures"][:5]
    assert imap.steps_without_interfaces() == []
    coverage = imap.coverage_summary()
    assert coverage["ok"]
    for row in coverage["rows"]:
        assert row["complete"], row
        assert len(row["modules"]) >= 2, f"{row['experiment_id']} touches fewer than two modules"


@pytest.mark.unit
def test_each_experiment_has_a_driver_entry() -> None:
    for mapping in imap.EXPERIMENTS:
        assert mapping.driver.startswith("scripts/experimental/run_e14.py --experiment")
        assert mapping.experiment_id in mapping.driver
    assert "E14-F1" in imap.mapping_for("E14-F1").driver
