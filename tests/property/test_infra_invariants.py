"""Property/invariant tests for the S13 infra layer.

Each test states an invariant that must hold for *every* input, not just for the
examples in the unit tests; failures here mean a contract the experiments rely on
(e.g. "a missing value is never a number", "an over-budget request is never
admitted") can be violated by some configuration.
"""

from __future__ import annotations

import random

import pytest

from hqsb.core.errors import ConfigError
from hqsb.infra import (
    artifacts,
    canary,
    capacity,
    contracts,
    identity,
    records,
    security,
)

SEEDS = (0, 1, 7, 42, 1337)
DIGEST = "sha256:" + "f" * 64


@pytest.mark.property
@pytest.mark.parametrize("seed", SEEDS)
def test_missing_identity_field_is_always_detected(seed: int) -> None:
    rng = random.Random(seed)
    required = list(identity.REQUIRED_IDENTITY_FIELDS)
    dropped = rng.sample(required, k=rng.randint(1, len(required)))
    payload = {
        "release_id": "rel",
        "source_commit": "commit",
        "image_index_digest": DIGEST,
        "platform_image_digests": {"linux/amd64": DIGEST},
        "service_config_hash": DIGEST,
        "model_artifact_id": "model",
        "tokenizer_id": "tok",
        "deployment_template_digest": DIGEST,
        "sbom_ids": ("sbom",),
    }
    for name in dropped:
        payload[name] = "" if name != "platform_image_digests" else {}
    bundle = identity.ReleaseBundle(**payload)
    problems = bundle.validate()
    if "platform_image_digests" in dropped:
        assert bundle.identity_complete() is False
    else:
        assert any(name in problem for problem in problems for name in dropped), (dropped, problems)
        assert bundle.identity_complete() is False


@pytest.mark.property
@pytest.mark.parametrize("seed", SEEDS)
def test_state_machine_walks_are_closed_under_legal_transitions(seed: int) -> None:
    rng = random.Random(seed)
    machine = records.REQUEST_STATE_MACHINE
    successors = {}
    for state_from, state_to in machine.transitions:
        successors.setdefault(state_from, []).append(state_to)
    path = ["RECEIVED"]
    for _ in range(rng.randint(1, 12)):
        options = successors.get(path[-1], [])
        if not options:
            break
        path.append(rng.choice(options))
    assert machine.walk(path)["ok"] is True
    # An injected illegal hop must be detected regardless of where it is inserted.
    illegal = ["RECEIVED", "ACCOUNTED_RELEASED"] + path[1:]
    assert machine.walk(illegal)["ok"] is False


@pytest.mark.property
def test_missingness_codes_are_never_numbers() -> None:
    for code in records.MISSINGNESS_CODES:
        with pytest.raises(ConfigError):
            records.numeric_value(code)
        with pytest.raises(ConfigError):
            records.numeric_value(1.0, state=code)


@pytest.mark.property
@pytest.mark.parametrize("seed", SEEDS)
def test_kv_estimate_is_monotone_and_respects_block_rounding(seed: int) -> None:
    rng = random.Random(seed)
    model = capacity.KVModel(
        layers=rng.randint(1, 64),
        kv_heads=rng.randint(1, 8),
        head_dim=rng.choice([64, 128, 256]),
        bytes_per_element=rng.choice([1, 2]),
        block_tokens=rng.choice([8, 16, 32]),
        block_bytes_overhead=rng.randint(1, 256),
    )
    budget = rng.randint(1, 4096)
    smaller = model.incremental_bytes(budget)
    larger = model.incremental_bytes(budget + model.block_tokens)
    assert larger > smaller
    assert smaller >= budget * model.bytes_per_token()
    shared = model.incremental_bytes(budget, prefix_shared_tokens=budget // 2)
    assert shared <= smaller


@pytest.mark.property
@pytest.mark.parametrize("seed", SEEDS)
def test_over_budget_requests_are_never_admitted(seed: int) -> None:
    rng = random.Random(seed)
    kv = capacity.KVModel(layers=8, kv_heads=4, head_dim=128, block_tokens=16, block_bytes_overhead=64)
    margin = rng.randint(1, 1 << 20)
    policy = capacity.AdmissionPolicy(
        policy_id="p", kind="MEMORY_KV_PREDICTED", safety_margin_bytes=margin,
        token_budget=10**9, version="v1",
    )
    usable = rng.randint(margin, margin + (1 << 20))
    decision = capacity.AdmissionDecision(
        decision_id="d", request_id="r", policy_version="v1",
        prompt_tokens=rng.randint(1, 4096), requested_output_tokens=rng.randint(1, 4096),
        usable_memory_bytes=usable, safety_margin_bytes=margin, release_id="rel", model_artifact_id="m",
    )
    capacity.admit(policy=policy, kv_model=kv, decision=decision)
    if decision.decision == "ADMIT_NOW":
        assert decision.predicted_peak_memory_bytes <= max(usable - margin, 0)
    else:
        assert decision.reason_code in capacity.REASON_CODES


@pytest.mark.property
@pytest.mark.parametrize("seed", SEEDS)
def test_admission_decision_validator_matches_the_admit_function(seed: int) -> None:
    rng = random.Random(seed)
    kv = capacity.KVModel(layers=4, kv_heads=2, head_dim=64, block_tokens=16, block_bytes_overhead=8)
    policy = capacity.AdmissionPolicy(policy_id="p", kind="MEMORY_KV_PREDICTED",
                                      safety_margin_bytes=1 << 10, version="v1")
    decision = capacity.AdmissionDecision(
        decision_id="d", request_id="r", policy_version="v1", prompt_tokens=rng.randint(1, 512),
        requested_output_tokens=rng.randint(1, 512), usable_memory_bytes=1 << 16,
        safety_margin_bytes=1 << 10, release_id="rel", model_artifact_id="m",
    )
    capacity.admit(policy=policy, kv_model=kv, decision=decision)
    result = contracts.validate_admission_decision(decision.as_decision_row())
    assert result.ok is True, result.as_dict()


@pytest.mark.property
@pytest.mark.parametrize("seed", SEEDS)
def test_canary_assignment_is_deterministic_and_monotone(seed: int) -> None:
    rng = random.Random(seed)
    policy = canary.AssignmentPolicy(
        policy_id="a", unit="SESSION_STICKY", hash_key="session_id", seed=str(seed),
        eligibility_rule="authenticated", retry_attribution="first attempt",
    )
    units = [f"session-{index}-{rng.randint(0, 10**6)}" for index in range(50)]
    low = {unit: policy.assign(unit_key=unit, traffic_fraction=0.1) for unit in units}
    high = {unit: policy.assign(unit_key=unit, traffic_fraction=0.4) for unit in units}
    again = {unit: policy.assign(unit_key=unit, traffic_fraction=0.1) for unit in units}
    assert low == again
    promoted = [unit for unit in units if low[unit] == "candidate" and high[unit] != "candidate"]
    assert promoted == []
    with pytest.raises(ConfigError):
        policy.assign(unit_key="x", traffic_fraction=1.5)


@pytest.mark.property
@pytest.mark.parametrize("seed", SEEDS)
def test_quota_ledger_oversell_matches_the_definition(seed: int) -> None:
    rng = random.Random(seed)
    limit = rng.randint(1, 10**6)
    committed = rng.randint(0, 2 * 10**6)
    ledger = security.QuotaLedger(tenant_id="t", resource="output_tokens", limit=float(limit),
                                  committed=float(committed))
    expected = max(committed - limit, 0)
    assert ledger.oversell() == expected
    assert (ledger.validate() == []) == (expected == 0)


@pytest.mark.property
def test_dropping_any_required_field_is_detected() -> None:
    sample_rows = {
        "admission_decision": {
            "decision_id": "d", "request_id": "r", "timestamp": 1.0, "policy_version": "v",
            "prompt_tokens": 1, "requested_output_tokens": 1, "predicted_incremental_kv_bytes": 1,
            "usable_memory_bytes": 1, "safety_margin_bytes": 1, "decision": "ADMIT_NOW",
            "reason_code": "ADMIT_WITHIN_BUDGET",
        },
        "fault_spec": {
            "fault_case_id": "F", "hypothesis": "h", "layer": "pod_container", "mechanism": "POD_DELETE",
            "target_selector": "pod/x", "resolved_targets": ("x",), "blast_radius": "one",
            "safety_policy_id": "s", "expected_detection": "d", "expected_degradation": "g",
            "expected_recovery": "r", "abort_threshold": "a",
        },
    }
    for table, row in sample_rows.items():
        assert records.validate_rows(table, [row]) == []
        for field_name in records.TABLE_SCHEMAS[table]:
            reduced = {key: value for key, value in row.items() if key != field_name}
            problems = records.validate_rows(table, [reduced])
            if field_name in row:
                assert problems, (table, field_name)
            else:
                assert problems, (table, field_name)


@pytest.mark.property
@pytest.mark.parametrize("seed", SEEDS)
def test_artifact_lifecycle_only_reaches_verified_states_legally(seed: int) -> None:
    rng = random.Random(seed)
    machine = records.ARTIFACT_STATE_MACHINE
    for _ in range(20):
        state_from = rng.choice(records.ARTIFACT_LIFECYCLE_STATES)
        state_to = rng.choice(records.ARTIFACT_LIFECYCLE_STATES)
        allowed = machine.allowed(state_from, state_to)
        event = artifacts.ArtifactLifecycleEvent(
            event_id="e", attempt_id="a", artifact_id="art", state_from=state_from, state_to=state_to,
            verification_status="OK" if state_to in ("VERIFIED_IMMUTABLE",) else "",
            durability_status="durable" if state_to == "VERIFIED_IMMUTABLE" else "",
        )
        problems = event.validate()
        if not allowed:
            assert any("illegal artifact transition" in problem for problem in problems)
        assert artifacts.assert_loadable(state_to)["ok"] == (state_to in records.LOADABLE_ARTIFACT_STATES)
