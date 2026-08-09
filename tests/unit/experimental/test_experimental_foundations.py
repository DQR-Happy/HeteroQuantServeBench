"""Foundations of the S14 layer: identity, vocabularies, contracts, gates.

These tests are where the task's hard rules become executable:

* **no silent degradation** — every fallback must carry requested/actual/reason;
* **missing is a state, not a zero** — a NaN cannot be canonicalised, an
  unmeasured budget entry is refused, an unmeasured resource key is reported;
* **quality before performance** — a performance row whose correctness or quality
  gate did not pass is refused;
* **no conclusion by default** — the triple gate refuses PASS/FAIL without
  ``--execute`` **and** satisfied prerequisites **and** raw samples, and the
  protocol tree under ``docs/stage_experiments/**`` is not writable from a run.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from hqsb.core.errors import ConfigError
from hqsb.experimental import campaign as camp
from hqsb.experimental import contracts as ct
from hqsb.experimental import dependencies as dep
from hqsb.experimental import experiment as exp
from hqsb.experimental import identity as ident
from hqsb.experimental import records as rec
from hqsb.experimental import specs as spec_mod
from hqsb.experimental import telemetry as tel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SPEC_DIR = os.path.join(REPO_ROOT, "configs", "experimental")


@pytest.mark.unit
def test_wheel_audit_normalises_distribution_name_case() -> None:
    policy = dep.DependencyPolicy(entries=(
        dep.DependencyEntry(
            name="PyYAML", layer="core", owner="hqsb.core", purpose="config",
            license="MIT", platforms=("any",), allowed_import_layer="module_level",
        ),
    ))
    metadata = dep.WheelMetadata(
        name="hqsb", version="0", requires_dist=("PyYAML>=5.4",),
        provides_extra=tuple(rec.EXTRA_FEATURE_MAPPING),
    )
    audit = dep.audit_wheel_metadata(metadata, policy)
    assert audit["leaks"] == []
    assert audit["missing_core_requirements"] == []


# ── identity ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_canonical_digest_is_order_independent_and_content_bound() -> None:
    assert ident.canonical_digest({"a": 1, "b": 2}) == ident.canonical_digest({"b": 2, "a": 1})
    assert ident.canonical_digest({"a": 1}) != ident.canonical_digest({"a": 2})
    assert ident.is_digest(ident.canonical_digest({"a": 1}))


@pytest.mark.unit
def test_non_finite_numbers_cannot_be_canonicalised() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ConfigError):
            ident.canonical_digest({"metric": value})


@pytest.mark.unit
def test_seed_bundle_requires_every_role() -> None:
    with pytest.raises(ConfigError):
        ident.seed_bundle(python=1, numpy=1)
    bundle = ident.seed_bundle(**{role: 7 for role in ident.SEED_ROLES})
    assert bundle.derive("python", rank=0) == bundle.derive("python", rank=0)
    assert bundle.derive("python", rank=1) != bundle.derive("python", rank=0)
    with pytest.raises(ConfigError):
        bundle.derive("does_not_exist")


@pytest.mark.unit
def test_rank_identity_rejects_a_shared_device() -> None:
    good = [
        ident.RankIdentity(0, 0, 2, "h1", "gpu0"),
        ident.RankIdentity(1, 1, 2, "h1", "gpu1"),
    ]
    assert ident.assert_rank_identities(good) == []
    bad = [
        ident.RankIdentity(0, 0, 2, "h1", "gpu0"),
        ident.RankIdentity(1, 1, 2, "h1", "gpu0"),
    ]
    problems = ident.assert_rank_identities(bad)
    assert any("more than one rank" in problem for problem in problems)


@pytest.mark.unit
def test_lineage_detects_cycles_and_duplicate_producers() -> None:
    chain = ident.LineageChain()
    chain.add("gather", "tool", ["checkpoint::" + "0" * 64], "serving::" + "1" * 64)
    # The source checkpoint is an external root, not a dangling input.
    assert chain.validate() == []
    assert chain.roots() == ["checkpoint::" + "0" * 64]
    assert chain.validate(strict_roots=True), "strict_roots must flag the external input"
    assert chain.reachable_from("checkpoint::" + "0" * 64)
    cyclic = ident.LineageChain()
    cyclic.add("a", "t", ["x"], "y")
    cyclic.add("b", "t", ["y"], "x")
    assert any("cycle" in problem for problem in cyclic.validate())


# ── records ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_state_machines_are_structurally_valid() -> None:
    assert rec.validate_state_machines() == []
    assert rec.is_valid_transition("conversion_node", "PINNED", "INPUTS_VERIFIED")
    assert not rec.is_valid_transition("conversion_node", "PINNED", "PUBLISHED")
    assert not rec.is_valid_transition("agent_workflow", "COMPLETED", "CREATED")


@pytest.mark.unit
def test_maturity_levels_are_ordered() -> None:
    assert rec.maturity_rank(rec.MATURITY_DESIGN_ONLY) < rec.maturity_rank(rec.MATURITY_KERNEL_PROFILED)
    assert rec.maturity_is_lower_than(rec.MATURITY_SOURCE_INTEGRATED, rec.MATURITY_SERVICE_PROFILED)
    assert set(rec.MATURITY_CLAIM) == set(rec.MATURITY_LEVELS)


@pytest.mark.unit
def test_negative_status_only_covers_performance_hypotheses() -> None:
    assert "quality-gate" in rec.NEGATIVE_NOT_ALLOWED_FOR
    assert "correctness" in rec.NEGATIVE_NOT_ALLOWED_FOR
    assert set(rec.S14_FAILURE_STATUSES) <= set(rec.ALL_STATUSES)
    # Bare FAIL comes from the manual §3 checklist; §21 of the S14 README speaks in
    # specific failures (FAIL_CORRECTNESS/FAIL_QUALITY/...).  Both are legitimate,
    # and every one of them is a conclusion the triple gate must refuse.
    assert {"FAIL", rec.STATUS_PASS, rec.STATUS_PASS_NEGATIVE} <= set(rec.CONCLUSION_STATUSES)
    assert rec.STATUS_FAIL_CORRECTNESS in rec.S14_FAILURE_STATUSES
    assert rec.STATUS_FAIL not in rec.S14_FAILURE_STATUSES


@pytest.mark.unit
def test_experiment_record_template_starts_at_not_started() -> None:
    record = rec.experiment_record_template("E14-F3")
    assert record["stage"] == "S14"
    assert record["experiment_id"] == "E14-F3"
    assert record["status"] == rec.STATUS_NOT_STARTED


# ── contracts ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_new_contracts_validate_and_reject_a_conclusion() -> None:
    good = ct.FrontierStudyContract(
        study_id="s1", selected_branch="E14-F2", primary_estimand="goodput change",
        primary_hypothesis="h", minimum_effect="1%", baseline_id="b", candidate_id="c",
        intended_difference="placement", quality_gate_id="q", holdout_id="ho",
        adoption_rule_id="a",
    )
    assert any("source version" in problem for problem in good.validate())
    bad = ct.FrontierStudyContract(
        study_id="s1", selected_branch="E14-F2", primary_estimand="x", primary_hypothesis="y",
        minimum_effect="1%", baseline_id="b", candidate_id="c", intended_difference="d",
        quality_gate_id="q", holdout_id="ho", adoption_rule_id="a", status=rec.STATUS_PASS,
    )
    assert any("conclusion" in problem for problem in bad.validate())


@pytest.mark.unit
def test_quality_must_precede_performance() -> None:
    assert ct.check_quality_before_performance(
        correctness_status=rec.STATUS_FAIL_CORRECTNESS, quality_status=rec.STATUS_PASS,
        performance_eligible=True,
    )
    assert ct.check_quality_before_performance(
        correctness_status=rec.STATUS_PASS, quality_status=rec.STATUS_FAIL_QUALITY,
        performance_eligible=True,
    )
    assert ct.check_quality_before_performance(
        correctness_status=rec.STATUS_PASS, quality_status=rec.STATUS_PASS, performance_eligible=True
    ) == []


@pytest.mark.unit
def test_actual_path_must_be_recorded() -> None:
    problems = ct.check_actual_path_recorded({"requested_implementation": "moe"})
    assert problems, "a sample without its actual path must be refused"
    assert ct.check_actual_path_recorded(
        {
            "requested_implementation": "moe",
            "actual_implementation": "moe",
            "feature_flags": {"frontier.moe": True},
        }
    ) == []


@pytest.mark.unit
def test_silent_degradation_is_detected() -> None:
    problems = ct.check_no_silent_degradation([{"fallback": True, "requested": "sparse"}])
    assert problems, "a fallback without actual/reason is silent degradation"
    assert ct.check_no_silent_degradation(
        [{"fallback": True, "requested": "sparse", "actual": "dense", "reason_code": "PATTERN_UNSUPPORTED"}]
    ) == []
    unknown = ct.check_no_silent_degradation(
        [{"fallback": True, "requested": "a", "actual": "b", "reason_code": "MADE_UP"}]
    )
    assert any("unknown reason code" in problem for problem in unknown)


@pytest.mark.unit
def test_negative_controls_must_cover_the_required_families() -> None:
    assert ct.check_negative_control_coverage(["baseline"])
    complete = ["baseline", "candidate", "feature_off", "wrong_artifact", "unsupported_shape"]
    assert ct.check_negative_control_coverage(complete) == []


@pytest.mark.unit
def test_two_layers_are_required_and_unknown_layers_are_refused() -> None:
    assert ct.check_profile_layering(["operator_kernel"])
    assert ct.check_profile_layering(["operator_kernel", "runtime"]) == []
    assert ct.check_profile_layering(["operator_kernel", "made_up"])


@pytest.mark.unit
def test_resource_ledger_keys_are_required_even_when_unknown() -> None:
    empty = ct.check_resource_ledger({})
    assert len(empty) >= 10
    full = {name: None for name in (
        "device_memory_allocated_bytes", "device_memory_reserved_bytes", "device_memory_peak_bytes",
        "host_memory_bytes", "pinned_memory_bytes", "kv_or_cache_bytes", "collective_bytes",
        "power_w", "energy_j", "device_count",
    )}
    assert ct.check_resource_ledger(full) == []


@pytest.mark.unit
def test_adoption_decision_refuses_claims_for_non_adoptions() -> None:
    decision = ct.AdoptionDecision(
        decision_id="d1", experiment_id="E14-F4", decision=rec.REJECT_NO_BENEFIT,
        allowed_claims=("we are faster",), evidence_refs=("raw/x.parquet",),
    )
    problems = decision.validate()
    assert any("may not carry allowed_claims" in problem for problem in problems)


# ── campaign ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_run_layout_is_under_artifacts_and_the_protocol_tree_is_read_only() -> None:
    layout = camp.run_layout("E14-03", "r1", root=REPO_ROOT)
    assert layout["_base"].endswith(os.path.join("artifacts", "S14", "E14-03", "r1"))
    camp.assert_writable(layout["raw"], root=REPO_ROOT)
    with pytest.raises(ConfigError):
        camp.assert_writable(os.path.join(camp.PROTOCOL_ROOT, "S14_实验清单.md"), root=REPO_ROOT)


@pytest.mark.unit
def test_budget_requires_every_dimension_and_stop_rules() -> None:
    empty = camp.RunBudget()
    problems = empty.validate()
    assert len(problems) >= len(camp.BUDGET_DIMENSIONS)
    complete = camp.RunBudget(
        values={dimension: 0 for dimension in camp.BUDGET_DIMENSIONS}, stop_rules=("quality gate fail",)
    )
    assert complete.validate() == []


@pytest.mark.unit
def test_campaign_requires_prohibitions_and_isolation_clauses() -> None:
    manifest = camp.CampaignManifest(
        campaign_id="c1",
        experiments=("E14-01",),
        hardware_and_power_mode="dev-machine",
        prohibited_actions=camp.default_prohibited_actions(),
        required_isolation=dict(camp.REQUIRED_ISOLATION),
        budget=camp.RunBudget(
            values={dimension: 0 for dimension in camp.BUDGET_DIMENSIONS}, stop_rules=("oom",)
        ),
    )
    assert manifest.validate() == []
    unsafe = camp.CampaignManifest(
        campaign_id="c2", experiments=("E14-01",), hardware_and_power_mode="x",
        prohibited_actions=("unknown_action",),
    )
    problems = unsafe.validate()
    assert any("unknown prohibited action" in problem for problem in problems)
    assert any("resource budget" in problem for problem in problems)


@pytest.mark.unit
def test_forbidden_binaries_may_not_be_committed() -> None:
    problems = camp.check_committable(["model.safetensors", "profiler.nsys-rep", ".env", "raw/sample.json"])
    assert len(problems) == 3
    assert all("sample.json" not in problem for problem in problems)


# ── telemetry ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_modality_metrics_refuse_llm_units() -> None:
    assert tel.modality_metric_name("audio", "wer") == "wer"
    with pytest.raises(ConfigError):
        tel.modality_metric_name("audio", "tokens_per_second")
    with pytest.raises(ConfigError):
        tel.modality_metric_name("vlm", "made_up_metric")


@pytest.mark.unit
def test_projections_mark_missing_fields_instead_of_zeroing_them() -> None:
    row = tel.project_speculation_cycle({"request_id": "r1", "target_calls": 1})
    assert "acceptance" in row["missing"]
    assert row["acceptance"] is None
    assert row["proposed_tokens"] == 0


@pytest.mark.unit
def test_map_only_edge_record_may_not_carry_measurements() -> None:
    row = tel.project_edge_record({"evidence_level": "MAP_ONLY", "energy_j": 1.0})
    assert row.get("violations"), "MAP_ONLY evidence carried a measurement"


@pytest.mark.unit
def test_identity_closure_reports_unresolved_rows() -> None:
    result = tel.identity_closure(
        [{"trajectory_id": "t1"}, {"policy_snapshot_id": "p1"}], required=("trajectory_id", "policy_snapshot_id")
    )
    assert result["closed"] is False
    assert len(result["unresolved"]) == 2


# ── specs ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_frozen_specs_are_present_and_audit_clean() -> None:
    documents = spec_mod.ExperimentalSpecs.load(SPEC_DIR)
    assert documents.missing_kinds() == [], f"missing spec kinds: {documents.missing_kinds()}"
    assert spec_mod.audit_all_ok(documents.audit())
    assert spec_mod.check_all(documents.documents) == []


@pytest.mark.unit
def test_spec_audit_refuses_a_measurement_and_an_absolute_path() -> None:
    with_measurement = {
        "kind": "dependency-policy", "version": "v1", "sources": ["x"],
        "layers": [], "rules": [], "dummy_latency_ms": 1.0,
    }
    problems = spec_mod._audit_document("dependency-policy", "<memory>", with_measurement)
    assert any("measurement-like key" in problem.problem for problem in problems)

    with_path = {
        "kind": "dependency-policy", "version": "v1", "sources": ["x"],
        "layers": ["/root/somewhere"], "rules": [],
    }
    problems = spec_mod._audit_document("dependency-policy", "<memory>", with_path)
    assert any("absolute path" in problem.problem for problem in problems)


# ── experiment scaffolding ─────────────────────────────────────────────────


@pytest.mark.unit
def test_triple_gate_refuses_a_conclusion(tmp_path: Any) -> None:
    run = exp.RunDirectory(str(tmp_path), "E14-02", "r1")
    run.create()
    unsatisfied = exp.check_prerequisites(str(tmp_path), probe=False)
    assert not unsatisfied.satisfied

    with pytest.raises(ConfigError):
        run.write_verdict(
            status=rec.STATUS_PASS, reason="x", prerequisites=unsatisfied,
            executed=False, allow_execute=False, raw_samples=0,
        )
    with pytest.raises(ConfigError):
        run.write_verdict(
            status=rec.STATUS_PASS, reason="x", prerequisites=unsatisfied,
            executed=True, allow_execute=True, raw_samples=10,
        )
    verdict = run.write_verdict(
        status=rec.STATUS_BLOCKED, reason="prerequisites unsatisfied", prerequisites=unsatisfied,
        executed=False, allow_execute=True, raw_samples=0,
    )
    assert verdict["status"] == rec.STATUS_BLOCKED
    assert verdict["conclusion"] is False


@pytest.mark.unit
def test_run_directory_refuses_an_unlisted_artefact(tmp_path: Any) -> None:
    run = exp.RunDirectory(str(tmp_path), "E14-01", "r2")
    run.create()
    with pytest.raises(ConfigError):
        run.write_json("not_in_the_layout.json", {})
    assert run.write_status(rec.STATUS_NOT_STARTED, "interface only")["status"] == rec.STATUS_NOT_STARTED


@pytest.mark.unit
def test_preregistration_must_declare_every_run_manifest_field(tmp_path: Any) -> None:
    run = exp.RunDirectory(str(tmp_path), "E14-F1", "r3")
    run.create()
    with pytest.raises(ConfigError):
        run.write_preregistration({"experiment_id": "E14-F1"})
    complete = {name: None for name in rec.RUN_MANIFEST_FIELDS}
    complete["experiment_id"] = "E14-F1"
    complete["status"] = rec.STATUS_NOT_RUN
    assert run.write_preregistration(complete)


@pytest.mark.unit
def test_evidence_manifest_requires_raw_artefacts_for_a_claim() -> None:
    manifest = exp.EvidenceManifest(
        run_id="r1", experiment_id="E14-03", claim_level="MODEL", commands=("cmd",)
    )
    assert any("raw artefact" in problem for problem in manifest.problems())
    unmeasured = exp.EvidenceManifest(
        run_id="r1", experiment_id="E14-03", claim_level="NOT_MEASURED", commands=("cmd",)
    )
    assert unmeasured.problems() == []


@pytest.mark.unit
def test_prerequisites_are_blocked_without_probing_the_machine() -> None:
    status = exp.check_prerequisites(REPO_ROOT, probe=False)
    assert not status.satisfied
    assert "raw_evidence_retention" in {check.name for check in status.checks}
    probed = exp.check_prerequisites(REPO_ROOT, probe=True)
    assert "distributed_launcher" in probed.states() or not probed.satisfied
