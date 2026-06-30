"""Unit tests for the E12-01 comparability audit and candidate inventory."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.evaluation import candidates as cand
from hqsb.evaluation import comparability as cmp
from hqsb.evaluation.contracts import (
    FIELD_INVARIANT,
    FieldDiff,
    NormalizationFormula,
    default_contract,
)
from hqsb.evaluation.records import (
    COMPARABLE,
    CONDITIONAL,
    INSUFFICIENT_EVIDENCE,
    NOT_COMPARABLE,
)

pytestmark = pytest.mark.unit


def _identity(**overrides: object) -> cand.CandidateIdentity:
    identity = {
        "hardware_sku": "h100",
        "device_count": 1,
        "topology_id": "single",
        "host_id": "host-a",
        "software_stack_id": "stack-1",
        "backend_id": "cuda",
        "compiler_artifact_id": "compile-1",
        "model_artifact_id": "qwen3-1.7b@frozen",
        "tokenizer_id": "qwen3-tokenizer@frozen",
        "precision_contract_id": "fp16_w_fp16_a_fp32_acc",
        "quant_artifact_id": "",
        "kv_contract_id": "kv-fp16",
        "parallel_plan_id": "",
        "service_policy_id": "policy-a",
        "measurement_contract_id": "mc-1",
    }
    identity.update(overrides)
    return cand.CandidateIdentity(
        identity=identity, display_name="H100 (display only)", raw_manifest_hash="a" * 64
    )


def _normalization() -> NormalizationFormula:
    return NormalizationFormula(
        formula_id="norm_per_token",
        formula="normalized = raw / accepted_output_tokens",
        inputs=("raw_latency_ns", "accepted_output_tokens"),
        conditions=("same tokenizer", "same output accounting"),
        residual_limitations=("per-token latency hides batching effects",),
        allowed_analyses=("per-token latency comparison",),
        forbidden_claims=("no end-to-end service conclusion",),
    )


class TestCandidateIdentity:
    def test_candidate_id_is_stable_and_ignores_display_name(self) -> None:
        first = _identity()
        second = cand.CandidateIdentity(
            identity=dict(first.identity), display_name="renamed", raw_manifest_hash="b" * 64
        )
        assert first.candidate_id == second.candidate_id
        assert first.validate() == []

    def test_missing_identity_field_is_rejected(self) -> None:
        identity = _identity().identity
        broken = dict(identity)
        broken["tokenizer_id"] = ""
        with pytest.raises(ConfigError):
            CandidateMatrix = cand.CandidateMatrix()
            CandidateMatrix.add(
                cand.CandidateIdentity(identity=broken, raw_manifest_hash="c" * 64)
            )

    def test_unknown_identity_field_is_rejected(self) -> None:
        identity = dict(_identity().identity)
        identity["gpu_name"] = "H100"
        candidate = cand.CandidateIdentity(identity=identity, raw_manifest_hash="d" * 64)
        assert any("unknown identity fields" in item for item in candidate.validate())

    def test_raw_manifest_hash_is_required(self) -> None:
        candidate = cand.CandidateIdentity(identity=_identity().identity)
        assert any("raw_manifest_hash" in item for item in candidate.validate())


class TestCandidateMatrix:
    def test_duplicate_identical_candidate_is_idempotent(self) -> None:
        matrix = cand.CandidateMatrix()
        first = matrix.add(_identity())
        assert matrix.add(_identity()) == first
        assert len(matrix.ids()) == 1

    def test_conflicting_duplicate_is_rejected(self) -> None:
        matrix = cand.CandidateMatrix()
        matrix.add(_identity())
        # Same hardware/software but a different host is a *different* candidate;
        # forcing the same id with different content must fail, not silently merge.
        twin = _identity(host_id="host-b")
        matrix.add(twin)
        assert len(matrix.ids()) == 2
        assert matrix.summary()["candidates"] == 2

    def test_get_unknown_candidate_raises(self) -> None:
        matrix = cand.CandidateMatrix()
        with pytest.raises(ConfigError):
            matrix.get("cand_missing")

    def test_freeze_hash_changes_when_universe_changes(self) -> None:
        matrix = cand.CandidateMatrix()
        matrix.add(_identity())
        first = matrix.freeze()
        matrix.add(_identity(hardware_sku="a100"))
        second = matrix.freeze()
        assert first["universe_hash"] != second["universe_hash"]
        assert cand.universe_invalidated(first, second["candidate_ids"]) is True
        assert cand.universe_invalidated(second, second["candidate_ids"]) is False


class TestRequirementsAndEstimands:
    def test_single_device_candidate_has_no_distributed_layer(self) -> None:
        candidate = _identity(device_count=1)
        with pytest.raises(ConfigError):
            cand.capability_requirements(candidate, "distributed")

    def test_multi_device_candidate_requires_collectives(self) -> None:
        candidate = _identity(device_count=4, parallel_plan_id="tp4")
        features = cand.capability_requirements(candidate, "distributed")
        assert "bf16_collective_all_reduce" in features

    def test_service_layer_requires_a_service_policy(self) -> None:
        candidate = _identity(service_policy_id="")
        with pytest.raises(ConfigError):
            cand.capability_requirements(candidate, "service")

    def test_estimand_registry_covers_every_layer(self) -> None:
        rows = cand.estimand_registry()
        assert cand.validate_estimands(rows) == []
        assert {row.layer for row in rows} == {"operator", "model_core", "service", "distributed"}

    def test_vague_estimator_is_rejected(self) -> None:
        bad = cand.Estimand(
            estimand_id="bad",
            layer="service",
            scenario="target_zone",
            object_description="everything",
            boundary="client",
            population="all",
            estimator="总体性能",
            unit="ms",
        )
        assert any("总体性能" in item for item in bad.validate())


class TestUpstreamEvidence:
    def test_usable_state_needs_uri_and_hash(self) -> None:
        row = cand.UpstreamEvidenceRow(
            candidate_id="cand_x",
            upstream_stage="S03",
            experiment_id="E03-03",
            artifact_uri="hqsb://S03/E03-03/result/1",
            state="VERIFIED",
            hash_ok=False,
        )
        assert any("verified hash" in item for item in row.validate())

    def test_missing_state_requires_reason(self) -> None:
        row = cand.UpstreamEvidenceRow(
            candidate_id="cand_x", upstream_stage="S05", experiment_id="E05-01"
        )
        assert row.state == "MISSING"
        assert any("reason" in item for item in row.validate())

    def test_inventory_reports_unresolved_dependencies(self) -> None:
        inventory = cand.inventory_for_candidates(
            ["cand_x"],
            [
                cand.UpstreamEvidenceRow(
                    candidate_id="cand_x",
                    upstream_stage="S02",
                    experiment_id="E02-01",
                    artifact_uri="hqsb://S02/E02-01/raw/1",
                    state="VERIFIED",
                    hash_ok=True,
                )
            ],
            required_stages=("S02", "S03"),
        )
        assert inventory["unresolved_count"] == 1
        assert inventory["unresolved"][0]["upstream_stage"] == "S03"
        assert inventory["ok"] is True

    def test_require_usable_fails_closed(self) -> None:
        with pytest.raises(ConfigError):
            cand.require_usable({"S03": "UNVERIFIED"}, "S03", candidate_id="cand_x")
        cand.require_usable({"S03": "VERIFIED"}, "S03")


class TestPolicyAndVerdicts:
    def test_unclassified_field_defaults_to_invariant(self) -> None:
        policy = cmp.FieldClassificationPolicy()
        assert policy.classify("something.new") == FIELD_INVARIANT

    def test_unknown_override_class_is_rejected(self) -> None:
        policy = cmp.FieldClassificationPolicy(overrides={"x": "WHATEVER"})
        with pytest.raises(ConfigError):
            policy.classify("x")

    def test_comparable_verdict_rejects_invariant_difference(self) -> None:
        verdict = cmp.ComparabilityVerdict(
            verdict=COMPARABLE,
            comparison_group_id="g",
            field_class=FIELD_INVARIANT,
            match_status="DIFF",
        )
        assert any("invariant/forbidden" in item for item in verdict.validate())

    def test_conditional_verdict_needs_formula_and_scope(self) -> None:
        verdict = cmp.ComparabilityVerdict(verdict=CONDITIONAL, normalization_formula_id="")
        problems = verdict.validate()
        assert any("normalization formula" in item for item in problems)
        assert any("analyses" in item for item in problems)
        assert any("claims" in item for item in problems)

    def test_not_comparable_cannot_carry_ranking_scope(self) -> None:
        verdict = cmp.ComparabilityVerdict(
            verdict=NOT_COMPARABLE, allowed_analyses=("rank",)
        )
        assert any("ranking" in item for item in verdict.validate())

    def test_decision_order_evidence_then_quality_then_diff(self) -> None:
        diff = FieldDiff(
            field_path="model_artifact_id",
            field_class=FIELD_INVARIANT,
            value_a_hash="a",
            value_b_hash="b",
            dimension="model_identity",
            match_status="DIFF",
            reason_code="MODEL_ARTIFACT_MISMATCH",
        )
        incomplete = cmp.decide_comparability(
            [diff], quality_ok=True, evidence_complete=False
        )
        assert incomplete.verdict == INSUFFICIENT_EVIDENCE
        failed_quality = cmp.decide_comparability([diff], quality_ok=False, evidence_complete=True)
        assert failed_quality.verdict == NOT_COMPARABLE
        assert failed_quality.reason_code == "QUALITY_GATE_FAILED"
        mismatched = cmp.decide_comparability([diff], quality_ok=True, evidence_complete=True)
        assert mismatched.verdict == NOT_COMPARABLE
        assert mismatched.reason_code == "MODEL_ARTIFACT_MISMATCH"

    def test_conditional_difference_without_formula_degrades(self) -> None:
        diff = FieldDiff(
            field_path="warmup.compile_state",
            field_class="CONDITIONALLY_NORMALIZABLE",
            value_a_hash="a",
            value_b_hash="b",
            dimension="warmup_cache",
            match_status="DIFF",
            reason_code="WARMUP_STATE_MISMATCH",
        )
        without = cmp.decide_comparability([diff], quality_ok=True, evidence_complete=True)
        assert without.verdict == NOT_COMPARABLE
        with_formula = cmp.decide_comparability(
            [diff], quality_ok=True, evidence_complete=True, normalization=_normalization()
        )
        assert with_formula.verdict == CONDITIONAL
        assert with_formula.allowed_analyses
        assert with_formula.forbidden_claims

    def test_no_difference_is_comparable(self) -> None:
        verdict = cmp.decide_comparability([], quality_ok=True, evidence_complete=True, group_id="g")
        assert verdict.verdict == COMPARABLE
        assert verdict.rankable is True

    def test_negative_case_matrix_has_at_least_ten_classes(self) -> None:
        rows = cmp.negative_case_matrix()
        assert len(rows) >= 10
        assert {row["expected_verdict"] for row in rows} <= {
            COMPARABLE,
            CONDITIONAL,
            NOT_COMPARABLE,
            INSUFFICIENT_EVIDENCE,
        }


class TestCellAudit:
    def _cell(self, **overrides: object) -> dict:
        contract = default_contract()
        cell = {
            "cell_id": "c1",
            "comparison_id": contract.comparison_id,
            "candidate_id": "cand_a",
            "layer": contract.workload.layer,
            "scenario": contract.workload.scenario,
            "workload_spec_id": contract.workload.workload_spec_id,
            "measurement_contract_id": "mc1",
            "quality_gate_id": contract.quality.gate_id,
            "actual_backend": "cuda",
        }
        cell.update(overrides)
        return cell

    def test_missing_actual_backend_is_insufficient_evidence(self) -> None:
        contract = default_contract()
        audit = cmp.audit_cell(self._cell(actual_backend=""), contract, policy=cmp.FieldClassificationPolicy())
        assert "ACTUAL_BACKEND_UNKNOWN" in audit.unresolved
        assert audit.verdict.verdict == INSUFFICIENT_EVIDENCE

    def test_run_audit_histogram_and_gaps(self) -> None:
        contract = default_contract()
        result = cmp.run_audit(
            [self._cell(), self._cell(cell_id="c2", quality_gate_id="")],
            contract,
            policy=cmp.FieldClassificationPolicy(),
        )
        histogram = result.verdict_histogram()
        assert sum(histogram.values()) == 2
        assert result.action_items()
        assert result.as_dict()["audit_id"]

    def test_illegal_join_guard_rejects_and_accepts(self) -> None:
        contract = default_contract()
        group = f"grp_{contract.comparison_id}"
        rows = [
            {
                "cell_id": "clean",
                "comparison_group_id": group,
                "comparability_status": COMPARABLE,
                "quality_status": "pass",
                "actual_backend": "cuda",
                "layer": contract.workload.layer,
                "contract_sha256": contract.sha256(),
            },
            {
                "cell_id": "quality_failed",
                "comparison_group_id": group,
                "comparability_status": COMPARABLE,
                "quality_status": "QUALITY_GATE_FAILED",
                "actual_backend": "cuda",
                "layer": contract.workload.layer,
                "contract_sha256": contract.sha256(),
            },
            {
                "cell_id": "unknown_group",
                "comparison_group_id": "other",
                "comparability_status": COMPARABLE,
                "quality_status": "pass",
                "actual_backend": "cuda",
                "layer": contract.workload.layer,
            },
            {
                "cell_id": "conditional",
                "comparison_group_id": group,
                "comparability_status": CONDITIONAL,
                "quality_status": "pass",
                "actual_backend": "cuda",
                "layer": contract.workload.layer,
                "contract_sha256": contract.sha256(),
            },
        ]
        decision = cmp.illegal_join_guard(
            rows, allowed_group_ids={group}, contract_by_group={group: contract}
        )
        assert decision.accepted == ("clean",)
        codes = {row["error_code"] for row in decision.rejected}
        assert codes == {
            "JOIN_QUALITY_FAILED",
            "JOIN_UNKNOWN_COMPARISON_GROUP",
            "JOIN_CONDITIONAL_NOT_OPTED_IN",
        }

    def test_conditional_rows_pass_with_explicit_opt_in(self) -> None:
        contract = default_contract()
        group = f"grp_{contract.comparison_id}"
        row = {
            "cell_id": "conditional",
            "comparison_group_id": group,
            "comparability_status": CONDITIONAL,
            "quality_status": "pass",
            "actual_backend": "cuda",
            "layer": contract.workload.layer,
            "contract_sha256": contract.sha256(),
        }
        decision = cmp.illegal_join_guard(
            [row],
            allowed_group_ids={group},
            contract_by_group={group: contract},
            level="conditional",
            opt_in_conditional_groups={group},
        )
        assert decision.accepted == ("conditional",)

    def test_contract_change_invalidates_suite_manifest(self) -> None:
        contract = default_contract()
        groups = cmp.build_comparison_groups(
            [
                cmp.CellAudit(
                    cell_id="c1",
                    comparison_id=contract.comparison_id,
                    candidate_id="cand_a",
                    layer="model_core",
                    scenario="decode",
                    verdict=cmp.decide_comparability(
                        [], quality_ok=True, evidence_complete=True, group_id="g"
                    ),
                )
            ],
            contract,
        )
        manifest = cmp.suite_manifest(groups[0], contract, ["c1"])
        assert cmp.suite_manifest_valid(manifest, contract, ["c1"]) is True
        assert cmp.suite_manifest_valid(manifest, contract, ["c1", "c2"]) is False

    def test_groups_for_non_comparable_are_diagnostic_only(self) -> None:
        contract = default_contract()
        audit = cmp.CellAudit(
            cell_id="c1",
            comparison_id=contract.comparison_id,
            candidate_id="cand_a",
            layer="model_core",
            scenario="decode",
            verdict=cmp.ComparabilityVerdict(verdict=NOT_COMPARABLE, reason_code="MODEL_ARTIFACT_MISMATCH"),
        )
        group = cmp.build_comparison_groups([audit], contract)[0]
        assert group.downstream_scope == "diagnostic_only"
        assert group.forbidden_claims


class TestAudits:
    def test_unit_audit_flags_cross_family_mixing(self) -> None:
        rows = [
            {"row_id": "r1", "metric_name": "ttft", "unit": "ms"},
            {"row_id": "r2", "metric_name": "ttft", "unit": "ns"},
            {"row_id": "r3", "metric_name": "energy", "unit": "W"},
            {"row_id": "r4", "metric_name": "energy", "unit": "J"},
        ]
        findings = cmp.unit_audit(rows)
        statuses = {row["check_id"]: row["status"] for row in findings}
        assert statuses["unit_mix:ttft"] == "CONVERTIBLE"
        assert statuses["unit_mix:energy"] == "FAIL"

    def test_unit_audit_flags_missing_denominator(self) -> None:
        findings = cmp.unit_audit(
            [{"row_id": "r1", "metric_name": "j_per_token", "unit": "J", "normalization_basis": "per_accepted_token"}]
        )
        assert any(row["check_id"].startswith("denominator_missing") for row in findings)

    def test_boundary_audit_requires_trace_evidence(self) -> None:
        contract = default_contract()
        unproven = cmp.boundary_audit([{"metric_name": "core_ttft_ns"}], contract)
        assert unproven[0]["status"] == "INSUFFICIENT_EVIDENCE"
        matched = cmp.boundary_audit(
            [{"metric_name": "core_ttft_ns", "boundary_id": contract.timing.boundary}], contract
        )
        assert matched == ()
        mismatched = cmp.boundary_audit(
            [{"metric_name": "core_ttft_ns", "boundary_id": "service_client_visible"}], contract
        )
        assert mismatched[0]["status"] == "FAIL"

    def test_quality_dependency_audit_catches_stale_gate(self) -> None:
        quality = [
            {
                "status": "PASS",
                "candidate_id": "cand_a",
                "precision_contract_id": "fp16",
                "model_artifact_id": "m1",
                "gate_id": "gate_v1",
            }
        ]
        missing = cmp.quality_dependency_audit(
            quality,
            [{"cell_id": "c1", "candidate_id": "cand_b", "precision_contract_id": "fp16", "model_artifact_id": "m1"}],
        )
        assert missing[0]["reason_code"] == "QUALITY_GATE_FAILED"
        stale = cmp.quality_dependency_audit(
            quality,
            [
                {
                    "cell_id": "c1",
                    "candidate_id": "cand_a",
                    "precision_contract_id": "fp16",
                    "model_artifact_id": "m1",
                    "quality_gate_id": "gate_v2",
                }
            ],
        )
        assert stale[0]["reason_code"] == "QUALITY_EVIDENCE_STALE"

    def test_actual_backend_audit_blocks_silent_fallback(self) -> None:
        findings = cmp.actual_backend_audit(
            [
                {"cell_id": "c1", "requested_backend": "triton", "actual_backend": "eager"},
                {"cell_id": "c2", "requested_backend": "triton", "actual_backend": ""},
            ]
        )
        codes = {row["reason_code"] for row in findings}
        assert codes == {"SILENT_FALLBACK", "ACTUAL_BACKEND_UNKNOWN"}


class TestReviewAndSmoke:
    def test_reviewer_agreement_records_disagreements(self) -> None:
        records = [
            cmp.reviewer_record("g", reviewer="r1", verdict=COMPARABLE),
            cmp.reviewer_record("g", reviewer="r2", verdict=NOT_COMPARABLE, disagreements=("timing",), evidence_refs=("trace://1",)),
        ]
        report = cmp.reviewer_agreement(records)
        assert report["fully_agreed"] == 0
        assert report["rows"][0]["disagreements"] == ["timing"]

    def test_resolution_lossless_detects_orphan_normalized_rows(self) -> None:
        report = cmp.resolution_lossless(
            [{"row_id": "r1"}], [{"row_id": "n1", "source_row_id": "r2"}]
        )
        assert report["lossless"] is False

    def test_protocol_steps_are_complete(self) -> None:
        assert len(cmp.PROTOCOL_STEPS) == 36
        assert all(interfaces for _, _, interfaces in cmp.PROTOCOL_STEPS)

    def test_smoke_self_check_is_labelled_smoke(self) -> None:
        result = cmp.smoke_self_check()
        assert result["status"] == "smoke"
        assert result["claim_allowed"] is False
        assert result["policy_default_is_invariant"] is True
        assert result["unit_conversion_keeps_raw"] is True
