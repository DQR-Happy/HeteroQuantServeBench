"""Unit tests for the S13 foundation layer (identity / records / contracts / campaign)."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.infra import campaign, contracts, identity, records, telemetry

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def _bundle(**overrides):
    payload = dict(
        release_id="rel-1",
        source_commit="a1da704",
        image_index_digest=DIGEST_A,
        platform_image_digests={"linux/amd64": DIGEST_B},
        service_config_hash=DIGEST_B,
        model_artifact_id="qwen3-1.7b@rev1",
        tokenizer_id="tok-1",
        deployment_template_digest=DIGEST_B,
        sbom_ids=("sbom-1",),
        status="DRAFT",
    )
    payload.update(overrides)
    return identity.ReleaseBundle(**payload)


@pytest.mark.unit
class TestIdentity:
    def test_digest_parsing_and_rejection(self) -> None:
        assert identity.parse_digest(DIGEST_A) == ("sha256", "a" * 64)
        with pytest.raises(ConfigError):
            identity.parse_digest("ubuntu:24.04")

    def test_tag_is_not_a_deployment_identity(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            identity.require_digest_reference("hqsb:latest")
        assert "digest" in str(excinfo.value)

    def test_release_bundle_requires_complete_identity(self) -> None:
        bundle = _bundle()
        assert bundle.validate() == []
        assert bundle.identity_complete() is True
        incomplete = _bundle(model_artifact_id="", status="DEPLOYABLE")
        problems = incomplete.validate()
        assert any("model_artifact_id" in problem for problem in problems)
        assert incomplete.identity_complete() is False

    def test_model_weights_may_not_be_embedded(self) -> None:
        problems = _bundle(model_embedded_in_image=True).validate()
        assert any("model weights" in problem for problem in problems)

    def test_oci_dag_compare_separates_bit_and_functional(self) -> None:
        left = identity.oci_digest_dag(
            DIGEST_A,
            platform_manifests={"linux/amd64": DIGEST_B},
            platform_configs={"linux/amd64": DIGEST_B},
            platform_layers={"linux/amd64": (DIGEST_B,)},
            diff_ids=(DIGEST_B,),
        )
        same = identity.compare_oci_dag(left, left)
        assert same["bit_reproducible"] is True
        other = identity.oci_digest_dag(
            DIGEST_A,
            platform_manifests={"linux/amd64": DIGEST_A},
            platform_configs={"linux/amd64": DIGEST_B},
            platform_layers={"linux/amd64": (DIGEST_B,)},
        )
        diff = identity.compare_oci_dag(left, other)
        assert diff["bit_reproducible"] is False
        assert "manifest" in diff["differing_levels"]

    def test_reproducibility_verdict_refuses_silent_upgrade(self) -> None:
        diff = {"bit_reproducible": False, "differing_levels": ["manifest"]}
        unexplained = identity.reproducibility_verdict(
            build_a_ok=True, build_b_ok=True, dag_diff=diff, filesystem_diff_rows=0,
            package_diff_rows=0, nondeterminism_sources=(),
        )
        assert unexplained["level"] == identity.REPRO_BUILD_REPEATABILITY
        explained = identity.reproducibility_verdict(
            build_a_ok=True, build_b_ok=True, dag_diff=diff, filesystem_diff_rows=0,
            package_diff_rows=0, nondeterminism_sources=("compression",),
        )
        assert explained["level"] == identity.REPRO_FUNCTIONAL_EQUIVALENCE
        with pytest.raises(ConfigError):
            identity.reproducibility_verdict(
                build_a_ok=True, build_b_ok=True, dag_diff=diff, filesystem_diff_rows=0,
                package_diff_rows=0, nondeterminism_sources=("vibes",),
            )

    def test_rollback_chain_needs_a_known_good_target(self) -> None:
        good = _bundle(release_id="rel-0", status="DEPLOYABLE")
        bad = _bundle(release_id="rel-1", rollback_parent_release_id="rel-0", status="DRAFT")
        report = identity.validate_rollback_chain([good, bad], active_release_id="rel-1")
        assert report["rollback_available"] is True
        missing = identity.validate_rollback_chain([bad], active_release_id="rel-1")
        assert missing["rollback_available"] is False

    def test_canary_change_scope_is_single_variable(self) -> None:
        control = _bundle()
        candidate = _bundle(release_id="rel-2", image_index_digest=DIGEST_B)
        ok = identity.validate_release_change_scope(control, candidate, declared_changes=["image_index_digest"])
        assert ok["single_variable_ok"] is True
        sneaky = _bundle(release_id="rel-3", image_index_digest=DIGEST_B, model_artifact_id="other-model")
        report = identity.validate_release_change_scope(control, sneaky, declared_changes=["image_index_digest"])
        assert report["single_variable_ok"] is False
        assert [row["field"] for row in report["undeclared_differences"]] == ["model_artifact_id"]


@pytest.mark.unit
class TestRecords:
    def test_state_machines_are_valid_and_enforce_transitions(self) -> None:
        assert records.validate_state_machines() == []
        records.ARTIFACT_STATE_MACHINE.assert_transition("VERIFYING", "VERIFIED_IMMUTABLE")
        with pytest.raises(ConfigError):
            records.ARTIFACT_STATE_MACHINE.assert_transition("ABSENT", "ACTIVE")
        assert records.REQUEST_STATE_MACHINE.walk(
            ["RECEIVED", "AUTHORIZED", "ADMITTED", "QUEUED", "BATCHED_OR_EXECUTING", "STREAMING",
             "COMPLETED", "ACCOUNTED_RELEASED"]
        )["ok"]

    def test_missing_is_not_zero(self) -> None:
        with pytest.raises(ConfigError):
            records.numeric_value(records.MISSING_MEASUREMENT_UNAVAILABLE)
        with pytest.raises(ConfigError):
            records.numeric_value(None)
        assert records.numeric_value(0.0) == 0.0

    def test_table_rows_require_every_field(self) -> None:
        problems = records.validate_rows("admission_decision", [{"decision_id": "d1"}])
        assert problems and "reason_code" in problems[0]["missing_fields"]

    def test_status_propagation_blocks_dependent_claims(self) -> None:
        report = records.propagate_statuses({"E13-01": records.STATUS_FAIL})
        claims = [claim for row in report["blocked"] for claim in row["blocked_claims"]]
        assert any("release admitted" in claim for claim in claims)
        assert records.propagate_statuses({})["all_green"] is False
        with pytest.raises(ConfigError):
            records.propagate_statuses({"E13-99": records.STATUS_PASS})


@pytest.mark.unit
class TestContracts:
    def test_readiness_needs_semantic_state_not_healthz(self) -> None:
        result = contracts.validate_readiness_claim({"ready": True, "healthz_status": 200, "model_state": "ABSENT"})
        assert result.ok is False
        assert contracts.RC_READINESS_WITHOUT_MODEL_STATE in result.reason_codes

    def test_liveness_must_not_be_load_coupled(self) -> None:
        result = contracts.validate_liveness_policy({"liveness_signals": ["queue_depth"], "failure_conditions": ["load"]})
        assert result.ok is False
        assert contracts.RC_LIVENESS_OVERLOAD_COUPLING in result.reason_codes

    def test_drain_after_point_rejects_new_work(self) -> None:
        result = contracts.validate_drain_semantics(draining=True, accepted_after_drain=1)
        assert result.ok is False

    def test_admission_decision_needs_budget_and_reason(self) -> None:
        minimal = contracts.validate_admission_decision({"decision": "ADMIT_NOW"})
        assert minimal.ok is False
        over = contracts.validate_admission_decision(
            {
                "decision": "ADMIT_NOW",
                "policy_version": "cap-v1",
                "reason_code": "ADMIT_WITHIN_BUDGET",
                "usable_memory_bytes": 100,
                "safety_margin_bytes": 10,
                "predicted_incremental_kv_bytes": 500,
            }
        )
        assert over.ok is False

    def test_stale_metric_may_not_drive_a_decision(self) -> None:
        fresh = contracts.validate_autoscaling_metric(
            {"metric_name": "queue_tokens", "event_ts": 1.0, "age_s": 5.0}, max_age_s=60.0
        )
        assert fresh.ok is True
        stale = contracts.validate_autoscaling_metric(
            {"metric_name": "queue_tokens", "event_s": 1.0, "event_ts": 1.0, "age_s": 600.0}, max_age_s=60.0
        )
        assert stale.ok is False
        assert contracts.RC_METRIC_STALE_USED_AS_FRESH in stale.reason_codes
        no_ts = contracts.validate_autoscaling_metric({"metric_name": "queue_tokens", "value": 1.0})
        assert no_ts.ok is False

    def test_fault_target_must_be_authorized_and_bounded(self) -> None:
        result = contracts.validate_fault_target(
            {"target_selector": "pod/*", "resolved_targets": (), "abort_threshold": "", "kill_switch": ""},
            {"namespaces": ["hqsb"], "max_targets": 1},
        )
        assert result.ok is False
        assert len(result.findings) >= 3

    def test_recovery_needs_three_axes(self) -> None:
        naive = contracts.validate_recovery_completion({"pod_running": True})
        assert naive.ok is False
        full = contracts.validate_recovery_completion(
            {
                "pod_running": True,
                "service": {"verified": True},
                "resource": {"verified": True},
                "state": {"verified": True},
                "correctness_rerun": True,
            }
        )
        assert full.ok is True

    def test_public_payload_scan_redacts(self) -> None:
        leaked = contracts.validate_public_payload("api_key = sk-live-abcdef0123456789")
        assert leaked.ok is False
        assert contracts.RC_SENSITIVE_DATA_IN_PUBLIC_ARTIFACT in leaked.reason_codes
        redacted = contracts.redact_payload("api_key = sk-live-abcdef0123456789")
        assert "sk-live" not in redacted

    def test_report_point_needs_evidence_reference(self) -> None:
        result = contracts.validate_report_point({"point_id": "p1", "metric_name": "ttft"})
        assert result.ok is False
        ok = contracts.validate_report_point(
            {"point_id": "p1", "evidence_refs": ["observability/traces/coverage.parquet#row12"]}
        )
        assert ok.ok is True


@pytest.mark.unit
class TestCampaign:
    def test_layout_matches_the_protocol_tree(self) -> None:
        assert len(campaign.ARTIFACT_LAYOUT) == 52
        assert "supply_chain/gates" in campaign.ARTIFACT_LAYOUT
        assert "security/quota" in campaign.ARTIFACT_LAYOUT
        assert campaign.logical_root("c1") == "artifacts/S13/c1"
        assert campaign.physical_root("/repo", "c1").endswith("experiment_results/S13/c1")

    def test_campaign_manifest_requires_frozen_claims_and_safety(self) -> None:
        with pytest.raises(ConfigError):
            campaign.campaign_manifest(campaign_id="c1")
        manifest = campaign.campaign_manifest(
            campaign_id="c1",
            image_index_digests={"hqsb": DIGEST_A},
            primary_sli=("slo_goodput",),
            slo_targets={"ttft_p99_seconds": 2.0},
            planned_repetitions=3,
            safety_policy_id="safety-1",
            allowed_claims=("clean deployment works",),
            forbidden_claims=("production reliability",),
        )
        assert manifest.manifest_hash().startswith("sha256:")

    def test_safety_policy_and_target_authorization(self) -> None:
        policy = campaign.SafetyPolicy(
            safety_policy_id="safety-1",
            cluster_id="test-cluster",
            namespaces=("hqsb-test",),
            authorized_nodes=("node-1",),
            max_targets=1,
            max_duration_s=600,
            max_replicas=2,
            error_budget_cap=0.1,
            kill_switch="delete-fault-job",
            operator="sre",
            target_inventory_recorded=True,
            fault_runs_separated_from_baseline=True,
        )
        assert campaign.validate_safety_policy(policy) == []
        assert campaign.authorize_targets(policy, ["node-1"])["authorized"] is True
        assert campaign.authorize_targets(policy, ["node-*"])["authorized"] is False
        assert campaign.authorize_targets(policy, ["node-2"])["authorized"] is False

    def test_report_obligations_are_experiment_dependent(self) -> None:
        report = campaign.report_obligations(experiments_executed=("E13-02",))
        assert "reports/reliability_report.md" in report["blocked"]
        assert "reports/production_architecture.md" in report["required_always"]


@pytest.mark.unit
class TestTelemetry:
    def test_projections_report_missing_identity_fields(self) -> None:
        bundle = _bundle()
        projection = telemetry.project_release_identity(bundle, sbom_ids=("sbom-1",), provenance_id="prov-1")
        assert projection["missing"] == []
        # A clean tree legitimately has no dirty-patch hash and the first release has
        # no parent: optional empties are reported, not treated as missing identity.
        assert "dirty_patch_hash" in projection["optional_missing"]
        assert "parent_release" in projection["optional_missing"]
        partial = telemetry.project_release_identity(
            _bundle(source_commit=""), sbom_ids=("sbom-1",), provenance_id="prov-1"
        )
        assert "source_commit" in partial["missing"]

    def test_coverage_report_lists_empty_tables(self) -> None:
        report = telemetry.coverage_report({"release_bundle": 3})
        assert report["filled"] == 1
        assert len(report["empty"]) == len(records.TABLE_SCHEMAS) - 1

    def test_experiment_tables_and_schema_audit(self) -> None:
        assert "release_bundle" in telemetry.experiment_tables("E13-01")
        assert telemetry.schema_audit()["ok"] is True

    def test_evidence_level_above_fault_validation_needs_production(self) -> None:
        levels = telemetry.evidence_axis_levels(
            {"fault": records.EVIDENCE_REPEATED_CONTROLLED,
             "reliability": records.EVIDENCE_PRODUCTION_OBSERVED}
        )
        assert levels["requires_production_evidence"] == ["reliability"]
