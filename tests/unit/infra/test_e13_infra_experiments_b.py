"""Negative-path tests for E13-06…E13-11 (capacity → multi-tenant security)."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.infra import autoscaling, canary, capacity, faults, observability, records, security

DIGEST = "sha256:" + "e" * 64


def _kv() -> capacity.KVModel:
    return capacity.KVModel(layers=28, kv_heads=4, head_dim=128, bytes_per_element=2,
                            block_tokens=16, block_bytes_overhead=64)


def _policy(**overrides) -> capacity.AdmissionPolicy:
    payload = dict(policy_id="p1", kind="MEMORY_KV_PREDICTED", safety_margin_bytes=1 << 30,
                   token_budget=100_000, version="cap-v1")
    payload.update(overrides)
    return capacity.AdmissionPolicy(**payload)


@pytest.mark.unit
class TestCapacity:
    def test_ledger_requires_every_component_and_per_rank(self) -> None:
        components = [
            capacity.MemoryComponent(component=name, bytes_value=1 << 20, rank_id="0", source="allocator")
            for name in records.RESOURCE_MEMORY_COMPONENTS
        ]
        assert capacity.memory_ledger(components, usable_memory_bytes=1 << 30)["ok"] is True
        partial = capacity.memory_ledger(
            components[:3], usable_memory_bytes=1 << 30
        )
        assert partial["missing_components"]

    def test_per_rank_limit_is_not_hidden_by_the_total(self) -> None:
        components = [
            capacity.MemoryComponent(component="M_weights_resident", bytes_value=90 << 30, rank_id="1",
                                     source="vendor"),
        ]
        report = capacity.memory_ledger(components, usable_memory_bytes=80 << 30)
        assert report["ok"] is False
        assert report["max_rank_bytes"] == 90 << 30

    def test_kv_model_needs_block_overhead(self) -> None:
        # A dataclass validates; only the factories raise. Both paths matter: a
        # zero block_tokens is structurally invalid, a zero overhead means the model
        # pretends the allocator is perfect.
        invalid = capacity.KVModel(layers=1, kv_heads=1, head_dim=1, block_tokens=0,
                                   block_bytes_overhead=0).validate()
        assert any("block_tokens" in problem for problem in invalid)
        assert any("block rounding" in problem for problem in invalid)
        with pytest.raises(ConfigError):
            capacity.KVModel(layers=1, kv_heads=1, head_dim=1, block_tokens=16).incremental_bytes(-1)

    def test_prefix_share_only_discounts_known_hits(self) -> None:
        kv = _kv()
        assert kv.incremental_bytes(100, prefix_shared_tokens=64) < kv.incremental_bytes(100)

    def test_admission_rejects_over_budget_before_execution(self) -> None:
        decision = capacity.AdmissionDecision(
            decision_id="d", request_id="r", policy_version="cap-v1", prompt_tokens=1,
            requested_output_tokens=1, usable_memory_bytes=1 << 20, safety_margin_bytes=1 << 19,
            release_id="rel", model_artifact_id="m",
        )
        capacity.admit(policy=_policy(), kv_model=_kv(), decision=decision)
        assert decision.decision.startswith("REJECT")
        assert decision.reason_code == "REJECT_MEMORY_PREDICTION_OVER_MARGIN"

    def test_oversell_and_cancel_leaks(self) -> None:
        oversell = capacity.reservation_atomicity(
            [{"trial_id": "t", "capacity_bytes": 10, "reserved_bytes": 9, "committed_bytes": 12}]
        )
        assert oversell["ok"] is False
        leak = capacity.cancel_release([{"stage": "decode", "budget_released": False, "kv_released": True,
                                         "slot_released": True}])
        assert leak["ok"] is False

    def test_residual_needs_an_explanation(self) -> None:
        report = capacity.prediction_residual(
            rows=[{"predicted_bytes": 100, "actual_bytes": 130, "component": "M_KV_active"}],
            s12_capacity_model_id="s12-cap-v1",
        )
        assert report["ok"] is False
        with pytest.raises(ConfigError):
            capacity.prediction_residual(rows=[], s12_capacity_model_id="")

    def test_false_reject_needs_a_counterfactual(self) -> None:
        report = capacity.false_decisions(
            decisions=[capacity.AdmissionDecision(decision_id="d", request_id="r", policy_version="p",
                                                 decision="REJECT_RETRYABLE_OVERLOAD",
                                                 reason_code="REJECT_TOKEN_BUDGET_EXCEEDED")],
            counterfactual_rows=[{"safe_under_counterfactual": True}],
            counterfactual_method="",
        )
        assert report["ok"] is False

    def test_margin_must_not_be_tuned_on_the_validation_data(self) -> None:
        report = capacity.margin_holdout_validation(
            holdout_rows=[{"oom": False}], selected_margin_bytes=1 << 20, tuned_on_same_data=True
        )
        assert report["ok"] is False

    def test_s12_feedback_keeps_the_original_model(self) -> None:
        report = capacity.s12_feedback(
            s12_capacity_model_id="s12-cap-v1",
            residuals={"rows": [{"component": "M_KV_active"}]},
            update={"model_version": "s12-cap-v1", "policy_version": "cap-v1"},
        )
        assert report["ok"] is False


@pytest.mark.unit
class TestAutoscaling:
    def test_desired_replicas_and_clamping(self) -> None:
        model = autoscaling.CapacityModel(safe_capacity_per_replica=10.0, workload_mix="mixed",
                                          capacity_policy_version="cap-v1", headroom_factor=1.2)
        assert model.validate() == []
        computed = autoscaling.desired_replicas(
            demand_estimate=45.0, capacity=model, current_replicas=4, min_replicas=2, max_replicas=8
        )
        assert computed["desired_raw"] == 6

    def test_stale_metric_fails_safe(self) -> None:
        policy = autoscaling.StaleMetricPolicy(
            policy_id="s", max_age_s={"queue_tokens": 30.0}, on_stale="FAILSAFE_STATIC",
            on_missing="FAILSAFE_STATIC", alert_id="alert-1",
        )
        assert policy.validate() == []
        stale = autoscaling.MetricSample(sample_id="x", metric_name="queue_tokens", value=1.0,
                                         unit="tokens", event_ts=0.0, query_ts=1000.0)
        assert autoscaling.classify_metric(stale, policy)["may_scale"] is False
        unsafe = autoscaling.StaleMetricPolicy(policy_id="s2", max_age_s={"queue_tokens": 1.0},
                                              on_stale="SCALE_TO_MIN", on_missing="HOLD", alert_id="a")
        assert any("fail-safe" in problem for problem in unsafe.validate())

    def test_high_cardinality_label_is_refused_on_a_metric(self) -> None:
        sample = autoscaling.MetricSample(sample_id="x", metric_name="ttft_burn_rate", value=1.0,
                                          unit="ratio", event_ts=1.0, labels={"request_id": "r"})
        assert any("unbounded" in problem for problem in sample.validate())

    def test_decision_requires_ages_and_blocks_zero_capacity(self) -> None:
        decision = autoscaling.AutoscalingDecision(
            decision_id="d", episode_id="e", policy_version="as-v1", action="SCALE_DOWN",
            current_replicas=1, ready_replicas=1, desired_stabilized=1,
            metric_name_values={"queue_tokens": 1.0}, metric_ages={"queue_tokens": 1.0},
        )
        assert any("zero ready capacity" in problem for problem in decision.validate())

    def test_control_metrics_detect_oscillation(self) -> None:
        report = autoscaling.control_metrics(
            episode_id="e",
            readiness_series=[{"ready_replicas": 2}, {"ready_replicas": 6}, {"ready_replicas": 2}],
            actions=[{"action": "SCALE_UP"}] * 40,
            recovery_within_s=3600.0,
        )
        assert report["oscillating"] is True

    def test_short_burst_cannot_be_covered_reactively(self) -> None:
        report = autoscaling.control_loop_delay(
            t_metric=10, t_decision=5, t_schedule=5, t_pull=30, t_model=90, t_compile=30,
            t_warmup=30, t_ready=10, burst_duration_s=30,
        )
        assert report["reactive_scaling_sufficient"] is False

    def test_scale_down_protections_and_failure_cases(self) -> None:
        selection = autoscaling.scale_down_candidate_selection(
            candidates=[{"pod_uid": "p1", "holds": ["longest_inflight_stream"], "selected": True}]
        )
        assert selection["ok"] is False
        failures = autoscaling.run_failure_cases(
            [{"kind": "METRIC_STALE", "scaled_to_min_or_zero": True, "alerted": False}]
        )
        assert failures["ok"] is False


@pytest.mark.unit
class TestObservability:
    def test_convention_set_needs_three_boundaries_and_units(self) -> None:
        minimal = [observability.SemanticConvention(name="hqsb_x", signal_class="metrics", unit="seconds",
                                                    version="1.0.0", boundary="server")]
        report = observability.validate_convention_set(minimal)
        assert report["ok"] is False
        bad_unit = observability.SemanticConvention(name="hqsb_y", signal_class="metrics", unit="ms",
                                                    version="1.0.0", boundary="core")
        assert any("base unit" in problem for problem in bad_unit.validate())

    def test_forbidden_label_must_be_declared_forbidden(self) -> None:
        convention = observability.SemanticConvention(
            name="hqsb_z", signal_class="metrics", unit="requests", version="1.0.0",
            attributes={"request_id": "required"},
        )
        assert any("unbounded" in problem for problem in convention.validate())
        documented = observability.SemanticConvention(
            name="hqsb_z", signal_class="metrics", unit="requests", version="1.0.0",
            attributes={"request_id": "forbidden", "outcome": "required"},
            required_attributes=("outcome",),
        )
        assert documented.validate() == []

    def test_histogram_buckets_and_metric_reconciliation(self) -> None:
        buckets = observability.validate_histogram_buckets(
            metric_name="hqsb_server_ttft_seconds",
            buckets=[0.1, 0.2, 0.5, 1.0, 2.0, float("inf")], observed_p99=5.0,
        )
        assert buckets["ok"] is False
        mismatch = observability.reconcile_metric_with_raw(
            metric_name="hqsb_requests_total", metric_value=10.0, raw_rows=[{"value": 1.0}]
        )
        assert mismatch["ok"] is False

    def test_rca_record_requires_alternatives_and_hides_ground_truth(self) -> None:
        record = observability.RCARecord(
            rca_id="rca-1", case_id="c1", investigator_id="inv", blind_status="BLIND",
            symptom="slow request", identified_layer="decode_runtime", confidence="high",
            evidence_refs=("traces/coverage.parquet#r1",), ground_truth="queue delay",
        )
        problems = record.validate()
        assert any("alternative" in problem for problem in problems)
        assert any("ground truth" in problem for problem in problems)

    def test_telemetry_loss_and_redaction(self) -> None:
        loss = observability.telemetry_failure_detection(
            [{"component": "exporter", "detected": False, "silently_zero": True}]
        )
        assert loss["ok"] is False
        leaked = observability.redaction_scan(
            [{"case_id": "c", "signal_class": "logs", "canary_id": "canary-1", "leaked": True}]
        )
        assert leaked["ok"] is False


@pytest.mark.unit
class TestFaults:
    def _spec(self, **overrides) -> faults.FaultSpec:
        payload = dict(
            fault_case_id="F-1", layer="pod_container", mechanism="POD_DELETE", hypothesis="h",
            target_selector="pod/replica-2", resolved_targets=("pod-replica-2",), namespace="hqsb-test",
            blast_radius="one replica", safety_policy_id="s1", duration_s=30.0,
            expected_detection="alert", expected_degradation="degraded", expected_recovery="ready again",
            expected_degradation_class="MASKED_REDUNDANCY", abort_threshold="budget<10%",
            kill_switch="delete job", repetitions=3, allowed_claims=("single replica loss",),
            forbidden_claims=("node loss",), health_gate=faults.PRE_INJECTION_GATES,
        )
        payload.update(overrides)
        return faults.FaultSpec(**payload)

    def test_wildcard_targets_and_missing_gates_are_refused(self) -> None:
        spec = self._spec(target_selector="pod/*", health_gate=())
        problems = spec.validate()
        assert any("wildcards" in problem for problem in problems)
        assert any("health gate" in problem for problem in problems)

    def test_semantics_changing_degradation_needs_quality_recheck(self) -> None:
        spec = self._spec(expected_degradation_class="DEGRADE_EXPLICIT_QUALITY_OR_FEATURE",
                          expected_recovery="retry")
        assert any("quality" in problem for problem in spec.validate())

    def test_episode_requires_effective_time_and_ground_truth(self) -> None:
        episode = faults.FaultEpisode(episode_id="e", fault_case_id="F-1")
        problems = episode.validate()
        assert any("effective" in problem for problem in problems)

    def test_reliability_limits_are_enforced(self) -> None:
        episode = faults.FaultEpisode(
            episode_id="e", fault_case_id="F-1", release_id="r", model_artifact_id="m",
            cluster_id="c", planned_ts=0.0, effective_ts=100.0, detection_ts=200.0,
            mitigation_ts=250.0, recovery_ts=900.0, ground_truth_probe="pid gone",
            verdict="RECOVERED_AUTOMATIC",
        )
        assert faults.reliability_metrics([episode], thresholds={"mttd_s": 50.0})["ok"] is False
        assert faults.reliability_metrics([episode], thresholds={"mttr_s": 1000.0})["ok"] is True
        assert faults.reliability_metrics([episode], thresholds={})["ok"] is False

    def test_silent_fallback_and_retry_storm(self) -> None:
        record = faults.RetryFallback(
            episode_id="e", request_id="r", retries=1, budget=2, backoff="exp",
            fallback="alternate", fallback_declared=False, fallback_quality_checked=False,
            fallback_capability_adequate=False,
        )
        assert len(record.validate()) >= 3

    def test_state_recovery_requires_pre_and_post(self) -> None:
        report = faults.validate_state_recovery(
            episode_id="e", pre_state={}, post_state={"kv_and_reservation_released": "released"}
        )
        assert report["ok"] is False

    def test_postmortem_needs_owned_actions(self) -> None:
        report = faults.postmortem(
            episode_id="e", verdict="FAILED_SLO", timeline=["t0"], impact="5% errors",
            root_cause="cache miss storm", contributing_factors=["cold start"],
            actions=[{"action_id": "a1"}],
        )
        assert report["ok"] is False


@pytest.mark.unit
class TestCanary:
    def _policy(self, **overrides) -> canary.CanaryPolicy:
        payload = dict(
            policy_id="canary-1", min_samples_per_stage=100, min_exposure_s=600,
            practical_delta={"ttft_seconds": 0.05}, alpha=0.05, beta=0.2,
            sequential_boundary="3 looks", multiple_metric_policy="hard gates first",
            max_traffic_fraction=0.25, max_error_budget_spend=0.2, assignment_unit="SESSION_STICKY",
            sticky=True,
        )
        payload.update(overrides)
        return canary.CanaryPolicy(**payload)

    def test_policy_requires_stickiness_and_registered_delta(self) -> None:
        assert self._policy().validate() == []
        lax = self._policy(sticky=False, practical_delta={})
        problems = lax.validate()
        assert any("sticky" in problem for problem in problems)
        assert any("practical" in problem for problem in problems)

    def test_state_machine_blocks_stage_skipping(self) -> None:
        assert canary.CANARY_STATE_MACHINE.validate() == []
        assert canary.CANARY_STATE_MACHINE.walk(list(canary.PROMOTION_PATH))["ok"] is True
        assert canary.CANARY_STATE_MACHINE.walk(
            ["CANDIDATE_REGISTERED", "PROMOTED"]
        )["ok"] is False

    def test_assignment_is_deterministic_and_unit_scoped(self) -> None:
        policy = canary.AssignmentPolicy(policy_id="a", unit="SESSION_STICKY", hash_key="session_id",
                                         seed="s", eligibility_rule="authenticated",
                                         retry_attribution="first attempt")
        assert policy.validate() == []
        assert policy.assign(unit_key="s1", traffic_fraction=0.5) == policy.assign(
            unit_key="s1", traffic_fraction=0.5
        )
        wrong = canary.AssignmentPolicy(policy_id="a2", unit="SESSION_STICKY", hash_key="request_id",
                                        seed="s", eligibility_rule="x", retry_attribution="y")
        assert any("session" in problem for problem in wrong.validate())

    def test_hard_gate_failure_stops_regardless_of_performance(self) -> None:
        gates = canary.evaluate_gates(
            results=[
                canary.GateResult(canary_run_id="run", stage="stage1", gate_id="G1_OFFLINE_CORRECTNESS",
                                  status="FAIL", hard_gate=True, evidence_count=5, reason="token mismatch"),
                canary.GateResult(canary_run_id="run", stage="stage1", gate_id="G5_LATENCY_SLO_GOODPUT",
                                  status="PASS", hard_gate=False, evidence_count=100),
            ],
            policy=self._policy(),
        )
        assert gates["decision"] == "STOP"

    def test_sequential_rule_blocks_premature_promotion(self) -> None:
        comparison = canary.paired_interval(control=[1.0] * 30, candidate=[1.0] * 30,
                                            direction="lower_is_better")
        decision = canary.sequential_decision(
            stage="stage1", metric="ttft_seconds", comparison=comparison, practical_delta=0.05,
            information_fraction=0.5, policy=self._policy(), error_budget_spent=0.0,
        )
        assert decision["decision"] == "CONTINUE"
        promoted = canary.CanaryDecisionRow(
            decision_id="d", canary_run_id="run", stage="stage1", control_release_id="a",
            candidate_release_id="b", information_fraction=1.0, decision="PROMOTE",
            reason_codes=("ALL_GATES_PASSED",), evidence_refs=("canary/analyses/x.parquet#1",),
        )
        premature = canary.CanaryDecisionRow(**{**promoted.__dict__, "information_fraction": 0.5})
        assert any("information fraction" in problem for problem in premature.validate())

    def test_override_cannot_bypass_gates(self) -> None:
        override = canary.OverrideAudit(override_id="o", actor="sre", action="FORCE_PROMOTE",
                                        reason="demo", canary_run_id="run", counts_as_automatic=True)
        assert len(override.validate()) >= 3

    def test_rollback_closure_and_false_rates(self) -> None:
        closure = canary.rollback_closure(
            rollback_id="rb", canary_run_id="run", decision_ts=1.0, traffic_stopped_ts=2.0,
            drain_done_ts=3.0, baseline_restored_ts=4000.0, request_integrity_ok=False,
            control_quality_verified=False, resource_state_consistent=False,
            residual_objects=("model-slot-B",), max_rollback_s=600.0,
        )
        assert closure["ok"] is False
        rates = canary.false_rate_accounting(
            outcomes=[{"kind": "CORRECTNESS_BAD", "decision": "PROMOTE", "canary_run_id": "run"}]
        )
        assert rates["false_negative"]


@pytest.mark.unit
class TestSecurity:
    def _threat_model(self, **overrides) -> security.ThreatModel:
        payload = dict(
            scope_id="scope", tenant_definition="namespace + api key", attacker_capability="authenticated tenant",
            trusted_components=("api-server",), protected_assets=("weights", "prompts"),
            boundaries=("namespace", "network"), non_goals=("host root",), failure_policy="fail_closed",
        )
        payload.update(overrides)
        return security.ThreatModel(**payload)

    def test_threat_model_requires_non_goals_and_fail_closed(self) -> None:
        assert self._threat_model().validate() == []
        naive = self._threat_model(non_goals=(), failure_policy="fail_open")
        assert len(naive.validate()) >= 2

    def test_indirect_escalation_is_flagged_for_tenant_subjects(self) -> None:
        report = security.rbac_graph_audit(
            rows=[security.RBACPermission(subject="tenant-a-runtime", verbs=("get",),
                                          resources=("secrets",), scope="namespace")],
            tenant_subjects=("tenant-a-runtime",),
        )
        assert report["ok"] is False

    def test_denied_case_passes_but_successful_attack_fails(self) -> None:
        denied = security.SecurityCase(
            case_id="c1", kind="AUTHZ", subject="tenant-a-user", action="get", resource="tenant-b/model",
            path="gateway-to-backend", authenticated_tenant="tenant-a", claimed_tenant="tenant-b",
            observed_status="403", decision_point="authorizer", audit_event_id="a1",
        )
        assert security.evaluate_security_cases([denied])["ok"] is True
        attacked = security.SecurityCase(
            case_id="c2", kind="AUTHZ", subject="tenant-a-user", action="get", resource="tenant-b/model",
            path="gateway-to-backend", authenticated_tenant="tenant-a", observed_status="200",
            decision_point="authorizer", audit_event_id="a2",
        )
        assert security.evaluate_security_cases([attacked])["ok"] is False

    def test_quota_oversell_and_race(self) -> None:
        ledger = security.QuotaLedger(tenant_id="tenant-a", resource="output_tokens", limit=10.0,
                                      committed=12.0)
        assert ledger.oversell() == 2.0
        race = security.quota_race_trials(
            [{"trial_id": "t", "accepted_cost": 12.0, "limit": 10.0, "rejected": 1, "final_balance": -2.0}]
        )
        assert race["max_oversell"] == 2.0 and race["ok"] is False

    def test_abuse_must_be_rejected_before_resource_allocation(self) -> None:
        report = security.abuse_trials(
            [{"kind": "GIANT_PROMPT", "bounded": True, "rejected_before_resource": False}]
        )
        assert report["ok"] is False

    def test_noisy_neighbor_uses_the_victim_metric(self) -> None:
        report = security.noisy_neighbor(
            trials=[{"trial_id": "t", "victim_alone": 1.0, "victim_with_attacker": 1.5}],
            victim_metric="ttft_seconds", isolation_budget=0.2,
        )
        assert report["ok"] is False
        assert report["max_interference_ratio"] == 1.5

    def test_network_policy_must_be_default_deny_and_tested(self) -> None:
        policy = security.NetworkPolicySpec(policy_id="np", cni="calico", default_ingress_deny=True,
                                            default_egress_deny=True, metadata_service_blocked=True,
                                            tested_paths=("direct_pod_ip",))
        problems = policy.validate()
        assert any("service_dns" in problem for problem in problems)
        assert any("enforcement" in problem for problem in problems)

    def test_invariants_without_evidence_are_not_run(self) -> None:
        report = security.invariant_verdicts(evidence={"MT-I01": {"verdict": "PASS"}})
        assert report["ok"] is False
        missing = [row for row in report["rows"] if row["verdict"] == "NOT_RUN"]
        assert len(missing) == len(records.TENANT_INVARIANTS) - 1

    def test_rotation_window_and_audit_coverage(self) -> None:
        rotation = security.rotation_recovery(
            rows=[{"case_id": "c", "kind": "revocation", "observed_window_s": 900.0, "consistent": False}],
            policy_window_s=300.0,
        )
        assert rotation["ok"] is False
        audit = security.audit_coverage_report(
            [security.AuditCoverage(event_kind="authn_failure", emitted=False)], max_lag_s=60.0
        )
        assert audit["ok"] is False
