"""Negative-path tests for E13-01…E13-05 (supply chain → lifecycle)."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.infra import artifacts, deployment, lifecycle, scheduling, supply_chain

DIGEST = "sha256:" + "c" * 64


def _runtime_context(**overrides):
    payload = dict(
        uid=10001,
        gid=10001,
        read_only_rootfs=True,
        capabilities=("CAP_CHOWN",),
        seccomp_profile="runtime/default",
        writable_paths=("/tmp", "/cache"),
        seccomp_inherited=True,
    )
    payload.pop("seccomp_inherited", None)
    payload.update(overrides)
    return supply_chain.RuntimeSecurityContext(**payload)


@pytest.mark.unit
class TestSupplyChain:
    def test_build_inputs_reject_floating_base(self) -> None:
        inputs = supply_chain.BuildInputs(
            base_image_digest="ubuntu:24.04", builder_image_digest=DIGEST,
            package_index_snapshot="idx", lockfiles={"l": "h"},
        )
        assert any("digest" in problem for problem in inputs.validate())

    def test_layer_plan_rejects_toolchain_and_model_in_runtime(self) -> None:
        plan = supply_chain.LayerPlan(
            stages={"base": ("libc",), "toolchain": ("gcc", "nvcc"), "runtime": ("gcc", "model-weights.safetensors")},
            copy_from_toolchain=(),
        )
        problems = plan.validate()
        assert any("toolchain" in problem for problem in problems)
        assert any("model" in problem for problem in problems)
        assert any("copy" in problem.lower() for problem in problems)

    def test_runtime_security_context_requires_non_root_and_readonly(self) -> None:
        assert _runtime_context().validate() == []
        problems = _runtime_context(uid=0, gid=0, read_only_rootfs=False, capabilities=("CAP_SYS_ADMIN",)).validate()
        assert any("non-root" in problem for problem in problems)
        assert any("read-only" in problem for problem in problems)
        assert any("CAP_SYS_ADMIN" in problem for problem in problems)

    def test_write_paths_and_capability_minimum(self) -> None:
        context = _runtime_context()
        report = supply_chain.validate_write_paths(context, ["/tmp/x", "/etc/passwd"])
        assert report["unexpected"] == ["/etc/passwd"]
        assert supply_chain.validate_device_capability_minimum(capabilities=["CAP_SYS_PTRACE"])["ok"] is False

    def test_scanner_unavailable_is_not_zero_findings(self) -> None:
        scan = supply_chain.VulnerabilityScan(
            scanner="s", scanner_version="1", db_snapshot="2026-09-01", scanner_status="unavailable"
        )
        report = supply_chain.scan_status_semantics(scan)
        assert report["status"] == "SCANNER_UNAVAILABLE"
        assert report["zero_findings"] is False

    def test_exception_needs_owner_and_expiry(self) -> None:
        exception = supply_chain.VulnerabilityException(finding_id="f1")
        assert len(exception.validate()) >= 5
        expired = supply_chain.VulnerabilityException(
            finding_id="f1", owner="sre", risk="low", compensating_control="egress blocked",
            expires_at="2020-01-01", retest_trigger="base image change",
        )
        assert expired.expired(now="2026-09-19") is True

    def test_sbom_completeness_finds_orphans(self) -> None:
        sbom = supply_chain.Sbom(
            sbom_id="sbom-1",
            image_digest=DIGEST,
            format="spdx",
            schema_version="3.0.1",
            tool="syft",
            tool_version="1",
            components=(supply_chain.SbomComponent(component_id="c1", name="libc"),),
        )
        report = supply_chain.sbom_completeness(
            sbom, filesystem_inventory={"libc.so.6": DIGEST, "vendor/libhqsb.so": DIGEST}
        )
        assert report["status"] == "INCOMPLETE"
        assert "vendor/libhqsb.so" in report["orphans"]

    def test_secret_and_model_scan_fail_the_gate(self) -> None:
        scan = supply_chain.scan_build_context(
            [{"surface": "image_history", "path": "/opt/app/.netrc", "content": "api_key = supersecretvalue"}],
            patterns=("*.safetensors",),
        )
        assert scan["secret_findings"] and scan["surfaces_missing"]
        gate = supply_chain.build_negative_gate(scan)
        assert gate["gate_decision"] == "FAIL"

    def test_attestation_closure_detects_wrong_subject(self) -> None:
        attestation = supply_chain.Attestation(
            attestation_id="a1", subject_digest="sha256:" + "d" * 64, signer_identity="ci",
            issuer="issuer", verified=True,
        )
        report = supply_chain.verify_attestation_closure(
            candidate_digest=DIGEST, sbom=None, provenance=None, attestation=attestation
        )
        assert report["status"] == "FAILED"
        assert any("different subject" in problem for problem in report["problems"])

    def test_negative_cases_must_be_rejected(self) -> None:
        gate = supply_chain.SupplyChainGateResult(
            release_id="rel-1", image_index_digest=DIGEST, gate_decision="FAIL", reason_codes=("SECRET_FOUND",)
        )
        report = supply_chain.run_negative_cases(
            [{"kind": "INJECTED_SECRET", "expected": "REJECT", "observed": "ACCEPT"}], gate=gate
        )
        assert report["all_rejected"] is False
        assert report["false_pass"]

    def test_size_inventory_is_diagnostic_only(self) -> None:
        report = supply_chain.image_size_inventory(
            [{"layer_id": "l1", "compressed_bytes": 10, "uncompressed_bytes": 20, "contains_toolchain": True}]
        )
        assert report["toolchain_leaks"] and "diagnostic" in report["note"]


@pytest.mark.unit
class TestDeployment:
    def test_preconditions_detect_contamination(self) -> None:
        scope = deployment.CleanScope(
            clean_level="L1_CLEAN_NAMESPACE", campaign_id="c1", cluster_id="cl", namespace="ns",
            budget_seconds=600, cleanup_policy="delete-namespace",
        )
        report = deployment.evaluate_preconditions({"warm_model_cache": "present"}, scope=scope)
        assert report["verdict"] == "CONTAMINATED"

    def test_clean_scope_rejects_overlapping_components(self) -> None:
        scope = deployment.CleanScope(
            clean_level="L2_CLEAN_WORKLOAD_CLUSTER", campaign_id="c", cluster_id="cl", namespace="ns",
            budget_seconds=1, cleanup_policy="delete",
            preexisting_components=("device_plugin",), workflow_installed_components=("device_plugin",),
        )
        assert any("both" in problem for problem in scope.validate())

    def test_illegal_deployment_transition_is_detected(self) -> None:
        events = [
            deployment.DeploymentStageEvent(
                deployment_run_id="r", event_id="e1", stage="ready", state_from="BASELINE_VERIFIED",
                state_to="ENDPOINT_READY",
            )
        ]
        report = deployment.deployment_timeline(events)
        assert report["ok"] is False

    def test_readiness_and_healthz(self) -> None:
        claim = deployment.ReadinessClaim(ready=True, healthz_status=200, model_state="ABSENT")
        assert deployment.readiness_verdict(claim)["ready"] is False
        good = deployment.ReadinessClaim(
            ready=True, release_id="r", model_artifact_id="m", backend_id="cuda", model_state="ACTIVE",
            release_digest_verified=True, warmup_complete=True, quality_probe_passed=True,
            minimum_capacity_available=True, engine_kernel_backend_identity_known=True,
            tokenizer_config_precision_compatible=True, endpoint_slice_targets=("pod-1",),
        )
        assert deployment.readiness_verdict(good)["ready"] is True

    def test_pre_ready_traffic_exclusion(self) -> None:
        report = deployment.pre_ready_traffic_exclusion(
            [{"request_id": "r1", "reached_pod": True, "pod_ready": False, "status": "200"}]
        )
        assert report["ok"] is False
        queued = deployment.pre_ready_traffic_exclusion(
            [{"request_id": "r2", "reached_pod": True, "pod_ready": False, "status": "503"}]
        )
        assert queued["ok"] is True

    def test_first_request_correctness_is_required(self) -> None:
        record = deployment.FirstRequestRecord(
            request_id="r1", release_id="rel", model_artifact_id="m", backend_id="cuda",
            output_hash="sha256:" + "1" * 64, reference_hash="sha256:" + "2" * 64,
            status="COMPLETED", stream_complete=True,
        )
        assert any("differs from the frozen reference" in problem for problem in record.validate())

    def test_cold_warm_mixing_is_refused(self) -> None:
        with pytest.raises(ConfigError):
            deployment.cold_warm_stage_times([], cache_state="mixed")

    def test_automation_verdict_and_cleanup(self) -> None:
        verdict = deployment.automation_verdict(
            [deployment.ManualIntervention(deployment_run_id="r", actor="author", command="kubectl edit")]
        )
        assert verdict["automated"] is False
        scope = deployment.CleanScope(
            clean_level="L1_CLEAN_NAMESPACE", campaign_id="c", cluster_id="cl", namespace="ns",
            budget_seconds=1, cleanup_policy="delete",
        )
        cleanup = deployment.evaluate_failure_cleanup(
            scope=scope, residuals=[{"kind": "finalizer", "object_id": "pod-1", "released": False}]
        )
        assert cleanup["ok"] is False


@pytest.mark.unit
class TestScheduling:
    def _node(self, **overrides):
        device = scheduling.DeviceRecord(
            device_id="GPU-0", vendor="NVIDIA", product="RTX3090", arch="sm_86",
            memory_bytes=24 << 30, driver_version="550", health="healthy", numa_node=0,
            link_domain="PCIe", source="device_plugin",
        )
        payload = dict(
            node_id="node-1", cpu_arch="x86_64", runtime_version="containerd", kubelet_version="v1.30",
            topology_policy="single-numa-node", cpu_manager_policy="static", labels={"vendor": "NVIDIA"},
            devices=(device,),
        )
        payload.update(overrides)
        return scheduling.NodeInventory(**payload)

    def _plan(self, **overrides):
        payload = dict(
            placement_plan_id="p1", workload_id="w1", required_vendor="NVIDIA", required_arch="sm_86",
            min_memory_bytes=16 << 30, device_count=1, rank_to_device={"0": "GPU-0"},
        )
        payload.update(overrides)
        return scheduling.PlacementPlan(**payload)

    def test_unknown_health_device_is_flagged(self) -> None:
        device = scheduling.DeviceRecord(
            device_id="GPU-1", vendor="NVIDIA", product="X", arch="sm_86", memory_bytes=1,
            driver_version="1", health="unknown", source="plugin",
        )
        assert any("health" in problem for problem in device.validate())

    def test_protected_label_requires_protection_and_source(self) -> None:
        label = scheduling.CapabilityLabel(
            node_id="n1", label="accel.arch", value="sm_86", label_class="arch", source="node",
            owner="sre", ttl_s=3600, protected=False, verified_at="2026-09-19T00:00:00Z",
        )
        assert any("protected" in problem for problem in label.validate())

    def test_hard_capability_is_not_a_soft_preference(self) -> None:
        plan = self._plan(soft_preferences=("sm_86",))
        assert any("soft preference" in problem for problem in plan.validate())

    def test_filter_rejects_wrong_arch_and_memory(self) -> None:
        feasible = scheduling.filter_nodes(self._plan(), [self._node()])
        assert feasible["feasible"] == ["node-1"]
        wrong = scheduling.filter_nodes(self._plan(required_arch="sm_90", min_memory_bytes=80 << 30), [self._node()])
        assert wrong["no_feasible_node"] is True
        assert wrong["unschedulable_reasons"]["node-1"]

    def test_visibility_leak_and_rank_mismatch(self) -> None:
        plan = self._plan()
        evidence = scheduling.PlacementEvidence(
            placement_run_id="r", placement_plan_id="p1", pod_uid="pod", scheduled_node="node-1",
            physical_or_partition_device_ids=("GPU-0",), runtime_visible_device_ids=("GPU-0", "GPU-1"),
            actual_execution_device_ids=("GPU-0",), rank_mapping={"0": "GPU-1"},
        )
        problems = evidence.validate(plan)
        assert any("not allocated" in problem for problem in problems)
        assert scheduling.validate_rank_mapping(plan, evidence)["ok"] is False

    def test_isolation_verdict_only_claims_tested_layers(self) -> None:
        verdict = scheduling.isolation_verdict(
            plan=self._plan(isolation_policy="exclusive"),
            exclusive_rows=[{"second_pod_admitted": True}],
            shared_rows=[],
            unauthorized={"ok": True},
            telemetry_visibility={"ok": False},
        )
        assert verdict["ok"] is False
        assert "scheduler_accounting" not in verdict["verified_layers"]


@pytest.mark.unit
class TestArtifacts:
    def _key(self, **overrides):
        payload = dict(
            artifact_id="qwen3", model_root="qwen3@rev1", tokenizer_id="tok", config_id="cfg",
            target_arch="sm_86", runtime_abi="cuda-12.4",
        )
        payload.update(overrides)
        return artifacts.CacheKey(**payload)

    def test_name_only_cache_key_is_rejected(self) -> None:
        weak = artifacts.CacheKey(artifact_id="qwen3", model_root="qwen3")
        assert len(weak.validate()) >= 3

    def test_download_must_not_write_the_active_path(self) -> None:
        download = artifacts.StagingDownload(
            attempt_id="a1", artifact_id="q", uri="s3://m", staging_path="/cache/active",
            active_path="/cache/active",
        )
        assert any("active path" in problem for problem in download.validate())

    def test_verification_must_pass_before_commit(self) -> None:
        download = artifacts.StagingDownload(
            attempt_id="a1", artifact_id="q", uri="s3://m", staging_path="/s", active_path="/v",
            bytes_expected=10, bytes_downloaded=10,
        )
        bad = artifacts.FileVerification(path="shard-2", expected_hash="sha256:" + "a" * 64,
                                         observed_hash="sha256:" + "b" * 64, expected_size=1, observed_size=1)
        report = artifacts.verify_download(
            download=download, files=[bad], aggregate_root_expected="x", aggregate_root_observed="x"
        )
        assert report["may_commit"] is False

    def test_marker_before_data_is_refused(self) -> None:
        commit = artifacts.AtomicCommit(attempt_id="a", artifact_id="q", data_durable=False,
                                        marker_written_after_data=False, rescanned_after_commit=False)
        assert len(commit.validate()) >= 3

    def test_gc_protects_pinned_and_active(self) -> None:
        safety = artifacts.gc_safety_check(
            deleted=["active-1"], protected=["active-1"], active="active-1", rollback_target="rb-1"
        )
        assert safety["ok"] is False

    def test_single_flight_and_version_consistency(self) -> None:
        race = artifacts.single_flight_outcome(
            [{"role": "leader", "committed": True, "cache_key": "k"},
             {"role": "leader", "committed": True, "cache_key": "k"}]
        )
        assert race["ok"] is False
        mixed = artifacts.validate_request_version_consistency(
            [{"request_id": "r", "artifact_ids_seen": ["v1", "v2"], "status": "COMPLETED"}]
        )
        assert mixed["ok"] is False

    def test_activation_requires_quality_and_retained_rollback(self) -> None:
        key = self._key()
        warmup = artifacts.WarmupResult(artifact_id="qwen3", shapes=("1x1024",), rounds=3,
                                        stable_per_round=True, quality_status="pass", capacity_available=True)
        report = artifacts.activate_version(
            cache_key=key, commit={"committed": True, "problems": []}, compatibility={"status": "COMPATIBLE"},
            warmup=warmup, capacity_available=True,
            prior=artifacts.ActivationGeneration(
                generation_id="g0", artifact_id="qwen3-old", activation_mode="POD_ROLLING",
                linearization_ts="t0", quality_passed=True, capacity_available=True,
            ),
            generation_id="g1", linearization_ts="t1", command_returned_ts="t1",
            existing_active_lease_ids=("lease-1",), retired_lease_ids=("qwen3-old",),
        )
        assert report["activated"] is False
        assert report["rollback_target_retained"] is False

    def test_fault_cases_must_not_serve(self) -> None:
        report = artifacts.run_fault_cases(
            [{"kind": "TAMPERED_SHARD", "served_traffic": True, "recovered": True}]
        )
        assert report["ok"] is False


@pytest.mark.unit
class TestLifecycle:
    def _policy(self, **overrides):
        payload = dict(
            policy_id="t1", grace_period_s=120.0, prestop_s=10.0, stop_accepting_s=1.0,
            max_generation_s=60.0, slow_client_policy="bounded", telemetry_flush_s=5.0,
            artifact_lease_release="after-inflight", inflight_policy="finish-or-cancel",
            retry_idempotency="before-first-token", completion_semantics=("COMPLETED_EXACT_VERSION",),
        )
        payload.update(overrides)
        return lifecycle.TerminationPolicy(**payload)

    def test_probe_endpoints_must_differ(self) -> None:
        config = lifecycle.ProbeConfig(
            probe_config_id="p", endpoints={"startup": "/healthz", "readiness": "/healthz", "liveness": "/healthz"},
            period_s={"startup": 1.0, "readiness": 1.0, "liveness": 1.0},
            failure_threshold={"startup": 1, "readiness": 1, "liveness": 1}, reason_codes=("x",),
            startup_covers_stages=("model_load", "warmup"),
        )
        assert any("same endpoint" in problem for problem in config.validate())

    def test_termination_budget_must_cover_generation_and_flush(self) -> None:
        assert self._policy().validate() == []
        over = self._policy(grace_period_s=30.0, prestop_s=30.0, max_generation_s=60.0)
        assert len(over.validate()) >= 2

    def test_post_drain_acceptance_is_detected(self) -> None:
        timeline = lifecycle.DrainTimeline(
            episode_id="e", delete_ts="t0", sigterm_ts="t1", not_ready_ts="t1", exit_ts="t2",
            accepted_after_drain=2,
        )
        assert any("after the drain point" in problem for problem in timeline.validate())

    def test_truncated_stream_is_not_complete(self) -> None:
        report = lifecycle.request_integrity_report(
            [lifecycle.TokenIntegrity(
                request_id="r", expected_tokens=(1, 2, 3), observed_tokens=(1, 2), prefix_ok=True,
                status="COMPLETED", model_version="v1", final_statuses=("COMPLETED",),
            )]
        )
        assert report["ok"] is False

    def test_post_first_token_retry_is_flagged(self) -> None:
        report = lifecycle.retry_duplicate_accounting(
            [{"request_id": "r", "retryable": True, "before_first_token": False}]
        )
        assert report["ok"] is False

    def test_forced_kill_is_not_a_graceful_pass(self) -> None:
        timeline = lifecycle.DrainTimeline(
            episode_id="e", delete_ts="t0", sigterm_ts="t1", not_ready_ts="t1", exit_ts="t2", forced_kill=True
        )
        report = lifecycle.forced_kill_accounting(
            timeline=timeline, lost_requests=1, partial_streams=1, retryable=0, slo_violated=False
        )
        assert report["ok"] is False
        assert report["verdict"] == "FORCED_WITH_DECLARED_VIOLATION"

    def test_resource_release_requires_every_target(self) -> None:
        report = lifecycle.resource_release_report(
            [{"resource": "device_context", "released": False}]
        )
        assert report["ok"] is False

    def test_progress_deadline_and_rolling_policy(self) -> None:
        policy = lifecycle.RollingPolicy(
            policy_id="rp", replicas=4, max_surge=1, max_unavailable=0, pod_disruption_budget=2,
            min_ready_seconds=30, progress_deadline_s=1800, termination_grace_s=120,
            rollback_conditions=("slo burn > budget",), serialize_with_autoscaler=True,
        )
        assert policy.validate() == []
        stuck = lifecycle.validate_progress_deadline(
            rollout_started="t0", stuck_candidate=True, deadline_s=1800, action_taken="NONE"
        )
        assert stuck["ok"] is False
