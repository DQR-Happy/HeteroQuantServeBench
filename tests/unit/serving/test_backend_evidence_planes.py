"""Backend + evidence plane tests: router, circuit, cache, faults, observability."""

from __future__ import annotations

import os

import pytest

from hqsb.core.errors import ConfigError
from hqsb.serving import cache_routing, circuit, faults, observability, router

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CONFIG = os.path.join(_REPO_ROOT, "configs", "serving")


def _record(instance_id: str, queue_depth: float = 0.0, aliases=("dummy-model",)):
    return router.BackendRecord(
        instance_id=instance_id,
        adapter_version="1",
        runtime_commit="c",
        hardware="jetson",
        device="cuda:0",
        parallel_degree=1,
        model_identity={"model_id": "m", "revision": "r", "model_manifest_sha256": "0" * 64},
        model_aliases=aliases,
        precision="float16",
        quality_class="fp16",
        max_context_tokens=32768,
        model_epoch=f"epoch-{instance_id}",
        ready=True,
        telemetry=router.Telemetry(monotonic_ns=0, queue_depth=queue_depth),
    )


def _spec():
    import yaml

    with open(os.path.join(_CONFIG, "routing_spec.yaml"), encoding="utf-8") as handle:
        return router.ScoreSpec.from_document(yaml.safe_load(handle))


def _request(alias="dummy-model", stream=True):
    return router.RouteRequest(
        request_id="r0",
        model_alias=alias,
        model_identity={"model_manifest_sha256": "0" * 64},
        precision="float16",
        quality_class="fp16",
        prompt_tokens=100,
        reserved_output_tokens=64,
        stream=stream,
    )


@pytest.mark.unit
class TestRouter:
    def test_registry_refuses_duplicate_ids(self):
        registry = router.BackendRegistry()
        registry.register(_record("a"))
        with pytest.raises(ConfigError):
            registry.register(_record("a"))

    def test_generation_update_is_atomic(self):
        registry = router.BackendRegistry()
        registry.register(_record("a", aliases=("x",)))
        with pytest.raises(ConfigError):
            registry.replace_generation([_record("b")], generation=1)  # not increasing
        registry.replace_generation(
            [_record("a", aliases=("x",)), _record("b", aliases=("y",))], generation=2
        )
        assert set(registry.snapshot().ids()) == {"a", "b"}

    def test_hard_filter_excludes_wrong_alias(self):
        registry = router.BackendRegistry()
        registry.register(_record("a"))
        feasible, rejected = router.hard_filter(
            _request(alias="ghost"),
            registry.snapshot(),
            now_ns=0,
            telemetry_ttl_ms=2000,
        )
        assert not feasible and rejected and rejected[0].failed_stage == "model_alias_resolve"

    def test_route_selects_lowest_score(self):
        registry = router.BackendRegistry()
        registry.register(_record("a", queue_depth=0.0))
        registry.register(_record("b", queue_depth=5.0))
        decision = router.route(_request(), registry, spec=_spec(), now_ns=1000)
        assert not decision.no_feasible
        assert decision.selected_instance_id == "a"

    def test_route_vs_actual_is_checked(self):
        registry = router.BackendRegistry()
        registry.register(_record("a"))
        decision = router.route(_request(), registry, spec=_spec(), now_ns=1000)
        assert router.route_vs_actual([decision], {"r0": ("a", "epoch-a")})["ok"]
        assert not router.route_vs_actual([decision], {"r0": ("b", "epoch-a")})["ok"]

    def test_no_feasible_is_a_stable_reject(self):
        registry = router.BackendRegistry()
        registry.register(_record("a", aliases=("other",)))
        decision = router.route(_request(), registry, spec=_spec(), now_ns=1000)
        assert decision.no_feasible and decision.fallback_level == "fail_closed_reject"

    def test_fallback_needs_permission(self):
        plan = router.fallback_plan(
            request=_request(), level="approved_alternate_precision", reason="busy"
        )
        assert not plan["permitted"]


@pytest.mark.unit
class TestCircuit:
    def _spec(self):
        import yaml

        with open(os.path.join(_CONFIG, "fault_spec.yaml"), encoding="utf-8") as handle:
            return circuit.CircuitSpec.from_document(yaml.safe_load(handle))

    def test_excluded_failures_do_not_count(self):
        breaker = circuit.CircuitBreaker(self._spec(), instance_id="a")
        assert breaker.record_failure("client_4xx", monotonic_ns=1) is None
        assert breaker.counters()["window_failures"] == 0

    def test_unknown_failure_class_is_refused(self):
        breaker = circuit.CircuitBreaker(self._spec(), instance_id="a")
        with pytest.raises(ConfigError):
            breaker.record_failure("mystery", monotonic_ns=1)

    def test_quarantine_is_not_backoff(self):
        breaker = circuit.CircuitBreaker(self._spec(), instance_id="a")
        breaker.record_failure("model_identity_mismatch", monotonic_ns=1)
        assert breaker.quarantined
        assert not breaker.allow_request(monotonic_ns=2)["allowed"]
        with pytest.raises(ConfigError):
            breaker.clear_quarantine(monotonic_ns=3, operator="")  # operator required

    def test_breaker_opens_and_recovers(self):
        spec = self._spec()
        breaker = circuit.CircuitBreaker(spec, instance_id="a")
        now = 0
        for _ in range(spec.consecutive_failure_threshold):
            now += 1_000_000
            breaker.record_failure("backend_unavailable", monotonic_ns=now)
        assert breaker.state == "OPEN"
        probe = breaker.allow_request(monotonic_ns=now + int(spec.open_duration_ms * 1e6) + 1)
        assert probe["allowed"] and breaker.state == "HALF_OPEN"
        for _ in range(spec.half_open_probes_required):
            now += 1_000_000
            breaker.record_success(monotonic_ns=now)
        assert breaker.state == "CLOSED"


@pytest.mark.unit
class TestCacheRouting:
    def test_identity_negative_fixtures(self):
        identity = cache_routing.CacheIdentity(
            fields={
                "model_id": "m",
                "weight_revision": "r1",
                "precision": "fp16",
                "quant_artifact_hash": "q",
                "adapter_hash": "a",
                "tokenizer_id": "t",
                "chat_template_hash": "c",
                "rope_config_hash": "r",
                "attention_config_hash": "at",
                "kv_dtype": "fp16",
                "kv_layout": "v1",
                "cache_format_version": "1",
                "tenant_sharing_policy": "default",
                "cache_epoch": "e1",
            }
        )
        fixtures = cache_routing.identity_negative_fixtures(identity)
        assert len(fixtures) == len(cache_routing.IDENTITY_FIELDS)
        assert all(item["digest_differs"] for item in fixtures)

    def test_version_invalidation(self):
        identity = cache_routing.CacheIdentity(
            fields={
                **{name: str(name) for name in cache_routing.IDENTITY_FIELDS},
                "cache_epoch": "e1",
            }
        )
        new = cache_routing.CacheIdentity(
            fields={**dict(identity.fields), "cache_epoch": "e2"}
        )
        report = cache_routing.version_invalidation_check(identity, new)
        assert "cache_epoch" in report["differing_fields"]
        assert report["old_cache_usable"] is False

    def test_matcher_oracle_agrees(self):
        identity = cache_routing.CacheIdentity(
            fields={name: str(name) for name in cache_routing.IDENTITY_FIELDS}
        )
        matcher = cache_routing.PrefixMatcher(block_size=4)
        matcher.insert(
            cache_routing.PrefixEntry(
                identity=identity, tokens=tuple(range(16)), block_size=4
            )
        )
        oracle = cache_routing.matcher_oracle(
            matcher, identity=identity, query=list(range(10))
        )
        assert oracle["ok"]

    def test_joint_policy_scores_net_value(self):
        import yaml

        with open(os.path.join(_CONFIG, "cache_routing_spec.yaml"), encoding="utf-8") as handle:
            weights = cache_routing.JointWeights.from_document(yaml.safe_load(handle))
        identity = cache_routing.CacheIdentity(
            fields={name: str(name) for name in cache_routing.IDENTITY_FIELDS}
        )
        matcher = cache_routing.PrefixMatcher(block_size=4)
        matcher.insert(
            cache_routing.PrefixEntry(identity=identity, tokens=tuple(range(16)), block_size=4)
        )
        request = cache_routing.CacheRouteRequest(
            request_id="r0",
            identity=identity,
            query_tokens=tuple(range(8)),
            queue_ms={"i0": 10.0, "i1": 1.0},
            decode_ms={"i0": 1.0, "i1": 1.0},
        )
        decision = cache_routing.cache_route(
            policy="joint_locality_load",
            request=request,
            matcher=matcher,
            telemetry=[
                cache_routing.CacheTelemetrySample(
                    instance_id="i0", monotonic_ns=0, cache_epoch="e1",
                    capacity_bytes=1000, free_bytes=100,
                ),
                cache_routing.CacheTelemetrySample(
                    instance_id="i1", monotonic_ns=0, cache_epoch="e1",
                    capacity_bytes=1000, free_bytes=100,
                ),
            ],
            instance_ids=["i0", "i1"],
            weights=weights,
            per_token_prefill_us=1.0,
            now_ns=0,
            ttl_ms=2000,
        )
        assert decision.selected_instance_id == "i1"  # least load, no locality benefit

    def test_skew_report_bounds(self):
        report = cache_routing.skew_report({"a": 10.0, "b": 10.0})
        assert report["ok"]
        report = cache_routing.skew_report({"a": 100.0, "b": 1.0})
        assert not report["ok"]


@pytest.mark.unit
class TestFaults:
    def test_commit_phase_boundary(self):
        assert faults.commit_phase(external_bytes_committed=0, backend_started=False) == "before_backend_start"
        assert (
            faults.commit_phase(external_bytes_committed=0, backend_started=True)
            == "backend_started_no_external_bytes"
        )
        assert faults.commit_phase(external_bytes_committed=1, backend_started=True) == "after_external_commit"

    def test_retry_is_forbidden_after_commit(self):
        decision = faults.retry_policy_after_fault(
            phase="after_external_commit",
            model_rng_identity_proven=True,
            cancel_and_cleanup_proven=True,
            budget_remaining=True,
            fallback_verified=True,
        )
        assert decision["retry_allowed"] is False

    def test_attempt_lineage_audit(self):
        attempts = [
            faults.Attempt(
                parent_request_id="p",
                attempt_id="p#1",
                instance_id="a",
                model_epoch="e",
                start_ns=0,
                end_ns=10,
                emitted_tokens=3,
                cleaned=True,
            )
        ]
        assert faults.attempt_lineage_audit(attempts)["ok"]
        attempts.append(
            faults.Attempt(
                parent_request_id="p",
                attempt_id="p#2",
                instance_id="b",
                model_epoch="e",
                start_ns=20,
                end_ns=30,
                emitted_tokens=3,
                cleaned=True,
            )
        )
        # retried after 3 externally visible tokens -> problem
        assert not faults.attempt_lineage_audit(attempts)["ok"]


@pytest.mark.unit
class TestObservability:
    def test_traceparent_validation(self):
        good = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
        parsed = observability.parse_traceparent(good, trusted_source=False)
        assert parsed["valid"] and not parsed["trusted"]
        assert not observability.parse_traceparent("bad", trusted_source=True)["valid"]

    def test_redaction_forbids_prompt(self):
        assert observability.redact_text("hello")["sha256"]
        audit = observability.redaction_audit([{"prompt": "secret"}])
        assert not audit["ok"]

    def test_histogram_quantiles_are_recomputed(self):
        histogram = observability.Histogram(
            name="client_ttft_ms", unit="ms", buckets=(10.0, 50.0, 100.0, 500.0)
        )
        for value in (5.0, 20.0, 30.0, 60.0):
            histogram.observe(value)
        assert histogram.quantile(0.5) == pytest.approx(25.0)

    def test_root_cause_needs_counterfactual(self):
        claim = observability.RootCauseClaim(
            request_id="r0",
            root_cause="admission_or_queue_hol",
            evidence_span="queue_wait",
        )
        assert claim.verdict()["status"] == "NOT_PROVEN"
        claim = observability.RootCauseClaim(
            request_id="r0",
            root_cause="admission_or_queue_hol",
            evidence_span="queue_wait",
            counterfactual_supported=True,
            counterfactual="removing the queue entry removed the delay",
        )
        assert claim.verdict()["status"] == "ROOT_CAUSE"
