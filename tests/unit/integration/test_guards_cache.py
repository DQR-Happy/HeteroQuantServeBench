"""Guard/recompile accounting and compile-cache tests (E06-05/E06-06)."""

from __future__ import annotations

import os
import stat

import pytest

from hqsb.core.errors import ConfigError, SchemaError
from hqsb.integration import cache, guards


@pytest.mark.unit
class TestGuardRecords:
    def test_unknown_guard_kind_refused(self):
        with pytest.raises(SchemaError):
            guards.GuardRecord(kind="vibes", expression="x", source="fake")

    def test_guard_record_carries_source_and_consequence(self):
        record = guards.GuardRecord(
            kind=guards.GuardKind.SHAPE,
            expression="S == 128",
            source="dynamo",
            first_input="B1/S128",
            failing_input="B1/S512",
            consequence="recompiled variant",
        )
        assert record.as_dict()["consequence"] == "recompiled variant"


@pytest.mark.unit
class TestCompileLedger:
    def test_five_way_counts_are_separate(self):
        ledger = guards.CompileLedger()
        ledger.record(guards.EventKind.GUARD_FAIL, 0, reason="shape")
        ledger.record(guards.EventKind.GUARD_FAIL, 1, reason="stride")
        ledger.record(guards.EventKind.RECOMPILE, 1, reason="new shape")
        ledger.record(guards.EventKind.GRAPH_BREAK, 1, reason="data dependent")
        ledger.record(guards.EventKind.RUNTIME_ASSERT, 2, reason="bound")
        ledger.record(guards.EventKind.FALLBACK, 3, reason="unsupported layout")
        payload = ledger.five_way().as_dict()
        assert payload["guard_failures"] == 2
        assert payload["recompiles"] == 1
        assert payload["graph_breaks"] == 1
        assert payload["runtime_asserts"] == 1
        assert payload["fallbacks"] == 1
        assert payload["requests"] == 4
        assert ledger.five_way().recompiles_per_request() == 0.25

    def test_unknown_event_kind_refused(self):
        with pytest.raises(SchemaError):
            guards.CompileLedger().record("recompile_maybe", 0)

    def test_timestamps_are_monotonic(self):
        ledger = guards.CompileLedger()
        ledger.record(guards.EventKind.CAPTURE, 0)
        ledger.record(guards.EventKind.KERNEL, 0)
        stamps = [event.timestamp_ns for event in ledger.events]
        assert stamps == sorted(stamps)

    def test_reasons_are_retrievable_per_kind(self):
        ledger = guards.CompileLedger()
        ledger.record(guards.EventKind.RECOMPILE, 0, graph_variant="v2", reason="shape")
        assert ledger.reasons_for(guards.EventKind.RECOMPILE) == (("v2", "shape"),)

    def test_jsonl_export_round_trips(self):
        ledger = guards.CompileLedger()
        ledger.record(guards.EventKind.FALLBACK, 0, reason="stride")
        assert "fallback" in ledger.to_jsonl()


@pytest.mark.unit
class TestDynamicPolicy:
    def test_dimension_bounds_are_validated(self):
        with pytest.raises(ConfigError):
            guards.DynamicDimensionSpec(name="batch", symbol="B", bounds=(0, 8))

    def test_bucket_outside_bounds_refused(self):
        with pytest.raises(ConfigError):
            guards.DynamicDimensionSpec(name="seq", symbol="S", bounds=(1, 128), buckets=(256,))

    def test_bucket_for_rounds_up(self):
        dim = guards.DynamicDimensionSpec(
            name="seq", symbol="S", bounds=(1, 512), buckets=(32, 128, 512)
        )
        assert dim.bucket_for(1) == 32
        assert dim.bucket_for(33) == 128
        assert dim.bucket_for(512) == 512

    def test_out_of_range_value_is_refused_not_padded(self):
        dim = guards.DynamicDimensionSpec(name="seq", symbol="S", bounds=(1, 128), buckets=(128,))
        with pytest.raises(ConfigError):
            dim.bucket_for(4096)

    def test_resolve_compiler_config_records_the_flag(self):
        assert guards.resolve_compiler_config(guards.DynamicPolicy.STATIC).dynamic_flag is False
        assert guards.resolve_compiler_config(guards.DynamicPolicy.DYNAMIC).dynamic_flag is True
        auto = guards.resolve_compiler_config(guards.DynamicPolicy.AUTO)
        assert auto.dynamic_flag is None
        assert auto.notes

    def test_bucketed_policy_requires_buckets(self):
        with pytest.raises(ConfigError):
            guards.resolve_compiler_config(guards.DynamicPolicy.BUCKETED)

    def test_unknown_policy_refused(self):
        with pytest.raises(ConfigError):
            guards.resolve_compiler_config("whatever")


@pytest.mark.unit
class TestShapeTrace:
    def test_trace_matches_the_protocol_sequence(self):
        trace = guards.ordered_shape_trace()
        assert len(trace) == 10
        assert trace[0].label == "B1/S32"
        assert trace[2].label == "B1/S32"
        assert trace[6].stride_class == "transposed"
        assert trace[7].dtype == "float64"

    def test_repeated_shape_evidence(self):
        trace = guards.ordered_shape_trace()
        records = (
            guards.TraceRequestRecord(point=trace[0], graph_variant="v1", guard_hit=True),
            guards.TraceRequestRecord(point=trace[1], graph_variant="v1", guard_hit=True),
            guards.TraceRequestRecord(point=trace[2], graph_variant="v1", guard_hit=True),
        )
        evidence = guards.reuse_evidence(records)
        assert evidence
        assert evidence[0]["reused"] is True
        assert evidence[0]["request_index"] == 2

    def test_recompile_breaks_reuse_evidence(self):
        trace = guards.ordered_shape_trace()
        records = (
            guards.TraceRequestRecord(point=trace[0], graph_variant="v1"),
            guards.TraceRequestRecord(point=trace[2], graph_variant="v2", recompiled=True),
        )
        evidence = guards.reuse_evidence(records)
        assert evidence[0]["reused"] is False


@pytest.mark.unit
class TestStorm:
    def _observation(self, **overrides):
        payload = {
            "requests": 1000,
            "recompiles": 2,
            "compile_wall_ms": 10.0,
            "total_wall_ms": 1000.0,
            "unique_shapes": 4,
            "graph_variants": 4,
            "fallbacks": 0,
            "baseline_p95_ms": 10.0,
            "observed_p95_ms": 10.0,
            "cache_entries_start": 0,
            "cache_entries_end": 4,
        }
        payload.update(overrides)
        return guards.StormObservation(**payload)

    def test_within_thresholds(self):
        report = guards.evaluate_storm(guards.StormThresholds(), self._observation())
        assert report.within_thresholds
        assert report.breaches == ()

    def test_recompile_rate_breach_is_reported(self):
        report = guards.evaluate_storm(
            guards.StormThresholds(), self._observation(recompiles=50)
        )
        assert not report.within_thresholds
        assert report.breaches[0]["reason"] == "RECOMPILE_RATE_ABOVE_PREREGISTERED"

    def test_recompile_limit_reached_is_a_breach(self):
        report = guards.evaluate_storm(
            guards.StormThresholds(), self._observation(recompile_limit_reached=True)
        )
        assert any(item["reason"] == "FRAMEWORK_RECOMPILE_LIMIT_REACHED" for item in report.breaches)

    def test_tail_amplification_is_reported(self):
        report = guards.evaluate_storm(
            guards.StormThresholds(), self._observation(observed_p95_ms=30.0)
        )
        assert any(item["reason"] == "TAIL_AMPLIFICATION_ABOVE_PREREGISTERED" for item in report.breaches)

    def test_thresholds_must_be_positive(self):
        with pytest.raises(ConfigError):
            guards.StormThresholds(max_p95_amplification=0.0)


@pytest.mark.unit
class TestGuardMinimality:
    def test_relaxation_without_evidence_is_flagged(self):
        guard = guards.GuardRecord(kind=guards.GuardKind.SHAPE, expression="S == 128", source="dynamo")
        review = guards.GuardReview(
            guard=guard,
            semantically_necessary=True,
            relaxed_and_verified=True,
            evidence="",
        )
        report = guards.audit_guard_minimality([review])
        assert not report["ok"]
        assert report["problems"][0]["reason"] == "RELAXED_WITHOUT_EVIDENCE"

    def test_relaxation_without_fallback_is_flagged(self):
        guard = guards.GuardRecord(kind=guards.GuardKind.SHAPE, expression="S == 128", source="dynamo")
        review = guards.GuardReview(
            guard=guard,
            semantically_necessary=True,
            relaxed_and_verified=True,
            evidence="bounds + fuzz",
            fallback_on_unsupported=False,
        )
        report = guards.audit_guard_minimality([review])
        assert report["problems"][0]["reason"] == "RELAXED_WITHOUT_FALLBACK"

    def test_classified_guards_pass(self):
        guard = guards.GuardRecord(kind=guards.GuardKind.SHAPE, expression="S == 128", source="dynamo")
        report = guards.audit_guard_minimality(
            [guards.GuardReview(guard=guard, semantically_necessary=True)]
        )
        assert report["ok"]

    def test_dynamic_correctness_points_cover_the_bounds(self):
        points = guards.dynamic_correctness_points((4, 64), example=16)
        assert points[0] == 4
        assert points[-1] == 65


@pytest.mark.unit
class TestCompileIdentity:
    def test_graph_identity_is_sensitive_to_every_factor(self):
        base = cache.graph_identity(
            "graph",
            op_schema_versions={"hqsb::rms_norm": "a"},
            tensor_metadata=("x:float16",),
            model_id="m",
            quant_policy_id="q",
            rewrite_spec_id="r",
        )
        variants = (
            cache.graph_identity("graph2", model_id="m", rewrite_spec_id="r"),
            cache.graph_identity("graph", model_id="m2", rewrite_spec_id="r"),
            cache.graph_identity("graph", model_id="m", quant_policy_id="q2", rewrite_spec_id="r"),
            cache.graph_identity("graph", model_id="m", rewrite_spec_id="r2"),
            cache.graph_identity(
                "graph",
                op_schema_versions={"hqsb::rms_norm": "b"},
                model_id="m",
                rewrite_spec_id="r",
            ),
        )
        for variant in variants:
            assert variant.digest != base.digest

    def test_compile_identity_adds_toolchain_factors(self):
        graph_id = cache.graph_identity("graph")
        first = cache.CompileIdentity(graph=graph_id, torch_version="2.5.0", target_arch="sm_86")
        second = cache.CompileIdentity(graph=graph_id, torch_version="2.6.0", target_arch="sm_86")
        third = cache.CompileIdentity(graph=graph_id, torch_version="2.5.0", target_arch="sm_87")
        assert first.digest != second.digest
        assert first.digest != third.digest


@pytest.mark.unit
class TestPhaseTimer:
    def test_phase_timer_records_known_phases(self):
        timer = cache.PhaseTimer()
        timer.record(cache.CompilePhase.DYNAMO_CAPTURE, 12.5)
        timer.record(cache.CompilePhase.CODEGEN, 3.0)
        payload = timer.as_dict()
        assert payload["total_ms"] == 15.5
        assert payload["phases_ms"][cache.CompilePhase.CODEGEN] == 3.0

    def test_unknown_phase_refused(self):
        with pytest.raises(ConfigError):
            cache.PhaseTimer().record("fast_magic", 1.0)

    def test_stop_without_start_refused(self):
        with pytest.raises(ConfigError):
            cache.PhaseTimer().stop(cache.CompilePhase.AUTOTUNE)


@pytest.mark.unit
class TestCacheEntryValidation:
    def _entry(self, root: str, **overrides) -> cache.CacheEntry:
        payload_path = os.path.join(root, "kernel.bin")
        with open(payload_path, "wb") as handle:
            handle.write(b"kernel-payload")
        payload = {
            "key_digest": "key",
            "layer": cache.CacheLayer.INDUCTOR_FX,
            "entry_hash": cache.sha256_file(payload_path),
            "payload_path": "kernel.bin",
            "size_bytes": os.path.getsize(payload_path),
            "target_arch": "sm_86",
            "abi_version": "1",
            "schema_hash": "schema",
        }
        payload.update(overrides)
        return cache.CacheEntry(**payload)

    def test_unknown_layer_refused(self):
        with pytest.raises(ConfigError):
            cache.CacheEntry(key_digest="k", layer="magic_cache")

    def test_clean_entry_validates(self, tmp_path):
        entry = self._entry(str(tmp_path))
        assert cache.validate_entry(entry, str(tmp_path), expected_arch="sm_86").ok

    def test_missing_payload_detected(self, tmp_path):
        entry = self._entry(str(tmp_path))
        os.remove(os.path.join(tmp_path, "kernel.bin"))
        result = cache.validate_entry(entry, str(tmp_path))
        assert not result.ok
        assert result.failure == cache.IntegrityFailure.MISSING_FILE

    def test_truncated_payload_detected(self, tmp_path):
        entry = self._entry(str(tmp_path))
        with open(os.path.join(tmp_path, "kernel.bin"), "wb") as handle:
            handle.write(b"x")
        result = cache.validate_entry(entry, str(tmp_path))
        assert not result.ok
        assert result.failure == cache.IntegrityFailure.PAYLOAD_TRUNCATED

    def test_bit_flip_detected(self, tmp_path):
        entry = self._entry(str(tmp_path))
        with open(os.path.join(tmp_path, "kernel.bin"), "r+b") as handle:
            handle.seek(0)
            handle.write(b"K")
        result = cache.validate_entry(entry, str(tmp_path))
        assert not result.ok
        assert result.failure == cache.IntegrityFailure.HASH_MISMATCH

    def test_foreign_arch_detected(self, tmp_path):
        entry = self._entry(str(tmp_path))
        result = cache.validate_entry(entry, str(tmp_path), expected_arch="sm_87")
        assert result.failure == cache.IntegrityFailure.FOREIGN_ARCH

    def test_stale_schema_detected(self, tmp_path):
        entry = self._entry(str(tmp_path))
        result = cache.validate_entry(entry, str(tmp_path), expected_schema_hash="other")
        assert result.failure == cache.IntegrityFailure.STALE_SCHEMA_ABI

    def test_incomplete_entry_detected(self, tmp_path):
        entry = self._entry(str(tmp_path), complete=False)
        result = cache.validate_entry(entry, str(tmp_path))
        assert result.failure == cache.IntegrityFailure.PARTIAL_WRITE

    def test_path_traversal_detected(self, tmp_path):
        entry = self._entry(str(tmp_path), payload_path="../../etc/passwd")
        result = cache.validate_entry(entry, str(tmp_path))
        assert result.failure == cache.IntegrityFailure.PATH_TRAVERSAL

    def test_untrusted_artifact_detected(self, tmp_path):
        entry = self._entry(str(tmp_path), trusted=False)
        result = cache.validate_entry(entry, str(tmp_path))
        assert result.failure == cache.IntegrityFailure.UNTRUSTED_ARTIFACT

    def test_unreadable_payload_detected(self, tmp_path):
        entry = self._entry(str(tmp_path))
        os.chmod(os.path.join(tmp_path, "kernel.bin"), 0o000)
        try:
            result = cache.validate_entry(entry, str(tmp_path))
        finally:
            os.chmod(os.path.join(tmp_path, "kernel.bin"), stat.S_IRUSR | stat.S_IWUSR)
        assert result.failure == cache.IntegrityFailure.BAD_PERMISSIONS

    def test_validation_is_always_pre_execution(self, tmp_path):
        entry = self._entry(str(tmp_path))
        assert cache.validate_entry(entry, str(tmp_path)).pre_execution is True


@pytest.mark.unit
class TestCorruptionFixture:
    def test_fixture_copies_before_mutating(self, tmp_path):
        golden = tmp_path / "golden"
        golden.mkdir()
        (golden / "kernel.bin").write_bytes(b"golden-payload")
        scratch = tmp_path / "scratch"
        fixture = cache.CorruptionFixture(source_dir=str(golden), scratch_root=str(scratch))
        work = fixture.prepare()
        fixture.bit_flip("kernel.bin", 0)
        assert (golden / "kernel.bin").read_bytes() == b"golden-payload"
        assert (tmp_path / "scratch" / "entry-under-test" / "kernel.bin").read_bytes() != b"golden-payload"
        assert work.endswith("entry-under-test")

    def test_fixture_refuses_to_mutate_outside_the_scratch_root(self, tmp_path):
        golden = tmp_path / "golden"
        golden.mkdir()
        (golden / "kernel.bin").write_bytes(b"payload")
        fixture = cache.CorruptionFixture(source_dir=str(golden), scratch_root=str(tmp_path / "scratch"))
        fixture.prepare()
        fixture.work_dir = str(golden)
        with pytest.raises(cache.RefusedMutationError):
            fixture.truncate("kernel.bin")

    def test_stale_lock_helper(self, tmp_path):
        golden = tmp_path / "golden"
        golden.mkdir()
        (golden / "kernel.bin").write_bytes(b"payload")
        fixture = cache.CorruptionFixture(source_dir=str(golden), scratch_root=str(tmp_path / "scratch"))
        fixture.prepare()
        path = fixture.stale_lock()
        assert os.path.isfile(path)


@pytest.mark.unit
class TestInvalidationMatrix:
    def test_default_matrix_covers_every_factor(self):
        matrix = cache.default_invalidation_matrix()
        assert len(matrix) >= 20
        assert any(not item.semantic for item in matrix)

    def test_expected_vs_actual_report_keeps_mismatches(self):
        factors = (
            cache.InvalidationFactor("dtype", cache.InvalidationAction.MISS),
            cache.InvalidationFactor("gpu_arch", cache.InvalidationAction.REJECT),
        )
        observations = (
            cache.InvalidationObservation("dtype", cache.InvalidationAction.MISS, cache.InvalidationAction.MISS),
            cache.InvalidationObservation("gpu_arch", cache.InvalidationAction.REJECT, cache.InvalidationAction.HIT),
        )
        report = cache.evaluate_invalidation(factors, observations)
        assert not report["ok"]
        assert report["mismatches"][0]["factor"] == "gpu_arch"

    def test_unobserved_factor_counts_as_not_observed(self):
        factors = (cache.InvalidationFactor("dtype", cache.InvalidationAction.MISS),)
        report = cache.evaluate_invalidation(factors, ())
        assert report["rows"][0]["actual_action"] == "<NOT_OBSERVED>"
        assert not report["ok"]

    def test_unknown_expected_action_refused(self):
        with pytest.raises(ConfigError):
            cache.InvalidationFactor("dtype", "MAYBE")


@pytest.mark.unit
class TestBreakEven:
    def test_break_even_is_ceiled(self):
        result = cache.compute_break_even(1000.0, 10.0, 8.0)
        assert result.requests == 500
        assert result.per_request_saving_ms == 2.0

    def test_no_amortisation_point_when_not_faster(self):
        result = cache.compute_break_even(1000.0, 8.0, 9.0)
        assert result.requests is None
        assert "DENOMINATOR_NON_POSITIVE" in result.reason

    def test_zero_extra_cost_is_reported(self):
        result = cache.compute_break_even(0.0, 10.0, 8.0)
        assert result.requests == 0
        assert result.reason == "NO_EXTRA_COMPILE_COST_RECORDED"


@pytest.mark.unit
class TestCacheManifestAndLock:
    def test_manifest_round_trip_and_atomic_write(self, tmp_path):
        manifest = cache.CacheManifest(state=cache.CacheState.COLD_EMPTY)
        manifest.add(
            cache.CacheEntry(
                key_digest="k", layer=cache.CacheLayer.DYNAMO_CODE, size_bytes=10
            )
        )
        path = cache.save_manifest(str(tmp_path / "manifest.json"), manifest)
        loaded = cache.load_manifest(path)
        assert loaded.entries[0].key_digest == "k"
        assert loaded.total_bytes == 10
        assert loaded.layers == (cache.CacheLayer.DYNAMO_CODE,)

    def test_unknown_state_refused(self):
        with pytest.raises(ConfigError):
            cache.CacheManifest(state="SOMEWHAT_WARM")

    def test_missing_manifest_is_an_artifact_error(self):
        with pytest.raises(Exception):
            cache.load_manifest("/nonexistent/manifest.json")

    def test_cache_lock_blocks_a_second_writer(self, tmp_path):
        lock = cache.CacheLock(str(tmp_path / "entry.lock"), stale_after_s=3600)
        with lock:
            with pytest.raises(cache.CacheLockError):
                cache.CacheLock(str(tmp_path / "entry.lock"), stale_after_s=3600).acquire()
        assert not os.path.exists(str(tmp_path / "entry.lock"))

    def test_stale_lock_is_replaced_deliberately(self, tmp_path):
        path = tmp_path / "entry.lock"
        path.write_text("stale", encoding="utf-8")
        os.utime(path, (0, 0))
        lock = cache.CacheLock(str(path), stale_after_s=1.0)
        lock.acquire()
        lock.release()

    def test_size_tracker_and_eviction(self):
        manifest = cache.CacheManifest(state=cache.CacheState.WARM_SAME_PROCESS)
        for index in range(5):
            manifest.add(
                cache.CacheEntry(
                    key_digest=f"k{index}",
                    layer=cache.CacheLayer.DYNAMO_CODE,
                    size_bytes=100,
                    created_ns=index,
                )
            )
        tracker = cache.CacheSizeTracker(cap_entries=3, cap_bytes=1000)
        assert tracker.evaluate(manifest)["over_cap"] is True
        evicted = cache.evict(manifest, 3)
        assert len(evicted) == 2
        assert len(manifest.entries) == 3
        assert manifest.entries[0].key_digest == "k2"
