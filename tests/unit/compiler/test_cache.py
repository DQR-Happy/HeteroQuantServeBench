"""Coverage for the E11-08 cache key/transaction/invalidation/safety interfaces.

Every test writes into ``tmp_path``: the corruption injections are applied to
isolated copies, never to a user cache.
"""

from __future__ import annotations

import json
import os

import pytest

from hqsb.core.errors import ConfigError
from hqsb.compiler import cache as ch


def _parts(spec: ch.CacheKeySpec, value: str = "v1") -> dict:
    return {name: f"{name}:{value}" for name in spec.fields}


def _store(tmp_path, spec: ch.CacheKeySpec | None = None) -> ch.EntryStore:
    spec = spec or ch.default_key_spec()
    return ch.EntryStore(str(tmp_path / "cache"), spec=spec)


def _publish(store: ch.EntryStore, *, entry_id: str = "e1", payload: bytes = b"payload") -> dict:
    spec = store.spec
    return store.publish(
        entry_id=entry_id,
        key=spec.compute(_parts(spec))["key"],
        layer="pass_ir",
        payload=payload,
        target_arch="sm_86",
        abi_version="1",
        guard_domain="B<=8",
    )


@pytest.mark.unit
class TestKeySpec:
    def test_default_spec_is_valid_and_vector_complete(self) -> None:
        spec = ch.default_key_spec()
        assert spec.validate() == []
        report = ch.evaluate_key_vectors(spec)
        assert report["all_ok"] is True

    def test_non_semantic_fields_are_forbidden_in_the_key(self) -> None:
        spec = ch.CacheKeySpec(
            spec_version="1.0.0", fields={"timestamp": "why", "semantic_graph": "graph"}
        )
        assert any("timestamp" in problem for problem in spec.validate())

    def test_missing_key_field_is_an_error_not_an_empty_string(self) -> None:
        spec = ch.default_key_spec()
        parts = _parts(spec)
        parts.pop("kernel_build_id")
        with pytest.raises(ConfigError):
            spec.compute(parts)

    def test_key_stability_to_noise(self) -> None:
        spec = ch.default_key_spec()
        report = ch.key_stability(spec, parts=_parts(spec))
        assert report["stable"] is True
        assert all(row["same_key"] for row in report["rows"])

    def test_key_changes_when_a_semantic_field_changes(self) -> None:
        spec = ch.default_key_spec()
        first = spec.compute(_parts(spec, "v1"))["key"]
        second = spec.compute(_parts(spec, "v2"))["key"]
        assert first != second


@pytest.mark.unit
class TestTransactions:
    def test_publish_commits_with_marker_last(self, tmp_path) -> None:
        store = _store(tmp_path)
        result = _publish(store)
        assert result["state"] == ch.STATE_COMMITTED
        assert result["stages"][-1] == "COMMITTED"
        assert os.path.isfile(os.path.join(store.entry_dir("e1"), ".committed"))

    def test_second_publisher_of_the_same_entry_is_rejected(self, tmp_path) -> None:
        store = _store(tmp_path)
        _publish(store)
        with pytest.raises(ConfigError):
            _publish(store)

    def test_writer_kill_leaves_no_committed_entry(self, tmp_path) -> None:
        store = _store(tmp_path)
        result = _publish(store)
        assert result["state"] == ch.STATE_COMMITTED
        killed_store = _store(tmp_path / "second")
        killed = killed_store.publish(
            entry_id="e2",
            key=killed_store.spec.compute(_parts(killed_store.spec))["key"],
            layer="pass_ir",
            payload=b"x",
            kill_at="before_marker",
        )
        assert killed["state"] == "KILLED"
        read = killed_store.read(
            "e2", expected_key=killed_store.spec.compute(_parts(killed_store.spec))["key"]
        )
        assert read.status == "reject"
        assert read.reason_code == "REJECT_NOT_COMMITTED"

    def test_cleanup_removes_temp_residue(self, tmp_path) -> None:
        store = _store(tmp_path)
        residue = os.path.join(store.root, ".tmp-residue")
        os.makedirs(residue)
        report = store.cleanup_temp()
        assert report["count"] == 1
        assert not os.path.isdir(residue)

    def test_unknown_layer_is_rejected(self, tmp_path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ConfigError):
            store.publish(entry_id="e", key="k", layer="magic", payload=b"")


@pytest.mark.unit
class TestSafeReader:
    def test_hit_when_everything_matches(self, tmp_path) -> None:
        store = _store(tmp_path)
        _publish(store)
        read = store.read(
            "e1",
            expected_key=store.spec.compute(_parts(store.spec))["key"],
            target_arch="sm_86",
            abi_version="1",
        )
        assert read.status == "hit"
        assert read.payload_bytes == b"payload"
        assert read.load_calls == 1
        assert all(step[1] for step in read.steps)

    def test_missing_entry_is_a_miss(self, tmp_path) -> None:
        store = _store(tmp_path)
        read = store.read("nope", expected_key="k")
        assert read.status == "miss" and read.reason_code == "MISS_NOT_FOUND"

    def test_key_mismatch_and_target_mismatch_are_rejects(self, tmp_path) -> None:
        store = _store(tmp_path)
        _publish(store)
        key_mismatch = store.read("e1", expected_key="other")
        assert key_mismatch.reason_code == "REJECT_KEY_MISMATCH"
        arch_mismatch = store.read(
            "e1", expected_key=store.spec.compute(_parts(store.spec))["key"], target_arch="sm_90"
        )
        assert arch_mismatch.reason_code == "REJECT_TARGET_INCOMPATIBLE"
        abi_mismatch = store.read(
            "e1", expected_key=store.spec.compute(_parts(store.spec))["key"], abi_version="2"
        )
        assert abi_mismatch.reason_code == "REJECT_ABI_MISMATCH"

    def test_guard_false_is_not_a_hit(self, tmp_path) -> None:
        store = _store(tmp_path)
        _publish(store)
        read = store.read(
            "e1",
            expected_key=store.spec.compute(_parts(store.spec))["key"],
            target_arch="sm_86",
            abi_version="1",
            guard_covers=False,
        )
        assert read.status == "reject"
        assert read.reason_code == "REJECT_GUARD_FALSE"
        assert read.load_calls == 0

    def test_unknown_schema_is_rejected(self, tmp_path) -> None:
        store = _store(tmp_path)
        _publish(store)
        read = store.read(
            "e1",
            expected_key=store.spec.compute(_parts(store.spec))["key"],
            schema_supported=lambda version: False,
        )
        assert read.reason_code == "REJECT_UNKNOWN_SCHEMA"

    def test_corrupted_entries_are_rejected_before_load(self, tmp_path) -> None:
        expectations = {
            "payload_bit_flip": "REJECT_PAYLOAD_HASH",
            "payload_truncation": "REJECT_PAYLOAD_HASH",
            "metadata_truncation": "REJECT_METADATA_PARSE",
            "metadata_field_edit": "REJECT_METADATA_HASH",
            "unknown_schema": "REJECT_UNKNOWN_SCHEMA",
            "missing_commit_marker": "REJECT_NOT_COMMITTED",
            "writer_killed": "REJECT_NOT_COMMITTED",
            "metadata_payload_swap": "REJECT_PAYLOAD_HASH",
        }
        for case, reason in expectations.items():
            store = _store(tmp_path / case)
            _publish(store)
            if case == "metadata_payload_swap":
                # a swap needs a second, legitimate entry to swap with
                _publish(store, entry_id="e2", payload=b"second-payload")
            ch.inject_corruption(store, "e1", case)
            read = store.read(
                "e1",
                expected_key=store.spec.compute(_parts(store.spec))["key"],
                target_arch="sm_86",
                abi_version="1",
            )
            assert read.status == "reject", case
            assert read.reason_code == reason, (case, read.reason_code)
            assert read.load_calls == 0, case

    def test_metadata_tamper_with_arch_change_is_caught_by_compatibility(self, tmp_path) -> None:
        store = _store(tmp_path / "arch_tamper")
        _publish(store)
        manifest_path = store.manifest_path("e1")
        with open(manifest_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["target_arch"] = "sm_999"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        read = store.read(
            "e1",
            expected_key=store.spec.compute(_parts(store.spec))["key"],
            target_arch="sm_86",
            abi_version="1",
        )
        assert read.status == "reject"
        assert read.reason_code == "REJECT_TARGET_INCOMPATIBLE"

    def test_quarantine_moves_the_entry(self, tmp_path) -> None:
        store = _store(tmp_path)
        _publish(store)
        ch.inject_corruption(store, "e1", "payload_bit_flip")
        read = store.read("e1", expected_key=store.spec.compute(_parts(store.spec))["key"])
        assert read.reason_code == "REJECT_PAYLOAD_HASH"
        assert store.entry_state("e1") == ch.STATE_QUARANTINED

    def test_unknown_corruption_case_is_rejected(self, tmp_path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ConfigError):
            ch.inject_corruption(store, "e1", "magic")


@pytest.mark.unit
class TestInvalidation:
    def test_matrix_table_matches_the_code_matrix(self) -> None:
        assert [row["change"] for row in ch.matrix_table()] == [
            row[0] for row in ch.INVALIDATION_MATRIX
        ]

    def test_noise_changes_must_hit(self) -> None:
        report = ch.evaluate_invalidation(
            change="timestamp_or_pid", expected="HIT", rewrote_key=False, recompile_occurred=False,
            actual_binary_same=True,
        )
        assert report["ok"] is True
        bad = ch.evaluate_invalidation(
            change="timestamp_or_pid", expected="HIT", rewrote_key=True, recompile_occurred=True,
            actual_binary_same=True,
        )
        assert bad["ok"] is False

    def test_semantic_changes_must_miss(self) -> None:
        report = ch.evaluate_invalidation(
            change="compiler_or_codegen_flags",
            expected="MISS",
            rewrote_key=True,
            recompile_occurred=True,
            actual_binary_same=False,
        )
        assert report["ok"] is True
        stale = ch.evaluate_invalidation(
            change="compiler_or_codegen_flags",
            expected="MISS",
            rewrote_key=False,
            recompile_occurred=False,
            actual_binary_same=True,
        )
        assert stale["ok"] is False

    def test_unregistered_change_is_reported(self) -> None:
        report = ch.evaluate_invalidation(
            change="phase_of_moon", expected="HIT", rewrote_key=False, recompile_occurred=False,
            actual_binary_same=True,
        )
        assert report["known"] is False


@pytest.mark.unit
class TestConcurrencyAndOmission:
    def test_partial_reads_fail_the_evaluation(self) -> None:
        report = ch.evaluate_concurrency(
            publishes=[{"entry_id": "e", "state": "COMMITTED", "compile_occurred": True}],
            reads=[{"status": "hit", "steps_committed": False}],
        )
        assert report["ok"] is False
        assert report["partial_reads"]

    def test_reader_writer_plan_documents_the_contract(self) -> None:
        plan = ch.reader_writer_plan(writers=2, readers=3)
        assert "never observe a temp directory" in plan["expected"]

    def test_key_omission_detector_requires_the_field_in_the_key(self) -> None:
        spec = ch.default_key_spec()
        ok = ch.key_omission_detector(
            spec=spec, base_parts=_parts(spec), hidden_config_field="compiler_flags"
        )
        assert ok["ok"] is True
        missing = ch.key_omission_detector(
            spec=spec, base_parts=_parts(spec), hidden_config_field="hidden_debug_flag"
        )
        assert missing["ok"] is False
        assert "add the field" in missing["action"]


@pytest.mark.unit
class TestEvictionAndPolicy:
    def test_eviction_never_touches_in_use_entries(self) -> None:
        policy = ch.EvictionPolicy(max_bytes=100, max_entries=1)
        entries = [
            {"entry_id": "in_use", "bytes": 80, "in_use": True, "state": "COMMITTED", "last_used_index": 0},
            {"entry_id": "old", "bytes": 80, "in_use": False, "state": "COMMITTED", "last_used_index": 1},
        ]
        report = policy.plan(entries)
        assert "in_use" not in report["evicted"]
        assert report["ok"] is True

    def test_eviction_policy_validation(self) -> None:
        assert any("strategy" in problem for problem in ch.EvictionPolicy(1, 1, "magic").validate())
        assert any("positive" in problem for problem in ch.EvictionPolicy(0, 0).validate())

    def test_layer_dependency_map_covers_all_layers(self) -> None:
        mapping = ch.layer_dependency_map()
        assert set(mapping) >= set(ch.CACHE_LAYERS)
        assert mapping["binary_package"] == []

    def test_policy_document_is_digestible(self) -> None:
        document = ch.cache_policy_document(
            spec=ch.default_key_spec(),
            eviction=ch.EvictionPolicy(1 << 20, 100),
            compat_policy={"driver_compat_class": "minor tolerated if verified"},
            quarantine_ttl_s=3600.0,
        )
        assert ch.cache_policy_digest(document) == ch.cache_policy_digest(document)
        assert document["defaults"]["corruption"] == "quarantine, never execute"


@pytest.mark.unit
class TestTelemetryAndMetrics:
    def test_hit_with_compile_is_rejected(self) -> None:
        event = ch.CacheEvent(layer="pass_ir", kind="hit", compile_occurred=True)
        assert any("not a hit" in problem for problem in event.validate())

    def test_unknown_event_kind_and_layer_are_rejected(self) -> None:
        event = ch.CacheEvent(layer="magic", kind="magic")
        problems = event.validate()
        assert any("layer" in problem for problem in problems)
        assert any("kind" in problem for problem in problems)

    def test_metrics_prioritise_safety(self) -> None:
        safe = ch.cache_metrics(
            eligible_lookups=10,
            compatible_guarded_hits=8,
            false_hits=0,
            false_misses=1,
            corrupt_executions=0,
            cross_process_recompiles=0,
            verification_overhead_s=0.1,
            cold_cost_saved_s=2.0,
        )
        assert safe["safe"] is True
        unsafe = ch.cache_metrics(
            eligible_lookups=10,
            compatible_guarded_hits=9,
            false_hits=1,
            false_misses=0,
            corrupt_executions=0,
            cross_process_recompiles=0,
            verification_overhead_s=0.1,
            cold_cost_saved_s=2.0,
        )
        assert unsafe["safe"] is False

    def test_timing_boundaries_record_the_reset_scope(self) -> None:
        report = ch.timing_boundaries(
            c0_cold_total_s=5.0,
            c1_memory_hit_s=0.01,
            c2_disk_hit_s=0.2,
            cache_reset_scope=["hqsb_project"],
            os_page_cache_cleared=False,
        )
        assert report["monotone_expected"] is True
        assert report["os_page_cache_cleared"] is False
        assert "project-cold" in report["note"]

    def test_telemetry_summary_groups_by_layer_and_kind(self) -> None:
        telemetry = ch.CacheTelemetry()
        telemetry.record(ch.CacheEvent(layer="pass_ir", kind="lookup"))
        telemetry.record(ch.CacheEvent(layer="pass_ir", kind="miss"))
        summary = telemetry.summary()
        assert summary["by_layer"] == {"pass_ir": 2}
        assert summary["by_kind"] == {"lookup": 1, "miss": 1}


@pytest.mark.unit
class TestEntryManifest:
    def test_manifest_validation(self) -> None:
        manifest = ch.EntryManifest(entry_id="e", key="k", layer="magic", state="MAGIC")
        problems = manifest.validate()
        assert any("layer" in problem for problem in problems)
        assert any("state" in problem for problem in problems)

    def test_committed_entry_requires_payload_hash(self) -> None:
        manifest = ch.EntryManifest(entry_id="e", key="k", layer="pass_ir", state="COMMITTED")
        assert any("payload hash" in problem for problem in manifest.validate())

    def test_manifest_round_trips_through_disk(self, tmp_path) -> None:
        store = _store(tmp_path)
        _publish(store)
        with open(store.manifest_path("e1"), encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["state"] == ch.STATE_VALIDATING or payload["state"] == ch.STATE_COMMITTED
        assert payload["key"] == store.spec.compute(_parts(store.spec))["key"]
