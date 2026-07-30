"""S07 experiment scaffolding: prerequisites, verdicts, configs, interface map."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from hqsb.core.errors import ConfigError
from hqsb.runtime import experiment as E
from hqsb.runtime import interface_map as IM
from hqsb.runtime import specs, telemetry

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs", "runtime")


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _complete_root(root: str) -> None:
    _write_json(
        os.path.join(root, E.S04_5_EVIDENCE_MARKER),
        {"stage": "S04.5", "overall": "PASS"},
    )
    for stage in ("S05", "S06"):
        _write_json(
            os.path.join(root, "docs", "stage_experiments", stage, "E00-01", "verdict.json"),
            {"stage": stage, "status": "PASS"},
        )
    _write_json(
        os.path.join(root, E.REQUEST_FIXTURE_CANDIDATES[0]), {"fixture": "frozen"}
    )
    _write_json(
        os.path.join(root, E.S07_EVIDENCE_DIR, E.CAPABILITY_PROBE_NAME),
        {
            "probe_complete": True,
            "verified_usable_engines": ["vllm"],
        },
    )
    _write_json(
        os.path.join(root, E.S07_EVIDENCE_DIR, E.MAIN_RUNTIME_SELECTION_NAME),
        {"selected_engine": "vllm", "selected_verified": True},
    )
    _write_json(
        os.path.join(root, "docs", "stage_experiments", "S07", "run-1", "environment_fingerprint.json"),
        {"python": "3.12"},
    )


@pytest.mark.unit
class TestPrerequisites:
    def test_current_repository_is_blocked(self):
        status = E.check_prerequisites(_REPO_ROOT)
        assert not status.satisfied
        assert "s04_5_model_reintegration_evidence" in status.missing
        assert "main_runtime_selection" in status.missing

    def test_negative_probe_does_not_create_a_verified_selection(self):
        """A completed negative probe is evidence, but it cannot select an engine."""
        status = E.check_prerequisites(_REPO_ROOT)
        checks = {check.name: check for check in status.checks}
        assert checks["runtime_capability_probe"].satisfied
        assert not checks["main_runtime_selection"].satisfied
        assert "no verified main-runtime selection" in checks["main_runtime_selection"].reason

    def test_placeholder_probe_and_selection_do_not_unlock_execution(self):
        with tempfile.TemporaryDirectory(prefix="hqsb-s07-placeholder-") as root:
            _write_json(
                os.path.join(root, E.S07_EVIDENCE_DIR, E.CAPABILITY_PROBE_NAME),
                {"probe": "done"},
            )
            _write_json(
                os.path.join(root, E.S07_EVIDENCE_DIR, E.MAIN_RUNTIME_SELECTION_NAME),
                {"selected": "vllm"},
            )
            checks = {check.name: check for check in E.check_prerequisites(root).checks}
            assert not checks["runtime_capability_probe"].satisfied
            assert not checks["main_runtime_selection"].satisfied

    def test_c6_c7_contract_is_available(self):
        checks = {check.name: check for check in E.check_prerequisites(_REPO_ROOT).checks}
        assert checks["c6_c7_schema_available"].satisfied

    def test_scaffolding_written_fingerprint_does_not_satisfy_the_gate(self):
        """A fingerprint this scaffolding writes must not unlock its own experiments."""
        with tempfile.TemporaryDirectory(prefix="hqsb-s07-fp-") as root:
            run = E.RunDirectory(root, "E07-01", "self")
            run.create()
            run.write_json("environment_fingerprint.json", E.environment_fingerprint())
            checks = {check.name: check for check in E.check_prerequisites(root).checks}
            assert not checks["frozen_environment_fingerprint"].satisfied

    def test_a_complete_evidence_chain_satisfies_every_check(self):
        with tempfile.TemporaryDirectory(prefix="hqsb-s07-prereq-") as root:
            _complete_root(root)
            status = E.check_prerequisites(root)
            assert status.satisfied, status.missing

    def test_missing_s06_verdict_keeps_the_chain_blocked(self):
        with tempfile.TemporaryDirectory(prefix="hqsb-s07-prereq-") as root:
            _complete_root(root)
            os.remove(
                os.path.join(
                    root, "docs", "stage_experiments", "S06", "E00-01", "verdict.json"
                )
            )
            assert "s06_stable_capability_evidence" in E.check_prerequisites(root).missing


@pytest.mark.unit
class TestPreregistration:
    def _kwargs(self, **overrides):
        payload = {
            "experiment_id": "E07-01",
            "question": "do the backends agree?",
            "hypothesis": "they agree on greedy tokens",
            "claim_boundary": "may not be used as a performance claim",
        }
        payload.update(overrides)
        return payload

    def test_preregistration_requires_question_and_hypothesis(self):
        with pytest.raises(ConfigError):
            E.Preregistration(**self._kwargs(question=""))

    def test_claim_boundary_is_mandatory(self):
        with pytest.raises(ConfigError):
            E.Preregistration(**self._kwargs(claim_boundary=""))

    def test_at_least_three_independent_processes(self):
        with pytest.raises(ConfigError):
            E.Preregistration(**self._kwargs(independent_processes=1))

    def test_unknown_experiment_refused(self):
        with pytest.raises(ConfigError):
            E.Preregistration(**self._kwargs(experiment_id="E07-11"))

    def test_hash_is_stable_across_identical_preregistrations(self):
        assert (
            E.Preregistration(**self._kwargs()).prereg_hash
            == E.Preregistration(**self._kwargs()).prereg_hash
        )


@pytest.mark.unit
class TestRunDirectory:
    def test_layout_matches_the_protocol(self):
        expected = {
            "requests",
            "iterations",
            "scheduler",
            "kv",
            "prefix",
            "graph",
            "kernels",
            "correctness",
            "performance",
            "memory",
            "energy",
            "errors",
            "profiler",
        }
        assert expected <= set(E.RUN_LAYOUT)

    def test_conclusion_without_execution_is_downgraded_to_blocked(self):
        with tempfile.TemporaryDirectory(prefix="hqsb-s07-run-") as root:
            run = E.RunDirectory(root, "E07-01", "run-1")
            run.create()
            prerequisites = E.check_prerequisites(_REPO_ROOT)
            payload = run.write_verdict(
                status=E.STATUS_PASS,
                reason="looks good",
                prerequisites=prerequisites,
                executed=True,
                raw_samples=10,
            )
            assert payload["status"] == E.STATUS_BLOCKED
            assert "execution is disabled" in payload["reason"]
            assert payload["requested_status"] == E.STATUS_PASS

    def test_conclusion_is_blocked_by_unsatisfied_prerequisites(self):
        with tempfile.TemporaryDirectory(prefix="hqsb-s07-run-") as root:
            run = E.RunDirectory(root, "E07-01", "run-2")
            run.create()
            payload = run.write_verdict(
                status=E.STATUS_PASS_NEGATIVE,
                reason="no benefit",
                prerequisites=E.check_prerequisites(_REPO_ROOT),
                executed=True,
                allow_execute=True,
                raw_samples=5,
            )
            assert payload["status"] == E.STATUS_BLOCKED
            assert "prerequisites unsatisfied" in payload["reason"]

    def test_conclusion_without_raw_samples_is_blocked(self):
        with tempfile.TemporaryDirectory(prefix="hqsb-s07-run-") as root:
            _complete_root(root)
            run = E.RunDirectory(root, "E07-01", "run-3")
            run.create()
            payload = run.write_verdict(
                status=E.STATUS_PASS,
                reason="",
                prerequisites=E.check_prerequisites(root),
                executed=True,
                allow_execute=True,
                raw_samples=0,
            )
            assert payload["status"] == E.STATUS_BLOCKED
            assert "no raw samples" in payload["reason"]

    def test_conclusion_with_full_evidence_is_written(self):
        with tempfile.TemporaryDirectory(prefix="hqsb-s07-run-") as root:
            _complete_root(root)
            run = E.RunDirectory(root, "E07-01", "run-4")
            run.create()
            payload = run.write_verdict(
                status=E.STATUS_PASS,
                reason="measured",
                prerequisites=E.check_prerequisites(root),
                executed=True,
                allow_execute=True,
                raw_samples=12,
            )
            assert payload["status"] == E.STATUS_PASS

    def test_status_writer_refuses_conclusion_statuses(self):
        with tempfile.TemporaryDirectory(prefix="hqsb-s07-run-") as root:
            run = E.RunDirectory(root, "E07-02", "run-5")
            run.create()
            with pytest.raises(ConfigError):
                run.write_status(
                    E.STATUS_PASS, "nope", E.check_prerequisites(_REPO_ROOT)
                )

    def test_report_skeleton_is_not_run_and_has_no_numbers(self):
        with tempfile.TemporaryDirectory(prefix="hqsb-s07-run-") as root:
            run = E.RunDirectory(root, "E07-03", "run-6")
            run.create()
            path = run.write_report_skeleton("E07-03", "KV capacity")
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            assert "NOT_RUN" in text
            assert "无。实验未执行" in text

    def test_interface_only_run_records_blocked_status(self):
        with tempfile.TemporaryDirectory(prefix="hqsb-s07-run-") as root:
            record = E.interface_only_run(root, "E07-04", run_id="iface")
            with open(
                os.path.join(record["run_dir"], "status.json"), encoding="utf-8"
            ) as handle:
                payload = json.load(handle)
            assert payload["status"] in (E.STATUS_BLOCKED, E.STATUS_NOT_STARTED)
            assert os.path.isdir(os.path.join(record["run_dir"], "kv"))


@pytest.mark.unit
class TestEvidenceManifest:
    def test_manifest_round_trips_the_s07_block(self):
        manifest = E.EvidenceManifest(
            run_id="run-1",
            experiment_id="E07-01",
            backend_id="vllm",
            runtime_version="0.6.3",
            runtime_commit="deadbeef",
            requested_capability={"prefix_cache": "SUPPORTED_EXACT"},
            actual_capability={"prefix_cache": "SUPPORTED_EXACT"},
            request_trace_hash="trace",
            iteration_ledger_uri="artifact://iterations",
        )
        payload = json.loads(manifest.to_json())
        assert payload["runtime"]["backend_id"] == "vllm"
        assert payload["runtime"]["iteration_ledger_uri"] == "artifact://iterations"
        assert payload["runtime"]["capability"]["requested"] == {
            "prefix_cache": "SUPPORTED_EXACT"
        }


@pytest.mark.unit
class TestConfigs:
    def test_every_document_loads_and_audits(self):
        loaded = specs.RuntimeSpecs.load(_CONFIG_DIR)
        assert loaded.ok, [audit for audit in loaded.audits if not audit["ok"]]
        assert set(loaded.documents) == set(specs.KINDS)

    def test_unknown_key_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="hqsb-s07-cfg-") as directory:
            path = os.path.join(directory, "broken.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("kind: hqsb.runtime.kv_spec\nblock_sizes: [16]\ntypo_key: 1\n")
            with pytest.raises(ConfigError):
                specs.load_yaml_document(path)

    def test_duplicate_kind_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="hqsb-s07-cfg-") as directory:
            for name in ("a.yaml", "b.yaml"):
                with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
                    handle.write("kind: hqsb.runtime.kv_spec\nblock_sizes: [16]\n")
            with pytest.raises(ConfigError):
                specs.RuntimeSpecs.load(directory)

    def test_scheduler_spec_builds_an_executable_object(self):
        loaded = specs.RuntimeSpecs.load(_CONFIG_DIR)
        spec = loaded.scheduler_spec()
        assert spec.mode in ("static", "continuous", "chunked_prefill")
        assert spec.max_batched_tokens > 0

    def test_prefix_spec_covers_every_identity_field(self):
        loaded = specs.RuntimeSpecs.load(_CONFIG_DIR)
        declared = set(
            loaded.documents[specs.KIND_PREFIX_SPEC].payload["key_fields"]
        )
        from hqsb.runtime import prefix_cache

        assert set(prefix_cache.IDENTITY_FIELDS) <= declared
        assert set(prefix_cache.SPAN_KEY_FIELDS) <= declared

    def test_graph_spec_buckets_match_the_file(self):
        loaded = specs.RuntimeSpecs.load(_CONFIG_DIR)
        graph_spec = loaded.graph_spec()
        declared = [
            item["name"]
            for item in loaded.documents[specs.KIND_GRAPH_ATTENTION_SPEC].payload["buckets"]
        ]
        assert [bucket.name for bucket in graph_spec.buckets] == declared

    def test_spec_decode_defaults_to_no_claim(self):
        loaded = specs.RuntimeSpecs.load(_CONFIG_DIR)
        contract = loaded.spec_decode()
        assert contract["claim"] is False
        assert 1 in contract["gamma"]["gammas"]

    def test_comparison_identity_is_bound_at_run_time(self):
        loaded = specs.RuntimeSpecs.load(_CONFIG_DIR)
        method = loaded.comparison_method()
        assert method["independent_processes"] >= 3
        spec = loaded.comparison_spec(
            model_id="Qwen/Qwen3-1.7B",
            model_manifest_sha256="a" * 64,
            precision="float16",
            hardware="host-a",
            request_trace_hash="trace",
        )
        assert spec.hardware == "host-a"
        with pytest.raises(ConfigError):
            loaded.comparison_spec()

    def test_kv_vocabulary_matches_the_code(self):
        loaded = specs.RuntimeSpecs.load(_CONFIG_DIR)
        audit = specs.audit_kv_spec(loaded.documents[specs.KIND_KV_SPEC])
        assert audit["ok"], audit["problems"]


@pytest.mark.unit
class TestInterfaceMap:
    def test_map_has_twenty_steps_per_experiment(self):
        for mapping in IM.EXPERIMENT_MAP:
            assert len(mapping.steps) == 20, mapping.experiment_id
            assert [step.step for step in mapping.steps] == list(range(1, 21))

    def test_every_symbol_resolves(self):
        resolved = IM.resolve_interfaces()
        assert resolved["ok"], resolved["failures"]
        assert resolved["steps"] == 200
        assert resolved["interfaces"] >= 200

    def test_missing_symbol_is_detected(self):
        with pytest.raises(AttributeError):
            IM.resolve_interface("hqsb.runtime.kv.definitely_not_there")

    def test_dataclass_fields_are_addressable(self):
        assert IM.resolve_interface("hqsb.runtime.comparison.ComparisonRow.metrics")
        assert IM.resolve_interface("hqsb.runtime.kv.BlockPool.events")

    def test_markdown_table_lists_every_experiment(self):
        table = IM.mapping_table_markdown()
        for mapping in IM.EXPERIMENT_MAP:
            assert mapping.experiment_id in table


@pytest.mark.unit
class TestExperimentRecord:
    """The execution handbook §4 unified record."""

    def _record(self, **overrides) -> E.ExperimentRecord:
        payload = {
            "experiment_id": "E07-01",
            "question": "do the backends agree?",
            "hypothesis": "greedy tokens match",
        }
        payload.update(overrides)
        return E.ExperimentRecord(**payload)

    def test_field_order_matches_the_handbook(self):
        assert E.EXPERIMENT_RECORD_FIELDS[:6] == (
            "stage",
            "experiment_id",
            "status",
            "question",
            "hypothesis",
            "run_id",
        )
        assert "requested_implementation" in E.EXPERIMENT_RECORD_FIELDS
        assert "performance_samples_uri" in E.EXPERIMENT_RECORD_FIELDS

    def test_question_and_hypothesis_are_mandatory(self):
        with pytest.raises(ConfigError):
            self._record(hypothesis="")
        with pytest.raises(ConfigError):
            self._record(experiment_id="E07-11")

    def test_unknown_status_refused(self):
        with pytest.raises(ConfigError):
            self._record(status="OK")

    def test_implementation_change_needs_a_fallback_reason(self):
        with pytest.raises(ConfigError):
            self._record(
                requested_implementation="hqsb_kernel",
                actual_implementation="runtime_default",
            )
        record = self._record(
            requested_implementation="hqsb_kernel",
            actual_implementation="runtime_default",
            fallback_reason="shape unsupported",
        )
        assert record.fallback_reason == "shape unsupported"

    def test_conclusion_needs_a_decision(self):
        with pytest.raises(ConfigError):
            self._record(status=E.STATUS_PASS, performance_samples_uri="artifact://x")

    def test_conclusion_needs_raw_evidence(self):
        with pytest.raises(ConfigError):
            self._record(status=E.STATUS_PASS_NEGATIVE, decision="no benefit")
        record = self._record(
            status=E.STATUS_PASS_NEGATIVE,
            decision="no benefit",
            performance_samples_uri="artifact://requests",
        )
        assert record.executed is False
        assert record.as_dict()["decision"] == "no benefit"

    def test_template_can_never_carry_a_conclusion(self):
        for status in (E.STATUS_PASS, E.STATUS_FAIL, E.STATUS_PASS_NEGATIVE):
            with pytest.raises(ConfigError):
                E.template_experiment_record("E07-01", status=status, reason="r")
        record = E.template_experiment_record(
            "E07-01", status=E.STATUS_BLOCKED, reason="prerequisites unmet"
        )
        assert record.status == E.STATUS_BLOCKED
        assert record.decision == ""
        assert record.limitations

    def test_record_writer_round_trips(self):
        with tempfile.TemporaryDirectory(prefix="hqsb-s07-record-") as root:
            run = E.RunDirectory(root, "E07-01", "run-1")
            run.create()
            path = run.write_experiment_record(
                E.template_experiment_record(
                    "E07-01", status=E.STATUS_NOT_STARTED, reason="template"
                )
            )
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            assert payload["experiment_id"] == "E07-01"
            assert payload["schema_version"] == "1.0.0"
            assert payload["status"] == E.STATUS_NOT_STARTED


@pytest.mark.unit
class TestTelemetryProjection:
    def test_c6_c7_coverage_is_complete(self):
        summary = telemetry.c6_c7_summary()
        assert summary["c6"]["ok"], summary["c6"]["missing"]
        assert summary["c7"]["ok"], summary["c7"]["invalid"]

    def test_c2_alignment_covers_every_workload_field(self):
        report = telemetry.c2_alignment()
        assert report["ok"], report["missing"]
        fields = {row["field"] for row in report["rows"]}
        assert {"input_tokens", "output_tokens", "sampling", "stop_condition"} <= fields
        assert all(row["carrier"] for row in report["rows"])

    def test_span_chain_matches_the_protocol(self):
        assert telemetry.SPAN_CHAIN[0] == "request"
        assert telemetry.SPAN_CHAIN[-1] == "cleanup"
        assert "kv_lookup_allocate_free" in telemetry.SPAN_CHAIN

    def test_silent_capability_change_is_refused(self):
        with pytest.raises(ConfigError):
            telemetry.S07ResultFields(
                requested_capability={"prefix_cache": "SUPPORTED_EXACT"},
                actual_capability={"prefix_cache": "UNSUPPORTED_REJECT"},
            )

    def test_capability_change_with_a_reason_is_accepted(self):
        fields = telemetry.S07ResultFields(
            requested_capability={"prefix_cache": "SUPPORTED_EXACT"},
            actual_capability={"prefix_cache": "UNSUPPORTED_REJECT"},
            capability_reasons={"prefix_cache": "no KV store"},
        )
        assert fields.as_dict()["capability_reasons"]["prefix_cache"] == "no KV store"

    def test_missing_actual_value_is_refused(self):
        with pytest.raises(ConfigError):
            telemetry.S07ResultFields(requested_capability={"prefix_cache": "SUPPORTED_EXACT"})

    def test_trace_collector_requires_a_request_id(self):
        collector = telemetry.TraceCollector(run_id="run")
        with pytest.raises(ConfigError):
            collector.emit("model_runner")

    def test_projection_validates_against_the_frozen_c6(self):
        result = telemetry.project_c6("run-1", telemetry.S07ResultFields(backend_id="vllm"))
        payload = result.model_dump()
        assert payload["summary"]["s07"]["backend"]["id"] == "vllm"

    def test_chain_coverage_reports_missing_spans(self):
        collector = telemetry.TraceCollector(run_id="run")
        collector.emit("request", request_id="r0")
        report = telemetry.chain_coverage(collector.records)
        assert not report["ok"]
        assert "cleanup" in report["missing"]
