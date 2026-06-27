"""S10 experiment scaffolding: gates, verdict refusal, interface map, configs."""

from __future__ import annotations

import json
import os

import pytest

from hqsb.core.errors import ConfigError
from hqsb.distributed import experiment as exp
from hqsb.distributed import interface_map as imap
from hqsb.distributed import specs

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@pytest.mark.unit
class TestPrerequisites:
    def test_all_experiments_are_known(self):
        assert set(exp.EXPERIMENTS) == {f"E10-{index:02d}" for index in range(1, 11)}

    def test_prerequisites_are_currently_blocked(self):
        status = exp.check_prerequisites(_REPO_ROOT)
        assert not status.satisfied
        assert "s07_p0_verdicts" in status.missing
        assert "two_accelerators_available" in status.missing
        assert "frozen_topology_manifest" in status.missing
        assert "frozen_collective_backend" in status.missing

    def test_prerequisites_do_not_self_unlock(self):
        # The evidence paths live under docs/stage_experiments/S10 — never under
        # experiment_results — so this interface layer cannot satisfy its own gate.
        assert exp.S10_EVIDENCE_DIR.startswith(os.path.join("docs", "stage_experiments"))
        assert exp.S10_EVIDENCE_DIR != os.path.join("experiment_results", "S10")
        fingerprints = []
        for root, _dirs, files in os.walk(os.path.join(_REPO_ROOT, "experiment_results")):
            fingerprints.extend(
                name for name in files if name == "environment_fingerprint.json"
            )
        assert all(
            not os.path.join("docs", "stage_experiments") in path for path in fingerprints
        )

    def test_minimum_accelerators_is_two(self):
        assert exp.MINIMUM_ACCELERATORS == 2


@pytest.mark.unit
class TestPreregistration:
    def test_preregistration_needs_hypothesis_claim_boundary_and_non_claims(self):
        with pytest.raises(ConfigError):
            exp.Preregistration(
                experiment_id="E10-05", question="q", hypothesis="", world_size=2,
                claim_boundary="x", non_claims=("y",),
            )
        with pytest.raises(ConfigError):
            exp.Preregistration(
                experiment_id="E10-05", question="q", hypothesis="h", world_size=2,
                claim_boundary="", non_claims=("y",),
            )
        record = exp.Preregistration(
            experiment_id="E10-05", question="q", hypothesis="h", world_size=2,
            claim_boundary="no 4-card claim", non_claims=("no multi-node claim",),
        )
        assert record.prereg_hash

    def test_unknown_experiment_is_refused(self):
        with pytest.raises(ConfigError):
            exp.Preregistration(
                experiment_id="E09-01", question="q", hypothesis="h", world_size=2,
                claim_boundary="x", non_claims=("y",),
            )


@pytest.mark.unit
class TestVerdictRefusal:
    def test_template_record_cannot_carry_a_conclusion(self):
        with pytest.raises(ConfigError):
            exp.template_experiment_record("E10-01", status="PASS", reason="?")

    def test_verdict_refuses_conclusion_without_execute(self, tmp_path):
        status = exp.PrerequisiteStatus(stage="S10", checks=[])
        run = exp.RunDirectory(str(tmp_path), "E10-01", "r0")
        run.create()
        verdict = run.write_verdict(
            status="PASS", reason="should be refused", prerequisites=status,
            executed=True, raw_samples=10, allow_execute=False,
        )
        assert verdict["status"] == exp.STATUS_BLOCKED
        assert "execution is disabled" in verdict["reason"]

    def test_verdict_refuses_conclusion_without_prerequisites(self, tmp_path):
        status = exp.PrerequisiteStatus(
            stage="S10",
            checks=[exp.PrerequisiteCheck(name="two_accelerators_available", satisfied=False)],
        )
        run = exp.RunDirectory(str(tmp_path), "E10-01", "r0")
        run.create()
        verdict = run.write_verdict(
            status="PASS", reason="should be refused", prerequisites=status,
            executed=True, raw_samples=10, allow_execute=True,
        )
        assert verdict["status"] == exp.STATUS_BLOCKED

    def test_verdict_refuses_conclusion_without_raw_samples(self, tmp_path):
        status = exp.PrerequisiteStatus(stage="S10", checks=[])
        run = exp.RunDirectory(str(tmp_path), "E10-01", "r0")
        run.create()
        verdict = run.write_verdict(
            status="PASS_NEGATIVE", reason="should be refused", prerequisites=status,
            executed=False, raw_samples=0, allow_execute=True,
        )
        assert verdict["status"] == exp.STATUS_BLOCKED

    def test_non_conclusion_status_is_writable(self, tmp_path):
        status = exp.PrerequisiteStatus(stage="S10", checks=[])
        run = exp.RunDirectory(str(tmp_path), "E10-02", "r0")
        run.create()
        payload = run.write_status(exp.STATUS_BLOCKED, "prerequisites unmet", status)
        assert payload["status"] == exp.STATUS_BLOCKED


@pytest.mark.unit
class TestRunDirectory:
    def test_layout_has_all_directories(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E10-01", "r0")
        run.create()
        for name in exp.RUN_LAYOUT:
            assert os.path.isdir(os.path.join(run.path, name))

    def test_report_skeleton_cannot_be_mistaken_for_a_result(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E10-02", "r0")
        run.create()
        path = run.write_report_skeleton("E10-02", "title")
        text = open(path, encoding="utf-8").read()
        assert "NOT_RUN" in text and "未执行" in text

    def test_record_and_manifest_round_trip(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E10-04", "r0")
        run.create()
        record = exp.template_experiment_record("E10-04", status="NOT_STARTED", reason="tpl")
        parsed = json.load(open(run.write_experiment_record(record), encoding="utf-8"))
        assert parsed["experiment_id"] == "E10-04"
        manifest = exp.EvidenceManifest(
            run_id="r0", world_size=2, node_count=1, backend="nccl", backend_version="2.27.3",
            topology_manifest_sha256="abc", scaleup_kind="strong",
        )
        payload = json.load(open(run.write_evidence_manifest(manifest), encoding="utf-8"))
        assert payload["distributed"]["world_size"] == 2
        assert payload["distributed"]["topology_manifest_sha256"] == "abc"

    def test_cpu_smoke_helpers_are_labelled(self):
        from hqsb.distributed import collectives as coll

        result = coll.LoopbackResult(op="all_reduce", outputs_by_rank={0: [1]})
        assert result.claim_allowed() is False
        assert result.as_dict()["simulated"] is True


@pytest.mark.unit
class TestInterfaceMap:
    def test_all_300_steps_are_mapped(self):
        assert sum(len(mapping.steps) for mapping in imap.EXPERIMENTS) == 300
        assert [mapping.experiment_id for mapping in imap.EXPERIMENTS] == [
            f"E10-{index:02d}" for index in range(1, 11)
        ]

    def test_every_referenced_symbol_resolves(self):
        report = imap.resolve_interfaces()
        assert report["ok"], report["failures"][:10]
        assert report["steps"] == 300
        assert report["experiments"] == 10
        assert report["interfaces"] > 300

    def test_markdown_table_is_emitted(self):
        table = imap.mapping_table_markdown()
        assert "| 实验 |" in table and "300" in table
        step_table = imap.step_table_for("E10-06")
        assert "E10-06" in step_table and "| 30 |" in step_table

    def test_unknown_experiment_is_refused(self):
        with pytest.raises(KeyError):
            imap.mapping_for("E10-11")


@pytest.mark.unit
class TestSpecs:
    def test_specs_load_and_audit(self):
        loaded = specs.DistributedSpecs.load(os.path.join(_REPO_ROOT, "configs", "distributed"))
        assert len(loaded.documents) == len(specs.KINDS)
        assert loaded.ok, [audit for audit in loaded.audits if not audit["ok"]]

    def test_specs_reject_unknown_keys(self, tmp_path):
        import yaml

        path = os.path.join(tmp_path, "bad.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {"kind": specs.KIND_TOPOLOGY_SPEC, "bogus_key": True}, handle
            )
        with pytest.raises(ConfigError):
            specs.load_yaml_document(path)

    def test_specs_reject_unknown_kind(self, tmp_path):
        import yaml

        path = os.path.join(tmp_path, "bad.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump({"kind": "hqsb.distributed.unknown_spec"}, handle)
        with pytest.raises(ConfigError):
            specs.load_yaml_document(path)

    def test_specs_refuse_missing_directory(self, tmp_path):
        with pytest.raises(ConfigError):
            specs.DistributedSpecs.load(str(tmp_path / "nope"))

    def test_fault_matrix_covers_every_fault(self):
        from hqsb.distributed import faults as ft

        loaded = specs.DistributedSpecs.load(os.path.join(_REPO_ROOT, "configs", "distributed"))
        rows = loaded.document(specs.KIND_FAULT_SPEC)["fault_matrix"]
        assert {row["fault"] for row in rows} == set(ft.MINIMUM_RECOVERY_BY_FAULT)
