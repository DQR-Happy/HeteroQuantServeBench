"""Scaffolding tests: specs audit, interface map, run directory and verdict gates."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from hqsb.core.errors import ConfigError
from hqsb.evaluation import experiment as exp
from hqsb.evaluation import interface_map as imap
from hqsb.evaluation import specs

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = REPO_ROOT / "configs" / "evaluation"

pytestmark = pytest.mark.unit


class TestInterfaceMapAndSpecs:
    def test_interface_map_resolves_every_step(self) -> None:
        result = imap.resolve_interfaces()
        assert result["ok"] is True, result["failures"]
        assert result["steps"] == 360
        assert result["expected_steps"] == 360
        assert result["interfaces"] >= 300

    def test_every_experiment_has_36_steps_and_a_claim_boundary(self) -> None:
        for mapping in imap.EXPERIMENTS:
            assert mapping.complete, (mapping.experiment_id, mapping.load_error)
            assert len(mapping.steps) == 36
            assert mapping.claim_boundary
            assert mapping.title

    def test_specs_load_and_audit_cleanly(self) -> None:
        documents = specs.EvaluationSpecs.load(str(SPEC_DIR))
        assert len(documents.documents) == 12
        reports = documents.audit()
        assert specs.audit_all_ok(reports), [report for report in reports if not report["ok"]]

    def test_spec_drift_is_detected(self) -> None:
        # A tampered document must fail the audit instead of silently passing.
        with tempfile.TemporaryDirectory() as tmp:
            import shutil

            shutil.copytree(SPEC_DIR, tmp, dirs_exist_ok=True)
            tampered = Path(tmp) / "comparability_spec.yaml"
            text = tampered.read_text(encoding="utf-8")
            tampered.write_text(text.replace("- NOT_COMPARABLE", "- NOT_COMPARABLE_AT_ALL"), encoding="utf-8")
            documents = specs.EvaluationSpecs.load(tmp)
            reports = documents.audit()
            assert not specs.audit_all_ok(reports)

    def test_unknown_spec_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "comparability_spec.yaml"
            bad.write_text(
                "kind: hqsb.evaluation.comparability_spec\nname: x\nmystery_key: 1\n",
                encoding="utf-8",
            )
            with pytest.raises(ConfigError):
                specs.load_yaml_document(str(bad))


class TestPrerequisites:
    def test_prerequisites_are_honest_about_the_repo_state(self) -> None:
        status = exp.check_prerequisites(str(REPO_ROOT))
        assert status.satisfied is False
        # s01/s02 are present; the cross-hardware chain is not (BLOCKED, never PASS)
        assert "s01_contracts_and_identity" not in status.missing
        assert "s02_model_workload_quality_freeze" not in status.missing
        assert "s03_s11_upstream_evidence_chain" in status.missing
        assert "multi_hardware_coverage" in status.missing

    def test_check_reports_evidence_pointers(self) -> None:
        status = exp.check_prerequisites(str(REPO_ROOT))
        satisfied = [check for check in status.checks if check.satisfied]
        assert all(check.evidence for check in satisfied)


class TestRunDirectory:
    def test_run_directory_writes_only_under_experiment_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = exp.RunDirectory(tmp, "E12-01", "run-1")
            assert run.path == str(Path(tmp) / "experiment_results" / "S12" / "E12-01" / "run-1")
            run.create()
            assert (Path(run.path) / "preregistration.json").parent.is_dir()

    def test_unknown_experiment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ConfigError):
                exp.RunDirectory(tmp, "E12-99", "run-1")

    def test_write_verdict_refuses_a_conclusion_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = exp.RunDirectory(tmp, "E12-01", "run-1")
            run.create()
            status = exp.check_prerequisites(str(REPO_ROOT))
            verdict = run.write_verdict(
                status=exp.STATUS_PASS,
                reason="looks good",
                prerequisites=status,
                executed=True,
                allow_execute=False,
                raw_samples=100,
            )
            assert verdict["status"] == exp.STATUS_BLOCKED
            assert "execution is disabled" in verdict["reason"]

    def test_write_verdict_refuses_without_raw_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = exp.RunDirectory(tmp, "E12-01", "run-1")
            run.create()
            satisfied = exp.PrerequisiteStatus(
                stage="S12",
                checks=[exp.PrerequisiteCheck(name="all", satisfied=True, evidence="x", required=True)],
            )
            verdict = run.write_verdict(
                status=exp.STATUS_PASS,
                reason="claims a result",
                prerequisites=satisfied,
                executed=True,
                allow_execute=True,
                raw_samples=0,
            )
            assert verdict["status"] == exp.STATUS_BLOCKED
            assert "raw evidence" in verdict["reason"]

    def test_experiment_record_rejects_silent_fallback(self) -> None:
        record = exp.ExperimentRecord(
            experiment_id="E12-03",
            requested_implementation="triton",
            actual_implementation="eager",
        )
        assert any("fallback" in item for item in record.validate())

    def test_evidence_manifest_requires_lineage_for_regeneration(self) -> None:
        manifest = exp.EvidenceManifest(run_id="run-1", regeneration_level="R3")
        assert any("lineage root" in item for item in manifest.validate())
        manifest.lineage_root = "hqsb://S12/c/lineage"
        manifest.correctness_status = "not_run"
        assert manifest.validate() == []

    def test_interface_only_run_is_not_a_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = exp.interface_only_run(tmp, "E12-01")
            verdict = json.loads((Path(result["run_dir"]) / "verdict.json").read_text(encoding="utf-8"))
            assert verdict["status"] == "BLOCKED"
            assert "interface-only" in verdict["reason"]
            assert verdict["raw_samples"] == 0
