"""Scaffolding and asset tests: interface map, specs, gates and the ``infra/`` tree."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from hqsb.core.errors import ConfigError
from hqsb.infra import experiment as exp
from hqsb.infra import identity, interface_map, observability, records, specs

REPO_ROOT = Path(__file__).resolve().parents[3]
INFRA_ASSETS = REPO_ROOT / "infra"


@pytest.mark.unit
class TestInterfaceMap:
    def test_every_experiment_maps_38_steps(self) -> None:
        assert len(interface_map.EXPERIMENTS) == 11
        assert interface_map.total_steps() == 418
        assert interface_map.steps_without_interfaces() == []
        for mapping in interface_map.EXPERIMENTS:
            assert mapping.complete is True, mapping.experiment_id
            assert mapping.claim_boundary, mapping.experiment_id

    def test_all_referenced_interfaces_resolve(self) -> None:
        result = interface_map.resolve_interfaces()
        assert result["ok"] is True, result["failures"][:5]
        assert result["steps"] == 418
        assert result["interfaces"] > 300

    def test_unknown_reference_is_reported_not_ignored(self) -> None:
        assert interface_map._resolve("supply_chain:does_not_exist") is not None
        assert interface_map._resolve("nope:thing") is not None
        assert interface_map._resolve("identity:parse_digest") is None

    def test_step_table_is_generated_from_the_mapping(self) -> None:
        table = interface_map.step_table_markdown("E13-01")
        assert "| 1 |" in table and "| 38 |" in table
        assert "identity:parse_digest" not in table or "supply_chain" in table

    def test_generated_report_table_is_not_stale(self) -> None:
        """The report cites a generated table; a stale one would cite the wrong code."""
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.infra import gen_interface_map  # type: ignore[import-not-found]

        path = REPO_ROOT / "docs" / "reports" / "S13_interface_map_generated.md"
        assert path.is_file()
        assert path.read_text(encoding="utf-8") == gen_interface_map.render()


@pytest.mark.unit
class TestSpecs:
    def test_all_thirteen_documents_load_and_audit(self) -> None:
        documents = specs.InfraSpecs.load(str(REPO_ROOT / "configs" / "infra"))
        assert len(documents.documents) == 13
        reports = documents.audit()
        assert specs.audit_all_ok(reports) is True, [r for r in reports if not r["ok"]]

    def test_documents_are_regenerated_from_the_code(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.infra.gen_infra_specs import check_documents  # type: ignore[import-not-found]

        ok, drifted = check_documents(str(REPO_ROOT / "configs" / "infra"))
        assert ok is True, drifted

    def test_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "release_identity_spec.yaml"
        path.write_text(
            yaml.safe_dump({"kind": "hqsb.infra.release_identity_spec", "name": "x", "bogus": 1}),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError):
            specs.load_yaml_document(str(path))

    def test_experiment_spec_declares_safety_requirement(self) -> None:
        documents = specs.InfraSpecs.load(str(REPO_ROOT / "configs" / "infra"))
        spec = documents.document(specs.KIND_EXPERIMENT)
        assert spec["safety_policy_required"] is True
        assert spec["removed_validation"]
        assert spec["run_root"] == "experiment_results/S13"


@pytest.mark.unit
class TestExperimentGates:
    def test_prerequisites_report_states_not_booleans_only(self) -> None:
        status = exp.check_prerequisites(str(REPO_ROOT), probe=False)
        payload = status.as_dict()
        assert payload["satisfied"] is False
        assert payload["missing"]
        for check in payload["checks"]:
            if not check["satisfied"] and check["required"]:
                assert check["state"] in records.PREREQUISITE_STATES, check

    def test_verdict_without_execute_is_blocked(self, tmp_path: Path) -> None:
        status = exp.check_prerequisites(str(REPO_ROOT), probe=False)
        run = exp.RunDirectory(str(tmp_path), "E13-01", "gate-test")
        run.create()
        verdict = run.write_verdict(
            status=records.STATUS_PASS, reason="looks fine", prerequisites=status,
            executed=False, allow_execute=False, raw_samples=0,
        )
        assert verdict["status"] == records.STATUS_BLOCKED
        assert "execution is disabled" in verdict["reason"]

    def test_verdict_with_execute_but_no_raw_samples_is_blocked(self, tmp_path: Path) -> None:
        status = exp.check_prerequisites(str(REPO_ROOT), probe=False)
        run = exp.RunDirectory(str(tmp_path), "E13-01", "gate-test-2")
        run.create()
        verdict = run.write_verdict(
            status=records.STATUS_PASS, reason="run finished", prerequisites=status,
            executed=True, allow_execute=True, raw_samples=0,
        )
        assert verdict["status"] == records.STATUS_BLOCKED
        assert "prerequisites unsatisfied" in verdict["reason"] or "no raw samples" in verdict["reason"]

    def test_preregistration_must_freeze_thresholds_and_non_claims(self, tmp_path: Path) -> None:
        run = exp.RunDirectory(str(tmp_path), "E13-06", "prereg")
        run.create()
        prereg = exp.Preregistration(
            experiment_id="E13-06", run_id="prereg", hypothesis="h", primary_metric="false_admit_rate",
            thresholds={}, claim_boundary="code level", non_claims=(),
        )
        with pytest.raises(ConfigError):
            run.write_preregistration(prereg)

    def test_evidence_manifest_refuses_unbacked_levels(self) -> None:
        manifest = exp.EvidenceManifest(run_id="r", evidence_level=records.EVIDENCE_TESTED_SINGLE_RUN)
        problems = manifest.validate()
        assert any("raw artifacts" in problem for problem in problems)
        assert any("release identity" in problem for problem in problems)

    def test_interface_only_run_writes_blocked_verdict(self, tmp_path: Path) -> None:
        result = exp.interface_only_run(str(tmp_path), "E13-02", run_id="iface-only", probe=False)
        assert result["verdict"]["status"] == records.STATUS_BLOCKED
        assert Path(result["run_dir"], "interface_map.json").is_file()
        assert Path(result["run_dir"], "prerequisites.json").is_file()

    def test_driver_rejects_conclusions_by_default(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/infra/run_e13.py", "--experiment", "E13-01", "--json"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["status"] == records.STATUS_BLOCKED
        assert payload["execution_allowed"] is False

    def test_driver_smoke_is_labelled(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.infra import run_e13  # type: ignore[import-not-found]

        collected: dict = {}
        run_e13.cmd_smoke.__wrapped__ if hasattr(run_e13.cmd_smoke, "__wrapped__") else None
        # Call the command through its public path so the printed payload is checked.
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            run_e13.cmd_smoke(True)
        collected = json.loads(buffer.getvalue())
        assert collected["status"] == "smoke"
        assert collected["claim_allowed"] is False
        assert collected["interface_map"]["ok"] is True
        assert collected["spec_audit_ok"] is True


@pytest.mark.unit
class TestInfraAssets:
    def test_dockerfile_is_multi_stage_non_root_and_model_free(self) -> None:
        text = (INFRA_ASSETS / "containers" / "Dockerfile.runtime").read_text(encoding="utf-8")
        for stage in ("AS base", "AS toolchain", "AS test", "AS runtime"):
            assert stage in text
        assert "USER ${HQSB_UID}:${HQSB_GID}" in text
        assert "safetensors" not in text.replace("*.safetensors", "")
        assert "COPY --from=toolchain /wheels/" in text

    def test_build_args_parse_and_expose_placeholders(self) -> None:
        document = yaml.safe_load((INFRA_ASSETS / "containers" / "build_args.yaml").read_text(encoding="utf-8"))
        assert document["base_image_digest"] == ""
        assert "reproducibility_target" in document
        assert document["status"] == "IMPLEMENTED_UNVERIFIED"

    def test_dockerignore_excludes_models_and_credentials(self) -> None:
        text = (INFRA_ASSETS / "containers" / ".dockerignore").read_text(encoding="utf-8")
        for pattern in ("*.safetensors", "*.pem", ".env", "build/"):
            assert pattern in text

    def test_helm_values_carry_policy_default_markers(self) -> None:
        values = yaml.safe_load(
            (INFRA_ASSETS / "deploy" / "helm" / "hqsb" / "values.yaml").read_text(encoding="utf-8")
        )
        assert values["status"] == "IMPLEMENTED_UNVERIFIED"
        assert values["namespace"]["labels"]["hqsb.dev/status"] == records.POLICY_DEFAULT_MARKER
        assert values["autoscaling"]["enabled"] is False  # elasticity is not claimed before E13-07
        assert values["release"]["imageIndexDigest"] == ""

    def test_helm_templates_separate_probes_and_pin_digest(self) -> None:
        workload = (INFRA_ASSETS / "deploy" / "helm" / "hqsb" / "templates" / "workload.yaml").read_text(
            encoding="utf-8"
        )
        assert workload.count("Probe:") == 3
        for probe in ("startup", "readiness", "liveness"):
            assert f".Values.probes.{probe}.path" in workload
        values = yaml.safe_load(
            (INFRA_ASSETS / "deploy" / "helm" / "hqsb" / "values.yaml").read_text(encoding="utf-8")
        )
        paths = {values["probes"][probe]["path"] for probe in ("startup", "readiness", "liveness")}
        assert len(paths) == 3, "the three probes must ask three different questions"
        assert "{{ .Values.release.platformImageDigest }}" in workload
        assert "runAsNonRoot: true" in workload
        policy = (INFRA_ASSETS / "deploy" / "helm" / "hqsb" / "templates" / "namespace-policy.yaml").read_text(
            encoding="utf-8"
        )
        assert 'policyTypes: ["Ingress", "Egress"]' in policy
        assert "ResourceQuota" in policy

    def test_semantic_conventions_load_and_validate(self) -> None:
        report = observability.load_semantic_conventions(
            str(INFRA_ASSETS / "observability" / "semantic_conventions.yaml")
        )
        assert report["ok"] is True, report["problems"]
        assert len(report["conventions"]) >= 15

    def test_alert_rules_have_runbooks_and_marked_thresholds(self) -> None:
        report = observability.load_alert_rules(str(INFRA_ASSETS / "observability" / "alerts.yaml"))
        assert report["ok"] is True, report["problems"]
        assert len(report["rules"]) >= 8
        for rule in report["rules"]:
            assert rule.runbook_id
            assert rule.threshold_source == records.POLICY_DEFAULT_MARKER

    def test_dashboard_panels_reference_sources(self) -> None:
        document = yaml.safe_load(
            (INFRA_ASSETS / "observability" / "dashboards.yaml").read_text(encoding="utf-8")
        )
        assert document["drilldown_order"] == ["user_sli", "queue_runtime", "device_system"]
        for panel in document["panels"]:
            assert panel["query"] and panel["unit"] and panel["source_metric"]
            assert panel["evidence_ref"]

    def test_ci_workflow_is_valid_yaml_and_fails_on_missing_tooling(self) -> None:
        document = yaml.safe_load((INFRA_ASSETS / "ci" / "release_gate.yaml").read_text(encoding="utf-8"))
        assert "jobs" in document
        text = (INFRA_ASSETS / "ci" / "release_gate.yaml").read_text(encoding="utf-8")
        assert "sys.exit(1 if missing else 0)" in text
        assert "if: false" in text  # cluster gate disabled until a safety policy exists

    def test_runbooks_referenced_by_alerts_exist(self) -> None:
        report = observability.load_alert_rules(str(INFRA_ASSETS / "observability" / "alerts.yaml"))
        for rule in report["rules"]:
            assert (INFRA_ASSETS / rule.runbook_id).is_file(), rule.runbook_id
        index = (INFRA_ASSETS / "runbooks" / "index.md").read_text(encoding="utf-8")
        assert "DESIGN_ONLY" in index

    def test_readme_states_the_unverified_status(self) -> None:
        text = (INFRA_ASSETS / "README.md").read_text(encoding="utf-8")
        assert "IMPLEMENTED_UNVERIFIED" in text or "DESIGN_ONLY" in text
        assert "POLICY_DEFAULT_UNVERIFIED" in text

    def test_no_credentials_or_weights_committed_under_infra(self) -> None:
        forbidden_suffixes = (".safetensors", ".bin", ".pt", ".pem", ".key", ".env")
        offenders = [
            str(path.relative_to(REPO_ROOT))
            for path in INFRA_ASSETS.rglob("*")
            if path.is_file() and path.suffix.lower() in forbidden_suffixes
        ]
        assert offenders == []


@pytest.mark.unit
class TestIdentityIntegration:
    def test_release_bundle_from_assets_is_rejected_until_frozen(self) -> None:
        values = yaml.safe_load(
            (INFRA_ASSETS / "deploy" / "helm" / "hqsb" / "values.yaml").read_text(encoding="utf-8")
        )
        bundle = identity.ReleaseBundle(
            release_id="rel-x",
            source_commit="a1da704",
            image_index_digest=values["release"]["imageIndexDigest"],
            model_artifact_id=values["release"]["modelArtifactId"],
        )
        problems = bundle.validate()
        assert any("image_index_digest" in problem for problem in problems)
        assert any("missing required identity field" in problem for problem in problems)
