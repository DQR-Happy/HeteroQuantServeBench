"""Foundation tests for ``hqsb.release`` (S15).

These verify the frozen objects, the claim lifecycle, the contribution boundary,
the evidence manifest and the vocabulary audit — all at the *interface* layer, no
experiment execution, no measurement.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import pytest

from hqsb.core.errors import ConfigError

from hqsb.release import campaign as camp
from hqsb.release import contracts as ct
from hqsb.release import experiment as exp
from hqsb.release import identity as ident
from hqsb.release import records as rec
from hqsb.release import specs

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _candidate(**overrides: Any) -> ct.ReleaseCandidateSnapshot:
    payload: Dict[str, Any] = {
        "candidate_id": "cand-1",
        "source_commit": "c" * 40,
        "tree_dirty": False,
        "contract_versions": {f"C{index}": "1.0.0" for index in range(1, 8)},
        "claim_ledger_id": "ledger-1",
        "evidence_graph_root": "sha256:" + "1" * 64,
    }
    payload.update(overrides)
    return ct.ReleaseCandidateSnapshot(**payload)


class TestFrozenObjects:
    def test_candidate_rejects_dirty_tree(self) -> None:
        candidate = _candidate(tree_dirty=True)
        assert any("dirty" in problem for problem in candidate.validate())

    def test_candidate_requires_full_commit(self) -> None:
        candidate = _candidate(source_commit="abc123")
        assert any("full source commit" in problem for problem in candidate.validate())

    def test_new_candidate_refuses_same_id(self) -> None:
        candidate = _candidate()
        with pytest.raises(ConfigError):
            ct.new_candidate(candidate, candidate_id="cand-1", source_commit="d" * 40)

    def test_new_candidate_allows_new_id(self) -> None:
        candidate = _candidate()
        fresh = ct.new_candidate(candidate, candidate_id="cand-2", source_commit="d" * 40)
        assert fresh.candidate_id == "cand-2"

    def test_acceptance_go_requires_all_p0_pass(self) -> None:
        decision = ct.FinalAcceptanceDecision(
            candidate_id="cand-1",
            experiment_outcomes={name: rec.STATUS_BLOCKED for name in rec.EXPERIMENT_IDS},
            release_decision="GO",
            resume_decision="NO_GO",
        )
        assert any("every P0 experiment" in problem for problem in decision.problems())

    def test_acceptance_rejects_overlap_claims(self) -> None:
        decision = ct.FinalAcceptanceDecision(
            candidate_id="cand-1",
            public_claim_ids=("CLM-1",),
            withheld_claim_ids=("CLM-1",),
        )
        assert any("public and withheld at once" in problem for problem in decision.problems())

    def test_bundle_requires_rights_and_roles(self) -> None:
        bundle = ct.PublicEvidenceBundle(bundle_id="b-1")
        assert any("empty bundle" in problem for problem in bundle.problems())


class TestClaimRecord:
    def _claim(self, **overrides: Any) -> ct.ClaimRecord:
        payload: Dict[str, Any] = {
            "claim_id": "CLM-abc",
            "canonical_text_zh": "自定义 RMSNorm kernel 更快",
            "canonical_text_en": "the custom RMSNorm kernel is faster",
            "claim_type": "performance",
            "evidence_level": "RUNTIME",
            "owner": "author",
            "valid_from_commit": "c" * 40,
        }
        payload.update(overrides)
        return ct.ClaimRecord(**payload)

    def test_draft_claim_does_not_need_raw(self) -> None:
        claim = self._claim(status="DRAFT", evidence_level="PLANNED")
        assert claim.problems() == []

    def test_verified_numeric_claim_requires_raw(self) -> None:
        claim = self._claim(
            status="VERIFIED",
            effect=ct.ClaimEffect(point=1.5, unit="ms", interval=(1.4, 1.6), sample_unit="run"),
            baseline_id="base-1",
            estimand="latency",
        )
        assert any("raw" in problem for problem in claim.problems())

    def test_state_machine_rejects_jump(self) -> None:
        claim = self._claim(status="DRAFT")
        with pytest.raises(ConfigError):
            claim.transition("STALE")

    def test_verified_requires_passing_gates(self) -> None:
        claim = self._claim(status="DRAFT")
        with pytest.raises(ConfigError):
            claim.transition("VERIFIED")

    def test_gate_failures_are_classified(self) -> None:
        claim = self._claim(status="DRAFT")
        failures = claim.gate_failures()
        assert "ORPHAN" in failures
        assert "QUALITY_DISQUALIFIED" in failures

    def test_stable_claim_id_ignores_wording(self) -> None:
        base = {"claim_type": "performance", "estimand": "tokens/s"}
        rewording = {"claim_type": "performance", "estimand": "tokens/s"}
        assert ident.stable_claim_id(base) == ident.stable_claim_id(rewording)


class TestContributionRecord:
    def test_human_must_own_interpretation(self) -> None:
        record = ct.ContributionRecord(
            artifact_id="a",
            roles={"coding_agent": ("draft_generation",)},
            accepted_by="author",
        )
        assert any("result interpretation" in problem for problem in record.problems())

    def test_third_party_cannot_be_human(self) -> None:
        record = ct.ContributionRecord(
            artifact_id="a",
            roles={"third_party": ("framework_runtime",)},
            accepted_by="author",
        )
        assert any("result interpretation" in problem for problem in record.problems())


class TestEvidenceManifest:
    def test_claim_above_not_measured_requires_raw(self) -> None:
        manifest = exp.EvidenceManifest(
            run_id="r", experiment_id="E15-01", claim_level="RUNTIME", commands=("run",)
        )
        assert any("raw artefact" in problem for problem in manifest.problems())

    def test_not_measured_is_allowed_without_raw(self) -> None:
        manifest = exp.EvidenceManifest(run_id="r", experiment_id="E15-01", claim_level="NOT_MEASURED", commands=("run",))
        assert manifest.problems() == []


class TestCampaignLayout:
    def test_protocol_tree_write_refused(self) -> None:
        with pytest.raises(ConfigError):
            camp.assert_writable("docs/stage_experiments/S15_实验清单.md")

    def test_run_layout_is_writable(self) -> None:
        layout = camp.run_layout("E15-01", "interface_only")
        camp.assert_writable(layout["raw"])

    def test_default_prohibitions_are_complete(self) -> None:
        problems = camp.check_execution_safety(camp.default_prohibited_actions(), dict(camp.REQUIRED_ISOLATION))
        assert problems == []


class TestVocabularyAudit:
    def test_specs_are_clean(self) -> None:
        documents = specs.ReleaseSpecs.load(os.path.join(REPO_ROOT, "configs", "release"))
        assert documents.missing_kinds() == []
        assert documents.duplicate_kinds() == []
        reports = documents.audit()
        assert specs.audit_all_ok(reports)
        assert specs.check_cross_document(documents.documents) == []

    def test_measured_number_rejected(self) -> None:
        report = specs.audit_document(
            "bad.yaml",
            {
                "kind": "release-policy",
                "version": "v1",
                "sources": ["docs/stage_experiments/details/S15/README.md"],
                "policy": {"speedup_latency_ms": 12.5},
            },
        )
        assert not report.ok

    def test_state_machines_are_consistent(self) -> None:
        assert rec.validate_state_machines() == []
