"""E07-08 speculative decoding / MTP contract and benefit model (P1)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from hqsb.core.errors import ConfigError, SchemaError
from hqsb.runtime import spec_decode as S


def _identity(**overrides) -> S.DraftTargetIdentity:
    payload = {
        "target_model_id": "Qwen/Qwen3-1.7B",
        "target_revision": "rev-1",
        "target_tokenizer_id": "Qwen/Qwen3-1.7B",
        "target_vocab_size": 151936,
        "target_precision": "float16",
    }
    payload.update(overrides)
    return S.DraftTargetIdentity(**payload)


def _cycle(**overrides) -> S.CycleRecord:
    payload = {
        "cycle_index": 0,
        "proposed": 3,
        "accepted": 2,
        "correction_token_committed": True,
        "advanced_tokens": 3,
        "draft_ms": 1.0,
        "verify_ms": 4.0,
    }
    payload.update(overrides)
    return S.CycleRecord(**payload)


@pytest.mark.unit
class TestDraftTargetIdentity:
    def test_matching_tokenizers_are_compatible(self):
        identity = _identity(draft_model_id="draft/tiny", draft_vocab_size=151936, draft_tokenizer_id="Qwen/Qwen3-1.7B")
        assert identity.compatibility()["compatible"]
        identity.require_compatible()

    def test_vocabulary_mismatch_without_mapping_is_refused(self):
        identity = _identity(draft_model_id="draft/tiny", draft_vocab_size=32000)
        assert not identity.compatibility()["compatible"]
        with pytest.raises(SchemaError):
            identity.require_compatible()

    def test_verified_mapping_allows_a_different_vocabulary(self):
        identity = _identity(
            draft_model_id="draft/tiny",
            draft_vocab_size=32000,
            token_mapping_verified=True,
        )
        assert identity.compatibility()["compatible"]

    def test_draft_without_vocab_size_refused(self):
        with pytest.raises(ConfigError):
            _identity(draft_model_id="draft/tiny")

    def test_gamma_must_be_at_least_one(self):
        with pytest.raises(ConfigError):
            _identity(gamma=0)


@pytest.mark.unit
class TestAcceptanceMathematics:
    def test_acceptance_probability_is_clamped(self):
        assert S.acceptance_probability(Fraction(1, 2), Fraction(1, 4)) == 1
        assert S.acceptance_probability(Fraction(1, 4), Fraction(1, 2)) == Fraction(1, 2)

    def test_zero_draft_probability_is_refused(self):
        with pytest.raises(ConfigError):
            S.acceptance_probability(Fraction(1, 2), Fraction(0))

    def test_residual_distribution_normalises_exactly(self):
        residual = S.residual_distribution(
            (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)),
            (Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)),
        )
        assert sum(residual) == 1
        assert S.residual_is_normalized(residual)
        assert residual[2] == 0

    def test_degenerate_residual_refused(self):
        with pytest.raises(ConfigError):
            S.residual_distribution((Fraction(1, 4),), (Fraction(1, 2),))

    def test_mismatched_vocabularies_refused(self):
        with pytest.raises(ConfigError):
            S.residual_distribution((Fraction(1, 2),), (Fraction(1, 2), Fraction(1, 2)))

    def test_verify_acceptance_matches_the_hand_computed_golden(self):
        target = (Fraction(1, 2), Fraction(3, 10), Fraction(1, 5))
        draft = (Fraction(1, 2), Fraction(1, 2), Fraction(0))
        accepted = S.verify_acceptance(target, draft, 0, Fraction(1, 10))
        assert accepted["accepted"]
        rejected = S.verify_acceptance(target, draft, 1, Fraction(9, 10))
        assert not rejected["accepted"]
        assert rejected["residual_normalized"]

    def test_proposed_token_outside_the_vocabulary_refused(self):
        with pytest.raises(ConfigError):
            S.verify_acceptance((Fraction(1, 2),), (Fraction(1, 2),), 5, Fraction(1, 2))

    def test_uniform_draw_must_be_a_probability(self):
        with pytest.raises(ConfigError):
            S.verify_acceptance((Fraction(1, 2),), (Fraction(1, 2),), 0, Fraction(3, 2))

    def test_golden_cases_cover_accept_reject_and_zero_draft(self):
        cases = {case.case for case in S.golden_acceptance_cases()}
        assert cases == {"all_accept", "first_reject", "draft_zero_probability"}


@pytest.mark.unit
class TestCyclesAndRollback:
    def test_advanced_tokens_must_equal_accepted_plus_correction(self):
        with pytest.raises(ConfigError):
            _cycle(advanced_tokens=4)

    def test_accepted_cannot_exceed_proposed(self):
        with pytest.raises(ConfigError):
            _cycle(proposed=2, accepted=3)

    def test_cycle_time_sums_every_component(self):
        cycle = _cycle(scheduler_ms=0.5, rollback_ms=0.25, accept_ms=0.25)
        assert cycle.cycle_time_ms == pytest.approx(6.0)

    def test_greedy_exactness_detects_a_mismatch(self):
        report = S.verify_greedy_exactness([1, 2, 3], [1, 2, 9], max_new_tokens=3)
        assert not report["ok"]
        assert report["first_mismatch_index"] == 2

    def test_greedy_exactness_detects_truncation(self):
        report = S.verify_greedy_exactness([1, 2, 3], [1, 2], max_new_tokens=3)
        assert report["truncated"]
        assert not report["ok"]

    def test_kv_commit_rollback_audit_closes_the_books(self):
        report = S.kv_commit_rollback_audit(
            proposed_positions=3,
            accepted_positions=2,
            committed_positions=3,
            rolled_back_positions=1,
        )
        assert report["ok"]

    def test_rollback_mismatch_is_reported(self):
        report = S.kv_commit_rollback_audit(
            proposed_positions=3,
            accepted_positions=2,
            committed_positions=3,
            rolled_back_positions=0,
        )
        assert not report["ok"]

    def test_stale_token_is_a_hard_failure(self):
        report = S.kv_commit_rollback_audit(
            proposed_positions=3,
            accepted_positions=2,
            committed_positions=3,
            rolled_back_positions=1,
            stale_token_detected=True,
        )
        assert not report["ok"]

    def test_committed_positions_cannot_exceed_accepted_plus_one(self):
        report = S.kv_commit_rollback_audit(
            proposed_positions=4,
            accepted_positions=1,
            committed_positions=3,
            rolled_back_positions=3,
        )
        assert not report["ok"]


@pytest.mark.unit
class TestBenefitModel:
    def test_effective_tpot_uses_advanced_tokens(self):
        model = S.BenefitModel(cycles=(_cycle(),), vanilla_tpot_ms=10.0)
        # draft 1.0 + verify 4.0 = 5.0 ms per cycle, advanced 3 tokens.
        assert model.total_cycle_time_ms == pytest.approx(5.0)
        assert model.effective_tpot_ms == pytest.approx(5.0 / 3.0)
        assert model.target_calls_per_output_token == pytest.approx(1 / 3)
        assert model.accepted_over_proposed == pytest.approx(2 / 3)
        assert model.speedup() == pytest.approx(10.0 / (5.0 / 3.0))

    def test_no_advanced_token_refuses_to_divide(self):
        cycle = S.CycleRecord(cycle_index=0, proposed=1, accepted=0, advanced_tokens=0)
        model = S.BenefitModel(cycles=(cycle,), vanilla_tpot_ms=10.0)
        with pytest.raises(ConfigError):
            model.effective_tpot_ms

    def test_benefit_model_needs_cycles_and_a_positive_baseline(self):
        with pytest.raises(ConfigError):
            S.BenefitModel(cycles=(), vanilla_tpot_ms=1.0)
        with pytest.raises(ConfigError):
            S.BenefitModel(cycles=(_cycle(),), vanilla_tpot_ms=0.0)

    def test_gamma_sweep_requires_the_control(self):
        with pytest.raises(ConfigError):
            S.gamma_sweep([2, 3])
        sweep = S.gamma_sweep([3, 1, 3])
        assert sweep["gammas"] == [1, 3]
        assert sweep["control"] == 1

    def test_distribution_gate_is_inconclusive_without_seeds(self):
        gate = S.distribution_gate([0.1], [0.1], pre_registered_bound=0.05)
        assert gate["verdict"] == "INCONCLUSIVE"

    def test_distribution_gate_decides_with_enough_seeds(self):
        gate = S.distribution_gate(
            [1.0] * 8, [1.0] * 8, pre_registered_bound=0.1, min_seeds=8
        )
        assert gate["verdict"] == "PASS"


@pytest.mark.unit
class TestMtpAndClaim:
    def test_mtp_requires_its_own_quality_contract(self):
        with pytest.raises(ConfigError):
            S.MtpContract(heads=1, quality_contract="")

    def test_mtp_may_not_borrow_the_strict_sampling_guarantee(self):
        contract = S.MtpContract(
            heads=1,
            quality_contract="own contract",
            borrows_strict_sampling_guarantee=True,
        )
        with pytest.raises(SchemaError):
            S.assert_mtp_does_not_borrow(contract)

    def test_mtp_may_borrow_when_it_proves_distribution_preservation(self):
        contract = S.MtpContract(
            heads=1,
            quality_contract="proved",
            borrows_strict_sampling_guarantee=True,
            target_distribution_preserved=True,
        )
        S.assert_mtp_does_not_borrow(contract)

    def test_claim_defaults_to_not_run(self):
        status = S.claim_status(S.SpeculativeEvidence(), executed=False, algorithm=S.GREEDY)
        assert status["status"] == "NOT_RUN"
        assert "P1" in status["reason"]

    def test_claim_with_incomplete_evidence_is_not_claimed(self):
        status = S.claim_status(
            S.SpeculativeEvidence(algorithm_frozen=True),
            executed=True,
            algorithm=S.STRICT_SAMPLING,
        )
        assert status["status"] == "NOT_CLAIMED"
        assert "identity_verified" in status["reason"]

    def test_unknown_algorithm_refused(self):
        with pytest.raises(ConfigError):
            S.claim_status(
                S.SpeculativeEvidence(), executed=False, algorithm="beam_speculation"
            )

    def test_full_evidence_claims_only_the_named_algorithm(self):
        evidence = S.SpeculativeEvidence(
            algorithm_frozen=True,
            identity_verified=True,
            vanilla_oracle_compared=True,
            acceptance_verified=True,
            kv_rollback_audited=True,
            cycle_cost_decomposed=True,
            workload_slices_measured=True,
            actual_path_recorded=True,
        )
        status = S.claim_status(evidence, executed=True, algorithm=S.GREEDY)
        assert status["status"] == "CLAIMED"

    def test_workload_slices_include_an_adversarial_case(self):
        slices = {item["slice"] for item in S.workload_slices()}
        assert {"predictable", "adversarial", "eos_near"} <= slices
