"""Property-style invariants for ``hqsb.release`` (S15).

These are not experiments: they prove that the *machinery* (digests, claim ids,
evidence graphs, sampling, Amdahl) obeys the invariants the protocol states, and
that the negative paths refuse invalid input rather than silently accepting it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from hqsb.core.errors import ConfigError
from hqsb.release import claims as cl
from hqsb.release import contracts as ct
from hqsb.release import figures as fg
from hqsb.release import hero_replay as hr
from hqsb.release import identity as ident


class TestDigestInvariants:
    def test_digest_is_order_independent(self) -> None:
        left = ident.canonical_digest({"a": 1, "b": 2})
        right = ident.canonical_digest({"b": 2, "a": 1})
        assert left == right

    def test_digest_refuses_nan(self) -> None:
        with pytest.raises(ConfigError):
            ident.canonical_digest({"x": float("nan")})

    def test_aggregate_refuses_empty(self) -> None:
        with pytest.raises(ConfigError):
            ident.content_address_aggregate({})

    def test_aggregate_refuses_non_digest(self) -> None:
        with pytest.raises(ConfigError):
            ident.content_address_aggregate({"a": "not-a-digest"})


class TestClaimIdInvariants:
    def test_wording_does_not_move_id(self) -> None:
        fact = {"claim_type": "performance", "estimand": "tokens/s"}
        assert ident.stable_claim_id(dict(fact, text="更快")) == ident.stable_claim_id(dict(fact, text="faster"))

    def test_scope_change_moves_id(self) -> None:
        fact = {"claim_type": "performance", "estimand": "tokens/s", "hardware_ids": ["a"]}
        assert ident.stable_claim_id(fact) != ident.stable_claim_id(dict(fact, hardware_ids=["b"]))


class TestEvidenceLevelOrder:
    def test_levels_are_strictly_ordered(self) -> None:
        levels = ident.EVIDENCE_LEVELS
        for index in range(len(levels) - 1):
            assert ident.level_index(levels[index]) < ident.level_index(levels[index + 1])

    def test_level_upgrade_refused_without_support(self) -> None:
        findings = ct.check_evidence_level_upgrade(claimed_level="SERVICE", cited_levels=["SOURCE"])
        assert findings


class TestEvidenceGraph:
    def test_cycle_is_rejected(self) -> None:
        graph = cl.EvidenceGraph()
        graph.add_entity("a")
        graph.add_entity("b")
        graph.add_edge("wasDerivedFrom", "a", "b")
        graph.add_edge("wasDerivedFrom", "b", "a")
        assert any("cycle" in problem for problem in graph.validate())

    def test_duplicate_producer_is_rejected(self) -> None:
        graph = cl.EvidenceGraph()
        graph.add_entity("a")
        graph.add_entity("b")
        graph.add_entity("c")
        graph.add_edge("wasDerivedFrom", "a", "c")
        graph.add_edge("wasDerivedFrom", "b", "c")
        assert any("producers" in problem for problem in graph.validate())


class TestSamplingInvariants:
    def _frame(self) -> List[fg.SamplingFrameRow]:
        return [
            fg.SamplingFrameRow(point_id=f"PT-{index}", figure_id="FIG-1", strata={"stage": str(index % 3)}, kinds=("hero_primary_claim",) if index == 0 else ())
            for index in range(10)
        ]

    def test_sampling_is_deterministic(self) -> None:
        frame = self._frame()
        first = fg.draw_sample(frame, seed=7)
        second = fg.draw_sample(frame, seed=7)
        assert first["selected"] == second["selected"]
        assert first["frame_sha256"] == second["frame_sha256"]

    def test_mandatory_samples_always_forced(self) -> None:
        frame = self._frame()
        draw = fg.draw_sample(frame, seed=7)
        assert "PT-0" in draw["selected"]  # the hero sample


class TestAmdahlInvariants:
    def test_formula_matches_hand_computation(self) -> None:
        model = hr.AmdahlModel(hotspot_share=0.3, local_speedup=2.0, recomputed_in_new_environment=True)
        expected = 1.0 / ((1.0 - 0.3) + 0.3 / 2.0)
        assert math.isclose(model.ideal_upper_bound(), expected)

    def test_copied_hotspot_rejected(self) -> None:
        model = hr.AmdahlModel(hotspot_share=0.3, local_speedup=2.0, recomputed_in_new_environment=False)
        assert any("recomputed" in problem for problem in model.problems())


class TestUnitInvariants:
    def test_cross_family_rejected(self) -> None:
        assert cl.check_units_and_magnitudes(left=(120.0, "ms"), right=(100.0, "gb"))

    def test_same_family_different_unit_flagged(self) -> None:
        assert cl.check_units_and_magnitudes(left=(120.0, "ms"), right=(1.0, "s"))

    def test_speedup_zero_denominator_rejected(self) -> None:
        assert cl.check_units_and_magnitudes(left=(1.0, "ms"), right=(0.0, "ms"), comparison="speedup")


class TestClaimGateInvariants:
    def _claim(self, **overrides: Any) -> ct.ClaimRecord:
        payload: Dict[str, Any] = {
            "claim_id": "CLM-1",
            "canonical_text_zh": "更快",
            "canonical_text_en": "faster",
            "claim_type": "performance",
            "evidence_level": "RUNTIME",
            "owner": "author",
            "valid_from_commit": "c" * 40,
        }
        payload.update(overrides)
        return ct.ClaimRecord(**payload)

    def test_gate_failures_are_idempotent(self) -> None:
        claim = self._claim()
        assert claim.gate_failures() == claim.gate_failures()

    def test_quality_fail_blocks_performance(self) -> None:
        claim = self._claim()
        assert "QUALITY_DISQUALIFIED" in claim.gate_failures()

    def test_p0_failures_block_release(self) -> None:
        claim = self._claim()
        assert claim.blocks_release()
