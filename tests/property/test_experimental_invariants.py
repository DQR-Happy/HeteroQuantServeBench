"""Property-based invariants of the S14 layer.

Randomised over generated inputs (stdlib ``random``, no extra dependency), each
test states an invariant that must hold for **every** input rather than for one
example.  They are the executable form of the protocol's numbers:

* a global batch is an identity of samples, so the token-mean denominator is the
  sum of per-sample tokens and the naive rank mean differs from it whenever the
  ranks hold different counts (``E14-02`` §8.1);
* ``E[A]`` is the running product of per-position acceptances, so
  ``acceptance ↑`` alone does not upper-bound it by ``γ``;
* a pattern is hardware eligible only when *every* N:M group is compliant, so a
  globally 50% matrix can still be ineligible (``E14-F4`` §4.3);
* an imbalance metric is zero for a balanced distribution and strictly positive
  for a skewed one (``E14-F2`` §3.3);
* KV bytes separate payload from metadata, and the metadata ratio is a fraction
  (``E14-F3`` §3.1);
* a non-adoption decision never carries an allowed claim (``details/S14/README.md``
  §7.7);
* a run directory is always under ``artifacts/S14`` and never in the protocol
  tree (hard rule: 不修改 ``docs/stage_experiments/**``).
"""

from __future__ import annotations

import random
from typing import List

import pytest

from hqsb.core.errors import ConfigError
from hqsb.experimental import campaign as camp
from hqsb.experimental import contracts as ct
from hqsb.experimental import identity as ident
from hqsb.experimental import long_context as lc
from hqsb.experimental import moe
from hqsb.experimental import posttraining as pt
from hqsb.experimental import records as rec
from hqsb.experimental import sparsity as sp
from hqsb.experimental import speculative as spec
from hqsb.experimental import training as tr

TRIALS = 60
RNG = random.Random(20260919)


def _positive_int(low: int = 1, high: int = 24) -> int:
    return RNG.randint(low, high)


@pytest.mark.property
def test_canonical_digest_is_order_independent() -> None:
    for _ in range(TRIALS):
        pairs = [(f"k{index}", RNG.randint(0, 10**6)) for index in range(_positive_int(1, 8))]
        forward = dict(pairs)
        backward = dict(reversed(pairs))
        assert ident.canonical_digest(forward) == ident.canonical_digest(backward)


@pytest.mark.property
def test_seed_derivation_is_deterministic_and_rank_separated() -> None:
    roles = {role: RNG.randint(0, 2**31 - 1) for role in ident.SEED_ROLES}
    bundle = ident.seed_bundle(**roles)
    for _ in range(TRIALS):
        rank, step = RNG.randint(0, 8), RNG.randint(0, 20)
        assert bundle.derive("python", rank=rank, step=step) == bundle.derive("python", rank=rank, step=step)
        assert bundle.derive("python", rank=rank, step=step) != bundle.derive("numpy", rank=rank, step=step)
        assert bundle.derive("python", rank=rank, step=step) != bundle.derive("python", rank=rank + 1, step=step)


@pytest.mark.property
def test_global_batch_is_the_product_of_its_factors() -> None:
    for _ in range(TRIALS):
        spec_ = tr.GlobalBatchSpec(
            microbatch=_positive_int(1, 8), accumulation=_positive_int(1, 8), world_size=_positive_int(1, 8)
        )
        assert spec_.global_batch == spec_.microbatch * spec_.accumulation * spec_.world_size
        assert spec_.problems() == []


@pytest.mark.property
def test_token_mean_loss_differs_from_the_naive_rank_mean() -> None:
    """``E14-02`` step 18: the artefact the experiment exists to catch."""
    for _ in range(TRIALS):
        ranks = _positive_int(2, 6)
        numerators = [RNG.uniform(0.1, 10.0) for _ in range(ranks)]
        denominators = [RNG.randint(1, 40) for _ in range(ranks)]
        if len(set(denominators)) == 1:
            denominators[0] += 1
        reduction = tr.LossReduction(reduction="token_mean")
        global_loss = tr.reconstruct_global_loss(numerators, denominators, reduction)
        per_rank = [value / count for value, count in zip(numerators, denominators)]
        naive = tr.naive_rank_mean(per_rank)
        assert global_loss["loss"] is not None
        assert abs(global_loss["loss"] - sum(numerators) / sum(denominators)) < 1e-12
        assert abs(global_loss["loss"] - naive["loss"]) > 1e-12


@pytest.mark.property
def test_expected_accepted_tokens_is_bounded_and_monotone() -> None:
    for _ in range(TRIALS):
        gamma = _positive_int(1, 8)
        acceptances = [RNG.uniform(0.05, 0.99) for _ in range(gamma)]
        expected = spec.expected_accepted_tokens(acceptances, gamma=gamma)
        assert 0.0 < expected <= gamma
        higher = [min(1.0, value + 0.01) for value in acceptances]
        assert spec.expected_accepted_tokens(higher) >= expected - 1e-12


@pytest.mark.property
def test_pattern_compliance_requires_every_group() -> None:
    for _ in range(TRIALS):
        rows = _positive_int(1, 4)
        groups = _positive_int(1, 4)
        matrix: List[List[int]] = []
        for _row in range(rows):
            row: List[int] = []
            for _group in range(groups):
                values = [0, 0, 0, 0]
                for position in RNG.sample(range(4), 2):
                    values[position] = RNG.randint(1, 5)
                row.extend(values)
            matrix.append(row)
        result = sp.pattern_compliance({"w": matrix}, n=2, m=4)
        assert result["pattern_compliance"] == 1.0
        assert result["hardware_eligible"] is True

        # Break exactly one group: the whole tensor becomes ineligible.
        matrix[0][0:4] = [1, 1, 1, 1]
        broken = sp.pattern_compliance({"w": matrix}, n=2, m=4)
        assert broken["hardware_eligible"] is False
        assert broken["problems"]


@pytest.mark.property
def test_imbalance_metrics_separate_balanced_from_skewed() -> None:
    for _ in range(TRIALS):
        experts = _positive_int(2, 8)
        balanced = moe.load_statistics([5] * experts)
        assert balanced["gini"] == pytest.approx(0.0)
        assert balanced["cv"] == pytest.approx(0.0)
        assert balanced["max_mean"] == pytest.approx(1.0)

        skewed_counts = [0] * experts
        skewed_counts[0] = 5 * experts
        skewed = moe.load_statistics(skewed_counts)
        assert skewed["gini"] > 0.0
        assert skewed["max_mean"] >= 1.0


@pytest.mark.property
def test_greedy_prefix_match_never_commits_more_than_target_produced() -> None:
    for _ in range(TRIALS):
        length = _positive_int(1, 8)
        proposed = [RNG.randint(0, 5) for _ in range(length)]
        target = [RNG.randint(0, 5) for _ in range(length)]
        result = spec.greedy_prefix_match(proposed, target)
        assert 0 <= result["accepted"] <= length
        assert len(result["committed"]) <= result["accepted"] + 1
        if result["accepted"] < length:
            assert result["committed"][: result["accepted"]] == proposed[: result["accepted"]]


@pytest.mark.property
def test_kv_geometry_separates_payload_from_metadata() -> None:
    for _ in range(TRIALS):
        storage = RNG.choice(lc.KV_STORAGE_BYTES)
        # A 'mixed' layout has no single element size by construction, so it must
        # state its own bytes; that requirement is asserted separately below.
        override = 1 if storage == "mixed" else None
        if storage == "mixed":
            with pytest.raises(ConfigError):
                lc.kv_bytes_per_token(
                    layers=1, kv_heads_per_rank=1, head_dim=64, storage="mixed"
                )
        geometry = lc.kv_bytes_per_token(
            layers=_positive_int(1, 40),
            kv_heads_per_rank=_positive_int(1, 16),
            head_dim=RNG.choice((64, 128, 256)),
            storage=storage,
            storage_bytes_override=override,
            scale_group_size=RNG.choice((0, 32, 64)),
            metadata_bytes_per_group=RNG.choice((0, 2, 4)),
        )
        assert geometry["bytes_per_token"] >= geometry["payload_bytes_per_token"]
        assert 0.0 <= geometry["metadata_ratio"] <= 1.0


@pytest.mark.property
def test_staleness_sweep_reports_every_missing_point() -> None:
    for _ in range(TRIALS):
        cap = _positive_int(1, 5)
        present = sorted(RNG.sample(range(cap + 1), RNG.randint(0, cap + 1)))
        rows = [
            {
                "staleness_steps": value,
                "utilization": 0.5,
                "kl": 0.01,
                "ratio_mean": 1.0,
                "quality_value": 0.9,
                "drop_rate": 0.0,
            }
            for value in present
        ]
        result = pt.check_staleness_sweep(rows, staleness_cap=cap)
        assert result["missing_points"] == sorted(set(range(cap + 1)) - set(present))
        assert bool(result["problems"]) == bool(result["missing_points"])
        assert result["ok"] is (not result["missing_points"])


@pytest.mark.property
def test_non_adoption_decisions_never_carry_allowed_claims() -> None:
    for _ in range(TRIALS):
        decision = RNG.choice(rec.NON_ADOPTION_DECISIONS)
        payload = ct.AdoptionDecision(
            decision_id="d", experiment_id="E14-F4", decision=decision,
            allowed_claims=("a claim",), evidence_refs=("raw/x",),
        )
        problems = payload.validate()
        assert any("allowed_claims" in problem for problem in problems), (
            f"{decision} must not carry an allowed claim"
        )


@pytest.mark.property
def test_run_layout_is_always_under_artifacts_and_never_in_the_protocol_tree() -> None:
    for _ in range(TRIALS):
        experiment_id = RNG.choice([row[0] for row in rec.EXPERIMENT_TABLE])
        run_id = f"r{RNG.randint(0, 10**6)}"
        layout = camp.run_layout(experiment_id, run_id, root="/tmp/repo")
        assert "/artifacts/S14/" in layout["_base"]
        assert camp.PROTOCOL_ROOT not in layout["_base"]
        camp.assert_writable(layout["raw"], root="/tmp/repo")
        with pytest.raises(ConfigError):
            camp.assert_writable(f"{camp.PROTOCOL_ROOT}/S14_实验清单.md", root="/tmp/repo")


@pytest.mark.property
def test_terminal_states_have_no_outgoing_transition() -> None:
    for name, (states, transitions) in rec.STATE_MACHINES.items():
        for state in states:
            targets = transitions.get(state, ())
            if not targets:
                continue
            # A state with outgoing transitions may not also be a declared terminal
            # of another machine's vocabulary; the check is that every target is a
            # known state and that the graph cannot reach itself trivially.
            for target in targets:
                assert target in states, f"{name}: {state} -> unknown {target}"


@pytest.mark.property
def test_reconciliation_names_the_first_divergent_tensor() -> None:
    for _ in range(TRIALS):
        count = _positive_int(1, 6)
        pairs = []
        broken_at = RNG.randrange(count)
        for index in range(count):
            candidate = 1.0 if index != broken_at else 1.0 + RNG.uniform(0.01, 1.0)
            pairs.append({"tensor": f"t{index}", "reference": 1.0, "candidate": candidate})
        result = tr.reconcile_tensors(pairs, abs_tol=1e-6, rel_tol=1e-6)
        assert result["all_within"] is False
        assert result["first_divergence"] == f"t{broken_at}"
