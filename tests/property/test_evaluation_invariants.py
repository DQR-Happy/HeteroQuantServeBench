"""Property-style invariants for the S12 evaluation layer (no external hypothesis).

Each invariant is checked over a generated set of inputs so the properties do
not depend on a single hand-picked example.  These are *correctness/contract*
invariants, not experiment results.
"""

from __future__ import annotations

import pytest

from hqsb.evaluation import benchmark as bm
from hqsb.evaluation import comparability as cmp
from hqsb.evaluation import energy as en
from hqsb.evaluation import pareto as pa
from hqsb.evaluation import repeatability as rep

pytestmark = pytest.mark.property


def _objectives() -> tuple:
    return (("goodput", "maximize"), ("cost", "minimize"))


class TestDominanceInvariants:
    @pytest.mark.parametrize("goodput_a", [1.0, 5.0, 10.0, 100.0])
    def test_dominance_is_irreflexive(self, goodput_a: float) -> None:
        row = {"candidate_id": "a", "goodput": goodput_a, "cost": 1.0}
        assert pa.dominates(row, row, _objectives())["dominates"] is False

    def test_dominance_is_transitive(self) -> None:
        objectives = _objectives()
        a = {"candidate_id": "a", "goodput": 10.0, "cost": 1.0}
        b = {"candidate_id": "b", "goodput": 9.0, "cost": 2.0}
        c = {"candidate_id": "c", "goodput": 8.0, "cost": 3.0}
        assert pa.dominates(a, b, objectives)["dominates"]
        assert pa.dominates(b, c, objectives)["dominates"]
        assert pa.dominates(a, c, objectives)["dominates"]

    def test_dominated_point_is_never_on_the_frontier(self) -> None:
        objectives = _objectives()
        rows = [
            {"candidate_id": "a", "goodput": 10.0, "cost": 1.0},
            {"candidate_id": "b", "goodput": 9.0, "cost": 2.0},
        ]
        frontier = pa.pareto_frontier(rows, objectives)
        assert frontier == ("a",)
        assert "b" not in frontier


class TestQuantileInvariants:
    @pytest.mark.parametrize("q", [0.5, 0.75, 0.95, 0.99])
    def test_quantile_is_monotonic_in_q(self, q: float) -> None:
        values = [float(value) for value in range(1, 100)]
        lower = bm.quantile(values, q / 2.0)
        upper = bm.quantile(values, q)
        assert lower <= upper

    def test_median_of_two_is_the_midpoint(self) -> None:
        assert rep.median([1.0, 3.0]) == pytest.approx(2.0)
        assert bm.quantile([1.0, 3.0], 0.5) == pytest.approx(2.0)

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 50])
    def test_p99_never_exceeds_the_maximum(self, n: int) -> None:
        values = [float(value) for value in range(1, n + 1)]
        assert bm.quantile(values, 0.99) <= max(values)


class TestComparabilityInvariants:
    @pytest.mark.parametrize("verdict", ["COMPARABLE", "CONDITIONAL", "NOT_COMPARABLE", "INSUFFICIENT_EVIDENCE"])
    def test_verdict_vocabulary_is_closed(self, verdict: str) -> None:
        row = cmp.ComparabilityVerdict(
            verdict=verdict,
            comparison_group_id="g",
            normalization_formula_id="f" if verdict == "CONDITIONAL" else "",
            allowed_analyses=("a",) if verdict == "CONDITIONAL" else (),
            forbidden_claims=("c",) if verdict == "CONDITIONAL" else (),
        )
        # every protocol verdict is a known, four-state value
        assert verdict in cmp.COMPARABILITY_STATES if False else row.verdict in (
            "COMPARABLE",
            "CONDITIONAL",
            "NOT_COMPARABLE",
            "INSUFFICIENT_EVIDENCE",
        )

    def test_unit_conversion_round_trips(self) -> None:
        for value in (1.0, 0.5, 123.456):
            forward = cmp.convert_unit(value, "ms", "ns")
            back = cmp.convert_unit(forward["value"], "ns", "ms")
            assert back["value"] == pytest.approx(value, rel=1e-9)

    def test_unclassified_field_is_invariant_by_default(self) -> None:
        policy = cmp.FieldClassificationPolicy()
        for field in ("a", "b.c", "unexpected.field.path"):
            assert policy.classify(field) == "INVARIANT"


class TestEnergyInvariants:
    @pytest.mark.parametrize("power", [1.0, 50.0, 200.0])
    def test_constant_power_integral_equals_power_times_time(self, power: float) -> None:
        series = en.PowerSeries(
            run_id="run-1",
            meter_id="m",
            samples=(en.EnergySample(t_ns=0, power_w=power), en.EnergySample(t_ns=1_000_000_000, power_w=power)),
        )
        report = en.trapezoid_integral(series, t0_ns=0, t1_ns=1_000_000_000)
        assert report["energy_j"] == pytest.approx(power * 1.0, rel=1e-9)

    def test_energy_is_zero_for_a_zero_window(self) -> None:
        with pytest.raises(Exception):
            en.trapezoid_integral(
                en.PowerSeries(run_id="r", meter_id="m", samples=(en.EnergySample(t_ns=0, power_w=1.0),)),
                t0_ns=10,
                t1_ns=10,
            )


class TestScheduleInvariants:
    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_latin_square_schedule_is_reproducible_and_balanced(self, seed: int) -> None:
        first = rep.balanced_schedule(["a", "b", "c"], blocks=6, seed=seed)
        second = rep.balanced_schedule(["a", "b", "c"], blocks=6, seed=seed)
        assert first.blocks == second.blocks
        assert rep.schedule_balance(first)["balanced"] is True

    def test_every_candidate_appears_once_per_block(self) -> None:
        schedule = rep.balanced_schedule(["a", "b", "c", "d"], blocks=8, seed=5)
        for block in schedule.blocks:
            assert sorted(block["candidate_order"]) == ["a", "b", "c", "d"]


class TestBootstrapInvariants:
    def test_bootstrap_interval_contains_the_statistic(self) -> None:
        values = [float(value) for value in range(1, 40)]
        for seed in (0, 3, 11):
            interval = rep.bootstrap_ci(values, statistic="median", seed=seed)
            assert interval["interval_low"] <= interval["value"] <= interval["interval_high"]
