"""Tests for calibration identity/leakage/selection and statistics (E05-03/E05-10)."""

from __future__ import annotations

import math

import pytest

from hqsb.core.errors import ConfigError
from hqsb.quant import stats as qstats
from hqsb.quant.calibration import (
    CandidateResult,
    DataSpec,
    SPLIT_CALIBRATION,
    SPLIT_FINAL_EVALUATION,
    SampleRecord,
    SplitManifest,
    audit_leakage,
    draw_subset,
    minhash_similarity,
    normalized_text_hash,
    select_minimum_sufficient,
    token_ids_hash,
)


def _sample(sample_id, tokens, split=SPLIT_CALIBRATION, **kwargs):
    return SampleRecord(
        dataset="d",
        revision="r1",
        split=split,
        sample_id=sample_id,
        text_hash=normalized_text_hash(" ".join(map(str, tokens))),
        token_ids=list(tokens),
        length_bucket="short" if len(tokens) < 30 else "long",
        domain="general",
        **kwargs,
    )


@pytest.mark.unit
class TestSampleIdentity:
    def test_token_hash_is_stable_and_order_sensitive(self):
        assert token_ids_hash([1, 2, 3]) == token_ids_hash([1, 2, 3])
        assert token_ids_hash([1, 2, 3]) != token_ids_hash([3, 2, 1])

    def test_normalized_text_hash_ignores_case_and_whitespace(self):
        assert normalized_text_hash("Hello  World") == normalized_text_hash("hello world")

    def test_unknown_split_is_refused(self):
        with pytest.raises(ConfigError, match="split"):
            SampleRecord(dataset="d", revision="r", split="nope", sample_id="s", text_hash="h", token_ids=[1])

    def test_empty_sample_id_is_refused(self):
        with pytest.raises(ConfigError):
            SampleRecord(dataset="d", revision="r", split=SPLIT_CALIBRATION, sample_id="", text_hash="h", token_ids=[1])


@pytest.mark.unit
class TestLeakageAudit:
    def test_planted_id_leak_is_found(self):
        cal = SplitManifest(split=SPLIT_CALIBRATION)
        cal.add(_sample("s0", [1, 2, 3]))
        fin = SplitManifest(split=SPLIT_FINAL_EVALUATION)
        fin.add(_sample("s0", [9, 9, 9], split=SPLIT_FINAL_EVALUATION))
        report = audit_leakage([cal, fin])
        assert report["leaked"] is True
        assert any(f["kind"] == "id_intersection" for f in report["findings"])

    def test_clean_manifests_report_no_leak(self):
        cal = SplitManifest(split=SPLIT_CALIBRATION)
        cal.add(_sample("a", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]))
        fin = SplitManifest(split=SPLIT_FINAL_EVALUATION)
        fin.add(_sample("b", [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104], split=SPLIT_FINAL_EVALUATION))
        report = audit_leakage([cal, fin])
        assert report["leaked"] is False

    def test_parent_document_leak_is_found(self):
        cal = SplitManifest(split=SPLIT_CALIBRATION)
        cal.add(_sample("a", [1, 2, 3], parent_id="doc-1"))
        fin = SplitManifest(split=SPLIT_FINAL_EVALUATION)
        fin.add(_sample("b", [4, 5, 6], split=SPLIT_FINAL_EVALUATION, parent_id="doc-1"))
        kinds = [f["kind"] for f in audit_leakage([cal, fin])["findings"]]
        assert "parent_document" in kinds

    def test_minhash_similarity_bounds(self):
        a = (1, 2, 3, 4, 5)
        assert minhash_similarity(a, a) == 1.0
        assert minhash_similarity(a, (9, 9, 9, 9, 9)) == 0.0

    def test_benchmark_contamination_without_lookup_is_reported(self):
        cal = SplitManifest(split=SPLIT_CALIBRATION)
        cal.add(_sample("a", [1, 2, 3]))
        report = audit_leakage([cal], benchmark_answer_strings=["secret"])
        assert report["benchmark_contamination"][0]["checked"] is False


@pytest.mark.unit
class TestSubsetSampling:
    def test_draw_is_deterministic(self):
        pool = [_sample(f"s{i}", [j for j in range(i, i + 10)]) for i in range(30)]
        a = draw_subset(pool, sample_count=10, seed=7)
        b = draw_subset(pool, sample_count=10, seed=7)
        assert a["sample_ids"] == b["sample_ids"]
        assert a["subset_hash"] == b["subset_hash"]

    def test_draw_different_seeds_differ(self):
        pool = [_sample(f"s{i}", [j for j in range(i, i + 10)]) for i in range(30)]
        a = draw_subset(pool, sample_count=10, seed=1)
        b = draw_subset(pool, sample_count=10, seed=2)
        assert a["sample_ids"] != b["sample_ids"]

    def test_token_budget_is_respected_without_truncation(self):
        pool = [_sample(f"s{i}", [1] * (10 + i)) for i in range(20)]
        result = draw_subset(pool, sample_count=20, seed=0, token_budget=50)
        assert result["valid_tokens"] <= 50
        assert result["token_budget"] == 50

    def test_bad_count_is_refused(self):
        with pytest.raises(ConfigError):
            draw_subset([_sample("a", [1])], sample_count=0, seed=0)


@pytest.mark.unit
class TestDataSpec:
    def test_spec_hash_is_stable(self):
        spec = DataSpec(name="t", sources=[{"dataset": "d"}], length_buckets={"short": (0, 30)})
        assert spec.data_spec_hash == spec.data_spec_hash
        other = DataSpec(name="t2", sources=[{"dataset": "d"}], length_buckets={"short": (0, 30)})
        assert spec.data_spec_hash != other.data_spec_hash

    def test_bucket_of(self):
        buckets = {"short": (0, 30), "long": (30, 1000)}
        assert DataSpec(name="t", sources=(), length_buckets=buckets) is not None
        from hqsb.quant.calibration import bucket_of

        assert bucket_of(15, buckets) == "short"
        assert bucket_of(50, buckets) == "long"
        assert bucket_of(5000, buckets) == "out-of-range"


@pytest.mark.unit
class TestStatistics:
    def test_summarize_distribution(self):
        summary = qstats.summarize_distribution([float(i) for i in range(100)])
        assert summary.count == 100
        assert summary.p50 == pytest.approx(49.5)
        assert summary.p999 > summary.p99 > summary.p90
        assert summary.outlier_rate >= 0

    def test_empty_distribution_is_nan_not_zero(self):
        summary = qstats.summarize_distribution([])
        assert summary.count == 0
        assert math.isnan(summary.mean)
        assert math.isnan(summary.p50)

    def test_bootstrap_ci_covers_the_point(self):
        interval = qstats.bootstrap_ci([float(i % 5) for i in range(200)], resamples=500, seed=0)
        assert interval.low <= interval.point <= interval.high
        assert interval.n_units == 200

    def test_paired_bootstrap(self):
        interval = qstats.paired_bootstrap_ci([0.1, 0.2, -0.05, 0.15], resamples=500, seed=1)
        assert interval.statistic == "paired_mean_delta"

    def test_cluster_bootstrap_resamples_clusters(self):
        interval = qstats.cluster_bootstrap_ci(
            [[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]], resamples=300, seed=0
        )
        assert interval.method == "cluster_bootstrap"
        assert interval.n_units == 3

    def test_cohens_d(self):
        assert qstats.cohens_d_paired([1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=0) or qstats.cohens_d_paired([1.0, 1.0, 1.0]) is None
        assert qstats.cohens_d_paired([2.0]) is None

    def test_non_inferiority_pass(self):
        verdict = qstats.non_inferiority_verdict(
            "ppl", qstats.LOWER_IS_BETTER, [10, 11, 12, 13], [10.1, 11.1, 12.1, 13.1], 0.5, resamples=400, seed=0
        )
        assert verdict.verdict == qstats.GATE_PASS

    def test_non_inferiority_fail(self):
        verdict = qstats.non_inferiority_verdict(
            "acc", qstats.HIGHER_IS_BETTER, [0.9, 0.9, 0.9, 0.9], [0.5, 0.5, 0.5, 0.5], 0.1, resamples=400, seed=0
        )
        assert verdict.verdict == qstats.GATE_FAIL

    def test_non_inferiority_inconclusive_with_no_samples(self):
        verdict = qstats.non_inferiority_verdict("ppl", qstats.LOWER_IS_BETTER, [], [], 0.5)
        assert verdict.verdict == qstats.GATE_INCONCLUSIVE

    def test_unknown_direction_is_refused(self):
        with pytest.raises(ConfigError):
            qstats.non_inferiority_verdict("ppl", "sideways", [1], [1], 0.5)

    def test_stable_subset_agreement(self):
        report = qstats.stable_subset_agreement([1.0, 2.0], [1.0, 2.0], tolerance=0.01)
        assert report["within_tolerance_fraction"] == 1.0
        assert report["cosine"] == pytest.approx(1.0)

    def test_histogram_sketch_merge(self):
        sketch = qstats.HistogramSketch(bin_edges=(0.0, 10.0, 5))
        sketch.add(2.0)
        other = qstats.HistogramSketch(bin_edges=(0.0, 10.0, 5))
        other.add(7.0)
        merged = sketch.merge(other)
        assert merged.as_dict()["total"] == 2

    def test_jaccard(self):
        assert qstats.jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)


@pytest.mark.unit
class TestSelectionRule:
    def _candidate(self, cid, source, count, quality, ci, seed=0, stability=1.0, cost=0.0):
        return CandidateResult(
            candidate_id=cid,
            source=source,
            sample_count=count,
            valid_tokens=count * 100,
            length_coverage="all",
            subset_seed=seed,
            quality=quality,
            quality_ci_low=ci[0],
            quality_ci_high=ci[1],
            statistics_stability=stability,
            offline_cost_s=cost,
        )

    def test_cheapest_stable_candidate_wins(self):
        spec = DataSpec(
            name="t",
            sources=[],
            length_buckets={"all": (0, 9999)},
            stabilization_epsilon=0.05,
            seed_variance_threshold=1.0,
            quality_margin=0.0,
        )
        candidates = [
            self._candidate("a0", "src", 32, 10.0, (9.9, 10.1), seed=0),
            self._candidate("a1", "src", 32, 10.01, (9.91, 10.11), seed=1),
            self._candidate("b0", "src", 64, 9.99, (9.89, 10.09), seed=0),
            self._candidate("b1", "src", 64, 9.99, (9.89, 10.09), seed=1),
        ]
        decision = select_minimum_sufficient(candidates, spec)
        assert decision.selected in ("a0", "a1")

    def test_zero_thresholds_refuse_to_select(self):
        spec = DataSpec(
            name="t", sources=[], length_buckets={"all": (0, 9999)},
            stabilization_epsilon=0.0, seed_variance_threshold=0.0,
        )
        candidates = [self._candidate("a", "src", 32, 10.0, (9.9, 10.1))]
        decision = select_minimum_sufficient(candidates, spec)
        assert decision.selected is None
        assert decision.reason

    def test_no_candidates(self):
        spec = DataSpec(name="t", sources=[], length_buckets={"all": (0, 9999)})
        decision = select_minimum_sufficient([], spec)
        assert decision.selected is None
