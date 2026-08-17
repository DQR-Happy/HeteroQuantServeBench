from hqsb.evaluation.observation_comparison import paired_uncertainty


def test_overhead_needs_independent_pairs_and_preserved_outputs():
    rows = [{"runtime_e2e_relative_delta": 0.01, "tokens_equal": True}] * 10
    assert paired_uncertainty(rows)["status"] == "passed"
    assert paired_uncertainty(rows[:3])["status"] == "insufficient_evidence"
    assert (
        paired_uncertainty(
            rows + [{"runtime_e2e_relative_delta": 0.01, "tokens_equal": False}]
        )["status"]
        == "failed"
    )
    assert (
        paired_uncertainty([{**r, "runtime_e2e_relative_delta": 0.2} for r in rows])[
            "status"
        ]
        == "failed"
    )
