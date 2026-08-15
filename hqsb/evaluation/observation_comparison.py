"""Core-only policy over injected observation facts; no evidence IO or devices."""

import random
import statistics


def paired_uncertainty(pairs, *, repetitions=2000, seed=0):
    if not pairs:
        return {"status": "insufficient_evidence", "sample_count": 0, "interval": None}
    values = [row["runtime_e2e_relative_delta"] for row in pairs]
    rng = random.Random(seed)
    samples = sorted(
        statistics.median([rng.choice(values) for _ in values])
        for _ in range(repetitions)
    )
    interval = [samples[int(repetitions * 0.025)], samples[int(repetitions * 0.975)]]
    passed = (
        interval[0] >= -0.05
        and interval[1] <= 0.05
        and all(p["tokens_equal"] for p in pairs)
    )
    return {
        "status": "insufficient_evidence"
        if len(pairs) < 10
        else "passed"
        if passed
        else "failed",
        "sample_count": len(pairs),
        "replication_unit": "independent_process_pair",
        "interval": interval,
        "method": "paired_bootstrap_median_relative_e2e",
        "seed": seed,
        "repetitions": repetitions,
        "scope": "e2e_only_not_full_low_overhead_gate",
    }
