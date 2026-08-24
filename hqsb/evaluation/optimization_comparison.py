"""Pure policy over paired facts. No device, archive, backend or HTTP imports."""

import random
import statistics


def performance_gate(deltas, *, maximum, independent, repetitions=2000):
    if not deltas:
        return {"status": "not_evaluated", "pairs": 0}
    rng = random.Random(0)
    estimates = sorted(statistics.median(rng.choices(deltas, k=len(deltas))) for _ in range(repetitions))
    interval = [estimates[int(repetitions * 0.025)], estimates[int(repetitions * 0.975)]]
    return {"status": "insufficient_evidence" if len(deltas) < 10 or not independent else
            "passed" if interval[1] <= maximum else "failed" if interval[0] > maximum else "inconclusive",
            "pairs": len(deltas), "median_relative_e2e": statistics.median(deltas),
            "interval_95": interval, "maximum": maximum, "independent_processes": independent,
            "repetitions": repetitions, "seed": 0, "method": "paired_bootstrap_median_relative_e2e"}
