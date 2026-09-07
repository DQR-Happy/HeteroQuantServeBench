"""Deterministic workloads shared by adapters, with an independent output oracle."""

import json
import math
import random


def latency_sample(index, seed, every=64):
    """SplitMix64 avalanche: low-bit multiplication aliases periodic snapshots."""
    mask = (1 << 64) - 1
    z = (index + seed + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    if every not in (1, 4, 16, 64, 1024):
        raise ValueError("sample_every must be one of 1,4,16,64,1024")
    return ((z ^ (z >> 31)) & (every - 1)) == 0


def offset_ns(index, case):
    """Open arrivals retain their original schedule even after a blocked call."""
    rate = case["rate"]
    arrival = case.get("arrival", "periodic")
    if arrival == "closed":
        return None
    if arrival == "burst":
        period = int(case.get("period_ms", 100) * 1_000_000)
        active = int(case.get("active_ms", 1) * 1_000_000)
        # Integer rational arithmetic avoids .3/.1 rounding into the wrong pulse.
        denominator = rate * period
        cycle, fraction = divmod(index * 1_000_000_000, denominator)
        return cycle * period + fraction * active // denominator
    return int(index * 1e9 / rate)


def plan(case):
    if case.get("phases"):
        result, start = [], 0
        for phase in case["phases"]:
            count = int(phase["duration_s"] * phase["rate"])
            result.extend(start + int(i * 1e9 / phase["rate"]) for i in range(count))
            start += int(phase["duration_s"] * 1e9)
        if len(result) > 2_000_000 or start != int(case["duration_s"] * 1e9):
            raise ValueError("invalid phased workload duration/receipt budget")
        return result
    count = case.get("events", int(case["duration_s"] * case["rate"]))
    if not 0 <= count <= 2_000_000:
        raise ValueError("receipt budget: events must be <= 2000000")
    arrival = case.get("arrival", "periodic")
    if arrival in ("poisson", "onoff"):
        rng = random.Random(case["seed"])
        result, t, credit = [], 0.0, 0.0
        horizon = case["duration_s"]
        while t < horizon:
            if arrival == "poisson":
                t += -math.log(max(rng.random(), 1e-15)) / case["rate"]
                if t < horizon:
                    result.append(int(t * 1e9))
            else:
                on = min(1.0, 0.001 * max(rng.random(), 1e-15) ** (-1 / 1.5))
                peak = 10 * case["rate"]
                available = min(on, horizon - t)
                k = int(available * peak + credit)
                result.extend(int((t + (j + 1 - credit) / peak) * 1e9) for j in range(k))
                credit = available * peak + credit - k
                t += 10 * on
            if len(result) > 2_000_000:
                raise ValueError("random schedule exceeds receipt budget")
        return [value for value in result if value < int(horizon * 1e9)]
    return [offset_ns(i, case) for i in range(count)]


def producer_for(index, case):
    if case.get("skew"):
        return 0 if index % 100 < 90 else 1 + (index % 100 - 90) % (case["producers"] - 1)
    return index % case.get("producers", 1)


def payload(index, case):
    """All libraries receive the same UTF-8 body, including dynamic metadata.

    High entropy uses a frozen 256-template corpus, not an IID byte claim.
    This generation occurs outside the timed adapter call, inside offered wall time.
    """
    size = case.get("body_bytes", 64)
    mix = case.get("mix")
    if mix:
        slot = index % 1000
        size = 64 if slot < 700 else 128 if slot < 900 else 1024 if slot < 999 else 1048576
    if case.get("giant_every") and index % case["giant_every"] == 0:
        size = case["giant_bytes"]
    kind = index % case.get("cardinality", 64)
    name = f"module_{kind}_".ljust(case.get("name_bytes", 16), "m")
    if case.get("entropy") == "high":
        rng = random.Random(case["seed"] * 1009 + index % 256)
        data = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=size))
    else:
        data = chr(97 + kind % 26) * size
    body = {"name": name, "data": data}
    if case.get("depth", 0):
        nested = list(range(case.get("list_entries", 1)))
        for _ in range(case["depth"]):
            nested = {"nested": nested}
        body["metadata"] = nested
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


def logical_bytes(body):
    return 64 + len(body.encode("utf-8"))
