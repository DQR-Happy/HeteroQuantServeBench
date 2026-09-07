"""Finite, versioned CPU case expansion; exclusions are output, not hidden."""

BASE = {"duration_s": 10, "rate": 10_000, "body_bytes": 64,
        "producers": 1, "cardinality": 64, "seed": 100, "arrival": "periodic", "pacer": "cooperative"}


def cases(tier):
    rows = []

    def add(workload, label, **changes):
        rows.append(dict(BASE, id=label, workload=workload, **changes))

    add("W01", "small-1k", rate=1000)
    add("W00", "idle", events=0, warmup_events=0)
    add("W01", "small-10k")
    add("W01", "small-100k", rate=100_000)
    add("W01", "small-closed", arrival="closed", events=100_000)
    for size in (256, 1024, 4096, 16384):
        add("W02", f"body-{size}", body_bytes=size, rate=min(10_000, 8_000_000 // size))
    add("W03", "mixed", mix=True, rate=1000)
    for size in (1, 8, 31):
        add("W04", f"giant-{size}MiB", rate=1000, giant_every=5000, giant_bytes=size * 1024**2)
    add("W05", "burst-10x", arrival="burst", period_ms=10, active_ms=1)
    add("W05", "burst-100x", arrival="burst", period_ms=100, active_ms=1)
    add("W06", "poisson", arrival="poisson")
    add("W06", "onoff", arrival="onoff")
    add("W07", "hot-source", producers=8, skew=True)
    for cardinality in (1, 4096, 100_000):
        add("W08", f"names-{cardinality}", cardinality=cardinality)
    for depth in (1, 4, 16):
        add("W09", f"nested-{depth}", depth=depth, list_entries=64, rate=1000)
    for p in (2, 4, 8):
        add("W10", f"threads-{p}", producers=p)
    add("W12", "high-entropy", entropy="high", body_bytes=256)
    for batch in (1, 64, 256):
        add("W12", f"batch-{batch}", batch_events=batch)
    for chunk in (1024**2, 64 * 1024**2):
        add("W13", f"chunk-{chunk}", chunk_bytes=chunk)
    add("W14", "slow-exporter", stall_ms=100)
    add("W15", "overload-recovery", duration_s=14, phases=[
        {"duration_s": 2, "rate": 1000}, {"duration_s": 2, "rate": 10_000},
        {"duration_s": 2, "rate": 100_000}, {"duration_s": 2, "rate": 200_000},
        {"duration_s": 6, "rate": 1000}])
    add("W16", "cpu-competitor", competitor={"MiB": 16, "duty": 0.5, "placement": "producer_core"})
    add("W16", "memory-competitor", competitor={"MiB": 256, "duty": 1, "placement": "separate_core"})
    add("W22", "record-policy", max_records=1000)
    add("W27", "soak", duration_s=1800, rate=1000)
    if tier == "smoke":
        selected = {"small-1k", "small-100k", "body-4096", "mixed", "burst-100x", "threads-4", "record-policy"}
        rows = [dict(row, duration_s=2, warmup_events=10) for row in rows if row["id"] in selected]
    elif tier == "screening":
        rows = [row for row in rows if row["id"] != "soak"]
    elif tier == "formal":
        selected = {"small-1k", "small-10k", "body-4096", "burst-100x", "threads-4", "high-entropy"}
        rows = [dict(row, duration_s=60, warmup_s=10) for row in rows if row["id"] in selected]
    elif tier == "soak":
        rows = [row for row in rows if row["id"] == "soak"]
    else:
        raise ValueError(tier)
    return rows


def applicability(adapter, case):
    if adapter == "nanolog":
        largest = max(case.get("body_bytes", 64), case.get("giant_bytes", 0), 1048576 if case.get("mix") else 0)
        if largest + 4096 >= 1024**2:
            return "unsupported", "upstream per-thread staging is 1MiB; adapter does not split original records"
    if "stall_ms" in case and adapter not in ("otel", "null"):
        return "unsupported", "stock writer has no equivalent injected consumer stall; do not monkeypatch timing path"
    if "batch_events" in case and adapter not in ("otel", "null"):
        return "unsupported", "this is an OTel export-batch parameter, not a common offer_batch API"
    if "chunk_bytes" in case and adapter not in ("hqsb", "null"):
        return "unsupported", "stock library has no equivalent chunk policy"
    if "max_records" in case and adapter not in ("hqsb", "null"):
        return "unsupported", "capture-count policy is HQSB-specific"
    return None
