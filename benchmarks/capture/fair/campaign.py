"""Freeze -> source controls -> bounded tuning -> held-out matched-contract runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ADAPTERS = ["nanolog", "hindsight", "quill", "perfetto", "lttng", "dftracer", "hqsb", "otel"]


def cases():
    def c(name, rate, seconds=1, producers=1, **kw):
        return dict(id=name, workload=name, rate=rate, duration_s=seconds,
                    producers=producers, body_bytes=64, **kw)
    rows = [c("steady-10k", 10000, 2), c("steady-100k", 100000, 2),
            c("steady-300k-p4", 300000, producers=4),
            c("steady-1m-p8", 1000000, producers=8),
            c("burst-10k", 10000, 2, 8, arrival="burst", period_ms=100, active_ms=10),
            c("burst-100k", 100000, 1, 8, arrival="burst", period_ms=100, active_ms=10),
            c("poisson-50k", 50000, 2, 4, arrival="poisson"),
            c("onoff-50k", 50000, 2, 4, arrival="onoff"),
            c("skew-100k", 100000, producers=8, skew=True),
            c("snapshot-64k", 1000, 2, giant_every=100, giant_bytes=65536),
            c("mixed-records", 10000, producers=4, mix=True),
            c("metadata-entropy", 10000, 2, 4, entropy="high", cardinality=1024,
              name_bytes=128, depth=4, list_entries=16),
            c("phase-transition", 1, 2, 4, phases=[dict(duration_s=.5, rate=1000),
               dict(duration_s=.5, rate=200000), dict(duration_s=1, rate=10000)]),
            c("high-cardinality", 100000, producers=4, cardinality=10000, name_bytes=128),
            c("giant-2m", 100, giant_every=100, giant_bytes=2097152)]
    rows[11]["body_bytes"] = 1024
    return rows


def fingerprint(work):
    paths = [p for folder in (work / "source/benchmarks/capture", work / "source/hqsb/observation", work / "build")
             for p in folder.rglob("*") if p.is_file() and p.suffix in {".py", ".cc", ".h", ".so"}]
    paths += [work / "build/capture-fair", work / "build/trace_processor_shell"]
    return {str(p.relative_to(work)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work, root = args.work.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    all_cases = cases()
    eligible = {a: [1] if a == "nanolog" else [0, 1, 2] for a in ADAPTERS}
    protocol = dict(version="native-fair-v1", frozen_at=datetime.now(timezone.utc).isoformat(),
        contract="FULL logical 5-field event; per-producer order; exact body; D2 flush/close/fsync; no sampling or aggregation",
        cpus=dict(producers=[192, 194, 195], collectors=[193], observer=[196]),
        budget=dict(aggregate_AS_MiB=1024, per_thread_stack_MiB=2, MALLOC_ARENA_MAX=2,
                    lttng_AS_split_MiB=dict(application=384, sessiond=256, consumerd=256,
                                            runas_helpers=[64, 64]),
                    resident_cgroup="unavailable (read-only); sampled tree PSS reported separately"),
        source_gate="2x input rate (each phase too), 3 trials; complete attempts and <=1% of dispatches >1ms late in EVERY control; failing case excluded from performance ranking",
        tuning=dict(seeds=[0, 1], max_profiles=3, eligible=eligible,
                    selection="fewest failed/corrupt; minimum mean scheduled loss; minimum sum H+drain; minimum lifecycle CPU; minimum peak tree PSS; profile ID"),
        test_seeds=[100, 101, 102], repetitions=3,
        limitations=["1-2s stress windows are not sustained capacity certification", "CPU event path only; no GPU completion or real model probe effect", "shared host not bare metal", "D2 file fsync is not power-loss validation"],
        source_and_binary_sha256=fingerprint(work), cases=all_cases)
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (root / "eligible-profiles.json").write_text(json.dumps(eligible))

    def phase(name, rows, adapters, profiles, repeats, passes="throughput"):
        if fingerprint(work) != protocol["source_and_binary_sha256"]:
            raise RuntimeError("frozen source/build changed")
        config = root / (name + "-cases.json")
        config.write_text(json.dumps(rows, indent=2))
        cmd = [sys.executable, "-m", "benchmarks.capture.fair.run", "--work", str(work),
               "--output", str(root / name), "--cases", str(config), "--adapters", ",".join(adapters),
               "--repeats", str(repeats), "--passes", passes, "--profile-map", str(profiles)]
        subprocess.run(cmd, cwd=work / "source", check=True)
        return json.loads((root / name / "all-runs.json").read_text())

    controls = []
    for row in all_cases:
        row = dict(row, rate=row["rate"] * 2)
        if row.get("phases"):
            row["phases"] = [dict(p, rate=p["rate"] * 2) for p in row["phases"]]
        controls.append(row)
    calibration = phase("calibration", controls, ["null"], root / "eligible-profiles.json", 3)
    gates = {c["id"]: all(r.get("source_gate_pass", False) for r in calibration if r["case_id"] == c["id"])
             and sum(r["case_id"] == c["id"] for r in calibration) == 3 for c in all_cases}
    (root / "source-gates.json").write_text(json.dumps(gates, indent=2))
    # All scheduled shapes are still measured; gates determine interpretation,
    # not whether an inconvenient failure disappears from the evidence.
    training = [dict(c, id=c["id"] + f"-train{s}", workload="tuning", duration_s=.5, seed=s)
                for c in (all_cases[0], all_cases[2], all_cases[11]) for s in (0, 1)]
    tuned = phase("tuning", training, ADAPTERS, root / "eligible-profiles.json", 1)
    selected, scores = {}, {}
    for adapter in ADAPTERS:
        scores[adapter] = {}
        for p in eligible[adapter]:
            rows = [r for r in tuned if r["adapter"] == adapter and r["profile"] == p]
            failed = sum(r["status"] not in {"complete", "lossy"} for r in rows)
            usable = [r for r in rows if r["status"] in {"complete", "lossy"}]
            scores[adapter][p] = [failed, sum(1-r["scheduled_delivery_ratio"] for r in usable) / max(1, len(usable)),
                                  sum(r["H_plus_drain_s"] for r in usable),
                                  sum(r["lifecycle_children_cpu_s"] for r in usable),
                                  max((r["sampled_peak_tree_PSS_KiB"] or 0 for r in usable), default=0), p]
        selected[adapter] = [min(eligible[adapter], key=lambda p: scores[adapter][p])]
    (root / "tuning-scores.json").write_text(json.dumps(scores, indent=2))
    (root / "selected-profiles.json").write_text(json.dumps(selected, indent=2))
    phase("test", all_cases, ADAPTERS, root / "selected-profiles.json", 3)
    phase("latency", [all_cases[i] for i in (1, 3, 7)], ADAPTERS, root / "selected-profiles.json", 3, "latency")
    (root / "completed.json").write_text(json.dumps(dict(completed_at=datetime.now(timezone.utc).isoformat(),
                                                        source_and_binary_sha256=fingerprint(work)), indent=2))


if __name__ == "__main__":
    main()
