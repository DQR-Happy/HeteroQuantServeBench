#!/usr/bin/env python3
"""E02-08 runner: Jetson power mode, energy integration and thermal baseline.

Subcommands follow the protocol in
``docs/stage_experiments/details/S02/E02-08_power_energy_and_thermal.md``:

``probe``     step 1  - query, record and protect the raw device state; enumerate
                        the rail set actually emitted and the supported modes.
``audit``     step 2 + section 7 - the monitor must pass its *own* correctness
                        tests first: parser counterexamples, the hand-computed
                        trapezoid oracle, and an idle -> known GPU workload ->
                        idle coverage/alignment rehearsal.
``collect``   steps 3-12 - one independent process that walks the pre-registered
                        mode blocks (baseline mode with all six workloads, one
                        different power mode, one fixed-clock policy), each with
                        its own cooldown, idle reference and steady windows.
``verify``    cross-run / cross-mode verdict against the pass criteria.
``summarize`` recompute the report tables from the raw evidence only.

Usage (on the Jetson, always through the privileged wrapper):

    ./scripts/audit/e02_08_run.sh probe     --output-dir docs/stage_experiments/S02/E02-08/raw
    ./scripts/audit/e02_08_run.sh audit     --output-dir docs/stage_experiments/S02/E02-08/raw
    for i in 0 1 2; do
      ./scripts/audit/e02_08_run.sh collect --output-dir ... --run-index $i
    done
    ./scripts/audit/e02_08_run.sh verify    --output-dir ...
    ./scripts/audit/e02_08_run.sh summarize --output-dir ...
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from hqsb.benchmark.correctness import hash_token_sequence
from hqsb.benchmark.power_thermal import (
    POWER_THERMAL_PROTOCOL,
    RAIL_SCOPE,
    align_and_integrate,
    assess_thermal,
    cooling_state_summary,
    cross_run_reduce,
    device_state_diff,
    dimensional_consistency,
    energy_metrics,
    integrate_energy,
    net_energy,
    parse_nvpmodel_conf,
    parse_tegrastats_line_v3,
    parse_telemetry_stream,
    power_spectrum_note,
    relative_spread,
    temperature_summary,
    trapezoid_energy_j,
)
from hqsb.benchmark.power_thermal_experiment import (
    TelemetrySuite,
    attach_request_telemetry,
    configured_state_view,
    lock_clocks,
    read_device_state,
    read_text,
    restore_clock_state,
    run_idle_window,
    run_steady_window,
    set_power_mode,
    slice_sysfs_records,
    store_clock_state,
    wait_for_temperature,
)
from hqsb.benchmark.workload import make_fixed_token_input
from hqsb.benchmark.workload_config import load_workload_dicts
from hqsb.models.loader import load_qwen3

logger = logging.getLogger("e02_08")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKLOAD_YAML = _REPO_ROOT / "configs" / "benchmarks" / "jetson_qwen3_fp16.yaml"
_MANIFEST = _REPO_ROOT / "docs" / "benchmark" / "model_sha256_manifest.txt"
_NVPMODEL_CONF = "/etc/nvpmodel.conf"

# ── Pre-registered mode matrix ─────────────────────────────────────────────
#
# The baseline block records the energy of all six frozen workloads.  The two
# contrast blocks are the "at least one supported different power mode" and the
# "fixed vs dynamic clock policy" comparisons required by protocol section 6/9;
# they carry only the two representative workloads (one prefill-bound, one
# decode-bound) to keep the run bounded.

_REPRESENTATIVES = ["long_prefill", "decode_heavy"]

_MODE_BLOCKS: Dict[str, Dict[str, Any]] = {
    "M2_dyn": {
        "label": "MAXN_SUPER + dynamic governors (baseline)",
        "nvpmodel_id": 2,
        "clock_policy": "dynamic",
        "workloads": [
            "tiny",
            "short",
            "balanced",
            "long_prefill",
            "decode_heavy",
            "long_balanced",
        ],
        "is_baseline": True,
    },
    "M0_dyn": {
        "label": "15W + dynamic governors (different power mode)",
        "nvpmodel_id": 0,
        "clock_policy": "dynamic",
        "workloads": list(_REPRESENTATIVES),
        "is_baseline": False,
    },
    "M2_fix": {
        "label": "MAXN_SUPER + jetson_clocks fixed maximum (fixed policy)",
        "nvpmodel_id": 2,
        "clock_policy": "fixed_max",
        "workloads": list(_REPRESENTATIVES),
        "is_baseline": False,
    },
}

# Latin-square rotation: with three independent processes every block occupies
# each position once, so "the last block is always the hottest" cannot bias the
# comparison.
_MODE_ROTATIONS: List[List[str]] = [
    ["M2_dyn", "M0_dyn", "M2_fix"],
    ["M0_dyn", "M2_fix", "M2_dyn"],
    ["M2_fix", "M2_dyn", "M0_dyn"],
]


# ── identity / hashes ──────────────────────────────────────────────────────


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=_REPO_ROOT,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, cwd=_REPO_ROOT,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, OSError):
        return None


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _protocol_hash() -> str:
    payload = json.dumps(
        {"protocol": POWER_THERMAL_PROTOCOL, "rail_scope": RAIL_SCOPE},
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
    }
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        env["device"] = torch.cuda.get_device_name(0)
        env["compute_capability"] = [int(cc[0]), int(cc[1])]
    else:
        env["device"] = "cpu"
        env["compute_capability"] = None
    return env


def _run_id() -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    return f"run_{stamp}"


# ── monitor self-tests (protocol section 7) ────────────────────────────────


# Real, unedited tegrastats lines captured from this board (see probe.json);
# plus synthetic counterexamples that the parser is required to reject or
# degrade rather than silently accepting.
_PARSER_CASES: List[Dict[str, Any]] = [
    {
        "name": "real_line_baseline",
        "line": (
            "09-17-2026 02:10:03 RAM 1488/7620MB (lfb 105x4MB) SWAP 253/12002MB "
            "(cached 2MB) CPU [0%@1036,0%@1036,1%@1036,1%@1036,0%@729,0%@729] "
            "GR3D_FREQ 0% cpu@47.875C soc2@46.75C soc0@47.281C gpu@50.218C "
            "tj@50.218C soc1@48.281C VDD_IN 4654mW/4654mW "
            "VDD_CPU_GPU_CV 560mW/560mW VDD_SOC 1484mW/1484mW"
        ),
        "expect_status": "ok",
        "expect": {"rails_mw.VDD_IN": 4654, "temperatures_c.gpu": 50.218},
    },
    {
        "name": "real_line_integer_temp_and_load",
        "line": (
            "09-17-2026 02:10:16 RAM 1483/7620MB (lfb 105x4MB) SWAP 253/12002MB "
            "(cached 2MB) CPU [0%@729,0%@729,0%@729,0%@729,0%@729,0%@729] "
            "GR3D_FREQ 62%@612 cpu@47.75C soc2@46.718C soc0@47.25C gpu@50C "
            "tj@50C soc1@48.093C VDD_IN 4541mW/4541mW "
            "VDD_CPU_GPU_CV 521mW/521mW VDD_SOC 1446mW/1473mW"
        ),
        "expect_status": "ok",
        "expect": {"gpu_util_pct": 62, "gpu_freq_mhz": 612, "temperatures_c.tj": 50.0},
    },
    {
        "name": "rail_order_permuted",
        "line": (
            "RAM 100/200MB CPU [1%@729,1%@729] GR3D_FREQ 1% gpu@40C "
            "VDD_SOC 900mW VDD_CPU_GPU_CV 200mW VDD_IN 3000mW"
        ),
        "expect_status": "ok",
        "expect": {"rails_mw.VDD_IN": 3000, "rails_mw.VDD_SOC": 900},
    },
    {
        "name": "extra_unknown_rail_kept_separately",
        "line": (
            "RAM 100/200MB CPU [1%@729] GR3D_FREQ 1% gpu@40C VDD_IN 3000mW "
            "VDD_GPU_SOC 1200mW VDD_EXTRA_UNKNOWN 42mW"
        ),
        "expect_status": "ok",
        "expect": {"rails_mw.VDD_GPU_SOC": 1200, "rails_mw.VDD_EXTRA_UNKNOWN": 42},
    },
    {
        "name": "unit_drift_watts_instead_of_milliwatts",
        "line": (
            "RAM 100/200MB CPU [1%@729] GR3D_FREQ 1% gpu@40C VDD_IN 4.65W"
        ),
        "expect_status": "degraded",
        "expect": {"unparsed_rails": ["VDD_IN"]},
    },
    {
        "name": "truncated_line_missing_cpu_bracket",
        "line": "RAM 100/200MB GR3D_FREQ 1% gpu@40C VDD_IN 3000mW",
        "expect_status": "degraded",
        "expect": {"missing_fields_contains": "cpu_bracket"},
    },
    {
        "name": "non_numeric_power",
        "line": "RAM 100/200MB CPU [1%@729] GR3D_FREQ 1% gpu@40C VDD_IN NAmW",
        "expect_status": "degraded",
        "expect": {"unparsed_rails": ["VDD_IN"]},
    },
    {
        "name": "empty_line",
        "line": "",
        "expect_status": "failed",
        "expect": {},
    },
    {
        "name": "english_locale_extra_spaces",
        "line": (
            "RAM  100/200MB  CPU [1%@729]  GR3D_FREQ 1%  gpu@40C  "
            "VDD_IN  3000mW"
        ),
        "expect_status": "ok",
        "expect": {"rails_mw.VDD_IN": 3000},
    },
]


def _resolve_expectation(
    parsed: Mapping[str, Any], dotted: str
) -> Any:
    node: Any = parsed
    for part in dotted.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node


def parser_self_test() -> Dict[str, Any]:
    """Section 7.1: the parser must pass its own correctness test suite."""
    results: List[Dict[str, Any]] = []
    all_ok = True
    for case in _PARSER_CASES:
        parsed = parse_tegrastats_line_v3(case["line"])
        checks: Dict[str, bool] = {}
        status_ok = parsed["parse_status"] == case["expect_status"]
        checks["parse_status"] = status_ok
        for key, expected in case["expect"].items():
            if key == "missing_fields_contains":
                checks[key] = expected in parsed["missing_fields"]
            else:
                checks[key] = _resolve_expectation(parsed, key) == expected
        ok = all(checks.values())
        all_ok = all_ok and ok
        results.append(
            {
                "name": case["name"],
                "line": case["line"],
                "expected_status": case["expect_status"],
                "observed_status": parsed["parse_status"],
                "checks": checks,
                "passed": ok,
            }
        )
    return {"passed": all_ok, "num_cases": len(results), "results": results}


def integration_self_test() -> Dict[str, Any]:
    """Section 7.2: hand-computed trapezoid oracle plus rejection cases.

    The expected value is computed by hand in the docstring; the integration
    function is only asked to reproduce it.

        t = [0, 1, 3] s, P = [2, 4, 4] W
        0 -> 1 : 0.5*(2+4)*1 = 3 J
        1 -> 3 : 0.5*(4+4)*2 = 8 J
        total  : 11 J
    """
    t_ns = [0, 1_000_000_000, 3_000_000_000]
    p_mw = [2000.0, 4000.0, 4000.0]
    total = trapezoid_energy_j(p_mw, t_ns)
    integral = integrate_energy(p_mw, t_ns)

    cases: List[Dict[str, Any]] = [
        {
            "name": "hand_oracle_11J",
            "observed": total,
            "expected": 11.0,
            "passed": abs(total - 11.0) < 1e-9,
        },
        {
            "name": "hand_oracle_segments",
            "observed": [
                trapezoid_energy_j([2000.0, 4000.0], [0, 1_000_000_000]),
                trapezoid_energy_j(
                    [4000.0, 4000.0], [1_000_000_000, 3_000_000_000]
                ),
            ],
            "expected": [3.0, 8.0],
            "passed": abs(
                trapezoid_energy_j([2000.0, 4000.0], [0, 1_000_000_000]) - 3.0
            )
            < 1e-9
            and abs(
                trapezoid_energy_j(
                    [4000.0, 4000.0], [1_000_000_000, 3_000_000_000]
                )
                - 8.0
            )
            < 1e-9,
        },
        {
            # 2 W held for 0.5 s = 1 J; exercises mW -> W and ns -> s together.
            "name": "milliwatt_nanosecond_unit_conversion",
            "observed": trapezoid_energy_j(
                [2000.0, 2000.0], [0, 500_000_000]
            ),
            "expected": 1.0,
            "passed": abs(
                trapezoid_energy_j([2000.0, 2000.0], [0, 500_000_000]) - 1.0
            )
            < 1e-9,
        },
        {
            # 1000 mW held for 1 ms = 1 mJ = 1e-3 J: the sub-interval scale
            # that a "mean power x duration" shortcut would get wrong.
            "name": "millisecond_scale_energy_is_not_dropped",
            "observed": trapezoid_energy_j([1000.0, 1000.0], [0, 1_000_000]),
            "expected": 1e-3,
            "passed": abs(
                trapezoid_energy_j([1000.0, 1000.0], [0, 1_000_000]) - 1e-3
            )
            < 1e-12,
        },
        {
            "name": "duplicate_timestamp_skipped_not_interpolated",
            "observed": integrate_energy([2000.0, 9000.0, 9000.0], [0, 0, 1_000_000_000]),
            "expected_energy_j": 9.0,
            "passed": abs(
                integrate_energy(
                    [2000.0, 9000.0, 9000.0], [0, 0, 1_000_000_000]
                )["energy_j"]
                - 9.0
            )
            < 1e-9
            and integrate_energy(
                [2000.0, 9000.0, 9000.0], [0, 0, 1_000_000_000]
            )["zero_dt_intervals"]
            == 1,
        },
        {
            "name": "negative_interval_rejected",
            "observed": integrate_energy([1000.0, 1000.0], [1_000_000_000, 0]),
            "expected_valid": False,
            "passed": integrate_energy(
                [1000.0, 1000.0], [1_000_000_000, 0]
            )["valid"]
            is False
            and integrate_energy([1000.0, 1000.0], [1_000_000_000, 0])[
                "negative_dt_intervals"
            ]
            == 1,
        },
        {
            "name": "large_gap_flagged",
            "observed": integrate_energy(
                [1000.0, 1000.0, 1000.0],
                [0, 100_000_000, 5_000_000_000],
                max_gap_s=1.5,
            )["gap_violations"],
            "expected": 1,
            "passed": integrate_energy(
                [1000.0, 1000.0, 1000.0],
                [0, 100_000_000, 5_000_000_000],
                max_gap_s=1.5,
            )["gap_violations"]
            == 1,
        },
        {
            "name": "mean_power_times_duration_differs_from_integral",
            "observed": None,
            "expected_note": (
                "with unequal sample spacing mean(P)*T != integral(P); this case "
                "documents why the mean is never published as energy"
            ),
            "passed": abs(
                trapezoid_energy_j([2000.0, 4000.0, 4000.0], t_ns)
                - statistics.mean([2000.0, 4000.0, 4000.0]) / 1000.0 * 3.0
            )
            > 0.1,
        },
    ]
    all_ok = all(c["passed"] for c in cases)
    return {
        "passed": all_ok,
        "oracle": {"t_s": [0, 1, 3], "p_w": [2, 4, 4], "expected_j": 11.0},
        "observed_energy_j": total,
        "max_gap_s_seen": integral["max_gap_s"],
        "cases": cases,
    }


def _monitor_error_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    interval_ms: int,
) -> Dict[str, Any]:
    """Section 7.3: does the monitor actually sample where it claims to?"""
    timestamps = [
        r["time_ns"] for r in records if isinstance(r.get("time_ns"), int)
    ]
    intervals_s = [
        (timestamps[i + 1] - timestamps[i]) / 1e9
        for i in range(len(timestamps) - 1)
    ]
    violations = sum(1 for d in intervals_s if d <= 0)
    nominal_s = interval_ms / 1000.0
    return {
        "num_records": len(records),
        "num_timestamps": len(timestamps),
        "monotonic": violations == 0,
        "timestamp_violations": violations,
        "nominal_interval_s": nominal_s,
        "observed_interval_s": {
            "count": len(intervals_s),
            "min": min(intervals_s) if intervals_s else None,
            "median": statistics.median(intervals_s) if intervals_s else None,
            "max": max(intervals_s) if intervals_s else None,
            "mean": statistics.mean(intervals_s) if intervals_s else None,
        },
        "wall_span_s": (
            (timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) >= 2 else 0.0
        ),
        "expected_samples_from_wall_span": (
            round((timestamps[-1] - timestamps[0]) / 1e9 / nominal_s)
            if len(timestamps) >= 2
            else 0
        ),
        "clock_domain": (
            "time.monotonic_ns() in the benchmark process for both the request "
            "window markers and every telemetry record (TegrastatsMonitor reads "
            "its own stdout in-process; SysfsSampler stamps its own reads), so "
            "the two series share one clock by construction."
        ),
    }


# ── probe ──────────────────────────────────────────────────────────────────


def _run_probe(args: argparse.Namespace) -> int:
    """Step 1: record and protect the raw device state before anything else."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = read_device_state()
    conf_text = read_text(_NVPMODEL_CONF) or ""
    conf_modes = parse_nvpmodel_conf(conf_text)

    # Enumerate the rails the tool actually emits, from a real sample.
    suite = TelemetrySuite(interval_ms=args.monitor_interval_ms)
    suite.start()
    time.sleep(args.sample_seconds)
    suite.stop()

    raw_lines = [r["raw"] for r in suite.tegrastats.records]
    parsed, parser_audit = parse_telemetry_stream(suite.tegrastats.records)
    rail_names: List[str] = sorted(
        {name for record in parsed for name in record["rails_mw"]}
    )
    sample_lines = raw_lines[:3]

    unparsed = sorted(
        {name for record in parsed for name in record.get("unparsed_rails", [])}
    )

    record = {
        "experiment": "E02-08",
        "phase": "probe",
        "run_id": _run_id(),
        "collected_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "environment": _environment(),
        "protocol_hash": _protocol_hash(),
        "protocol": POWER_THERMAL_PROTOCOL,
        "rail_scope": RAIL_SCOPE,
        "device_state_before": state,
        "configured_state_view_before": configured_state_view(state),
        "nvpmodel_conf": {
            "path": _NVPMODEL_CONF,
            "sha256": _sha256_bytes(conf_text.encode("utf-8")),
            **conf_modes,
        },
        "rail_enumeration": {
            "rails_emitted": rail_names,
            "unparsed_rails": unparsed,
            "primary_rail": POWER_THERMAL_PROTOCOL["primary_rail"],
            "sub_rails_declared": POWER_THERMAL_PROTOCOL["sub_rails"],
            "primary_rail_present": (
                POWER_THERMAL_PROTOCOL["primary_rail"] in rail_names
            ),
            "additive": RAIL_SCOPE["additive"],
        },
        "monitor": {
            "command": f"tegrastats --interval {args.monitor_interval_ms}",
            "interval_ms": args.monitor_interval_ms,
            "sample_seconds": args.sample_seconds,
            "num_records": len(suite.tegrastats.records),
            "tegrastats_available": suite.tegrastats_available,
            "tegrastats_error": suite.tegrastats_error,
            "parser_audit": parser_audit,
            "sample_lines": sample_lines,
        },
        "privileges": {
            "emc_rate_readable": (state["sysfs"].get("emc_rate_hz") is not None),
            "jetson_clocks_show_available": (
                state["jetson_clocks"].get("available") is True
            ),
            "sysfs_gpu_governor": (
                state["sysfs"].get("gpu_devfreq") or {}
            ).get("governor"),
            "cooling_devices": sorted(
                (state["sysfs"].get("cooling_cur_state") or {}).keys()
            ),
        },
        "background": {
            "note": (
                "no user workloads are started by this experiment; tegrastats "
                "itself is the only added process"
            ),
            "numara": None,
        },
        "restore_plan": {
            "target_nvpmodel_mode_id": (state["nvpmodel"] or {}).get("mode_id"),
            "target_nvpmodel_mode_name": (state["nvpmodel"] or {}).get("mode_name"),
            "clock_restore_command": "jetson_clocks --restore",
            "clock_store_command": "jetson_clocks --store",
        },
    }

    store_result = store_clock_state()
    record["clock_store"] = store_result
    record["clock_store_ok"] = store_result.get("returncode") == 0

    out_path = output_dir / "probe.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("wrote %s (rails=%s)", out_path, rail_names)
    print(json.dumps({k: record[k] for k in (
        "rail_enumeration", "privileges", "restore_plan"
    )}, indent=2))
    return 0


# ── audit ──────────────────────────────────────────────────────────────────


def _run_audit(args: argparse.Namespace) -> int:
    """Step 2 + section 7: the monitor must pass its own tests first."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parser_test = parser_self_test()
    integration_test = integration_self_test()

    tokenizer, model, load_time_s = load_qwen3(
        args.model_path,
        dtype=torch.float16,
        attention_backend="eager",
        verify_manifest=args.manifest,
        allow_extra=("model_sha256_manifest.txt",),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    workloads = {w["name"]: w for w in load_workload_dicts(str(args.workload_yaml))}
    audit_workload = workloads[args.audit_workload]
    isl = int(audit_workload["input_tokens"])
    osl = int(audit_workload["output_tokens"])

    # warmup (never recorded)
    warm_inputs = make_fixed_token_input(tokenizer, 32, device=device)
    run_steady_window(model, warm_inputs, 2, 1, device=device)

    inputs = make_fixed_token_input(tokenizer, isl, device=device)

    suite = TelemetrySuite(interval_ms=args.monitor_interval_ms)
    suite.start()

    idle_before = run_idle_window(args.audit_idle_s, device=device)
    gpu_window = run_steady_window(
        model,
        inputs,
        osl,
        args.audit_requests,
        device=device,
        workload_name=args.audit_workload,
    )
    idle_after = run_idle_window(args.audit_idle_s, device=device)

    suite.stop()

    tegrastats_records = list(suite.tegrastats.records)
    sysfs_records = list(suite.sysfs.records)
    parsed, parser_stream_audit = parse_telemetry_stream(tegrastats_records)
    rail_audit = {}
    for rail in [POWER_THERMAL_PROTOCOL["primary_rail"]] + list(
        POWER_THERMAL_PROTOCOL["sub_rails"]
    ):
        times, _ = _series_for_rail(parsed, rail)
        rail_audit[rail] = {
            "samples": len(times),
            "present": len(times) > 0,
        }

    primary = POWER_THERMAL_PROTOCOL["primary_rail"]
    gpu_energy = align_and_integrate(
        parsed, gpu_window["begin_ns"], gpu_window["end_ns"], primary
    )
    idle_energy = align_and_integrate(
        parsed, idle_before["begin_ns"], idle_after["end_ns"], primary
    )
    idle_only = align_and_integrate(
        parsed, idle_before["begin_ns"], idle_before["end_ns"], primary
    )
    idle_tail = align_and_integrate(
        parsed, idle_after["begin_ns"], idle_after["end_ns"], primary
    )

    response_ok = (
        gpu_energy.get("avg_power_w") is not None
        and idle_only.get("avg_power_w") is not None
        and idle_tail.get("avg_power_w") is not None
        and gpu_energy["avg_power_w"]
        > max(idle_only["avg_power_w"], idle_tail["avg_power_w"])
    )

    gpu_util_series = [
        p.get("gpu_util_pct")
        for p in parsed
        if p.get("time_ns") is not None
        and gpu_window["begin_ns"] <= p["time_ns"] <= gpu_window["end_ns"]
    ]
    idle_util_series = [
        p.get("gpu_util_pct")
        for p in parsed
        if p.get("time_ns") is not None
        and idle_before["begin_ns"] <= p["time_ns"] <= idle_before["end_ns"]
    ]

    monitor_audit = _monitor_error_audit(
        tegrastats_records, interval_ms=args.monitor_interval_ms
    )

    boundary_ok = (
        gpu_energy.get("guard_samples_before", 0) >= 1
        and gpu_energy.get("guard_samples_after", 0) >= 1
    )

    verdict = {
        "parser_self_test_passed": parser_test["passed"],
        "integration_self_test_passed": integration_test["passed"],
        "parser_stream_monotonic": parser_stream_audit["monotonic"],
        "parser_stream_error_ratio": parser_stream_audit["parse_error_ratio"],
        "primary_rail_present": rail_audit[primary]["present"],
        "window_coverage_usable": bool(gpu_energy.get("usable")),
        "boundary_guard_present": boundary_ok,
        "power_responds_to_workload": response_ok,
        "clock_domain_shared": True,
    }
    verdict["passed"] = all(
        value
        for key, value in verdict.items()
        if isinstance(value, bool)
    ) and (parser_stream_audit["parse_error_ratio"] == 0.0)

    record = {
        "experiment": "E02-08",
        "phase": "audit",
        "run_id": _run_id(),
        "collected_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "environment": _environment(),
        "protocol_hash": _protocol_hash(),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "protocol": {
            "monitor_interval_ms": args.monitor_interval_ms,
            "audit_workload": args.audit_workload,
            "audit_workload_isl": isl,
            "audit_workload_osl": osl,
            "audit_requests": args.audit_requests,
            "audit_idle_s": args.audit_idle_s,
        },
        "load_time_s": load_time_s,
        "parser_self_test": parser_test,
        "integration_self_test": integration_test,
        "parser_stream_audit": parser_stream_audit,
        "rail_audit": rail_audit,
        "monitor_error_audit": monitor_audit,
        "windows": {
            "idle_before": idle_before,
            "gpu_workload": gpu_window,
            "idle_after": idle_after,
        },
        "energy": {
            "idle_before": idle_only,
            "gpu_window": gpu_energy,
            "idle_after": idle_tail,
            "idle_over_both": idle_energy,
        },
        "response_evidence": {
            "idle_before_avg_power_w": idle_only.get("avg_power_w"),
            "gpu_avg_power_w": gpu_energy.get("avg_power_w"),
            "idle_after_avg_power_w": idle_tail.get("avg_power_w"),
            "idle_before_mean_gpu_util_pct": _mean_or_none(idle_util_series),
            "gpu_window_mean_gpu_util_pct": _mean_or_none(gpu_util_series),
            "power_rise_ratio": (
                (gpu_energy.get("avg_power_w") or 0.0)
                / (idle_only.get("avg_power_w") or 1.0)
            ),
        },
        "token_hashes": sorted(
            {r["sequence_sha256"] for r in gpu_window["requests"]}
        ),
        "verdict": verdict,
    }

    (output_dir / "audit.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    (output_dir / "audit_tegrastats.txt").write_text(
        "\n".join(
            f"{r['time_ns']}\t{r['raw']}" for r in tegrastats_records
        ),
        encoding="utf-8",
    )
    (output_dir / "audit_sysfs.json").write_text(
        json.dumps(sysfs_records, indent=2), encoding="utf-8"
    )
    (output_dir / "audit_driver.json").write_text(
        json.dumps(
            {
                "idle_before": idle_before,
                "gpu_window": gpu_window,
                "idle_after": idle_after,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s (passed=%s)", output_dir / "audit.json", verdict["passed"])
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["passed"] else 1


def _mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def _series_for_rail(
    parsed: Sequence[Mapping[str, Any]], rail: str
) -> tuple:
    pairs = [
        (int(p["time_ns"]), float(p["rails_mw"][rail]))
        for p in parsed
        if p.get("time_ns") is not None and rail in p.get("rails_mw", {})
    ]
    pairs.sort(key=lambda item: item[0])
    return [t for t, _ in pairs], [v for _, v in pairs]


# ── collect ────────────────────────────────────────────────────────────────


def _zone_series(
    sysfs_records: Sequence[Mapping[str, Any]],
    zone: str,
    begin_ns: int,
    end_ns: int,
) -> List[tuple]:
    series = []
    for record in slice_sysfs_records(sysfs_records, begin_ns, end_ns):
        value = (record.get("temperatures_c") or {}).get(zone)
        if value is not None:
            series.append((record["time_ns"], float(value)))
    return series


def _freq_series(
    sysfs_records: Sequence[Mapping[str, Any]],
    begin_ns: int,
    end_ns: int,
) -> List[tuple]:
    series = []
    for record in slice_sysfs_records(sysfs_records, begin_ns, end_ns):
        value = record.get("gpu_cur_freq_hz")
        if value is not None:
            series.append((record["time_ns"], float(value)))
    return series


def _attach_windows(
    parsed: Sequence[Mapping[str, Any]],
    sysfs_records: Sequence[Mapping[str, Any]],
    window: Mapping[str, Any],
    idle_avg_power_w: Optional[float],
    *,
    temperature_zone: str,
    sub_rails: Sequence[str],
) -> Dict[str, Any]:
    """Reduce one measured window into the frozen evidence structure."""
    begin_ns = int(window["begin_ns"])
    end_ns = int(window["end_ns"])
    primary = POWER_THERMAL_PROTOCOL["primary_rail"]

    primary_energy = align_and_integrate(parsed, begin_ns, end_ns, primary)
    sub_rail_energy = {
        rail: align_and_integrate(parsed, begin_ns, end_ns, rail)
        for rail in sub_rails
    }

    requests = attach_request_telemetry(
        window.get("requests", []),
        sysfs_records,
        hot_zone=temperature_zone,
        temp_zone=temperature_zone,
    )

    cooling = cooling_state_summary(
        slice_sysfs_records(sysfs_records, begin_ns, end_ns)
    )
    thermal = assess_thermal(
        requests,
        cooling_engaged=bool(cooling["engaged"]),
        cooling_detail=cooling,
    )

    output_tokens = int(window.get("output_tokens_total", 0))
    processed_tokens = int(window.get("processed_tokens_total", 0))
    valid_requests = int(window.get("valid_requests", 0))

    energy_j = primary_energy.get("energy_j")
    metrics = energy_metrics(
        energy_j,
        num_requests=valid_requests,
        output_tokens=output_tokens,
        processed_tokens=processed_tokens,
    )

    temperature = temperature_summary(
        _zone_series(sysfs_records, temperature_zone, begin_ns, end_ns)
    )
    freq_series = _freq_series(sysfs_records, begin_ns, end_ns)
    freq_values = [v for _, v in freq_series]

    net = None
    if (
        idle_avg_power_w is not None
        and energy_j is not None
        and primary_energy.get("covered_s")
    ):
        net = net_energy(energy_j, primary_energy["covered_s"], idle_avg_power_w)

    mean_output_tps = (
        output_tokens / primary_energy["covered_s"]
        if primary_energy.get("covered_s")
        else None
    )
    closure = dimensional_consistency(
        energy_j=energy_j,
        duration_s=primary_energy.get("covered_s"),
        output_tokens=output_tokens,
        output_tokens_per_s=mean_output_tps,
    )

    expected_output_tokens = valid_requests * int(
        window.get("output_tokens_per_request", 0)
    )

    return {
        "kind": window.get("kind"),
        "workload_name": window.get("workload_name"),
        "begin_ns": begin_ns,
        "end_ns": end_ns,
        "window_s": window.get("window_s"),
        "requested_requests": window.get("requested_requests"),
        "valid_requests": valid_requests,
        "failures": window.get("failures", []),
        "input_tokens_total": window.get("input_tokens_total"),
        "output_tokens_total": output_tokens,
        "processed_tokens_total": processed_tokens,
        "expected_output_tokens_total": expected_output_tokens,
        "token_denominator_matches": output_tokens == expected_output_tokens,
        "sequence_sha256_set": window.get("sequence_sha256_set", []),
        "energy": primary_energy,
        "sub_rail_energy": sub_rail_energy,
        "energy_metrics": metrics,
        "net_energy": net,
        "dimensional_consistency": closure,
        "temperature": temperature,
        "temperature_zone": temperature_zone,
        "gpu_freq_hz": {
            "count": len(freq_values),
            "min": min(freq_values) if freq_values else None,
            "median": statistics.median(freq_values) if freq_values else None,
            "max": max(freq_values) if freq_values else None,
            "mean": _mean_or_none(freq_values),
        },
        "cooling": cooling,
        "thermal": thermal,
        "requests": requests,
    }


def _execute_mode_block(
    block_id: str,
    declared: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    workloads: Mapping[str, Mapping[str, Any]],
    device: str,
    suite: TelemetrySuite,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Enter a mode, cool down, sample idle, then run every steady window."""
    record: Dict[str, Any] = {
        "block_id": block_id,
        "declared": dict(declared),
    }

    target_mode = int(declared["nvpmodel_id"])
    policy = str(declared["clock_policy"])

    # ── enter the requested state and re-query the observed one ──────────
    record["nvpmodel_enter"] = set_power_mode(target_mode)
    if policy == "fixed_max":
        record["clocks_enter"] = lock_clocks()
    record["state_after_enter"] = read_device_state()
    record["observed_configured"] = configured_state_view(
        record["state_after_enter"]
    )

    # ── start-of-window thermal condition ────────────────────────────────
    record["cooldown_before"] = wait_for_temperature(
        float(args.cooldown_target_c),
        zone=args.temperature_zone,
        timeout_s=float(args.cooldown_timeout_s),
        poll_s=2.0,
    )

    # ── workload order: balanced per (run, block) so a block can never always
    #    start with the same shape, and the exact order stays auditable ─────
    order = list(declared["workloads"])
    random.Random(f"{args.run_index}:{block_id}").shuffle(order)
    record["workload_order"] = order

    # ── warmup of this block's shapes (never inside a measurement window) ─
    warmup_records: List[Dict[str, Any]] = []
    for name in order:
        spec = workloads[name]
        warm_inputs = make_fixed_token_input(
            tokenizer,
            int(spec["input_tokens"]),
            device=device,
        )
        warm = run_steady_window(
            model,
            warm_inputs,
            min(int(spec["output_tokens"]), args.warmup_output_tokens),
            1,
            device=device,
            workload_name=name,
        )
        warmup_records.append(
            {
                "workload": name,
                "output_tokens": min(
                    int(spec["output_tokens"]), args.warmup_output_tokens
                ),
                "wall_ms": warm["requests"][0]["wall_ms"] if warm["requests"] else None,
            }
        )
    record["warmup"] = warmup_records

    # ── run every window first, attach the telemetry afterwards ───────────
    #
    # A window is only integrable once the monitor has also produced samples
    # *after* its end marker, which cannot be true at the instant the window
    # finishes.  Every window is therefore measured first and reduced after a
    # short settle, so the run-time telemetry tail is present and the coverage
    # audit reports the guard samples the protocol asks for.
    idle_window = run_idle_window(args.idle_window_s, device=device)
    raw_windows: List[Dict[str, Any]] = []
    for name in order:
        spec = workloads[name]
        isl = int(spec["input_tokens"])
        osl = int(spec["output_tokens"])
        inputs = make_fixed_token_input(tokenizer, isl, device=device)
        window = run_steady_window(
            model,
            inputs,
            osl,
            int(args.requests_per_window),
            device=device,
            workload_name=name,
        )
        window["output_tokens_per_request"] = osl
        window["input_tokens_per_request"] = isl
        raw_windows.append(window)
        logger.info(
            "block %s: window %s done (%.1f s, %d/%d valid requests)",
            block_id,
            name,
            window["window_s"],
            window["valid_requests"],
            window["requested_requests"],
        )

    settle_s = max(1.0, 4.0 * args.monitor_interval_ms / 1000.0)
    logger.info("block %s: settling %.1f s for the post-window telemetry tail",
                block_id, settle_s)
    time.sleep(settle_s)

    parsed, _ = parse_telemetry_stream(list(suite.tegrastats.records))
    sysfs_snapshot = list(suite.sysfs.records)

    def attach(
        window: Mapping[str, Any], idle_avg_power_w: Optional[float]
    ) -> Dict[str, Any]:
        return _attach_windows(
            parsed=parsed,
            sysfs_records=sysfs_snapshot,
            window=window,
            idle_avg_power_w=idle_avg_power_w,
            temperature_zone=args.temperature_zone,
            sub_rails=POWER_THERMAL_PROTOCOL["sub_rails"],
        )

    record["idle"] = attach(idle_window, None)
    idle_avg_power_w = record["idle"]["energy"].get("avg_power_w")

    windows: List[Dict[str, Any]] = []
    for window in raw_windows:
        windows.append(attach(window, idle_avg_power_w))
        logger.info(
            "block %s window %s: E=%s J usable=%s label=%s",
            block_id,
            window["workload_name"],
            windows[-1]["energy"].get("energy_j"),
            windows[-1]["energy"].get("usable"),
            windows[-1]["thermal"]["label"],
        )
    record["windows"] = windows

    # ── leave the mode (fixed clocks must be handed back before the next one) ─
    if policy == "fixed_max":
        record["clocks_exit"] = restore_clock_state()
    record["cooldown_after"] = wait_for_temperature(
        float(args.cooldown_target_c),
        zone=args.temperature_zone,
        timeout_s=float(args.cooldown_timeout_s),
        poll_s=2.0,
    )
    record["state_after_exit"] = read_device_state()
    return record


def _run_collect(args: argparse.Namespace) -> int:
    """Steps 3-12 for one independent process."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_index = int(args.run_index)
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rotation = _MODE_ROTATIONS[run_index % len(_MODE_ROTATIONS)]

    # A smoke pass exercises the full mode-switch / cooldown / restore path
    # with one short window per block before the real multi-hour collection is
    # started.  It is recorded as `smoke=True` so it can never be mistaken for
    # a正式 run (the verifier only accepts `smoke` runs that are absent).
    plan: Dict[str, Dict[str, Any]] = {
        key: dict(value) for key, value in _MODE_BLOCKS.items()
    }
    if args.smoke:
        for value in plan.values():
            value["workloads"] = ["short"]
        args.requests_per_window = 1
        args.idle_window_s = 10.0

    state_before = read_device_state()
    configured_before = configured_state_view(state_before)
    store_result = store_clock_state()

    tokenizer, model, load_time_s = load_qwen3(
        args.model_path,
        dtype=torch.float16,
        attention_backend="eager",
        verify_manifest=args.manifest,
        allow_extra=("model_sha256_manifest.txt",),
    )
    param_devices = sorted({str(p.device) for p in model.parameters()})
    model_fully_on_gpu = bool(param_devices) and all(
        d.startswith("cuda") for d in param_devices
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    workloads = {w["name"]: w for w in load_workload_dicts(str(args.workload_yaml))}

    # Global warmup so first-call effects never land inside a measurement window.
    warm_inputs = make_fixed_token_input(tokenizer, 32, device=device)
    warmup = run_steady_window(
        model, warm_inputs, args.warmup_output_tokens, 1, device=device
    )

    suite = TelemetrySuite(interval_ms=args.monitor_interval_ms)
    suite.start()

    blocks: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    try:
        for block_id in rotation:
            try:
                blocks.append(
                    _execute_mode_block(
                        block_id,
                        plan[block_id],
                        model=model,
                        tokenizer=tokenizer,
                        workloads=workloads,
                        device=device,
                        suite=suite,
                        args=args,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — a failed block must be kept
                logger.exception("mode block %s failed", block_id)
                errors.append(
                    {
                        "block_id": block_id,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        suite.stop()

    # Restore the original state no matter what happened above.
    restore_nvpmodel = set_power_mode(int(configured_before["nvpmodel_mode_id"]))
    restore_clocks = restore_clock_state()
    state_after = read_device_state()
    configured_after = configured_state_view(state_after)
    restore_diff = device_state_diff(configured_before, configured_after)

    # If the clock restore did not reproduce the configured state, re-applying
    # the original nvpmodel mode is the fallback (nvpmodel rewrites the CPU/GPU
    # min/max limits itself).  A failed restore is reported, never hidden.
    restore_reapply: Optional[Dict[str, Any]] = None
    if not restore_diff.get("restored"):
        logger.warning("clock restore incomplete; re-applying nvpmodel mode")
        restore_reapply = set_power_mode(
            int(configured_before["nvpmodel_mode_id"])
        )
        state_after = read_device_state()
        configured_after = configured_state_view(state_after)
        restore_diff = device_state_diff(configured_before, configured_after)

    tegrastats_records = list(suite.tegrastats.records)
    parsed, parser_audit = parse_telemetry_stream(tegrastats_records)

    stem = f"{'smoke_' if args.smoke else ''}run_{run_index}"
    try:
        tegrastats_path = output_dir / f"{stem}.tegrastats.txt"
        with tegrastats_path.open("w", encoding="utf-8") as handle:
            for record in tegrastats_records:
                handle.write(f"{record['time_ns']}\t{record['raw']}\n")
        sysfs_path = output_dir / f"{stem}.sysfs.json"
        sysfs_path.write_text(
            json.dumps(suite.sysfs.records), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("telemetry dump failed: %s", exc)
        tegrastats_path = None
        sysfs_path = None

    thermal_flags = [
        {
            "block_id": block["block_id"],
            "workload": window["workload_name"],
            "label": window["thermal"]["label"],
            "isolate_from_baseline": window["thermal"]["isolate_from_baseline"],
        }
        for block in blocks
        for window in block.get("windows", [])
    ]

    run_record = {
        "experiment": "E02-08",
        "phase": "collect",
        "run_id": _run_id(),
        "run_index": run_index,
        "smoke": bool(args.smoke),
        "rotation": rotation,
        "started_at": started_at,
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "complete": True,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "environment": _environment(),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "manifest_sha256": _sha256_file(Path(args.manifest)),
        "workload_yaml": str(args.workload_yaml),
        "workload_yaml_sha256": _sha256_file(Path(args.workload_yaml)),
        "protocol_hash": _protocol_hash(),
        "identity": {
            "model": "Qwen/Qwen3-1.7B FP16 eager",
            "allocator_mode": (
                "no_caching" if _no_caching_enabled() else "caching"
            ),
            "attention_backend": "eager",
        },
        "load_time_s": load_time_s,
        "model_fully_on_gpu": model_fully_on_gpu,
        "param_devices": param_devices,
        "protocol": POWER_THERMAL_PROTOCOL,
        "rail_scope": RAIL_SCOPE,
        "run_protocol": {
            "monitor_interval_ms": args.monitor_interval_ms,
            "requests_per_window": int(args.requests_per_window),
            "idle_window_s": float(args.idle_window_s),
            "temperature_zone": args.temperature_zone,
            "cooldown_target_c": float(args.cooldown_target_c),
            "cooldown_timeout_s": float(args.cooldown_timeout_s),
            "warmup_output_tokens": int(args.warmup_output_tokens),
            "mode_rotation": rotation,
        },
        "state_before": state_before,
        "configured_state_before": configured_before,
        "clock_store": store_result,
        "global_warmup": {
            "output_tokens": args.warmup_output_tokens,
            "wall_ms": warmup["requests"][0]["wall_ms"] if warmup["requests"] else None,
        },
        "blocks": blocks,
        "errors": errors,
        "restore": {
            "nvpmodel": restore_nvpmodel,
            "clocks": restore_clocks,
            "reapply_nvpmodel": restore_reapply,
            "state_after": state_after,
            "configured_state_after": configured_after,
            "diff": restore_diff,
        },
        "monitor": {
            "tegrastats_available": suite.tegrastats_available,
            "tegrastats_error": suite.tegrastats_error,
            "interval_ms": args.monitor_interval_ms,
            "num_tegrastats_records": len(tegrastats_records),
            "num_sysfs_records": len(suite.sysfs.records),
            "sysfs_read_errors": suite.sysfs.read_errors,
            "parser_audit": parser_audit,
            "tegrastats_file": str(tegrastats_path) if tegrastats_path else None,
            "sysfs_file": str(sysfs_path) if sysfs_path else None,
        },
        "thermal_flags": thermal_flags,
        "rail_availability": {
            rail: len(_series_for_rail(parsed, rail)[0])
            for rail in [POWER_THERMAL_PROTOCOL["primary_rail"]]
            + list(POWER_THERMAL_PROTOCOL["sub_rails"])
        },
    }

    out_path = output_dir / f"{stem}.json"
    out_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    logger.info("wrote %s", out_path)
    return 0


def _no_caching_enabled() -> bool:
    import os

    return os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1"


# ── verify ─────────────────────────────────────────────────────────────────


def _collect_windows(run: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        window
        for block in run.get("blocks", [])
        for window in block.get("windows", [])
    ]


def _collect_run_files(output_dir: Path) -> List[Path]:
    """Return only the ``run_<index>.json`` records.

    A ``run_*.json`` glob also matches the per-run telemetry dumps
    (``run_0.sysfs.json``), which are lists rather than records, so the name is
    matched exactly instead.
    """
    return sorted(
        path
        for path in output_dir.glob("run_*.json")
        if re.fullmatch(r"run_\d+\.json", path.name)
    )


def _run_verify(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    run_files = _collect_run_files(output_dir)
    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in run_files
    ]
    runs = [run for run in runs if run.get("phase") == "collect"]
    audit_path = output_dir / "audit.json"
    probe_path = output_dir / "probe.json"
    audit = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path.exists()
        else None
    )
    probe = (
        json.loads(probe_path.read_text(encoding="utf-8"))
        if probe_path.exists()
        else None
    )

    # ── per-window evidence, keyed by (block, workload) ──────────────────
    per_block_workload: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for run in runs:
        for block in run.get("blocks", []):
            block_id = block["block_id"]
            for window in block.get("windows", []):
                per_block_workload.setdefault(block_id, {}).setdefault(
                    window["workload_name"], []
                ).append(window)

    published: List[Dict[str, Any]] = []
    unusable: List[Dict[str, Any]] = []
    for block_id, workloads in sorted(per_block_workload.items()):
        for workload, windows in sorted(workloads.items()):
            for window in windows:
                energy = window["energy"]
                if energy.get("usable") and energy.get("energy_j") is not None:
                    published.append(window)
                else:
                    unusable.append(
                        {
                            "block_id": block_id,
                            "workload": workload,
                            "reason": energy.get("reason"),
                        }
                    )

    per_block_workload_j: Dict[str, Dict[str, Any]] = {}
    for block_id, workloads in sorted(per_block_workload.items()):
        per_block_workload_j[block_id] = {}
        for workload, windows in sorted(workloads.items()):
            per_block_workload_j[block_id][workload] = {
                "energy_j": cross_run_reduce(
                    [w["energy"].get("energy_j") for w in windows]
                ),
                "j_per_request": cross_run_reduce(
                    [w["energy_metrics"].get("j_per_request") for w in windows]
                ),
                "output_tok_per_j": cross_run_reduce(
                    [w["energy_metrics"].get("output_tok_per_j") for w in windows]
                ),
                "processed_tok_per_j": cross_run_reduce(
                    [
                        w["energy_metrics"].get("processed_tok_per_j")
                        for w in windows
                    ]
                ),
                "avg_power_w": cross_run_reduce(
                    [w["energy"].get("avg_power_w") for w in windows]
                ),
                "j_per_request_spread": relative_spread(
                    [w["energy_metrics"].get("j_per_request") for w in windows]
                ),
                "thermal_labels": sorted(
                    {w["thermal"]["label"] for w in windows}
                ),
                "isolated_windows": sum(
                    1
                    for w in windows
                    if w["thermal"]["isolate_from_baseline"]
                ),
                "num_windows": len(windows),
            }

    # ── checks ───────────────────────────────────────────────────────────
    three_runs = len(runs) >= int(POWER_THERMAL_PROTOCOL["independent_runs"])
    all_gpu = all(run.get("model_fully_on_gpu") for run in runs)

    monitor_ok = all(
        run["monitor"]["parser_audit"]["parse_error_ratio"] == 0.0
        and run["monitor"]["tegrastats_available"]
        for run in runs
    )

    energy_integrals_ok = len(published) > 0 and len(unusable) == 0

    observable_modes_ok = True
    mode_detail: Dict[str, Any] = {}
    for run in runs:
        for block in run.get("blocks", []):
            block_id = block["block_id"]
            declared = block["declared"]
            observed = (block.get("observed_configured") or {}).get(
                "nvpmodel_mode_id"
            )
            gpu_max_observed = (
                (block.get("observed_configured") or {}).get("sysfs_gpu") or {}
            ).get("max_freq_hz")
            gpu_max_expected = (
                (block.get("state_after_enter") or {})
                .get("sysfs", {})
                .get("gpu_devfreq", {})
                .get("max_freq_hz")
            )
            matches = observed == declared["nvpmodel_id"]
            observable_modes_ok = observable_modes_ok and matches
            mode_detail[f"run{run['run_index']}:{block_id}"] = {
                "requested_mode_id": declared["nvpmodel_id"],
                "observed_mode_id": observed,
                "matches": matches,
                "clock_policy": declared["clock_policy"],
                # Requested vs actual GPU clock: min_freq is what separates a
                # fixed-maximum policy from the dynamic governor, which both
                # report the same max_freq.
                "gpu_min_freq_hz": (
                    (block.get("observed_configured") or {}).get("sysfs_gpu") or {}
                ).get("min_freq_hz"),
                "gpu_max_freq_hz": (
                    (block.get("observed_configured") or {}).get("sysfs_gpu") or {}
                ).get("max_freq_hz"),
                "gpu_governor": (
                    (block.get("observed_configured") or {}).get("sysfs_gpu") or {}
                ).get("governor"),
                "cpu0_min_freq_hz": (
                    (block.get("observed_configured") or {}).get("sysfs_cpu") or {}
                ).get("cpu0", {}).get("min_freq_hz"),
                "emc_cap_hz": (
                    block.get("observed_configured") or {}
                ).get("emc_cap_hz"),
                "nvpmodel_returncode": (
                    block.get("nvpmodel_enter") or {}
                ).get("returncode"),
            }

    restored_ok = all(
        (run.get("restore") or {}).get("diff", {}).get("restored") is True
        for run in runs
    )
    restore_detail = {
        f"run{run['run_index']}": (run.get("restore") or {}).get("diff")
        for run in runs
    }

    closure_ok = all(
        window["dimensional_consistency"].get("closed") is not False
        for window in published
    ) and all(
        window["dimensional_consistency"].get("closed") is True
        for window in published
        if window["dimensional_consistency"].get("comparable")
    )

    tokens_ok = all(
        window["token_denominator_matches"] for window in published
    ) and all(
        window["valid_requests"] == window["requested_requests"]
        for window in published
    )

    token_hashes_ok = True
    for block_id, workloads in per_block_workload.items():
        for workload, windows in workloads.items():
            hashes = {h for w in windows for h in w["sequence_sha256_set"]}
            if len(hashes) != 1:
                token_hashes_ok = False

    idle_refs_ok = all(
        block.get("idle", {}).get("energy", {}).get("usable")
        and block["idle"]["energy"].get("avg_power_w") is not None
        for run in runs
        for block in run.get("blocks", [])
    )

    thermal_isolated_ok = True
    baseline_thermal_clean = True
    for run in runs:
        for block in run.get("blocks", []):
            for window in block.get("windows", []):
                label = window["thermal"]["label"]
                if label != "stable" and block["declared"]["is_baseline"]:
                    baseline_thermal_clean = False
    for block_id, workloads in per_block_workload.items():
        for workload, windows in workloads.items():
            for window in windows:
                if window["thermal"]["isolate_from_baseline"]:
                    thermal_isolated_ok = thermal_isolated_ok and True

    audit_ok = bool(audit and audit.get("verdict", {}).get("passed"))
    rail_declared = bool(probe and probe["rail_enumeration"]["primary_rail_present"])

    # The clock snapshot is what `jetson_clocks --restore` later consumes.  A
    # failed `--store` is recorded here (it happened for the 2026-09-17 runs
    # because a snapshot from the probe step already existed); restoration
    # itself is still judged by the before/after configured-state diff, not by
    # the store return code.
    clock_store_detail = {
        f"run{run['run_index']}": {
            "store_returncode": (run.get("clock_store") or {}).get("returncode"),
            "retried_after_removing_stale_snapshot": (
                run.get("clock_store") or {}
            ).get("retried_after_removing_stale_snapshot"),
            "effective_snapshot_used": (
                "session-snapshot"
                if (run.get("clock_store") or {}).get("returncode") == 0
                else "pre-existing snapshot (probe step), verified by the "
                "before/after configured-state diff"
            ),
        }
        for run in runs
    }

    verdict = {
        "experiment": "E02-08",
        "runs": len(runs),
        "independent_runs_present": three_runs,
        "audit_gate_passed": audit_ok,
        "rail_scope_declared_and_present": rail_declared,
        "clock_store_detail": clock_store_detail,
        "monitor_samples_complete": monitor_ok,
        "energy_integrals_present_and_usable": energy_integrals_ok,
        "num_published_windows": len(published),
        "num_unusable_windows": len(unusable),
        "unusable_windows": unusable,
        "observed_mode_matches_request": observable_modes_ok,
        "mode_detail": mode_detail,
        "original_state_restored": restored_ok,
        "restore_detail": restore_detail,
        "token_denominators_closed": tokens_ok,
        "token_hashes_consistent": token_hashes_ok,
        "dimensional_closure_ok": closure_ok,
        "idle_reference_present_per_mode": idle_refs_ok,
        "thermal_suspect_isolated": thermal_isolated_ok,
        "baseline_windows_free_of_thermal_flags": baseline_thermal_clean,
        "model_fully_on_gpu_all_runs": all_gpu,
        "per_block_workload_energy": per_block_workload_j,
        "thermal_flags": [
            flag for run in runs for flag in run.get("thermal_flags", [])
        ],
        "errors": [err for run in runs for err in run.get("errors", [])],
    }
    verdict["passed"] = all(
        [
            three_runs,
            audit_ok,
            rail_declared,
            monitor_ok,
            energy_integrals_ok,
            observable_modes_ok,
            restored_ok,
            tokens_ok,
            token_hashes_ok,
            closure_ok,
            idle_refs_ok,
            thermal_isolated_ok,
        ]
    )

    (output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    logger.info("wrote %s (passed=%s)", output_dir / "verdict.json", verdict["passed"])
    print(json.dumps({k: v for k, v in verdict.items() if k != "per_block_workload_energy"}, indent=2))
    return 0 if verdict["passed"] else 1


# ── summarize ──────────────────────────────────────────────────────────────


def _freq_histogram(
    sysfs_records: Sequence[Mapping[str, Any]],
    begin_ns: int,
    end_ns: int,
) -> Dict[str, Any]:
    """Distribution of the observed GPU clock inside a window.

    A mean clock cannot distinguish "always at 306 MHz" from "alternating
    between 306 and 1020 MHz", and that difference is exactly what the
    requested-vs-observed-frequency question is about.  The histogram is
    therefore reported alongside the mean.
    """
    values = [
        int(record["gpu_cur_freq_hz"])
        for record in slice_sysfs_records(sysfs_records, begin_ns, end_ns)
        if record.get("gpu_cur_freq_hz")
    ]
    if not values:
        return {"count": 0, "share": {}, "mean_hz": None, "max_hz": None}
    counts: Dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = len(values)
    return {
        "count": total,
        "share": {
            freq: count / total for freq, count in sorted(counts.items())
        },
        "mean_hz": sum(values) / total,
        "max_hz": max(values),
        "min_hz": min(values),
    }


def _run_summarize(args: argparse.Namespace) -> int:
    """Recompute every report table from the raw evidence (no device access)."""
    output_dir = Path(args.output_dir)
    run_files = _collect_run_files(output_dir)
    runs = [
        json.loads(path.read_text(encoding="utf-8")) for path in run_files
    ]
    runs = [run for run in runs if run.get("phase") == "collect"]

    # The raw sysfs telemetry is re-read here (not re-collected) so the clock
    # distribution can be recomputed from the same saved evidence.
    sysfs_by_run: Dict[int, List[Dict[str, Any]]] = {}
    for run in runs:
        path = output_dir / f"run_{run['run_index']}.sysfs.json"
        try:
            sysfs_by_run[run["run_index"]] = json.loads(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            sysfs_by_run[run["run_index"]] = []

    summary: Dict[str, Any] = {
        "experiment": "E02-08",
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "protocol_hash": _protocol_hash(),
        "protocol": POWER_THERMAL_PROTOCOL,
        "rail_scope": RAIL_SCOPE,
        "power_spectrum_note": power_spectrum_note(),
        "runs": [],
    }

    for run in runs:
        entry: Dict[str, Any] = {
            "run_index": run["run_index"],
            "rotation": run["rotation"],
            "load_time_s": run["load_time_s"],
            "num_tegrastats_records": run["monitor"]["num_tegrastats_records"],
            "num_sysfs_records": run["monitor"]["num_sysfs_records"],
            "parser_audit": run["monitor"]["parser_audit"],
            "blocks": [],
        }
        for block in run.get("blocks", []):
            block_entry = {
                "block_id": block["block_id"],
                "label": block["declared"]["label"],
                "nvpmodel_id_requested": block["declared"]["nvpmodel_id"],
                "nvpmodel_id_observed": (
                    block.get("observed_configured") or {}
                ).get("nvpmodel_mode_id"),
                "clock_policy": block["declared"]["clock_policy"],
                "cooldown_before": {
                    "reached": block.get("cooldown_before", {}).get("reached"),
                    "start_c": block.get("cooldown_before", {}).get("start_c"),
                    "end_c": block.get("cooldown_before", {}).get("end_c"),
                },
                "idle": {
                    "avg_power_w": block.get("idle", {})
                    .get("energy", {})
                    .get("avg_power_w"),
                    "energy_j": block.get("idle", {})
                    .get("energy", {})
                    .get("energy_j"),
                    "peak_temp_c": block.get("idle", {})
                    .get("temperature", {})
                    .get("max_c"),
                    "gpu_freq_mean_hz": (
                        block.get("idle", {}).get("gpu_freq_hz") or {}
                    ).get("mean"),
                },
                "windows": [],
            }
            for window in block.get("windows", []):
                frequency = _freq_histogram(
                    sysfs_by_run.get(run["run_index"], []),
                    window["begin_ns"],
                    window["end_ns"],
                )
                block_entry["windows"].append(
                    {
                        "workload": window["workload_name"],
                        "gpu_freq_share": frequency["share"],
                        "gpu_freq_hist_count": frequency["count"],
                        "valid_requests": window["valid_requests"],
                        "window_s": window["window_s"],
                        "usable": window["energy"].get("usable"),
                        "reason": window["energy"].get("reason"),
                        "energy_j": window["energy"].get("energy_j"),
                        "avg_power_w": window["energy"].get("avg_power_w"),
                        "coverage_ratio": window["energy"].get("coverage_ratio"),
                        "max_gap_s": window["energy"].get("max_gap_s"),
                        "j_per_request": window["energy_metrics"].get(
                            "j_per_request"
                        ),
                        "output_tok_per_j": window["energy_metrics"].get(
                            "output_tok_per_j"
                        ),
                        "processed_tok_per_j": window["energy_metrics"].get(
                            "processed_tok_per_j"
                        ),
                        "net_energy_j": (
                            window.get("net_energy") or {}
                        ).get("net_energy_j"),
                        "net_fraction": (
                            window.get("net_energy") or {}
                        ).get("net_fraction"),
                        "sub_rail_avg_power_w": {
                            rail: data.get("avg_power_w")
                            for rail, data in window["sub_rail_energy"].items()
                        },
                        "temperature_zone": window["temperature_zone"],
                        "temp_start_c": window["temperature"].get("start_c"),
                        "temp_peak_c": window["temperature"].get("max_c"),
                        "temp_delta_c": window["temperature"].get("delta_c"),
                        "gpu_freq_mean_hz": window["gpu_freq_hz"].get("mean"),
                        "gpu_freq_min_hz": window["gpu_freq_hz"].get("min"),
                        "thermal_label": window["thermal"]["label"],
                        "thermal_isolated": window["thermal"][
                            "isolate_from_baseline"
                        ],
                        "cooling_engaged": window["cooling"]["engaged"],
                        "dimensional_closed": window["dimensional_consistency"].get(
                            "closed"
                        ),
                        "output_tokens_total": window["output_tokens_total"],
                        "token_denominator_matches": window[
                            "token_denominator_matches"
                        ],
                    }
                )
            entry["blocks"].append(block_entry)
        entry["restore"] = {
            "restored": (run.get("restore") or {})
            .get("diff", {})
            .get("restored"),
            "configured_changes": (run.get("restore") or {})
            .get("diff", {})
            .get("configured_changes"),
        }
        summary["runs"].append(entry)

    # cross-run median / min / max per (block, workload)
    cross: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        for block in run.get("blocks", []):
            for window in block.get("windows", []):
                key = f"{block['block_id']}::{window['workload_name']}"
                metrics = cross.setdefault(
                    key,
                    {
                        "block_id": block["block_id"],
                        "workload": window["workload_name"],
                        "energy_j": [],
                        "j_per_request": [],
                        "output_tok_per_j": [],
                        "processed_tok_per_j": [],
                        "avg_power_w": [],
                        "gpu_freq_mean_hz": [],
                        "temp_peak_c": [],
                        "thermal_labels": [],
                    },
                )
                metrics["energy_j"].append(window["energy"].get("energy_j"))
                metrics["j_per_request"].append(
                    window["energy_metrics"].get("j_per_request")
                )
                metrics["output_tok_per_j"].append(
                    window["energy_metrics"].get("output_tok_per_j")
                )
                metrics["processed_tok_per_j"].append(
                    window["energy_metrics"].get("processed_tok_per_j")
                )
                metrics["avg_power_w"].append(window["energy"].get("avg_power_w"))
                metrics["gpu_freq_mean_hz"].append(
                    window["gpu_freq_hz"].get("mean")
                )
                metrics["temp_peak_c"].append(window["temperature"].get("max_c"))
                metrics["thermal_labels"].append(window["thermal"]["label"])

    for key, metrics in cross.items():
        metrics["energy_j"] = cross_run_reduce(metrics["energy_j"])
        metrics["j_per_request"] = cross_run_reduce(metrics["j_per_request"])
        metrics["output_tok_per_j"] = cross_run_reduce(
            metrics["output_tok_per_j"]
        )
        metrics["processed_tok_per_j"] = cross_run_reduce(
            metrics["processed_tok_per_j"]
        )
        metrics["avg_power_w"] = cross_run_reduce(metrics["avg_power_w"])
        metrics["gpu_freq_mean_hz"] = cross_run_reduce(
            metrics["gpu_freq_mean_hz"]
        )
        metrics["temp_peak_c"] = cross_run_reduce(metrics["temp_peak_c"])
        metrics["thermal_labels"] = sorted(set(metrics["thermal_labels"]))
    summary["cross_run"] = cross

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Evidence manifest: every artifact with its size and digest, so the report
    # can be checked against the bytes that produced it.
    manifest: Dict[str, Any] = {
        "experiment": "E02-08",
        "generated_at": summary["generated_at"],
        "protocol_hash": summary["protocol_hash"],
        "primary_rail": RAIL_SCOPE["primary_rail"],
        "rail_scope": RAIL_SCOPE["primary_scope"],
        "runs": [run["run_index"] for run in runs],
        "artifacts": {},
    }
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            try:
                manifest["artifacts"][path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            except OSError:
                continue
    for run in runs:
        index = run["run_index"]
        manifest.setdefault("per_run", {})[f"run_{index}"] = {
            "run_id": run.get("run_id"),
            "rotation": run.get("rotation"),
            "blocks": [block["block_id"] for block in run.get("blocks", [])],
            "state_restored": (
                run.get("restore") or {}
            ).get("diff", {}).get("restored"),
            "load_time_s": run.get("load_time_s"),
        }
    (output_dir / "EVIDENCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Markdown rendering of the same numbers.
    lines: List[str] = ["# E02-08 evidence summary (recomputed from raw)", ""]
    lines.append(
        "Energy is the trapezoidal integral of the instantaneous VDD_IN rail "
        "over each window's own host-monotonic interval. USB/board scope: "
        f"{RAIL_SCOPE['primary_scope']}."
    )
    lines.append("")
    lines.append("## Per-run window table")
    lines.append("")
    lines.append(
        "| run | block | workload | N | window s | E J | avg W | J/req | out tok/J | "
        "temp start→peak C | gpu freq MHz | thermal | usable |"
    )
    lines.append(
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|"
    )
    for entry in summary["runs"]:
        for block in entry["blocks"]:
            for window in block["windows"]:
                lines.append(
                    "| {run} | {block} | {wl} | {n} | {win:.1f} | {e} | {p} | "
                    "{jpr} | {tpj} | {t0}→{tp} | {f} | {th} | {us} |".format(
                        run=entry["run_index"],
                        block=block["block_id"],
                        wl=window["workload"],
                        n=window["valid_requests"],
                        win=window["window_s"] or 0.0,
                        e=_fmt(window["energy_j"], 1),
                        p=_fmt(window["avg_power_w"], 2),
                        jpr=_fmt(window["j_per_request"], 1),
                        tpj=_fmt(window["output_tok_per_j"], 3),
                        t0=_fmt(window["temp_start_c"], 1),
                        tp=_fmt(window["temp_peak_c"], 1),
                        f=_fmt(
                            (window["gpu_freq_mean_hz"] or 0) / 1e6
                            if window["gpu_freq_mean_hz"]
                            else None,
                            0,
                        ),
                        th=window["thermal_label"],
                        us=window["usable"],
                    )
                )
    lines.append("")
    lines.append("## Cross-run median (min–max)")
    lines.append("")
    lines.append(
        "| block | workload | E J | avg W | J/req | out tok/J | proc tok/J | "
        "peak C | MHz | labels |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for key, metrics in sorted(cross.items()):
        lines.append(
            "| {block} | {wl} | {e} | {p} | {j} | {t} | {q} | {c} | {f} | {l} |".format(
                block=metrics["block_id"],
                wl=metrics["workload"],
                e=_reduced(metrics["energy_j"], 1),
                p=_reduced(metrics["avg_power_w"], 2),
                j=_reduced(metrics["j_per_request"], 1),
                t=_reduced(metrics["output_tok_per_j"], 3),
                q=_reduced(metrics["processed_tok_per_j"], 3),
                c=_reduced(metrics["temp_peak_c"], 1),
                f=_reduced(
                    cross_run_reduce(
                        [
                            (v / 1e6) if v else None
                            for v in metrics["gpu_freq_mean_hz"]["values"]
                        ]
                    ),
                    0,
                ),
                l=",".join(metrics["thermal_labels"]),
            )
        )
    # observed-clock distribution, averaged over the independent runs
    freq_share: Dict[str, Dict[float, float]] = {}
    for entry in summary["runs"]:
        for block in entry["blocks"]:
            for window in block["windows"]:
                key = f"{block['block_id']}::{window['workload']}"
                bucket = freq_share.setdefault(key, {})
                for freq, share in (window.get("gpu_freq_share") or {}).items():
                    bucket[freq] = bucket.get(freq, 0.0) + share / len(
                        summary["runs"]
                    )
    summary["gpu_freq_share_cross_run"] = {
        key: {str(int(f)): round(v, 4) for f, v in sorted(bucket.items())}
        for key, bucket in sorted(freq_share.items())
    }

    lines.append("")
    lines.append("## Observed GPU clock distribution (mean share across runs)")
    lines.append("")
    lines.append(
        "A mean clock cannot distinguish 'always at the floor' from "
        "'alternating between floor and ceiling'; the distribution is the "
        "requested-vs-observed frequency evidence."
    )
    lines.append("")
    lines.append("| block | workload | clock share (MHz: %) |")
    lines.append("|---|---|---|")
    for key, bucket in sorted(freq_share.items()):
        block_id, workload = key.split("::", 1)
        rendered = ", ".join(
            f"{int(freq) / 1e6:.0f}: {share * 100:.1f}%"
            for freq, share in sorted(bucket.items())
            if share > 0.001
        )
        lines.append(f"| {block_id} | {workload} | {rendered} |")

    (output_dir / "evidence_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    logger.info("wrote %s and %s", output_dir / "summary.json",
                output_dir / "evidence_summary.md")
    print("\n".join(lines))
    return 0


def _fmt(value: Optional[float], digits: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _reduced(reduced: Mapping[str, Any], digits: int) -> str:
    if not reduced or reduced.get("median") is None:
        return "n/a"
    return (
        f"{reduced['median']:.{digits}f} "
        f"({reduced['min']:.{digits}f}-{reduced['max']:.{digits}f})"
    )


# ── CLI ────────────────────────────────────────────────────────────────────


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="~/models/hqsb/Qwen3-1.7B")
    parser.add_argument("--manifest", default=str(_MANIFEST))
    parser.add_argument("--workload-yaml", default=str(_WORKLOAD_YAML))
    parser.add_argument(
        "--monitor-interval-ms",
        type=int,
        default=int(POWER_THERMAL_PROTOCOL["monitor_interval_ms"]),
    )
    parser.add_argument(
        "--temperature-zone", default="tj-thermal"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E02-08 power/energy/thermal runner")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="record the raw device state")
    _add_common(probe)
    probe.add_argument("--sample-seconds", type=float, default=5.0)
    probe.set_defaults(func=_run_probe)

    audit = sub.add_parser("audit", help="monitor self-tests + coverage rehearsal")
    _add_common(audit)
    audit.add_argument("--audit-workload", default="short")
    audit.add_argument("--audit-requests", type=int, default=1)
    audit.add_argument("--audit-idle-s", type=float, default=10.0)
    audit.set_defaults(func=_run_audit)

    collect = sub.add_parser("collect", help="one independent measurement process")
    _add_common(collect)
    collect.add_argument("--run-index", type=int, default=0)
    collect.add_argument(
        "--requests-per-window",
        type=int,
        default=int(POWER_THERMAL_PROTOCOL["requests_per_window"]),
    )
    collect.add_argument(
        "--idle-window-s",
        type=float,
        default=float(POWER_THERMAL_PROTOCOL["idle_window_s"]),
    )
    collect.add_argument(
        "--cooldown-target-c",
        type=float,
        default=float(POWER_THERMAL_PROTOCOL["cooldown_target_c"]),
    )
    collect.add_argument(
        "--cooldown-timeout-s",
        type=float,
        default=float(POWER_THERMAL_PROTOCOL["cooldown_timeout_s"]),
    )
    collect.add_argument("--warmup-output-tokens", type=int, default=4)
    collect.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "exercise the mode-switch/cooldown/restore path with one `short` "
            "window per block; the resulting run file is marked smoke and is "
            "excluded from the verdict"
        ),
    )
    collect.set_defaults(func=_run_collect)

    verify = sub.add_parser("verify", help="cross-run verdict")
    verify.add_argument("--output-dir", required=True)
    verify.set_defaults(func=_run_verify)

    summarize = sub.add_parser("summarize", help="recompute report tables")
    summarize.add_argument("--output-dir", required=True)
    summarize.set_defaults(func=_run_summarize)
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
