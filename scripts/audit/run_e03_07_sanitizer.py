#!/usr/bin/env python3
"""Parse E03-07 isolated sanitizer logs into auditable raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


CASE_RE = re.compile(
    r"HQSB_CASE id=(\S+) status=(\S+) launch=(-?\d+) completion=(-?\d+) "
    r"guards=(\S+) detail=(\S+)"
)
ERROR_RE = re.compile(r"ERROR SUMMARY:\s*(\d+)\s+error", re.IGNORECASE)
RACE_RE = re.compile(r"RACECHECK SUMMARY:.*?\((\d+)\s+error", re.IGNORECASE)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir

    runs: list[dict[str, object]] = []
    with (out / "index.tsv").open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            log_path = out / row["log"]
            text = log_path.read_text(errors="replace") if log_path.exists() else ""
            summaries = [int(x) for x in ERROR_RE.findall(text)] + [
                int(x) for x in RACE_RE.findall(text)
            ]
            cases = [
                {
                    "id": m.group(1),
                    "status": m.group(2),
                    "launch_status": int(m.group(3)),
                    "completion_status": int(m.group(4)),
                    "guards": m.group(5),
                    "detail": m.group(6),
                }
                for m in CASE_RE.finditer(text)
            ]
            exit_code = int(row["exit_code"])
            expectation = row["expectation"]
            issue_count = max(summaries) if summaries else None
            if expectation == "detect":
                expectation_met = exit_code == 86 and issue_count is not None and issue_count > 0
            elif row["tool"] == "ordinary":
                expectation_met = exit_code == 0 and all(
                    case["status"] in {"PASS", "DONE"} for case in cases
                )
            else:
                expectation_met = (
                    exit_code == 0
                    and issue_count == 0
                    and bool(cases)
                    and all(case["status"] in {"PASS", "DONE"} for case in cases)
                )
            runs.append(
                {
                    **row,
                    "exit_code": exit_code,
                    "elapsed_s": int(row["elapsed_s"]),
                    "sanitizer_error_summaries": summaries,
                    "issue_count": issue_count,
                    "cases": cases,
                    "expectation_met": expectation_met,
                    "log_sha256": digest(log_path) if log_path.exists() else None,
                }
            )

    by_id = {str(run["run_id"]): run for run in runs}
    tools = ("memcheck", "racecheck", "initcheck", "synccheck")
    controls_valid = all(
        by_id.get(f"control_{tool}_clean", {}).get("expectation_met")
        and by_id.get(f"control_{tool}_negative", {}).get("expectation_met")
        for tool in tools
    )
    production_clean = all(
        by_id.get(f"{tool}_{family}_matrix", {}).get("expectation_met")
        for tool in tools
        for family in ("rms", "fused")
    )
    api_safe = bool(by_id.get("api_negative", {}).get("expectation_met"))
    destroyed_safe = bool(by_id.get("destroyed_stream", {}).get("expectation_met"))
    host_pointer_safe = bool(by_id.get("host_pointer", {}).get("expectation_met"))
    lifecycle_stable = bool(by_id.get("lifecycle", {}).get("expectation_met"))
    leak_clean = bool(by_id.get("memcheck_lifecycle", {}).get("expectation_met"))
    danger_memcheck_clean = all(
        by_id.get(name, {}).get("expectation_met")
        for name in ("memcheck_api_negative", "memcheck_destroyed_stream", "memcheck_host_pointer")
    )

    production_cases = [
        case
        for run in runs
        if str(run["run_id"]).endswith(("rms_matrix", "fused_matrix"))
        for case in run["cases"]  # type: ignore[index]
    ]
    coverage_ids = {str(case["id"]) for case in production_cases}
    coverage = {
        "minimum_h1": any("_h1" in case for case in coverage_ids),
        "tail_31_33_127_129_2047_2049": all(
            any(f"_h{h}" in case for case in coverage_ids)
            for h in (31, 33, 127, 129, 2047, 2049)
        ),
        "aligned_32_128_2048": all(
            any(f"_h{h}" in case for case in coverage_ids) for h in (32, 128, 2048)
        ),
        "maximum_safely_allocated_h8192": any("_h8192" in case for case in coverage_ids),
        "maximum_declared_h2147483647_executed": False,
        "real_prefill_rows1024": all(
            any(token in case for case in coverage_ids)
            for token in ("rms_fp32_prefill1024", "rms_fp16_prefill1024",
                          "fused_fp32_prefill1024", "fused_fp16_prefill1024")
        ),
        "fp16_fp32": any("fp16" in case for case in coverage_ids)
        and any("fp32" in case for case in coverage_ids),
        "forced_v0_v1_vector_scalar": all(
            any(token in case for case in coverage_ids)
            for token in ("forced_v0", "forced_v1", "_h4", "_h3")
        ),
        "auto_misaligned_fallback": all(
            name in coverage_ids
            for name in ("rms_fp32_misaligned_auto", "rms_fp16_misaligned_auto")
        ),
        "exact_inplace": all(
            name in coverage_ids
            for name in ("rms_fp32_exact_inplace", "rms_fp16_exact_inplace")
        ),
        "fused": any(case.startswith("fused_") for case in coverage_ids),
        "nondefault_stream": True,
    }
    # The frozen RMS contract declares INT_MAX H, which cannot be allocated on
    # the 8 GB target. Preserve this as an explicit coverage/spec failure; do
    # not silently redefine H=8192 as the declared maximum.
    coverage_complete = all(coverage.values())

    conditions = {
        "1_sanitizer_positive_negative_controls_valid": controls_valid,
        "2_supported_cases_clean_under_applicable_tools": production_clean,
        "3_required_path_coverage_executed": coverage_complete,
        "4_invalid_inputs_prelaunch_or_stable_reject": api_safe and destroyed_safe and host_pointer_safe,
        "5_alias_policy_enforced_and_allowed_alias_clean": api_safe
        and coverage["exact_inplace"],
        "6_workspace_zero_no_hidden_allocation_and_lifetime_valid": lifecycle_stable,
        "7_immediate_and_async_errors_not_swallowed": destroyed_safe and host_pointer_safe,
        "8_repeated_lifecycle_no_project_growth_or_leak": lifecycle_stable and leak_clean,
        "9_raw_logs_commands_exit_codes_and_binary_identity_complete": bool(runs)
        and (out / "provenance.txt").exists(),
    }
    tool_available = "Compute Sanitizer" in (out / "provenance.txt").read_text(errors="replace")
    if not tool_available or not controls_valid:
        overall = "BLOCKED"
    else:
        overall = "PASS" if all(conditions.values()) and danger_memcheck_clean else "FAIL"

    protocol = {
        "schema_version": "hqsb.s03.e03_07.protocol/v1",
        "experiment_id": "E03-07",
        "frozen_before_formal_run": True,
        "tools": list(tools),
        "sanitizer_path": "/usr/local/cuda-12.6/bin/compute-sanitizer",
        "timeout_s": 600,
        "production_build": "Release -O3 -lineinfo, sm_87",
        "workspace_bytes": 0,
        "lifecycle_cuda_mem_get_info_envelope_bytes": 2 * 1024 * 1024,
        "lifecycle_hard_leak_rule": "memcheck leak report must be 0 bytes in 0 allocations",
        "alias_policy": {
            "rmsnorm_exact_input_output": "allowed",
            "rmsnorm_partial_input_output": "reject",
            "rmsnorm_output_weight": "reject",
            "fused_all_output_input_and_output_output_overlap": "reject",
        },
        "pass_rule": "all nine conditions true; controls mandatory; no PASS_NEGATIVE",
        "max_shape_note": "H=8192 is the largest safely allocated audit case; raw RMS ABI theoretical INT_MAX H is validation-only",
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "runs.json", runs)
    write_json(out / "coverage.json", {"case_ids": sorted(coverage_ids), "conditions": coverage})
    verdict = {
        "schema_version": "hqsb.s03.e03_07.v1",
        "experiment_id": "E03-07",
        "verified_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "overall": overall,
        "conditions": conditions,
        "failed_conditions": [name for name, value in conditions.items() if not value],
        "tool_available": tool_available,
        "controls_valid": controls_valid,
        "dangerous_memcheck_cases_clean": danger_memcheck_clean,
        "run_count": len(runs),
        "production_case_observations": len(production_cases),
        "stage_gate": "open for E03-08" if overall == "PASS" else "closed",
    }
    write_json(out / "verdict.json", verdict)

    manifest = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "EVIDENCE_MANIFEST.json":
            manifest.append(
                {
                    "path": str(path.relative_to(out)),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    write_json(
        out / "EVIDENCE_MANIFEST.json",
        {"schema_version": "hqsb.evidence-manifest/v1", "files": manifest},
    )
    print(json.dumps(verdict, ensure_ascii=False))
    return 0 if overall == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
