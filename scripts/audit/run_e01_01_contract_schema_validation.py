#!/usr/bin/env python3
"""E01-01 — C1–C7 versioned contract schema validation.

Question
--------
Are C1–C7 *executable, versioned, error-rejecting data contracts*, so that a
legal payload round-trips losslessly while missing / unknown / wrong-typed /
local-semantic / future-version / old-version payloads are all rejected at
the parse boundary with a field-localized, stable error?

Hypothesis (falsifiable, pre-registered)
----------------------------------------
H1  Every contract (C1–C7) has a current-version canonical fixture that
    round-trips losslessly; all six negative input classes are rejected at
    the entry boundary with a stable error code and a field path; future
    versions are refused by the version gate; old versions are refused with
    a ``requires_migration`` signal (never silently reinterpreted).
H0  At least one negative class is accepted, or is only detected later, or a
    versioned payload is silently coerced to the current schema.

Design
------
* Step 1: load the C1–C7 contract registry and verify its integrity.
* Step 2: legal round-trip (parse → serialize → parse) + canonical hash.
* Steps 3–8: missing_required / unknown_field / wrong_type / local-semantic /
  future_version / old_version negative cases.
* Step 9: serialization stability across a second round-trip.
* Step 10: structured per-case records, error matrix, verdict.

This is pure CPU: no torch, no GPU, no model weights.

Raw output (under <out>/)
-------------------------
``e01_01_run_<run_id>.json``         full record (registry + every case)
``e01_01_run_<run_id>_env.json``     frozen environment / git identity
``E01-01_case_results.jsonl``        one JSON line per case
``E01-01_roundtrip_hashes.json``     per-contract canonical round-trip hash
``E01-01_error_matrix.csv``          contract × case-kind matrix
``E01-01_negative_cases.jsonl``      negative-case records only
``verdict.json``                     pass criteria + overall verdict
``fixtures/contracts/<C#>/<id>.json`` materialized fixtures (reviewable)

Usage
-----
    python3 scripts/audit/run_e01_01_contract_schema_validation.py \
        --output-dir docs/stage_experiments/S01/E01-01/raw
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hqsb.core.contracts.registry import load_contract_registry, registry_checks  # noqa: E402
from hqsb.core.errors import (  # noqa: E402
    ExitCode,
    SchemaMigrationRequiredError,
    UnsupportedSchemaVersionError,
    exit_code_for,
)
from hqsb.core.ids import new_run_id  # noqa: E402

EXPERIMENT_ID = "E01-01"
STAGE = "S01"
CURRENT_VERSION = "1.0.0"
FUTURE_VERSION = "2.0.0"
OLD_VERSION = "0.9.0"

#: Canonical fixture IDs follow ``<C#>-<case>-v1``.
#: Error codes are stable machine-readable labels (not free-text messages).
CODE_MISSING = "missing_field"
CODE_UNKNOWN = "unknown_field"
CODE_WRONG_TYPE = "type_error"
CODE_LOCAL_RULE = "local_rule"
CODE_FUTURE = "unsupported_future_schema_version"
CODE_MIGRATION = "requires_migration"

_SHA = "a" * 64

# ── Fixture definitions ─────────────────────────────────────────────────
# Each contract: canonical valid payload, the field deleted for the
# missing-required case, the (name, value) unknown field, the (name, value)
# wrong-type mutation, and one local-semantic payload with its expected field
# path.
FIXTURES: Dict[str, Dict[str, Any]] = {
    "C1": {
        "name": "ModelArtifact",
        "valid": {
            "schema_version": CURRENT_VERSION,
            "model_id": "Qwen/Qwen3-1.7B",
            "source": "modelscope",
            "architecture": "Qwen3ForCausalLM",
            "dtype": "float16",
            "revision": None,
            "file_hashes": {"config.json": _SHA},
            "layout": "dense",
            "context_length": 32768,
        },
        "missing_field": "architecture",
        "unknown": ("experimental_magic", True),
        "wrong_type": ("dtype", 12345),
        "domain": {
            "schema_version": CURRENT_VERSION,
            "model_id": "Qwen/Qwen3-1.7B",
            "source": "modelscope",
            "architecture": "Qwen3ForCausalLM",
            "dtype": "float16",
            "file_hashes": {"config.json": "not-a-sha256"},
        },
        "domain_path": "file_hashes",
    },
    "C2": {
        "name": "WorkloadSpec",
        "valid": {
            "schema_version": CURRENT_VERSION,
            "name": "short",
            "batch_size": 1,
            "input_tokens": 128,
            "output_tokens": 32,
            "seed": 0,
            "sampling": "greedy",
            "warmup": 1,
            "repetitions": 1,
            "concurrency": 1,
            "arrival_process": "fixed",
            "stop_condition": "output_tokens",
        },
        "missing_field": "input_tokens",
        "unknown": ("experimental_magic", True),
        "wrong_type": ("input_tokens", "not-an-int"),
        "domain": {
            "schema_version": CURRENT_VERSION,
            "name": "short",
            "input_tokens": 0,
            "output_tokens": 32,
        },
        "domain_path": "input_tokens",
    },
    "C3": {
        "name": "OperatorSpec",
        "valid": {
            "schema_version": CURRENT_VERSION,
            "name": "rmsnorm",
            "semantic_version": "1.0.0",
            "inputs": [{"name": "x", "dtype": "float32", "shape": [-1, 1024]}],
            "outputs": [{"name": "y", "dtype": "float32", "shape": [-1, 1024]}],
            "device": "cuda",
            "workspace_bytes": 0,
            "deterministic": True,
            "tolerance": 0.0,
            "implementation": "v0_shared",
        },
        "missing_field": "implementation",
        "unknown": ("experimental_magic", True),
        "wrong_type": ("workspace_bytes", "not-an-int"),
        "domain": {
            "schema_version": CURRENT_VERSION,
            "name": "rmsnorm",
            "semantic_version": "1.0.0",
            "inputs": [{"name": "x", "dtype": "float32", "shape": [-1, 1024]}],
            "outputs": [{"name": "y", "dtype": "float32", "shape": [-1, 1024]}],
            "implementation": "v0_shared",
            "workspace_bytes": -1,
        },
        "domain_path": "workspace_bytes",
    },
    "C4": {
        "name": "BackendCapability",
        "valid": {
            "schema_version": CURRENT_VERSION,
            "name": "dummy",
            "supported_dtypes": ["float16"],
            "max_batch": 1,
            "streaming": False,
            "quantization": [],
            "distributed": False,
        },
        "missing_field": "name",
        "unknown": ("experimental_magic", True),
        "wrong_type": ("max_batch", "not-an-int"),
        "domain": {
            "schema_version": CURRENT_VERSION,
            "name": "dummy",
            "supported_dtypes": ["float16"],
            "max_batch": 0,
        },
        "domain_path": "max_batch",
    },
    "C5": {
        "name": "QuantArtifact",
        "valid": {
            "schema_version": CURRENT_VERSION,
            "algorithm": "rtn",
            "bits": 4,
            "granularity": "per-channel",
            "symmetric": True,
            "packing": "native",
            "kernel_compatibility": "unknown",
        },
        "missing_field": "bits",
        "unknown": ("experimental_magic", True),
        "wrong_type": ("bits", "not-an-int"),
        "domain": {
            "schema_version": CURRENT_VERSION,
            "algorithm": "rtn",
            "bits": 9,
            "granularity": "per-channel",
        },
        "domain_path": "bits",
    },
    "C6": {
        "name": "BenchmarkResult",
        "valid": {
            "schema_version": CURRENT_VERSION,
            "run_id": "run_fixture_0001",
            "timestamp": 1700000000.0,
            "environment": {"platform": "test", "device": "cpu"},
            "raw_samples": [],
            "summary": {},
        },
        "missing_field": "run_id",
        "unknown": ("experimental_magic", True),
        "wrong_type": ("timestamp", "not-a-number"),
        "domain": {
            "schema_version": CURRENT_VERSION,
            "run_id": "run_fixture_0002",
            "timestamp": 1700000000.0,
            "environment": {"platform": "test"},
            "resource": {"peak_cuda_allocated_mb": -1},
        },
        "domain_path": "resource.peak_cuda_allocated_mb",
    },
    "C7": {
        "name": "TraceEvent",
        "valid": {
            "schema_version": CURRENT_VERSION,
            "event_type": "prefill",
            "timestamp_ns": 10,
            "trace_id": "t1",
            "span_id": "s1",
        },
        "missing_field": "trace_id",
        "unknown": ("experimental_magic", True),
        "wrong_type": ("timestamp_ns", "not-an-int"),
        "domain": {
            "schema_version": CURRENT_VERSION,
            "event_type": "prefill",
            "timestamp_ns": -1,
            "trace_id": "t1",
            "span_id": "s1",
        },
        "domain_path": "timestamp_ns",
    },
}

#: Case kinds in execution order.
CASE_KINDS = (
    "valid_current",
    "missing_required",
    "unknown_field",
    "wrong_type",
    "local_semantic",
    "future_version",
    "old_version",
)


# ── Helpers ─────────────────────────────────────────────────────────────


def _git(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(_REPO_ROOT), capture_output=True, text=True
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except OSError:
        return ""


def _canonical(obj: Any) -> str:
    """Canonical JSON (sorted keys, compact) of a model's JSON dump."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _canonical_hash(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _loc_path(loc: Tuple[Any, ...]) -> str:
    """Render a pydantic error ``loc`` tuple as a JSON-path-like string."""
    parts: List[str] = []
    for item in loc:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        else:
            parts.append(str(item))
    if not parts:
        return "$"
    return "$." + ".".join(p for p in parts if not p.startswith("["))


def _classify(exc: BaseException) -> Tuple[str, str, str]:
    """Return ``(error_code, error_path, error_kind)`` for a raised error.

    ``error_kind`` is the concrete exception class name (stable taxonomy).
    """
    if isinstance(exc, UnsupportedSchemaVersionError):
        return CODE_FUTURE, "$.schema_version", type(exc).__name__
    if isinstance(exc, SchemaMigrationRequiredError):
        return CODE_MIGRATION, "$.schema_version", type(exc).__name__
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        if errors:
            first = errors[0]
            ptype = first.get("type", "")
            loc = _loc_path(first.get("loc", ()))
            if ptype == "missing":
                return CODE_MISSING, loc, "ValidationError"
            if ptype == "extra_forbidden":
                return CODE_UNKNOWN, loc, "ValidationError"
            if ptype in {"value_error", "enum", "greater_than_equal",
                         "less_than_equal", "less_than", "greater_than",
                         "int_from_float"}:
                return CODE_LOCAL_RULE, loc, "ValidationError"
            return CODE_WRONG_TYPE, loc, "ValidationError"
        return CODE_WRONG_TYPE, "$", "ValidationError"
    return "unexpected_error", "$", type(exc).__name__


def _expected_for(case_kind: str) -> Tuple[str, str]:
    """Return the ``(expected_verdict, expected_error_code)`` per kind."""
    if case_kind == "valid_current":
        return "accept", ""
    if case_kind == "missing_required":
        return "reject", CODE_MISSING
    if case_kind == "unknown_field":
        return "reject", CODE_UNKNOWN
    if case_kind == "wrong_type":
        return "reject", CODE_WRONG_TYPE
    if case_kind == "local_semantic":
        return "reject", CODE_LOCAL_RULE
    if case_kind == "future_version":
        return "reject", CODE_FUTURE
    if case_kind == "old_version":
        return "reject", CODE_MIGRATION
    raise AssertionError(case_kind)


# ── Case execution ──────────────────────────────────────────────────────


def _build_payload(spec: Dict[str, Any], case_kind: str) -> Dict[str, Any]:
    if case_kind == "valid_current":
        return copy.deepcopy(spec["valid"])
    payload = copy.deepcopy(spec["valid"])
    if case_kind == "missing_required":
        payload.pop(spec["missing_field"])
    elif case_kind == "unknown_field":
        key, value = spec["unknown"]
        payload[key] = value
    elif case_kind == "wrong_type":
        key, value = spec["wrong_type"]
        payload[key] = value
    elif case_kind == "local_semantic":
        return copy.deepcopy(spec["domain"])
    elif case_kind == "future_version":
        payload["schema_version"] = FUTURE_VERSION
    elif case_kind == "old_version":
        payload["schema_version"] = OLD_VERSION
    return payload


def _run_case(model_cls: Any, case_kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one payload against ``model_cls`` and record the verdict."""
    expected_verdict, expected_code = _expected_for(case_kind)
    record: Dict[str, Any] = {
        "case_kind": case_kind,
        "expected_verdict": expected_verdict,
        "expected_error_code": expected_code,
        "input_schema_version": payload.get("schema_version", "<omitted>"),
    }
    try:
        obj = model_cls.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - classify any rejection
        code, path, kind = _classify(exc)
        record.update(
            {
                "actual_verdict": "reject",
                "error_code": code,
                "error_path": path,
                "error_kind": kind,
                "exit_code": exit_code_for(exc),
                "message": str(exc)[:200],
                "roundtrip_hash": None,
            }
        )
        record["status"] = (
            "PASS"
            if (code == expected_code and record["actual_verdict"] == expected_verdict)
            else "FAIL"
        )
        return record

    # Accepted: verify a stable canonical round-trip.
    h1 = _canonical_hash(obj)
    h2 = _canonical_hash(model_cls.model_validate(json.loads(obj.model_dump_json())))
    record.update(
        {
            "actual_verdict": "accept",
            "error_code": "",
            "error_path": "",
            "error_kind": "",
            "exit_code": ExitCode.SUCCESS,
            "message": "",
            "roundtrip_hash": h1,
            "roundtrip_hash_stable": h1 == h2,
        }
    )
    if case_kind == "valid_current":
        record["status"] = "PASS" if h1 == h2 else "FAIL"
    else:
        # A negative case that was accepted is a failure.
        record["status"] = "FAIL"
    return record


# ── Drivers ─────────────────────────────────────────────────────────────


def collect_environment() -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_commit_short": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": str(_REPO_ROOT),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write_fixtures(out_dir: Path) -> None:
    fixtures_dir = out_dir / "fixtures" / "contracts"
    for contract, spec in FIXTURES.items():
        for case_kind in CASE_KINDS:
            payload = _build_payload(spec, case_kind)
            fid = f"{contract}-{case_kind}-v1"
            d = fixtures_dir / contract
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{fid}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )


def run(entries: List[Any], out_dir: Path) -> Dict[str, Any]:
    # Step 1: registry integrity.
    checks = registry_checks(entries)
    registry_ok = all(checks.values())

    cases: List[Dict[str, Any]] = []
    roundtrip_hashes: Dict[str, Any] = {}
    for entry in entries:
        contract = entry.contract
        model_cls = entry.model_cls
        spec = FIXTURES[contract]
        for case_kind in CASE_KINDS:
            payload = _build_payload(spec, case_kind)
            rec = _run_case(model_cls, case_kind, payload)
            rec.update(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "contract": contract,
                    "contract_name": entry.name,
                    "fixture_id": f"{contract}-{case_kind}-v1",
                    "schema_version": entry.schema_version,
                    "validation_layer": "schema",
                    "parser_version": entry.parser,
                    "schema_sha256": entry.schema_sha256,
                }
            )
            cases.append(rec)
            if case_kind == "valid_current":
                roundtrip_hashes[contract] = {
                    "contract_name": entry.name,
                    "schema_sha256": entry.schema_sha256,
                    "roundtrip_hash": rec["roundtrip_hash"],
                    "roundtrip_hash_stable": rec["roundtrip_hash_stable"],
                }

    negative = [c for c in cases if c["case_kind"] != "valid_current"]
    passed = sum(1 for c in cases if c["status"] == "PASS")
    valid_passed = sum(
        1 for c in cases if c["case_kind"] == "valid_current" and c["status"] == "PASS"
    )
    negative_passed = sum(1 for c in negative if c["status"] == "PASS")

    verdict = {
        "registry_ok": registry_ok,
        "registry_checks": checks,
        "all_valid_roundtrip": valid_passed == len(entries),
        "all_negative_rejected": negative_passed == len(negative),
        "total_cases": len(cases),
        "passed_cases": passed,
        "valid_cases": len(entries),
        "negative_cases": len(negative),
        "overall": "PASS"
        if (registry_ok and valid_passed == len(entries)
            and negative_passed == len(negative))
        else "FAIL",
    }
    return {
        "cases": cases,
        "roundtrip_hashes": roundtrip_hashes,
        "verdict": verdict,
    }


def _write_csv(path: Path, cases: List[Dict[str, Any]]) -> None:
    cols = [
        "contract", "fixture_id", "case_kind", "expected_verdict",
        "actual_verdict", "validation_layer", "error_code", "error_path",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for c in cases:
            writer.writerow({k: c.get(k, "") for k in cols})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E01-01 contract schema validation.")
    parser.add_argument(
        "--output-dir",
        default="docs/stage_experiments/S01/E01-01/raw",
        help="Directory for raw JSON/JSONL/CSV artifacts.",
    )
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or new_run_id()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_contract_registry()
    result = run(entries, out_dir)
    cases = result["cases"]

    # Full run record + env.
    record = {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "environment": collect_environment(),
        "registry": [
            {
                "contract": e.contract,
                "name": e.name,
                "schema_version": e.schema_version,
                "module_path": e.module_path,
                "schema_sha256": e.schema_sha256,
                "unknown_field_policy": e.unknown_field_policy,
                "legacy_policy": e.legacy_policy,
                "parser": e.parser,
                "serializer": e.serializer,
            }
            for e in entries
        ],
        "verdict": result["verdict"],
        "roundtrip_hashes": result["roundtrip_hashes"],
        "cases": cases,
    }
    (out_dir / f"e01_01_{run_id}.json").write_text(
        json.dumps(record, indent=2, sort_keys=False, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / f"e01_01_{run_id}_env.json").write_text(
        json.dumps(record["environment"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Protocol §10 artifacts.
    with (out_dir / "E01-01_case_results.jsonl").open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n")
    with (out_dir / "E01-01_negative_cases.jsonl").open("w", encoding="utf-8") as fh:
        for c in cases:
            if c["case_kind"] != "valid_current":
                fh.write(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n")
    (out_dir / "E01-01_roundtrip_hashes.json").write_text(
        json.dumps(result["roundtrip_hashes"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(out_dir / "E01-01_error_matrix.csv", cases)
    (out_dir / "verdict.json").write_text(
        json.dumps(result["verdict"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_fixtures(out_dir)

    # Console report.
    v = result["verdict"]
    print(f"[{EXPERIMENT_ID}] run_id={run_id}")
    print(f"[{EXPERIMENT_ID}] commit={record['environment']['git_commit_short']} "
          f"dirty={record['environment']['git_dirty']}")
    print(f"[{EXPERIMENT_ID}] registry_ok={v['registry_ok']} "
          f"{v['registry_checks']}")
    print(f"[{EXPERIMENT_ID}] cases={v['passed_cases']}/{v['total_cases']} pass; "
          f"valid_roundtrip={v['all_valid_roundtrip']} "
          f"negative_rejected={v['all_negative_rejected']}")
    print(f"[{EXPERIMENT_ID}] verdict={v['overall']}")
    print()
    hdr = f"{'contract':<4} {'case_kind':<16} {'exp':<7} {'act':<7} {'code':<28} {'path':<32} {'status':<5}"
    print(hdr)
    print("-" * len(hdr))
    for c in cases:
        print(
            f"{c['contract']:<4} {c['case_kind']:<16} {c['expected_verdict']:<7} "
            f"{c['actual_verdict']:<7} {c['error_code']:<28} "
            f"{c['error_path']:<32} {c['status']:<5}"
        )
    return 0 if v["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
