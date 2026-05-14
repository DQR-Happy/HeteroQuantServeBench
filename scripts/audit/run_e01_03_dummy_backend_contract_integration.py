#!/usr/bin/env python3
"""E01-03 — DummyBackend registration, execution, result & trace closed loop.

Question
--------
Can HQSB discover a C4-compliant backend through the *formal registry*,
confirm its capabilities, execute a :class:`WorkloadSpec` (C2) through the
*public benchmark engine*, and produce a verifiable :class:`BenchmarkResult`
(C6) plus correlated :class:`TraceEvent` (C7) — all without importing any
concrete Qwen loader or ML framework?

Hypothesis (falsifiable, pre-registered)
----------------------------------------
H1  The DummyBackend runs the full registry→C4→C2→C6/C7 chain end-to-end;
    duplicate-name / version conflicts and capability shortfalls are
    rejected explicitly; lifecycle failures are diagnosable and resource
    state is cleaned up; no Qwen/torch/transformers import is triggered.
H0  Somewhere the chain is bypassed, a conflict silently overwrites an
    entry, an unsupported request is silently ignored, or a failure yields
    a success result.

Design (protocol §6 steps 1–10)
-------------------------------
* dependency isolation: assert the Dummy path imports no Qwen/ML modules.
* registry register/discover + snapshots + lazy-factory check.
* end-to-end run: golden-token cross-check, call ledger, C6 + C7 trace.
* supported-parameter variation; unsupported-capability rejection.
* registry conflicts (same object / different factory / different version),
  unknown name, unregister, incompatible C4 version.
* capability fallback (forced) and lifecycle fault injection.

Pure CPU: no torch, no GPU, no model weights.

Raw output (under <out>/)
-------------------------
``e01_03_<run_id>.json``            full record (cases + extras + verdict)
``e01_03_<run_id>_env.json``        frozen environment / git identity
``E01-03_case_results.jsonl``       one JSON line per case
``E01-03_registry_snapshots.json``  before/after registry snapshots
``E01-03_capabilities.json``        declared capabilities
``E01-03_call_ledger.jsonl``        per-case call counts
``E01-03_traces.jsonl``             C7 trace events (run_id-correlated)
``results/<run_id>.json``           C6 BenchmarkResult (end-to-end)
``verdict.json``                    pass criteria + overall verdict

Usage
-----
    python3 scripts/audit/run_e01_03_dummy_backend_contract_integration.py \
        --output-dir docs/stage_experiments/S01/E01-03/raw
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hqsb.backends import DummyBackend, make_dummy_backend  # noqa: E402
from hqsb.benchmark.engine import BenchmarkEngine  # noqa: E402
from hqsb.core.contracts import (  # noqa: E402
    BackendCapability,
    ModelArtifact,
    WorkloadSpec,
)
from hqsb.core.errors import (  # noqa: E402
    BackendError,
    CapabilityError,
    DuplicateRegistrationError,
    RegistryLookupError,
    UnsupportedSchemaVersionError,
    exit_code_for,
)
from hqsb.core.ids import new_run_id  # noqa: E402
from hqsb.core.registry import Registry, RegistryHub  # noqa: E402

EXPERIMENT_ID = "E01-03"
STAGE = "S01"

#: Concrete-model / ML modules that must NOT be imported by the Dummy path.
_FORBIDDEN_IMPORTS = ("torch", "transformers", "modelscope", "qwen")

#: Independent golden source: reproduce DummyBackend's PRNG rule by hand so
#: the expected output never calls DummyBackend itself.
_TOKEN_VOCAB = 50_000


def golden_token_ids(seed: int, output_tokens: int) -> List[int]:
    """Hand-computed expected token sequence for the DummyBackend rule."""
    rng = random.Random(seed)
    return [rng.randrange(_TOKEN_VOCAB) for _ in range(output_tokens)]


def _git(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(_REPO_ROOT), capture_output=True, text=True
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except OSError:
        return ""


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


def _synthetic_artifact() -> ModelArtifact:
    """A synthetic fixture — explicitly NOT a real Qwen identity."""
    return ModelArtifact(
        model_id="fixture/dummy-synthetic-v1",
        source="local",
        architecture="DummyForCausalLM",
        dtype="float16",
    )


def _base_workload(**overrides: Any) -> WorkloadSpec:
    payload: Dict[str, Any] = {
        "name": "short",
        "input_tokens": 128,
        "output_tokens": 32,
        "seed": 42,
        "warmup": 1,
        "repetitions": 3,
    }
    payload.update(overrides)
    return WorkloadSpec(**payload)


# ── Case executors ──────────────────────────────────────────────────────


def _case(case_id: str, kind: str, expected: str) -> Dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "case_id": case_id,
        "case_kind": kind,
        "expected": expected,
    }


def _finish(rec: Dict[str, Any], status: str, **observed: Any) -> Dict[str, Any]:
    rec["status"] = status
    rec["observed"] = observed
    return rec


def case_dependency_isolation() -> Dict[str, Any]:
    rec = _case("dependency_isolation", "negative", "no Qwen/ML import")
    # Force the full Dummy path to import (registry → backend → engine).
    hub = RegistryHub()
    hub.backends.register("dummy", make_dummy_backend, version="1.0.0")
    backend = hub.backends.get("dummy")()
    BenchmarkEngine(backend)
    present = sorted(
        m for m in sys.modules if any(m.startswith(f) for f in _FORBIDDEN_IMPORTS)
    )
    return _finish(rec, "PASS" if not present else "FAIL", forbidden_imports=present)


def case_register_discover() -> Dict[str, Any]:
    rec = _case("register_discover", "positive", "exact entry added")
    hub = RegistryHub()
    before = list(hub.backends.names())
    hub.backends.register("dummy", make_dummy_backend, version="1.0.0")
    after = list(hub.backends.names())
    factory = hub.backends.get("dummy")
    is_lazy = callable(factory) and not isinstance(factory, DummyBackend)
    return _finish(
        rec,
        "PASS" if (before == [] and after == ["dummy"] and is_lazy) else "FAIL",
        snapshot_before=before,
        snapshot_after=after,
        entry_kind=type(factory).__name__,
        lazy_factory=is_lazy,
    )


def case_end_to_end() -> Dict[str, Any]:
    rec = _case("end_to_end_run", "positive", "full C6/C7 closed loop")
    backend = make_dummy_backend()
    artifact = _synthetic_artifact()
    workload = _base_workload()
    engine = BenchmarkEngine(backend)
    result = engine.run(workload, artifact=artifact)

    golden = golden_token_ids(workload.seed, workload.output_tokens)
    samples = result.raw_samples
    tokens_match = all(
        s["generated_token_ids"] == golden for s in samples
    )
    counts = backend.call_counts
    trace = backend.trace_events
    trace_ids = {e.trace_id for e in trace}
    event_types = [e.event_type.value for e in trace]
    has_parent = any(e.parent_span_id is not None for e in trace)
    ok = (
        result.schema_version == "1.0.0"
        and result.correctness.passed is True
        and len(samples) == workload.repetitions
        and tokens_match
        and counts == {"load": 1, "warmup": 1, "generate": 1, "close": 0}
        and len(trace_ids) == 1
        and "model_load" in event_types
        and "prefill" in event_types
        and event_types.count("decode") == workload.repetitions
        and "output" in event_types
        and has_parent
    )
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        run_id=result.run_id,
        config_hash=result.config_hash,
        model_artifact_hash=result.model_artifact_hash,
        correctness=result.correctness.passed,
        sample_count=len(samples),
        tokens_match=tokens_match,
        golden_tokens=golden,
        call_counts=counts,
        trace_event_types=event_types,
        trace_id_count=len(trace_ids),
        has_parent_span=has_parent,
        result=result.model_dump(mode="json"),
        trace_events=[e.model_dump(mode="json") for e in trace],
    )


def case_requested_actual_reason() -> Dict[str, Any]:
    rec = _case("requested_actual_reason", "positive", "requested == actual")
    hub = RegistryHub()
    hub.backends.register("dummy", make_dummy_backend, version="1.0.0")
    requested = "dummy"
    backend = hub.backends.get(requested)()
    cap = backend.capabilities()
    actual = backend.name
    reason = (
        f"registry lookup {requested!r} → factory instantiated; "
        f"capability check satisfied (max_batch={cap.max_batch}, "
        f"dtypes={cap.supported_dtypes})"
    )
    ok = requested == actual and backend.name == "dummy"
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        requested_backend=requested,
        actual_backend=actual,
        reason=reason,
        capability=cap.model_dump(mode="json"),
    )


def case_supported_variation() -> Dict[str, Any]:
    rec = _case("supported_variation", "positive", "params affect output")
    backend = make_dummy_backend()
    workload = _base_workload(seed=7, output_tokens=16, repetitions=1)
    result = BenchmarkEngine(backend).run(workload, artifact=_synthetic_artifact())
    got = result.raw_samples[0]["generated_token_ids"]
    golden = golden_token_ids(7, 16)
    return _finish(
        rec,
        "PASS" if (len(got) == 16 and got == golden) else "FAIL",
        seed=7,
        output_tokens=16,
        generated=got,
        golden=golden,
    )


def case_unsupported_capability() -> Dict[str, Any]:
    rec = _case("unsupported_capability", "negative", "explicit rejection")
    backend = make_dummy_backend()
    workload = _base_workload(batch_size=8)
    error_code = None
    try:
        BenchmarkEngine(backend).run(workload, artifact=_synthetic_artifact())
    except CapabilityError as exc:
        error_code = exit_code_for(exc)
    ok = error_code == 7 and backend.call_counts["generate"] == 0
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        requested_batch=8,
        max_batch=backend.capabilities().max_batch,
        error_code=error_code,
        generate_count=backend.call_counts["generate"],
    )


def case_streaming_unsupported() -> Dict[str, Any]:
    rec = _case("streaming_unsupported", "negative", "explicit rejection")
    backend = make_dummy_backend()
    workload = _base_workload()
    rejected = False
    try:
        backend.stream(workload, None)
    except NotImplementedError:
        rejected = True
    cap = backend.capabilities()
    ok = rejected and cap.streaming is False
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        streaming_declared=cap.streaming,
        stream_raised_notimplemented=rejected,
    )


def case_duplicate_same_object() -> Dict[str, Any]:
    rec = _case("duplicate_same_object", "positive", "idempotent re-register")
    hub = RegistryHub()
    hub.backends.register("dummy", make_dummy_backend, version="1.0.0")
    raised = False
    try:
        hub.backends.register("dummy", make_dummy_backend, version="1.0.0")
    except DuplicateRegistrationError:
        raised = True
    return _finish(
        rec,
        "PASS" if (not raised and len(hub.backends) == 1) else "FAIL",
        raised=raised,
        registry_size=len(hub.backends),
    )


def case_duplicate_diff_factory() -> Dict[str, Any]:
    rec = _case("duplicate_diff_factory", "negative", "conflict raised")
    hub = RegistryHub()
    hub.backends.register("dummy", make_dummy_backend, version="1.0.0")
    raised = False
    try:
        hub.backends.register(
            "dummy", lambda: DummyBackend(name="other"), version="1.0.0"
        )
    except DuplicateRegistrationError:
        raised = True
    still_original = hub.backends.get("dummy") is make_dummy_backend
    return _finish(
        rec,
        "PASS" if (raised and still_original) else "FAIL",
        raised=raised,
        original_preserved=still_original,
    )


def case_duplicate_diff_version() -> Dict[str, Any]:
    rec = _case("duplicate_diff_version", "negative", "conflict raised")
    hub = RegistryHub()
    hub.backends.register("dummy", make_dummy_backend, version="1.0.0")
    raised = False
    try:
        hub.backends.register(
            "dummy", make_dummy_backend, version="2.0.0"
        )
    except DuplicateRegistrationError:
        raised = True
    return _finish(
        rec,
        "PASS" if raised else "FAIL",
        raised=raised,
        policy="different object under same name rejected regardless of version",
    )


def case_unknown_name() -> Dict[str, Any]:
    rec = _case("unknown_name", "negative", "stable registry error")
    hub = RegistryHub()
    raised = False
    try:
        hub.backends.get("nope")
    except RegistryLookupError:
        raised = True
    return _finish(rec, "PASS" if raised else "FAIL", raised=raised)


def case_unregister() -> Dict[str, Any]:
    rec = _case("unregister", "positive", "removed state queryable")
    hub = RegistryHub()
    hub.backends.register("dummy", make_dummy_backend, version="1.0.0")
    hub.backends.unregister("dummy")
    gone = "dummy" not in hub.backends
    raised = False
    try:
        hub.backends.get("dummy")
    except RegistryLookupError:
        raised = True
    return _finish(
        rec,
        "PASS" if (gone and raised) else "FAIL",
        gone=gone,
        lookup_raises=raised,
    )


def case_incompatible_c4_version() -> Dict[str, Any]:
    rec = _case("incompatible_c4_version", "negative", "rejected pre-attach")
    raised = False
    try:
        BackendCapability(
            schema_version="2.0.0",
            name="dummy",
            supported_dtypes=["float16"],
        )
    except UnsupportedSchemaVersionError:
        raised = True
    return _finish(rec, "PASS" if raised else "FAIL", raised=raised)


def case_lifecycle_fault_load() -> Dict[str, Any]:
    rec = _case("lifecycle_fault_load", "negative", "diagnosable + no run")
    backend = make_dummy_backend(fail_at="load")
    raised = False
    try:
        BenchmarkEngine(backend).run(
            _base_workload(), artifact=_synthetic_artifact()
        )
    except BackendError:
        raised = True
    counts = backend.call_counts
    ok = raised and counts["warmup"] == 0 and counts["generate"] == 0
    return _finish(rec, "PASS" if ok else "FAIL", raised=raised, call_counts=counts)


def case_lifecycle_fault_generate() -> Dict[str, Any]:
    rec = _case("lifecycle_fault_generate", "negative", "failed status in trace")
    backend = make_dummy_backend(fail_at="generate")
    raised = False
    try:
        BenchmarkEngine(backend).run(
            _base_workload(), artifact=_synthetic_artifact()
        )
    except BackendError:
        raised = True
    counts = backend.call_counts
    names = [e.name or "" for e in backend.trace_events]
    has_failed = any("generate.failed" in n for n in names)
    ok = (
        raised
        and counts["load"] == 1
        and counts["warmup"] == 1
        and has_failed
    )
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        raised=raised,
        call_counts=counts,
        has_failed_event=has_failed,
    )


def case_lifecycle_fault_close() -> Dict[str, Any]:
    rec = _case("lifecycle_fault_close", "negative", "original error preserved")
    backend = make_dummy_backend(fail_at="close")
    backend.load(_synthetic_artifact())
    backend.warmup(_base_workload())
    output = backend.generate(_base_workload(), None)
    produced = len(output.samples)
    raised = False
    try:
        backend.close()
    except BackendError:
        raised = True
    ok = raised and produced > 0
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        raised=raised,
        samples_before_close=produced,
        closed_after_failure=backend.health(),
    )


def case_fresh_rerun() -> Dict[str, Any]:
    rec = _case("fresh_rerun", "positive", "success independent of prior state")
    # New registry, new backend, no shared state with previous cases.
    hub = RegistryHub()
    hub.backends.register("dummy", make_dummy_backend, version="1.0.0")
    backend = hub.backends.get("dummy")()
    result = BenchmarkEngine(backend).run(
        _base_workload(), artifact=_synthetic_artifact()
    )
    ok = result.correctness.passed is True
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        run_id=result.run_id,
        correctness=result.correctness.passed,
        registry_size=len(hub.backends),
    )


# ── Drivers ─────────────────────────────────────────────────────────────


_CASES = (
    case_dependency_isolation,
    case_register_discover,
    case_end_to_end,
    case_requested_actual_reason,
    case_supported_variation,
    case_unsupported_capability,
    case_streaming_unsupported,
    case_duplicate_same_object,
    case_duplicate_diff_factory,
    case_duplicate_diff_version,
    case_unknown_name,
    case_unregister,
    case_incompatible_c4_version,
    case_lifecycle_fault_load,
    case_lifecycle_fault_generate,
    case_lifecycle_fault_close,
    case_fresh_rerun,
)


def run_cases() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for fn in _CASES:
        try:
            cases.append(fn())
        except Exception as exc:  # noqa: BLE001 - record unexpected failures
            cases.append(
                _finish(
                    _case(fn.__name__, "unexpected", "no exception"),
                    "FAIL",
                    unexpected=type(exc).__name__,
                    message=str(exc)[:200],
                )
            )
    passed = sum(1 for c in cases if c["status"] == "PASS")
    verdict = {
        "total_cases": len(cases),
        "passed_cases": passed,
        "overall": "PASS" if passed == len(cases) else "FAIL",
    }
    return cases, verdict


def _write_traces(cases: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for c in cases:
            for ev in c.get("observed", {}).get("trace_events", []):
                fh.write(
                    json.dumps(
                        {"run_id": c["observed"].get("run_id"), "event": ev},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )


def _write_call_ledger(cases: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for c in cases:
            counts = c.get("observed", {}).get("call_counts")
            if counts:
                fh.write(
                    json.dumps(
                        {"case_id": c["case_id"], "call_counts": counts},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or new_run_id()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cases, verdict = run_cases()

    # Registry snapshots (recompute for the record).
    hub = RegistryHub()
    snap_before = list(hub.backends.names())
    hub.backends.register("dummy", make_dummy_backend, version="1.0.0")
    snap_after = list(hub.backends.names())

    record = {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "environment": collect_environment(),
        "registry_snapshots": {"before": snap_before, "after": snap_after},
        "verdict": verdict,
        "cases": cases,
    }

    (out_dir / f"e01_03_{run_id}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / f"e01_03_{run_id}_env.json").write_text(
        json.dumps(record["environment"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (out_dir / "E01-03_case_results.jsonl").open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n")
    (out_dir / "E01-03_registry_snapshots.json").write_text(
        json.dumps(record["registry_snapshots"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_traces(cases, out_dir / "E01-03_traces.jsonl")
    _write_call_ledger(cases, out_dir / "E01-03_call_ledger.jsonl")

    # C6 result for the end-to-end case.
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for c in cases:
        if c["case_id"] == "end_to_end_run" and "result" in c.get("observed", {}):
            rid = c["observed"]["run_id"]
            (results_dir / f"{rid}.json").write_text(
                json.dumps(c["observed"]["result"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    (out_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Console report.
    print(f"[{EXPERIMENT_ID}] run_id={run_id}")
    env = record["environment"]
    print(f"[{EXPERIMENT_ID}] commit={env['git_commit_short']} dirty={env['git_dirty']}")
    print(f"[{EXPERIMENT_ID}] cases={verdict['passed_cases']}/{verdict['total_cases']} pass")
    print(f"[{EXPERIMENT_ID}] verdict={verdict['overall']}")
    print()
    for c in cases:
        mark = "PASS" if c["status"] == "PASS" else "FAIL"
        print(f"  [{mark}] {c['case_id']:<32} ({c['case_kind']})")
    return 0 if verdict["overall"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E01-03 dummy backend closed loop.")
    parser.add_argument(
        "--output-dir",
        default="docs/stage_experiments/S01/E01-03/raw",
        help="Directory for raw JSON/JSONL artifacts.",
    )
    parser.add_argument("--run-id", default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
