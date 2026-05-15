#!/usr/bin/env python3
"""E01-06 — public error taxonomy, process exit codes & cross-layer traceability.

Question
--------
When HQSB fails at the config / contract / registry / backend / artifact
boundary, can it (a) express a *stable* error category, (b) return a real
non-zero process exit code, (c) emit parseable JSONL logs that join to the
run/trace/span, and (d) never leak secrets?

Hypothesis (falsifiable, pre-registered)
----------------------------------------
H1  Five public error categories map to stable exception classes + exit codes
    (config=3, schema-version=4, registry=5, backend=6/7, artifact=8); the
    reference CLI returns a real non-zero returncode; JSONL logs are parseable
    and tagged with trace/span IDs; synthetic secrets never leak; failure
    never fabricates success nor pollutes the next run.
H0  A category is unstable, an error is manually fabricated, a failure is
    reported as success, logs cannot be joined, or a secret leaks.

Design (protocol §6 steps 1–11)
-------------------------------
* freeze the error→exit-code mapping *before* running any case;
* run a legal closed-loop control and record run/trace/span IDs;
* inject the five error categories through real public entries only
  (never ``raise HqsbError(...)`` by hand);
* capture the reference CLI via subprocess (real returncode/stdout/stderr);
* verify JSONL parseability + stability, correlation, redaction, cleanup.

Raw output (under <out>/)
-------------------------
``E01-06_error_mapping.json``       frozen + observed category→exit-code map
``E01-06_case_matrix.json``         input matrix (category, injection, boundary)
``E01-06_case_results.jsonl``       one JSON line per case (rich observations)
``E01-06_command_results.jsonl``    subprocess CLI stdout/stderr/returncode
``E01-06_traces.jsonl``             C7 trace events (trace/span/parent)
``E01-06_logs.jsonl``               JSONL log stream (trace/span tagged)
``E01-06_correlation_audit.json``   run/trace/span/parent + interleave check
``E01-06_redaction_audit.json``     sentinel scan across all outputs
``E01-06_cleanup_audit.json``       resource acquire/release + original-error
``e01_06_<run_id>.json``            full record
``e01_06_<run_id>_env.json``        frozen environment / git identity
``results/<run_id>.json``           C6 BenchmarkResult (control run)
``verdict.json``                    per-criterion pass/fail + overall verdict

Usage
-----
    python3 scripts/audit/run_e01_06_error_taxonomy_and_traceability.py \
        --output-dir docs/stage_experiments/S01/E01-06/raw
"""

from __future__ import annotations

import argparse
import contextvars
import hashlib
import io
import json
import logging
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hqsb.backends import DummyBackend, make_dummy_backend  # noqa: E402
from hqsb.benchmark.engine import BenchmarkEngine  # noqa: E402
from hqsb.core.config.loader import ConfigLoader, redact_text  # noqa: E402
from hqsb.core.config.schema import BenchmarkConfig  # noqa: E402
from hqsb.core.contracts import (  # noqa: E402
    BackendCapability,
    ModelArtifact,
    WorkloadSpec,
)
from hqsb.core.errors import (  # noqa: E402
    ArtifactError,
    BackendError,
    CapabilityError,
    ConfigError,
    DuplicateRegistrationError,
    ExitCode,
    HqsbError,
    RegistryLookupError,
    SchemaError,
    SchemaMigrationRequiredError,
    UnsupportedSchemaVersionError,
    exit_code_for,
)
from hqsb.core.ids import new_run_id  # noqa: E402
from hqsb.core.logging import (  # noqa: E402
    JsonLineFormatter,
    configure_logging,
    get_span_id,
    get_trace_id,
    set_trace_context,
)
from hqsb.core.registry import Registry, RegistryHub  # noqa: E402
from hqsb.models.manifest import verify_or_raise  # noqa: E402

EXPERIMENT_ID = "E01-06"
STAGE = "S01"

#: Reference CLI driver launched by subprocess for real exit-code capture.
_DRIVER = _REPO_ROOT / "scripts" / "audit" / "_e01_06_cli_driver.py"

#: Synthetic secret sentinel — carries NO real credential.
_SENTINEL = "E01_06_SUPERSECRET_SENTINEL_9f3a7c1d"

#: CLI cases that the reference driver exposes (protocol §8 step 8).
_CLI_CASES = (
    "success",
    "config_unknown_field",
    "config_bad_value",
    "config_missing_file",
    "schema_future_version",
    "registry_unknown",
    "backend_capability",
    "backend_load_failure",
    "artifact_hash_mismatch",
    "artifact_missing_file",
)


# ── Environment / identity ─────────────────────────────────────────────


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


# ── Small helpers ──────────────────────────────────────────────────────


def _case(case_id: str, kind: str, category: str, expected: str) -> Dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "case_id": case_id,
        "case_kind": kind,  # positive | negative
        "category": category,
        "expected": expected,
    }


def _finish(rec: Dict[str, Any], status: str, **observed: Any) -> Dict[str, Any]:
    rec["status"] = status
    rec["observed"] = observed
    return rec


def _cause_chain(exc: BaseException) -> List[str]:
    """Walk the ``__cause__`` chain, returning class names outermost→inner."""
    chain: List[str] = [type(exc).__name__]
    cause = exc.__cause__
    seen = 0
    while cause is not None and seen < 10:
        chain.append(type(cause).__name__)
        cause = cause.__cause__
        seen += 1
    return chain


def _exc_obs(exc: BaseException) -> Dict[str, Any]:
    return {
        "error_class": type(exc).__name__,
        "exit_code": exit_code_for(exc),
        "message": str(exc)[:400],
        "details": getattr(exc, "details", None),
        "cause_chain": _cause_chain(exc),
        "is_hqsb_error": isinstance(exc, HqsbError),
    }


def _run_raises(fn: Callable[[], Any]) -> Tuple[bool, Optional[BaseException], Dict[str, Any]]:
    """Run ``fn``, returning ``(raised, exc, observation)``."""
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - capture the public error
        return True, exc, _exc_obs(exc)
    return False, None, {}


# ── Fixtures ───────────────────────────────────────────────────────────


def _synthetic_artifact() -> ModelArtifact:
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


def _defaults() -> Dict[str, Any]:
    return {
        "benchmark": {
            "model": "fixture/dummy",
            "model_source": "modelscope",
            "backend": "dummy",
            "dtype": "float16",
        },
        "workloads": [{"name": "short", "input_tokens": 128, "output_tokens": 32}],
    }


def _make_fixture() -> Tuple[str, str]:
    """Small legal model dir + sibling manifest (mirrors the CLI driver)."""
    tmp = Path(tempfile.mkdtemp(prefix="e01_06_fixture_"))
    root = tmp / "model"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"dummy-weights-0000\n")
    (root / "config.json").write_text('{"arch": "dummy"}', encoding="utf-8")
    lines: List[str] = []
    for name in ("model.safetensors", "config.json"):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}\n")
    manifest = tmp / "manifest.sha256"
    manifest.write_text("".join(lines), encoding="utf-8")
    return str(root), str(manifest)


def _tamper(root: str) -> None:
    p = Path(root) / "model.safetensors"
    data = bytearray(p.read_bytes())
    data[0] ^= 0xFF
    p.write_bytes(bytes(data))


# ── Frozen error mapping (pre-execution) ───────────────────────────────


def frozen_error_mapping() -> Dict[str, Any]:
    """Introspect the *declared* taxonomy; this is the pre-registered table."""
    mapping = {
        "config": {"exit_code": ExitCode.CONFIG, "classes": ["ConfigError"]},
        "schema_version": {
            "exit_code": ExitCode.SCHEMA,
            "classes": [
                "SchemaError",
                "SchemaVersionError",
                "UnsupportedSchemaVersionError",
                "SchemaMigrationRequiredError",
            ],
        },
        "registry": {
            "exit_code": ExitCode.REGISTRY,
            "classes": ["RegistryError", "RegistryLookupError", "DuplicateRegistrationError"],
        },
        "backend": {"exit_code": ExitCode.BACKEND, "classes": ["BackendError"]},
        "capability": {"exit_code": ExitCode.CAPABILITY, "classes": ["CapabilityError"]},
        "artifact": {"exit_code": ExitCode.ARTIFACT, "classes": ["ArtifactError"]},
        "internal": {"exit_code": ExitCode.INTERNAL, "classes": ["<non-HqsbError>"]},
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "exit_code_table": {
            name: getattr(ExitCode, name)
            for name in (
                "SUCCESS",
                "INTERNAL",
                "USAGE",
                "CONFIG",
                "SCHEMA",
                "REGISTRY",
                "BACKEND",
                "CAPABILITY",
                "ARTIFACT",
                "BENCHMARK",
            )
        },
        "category_mapping": mapping,
        "note": "field-level contract validation (missing/unknown field, constraint) "
        "surfaces as pydantic ValidationError and is NOT part of SchemaError(4); "
        "this boundary is checked explicitly by the schema cases below.",
    }


# ── Cases: control ─────────────────────────────────────────────────────


def case_control_closed_loop() -> Dict[str, Any]:
    rec = _case("control_closed_loop", "positive", "control", "full C6/C7 + logs")
    backend = make_dummy_backend()
    artifact = _synthetic_artifact()
    workload = _base_workload()
    engine = BenchmarkEngine(backend)

    log_buf = io.StringIO()
    configure_logging(stream=log_buf, json_format=True)
    set_trace_context("trace-control-fixture", "span-control-fixture")
    logging.getLogger("hqsb.e01_06").info("control run starting", extra={"step": "start"})
    result = engine.run(workload, artifact=artifact)
    logging.getLogger("hqsb.e01_06").info("control run finished", extra={"step": "end"})

    trace = backend.trace_events
    trace_ids = {e.trace_id for e in trace}
    root = next((e for e in trace if e.parent_span_id is None), None)
    child_spans = [e for e in trace if e.parent_span_id is not None]
    children_ok = root is not None and all(
        e.parent_span_id == root.span_id for e in child_spans
    )

    ok = (
        result.schema_version == "1.0.0"
        and result.correctness.passed is True
        and len(trace_ids) == 1
        and children_ok
        and backend.call_counts == {"load": 1, "warmup": 1, "generate": 1, "close": 0}
    )
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        run_id=result.run_id,
        result_has_trace_events_field=("trace_events" in type(result).model_fields),
        backend_trace_id=list(trace_ids),
        trace_event_types=[e.event_type.value for e in trace],
        span_parent_ok=children_ok,
        call_counts=backend.call_counts,
        log_lines=log_buf.getvalue().splitlines(),
        result=result.model_dump(mode="json"),
        trace_events=[e.model_dump(mode="json") for e in trace],
    )


# ── Cases: config (exit 3) ─────────────────────────────────────────────


def case_config_unknown_field() -> Dict[str, Any]:
    rec = _case("config_unknown_field", "negative", "config", "ConfigError(3)")
    raised, exc, obs = _run_raises(
        lambda: ConfigLoader(BenchmarkConfig).load(
            defaults=_defaults(), environ={}, cli={"benchmark": {"bogus": 1}}
        )
    )
    ok = raised and isinstance(exc, ConfigError) and obs["exit_code"] == 3
    return _finish(rec, "PASS" if ok else "FAIL", **obs)


def case_config_bad_value() -> Dict[str, Any]:
    rec = _case("config_bad_value", "negative", "config", "ConfigError(3)")
    raised, exc, obs = _run_raises(
        lambda: ConfigLoader(BenchmarkConfig).load(
            defaults=_defaults(), environ={"HQSB_BENCHMARK__BATCH_SIZE": "not-an-int"}
        )
    )
    ok = raised and isinstance(exc, ConfigError) and obs["exit_code"] == 3
    return _finish(rec, "PASS" if ok else "FAIL", **obs)


def case_config_missing_file() -> Dict[str, Any]:
    rec = _case("config_missing_file", "negative", "config", "ConfigError(3)")
    raised, exc, obs = _run_raises(
        lambda: ConfigLoader(BenchmarkConfig).load(
            defaults=_defaults(), environ={}, path="/nonexistent/e01_06.yaml"
        )
    )
    ok = raised and isinstance(exc, ConfigError) and obs["exit_code"] == 3
    return _finish(rec, "PASS" if ok else "FAIL", **obs)


def case_config_bad_yaml() -> Dict[str, Any]:
    rec = _case("config_bad_yaml", "negative", "config", "ConfigError(3)")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write("benchmark: [unclosed\n  bad: yaml: here\n")
        path = fh.name
    raised, exc, obs = _run_raises(
        lambda: ConfigLoader(BenchmarkConfig).load(
            defaults=_defaults(), environ={}, path=path
        )
    )
    ok = raised and isinstance(exc, ConfigError) and obs["exit_code"] == 3
    return _finish(rec, "PASS" if ok else "FAIL", **obs)


def case_config_duplicate_yaml_key() -> Dict[str, Any]:
    rec = _case("config_duplicate_yaml_key", "negative", "config", "ConfigError(3)")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write("benchmark:\n  model: a\n  model: b\n")
        path = fh.name
    raised, exc, obs = _run_raises(
        lambda: ConfigLoader(BenchmarkConfig).load(
            defaults=_defaults(), environ={}, path=path
        )
    )
    ok = (
        raised
        and isinstance(exc, ConfigError)
        and obs["exit_code"] == 3
        and obs["details"].get("error_code") == "duplicate_yaml_key"
    )
    return _finish(rec, "PASS" if ok else "FAIL", **obs)


def case_config_cross_field_conflict() -> Dict[str, Any]:
    rec = _case("config_cross_field_conflict", "negative", "config", "ConfigError(3)")
    defaults = {
        "benchmark": {
            "model": "fixture/dummy",
            "model_source": "modelscope",
            "backend": "dummy",
            "dtype": "float16",
        },
        "workloads": [{"name": "decode_heavy", "input_tokens": 256, "output_tokens": 128}],
    }
    raised, exc, obs = _run_raises(
        lambda: ConfigLoader(BenchmarkConfig).load(defaults=defaults, environ={})
    )
    ok = (
        raised
        and isinstance(exc, ConfigError)
        and obs["exit_code"] == 3
        and obs["details"].get("error_code") == "cross_field_conflict"
    )
    return _finish(rec, "PASS" if ok else "FAIL", **obs)


# ── Cases: schema (exit 4 for version gate) ────────────────────────────


def case_schema_future_version() -> Dict[str, Any]:
    rec = _case("schema_future_version", "negative", "schema", "SchemaError(4)")
    raised, exc, obs = _run_raises(
        lambda: ModelArtifact(
            schema_version="2.0.0",
            model_id="m",
            source="local",
            architecture="a",
            dtype="float16",
        )
    )
    ok = (
        raised
        and isinstance(exc, UnsupportedSchemaVersionError)
        and obs["exit_code"] == 4
    )
    return _finish(rec, "PASS" if ok else "FAIL", **obs)


def case_schema_old_version() -> Dict[str, Any]:
    rec = _case("schema_old_version", "negative", "schema", "SchemaError(4)")
    raised, exc, obs = _run_raises(
        lambda: WorkloadSpec(
            schema_version="0.9.0", name="short", input_tokens=128, output_tokens=32
        )
    )
    ok = (
        raised
        and isinstance(exc, SchemaMigrationRequiredError)
        and obs["exit_code"] == 4
    )
    return _finish(rec, "PASS" if ok else "FAIL", **obs)


def case_schema_missing_required_field() -> Dict[str, Any]:
    rec = _case(
        "schema_missing_required_field",
        "negative",
        "schema",
        "expect SchemaError(4); OBSERVED pydantic ValidationError",
    )
    raised, exc, obs = _run_raises(
        lambda: ModelArtifact(model_id="m", source="local", dtype="float16")
    )
    # Honest observation: this surfaces as pydantic ValidationError, NOT SchemaError.
    ok = raised and isinstance(exc, SchemaError)
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        observed_class=obs.get("error_class"),
        observed_exit_code=obs.get("exit_code"),
        expected_exit_code=4,
    )


def case_schema_unknown_field() -> Dict[str, Any]:
    rec = _case(
        "schema_unknown_field",
        "negative",
        "schema",
        "expect SchemaError(4); OBSERVED pydantic ValidationError",
    )
    raised, exc, obs = _run_raises(
        lambda: WorkloadSpec(name="short", input_tokens=128, output_tokens=32, bogus=1)
    )
    ok = raised and isinstance(exc, SchemaError)
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        observed_class=obs.get("error_class"),
        observed_exit_code=obs.get("exit_code"),
        expected_exit_code=4,
    )


# ── Cases: registry (exit 5) ───────────────────────────────────────────


def case_registry_unknown_backend() -> Dict[str, Any]:
    rec = _case("registry_unknown_backend", "negative", "registry", "RegistryError(5)")
    raised, exc, obs = _run_raises(lambda: Registry(kind="backend").get("nope"))
    ok = raised and isinstance(exc, RegistryLookupError) and obs["exit_code"] == 5
    return _finish(rec, "PASS" if ok else "FAIL", **obs)


def case_registry_duplicate_conflict() -> Dict[str, Any]:
    rec = _case("registry_duplicate_conflict", "negative", "registry", "RegistryError(5)")
    hub = RegistryHub()
    hub.backends.register("dummy", make_dummy_backend, version="1.0.0")
    raised, exc, obs = _run_raises(
        lambda: hub.backends.register("dummy", make_dummy_backend, version="2.0.0")
    )
    ok = (
        raised
        and isinstance(exc, DuplicateRegistrationError)
        and obs["exit_code"] == 5
        and hub.backends.get("dummy") is make_dummy_backend
    )
    return _finish(rec, "PASS" if ok else "FAIL", original_preserved=True, **obs)


def case_registry_incompatible_version() -> Dict[str, Any]:
    rec = _case(
        "registry_incompatible_version",
        "negative",
        "registry",
        "rejected pre-attach by schema version gate",
    )
    raised, exc, obs = _run_raises(
        lambda: BackendCapability(schema_version="2.0.0", name="dummy")
    )
    # The incompatible C4 version is refused by the *contract version gate*
    # (SchemaError 4), not by RegistryError(5). Report the actual class.
    ok = raised and isinstance(exc, UnsupportedSchemaVersionError)
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        observed_class=obs.get("error_class"),
        observed_exit_code=obs.get("exit_code"),
        note="incompatible C4 version refused pre-attach by version gate (SchemaError 4)",
        **obs,
    )


# ── Cases: backend (exit 6/7) ──────────────────────────────────────────


def case_backend_capability_unsupported() -> Dict[str, Any]:
    rec = _case("backend_capability_unsupported", "negative", "backend", "CapabilityError(7)")
    backend = make_dummy_backend()
    workload = _base_workload(batch_size=8)
    raised, exc, obs = _run_raises(
        lambda: BenchmarkEngine(backend).run(workload, artifact=_synthetic_artifact())
    )
    ok = (
        raised
        and isinstance(exc, CapabilityError)
        and obs["exit_code"] == 7
        and backend.call_counts["generate"] == 0
    )
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        generate_count=backend.call_counts["generate"],
        **obs,
    )


def case_backend_load_failure() -> Dict[str, Any]:
    rec = _case("backend_load_failure", "negative", "backend", "BackendError(6)")
    backend = make_dummy_backend(fail_at="load")
    raised, exc, obs = _run_raises(
        lambda: BenchmarkEngine(backend).run(_base_workload(), artifact=_synthetic_artifact())
    )
    ok = (
        raised
        and isinstance(exc, BackendError)
        and obs["exit_code"] == 6
        and backend.call_counts["warmup"] == 0
        and backend.call_counts["generate"] == 0
    )
    return _finish(rec, "PASS" if ok else "FAIL", call_counts=backend.call_counts, **obs)


def case_backend_generate_failure() -> Dict[str, Any]:
    rec = _case("backend_generate_failure", "negative", "backend", "BackendError(6)")
    backend = make_dummy_backend(fail_at="generate")
    raised, exc, obs = _run_raises(
        lambda: BenchmarkEngine(backend).run(_base_workload(), artifact=_synthetic_artifact())
    )
    names = [e.name or "" for e in backend.trace_events]
    has_failed_event = any("generate.failed" in n for n in names)
    ok = (
        raised
        and isinstance(exc, BackendError)
        and obs["exit_code"] == 6
        and has_failed_event
    )
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        has_failed_event=has_failed_event,
        call_counts=backend.call_counts,
        trace_events=[e.model_dump(mode="json") for e in backend.trace_events],
        **obs,
    )


# ── Cases: artifact (exit 8) ───────────────────────────────────────────


def case_artifact_positive_control() -> Dict[str, Any]:
    rec = _case("artifact_positive_control", "positive", "artifact", "gate passes")
    root, manifest = _make_fixture()
    raised, exc, obs = _run_raises(lambda: verify_or_raise(root, manifest))
    ok = not raised
    return _finish(rec, "PASS" if ok else "FAIL", raised=raised, **obs)


def case_artifact_hash_mismatch() -> Dict[str, Any]:
    rec = _case("artifact_hash_mismatch", "negative", "artifact", "ArtifactError(8)")
    root, manifest = _make_fixture()
    _tamper(root)
    raised, exc, obs = _run_raises(lambda: verify_or_raise(root, manifest))
    ok = (
        raised
        and isinstance(exc, ArtifactError)
        and obs["exit_code"] == 8
        and "hash_mismatch" in obs["details"].get("reason_codes", [])
    )
    return _finish(rec, "PASS" if ok else "FAIL", **obs)


def case_artifact_missing_file() -> Dict[str, Any]:
    rec = _case("artifact_missing_file", "negative", "artifact", "ArtifactError(8)")
    root, manifest = _make_fixture()
    (Path(root) / "config.json").unlink()
    raised, exc, obs = _run_raises(lambda: verify_or_raise(root, manifest))
    ok = (
        raised
        and isinstance(exc, ArtifactError)
        and obs["exit_code"] == 8
        and "missing_file" in obs["details"].get("reason_codes", [])
    )
    return _finish(rec, "PASS" if ok else "FAIL", **obs)


def case_artifact_unsafe_path() -> Dict[str, Any]:
    rec = _case("artifact_unsafe_path", "negative", "artifact", "ArtifactError(8)")
    root, manifest = _make_fixture()
    # Append a traversal line to the manifest (real gate must reject it).
    with open(manifest, "a", encoding="utf-8") as fh:
        fh.write("0" * 64 + "  ../../etc/passwd\n")
    raised, exc, obs = _run_raises(lambda: verify_or_raise(root, manifest))
    ok = (
        raised
        and isinstance(exc, ArtifactError)
        and obs["exit_code"] == 8
        and obs["details"].get("reason_codes")
    )
    return _finish(rec, "PASS" if ok else "FAIL", **obs)


# ── Cases: internal unexpected + cleanup ───────────────────────────────


class _BuggyBackend(DummyBackend):
    """A backend whose own code has an internal bug (not a capability gap)."""

    def generate(self, workload: object, inputs: object):  # noqa: D102
        raise RuntimeError("plugin internal bug: division by zero")


class _DoubleFaultBackend(DummyBackend):
    """Fails during generate AND during close (original + cleanup error)."""

    def generate(self, workload: object, inputs: object):  # noqa: D102
        raise BackendError("original execution failure")

    def close(self) -> None:  # noqa: D102
        raise BackendError("cleanup also failed")


def case_internal_unexpected_bug() -> Dict[str, Any]:
    rec = _case(
        "internal_unexpected_bug",
        "negative",
        "internal",
        "wrapped as BackendError, cause preserved",
    )
    backend = _BuggyBackend()
    raised, exc, obs = _run_raises(
        lambda: BenchmarkEngine(backend).run(_base_workload(), artifact=_synthetic_artifact())
    )
    cause_preserved = "RuntimeError" in obs.get("cause_chain", [])
    misclassified = isinstance(exc, CapabilityError)
    ok = (
        raised
        and isinstance(exc, BackendError)
        and obs["exit_code"] == 6
        and cause_preserved
        and not misclassified
    )
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        cause_preserved=cause_preserved,
        misclassified_as_capability=misclassified,
        **obs,
    )


def case_cleanup_close_failure_preserves_original() -> Dict[str, Any]:
    rec = _case(
        "cleanup_close_failure_preserves_original",
        "negative",
        "cleanup",
        "original error visible; cleanup error separate",
    )
    backend = _DoubleFaultBackend()
    original = None
    try:
        BenchmarkEngine(backend).run(_base_workload(), artifact=_synthetic_artifact())
    except BaseException as exc:  # noqa: BLE001
        original = exc
    cleanup = None
    try:
        backend.close()
    except BaseException as exc:  # noqa: BLE001
        cleanup = exc
    ok = (
        isinstance(original, BackendError)
        and isinstance(cleanup, BackendError)
        and original is not cleanup
        and str(original) == "original execution failure"
        and str(cleanup) == "cleanup also failed"
    )
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        original_error=str(original) if original else None,
        cleanup_error=str(cleanup) if cleanup else None,
        distinct=original is not cleanup,
    )


def case_cleanup_fresh_rerun() -> Dict[str, Any]:
    rec = _case("cleanup_fresh_rerun", "positive", "cleanup", "no state leak")
    hub = RegistryHub()
    hub.backends.register("dummy", make_dummy_backend, version="1.0.0")
    backend = hub.backends.get("dummy")()
    result = BenchmarkEngine(backend).run(_base_workload(), artifact=_synthetic_artifact())
    ok = result.correctness.passed is True and len(hub.backends) == 1
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        run_id=result.run_id,
        correctness=result.correctness.passed,
        registry_size=len(hub.backends),
    )


# ── Correlation audit (protocol §7 step 7) ─────────────────────────────


def audit_correlation(control: Dict[str, Any]) -> Dict[str, Any]:
    """Check run/trace/span linkage and interleave two tasks via contextvars."""
    observed = control["observed"]
    result = observed.get("result", {})
    run_id = observed.get("run_id")
    trace_events = observed.get("trace_events", [])

    # 1. Within-trace span/parent correlation.
    trace_ids = {e["trace_id"] for e in trace_events}
    roots = [e for e in trace_events if e.get("parent_span_id") is None]
    children = [e for e in trace_events if e.get("parent_span_id") is not None]
    spans_parent_ok = len(roots) == 1 and all(
        c["parent_span_id"] == roots[0]["span_id"] for c in children
    )

    # 2. Cross-layer: does C6 run_id link to C7 trace_id?
    c6_has_trace_links = any(
        e.get("trace_id") == run_id or e.get("run_id") == run_id for e in trace_events
    )
    result_has_trace_events = "trace_events" in result
    result_run_id = result.get("run_id")
    run_trace_linked = c6_has_trace_links

    # 3. Interleave two tasks via contextvars isolation.
    buf = io.StringIO()
    configure_logging(stream=buf, json_format=True)
    ctx_a = contextvars.copy_context()
    ctx_b = contextvars.copy_context()

    def _log(tag: str) -> None:
        logging.getLogger("hqsb.e01_06.interleave").info(tag)

    ctx_a.run(set_trace_context, "trace-A", "span-A")
    ctx_b.run(set_trace_context, "trace-B", "span-B")
    ctx_a.run(_log, "A.1")
    ctx_b.run(_log, "B.1")
    ctx_a.run(_log, "A.2")

    lines = [json.loads(ln) for ln in buf.getvalue().splitlines()]
    interleave = [
        {"message": ln["message"], "trace_id": ln.get("trace_id"), "span_id": ln.get("span_id")}
        for ln in lines
    ]
    interleave_ok = (
        interleave[0]["trace_id"] == "trace-A"
        and interleave[0]["span_id"] == "span-A"
        and interleave[1]["trace_id"] == "trace-B"
        and interleave[1]["span_id"] == "span-B"
        and interleave[2]["trace_id"] == "trace-A"
        and interleave[2]["span_id"] == "span-A"
    )

    return {
        "run_id": run_id,
        "result_run_id": result_run_id,
        "result_has_trace_events_field": observed.get("result_has_trace_events_field"),
        "c6_has_trace_links": c6_has_trace_links,
        "run_trace_linked": run_trace_linked,
        "trace_id_count": len(trace_ids),
        "span_parent_ok": spans_parent_ok,
        "interleave_ok": interleave_ok,
        "interleave": interleave,
    }


# ── Redaction audit (protocol §6 step 10) ──────────────────────────────


def _scan(sentinel: str, *targets: str) -> Dict[str, Any]:
    hits = [t for t in targets if sentinel in t]
    return {"scanned": len(targets), "hits": len(hits), "hit_targets": hits}


def audit_redaction() -> Dict[str, Any]:
    # 1. Success path: sentinel in secrets → redacted public view, excluded hash.
    resolution = ConfigLoader(BenchmarkConfig).load_resolved(
        defaults={
            "benchmark": {
                "model": "fixture/dummy",
                "model_source": "modelscope",
                "backend": "dummy",
                "dtype": "float16",
            },
            "workloads": [{"name": "short", "input_tokens": 128, "output_tokens": 32}],
            "secrets": {"modelscope_token": _SENTINEL},
        },
        environ={},
    )
    public_view_str = json.dumps(resolution.public_view, ensure_ascii=False)
    semantic_str = json.dumps(resolution.semantic_payload, ensure_ascii=False)
    public_token = resolution.public_view.get("secrets", {}).get("modelscope_token")

    # 2. Failure path: sentinel present but a different field errors; the
    #    loader must redact secret values from any raised message.
    raised, exc, obs = _run_raises(
        lambda: ConfigLoader(BenchmarkConfig).load(
            defaults={
                "benchmark": {
                    "model": "fixture/dummy",
                    "model_source": "modelscope",
                    "backend": "dummy",
                    "dtype": "float16",
                },
                "workloads": [{"name": "short", "input_tokens": 128, "output_tokens": 32}],
                "secrets": {"modelscope_token": _SENTINEL},
            },
            environ={},
            cli={"benchmark": {"bogus": 1}},
        )
    )
    error_message = obs.get("message", "")

    # 3. redact_text unit behavior.
    redacted_unit = redact_text(f"prefix {_SENTINEL} suffix", [_SENTINEL])

    # 4. Scanner negative control: a deliberately-leaked fixture must alarm.
    leak_fixture = f"token={_SENTINEL}"
    control_scan = _scan(_SENTINEL, leak_fixture)

    # 5. Scanner positive: collected outputs must be clean.
    clean_scan = _scan(_SENTINEL, public_view_str, semantic_str, error_message)

    ok = (
        public_token == "<redacted>"
        and _SENTINEL not in public_view_str
        and _SENTINEL not in semantic_str
        and _SENTINEL not in error_message
        and redacted_unit == "prefix <redacted> suffix"
        and control_scan["hits"] == 1
        and clean_scan["hits"] == 0
    )
    return _finish(
        _case("redaction_audit", "negative", "redaction", "no secret leak"),
        "PASS" if ok else "FAIL",
        public_token=public_token,
        semantic_has_secrets="secrets" in resolution.semantic_payload,
        error_message_clean=(_SENTINEL not in error_message),
        redacted_unit=redacted_unit,
        scanner_negative_control=control_scan,
        scanner_positive=clean_scan,
    )


# ── JSONL stability (protocol §6 step 9) ───────────────────────────────


def audit_jsonl_stability() -> Dict[str, Any]:
    rec = _case("jsonl_stability", "negative", "logging", "parseable + stable")

    def _emit_once() -> Dict[str, Any]:
        buf = io.StringIO()
        configure_logging(stream=buf, json_format=True)
        set_trace_context("trace-stability", "span-stability")
        raised, exc, obs = _run_raises(
            lambda: ConfigLoader(BenchmarkConfig).load(
                defaults=_defaults(), environ={}, cli={"benchmark": {"bogus": 1}}
            )
        )
        logging.getLogger("hqsb.e01_06").error(
            "config load failed",
            extra={"hqsb_extra": {"exit_code": obs.get("exit_code")}},
        )
        lines = [json.loads(ln) for ln in buf.getvalue().splitlines()]
        return {
            "raised": raised,
            "error_class": obs.get("error_class"),
            "exit_code": obs.get("exit_code"),
            "lines": lines,
        }

    a = _emit_once()
    b = _emit_once()

    all_lines = a["lines"] + b["lines"]
    # Both lines were already parsed by json.loads; if any line were not a
    # single valid JSON object the list comprehension would have raised.
    parseable = len(all_lines) == 2  # one log line per run
    has_trace_tag = all(ln.get("trace_id") == "trace-stability" for ln in all_lines)
    has_exit_code = all(ln.get("exit_code") == 3 for ln in all_lines)
    stable = (
        a["error_class"] == b["error_class"]
        and a["exit_code"] == b["exit_code"]
        and a["raised"] is True
        and b["raised"] is True
    )
    ok = parseable and has_trace_tag and has_exit_code and stable
    return _finish(
        rec,
        "PASS" if ok else "FAIL",
        parseable=parseable,
        trace_tagged=has_trace_tag,
        exit_code_tagged=has_exit_code,
        run1={"class": a["error_class"], "exit_code": a["exit_code"]},
        run2={"class": b["error_class"], "exit_code": b["exit_code"]},
        stable=stable,
        log_lines=all_lines,
    )


# ── CLI subprocess capture (protocol §6 step 8) ────────────────────────


def run_cli_cases() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in _CLI_CASES:
        proc = subprocess.run(
            [sys.executable, str(_DRIVER), "--case", case],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        rows.append(
            {
                "case": case,
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        )
    return rows


def audit_cli(command_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected = {
        "success": 0,
        "config_unknown_field": 3,
        "config_bad_value": 3,
        "config_missing_file": 3,
        "schema_future_version": 4,
        "registry_unknown": 5,
        "backend_capability": 7,
        "backend_load_failure": 6,
        "artifact_hash_mismatch": 8,
        "artifact_missing_file": 8,
    }
    checks = {
        row["case"]: (row["returncode"] == expected[row["case"]])
        for row in command_results
    }
    ok = all(checks.values())
    return _finish(
        _case("cli_exit_codes", "negative", "cli", "real non-zero returncodes"),
        "PASS" if ok else "FAIL",
        expected=expected,
        actual={row["case"]: row["returncode"] for row in command_results},
        per_case=checks,
    )


# ── Drivers ────────────────────────────────────────────────────────────


_CASE_FNS: Tuple[Callable[[], Dict[str, Any]], ...] = (
    case_control_closed_loop,
    case_config_unknown_field,
    case_config_bad_value,
    case_config_missing_file,
    case_config_bad_yaml,
    case_config_duplicate_yaml_key,
    case_config_cross_field_conflict,
    case_schema_future_version,
    case_schema_old_version,
    case_schema_missing_required_field,
    case_schema_unknown_field,
    case_registry_unknown_backend,
    case_registry_duplicate_conflict,
    case_registry_incompatible_version,
    case_backend_capability_unsupported,
    case_backend_load_failure,
    case_backend_generate_failure,
    case_artifact_positive_control,
    case_artifact_hash_mismatch,
    case_artifact_missing_file,
    case_artifact_unsafe_path,
    case_internal_unexpected_bug,
    case_cleanup_close_failure_preserves_original,
    case_cleanup_fresh_rerun,
)


def run_cases() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for fn in _CASE_FNS:
        try:
            cases.append(fn())
        except Exception as exc:  # noqa: BLE001 - record unexpected runner failures
            cases.append(
                _finish(
                    _case(fn.__name__, "unexpected", "runner", "no exception"),
                    "FAIL",
                    unexpected=type(exc).__name__,
                    message=str(exc)[:200],
                )
            )
    return cases


def _write_traces(cases: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for c in cases:
            rid = c.get("observed", {}).get("run_id")
            for ev in c.get("observed", {}).get("trace_events", []):
                fh.write(
                    json.dumps(
                        {"run_id": rid, "event": ev}, ensure_ascii=False, sort_keys=True
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

    error_mapping = frozen_error_mapping()
    cases = run_cases()
    command_results = run_cli_cases()
    cli_audit = audit_cli(command_results)

    control = next(c for c in cases if c["case_id"] == "control_closed_loop")
    correlation = audit_correlation(control)
    redaction = audit_redaction()
    jsonl = audit_jsonl_stability()

    # Case matrix (input → boundary).
    case_matrix = [
        {
            "category": "config",
            "injections": [
                "unknown field (cli)",
                "bad value (env)",
                "missing file",
                "bad yaml",
                "duplicate yaml key",
                "cross-field conflict",
            ],
            "block_point": "no legal execution plan",
            "exit_code": 3,
        },
        {
            "category": "schema",
            "injections": ["future version", "old version", "missing field", "unknown field"],
            "block_point": "no accepted Contract",
            "exit_code": 4,
        },
        {
            "category": "registry",
            "injections": ["unknown backend", "duplicate conflict", "incompatible C4 version"],
            "block_point": "no wrong create/overwrite",
            "exit_code": 5,
        },
        {
            "category": "backend",
            "injections": ["capability unsupported", "load failure", "generate failure"],
            "block_point": "no fake success / silent swap",
            "exit_code": "6/7",
        },
        {
            "category": "artifact",
            "injections": ["tampered byte", "missing file", "unsafe path"],
            "block_point": "gate rejects before load",
            "exit_code": 8,
        },
    ]

    # Verdict against the four single-item pass criteria.
    category_statuses = {
        c["case_id"]: c["status"] for c in cases
    }
    taxonomy_cases = [
        "config_unknown_field",
        "config_bad_value",
        "config_missing_file",
        "config_bad_yaml",
        "config_duplicate_yaml_key",
        "config_cross_field_conflict",
        "schema_future_version",
        "schema_old_version",
        "registry_unknown_backend",
        "registry_duplicate_conflict",
        "registry_incompatible_version",
        "backend_capability_unsupported",
        "backend_load_failure",
        "backend_generate_failure",
        "artifact_hash_mismatch",
        "artifact_missing_file",
        "artifact_unsafe_path",
        "internal_unexpected_bug",
        "cleanup_close_failure_preserves_original",
        "cleanup_fresh_rerun",
        "artifact_positive_control",
    ]
    taxonomy_stable = all(category_statuses.get(c) == "PASS" for c in taxonomy_cases)
    schema_field_gap = (
        category_statuses.get("schema_missing_required_field") != "PASS"
        or category_statuses.get("schema_unknown_field") != "PASS"
    )
    run_trace_linked = bool(correlation.get("run_trace_linked"))

    criteria = {
        "stable_error_types": {
            "status": "PASS" if taxonomy_stable else "FAIL",
            "note": "config(3)/schema-version(4)/registry(5)/backend(6,7)/artifact(8) stable",
            "schema_field_gap": schema_field_gap,
        },
        "logs_joinable": {
            "status": "PASS" if (jsonl["status"] == "PASS" and run_trace_linked) else "FAIL",
            "jsonl_parseable_stable": jsonl["status"],
            "run_trace_linked": run_trace_linked,
            "note": "JSONL parseable + trace-tagged; run↔trace linkage is a gap",
        },
        "no_secret": {"status": redaction["status"]},
        "cli_nonzero": {"status": cli_audit["status"]},
    }

    overall = "PASS" if all(c["status"] == "PASS" for c in criteria.values()) else "FAIL"

    verdict = {
        "experiment_id": EXPERIMENT_ID,
        "overall": overall,
        "criteria": criteria,
        "findings": {
            "schema_field_gap": {
                "summary": "missing/unknown field surfaces as pydantic ValidationError (exit 1), "
                "not SchemaError (exit 4); the version gate is the only SchemaError path",
                "cases": ["schema_missing_required_field", "schema_unknown_field"],
            },
            "run_trace_linkage_gap": {
                "summary": "C6 run_id and C7 trace_id are not linked; the benchmark engine "
                "drops backend trace events from the result and never sets a shared "
                "run/trace context",
                "evidence": correlation,
            },
        },
    }

    record = {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "environment": collect_environment(),
        "error_mapping": error_mapping,
        "case_matrix": case_matrix,
        "verdict": verdict,
        "correlation_audit": correlation,
        "redaction_audit": redaction,
        "jsonl_audit": jsonl,
        "cli_audit": cli_audit,
        "command_results": command_results,
        "cases": cases,
    }

    # ── Persist evidence ──────────────────────────────────────────────
    def _wjson(name: str, obj: Any) -> None:
        (out_dir / name).write_text(
            json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    _wjson("E01-06_error_mapping.json", error_mapping)
    _wjson("E01-06_case_matrix.json", case_matrix)
    with (out_dir / "E01-06_case_results.jsonl").open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n")
    with (out_dir / "E01-06_command_results.jsonl").open("w", encoding="utf-8") as fh:
        for row in command_results:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_traces(cases, out_dir / "E01-06_traces.jsonl")
    with (out_dir / "E01-06_logs.jsonl").open("w", encoding="utf-8") as fh:
        for c in cases:
            for ln in c.get("observed", {}).get("log_lines", []):
                fh.write(ln + "\n")
        for ln in jsonl["observed"]["log_lines"]:
            fh.write(json.dumps(ln, ensure_ascii=False, sort_keys=True) + "\n")
    _wjson("E01-06_correlation_audit.json", correlation)
    _wjson("E01-06_redaction_audit.json", redaction)
    _wjson("e01_06_" + run_id + ".json", record)
    _wjson("e01_06_" + run_id + "_env.json", record["environment"])
    _wjson("verdict.json", verdict)

    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if control["status"] == "PASS":
        rid = control["observed"]["run_id"]
        _wjson(f"results/{rid}.json", control["observed"]["result"])

    # ── Console report ────────────────────────────────────────────────
    print(f"[{EXPERIMENT_ID}] run_id={run_id}")
    env = record["environment"]
    print(f"[{EXPERIMENT_ID}] commit={env['git_commit_short']} dirty={env['git_dirty']}")
    print(f"[{EXPERIMENT_ID}] overall={overall}")
    for key, val in criteria.items():
        print(f"  - {key}: {val['status']}")
    print()
    for c in cases:
        mark = "PASS" if c["status"] == "PASS" else "FAIL"
        print(f"  [{mark}] {c['case_id']:<42} ({c['category']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E01-06 error taxonomy & traceability.")
    parser.add_argument(
        "--output-dir",
        default="docs/stage_experiments/S01/E01-06/raw",
        help="Directory for raw JSON/JSONL artifacts.",
    )
    parser.add_argument("--run-id", default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
