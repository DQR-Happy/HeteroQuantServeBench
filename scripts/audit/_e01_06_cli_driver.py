#!/usr/bin/env python3
"""E01-06 reference CLI driver for real process exit-code capture.

This is **not** a production CLI. It is the public-entry boundary that
E01-06 uses to prove the error taxonomy maps to real, non-zero *process*
returncodes. Every case runs only real public entry points:

* ``ConfigLoader`` (config),
* the ``VersionedModel`` construction gate (schema/contract version),
* ``Registry`` (registry),
* ``BenchmarkEngine`` (backend),
* ``verify_or_raise`` (artifact gate).

Any :class:`~hqsb.core.errors.HqsbError` is mapped to its canonical exit
code via :func:`~hqsb.core.errors.exit_code_for` and the process exits with
that code. A non-HQSB exception is treated as an internal error (exit 1).
The diagnostic record is printed to **stderr**; only the success payload is
printed to **stdout**, so the parent can separate machine payload from the
human-facing stream.

Usage
-----
    python3 scripts/audit/_e01_06_cli_driver.py --case <case>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hqsb.backends import make_dummy_backend  # noqa: E402
from hqsb.benchmark.engine import BenchmarkEngine  # noqa: E402
from hqsb.core.config.loader import ConfigLoader  # noqa: E402
from hqsb.core.config.schema import BenchmarkConfig  # noqa: E402
from hqsb.core.contracts import ModelArtifact, WorkloadSpec  # noqa: E402
from hqsb.core.errors import HqsbError, exit_code_for  # noqa: E402
from hqsb.core.registry import Registry  # noqa: E402
from hqsb.models.manifest import verify_or_raise  # noqa: E402


def _emit(record: Dict[str, Any], *, to_stderr: bool = False) -> None:
    stream = sys.stderr if to_stderr else sys.stdout
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), file=stream)


def _success(payload: Dict[str, Any]) -> int:
    _emit({"status": "success", "exit_code": 0, **payload})
    return 0


def _failure(exc: BaseException, context: Dict[str, Any]) -> int:
    code = exit_code_for(exc)
    _emit(
        {
            "status": "failure",
            "exit_code": code,
            "error_class": type(exc).__name__,
            "message": str(exc),
            "details": getattr(exc, "details", None),
            "context": context,
        },
        to_stderr=True,
    )
    return code


# ── Fixtures ────────────────────────────────────────────────────────────


def _make_fixture() -> Tuple[str, str]:
    """Create a small, legal model dir + manifest, returning both paths.

    The manifest is a *sibling* of the model directory (not inside it) so
    that strict-extra verification reports exactly the intended fault class
    and not a spurious ``extra_file`` for the manifest itself.
    """
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


# ── Cases (real public entries only) ────────────────────────────────────


def _case_success() -> int:
    backend = make_dummy_backend()
    artifact = ModelArtifact(
        model_id="fixture/dummy-synthetic-v1",
        source="local",
        architecture="DummyForCausalLM",
        dtype="float16",
    )
    workload = WorkloadSpec(
        name="short", input_tokens=128, output_tokens=32, seed=42, repetitions=1
    )
    result = BenchmarkEngine(backend).run(workload, artifact=artifact)
    return _success(
        {
            "run_id": result.run_id,
            "correctness_passed": result.correctness.passed,
            "sample_count": len(result.raw_samples),
        }
    )


def _case_config_unknown_field() -> int:
    try:
        ConfigLoader(BenchmarkConfig).load(
            defaults=_defaults(), environ={}, cli={"benchmark": {"bogus": 1}}
        )
    except HqsbError as exc:
        return _failure(exc, {"case": "config_unknown_field"})
    return _failure(_NoError("expected ConfigError"), {"case": "config_unknown_field"})


def _case_config_bad_value() -> int:
    try:
        ConfigLoader(BenchmarkConfig).load(
            defaults=_defaults(),
            environ={"HQSB_BENCHMARK__BATCH_SIZE": "not-an-int"},
        )
    except HqsbError as exc:
        return _failure(exc, {"case": "config_bad_value"})
    return _failure(_NoError("expected ConfigError"), {"case": "config_bad_value"})


def _case_schema_future_version() -> int:
    try:
        ModelArtifact(
            schema_version="2.0.0",
            model_id="m",
            source="local",
            architecture="a",
            dtype="float16",
        )
    except HqsbError as exc:
        return _failure(exc, {"case": "schema_future_version"})
    return _failure(_NoError("expected schema version error"), {"case": "schema_future_version"})


def _case_registry_unknown() -> int:
    try:
        Registry(kind="backend").get("does-not-exist")
    except HqsbError as exc:
        return _failure(exc, {"case": "registry_unknown"})
    return _failure(_NoError("expected RegistryLookupError"), {"case": "registry_unknown"})


def _case_backend_capability() -> int:
    try:
        backend = make_dummy_backend()
        workload = WorkloadSpec(
            name="short", input_tokens=128, output_tokens=32, batch_size=8
        )
        BenchmarkEngine(backend).run(workload)
    except HqsbError as exc:
        return _failure(exc, {"case": "backend_capability"})
    return _failure(_NoError("expected CapabilityError"), {"case": "backend_capability"})


def _case_backend_load_failure() -> int:
    try:
        backend = make_dummy_backend(fail_at="load")
        workload = WorkloadSpec(
            name="short", input_tokens=128, output_tokens=32
        )
        BenchmarkEngine(backend).run(
            workload,
            artifact=ModelArtifact(
                model_id="m", source="local", architecture="a", dtype="float16"
            ),
        )
    except HqsbError as exc:
        return _failure(exc, {"case": "backend_load_failure"})
    return _failure(_NoError("expected BackendError"), {"case": "backend_load_failure"})


def _case_artifact_hash_mismatch() -> int:
    root, manifest = _make_fixture()
    _tamper(root)
    try:
        verify_or_raise(root, manifest)
    except HqsbError as exc:
        return _failure(exc, {"case": "artifact_hash_mismatch"})
    return _failure(_NoError("expected ArtifactError"), {"case": "artifact_hash_mismatch"})


def _case_artifact_missing_file() -> int:
    root, manifest = _make_fixture()
    (Path(root) / "config.json").unlink()
    try:
        verify_or_raise(root, manifest)
    except HqsbError as exc:
        return _failure(exc, {"case": "artifact_missing_file"})
    return _failure(_NoError("expected ArtifactError"), {"case": "artifact_missing_file"})


def _case_config_missing_file() -> int:
    try:
        ConfigLoader(BenchmarkConfig).load(
            defaults=_defaults(), environ={}, path="/nonexistent/e01_06.yaml"
        )
    except HqsbError as exc:
        return _failure(exc, {"case": "config_missing_file"})
    return _failure(_NoError("expected ConfigError"), {"case": "config_missing_file"})


class _NoError(HqsbError):
    """Placeholder raised when an expected failure did not occur."""


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


_CASES = {
    "success": _case_success,
    "config_unknown_field": _case_config_unknown_field,
    "config_bad_value": _case_config_bad_value,
    "config_missing_file": _case_config_missing_file,
    "schema_future_version": _case_schema_future_version,
    "registry_unknown": _case_registry_unknown,
    "backend_capability": _case_backend_capability,
    "backend_load_failure": _case_backend_load_failure,
    "artifact_hash_mismatch": _case_artifact_hash_mismatch,
    "artifact_missing_file": _case_artifact_missing_file,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="E01-06 reference CLI driver.")
    parser.add_argument("--case", required=True, choices=sorted(_CASES))
    args = parser.parse_args()
    try:
        return _CASES[args.case]()
    except HqsbError as exc:  # unexpected path inside a case executor
        return _failure(exc, {"case": args.case})
    except Exception as exc:  # noqa: BLE001 - internal error, exit 1
        return _failure(exc, {"case": args.case})


if __name__ == "__main__":
    raise SystemExit(main())
