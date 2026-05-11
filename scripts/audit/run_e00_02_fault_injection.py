#!/usr/bin/env python3
"""E00-02 — ModelArtifact fault injection.

Question
--------
Can a wrong, partial, or hostile model artifact reach a backend, or is it
rejected before any weight is loaded?

Hypothesis (falsifiable, pre-registered)
----------------------------------------
H1  Every injected fault (missing file, tampered content, empty path,
    duplicate path, path traversal, extra file) is rejected by the
    artifact gate *before* ``load``, with a stable machine-readable reason
    code and a first-bad-file pointer.
H0  At least one fault class is accepted, or is only detected after
    weights have been loaded.

Design
------
Two independent gates are exercised for every case:

* **manifest gate** — :func:`hqsb.models.manifest.verify_or_raise`, the
  function a loader calls before touching weights. It is the gate that
  must fail closed.
* **contract gate** — :class:`hqsb.core.contracts.model.ModelArtifact`
  construction from the manifest's own ``file_hashes``. Declaration faults
  (empty/duplicate/traversal paths) must be rejected here too, so an
  unsafe artifact cannot even be represented in memory.

A third, non-gate observation — **legacy gate** — replays the pre-E00-02
algorithm (``os.path.join`` + ``isfile`` + digest compare, no path checks,
no extra-file scan) to document what each fault used to do. ``legacy_ok``
is *not* a pass/fail signal; it is the vulnerability evidence.

The experiment is pure CPU: no torch, no GPU, no real weights.

Raw output
----------
``<out>/e00_02_<run_id>.json``      full record (env + every case + samples)
``<out>/e00_02_<run_id>_cases.csv`` one row per case
``<out>/e00_02_<run_id>_env.json``  frozen environment / identity block

Usage
-----
    python3 scripts/audit/run_e00_02_fault_injection.py \\
        --output-dir docs/stage_experiments/S00/E00-02/raw

    # additionally verify the real snapshot (hashes ~4 GB, slower):
    python3 scripts/audit/run_e00_02_fault_injection.py \\
        --output-dir ... \\
        --real-model-path ~/models/hqsb/Qwen3-1.7B \\
        --real-manifest docs/benchmark/model_sha256_manifest.txt \\
        --real-allow-extra model_sha256_manifest.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hqsb.core.contracts import ModelArtifact  # noqa: E402
from hqsb.core.errors import ArtifactError, ExitCode, exit_code_for  # noqa: E402
from hqsb.core.ids import new_run_id  # noqa: E402
from hqsb.models.manifest import (  # noqa: E402
    ManifestError,
    compute_sha256,
    parse_manifest,
    verify_or_raise,
)

EXPERIMENT_ID = "E00-02"
STAGE = "S00"

# Pre-registered pass criteria.
EXPECTED_EXIT_CODE_REJECTED = ExitCode.ARTIFACT  # 8
EXPECTED_EXIT_CODE_ACCEPTED = ExitCode.SUCCESS  # 0

_GOOD = b'{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}'
_TOKENIZER = b'{"version":"1.0","model":{"type":"BPE"}}'
_WEIGHTS = b"HQSB-FAKE-WEIGHTS-" * 256
_SECRET = b"TOP-SECRET-OUTSIDE-THE-MODEL-ROOT\n"

# C10 needs an absolute path outside the model root. It is anchored to a
# fixed location (not the per-run temp dir) so repeated in-process runs
# produce byte-identical manifests and can be compared directly.
_FIXTURE_OUTSIDE = Path(tempfile.gettempdir()) / "hqsb_e00_02_fixture" / "outside"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest_from(files: Dict[str, bytes]) -> str:
    """Render a ``sha256sum``-style manifest for ``{path: content}``."""
    return "".join(
        f"{_sha(content)}  ./{rel}\n"
        for rel, content in sorted(files.items())
    )


@dataclass
class BuiltCase:
    """One materialized fault-injection case."""

    case_id: str
    fault_class: str
    fault_input: str
    expected_reason: str
    expected_rejected: bool
    model_dir: Path
    manifest_path: Path
    manifest_text: str
    declared_files: Dict[str, bytes]
    outside_files: Dict[str, bytes] = field(default_factory=dict)
    notes: str = ""


# ── Case builders ───────────────────────────────────────────────────────


def _base_files() -> Dict[str, bytes]:
    return {
        "config.json": _GOOD,
        "tokenizer.json": _TOKENIZER,
        "model.safetensors": _WEIGHTS,
    }


def _materialize(root: Path, case_id: str, files: Dict[str, bytes]) -> Path:
    model_dir = root / case_id / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        target = model_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return model_dir


def _write_manifest(root: Path, case_id: str, text: str) -> Path:
    path = root / case_id / "manifest.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _build(root: Path, case_id: str) -> BuiltCase:
    """Dispatch table: every case is materialized the same way."""
    files = _base_files()

    if case_id == "C01":
        # Control: a legal artifact. Must be ACCEPTED.
        model_dir = _materialize(root, case_id, files)
        text = _manifest_from(files)
        return BuiltCase(
            case_id, "none_control", "legal 3-file artifact",
            "none", False, model_dir, _write_manifest(root, case_id, text),
            text, files,
        )

    if case_id == "C02":
        # 缺文件: declared in the manifest, absent on disk.
        model_dir = _materialize(root, case_id, files)
        text = _manifest_from(files)
        (model_dir / "model.safetensors").unlink()
        return BuiltCase(
            case_id, "missing_file",
            "manifest declares ./model.safetensors, file deleted from disk",
            "missing_file", True, model_dir,
            _write_manifest(root, case_id, text), text, files,
        )

    if case_id == "C03":
        # 内容篡改: content changed after the manifest was written.
        model_dir = _materialize(root, case_id, files)
        text = _manifest_from(files)
        (model_dir / "tokenizer.json").write_bytes(_TOKENIZER + b',"tampered":true}')
        return BuiltCase(
            case_id, "hash_mismatch",
            "tokenizer.json rewritten after manifest generation",
            "hash_mismatch", True, model_dir,
            _write_manifest(root, case_id, text), text, files,
        )

    if case_id == "C04":
        # 空路径: '<sha>  ./' normalizes to the model root itself.
        model_dir = _materialize(root, case_id, files)
        text = f"{_sha(_GOOD)}  ./\n" + _manifest_from(
            {"tokenizer.json": _TOKENIZER, "model.safetensors": _WEIGHTS}
        )
        return BuiltCase(
            case_id, "path_empty", "manifest line '<sha256>  ./'",
            "path_empty", True, model_dir,
            _write_manifest(root, case_id, text), text, files,
        )

    if case_id == "C05":
        # 重复路径: same path declared twice with the same digest.
        model_dir = _materialize(root, case_id, files)
        text = _manifest_from(files) + f"{_sha(_GOOD)}  ./config.json\n"
        return BuiltCase(
            case_id, "path_duplicate",
            "config.json declared twice (identical digests)",
            "path_duplicate", True, model_dir,
            _write_manifest(root, case_id, text), text, files,
        )

    if case_id == "C06":
        # 目录穿越: '../outside/secret.bin' escapes the model root.
        model_dir = _materialize(root, case_id, files)
        outside_dir = root / case_id / "outside"
        outside_dir.mkdir(parents=True, exist_ok=True)
        (outside_dir / "secret.bin").write_bytes(_SECRET)
        text = _manifest_from(files) + f"{_sha(_SECRET)}  ../outside/secret.bin\n"
        return BuiltCase(
            case_id, "path_traversal",
            "manifest declares ../outside/secret.bin (exists, digest matches)",
            "path_traversal", True, model_dir,
            _write_manifest(root, case_id, text), text, files,
            outside_files={"outside/secret.bin": _SECRET},
            notes="traversal target exists with a matching digest: a "
                  "join-only verifier would verify a file outside the root.",
        )

    if case_id == "C07":
        # 额外文件: present under the root, not declared by the manifest.
        model_dir = _materialize(root, case_id, files)
        text = _manifest_from(files)
        (model_dir / "leftover.tmp").write_bytes(b"partial-download-fragment")
        return BuiltCase(
            case_id, "extra_file",
            "undeclared ./leftover.tmp written into the model root",
            "extra_file", True, model_dir,
            _write_manifest(root, case_id, text), text, files,
        )

    # ── Supplementary coverage (beyond the six required classes) ────────

    if case_id == "C08":
        model_dir = _materialize(root, case_id, files)
        text = _manifest_from(files) + f"{_sha(b'x')}  ./nested//config.json\n"
        return BuiltCase(
            case_id, "path_empty",
            "manifest declares ./nested//config.json (empty component)",
            "path_empty", True, model_dir,
            _write_manifest(root, case_id, text), text, files,
        )

    if case_id == "C09":
        model_dir = _materialize(root, case_id, files)
        text = _manifest_from(files) + f"{_sha(_GOOD)}  ./config.json\n" + (
            f"{_sha(b'totally-different')}  ./config.json\n"
        )
        return BuiltCase(
            case_id, "path_duplicate",
            "config.json declared twice with CONFLICTING digests",
            "path_duplicate", True, model_dir,
            _write_manifest(root, case_id, text), text, files,
        )

    if case_id == "C10":
        model_dir = _materialize(root, case_id, files)
        outside_dir = _FIXTURE_OUTSIDE
        outside_dir.mkdir(parents=True, exist_ok=True)
        (outside_dir / "secret.bin").write_bytes(_SECRET)
        text = _manifest_from(files) + f"{_sha(_SECRET)}  {outside_dir / 'secret.bin'}\n"
        return BuiltCase(
            case_id, "path_absolute",
            "manifest declares an absolute path outside the model root",
            "path_absolute", True, model_dir,
            _write_manifest(root, case_id, text), text, files,
            outside_files={"outside/secret.bin": _SECRET},
        )

    if case_id == "C11":
        model_dir = _materialize(root, case_id, files)
        text = _manifest_from(files) + f"{_sha(_TOKENIZER)}  ./sub/./tokenizer.json\n"
        return BuiltCase(
            case_id, "path_not_normalized",
            "manifest declares ./sub/./tokenizer.json",
            "path_not_normalized", True, model_dir,
            _write_manifest(root, case_id, text), text, files,
        )

    if case_id == "C12":
        model_dir = _materialize(root, case_id, files)
        text = _manifest_from(files) + f"{_sha(_SECRET)}  ..\\outside\\secret.bin\n"
        return BuiltCase(
            case_id, "path_backslash",
            "manifest declares ..\\outside\\secret.bin (Windows traversal)",
            "path_backslash", True, model_dir,
            _write_manifest(root, case_id, text), text, files,
            notes="rejected on every platform so one manifest cannot be "
                  "safe on POSIX and unsafe on Windows.",
        )

    if case_id == "C13":
        files = dict(files)
        files["Config.json"] = b'{"different":"content"}'
        model_dir = _materialize(root, case_id, files)
        text = _manifest_from(files)
        return BuiltCase(
            case_id, "path_duplicate_casefold",
            "Config.json and config.json declared together",
            "path_duplicate_casefold", True, model_dir,
            _write_manifest(root, case_id, text), text, files,
            notes="collides on case-insensitive filesystems (macOS/Windows).",
        )

    if case_id == "C14":
        model_dir = _materialize(root, case_id, files)
        text = _manifest_from(files) + "not-a-sha256  ./config.json\n"
        return BuiltCase(
            case_id, "digest_invalid",
            "manifest line with a non-SHA256 digest",
            "digest_invalid", True, model_dir,
            _write_manifest(root, case_id, text), text, files,
        )

    if case_id == "C15":
        model_dir = _materialize(root, case_id, files)
        text = _manifest_from(files) + "only-one-column\n"
        return BuiltCase(
            case_id, "line_malformed",
            "manifest line with a digest but no path",
            "line_malformed", True, model_dir,
            _write_manifest(root, case_id, text), text, files,
        )

    raise ValueError(f"unknown case_id: {case_id}")


CASE_IDS = [f"C{i:02d}" for i in range(1, 16)]

#: The six fault classes mandated by the S00 experiment list, mapped to the
#: case IDs that inject them (plus the clean control).
REQUIRED_FAULT_CLASSES = {
    "缺文件 (missing file)": "C02",
    "内容篡改 (content tampering)": "C03",
    "空路径 (empty path)": "C04",
    "重复路径 (duplicate path)": "C05",
    "目录穿越 (directory traversal)": "C06",
    "额外文件 (extra file)": "C07",
}


# ── Gates ───────────────────────────────────────────────────────────────


def _escaping_entries(model_dir: Path, text: str) -> List[str]:
    """Paths in ``text`` that resolve outside ``model_dir``.

    Faults are injected through the manifest, so "would this have read a
    file outside the artifact root?" is measured by resolving each declared
    path, not by trusting the manifest's spelling.
    """
    try:
        entries = parse_manifest(text, validate_paths=False)
    except ManifestError:
        return []
    root = os.path.realpath(str(model_dir))
    escaped: List[str] = []
    for entry in entries:
        joined = os.path.realpath(os.path.join(str(model_dir), entry.normalized_path))
        if joined != root and not joined.startswith(root + os.sep):
            escaped.append(entry.normalized_path)
    return escaped


def _load_prefix_manifest() -> Tuple[Optional[Any], str]:
    """Load ``hqsb/models/manifest.py`` as it exists at ``git HEAD``.

    Provides a ground-truth "before" baseline instead of a hand-written
    reimplementation. Returns ``(None, reason)`` when unavailable, e.g.
    once the hardened manifest has been committed.
    """
    source = _git("show", "HEAD:hqsb/models/manifest.py")
    if not source:
        return None, "unavailable: git HEAD manifest source could not be read"
    if "validate_relative_paths" in source:
        return None, (
            "unavailable: HEAD already contains the hardened manifest "
            "(re-run against the pre-fix commit for a live comparison)"
        )
    try:
        tmpdir = tempfile.mkdtemp(prefix="hqsb_prefix_manifest_")
        module_path = os.path.join(tmpdir, "hqsb_prefix_manifest.py")
        with open(module_path, "w", encoding="utf-8") as fh:
            fh.write(source)
        spec = importlib.util.spec_from_file_location(
            "hqsb_prefix_manifest", module_path
        )
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            return None, "unavailable: could not build an import spec"
        module = importlib.util.module_from_spec(spec)
        # ``dataclasses`` resolves annotations via ``sys.modules[cls.__module__]``,
        # so the module must be registered *before* its body executes.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
        return module, f"loaded from git HEAD ({spec.name}@{module_path})"
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"unavailable: {type(exc).__name__}: {exc}"


_PREFIX_MODULE: Optional[Any] = None
_PREFIX_STATUS = "not loaded"


def init_prefix_baseline() -> str:
    """Load the pre-fix manifest module once and report its status."""
    global _PREFIX_MODULE, _PREFIX_STATUS
    _PREFIX_MODULE, _PREFIX_STATUS = _load_prefix_manifest()
    return _PREFIX_STATUS


def _run_prefix_gate(model_dir: Path, manifest_path: Path, text: str) -> Dict[str, Any]:
    """Run the *pre-fix* verifier on this case (vulnerability baseline)."""
    escaped = _escaping_entries(model_dir, text)
    if _PREFIX_MODULE is None:
        return {"available": False, "status": _PREFIX_STATUS, "escaped_root": escaped}
    try:
        result = _PREFIX_MODULE.verify_model_files(
            str(model_dir), str(manifest_path)
        )
    except Exception as exc:
        return {
            "available": True,
            "status": _PREFIX_STATUS,
            "raised": type(exc).__name__,
            "ok": False,
            "message": str(exc)[:400],
            "missing": [],
            "mismatched": [],
            "escaped_root": escaped,
            "would_read_outside_root": bool(escaped),
        }
    return {
        "available": True,
        "status": _PREFIX_STATUS,
        "raised": None,
        "ok": bool(result.ok),
        "describe": result.describe(),
        "missing": list(result.missing_files),
        "mismatched": [path for path, _e, _a in result.mismatched_files],
        "escaped_root": escaped,
        "would_read_outside_root": bool(escaped),
    }


def _legacy_verify(model_dir: Path, text: str) -> Dict[str, Any]:
    """Replay the pre-E00-02 algorithm in-process: join, stat, compare.

    Independent of :func:`_run_prefix_gate`, which runs the real historical
    module. Agreement between the two is itself a sanity check that the
    reimplementation is faithful.
    """
    try:
        entries = parse_manifest(text, validate_paths=False)
    except ManifestError as exc:
        return {
            "parsed": False,
            "ok": False,
            "error": type(exc).__name__,
            "reasons": list(exc.reasons),
            "missing": [],
            "mismatched": [],
            "escaped_root": _escaping_entries(model_dir, text),
            "would_read_outside_root": bool(_escaping_entries(model_dir, text)),
        }

    missing: List[str] = []
    mismatched: List[str] = []
    for entry in entries:
        joined = os.path.join(str(model_dir), entry.normalized_path)
        if not os.path.isfile(joined):
            missing.append(entry.normalized_path)
            continue
        if compute_sha256(joined) != entry.sha256:
            mismatched.append(entry.normalized_path)
    escaped = _escaping_entries(model_dir, text)
    return {
        "parsed": True,
        "ok": not missing and not mismatched,
        "missing": missing,
        "mismatched": mismatched,
        "escaped_root": escaped,
        "would_read_outside_root": bool(escaped),
    }


def _run_manifest_gate(model_dir: Path, manifest_path: Path) -> Dict[str, Any]:
    """Run the loader-facing gate and record the raw verdict."""
    try:
        result = verify_or_raise(str(model_dir), str(manifest_path))
    except ArtifactError as exc:
        details = dict(exc.details or {})
        return {
            "rejected": True,
            "error_type": type(exc).__name__,
            "exit_code": exit_code_for(exc),
            "reason_codes": details.get("reason_codes", []),
            "first_bad_file": details.get("first_bad_file"),
            "missing_files": details.get("missing_files", []),
            "mismatched_files": details.get("mismatched_files", []),
            "extra_files": details.get("extra_files", []),
            "message": str(exc),
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "rejected": True,
            "error_type": type(exc).__name__,
            "exit_code": exit_code_for(exc),
            "reason_codes": ["unexpected_exception"],
            "first_bad_file": None,
            "message": str(exc),
        }
    return {
        "rejected": False,
        "error_type": None,
        "exit_code": ExitCode.SUCCESS,
        "reason_codes": [],
        "first_bad_file": None,
        "describe": result.describe(),
        "verified_files": result.verified_files,
        "total_files": result.total_files,
    }


def _run_contract_gate(text: str) -> Dict[str, Any]:
    """Build a ModelArtifact from the manifest's own file_hashes.

    Path validation is deliberately disabled while *parsing* so the raw,
    unvalidated keys reach the contract: this measures whether the contract
    itself refuses them, independent of the manifest layer.
    """
    try:
        entries = parse_manifest(text, validate_paths=False)
    except ManifestError as exc:
        return {
            "applicable": False,
            "rejected": True,
            "error_type": type(exc).__name__,
            "exit_code": ExitCode.ARTIFACT,
            "reasons": list(exc.reasons),
            "message": str(exc),
        }

    file_hashes = {entry.normalized_path: entry.sha256 for entry in entries}
    try:
        artifact = ModelArtifact(
            model_id="hqsb-test/tiny-qwen3",
            source="local",
            architecture="Qwen3ForCausalLM",
            dtype="float16",
            file_hashes=file_hashes,
        )
    except Exception as exc:
        return {
            "applicable": True,
            "rejected": True,
            "error_type": type(exc).__name__,
            "exit_code": exit_code_for(exc),
            "file_hashes": file_hashes,
            "identity_hash": None,
            "message": str(exc),
        }
    return {
        "applicable": True,
        "rejected": False,
        "error_type": None,
        "exit_code": ExitCode.SUCCESS,
        "file_hashes": file_hashes,
        "identity_hash": artifact.identity_hash(),
    }


def _stabilize(value: Any, work_root: Path) -> Any:
    """Replace volatile absolute fixture paths with a stable placeholder.

    Case C10 declares an absolute path inside the per-run temp directory, so
    byte-identical repetition comparison would fail on the fixture path
    alone. Every verdict field is preserved; only the volatile location is
    normalized away.
    """
    token = str(work_root)
    if isinstance(value, str):
        value = value.replace(token, "<WORK_ROOT>")
        # Each in-process repetition materializes under ``<root>/repN``.
        return re.sub(r"/rep\d+/", "/repN/", value)
    if isinstance(value, list):
        return [_stabilize(item, work_root) for item in value]
    if isinstance(value, tuple):
        return [_stabilize(item, work_root) for item in value]
    if isinstance(value, dict):
        return {key: _stabilize(item, work_root) for key, item in value.items()}
    return value


def _display_path(value: Optional[str]) -> str:
    """Render a first-bad-file value for tables (empty path is not blank)."""
    if value is None:
        return ""
    if value == "":
        return "<empty-path>"
    return value


def run_case(root: Path, case_id: str) -> Dict[str, Any]:
    """Materialize, gate, and observe one fault case."""
    case = _build(root, case_id)
    record: Dict[str, Any] = {
        "case_id": case.case_id,
        "fault_class": case.fault_class,
        "fault_input": case.fault_input,
        "notes": case.notes,
        "expected": {
            "rejected": case.expected_rejected,
            "reason": case.expected_reason,
            "exit_code": (
                EXPECTED_EXIT_CODE_REJECTED
                if case.expected_rejected
                else EXPECTED_EXIT_CODE_ACCEPTED
            ),
        },
        "manifest_sha256": compute_sha256(str(case.manifest_path)),
        "manifest_text": case.manifest_text,
        "declared_files": {
            rel: _sha(content) for rel, content in sorted(case.declared_files.items())
        },
        "outside_files": {
            rel: _sha(content) for rel, content in sorted(case.outside_files.items())
        },
    }
    record["manifest_gate"] = _run_manifest_gate(case.model_dir, case.manifest_path)
    record["contract_gate"] = _run_contract_gate(case.manifest_text)
    record["legacy_gate"] = _legacy_verify(case.model_dir, case.manifest_text)
    record["prefix_gate"] = _run_prefix_gate(
        case.model_dir, case.manifest_path, case.manifest_text
    )
    record["actual"] = {
        "rejected": record["manifest_gate"]["rejected"],
        "exit_code": record["manifest_gate"]["exit_code"],
        "reason_codes": record["manifest_gate"]["reason_codes"],
        "first_bad_file": record["manifest_gate"]["first_bad_file"],
    }
    record["pass"] = (
        record["actual"]["rejected"] == record["expected"]["rejected"]
        and record["actual"]["exit_code"] == record["expected"]["exit_code"]
        and (
            not record["expected"]["rejected"]
            or record["expected"]["reason"] in record["actual"]["reason_codes"]
        )
        and (
            not record["expected"]["rejected"]
            or record["actual"]["first_bad_file"] is not None
        )
    )
    return record


# ── Hash stability ──────────────────────────────────────────────────────

_HASH_CHILD = r"""
import json, sys
sys.path.insert(0, {root!r})
from hqsb.core.contracts import ModelArtifact
payload = json.loads(sys.stdin.read())
print(ModelArtifact(**payload).identity_hash())
"""


def _clean_artifact() -> ModelArtifact:
    files = _base_files()
    return ModelArtifact(
        model_id="hqsb-test/tiny-qwen3",
        source="local",
        architecture="Qwen3ForCausalLM",
        dtype="float16",
        file_hashes={rel: _sha(content) for rel, content in files.items()},
    )


def run_hash_stability(repeats: int = 5) -> Dict[str, Any]:
    """Verify a legal artifact always yields the same identity hash.

    Three independent sources of nondeterminism are controlled:
    in-process repetition, ``file_hashes`` insertion order, and a fresh
    interpreter with a different ``PYTHONHASHSEED`` per process.
    """
    artifact = _clean_artifact()
    in_process = [artifact.identity_hash() for _ in range(repeats)]

    payload = artifact.identity_payload()
    permutations: List[str] = []
    rng = random.Random(20240903)
    for _ in range(repeats):
        items = list(payload["file_hashes"].items())
        rng.shuffle(items)
        shuffled = dict(payload)
        shuffled["file_hashes"] = dict(items)
        permutations.append(ModelArtifact(**shuffled).identity_hash())

    subprocess_env: List[Dict[str, Any]] = []
    for seed in ("0", "1", "2"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        proc = subprocess.run(
            [sys.executable, "-c", _HASH_CHILD.format(root=str(_REPO_ROOT))],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=env,
        )
        subprocess_env.append(
            {
                "pythonhashseed": seed,
                "identity_hash": proc.stdout.strip(),
                "returncode": proc.returncode,
                "stderr": proc.stderr.strip()[-500:],
            }
        )

    observed = in_process + permutations + [
        item["identity_hash"] for item in subprocess_env
    ]
    distinct = sorted(set(observed))
    return {
        "artifact_identity_payload": payload,
        "in_process_repeats": in_process,
        "insertion_order_permutations": permutations,
        "subprocess_runs": subprocess_env,
        "samples": observed,
        "distinct_hashes": distinct,
        "stable": len(distinct) == 1,
        "pass": len(distinct) == 1
        and all(item["returncode"] == 0 for item in subprocess_env),
    }


# ── Real snapshot (optional) ────────────────────────────────────────────


def run_real_snapshot(
    model_path: str,
    manifest_path: str,
    allow_extra: Tuple[str, ...],
) -> Dict[str, Any]:
    """Apply the same gate to the real Qwen3-1.7B snapshot."""
    from hqsb.models.manifest import verify_model_files

    observation: Dict[str, Any] = {
        "model_path": os.path.abspath(os.path.expanduser(model_path)),
        "manifest_path": os.path.abspath(os.path.expanduser(manifest_path)),
        "allow_extra": list(allow_extra),
    }

    # Two arms: strict (no exemptions) and strict-with-exemptions. Both keep
    # ``strict_extra=True`` so the only difference is the allow-list, which
    # is what makes the exemption explicit and auditable.
    for label, allow in (("strict", ()), ("with_allow_extra", allow_extra)):
        started = time.perf_counter()
        result = verify_model_files(
            model_path,
            manifest_path,
            strict_extra=True,
            allow_extra=allow,
        )
        observation[label] = {
            "allow_extra_used": list(allow),
            "ok": result.ok,
            "describe": result.describe(),
            "first_bad_file": result.first_bad_file,
            "reason_codes": result.reason_codes,
            "extra_files": list(result.extra_files),
            "allowed_extra_files": list(result.allowed_extra_files),
            "missing_files": list(result.missing_files),
            "mismatched_files": [
                {"path": p, "expected": e, "actual": a}
                for p, e, a in result.mismatched_files
            ],
            "manifest_sha256": result.manifest_sha256,
            "elapsed_s": round(time.perf_counter() - started, 3),
        }
    return observation


# ── Drivers ─────────────────────────────────────────────────────────────


def _git(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except OSError:  # pragma: no cover - git missing
        return ""


def _case_fingerprint(case: Dict[str, Any]) -> Dict[str, Any]:
    """Stable verdict-only projection of one case record.

    Drops everything that legitimately varies between processes (run id,
    timestamps, fixture directories) and keeps only the verdicts, so two
    independent runs can be compared for agreement.
    """
    return {
        "case_id": case["case_id"],
        "fault_class": case["fault_class"],
        "expected": case["expected"],
        "manifest_gate": {
            "rejected": case["manifest_gate"]["rejected"],
            "exit_code": case["manifest_gate"]["exit_code"],
            "reason_codes": case["manifest_gate"]["reason_codes"],
        },
        "contract_gate": {
            "rejected": case["contract_gate"]["rejected"],
            "exit_code": case["contract_gate"]["exit_code"],
        },
        "legacy_gate": {
            "ok": case["legacy_gate"].get("ok"),
            "escaped_root": sorted(case["legacy_gate"].get("escaped_root") or []),
        },
        "prefix_gate": {
            "available": case["prefix_gate"].get("available"),
            "ok": case["prefix_gate"].get("ok"),
            "escaped_root": sorted(case["prefix_gate"].get("escaped_root") or []),
        },
        "pass": case["pass"],
    }


def compare_runs(out_dir: Path, record: Dict[str, Any]) -> Dict[str, Any]:
    """Compare this run's verdicts against every earlier run in ``out_dir``.

    Writes ``cross_process_agreement.json`` so a reader can verify the
    "3 independent process runs" rule without re-running anything.
    """
    fingerprint = [
        _case_fingerprint(case) for case in record["repetitions"][0]["cases"]
    ]
    runs: List[Dict[str, Any]] = []
    for path in sorted(out_dir.glob("e00_02_run_*.json")):
        if path.name.endswith("_env.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "repetitions" not in payload:
            continue
        cases = payload["repetitions"][0]["cases"]
        runs.append(
            {
                "file": path.name,
                "sha256": compute_sha256(str(path)),
                "run_id": payload.get("run_id"),
                "started_at_utc": payload.get("environment", {}).get(
                    "started_at_utc"
                ),
                "repeat": len(payload.get("repetitions", [])),
                "suite_stable_across_repetitions": payload.get(
                    "suite_stable_across_repetitions"
                ),
                "identity_hash_stable": payload.get("hash_stability", {}).get(
                    "stable"
                ),
                "cases_passed": sum(1 for c in cases if c.get("pass")),
                "fingerprint_sha256": hashlib.sha256(
                    json.dumps(
                        [_case_fingerprint(c) for c in cases],
                        sort_keys=True,
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )

    distinct = sorted({run["fingerprint_sha256"] for run in runs})
    agreement = {
        "experiment_id": EXPERIMENT_ID,
        "current_run_id": record["run_id"],
        "current_fingerprint_sha256": hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest(),
        "runs": runs,
        "run_count": len(runs),
        "distinct_fingerprints": distinct,
        "agreement": len(distinct) <= 1,
    }
    (out_dir / "cross_process_agreement.json").write_text(
        json.dumps(agreement, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return agreement


def collect_environment() -> Dict[str, Any]:
    """Freeze the identity inputs required by the experiment record schema."""
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
        "system": f"{platform.system()} {platform.release()}",
        "cwd": str(_REPO_ROOT),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E00-02 ModelArtifact fault injection.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/stage_experiments/S00/E00-02/raw",
        help="Directory for raw JSON/CSV artifacts.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the whole case suite this many times in-process.",
    )
    parser.add_argument("--real-model-path", default=None)
    parser.add_argument("--real-manifest", default=None)
    parser.add_argument(
        "--real-allow-extra",
        action="append",
        default=[],
        help="Allow-listed undeclared files for the real snapshot.",
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

    work_root = Path(os.environ.get("TMPDIR", "/tmp")) / f"hqsb_{EXPERIMENT_ID}_{run_id}"
    work_root.mkdir(parents=True, exist_ok=True)

    prefix_status = init_prefix_baseline()

    repetitions: List[Dict[str, Any]] = []
    for index in range(max(1, args.repeat)):
        case_root = work_root / f"rep{index}"
        case_root.mkdir(parents=True, exist_ok=True)
        repetitions.append(
            {
                "repetition": index,
                "cases": [run_case(case_root, case_id) for case_id in CASE_IDS],
            }
        )

    cases = repetitions[0]["cases"]
    stabilized = [_stabilize(case, work_root) for case in cases]
    suite_stable = all(
        [_stabilize(case, work_root) for case in rep["cases"]] == stabilized
        for rep in repetitions[1:]
    )

    hash_stability = run_hash_stability()

    record: Dict[str, Any] = {
        "stage": STAGE,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "environment": collect_environment(),
        "prefix_baseline_status": prefix_status,
        "required_fault_classes": REQUIRED_FAULT_CLASSES,
        "repetitions": repetitions,
        "suite_stable_across_repetitions": suite_stable,
        "hash_stability": hash_stability,
        "summary": {
            "total_cases": len(cases),
            "passed_cases": sum(1 for c in cases if c["pass"]),
            "rejected_before_load": sum(
                1 for c in cases if c["fault_class"] != "none_control"
                and c["manifest_gate"]["rejected"]
            ),
            "fault_cases": sum(
                1 for c in cases if c["fault_class"] != "none_control"
            ),
            # Pre-fix baseline: the real historical module when available,
            # otherwise the in-process reimplementation.
            "prefix_would_accept": [
                c["case_id"]
                for c in cases
                if c["fault_class"] != "none_control"
                and (
                    c["prefix_gate"].get("ok")
                    if c["prefix_gate"].get("available")
                    else c["legacy_gate"].get("ok")
                )
            ],
            "prefix_would_read_outside_root": [
                c["case_id"]
                for c in cases
                if (
                    c["prefix_gate"].get("would_read_outside_root")
                    if c["prefix_gate"].get("available")
                    else c["legacy_gate"].get("would_read_outside_root")
                )
            ],
            "contract_gate_rejections": sum(
                1 for c in cases if c["contract_gate"]["rejected"]
            ),
        },
    }

    if args.real_model_path and args.real_manifest:
        record["real_snapshot"] = run_real_snapshot(
            args.real_model_path,
            args.real_manifest,
            tuple(args.real_allow_extra),
        )

    json_path = out_dir / f"e00_02_{run_id}.json"
    json_path.write_text(
        json.dumps(record, indent=2, sort_keys=False, ensure_ascii=False),
        encoding="utf-8",
    )
    env_path = out_dir / f"e00_02_{run_id}_env.json"
    env_path.write_text(
        json.dumps(record["environment"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    csv_path = out_dir / f"e00_02_{run_id}_cases.csv"
    _write_csv(csv_path, cases)

    # Console report.
    print(f"[{EXPERIMENT_ID}] run_id={run_id}")
    print(f"[{EXPERIMENT_ID}] commit={record['environment']['git_commit_short']} "
          f"dirty={record['environment']['git_dirty']}")
    print(f"[{EXPERIMENT_ID}] cases={record['summary']['passed_cases']}"
          f"/{record['summary']['total_cases']} pass; "
          f"rejected_before_load="
          f"{record['summary']['rejected_before_load']}"
          f"/{record['summary']['fault_cases']}")
    print(f"[{EXPERIMENT_ID}] prefix_baseline: {prefix_status}")
    print(f"[{EXPERIMENT_ID}] prefix_would_accept="
          f"{record['summary']['prefix_would_accept']}")
    print(f"[{EXPERIMENT_ID}] prefix_would_read_outside_root="
          f"{record['summary']['prefix_would_read_outside_root']}")
    print(f"[{EXPERIMENT_ID}] contract_gate_rejections="
          f"{record['summary']['contract_gate_rejections']}")
    print(f"[{EXPERIMENT_ID}] suite_stable_across_repetitions={suite_stable}")
    print(f"[{EXPERIMENT_ID}] identity_hash_stable={hash_stability['stable']} "
          f"hash={hash_stability['distinct_hashes'][0] if hash_stability['distinct_hashes'] else None}")
    print()
    header = (
        f"{'case':<5} {'fault_class':<24} {'exp':<4} {'act':<4} "
        f"{'exit':<5} {'first_bad_file':<28} {'pass':<5}"
    )
    print(header)
    print("-" * len(header))
    for case in cases:
        print(
            f"{case['case_id']:<5} {case['fault_class']:<24} "
            f"{str(case['expected']['rejected']):<4} "
            f"{str(case['actual']['rejected']):<4} "
            f"{case['actual']['exit_code']:<5} "
            f"{_display_path(case['actual']['first_bad_file']):<28} "
            f"{str(case['pass']):<5}"
        )
    agreement = compare_runs(out_dir, record)

    print()
    print(f"[{EXPERIMENT_ID}] raw: {json_path}")
    print(f"[{EXPERIMENT_ID}] csv: {csv_path}")
    print(
        f"[{EXPERIMENT_ID}] cross-process runs={agreement['run_count']} "
        f"agreement={agreement['agreement']}"
    )

    ok = (
        all(case["pass"] for case in cases)
        and suite_stable
        and hash_stability["pass"]
    )

    # Clean up the materialized fixtures; the raw record is self-contained.
    import shutil

    shutil.rmtree(work_root, ignore_errors=True)

    return ExitCode.SUCCESS if ok else ExitCode.BENCHMARK


_CSV_FIELDS = [
    "case_id",
    "fault_class",
    "fault_input",
    "expected_rejected",
    "expected_reason",
    "expected_exit_code",
    "actual_rejected",
    "actual_exit_code",
    "actual_reason_codes",
    "first_bad_file",
    "manifest_sha256",
    "contract_gate_rejected",
    "contract_gate_error",
    "prefix_ok",
    "prefix_would_read_outside_root",
    "legacy_ok",
    "pass",
]


def _write_csv(path: Path, cases: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "fault_class": case["fault_class"],
                    "fault_input": case["fault_input"],
                    "expected_rejected": case["expected"]["rejected"],
                    "expected_reason": case["expected"]["reason"],
                    "expected_exit_code": case["expected"]["exit_code"],
                    "actual_rejected": case["actual"]["rejected"],
                    "actual_exit_code": case["actual"]["exit_code"],
                    "actual_reason_codes": "|".join(case["actual"]["reason_codes"]),
                    "first_bad_file": _display_path(case["actual"]["first_bad_file"]),
                    "manifest_sha256": case["manifest_sha256"],
                    "contract_gate_rejected": case["contract_gate"]["rejected"],
                    "contract_gate_error": case["contract_gate"]["error_type"] or "",
                    "prefix_ok": (
                        case["prefix_gate"].get("ok")
                        if case["prefix_gate"].get("available")
                        else ""
                    ),
                    "prefix_would_read_outside_root": (
                        case["prefix_gate"].get("would_read_outside_root")
                        if case["prefix_gate"].get("available")
                        else ""
                    ),
                    "legacy_ok": case["legacy_gate"].get("ok"),
                    "pass": case["pass"],
                }
            )


if __name__ == "__main__":
    sys.exit(main())
