"""Build/ABI identity and pre-load compatibility (E06-01 §11, E06-09 §5).

The protocol is explicit: an ABI mismatch must be discovered from the manifest
**before** ``dlopen``/kernel launch, not by a crashing kernel (E06-09 §5).  This
module freezes the compatibility requirements, collects the build identity
lazily (a missing toolchain is reported, never guessed), and evaluates the
matrix with structured reasons.  :func:`simulate_identity` builds negative
fixtures without touching the real environment.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

# ── build identity ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BuildIdentity:
    """Machine-readable build/ABI identity of one extension or wheel."""

    python_version: str = ""
    torch_version: str = ""
    torch_build_config: str = ""
    cxx_compiler: str = ""
    cxx_abi_version: str = ""
    cuda_toolkit: str = ""
    cuda_runtime: str = ""
    cuda_driver: str = ""
    gpu_arch: str = ""
    fatbin_targets: Tuple[str, ...] = ()
    extension_soname: str = ""
    extension_sha256: str = ""
    compile_flags: Tuple[str, ...] = ()
    operator_schema_hash: str = ""
    git_commit: str = ""
    dependent_libraries: Tuple[str, ...] = ()
    unknown: Tuple[str, ...] = ()

    def get(self, name: str) -> Any:
        """Read one identity field by name (used by the requirement matrix)."""
        return getattr(self, name)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "torch_build_config": self.torch_build_config,
            "cxx_compiler": self.cxx_compiler,
            "cxx_abi_version": self.cxx_abi_version,
            "cuda_toolkit": self.cuda_toolkit,
            "cuda_runtime": self.cuda_runtime,
            "cuda_driver": self.cuda_driver,
            "gpu_arch": self.gpu_arch,
            "fatbin_targets": list(self.fatbin_targets),
            "extension_soname": self.extension_soname,
            "extension_sha256": self.extension_sha256,
            "compile_flags": list(self.compile_flags),
            "operator_schema_hash": self.operator_schema_hash,
            "git_commit": self.git_commit,
            "dependent_libraries": list(self.dependent_libraries),
            "unknown": list(self.unknown),
        }

    def digest(self) -> str:
        import json

        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()


def _run(command: Sequence[str]) -> Tuple[str, str]:
    try:
        proc = subprocess.run(
            list(command), capture_output=True, text=True, check=False, timeout=5
        )
        if proc.returncode != 0:
            return "", proc.stderr.strip() or f"exit {proc.returncode}"
        return proc.stdout.strip(), ""
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env
        return "", str(exc)


def collect_build_identity(
    *,
    extension_path: str = "",
    compile_flags: Sequence[str] = (),
    operator_schema_hash: str = "",
    cuda_toolkit: str = "",
) -> BuildIdentity:
    """Collect the identity lazily; unavailable pieces are reported, not guessed."""
    unknown: List[str] = []
    torch_version = ""
    torch_config = ""
    cuda_runtime = ""
    gpu_arch = ""
    try:  # pragma: no cover - requires torch
        import torch

        torch_version = str(getattr(torch, "__version__", ""))
        config_fn = getattr(getattr(torch, "utils", None), "collect_env", None)
        if callable(config_fn):
            for line in str(config_fn()).splitlines():
                if line.startswith("PyTorch version"):
                    torch_config = line.split(":", 1)[-1].strip()
        cuda_runtime = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            gpu_arch = f"sm_{major}{minor}"
    except Exception as exc:  # noqa: BLE001 - identity collection never blocks
        unknown.extend(["torch_version", "cuda_runtime", "gpu_arch"])
        torch_config = f"unavailable: {type(exc).__name__}"
    if not torch_version:
        unknown.append("torch_version")
    if not cuda_runtime:
        unknown.append("cuda_runtime")
    if not gpu_arch:
        unknown.append("gpu_arch")

    compiler, compiler_error = _run(["c++", "--version"])
    if compiler_error:
        unknown.append("cxx_compiler")
        compiler = ""
    cxx_abi = ""
    for line in compiler.splitlines():
        match = re.search(r"\bGLIBCXX_(\d+\.\d+\.\d+)", line)
        if match:
            cxx_abi = match.group(1)
    if not cxx_abi:
        unknown.append("cxx_abi_version")

    toolchain = cuda_toolkit
    if not toolchain:
        toolchain, error = _run(["nvcc", "--version"])
        if error:
            unknown.append("cuda_toolkit")
            toolchain = ""
        else:
            match = re.search(r"release (\d+\.\d+)", toolchain)
            toolchain = match.group(1) if match else toolchain.splitlines()[-1]

    driver, driver_error = _run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if driver_error:
        unknown.append("cuda_driver")
        driver = ""
    driver = driver.splitlines()[0].strip() if driver else ""

    extension_sha = ""
    soname = ""
    if extension_path:
        if os.path.isfile(extension_path):
            digest = hashlib.sha256()
            with open(extension_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            extension_sha = digest.hexdigest()
            soname = os.path.basename(extension_path) if not soname else soname
            targets, error = _run(["cuobjdump", "--list-elf", extension_path])
            if not error:
                archs = tuple(sorted(set(re.findall(r"sm_(\d+)", targets))))
                fatbin = tuple(f"sm_{arch}" for arch in archs)
            else:
                fatbin = ()
        else:
            unknown.append("extension_sha256")
            fatbin = ()
    else:
        unknown.append("extension_sha256")
        fatbin = ()
        soname = ""

    commit, _ = _run(["git", "rev-parse", "HEAD"])
    if not commit:
        unknown.append("git_commit")

    return BuildIdentity(
        python_version=platform.python_version(),
        torch_version=torch_version,
        torch_build_config=torch_config,
        cxx_compiler=compiler.splitlines()[0] if compiler else "",
        cxx_abi_version=cxx_abi,
        cuda_toolkit=toolchain,
        cuda_runtime=cuda_runtime,
        cuda_driver=driver,
        gpu_arch=gpu_arch,
        fatbin_targets=fatbin,
        extension_soname=soname,
        extension_sha256=extension_sha,
        compile_flags=tuple(compile_flags),
        operator_schema_hash=operator_schema_hash,
        git_commit=commit,
        dependent_libraries=(),
        unknown=tuple(sorted(set(unknown))),
    )


def simulate_identity(base: Optional[BuildIdentity] = None, **overrides: Any) -> BuildIdentity:
    """Build a (negative) fixture identity by overriding fields."""
    identity = base if base is not None else BuildIdentity(
        python_version="3.12.0",
        torch_version="2.5.0",
        cxx_compiler="c++ 13.2.0",
        cxx_abi_version="3.4.30",
        cuda_toolkit="12.6",
        cuda_runtime="12.6",
        cuda_driver="560.35",
        gpu_arch="sm_86",
        fatbin_targets=("sm_86",),
        extension_soname="libhqsb_ops.so",
        extension_sha256="0" * 64,
        operator_schema_hash="1" * 64,
        git_commit="0" * 40,
    )
    for key in overrides:
        if key not in identity.as_dict():
            raise ConfigError(
                f"cannot simulate unknown identity field {key!r}",
                details={"field": key},
            )
    return replace(identity, **overrides)


# ── compatibility matrix ──────────────────────────────────────────────────


class CompatibilityRule:
    EXACT = "exact"
    PREFIX = "prefix"
    MIN = "min"
    IN_SET = "in_set"
    ARCH_SUPPORTED = "arch_supported"

    ALL = (EXACT, PREFIX, MIN, IN_SET, ARCH_SUPPORTED)


@dataclass(frozen=True)
class CompatibilityRequirement:
    """One requirement on one identity field."""

    field_name: str
    rule: str
    expected: Any
    reason_code: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.rule not in CompatibilityRule.ALL:
            raise ConfigError(
                f"{self.field_name}: unknown compatibility rule {self.rule!r}",
                details={"field": "rule", "allowed": list(CompatibilityRule.ALL)},
            )

    def check(self, identity: BuildIdentity) -> Optional[Dict[str, Any]]:
        actual = identity.get(self.field_name)
        ok = True
        if self.rule == CompatibilityRule.EXACT:
            ok = str(actual) == str(self.expected)
        elif self.rule == CompatibilityRule.PREFIX:
            ok = str(actual).startswith(str(self.expected))
        elif self.rule == CompatibilityRule.MIN:
            ok = _version_ge(str(actual), str(self.expected))
        elif self.rule == CompatibilityRule.IN_SET:
            ok = str(actual) in {str(item) for item in self.expected}
        elif self.rule == CompatibilityRule.ARCH_SUPPORTED:
            targets = tuple(str(item) for item in identity.fatbin_targets)
            ok = str(actual) in targets if targets else False
        if ok:
            return None
        return {
            "field": self.field_name,
            "rule": self.rule,
            "expected": self.expected,
            "actual": actual,
            "reason_code": self.reason_code,
            "note": self.note,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field_name,
            "rule": self.rule,
            "expected": self.expected,
            "reason_code": self.reason_code,
            "note": self.note,
        }


def _version_ge(actual: str, expected: str) -> bool:
    def parts(value: str) -> Tuple[int, ...]:
        return tuple(int(item) for item in re.findall(r"\d+", value)[:4])

    return parts(actual) >= parts(expected)


@dataclass(frozen=True)
class CompatibilityReport:
    """``ok``/``failures`` with pre-load semantics made explicit."""

    ok: bool
    failures: Tuple[Dict[str, Any], ...] = ()
    pre_load: bool = True

    def codes(self) -> Tuple[str, ...]:
        return tuple(str(item["reason_code"]) for item in self.failures)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "pre_load": self.pre_load,
            "failures": [dict(item) for item in self.failures],
            "codes": list(self.codes()),
        }


#: The frozen requirement set (a version change here is a schema-level decision).
ABI_REQUIREMENTS: Tuple[CompatibilityRequirement, ...] = (
    CompatibilityRequirement(
        "python_version", CompatibilityRule.PREFIX, "3.1", "ABI_MISMATCH",
        note="wheel tags are cp3x; a mismatch must be refused before load",
    ),
    CompatibilityRequirement(
        "torch_version", CompatibilityRule.MIN, "2.4", "ABI_MISMATCH",
        note="custom ops link against the PyTorch C++ ABI",
    ),
    CompatibilityRequirement(
        "cxx_abi_version", CompatibilityRule.MIN, "3.4.0", "ABI_MISMATCH",
        note="libstdc++ ABI must be forward compatible",
    ),
    CompatibilityRequirement(
        "cuda_runtime", CompatibilityRule.MIN, "12.0", "ABI_MISMATCH",
        note="CUDA runtime must satisfy the toolkit the binary was built with",
    ),
    CompatibilityRequirement(
        "gpu_arch", CompatibilityRule.ARCH_SUPPORTED, "sm_86", "ARCH_UNSUPPORTED",
        note="the binary must declare a fatbin target for the device arch",
    ),
)


@dataclass(frozen=True)
class CompatibilityMatrix:
    """Evaluate requirements against one identity (never guesses a field)."""

    requirements: Tuple[CompatibilityRequirement, ...] = ABI_REQUIREMENTS
    allow_unknown_fields: bool = False

    def check(self, identity: BuildIdentity) -> CompatibilityReport:
        failures: List[Dict[str, Any]] = []
        for requirement in self.requirements:
            failure = requirement.check(identity)
            if failure:
                failures.append(failure)
                continue
            if requirement.field_name in identity.unknown and not self.allow_unknown_fields:
                failures.append(
                    {
                        "field": requirement.field_name,
                        "rule": "known",
                        "expected": requirement.expected,
                        "actual": "<unknown>",
                        "reason_code": "ABI_MISMATCH",
                        "note": "identity field could not be collected; refusing to assume",
                    }
                )
        return CompatibilityReport(ok=not failures, failures=tuple(failures))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requirements": [item.as_dict() for item in self.requirements],
            "allow_unknown_fields": self.allow_unknown_fields,
        }


# ── binary manifest / symbol audit ────────────────────────────────────────


@dataclass(frozen=True)
class BinaryManifest:
    """Declared facts about one compiled extension, checked before load."""

    path: str
    sha256: str
    target_arch: str
    abi_version: str
    operator_schema_hash: str = ""
    required_symbols: Tuple[str, ...] = ()
    provided_symbols: Tuple[str, ...] = ()
    dependent_libraries: Tuple[str, ...] = ()

    def verify_file(self) -> Dict[str, Any]:
        """Verify existence/permissions/hash before anything is imported."""
        if not os.path.isfile(self.path):
            return {
                "ok": False,
                "reason_code": "LINK_LOAD_FAILED",
                "detail": f"missing binary: {self.path}",
            }
        if not os.access(self.path, os.R_OK):
            return {
                "ok": False,
                "reason_code": "LINK_LOAD_FAILED",
                "detail": f"unreadable binary: {self.path}",
            }
        digest = hashlib.sha256()
        with open(self.path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != self.sha256:
            return {
                "ok": False,
                "reason_code": "ABI_MISMATCH",
                "detail": "binary hash does not match the manifest",
                "expected": self.sha256,
                "actual": actual,
            }
        return {"ok": True, "sha256": actual}

    def symbol_audit(self) -> Dict[str, Any]:
        missing = sorted(set(self.required_symbols) - set(self.provided_symbols))
        return {
            "ok": not missing,
            "missing": missing,
            "reason_code": "SYMBOL_MISSING" if missing else "",
            "required": len(self.required_symbols),
            "provided": len(self.provided_symbols),
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "target_arch": self.target_arch,
            "abi_version": self.abi_version,
            "operator_schema_hash": self.operator_schema_hash,
            "required_symbols": list(self.required_symbols),
            "provided_symbols": list(self.provided_symbols),
            "dependent_libraries": list(self.dependent_libraries),
        }


def preload_check(
    identity: BuildIdentity,
    manifest: Optional[BinaryManifest] = None,
    matrix: Optional[CompatibilityMatrix] = None,
) -> Dict[str, Any]:
    """The single pre-load gate: identity matrix then manifest then symbols."""
    report = (matrix or CompatibilityMatrix()).check(identity)
    payload: Dict[str, Any] = {
        "identity": report.as_dict(),
        "stage": "binary_abi_load",
        "pre_kernel": True,
    }
    if not report.ok:
        payload["ok"] = False
        return payload
    if manifest is not None:
        file_result = manifest.verify_file()
        symbol_result = manifest.symbol_audit()
        payload["manifest"] = {"file": file_result, "symbols": symbol_result}
        payload["ok"] = bool(file_result["ok"] and symbol_result["ok"])
        return payload
    payload["ok"] = True
    return payload


def py_abi_matches(python_version: str, wheel_tag: str) -> bool:
    """Check a wheel tag (``cp312``) against the interpreter version."""
    major, minor = python_version.split(".")[:2]
    return wheel_tag.lower() == f"cp{major}{minor}"


__all__ = [
    "ABI_REQUIREMENTS",
    "BinaryManifest",
    "BuildIdentity",
    "CompatibilityMatrix",
    "CompatibilityReport",
    "CompatibilityRequirement",
    "CompatibilityRule",
    "collect_build_identity",
    "preload_check",
    "py_abi_matches",
    "simulate_identity",
]
