"""Target snapshots and capability analysis for lowering.

Protocol anchors: ``details/S11/E11-03`` §3.3 (capability vs per-call guard),
steps 8 and 12 (target snapshot, capability analysis before materialisation),
``details/S11/E11-04`` step 1 (freeze the target identity) and §13 of the
details README (corruption/ABI failures).

Design rules enforced here:

* a target snapshot must freeze exact versions — ``latest``/``main`` are
  rejected (mirrors the S10 identity discipline);
* probing is read-only and never raises: an absent tool yields
  ``UNAVAILABLE(reason)``, never ``0`` and never a fabricated value;
* capability is a *class* property (arch/dtype/features/ABI), while
  per-call conditions (shape/stride/eps) belong to guards (E11-05).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import (
    canonical_json,
    require_frozen_versions,
    sha256_text,
    version_is_frozen,
)

# ── target snapshot ────────────────────────────────────────────────────────

BACKEND_KINDS: Tuple[str, ...] = ("cuda", "triton", "cutlass", "inductor", "ascend", "cpu_reference")

#: Capability reject reasons (structured, never a bare boolean).
CAPABILITY_REASONS: Tuple[str, ...] = (
    "TARGET_UNSUPPORTED",
    "DTYPE_UNSUPPORTED",
    "LAYOUT_UNSUPPORTED",
    "FEATURE_MISSING",
    "RESOURCE_INSUFFICIENT",
    "ALIGNMENT_REQUIRED",
    "WORKSPACE_EXCEEDED",
    "ABI_MISMATCH",
    "PACKAGE_MISSING",
    "EVIDENCE_MISSING",
)


@dataclass
class TargetSnapshot:
    """A frozen description of what the target can execute (E11-03 step 8)."""

    target_id: str
    device_kind: str
    device_name: str = ""
    device_uuid: str = ""
    arch: str = ""
    driver_version: str = ""
    runtime_version: str = ""
    compiler_version: str = ""
    backend_versions: Mapping[str, str] = field(default_factory=dict)
    dtypes: Tuple[str, ...] = ()
    features: Tuple[str, ...] = ()
    shared_memory_per_block_bytes: Optional[int] = None
    registers_per_thread: Optional[int] = None
    max_threads_per_block: Optional[int] = None
    abi_version: str = "1"
    target_triple: str = ""
    collector: str = ""
    collected_at: str = ""
    unavailable: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("target_id", "device_kind", "arch"):
            if not getattr(self, name):
                problems.append(f"target snapshot missing {name!r}")
        if self.device_kind not in BACKEND_KINDS:
            problems.append(f"unknown device_kind {self.device_kind!r}")
        if self.device_kind == "cpu_reference":
            return problems
        vague = require_frozen_versions(
            {
                "driver_version": self.driver_version,
                "runtime_version": self.runtime_version,
                "compiler_version": self.compiler_version,
                **{f"backend:{k}": v for k, v in self.backend_versions.items()},
            }
        )
        for name in vague:
            problems.append(f"{name} is not a frozen version (no latest/main/empty)")
        if not self.dtypes:
            problems.append("target snapshot must list the approved dtypes")
        if not self.abi_version:
            problems.append("target snapshot must declare an ABI version")
        for key, reason in self.unavailable.items():
            if not reason:
                problems.append(f"unavailable field {key!r} needs a reason")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "device_kind": self.device_kind,
            "device_name": self.device_name,
            "device_uuid": self.device_uuid,
            "arch": self.arch,
            "driver_version": self.driver_version,
            "runtime_version": self.runtime_version,
            "compiler_version": self.compiler_version,
            "backend_versions": dict(sorted(self.backend_versions.items())),
            "dtypes": list(self.dtypes),
            "features": list(self.features),
            "shared_memory_per_block_bytes": self.shared_memory_per_block_bytes,
            "registers_per_thread": self.registers_per_thread,
            "max_threads_per_block": self.max_threads_per_block,
            "abi_version": self.abi_version,
            "target_triple": self.target_triple,
            "collector": self.collector,
            "collected_at": self.collected_at,
            "unavailable": dict(sorted(self.unavailable.items())),
        }

    def sha256(self) -> str:
        return sha256_text(canonical_json(self.as_dict()))

    @property
    def fingerprint(self) -> str:
        """Short form used in reports (full hash stays in the manifest)."""
        return self.sha256()[:12]


# ── read-only probes ───────────────────────────────────────────────────────

PROBE_TIMEOUT_S = 10.0


def _run(command: Sequence[str]) -> Tuple[int, str, str]:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            list(command),
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    return completed.returncode, completed.stdout, completed.stderr


def cuda_target_snapshot(*, target_id: str = "cuda-device") -> TargetSnapshot:
    """Probe the CUDA device read-only (never raises, never fabricates)."""
    unavailable: Dict[str, str] = {}
    name = uuid = driver = smi_arch = ""

    if shutil.which("nvidia-smi") is None:
        unavailable["nvidia-smi"] = "tool_not_installed"
    else:
        code, out, err = _run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,compute_cap",
                "--format=csv,noheader",
            ]
        )
        if code != 0 or not out.strip():
            unavailable["nvidia-smi"] = f"query_failed:{code}:{err.strip()[:80]}"
        else:
            first = out.strip().splitlines()[0]
            parts = [item.strip() for item in first.split(",")]
            name = parts[0] if parts else ""
            uuid = parts[1] if len(parts) > 1 else ""
            driver = parts[2] if len(parts) > 2 else ""
            if len(parts) > 3 and parts[3]:
                compute_cap = parts[3].replace(".", "")  # e.g. 8.6 -> 86
                smi_arch = f"sm_{compute_cap}"

    torch_version = ""
    arch = ""
    if shutil.which("nvcc"):
        code, out, _ = _run(["nvcc", "--version"])
        if code == 0:
            compiler_version = _parse_nvcc_version(out)
        else:
            compiler_version = ""
            unavailable["nvcc"] = "version_query_failed"
    else:
        compiler_version = ""
        unavailable["nvcc"] = "tool_not_installed"
    try:  # lazy optional dependency; the package must import without torch
        import importlib

        torch = importlib.import_module("torch")
        torch_version = str(getattr(torch, "__version__", ""))
        if getattr(torch.version, "cuda", None):
            unavailable["torch_cuda"] = ""
        if torch.cuda.is_available():
            arch = f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}"
    except Exception as exc:
        unavailable["torch"] = f"NOT_INSTALLED:{type(exc).__name__}"

    if not arch and smi_arch and smi_arch != "sm_unknown":
        arch = smi_arch
    if not arch:
        unavailable["arch"] = "no_probe_available"

    backend_versions = {}
    if torch_version and version_is_frozen(torch_version):
        backend_versions["torch"] = torch_version
    if compiler_version:
        backend_versions["nvcc"] = compiler_version
    triton_version = ""
    try:
        import importlib

        triton = importlib.import_module("triton")
        triton_version = str(getattr(triton, "__version__", ""))
        if triton_version:
            backend_versions["triton"] = triton_version
    except Exception:  # pragma: no cover - depends on environment
        triton_version = ""

    return TargetSnapshot(
        target_id=target_id,
        device_kind="cuda",
        device_name=name,
        device_uuid=uuid,
        arch=arch or "unknown",
        driver_version=driver,
        runtime_version=torch_version,
        compiler_version=compiler_version,
        backend_versions=backend_versions,
        dtypes=("fp16", "fp32", "bf16"),
        features=_arch_features(arch),
        abi_version="1",
        target_triple="x86_64-linux-gnu",
        collector="hqsb.compiler.targets.cuda_target_snapshot",
        unavailable=unavailable,
    )


def _arch_features(arch: str) -> Tuple[str, ...]:
    """Derive the feature clauses implied by an SM arch (no guessing beyond it)."""
    if not arch.startswith("sm_"):
        return ()
    try:
        number = int(arch.split("_")[1])
    except (IndexError, ValueError):
        return ()
    features: List[str] = []
    if number >= 70:
        features.append("tensor_core")
    if number >= 75:
        features.append("tensor_core_fp16")
    if number >= 80:
        features.append("tf32")
    if number >= 90:
        features.append("tma")
    return tuple(features)


def _parse_nvcc_version(text: str) -> str:
    for line in text.splitlines():
        token = line.split("release")
        if len(token) == 2:
            return token[1].split(",")[0].strip()
    return ""


def cpu_target_snapshot(*, target_id: str = "cpu-reference") -> TargetSnapshot:
    return TargetSnapshot(
        target_id=target_id,
        device_kind="cpu_reference",
        device_name="cpu",
        arch="x86_64",
        dtypes=("fp32", "fp16"),
        features=(),
        abi_version="1",
        target_triple="x86_64-linux-gnu",
        collector="hqsb.compiler.targets.cpu_target_snapshot",
        unavailable={"device_timing": "cpu reference target has no device timers"},
    )


def ascend_target_snapshot(*, target_id: str = "ascend-npu") -> TargetSnapshot:
    """Ascend branch: explicit UNAVAILABLE unless the CANN probe succeeds."""
    unavailable: Dict[str, str] = {}
    if shutil.which("npu-smi") is None:
        unavailable["npu-smi"] = "tool_not_installed"
    return TargetSnapshot(
        target_id=target_id,
        device_kind="ascend",
        device_name="ascend-npu",
        arch="ascend-unknown",
        driver_version="",
        runtime_version="",
        compiler_version="",
        dtypes=("fp16", "fp32"),
        features=(),
        abi_version="1",
        target_triple="aarch64-linux-gnu",
        collector="hqsb.compiler.targets.ascend_target_snapshot",
        unavailable={**unavailable, "cann_runtime": "NOT_RUN_TOOL_UNAVAILABLE"},
    )


def probe_target(device_kind: str = "cuda") -> TargetSnapshot:
    if device_kind == "cuda":
        return cuda_target_snapshot()
    if device_kind == "cpu_reference":
        return cpu_target_snapshot()
    if device_kind == "ascend":
        return ascend_target_snapshot()
    raise ConfigError(f"unknown device kind {device_kind!r}")


# ── capability requirements ────────────────────────────────────────────────


@dataclass
class CapabilityRequirement:
    """What one lowering entry needs from the target (E11-03 step 9)."""

    requirement_id: str
    backends: Tuple[str, ...] = ()
    dtypes: Tuple[str, ...] = ()
    layouts: Tuple[str, ...] = ()
    archs: Tuple[str, ...] = ()
    features: Tuple[str, ...] = ()
    min_arch: str = ""
    alignment_bytes: int = 0
    workspace_bytes: int = 0
    shared_memory_bytes: int = 0
    registers_per_thread: int = 0
    abi_version: str = ""
    packages: Tuple[str, ...] = ()
    evidence_scope: str = ""
    reason_on_failure: str = "TARGET_UNSUPPORTED"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.reason_on_failure not in CAPABILITY_REASONS:
            problems.append(f"unknown capability reason {self.reason_on_failure!r}")
        if not self.evidence_scope:
            problems.append(
                f"{self.requirement_id}: capability must reference a correctness evidence scope"
            )
        for backend in self.backends:
            if backend not in BACKEND_KINDS:
                problems.append(f"{self.requirement_id}: unknown backend {backend!r}")
        return problems

    def check(self, snapshot: TargetSnapshot) -> "CapabilityOutcome":
        """Evaluate the requirement against a snapshot (structured reject reason)."""
        def reject(reason: str, detail: str) -> "CapabilityOutcome":
            return CapabilityOutcome(
                requirement_id=self.requirement_id,
                available=False,
                reason_code=reason,
                detail=detail,
            )

        if self.backends and snapshot.device_kind not in self.backends:
            return reject(
                "TARGET_UNSUPPORTED",
                f"device {snapshot.device_kind!r} not in {list(self.backends)}",
            )
        if self.archs or self.min_arch:
            arch = snapshot.arch
            if self.archs and arch not in self.archs:
                return reject(
                    "TARGET_UNSUPPORTED",
                    f"arch {arch!r} not in {list(self.archs)}",
                )
            if self.min_arch and arch and arch.startswith("sm_"):
                try:
                    if int(arch.split("_")[1]) < int(self.min_arch.split("_")[1]):
                        return reject(
                            "TARGET_UNSUPPORTED",
                            f"arch {arch} below minimum {self.min_arch}",
                        )
                except (IndexError, ValueError):
                    return reject("TARGET_UNSUPPORTED", f"unparsable arch {arch!r}")
        if self.dtypes:
            missing = [dtype for dtype in self.dtypes if dtype not in snapshot.dtypes]
            if missing:
                return reject("DTYPE_UNSUPPORTED", f"dtypes not in snapshot: {missing}")
        if self.features:
            missing = [feature for feature in self.features if feature not in snapshot.features]
            if missing:
                return reject("FEATURE_MISSING", f"features missing: {missing}")
        if self.abi_version and self.abi_version != snapshot.abi_version:
            return reject(
                "ABI_MISMATCH",
                f"artifact ABI {self.abi_version} vs target ABI {snapshot.abi_version}",
            )
        if self.shared_memory_bytes:
            limit = snapshot.shared_memory_per_block_bytes
            if limit is None:
                return reject("RESOURCE_INSUFFICIENT", "shared-memory limit unknown on target")
            if self.shared_memory_bytes > limit:
                return reject(
                    "RESOURCE_INSUFFICIENT",
                    f"shared {self.shared_memory_bytes} > limit {limit}",
                )
        if self.registers_per_thread:
            limit = snapshot.registers_per_thread
            if limit is not None and self.registers_per_thread > limit:
                return reject(
                    "RESOURCE_INSUFFICIENT",
                    f"registers {self.registers_per_thread} > limit {limit}",
                )
        if self.packages:
            missing = [name for name in self.packages if name not in snapshot.backend_versions]
            if missing:
                return reject("PACKAGE_MISSING", f"backend packages missing: {missing}")
        return CapabilityOutcome(
            requirement_id=self.requirement_id,
            available=True,
            reason_code="",
            detail="all capability conditions satisfied",
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "backends": list(self.backends),
            "dtypes": list(self.dtypes),
            "layouts": list(self.layouts),
            "archs": list(self.archs),
            "features": list(self.features),
            "min_arch": self.min_arch,
            "alignment_bytes": self.alignment_bytes,
            "workspace_bytes": self.workspace_bytes,
            "shared_memory_bytes": self.shared_memory_bytes,
            "registers_per_thread": self.registers_per_thread,
            "abi_version": self.abi_version,
            "packages": list(self.packages),
            "evidence_scope": self.evidence_scope,
            "reason_on_failure": self.reason_on_failure,
        }


@dataclass
class CapabilityOutcome:
    requirement_id: str
    available: bool
    reason_code: str = ""
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "available": self.available,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }


def capability_matrix(
    requirements: Sequence[CapabilityRequirement], snapshot: TargetSnapshot
) -> Dict[str, Any]:
    """Per-candidate capability outcomes with structured reject reasons."""
    rows = []
    problems: List[str] = []
    for requirement in requirements:
        problems.extend(requirement.validate())
        rows.append({**requirement.as_dict(), **requirement.check(snapshot).as_dict()})
    return {
        "target_id": snapshot.target_id,
        "target_fingerprint": snapshot.fingerprint,
        "rows": rows,
        "available": [row["requirement_id"] for row in rows if row["available"]],
        "rejected": [
            {"id": row["requirement_id"], "reason": row["reason_code"]}
            for row in rows
            if not row["available"]
        ],
        "problems": problems,
    }


def target_field_diff(before: TargetSnapshot, after: TargetSnapshot) -> List[Dict[str, Any]]:
    """Field-level diff for drift/invalidation reports."""
    rows: List[Dict[str, Any]] = []
    left, right = before.as_dict(), after.as_dict()
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            rows.append({"field": key, "before": left.get(key), "after": right.get(key)})
    return rows


def snapshot_has_unavailable_required_fields(snapshot: TargetSnapshot) -> List[str]:
    """Fields a *claim* depends on; a report must list them as limitations."""
    required = ("device_uuid", "driver_version", "runtime_version", "compiler_version")
    return [name for name in required if not getattr(snapshot, name)]


def toolchain_export_capability(snapshot: TargetSnapshot) -> Dict[str, Any]:
    """Which IR/binary export tools exist for the attribution chain (E11-04)."""
    tools = {
        "nvcc": shutil.which("nvcc") or "",
        "cuobjdump": shutil.which("cuobjdump") or "",
        "nvdisasm": shutil.which("nvdisasm") or "",
        "nsys": shutil.which("nsys") or "",
        "ncu": shutil.which("ncu") or "",
    }
    missing = [name for name, path in tools.items() if not path]
    return {
        "target_id": snapshot.target_id,
        "tools": dict(sorted(tools.items())),
        "missing": missing,
        "status": "OK" if not missing else "NOT_RUN_TOOL_UNAVAILABLE",
        "rule": "a missing tool lowers the claim (unavailable layer), it does not become an estimate",
    }


def environment_lock(extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Minimal environment record for the target snapshot sidecar."""
    payload = {
        "python": sys.version.split()[0],
        "compiler_root": os.environ.get("CUDA_HOME", ""),
        "path_has_nvcc": bool(shutil.which("nvcc")),
        "extra": dict(extra or {}),
    }
    return payload
