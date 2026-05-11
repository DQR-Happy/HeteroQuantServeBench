"""Run / environment fingerprinting (S00 E00-03).

A *run fingerprint* is a canonical, deterministic SHA256 identity for one
benchmark run, derived from the environment and inputs that the experiment
record must freeze (see ``docs/stage_experiments/README.md`` §4 and the S00
experiment list E00-03).

The fingerprint separates two halves of a run's identity:

* **environment sections** — ``os``, ``device``, ``python``, ``packages``,
  ``power`` — the "same machine / same configuration" half; and
* **input sections** — ``config``, ``model``, ``commit`` — the "what exactly
  did we run" half.

Each section is hashed independently, so a single changed field is traceable
to exactly one section digest. Two aggregate roots are then derived:

* ``environment_fingerprint`` — identity of the environment half only; and
* ``run_fingerprint`` — identity over *all* section digests.

Clock / temperature / "observed at" values are deliberately excluded from the
identity roots and collected into :class:`VolatileObservations` instead: they
legitimately vary between two consecutive runs, but the experiment still
requires them to be *recorded* (and their changes to be observable through
``volatile_digest``) without corrupting the run identity.

This module is part of the dependency-free ``hqsb.core`` layer: it imports
only the standard library and Pydantic. Concrete device facts (CUDA runtime,
device names) are injected by the caller through ``device_probe`` so that
``core`` never imports torch or any backend.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import Field

from hqsb.core.contracts.base import VersionedModel

#: Ordering of sections used when computing aggregate digests. ``packages``
#: is a plain mapping, all others are Pydantic section models.
SECTION_NAMES: Tuple[str, ...] = (
    "os",
    "device",
    "python",
    "packages",
    "power",
    "config",
    "model",
    "commit",
)

#: The subset of sections that describe the *environment* (same machine /
#: same software / same power state). Excludes the run-input sections.
ENVIRONMENT_SECTION_NAMES: Tuple[str, ...] = (
    "os",
    "device",
    "python",
    "packages",
    "power",
)

#: Packages probed by default for the ``packages`` section. These are
#: distribution names; :func:`collect_packages` falls back to a module-level
#: ``__version__`` lookup when no distribution metadata is present.
DEFAULT_PACKAGE_WATCHLIST: Tuple[str, ...] = (
    "accelerate",
    "modelscope",
    "numpy",
    "onnx",
    "onnxruntime",
    "pydantic",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
    "triton",
    "psutil",
)

_MISSING_PACKAGE = "<missing>"

# ── Canonical serialization ─────────────────────────────────────────────


def canonical_json(obj: Any) -> str:
    """Return a deterministic JSON encoding (sorted keys, compact separators).

    This is the single serialization used for every digest, so identical
    payloads always hash identically across processes, machines, and dict
    insertion orders.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    """Return the lowercase SHA256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Section models ──────────────────────────────────────────────────────


class OsSection(VersionedModel):
    SCHEMA_VERSION = "1.0.0"

    system: str = Field("", description="platform.system(), e.g. 'Linux'.")
    release: str = Field("", description="Kernel release, e.g. '5.15.148-tegra'.")
    machine: str = Field("", description="Architecture, e.g. 'aarch64'.")
    platform: str = Field("", description="platform.platform() full string.")
    libc: str = Field("", description="libc family/version, e.g. 'glibc 2.35'.")


class DeviceSection(VersionedModel):
    SCHEMA_VERSION = "1.0.0"

    cuda_available: bool = Field(False, description="Whether CUDA is available.")
    device_count: int = Field(0, description="Number of visible devices.")
    device_names: List[str] = Field(default_factory=list, description="Device names.")
    compute_capabilities: List[List[int]] = Field(
        default_factory=list, description="[major, minor] per device."
    )
    cuda_runtime_version: str = Field("", description="CUDA runtime version.")
    cuda_driver_version: str = Field("", description="NVIDIA driver version.")
    nvcc_version: str = Field("", description="nvcc compiler release.")
    board_compatible: str = Field("", description="Device-tree compatible string.")
    l4t_release: str = Field("", description="L4T/JetPack release string.")


class PythonSection(VersionedModel):
    SCHEMA_VERSION = "1.0.0"

    version: str = Field("", description="Python version, e.g. '3.10.12'.")
    implementation: str = Field("", description="CPython / PyPy, ...")
    executable: str = Field("", description="Interpreter path.")


class PowerSection(VersionedModel):
    SCHEMA_VERSION = "1.0.0"

    nvpmodel_mode: Optional[int] = Field(None, description="nvpmodel mode id.")
    nvpmodel_name: Optional[str] = Field(None, description="nvpmodel mode name.")
    jetson_clocks_active: Optional[bool] = Field(
        None, description="Whether jetson_clocks is active (inferred)."
    )
    cpu_governor: Optional[str] = Field(None, description="CPU scaling governor.")
    gpu_governor: Optional[str] = Field(None, description="GPU devfreq governor.")


class ConfigSection(VersionedModel):
    SCHEMA_VERSION = "1.0.0"

    config_path: str = Field("", description="Repo-relative config file path.")
    config_sha256: str = Field("", description="SHA256 of the raw config file.")
    config_hash: str = Field("", description="Semantic hash of parsed config.")


class ModelSection(VersionedModel):
    SCHEMA_VERSION = "1.0.0"

    model_id: str = Field("", description="Model identifier.")
    manifest_path: str = Field("", description="Repo-relative manifest path.")
    manifest_sha256: str = Field("", description="SHA256 of the manifest file.")
    model_hash: str = Field("", description="ModelArtifact.identity_hash().")


class CommitSection(VersionedModel):
    SCHEMA_VERSION = "1.0.0"

    commit: str = Field("", description="Full git commit SHA.")
    commit_short: str = Field("", description="Short git commit SHA.")
    dirty: bool = Field(False, description="Working tree dirty flag.")


class VolatileObservations(VersionedModel):
    SCHEMA_VERSION = "1.0.0"

    observed_at_utc: str = Field("", description="Observation timestamp (UTC).")
    cpu_cur_freq_khz: Optional[int] = Field(None, description="Current CPU freq (kHz).")
    gpu_cur_freq_hz: Optional[int] = Field(None, description="Current GPU freq (Hz).")
    max_temperature_c: Optional[float] = Field(None, description="Max thermal zone (C).")


class FingerprintSections(VersionedModel):
    """The eight identity sections that together define a run fingerprint."""

    SCHEMA_VERSION = "1.0.0"

    os: OsSection = Field(default_factory=OsSection)
    device: DeviceSection = Field(default_factory=DeviceSection)
    python: PythonSection = Field(default_factory=PythonSection)
    packages: Dict[str, str] = Field(default_factory=dict)
    power: PowerSection = Field(default_factory=PowerSection)
    config: ConfigSection = Field(default_factory=ConfigSection)
    model: ModelSection = Field(default_factory=ModelSection)
    commit: CommitSection = Field(default_factory=CommitSection)


class RunFingerprint(VersionedModel):
    """A canonical run fingerprint plus its per-section digests."""

    SCHEMA_VERSION = "1.0.0"

    sections: FingerprintSections = Field(..., description="The eight sections.")
    volatile: VolatileObservations = Field(
        default_factory=VolatileObservations, description="Volatile observations."
    )
    section_digests: Dict[str, str] = Field(
        default_factory=dict, description="SHA256 per section name."
    )
    environment_fingerprint: str = Field(
        "", description="Aggregate identity of the environment sections."
    )
    volatile_digest: str = Field(
        "", description="SHA256 of the volatile observations."
    )
    run_fingerprint: str = Field(
        "", description="Aggregate identity over all section digests."
    )

    def project(self, *section_names: str) -> Dict[str, str]:
        """Return the requested ``name -> digest`` subset."""
        return {name: self.section_digests[name] for name in section_names}

    def environment_matches(self, other: "RunFingerprint") -> bool:
        """True when the environment half of two fingerprints is identical."""
        return self.environment_fingerprint == other.environment_fingerprint

    def verify(self) -> None:
        """Recompute all digests from ``sections`` and raise on any drift.

        This is a self-check used by the experiment: it proves the stored
        aggregate digests are always recomputable from the sections, not
        copy-pasted numbers.
        """
        expected = compute_run_fingerprint(self.sections, self.volatile)
        if self.model_dump(mode="json") != expected.model_dump(mode="json"):
            raise ValueError("fingerprint digests drifted from their sections")


# ── Digest computation ──────────────────────────────────────────────────


def section_payload(sections: FingerprintSections) -> Dict[str, Any]:
    """Return the canonical, JSON-serializable per-section payload."""
    return {
        "os": sections.os.model_dump(mode="json"),
        "device": sections.device.model_dump(mode="json"),
        "python": sections.python.model_dump(mode="json"),
        "packages": sections.packages,
        "power": sections.power.model_dump(mode="json"),
        "config": sections.config.model_dump(mode="json"),
        "model": sections.model.model_dump(mode="json"),
        "commit": sections.commit.model_dump(mode="json"),
    }


def section_digests(sections: FingerprintSections) -> Dict[str, str]:
    """Compute the independent SHA256 digest of each section."""
    payload = section_payload(sections)
    return {name: sha256_hex(canonical_json(payload[name])) for name in SECTION_NAMES}


def compute_run_fingerprint(
    sections: FingerprintSections, volatile: VolatileObservations
) -> RunFingerprint:
    """Build a :class:`RunFingerprint` from sections and volatile data.

    ``environment_fingerprint`` covers only the environment sections;
    ``run_fingerprint`` covers all eight sections. ``volatile_digest``
    records (but does not identify) the volatile observations.
    """
    digests = section_digests(sections)
    environment_fingerprint = sha256_hex(
        canonical_json({name: digests[name] for name in ENVIRONMENT_SECTION_NAMES})
    )
    run_fingerprint = sha256_hex(canonical_json(digests))
    volatile_digest = sha256_hex(canonical_json(volatile.model_dump(mode="json")))
    return RunFingerprint(
        sections=sections,
        volatile=volatile,
        section_digests=digests,
        environment_fingerprint=environment_fingerprint,
        volatile_digest=volatile_digest,
        run_fingerprint=run_fingerprint,
    )


def diff_sections(a: RunFingerprint, b: RunFingerprint) -> Dict[str, str]:
    """Return the section names whose digests differ between two fingerprints."""
    return sorted(
        name
        for name in SECTION_NAMES
        if a.section_digests.get(name) != b.section_digests.get(name)
    )


# ── Collectors (environment half; dependency-free) ──────────────────────


def _read_text(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        if not raw:
            return ""
        return raw.decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _run(cmd: Sequence[str], timeout_s: float = 10.0) -> str:
    try:
        proc = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=timeout_s
        )
        return proc.stdout or ""
    except (OSError, subprocess.SubprocessError):
        return ""


def collect_os() -> OsSection:
    libc = platform.libc_ver() if hasattr(platform, "libc_ver") else ("", "")
    return OsSection(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        platform=platform.platform(),
        libc=" ".join(part for part in libc if part),
    )


def collect_python() -> PythonSection:
    return PythonSection(
        version=sys.version.split()[0],
        implementation=platform.python_implementation(),
        executable=sys.executable,
    )


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        pass
    # Fallback: import the module and read its __version__.
    try:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", _MISSING_PACKAGE))
    except Exception:
        return _MISSING_PACKAGE


def collect_packages(watchlist: Sequence[str] = DEFAULT_PACKAGE_WATCHLIST) -> Dict[str, str]:
    return {name: _distribution_version(name) for name in watchlist}


def _nvpmodel_mode_and_name() -> Tuple[Optional[int], Optional[str]]:
    out = _run(["nvpmodel", "-q"])
    name: Optional[str] = None
    mode: Optional[int] = None
    for line in out.splitlines():
        if "Power Mode" in line:
            candidate = line.split(":", 1)[-1].strip()
            if candidate:
                name = candidate
    for line in reversed(out.splitlines()):
        token = line.strip()
        if token.isdigit():
            mode = int(token)
            break
    if mode is None:
        text = _read_text("/var/lib/nvpmodel/status")
        match = re.search(r"(?:pmode|Mode)\s*[:=]?\s*(\d+)", text)
        if match:
            mode = int(match.group(1))
    return mode, name


def _cpu_governor() -> Optional[str]:
    value = _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return value or None


def _cpu_cur_freq_khz() -> Optional[int]:
    value = _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    return int(value) if value.isdigit() else None


def _cpu_max_freq_khz() -> Optional[int]:
    value = _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
    return int(value) if value.isdigit() else None


def _gpu_devfreq_dir() -> Optional[str]:
    try:
        names = sorted(glob.glob("/sys/class/devfreq/*/"))
    except OSError:
        return None
    for name in names:
        if "gpu" in name.lower():
            return name
    return None


def _gpu_governor() -> Optional[str]:
    directory = _gpu_devfreq_dir()
    if not directory:
        return None
    value = _read_text(directory + "governor")
    return value or None


def _gpu_cur_freq_hz() -> Optional[int]:
    directory = _gpu_devfreq_dir()
    if not directory:
        return None
    value = _read_text(directory + "cur_freq")
    return int(value) if value.isdigit() else None


def _jetson_clocks_active() -> Optional[bool]:
    governor = _cpu_governor()
    current = _cpu_cur_freq_khz()
    maximum = _cpu_max_freq_khz()
    if governor is None or current is None or maximum is None:
        return None
    if governor == "userspace" or (governor == "performance" and current == maximum):
        return True
    if governor in ("schedutil", "ondemand", "powersave", "conservative"):
        return False
    return None


def collect_power() -> PowerSection:
    mode, name = _nvpmodel_mode_and_name()
    return PowerSection(
        nvpmodel_mode=mode,
        nvpmodel_name=name,
        jetson_clocks_active=_jetson_clocks_active(),
        cpu_governor=_cpu_governor(),
        gpu_governor=_gpu_governor(),
    )


def _max_temperature_c() -> Optional[float]:
    best: Optional[float] = None
    for path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
            if not raw:
                continue
            value = int(raw.decode("ascii").strip()) / 1000.0
            if best is None or value > best:
                best = value
        except (OSError, ValueError, TypeError, UnicodeDecodeError):
            continue
    return best


def collect_volatile() -> VolatileObservations:
    return VolatileObservations(
        observed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        cpu_cur_freq_khz=_cpu_cur_freq_khz(),
        gpu_cur_freq_hz=_gpu_cur_freq_hz(),
        max_temperature_c=_max_temperature_c(),
    )


def _board_compatible() -> str:
    try:
        with open("/proc/device-tree/compatible", "rb") as fh:
            return fh.read().decode("utf-8", errors="replace").replace("\x00", " ").strip()
    except OSError:
        return ""


def _l4t_release() -> str:
    return _read_text("/etc/nv_tegra_release")


def _nvidia_driver_version() -> str:
    text = _read_text("/proc/driver/nvidia/version")
    match = re.search(r"(\d+\.\d+(?:\.\d+)?)", text)
    return match.group(1) if match else ""


def _nvcc_version() -> str:
    out = _run(["nvcc", "--version"])
    match = re.search(r"release\s+([0-9.]+)", out)
    return match.group(1) if match else ""


def collect_device_basic(
    device_probe: Optional[Mapping[str, Any]] = None,
) -> DeviceSection:
    """Collect the torch-independent device facts.

    CUDA-runtime/device-list facts are supplied through ``device_probe``
    (see :mod:`hqsb.hardware.probe`) so this module stays torch-free.
    """
    cuda_available = False
    device_count = 0
    device_names: List[str] = []
    compute_capabilities: List[List[int]] = []
    cuda_runtime_version = ""
    if device_probe:
        cuda_available = bool(device_probe.get("cuda_available", False))
        device_count = int(device_probe.get("device_count", 0) or 0)
        device_names = list(device_probe.get("device_names") or [])
        for cap in device_probe.get("compute_capabilities") or []:
            compute_capabilities.append([int(x) for x in cap])
        cuda_runtime_version = str(device_probe.get("cuda_runtime_version") or "")

    return DeviceSection(
        cuda_available=cuda_available,
        device_count=device_count,
        device_names=device_names,
        compute_capabilities=compute_capabilities,
        cuda_runtime_version=cuda_runtime_version,
        cuda_driver_version=_nvidia_driver_version(),
        nvcc_version=_nvcc_version(),
        board_compatible=_board_compatible(),
        l4t_release=_l4t_release(),
    )


def _git(repo_path: str, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=repo_path, capture_output=True, text=True
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except OSError:
        return ""


def collect_commit(repo_path: str) -> CommitSection:
    return CommitSection(
        commit=_git(repo_path, "rev-parse", "HEAD"),
        commit_short=_git(repo_path, "rev-parse", "--short", "HEAD"),
        dirty=bool(_git(repo_path, "status", "--porcelain")),
    )


__all__ = [
    "CommitSection",
    "ConfigSection",
    "DEFAULT_PACKAGE_WATCHLIST",
    "DeviceSection",
    "ENVIRONMENT_SECTION_NAMES",
    "FingerprintSections",
    "ModelSection",
    "OsSection",
    "PowerSection",
    "PythonSection",
    "RunFingerprint",
    "SECTION_NAMES",
    "VolatileObservations",
    "canonical_json",
    "collect_commit",
    "collect_device_basic",
    "collect_os",
    "collect_packages",
    "collect_power",
    "collect_python",
    "collect_volatile",
    "compute_run_fingerprint",
    "diff_sections",
    "section_digests",
    "section_payload",
    "sha256_hex",
]
