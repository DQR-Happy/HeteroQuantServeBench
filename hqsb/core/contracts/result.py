"""C6 — BenchmarkResult contract.

The single, versioned envelope for every benchmark outcome. It binds raw
samples to their environment, git commit, and artifact/config hashes, and
keeps correctness and summary separate from raw data (top-level
architecture §5 C6 and §3.3).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from hqsb.core.contracts.base import VersionedModel


class EnvironmentInfo(VersionedModel):
    """Hardware/software environment captured at run time."""

    SCHEMA_VERSION = "1.0.0"

    platform: str = Field("", description="Platform string (platform.platform()).")
    device: str = Field("", description="Accelerator device name, e.g. 'Orin'.")
    compute_capability: Optional[List[int]] = Field(
        default=None, description="Device compute capability [major, minor]."
    )
    python_version: str = Field("", description="Python version.")
    torch_version: str = Field("", description="PyTorch version.")
    cuda_version: str = Field("", description="CUDA version.")
    framework_versions: Dict[str, str] = Field(
        default_factory=dict, description="Additional framework versions."
    )
    power_mode: Optional[str] = Field(default=None, description="nvpmodel mode.")


class CorrectnessReport(VersionedModel):
    """Correctness gate outcome for a benchmark pass."""

    SCHEMA_VERSION = "1.0.0"

    passed: bool = Field(..., description="Whether the correctness gate passed.")
    method: str = Field("", description="How correctness was assessed.")
    details: Dict[str, Any] = Field(default_factory=dict)


class ResourceUsage(VersionedModel):
    """Peak/aggregate resource metrics for a benchmark pass."""

    SCHEMA_VERSION = "1.0.0"

    peak_cuda_allocated_mb: float = Field(0.0, ge=0)
    peak_cuda_reserved_mb: float = Field(0.0, ge=0)
    process_rss_mb: float = Field(0.0, ge=0)
    avg_power_w: Optional[float] = Field(default=None, ge=0)
    peak_power_w: Optional[float] = Field(default=None, ge=0)
    energy_j: Optional[float] = Field(default=None, ge=0)


class BenchmarkResult(VersionedModel):
    """A versioned benchmark outcome with raw samples and summary."""

    SCHEMA_VERSION = "1.0.0"

    run_id: str = Field(..., description="Unique run identifier.")
    timestamp: float = Field(..., description="Unix timestamp of the run.")
    environment: EnvironmentInfo = Field(..., description="Run environment.")
    git_commit: Optional[str] = Field(default=None, description="Git SHA.")
    git_dirty: Optional[bool] = Field(default=None, description="Working tree state.")
    model_artifact_hash: Optional[str] = Field(
        default=None, description="Hash of the model artifact (weights/config)."
    )
    config_hash: Optional[str] = Field(
        default=None, description="Hash of the benchmark configuration."
    )
    workload: Optional["WorkloadSpec"] = Field(  # noqa: F821
        default=None, description="Workload that produced this result."
    )
    raw_samples: List[Dict[str, Any]] = Field(
        default_factory=list, description="Raw per-pass samples (not summarized)."
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Statistical summary derived from samples."
    )
    correctness: Optional[CorrectnessReport] = Field(default=None)
    resource: Optional[ResourceUsage] = Field(default=None)
    error: Optional[str] = Field(default=None, description="Error message if failed.")
    artifact_links: Dict[str, str] = Field(
        default_factory=dict, description="Mapping name -> artifact path/hash."
    )


# Late import to satisfy forward reference without a circular import at
# module load time (WorkloadSpec lives in a sibling module).
from hqsb.core.contracts.workload import WorkloadSpec  # noqa: E402,F401

BenchmarkResult.model_rebuild()


__all__ = [
    "BenchmarkResult",
    "CorrectnessReport",
    "EnvironmentInfo",
    "ResourceUsage",
]
