"""Bounded Nsight jobs reference an immutable, previously measured workload."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DiagnosticPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    source_run_id: str = Field(pattern=r"^run_[0-9a-f]{32}$")
    tool: Literal["nsys", "ncu", "eretrace"] = "nsys"
    kernel_filter: str = Field(".*", min_length=1, max_length=256)
    launch_skip: int = Field(0, ge=0, le=100000)
    launch_count: int = Field(3, ge=1, le=32)
    sections: Literal["memory", "compute", "balanced"] = "balanced"
    timeout_s: int = Field(900, ge=30, le=1800)
    artifact_limit_mb: int = Field(256, ge=32, le=1024)
    ere_kernel: str = Field("_fwd_kernel", min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_.$]+$")
    ere_codec: Literal["reference", "raw"] = "reference"
    ere_verify: bool = False

    @model_validator(mode="after")
    def expression(self):
        import re
        try:
            re.compile(self.kernel_filter)
        except re.error as exc:
            raise ValueError(f"KERNEL_FILTER_INVALID:{exc}") from exc
        return self


class DiagnosticComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    baseline_run_id: str = Field(pattern=r"^run_[0-9a-f]{32}$")
    candidate_run_id: str = Field(pattern=r"^run_[0-9a-f]{32}$")


def profiler_arguments(plan, executable, prefix, command):
    """All commands/paths are server-owned; the browser supplies a typed plan."""
    if plan.tool == "eretrace":
        return command
    if plan.tool == "nsys":
        return [executable, "profile", "--trace=cuda,nvtx,osrt", "--sample=none",
                "--cpuctxsw=none", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                "--output", str(prefix), *command]
    sections = {"memory": ["MemoryWorkloadAnalysis"], "compute": ["SpeedOfLight", "Occupancy"],
                "balanced": ["SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy"]}[plan.sections]
    return [executable, "--target-processes", "all", "--profile-from-start", "off",
            "--replay-mode", "application", "--cache-control", "none", "--clock-control", "none",
            "--launch-skip", str(plan.launch_skip), "--launch-count", str(plan.launch_count),
            "--kernel-name-base", "demangled", "--kernel-name", "regex:" + plan.kernel_filter,
            *[value for section in sections for value in ("--section", section)],
            "--export", str(prefix), *command]
