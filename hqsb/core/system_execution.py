"""L1 bounded, process-scoped settings and paired experiment contract."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SystemProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cpu_ids: list[int] | None = Field(None, min_length=1, max_length=512)
    intra_threads: int = Field(2, ge=1, le=64)
    interop_threads: int = Field(1, ge=1, le=16)
    numa_node: int | None = Field(None, ge=0, le=63)
    power_limit_w: int | None = Field(None, ge=1, le=1000)

    @model_validator(mode="after")
    def valid(self):
        if self.cpu_ids is not None and (
            len(set(self.cpu_ids)) != len(self.cpu_ids)
            or any(i < 0 or i > 8191 for i in self.cpu_ids)
        ):
            raise ValueError("L1_CPU_IDS_MUST_BE_UNIQUE_NONNEGATIVE")
        return self


class SystemPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    baseline: SystemProfile = Field(default_factory=SystemProfile)
    candidate: SystemProfile = Field(default_factory=SystemProfile)
    device: int = Field(0, ge=0, le=63)
    precision: Literal["float16", "bfloat16", "float32"] = "float16"
    prompts: list[str] = Field(
        default_factory=lambda: [
            "Once upon a time, there was a little fox.",
            "Explain why the sky is blue in one sentence.",
        ],
        min_length=1,
        max_length=4,
    )
    max_output_tokens: int = Field(32, ge=2, le=128)
    rounds: int = Field(2, ge=1, le=4)
    timeout_s: int = Field(300, ge=30, le=1200)

    @model_validator(mode="after")
    def valid(self):
        if any(not p.strip() or len(p) > 4096 for p in self.prompts):
            raise ValueError("L1_BOUNDED_NONEMPTY_PROMPTS_REQUIRED")
        return self


def validate_system(plan, hardware):
    if plan.device >= len(hardware["devices"]):
        raise ValueError("L1_GPU_NOT_VISIBLE: choose another node on /machines")
    for profile in (plan.baseline, plan.candidate):
        if profile.cpu_ids and not set(profile.cpu_ids) <= set(
            hardware["allowed_cpus"]
        ):
            raise ValueError("L1_CPU_OUTSIDE_EFFECTIVE_AFFINITY")
        if profile.numa_node is not None:
            if not hardware["numa_policy"]["available"]:
                raise ValueError(
                    "L1_NUMA_UNAVAILABLE: " + hardware["numa_policy"]["reason"]
                )
            if profile.numa_node not in [n["id"] for n in hardware["numa_nodes"]]:
                raise ValueError("L1_NUMA_NODE_NOT_VISIBLE")
        if profile.power_limit_w is not None:
            gpu = hardware["devices"][plan.device]
            limits = gpu.get("power_constraints_w")
            if not limits:
                raise ValueError(
                    "L1_POWER_CONTROL_UNAVAILABLE: "
                    + gpu.get("power_reason", "NVML missing")
                )
            if not limits[0] <= profile.power_limit_w <= limits[1]:
                raise ValueError("L1_POWER_LIMIT_OUTSIDE_HARDWARE_RANGE")


def summarize(trials):
    """Descriptive paired deltas; quality remains distinct from performance."""
    from statistics import median

    a = [t for t in trials if t["label"] == "baseline"]
    b = [t for t in trials if t["label"] == "candidate"]
    reference = [r["token_ids"] for r in a[0]["requests"]] if a else None
    exact = bool(a and b) and all(
        [r["token_ids"] for r in t["requests"]] == reference for t in trials
    )
    metrics = {}
    for key in (
        "token_s",
        "first_token_ms",
        "peak_allocated_bytes",
        "rss_bytes",
        "energy_j",
    ):
        vals = [[t["metrics"].get(key) for t in rows] for rows in (a, b)]
        med = [median(v) if v and all(x is not None for x in v) else None for v in vals]
        metrics[key] = {
            "baseline": med[0],
            "candidate": med[1],
            "delta": med[1] - med[0] if None not in med else None,
            "delta_percent": (med[1] / med[0] - 1) * 100
            if None not in med and med[0]
            else None,
        }
    return {
        "greedy_tokens_exact": exact,
        "metrics": metrics,
        "repetitions_per_profile": [len(a), len(b)],
        "claim": "descriptive_medians_no_general_speedup_claim",
        "quality_scope": "fixed prompt greedy token agreement; not independent task accuracy",
    }
