"""Server-owned model configuration and operator-managed node connections."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from hqsb.console.machine_registry import MachineConfig


class DeploymentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    name: str
    provider: Literal["pytorch", "openai", "vllm", "sglang"] = "pytorch"
    platform: str = "CUDA / Jetson"
    model: str = "Qwen/Qwen3-1.7B"
    model_path: str = "~/models/hqsb/Qwen3-1.7B"
    attention: Literal["eager", "sdpa"] = "eager"
    precision: str = "float16"
    kernel: str = "framework-native"
    kernel_profile: str = "framework-native"
    input_mode: Literal["chat", "completion"] = "chat"
    kernel_manifest: str | None = None
    context_limit: int = Field(2048, ge=32, le=131072)
    max_output_tokens: int = Field(512, ge=1, le=4096)
    manifest: str | None = None
    manifest_allow_extra: list[str] = Field(default_factory=list)
    device_lock_path: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    engine_python: str | None = None
    engine_options: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_provider(self):
        from hqsb.core.kernel_profiles import profile

        profile(self.kernel_profile)
        if self.provider in {"vllm", "sglang"}:
            from hqsb.core.native_engine import resolve_options
            self.engine_options = resolve_options(self.provider, self.engine_options)
            if not self.engine_python or not self.manifest or not self.device_lock_path:
                raise ValueError("Native engine requires engine_python, manifest and device_lock_path")
        if self.provider != "pytorch" and self.kernel_profile != "framework-native":
            raise ValueError("kernel profiles require a local PyTorch worker")
        if self.provider == "pytorch" and (
            self.precision not in {"float32", "float16", "bfloat16"} or self.kernel != "framework-native"
        ):
            raise ValueError(
                "interactive reference supports FP32/FP16/BF16 framework-native only"
            )
        if self.provider == "pytorch" and self.precision != "float16" and self.kernel_profile != "framework-native":
            raise ValueError("Custom kernel profiles currently require FP16")
        if self.provider == "openai" and not self.base_url:
            raise ValueError("OpenAI provider requires a server-configured base_url")
        if self.provider == "openai" and self.input_mode != "chat":
            raise ValueError("External provider currently supports chat input only")
        return self

    def public(self) -> dict:
        return self.model_dump(
            exclude={
                "model_path",
                "manifest",
                "manifest_allow_extra",
                "base_url",
                "api_key_env",
                "device_lock_path",
                "kernel_manifest",
                "engine_python",
            }
        )


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_dir: Path
    evidence_root: Path
    web_dist: Path
    deployments: list[DeploymentConfig]
    machine_name: str = "本机 HQSB"
    machines: list[MachineConfig] = Field(default_factory=list)
    max_pending: int = Field(8, ge=1, le=64)
    request_deadline_ms: int = Field(120000, ge=1000, le=600000)
    max_body_bytes: int = Field(262144, ge=4096, le=1048576)
    secure_cookie: bool = False
    # Test workspaces open directly; authentication is an explicit opt-in.
    auth_required: bool = False
    # Explicit operator opt-in for Tegra CUPTI; the network API stays unprivileged.
    privileged_worker: bool = False
    profile_min_available_bytes: int = Field(768 * 1024 * 1024, ge=0)
    artifact_min_available_bytes: int = Field(384 * 1024 * 1024, ge=0)
    artifact_budget_bytes: int = Field(8 * 1024**3, ge=1024**2)
    quantization_timeout_s: int = Field(900, ge=30, le=3600)
    model_download_dir: Path | None = None
    model_download_reserve_bytes: int = Field(4 * 1024**3, ge=1024**3)
    native_engine_pythons: dict[str, str] = Field(default_factory=lambda: {
        "vllm": ".venv-vllm/bin/python", "sglang": ".venv-sglang/bin/python"})
    native_engine_cache_dir: Path | None = None
    diagnostics_dir: Path | None = None
    diagnostic_tools: dict[Literal["nsys", "ncu"], str] = Field(default_factory=dict)
    eretrace_runtime: Path | None = None

    @model_validator(mode="after")
    def unique_ids(self):
        machine_ids = [item.id for item in self.machines]
        if len(machine_ids) != len(set(machine_ids)) or len(machine_ids) > 32:
            raise ValueError("Machine connection IDs must be unique; at most 32")
        ids = [item.id for item in self.deployments]
        if len(ids) != len(set(ids)):
            raise ValueError("deployment IDs must be unique")
        if len(ids) > 16:
            raise ValueError(
                "Console supports at most sixteen configured deployment workers"
            )
        return self


def load_settings(path: Path | None = None) -> Settings:
    root = Path(__file__).resolve().parents[2]
    cfg = path or root / "configs/console/default.yaml"
    raw = yaml.safe_load(cfg.read_text()) if cfg.exists() else {}
    raw.setdefault(
        "deployments", [{"id": "qwen-reference", "name": "Qwen3 · FP16 Reference"}]
    )
    for deployment in raw["deployments"]:
        if deployment.get("provider", "pytorch") == "pytorch":
            deployment.setdefault("device_lock_path", str(Path.home() / ".cache/hqsb/device0.lock"))
    raw["data_dir"] = (
        Path(os.environ.get("HQSB_CONSOLE_DATA", str(root / ".console")))
        .expanduser()
        .resolve()
    )
    raw["evidence_root"] = (
        Path(os.environ.get("HQSB_EVIDENCE_ROOT", str(root))).expanduser().resolve()
    )
    raw["web_dist"] = root / "web/console/dist"
    return Settings.model_validate(raw)
