"""Server-owned configuration. Browser clients never select paths or URLs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DeploymentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    name: str
    provider: Literal["pytorch", "openai"] = "pytorch"
    platform: str = "CUDA / Jetson"
    model: str = "Qwen/Qwen3-1.7B"
    model_path: str = "~/models/hqsb/Qwen3-1.7B"
    attention: Literal["eager", "sdpa"] = "eager"
    precision: str = "float16"
    kernel: str = "framework-native"
    context_limit: int = Field(2048, ge=32, le=131072)
    max_output_tokens: int = Field(512, ge=1, le=4096)
    manifest: str | None = None
    manifest_allow_extra: list[str] = Field(default_factory=list)
    base_url: str | None = None
    api_key_env: str | None = None

    @model_validator(mode="after")
    def valid_provider(self):
        if self.provider == "pytorch" and (
            self.precision != "float16" or self.kernel != "framework-native"
        ):
            raise ValueError(
                "interactive reference currently supports FP16 framework-native only"
            )
        if self.provider == "openai" and not self.base_url:
            raise ValueError("OpenAI provider requires a server-configured base_url")
        return self

    def public(self) -> dict:
        return self.model_dump(
            exclude={
                "model_path",
                "manifest",
                "manifest_allow_extra",
                "base_url",
                "api_key_env",
            }
        )


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_dir: Path
    evidence_root: Path
    web_dist: Path
    deployments: list[DeploymentConfig]
    max_pending: int = Field(8, ge=1, le=64)
    request_deadline_ms: int = Field(120000, ge=1000, le=600000)
    max_body_bytes: int = Field(262144, ge=4096, le=1048576)
    secure_cookie: bool = False
    # Explicit operator opt-in for Tegra CUPTI; the network API stays unprivileged.
    privileged_worker: bool = False
    profile_min_available_bytes: int = Field(768 * 1024 * 1024, ge=0)
    artifact_min_available_bytes: int = Field(384 * 1024 * 1024, ge=0)
    artifact_budget_bytes: int = Field(8 * 1024**3, ge=1024**2)
    quantization_timeout_s: int = Field(900, ge=30, le=3600)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [item.id for item in self.deployments]
        if len(ids) != len(set(ids)):
            raise ValueError("deployment IDs must be unique")
        if len(ids) > 4:
            raise ValueError(
                "Console supports at most four configured deployment workers"
            )
        return self


def load_settings(path: Path | None = None) -> Settings:
    root = Path(__file__).resolve().parents[2]
    cfg = path or root / "configs/console/default.yaml"
    raw = yaml.safe_load(cfg.read_text()) if cfg.exists() else {}
    raw.setdefault(
        "deployments", [{"id": "qwen-reference", "name": "Qwen3 · FP16 Reference"}]
    )
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
