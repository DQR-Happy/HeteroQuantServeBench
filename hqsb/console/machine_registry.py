"""Persistent node connection profiles; no model or transport resources live here."""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MachineError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.status = code, status


class MachineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    name: str = Field(min_length=1, max_length=120)
    transport: Literal["http", "ssh"]
    base_url: str | None = Field(None, max_length=2048)
    ssh_host: str | None = Field(None, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,252}$")
    ssh_user: str | None = Field(None, pattern=r"^[a-zA-Z_][a-zA-Z0-9_-]{0,63}$")
    ssh_port: int | None = Field(None, ge=1, le=65535)
    identity_file: str | None = Field(None, max_length=1024)
    remote_port: int = Field(8768, ge=1, le=65535)
    token_env: str | None = Field(None, pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

    @model_validator(mode="after")
    def check_transport(self):
        if self.id == "local":
            raise ValueError("local is reserved for this server")
        if self.transport == "http":
            if any(char.isspace() or ord(char) < 32 for char in (self.base_url or "")):
                raise ValueError(
                    "base_url cannot contain whitespace or control characters"
                )
            parts = urlsplit(self.base_url or "")
            if (
                parts.scheme not in {"http", "https"}
                or not parts.hostname
                or parts.username
                or parts.password
                or parts.query
                or parts.fragment
                or parts.path not in {"", "/"}
            ):
                raise ValueError(
                    "base_url must be an HTTP(S) origin without credentials or path"
                )
            # Force port parsing here, before a profile can be persisted.
            _ = parts.port
            if self.ssh_host or self.ssh_user or self.ssh_port or self.identity_file:
                raise ValueError("HTTP profiles cannot contain SSH options")
        elif not self.ssh_host or self.base_url:
            raise ValueError("SSH requires ssh_host and does not use base_url")
        if self.identity_file and (
            "\n" in self.identity_file or "\x00" in self.identity_file
        ):
            raise ValueError("Invalid identity file path")
        return self


class MachineRegistry:
    def __init__(self, root: Path, name: str, seeds: list[MachineConfig]):
        self.path = root / "machines.json"
        self.name = name
        self.lock = threading.RLock()
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            if raw.get("schema_version") != 1:
                raise ValueError("Unsupported machine registry schema")
            self.node_id = raw["node_id"]
            self.retired = set(raw.get("retired_ids", []))
            self.items = {
                row["id"]: MachineConfig.model_validate(row) for row in raw["items"]
            }
        else:
            self.node_id = uuid.uuid4().hex
            self.retired: set[str] = set()
            self.items = {row.id: row for row in seeds}
            self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.path.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(
                {
                    "schema_version": 1,
                    "node_id": self.node_id,
                    "retired_ids": sorted(self.retired),
                    "items": [row.model_dump() for row in self.items.values()],
                },
                stream,
                indent=2,
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)

    def get(self, identity: str) -> MachineConfig:
        with self.lock:
            if identity not in self.items:
                raise MachineError(
                    "MACHINE_NOT_FOUND", "Machine connection not found", 404
                )
            return self.items[identity]

    def add(self, config: MachineConfig):
        with self.lock:
            if config.id in self.items or config.id in self.retired:
                raise MachineError(
                    "MACHINE_EXISTS",
                    "Choose a new machine ID; connection IDs cannot be reused",
                    409,
                )
            if len(self.items) >= 32:
                raise MachineError(
                    "MACHINE_LIMIT", "At most 32 remote connections are supported", 409
                )
            self.items[config.id] = config
            try:
                self._save()
            except Exception:
                del self.items[config.id]
                raise

    def remove(self, identity: str):
        with self.lock:
            config = self.get(identity)
            del self.items[identity]
            # Old tabs and saved API URLs must fail rather than submit work to
            # a different server later registered under the same connection ID.
            self.retired.add(identity)
            try:
                self._save()
            except Exception:
                self.items[identity] = config
                self.retired.remove(identity)
                raise

    def public(self, selected: str):
        with self.lock:
            return {
                "selected_id": selected,
                "control_node_id": self.node_id,
                "items": [
                    {"id": "local", "name": self.name, "transport": "local"},
                    *[row.model_dump(exclude_none=True) for row in self.items.values()],
                ],
            }
