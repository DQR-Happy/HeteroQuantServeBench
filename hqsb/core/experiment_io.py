"""Shared storage for stage runs, independent of stage verdict policy.

Only file/command persistence is shared. Prerequisites and PASS/FAIL decisions
remain in their stage modules so a refactor cannot weaken scientific gates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from hqsb.core.errors import ConfigError


def run_directory_path(root: str, stage: str, experiment_id: str, run_id: str) -> str:
    """Resolve one run; identifiers cannot select a parent or absolute path."""
    for label, value in (
        ("stage", stage),
        ("experiment_id", experiment_id),
        ("run_id", run_id),
    ):
        if (
            not value
            or value in (".", "..")
            or any(c in value for c in ("/", "\\", "\0"))
        ):
            raise ConfigError(f"{label} must be one non-empty path component")
    base = (Path(root) / "experiment_results").resolve()
    target = (base / stage / experiment_id / run_id).resolve()
    if not target.is_relative_to(base):
        raise ConfigError("run directory escapes experiment_results")
    return str(target)


class RunStorage:
    """File persistence mixin for a stage-owned ``RunDirectory.path``."""

    path: str

    def _output_path(self, relative: str) -> Path:
        base = Path(self.path).resolve()
        requested = Path(relative)
        target = (base / requested).resolve()
        if requested.is_absolute() or not target.is_relative_to(base) or target == base:
            raise ConfigError("output must be a file inside the run directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_json(self, relative: str, payload: Any) -> str:
        target = self._output_path(relative)
        with target.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False)
        return str(target)

    def write_text(self, relative: str, text: str) -> str:
        target = self._output_path(relative)
        target.write_text(text, encoding="utf-8")
        return str(target)

    def write_jsonl(self, relative: str, rows: Sequence[Mapping[str, Any]]) -> str:
        target = self._output_path(relative)
        with target.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
                )
        return str(target)

    def record_command(
        self,
        index: int,
        command: Sequence[str],
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
    ) -> dict[str, Any]:
        """Save original argv in JSON; filename characters never become paths."""
        label = "_".join(part for part in command[:3] if part)
        name = f"{index:02d}_{re.sub(r'[^A-Za-z0-9_.-]', '_', label)}"[:60]
        self.write_json(
            f"commands/{name}.json",
            {"command": list(command), "returncode": returncode, "cwd": os.getcwd()},
        )
        if stdout:
            self.write_text(f"stdout/{name}.stdout", stdout)
        if stderr:
            self.write_text(f"stderr/{name}.stderr", stderr)
        return {
            "command": list(command),
            "returncode": returncode,
            "stdout": f"stdout/{name}.stdout" if stdout else "",
            "stderr": f"stderr/{name}.stderr" if stderr else "",
        }
