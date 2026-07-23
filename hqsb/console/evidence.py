"""Read-only, bounded index of existing stage evidence; no verdict rewriting."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path


class EvidenceCatalog:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.lock = threading.RLock()
        self.scanned = 0.0
        self.items = []
        self.files = {}
        self.sources = []

    def _register(self, path):
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            return None
        relative = str(resolved.relative_to(self.root))
        key = hashlib.sha256(relative.encode()).hexdigest()[:24]
        self.files[key] = resolved
        return {
            "id": key,
            "name": resolved.name,
            "relative_path": relative,
            "bytes": resolved.stat().st_size,
        }

    def scan(self, refresh=False):
        with self.lock:
            if not refresh and time.monotonic() - self.scanned < 30:
                return self.items
            self.items, self.files, self.sources = [], {}, []
            base = self.root / "docs/stage_experiments"
            for path in sorted(base.glob("*/*/raw/verdict.json")):
                if path.stat().st_size > 4_000_000:
                    continue
                try:
                    verdict = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if not isinstance(verdict, dict):
                    continue
                record = self._register(path)
                if not record:
                    continue
                stage, experiment = path.parts[-4:-2]
                status = (
                    verdict.get("overall")
                    or verdict.get("verdict")
                    or verdict.get("status")
                    or "UNKNOWN"
                )
                if not isinstance(status, str):
                    status = "UNKNOWN"
                refs = [record]
                for candidate in sorted(path.parent.glob("*.json")):
                    if candidate != path and candidate.stat().st_size <= 8_000_000:
                        item = self._register(candidate)
                        if item:
                            refs.append(item)
                for candidate in path.parent.parent.glob("*.md"):
                    item = self._register(candidate)
                    if item:
                        refs.append(item)
                self.items.append(
                    {
                        "id": record["id"],
                        "stage": stage,
                        "experiment": experiment,
                        "status": status,
                        "scientific": verdict.get(
                            "scientific_execution_verdict", verdict.get("scientific")
                        ),
                        "updated_at": path.stat().st_mtime,
                        "source": "historical_evidence",
                        "files": refs,
                    }
                )
            for directory in ("ops", "configs/quantization", "configs/models"):
                for path in sorted((self.root / directory).rglob("*")):
                    if (
                        path.is_file()
                        and path.suffix in {".py", ".cu", ".cuh", ".cpp", ".h", ".yaml"}
                        and "__pycache__" not in path.parts
                        and path.stat().st_size <= 500_000
                    ):
                        item = self._register(path)
                        if item:
                            self.sources.append(
                                {
                                    **item,
                                    "category": "kernel"
                                    if directory == "ops"
                                    else "configuration",
                                    "verification": "source_only",
                                }
                            )
            self.scanned = time.monotonic()
            return self.items

    def file(self, key):
        with self.lock:
            self.scan()
            path = self.files.get(key)
            if (
                path is None
                or not path.resolve().is_relative_to(self.root)
                or not path.is_file()
            ):
                raise KeyError(key)
            if path.stat().st_size > 8_000_000:
                raise ValueError("Evidence exceeds the 8 MB browser limit")
            data = path.read_bytes()
            return path, data, hashlib.sha256(data).hexdigest()

    def detail(self, key):
        path, data, digest = self.file(key)
        return {
            "id": key,
            "name": path.name,
            "sha256": digest,
            "bytes": len(data),
            "content": json.loads(data)
            if path.suffix == ".json"
            else data.decode("utf-8"),
            "format": path.suffix.lstrip("."),
            "source": "historical_evidence",
        }
