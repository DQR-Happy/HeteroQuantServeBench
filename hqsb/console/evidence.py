"""Read-only, bounded index of existing stage evidence; no verdict rewriting."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path


MAX_EVIDENCE_BYTES = 8_000_000
EVIDENCE_TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".md"}


class EvidenceCatalog:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.lock = threading.RLock()
        self.scanned = 0.0
        self.items = []
        self.files = {}
        self.file_roots = {}
        self.sources = []

    def _register(self, path, *, boundary=None, max_bytes=MAX_EVIDENCE_BYTES):
        try:
            resolved = path.resolve()
            boundary = boundary.resolve() if boundary is not None else self.root
            if (
                not resolved.is_relative_to(self.root)
                or not resolved.is_relative_to(boundary)
                or not resolved.is_file()
            ):
                return None
            size = resolved.stat().st_size
        except (OSError, RuntimeError):
            return None
        if size > max_bytes:
            return None
        relative = str(resolved.relative_to(self.root))
        key = hashlib.sha256(relative.encode()).hexdigest()[:24]
        self.files[key] = resolved
        self.file_roots[key] = boundary
        return {
            "id": key,
            "name": resolved.name,
            "relative_path": relative,
            "bytes": size,
        }

    def scan(self, refresh=False):
        with self.lock:
            if not refresh and time.monotonic() - self.scanned < 30:
                return self.items
            self.items, self.files, self.file_roots, self.sources = [], {}, {}, []
            base = self.root / "docs/stage_experiments"
            for path in sorted(base.glob("*/*/raw/verdict.json")):
                experiment_root = path.parent.parent
                record = self._register(
                    path, boundary=experiment_root, max_bytes=4_000_000
                )
                if not record:
                    continue
                try:
                    with path.open("rb") as stream:
                        raw_verdict = stream.read(4_000_001)
                    if len(raw_verdict) > 4_000_000:
                        raise ValueError("Verdict exceeds the parsing budget")
                    verdict = json.loads(raw_verdict)
                    if not isinstance(verdict, dict):
                        raise ValueError("Verdict must be a JSON object")
                except (OSError, ValueError):
                    self.files.pop(record["id"], None)
                    self.file_roots.pop(record["id"], None)
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
                # S05 quality, calibration, kernel and compatibility evidence
                # lives below raw/. Index text attachments without parsing them;
                # model tensors and native profiler binaries remain excluded.
                seen = {record["id"]}
                evidence_paths = [
                    candidate
                    for directory in (path.parent, experiment_root / "post_fix", experiment_root / "model_diagnostics")
                    for candidate in directory.rglob("*")
                ]
                for candidate in sorted(evidence_paths):
                    if (
                        candidate != path
                        and candidate.suffix.lower() in EVIDENCE_TEXT_SUFFIXES
                    ):
                        item = self._register(candidate, boundary=experiment_root)
                        if item and item["id"] not in seen:
                            refs.append(item)
                            seen.add(item["id"])
                for candidate in sorted(experiment_root.glob("*.md")):
                    item = self._register(candidate, boundary=experiment_root)
                    if item and item["id"] not in seen:
                        refs.append(item)
                        seen.add(item["id"])
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
                or not path.resolve().is_relative_to(self.file_roots[key])
                or not path.is_file()
            ):
                raise KeyError(key)
            if path.stat().st_size > MAX_EVIDENCE_BYTES:
                raise ValueError("Evidence exceeds the 8 MB browser limit")
            with path.open("rb") as stream:
                data = stream.read(MAX_EVIDENCE_BYTES + 1)
            if len(data) > MAX_EVIDENCE_BYTES:
                raise ValueError("Evidence exceeds the 8 MB browser limit")
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
