"""Actual deployed source identity, not just a possibly stale Git HEAD."""

import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform

from hqsb.observation.contracts import digest
from hqsb.observation.procfs import process_key


def environment(root):
    root = Path(root).resolve()
    files = {}
    for prefix in ("hqsb", "ops", "scripts", "configs"):
        for path in sorted((root / prefix).rglob("*")):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.suffix in {".py", ".cu", ".h", ".cpp", ".yaml", ".json"}
            ):
                files[str(path.relative_to(root))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    versions = {}
    for name in ("torch", "transformers", "triton", "fastapi"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "process": process_key(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "packages": versions,
        "source_digest": digest(files),
        "source_files": files,
        "allocator_environment": {
            k: os.environ.get(k)
            for k in ("PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_CUDA_ALLOC_CONF")
        },
    }
