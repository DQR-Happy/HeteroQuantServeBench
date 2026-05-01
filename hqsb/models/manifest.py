"""Model artifact manifest parsing and SHA256 verification.

A *manifest* is a plain-text file in the ``sha256sum`` format where every
non-empty, non-comment line has the shape::

    <sha256_hex>  <relative_path>

The relative path may carry a leading ``./`` (as produced by
``sha256sum``/``find``) and is normalized to the platform path separator
before being joined with the model root. Manifest verification is the
correctness gate that guarantees a benchmark references an unchanged,
fully-materialized model snapshot.

This module is deliberately free of PyTorch/ModelScope imports so that
artifact integrity can be checked on a CPU-only machine and unit-tested
without a GPU or model weights.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple

# 64 hexadecimal characters.
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# Streaming read chunk size for hashing (1 MiB).
_DEFAULT_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class ManifestEntry:
    """A single ``<sha256>  <path>`` manifest line."""

    sha256: str
    path: str

    @property
    def normalized_path(self) -> str:
        """Return the entry path with any leading ``./`` stripped."""
        path = self.path
        while path.startswith("./"):
            path = path[2:]
        return path


def parse_manifest(text: str) -> List[ManifestEntry]:
    """Parse manifest text into an ordered list of entries.

    Blank lines and lines whose first non-whitespace character is ``#``
    are ignored. Every other line must contain at least a SHA256 digest
    and a relative path separated by whitespace.

    Args:
        text: Raw manifest file contents.

    Returns:
        Ordered list of :class:`ManifestEntry`.

    Raises:
        ValueError: If a non-blank, non-comment line is malformed or its
            digest is not a valid 64-character hexadecimal SHA256.
    """
    entries: List[ManifestEntry] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            raise ValueError(
                f"manifest line {line_number}: expected '<sha256> <path>', "
                f"got {raw_line!r}"
            )

        digest, path = parts[0], parts[1]
        if not _SHA256_RE.match(digest):
            raise ValueError(
                f"manifest line {line_number}: invalid SHA256 digest {digest!r}"
            )
        entries.append(ManifestEntry(sha256=digest.lower(), path=path))
    return entries


def load_manifest(path: str) -> List[ManifestEntry]:
    """Read and parse a manifest file from disk.

    Args:
        path: Path to the manifest file. ``~`` and environment variables
            are expanded.

    Returns:
        Ordered list of :class:`ManifestEntry`.
    """
    resolved = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    with open(resolved, encoding="utf-8") as fh:
        return parse_manifest(fh.read())


def compute_sha256(path: str, chunk_bytes: int = _DEFAULT_CHUNK_BYTES) -> str:
    """Compute the lowercase SHA256 hex digest of a file by streaming it.

    Args:
        path: Path to the file to hash.
        chunk_bytes: Read chunk size in bytes.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class VerificationResult:
    """Outcome of verifying a model directory against a manifest.

    Attributes:
        model_path: Absolute path to the verified model directory.
        manifest_path: Absolute path to the manifest used.
        entries: Parsed manifest entries.
        missing_files: Relative paths listed in the manifest but absent.
        mismatched_files: ``(relative_path, expected_sha256, actual_sha256)``
            triples for files whose digest does not match.
    """

    model_path: str
    manifest_path: str
    entries: List[ManifestEntry] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    mismatched_files: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        """Total number of files described by the manifest."""
        return len(self.entries)

    @property
    def verified_files(self) -> int:
        """Number of files that were present and digest-verified."""
        return (
            self.total_files
            - len(self.missing_files)
            - len(self.mismatched_files)
        )

    @property
    def ok(self) -> bool:
        """Whether every manifest entry was present and matched."""
        return not self.missing_files and not self.mismatched_files

    def describe(self) -> str:
        """Render a human-readable, single-line summary of the outcome."""
        return (
            f"{self.verified_files}/{self.total_files} verified, "
            f"{len(self.missing_files)} missing, "
            f"{len(self.mismatched_files)} mismatched"
        )


def verify_model_files(model_path: str, manifest_path: str) -> VerificationResult:
    """Verify every file listed in ``manifest_path`` under ``model_path``.

    The verification never raises on hash mismatch or missing files; it
    records them in the returned :class:`VerificationResult`. A missing
    model directory or manifest file raises :class:`FileNotFoundError`, and
    a malformed manifest raises :class:`ValueError`.

    Args:
        model_path: Root directory containing the model files.
        manifest_path: Path to the SHA256 manifest.

    Returns:
        A :class:`VerificationResult` whose ``ok`` attribute indicates
        whether the snapshot is intact.
    """
    model_path = os.path.abspath(
        os.path.expanduser(os.path.expandvars(model_path))
    )
    manifest_path = os.path.abspath(
        os.path.expanduser(os.path.expandvars(manifest_path))
    )

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    entries = load_manifest(manifest_path)
    result = VerificationResult(
        model_path=model_path,
        manifest_path=manifest_path,
        entries=entries,
    )

    for entry in entries:
        relative = entry.normalized_path
        file_path = os.path.join(model_path, relative)

        if not os.path.isfile(file_path):
            result.missing_files.append(relative)
            continue

        actual = compute_sha256(file_path)
        if actual != entry.sha256:
            result.mismatched_files.append((relative, entry.sha256, actual))

    return result
