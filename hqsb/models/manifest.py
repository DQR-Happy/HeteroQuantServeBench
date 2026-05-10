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

Integrity model (E00-02)
------------------------
Verification rejects six classes of artifact fault *before* any weight is
loaded:

1. ``missing_file``     — a manifest entry has no file on disk;
2. ``hash_mismatch``    — a file's digest differs from the manifest;
3. ``path_empty``       — the path is empty (or an empty component).
                          An empty path silently collapses to the model
                          root when joined, so it must never be accepted;
4. ``path_duplicate``   — the same normalized path is declared twice, which
                          makes "the" digest of that file ambiguous;
5. ``path_traversal``   — ``..``/absolute/backslash escapes would let a
                          manifest reference files *outside* the model root;
6. ``extra_file``       — a file exists under the model root that the
                          manifest does not declare.

Path faults (3-5) are *manifest* faults and raise :class:`ManifestError`,
a :class:`ValueError` subclass; content faults (1, 2, 6) are recorded in
:class:`VerificationResult`. :func:`verify_or_raise` collapses both into a
single :class:`~hqsb.core.errors.ArtifactError` gate (exit code 8) so a
corrupt artifact can never reach a loader.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Collection, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from hqsb.core.artifact_path import (
    REASON_PATH_ABSOLUTE,
    REASON_PATH_BACKSLASH,
    REASON_PATH_DUPLICATE,
    REASON_PATH_DUPLICATE_CASEFOLD,
    REASON_PATH_EMPTY,
    REASON_PATH_NOT_NORMALIZED,
    REASON_PATH_TRAVERSAL,
    classify_relative_path,
    validate_relative_paths,
)

# 64 hexadecimal characters.
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# Streaming read chunk size for hashing (1 MiB).
_DEFAULT_CHUNK_BYTES = 1 << 20

# ── Reason codes (stable, machine-readable) ─────────────────────────────
# The path_* codes above are owned by ``hqsb.core.artifact_path`` (single
# source of truth, shared with the ModelArtifact contract) and re-exported
# here so manifest callers only need one import.
REASON_LINE_MALFORMED = "line_malformed"
REASON_DIGEST_INVALID = "digest_invalid"
REASON_MISSING_FILE = "missing_file"
REASON_HASH_MISMATCH = "hash_mismatch"
REASON_EXTRA_FILE = "extra_file"

#: Reason codes that make a manifest structurally unusable.
MANIFEST_REASON_CODES = frozenset(
    {
        REASON_LINE_MALFORMED,
        REASON_DIGEST_INVALID,
        REASON_PATH_EMPTY,
        REASON_PATH_ABSOLUTE,
        REASON_PATH_TRAVERSAL,
        REASON_PATH_NOT_NORMALIZED,
        REASON_PATH_BACKSLASH,
        REASON_PATH_DUPLICATE,
        REASON_PATH_DUPLICATE_CASEFOLD,
    }
)


@dataclass(frozen=True)
class ManifestIssue:
    """A structured reason a manifest (or one of its entries) is invalid."""

    line_number: int
    path: str
    reason: str
    message: str

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable view of the issue."""
        return {
            "line_number": self.line_number,
            "path": self.path,
            "reason": self.reason,
            "message": self.message,
        }


class ManifestError(ValueError):
    """A manifest is malformed or declares an unsafe path.

    Subclasses :class:`ValueError` so existing callers that catch
    ``ValueError`` keep working, while new callers can read
    :attr:`issues` for machine-consumed diagnostics.
    """

    def __init__(self, message: str, *, issues: Sequence[ManifestIssue] = ()) -> None:
        super().__init__(message)
        self.message = message
        self.issues: Tuple[ManifestIssue, ...] = tuple(issues)

    @property
    def reasons(self) -> Tuple[str, ...]:
        """Distinct reason codes, in the order they were encountered."""
        seen: Dict[str, None] = {}
        for issue in self.issues:
            seen.setdefault(issue.reason, None)
        return tuple(seen)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


@dataclass(frozen=True)
class ManifestEntry:
    """A single ``<sha256>  <path>`` manifest line."""

    sha256: str
    path: str
    line_number: int = 0

    @property
    def normalized_path(self) -> str:
        """Return the entry path with any leading ``./`` stripped."""
        path = self.path
        while path.startswith("./"):
            path = path[2:]
        return path


def classify_path(path: str) -> Optional[Tuple[str, str]]:
    """Classify a manifest-relative path as safe or unsafe.

    Thin alias over :func:`hqsb.core.artifact_path.classify_relative_path`
    so the rule exists in exactly one place.

    Args:
        path: A path already stripped of its leading ``./``.

    Returns:
        ``None`` when the path is safe, otherwise a ``(reason, message)``
        pair describing why it must be rejected.
    """
    return classify_relative_path(path)


def validate_manifest_paths(entries: Iterable[ManifestEntry]) -> List[ManifestIssue]:
    """Return every structural problem in a parsed manifest.

    Checks each entry's normalized path for emptiness, absoluteness,
    traversal, non-normalized components, and duplicates (exact and
    case-insensitive, the latter because a manifest produced on a
    case-sensitive filesystem can silently collide on macOS/Windows).

    Args:
        entries: Parsed manifest entries, in file order.

    Returns:
        Issues in manifest order; empty when every path is safe.
    """
    entry_list = list(entries)
    problems = validate_relative_paths(
        entry.normalized_path for entry in entry_list
    )
    line_of = {
        entry.normalized_path: entry.line_number for entry in entry_list
    }
    return [
        ManifestIssue(
            line_number=line_of.get(path, 0),
            path=path,
            reason=reason,
            message=f"manifest line {line_of.get(path, 0)}: {message}",
        )
        for path, (reason, message) in problems.items()
    ]


def parse_manifest(text: str, *, validate_paths: bool = True) -> List[ManifestEntry]:
    """Parse manifest text into an ordered list of entries.

    Blank lines and lines whose first non-whitespace character is ``#``
    are ignored. Every other line must contain at least a SHA256 digest
    and a relative path separated by whitespace.

    Args:
        text: Raw manifest file contents.
        validate_paths: When true (default), reject empty, absolute,
            traversing, non-normalized, and duplicate paths.

    Returns:
        Ordered list of :class:`ManifestEntry`.

    Raises:
        ManifestError: If a non-blank, non-comment line is malformed, its
            digest is not a valid 64-character hexadecimal SHA256, or its
            path is unsafe. :class:`ManifestError` derives from
            :class:`ValueError`.
    """
    entries: List[ManifestEntry] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            raise ManifestError(
                f"manifest line {line_number}: expected '<sha256> <path>', "
                f"got {raw_line!r}",
                issues=[
                    ManifestIssue(
                        line_number=line_number,
                        path="",
                        reason=REASON_LINE_MALFORMED,
                        message=(
                            f"manifest line {line_number}: expected "
                            f"'<sha256> <path>', got {raw_line!r}"
                        ),
                    )
                ],
            )

        digest, path = parts[0], parts[1]
        if not _SHA256_RE.match(digest):
            raise ManifestError(
                f"manifest line {line_number}: invalid SHA256 digest {digest!r}",
                issues=[
                    ManifestIssue(
                        line_number=line_number,
                        path=path,
                        reason=REASON_DIGEST_INVALID,
                        message=(
                            f"manifest line {line_number}: invalid SHA256 "
                            f"digest {digest!r}"
                        ),
                    )
                ],
            )
        entries.append(
            ManifestEntry(sha256=digest.lower(), path=path, line_number=line_number)
        )

    if validate_paths:
        issues = validate_manifest_paths(entries)
        if issues:
            joined = "; ".join(issue.message for issue in issues)
            raise ManifestError(f"invalid manifest: {joined}", issues=issues)

    return entries


def _require_non_empty(value: str, what: str) -> str:
    """Reject empty/whitespace path arguments before they resolve to ``cwd``."""
    if value is None or str(value).strip() == "":
        raise ManifestError(
            f"{what} is empty: an empty path resolves to the current "
            f"working directory and would verify the wrong tree",
            issues=[
                ManifestIssue(
                    line_number=0,
                    path="",
                    reason=REASON_PATH_EMPTY,
                    message=f"{what} is empty",
                )
            ],
        )
    return str(value)


def load_manifest(path: str) -> List[ManifestEntry]:
    """Read and parse a manifest file from disk.

    Args:
        path: Path to the manifest file. ``~`` and environment variables
            are expanded.

    Returns:
        Ordered list of :class:`ManifestEntry`.
    """
    resolved = os.path.abspath(
        os.path.expanduser(os.path.expandvars(_require_non_empty(path, "manifest path")))
    )
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


def _is_allowed(relative: str, allow_extra: Collection[str]) -> bool:
    """Whether ``relative`` is explicitly allow-listed as an extra file."""
    for pattern in allow_extra:
        if pattern == relative:
            return True
        if fnmatch.fnmatch(relative, pattern):
            return True
    return False


def _collect_present_files(model_root: str) -> List[str]:
    """List every regular file under ``model_root`` as a POSIX relative path."""
    present: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(model_root):
        for filename in filenames:
            absolute = os.path.join(dirpath, filename)
            if not os.path.isfile(absolute):
                continue
            relative = os.path.relpath(absolute, model_root).replace(os.sep, "/")
            present.append(relative)
    return present


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
        extra_files: Files present under ``model_path`` but not declared by
            the manifest (after the ``allow_extra`` filter).
        allowed_extra_files: Files that matched ``allow_extra`` and were
            therefore not treated as faults.
        strict_extra: Whether extra files make the result not-``ok``.
    """

    model_path: str
    manifest_path: str
    entries: List[ManifestEntry] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    mismatched_files: List[Tuple[str, str, str]] = field(default_factory=list)
    extra_files: List[str] = field(default_factory=list)
    allowed_extra_files: List[str] = field(default_factory=list)
    strict_extra: bool = True

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
    def declared_paths(self) -> Set[str]:
        """Normalized relative paths declared by the manifest."""
        return {entry.normalized_path for entry in self.entries}

    @property
    def manifest_sha256(self) -> str:
        """SHA256 of the manifest file itself (identity of the file list)."""
        return compute_sha256(self.manifest_path)

    @property
    def reason_codes(self) -> List[str]:
        """Stable, ordered reason codes describing why verification failed."""
        codes: List[str] = []
        if self.missing_files:
            codes.append(REASON_MISSING_FILE)
        if self.mismatched_files:
            codes.append(REASON_HASH_MISMATCH)
        if self.extra_files and self.strict_extra:
            codes.append(REASON_EXTRA_FILE)
        return codes

    @property
    def first_bad_file(self) -> Optional[str]:
        """First offending relative path, or ``None`` when the artifact is ok.

        Ordering is deterministic: manifest order for missing/mismatched
        files (so the reported file is the one an operator sees first in
        the manifest), then lexicographic order for extra files.
        """
        missing = set(self.missing_files)
        mismatched = {path for path, _expected, _actual in self.mismatched_files}
        for entry in self.entries:
            normalized = entry.normalized_path
            if normalized in missing or normalized in mismatched:
                return normalized
        if self.extra_files:
            return sorted(self.extra_files)[0]
        return None

    @property
    def ok(self) -> bool:
        """Whether every manifest entry was present and matched.

        Extra files only fail the gate when ``strict_extra`` is set, which
        is the default: an undeclared file is exactly the "silent extra
        weight" case this gate exists to catch.
        """
        if self.missing_files or self.mismatched_files:
            return False
        return not (self.strict_extra and self.extra_files)

    def describe(self) -> str:
        """Render a human-readable, single-line summary of the outcome."""
        return (
            f"{self.verified_files}/{self.total_files} verified, "
            f"{len(self.missing_files)} missing, "
            f"{len(self.mismatched_files)} mismatched, "
            f"{len(self.extra_files)} extra"
        )

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable view of the full outcome."""
        return {
            "model_path": self.model_path,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "total_files": self.total_files,
            "verified_files": self.verified_files,
            "ok": self.ok,
            "strict_extra": self.strict_extra,
            "first_bad_file": self.first_bad_file,
            "reason_codes": self.reason_codes,
            "missing_files": list(self.missing_files),
            "mismatched_files": [
                {"path": path, "expected": expected, "actual": actual}
                for path, expected, actual in self.mismatched_files
            ],
            "extra_files": list(self.extra_files),
            "allowed_extra_files": list(self.allowed_extra_files),
            "describe": self.describe(),
        }


def verify_model_files(
    model_path: str,
    manifest_path: str,
    *,
    strict_extra: bool = True,
    allow_extra: Collection[str] = (),
    collect_extra: bool = True,
) -> VerificationResult:
    """Verify every file listed in ``manifest_path`` under ``model_path``.

    The verification never raises on hash mismatch, missing files, or extra
    files; it records them in the returned :class:`VerificationResult`.
    A missing model directory or manifest file raises
    :class:`FileNotFoundError`, and a malformed or unsafe manifest raises
    :class:`ManifestError` (a :class:`ValueError`).

    Args:
        model_path: Root directory containing the model files.
        manifest_path: Path to the SHA256 manifest.
        strict_extra: When true (default), an undeclared file under
            ``model_path`` fails the gate.
        allow_extra: Relative paths or ``fnmatch`` globs that are permitted
            to exist without being declared (e.g. a copy of the manifest
            shipped inside the snapshot for convenience).
        collect_extra: When false, skip the directory walk entirely (the
            cheap path, used when extra files are irrelevant).

    Returns:
        A :class:`VerificationResult` whose ``ok`` attribute indicates
        whether the snapshot is intact.
    """
    model_path = os.path.abspath(
        os.path.expanduser(
            os.path.expandvars(_require_non_empty(model_path, "model path"))
        )
    )
    manifest_path = os.path.abspath(
        os.path.expanduser(
            os.path.expandvars(_require_non_empty(manifest_path, "manifest path"))
        )
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
        strict_extra=strict_extra,
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

    if collect_extra:
        declared = result.declared_paths
        for relative in sorted(_collect_present_files(model_path)):
            if relative in declared:
                continue
            if _is_allowed(relative, allow_extra):
                result.allowed_extra_files.append(relative)
            else:
                result.extra_files.append(relative)

    return result


def verify_or_raise(
    model_path: str,
    manifest_path: str,
    **kwargs: object,
) -> VerificationResult:
    """Run :func:`verify_model_files` and raise on any integrity failure.

    This is the single gate a loader should call: every E00-02 fault class
    (missing, tampered, empty/duplicate/traversal path, extra file) leaves
    this function via one :class:`~hqsb.core.errors.ArtifactError`, so a
    bad artifact can never reach a backend.

    Args:
        model_path: Root directory containing the model files.
        manifest_path: Path to the SHA256 manifest.
        **kwargs: Forwarded to :func:`verify_model_files`.

    Returns:
        The successful :class:`VerificationResult`.

    Raises:
        ArtifactError: On any missing/mismatched/extra file or malformed
            manifest. ``details`` carries ``reason_codes``,
            ``first_bad_file``, and the per-class file lists.
        FileNotFoundError: If the model directory or manifest is absent.
    """
    # Imported lazily: hqsb.core must stay importable without hqsb.models,
    # and this module must stay importable without pulling in pydantic at
    # module-import time on minimal installs.
    from hqsb.core.errors import ArtifactError

    try:
        result = verify_model_files(model_path, manifest_path, **kwargs)  # type: ignore[arg-type]
    except ManifestError as exc:
        raise ArtifactError(
            f"model artifact manifest rejected: {exc}",
            details={
                "model_path": os.path.abspath(os.path.expanduser(str(model_path))),
                "manifest_path": os.path.abspath(
                    os.path.expanduser(str(manifest_path))
                ),
                "reason_codes": list(exc.reasons),
                "issues": [issue.as_dict() for issue in exc.issues],
                "first_bad_file": exc.issues[0].path if exc.issues else None,
            },
        ) from exc

    if not result.ok:
        raise ArtifactError(
            f"model artifact verification failed ({result.describe()}): "
            f"first bad file {result.first_bad_file!r}",
            details=result.as_dict(),
        )
    return result


__all__ = [
    "MANIFEST_REASON_CODES",
    "ManifestEntry",
    "ManifestError",
    "ManifestIssue",
    "VerificationResult",
    "classify_path",
    "compute_sha256",
    "load_manifest",
    "parse_manifest",
    "validate_manifest_paths",
    "verify_model_files",
    "verify_or_raise",
]
