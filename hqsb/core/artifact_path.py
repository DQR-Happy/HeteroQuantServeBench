"""Path-safety rules for artifact-relative file paths (C1 / E00-02).

Both :class:`~hqsb.core.contracts.model.ModelArtifact` (a ``file_hashes``
mapping) and :mod:`hqsb.models.manifest` (a ``sha256sum``-style manifest)
declare files by *relative path*. A relative path is only trustworthy if it
is confined to the artifact root, so this module is the single place where
that rule is defined and enforced — ``core`` must never import
``hqsb.models``, and ``hqsb.models`` must not re-invent the rule.

Rejected classes:

* ``path_empty``              — empty path or empty component; an empty path
  silently resolves to the artifact root when joined.
* ``path_absolute``           — ``/x`` or ``C:x`` escapes the root.
* ``path_traversal``          — a ``..`` component escapes the root.
* ``path_not_normalized``     — ``.`` component or trailing separator; two
  spellings of one file break digest de-duplication.
* ``path_backslash``          — ``\\`` is a separator on Windows and a legal
  character on POSIX, so ``..\\..\\secret`` would only be dangerous on one
  of the two platforms. Rejecting it keeps one manifest verifiable on both.
* ``path_duplicate``          — the same path declared twice.
* ``path_duplicate_casefold`` — collides case-insensitively (macOS/Windows).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Tuple

REASON_PATH_EMPTY = "path_empty"
REASON_PATH_ABSOLUTE = "path_absolute"
REASON_PATH_TRAVERSAL = "path_traversal"
REASON_PATH_NOT_NORMALIZED = "path_not_normalized"
REASON_PATH_BACKSLASH = "path_backslash"
REASON_PATH_DUPLICATE = "path_duplicate"
REASON_PATH_DUPLICATE_CASEFOLD = "path_duplicate_casefold"

#: Every path-safety reason code, for pre-registration and table-driven tests.
PATH_REASON_CODES = frozenset(
    {
        REASON_PATH_EMPTY,
        REASON_PATH_ABSOLUTE,
        REASON_PATH_TRAVERSAL,
        REASON_PATH_NOT_NORMALIZED,
        REASON_PATH_BACKSLASH,
        REASON_PATH_DUPLICATE,
        REASON_PATH_DUPLICATE_CASEFOLD,
    }
)

_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def classify_relative_path(path: str) -> Optional[Tuple[str, str]]:
    """Classify one artifact-relative path.

    Args:
        path: A POSIX-style path relative to the artifact root, already
            stripped of any leading ``./``.

    Returns:
        ``None`` when the path is safe to join with the artifact root,
        otherwise ``(reason_code, human_readable_message)``.
    """
    if path == "":
        return (
            REASON_PATH_EMPTY,
            "empty path: an empty path resolves to the artifact root",
        )

    if "\\" in path:
        return (
            REASON_PATH_BACKSLASH,
            f"backslash separator in path {path!r}: not portable and "
            f"ambiguous across platforms",
        )

    if path.startswith("/"):
        return (
            REASON_PATH_ABSOLUTE,
            f"absolute path {path!r}: artifact paths must be relative to the "
            f"artifact root",
        )
    if _DRIVE_RE.match(path):
        return (
            REASON_PATH_ABSOLUTE,
            f"drive-absolute path {path!r}: artifact paths must be relative "
            f"to the artifact root",
        )

    parts = path.split("/")
    for part in parts:
        if part == "":
            return (
                REASON_PATH_EMPTY,
                f"empty path component in {path!r}",
            )
        if part == "..":
            return (
                REASON_PATH_TRAVERSAL,
                f"path traversal in {path!r}: '..' escapes the artifact root",
            )
        if part == ".":
            return (
                REASON_PATH_NOT_NORMALIZED,
                f"non-normalized path {path!r}: contains a '.' component",
            )
    if len(parts) > 1 and parts[-1] == "":
        return (
            REASON_PATH_NOT_NORMALIZED,
            f"non-normalized path {path!r}: trailing separator",
        )
    return None


def validate_relative_paths(paths: Iterable[str]) -> Dict[str, Tuple[str, str]]:
    """Return ``{path: (reason, message)}`` for every unsafe path.

    Performs both per-path classification and cross-path duplicate
    detection. Ordering: the returned dict preserves first-encounter order
    of the offending paths.

    Args:
        paths: Relative paths in declaration order.

    Returns:
        Empty dict when every path is safe, unique, and normalized.
    """
    problems: Dict[str, Tuple[str, str]] = {}
    seen: Dict[str, str] = {}
    seen_casefold: Dict[str, str] = {}

    for path in paths:
        if path in problems:
            continue

        verdict = classify_relative_path(path)
        if verdict is not None:
            problems[path] = verdict
            continue

        if path in seen:
            problems[path] = (
                REASON_PATH_DUPLICATE,
                f"duplicate path {path!r} declared more than once",
            )
            continue
        seen[path] = path

        folded = path.casefold()
        if folded in seen_casefold:
            problems[path] = (
                REASON_PATH_DUPLICATE_CASEFOLD,
                f"path {path!r} collides case-insensitively with "
                f"{seen_casefold[folded]!r}",
            )
            continue
        seen_casefold[folded] = path

    return problems


__all__ = [
    "PATH_REASON_CODES",
    "REASON_PATH_ABSOLUTE",
    "REASON_PATH_BACKSLASH",
    "REASON_PATH_DUPLICATE",
    "REASON_PATH_DUPLICATE_CASEFOLD",
    "REASON_PATH_EMPTY",
    "REASON_PATH_NOT_NORMALIZED",
    "REASON_PATH_TRAVERSAL",
    "classify_relative_path",
    "validate_relative_paths",
]
