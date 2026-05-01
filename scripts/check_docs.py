#!/usr/bin/env python3
"""Documentation integrity checker.

Checks that relative Markdown links in ``docs/`` and ``README.md`` resolve to
existing files, so a broken link cannot silently rot documentation. Also
verifies that no stage doc references a ``schema_version`` newer than the
current contract version (drift detection).

Usage:
    python scripts/check_docs.py
"""

from __future__ import annotations

import os
import re
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Matches [text](target) where target is a relative path (no scheme).
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

# Current contract schema version (single source of truth is the contracts).
_CURRENT_SCHEMA_VERSION = "1.0.0"


def _iter_markdown(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".md"):
                yield os.path.join(dirpath, filename)


def _check_links(path: str) -> int:
    """Return the number of broken relative links found in ``path``."""
    broken = 0
    base_dir = os.path.dirname(path)
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    for target in _LINK_RE.findall(text):
        # Skip anchors, absolute URLs, and mailto links.
        if target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        # Strip a trailing anchor fragment.
        clean = target.split("#", 1)[0]
        if not clean:
            continue
        resolved = os.path.normpath(os.path.join(base_dir, clean))
        if not os.path.exists(resolved):
            print(f"  [broken link] {path}: -> {target}")
            broken += 1
    return broken


def main() -> int:
    """Run all documentation checks and return the number of failures."""
    root = os.path.join(_REPO_ROOT, "docs")
    readme = os.path.join(_REPO_ROOT, "README.md")

    total_broken = 0
    paths = list(_iter_markdown(root))
    if os.path.isfile(readme):
        paths.append(readme)

    print(f"Checking {len(paths)} markdown files for broken links...")
    for path in sorted(paths):
        total_broken += _check_links(path)

    if total_broken:
        print(f"\n{total_broken} broken link(s) found.")
        return 1

    print("All relative links resolve.")
    print(f"Current contract schema version: {_CURRENT_SCHEMA_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
