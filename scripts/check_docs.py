#!/usr/bin/env python3
"""Documentation integrity checker.

Checks that relative Markdown links in ``docs/`` and ``README.md`` resolve to
existing files, so a broken link cannot silently rot documentation.
Private experiment archives are checked separately with --include-private;
they are intentionally absent from a public checkout. This checker does not
claim to validate every prose schema version or execute documentation commands.

Usage:
    python scripts/check_docs.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from urllib.parse import unquote

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Matches [text](target) where target is a relative path (no scheme).
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

def _iter_markdown(root: str, *, include_private: bool = False):
    for dirpath, dirnames, filenames in os.walk(root):
        if not include_private:
            dirnames[:] = [name for name in dirnames if name != "stage_experiments"]
        for filename in filenames:
            if filename.endswith(".md"):
                yield os.path.join(dirpath, filename)


def _check_links(path: str) -> int:
    """Return the number of broken relative links found in ``path``."""
    broken = 0
    base_dir = os.path.dirname(path)
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    # Code examples are not links and may intentionally demonstrate bad input.
    text = re.sub(r"```.*?```", "", text, flags=re.S)

    for target in _LINK_RE.findall(text):
        # Skip anchors, absolute URLs, and mailto links.
        if target.startswith("#") or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
            continue
        # Strip a trailing anchor fragment.
        clean = unquote(target.strip("<>").split("#", 1)[0])
        if not clean:
            continue
        resolved = os.path.normpath(os.path.join(base_dir, clean))
        if not os.path.exists(resolved):
            print(f"  [broken link] {path}: -> {target}")
            broken += 1
    return broken


def main() -> int:
    """Run all documentation checks and return the number of failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-private", action="store_true")
    args = parser.parse_args()
    root = os.path.join(_REPO_ROOT, "docs")
    readme = os.path.join(_REPO_ROOT, "README.md")

    total_broken = 0
    paths = list(_iter_markdown(root, include_private=args.include_private))
    if os.path.isfile(readme):
        paths.append(readme)

    print(f"Checking {len(paths)} markdown files for broken links...")
    for path in sorted(paths):
        total_broken += _check_links(path)

    if total_broken:
        print(f"\n{total_broken} broken link(s) found.")
        return 1

    print("All relative links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
