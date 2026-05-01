"""Pytest configuration and shared fixtures for the HQSB test suite.

This module guarantees that ``import hqsb`` resolves to the repository
root regardless of how the test suite is invoked, so tests never depend
on an externally managed ``PYTHONPATH``.
"""

from __future__ import annotations

import os
import sys

# Repository root is the parent of this ``tests/`` directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
