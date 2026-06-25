"""The reference path must not require any optional DSL package."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


@pytest.mark.unit
def test_reference_has_no_module_scope_optional_import():
    path = Path(__file__).resolve().parents[3] / "ops/reference.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported.intersection({"torch", "triton", "tilelang", "cutlass"})
