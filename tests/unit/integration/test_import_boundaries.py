"""Dependency-boundary tests for ``hqsb.integration`` (module ownership).

The integration layer must:

* import on the CPU-minimal installation (no torch, no triton, no model
  loader) — a missing accelerator is a structured reason, never an import
  failure;
* never be depended upon by ``hqsb.core``, ``hqsb.models`` or
  ``hqsb.benchmark`` (the dependency arrow points the other way);
* declare no module-level torch/triton import anywhere in the package.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_INTEGRATION_DIR = os.path.join(_REPO_ROOT, "hqsb", "integration")
_UPSTREAM_PACKAGES = ("hqsb/core", "hqsb/models", "hqsb/benchmark")


def _python_files(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


def _imported_modules(path: str):
    """Yield ``(module, line)`` for every import in the file (any depth)."""
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module, node.lineno


def _module_level_imports(path: str):
    """Yield imports that execute at import time (module scope, incl. if/try)."""
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    def walk_top_level(statements):
        for node in statements:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield alias.name, node.lineno
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    yield node.module, node.lineno
            elif isinstance(node, ast.If):
                yield from walk_top_level(node.body)
                yield from walk_top_level(node.orelse)
            elif isinstance(node, ast.Try):
                yield from walk_top_level(node.body)
                yield from walk_top_level(node.orelse)
                yield from walk_top_level(node.finalbody)
                for handler in node.handlers:
                    yield from walk_top_level(handler.body)

    yield from walk_top_level(tree.body)


@pytest.mark.unit
class TestIntegrationBoundaries:
    def test_no_module_level_heavy_imports(self):
        """Lazy imports inside functions are allowed; module-scope ones are not."""
        offenders = []
        for path in _python_files(_INTEGRATION_DIR):
            for module, line in _module_level_imports(path):
                top = module.split(".")[0]
                if top in ("torch", "triton", "transformers", "modelscope", "numpy"):
                    offenders.append(
                        f"{os.path.relpath(path, _REPO_ROOT)}:{line} imports {module}"
                    )
        assert offenders == [], "\n".join(offenders)

    def test_lazy_imports_are_documented_as_such(self):
        """Every heavy import in the package must sit inside a function."""
        module_level = set()
        for path in _python_files(_INTEGRATION_DIR):
            module_level.update(
                module.split(".")[0]
                for module, _line in _module_level_imports(path)
            )
        assert not module_level & {"torch", "triton", "numpy"}

    def test_integration_is_importable_without_torch(self):
        probe = textwrap.dedent(
            """\
            import importlib
            import sys
            for name in (
                "hqsb.integration.specs",
                "hqsb.integration.dispatch",
                "hqsb.integration.meta",
                "hqsb.integration.graph",
                "hqsb.integration.patterns",
                "hqsb.integration.guards",
                "hqsb.integration.cache",
                "hqsb.integration.lowering",
                "hqsb.integration.cuda_graph",
                "hqsb.integration.taxonomy",
                "hqsb.integration.abi",
                "hqsb.integration.lifecycle",
                "hqsb.integration.adapter",
                "hqsb.integration.differential",
                "hqsb.integration.telemetry",
                "hqsb.integration.policies",
                "hqsb.integration.experiment",
                "hqsb.integration.interface_map",
            ):
                importlib.import_module(name)
            bad = [m for m in sys.modules
                   if m == "torch" or m.startswith("torch.")
                   or m == "triton" or m.startswith("triton.")]
            print("BAD_IMPORTS=" + ",".join(bad))
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env=env,
            cwd=_REPO_ROOT,
        )
        assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        assert tail == "BAD_IMPORTS=", f"integration transitively imported torch/triton: {tail}"

    def test_upstream_packages_do_not_depend_on_integration(self):
        violations = []
        for relative in _UPSTREAM_PACKAGES:
            for path in _python_files(os.path.join(_REPO_ROOT, relative)):
                for module, line in _imported_modules(path):
                    if module.startswith("hqsb.integration"):
                        violations.append(
                            f"{os.path.relpath(path, _REPO_ROOT)}:{line} imports {module}"
                        )
        assert violations == [], "\n".join(violations)

    def test_core_does_not_import_integration(self):
        for path in _python_files(os.path.join(_REPO_ROOT, "hqsb", "core")):
            for module, line in _imported_modules(path):
                assert not module.startswith("hqsb.integration"), (
                    f"{path}:{line} imports {module}"
                )

    def test_ops_layer_is_not_imported_by_integration(self):
        offenders = []
        for path in _python_files(_INTEGRATION_DIR):
            for module, line in _imported_modules(path):
                if module == "ops" or module.startswith("ops."):
                    offenders.append(f"{os.path.relpath(path, _REPO_ROOT)}:{line}")
        assert offenders == [], "\n".join(offenders)
