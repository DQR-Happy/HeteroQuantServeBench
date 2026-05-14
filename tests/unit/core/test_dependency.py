"""Dependency-rule enforcement tests.

Verifies the architectural invariant that ``hqsb.core`` never imports a
concrete backend, operator implementation, model loader, or serving module
(top-level architecture §3.1/§6: ``core`` depends on nothing concrete).
"""

from __future__ import annotations

import ast
import os

import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_CORE_DIR = os.path.join(_REPO_ROOT, "hqsb", "core")

# Submodule prefixes that ``core`` must never import. ``hqsb.core.*``
# (including ``hqsb.core.contracts``) is allowed.
_FORBIDDEN_PREFIXES = (
    "hqsb.backends",
    "hqsb.models",
    "hqsb.quant",
    "hqsb.serving",
    "hqsb.benchmark",
    "ops.",
    "ops ",
)


def _python_files(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


def _imported_modules(path: str):
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module


@pytest.mark.unit
class TestCoreDependencyRules:
    def test_core_does_not_import_concrete_modules(self):
        violations = []
        for path in _python_files(_CORE_DIR):
            rel = os.path.relpath(path, _REPO_ROOT)
            for module in _imported_modules(path):
                if module.startswith(_FORBIDDEN_PREFIXES):
                    violations.append(f"{rel} imports {module}")
        assert violations == [], "\n".join(violations)

    def test_core_imports_only_core_and_stdlib(self):
        # Allow `hqsb.core.*`, `pydantic`, `yaml`, and stdlib only.
        allowed_tops = {
            "hqsb", "pydantic", "yaml", "typing", "abc", "enum", "json",
            "hashlib", "os", "sys", "time", "uuid", "re", "contextvars",
            "logging", "functools", "random", "dataclasses", "collections",
            "glob", "importlib", "platform", "subprocess", "pathlib",
            "types", "__future__",
        }
        for path in _python_files(_CORE_DIR):
            for module in _imported_modules(path):
                top = module.split(".")[0]
                assert top in allowed_tops, (
                    f"{os.path.relpath(path, _REPO_ROOT)} imports {module}"
                )

    def test_core_package_is_importable_standalone(self):
        # Importing core must not transitively require torch or modelscope.
        # The probe runs in a *fresh* interpreter so that earlier test modules
        # importing torch in this process can never make the gate
        # tautological (E01-05: replaces the former ``assert ... or True``).
        import subprocess
        import sys
        import textwrap

        probe = textwrap.dedent(
            """\
            import sys
            import hqsb.core  # noqa: F401
            bad = [m for m in sys.modules
                   if m == "torch" or m.startswith("torch.")
                   or m == "modelscope" or m.startswith("modelscope.")]
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
        assert tail == "BAD_IMPORTS=", (
            f"hqsb.core import transitively loaded forbidden modules: {tail}"
        )
