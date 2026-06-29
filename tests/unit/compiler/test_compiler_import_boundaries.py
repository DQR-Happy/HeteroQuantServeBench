"""Architecture boundary tests for ``hqsb.compiler`` (S11).

The S11 layer must stay importable on the CPU-minimal installation and must
never pull in ``ops``; no lower region may depend on it.  The checks are static
(AST) plus one subprocess probe so that the result does not depend on what the
current test process already imported.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPILER_DIR = REPO_ROOT / "hqsb" / "compiler"

FORBIDDEN_MODULE_LEVEL = ("torch", "triton", "numpy", "scipy", "transformers")

LOWER_REGIONS = (
    "hqsb/core",
    "hqsb/models",
    "hqsb/benchmark",
    "hqsb/backends",
    "hqsb/hardware",
    "hqsb/quant",
    "hqsb/integration",
    "hqsb/runtime",
    "hqsb/serving",
    "hqsb/distributed",
)


def _python_files() -> list[Path]:
    return sorted(path for path in COMPILER_DIR.glob("*.py"))


def _module_level_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.append(node.module)
    return names


@pytest.mark.unit
class TestCompilerBoundaries:
    def test_no_module_level_heavy_dependencies(self) -> None:
        offenders: list[tuple[str, str]] = []
        for path in _python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for name in _module_level_imports(tree):
                root = name.split(".")[0]
                if root in FORBIDDEN_MODULE_LEVEL:
                    offenders.append((path.name, name))
        assert offenders == []

    def test_package_init_does_not_import_submodules(self) -> None:
        init = COMPILER_DIR / "__init__.py"
        tree = ast.parse(init.read_text(encoding="utf-8"))
        for name in _module_level_imports(tree):
            assert not name.startswith("hqsb.compiler."), name
        # the lazy table must cover every documented module
        text = init.read_text(encoding="utf-8")
        for module in (
            "identity",
            "ir",
            "records",
            "capture",
            "guards",
            "rewrite",
            "lowering",
            "backend",
            "codegen",
            "autotune",
            "costmodel",
            "cache",
            "portable",
            "aigate",
            "telemetry",
            "specs",
            "experiment",
            "interface_map",
        ):
            assert f'"{module}"' in text

    def test_compiler_does_not_import_ops(self) -> None:
        offenders: list[tuple[str, str]] = []
        for path in _python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    offenders.extend(
                        (path.name, alias.name) for alias in node.names if alias.name.split(".")[0] == "ops"
                    )
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    if node.module.split(".")[0] == "ops":
                        offenders.append((path.name, node.module))
        assert offenders == []

    def test_lower_regions_do_not_import_compiler(self) -> None:
        offenders: list[str] = []
        for region in LOWER_REGIONS:
            region_dir = REPO_ROOT / region
            for path in region_dir.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                if "hqsb.compiler" in text:
                    offenders.append(str(path.relative_to(REPO_ROOT)))
        assert offenders == []

    def test_compiler_imports_only_core_and_stdlib(self) -> None:
        """Intra-project imports: core and the layer itself only.

        Heavy third-party imports are allowed *inside functions* (lazy optional
        dependencies); at module level only stdlib/yaml are permitted, which
        ``test_no_module_level_heavy_dependencies`` pins.
        """
        allowed_prefixes = ("hqsb.core", "hqsb.compiler")
        offenders: list[tuple[str, str]] = []
        for path in _python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = [node.module]
                for name in modules:
                    if not name.startswith("hqsb."):
                        continue
                    if not name.startswith(allowed_prefixes):
                        offenders.append((path.name, name))
        assert offenders == []

    def test_module_level_imports_are_stdlib_or_yaml(self) -> None:
        allowed = {
            "__future__", "yaml", "dataclasses", "typing", "json", "hashlib", "os", "re", "math",
            "random", "subprocess", "shutil", "sys", "platform", "time", "tempfile", "importlib",
            "glob", "collections", "pathlib", "functools", "itertools", "struct", "enum", "abc",
            "copy", "uuid", "string", "warnings", "textwrap", "hqsb",
        }
        offenders: list[tuple[str, str]] = []
        for path in _python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for name in _module_level_imports(tree):
                if name.split(".")[0] not in allowed:
                    offenders.append((path.name, name))
        assert offenders == []

    def test_layer_imports_without_torch(self) -> None:
        script = (
            "import sys;"
            "sys.modules['torch'] = None;"
            "sys.modules['triton'] = None;"
            "sys.modules['numpy'] = None;"
            "import hqsb.compiler as c;"
            "from hqsb.compiler import capture, guards, ir, rewrite, lowering, targets;"
            "from hqsb.compiler import backend, codegen, autotune, costmodel, cache;"
            "from hqsb.compiler import portable, aigate, telemetry, specs, identity, records;"
            "from hqsb.compiler import experiment as e;"
            "from hqsb.compiler import interface_map as m;"
            "print('OK', len(m.EXPERIMENTS), c.STAGE)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip().startswith("OK 10 S11")

    def test_dependency_gate_covers_the_compiler_region(self) -> None:
        gate = (REPO_ROOT / "scripts" / "audit" / "import_dependency_gate.py").read_text(encoding="utf-8")
        assert '"compiler"' in gate
        assert "hqsb.compiler" in gate
        assert "R9" in gate and "R10" in gate
