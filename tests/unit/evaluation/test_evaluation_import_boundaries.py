"""Architecture boundary tests for ``hqsb.evaluation`` (S12).

The S12 layer sits above every other region and must stay importable on the
CPU-minimal installation, never pull in ``ops``, and never be imported by any
lower region.  The checks are static (AST) plus one subprocess probe so the
result does not depend on what the current test process already imported.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_DIR = REPO_ROOT / "hqsb" / "evaluation"

FORBIDDEN_MODULE_LEVEL = ("torch", "triton", "numpy", "scipy", "transformers", "pandas", "pyarrow")

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
    "hqsb/compiler",
)

EVALUATION_MODULES = (
    "identity",
    "records",
    "contracts",
    "layers",
    "campaign",
    "candidates",
    "comparability",
    "platform",
    "capability",
    "benchmark",
    "repeatability",
    "roofline",
    "energy",
    "cost",
    "pareto",
    "maturity",
    "lineage",
    "telemetry",
    "specs",
    "experiment",
    "interface_map",
)


def _python_files() -> list[Path]:
    return sorted(path for path in EVALUATION_DIR.glob("*.py"))


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
class TestEvaluationBoundaries:
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
        init = EVALUATION_DIR / "__init__.py"
        tree = ast.parse(init.read_text(encoding="utf-8"))
        for name in _module_level_imports(tree):
            assert not name.startswith("hqsb.evaluation."), name
        text = init.read_text(encoding="utf-8")
        for module in EVALUATION_MODULES:
            assert f'"{module}"' in text, module

    def test_evaluation_does_not_import_ops(self) -> None:
        offenders: list[tuple[str, str]] = []
        for path in _python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    offenders.extend(
                        (path.name, alias.name)
                        for alias in node.names
                        if alias.name.split(".")[0] == "ops"
                    )
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    if node.module.split(".")[0] == "ops":
                        offenders.append((path.name, node.module))
        assert offenders == []

    def test_lower_regions_do_not_import_evaluation(self) -> None:
        offenders: list[str] = []
        for region in LOWER_REGIONS:
            region_dir = REPO_ROOT / region
            if not region_dir.is_dir():
                continue
            for path in region_dir.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                if "hqsb.evaluation" in text:
                    offenders.append(str(path.relative_to(REPO_ROOT)))
        assert offenders == []

    def test_evaluation_imports_only_core_and_itself(self) -> None:
        allowed_prefixes = ("hqsb.core", "hqsb.evaluation")
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
            "copy", "uuid", "string", "warnings", "textwrap", "graphlib", "bisect", "statistics",
            "hqsb",
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
            "import hqsb.evaluation as e;"
            "from hqsb.evaluation import identity, records, contracts, layers, campaign;"
            "from hqsb.evaluation import candidates, comparability, platform, capability;"
            "from hqsb.evaluation import benchmark, repeatability, roofline, energy, cost;"
            "from hqsb.evaluation import pareto, maturity, lineage, telemetry, specs;"
            "from hqsb.evaluation import experiment as ex;"
            "from hqsb.evaluation import interface_map as im;"
            "print('OK', len(im.EXPERIMENTS), e.STAGE)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip().startswith("OK 10 S12")

    def test_dependency_gate_covers_the_evaluation_region(self) -> None:
        gate = (REPO_ROOT / "scripts" / "audit" / "import_dependency_gate.py").read_text(encoding="utf-8")
        assert '"evaluation"' in gate
        assert "hqsb.evaluation" in gate
        assert "R11" in gate and "R12" in gate
