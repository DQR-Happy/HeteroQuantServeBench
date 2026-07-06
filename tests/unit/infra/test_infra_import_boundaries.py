"""Architecture boundary tests for ``hqsb.infra`` (S13).

The infra layer sits above every other region and must stay importable on the
CPU-minimal installation, never pull in ``ops``, and never be imported by any lower
region.  The checks are static (AST) plus one subprocess probe so the result does
not depend on what the current test process already imported.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INFRA_DIR = REPO_ROOT / "hqsb" / "infra"

FORBIDDEN_MODULE_LEVEL = ("torch", "triton", "numpy", "scipy", "transformers", "pandas", "pyarrow", "kubernetes")

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
    "hqsb/evaluation",
)

INFRA_MODULES = (
    "identity",
    "records",
    "contracts",
    "campaign",
    "supply_chain",
    "deployment",
    "scheduling",
    "artifacts",
    "lifecycle",
    "capacity",
    "autoscaling",
    "observability",
    "faults",
    "canary",
    "security",
    "telemetry",
    "specs",
    "experiment",
    "interface_map",
)


def _python_files() -> list[Path]:
    return sorted(INFRA_DIR.glob("*.py"))


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
class TestInfraBoundaries:
    def test_no_module_level_heavy_dependencies(self) -> None:
        offenders: list[tuple[str, str]] = []
        for path in _python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for name in _module_level_imports(tree):
                if name.split(".")[0] in FORBIDDEN_MODULE_LEVEL:
                    offenders.append((path.name, name))
        assert offenders == []

    def test_package_init_does_not_import_submodules(self) -> None:
        init = INFRA_DIR / "__init__.py"
        tree = ast.parse(init.read_text(encoding="utf-8"))
        for name in _module_level_imports(tree):
            assert not name.startswith("hqsb.infra."), name
        text = init.read_text(encoding="utf-8")
        for module in INFRA_MODULES:
            assert f'"{module}"' in text, module

    def test_infra_does_not_import_ops(self) -> None:
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

    def test_lower_regions_do_not_import_infra(self) -> None:
        offenders: list[str] = []
        for region in LOWER_REGIONS:
            region_dir = REPO_ROOT / region
            if not region_dir.is_dir():
                continue
            for path in region_dir.rglob("*.py"):
                if "hqsb.infra" in path.read_text(encoding="utf-8"):
                    offenders.append(str(path.relative_to(REPO_ROOT)))
        assert offenders == []

    def test_infra_imports_only_core_and_itself(self) -> None:
        """The layer may read S08/S12 evidence only through records it owns."""
        allowed = ("hqsb.core", "hqsb.infra")
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
                    if not name.startswith(allowed):
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

    def test_layer_imports_without_torch_or_kubernetes(self) -> None:
        script = (
            "import sys;"
            "sys.modules['torch'] = None;"
            "sys.modules['triton'] = None;"
            "sys.modules['numpy'] = None;"
            "sys.modules['kubernetes'] = None;"
            "import hqsb.infra as infra;"
            "from hqsb.infra import identity, records, contracts, campaign;"
            "from hqsb.infra import supply_chain, deployment, scheduling, artifacts, lifecycle;"
            "from hqsb.infra import capacity, autoscaling, observability, faults, canary, security;"
            "from hqsb.infra import telemetry, specs, experiment as ex, interface_map as im;"
            "print('OK', len(im.EXPERIMENTS), im.total_steps(), infra.STAGE)"
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
        assert completed.stdout.strip().startswith("OK 11 418 S13")

    def test_dependency_gate_covers_the_infra_region(self) -> None:
        gate = (REPO_ROOT / "scripts" / "audit" / "import_dependency_gate.py").read_text(encoding="utf-8")
        assert '"infra"' in gate
        assert "hqsb.infra" in gate
        assert "R13" in gate and "R14" in gate

    def test_ownership_document_registers_the_region(self) -> None:
        """The ownership document must declare *the version the gate actually enforces*.

        This assertion used to hard-code ``1.7.0``.  S14 added the ``experimental``
        region and rules R15/R16, which legitimately advances the rules version, so
        the literal went stale — a document literal cannot stay correct across
        stages.  The replacement is **stricter, not looser**: the version is read
        from the gate itself and compared with the document, so the two can no
        longer drift apart (previously both could say ``1.7.0`` while disagreeing
        about which rules exist).  The negative control below proves the comparison
        still rejects a mismatched document.
        """
        import re

        ownership = (REPO_ROOT / "docs" / "architecture" / "module_ownership.md").read_text(encoding="utf-8")
        gate_source = (REPO_ROOT / "scripts" / "audit" / "import_dependency_gate.py").read_text(encoding="utf-8")
        gate_version = re.search(r'RULES_VERSION\s*=\s*"(\d+\.\d+\.\d+)"', gate_source)
        assert gate_version, "the gate no longer declares RULES_VERSION"
        declared = re.search(r"版本：(\d+\.\d+\.\d+)", ownership)
        assert declared, "the ownership document no longer declares a rules version"
        assert declared.group(1) == gate_version.group(1), (
            f"module_ownership.md declares rules version {declared.group(1)} but the gate enforces "
            f"{gate_version.group(1)}: the document and the enforced rules have drifted apart"
        )
        assert "hqsb.infra" in ownership
        assert "hqsb.experimental" in ownership
        for rule in ("R13", "R14", "R15", "R16"):
            assert rule in ownership, f"the ownership document does not register {rule}"

        # Negative control: a document whose declared version disagrees with the
        # gate must fail the very comparison above.
        stale = ownership.replace(f"版本：{declared.group(1)}", "版本：0.0.1")
        stale_declared = re.search(r"版本：(\d+\.\d+\.\d+)", stale)
        assert stale_declared and stale_declared.group(1) != gate_version.group(1)
