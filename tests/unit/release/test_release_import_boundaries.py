"""Import boundaries for ``hqsb.release`` (S15, gate rules R17/R18).

Four independent checks, because each alone is escapable:

1. the gate's rule table really contains R17/R18 and the ``release`` region (a
   rule that was deleted would otherwise let the layer drift silently);
2. a static AST scan of ``hqsb/release/*.py`` finds no module-level import of a
   heavy framework (``torch``/``triton``/``numpy``/``requests``/``httpx``/
   ``aiohttp``) — a function-level probe is allowed and is exactly the pattern
   the protocol requires;
3. no lower region imports ``hqsb.release`` (the reverse direction is the reason
   R17 exists);
4. a **subprocess** probe imports ``hqsb.release`` and reports the set of heavy
   modules it pulled in, so the answer does not depend on what the test process
   already imported.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from typing import List, Set

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RELEASE_DIR = os.path.join(REPO_ROOT, "hqsb", "release")
GATE_PATH = os.path.join(REPO_ROOT, "scripts", "audit", "import_dependency_gate.py")

#: Frameworks this layer must never import at module level (R18).
HEAVY_MODULES: Set[str] = {
    "torch",
    "triton",
    "numpy",
    "requests",
    "httpx",
    "aiohttp",
    "transformers",
    "ray",
    "vllm",
    "sglang",
    "docker",
    "twine",
    "build",
}

#: Regions that must not import ``hqsb.release`` (R17).
LOWER_REGIONS: List[str] = [
    "hqsb/core/__init__.py",
    "hqsb/models",
    "hqsb/benchmark",
    "hqsb/backends",
    "hqsb/quant",
    "hqsb/runtime",
    "hqsb/serving",
    "hqsb/distributed",
    "hqsb/integration",
    "hqsb/infra",
    "hqsb/experimental",
    "hqsb/compiler",
    "hqsb/evaluation",
]


def _release_modules() -> List[str]:
    return sorted(
        os.path.join(RELEASE_DIR, name)
        for name in os.listdir(RELEASE_DIR)
        if name.endswith(".py")
    )


def _module_level_imports(path: str) -> List[ast.Import]:
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    return [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]


def _imported_roots(node: ast.AST) -> List[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.module is None or node.level:
            return []
        return [node.module.split(".")[0]]
    return []


def _imports_release(node: ast.AST) -> bool:
    """True if this node imports (part of) ``hqsb.release``."""
    if isinstance(node, ast.Import):
        return any(alias.name == "hqsb.release" or alias.name.startswith("hqsb.release.") for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return bool(node.module) and (node.module == "hqsb.release" or node.module.startswith("hqsb.release."))
    return False


class TestReleaseImportBoundaries:
    def test_dependency_gate_covers_the_release_region(self) -> None:
        gate = open(GATE_PATH, encoding="utf-8").read()
        assert '"release"' in gate
        assert "hqsb.release" in gate
        assert "R17" in gate and "R18" in gate

    def test_ownership_document_registers_the_region(self) -> None:
        import re

        ownership = open(os.path.join(REPO_ROOT, "docs", "architecture", "module_ownership.md"), encoding="utf-8").read()
        gate_source = open(GATE_PATH, encoding="utf-8").read()
        gate_version = re.search(r'RULES_VERSION\s*=\s*"(\d+\.\d+\.\d+)"', gate_source)
        assert gate_version
        declared = re.search(r"版本：(\d+\.\d+\.\d+)", ownership)
        assert declared
        assert declared.group(1) == gate_version.group(1)
        assert "hqsb.release" in ownership
        for rule in ("R17", "R18"):
            assert rule in ownership

    def test_no_module_level_heavy_imports(self) -> None:
        offenders: List[str] = []
        for path in _release_modules():
            for node in _module_level_imports(path):
                for root in _imported_roots(node):
                    if root in HEAVY_MODULES:
                        offenders.append(f"{path}: {root}")
        assert not offenders, f"module-level heavy imports: {offenders}"

    def test_lower_regions_do_not_import_release(self) -> None:
        for region in LOWER_REGIONS:
            region_path = os.path.join(REPO_ROOT, region)
            if os.path.isfile(region_path):
                paths = [region_path]
            elif os.path.isdir(region_path):
                paths = [
                    os.path.join(dirpath, name)
                    for dirpath, _dirs, files in os.walk(region_path)
                    for name in files
                    if name.endswith(".py")
                ]
            else:
                continue
            for path in paths:
                for node in _module_level_imports(path):
                    if _imports_release(node):
                        pytest.fail(f"{path} imports hqsb.release (R17)")


def test_cpu_minimal_import_in_a_subprocess() -> None:
    """A subprocess import must pull in zero heavy frameworks (R18)."""
    code = (
        "import json, sys; import hqsb.release; "
        "heavy = [m for m in sys.modules if m.split('.')[0] in "
        "{'torch','triton','numpy','requests','httpx','transformers','ray','vllm'}]; "
        "print(json.dumps(sorted(heavy)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    heavy = json.loads(completed.stdout.strip())
    assert heavy == [], f"import hqsb.release pulled heavy frameworks: {heavy}"
