"""Import boundaries for ``hqsb.experimental`` (S14, gate rules R15/R16).

Four independent checks, because each alone is escapable:

1. the gate's rule table really contains R15/R16 and the ``experimental`` region
   (a rule that was deleted would otherwise let the layer drift silently);
2. a static AST scan of ``hqsb/experimental/*.py`` finds no module-level import of
   a heavy framework (``torch``/``triton``/``numpy``/``transformers``/``ray``/
   ``vllm``/``tensorflow``/``onnx``) — a function-level probe is allowed and is
   exactly the pattern ``E14-01`` §3.3 requires;
3. no lower region imports ``hqsb.experimental`` (the reverse direction is the
   reason R15 exists);
4. a **subprocess** probe imports ``hqsb.experimental`` and reports the set of
   heavy modules it pulled in, so the answer does not depend on what the test
   process already imported.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Set

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXPERIMENTAL_DIR = os.path.join(REPO_ROOT, "hqsb", "experimental")
GATE_PATH = os.path.join(REPO_ROOT, "scripts", "audit", "import_dependency_gate.py")

#: Frameworks this layer must never import at module level (R16).
HEAVY_MODULES: Set[str] = {
    "torch",
    "triton",
    "numpy",
    "transformers",
    "ray",
    "vllm",
    "sglang",
    "tensorflow",
    "onnx",
    "onnxruntime",
    "diffusers",
    "librosa",
    "torchaudio",
    "deepspeed",
    "megatron",
    "peft",
    "trl",
}

#: Regions that must not import ``hqsb.experimental`` (R15).
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
]


def _experimental_modules() -> List[str]:
    return sorted(
        os.path.join(EXPERIMENTAL_DIR, name)
        for name in os.listdir(EXPERIMENTAL_DIR)
        if name.endswith(".py")
    )


def test_the_layer_has_modules_to_check() -> None:
    modules = _experimental_modules()
    assert len(modules) >= 19, f"expected the full S14 layer, found {len(modules)} modules"


def test_gate_declares_the_experimental_region_and_rules() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("_hqsb_gate", GATE_PATH)
    assert spec and spec.loader
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    regions = {region for region, _prefix in gate.REGION_PREFIXES}
    assert "experimental" in regions, "the gate does not know the experimental region"
    assert gate.region_of("hqsb.experimental.training") == "experimental"
    assert gate.region_of("hqsb.experimental") == "experimental"

    rules = {rule["id"]: rule for rule in gate.RULES}
    assert "R15" in rules, "R15 (lower layers must not import experimental) is missing"
    assert "R16" in rules, "R16 (experimental must not import ops) is missing"
    assert rules["R15"]["forbidden_target_regions"] == ["experimental"]
    assert rules["R16"]["forbidden_target_regions"] == ["ops"]
    assert gate.RULES_VERSION >= "1.8.0"


def _module_level_imports(path: str) -> List[str]:
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    found: List[str] = []

    def walk(node: ast.AST, nested: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                if not nested:
                    found.extend(alias.name for alias in child.names)
                continue
            if isinstance(child, ast.ImportFrom):
                if not nested and child.module:
                    found.append(child.module)
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(child, True)
                continue
            walk(child, nested)

    walk(tree, False)
    return found


@pytest.mark.unit
def test_no_module_level_heavy_imports() -> None:
    offenders: Dict[str, List[str]] = {}
    for path in _experimental_modules():
        for module in _module_level_imports(path):
            root = module.split(".", 1)[0]
            if root in HEAVY_MODULES:
                offenders.setdefault(os.path.basename(path), []).append(module)
    assert not offenders, (
        "hqsb.experimental must stay importable on the CPU-minimal installation "
        f"(E14-01 §3.3); module-level heavy imports found: {offenders}"
    )


@pytest.mark.unit
def test_package_init_is_lazy_only() -> None:
    with open(os.path.join(EXPERIMENTAL_DIR, "__init__.py"), encoding="utf-8") as handle:
        source = handle.read()
    assert "_LAZY" in source, "__init__ must expose a PEP 562 _LAZY map"
    assert "importlib.import_module(f\"{__name__}.{_LAZY[name]}\")" in source, (
        "the lazy accessor must be the only import path for submodules"
    )
    # A package-level `from . import x` or `from hqsb.experimental.x import y`
    # would create the cycle the gate rejects.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            pytest.fail(f"__init__ imports a submodule at module level: line {node.lineno}")


@pytest.mark.unit
def test_lower_regions_do_not_import_experimental() -> None:
    offenders: List[str] = []
    for relative in LOWER_REGIONS:
        path = os.path.join(REPO_ROOT, relative)
        targets: List[str] = []
        if os.path.isfile(path):
            targets = [path]
        elif os.path.isdir(path):
            for dirpath, _dirnames, filenames in os.walk(path):
                targets.extend(
                    os.path.join(dirpath, name) for name in filenames if name.endswith(".py")
                )
        for target in targets:
            for module in _module_level_imports(target):
                if module == "hqsb.experimental" or module.startswith("hqsb.experimental."):
                    offenders.append(f"{os.path.relpath(target, REPO_ROOT)} imports {module}")
    assert not offenders, f"R15 violation: {offenders}"


PROBE = """
import json, sys
import hqsb.experimental  # noqa: F401
heavy = {m for m in sys.modules if m.split('.')[0] in %(heavy)r}
print(json.dumps({"heavy": sorted(heavy), "lazy_present": "hqsb.experimental.training" in sys.modules}))
"""


@pytest.mark.unit
def test_cpu_minimal_import_in_a_subprocess() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", PROBE % {"heavy": sorted(HEAVY_MODULES)}],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    payload: Dict[str, Any] = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["heavy"] == [], (
        f"`import hqsb.experimental` pulled in heavy frameworks: {payload['heavy']}"
    )
    assert payload["lazy_present"] is False, (
        "the package-level import must not eagerly load a heavy submodule"
    )
