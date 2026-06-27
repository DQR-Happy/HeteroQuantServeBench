"""Dependency-boundary tests for ``hqsb.distributed`` (module ownership, R7/R8).

The distributed layer sits above ``runtime``/``serving``:

    core ← models/benchmark ← backends/integration/quant ← runtime ← serving
         ← distributed

and therefore must:

* import on the CPU-minimal installation (no torch/triton/numpy at module level);
* never be depended upon by any lower layer (the arrow points one way);
* never import ``ops`` (kernels are addressed through capability names);
* only reach runtime/serving through their public contract objects.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DISTRIBUTED_DIR = os.path.join(_REPO_ROOT, "hqsb", "distributed")

#: Every package below the distributed layer in the ownership graph.
_DOWNSTREAM_PACKAGES = (
    "hqsb/core",
    "hqsb/models",
    "hqsb/benchmark",
    "hqsb/backends",
    "hqsb/hardware",
    "hqsb/integration",
    "hqsb/quant",
    "hqsb/runtime",
    "hqsb/serving",
    "hqsb/ascend",
)

_DISTRIBUTED_MODULES = (
    "hqsb.distributed.topology",
    "hqsb.distributed.ranks",
    "hqsb.distributed.placement",
    "hqsb.distributed.probes",
    "hqsb.distributed.backend",
    "hqsb.distributed.collectives",
    "hqsb.distributed.faults",
    "hqsb.distributed.sequence",
    "hqsb.distributed.parallel_plan",
    "hqsb.distributed.ledger",
    "hqsb.distributed.scaling",
    "hqsb.distributed.overlap",
    "hqsb.distributed.boundary",
    "hqsb.distributed.moe",
    "hqsb.distributed.traces",
    "hqsb.distributed.telemetry",
    "hqsb.distributed.specs",
    "hqsb.distributed.experiment",
    "hqsb.distributed.interface_map",
)


def _python_files(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


def _imported_modules(path: str):
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
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    def walk(statements):
        for node in statements:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield alias.name, node.lineno
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    yield node.module, node.lineno
            elif isinstance(node, ast.If):
                yield from walk(node.body)
                yield from walk(node.orelse)
            elif isinstance(node, ast.Try):
                yield from walk(node.body)
                yield from walk(node.orelse)
                yield from walk(node.finalbody)
                for handler in node.handlers:
                    yield from walk(handler.body)

    yield from walk(tree.body)


@pytest.mark.unit
class TestDistributedBoundaries:
    def test_no_module_level_heavy_imports(self):
        offenders = []
        for path in _python_files(_DISTRIBUTED_DIR):
            for module, line in _module_level_imports(path):
                top = module.split(".")[0]
                if top in ("torch", "triton", "transformers", "modelscope", "numpy"):
                    offenders.append(
                        f"{os.path.relpath(path, _REPO_ROOT)}:{line} imports {module}"
                    )
        assert offenders == [], "\n".join(offenders)

    def test_distributed_is_importable_without_torch(self):
        probe = textwrap.dedent(
            """\
            import importlib
            import sys
            for name in %r:
                importlib.import_module(name)
            bad = [m for m in sys.modules
                   if m == "torch" or m.startswith("torch.")
                   or m == "triton" or m.startswith("triton.")
                   or m == "numpy" or m.startswith("numpy.")]
            print("BAD_IMPORTS=" + ",".join(bad))
            """
            % (_DISTRIBUTED_MODULES,)
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
        assert tail == "BAD_IMPORTS=", f"distributed transitively imported torch/triton/numpy: {tail}"

    def test_downstream_packages_do_not_depend_on_distributed(self):
        violations = []
        for relative in _DOWNSTREAM_PACKAGES:
            root = os.path.join(_REPO_ROOT, relative)
            if not os.path.isdir(root):
                continue
            for path in _python_files(root):
                for module, line in _imported_modules(path):
                    if module.startswith("hqsb.distributed"):
                        violations.append(
                            f"{os.path.relpath(path, _REPO_ROOT)}:{line} imports {module}"
                        )
        assert violations == [], "\n".join(violations)

    def test_ops_layer_is_not_imported(self):
        offenders = []
        for path in _python_files(_DISTRIBUTED_DIR):
            for module, line in _imported_modules(path):
                if module == "ops" or module.startswith("ops."):
                    offenders.append(f"{os.path.relpath(path, _REPO_ROOT)}:{line}")
        assert offenders == [], "\n".join(offenders)

    def test_distributed_only_reaches_lower_layers_through_public_modules(self):
        # The lower-layer dependency is limited to the stable public surface the
        # S10 evidence layer consumes (C6/C7 contracts, runtime experiment record,
        # runtime/metrics-equivalent helpers).  No private engine internals.
        allowed_prefixes = (
            "hqsb.core",
            "hqsb.runtime.experiment",
        )
        offenders = []
        for path in _python_files(_DISTRIBUTED_DIR):
            for module, line in _imported_modules(path):
                if not module.startswith("hqsb."):
                    continue
                if module.startswith("hqsb.distributed"):
                    continue
                if not module.startswith(allowed_prefixes):
                    offenders.append(
                        f"{os.path.relpath(path, _REPO_ROOT)}:{line} imports {module}"
                    )
        assert offenders == [], "\n".join(offenders)

    def test_package_exports_experiments_and_stage(self):
        import hqsb.distributed as distributed

        assert distributed.STAGE == "S10"
        assert distributed.EXPERIMENTS == tuple(f"E10-{index:02d}" for index in range(1, 11))
