"""Dependency-boundary tests for ``hqsb.serving`` (module ownership, rule R5).

The serving layer is the top of the dependency graph

    core ← models/benchmark ← backends/integration/quant ← runtime ← serving

and therefore must:

* import on the CPU-minimal installation (no torch/triton/numpy at module level);
* never be depended upon by any lower layer (the arrow points one way);
* never import ``ops`` (kernels are addressed through capability names);
* only reach the Runtime through its public contract objects.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SERVING_DIR = os.path.join(_REPO_ROOT, "hqsb", "serving")

#: Every package below serving in the ownership graph.
_DOWNSTREAM_PACKAGES = (
    "hqsb/core",
    "hqsb/models",
    "hqsb/benchmark",
    "hqsb/backends",
    "hqsb/hardware",
    "hqsb/integration",
    "hqsb/quant",
    "hqsb/runtime",
)

_SERVING_MODULES = (
    "hqsb.serving.protocol",
    "hqsb.serving.sse",
    "hqsb.serving.timing",
    "hqsb.serving.pipeline",
    "hqsb.serving.transport",
    "hqsb.serving.transport_http",
    "hqsb.serving.clients",
    "hqsb.serving.gateway",
    "hqsb.serving.dummy_backend",
    "hqsb.serving.slo",
    "hqsb.serving.arrival",
    "hqsb.serving.loadgen",
    "hqsb.serving.fairness",
    "hqsb.serving.policies",
    "hqsb.serving.admission",
    "hqsb.serving.router",
    "hqsb.serving.circuit",
    "hqsb.serving.cache_routing",
    "hqsb.serving.faults",
    "hqsb.serving.observability",
    "hqsb.serving.service_ab",
    "hqsb.serving.telemetry",
    "hqsb.serving.specs",
    "hqsb.serving.experiment",
    "hqsb.serving.interface_map",
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
class TestServingBoundaries:
    def test_no_module_level_heavy_imports(self):
        offenders = []
        for path in _python_files(_SERVING_DIR):
            for module, line in _module_level_imports(path):
                top = module.split(".")[0]
                if top in ("torch", "triton", "transformers", "modelscope", "numpy"):
                    offenders.append(
                        f"{os.path.relpath(path, _REPO_ROOT)}:{line} imports {module}"
                    )
        assert offenders == [], "\n".join(offenders)

    def test_serving_is_importable_without_torch(self):
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
            % (_SERVING_MODULES,)
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
        assert tail == "BAD_IMPORTS=", f"serving transitively imported torch/triton/numpy: {tail}"

    def test_downstream_packages_do_not_depend_on_serving(self):
        violations = []
        for relative in _DOWNSTREAM_PACKAGES:
            root = os.path.join(_REPO_ROOT, relative)
            if not os.path.isdir(root):
                continue
            for path in _python_files(root):
                for module, line in _imported_modules(path):
                    if module.startswith("hqsb.serving"):
                        violations.append(
                            f"{os.path.relpath(path, _REPO_ROOT)}:{line} imports {module}"
                        )
        assert violations == [], "\n".join(violations)

    def test_ops_layer_is_not_imported(self):
        offenders = []
        for path in _python_files(_SERVING_DIR):
            for module, line in _imported_modules(path):
                if module == "ops" or module.startswith("ops."):
                    offenders.append(f"{os.path.relpath(path, _REPO_ROOT)}:{line}")
        assert offenders == [], "\n".join(offenders)

    def test_serving_only_reaches_runtime_through_public_modules(self):
        # The runtime dependency is limited to the stable public surface used by
        # the S08 evidence layer (metrics, policy_ab, request, prefix_cache,
        # comparison, telemetry, experiment) — no private engine internals.
        allowed_prefixes = (
            "hqsb.runtime.metrics",
            "hqsb.runtime.policy_ab",
            "hqsb.runtime.request",
            "hqsb.runtime.prefix_cache",
            "hqsb.runtime.comparison",
            "hqsb.runtime.telemetry",
            "hqsb.runtime.experiment",
        )
        offenders = []
        for path in _python_files(_SERVING_DIR):
            for module, line in _imported_modules(path):
                if module.startswith("hqsb.runtime") and not module.startswith(allowed_prefixes):
                    offenders.append(
                        f"{os.path.relpath(path, _REPO_ROOT)}:{line} imports {module}"
                    )
        assert offenders == [], "\n".join(offenders)
