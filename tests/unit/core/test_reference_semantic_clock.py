"""E02-01 guards for the reference semantic and clock convention.

Two properties must hold for the S02 reference runtime to be trustworthy:

1. The model-core clock (``decode_total_ms`` / ``model_core_ttft_ms`` /
   ``model_core_e2e_ms``) has **one** implementation —
   :func:`hqsb.benchmark.metrics.model_core_timings` — and both the
   model-core engine and the benchmark engine consume it instead of
   re-deriving the relationship inline.

2. Every timed region in ``benchmark_model_core`` is bracketed by
   ``torch.cuda.synchronize()`` (before start and after the launch), so an
   asynchronous CUDA launch is never mis-recorded as completion.

These are static guards: they read the source files, so they pass on the
CPU CI without a GPU or model weights while still catching drift in the
two files that define the reference convention.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODEL_CORE = _REPO_ROOT / "hqsb" / "benchmark" / "model_core.py"
_ENGINE = _REPO_ROOT / "hqsb" / "benchmark" / "engine.py"
_METRICS = _REPO_ROOT / "hqsb" / "benchmark" / "metrics.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(_source(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
    return names


class TestSingleFormulaImplementation:
    def test_model_core_imports_single_formula(self):
        assert "model_core_timings" in _imports(_MODEL_CORE)

    def test_engine_imports_single_formula(self):
        assert "model_core_timings" in _imports(_ENGINE)

    def test_no_inline_formula_outside_metrics(self):
        # The raw ``prefill + selection`` relationship may appear only in the
        # metrics helper's docstring (the single source of truth).
        inline = "prefill_forward_ms + first_token_selection_ms"
        assert inline not in _source(_MODEL_CORE)
        assert inline not in _source(_ENGINE)

    def test_metrics_defines_single_formula(self):
        assert "def model_core_timings" in _source(_METRICS)


class TestSyncPointBracketing:
    """Timed regions in ``benchmark_model_core`` must sync before/after."""

    def _function_source(self) -> ast.FunctionDef:
        tree = ast.parse(_source(_MODEL_CORE))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "benchmark_model_core":
                return node
        raise AssertionError("benchmark_model_core not found")

    def test_benchmark_model_core_exists(self):
        self._function_source()

    def test_timed_regions_are_cuda_synchronized(self):
        """The three timed regions each contain a ``torch.cuda.synchronize``.

        We assert the function body references ``torch.cuda.synchronize`` a
        sufficient number of times (prefill before+after, selection after,
        and each decode step before+after), which is the mechanism that keeps
        ``time.perf_counter`` from treating an async launch as completion.
        """
        body = ast.unparse(self._function_source())
        # Occurrences: prefill before + prefill after + selection after +
        # decode-loop before + decode-loop after == 5 sync calls.
        assert body.count("torch.cuda.synchronize()") >= 5

    def test_perf_counter_marks_clock_regions(self):
        body = ast.unparse(self._function_source())
        assert "prefill_start = time.perf_counter()" in body
        assert "first_token_start = time.perf_counter()" in body
        assert "decode_start = time.perf_counter()" in body
