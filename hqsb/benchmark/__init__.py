"""HQSB Benchmark module.

Provides the backend-interface benchmark engine, workload generation,
metrics computation, resource monitoring (tegrastats), and power/energy
analysis. The engine depends only on ``hqsb.core`` contracts and the
``Backend`` interface; concrete backends are injected by callers.
"""

from hqsb.benchmark.metrics import (
    latency_summary,
    numerical_diff_summary,
    percentile,
)
from hqsb.benchmark.engine import BenchmarkEngine, run_backend

__all__ = [
    "BenchmarkEngine",
    "run_backend",
    "latency_summary",
    "numerical_diff_summary",
    "percentile",
    "make_fixed_token_input",
    "benchmark_model_core",
]


def __getattr__(name: str):
    """Lazily expose torch-dependent helpers.

    ``make_fixed_token_input`` and ``benchmark_model_core`` require
    ``torch``; importing them eagerly from the package ``__init__`` would
    pollute the pure engine path (E01-03 dependency isolation). They are
    imported only when explicitly requested.
    """
    if name == "make_fixed_token_input":
        from hqsb.benchmark.workload import make_fixed_token_input

        globals()[name] = make_fixed_token_input
        return make_fixed_token_input
    if name == "benchmark_model_core":
        from hqsb.benchmark.model_core import benchmark_model_core

        globals()[name] = benchmark_model_core
        return benchmark_model_core
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
