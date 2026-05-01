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
from hqsb.benchmark.workload import make_fixed_token_input
from hqsb.benchmark.model_core import benchmark_model_core
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
