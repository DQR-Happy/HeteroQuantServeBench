"""HQSB Benchmark module.

Provides model-core benchmark, workload generation, metrics computation,
resource monitoring (tegrastats), and power/energy analysis for LLM
inference on Jetson and datacenter platforms.
"""

from hqsb.benchmark.metrics import latency_summary, percentile
from hqsb.benchmark.workload import make_fixed_token_input
from hqsb.benchmark.model_core import benchmark_model_core

__all__ = [
    "latency_summary",
    "percentile",
    "make_fixed_token_input",
    "benchmark_model_core",
]
