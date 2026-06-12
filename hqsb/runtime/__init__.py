"""HQSB inference runtime layer (S07).

The package owns request state, KV management, batching/scheduling, prefix
reuse, graph/attention routing, speculative decoding, failure handling and the
runtime side of the backend contract.  It is the layer **above**
``hqsb.integration`` and **below** ``hqsb.serving`` in the dependency graph:

    core ← models/benchmark ← backends/integration/quant ← runtime ← serving

Design rules (see ``docs/architecture/module_ownership.md``):

* no module-level ``torch``/``triton``/``numpy`` import — heavy engines are
  probed and imported lazily, and a missing engine is a *structured* reason,
  never an ``ImportError``;
* ``ops`` is never imported (kernels are addressed through capability and
  provider names, exactly as in ``hqsb.integration``);
* nothing below this layer may import it (gate rule R3).

Everything here is interface/scaffolding code.  Experiment execution is gated by
:mod:`hqsb.runtime.experiment` and is currently ``BLOCKED`` (see the S07
development report); no module in this package produces a result on its own.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "STAGE",
    "__version__",
]

#: The stage this package implements (protocol statuses live in ``experiment``).
STAGE = "S07"

__version__ = "0.1.0"

#: Public API re-exported lazily so that ``import hqsb.runtime`` stays cheap and
#: free of optional dependencies (mirrors ``hqsb.integration``).
_LAZY: dict[str, str] = {
    "adapter": "hqsb.runtime.adapter",
    "comparison": "hqsb.runtime.comparison",
    "experiment": "hqsb.runtime.experiment",
    "failure": "hqsb.runtime.failure",
    "graph_route": "hqsb.runtime.graph_route",
    "interface_map": "hqsb.runtime.interface_map",
    "kv": "hqsb.runtime.kv",
    "metrics": "hqsb.runtime.metrics",
    "parity": "hqsb.runtime.parity",
    "policy_ab": "hqsb.runtime.policy_ab",
    "prefix_cache": "hqsb.runtime.prefix_cache",
    "request": "hqsb.runtime.request",
    "scheduler": "hqsb.runtime.scheduler",
    "spec_decode": "hqsb.runtime.spec_decode",
    "specs": "hqsb.runtime.specs",
    "telemetry": "hqsb.runtime.telemetry",
    "trace": "hqsb.runtime.trace",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy loader
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    globals()[name] = module
    return module


def __dir__() -> list[str]:  # pragma: no cover - introspection helper
    return sorted(set(__all__) | set(_LAZY))
