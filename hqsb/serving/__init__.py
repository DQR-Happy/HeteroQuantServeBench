"""HQSB ServeFabric: protocol, gateway, policy and evidence planes (S08).

The package sits **above** :mod:`hqsb.runtime` and **below** nothing in the
Python dependency graph:

    core ← models/benchmark ← backends/integration/quant ← runtime ← serving

Layering (details README §6) — dependencies point downwards only:

* **Protocol plane** — ``protocol`` (frozen subset, error catalog, canonical
  request path), ``sse`` (raw bytes and framing);
* **Gateway plane** — ``gateway`` (request state machine, streaming, cancel,
  drain), ``transport`` / ``transport_http`` (connection and backpressure),
  ``timing`` and ``pipeline`` (time boundaries and the delivery ledger);
* **Policy plane** — ``admission``, ``policies``, ``fairness``, ``router``,
  ``circuit``, ``cache_routing``, ``slo``, ``arrival``, ``loadgen``, ``clients``;
* **Backend plane** — the S07 Backend Contract only (``dummy_backend`` is the
  model-free fixture used to prove protocol behaviour);
* **Evidence plane** — ``observability``, ``telemetry``, ``faults``,
  ``service_ab``, ``experiment``, ``specs``, ``interface_map``.

Design rules (see ``docs/architecture/module_ownership.md``):

* no module-level ``torch``/``triton``/``numpy`` import — the service core runs
  on the CPU-minimal installation;
* ``ops`` is never imported (kernels are addressed through capability names);
* runtime/contracts are consumed through their public objects, never through a
  private engine SDK.

Everything here is interface/scaffolding code.  Experiment execution is gated by
:mod:`hqsb.serving.experiment` and is currently ``BLOCKED`` (see the S08
development report); no module in this package produces a result on its own.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "STAGE",
    "__version__",
]

#: The stage this package implements (protocol statuses live in ``experiment``).
STAGE = "S08"

__version__ = "0.1.0"

#: Public API re-exported lazily so that ``import hqsb.serving`` stays cheap and
#: free of optional dependencies (mirrors ``hqsb.runtime``).
_LAZY: dict[str, str] = {
    "admission": "hqsb.serving.admission",
    "arrival": "hqsb.serving.arrival",
    "cache_routing": "hqsb.serving.cache_routing",
    "circuit": "hqsb.serving.circuit",
    "clients": "hqsb.serving.clients",
    "dummy_backend": "hqsb.serving.dummy_backend",
    "experiment": "hqsb.serving.experiment",
    "fairness": "hqsb.serving.fairness",
    "faults": "hqsb.serving.faults",
    "gateway": "hqsb.serving.gateway",
    "interface_map": "hqsb.serving.interface_map",
    "loadgen": "hqsb.serving.loadgen",
    "observability": "hqsb.serving.observability",
    "pipeline": "hqsb.serving.pipeline",
    "policies": "hqsb.serving.policies",
    "protocol": "hqsb.serving.protocol",
    "router": "hqsb.serving.router",
    "service_ab": "hqsb.serving.service_ab",
    "slo": "hqsb.serving.slo",
    "specs": "hqsb.serving.specs",
    "sse": "hqsb.serving.sse",
    "telemetry": "hqsb.serving.telemetry",
    "timing": "hqsb.serving.timing",
    "transport": "hqsb.serving.transport",
    "transport_http": "hqsb.serving.transport_http",
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
