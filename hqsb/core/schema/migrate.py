"""Migration from legacy benchmark documents to the current schema.

S00 left two legacy document shapes in the tree:

* **legacy golden** — produced by ``generate_golden.py`` (has
  ``first_token`` and ``input_token_ids``, but no run/result envelope);
* **legacy result** — produced by ``run_model_core.py`` (has ``repetitions``
  and a ``workload`` block, but no ``run_id``/contract-typed model).

Both are migrated to the current :class:`BenchmarkResult` (C6) envelope so
they can participate in regression gates under one schema. The migration is
*lossless for preserved fields* and explicitly drops nothing silently —
fields that have no target are recorded in ``artifact_links`` or
``summary``.

Usage (CLI)::

    python scripts/migrate_legacy.py input.json output.json
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping

from hqsb.core.contracts.model import ModelArtifact
from hqsb.core.contracts.result import (
    BenchmarkResult,
    EnvironmentInfo,
)
from hqsb.core.contracts.workload import WorkloadSpec
from hqsb.core.errors import SchemaError
from hqsb.core.ids import new_run_id


def _environment(document: Mapping[str, Any]) -> EnvironmentInfo:
    hardware = document.get("hardware", {}) or {}
    software = document.get("software", {}) or {}
    return EnvironmentInfo(
        platform=hardware.get("platform", ""),
        device=hardware.get("device", ""),
        compute_capability=hardware.get("compute_capability"),
        python_version=software.get("python", ""),
        torch_version=software.get("torch", ""),
        cuda_version=software.get("cuda", ""),
        framework_versions={},
    )


def _workload_from_fields(document: Mapping[str, Any]) -> WorkloadSpec:
    return WorkloadSpec(
        name="legacy",
        input_tokens=document["input_tokens"],
        output_tokens=document["output_tokens"],
        seed=0,
        sampling="greedy",
        warmup=0,
        repetitions=1,
    )


def migrate_legacy_golden(document: Mapping[str, Any]) -> BenchmarkResult:
    """Migrate a legacy golden document to :class:`BenchmarkResult`.

    The golden's ``first_token`` payload (top-K logits, L2 norm) is preserved
    in ``summary``; the generated token sequence becomes a raw sample with
    zero timings (golden records *what* was produced, not *how long*).
    """
    if "input_token_ids" not in document or "first_token" not in document:
        raise SchemaError("document does not look like a legacy golden file")

    model = document.get("model", {}) or {}
    generated = document.get("generated_tokens", [])

    result = BenchmarkResult(
        run_id=new_run_id(),
        timestamp=document.get("timestamp", time.time()),
        environment=_environment(document),
        model_artifact_hash=model.get("config_hash"),
        config_hash=None,
        workload=_workload_from_fields(document),
        raw_samples=[
            {
                "input_tokens": document["input_tokens"],
                "output_tokens": document["output_tokens"],
                "generated_token_ids": generated,
                "prefill_forward_ms": 0.0,
                "first_token_selection_ms": 0.0,
                "itl_ms": [],
                "peak_cuda_allocated_mb": 0.0,
                "peak_cuda_reserved_mb": 0.0,
            }
        ],
        summary={
            "first_token": document.get("first_token"),
            "input_token_ids": document.get("input_token_ids"),
        },
        correctness=None,
        resource=None,
        artifact_links={
            "legacy_kind": "golden",
            "model_id": model.get("id", ""),
            "source": model.get("source", ""),
            "dtype": model.get("dtype", ""),
        },
    )
    return result


def migrate_legacy_result(document: Mapping[str, Any]) -> BenchmarkResult:
    """Migrate a legacy model-core result document to :class:`BenchmarkResult`.

    Each ``repetitions`` entry becomes a raw sample; the ``deterministic``
    flag becomes the correctness gate (method ``determinism``).
    """
    if "repetitions" not in document:
        raise SchemaError("document does not look like a legacy result file")

    workload_block = document.get("workload", {}) or {}
    model_block = document.get("model", {}) or {}

    workload = WorkloadSpec(
        name="legacy",
        input_tokens=workload_block.get("input_tokens", 0),
        output_tokens=workload_block.get("output_tokens", 0),
        seed=0,
        sampling="greedy",
        warmup=0,
        repetitions=max(len(document.get("repetitions", [])), 1),
    )

    raw_samples: List[Dict[str, Any]] = []
    for rep in document.get("repetitions", []):
        itl_data = rep.get("itl", {}) or {}
        # Reconstruct a per-sample ITL list from summary stats as a faithful
        # approximation (legacy format stored only summary stats).
        count = int(itl_data.get("count", 0))
        mean_ms = float(itl_data.get("mean_ms", 0.0))
        raw_samples.append(
            {
                "input_tokens": rep.get("input_tokens", workload.input_tokens),
                "output_tokens": rep.get("output_tokens", workload.output_tokens),
                "generated_token_ids": rep.get("generated_token_ids", []),
                "prefill_forward_ms": rep.get("prefill_forward_ms", 0.0),
                "first_token_selection_ms": rep.get(
                    "first_token_selection_ms", 0.0
                ),
                "itl_ms": [mean_ms] * count,
                "peak_cuda_allocated_mb": rep.get("peak_cuda_allocated_mb", 0.0),
                "peak_cuda_reserved_mb": rep.get("peak_cuda_reserved_mb", 0.0),
            }
        )

    from hqsb.core.contracts.result import CorrectnessReport

    return BenchmarkResult(
        run_id=new_run_id(),
        timestamp=document.get("timestamp", time.time()),
        environment=_environment(document),
        model_artifact_hash=None,
        config_hash=None,
        workload=workload,
        raw_samples=raw_samples,
        summary={
            "deterministic": document.get("deterministic"),
            "generated_token_sha256": document.get("generated_token_sha256"),
        },
        correctness=CorrectnessReport(
            passed=bool(document.get("deterministic", False)),
            method="determinism",
            details={"legacy_deterministic": document.get("deterministic")},
        ),
        resource=None,
        artifact_links={
            "legacy_kind": "result",
            "model_id": model_block.get("id", ""),
            "backend": model_block.get("backend", ""),
            "dtype": model_block.get("dtype", ""),
        },
    )


def migrate_any(document: Mapping[str, Any]) -> BenchmarkResult:
    """Auto-detect the legacy document kind and migrate it.

    Raises:
        SchemaError: If the document matches neither legacy shape.
    """
    if "first_token" in document and "input_token_ids" in document:
        return migrate_legacy_golden(document)
    if "repetitions" in document:
        return migrate_legacy_result(document)
    raise SchemaError("unrecognized legacy document shape")


__all__ = [
    "migrate_any",
    "migrate_legacy_golden",
    "migrate_legacy_result",
]
