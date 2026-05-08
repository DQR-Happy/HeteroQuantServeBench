"""Workload YAML single-source-of-truth loading.

The six benchmark workloads are defined *only* in
``configs/benchmarks/jetson_qwen3_fp16.yaml``. This module loads that file
and validates each entry against :class:`WorkloadSpec`, so the orchestrator,
golden generator, and profiler all consume the same definitions and can
never drift from one another (S02 execution step 5).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import yaml

from hqsb.core.contracts.workload import WorkloadSpec
from hqsb.core.errors import ConfigError


def load_workload_specs(path: str) -> List[WorkloadSpec]:
    """Load and validate the ``workloads`` list from a benchmark YAML file.

    Args:
        path: Path to a YAML file with a top-level ``workloads:`` list of
            mappings, each containing at least ``name``, ``input_tokens``,
            and ``output_tokens``.

    Returns:
        A list of validated :class:`WorkloadSpec` instances in file order.

    Raises:
        ConfigError: If the file is missing/malformed, ``workloads`` is not
            a list, or any entry fails :class:`WorkloadSpec` validation.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            document = yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigError(f"workload config file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(document, Mapping):
        raise ConfigError(f"workload config {path} must be a mapping")

    raw_workloads = document.get("workloads")
    if not isinstance(raw_workloads, list) or not raw_workloads:
        raise ConfigError(
            f"workload config {path} must contain a non-empty 'workloads' list"
        )

    specs: List[WorkloadSpec] = []
    for index, entry in enumerate(raw_workloads):
        if not isinstance(entry, Mapping):
            raise ConfigError(
                f"workloads[{index}] in {path} must be a mapping"
            )
        try:
            specs.append(WorkloadSpec.model_validate(dict(entry)))
        except Exception as exc:
            raise ConfigError(
                f"workloads[{index}] in {path} is invalid: {exc}"
            ) from exc
    return specs


def workload_specs_by_name(
    specs: Sequence[WorkloadSpec],
) -> Dict[str, WorkloadSpec]:
    """Index a list of workload specs by name (for lookups in reports)."""
    return {spec.name: spec for spec in specs}


def load_workload_dicts(path: str) -> List[Dict[str, Any]]:
    """Return the raw workload dicts (unvalidated) for lightweight readers.

    Used by scripts that only need ``input_tokens``/``output_tokens`` and do
    not need the full :class:`WorkloadSpec` validation.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            document = yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigError(f"workload config file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    workloads = document.get("workloads", [])
    if not isinstance(workloads, list):
        raise ConfigError(f"workload config {path}: 'workloads' must be a list")
    return [dict(w) for w in workloads]


__all__ = [
    "load_workload_dicts",
    "load_workload_specs",
    "workload_specs_by_name",
]
