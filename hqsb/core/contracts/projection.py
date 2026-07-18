"""Serialize domain evidence inside the frozen C6 extension points.

A field-coverage report is not a BenchmarkResult. This boundary makes the
conversion explicit while leaving stage-specific validation in its owner.
It neither invents raw samples nor promotes a projection to a measured claim.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from hqsb.core.contracts.result import (
    BenchmarkResult,
    CorrectnessReport,
    EnvironmentInfo,
)
from hqsb.core.errors import ConfigError


def result_from_projection(
    projection: Mapping[str, Any],
    *,
    namespace: str,
    run_id: str,
    timestamp: float,
    raw_samples: Sequence[Mapping[str, Any]] = (),
    correctness: CorrectnessReport | None = None,
    **metadata: Any,
) -> BenchmarkResult:
    """Wrap a validated ``fields/ok/problems`` report as a serializable C6.

    ``timestamp`` and identity/environment metadata are supplied by the caller;
    projection time is not substituted for measurement time. Correctness is
    absent unless the caller provides an independent correctness report.
    """
    if projection.get("ok") is not True or projection.get("problems"):
        raise ConfigError(
            "cannot export an invalid projection", details=dict(projection)
        )
    if not namespace or not run_id or not isinstance(projection.get("fields"), Mapping):
        raise ConfigError("projection export requires namespace, run_id and fields")
    if "summary" in metadata:
        raise ConfigError("summary is owned by the projection exporter")
    metadata.setdefault("environment", EnvironmentInfo())
    return BenchmarkResult(
        run_id=run_id,
        timestamp=timestamp,
        raw_samples=[dict(sample) for sample in raw_samples],
        summary={namespace: dict(projection["fields"]), "claim_level": "SOURCE"},
        correctness=correctness,
        **metadata,
    )
