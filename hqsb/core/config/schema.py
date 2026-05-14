"""Formal S01 experiment configuration schema (E01-02).

This module freezes the *official* benchmark experiment configuration model
that the unified :class:`~hqsb.core.config.loader.ConfigLoader` validates and
resolves.  It mirrors the real repository config files under ``configs/``
(``configs/benchmarks/jetson_qwen3_fp16.yaml``), so the existing legal YAML is
a regression fixture, while adding the field categories E01-02 must prove:

* ``benchmark`` — experiment-semantic execution parameters (model identity,
  dtype, backend, batch size, warmup switch, repetition count);
* ``workloads`` — the C2 :class:`WorkloadSpec` suite (six official cases);
* ``secrets`` — credential *references* held separately from experiment
  semantics; they are redacted from every public view and excluded from the
  semantic identity hash;
* ``run`` — operational parameters (output dir, log level) that do not change
  execution semantics and are excluded from the semantic hash.

Identity rules (E01-02 protocol §7.3):

* ``schema_version``, ``benchmark.*`` and ``workloads[*].*`` are part of the
  semantic payload that feeds :func:`config_identity_hash`.
* ``run.*`` and ``secrets.*`` are **not** part of the semantic identity.
* ``run.run_id`` is an audit field; recording it never changes the hash.
* Secret plaintext never appears in the public/redacted view, in the semantic
  payload, in logs, or in raised error messages.

The schema declares its own version and rejects unknown fields everywhere
(``extra="forbid"``), which is what makes every provided source auditable.
"""

from __future__ import annotations

from typing import ClassVar, Dict, List, Literal, Optional, Tuple

from pydantic import Field, field_validator, model_validator

from hqsb.core.contracts.base import VersionedModel
from hqsb.core.contracts.workload import WorkloadSpec
from hqsb.core.errors import ConfigError

__all__ = [
    "BenchmarkConfig",
    "BenchmarkSection",
    "OPERATIONAL_FIELD_PATHS",
    "RunSection",
    "SECRET_FIELD_PATHS",
    "SecretsSection",
    "cross_field_validate",
    "workload_names_duplicates",
]


#: Dot paths of sensitive credential fields.  Loader redacts these paths and
#: never lets their plaintext reach public views, hashes, or error messages.
SECRET_FIELD_PATHS: ClassVar[Tuple[str, ...]] = (
    "secrets.modelscope_token",
    "secrets.hf_token",
)

#: Dot paths of operational (non-semantic) fields.  These are excluded from
#: the semantic identity payload but still recorded in the resolved snapshot.
OPERATIONAL_FIELD_PATHS: ClassVar[Tuple[str, ...]] = (
    "run.run_id",
    "run.output_dir",
    "run.log_level",
    "secrets",
)


class BenchmarkSection(VersionedModel):
    """Execution-level experiment semantics (mirrors ``configs/benchmarks``)."""

    SCHEMA_VERSION = "1.0.0"

    model: str = Field(..., description="Model identifier, e.g. 'Qwen/Qwen3-1.7B'.")
    model_source: Literal["modelscope", "huggingface"] = Field(
        ..., description="Model artifact source."
    )
    backend: str = Field(..., description="Requested execution backend.")
    dtype: str = Field(..., description="Weight/compute dtype, e.g. 'float16'.")
    attention_backend: str = Field("eager", description="Attention implementation.")
    batch_size: int = Field(1, ge=1, description="Inference batch size.")
    warmup: bool = Field(True, description="Global warmup switch for the run.")
    repetitions: int = Field(1, ge=1, description="Timed repetitions per case.")


class SecretsSection(VersionedModel):
    """Credential references used by artifact download / service access.

    E01-02 only ever exercises these fields with synthetic sentinel values.
    The plaintext never leaves a controlled channel and never appears in any
    public view, error, log, or report produced by the experiment.
    """

    SCHEMA_VERSION = "1.0.0"

    modelscope_token: Optional[str] = Field(
        default=None, description="ModelScope access token (secret reference)."
    )
    hf_token: Optional[str] = Field(
        default=None, description="Hugging Face access token (secret reference)."
    )


class RunSection(VersionedModel):
    """Operational, non-semantic run metadata."""

    SCHEMA_VERSION = "1.0.0"

    run_id: str = Field("", description="Audit run identifier.")
    output_dir: str = Field("reports/dev", description="Output directory.")
    log_level: str = Field("INFO", description="Console log level.")


class BenchmarkConfig(VersionedModel):
    """Formal S01 benchmark experiment configuration document.

    The semantic identity covers ``schema_version``, ``benchmark`` and
    ``workloads`` only.  ``run`` and ``secrets`` are excluded (see
    :data:`OPERATIONAL_FIELD_PATHS`), and secret values are redacted by the
    loader before any public serialization.
    """

    SCHEMA_VERSION = "1.0.0"

    #: ClassVar mirrors so a generic loader can discover the identity rules.
    SECRET_FIELD_PATHS: ClassVar[Tuple[str, ...]] = SECRET_FIELD_PATHS
    OPERATIONAL_FIELD_PATHS: ClassVar[Tuple[str, ...]] = OPERATIONAL_FIELD_PATHS

    benchmark: BenchmarkSection = Field(
        ..., description="Execution-level experiment semantics."
    )
    workloads: List[WorkloadSpec] = Field(
        ...,
        min_length=1,
        description="C2 workload suite (six official cases in real config).",
    )
    secrets: SecretsSection = Field(
        default_factory=SecretsSection,
        description="Credential references (redacted, never semantic).",
    )
    run: RunSection = Field(
        default_factory=RunSection,
        description="Operational metadata (excluded from semantic identity).",
    )

    @field_validator("workloads")
    @classmethod
    def _workload_names_unique(cls, value: List[WorkloadSpec]) -> List[WorkloadSpec]:
        """Reject duplicate case names (E01-02 step 4).

        A repeated name would make the resolved case ambiguous, so it is a
        structural error at the config boundary instead of a silent override.
        """
        dup = workload_names_duplicates([w.name for w in value])
        if dup is not None:
            raise ValueError(f"duplicate workload case name {dup!r} in workloads")
        return value

    @model_validator(mode="after")
    def _cross_field_checks(self) -> "BenchmarkConfig":
        """Run final-document semantic checks on the fully merged object."""
        cross_field_validate(self)
        return self


def workload_names_duplicates(names: List[str]) -> Optional[str]:
    """Return the first duplicate workload case name, or ``None``."""
    seen: Dict[str, None] = {}
    for name in names:
        if name in seen:
            return name
        seen[name] = None
    return None


def cross_field_validate(config: "BenchmarkConfig") -> None:
    """Final merged-document semantic checks (protocol step 5/6 table).

    These checks intentionally run *after* the complete merged object exists:
    a per-source overlay is allowed to carry partial fields, but the fully
    merged request must be internally consistent.  Violations are reported as
    :class:`ConfigError` with a stable error code and field-localized message.

    The frozen cross-field rule here is: a workload case named
    ``decode_heavy`` must actually be decode-bound (OSL > ISL).  The name
    describes a workload family, not an arbitrary label, so contradicting it
    is an internally-inconsistent request rather than a local type error.
    """
    by_name = {w.name: i for i, w in enumerate(config.workloads)}
    for spec in config.workloads:
        index = by_name[spec.name]
        if spec.name == "decode_heavy" and spec.output_tokens <= spec.input_tokens:
            raise ConfigError(
                f"workload {spec.name!r} must be decode-bound "
                f"(output_tokens > input_tokens); got OSL={spec.output_tokens} "
                f"ISL={spec.input_tokens} (field path: workloads[{index}])",
                details={
                    "error_code": "cross_field_conflict",
                    "source": "merged",
                    "field_path": f"workloads[{index}]",
                },
            )


__all__ = [
    "BenchmarkConfig",
    "BenchmarkSection",
    "OPERATIONAL_FIELD_PATHS",
    "RunSection",
    "SECRET_FIELD_PATHS",
    "SecretsSection",
    "cross_field_validate",
    "workload_names_duplicates",
]
