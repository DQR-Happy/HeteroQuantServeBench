"""C1 — ModelArtifact contract.

Describes a fully-materialized, hash-pinned model snapshot. A benchmark
result references a :class:`ModelArtifact` so that its correctness and
performance can always be traced back to the exact weights, tokenizer, and
configuration that produced it (top-level architecture §5 C1).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Optional

from pydantic import Field, field_validator

from hqsb.core.artifact_path import validate_relative_paths
from hqsb.core.contracts.base import VersionedModel

# Lowercase 64-character hex, as emitted by ``sha256sum`` / ``hashlib``.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Fields that define an artifact's *identity*: two artifacts with equal
#: values for all of these describe the same weights, so they must hash
#: identically. Deliberately excludes provenance-only fields such as
#: ``tool_version`` or ``license``, which may change without changing the
#: bytes a benchmark actually reads.
_IDENTITY_FIELDS = (
    "model_id",
    "source",
    "revision",
    "architecture",
    "dtype",
    "file_hashes",
)


class ModelArtifact(VersionedModel):
    """A versioned, hash-pinned description of a model snapshot.

    Construction is itself an integrity gate (E00-02): ``file_hashes`` keys
    must be safe, normalized, unique relative paths and values must be
    lowercase 64-character SHA256 digests. Unsafe artifacts are therefore
    rejected at parse time, long before any loader touches the filesystem.
    """

    SCHEMA_VERSION = "1.0.0"

    model_id: str = Field(..., description="Model identifier, e.g. 'Qwen/Qwen3-1.7B'.")
    source: str = Field(
        ...,
        description="Model source: 'modelscope', 'huggingface', or 'local'.",
    )
    revision: Optional[str] = Field(
        default=None,
        description="Upstream revision/tag/commit, when applicable.",
    )
    file_hashes: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from relative file path to lowercase SHA256 hex.",
    )
    architecture: str = Field(
        ...,
        description="Model architecture family, e.g. 'Qwen3ForCausalLM'.",
    )
    dtype: str = Field(
        ...,
        description="Weight precision, e.g. 'float16', 'bfloat16', 'float32'.",
    )
    quantization: Optional[str] = Field(
        default=None,
        description="Quantization scheme, if any (e.g. 'int4-gptq').",
    )
    layout: str = Field(
        default="dense",
        description="Parameter layout ('dense', 'moe', ...).",
    )
    context_length: Optional[int] = Field(
        default=None,
        description="Maximum supported context length in tokens.",
    )
    license: Optional[str] = Field(
        default=None,
        description="Model license identifier or URL.",
    )
    tool_version: Optional[str] = Field(
        default=None,
        description="Version of the tool/loader that produced this artifact.",
    )

    @field_validator("file_hashes")
    @classmethod
    def _validate_file_hashes(cls, value: Dict[str, str]) -> Dict[str, str]:
        """Reject unsafe paths and non-digest values in ``file_hashes``.

        Raises:
            ValueError: On an empty/absolute/traversing/non-normalized/
                duplicate path, or a value that is not a lowercase
                64-character SHA256 hex digest.
        """
        if not value:
            return value

        problems = validate_relative_paths(value.keys())
        if problems:
            path, (reason, message) = next(iter(problems.items()))
            raise ValueError(
                f"ModelArtifact.file_hashes: unsafe path {path!r} "
                f"[{reason}]: {message}"
            )

        for path, digest in value.items():
            if not isinstance(digest, str) or not _SHA256_RE.match(digest):
                raise ValueError(
                    f"ModelArtifact.file_hashes[{path!r}]: expected a "
                    f"lowercase 64-character SHA256 hex digest, got "
                    f"{digest!r}"
                )
        return value

    # ── Identity ────────────────────────────────────────────────────────

    def identity_payload(self) -> Dict[str, Any]:
        """Return the canonical, JSON-serializable identity mapping."""
        return {name: getattr(self, name) for name in _IDENTITY_FIELDS}

    def identity_hash(self) -> str:
        """Return the stable SHA256 of this artifact's identity.

        The digest is computed over a canonical JSON encoding
        (sorted keys, compact separators) of the identity fields, so the
        same artifact yields the same digest across processes, machines,
        and dict insertion orders.

        Returns:
            Lowercase 64-character hexadecimal SHA256 digest.
        """
        canonical = json.dumps(
            self.identity_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["ModelArtifact"]
