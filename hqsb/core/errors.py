"""Unified error taxonomy and process exit codes for HQSB.

Every public failure path in HQSB raises a subclass of :class:`HqsbError`
so callers can catch, classify, and map errors to stable process exit
codes without string matching. The taxonomy mirrors the module boundaries
defined in the top-level architecture (contracts, config, registry,
backend, benchmark) so that errors are diagnosable by their origin.

Exit code policy (0-99 reserved by HQSB; 100+ free for application use):

========  ==============================================================
code      meaning
========  ==============================================================
0         success
1         generic / internal error (unexpected)
2         usage / CLI argument error
3         configuration error (invalid value, unknown field, bad file)
4         contract / schema validation error (incl. version rejection)
5         registry error (duplicate, conflict, missing, unload)
6         backend / runtime error
7         capability not supported (structured fallback expected)
8         model artifact error (missing file, hash mismatch)
9         benchmark execution error
========  ==============================================================
"""

from __future__ import annotations

from typing import Optional


class ExitCode:
    """Canonical process exit codes reserved by HQSB."""

    SUCCESS = 0
    INTERNAL = 1
    USAGE = 2
    CONFIG = 3
    SCHEMA = 4
    REGISTRY = 5
    BACKEND = 6
    CAPABILITY = 7
    ARTIFACT = 8
    BENCHMARK = 9


class HqsbError(Exception):
    """Base class for all HQSB errors.

    Attributes:
        exit_code: Process exit code associated with this error class.
        message: Human-readable error message.
        details: Optional structured context (dict) attached to the error,
            useful for machine-consumed diagnostics without parsing text.
    """

    exit_code: int = ExitCode.INTERNAL

    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class UsageError(HqsbError):
    """Invalid command-line usage or argument value."""

    exit_code = ExitCode.USAGE


class ConfigError(HqsbError):
    """Configuration is missing, malformed, or violates the schema."""

    exit_code = ExitCode.CONFIG


class SchemaError(HqsbError):
    """A versioned schema document failed validation or migration."""

    exit_code = ExitCode.SCHEMA


class SchemaVersionError(SchemaError):
    """A schema document has a missing, malformed, or future version."""


class UnsupportedSchemaVersionError(SchemaVersionError):
    """A payload declares a schema version newer than this implementation.

    Refusing forward-incompatible input is intentional: guessing a future
    schema would silently reinterpret fields the current code cannot know.
    """


class SchemaMigrationRequiredError(SchemaVersionError):
    """A payload declares an older schema version and must be migrated.

    The payload is refused until a registered migration lifts it to the
    current schema version; no field is silently dropped or default-filled.
    """


class RegistryError(HqsbError):
    """A registry operation failed (duplicate, conflict, missing, unload)."""

    exit_code = ExitCode.REGISTRY


class DuplicateRegistrationError(RegistryError):
    """An entry with the same name (and incompatible version) already exists."""


class RegistryLookupError(RegistryError):
    """A requested entry is not present in the registry."""


class BackendError(HqsbError):
    """A backend failed during its lifecycle (load/warmup/generate/...)."""

    exit_code = ExitCode.BACKEND


class CapabilityError(HqsbError):
    """A requested capability is not supported by a backend/operator.

    This is a *structured* fallback signal: it should carry enough
    ``details`` for callers to select an alternative implementation
    instead of silently degrading a benchmark.
    """

    exit_code = ExitCode.CAPABILITY


class ArtifactError(HqsbError):
    """A model artifact is missing, corrupted, or fails hash verification."""

    exit_code = ExitCode.ARTIFACT


class BenchmarkError(HqsbError):
    """A benchmark run failed to execute or produce a valid result."""

    exit_code = ExitCode.BENCHMARK


_EXIT_CODE_TO_CLASS = {
    cls.exit_code: cls
    for cls in (
        UsageError,
        ConfigError,
        SchemaError,
        RegistryError,
        BackendError,
        CapabilityError,
        ArtifactError,
        BenchmarkError,
    )
}


def exit_code_for(exc: BaseException) -> int:
    """Return the stable exit code for an exception.

    Any :class:`HqsbError` maps to its class ``exit_code``; everything else
    maps to :data:`ExitCode.INTERNAL`.
    """
    if isinstance(exc, HqsbError):
        return exc.exit_code
    return ExitCode.INTERNAL
