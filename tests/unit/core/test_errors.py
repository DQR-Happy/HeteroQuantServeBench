"""Unit tests for the unified error taxonomy and exit codes."""

from __future__ import annotations

import pytest

from hqsb.core.errors import (
    ArtifactError,
    BackendError,
    BenchmarkError,
    CapabilityError,
    ConfigError,
    ExitCode,
    HqsbError,
    RegistryError,
    SchemaError,
    UsageError,
    exit_code_for,
)


@pytest.mark.unit
class TestExitCodes:
    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (UsageError("x"), ExitCode.USAGE),
            (ConfigError("x"), ExitCode.CONFIG),
            (SchemaError("x"), ExitCode.SCHEMA),
            (RegistryError("x"), ExitCode.REGISTRY),
            (BackendError("x"), ExitCode.BACKEND),
            (CapabilityError("x"), ExitCode.CAPABILITY),
            (ArtifactError("x"), ExitCode.ARTIFACT),
            (BenchmarkError("x"), ExitCode.BENCHMARK),
        ],
    )
    def test_maps_to_stable_code(self, exc, expected):
        assert exc.exit_code == expected
        assert exit_code_for(exc) == expected

    def test_unknown_exception_maps_to_internal(self):
        assert exit_code_for(RuntimeError("boom")) == ExitCode.INTERNAL

    def test_base_hqsb_error_is_internal(self):
        assert exit_code_for(HqsbError("generic")) == ExitCode.INTERNAL

    def test_exit_codes_are_unique(self):
        codes = [
            ExitCode.USAGE,
            ExitCode.CONFIG,
            ExitCode.SCHEMA,
            ExitCode.REGISTRY,
            ExitCode.BACKEND,
            ExitCode.CAPABILITY,
            ExitCode.ARTIFACT,
            ExitCode.BENCHMARK,
        ]
        assert len(codes) == len(set(codes))


@pytest.mark.unit
class TestErrorDetails:
    def test_error_carries_structured_details(self):
        err = CapabilityError(
            "unsupported", details={"requested": "int4", "supported": []}
        )
        assert err.details == {"requested": "int4", "supported": []}

    def test_error_message_is_str(self):
        err = ConfigError("bad config")
        assert str(err) == "bad config"
