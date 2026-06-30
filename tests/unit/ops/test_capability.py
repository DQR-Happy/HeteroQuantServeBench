"""Unit tests for capability detection (S04).

Verifies the detection structure, the defensive "never raises" contract, and
the negative case (CUTLASS not installed reports False rather than failing).
"""

from __future__ import annotations

import pytest

from hqsb.core.errors import BackendError, CapabilityError
from ops.capability import (
    BackendCapabilities,
    CapabilityCache,
    CapabilityCacheCorruption,
    CapabilityIdentity,
    CapabilityReason,
    CapabilityResult,
    CapabilityStage,
    detect_capabilities,
    resolve_backend,
    unavailable_result,
)


@pytest.mark.unit
class TestBackendCapabilities:
    def test_as_dict_serializable(self):
        caps = BackendCapabilities(
            cuda_available=True,
            device_capability=(8, 7),
            triton_available=True,
            triton_version="3.7.1",
            cutlass_available=False,
            cutlass_include_dir=None,
            tilelang_available=False,
            tilelang_version=None,
            cublas_available=True,
            cuda_rmsnorm_available=True,
            cuda_rmsnorm_lib="/fake/lib.so",
            notes=("note1",),
        )
        d = caps.as_dict()
        assert d["device_capability"] == [8, 7]
        assert d["notes"] == ["note1"]
        import json

        json.dumps(d)  # must not raise


@pytest.mark.unit
class TestDetectCapabilities:
    def test_returns_capabilities_without_raising(self):
        # Detection must never raise (defensive contract), even on a machine
        # missing Triton/CUTLASS/GPU.
        caps = detect_capabilities()
        assert isinstance(caps, BackendCapabilities)
        assert isinstance(caps.cuda_available, bool)
        assert isinstance(caps.triton_available, bool)
        assert isinstance(caps.cutlass_available, bool)
        assert isinstance(caps.tilelang_available, bool)
        assert isinstance(caps.cublas_available, bool)
        assert isinstance(caps.cuda_rmsnorm_available, bool)

    def test_cached(self):
        assert detect_capabilities() is detect_capabilities()

    def test_cutlass_negative_case(self):
        # Negative path: if CUTLASS is absent the detector must report False
        # and record a note (not raise). On hosts where it IS present, the
        # include dir must be non-empty and point at a real checkout.
        caps = detect_capabilities()
        if not caps.cutlass_available:
            assert any("CUTLASS" in n for n in caps.notes)
        else:
            assert caps.cutlass_include_dir
            assert "cutlass" in caps.cutlass_include_dir

    def test_tilelang_negative_case(self):
        # Same defensive contract for TileLang: absent -> False + note.
        caps = detect_capabilities()
        if not caps.tilelang_available:
            assert any("TileLang" in n for n in caps.notes)

    def test_cuda_lib_build_arch_is_exposed_or_explained(self):
        # The dispatcher gates on the library's *real* build arch, so either
        # the arch is reported or a note explains why it is not -- never a
        # silent hard-coded assumption.
        caps = detect_capabilities()
        if not caps.cuda_rmsnorm_available:
            return
        if caps.cuda_rmsnorm_build_arch is None:
            assert any("build arch" in n for n in caps.notes)
        else:
            major, minor = caps.cuda_rmsnorm_build_arch
            assert major >= 1 and 0 <= minor <= 9


def _identity(**overrides):
    values = dict(
        device_identity="gpu-uuid-1",
        arch=(8, 7),
        package_version="3.7.1",
        runtime_version="12.6",
        compiler_version="12.6.85",
        build_identity="build-a",
    )
    values.update(overrides)
    return CapabilityIdentity(**values)


def _failure(reason, stage, *, retryable=False, identity=None):
    return unavailable_result(
        "triton",
        identity or _identity(),
        stage,
        reason,
        "injected failure",
        retryable=retryable,
        cause_chain=("InjectedFault",),
    )


@pytest.mark.unit
class TestStructuredCapability:
    def test_reason_roundtrip_and_exit_mapping(self):
        unsupported = _failure(
            CapabilityReason.ARCH_UNSUPPORTED, CapabilityStage.DEVICE
        )
        runtime = _failure(
            CapabilityReason.RUNTIME_FAILED,
            CapabilityStage.EXECUTE,
            retryable=True,
        )
        assert unsupported.as_dict()["reason_code"] == "ARCH_UNSUPPORTED"
        assert unsupported.exit_code == 7
        assert runtime.exit_code == 6
        assert runtime.retryable

    def test_auto_fallback_records_all_candidates(self):
        triton = _failure(
            CapabilityReason.PACKAGE_NOT_INSTALLED, CapabilityStage.DISCOVERY
        )
        cutlass = unavailable_result(
            "cutlass",
            _identity(package_version="4.7.0"),
            CapabilityStage.DEVICE,
            CapabilityReason.ARCH_UNSUPPORTED,
            "sm_87 excluded by injected policy",
            retryable=False,
        )
        decision = resolve_backend("auto", [triton, cutlass])
        assert decision.actual == "reference"
        assert decision.fallback
        assert [r.backend for r in decision.candidate_results] == ["triton", "cutlass"]

    def test_forced_never_silently_falls_back(self):
        deterministic = _failure(
            CapabilityReason.VERSION_INCOMPATIBLE, CapabilityStage.VERSION
        )
        with pytest.raises(CapabilityError) as unsupported:
            resolve_backend("triton", [deterministic])
        assert unsupported.value.exit_code == 7
        operational = _failure(
            CapabilityReason.COMPILE_FAILED, CapabilityStage.COMPILE, retryable=True
        )
        with pytest.raises(BackendError) as failed:
            resolve_backend("triton", [operational])
        assert failed.value.exit_code == 6


@pytest.mark.unit
class TestCapabilityCache:
    def test_deterministic_failure_cached_but_transient_retried(self):
        identity = _identity()
        cache = CapabilityCache()
        calls = {"n": 0}

        def deterministic_probe():
            calls["n"] += 1
            return _failure(
                CapabilityReason.VERSION_INCOMPATIBLE,
                CapabilityStage.VERSION,
                identity=identity,
            )

        assert not cache.get_or_probe("triton", identity, deterministic_probe).from_cache
        assert cache.get_or_probe("triton", identity, deterministic_probe).from_cache
        assert calls["n"] == 1

        cache.clear()
        calls["n"] = 0

        def transient_probe():
            calls["n"] += 1
            return _failure(
                CapabilityReason.TIMEOUT,
                CapabilityStage.COMPILE,
                retryable=True,
                identity=identity,
            )

        cache.get_or_probe("triton", identity, transient_probe)
        cache.get_or_probe("triton", identity, transient_probe)
        assert calls["n"] == 2

    def test_identity_change_invalidates_and_corruption_detected(self):
        cache = CapabilityCache()
        calls = {"n": 0}

        def probe(identity):
            def run():
                calls["n"] += 1
                return _failure(
                    CapabilityReason.ARCH_UNSUPPORTED,
                    CapabilityStage.DEVICE,
                    identity=identity,
                )

            return run

        first = _identity()
        changed = _identity(build_identity="build-b")
        cache.get_or_probe("triton", first, probe(first))
        cache.get_or_probe("triton", changed, probe(changed))
        assert calls["n"] == 2
        payload = cache.to_payload()
        CapabilityCache.validate_payload(payload)
        payload["values"][next(iter(payload["values"]))]["detail"] = "tampered"
        with pytest.raises(CapabilityCacheCorruption):
            CapabilityCache.validate_payload(payload)

    def test_concurrent_probe_is_single_flight(self):
        import concurrent.futures
        import time

        cache = CapabilityCache()
        identity = _identity()
        calls = {"n": 0}

        def probe():
            calls["n"] += 1
            time.sleep(0.02)
            return CapabilityResult(
                backend="triton",
                available=True,
                stage=CapabilityStage.EXECUTE,
                reason_code=CapabilityReason.AVAILABLE,
                detail="ok",
                retryable=False,
                identity=identity,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda _: cache.get_or_probe("triton", identity, probe), range(8)
                )
            )
        assert calls["n"] == 1
        assert sum(result.from_cache for result in results) == 7
