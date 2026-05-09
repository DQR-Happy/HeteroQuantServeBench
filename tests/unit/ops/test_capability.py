"""Unit tests for capability detection (S04).

Verifies the detection structure, the defensive "never raises" contract, and
the negative case (CUTLASS not installed reports False rather than failing).
"""

from __future__ import annotations

import pytest

from ops.capability import BackendCapabilities, detect_capabilities


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
