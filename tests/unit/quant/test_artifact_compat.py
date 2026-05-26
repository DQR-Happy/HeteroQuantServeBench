"""Tests for QuantArtifact persistence, compatibility and fault injection."""

from __future__ import annotations

import json
import os

import pytest

from hqsb.core.errors import ArtifactError, ConfigError
from hqsb.quant import compat
from hqsb.quant.artifact import (
    MANIFEST_FILENAME,
    ModelIdentity,
    load_document,
    validate_artifact_dir,
    verify_variant_against_canonical,
)
from hqsb.quant.fixtures import build_tiny_artifact, save_golden_artifacts
from hqsb.quant.packing import LAYOUT_W4A16_ROWMAJOR_NK_V1


def _save_doc(tmp, **kwargs):
    doc = build_tiny_artifact(**kwargs)
    path = os.path.join(tmp, "artifact")
    doc.save(path)
    return doc, path


@pytest.mark.unit
class TestPersistence:
    def test_round_trip_identity_stability(self, tmp_path):
        doc, path = _save_doc(str(tmp_path))
        loaded = load_document(path)
        assert loaded.identity_hash() == doc.identity_hash()
        assert loaded.canonical_hash() == doc.canonical_hash()
        assert loaded.values == doc.values
        assert loaded.scheme.scheme_hash() == doc.scheme.scheme_hash()

    def test_validate_accepts_golden(self, tmp_path):
        _doc, path = _save_doc(str(tmp_path))
        report = validate_artifact_dir(path)
        assert report["accepted"] is True
        assert [stage["stage"] for stage in report["stages"]] == [
            "schema",
            "model",
            "quant",
            "pack",
        ]

    def test_save_refuses_overwrite(self, tmp_path):
        doc = build_tiny_artifact()
        path = os.path.join(str(tmp_path), "artifact")
        doc.save(path)
        with pytest.raises(ArtifactError, match="overwrite"):
            doc.save(path)

    def test_save_is_atomic_and_creates_manifest(self, tmp_path):
        _doc, path = _save_doc(str(tmp_path))
        assert os.path.isfile(os.path.join(path, MANIFEST_FILENAME))
        manifest = json.load(open(os.path.join(path, MANIFEST_FILENAME)))
        assert manifest["kind"].startswith("hqsb.quant")
        assert manifest["canonical_hash"]

    def test_fp16_equivalent_and_size_breakdown(self, tmp_path):
        doc, path = _save_doc(str(tmp_path))
        breakdown = doc.size_breakdown()
        assert breakdown.canonical_total > 0
        assert breakdown.packed_variant_bytes > 0
        assert doc.fp16_equivalent_bytes() > breakdown.canonical_q_bytes

    def test_to_c5_projection(self, tmp_path):
        doc, path = _save_doc(str(tmp_path))
        c5 = doc.to_c5()
        assert c5.bits == doc.scheme.bits
        assert c5.algorithm == doc.method
        assert c5.granularity == doc.scheme.granularity
        assert c5.scale.startswith("sha256:")
        assert c5.SCHEMA_VERSION == "1.0.0"

    def test_variant_matches_canonical(self, tmp_path):
        doc, path = _save_doc(str(tmp_path))
        loaded = load_document(path)
        for record in loaded.variants:
            report = verify_variant_against_canonical(loaded, record)
            assert report["values_checked"] == len(loaded.values)


@pytest.mark.unit
class TestNegativeArtifact:
    def test_missing_required_field(self, tmp_path):
        _doc, path = _save_doc(str(tmp_path))
        manifest = json.load(open(os.path.join(path, MANIFEST_FILENAME)))
        del manifest["canonical_count"]
        json.dump(manifest, open(os.path.join(path, MANIFEST_FILENAME), "w"))
        report = validate_artifact_dir(path)
        assert report["accepted"] is False
        assert report["reason_code"] == "manifest_required_field_invalid"

    def test_unknown_scheme_field(self, tmp_path):
        _doc, path = _save_doc(str(tmp_path))
        manifest = json.load(open(os.path.join(path, MANIFEST_FILENAME)))
        manifest["scheme"]["mystery"] = 1
        json.dump(manifest, open(os.path.join(path, MANIFEST_FILENAME), "w"))
        report = validate_artifact_dir(path)
        assert report["accepted"] is False

    def test_payload_bitflip_detected(self, tmp_path):
        _doc, path = _save_doc(str(tmp_path))
        with open(os.path.join(path, "canonical.bin"), "rb") as handle:
            data = bytearray(handle.read())
        data[0] ^= 0x01
        with open(os.path.join(path, "canonical.bin"), "wb") as handle:
            handle.write(bytes(data))
        report = validate_artifact_dir(path)
        assert report["accepted"] is False
        assert report["reason_code"] == "payload_hash_mismatch"

    def test_symmetric_artifact_with_zeros_refused(self, tmp_path):
        _doc, path = _save_doc(str(tmp_path))
        manifest = json.load(open(os.path.join(path, MANIFEST_FILENAME)))
        manifest["zero_count"] = 1
        manifest["zeros_file"] = {"filename": "zeros.bin", "sha256": "0" * 64}
        json.dump(manifest, open(os.path.join(path, MANIFEST_FILENAME), "w"))
        report = validate_artifact_dir(path)
        assert report["accepted"] is False


@pytest.mark.unit
class TestCompatibility:
    def _cap(self, **overrides):
        base = dict(
            kernel_id="hqsb.w4a16.triton",
            provider="triton",
            layouts=(LAYOUT_W4A16_ROWMAJOR_NK_V1,),
            bits=(4,),
            group_sizes=(128, None),
            target_arch="sm_86",
            abi_version="1",
            dtype="float16",
        )
        base.update(overrides)
        return compat.KernelCapability(**base)

    def test_direct_load(self, tmp_path):
        doc, _path = _save_doc(str(tmp_path))
        decision = compat.check_compatibility(
            doc,
            self._cap(),
            expected_model=doc.model,
            tensor_name=doc.tensor.name,
            tensor_shape=doc.tensor.shape,
        )
        assert decision.status == compat.DIRECT_LOAD
        assert decision.accepted and decision.allows_low_bit_claim

    def test_bits_mismatch_is_requantize(self, tmp_path):
        doc, _path = _save_doc(str(tmp_path))
        decision = compat.check_compatibility(doc, self._cap(bits=(8,)))
        assert decision.status == compat.REQUANTIZE_REQUIRED
        assert decision.reason_code == compat.REASON_BITS_UNSUPPORTED

    def test_model_revision_mismatch(self, tmp_path):
        doc, _path = _save_doc(str(tmp_path))
        other = ModelIdentity(model_id=doc.model.model_id, revision="different")
        decision = compat.check_compatibility(doc, self._cap(), expected_model=other)
        assert decision.status == compat.REJECT
        assert decision.reason_code == compat.REASON_MODEL_REVISION

    def test_layout_unsupported_is_repack(self, tmp_path):
        doc, _path = _save_doc(str(tmp_path))
        decision = compat.check_compatibility(
            doc, self._cap(layouts=("hqsb.other.layout.v1",))
        )
        assert decision.status == compat.REPACK_REQUIRED

    def test_explicit_fallback_mode(self, tmp_path):
        doc, _path = _save_doc(str(tmp_path))
        decision = compat.check_compatibility(
            doc, self._cap(bits=(8,)), mode=compat.MODE_EXPLICIT_FALLBACK
        )
        assert decision.status == compat.EXPLICIT_FALLBACK
        assert not decision.allows_low_bit_claim
        assert decision.fallback_label

    def test_strict_mode_does_not_fallback(self, tmp_path):
        doc, _path = _save_doc(str(tmp_path))
        decision = compat.check_compatibility(doc, self._cap(bits=(8,)), mode=compat.MODE_STRICT)
        assert decision.status == compat.REQUANTIZE_REQUIRED

    def test_unknown_mode_rejected(self, tmp_path):
        doc, _path = _save_doc(str(tmp_path))
        with pytest.raises(ConfigError):
            compat.check_compatibility(doc, self._cap(), mode="yolo")

    def test_is_requantize_classification(self, tmp_path):
        doc, _path = _save_doc(str(tmp_path))
        assert compat.is_requantize(doc, 8, 128) is True
        assert compat.is_requantize(doc, 4, 64) is True
        assert compat.is_requantize(doc, 4, 128) is False


@pytest.mark.unit
class TestRepack:
    def test_repack_preserves_invariants(self, tmp_path):
        doc, _path = _save_doc(str(tmp_path))
        new_doc, provenance = compat.repack(doc, LAYOUT_W4A16_ROWMAJOR_NK_V1)
        assert all(provenance.invariants.values())
        assert provenance.parent_artifact_id == doc.artifact_id()
        assert new_doc.values == doc.values

    def test_migration_plan(self):
        assert compat.plan_migration("1.0.0", "1.0.0")["status"] == "no_migration_required"
        future = compat.plan_migration("2.0.0", "1.0.0")
        assert future["status"] == "reject_future_version"


@pytest.mark.unit
class TestFaultMatrix:
    def test_all_cases_caught(self, tmp_path):
        paths = save_golden_artifacts(str(tmp_path / "golden"))
        capability = compat.KernelCapability(
            kernel_id="hqsb.w4a16.triton",
            provider="triton",
            layouts=(LAYOUT_W4A16_ROWMAJOR_NK_V1,),
            bits=(4,),
            group_sizes=(128, None),
            target_arch="sm_86",
            abi_version="1",
        )
        from hqsb.quant import faults

        results = faults.run_fault_matrix(
            paths["w4"], str(tmp_path / "scratch"), capability
        )
        summary = faults.summarize_fault_matrix(results)
        assert summary["uncaught"] == []
        assert summary["total_cases"] >= 30
        assert summary["all_prelaunch"] is True
        assert summary["cleanup_ok"] is True
        # golden untouched
        assert os.path.isfile(os.path.join(paths["w4"], MANIFEST_FILENAME))

    def test_matrix_plan_is_deterministic(self):
        from hqsb.quant import faults

        assert faults.matrix_plan_json() == faults.matrix_plan_json()
