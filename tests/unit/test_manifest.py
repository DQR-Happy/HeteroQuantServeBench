"""Unit tests for :mod:`hqsb.models.manifest`.

Validates SHA256 manifest parsing and the artifact integrity gate used by
the model loader. No PyTorch, GPU, or model weights are required.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from hqsb.models.manifest import (
    ManifestEntry,
    compute_sha256,
    parse_manifest,
    verify_model_files,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Valid 64-character lowercase hex digests for parser unit tests.
_H1 = "a" * 64
_H2 = "b" * 64


class TestParseManifest:
    def test_parses_entries(self):
        entries = parse_manifest(
            f"{_H1}  ./config.json\n"
            f"{_H2}  ./model.safetensors\n"
        )
        assert len(entries) == 2
        assert entries[0].sha256 == _H1
        assert entries[0].normalized_path == "config.json"

    def test_ignores_blank_and_comment_lines(self):
        text = f"\n# comment\n  \n{_H1}  ./a.bin\n"
        assert len(parse_manifest(text)) == 1

    def test_rejects_invalid_digest(self):
        with pytest.raises(ValueError):
            parse_manifest("not-a-hash  ./a.bin\n")

    def test_rejects_short_digest(self):
        with pytest.raises(ValueError):
            parse_manifest("abc123  ./a.bin\n")

    def test_rejects_malformed_line(self):
        with pytest.raises(ValueError):
            parse_manifest("only-one-column\n")


class TestComputeSha256:
    def test_matches_hashlib(self, tmp_path):
        payload = b"hello, artifact integrity\n"
        path = tmp_path / "file.bin"
        path.write_bytes(payload)
        assert compute_sha256(str(path)) == _sha256(payload)

    def test_streams_large_file(self, tmp_path):
        # 3 MiB forces multiple streaming chunks (chunk size is 1 MiB).
        payload = os.urandom(3 * 1024 * 1024)
        path = tmp_path / "large.bin"
        path.write_bytes(payload)
        assert compute_sha256(str(path)) == _sha256(payload)


class TestVerifyModelFiles:
    def _write_manifest(self, tmp_path, mapping):
        lines = [f"{digest}  ./{name}" for name, digest in mapping.items()]
        manifest = tmp_path / "manifest.txt"
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(manifest)

    def test_all_files_match(self, tmp_path):
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_bytes(b"{}")
        manifest = self._write_manifest(
            tmp_path, {"config.json": _sha256(b"{}")}
        )
        result = verify_model_files(str(model), manifest)
        assert result.ok
        assert result.verified_files == 1

    def test_missing_file_recorded(self, tmp_path):
        model = tmp_path / "model"
        model.mkdir()
        manifest = self._write_manifest(
            tmp_path, {"config.json": _sha256(b"{}")}
        )
        result = verify_model_files(str(model), manifest)
        assert not result.ok
        assert result.missing_files == ["config.json"]

    def test_mismatched_file_recorded(self, tmp_path):
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_bytes(b"wrong")
        manifest = self._write_manifest(
            tmp_path, {"config.json": _sha256(b"right")}
        )
        result = verify_model_files(str(model), manifest)
        assert not result.ok
        assert len(result.mismatched_files) == 1
        assert result.mismatched_files[0][0] == "config.json"

    def test_missing_model_directory_raises(self, tmp_path):
        manifest = self._write_manifest(tmp_path, {"a": _sha256(b"a")})
        with pytest.raises(FileNotFoundError):
            verify_model_files(str(tmp_path / "nope"), manifest)

    def test_missing_manifest_raises(self, tmp_path):
        model = tmp_path / "model"
        model.mkdir()
        with pytest.raises(FileNotFoundError):
            verify_model_files(str(model), str(tmp_path / "nope.txt"))
