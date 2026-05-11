"""E00-02 — ModelArtifact fault-injection regression tests.

These tests are the permanent, CPU-only regression net for the six fault
classes mandated by the S00 experiment list: missing file, tampered
content, empty path, duplicate path, path traversal, and extra file. They
assert the *gate* behavior, not the experiment driver: every fault must be
rejected before a loader runs, and a legal artifact must always produce the
same identity hash.

The driver that produces the raw evidence lives at
``scripts/audit/run_e00_02_fault_injection.py``.
"""

from __future__ import annotations

import hashlib
import os

import pytest
from pydantic import ValidationError

from hqsb.core.contracts import ModelArtifact
from hqsb.core.errors import ArtifactError, ExitCode, exit_code_for
from hqsb.models.manifest import (
    ManifestError,
    classify_path,
    parse_manifest,
    validate_manifest_paths,
    verify_model_files,
    verify_or_raise,
)

GOOD = b'{"model_type":"qwen3"}'
TOKENIZER = b'{"version":"1.0"}'
WEIGHTS = b"HQSB-FAKE-WEIGHTS-" * 64
SECRET = b"TOP-SECRET-OUTSIDE-THE-MODEL-ROOT\n"

H1 = "a" * 64
H2 = "b" * 64


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(files: dict) -> str:
    return "".join(
        f"{_sha(content)}  ./{rel}\n" for rel, content in sorted(files.items())
    )


def _write_manifest(tmp_path, text: str) -> str:
    path = tmp_path / "manifest.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture()
def base_files() -> dict:
    return {
        "config.json": GOOD,
        "tokenizer.json": TOKENIZER,
        "model.safetensors": WEIGHTS,
    }


@pytest.fixture()
def model_dir(tmp_path, base_files):
    root = tmp_path / "model"
    root.mkdir()
    for rel, content in base_files.items():
        (root / rel).write_bytes(content)
    return root


def _artifact(**overrides) -> ModelArtifact:
    payload = {
        "model_id": "hqsb-test/tiny-qwen3",
        "source": "local",
        "architecture": "Qwen3ForCausalLM",
        "dtype": "float16",
    }
    payload.update(overrides)
    return ModelArtifact(**payload)


# ── The six mandated fault classes ──────────────────────────────────────


@pytest.mark.unit
class TestMissingFile:
    def test_rejected_before_load(self, tmp_path, model_dir, base_files):
        manifest = _write_manifest(tmp_path, _manifest(base_files))
        (model_dir / "model.safetensors").unlink()

        with pytest.raises(ArtifactError) as excinfo:
            verify_or_raise(str(model_dir), manifest)

        assert exit_code_for(excinfo.value) == ExitCode.ARTIFACT
        assert excinfo.value.details["first_bad_file"] == "model.safetensors"
        assert excinfo.value.details["reason_codes"] == ["missing_file"]

    def test_recorded_without_raising(self, tmp_path, model_dir, base_files):
        manifest = _write_manifest(tmp_path, _manifest(base_files))
        (model_dir / "tokenizer.json").unlink()
        result = verify_model_files(str(model_dir), manifest)
        assert not result.ok
        assert result.missing_files == ["tokenizer.json"]
        assert result.first_bad_file == "tokenizer.json"


@pytest.mark.unit
class TestTamperedContent:
    def test_rejected_before_load(self, tmp_path, model_dir, base_files):
        manifest = _write_manifest(tmp_path, _manifest(base_files))
        (model_dir / "tokenizer.json").write_bytes(TOKENIZER + b',"tampered":true}')

        with pytest.raises(ArtifactError) as excinfo:
            verify_or_raise(str(model_dir), manifest)

        assert exit_code_for(excinfo.value) == ExitCode.ARTIFACT
        assert excinfo.value.details["first_bad_file"] == "tokenizer.json"
        assert excinfo.value.details["reason_codes"] == ["hash_mismatch"]

    def test_expected_and_actual_digests_are_reported(
        self, tmp_path, model_dir, base_files
    ):
        manifest = _write_manifest(tmp_path, _manifest(base_files))
        (model_dir / "config.json").write_bytes(b"tampered")
        result = verify_model_files(str(model_dir), manifest)
        path, expected, actual = result.mismatched_files[0]
        assert path == "config.json"
        assert expected == _sha(GOOD)
        assert actual == _sha(b"tampered")


@pytest.mark.unit
class TestEmptyPath:
    @pytest.mark.parametrize(
        "declared",
        ["./", "./nested//config.json"],
    )
    def test_rejected_before_load(self, tmp_path, model_dir, base_files, declared):
        manifest = _write_manifest(
            tmp_path, _manifest(base_files) + f"{_sha(GOOD)}  {declared}\n"
        )
        with pytest.raises(ArtifactError) as excinfo:
            verify_or_raise(str(model_dir), manifest)
        assert "path_empty" in excinfo.value.details["reason_codes"]

    def test_empty_path_never_hashes_the_model_root(self, tmp_path, model_dir):
        """An empty path resolves to the root: it must not be silently ok."""
        manifest = _write_manifest(tmp_path, f"{_sha(GOOD)}  ./\n")
        with pytest.raises(ArtifactError):
            verify_or_raise(str(model_dir), manifest)

    def test_empty_key_rejected_by_contract(self):
        with pytest.raises(ValidationError):
            _artifact(file_hashes={"": H1})


@pytest.mark.unit
class TestDuplicatePath:
    def test_exact_duplicate_rejected_before_load(
        self, tmp_path, model_dir, base_files
    ):
        manifest = _write_manifest(
            tmp_path, _manifest(base_files) + f"{_sha(GOOD)}  ./config.json\n"
        )
        with pytest.raises(ArtifactError) as excinfo:
            verify_or_raise(str(model_dir), manifest)
        assert excinfo.value.details["reason_codes"] == ["path_duplicate"]
        assert excinfo.value.details["first_bad_file"] == "config.json"

    def test_conflicting_duplicate_rejected(self, tmp_path, model_dir, base_files):
        manifest = _write_manifest(
            tmp_path,
            _manifest(base_files)
            + f"{_sha(GOOD)}  ./config.json\n"
            + f"{_sha(b'other')}  ./config.json\n",
        )
        with pytest.raises(ArtifactError):
            verify_or_raise(str(model_dir), manifest)

    def test_casefold_duplicate_rejected(self, tmp_path, model_dir, base_files):
        files = dict(base_files)
        files["Config.json"] = b'{"different":"content"}'
        for rel, content in files.items():
            (model_dir / rel).write_bytes(content)
        manifest = _write_manifest(tmp_path, _manifest(files))
        with pytest.raises(ArtifactError) as excinfo:
            verify_or_raise(str(model_dir), manifest)
        assert "path_duplicate_casefold" in excinfo.value.details["reason_codes"]

    def test_duplicate_is_a_manifest_layer_fault(self):
        """A dict cannot hold duplicates, so only the manifest can catch them."""
        entries = parse_manifest(f"{H1}  ./a.bin\n{H2}  ./a.bin\n", validate_paths=False)
        issues = validate_manifest_paths(entries)
        assert [issue.reason for issue in issues] == ["path_duplicate"]


@pytest.mark.unit
class TestPathTraversal:
    def test_relative_traversal_rejected_before_load(
        self, tmp_path, model_dir, base_files
    ):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.bin").write_bytes(SECRET)
        manifest = _write_manifest(
            tmp_path,
            _manifest(base_files) + f"{_sha(SECRET)}  ../outside/secret.bin\n",
        )
        with pytest.raises(ArtifactError) as excinfo:
            verify_or_raise(str(model_dir), manifest)
        assert excinfo.value.details["reason_codes"] == ["path_traversal"]
        assert excinfo.value.details["first_bad_file"] == "../outside/secret.bin"

    def test_absolute_path_rejected(self, tmp_path, model_dir, base_files):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.bin").write_bytes(SECRET)
        manifest = _write_manifest(
            tmp_path,
            _manifest(base_files) + f"{_sha(SECRET)}  {outside / 'secret.bin'}\n",
        )
        with pytest.raises(ArtifactError) as excinfo:
            verify_or_raise(str(model_dir), manifest)
        assert "path_absolute" in excinfo.value.details["reason_codes"]

    def test_backslash_traversal_rejected(self, tmp_path, model_dir, base_files):
        manifest = _write_manifest(
            tmp_path, _manifest(base_files) + f"{_sha(SECRET)}  ..\\outside\\s.bin\n"
        )
        with pytest.raises(ArtifactError) as excinfo:
            verify_or_raise(str(model_dir), manifest)
        assert "path_backslash" in excinfo.value.details["reason_codes"]

    def test_no_file_outside_the_root_is_ever_hashed(
        self, tmp_path, model_dir, base_files
    ):
        """The traversal target exists and matches: it must still be refused."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.bin").write_bytes(SECRET)
        manifest = _write_manifest(
            tmp_path,
            _manifest(base_files) + f"{_sha(SECRET)}  ../outside/secret.bin\n",
        )
        with pytest.raises(ArtifactError):
            verify_or_raise(str(model_dir), manifest)
        # The file is still intact, proving it was never consumed.
        assert (outside / "secret.bin").read_bytes() == SECRET

    @pytest.mark.parametrize(
        "path",
        [
            "../outside/secret.bin",
            "/etc/passwd",
            "C:/windows/win.ini",
            "..\\outside\\secret.bin",
            "nested/./config.json",
            "nested//config.json",
            "",
        ],
    )
    def test_unsafe_paths_are_classified(self, path):
        assert classify_path(path) is not None

    def test_safe_paths_are_accepted(self):
        for path in ("config.json", "sub/dir/weights.safetensors", ".msc"):
            assert classify_path(path) is None

    def test_unsafe_key_rejected_by_contract(self):
        with pytest.raises(ValidationError):
            _artifact(file_hashes={"../outside/secret.bin": H1})


@pytest.mark.unit
class TestExtraFile:
    def test_undeclared_file_rejected_before_load(
        self, tmp_path, model_dir, base_files
    ):
        manifest = _write_manifest(tmp_path, _manifest(base_files))
        (model_dir / "leftover.tmp").write_bytes(b"partial-download-fragment")

        with pytest.raises(ArtifactError) as excinfo:
            verify_or_raise(str(model_dir), manifest)

        assert excinfo.value.details["reason_codes"] == ["extra_file"]
        assert excinfo.value.details["first_bad_file"] == "leftover.tmp"

    def test_nested_undeclared_file_rejected(self, tmp_path, model_dir, base_files):
        manifest = _write_manifest(tmp_path, _manifest(base_files))
        nested = model_dir / "._____temp"
        nested.mkdir()
        (nested / "chunk.bin").write_bytes(b"x")

        result = verify_model_files(str(model_dir), manifest)
        assert result.extra_files == ["._____temp/chunk.bin"]
        assert not result.ok

    def test_allow_extra_exempts_declared_exceptions(
        self, tmp_path, model_dir, base_files
    ):
        manifest = _write_manifest(tmp_path, _manifest(base_files))
        (model_dir / "model_sha256_manifest.txt").write_bytes(b"copy of manifest")

        result = verify_model_files(
            str(model_dir), manifest, allow_extra=("model_sha256_manifest.txt",)
        )
        assert result.ok
        assert result.allowed_extra_files == ["model_sha256_manifest.txt"]
        assert result.extra_files == []

    def test_allow_extra_supports_globs(self, tmp_path, model_dir, base_files):
        manifest = _write_manifest(tmp_path, _manifest(base_files))
        (model_dir / "cache.tmp").write_bytes(b"x")
        result = verify_model_files(str(model_dir), manifest, allow_extra=("*.tmp",))
        assert result.ok

    def test_non_strict_extra_only_records(self, tmp_path, model_dir, base_files):
        manifest = _write_manifest(tmp_path, _manifest(base_files))
        (model_dir / "leftover.tmp").write_bytes(b"x")
        result = verify_model_files(str(model_dir), manifest, strict_extra=False)
        assert result.ok
        assert result.extra_files == ["leftover.tmp"]


# ── Malformed manifests ─────────────────────────────────────────────────


@pytest.mark.unit
class TestMalformedManifest:
    def test_invalid_digest(self):
        with pytest.raises(ManifestError) as excinfo:
            parse_manifest("not-a-hash  ./config.json\n")
        assert "digest_invalid" in excinfo.value.reasons

    def test_single_column_line(self):
        with pytest.raises(ManifestError) as excinfo:
            parse_manifest("only-one-column\n")
        assert "line_malformed" in excinfo.value.reasons

    def test_manifest_error_is_a_value_error(self):
        assert issubclass(ManifestError, ValueError)

    def test_empty_model_path_rejected(self, tmp_path):
        """An empty path resolves to cwd and would verify the wrong tree."""
        manifest = _write_manifest(tmp_path, f"{_sha(GOOD)}  ./config.json\n")
        with pytest.raises(ManifestError):
            verify_model_files("", manifest)

    def test_empty_manifest_path_rejected(self, tmp_path):
        with pytest.raises(ManifestError):
            verify_model_files(str(tmp_path), "   ")


# ── Positive control and hash stability ─────────────────────────────────


@pytest.mark.unit
class TestLegalArtifact:
    def test_clean_snapshot_passes(self, tmp_path, model_dir, base_files):
        manifest = _write_manifest(tmp_path, _manifest(base_files))
        result = verify_or_raise(str(model_dir), manifest)
        assert result.ok
        assert result.first_bad_file is None
        assert result.extra_files == []
        assert result.reason_codes == []

    def test_identity_hash_is_stable(self, base_files, tmp_path):
        hashes = {rel: _sha(content) for rel, content in base_files.items()}
        first = _artifact(file_hashes=hashes).identity_hash()
        for _ in range(5):
            assert _artifact(file_hashes=hashes).identity_hash() == first

    def test_identity_hash_ignores_insertion_order(self, base_files):
        hashes = {rel: _sha(content) for rel, content in base_files.items()}
        forward = _artifact(file_hashes=hashes).identity_hash()
        reversed_hashes = dict(reversed(list(hashes.items())))
        assert (
            _artifact(file_hashes=reversed_hashes).identity_hash() == forward
        )

    def test_identity_hash_changes_with_content(self, base_files):
        hashes = {rel: _sha(content) for rel, content in base_files.items()}
        baseline = _artifact(file_hashes=hashes).identity_hash()
        tampered = dict(hashes)
        tampered["config.json"] = _sha(b"tampered")
        assert _artifact(file_hashes=tampered).identity_hash() != baseline

    def test_identity_hash_ignores_provenance_fields(self, base_files):
        hashes = {rel: _sha(content) for rel, content in base_files.items()}
        baseline = _artifact(file_hashes=hashes).identity_hash()
        annotated = _artifact(file_hashes=hashes, tool_version="1.2.3", license="MIT")
        assert annotated.identity_hash() == baseline

    def test_non_lowercase_digest_rejected(self):
        with pytest.raises(ValidationError):
            _artifact(file_hashes={"config.json": "A" * 64})

    def test_manifest_and_roundtrip_agree(self, tmp_path, model_dir, base_files):
        """A manifest-verified snapshot yields the same artifact identity."""
        manifest = _write_manifest(tmp_path, _manifest(base_files))
        result = verify_or_raise(str(model_dir), manifest)
        from_manifest = _artifact(
            file_hashes={e.normalized_path: e.sha256 for e in result.entries}
        )
        direct = _artifact(
            file_hashes={rel: _sha(c) for rel, c in base_files.items()}
        )
        assert from_manifest.identity_hash() == direct.identity_hash()


@pytest.mark.unit
class TestResultSerialization:
    def test_as_dict_is_json_serializable(self, tmp_path, model_dir, base_files):
        import json

        manifest = _write_manifest(tmp_path, _manifest(base_files))
        result = verify_or_raise(str(model_dir), manifest)
        payload = json.dumps(result.as_dict())
        assert result.manifest_sha256 in payload

    def test_manifest_sha256_identifies_the_file_list(
        self, tmp_path, model_dir, base_files
    ):
        manifest = _write_manifest(tmp_path, _manifest(base_files))
        result = verify_model_files(str(model_dir), manifest)
        expected = hashlib.sha256(
            open(manifest, "rb").read()
        ).hexdigest()
        assert result.manifest_sha256 == expected
        assert os.path.isfile(manifest)
