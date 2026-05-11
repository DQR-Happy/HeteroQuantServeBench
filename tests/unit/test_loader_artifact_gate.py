"""E00-02 — the artifact gate must reject bad snapshots *inside* the loader.

The fault-injection suite in ``test_artifact_fault_injection.py`` exercises
the gate functions directly. This module closes the last gap: it proves
:func:`hqsb.models.loader.load_qwen3` runs the gate **before** it asks
ModelScope for a tokenizer or any weights, so a bad artifact cannot reach a
backend.

A fake ``AutoModelForCausalLM`` / ``AutoTokenizer`` is injected so the test
never touches real weights: if the gate were skipped, the sentinel would
raise and the test would fail loudly instead of silently loading 3.4 GB.
"""

from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch", reason="requires the benchmark extra")
pytest.importorskip("modelscope", reason="requires the benchmark extra")

import hqsb.models.loader as loader_module  # noqa: E402
from hqsb.core.errors import ArtifactError, ExitCode  # noqa: E402

GOOD = b'{"model_type":"qwen3"}'
TOKENIZER = b'{"version":"1.0"}'
WEIGHTS = b"HQSB-FAKE-WEIGHTS-" * 8


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def snapshot(tmp_path):
    """A minimal model directory plus a matching SHA256 manifest."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    files = {
        "config.json": GOOD,
        "tokenizer.json": TOKENIZER,
        "model.safetensors": WEIGHTS,
    }
    for rel, content in files.items():
        (model_dir / rel).write_bytes(content)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "".join(f"{_sha(c)}  ./{rel}\n" for rel, c in sorted(files.items())),
        encoding="utf-8",
    )
    return model_dir, str(manifest), files


class _Refuse:
    """Stand-in for ModelScope: reaching it means the gate was bypassed."""

    def __init__(self) -> None:
        self.calls = 0

    def from_pretrained(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        raise AssertionError(
            "the loader reached ModelScope: the artifact gate did not run "
            "before load"
        )


@pytest.fixture()
def refuse(monkeypatch):
    """Replace both ModelScope entry points with refusals."""
    model_cls = _Refuse()
    tokenizer_cls = _Refuse()
    monkeypatch.setattr(loader_module, "AutoModelForCausalLM", model_cls)
    monkeypatch.setattr(loader_module, "AutoTokenizer", tokenizer_cls)
    return model_cls, tokenizer_cls


@pytest.mark.unit
class TestLoaderArtifactGate:
    def test_missing_file_rejected_before_load(self, snapshot, refuse):
        model_dir, manifest, _files = snapshot
        (model_dir / "model.safetensors").unlink()

        with pytest.raises(ArtifactError) as excinfo:
            loader_module.load_qwen3(str(model_dir), verify_manifest=manifest)

        assert excinfo.value.exit_code == ExitCode.ARTIFACT
        assert excinfo.value.details["first_bad_file"] == "model.safetensors"
        # Nothing was ever requested from ModelScope.
        assert refuse[0].calls == 0 and refuse[1].calls == 0

    def test_tampered_file_rejected_before_load(self, snapshot, refuse):
        model_dir, manifest, _files = snapshot
        (model_dir / "config.json").write_bytes(b"tampered")

        with pytest.raises(ArtifactError) as excinfo:
            loader_module.load_qwen3(str(model_dir), verify_manifest=manifest)

        assert excinfo.value.details["reason_codes"] == ["hash_mismatch"]
        assert refuse[0].calls == 0 and refuse[1].calls == 0

    def test_extra_file_rejected_before_load(self, snapshot, refuse):
        model_dir, manifest, _files = snapshot
        (model_dir / "leftover.tmp").write_bytes(b"partial-download-fragment")

        with pytest.raises(ArtifactError) as excinfo:
            loader_module.load_qwen3(str(model_dir), verify_manifest=manifest)

        assert excinfo.value.details["reason_codes"] == ["extra_file"]
        assert refuse[0].calls == 0 and refuse[1].calls == 0

    def test_traversal_manifest_rejected_before_load(self, snapshot, refuse):
        model_dir, manifest, _files = snapshot
        outside = model_dir.parent / "outside"
        outside.mkdir()
        secret = b"TOP-SECRET\n"
        (outside / "secret.bin").write_bytes(secret)
        digest = _sha(secret)
        with open(manifest, "a", encoding="utf-8") as fh:
            fh.write(f"{digest}  ../outside/secret.bin\n")

        with pytest.raises(ArtifactError) as excinfo:
            loader_module.load_qwen3(str(model_dir), verify_manifest=manifest)

        assert "path_traversal" in excinfo.value.details["reason_codes"]
        assert refuse[0].calls == 0 and refuse[1].calls == 0

    def test_allow_extra_lets_a_declared_exception_through_gate(
        self, snapshot, refuse
    ):
        """With an explicit exemption the gate passes and loading proceeds.

        The sentinel raising proves the loader really did move past the
        gate, i.e. the exemption is honoured rather than always-fail.
        """
        model_dir, manifest, _files = snapshot
        (model_dir / "model_sha256_manifest.txt").write_bytes(b"copy")

        with pytest.raises(AssertionError):
            loader_module.load_qwen3(
                str(model_dir),
                verify_manifest=manifest,
                allow_extra=("model_sha256_manifest.txt",),
            )
        assert refuse[1].calls == 1
