"""A selectable profile needs files; successful loading remains a separate claim."""
import hashlib
import json

from hqsb.console.config import DeploymentConfig
from hqsb.console.kernel_inventory import deployment_catalogue, native_inventory


def test_missing_artifact_disables_only_native_profiles():
    config = DeploymentConfig(id="reference", name="test")
    catalog = deployment_catalogue(config)
    profiles = {item["profile_id"]: item for item in catalog["profiles"]}
    assert profiles["framework-native"]["availability"]["loadable"]
    assert profiles["qwen3-attention-math-v1"]["availability"]["loadable"]
    assert not profiles["qwen3-norm-vector-v2"]["availability"]["loadable"]
    assert profiles["qwen3-norm-vector-v2"]["availability"]["reason"] == "KERNEL_NATIVE_MANIFEST_REQUIRED"
    assert profiles["qwen3-norm-vector-v2"]["label"].endswith("v2")


def test_external_provider_cannot_claim_local_kernel_selection():
    catalog = deployment_catalogue(DeploymentConfig(id="external", name="test", provider="openai", base_url="http://example.invalid"))
    assert all(not item["availability"]["loadable"] for item in catalog["profiles"])


def test_other_model_families_and_sizes_never_offer_qwen17b_kernels():
    for model in ("google/gemma-3-1b-it", "LLM-Research/Llama-3.2-1B-Instruct", "Qwen/Qwen3-8B"):
        result = deployment_catalogue(DeploymentConfig(id="test", name="test", model=model))
        for row in result["profiles"]:
            assert row["availability"]["loadable"] == (row["profile_id"] == "framework-native")
            if row["operators"]:
                assert row["availability"]["reason"] == "KERNEL_MODEL_SHAPE_UNSUPPORTED"


def test_actual_config_overrides_mistaken_qwen17b_model_label(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    config = DeploymentConfig(id="test", name="test", model_path=str(tmp_path))
    result = deployment_catalogue(config)
    assert all(not row["availability"]["loadable"] for row in result["profiles"] if row["operators"])


def test_file_identity_check_rejects_tampering_and_source_escape(tmp_path):
    source = tmp_path / "kernel.cu"
    library = tmp_path / "kernel.so"
    source.write_bytes(b"kernel source")
    library.write_bytes(b"test artifact: never loaded")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    document = {"schema_version": "l2-native-v1", "abi": 1,
                "library": str(library), "library_sha256": digest(library),
                "sources": {"kernel.cu": digest(source)}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    config = DeploymentConfig(id="test", name="test", kernel_manifest=str(path))
    verified = native_inventory(config, source_root=tmp_path)
    assert verified["status"] == "files_verified"
    assert verified["runtime_validation"] == "required_on_worker_load"
    assert verified["manifest_sha256"] == digest(path)
    library.write_bytes(b"tampered")
    assert native_inventory(config, source_root=tmp_path)["status"] == "invalid"
    library.write_bytes(b"test artifact: never loaded")
    source.write_bytes(b"changed source")
    assert native_inventory(config, source_root=tmp_path)["status"] == "invalid"
    document["sources"] = {"../outside": "invalid"}
    path.write_text(json.dumps(document))
    assert native_inventory(config, source_root=tmp_path)["status"] == "invalid"
