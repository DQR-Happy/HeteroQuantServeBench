"""L4 pure control contracts, existing lifecycle and plaintext redaction."""

import json
import time
import pytest
from fastapi.testclient import TestClient
from hqsb.core.precision import PrecisionPlan, catalogue
from hqsb.console.app import create_app
from hqsb.console.config import Settings, DeploymentConfig
from hqsb.console.service import ConsoleService, ServiceError


class TestActor:
    __test__ = False

    def __init__(self, config):
        self.config = config

    def execute(self, kind, payload, *args):
        return {"weight_verification": "test_only"}

    def close(self):
        pass


def test_capacity_uses_compact_tensor_bytes_and_checks_store_and_disk(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from hqsb.backends import precision_artifacts as artifacts

    monkeypatch.setattr(
        artifacts.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024**3)
    )
    destination = tmp_path / "partial"
    metadata = {
        "source_weight_bytes": 14 * 1024**3,
        "max_artifact_bytes": 128 * 1024**2,
    }
    artifacts.ensure_capacity(destination, 64 * 1024**2, metadata)
    with pytest.raises(ValueError, match="STORE_BUDGET"):
        artifacts.ensure_capacity(destination, 128 * 1024**2, metadata)
    monkeypatch.setattr(
        artifacts.shutil, "disk_usage", lambda _: SimpleNamespace(free=500 * 1024**2)
    )
    with pytest.raises(ValueError, match="INSUFFICIENT_DISK"):
        artifacts.ensure_capacity(destination, 64 * 1024**2, metadata)


def wait(service, run):
    for _ in range(200):
        row = service.store.get(run["id"])
        if row["state"] in {"completed", "failed"}:
            return row
        time.sleep(0.01)
    raise TimeoutError(run["id"])


@pytest.mark.parametrize(
    "recipe",
    [
        {"method": "awq"},
        {"method": "gptq", "bits": 8},
        {"method": "smoothquant"},
        {"method": "awq", "bits": 8, "calibration": ["hello"]},
        {"method": "rtn", "group_size": 16},
        {"method": "bnb_nf4", "bits": 8},
        {"method": "none", "calibration": ["unexpected"]},
        {"kv_bits": 8},
        {"method": "gptq", "calibration": [""]},
        {"exclude_modules": ["unsafe/*"]},
    ],
)
def test_invalid_precision_recipes_are_explicitly_rejected(recipe):
    with pytest.raises(ValueError):
        PrecisionPlan.model_validate(recipe)


def test_method_semantics_and_calibration_identity_are_preserved():
    plan = PrecisionPlan(
        method="smoothquant", calibration=["private calibration sample"]
    )
    assert plan.bits == 8
    assert "private calibration sample" not in json.dumps(plan.identity())
    assert (
        plan.identity()
        != PrecisionPlan(method="smoothquant", calibration=["other data"]).identity()
    )
    assert {m["id"] for m in catalogue()["methods"]} >= {
        "gptq",
        "awq",
        "smoothquant",
        "bnb_int8",
        "bnb_nf4",
        "fp8",
    }


def test_exclusions_resolve_suffixes_to_exact_linear_paths_and_reject_typos():
    from hqsb.core.precision import resolve_exclusions

    names = ["model.layers.0.q_proj", "model.layers.1.q_proj", "lm_head"]
    assert resolve_exclusions(names, ["q_proj", "lm_head"]) == sorted(names)
    assert resolve_exclusions(names, ["model.layers.0.q_proj"]) == names[:1]
    with pytest.raises(ValueError, match="NOT_MATCHED"):
        resolve_exclusions(names, ["q_projj"])


def test_quantized_artifact_tampering_and_extra_files_are_rejected(tmp_path):
    from hqsb.backends.precision_artifacts import publish, verify

    partial, destination = tmp_path / "partial", tmp_path / "published"
    partial.mkdir()
    (partial / "payload").write_bytes(b"1234")
    publish(partial, destination, {"artifact_id": "a" * 64})
    assert verify(destination)["disk_bytes"] == 4
    (destination / "payload").write_bytes(b"1235")
    with pytest.raises(ValueError, match="INTEGRITY"):
        verify(destination)
    (destination / "payload").write_bytes(b"1234")
    (destination / "extra").write_bytes(b"unregistered")
    with pytest.raises(ValueError, match="UNDECLARED"):
        verify(destination)


def test_load_configuration_reaches_actor_but_raw_calibration_is_not_saved(tmp_path):
    received = []

    class Actor(TestActor):
        def execute(self, kind, payload, *args):
            received.append((kind, payload))
            return super().execute(kind, payload, *args)

    settings = Settings(
        data_dir=tmp_path,
        evidence_root=tmp_path,
        web_dist=tmp_path,
        deployments=[DeploymentConfig(id="test", name="test")],
    )
    service = ConsoleService(settings, actor_factory=Actor)
    try:
        plan = {
            "method": "gptq",
            "calibration": ["private sample must not appear in stored run"],
        }
        run = wait(
            service,
            service.submit(
                "load",
                "test",
                {"precision": "float16", "quantization": plan},
                "test-key",
            ),
        )
        assert run["state"] == "completed"
        assert received[0][1]["quantization"]["calibration"] == plan["calibration"]
        assert received[0][1]["precision"] == "float16"
        assert "private sample" not in json.dumps(run)
        assert run["config"]["quantization"]["calibration_count"] == 1
        with TestClient(create_app(settings, service=service, monitor=False)) as client:
            assert client.get("/api/console/v1/precision/catalog").status_code == 200
            response = client.post(
                "/api/console/v1/deployments/test/unload",
                json={"precision": "float32"},
                headers={"Idempotency-Key": "wrong-action", "X-HQSB-Client": "console"},
            )
            assert response.status_code == 422
    finally:
        service.close()


def test_omitted_precision_is_resolved_again_on_each_load(tmp_path):
    received = []

    class Actor(TestActor):
        def execute(self, kind, payload, *args):
            received.append((kind, payload))
            return super().execute(kind, payload, *args)

    settings = Settings(
        data_dir=tmp_path,
        evidence_root=tmp_path,
        web_dist=tmp_path,
        deployments=[DeploymentConfig(id="test", name="test")],
    )
    service = ConsoleService(settings, actor_factory=Actor)
    try:
        assert (
            wait(
                service,
                service.submit("load", "test", {"precision": "float32"}, "fp32"),
            )["state"]
            == "completed"
        )
        assert (
            wait(service, service.submit("unload", "test", {}, "unload"))["state"]
            == "completed"
        )
        assert (
            wait(service, service.submit("load", "test", {}, "default"))["state"]
            == "completed"
        )
        assert received[-1][1]["precision"] == "float16"
        assert service.deployments()[0]["precision"] == "float16"
    finally:
        service.close()


@pytest.mark.parametrize(
    "accuracy_limit,accuracy,status",
    [(None, None, "passed"), (0.1, None, "inconclusive"), (0.1, 1.0, "passed")],
)
def test_quality_gate_requires_labels_only_when_accuracy_is_declared(
    tmp_path, accuracy_limit, accuracy, status
):
    import struct
    from hqsb.experiments.artifacts import seal
    from hqsb.observation.storage import sha256

    settings = Settings(
        data_dir=tmp_path, evidence_root=tmp_path, web_dist=tmp_path, deployments=[]
    )
    service = ConsoleService(settings, actor_factory=TestActor)
    limits = dict(
        max_logit_abs_error=1.0,
        max_kl=1.0,
        max_nll_increase=1.0,
        max_accuracy_drop=accuracy_limit,
    )
    ids = []
    try:
        for key in ("baseline", "candidate"):
            run, _ = service.store.create(
                "generate", {"quality_limits": limits}, key, key
            )
            identity = run["id"]
            directory = tmp_path / "captures" / identity
            directory.mkdir(parents=True)
            row = directory / "quality-logits-0000.f32"
            row.write_bytes(struct.pack("<ff", 1, 2))
            receipt = dict(
                run_id=identity,
                model_identity={"hash": "same"},
                tokenizer_identity={"hash": "same"},
                source_digest="same",
                effective={"attention_backend": "eager"},
                precision_receipt={},
            )
            quality = dict(
                run_id=identity,
                status="complete",
                dataset_sha256="same",
                protocol={"top_k": 1, "generation_tokens": 4},
                positions=[
                    dict(
                        sample_id="sample",
                        position=0,
                        slice="all",
                        vocab_size=2,
                        path=row.name,
                        bytes=8,
                        sha256=sha256(row),
                        nll=0.3,
                    )
                ],
                samples=[{"id": "sample", "output_token_sha256": "same"}],
                perplexity=1.35,
                task_accuracy=accuracy,
            )
            (directory / "execution-receipt.json").write_text(json.dumps(receipt))
            (directory / "quality-result.json").write_text(json.dumps(quality))
            seal(directory, run_id=identity)
            service.store.update(
                identity, {"state": "completed", "cleanup": "succeeded"}
            )
            ids.append(identity)
        with TestClient(create_app(settings, service=service, monitor=False)) as client:
            response = client.post(
                "/api/console/v1/precision/comparisons",
                json={"baseline_run_id": ids[0], "candidate_run_ids": ids[1:]},
                headers={"X-HQSB-Client": "console"},
            )
            assert response.status_code == 201, response.text
            assert response.json()["candidates"][0]["quality"]["status"] == status
    finally:
        service.close()


def test_quantization_never_silently_combines_with_unverified_l2_profile(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        evidence_root=tmp_path,
        web_dist=tmp_path,
        deployments=[DeploymentConfig(id="test", name="test")],
    )
    service = ConsoleService(settings, actor_factory=TestActor)
    try:
        with pytest.raises(ServiceError, match="L4_CUSTOM_L2_KERNEL"):
            service.submit(
                "load",
                "test",
                {
                    "kernel_profile": "qwen3-norm-vector-v2",
                    "quantization": {"method": "rtn"},
                },
                "bad-profile",
            )
    finally:
        service.close()
