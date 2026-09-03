"""L2 plans must remain device-independent and immutable across deployment epochs."""
import pytest

from hqsb.core.kernel_profiles import catalogue, profile
from hqsb.core.optimization import resolve_plan, validate_quality_plan
from hqsb.backends.kernel_session import KernelSession
from hqsb.experiments.plans import validate_spec
from hqsb.experiments.artifacts import seal, load_bundle
from hqsb.observation.storage import atomic_json


def test_profiles_are_detached_and_do_not_promise_capability():
    item = profile("qwen3-local-fusions-v1")
    item["operators"].clear()
    assert len(profile("qwen3-local-fusions-v1")["operators"]) == 4
    assert all(p["capability"] == "requires_worker_validation" for p in catalogue()["profiles"])
    with pytest.raises(ValueError, match="UNKNOWN"):
        profile("../../unsafe.so")


def test_kernel_selection_requires_reload():
    defaults = {"attention_backend": "eager"}
    with pytest.raises(ValueError, match="REQUIRES_RELOAD"):
        resolve_plan({"changes": {"kernel_profile": "qwen3-norm-vector-v2"}}, defaults,
                     deployment_kernel_profile="framework-native")
    result = resolve_plan({"changes": {"kernel_profile": "qwen3-norm-vector-v2"}}, defaults,
                          deployment_kernel_profile="qwen3-norm-vector-v2")
    assert result["status"] == "resolved_not_applied"


def test_instance_binding_restores_class_and_instance_methods():
    class Module:
        def forward(self, x):
            return x + 1

    session = KernelSession(None, {})
    original, other = Module(), Module()
    session.install(original, lambda self,x: x + 2)
    assert original.forward(1) == 3 and other.forward(1) == 2
    session.close()
    assert original.forward(1) == 2 and "forward" not in original.__dict__


def test_phase_counters_and_alternative_linear_coverage():
    session = KernelSession(None, {})
    first = session.site("linear", "linear", "lt")
    second = session.site("linear", "linear_small_m", "gemv")
    for row in session.sites:
        row["coverage_group"] = "linear"
    session.set_phase("decode")
    session.hit(second)
    session.binding_hash = "test"
    assert session.summary()["status"] == "invoked"
    assert session.summary()["sites"][first]["calls"]["decode"] == 0
    session.reset()
    assert session.summary()["status"] == "not_exercised"


def test_kernel_experiment_normalizes_baseline_and_requires_declared_change():
    document = {"candidate": {"changes": {"kernel_profile": "qwen3-rope-v1"}},
                "declared_changes": ["kernel_profile"], "quality_plan": {"samples": [{"text": "Hello world."}]}}
    plan = validate_spec(document)
    assert plan["spec"]["baseline"]["changes"]["kernel_profile"] == "framework-native"
    document["declared_changes"] = []
    with pytest.raises(ValueError, match="UNDECLARED"):
        validate_spec(document)


def test_quality_forward_protocol_is_explicit():
    raw = {"samples": [{"text": "Hello world"}], "forward_mode": "full_prefill"}
    assert validate_quality_plan(raw)["forward_mode"] == "full_prefill"
    with pytest.raises(ValueError, match="FORWARD_MODE"):
        validate_quality_plan({**raw, "forward_mode": "pretend"})


def test_seal_includes_kernel_evidence(tmp_path):
    root = tmp_path / ("run_" + "0" * 32)
    root.mkdir()
    for name in ("execution-receipt", "kernel-plan", "kernel-bindings", "kernel-summary"):
        atomic_json(root / (name + ".json"), {})
    manifest = seal(root, run_id=root.name)
    assert {x["path"] for x in manifest["artifacts"]} == {name + ".json" for name in (
        "execution-receipt", "kernel-plan", "kernel-bindings", "kernel-summary")}
    (root / "kernel-summary.json").write_text("tampered")
    with pytest.raises(ValueError):
        load_bundle(root)


def test_kernel_mechanism_projection_keeps_host_and_shape_scope():
    from hqsb.experiments.comparison import _kernel_metrics

    session = KernelSession(None,{})
    site = session.site("norm","rms_norm","cuda-hf-scalar-v1")
    session.set_phase("prefill")
    session.hit(site)
    metrics = _kernel_metrics({"kernel_receipt":session.summary()})
    assert next(m for m in metrics if m["metric_id"]=="kernel.logical_site_calls" and m["label"]=="prefill")["value"]==1
    assert all("hardware_counter" in m["evidence_scope"] for m in metrics)
    assert _kernel_metrics({})==[]


def test_quality_modes_cannot_be_compared_as_same_protocol():
    from hqsb.experiments.comparison import _quality

    left = {"status":"complete","dataset_sha256":"same","protocol":{"name":"raw-token-next-token-v1"}}
    right = {**left,"protocol":{"name":"raw-full-prefill-v1"}}
    assert _quality(None,None,left,right,{})["status"]=="not_comparable"
