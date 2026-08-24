"""Pre-registered experiment recipes shared by the CLI and authenticated API."""

from hqsb.core.optimization import VERSION, fingerprint, resolve_plan, validate_quality_plan
from hqsb.observation.contracts import resolve_observation


def validate_spec(raw):
    allowed = {"baseline", "candidate", "declared_changes", "quality_plan", "observation_plan",
               "performance_pairs", "input_tokens", "output_tokens", "max_relative_e2e",
               "quality_limits", "mechanism_expectations"}
    allowed |= {"control_changes", "output_contract"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("EXPERIMENT_SPEC_INVALID")
    spec = {"baseline": {"changes": {}}, "candidate": {"changes": {}},
            "declared_changes": [], "performance_pairs": 10, "input_tokens": 32,
            "output_tokens": 8, "max_relative_e2e": 0.0,
            "observation_plan": {"mode": "memory"},
            "quality_limits": {"max_logit_abs_error": 0.0, "max_kl": 0.0,
                               "max_nll_increase": 0.0, "max_accuracy_drop": 0.0},
            "mechanism_expectations": [], **raw}
    spec.setdefault("control_changes", {})
    spec.setdefault("output_contract", "exact_tokens")
    if not isinstance(spec["output_contract"], str) or spec["output_contract"] not in {"exact_tokens", "quality_tolerant"}:
        raise ValueError("EXPERIMENT_OUTPUT_CONTRACT_INVALID")
    controls = spec["control_changes"]
    if not isinstance(controls, dict) or set(controls) - {"model_identity", "dtype", "kernel", "source_files"}:
        raise ValueError("EXPERIMENT_CONTROL_CHANGES_INVALID")
    for key, value in controls.items():
        if key == "source_files":
            if not isinstance(value, dict) or not 1 <= len(value) <= 64:
                raise ValueError("EXPERIMENT_SOURCE_PATCH_INVALID")
            for path, hashes in value.items():
                if not isinstance(path, str) or ".." in path.split("/") or path.startswith("/") or path.split("/")[0] not in {"hqsb", "ops", "scripts", "configs"}:
                    raise ValueError("EXPERIMENT_SOURCE_PATCH_PATH")
                if not isinstance(hashes, dict) or set(hashes) != {"baseline", "candidate"} or any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes.values()):
                    raise ValueError("EXPERIMENT_SOURCE_PATCH_HASH")
        elif not isinstance(value, dict) or set(value) != {"baseline", "candidate"} or value["baseline"] == value["candidate"]:
            raise ValueError("EXPERIMENT_CONTROL_VALUE_INVALID")
    for name, low, high in (("performance_pairs", 1, 20), ("input_tokens", 2, 512), ("output_tokens", 1, 32)):
        if type(spec[name]) is not int or not low <= spec[name] <= high:
            raise ValueError("EXPERIMENT_BOUND_INVALID:" + name)
    changes = spec["declared_changes"]
    if not isinstance(changes, list) or not all(isinstance(item, str) for item in changes) or len(set(changes)) != len(changes) or set(changes) - {"cleanup_policy", "attention_backend", "output_budget"}:
        raise ValueError("EXPERIMENT_DECLARED_CHANGES_INVALID")
    defaults = {"cleanup_policy": "empty_cache", "attention_backend": "eager", "output_budget": spec["output_tokens"]}
    a, b = (resolve_plan(spec[arm], defaults)["resolved"] for arm in ("baseline", "candidate"))
    actual = {key for key in a if a[key] != b[key]}
    if actual != set(changes):
        raise ValueError("EXPERIMENT_UNDECLARED_OR_UNUSED_CHANGE")
    if type(spec["max_relative_e2e"]) not in (float, int) or not -0.9 <= spec["max_relative_e2e"] <= 1:
        raise ValueError("EXPERIMENT_PERFORMANCE_MARGIN_INVALID")
    limits = spec["quality_limits"]
    if not isinstance(limits, dict) or set(limits) != {"max_logit_abs_error", "max_kl", "max_nll_increase", "max_accuracy_drop"}:
        raise ValueError("EXPERIMENT_QUALITY_LIMITS_INVALID")
    if any(type(value) not in (int, float) or not 0 <= value <= 100 for value in limits.values()):
        raise ValueError("EXPERIMENT_QUALITY_MARGIN_INVALID")
    spec["quality_plan"] = validate_quality_plan(spec.get("quality_plan"))
    observation = spec["observation_plan"]
    if not isinstance(observation, dict) or observation.get("mode", "basic") == "off":
        raise ValueError("EXPERIMENT_DIAGNOSTIC_REQUIRED")
    spec["observation_plan"] = resolve_observation(observation, observation.get("mode", "basic")).as_dict()
    expected = spec["mechanism_expectations"]
    if not isinstance(expected, list) or len(expected) > 16:
        raise ValueError("EXPERIMENT_MECHANISM_LIMIT")
    for item in expected:
        if not isinstance(item, dict) or set(item) != {"metric_id", "label", "direction", "minimum_delta"} or item["direction"] not in {"increase", "decrease"}:
            raise ValueError("EXPERIMENT_MECHANISM_INVALID")
        if not isinstance(item["metric_id"], str) or not isinstance(item["label"], str) or type(item["minimum_delta"]) not in (int, float) or not 0 <= item["minimum_delta"] <= 10**15:
            raise ValueError("EXPERIMENT_MECHANISM_INVALID")
    return {"schema_version": VERSION, "spec": spec, "spec_sha256": fingerprint(spec)}


def role_request(plan, arm, role):
    if arm not in {"baseline", "candidate"} or role not in {"performance", "diagnostic", "quality"}:
        raise ValueError("EXPERIMENT_ROLE_INVALID")
    spec = plan["spec"]
    output = spec["quality_plan"]["generation_tokens"] if role == "quality" else spec["output_tokens"]
    optimization = spec[arm]
    return {"optimization": optimization, "max_output_tokens": output,
            "observation_mode": spec["observation_plan"]["mode"] if role == "diagnostic" else "off",
            "observation_plan": spec["observation_plan"] if role == "diagnostic" else None,
            "experiment_role": role,
            **({"quality_plan": spec["quality_plan"]} if role == "quality" else {})}
