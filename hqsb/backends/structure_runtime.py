"""Invocation proof and immutable lineage of the loaded L5 model."""


def install(provider):
    provider.structure_handles = []
    for row in provider.structure_load.get("transforms", []):
        module = provider.model.get_submodule(row["name"])
        module._hqsb_l5_calls = 0

        def count(owner, _args):
            owner._hqsb_l5_calls += 1

        provider.structure_handles.append(module.register_forward_pre_hook(count))


def reset(provider):
    for row in getattr(provider, "structure_load", {}).get("transforms", []):
        provider.model.get_submodule(row["name"])._hqsb_l5_calls = 0


def receipt(provider):
    doc = getattr(provider, "structure_load", {})
    if not doc:
        return {"status": "original", "artifact_id": None, "transforms": []}
    modules = [
        {
            **row,
            "calls": getattr(
                provider.model.get_submodule(row["name"]), "_hqsb_l5_calls", 0
            ),
        }
        for row in doc["transforms"]
    ]
    return {
        "schema_version": "l5-receipt-v1",
        "artifact_id": doc["artifact_id"],
        "artifact_manifest_sha256": doc["manifest_sha256"],
        "source_model_identity": doc["source_model_identity"],
        "status": "invoked"
        if all(m["calls"] > 0 for m in modules)
        else "loaded_not_exercised",
        "recipe": doc["recipe"],
        "statistics": doc["statistics"],
        "transforms": modules,
        "calibration_text_sha256": doc.get("calibration_text_sha256", []),
        "execution": "compact_dense_gemms"
        if doc["recipe"]["method"] in {"svd_low_rank", "mlp_channels"}
        else "masked_weights_dense_gemms",
        "proof_scope": "bound_modules_and_invocations_not_sparse_instruction_trace",
    }
