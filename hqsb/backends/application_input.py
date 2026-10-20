"""L10 resident tokenizer preparation; explicit lossy transformations, no forward."""

from hqsb.backends.prompting import prepare_prompt
from hqsb.core.optimization import fingerprint


def prefix_scope(provider, owner, begin):
    """Swap an isolated L10 prefix store; preserve the caller's prior KV cache."""
    import torch
    from hqsb.backends.prefix_store import PrefixStore

    active = getattr(provider, "_l10_prefix_scope", None)
    if active:
        if active[0] != owner:
            raise ValueError("L10_PREFIX_SCOPE_OWNERSHIP")
        if begin:
            return {"scope": "already_active"}
        torch.cuda.synchronize()
        released = provider.prefix_store.clear()
        if active[1]:
            provider.prefix_store = active[2]
        else:
            del provider.prefix_store
        del provider._l10_prefix_scope
        torch.cuda.empty_cache()
        return {"scope": "restored", **released}
    if begin:
        provider._l10_prefix_scope = (
            owner,
            hasattr(provider, "prefix_store"),
            getattr(provider, "prefix_store", None),
        )
        provider.prefix_store = PrefixStore()
        return {"scope": "new_isolated_store"}
    return {"scope": "already_restored"}


def metadata(provider):
    from hqsb.backends.memory_session import capabilities
    import torch
    import transformers

    return {
        "model_identity": provider.source_identity,
        "tokenizer_identity": provider.tokenizer_identity,
        "execution_identity": {
            "kernel_binding_sha256": getattr(
                provider.kernel_session, "binding_hash", None
            ),
            "precision_plan": provider.precision_plan.identity(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "prefix_capability": capabilities(provider),
        "source_digest": provider.source_environment["source_digest"],
    }


def prepare(provider, payload):
    item, policy = payload["input"], payload["policy"]
    request = (
        {"prompt": item["prompt"]}
        if item.get("prompt") is not None
        else {"messages": item["messages"]}
    )

    def encoded(req):
        prompt, special = prepare_prompt(
            provider.tokenizer, req, provider.model.config.model_type
        )
        return provider.tokenizer.encode(prompt, add_special_tokens=special)

    before = encoded(request)
    if (
        policy["input_transform"] == "head_tail_tokens"
        and len(before) > policy["input_token_cap"]
    ):
        cap = policy["input_token_cap"]
        # Re-tokenization is verified: decoding arbitrary token slices can alter
        # boundary merges; never silently exceed the promised token cap.
        for size in range(cap, 0, -1):
            a = size // 2
            request = {
                "prompt": provider.tokenizer.decode(
                    before[:a] + before[-(size - a) :], skip_special_tokens=True
                )
            }
            if len(encoded(request)) <= cap:
                break
    if policy["input_transform"] == "recent_messages":
        system = [m for m in item["messages"] if m["role"] == "system"]
        rest = [m for m in item["messages"] if m["role"] != "system"]
        request = {"messages": system + rest[-policy["recent_messages"] :]}
    ids = encoded(request)
    if not ids or ("prompt" in request and not request["prompt"].strip()):
        raise ValueError("L10_INPUT_TRANSFORMATION_EMPTY")
    native = getattr(
        provider.model.config,
        "max_position_embeddings",
        provider.config["context_limit"],
    )
    limit = min(native, provider.config["context_limit"])
    requested = min(
        item["max_output_tokens"], policy["output_cap"] or item["max_output_tokens"]
    )
    effective = min(requested, provider.config["max_output_tokens"], limit - len(ids))
    if effective < 1:
        raise ValueError(
            f"L10_NO_OUTPUT_CONTEXT_ROOM: input={len(ids)}, context={limit}"
        )
    return {
        "payload": request,
        "input_tokens_before": len(before),
        "input_tokens_after": len(ids),
        "effective_input_sha256": fingerprint(ids),
        "original_input_sha256": fingerprint(before),
        "requested_output_tokens": item["max_output_tokens"],
        "effective_output_tokens": effective,
        "lossy": before != ids,
        "context_limit": limit,
    }
