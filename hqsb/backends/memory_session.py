"""Actual single-request cache lifetime and evidence. No scheduling policy here."""
import hashlib
import json
import resource
import time
from pathlib import Path

from hqsb.core.memory_plan import MemoryPlan, catalogue
from hqsb.backends.prefix_store import PrefixStore


class MemoryPlanError(ValueError):
    pass


def capabilities(provider):
    cfg = provider.model.config.get_text_config(decoder=True)
    hybrid = bool(getattr(cfg, "sliding_window", None) or getattr(cfg, "attention_chunk_size", None)
                  or any(x != "full_attention" for x in (getattr(cfg, "layer_types", None) or [])))
    quant = bool(getattr(getattr(provider, "precision_plan", None), "kv_bits", None))
    native = not provider.kernel_session or provider.kernel_session.profile["profile_id"] == "framework-native"
    return {**catalogue(), "availability": "resident_worker_inspected", "model_class": type(provider.model).__name__,
            "layer_types": getattr(cfg, "layer_types", None), "hybrid": hybrid,
            "modes": [{**row, "supported": native and (not quant or row["id"] == "model_default"),
                       "reason": "当前自定义算子尚未验证 L6 缓存组合；请在设备页重载框架原生算子。" if not native else
                                 "当前 L4 KV 量化仅支持模型默认 KV；请在精度页关闭 KV 量化。" if quant and row["id"] != "model_default" else None}
                      for row in catalogue()["modes"]],
            "prefix_supported": native and not hybrid and not quant,
            "prefix_reason": "前缀适配器目前要求原生算子、未量化、全部为 full attention 的 KV。" if hybrid or quant or not native else None,
            "prefix_retained_bytes": getattr(getattr(provider, "prefix_store", None), "retained_bytes", 0)}


def clear_prefix(provider):
    import torch
    torch.cuda.synchronize()
    store = getattr(provider, "prefix_store", None)
    result = store.clear() if store else {"released_entries": 0, "released_bytes": 0}
    torch.cuda.empty_cache()
    return {**result, "completion": "cuda_synchronized", "memory": provider.memory()}


def inventory(cache):
    rows, seen, totals = [], set(), {"cpu": 0, "cuda": 0}
    def walk(tensor, index, role):
        flatten = getattr(tensor, "__tensor_flatten__", None)
        if flatten is not None:
            names, _ = flatten()
            for name in names:
                walk(getattr(tensor, name), index, role + "." + name)
        elif tensor is not None and hasattr(tensor, "untyped_storage"):
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            size = storage.nbytes() if key not in seen else 0
            seen.add(key)
            totals[tensor.device.type] = totals.get(tensor.device.type, 0) + size
            rows.append({"layer": index, "role": role, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                         "device": str(tensor.device), "logical_bytes": tensor.numel() * tensor.element_size(),
                         "unique_storage_bytes": size})
    for index, layer in enumerate(getattr(cache, "layers", [])):
        for role in ("keys", "values", "_quantized_keys", "_quantized_values"):
            walk(getattr(layer, role, None), index, role)
    return {"scope": "cache_tensor_storage_not_attention_workspace_or_physical_pages", "bytes_by_device": totals, "tensors": rows}


class _Transfers:
    """Observe upstream cache API transfers, without altering stream semantics."""
    def offload(self, layer_idx, *args, **kwargs):
        layer = self.layers[layer_idx]
        tensors = [getattr(layer, k, None) for k in ("keys", "values")]
        size = sum(t.numel() * t.element_size() for t in tensors if t is not None and t.device.type == "cuda")
        result = super().offload(layer_idx, *args, **kwargs)
        if size and getattr(layer, "keys", None).device.type == "cpu":
            self.l6_transfers["d2h_bytes"] += size
            self.l6_transfers["offload_calls"] += 1
        return result

    def prefetch(self, layer_idx, *args, **kwargs):
        # Upstream may skip sliding layers and wrap the index. Count all actual
        # CPU -> device tensor changes, not the requested layer number.
        before = [(layer, getattr(layer, "keys", None), getattr(layer, "values", None)) for layer in self.layers]
        result = super().prefetch(layer_idx, *args, **kwargs)
        for layer, keys, values in before:
            if keys is not None and keys.device.type == "cpu" and layer.keys.device.type == "cuda":
                self.l6_transfers["h2d_bytes"] += keys.numel() * keys.element_size() + values.numel() * values.element_size()
                self.l6_transfers["prefetch_calls"] += 1
        return result


class MemorySession:
    def __init__(self, provider, request):
        self.provider, self.request = provider, request
        self.plan = MemoryPlan.model_validate(request.get("memory_plan") or {}).model_dump()
        self.enabled = request.get("memory_plan") is not None
        self.started = time.monotonic()
        self.before = resource.getrusage(resource.RUSAGE_SELF)
        self.cache = None
        self.tokens = ()
        self.reused = 0
        self.processed_positions = 0
        self.forwards = 0
        self.selected_logprobs = []
        self.transfer = {"d2h_bytes": 0, "h2d_bytes": 0, "offload_calls": 0, "prefetch_calls": 0}
        self.prefix = {"lookup": "disabled", "reused_tokens": 0, "restore_copy_bytes": 0, "snapshot_copy_bytes": 0}
        self.actual_class = None
        self.final_inventory = None
        self.identity = None
        self.evictions_before = 0
        self.prefix_preparation_ms = 0

    @property
    def use_cache(self):
        return self.plan["mode"] != "recompute"

    def prepare(self, token_ids, limit):
        from transformers import DynamicCache, StaticCache
        from hqsb.backends.precision_runtime import new_cache
        self.tokens = tuple(token_ids)
        cap = capabilities(self.provider)
        mode = self.plan["mode"]
        if self.enabled and not next(row for row in cap["modes"] if row["id"] == mode)["supported"]:
            raise MemoryPlanError("L6_CACHE_COMPOSITION_UNSUPPORTED")
        if self.plan["prefix_reuse"] and not cap["prefix_supported"]:
            raise MemoryPlanError("L6_PREFIX_REQUIRES_FULL_ATTENTION_DYNAMIC_CACHE")
        required = len(self.tokens) + limit - 1
        capacity = self.plan["max_cache_tokens"] or required
        if capacity < required or capacity > self.provider.config["context_limit"]:
            raise MemoryPlanError(f"L6_STATIC_CAPACITY: required={required}, capacity={capacity}, context={self.provider.config['context_limit']}")
        cfg = self.provider.model.config.get_text_config(decoder=True)
        if mode == "model_default":
            self.cache = new_cache(self.provider)
        elif mode == "recompute":
            self.cache = None
        elif mode in {"static", "offloaded_static"}:
            cls = type("TrackedStaticCache", (_Transfers, StaticCache), {})
            self.cache = cls(config=cfg, max_cache_len=capacity, offloading=mode.startswith("offloaded"), offload_only_non_sliding=False)
        else:
            cls = type("TrackedDynamicCache", (_Transfers, DynamicCache), {})
            self.cache = cls(config=cfg, offloading=mode.startswith("offloaded"), offload_only_non_sliding=False)
        if self.cache is not None:
            self.cache.l6_transfers = self.transfer
        self.capacity = capacity if "static" in mode else None
        if self.plan["prefix_reuse"]:
            self._restore_prefix()
        self.actual_class = type(self.cache).__name__ if self.cache is not None else None
        return self.cache, self.reused

    def _restore_prefix(self):
        from transformers import DynamicCache
        start = time.monotonic()
        provider = self.provider
        if not hasattr(provider, "prefix_store"):
            provider.prefix_store = PrefixStore()
        store = provider.prefix_store
        self.evictions_before = store.evictions
        store.trim(self.plan["prefix_budget_bytes"], self.plan["prefix_max_entries"])
        identity = {"model": provider.source_identity, "tokenizer": provider.tokenizer_identity,
                    "source": provider.source_environment["source_digest"], "precision": provider.config.get("precision"),
                    "precision_plan": provider.precision_plan.identity(), "namespace": self.plan["prefix_namespace"],
                    "layout": "hf-dynamic-full-attention-v1", "attention": provider.config["attention"]}
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        hit = store.lookup(self.identity, self.tokens)
        self.prefix["lookup"] = "miss"
        if hit:
            tokens, data, size = hit
            self.cache = DynamicCache(ddp_cache_data=data)
            self.cache.l6_transfers = self.transfer
            self.reused = len(tokens)
            self.prefix.update(lookup="hit", reused_tokens=self.reused, restore_copy_bytes=size)
        self.prefix_preparation_ms = (time.monotonic() - start) * 1000

    def after_forward(self, cache, input_positions, logprob):
        self.cache = cache
        self.actual_class = type(cache).__name__ if cache is not None else None
        self.forwards += 1
        self.processed_positions += input_positions
        self.selected_logprobs.append(logprob)

    def finish(self, success):
        import torch
        torch.cuda.synchronize()  # Includes offload/prefetch streams before release.
        self.final_inventory = inventory(self.cache)
        if success and self.plan["prefix_reuse"] and self.cache is not None:
            count = min(len(self.tokens) - 1, self.plan["prefix_max_tokens"])
            with torch.inference_mode():
                self.prefix["snapshot_copy_bytes"] = self.provider.prefix_store.publish(
                    self.identity, self.tokens[:count], self.cache, self.plan["prefix_budget_bytes"], self.plan["prefix_max_entries"])
            torch.cuda.synchronize()
        store = getattr(self.provider, "prefix_store", None)
        self.prefix.update(retained_bytes=store.retained_bytes if store else 0,
                           retained_entries=len(store.entries) if store else 0,
                           evictions=store.evictions - self.evictions_before if store and self.plan["prefix_reuse"] else 0)
        self.cache = None
        usage = resource.getrusage(resource.RUSAGE_SELF)
        doc = {"schema_version": "l6-v1", "run_id": self.request.get("capture_id"), "epoch": self.request.get("expected_epoch"),
               "plan": self.plan, "status": "applied" if success and self.forwards else "not_fully_applied",
               "cache_class": self.actual_class, "use_cache": self.use_cache, "static_capacity_tokens": self.capacity if hasattr(self, "capacity") else None,
               "forward_calls": self.forwards, "processed_input_positions": self.processed_positions,
               "selected_token_logprobs": self.selected_logprobs, "prefix": self.prefix, "transfers": self.transfer,
               "prefix_preparation_host_ms": self.prefix_preparation_ms,
               "storage_before_release": self.final_inventory,
               "allocator_after_prefix_publish": self.provider.memory() if hasattr(self.provider, "memory") else None,
               "process_faults": {"scope": "worker_request_including_tokenization_and_cleanup_start_not_GPU_pages",
                                  "minor": usage.ru_minflt - self.before.ru_minflt, "major": usage.ru_majflt - self.before.ru_majflt},
               "completion": "cuda_all_streams_synchronized",
               "limitations": catalogue()["limitations"]}
        self.request["_memory_receipt"] = doc
        if self.enabled and self.request.get("capture_dir"):
            from hqsb.observation.storage import atomic_json
            atomic_json(Path(self.request["capture_dir"]) / "memory-cache.json", doc)
        return doc
