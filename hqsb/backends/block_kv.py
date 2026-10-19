"""Real bounded device block storage with dense Attention materialization.

Blocks are tensor slices, never CUDA virtual/physical pages. This reduces tail
scatter copies; it deliberately does not implement a PagedAttention kernel.
Only the synchronous StepExecutor owns and releases these request mappings.
"""
import math


class BlockKV:
    def __init__(self, torch, *, block_size, capacity_blocks, max_bytes):
        self.torch, self.block_size = torch, block_size
        self.capacity_blocks, self.max_bytes = capacity_blocks, max_bytes
        self.rows, self.slabs = {}, None
        self.free = list(reversed(range(capacity_blocks)))
        self.allocations = self.releases = self.reuses = 0
        self.used_before = set()
        self.bytes_per_token = 0

    def refusal(self, identities, lengths, additions):
        needed = 0
        for identity, start, added in zip(identities, lengths, additions):
            row = self.rows.get(identity, {"valid": 0, "blocks": []})
            if row["valid"] != start:
                return "BLOCK_KV_STALE_LENGTH"
            needed += math.ceil((start + added) / self.block_size) - len(row["blocks"])
        return "BLOCK_POOL_CAPACITY" if needed > len(self.free) else None

    def pack(self, identities, lengths):
        from transformers import DynamicCache
        width = max(lengths)
        if not width:
            return DynamicCache(), 0
        data, copied = [], 0
        for layer in range(self.slabs.shape[1]):
            pair = []
            for component in range(2):
                ref = self.slabs[component, layer]
                tensor = ref.new_zeros((len(identities), ref.shape[1], width, ref.shape[3]))
                for row_index, (identity, valid) in enumerate(zip(identities, lengths)):
                    if not valid:
                        continue
                    entry = self.rows[identity]
                    if entry["valid"] != valid:
                        raise ValueError("BLOCK_KV_STALE_LENGTH")
                    for logical, physical in enumerate(entry["blocks"]):
                        offset = logical * self.block_size
                        count = min(self.block_size, valid - offset)
                        tensor[row_index, :, width - valid + offset:width - valid + offset + count, :].copy_(ref[physical, :, :count, :])
                        copied += ref[physical, :, :count, :].numel() * ref.element_size()
                pair.append(tensor)
            data.append(tuple(pair))
        return DynamicCache(ddp_cache_data=data), copied

    def commit(self, cache, identities, lengths, additions):
        reason = self.refusal(identities, lengths, additions)
        if reason:
            raise ValueError(reason)
        if not cache.layers or any(type(layer).__name__ != "DynamicLayer" for layer in cache.layers):
            raise ValueError("BLOCK_POOL_REQUIRES_FULL_ATTENTION_DYNAMIC_LAYERS")
        sample = cache.layers[0].keys
        layers, heads, dim = len(cache.layers), sample.shape[1], sample.shape[-1]
        if any(tuple(t.shape[1::2]) != (heads, dim) or t.dtype != sample.dtype or t.device != sample.device
               for layer in cache.layers for t in (layer.keys, layer.values)):
            raise ValueError("BLOCK_POOL_NONUNIFORM_GEOMETRY")
        if self.slabs is None:
            self.bytes_per_token = 2 * layers * heads * dim * sample.element_size()
            size = self.capacity_blocks * self.block_size * self.bytes_per_token
            if size > self.max_bytes:
                raise ValueError("BLOCK_POOL_BYTE_BUDGET")
            self.slabs = sample.new_empty((2, layers, self.capacity_blocks, heads, self.block_size, dim))
        copied, past, query = 0, max(lengths), max(additions)
        for index, (identity, start, added) in enumerate(zip(identities, lengths, additions)):
            entry = self.rows.setdefault(identity, {"valid": 0, "blocks": []})
            while len(entry["blocks"]) < math.ceil((start + added) / self.block_size):
                physical = self.free.pop()
                self.reuses += physical in self.used_before
                self.used_before.add(physical)
                self.allocations += 1
                entry["blocks"].append(physical)
            # Scatter only newly computed K/V. Old blocks remain private and intact.
            offset = 0
            while offset < added:
                logical, within = divmod(start + offset, self.block_size)
                count = min(added - offset, self.block_size - within)
                physical = entry["blocks"][logical]
                source = past + query - added + offset
                for layer_idx, layer in enumerate(cache.layers):
                    for component, tensor in enumerate((layer.keys, layer.values)):
                        value = tensor[index, :, source:source + count, :]
                        self.slabs[component, layer_idx, physical, :, within:within + count, :].copy_(value)
                        copied += value.numel() * value.element_size()
                offset += count
            entry["valid"] = start + added
        return copied

    def release(self, identity):
        entry = self.rows.pop(identity, None)
        if entry:
            self.free.extend(entry["blocks"])
            self.releases += len(entry["blocks"])

    def bytes(self):
        return self.slabs.untyped_storage().nbytes() if self.slabs is not None else 0

    def receipt(self):
        active = sum(len(row["blocks"]) for row in self.rows.values())
        valid = sum(row["valid"] for row in self.rows.values())
        return {"layout": "block_pool_dense_attention", "paged_attention_kernel": False,
                "block_tokens": self.block_size, "capacity_blocks": self.capacity_blocks,
                "active_blocks": active, "free_blocks": len(self.free), "valid_tokens": valid,
                "tail_waste_tokens": active * self.block_size - valid,
                "pool_storage_bytes": self.bytes(), "active_capacity_bytes": active * self.block_size * self.bytes_per_token,
                "valid_kv_bytes": valid * self.bytes_per_token, "allocations": self.allocations,
                "released_blocks": self.releases, "reused_blocks": self.reuses,
                "page_scope": "tensor_slices_not_OS_or_CUDA_virtual_pages"}

    def clear(self):
        for identity in list(self.rows):
            self.release(identity)
        self.slabs = None
