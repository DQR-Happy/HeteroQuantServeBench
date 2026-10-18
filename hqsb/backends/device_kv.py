"""Dense full-attention KV: independent compact rows, transient left-padded batch.

Only completed synchronous forwards may be committed/released. No physical
address, Linux page or paged-attention claim is made by this adapter.
"""


class DenseKV:
    def __init__(self, torch):
        self.torch, self.rows = torch, {}

    def pack(self, identities, lengths):
        from transformers import DynamicCache

        width = max(lengths)
        if not width:
            return DynamicCache(), 0
        template = next(self.rows[i] for i, n in zip(identities, lengths) if n)
        data, copied = [], 0
        for layer in range(len(template)):
            pair = []
            for component in range(2):
                ref = template[layer][component]
                packed = ref.new_zeros((len(identities), ref.shape[1], width, ref.shape[3]))
                for row, (identity, count) in enumerate(zip(identities, lengths)):
                    if count:
                        tensor = self.rows[identity][layer][component]
                        if tensor.shape[-2] != count:
                            raise ValueError("DEVICE_KV_LENGTH_MISMATCH")
                        packed[row:row + 1, :, width - count:, :].copy_(tensor)
                        copied += tensor.numel() * tensor.element_size()
                pair.append(packed)
            data.append(tuple(pair))
        return DynamicCache(ddp_cache_data=data), copied

    def commit(self, cache, identities, lengths, additions):
        torch = self.torch
        past, query = max(lengths), max(additions)
        copied = 0
        for row, (identity, count, added) in enumerate(zip(identities, lengths, additions)):
            layers = []
            for layer in cache.layers:
                if type(layer).__name__ != "DynamicLayer":
                    raise ValueError("UNSUPPORTED_DEVICE_CACHE_LAYOUT")
                pair = []
                for tensor in (layer.keys, layer.values):
                    # Never retain a view into another request's batch storage.
                    compact = torch.cat((tensor[row:row + 1, :, past - count:past, :],
                                         tensor[row:row + 1, :, past + query - added:past + query, :]), dim=-2)
                    pair.append(compact)
                    copied += compact.numel() * compact.element_size()
                layers.append(tuple(pair))
            self.rows[identity] = layers
        return copied

    def release(self, identity):
        self.rows.pop(identity, None)

    def bytes(self):
        return sum(t.numel() * t.element_size() for layers in self.rows.values() for pair in layers for t in pair)

    def clear(self):
        self.rows.clear()
