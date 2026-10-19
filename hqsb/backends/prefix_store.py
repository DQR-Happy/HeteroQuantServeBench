"""Bounded immutable GPU prefix snapshots, owned by one serial model worker."""
from collections import OrderedDict


class PrefixStore:
    def __init__(self):
        self.entries = OrderedDict()
        self.retained_bytes = 0
        self.evictions = 0

    def trim(self, budget, max_entries):
        while self.entries and (self.retained_bytes > budget or len(self.entries) > max_entries):
            _, (_, _, size) = self.entries.popitem(last=False)
            self.retained_bytes -= size
            self.evictions += 1

    def lookup(self, identity, tokens):
        best = None
        for key, (stored, pairs, size) in self.entries.items():
            if key[0] == identity and len(stored) < len(tokens) and tokens[:len(stored)] == stored:
                if best is None or len(stored) > len(best[1]):
                    best = (key, stored, pairs, size)
        if best is None:
            return None
        self.entries.move_to_end(best[0])
        return best[1], [(k.clone(), v.clone()) for k, v in best[2]], best[3]

    def publish(self, identity, tokens, cache, budget, max_entries):
        if not tokens:
            return 0
        count = len(tokens)
        pairs = [(layer.keys[..., :count, :], layer.values[..., :count, :]) for layer in cache.layers]
        size = sum(t.numel() * t.element_size() for pair in pairs for t in pair)
        if size > budget:
            return 0
        key = (identity, tokens)
        if key in self.entries:
            self.entries.move_to_end(key)
            return 0
        # Evict before cloning: retention never exceeds its byte/entry budget.
        self.trim(budget - size, max_entries - 1)
        self.entries[key] = (tokens, [(k.detach().clone(), v.detach().clone()) for k, v in pairs], size)
        self.retained_bytes += size
        return size

    def clear(self):
        size, entries = self.retained_bytes, len(self.entries)
        self.entries.clear()
        self.retained_bytes = 0
        return {"released_bytes": size, "released_entries": entries}
