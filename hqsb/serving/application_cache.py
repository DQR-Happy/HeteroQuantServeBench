"""Bounded exact-result LRU; ownership is one L10 variant, tenants never share."""

from collections import OrderedDict
import copy
import json
import threading
import time
from hqsb.core.optimization import fingerprint


def result_key(item, policy, snapshot, fixed):
    return fingerprint(
        {
            "tenant": item.tenant,
            "prompt": item.prompt,
            "messages": [m.model_dump() for m in item.messages],
            "output": item.max_output_tokens,
            "policy": policy.model_dump(),
            "runtime": snapshot["identity"],
            "epoch": snapshot["epoch"],
            "sampling": "greedy",
            "fixed_output_length": fixed,
        }
    )


class ResultCache:
    def __init__(self, policy):
        self.policy, self.lock = policy, threading.RLock()
        self.entries, self.bytes = OrderedDict(), 0
        self.hits, self.misses, self.evictions = 0, 0, 0

    def _lookup(self, key):
        row = self.entries.get(key)
        if row and time.monotonic() >= row[0]:
            self.bytes -= self.entries.pop(key)[2]
            self.evictions += 1
            return None
        return row

    def contains(self, key):
        with self.lock:
            return self._lookup(key) is not None

    def get(self, key):
        with self.lock:
            row = self._lookup(key)
            if row:
                self.hits += 1
                self.entries.move_to_end(key)
                return copy.deepcopy(row[1])
            self.misses += 1
            return None

    def put(self, key, value):
        size = len(json.dumps(value, ensure_ascii=False).encode())
        with self.lock:
            if size > self.policy.cache_max_bytes:
                return False
            old = self.entries.pop(key, None)
            if old:
                self.bytes -= old[2]
            while self.entries and (
                len(self.entries) >= self.policy.cache_max_entries
                or self.bytes + size > self.policy.cache_max_bytes
            ):
                self.bytes -= self.entries.popitem(last=False)[1][2]
                self.evictions += 1
            self.entries[key] = (
                time.monotonic() + self.policy.cache_ttl_s,
                copy.deepcopy(value),
                size,
            )
            self.bytes += size
            return True

    def summary(self):
        with self.lock:
            return {
                "entries": len(self.entries),
                "bytes": self.bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "scope": "variant-local CPU serialized-result cache; not KV or GPU memory",
            }
