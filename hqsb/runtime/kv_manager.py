"""Full-capacity reservations and generation-checked logical ownership.

This is not a paged device KV allocator. Device storage lives in the executor.
"""
from dataclasses import dataclass


@dataclass
class Lease:
    generation: int
    capacity: int
    valid: int = 0


class KVManager:
    def __init__(self, capacity):
        self.capacity, self.generation, self.peak = capacity, 0, 0
        self.leases = {}

    @property
    def reserved(self):
        return sum(x.capacity for x in self.leases.values())

    def reserve(self, identity, capacity):
        if identity in self.leases:
            raise ValueError("DUPLICATE_KV_LEASE")
        if capacity > self.capacity - self.reserved:
            return None
        self.generation += 1
        lease = Lease(self.generation, capacity)
        self.leases[identity] = lease
        self.peak = max(self.peak, self.reserved)
        return lease

    def commit(self, identity, generation, valid):
        lease = self.leases[identity]
        if generation != lease.generation or not lease.valid <= valid <= lease.capacity:
            raise ValueError("STALE_OR_INVALID_KV_COMMIT")
        lease.valid = valid

    def release(self, identity, generation):
        if self.leases[identity].generation != generation:
            raise ValueError("STALE_KV_RELEASE")
        return self.leases.pop(identity)
