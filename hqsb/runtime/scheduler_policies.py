"""Deterministic ordering and token allocation; no tensor or IO ownership."""
from hqsb.core.contracts.scheduling import Step


def ordered(sequences, policy, now, aging_ms, service, weights=None):
    if policy == "priority_aging":
        return sorted(sequences, key=lambda r: (
            -(r.priority + int((now - r.received) * 1000 / aging_ms)), r.received, r.id))
    if policy == "tenant_fair":
        return sorted(sequences, key=lambda r: (service.get(r.tenant, 0) / (weights or {}).get(r.tenant, 1), r.received, r.id))
    return sorted(sequences, key=lambda r: (r.received, r.id))


def plan(sequences, limits, now, service, mode, tick=0):
    budget = limits.max_batched_tokens
    selected = []
    rows = ordered(sequences, limits.policy, now, limits.aging_ms, service, limits.tenant_weights)
    row_limit = 1 if mode == "serial" else limits.max_batch_sequences

    def allocate(candidates, available, slots):
        nonlocal budget
        for r in candidates:
            if slots <= 0 or available <= 0:
                break
            # Teacher force committed history after preemption; never re-emit it.
            history = r.history
            remaining = len(history) - r.consumed
            chunk = min(remaining, limits.prefill_chunk_tokens, available) if limits.chunked_prefill else remaining
            if chunk > available:
                continue
            end = r.consumed + chunk
            selected.append(Step(r.id, r.consumed, tuple(history[r.consumed:end]),
                                 end == len(history), "decode" if r.output and remaining == 1 else "prefill"))
            budget -= chunk
            available -= chunk
            slots -= 1

    if limits.phase_policy == "decode_first":
        decode = [r for r in rows if r.output and len(r.history) - r.consumed == 1]
        prefill = [r for r in rows if r not in decode]
        if decode and prefill:
            if row_limit == 1 or budget == 1:
                # One physical row cannot run two phases together. Alternate so
                # a stream of decode requests cannot starve waiting prefill.
                allocate(prefill if tick % 2 else decode, budget, row_limit)
            else:
                quota = min(limits.prefill_quota_tokens, budget - 1)
                if not limits.chunked_prefill:
                    quota = min(len(prefill[0].history) - prefill[0].consumed, budget)
                allocate(decode, budget - quota, row_limit - 1)
                allocate(prefill, budget, row_limit - len(selected))
                chosen = {s.request_id for s in selected}
                allocate([r for r in decode if r.id not in chosen], budget, row_limit - len(selected))
        else:
            allocate(rows, budget, row_limit)
    else:
        allocate(rows, budget, row_limit)
    return selected
