"""Single-writer live scheduler. No HTTP, tokenizer or tensor ownership.

Executor calls return only after device completion. A failed submitted forward
invalidates the session; it is never blindly replayed as isolated row failures.
"""
import time
from collections import defaultdict

from hqsb.runtime.kv_manager import KVManager
from hqsb.runtime.scheduler_policies import ordered, plan


class LiveEngine:
    def __init__(self, executor, limits, mode, emit, session_id, eos_ids=()):
        self.executor, self.limits, self.mode, self.emit = executor, limits, mode, emit
        self.session_id, self.eos_ids = session_id, set(eos_ids)
        self.rows, self.kv = {}, KVManager(limits.kv_capacity_tokens)
        self.service = defaultdict(int)
        self.tick = self.forwards = self.preemptions = 0
        self.live_join_ids, self.confirmed_join_ids, self.executed_ids = set(), set(), set()

    def event(self, kind, **body):
        self.emit(kind, {"session_id": self.session_id, "time": time.monotonic(), **body})

    def release(self, row, cause):
        if row.id in self.kv.leases:
            lease = self.kv.leases[row.id]
            self.executor.release(row.id)
            self.kv.release(row.id, lease.generation)
            self.event("kv.release", request_id=row.id, generation=lease.generation,
                       valid=lease.valid, cause=cause, completion="executor_completed")

    def finish(self, row, reason, error=None):
        row.state, row.reason, row.error, row.ended = "terminal", reason, error, time.monotonic()
        self.release(row, reason)
        self.event("request.terminal", request_id=row.id, reason=reason, error=error,
                   output_tokens=len(row.output), consumed=row.consumed)

    def accept(self, row):
        if row.id in self.rows:
            self.event("request.refused", request_id=row.id, reason="DUPLICATE_ID")
            return
        self.rows[row.id] = row
        self.event("request.accepted", request_id=row.id, input_tokens=len(row.input_ids),
                   output_limit=row.limit, received=row.received, tenant=row.tenant, priority=row.priority)
        needed = len(row.input_ids) + row.limit - 1
        if needed > self.kv.capacity:
            self.finish(row, "rejected", "KV_CAPACITY")
        elif not self.limits.chunked_prefill and len(row.input_ids) > self.limits.max_batched_tokens:
            self.finish(row, "rejected", "FULL_PREFILL_TOKEN_BUDGET")
        elif len(row.input_ids) + row.limit > self.executor.context_limit:
            self.finish(row, "rejected", "CONTEXT_LIMIT")
        elif len(row.input_ids) + row.limit - 1 > self.limits.max_padded_positions:
            self.finish(row, "rejected", "PADDED_POSITION_BUDGET")

    def run(self, receive, cancelled, deadline, on_terminal):
        sealed = False
        notified = set()
        while True:
            self.tick += 1
            planning_start = time.monotonic()
            for kind, value in receive():
                if kind == "submit":
                    self.accept(value)
                elif kind == "seal":
                    sealed = True
                elif kind == "cancel":
                    row = self.rows.get(value)
                    if row and row.state != "terminal":
                        self.finish(row, "cancelled")
                    else:
                        self.event("control.refused", request_id=value, reason="UNKNOWN_OR_TERMINAL")
            now = time.monotonic()
            stopping = cancelled() or now >= deadline
            for row in self.rows.values():
                if row.state == "terminal":
                    if row.id not in notified:
                        notified.add(row.id)
                        on_terminal(row.id)
                    continue
                if stopping or (row.cancel_at is not None and now >= row.cancel_at):
                    self.finish(row, "timed_out" if now >= deadline else "cancelled")
                elif now >= row.deadline:
                    self.finish(row, "timed_out")
            if stopping:
                break
            live = [r for r in self.rows.values() if r.state != "terminal"]
            if sealed and not live:
                break
            active = [r for r in live if r.state == "active"]
            waiting = ordered([r for r in live if r.state == "waiting"], self.limits.policy,
                              now, self.limits.aging_ms, self.service, self.limits.tenant_weights)
            # Controlled priority preemption only at a completed forward boundary.
            if (self.mode == "continuous" and waiting and active and self.limits.preemption == "recompute"
                    and self.preemptions < self.limits.max_preemptions):
                victim = min(active, key=lambda r: (r.priority, -r.received))
                if waiting[0].priority > victim.priority:
                    self.release(victim, "preempted")
                    victim.state, victim.consumed = "waiting", 0
                    victim.preemptions += 1
                    self.preemptions += 1
                    active.remove(victim)
                    self.event("request.preempted", request_id=victim.id, committed_output=len(victim.output))
            cap = 1 if self.mode == "serial" else self.limits.max_active_sequences
            if self.mode != "static" or not active:
                for row in waiting:
                    if len(active) >= cap:
                        break
                    lease = self.kv.reserve(row.id, len(row.input_ids) + row.limit - 1)
                    if lease is None:
                        continue
                    if any(r.id in self.executed_ids for r in active) and row.admitted is None:
                        self.live_join_ids.add(row.id)
                    row.state, row.admitted = "active", row.admitted or now
                    active.append(row)
                    self.event("kv.reserve", request_id=row.id, generation=lease.generation,
                               capacity=lease.capacity, valid=0, reserved_total=self.kv.reserved)
            steps = plan(active, self.limits, now, self.service, self.mode, self.tick)
            # Split overly padded groups without changing request order. A single
            # row has already passed admission; shrinking a batch always progresses.
            while len(steps) > 1 and len(steps) * (max(s.start for s in steps) + max(len(s.token_ids) for s in steps)) > self.limits.max_padded_positions:
                steps.pop()
            while steps:
                refusal = self.executor.prepare(steps)
                if refusal is None:
                    break
                if len(steps) > 1:
                    steps.pop()
                else:
                    self.finish(self.rows[steps[0].request_id], "rejected", refusal)
                    steps.clear()
            if not steps:
                if any(r.state == "active" for r in active):
                    raise RuntimeError("SCHEDULER_NO_PROGRESS")
                time.sleep(0.001)
                continue
            self.forwards += 1
            forward_id = f"{self.session_id}:f{self.forwards}"
            self.event("scheduler.plan", tick_id=self.tick, forward_id=forward_id,
                       waiting=[r.id for r in waiting], selected=[s.request_id for s in steps],
                       reserved_tokens=self.kv.reserved, policy=self.limits.policy,
                       planning_host_ms=(time.monotonic() - planning_start) * 1000,
                       valid_token_budget_used=sum(len(s.token_ids) for s in steps),
                       waiting_reasons={r.id: "ACTIVE_SEQUENCE_LIMIT" if len(active) >= cap else "KV_RESERVATION_CAPACITY"
                                        for r in waiting if r.state == "waiting"},
                       deferred_active=[r.id for r in active if r.id not in {s.request_id for s in steps}],
                       leases={s.request_id: self.kv.leases[s.request_id].generation for s in steps})
            result = self.executor.execute(steps, forward_id)
            self.event("forward", tick_id=self.tick, **result.receipt)
            members = {s.request_id for s in steps}
            self.confirmed_join_ids.update(i for i in members & self.live_join_ids
                                           if (members - {i}) & self.executed_ids)
            self.executed_ids.update(members)
            ready = time.monotonic()
            for row_index, (step, token) in enumerate(zip(steps, result.tokens)):
                row = self.rows[step.request_id]
                row.max_unscheduled_ms = max(row.max_unscheduled_ms, (ready - (row.last_scheduled or row.received)) * 1000)
                row.last_scheduled = ready
                old_target = row.target
                if row.preemptions:
                    row.recomputed += max(0, min(step.start + len(step.token_ids), old_target) - step.start)
                row.consumed += len(step.token_ids)
                lease = self.kv.leases[row.id]
                self.kv.commit(row.id, lease.generation, row.consumed)
                self.service[row.tenant] += len(step.token_ids)
                self.event("kv.commit", request_id=row.id, forward_id=forward_id,
                           generation=lease.generation, valid=row.consumed)
                if ready >= row.deadline or (row.cancel_at is not None and ready >= row.cancel_at):
                    self.finish(row, "timed_out" if ready >= row.deadline else "cancelled")
                    continue
                if step.emit:
                    row.output.append(token)
                    row.times.append(ready)
                    if "selected_logprobs" in result.receipt:
                        row.logprobs.append(result.receipt["selected_logprobs"][row_index])
                        row.margins.append(result.receipt["top2_logit_margins"][row_index])
                    self.event("token", request_id=row.id, forward_id=forward_id,
                               token_id=token, output_index=len(row.output), ready=ready)
                    if len(row.output) >= row.limit or (token in self.eos_ids and not row.fixed):
                        self.finish(row, "length" if len(row.output) >= row.limit else "stop")
        for row in self.rows.values():
            if row.state != "terminal":
                self.finish(row, "cancelled")
        if self.kv.leases:
            raise RuntimeError("KV_LEASE_LEAK")
        return list(self.rows.values())

    def close(self):
        # Also used after a failed forward; executor synchronizes before clearing.
        self.executor.quiesce()
        for row in self.rows.values():
            self.release(row, "session_cleanup")
        return self.executor.close()
