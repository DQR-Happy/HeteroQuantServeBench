import time
from types import SimpleNamespace

import pytest

from hqsb.core.contracts.scheduling import SchedulingLimits, SchedulingExperiment, Sequence
from hqsb.runtime.kv_manager import KVManager
from hqsb.runtime.live_engine import LiveEngine
from hqsb.serving.scheduling_loadgen import WorkloadSender


class Executor:
    context_limit = 4096

    def __init__(self):
        self.rows = {}
        self.batches = []

    def execute(self, steps, identity):
        self.batches.append([s.request_id for s in steps])
        values = []
        for s in steps:
            old = self.rows.get(s.request_id, [])
            assert len(old) == s.start
            self.rows[s.request_id] = old + list(s.token_ids)
            values.append(sum(self.rows[s.request_id]) % 97)
        return SimpleNamespace(tokens=values, receipt={"forward_id": identity, "members": self.batches[-1]})

    def prepare(self, steps):
        return None

    def release(self, identity):
        self.rows.pop(identity, None)

    def close(self):
        self.rows.clear()
        return {"device_kv_rows": 0}

    def quiesce(self):
        pass


def test_generation_fences_capacity_and_monotonicity():
    kv = KVManager(10)
    one = kv.reserve("a", 8)
    assert kv.reserve("b", 3) is None
    kv.commit("a", one.generation, 4)
    with pytest.raises(ValueError):
        kv.commit("a", one.generation, 3)
    kv.release("a", one.generation)
    two = kv.reserve("a", 8)
    with pytest.raises(ValueError):
        kv.release("a", one.generation)
    assert two.generation > one.generation


@pytest.mark.parametrize("mode", ["serial", "static", "continuous"])
def test_real_engine_membership_and_token_conservation(mode):
    ex, events = Executor(), []
    limits = SchedulingLimits(max_active_sequences=3, max_batch_sequences=3, prefill_chunk_tokens=2)
    engine = LiveEngine(ex, limits, mode, lambda k, b: events.append((k, b)), "test")
    clock = time.monotonic()
    a = Sequence("a", [1, 2, 3], 6, clock, clock + 10)
    b = Sequence("b", [8], 2, clock + .001, clock + 10)
    c = Sequence("c", [3, 9], 3, clock + .002, clock + 10)
    iteration = 0

    def receive():
        nonlocal iteration
        iteration += 1
        return [("submit", a), ("submit", b)] if iteration == 1 else [("submit", c), ("seal", None)] if iteration == 3 else []

    rows = engine.run(receive, lambda: False, clock + 10, lambda _: None)
    assert not engine.kv.leases and not ex.rows
    for row in rows:
        assert row.reason == "length" and len(row.output) == row.limit
        history = row.input_ids[:]
        expected = []
        for _ in range(row.limit):
            token = sum(history) % 97
            expected.append(token)
            history.append(token)
        assert row.output == expected
        assert row.consumed == len(row.input_ids) + len(row.output) - 1
    if mode == "continuous":
        assert any("a" in batch and "c" in batch for batch in ex.batches)
        assert "c" in engine.confirmed_join_ids
    if mode == "static":
        assert not any("a" in batch and "c" in batch for batch in ex.batches)
        assert not engine.confirmed_join_ids
    if mode == "serial":
        assert all(len(batch) == 1 for batch in ex.batches)


def test_cancel_deadline_rejection_and_recompute_keep_committed_output():
    ex = Executor()
    limits = SchedulingLimits(max_active_sequences=1, max_batch_sequences=1,
        kv_capacity_tokens=32, prefill_chunk_tokens=2, preemption="recompute")
    engine = LiveEngine(ex, limits, "continuous", lambda *_: None, "test")
    clock = time.monotonic()
    a = Sequence("a", [1, 2, 3], 6, clock, clock + 10)
    b = Sequence("b", [8], 2, clock + .001, clock + 10, priority=9)
    reject = Sequence("reject", [1] * 40, 1, clock, clock + 10)
    expired = Sequence("expired", [1], 1, clock, clock - 1)
    iteration = 0

    def receive():
        nonlocal iteration
        iteration += 1
        return [("submit", a)] if iteration == 1 else [("submit", b), ("submit", reject), ("submit", expired), ("seal", None)] if iteration == 4 else []

    engine.run(receive, lambda: False, clock + 10, lambda _: None)
    assert reject.reason == "rejected" and expired.reason == "timed_out"
    assert a.preemptions == 1 and a.recomputed > 0
    history, expected = a.input_ids[:], []
    for _ in range(6):
        token = sum(history) % 97
        expected.append(token)
        history.append(token)
    assert a.output == expected and not engine.kv.leases


def test_open_loop_sender_does_not_wait_for_completions():
    sender = WorkloadSender([{"id": str(i), "arrival_ms": i * 2} for i in range(5)], "open_loop", 1)
    sender.start()
    sender.thread.join(1)
    messages = sender.drain()
    assert sum(k == "submit" for k, _ in messages) == 5
    assert sender.close()["validity"] == "VALID"


def test_closed_loop_obeys_feedback_and_shutdown():
    sender = WorkloadSender([{"id": str(i), "arrival_ms": 0} for i in range(3)], "closed_loop", 1)
    sender.start()
    time.sleep(.01)
    assert len(sender.records) == 1
    sender.complete("0")
    time.sleep(.01)
    assert len(sender.records) == 2
    sender.complete("1")
    sender.thread.join(1)
    assert sender.close()["validity"] == "VALID"


def test_contract_rejects_duplicate_ids_and_invalid_chunk():
    with pytest.raises(ValueError):
        SchedulingExperiment(deployment_id="a", expected_epoch="a", requests=[{"id": "a", "prompt": "x"}] * 2)
    with pytest.raises(ValueError):
        SchedulingExperiment(deployment_id="a", expected_epoch="a", requests=[{"id": "a", "prompt": "x"}],
                             limits={"max_batched_tokens": 1, "prefill_chunk_tokens": 2})


def test_decode_first_keeps_prefill_quota_and_single_row_progress():
    from hqsb.runtime.scheduler_policies import plan, ordered
    now = time.monotonic()
    decode = Sequence("decode", [1, 2], 32, now, now + 10, tenant="a", output=[3], consumed=2)
    prefill = Sequence("prefill", list(range(20)), 8, now, now + 10, tenant="b")
    limits = SchedulingLimits(phase_policy="decode_first", max_batched_tokens=8,
        prefill_chunk_tokens=8, prefill_quota_tokens=3)
    steps = plan([decode, prefill], limits, now, {}, "continuous")
    assert {s.request_id for s in steps} == {"decode", "prefill"}
    assert sum(len(s.token_ids) for s in steps) <= 8
    one = limits.model_copy(update={"max_batch_sequences": 1})
    assert plan([decode, prefill], one, now, {}, "continuous", 0)[0].request_id == "decode"
    assert plan([decode, prefill], one, now, {}, "continuous", 1)[0].request_id == "prefill"
    assert ordered([decode, prefill], "tenant_fair", now, 100, {"a": 10, "b": 8}, {"a": 2})[0].id == "decode"
    aged = Sequence("aged", [1], 1, now - 5, now + 10, priority=0)
    fresh = Sequence("fresh", [1], 1, now, now + 10, priority=9)
    assert ordered([fresh, aged], "priority_aging", now, 100, {})[0].id == "aged"


def test_failed_executor_is_quiesced_before_reclaim():
    class Failed(Executor):
        def __init__(self):
            super().__init__()
            self.safe = False

        def execute(self, steps, identity):
            raise RuntimeError("submitted device failure")

        def quiesce(self):
            self.safe = True

        def release(self, identity):
            assert self.safe
            super().release(identity)

    ex = Failed()
    now = time.monotonic()
    engine = LiveEngine(ex, SchedulingLimits(), "continuous", lambda *_: None, "failure")
    with pytest.raises(RuntimeError, match="device failure"):
        engine.run(lambda: [("submit", Sequence("a", [1], 1, now, now + 10)), ("seal", None)],
                   lambda: False, now + 10, lambda _: None)
    assert engine.close()["device_kv_rows"] == 0 and not engine.kv.leases
