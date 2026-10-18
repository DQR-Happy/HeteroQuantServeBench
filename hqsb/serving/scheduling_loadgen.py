"""Actual timed in-process ingress replay, independent of GPU completion.

The receiver queue is bounded. Rejected sends are recorded without retries;
closed-loop uses completion feedback and does not claim identical arrivals.
"""
import queue
import threading
import time


class WorkloadSender:
    def __init__(self, items, mode, concurrency, *, lag_limit_ms=20):
        self.items, self.mode, self.concurrency = items, mode, concurrency
        self.queue = queue.Queue(maxsize=256)
        self.stop = threading.Event()
        self.condition = threading.Condition()
        self.outstanding = set()
        self.records, self.lag_limit_ms = [], lag_limit_ms
        self.error = None
        self.thread = threading.Thread(target=self._run, name="hqsb-l8-loadgen", daemon=True)

    def start(self):
        self.origin = time.monotonic()
        self.thread.start()

    def complete(self, identity):
        with self.condition:
            self.outstanding.discard(identity)
            self.condition.notify_all()

    def _run(self):
        try:
            for item in sorted(self.items, key=lambda x: (x["arrival_ms"], x["id"])):
                planned = self.origin + item["arrival_ms"] / 1000
                if self.stop.wait(max(0, planned - time.monotonic())):
                    return
                if self.mode == "closed_loop":
                    with self.condition:
                        while len(self.outstanding) >= self.concurrency and not self.stop.is_set():
                            self.condition.wait(0.05)
                if self.stop.is_set():
                    return
                sent = time.monotonic()
                with self.condition:
                    self.outstanding.add(item["id"])
                accepted = True
                try:
                    self.queue.put_nowait(("submit", {**item, "sent": sent, "planned": planned}))
                except queue.Full:
                    accepted = False
                    self.complete(item["id"])
                self.records.append({"id": item["id"], "planned": planned, "sent": sent,
                                     "sender_lag_ms": (sent - planned) * 1000, "enqueued": accepted})
        except Exception as exc:
            self.error = str(exc)[:400]
        finally:
            # Separate flag, not an end marker that could overtake queued items.
            self.finished = True

    def drain(self):
        out = []
        for _ in range(256):
            try:
                out.append(self.queue.get_nowait())
            except queue.Empty:
                break
        if getattr(self, "finished", False) and self.queue.empty():
            out.append(("seal", None))
        return out

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(2)
        if self.thread.is_alive():
            raise RuntimeError("LOADGEN_CLEANUP_UNKNOWN")
        lags = sorted(r["sender_lag_ms"] for r in self.records)
        p95 = lags[min(len(lags) - 1, int(len(lags) * .95))] if lags else None
        drops = sum(not r["enqueued"] for r in self.records)
        valid = not self.error and not drops and len(self.records) == len(self.items) and (
            self.mode != "open_loop" or (p95 is not None and p95 <= self.lag_limit_ms))
        return {"mode": self.mode, "validity": "VALID" if valid else "LOADGEN_INVALID",
                "sent": len(self.records), "offered": len(self.items), "dropped": drops,
                "sender_lag_p95_ms": p95, "error": self.error,
                "same_arrivals_across_candidates": self.mode == "open_loop"}
