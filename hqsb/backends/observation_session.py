"""One worker-owned capture scope; metadata leaves it, Tensor owners never do."""

import threading
import time

from hqsb.backends.tensor_memory import cache_inventory, parameter_inventory
from hqsb.observation.contracts import ObservationPlan
from hqsb.observation.procfs import (
    page_sample,
    process_key,
    sample_process,
    vma_snapshot,
)
from hqsb.observation.storage import EvidenceWriter


class CaptureSession:
    def __init__(self, torch, directory, context, mode):
        self.torch, self.plan = torch, ObservationPlan(mode=mode)
        self.key = process_key()
        context = {**context, "process": self.key, "plan": self.plan.as_dict()}
        self.writer = EvidenceWriter(
            directory,
            context,
            max_records=self.plan.max_records,
            max_bytes=self.plan.max_bytes,
        )
        self.stop_event = threading.Event()
        self.problems, self.history = [], False
        self.closed = False
        self.thread = threading.Thread(
            target=self._sample_loop, daemon=True, name="hqsb-os-observer"
        )

    def emit(self, kind, payload):
        receipt = self.writer.offer(kind, payload)
        if receipt != "accepted" and receipt not in self.problems:
            self.problems.append(receipt)

    def _problem(self, source, exc):
        reason = f"{source}: {type(exc).__name__}: {exc}"[:400]
        self.problems.append(reason)
        self.emit(
            "source_status",
            {"source": source, "availability": "not_collected", "reason": reason},
        )

    def start(self, model, inventory=None):
        ledger = (
            parameter_inventory(model, include_addresses=True)
            if self.plan.mode == "memory"
            else inventory or parameter_inventory(model)
        )
        self.emit("tensor_inventory", {"label": "model_loaded", "inventory": ledger})
        if self.plan.mode == "memory":
            try:
                # Version-sensitive API verified against the vendor wheel on target.
                self.torch.cuda.memory._record_memory_history(
                    enabled="all", context=None, stacks="python", max_entries=100_000
                )
                self.history = True
                self.allocator_snapshot("before_request")
            except (RuntimeError, AttributeError, TypeError) as exc:
                self._problem("allocator_history", exc)
        self.boundary("before_request", detailed=True)
        self.thread.start()

    def _sample_loop(self):
        while not self.stop_event.wait(self.plan.os_interval_ms / 1000):
            try:
                self.emit("os_sample", sample_process(self.key, with_rollup=False))
            except (OSError, ValueError) as exc:
                self._problem("os_periodic", exc)
                break

    def boundary(self, label, *, cache=None, step=None, detailed=False):
        self.emit(
            "boundary",
            {
                "label": label,
                "step": step,
                "timestamp_ns": str(time.monotonic_ns()),
                "clock_domain": "host_monotonic",
            },
        )
        try:
            self.emit("os_sample", {"label": label, **sample_process(self.key)})
            if detailed and self.plan.mode in {"memory", "pages"}:
                vmas = vma_snapshot(self.key)
                self.emit("vma_snapshot", {"label": label, **vmas})
                if self.plan.mode == "pages":
                    self.emit(
                        "page_sample",
                        {
                            "label": label,
                            **page_sample(
                                self.key, vmas["vmas"], max_pages=self.plan.max_pages
                            ),
                        },
                    )
        except (OSError, ValueError) as exc:
            self._problem("os_boundary", exc)
        if cache is not None:
            self.emit(
                "kv_inventory",
                {
                    "label": label,
                    "step": step,
                    "inventory": cache_inventory(
                        cache, include_addresses=self.plan.mode == "memory"
                    ),
                },
            )

    def allocator_snapshot(self, label):
        if not self.history:
            return
        try:
            snapshot = self.torch.cuda.memory._snapshot()
            # Version 2.5 trace sequence does not supply calibrated timestamps.
            traces = snapshot.get("device_traces", [])
            overwritten = any(len(t) >= 100_000 for t in traces)
            if (
                overwritten
                and "ALLOCATOR_HISTORY_RING_OVERWRITTEN" not in self.problems
            ):
                self.problems.append("ALLOCATOR_HISTORY_RING_OVERWRITTEN")
            self.emit(
                "allocator_snapshot",
                {
                    "label": label,
                    "snapshot": snapshot,
                    "time_source": "snapshot_anchor_only",
                    "allocation_stacks": "not_collected_to_bound_capture_memory",
                    "history_prefix_may_be_overwritten": overwritten,
                    "timestamp_ns": str(time.monotonic_ns()),
                    "scope": "torch_allocator_not_all_driver_allocations",
                },
            )
        except (RuntimeError, AttributeError, TypeError) as exc:
            self._problem("allocator_snapshot", exc)

    def finish(self, execution_status, cleanup_status):
        if self.closed:
            return self.report
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.problems.append("OS_SAMPLER_CLEANUP_UNKNOWN")
        self.allocator_snapshot("after_cleanup")
        if self.history:
            try:
                self.torch.cuda.memory._record_memory_history(enabled=None)
            except (RuntimeError, AttributeError, TypeError) as exc:
                self._problem("allocator_cleanup", exc)
        self.emit(
            "coverage",
            {
                "problems": self.problems,
                "gpu_page_tables": "unsupported",
                "dram_counters": "not_collected",
                "physical_page_mapping": "not_collected",
            },
        )
        self.report = self.writer.close(
            execution_status=execution_status,
            cleanup_status=cleanup_status,
            source_problems=self.problems,
        )
        self.closed = True
        return self.report
