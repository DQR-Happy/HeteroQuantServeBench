"""Bounded worker observations; importing this module never imports torch.

Host spans describe dispatch and waiting, not device execution. Operator capture
is opt-in and kept separate from request observations so missing CUPTI support
does not turn successful inference into a failed request.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path


def host_memory() -> dict:
    """Read availability without probing or initializing an accelerator."""
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            if name in {"MemTotal", "MemAvailable"}:
                values[name] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return {
        "host_total_bytes": values.get("MemTotal"),
        "host_available_bytes": values.get("MemAvailable"),
    }


def _device_milliseconds(event, field):
    """Read current or older PyTorch device fields without inventing zeroes."""
    value = getattr(event, field, None)
    if value is None:
        value = getattr(event, field.replace("device", "cuda"), None)
    return value / 1000 if value is not None else None


class ObservationRecorder:
    """An append-only, per-request record bounded by the output-token limit."""

    def __init__(self, mode="basic", *, start=None, clock=None, execution=None):
        if mode not in {"off", "basic", "operators"}:
            raise ValueError("Unknown observation mode")
        self.mode = mode
        self.clock = clock or time.monotonic
        self.start = self.clock() if start is None else start
        self.data = {
            "schema_version": "1.0",
            "mode": mode,
            "clock_domain": "worker_monotonic",
            "time_origin": "request_received_by_worker",
            "phases": [],
            "tokens": [],
            "memory_snapshots": [],
            "execution": execution or {},
            "limitations": [
                "Host spans include dispatch and synchronization waits; they are not GPU kernel durations.",
                "Nested phases and operator totals must not be summed as wall-clock latency.",
                "Observation overhead is not calibrated against an unobserved baseline.",
            ],
        }

    def now_ms(self):
        return (self.clock() - self.start) * 1000

    @contextmanager
    def phase(self, name):
        if self.mode == "off":
            yield
            return
        start = self.now_ms()
        try:
            yield
        finally:
            self.data["phases"].append(
                {
                    "name": name,
                    "start_ms": start,
                    "duration_ms": max(0.0, self.now_ms() - start),
                    "source": "host_monotonic",
                }
            )

    def token(self, **fields):
        if self.mode != "off":
            self.data["tokens"].append({**fields, "source": "host_monotonic"})

    def snapshot(self, label, allocator):
        if self.mode == "off":
            return
        try:
            memory = allocator()
        except Exception as exc:
            memory = {"unavailable_reason": f"{type(exc).__name__}: {exc}"[:300]}
        self.data["memory_snapshots"].append(
            {
                "label": label,
                "time": time.time(),
                "start_ms": self.now_ms(),
                **host_memory(),
                "allocated_bytes": memory.get("allocated_bytes"),
                "reserved_bytes": memory.get("reserved_bytes"),
                "peak_allocated_bytes": memory.get("peak_allocated_bytes"),
                "scope": "worker_allocator_and_worker_host",
                "source": "torch_cuda_allocator + procfs",
                **(
                    {"unavailable_reason": memory["unavailable_reason"]}
                    if "unavailable_reason" in memory
                    else {}
                ),
            }
        )


class OperatorCapture:
    """Collect one prefill and at most eight decode steps, without op barriers.

    Only server-selected absolute output directories are accepted. A trace may
    be large even with a step bound; byte limits here cap retained artifacts, not
    the profiler's transient native buffers.
    """

    MAX_DECODE_STEPS = 8
    MAX_OPERATORS = 300
    MAX_TRACE_BYTES = 128 * 1024 * 1024

    def __init__(self, torch, *, enabled=False, capture_dir=None, capture_id=None):
        self.torch = torch
        self.enabled = enabled
        self.profiler = None
        self.active = False
        self.finished = False
        self.steps = 0
        self._hooks = []
        self._ranges = []
        self._thread_id = None
        self.directory = Path(capture_dir) if capture_dir else None
        if self.directory is not None and not self.directory.is_absolute():
            raise ValueError("capture_dir must be an absolute server-selected path")
        self.summary = {
            "status": "not_requested" if not enabled else "pending",
            "source": "torch.profiler",
            "capture_id": capture_id,
            "activities_requested": [],
            "activities": [],
            "operators": [],
            "trace_available": False,
            "coverage": {
                "prefill": False,
                "decode_steps": 0,
                "max_decode_steps": self.MAX_DECODE_STEPS,
                "scope": "first_prefill_and_bounded_decode_window",
            },
            "limitations": [],
        }

    def _problem(self, message):
        self.summary["limitations"].append(str(message)[:500])

    def start(self):
        if not self.enabled or self.active or self.finished:
            return
        try:
            api = self.torch.profiler
            cpu, cuda = api.ProfilerActivity.CPU, api.ProfilerActivity.CUDA
            activities = [cpu]
            if cuda in api.supported_activities():
                activities.append(cuda)
            else:
                self._problem(
                    "CUDA profiler activity is unsupported; collecting CPU only."
                )
            self.summary["activities_requested"] = [
                "CPU" if item == cpu else "CUDA" for item in activities
            ]
            try:
                self._start_with(activities)
            except Exception as exc:
                self._discard_failed_profiler()
                if cuda not in activities:
                    raise
                self._problem(f"CUDA capture startup failed; retrying CPU: {exc}")
                self._start_with([cpu])
            self.summary["status"] = "collecting"
        except Exception as exc:
            self._discard_failed_profiler()
            self.summary["status"] = "unavailable"
            self._problem(f"Profiler unavailable: {type(exc).__name__}: {exc}")
            self.finished = True

    def _start_with(self, activities):
        self.profiler = self.torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        self.profiler.start()
        self.active = True

    def _discard_failed_profiler(self):
        if self.profiler is not None:
            try:
                self.profiler.stop()
            except Exception:
                pass
        self.profiler = None
        self.active = False

    def region(self, name):
        if self.active:
            return self.torch.profiler.record_function(name)
        return nullcontext()

    def attach_modules(self, model):
        """Annotate the first decoder layer only; never mutate its computation."""
        if not self.active:
            return
        modules = list(model.named_modules())
        roots = [name for name, _ in modules if name.endswith("layers.0")]
        if not roots:
            self._problem("Decoder layers.0 not found; module ranges are unavailable.")
            return
        root = roots[0]
        selected = [
            (name, module)
            for name, module in modules
            if name == root or name.startswith(root + ".")
        ]
        self._thread_id = threading.get_ident()
        self.summary["coverage"]["module_scope"] = "first_decoder_layer_only"
        self.summary["coverage"]["module_names"] = [name for name, _ in selected]
        for name, module in selected:

            def before(target, args, range_name=name):
                if not self.active or threading.get_ident() != self._thread_id:
                    return
                try:
                    region = self.torch.profiler.record_function(
                        f"hqsb.module:{range_name}"
                    )
                    region.__enter__()
                    self._ranges.append((target, region))
                except Exception as exc:
                    self._problem(f"Module annotation failed: {exc}")

            def after(target, args, output):
                if threading.get_ident() != self._thread_id:
                    return
                if self._ranges and self._ranges[-1][0] is target:
                    _, region = self._ranges.pop()
                    try:
                        region.__exit__(None, None, None)
                    except Exception as exc:
                        self._problem(f"Module annotation cleanup failed: {exc}")

            try:
                self._hooks.append(module.register_forward_pre_hook(before))
                try:
                    self._hooks.append(
                        module.register_forward_hook(after, always_call=True)
                    )
                except TypeError:
                    self._hooks.append(module.register_forward_hook(after))
            except Exception as exc:
                self._problem(f"Cannot register module hook {name}: {exc}")
                self._clear_module_ranges()
                break

    def _clear_module_ranges(self):
        for handle in self._hooks:
            try:
                handle.remove()
            except Exception as exc:
                self._problem(f"Module hook removal failed: {exc}")
        self._hooks.clear()
        while self._ranges:
            _, region = self._ranges.pop()
            try:
                region.__exit__(None, None, None)
            except Exception as exc:
                self._problem(f"Unfinished module range cleanup failed: {exc}")

    def step_completed(self):
        if not self.active:
            return
        self.steps += 1
        self.summary["coverage"].update(
            prefill=self.steps > 0, decode_steps=max(0, self.steps - 1)
        )
        if self.steps >= self.MAX_DECODE_STEPS + 1:
            self.finish()

    def finish(self, *, total_tokens=None):
        if total_tokens is not None:
            self.summary["coverage"]["total_output_tokens"] = total_tokens
        if not self.enabled or self.finished:
            return self.summary
        self.finished = True
        if not self.active:
            self.summary["status"] = "not_collected"
            self._problem("Request ended before the first model step.")
            return self.summary
        self.active = False
        try:
            self._clear_module_ranges()
            self.profiler.stop()
            events = self.profiler.events()
            cuda_observed = any("CUDA" in str(e.device_type) for e in events)
            self.summary["activities"] = ["CPU"] + (["CUDA"] if cuda_observed else [])
            if not cuda_observed:
                self._problem(
                    "No CUDA device events were observed; GPU operator times are unavailable."
                )
            rows = []
            for event in self.profiler.key_averages(group_by_input_shape=True):
                rows.append(
                    {
                        "name": event.key,
                        "calls": event.count,
                        "cpu_ms": event.cpu_time_total / 1000,
                        "self_cpu_ms": event.self_cpu_time_total / 1000,
                        "cuda_ms": _device_milliseconds(event, "device_time_total")
                        if cuda_observed
                        else None,
                        "self_cuda_ms": _device_milliseconds(
                            event, "self_device_time_total"
                        )
                        if cuda_observed
                        else None,
                        "cpu_memory_bytes": getattr(event, "cpu_memory_usage", None),
                        "device_memory_bytes": getattr(
                            event, "device_memory_usage", None
                        )
                        if cuda_observed
                        else None,
                        "input_shapes": event.input_shapes,
                        "source": "torch_profiler_key_averages",
                    }
                )
            rows.sort(
                key=lambda row: row["self_cuda_ms"] or row["self_cpu_ms"], reverse=True
            )
            self.summary["operators"] = rows[: self.MAX_OPERATORS]
            self.summary["operator_groups_total"] = len(rows)
            self.summary["operator_groups_truncated"] = len(rows) > self.MAX_OPERATORS
            self.summary["status"] = "complete" if cuda_observed else "partial"
            self._export()
        except Exception as exc:
            self.summary["status"] = "error"
            self._problem(f"Capture finalization failed: {type(exc).__name__}: {exc}")
        finally:
            # Release references retained by shape/memory recording promptly.
            self.profiler = None
        return self.summary

    def _export(self):
        if self.directory is None:
            self._problem(
                "No capture directory was provided; only operator summary is retained."
            )
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        trace = self.directory / "trace.json"
        try:
            self.profiler.export_chrome_trace(str(trace))
            size = trace.stat().st_size
            self.summary["trace_bytes"] = size
            if size > self.MAX_TRACE_BYTES:
                trace.unlink()
                self._problem(
                    f"Trace exceeds retained artifact limit ({self.MAX_TRACE_BYTES} bytes)."
                )
            else:
                self.summary["trace_available"] = True
        except Exception as exc:
            trace.unlink(missing_ok=True)
            self._problem(f"Chrome trace export failed: {type(exc).__name__}: {exc}")

    def save_summary(self):
        """Best-effort evidence export must never hide an inference exception."""
        if self.enabled and self.directory is not None:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                (self.directory / "profile.json").write_text(
                    json.dumps(self.summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                self._problem(
                    f"Profile summary export failed: {type(exc).__name__}: {exc}"
                )
