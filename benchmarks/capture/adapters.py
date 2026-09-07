"""Real-library adapters. Library limitations remain explicit capabilities."""

import ctypes
import json
import os
import time


def body_text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class NullAdapter:
    name = "null"
    acknowledgement = "discard_control"

    def __init__(self, directory, case, upstream):
        self.directory = directory

    def offer(self, event):
        return "discarded"

    def close(self):
        return {"endpoint": "discard_control", "close_end_ns": time.monotonic_ns()}


class HQSBAdapter:
    name = "hqsb"
    acknowledgement = "explicit"

    def __init__(self, directory, case, upstream):
        from hqsb.observation.storage import EvidenceWriter

        self.directory = directory
        # Only capture lifetime limits are enlarged in CORE. Queue remains 256.
        self.writer = EvidenceWriter(
            directory,
            {"run_id": case["run_id"], "benchmark": "capture-cpu"},
            max_records=case.get("max_records", 2_000_001 + case.get("warmup_events", 100)),
            max_bytes=case.get("max_bytes", 1024**3),
            chunk_bytes=case.get("chunk_bytes", 8 * 1024**2),
        )

    def offer(self, event):
        # Event and immutable strings are fresh/owned for the existing API.
        return self.writer.offer("capture_benchmark", event)

    def close(self):
        started = time.monotonic_ns()
        problems = getattr(self, "source_problems", [])
        report = self.writer.close(execution_status="failed" if problems else "succeeded",
                                   cleanup_status="succeeded", source_problems=problems)
        ended = time.monotonic_ns()
        result = {"endpoint": "D2-close", "close_start_ns": started,
                  "close_end_ns": ended, "writer_report": report}
        return result


class NanoLogAdapter:
    name = "nanolog"
    acknowledgement = "unavailable_void_api"

    def __init__(self, directory, case, upstream):
        self.directory = directory
        self.library = ctypes.CDLL(str(upstream.parent / "build" / "libcapture_nanolog.so"))
        self.library.hf_open.argtypes = [ctypes.c_char_p]
        self.library.hf_open.restype = ctypes.c_int
        self.library.hf_offer.argtypes = [ctypes.c_char_p, ctypes.c_uint32,
                                         ctypes.c_uint64, ctypes.c_uint64, ctypes.c_char_p]
        self.library.hf_offer.restype = ctypes.c_int
        self.library.hf_close.restype = ctypes.c_int
        if self.library.hf_open(os.fsencode(directory / "nanolog.bin")):
            raise RuntimeError("NanoLog open failed")

    def offer(self, event):
        result = self.library.hf_offer(event["run_id"].encode(), event["producer_id"],
                                       event["source_seq"], event["source_timestamp"],
                                       body_text(event["body"]).encode())
        return "submitted_unknown" if result == 0 else "adapter_error"

    def close(self):
        started = time.monotonic_ns()
        if self.library.hf_close():
            raise RuntimeError(f"{self.name} close failed")
        return {"endpoint": "D2-close", "close_start_ns": started,
                "close_end_ns": time.monotonic_ns(),
                "durability": "NanoLog sync plus adapter fsync; no per-event checkpoint ack"}


class OTelAdapter:
    name = "otel"
    acknowledgement = "unavailable_void_api"

    def __init__(self, directory, case, upstream):
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SpanExporter,
            SpanExportResult,
        )

        class FileExporter(SpanExporter):
            def __init__(self):
                self.stream = (directory / "otel.jsonl").open("xb")

            def export(self, spans):
                if case.get("stall_ms"):
                    time.sleep(case["stall_ms"] / 1000)
                for span in spans:
                    self.stream.write((json.dumps(dict(span.attributes), separators=(",", ":")) + "\n").encode())
                return SpanExportResult.SUCCESS

            def shutdown(self):
                self.stream.flush()
                os.fsync(self.stream.fileno())
                self.stream.close()

        self.exporter = FileExporter()
        self.provider = TracerProvider(resource=Resource.create({"service.name": "hqsb-capture-benchmark"}),
                                       shutdown_on_exit=False)
        self.provider.add_span_processor(BatchSpanProcessor(
            self.exporter, max_queue_size=case.get("queue_records", 256),
            max_export_batch_size=min(case.get("batch_events", 64), case.get("queue_records", 256)),
            schedule_delay_millis=10))
        self.tracer = self.provider.get_tracer("hqsb.capture")

    def offer(self, event):
        attributes = event if isinstance(event["body"], str) else dict(event, body=body_text(event["body"]))
        span = self.tracer.start_span("capture_benchmark", attributes=attributes)
        span.end()
        return "submitted_unknown"

    def close(self):
        started = time.monotonic_ns()
        flushed = self.provider.force_flush(timeout_millis=30_000)
        self.provider.shutdown()
        return {"endpoint": "D2-close", "close_start_ns": started,
                "close_end_ns": time.monotonic_ns(), "flush_ok": flushed,
                "scope": "real OTel SDK BatchSpanProcessor + benchmark file exporter; not OTAP engine"}


class HindsightAdapter(NanoLogAdapter):
    name = "hindsight"

    def __init__(self, directory, case, upstream):
        self.directory = directory
        self.library = ctypes.CDLL(str(upstream.parent / "build" / "libcapture_hindsight.so"))
        self.library.hf_offer = self.library.hh_offer
        self.library.hf_close = self.library.hh_close
        self.library.hh_open.argtypes = [ctypes.c_char_p]
        self.library.hh_open.restype = ctypes.c_int
        self.library.hf_offer.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint64,
                                         ctypes.c_uint64, ctypes.c_char_p]
        self.library.hf_offer.restype = ctypes.c_int
        self.library.hf_close.restype = ctypes.c_int
        if self.library.hh_open(os.fsencode(directory / "hindsight.blocks")):
            raise RuntimeError("Hindsight open failed")

    def close(self):
        result = super().close()
        result["scope"] = "upstream C client + agent API, benchmark full-stream file sink; not distributed trigger system"
        return result


ADAPTERS = {cls.name: cls for cls in (NullAdapter, HQSBAdapter, NanoLogAdapter, OTelAdapter, HindsightAdapter)}
