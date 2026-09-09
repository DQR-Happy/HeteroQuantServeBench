"""Run unmodified MooBench Python SuT; no Docker/R or fake pipeline substitute.

The Zipkin endpoint is a local benchmark receiver, not a Zipkin storage service.
Original recursion and span instrumentation are unchanged. Output stays separate.
"""

import argparse
import fcntl
import gzip
import importlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from benchmarks.capture.oracle import quantiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    root, upstream = args.output.resolve(), args.upstream.resolve()
    root.mkdir(parents=True, exist_ok=False)
    # Upstream silently falls back when these imports fail: prevent fake reproduction.
    for module, attribute in (("opentelemetry.sdk.trace", "TracerProvider"),
                              ("opentelemetry.sdk.trace.export", "BatchSpanProcessor"),
                              ("opentelemetry.exporter.zipkin.json", "ZipkinExporter"),
                              ("opentelemetry.sdk.resources", "Resource"),
                              ("opentelemetry.semconv.resource", "ResourceAttributes")):
        getattr(importlib.import_module(module), attribute)
    lock = (upstream.parent / "measurement.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    worker = upstream / "moobench/tools/SuT-python/benchmark.py"
    rows = []
    cpus = sorted(os.sched_getaffinity(0))
    rng = random.Random(20261017)
    for depth in (1, 10, 100):
        for repeat in range(args.repeats):
            modes = ["off", "no-export", "zipkin-local-receiver"]
            rng.shuffle(modes)
            for mode in modes:
                directory = root / f"depth{depth}-{mode}-r{repeat}"
                directory.mkdir()
                counts = {"spans": 0, "requests": 0, "bytes": 0, "duplicates": 0}
                seen = set()
                stream = (directory / "received.jsonl").open("wb")

                class Receiver(BaseHTTPRequestHandler):
                    capture_counts = counts
                    capture_seen = seen
                    capture_stream = stream

                    def do_POST(self):
                        data = self.rfile.read(int(self.headers["Content-Length"]))
                        if self.headers.get("Content-Encoding") == "gzip":
                            data = gzip.decompress(data)
                        batch = json.loads(data)
                        for span in batch:
                            key = (span["traceId"], span["id"])
                            self.capture_counts["duplicates"] += key in self.capture_seen
                            self.capture_seen.add(key)
                        self.capture_counts["spans"] += len(batch)
                        self.capture_counts["requests"] += 1
                        self.capture_counts["bytes"] += len(data)
                        self.capture_stream.write(data + b"\n")
                        self.send_response(202)
                        self.send_header("Content-Length", "0")
                        self.end_headers()

                    def log_message(self, *args):
                        pass

                server = HTTPServer(("127.0.0.1", 0), Receiver)

                def serve(server=server):
                    os.sched_setaffinity(0, {cpus[1]})
                    server.serve_forever(poll_interval=0.05)

                thread = threading.Thread(target=serve)
                thread.start()
                calls = 10_000 if depth <= 10 else 2000
                config = directory / "config.ini"
                config.write_text("[Benchmark]\n" + f"total_calls={calls}\nrecursion_depth={depth}\nmethod_time=0\n"
                                  + f"output_filename={directory / 'raw.csv'}\nconfig_path={directory / 'unused.ini'}\n"
                                  + "inactive=False\ninstrumentation_on=False\napproach=1\n")
                env = dict(os.environ, ENABLE_OTEL="false" if mode == "off" else "true",
                           OTEL_TRACES_EXPORTER="zipkin" if mode == "zipkin-local-receiver" else "none",
                           OTEL_METRICS_EXPORTER="none", OTEL_LOGS_EXPORTER="none",
                           OTEL_EXPORTER_ZIPKIN_ENDPOINT=f"http://127.0.0.1:{server.server_port}/api/v2/spans")
                started = time.monotonic()
                try:
                    with (directory / "stdout.log").open("wb") as log:
                        process = subprocess.run(["taskset", "-c", str(cpus[0]), sys.executable,
                                                  str(worker), str(config)], env=env, stdout=log,
                                                 stderr=subprocess.STDOUT, timeout=120, check=False)
                finally:
                    server.shutdown()
                    thread.join()
                    server.server_close()
                    stream.close()
                duration = time.monotonic() - started
                values = []
                if (directory / "raw.csv").exists():
                    with (directory / "raw.csv").open() as data:
                        values = [int(line.split(";")[1]) for line in data]
                row = {"depth": depth, "mode": mode, "repeat": repeat, "calls": calls,
                       "recorded_calls": len(values), "exit_code": process.returncode,
                       "method_call_latency": quantiles(values), "process_wall_s": duration,
                       "mean_method_ns": sum(values) / len(values) if values else None,
                       "expected_spans": calls * (depth + 1) if mode == "zipkin-local-receiver" else None,
                       "receiver": counts, "scope": "upstream recursive SuT; local Zipkin-protocol receiver; no disk durability claim"}
                (directory / "result.json").write_text(json.dumps(row, indent=2))
                rows.append(row)
                (root / "all-runs.json").write_text(json.dumps(rows, indent=2))
                print(depth, repeat, mode, process.returncode, len(values), counts["spans"], flush=True)


if __name__ == "__main__":
    main()
