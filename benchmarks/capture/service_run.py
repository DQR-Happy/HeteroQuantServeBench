"""OTAP real HTTP pipeline, with a materializing fsync sink and direct control.

SERVICE track includes codecs/loopback HTTP. Never rank against in-process
offer() nanoseconds. Completion timestamps come from the independent sink.
"""

import argparse
import fcntl
import gzip
import json
import os
import signal
import socket
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests

from benchmarks.capture.oracle import quantiles
from benchmarks.capture.schedule import payload


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def execute(root, upstream, rate, batch, mode, repeat, seconds):
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
        ExportLogsServiceRequest,
    )

    directory = root / f"{mode}-{rate}-batch{batch}-r{repeat}"
    directory.mkdir()
    case = {"seed": 100 + repeat, "body_bytes": 64}
    run_id = "run_" + uuid.uuid4().hex
    counts = {"valid": 0, "corrupt": 0, "duplicate": 0, "foreign": 0}
    seen, completions = set(), []
    sink_data = (directory / "received.jsonl").open("wb")
    cpus = sorted(os.sched_getaffinity(0))
    t0 = [0]
    n = int(rate * seconds)

    class Sink(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                chunks = []
                while True:
                    length = int(self.rfile.readline().split(b";", 1)[0], 16)
                    if length == 0:
                        while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                            pass
                        break
                    chunks.append(self.rfile.read(length))
                    if self.rfile.read(2) != b"\r\n":
                        raise ValueError("invalid HTTP chunk delimiter")
                body = b"".join(chunks)
            else:
                body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            request = ExportLogsServiceRequest.FromString(body)
            delivered = []
            for resource in request.resource_logs:
                for scope in resource.scope_logs:
                    for record in scope.log_records:
                        event = json.loads(record.body.string_value)
                        i = event["source_seq"]
                        if event["run_id"] != run_id or not 0 <= i < n:
                            counts["foreign"] += 1
                        elif i in seen:
                            counts["duplicate"] += 1
                        elif event["body"] != payload(i, case) or event["source_timestamp"] != t0[0] + int(i * 1e9 / rate):
                            counts["corrupt"] += 1
                        else:
                            seen.add(i)
                            delivered.append(i)
                            sink_data.write((json.dumps(event, separators=(",", ":")) + "\n").encode())
            sink_data.flush()
            os.fsync(sink_data.fileno())
            durable = time.monotonic_ns()
            completions.extend((i, durable) for i in delivered)
            counts["valid"] += len(delivered)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    sink = HTTPServer(("127.0.0.1", 0), Sink)

    def serve():
        os.sched_setaffinity(0, {cpus[1]})
        sink.serve_forever(poll_interval=0.05)

    thread = threading.Thread(target=serve)
    thread.start()
    engine, engine_log = None, None
    target = sink.server_port
    results = None
    try:
        if mode == "otap":
            target = unused_port()
            config = directory / "engine.yaml"
            config.write_text(f"""version: otel_dataflow/v1
engine: {{}}
policies:
  resources:
    core_allocation:
      type: core_count
      count: 1
groups:
  default:
    pipelines:
      main:
        nodes:
          receiver:
            type: receiver:otlp
            config:
              protocols:
                http:
                  listening_addr: "127.0.0.1:{target}"
                  wait_for_result: true
          exporter:
            type: exporter:otlp_http
            config:
              http: {{}}
              endpoint: "http://127.0.0.1:{sink.server_port}"
              client_pool_size: 1
        connections:
          - from: receiver
            to: exporter
""")
            binary = upstream / "otel-arrow/rust/otap-dataflow/target/release/df_engine"
            engine_log = (directory / "engine.log").open("wb")
            engine = subprocess.Popen(["taskset", "-c", str(cpus[2]), str(binary), "-c", str(config),
                "--http-admin-bind", "127.0.0.1:0"], stdout=engine_log, stderr=subprocess.STDOUT,
                start_new_session=True)
            deadline = time.monotonic() + 20
            while True:
                if engine.poll() is not None:
                    raise RuntimeError(f"OTAP startup exit {engine.returncode}")
                try:
                    with socket.create_connection(("127.0.0.1", target), timeout=0.1):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("OTAP listener startup timeout")
                    time.sleep(0.1)
        os.sched_setaffinity(0, {cpus[0]})
        session = requests.Session()
        session.trust_env = False
        attempts = accepted = 0
        request_times, dispatch = [], []
        t0[0] = time.monotonic_ns() + 50_000_000
        wall_epoch = time.time_ns() + 50_000_000
        hard_end = t0[0] + int((seconds + 1) * 1e9)
        for start in range(0, n, batch):
            end = min(n, start + batch)
            scheduled = t0[0] + int((end - 1) * 1e9 / rate)
            sleep = scheduled - time.monotonic_ns()
            if sleep > 0:
                time.sleep(sleep / 1e9)
            if time.monotonic_ns() > hard_end:
                break
            request = ExportLogsServiceRequest()
            scope = request.resource_logs.add().scope_logs.add()
            for i in range(start, end):
                event = {"run_id": run_id, "producer_id": 0, "source_seq": i,
                         "source_timestamp": t0[0] + int(i * 1e9 / rate), "body": payload(i, case)}
                log = scope.log_records.add()
                log.time_unix_nano = wall_epoch + int(i * 1e9 / rate)
                log.body.string_value = json.dumps(event, separators=(",", ":"))
            encoded = request.SerializeToString()
            begin = time.monotonic_ns()
            dispatch.append(begin - scheduled)
            response = session.post(f"http://127.0.0.1:{target}/v1/logs", data=encoded,
                                    headers={"Content-Type": "application/x-protobuf"}, timeout=30)
            request_times.append(time.monotonic_ns() - begin)
            attempts += end - start
            if response.status_code == 200:
                accepted += end - start
            else:
                raise RuntimeError(f"HTTP status {response.status_code}: {response.text[:200]}")
        session.close()
        deadline = time.monotonic() + 30
        while counts["valid"] < accepted and time.monotonic() < deadline:
            time.sleep(0.01)
        endpoint = max((t for i, t in completions), default=time.monotonic_ns())
        elapsed = max(seconds, (endpoint - t0[0]) / 1e9)
        latencies = [t - (t0[0] + int(i * 1e9 / rate)) for i, t in completions]
        results = {"track": "SERVICE", "mode": mode, "rate": rate, "batch_events": batch, "repeat": repeat,
                   "scheduled": n, "attempted": attempts, "HTTP_200_events": accepted,
                   "scheduled_not_attempted": n - attempts, "counts": counts,
                   "H_plus_drain_s": elapsed, "durable_goodput": counts["valid"] / elapsed,
                   "scheduled_delivery_ratio": counts["valid"] / n,
                   "scheduled_durable_latency": quantiles(latencies), "HTTP_batch_latency": quantiles(request_times),
                   "dispatch_lag": quantiles(dispatch),
                   "deadline_success": {str(ms): sum(x <= ms * 1e6 for x in latencies) / n for ms in (1, 10, 100, 1000)},
                   "status": "complete" if counts["valid"] == n and not any(counts[k] for k in ("corrupt", "duplicate", "foreign")) else "lossy_or_invalid",
                   "scope": "OTLP bytes over loopback HTTP, one sequential sender, one-core engine, independent decoding fsync sink; direct is generator/codec/sink control; not Arrow conversion or in-process offer ranking"}
        (directory / "completion-timestamps.json").write_text(json.dumps(completions))
    finally:
        if engine is not None and engine.poll() is None:
            os.killpg(engine.pid, signal.SIGINT)
            try:
                engine.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(engine.pid, signal.SIGKILL)
                engine.wait()
        if engine_log:
            engine_log.close()
        if results is not None and engine is not None:
            results["engine_exit_code"] = engine.returncode
        sink.shutdown()
        thread.join()
        sink.server_close()
        sink_data.close()
        os.sched_setaffinity(0, set(cpus))
    (directory / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--modes", default="direct,otap")
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--rates", default="1000,10000,100000")
    p.add_argument("--batches", default="1,64,512")
    args = p.parse_args()
    root, upstream = args.output.resolve(), args.upstream.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if "otap" in args.modes.split(",") and not (upstream / "otel-arrow/rust/otap-dataflow/target/release/df_engine").is_file():
        raise RuntimeError("OTAP binary unavailable; no SDK substitution permitted")
    rows = []
    with (upstream.parent / "measurement.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for rate in map(int, args.rates.split(",")):
            for batch in map(int, args.batches.split(",")):
                for repeat in range(args.repeats):
                    for mode in args.modes.split(","):
                        try:
                            row = execute(root, upstream, rate, batch, mode, repeat, args.seconds)
                        except Exception as exc:  # noqa: BLE001 - retain failed configurations, continue campaign
                            row = {"mode": mode, "rate": rate, "batch_events": batch, "repeat": repeat,
                                   "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                        rows.append(row)
                        (root / "all-runs.json").write_text(json.dumps(rows, indent=2))
                        print(mode, rate, batch, repeat, row["status"], flush=True)


if __name__ == "__main__":
    main()
