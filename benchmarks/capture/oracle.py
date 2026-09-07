"""Post-run decoding and semantic verification; never used as producer truth."""

import hashlib
import json
import struct
import subprocess
import time
from collections import Counter

from benchmarks.capture.schedule import payload, producer_for
from benchmarks.capture.worker import RECEIPT


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values):
    values = sorted(values)
    if not values:
        return {"samples": 0}
    result = {"samples": len(values), "max_ns": values[-1]}
    for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("p999", 0.999)):
        result[name + "_ns"] = values[min(len(values) - 1, int(q * len(values)))] if len(values) * (1 - q) >= 100 - 1e-8 else None
    return result


def decoded(directory, adapter, upstream):
    if adapter == "hqsb":
        previous = 0
        for path in sorted((directory / "observations").glob("records-*.jsonl")):
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if (row.get("schema_version") != "1.0.0" or type(row.get("seq")) is not int
                            or row["seq"] <= previous or row.get("kind") != "capture_benchmark"):
                        raise ValueError("HQSB envelope schema/sequence invalid")
                    previous = row["seq"]
                    yield row["payload"]
    elif adapter == "otel":
        with (directory / "otel.jsonl").open() as stream:
            for line in stream:
                yield json.loads(line)
    elif adapter == "nanolog":
        output, error = directory / "decoded.txt", directory / "decode.stderr"
        with output.open("wb") as out, error.open("wb") as err:
            proc = subprocess.run([str(upstream / "NanoLog/runtime/decompressor"), "decompress",
                                   str(directory / "nanolog.bin")], stdout=out, stderr=err, timeout=60, check=False)
        if proc.returncode:
            raise RuntimeError(f"NanoLog decompressor exit {proc.returncode}")
        with output.open() as stream:
            for line in stream:
                if "HQSB|" not in line:
                    continue
                fields = line.split("HQSB|", 1)[1].rstrip("\n").split("|", 4)
                if len(fields) != 5:
                    raise ValueError("truncated NanoLog record")
                run, producer, seq, scheduled, body = fields
                yield {"run_id": run, "producer_id": int(producer), "source_seq": int(seq),
                       "source_timestamp": int(scheduled), "body": body}
    elif adapter == "hindsight":
        pending = {}
        trace_header = struct.Struct("<QQiiIhh")
        event_header = struct.Struct("<64sIQQI8x")
        with (directory / "hindsight.blocks").open("rb") as stream:
            while prefix := stream.read(4):
                if len(prefix) != 4:
                    raise ValueError("truncated Hindsight block length")
                length = struct.unpack("<I", prefix)[0]
                if not trace_header.size <= length <= 4096:
                    raise ValueError("Hindsight block size invalid")
                block = stream.read(length)
                if len(block) != length:
                    raise ValueError("truncated Hindsight block")
                trace, _acquired, _buf, _prev, _size, number, _nulls = trace_header.unpack_from(block)
                fragment = block[trace_header.size:]
                if number == 0:
                    pending[trace] = (1, bytearray(fragment))
                elif trace in pending and pending[trace][0] == number:
                    expected, data = pending[trace]
                    data.extend(fragment)
                    pending[trace] = (expected + 1, data)
                else:
                    pending.pop(trace, None)
                    continue
                _, data = pending[trace]
                if len(data) < event_header.size:
                    continue
                run, producer, seq, scheduled, body_size = event_header.unpack_from(data)
                if body_size > 32 * 1024**2:
                    raise ValueError("invalid Hindsight event length")
                if len(data) == event_header.size + body_size:
                    yield {"run_id": run.split(b"\0", 1)[0].decode(), "producer_id": producer,
                           "source_seq": seq, "source_timestamp": scheduled,
                           "body": data[event_header.size:].decode()}
                    del pending[trace]
                elif len(data) > event_header.size + body_size:
                    raise ValueError("Hindsight event overrun")


def verify(directory, upstream, decode_events=None):
    started = time.monotonic_ns()
    raw = json.loads((directory / "worker.json").read_text())
    case, adapter = raw["case"], raw["adapter"]
    n = raw["scheduled"]
    # Bounded <=2M identities; no full decoded body corpus retained.
    attempted, valid, codes = bytearray(n), bytearray(n), bytearray(n)
    schedules = [0] * n
    call, lag, by_code = [], [], Counter()
    for path in sorted(directory.glob("receipts-*.bin")):
        with path.open("rb") as stream:
            while block := stream.read(RECEIPT.size):
                if len(block) != RECEIPT.size:
                    raise ValueError("truncated receipt")
                i, begin, scheduled, duration, code = RECEIPT.unpack(block)
                if i >= n or attempted[i]:
                    raise ValueError("invalid or duplicate receipt")
                attempted[i], codes[i], schedules[i] = 1, code, scheduled
                by_code[str(code)] += 1
                if begin:
                    call.append(duration)
                    lag.append(begin - scheduled)
    duplicate = corruption = foreign = order_errors = delivered_bytes = warmup = 0
    per_source = Counter()
    previous = {}
    decode_error = None
    try:
        if adapter == "hqsb":
            manifest = raw["terminal"]["writer_report"]
            for artifact in manifest["artifacts"]:
                path = directory / artifact["path"]
                if path.stat().st_size != artifact["bytes"] or file_sha256(path) != artifact["sha256"]:
                    raise ValueError("HQSB writer artifact hash/size mismatch")
        for event in (decode_events or decoded)(directory, adapter, upstream):
            if event.get("run_id") == "warmup":
                warmup += 1
                continue
            i = event.get("source_seq")
            p = event.get("producer_id")
            if (event.get("run_id") != case["run_id"] or type(i) is not int
                    or type(p) is not int or not 0 <= i < n or not attempted[i] or p != producer_for(i, case)):
                foreign += 1
                continue
            if valid[i]:
                duplicate += 1
                continue
            expected_body = payload(i, case)
            expected_size = 64 + len(expected_body.encode())
            actual_body = event.get("body")
            if case.get("structured"):
                if isinstance(actual_body, str):
                    actual_body = json.loads(actual_body)
                actual_body = json.dumps(actual_body, sort_keys=True, allow_nan=False, separators=(",", ":"))
                expected_body = json.dumps(json.loads(expected_body), sort_keys=True, allow_nan=False, separators=(",", ":"))
            if type(event.get("source_timestamp")) is not int or event.get("source_timestamp") != schedules[i] or actual_body != expected_body:
                corruption += 1
                continue
            if i <= previous.get(p, -1):
                order_errors += 1
            previous[p] = i
            valid[i] = 1
            per_source[p] += 1
            delivered_bytes += expected_size
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        decode_error = f"{type(exc).__name__}: {exc}"
    H = raw["H_ns"] / 1e9
    elapsed = max(raw["H_ns"], raw["endpoint_end_ns"] - raw["t0_ns"]) / 1e9
    N_A, N_V = sum(attempted), sum(valid)
    accepted = sum(1 for i in range(n) if attempted[i] and codes[i] == 0)
    missing_accepted = sum(1 for i in range(n) if attempted[i] and codes[i] == 0 and not valid[i])
    offered_per_source = Counter(producer_for(i, case) for i in range(n))
    per_source_ratios = [per_source[p] / offered for p, offered in offered_per_source.items()]
    squares = sum(x * x for x in per_source_ratios)
    # Do not include observer sidecars or decompressed duplicates as encoded output.
    if raw.get("encoded_output_files") is not None:
        artifacts = [directory / name for name in raw["encoded_output_files"]]
    elif adapter == "hqsb":
        artifacts = sorted((directory / "observations").glob("*"))
    else:
        artifacts = [directory / {"nanolog": "nanolog.bin", "hindsight": "hindsight.blocks"}.get(adapter, "otel.jsonl")]
    artifacts = [path for path in artifacts if path.is_file()]
    encoded_bytes = sum(path.stat().st_size for path in artifacts)
    accepted_known = raw["acknowledgement"] == "explicit"
    failure = decode_error or corruption or duplicate or foreign or order_errors or raw["errors"]
    output = {
        "run_id": case["run_id"], "case_id": case["id"], "workload": case["workload"],
        "adapter": adapter, "track": raw["track"], "endpoint": raw["terminal"]["endpoint"],
        "measurement_pass": case.get("measurement_pass", "latency"),
        "status": "control" if adapter == "null" else "correctness_fail" if failure else "lossy" if N_V != n else "complete",
        "scheduled": n, "attempted": N_A, "valid_unique": N_V,
        "accepted": accepted if accepted_known else None,
        "accepted_not_valid": missing_accepted if accepted_known else None,
        "scheduled_not_attempted": n - N_A, "warmup_records": warmup,
        "retained_ratio": N_V / N_A if N_A else None,
        "scheduled_delivery_ratio": N_V / n if n else None,
        "offered_rate": n / H, "attempt_rate": N_A / H,
        "cohort_completion_rate": N_V / elapsed,
        "logical_bandwidth_Bps": delivered_bytes / elapsed,
        "H_s": H, "H_plus_drain_s": elapsed,
        "call_latency": quantiles(call), "dispatch_lag": quantiles(lag),
        "cpu_seconds": raw["cpu_seconds"], "combined_cpu_cores": raw["cpu_seconds"] / elapsed,
        "cpu_ns_per_valid_event": raw["cpu_seconds"] * 1e9 / N_V if N_V else None,
        "cpu_ns_per_logical_byte": raw["cpu_seconds"] * 1e9 / delivered_bytes if delivered_bytes else None,
        "process_maxrss_KiB": raw["maxrss_KiB"], "encoded_bytes": encoded_bytes,
        "encoding_space_ratio": delivered_bytes / encoded_bytes if encoded_bytes else None,
        "startup_ms": raw["startup_ns"] / 1e6,
        "close_ms": (raw["terminal"]["close_end_ns"] - raw["terminal"].get("close_start_ns", raw["terminal"]["close_end_ns"])) / 1e6,
        "publication_ms": (raw["terminal"]["published_ns"] - raw["terminal"]["close_start_ns"]) / 1e6 if "published_ns" in raw["terminal"] else None,
        "supervisor_publication_ms": (raw["terminal"]["published_ns"] - raw["terminal"]["publication_start_ns"]) / 1e6 if "publication_start_ns" in raw["terminal"] else None,
        "duplicate_records": duplicate, "semantic_corruption": corruption,
        "foreign_records": foreign, "ordering_violations": order_errors,
        "decode_error": decode_error, "per_producer_valid": dict(per_source),
        "normalized_jain_fairness": sum(per_source_ratios)**2 / (len(per_source_ratios) * squares) if squares else None,
        "receipt_codes": dict(by_code),
        "minor_faults": raw["minor_faults"], "major_faults": raw["major_faults"],
        "voluntary_switches": raw["voluntary_switches"], "involuntary_switches": raw["involuntary_switches"],
        "verification_s": (time.monotonic_ns() - started) / 1e9,
        "unavailable": {
            "per_event_ready_durable_latency_and_deadline_goodput": "stock APIs do not expose per-event completion; do not substitute offer return or close time",
            "collector_owned_bytes_queue_bytes": "stock libraries lack equivalent exact allocation/queue accounting; process samples separate",
            "PMU_energy": "not collected in timing run",
            "accepted": None if accepted_known else "void library API; submitted is not accepted",
            "capacity": "requires independent search and sustained confirmations, not inferred from one load point"},
        "encoded_artifacts": [{"path": str(p.relative_to(directory)), "bytes": p.stat().st_size,
                               "sha256": file_sha256(p)} for p in artifacts],
    }
    output["verification_s"] = (time.monotonic_ns() - started) / 1e9
    output["goodput_including_benchmark_verification"] = N_V / (elapsed + output["verification_s"])
    (directory / "metrics.json").write_text(json.dumps(output, indent=2))
    return output
