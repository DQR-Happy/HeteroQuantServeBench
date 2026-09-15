"""Independent official-format decoders, outside all SUT timing windows."""

import csv
import gzip
import json
import os
from pathlib import Path
import subprocess

from benchmarks.capture.oracle import decoded as legacy_decode
from benchmarks.capture.fair.wire_semantics import quill_json_text


def sql_uint64(value):
    """Perfetto SQL exposes the original uint64 bits as signed int64."""
    integer = int(value)
    if not -(2**63) <= integer < 2**64:
        raise ValueError("integer outside uint64/int64 representation")
    return integer + 2**64 if integer < 0 else integer


def events(directory, adapter, upstream):
    if adapter in {"null", "hindsight", "hqsb", "otel"}:
        yield from legacy_decode(directory, adapter, upstream)
        return
    csv.field_size_limit(64 * 1024**2)
    if adapter == "nanolog":
        # A default decoder can silently print zero records for a larger
        # configured writer. Receiver configuration is part of the experiment.
        decoder = os.environ.get("HQSB_NANOLOG_DECOMPRESSOR", str(upstream / "NanoLog/runtime/decompressor"))
        with (directory / "decoded.txt").open("wb") as output, (directory / "decode.stderr").open("wb") as error:
            result = subprocess.run([decoder, "decompress", str(directory / "nanolog.bin")],
                                    stdout=output, stderr=error, timeout=120, check=False)
        if result.returncode:
            raise RuntimeError(f"NanoLog decompressor exit {result.returncode}")
        with (directory / "decoded.txt").open() as stream:
            for line in stream:
                if "HQSB|" not in line:
                    continue
                parts = line.split("HQSB|", 1)[1].rstrip("\n").split("|", 4)
                if len(parts) != 5:
                    raise ValueError("truncated NanoLog record")
                run, p, seq, ts, body = parts
                yield dict(run_id=run, producer_id=int(p), source_seq=int(seq),
                           source_timestamp=int(ts), body=body)
    elif adapter == "dftracer":
        for path in sorted(directory.glob("dftrace*")):
            if not path.is_file() or path.suffix not in {".pfw", ".gz"}:
                continue
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt") as stream:
                for line in stream:
                    text = line.strip().strip(",")
                    if text in {"[", "]", ""}:
                        continue
                    row = json.loads(text)
                    if row.get("name") != "capture":
                        continue
                    args = row["args"]
                    yield {"run_id": args["run_id"], "producer_id": int(args["producer_id"]),
                           "source_seq": int(args["source_seq"]), "source_timestamp": int(args["source_timestamp"]), "body": args["body"]}
    elif adapter == "quill":
        with (directory / "quill.log").open() as stream:
            for line in stream:
                if not line.startswith("HQSB|"):
                    raise ValueError("unexpected Quill output")
                run, p, seq, ts, body = line[5:].rstrip("\n").split("|", 4)
                yield {"run_id": run, "producer_id": int(p), "source_seq": int(seq),
                       "source_timestamp": int(ts), "body": quill_json_text(body)}
    elif adapter == "perfetto":
        # trace_processor's CLI CSV does not escape embedded JSON quotes.
        # Hex strings preserve bytes through its public SQL interface.
        sql = "SELECT hex(EXTRACT_ARG(arg_set_id,'debug.run_id')) AS run_id, EXTRACT_ARG(arg_set_id,'debug.producer_id') AS producer_id, EXTRACT_ARG(arg_set_id,'debug.source_seq') AS source_seq, EXTRACT_ARG(arg_set_id,'debug.source_timestamp') AS source_timestamp, hex(EXTRACT_ARG(arg_set_id,'debug.body')) AS body FROM slice WHERE name='capture' ORDER BY ts,id"
        with (directory / "decoded.csv").open("w") as output, (directory / "decode.stderr").open("w") as error:
            subprocess.run([os.environ["HQSB_TRACE_PROCESSOR"], str(directory / "perfetto.pftrace"),
                            "-Q", sql], stdout=output, stderr=error, timeout=120, check=True)
        with (directory / "decoded.csv").open() as stream:
            for row in csv.DictReader(stream):
                yield {"run_id": bytes.fromhex(row["run_id"]).decode(), "producer_id": int(row["producer_id"]),
                       "source_seq": sql_uint64(row["source_seq"]), "source_timestamp": sql_uint64(row["source_timestamp"]), "body": bytes.fromhex(row["body"]).decode()}
    elif adapter == "lttng":
        # bt2 is the distribution's real Babeltrace2 CTF decoder, not a fake codec.
        script = Path(__file__).with_name("read_ctf.py")
        with (directory / "decoded.jsonl").open("w") as output, (directory / "decode.stderr").open("w") as error:
            subprocess.run(["/usr/bin/python3", str(script), str(directory / "ctf")],
                           stdout=output, stderr=error, check=True, timeout=120)
        with (directory / "decoded.jsonl").open() as stream:
            for line in stream:
                yield json.loads(line)
    else:
        raise ValueError(f"no independent decoder for {adapter}")
