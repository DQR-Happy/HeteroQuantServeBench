"""Real C ABI ownership/64-bit/UTF-8 probes, outside timed campaigns."""

import argparse
import ctypes
import json
from pathlib import Path

from benchmarks.capture.adapters import HindsightAdapter, NanoLogAdapter
from benchmarks.capture.oracle import decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, upstream = args.output.resolve(), args.upstream.resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for cls in (NanoLogAdapter, HindsightAdapter):
        directory = root / cls.name
        directory.mkdir()
        adapter = cls(directory, {}, upstream)
        expected = []
        for i, value in enumerate((2**32 - 1, 2**32 + 1, 2**53 + 1, 2**63 + 1)):
            body = json.dumps({"name": "推理层µ", "address": value, "data": "x" * 1024}, ensure_ascii=False)
            mutable = ctypes.create_string_buffer(body.encode())
            result = adapter.library.hf_offer(b"run_numeric", i, value, value + 123, mutable)
            assert result == 0
            # Reuse the same allocation immediately, before background consumption.
            mutable.value = b"MUTATED"
            expected.append({"run_id": "run_numeric", "producer_id": i, "source_seq": value,
                             "source_timestamp": value + 123, "body": body})
        adapter.close()
        actual = list(decoded(directory, cls.name, upstream))
        actual.sort(key=lambda row: row["producer_id"])
        assert actual == expected
        rows.append({"adapter": cls.name, "workloads": ["W21", "W30"], "status": "pass",
                     "observed": "mutable C source reused after return; original UTF-8 body and >2^63 integer headers preserved"})
    (root / "all-probes.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
