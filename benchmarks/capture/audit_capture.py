"""Read-only post-audit for stock HQSB manifests, including expanded CORE files."""

import argparse
import json
from pathlib import Path

from benchmarks.capture.oracle import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.campaign.glob("*/worker.json")):
        raw = json.loads(path.read_text())
        if raw["adapter"] != "hqsb":
            continue
        manifest = raw["terminal"]["writer_report"]
        count, previous, error = 0, 0, None
        try:
            for item in manifest["artifacts"]:
                artifact = path.parent / item["path"]
                assert artifact.stat().st_size == item["bytes"]
                assert file_sha256(artifact) == item["sha256"]
                with artifact.open() as stream:
                    for line in stream:
                        record = json.loads(line)
                        assert record["schema_version"] == "1.0.0"
                        assert type(record["seq"]) is int and record["seq"] > previous
                        assert record["kind"] == "capture_benchmark"
                        previous = record["seq"]
                        count += 1
            assert count == manifest["counts"]["written"]
            c = manifest["counts"]
            assert c["attempted"] == c["accepted"] + c["dropped"]
            assert c["accepted"] == c["written"] + c["accepted_not_written"]
        except (AssertionError, OSError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        rows.append({"run_id": raw["case"]["run_id"], "status": "pass" if error is None else "fail",
                     "rows_including_warmup": count, "error": error, "path": str(path.parent)})
    args.output.write_text(json.dumps(rows, indent=2))
    print(json.dumps({"audited": len(rows), "failed": sum(r["status"] == "fail" for r in rows)}))


if __name__ == "__main__":
    main()
