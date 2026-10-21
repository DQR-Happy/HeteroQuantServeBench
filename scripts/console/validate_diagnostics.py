"""Target-only acceptance through public Console APIs and real engine workers."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--tool", choices=["nsys", "ncu"], default="nsys")
    parser.add_argument("--url", default="http://127.0.0.1:8768")
    parser.add_argument("--expect-counter-permission-error", action="store_true")
    args = parser.parse_args()
    root = Path("reports/console/nsight")
    root.mkdir(parents=True, exist_ok=True)

    def call(path, body=None):
        request = urllib.request.Request(args.url + "/api/console/v1" + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json", "X-HQSB-Client": "console", "Idempotency-Key": uuid.uuid4().hex})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    run = call("/diagnostics/jobs", {"source_run_id": args.source, "tool": args.tool,
                "launch_count": 1, "sections": "compute", "timeout_s": 900})
    print("submitted", run["id"], flush=True)
    deadline = time.monotonic() + 1050
    last = None
    while time.monotonic() < deadline:
        run = call("/runs/" + run["id"])
        current = (run["state"], run.get("progress", {}).get("stage"))
        if current != last:
            print(current, flush=True)
            last = current
        if run["state"] in {"completed", "failed", "timed_out", "cancelled", "interrupted"}:
            break
        time.sleep(2)
    (root / (run["id"] + ".json")).write_text(json.dumps(run, ensure_ascii=False, indent=2))
    if args.expect_counter_permission_error:
        assert run["state"] == "failed" and "GPU_COUNTER_PERMISSION_REQUIRED" in (run["error"] or ""), run.get("error")
        print("permission failure recorded; hardware counters remain unavailable", flush=True)
    else:
        assert run["state"] == "completed", run.get("error")
        receipt = run["metrics"]["diagnostic"]
        assert receipt["panel"]["availability"] == "observed"
        assert receipt["replay_workloads_match_source"]
        assert receipt["attempts"] and all(a["engine"]["diagnostic_instrumentation"] for a in receipt["attempts"])
        assert receipt["attempts"][0]["engine"]["model_manifest_sha256"] == receipt["source_engine"]["model_manifest_sha256"]
        print("captured", receipt["panel"].get("counts", receipt["panel"].get("metric_rows")), "outputs_match", receipt["replay_outputs_match_source"], flush=True)
    assert run["cleanup"] == "succeeded", run["cleanup"]
    assert not call("/overview")["pending"]
    print("result", root / (run["id"] + ".json"), flush=True)


if __name__ == "__main__":
    main()
