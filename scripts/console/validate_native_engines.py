"""Remote real-model acceptance through the same HTTP APIs used by the console."""
import argparse
import json
import os
import subprocess
from pathlib import Path
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8768")
    parser.add_argument("--source", default="qwen-reference")
    parser.add_argument("--engine", choices=["vllm", "sglang", "both"], default="both")
    parser.add_argument("--prefix-cache", choices=["off", "on", "both"], default="both")
    parser.add_argument("--official-crosscheck", action="store_true")
    args = parser.parse_args()
    root = Path("reports/console/native-engines")
    root.mkdir(parents=True, exist_ok=True)
    def call(path, body=None):
        req = urllib.request.Request(args.url + "/api/console/v1" + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json", "X-HQSB-Client": "console", "Idempotency-Key": uuid.uuid4().hex})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    def task(path, body):
        run = call(path, body)
        print("submitted", path, run["id"], flush=True)
        for _ in range(350):
            row = call("/runs/" + run["id"])
            if row["state"] in {"completed", "failed", "timed_out", "cancelled", "interrupted"}:
                (root / (row["id"] + ".json")).write_text(json.dumps(row, ensure_ascii=False, indent=2))
                assert row["state"] == "completed", row.get("error")
                return row
            time.sleep(1)
        raise TimeoutError(run["id"])
    rows = call("/deployments")["items"]
    assert not call("/overview")["pending"]
    assert not any(row["state"] not in {"unloaded", "failed"} for row in rows), "Unload resident models before acceptance"
    outputs = []
    for engine in (["vllm", "sglang"] if args.engine == "both" else [args.engine]):
        deployment = call("/native-engines/deployments", {"source_deployment_id": args.source, "engine": engine})
        identity = deployment["id"]
        options = {"max_batch_tokens": 1024, "max_running_requests": 8, "memory_fraction": 0.5,
                   "dtype": deployment["engine_options"]["dtype"]}
        if engine == "sglang":
            options.update(attention_backend="triton", block_size=1)
        try:
            for tuning in ([False, True] if args.prefix_cache == "both" else [args.prefix_cache == "on"]):
                options["prefix_cache"] = tuning
                loaded = task(f"/deployments/{identity}/load", {"engine_options": options})
                assert loaded["result"]["native_engine_receipt"]["requested"]["prefix_cache"] is tuning
                if engine == "sglang":
                    assert loaded["result"]["native_engine_receipt"]["effective_engine_config"]["disable_radix_cache"] is not tuning
                state = next(row for row in call("/deployments")["items"] if row["id"] == identity)
                generated = task("/requests", {"deployment_id": identity, "expected_epoch": state["epoch"],
                    "messages": [{"role": "user", "content": "Compute 17 + 25. Give only the number."}],
                    "max_output_tokens": 16, "observation_mode": "basic"})
                assert generated["output"].strip(), generated
                bench = task("/native-engines/benchmarks", {"deployment_id": identity, "expected_epoch": state["epoch"],
                    "prompts": ["Compute 17 + 25. Give only the number.", "Compute 2 + 2. Give only the number."],
                    "expected_outputs": ["42", "4"], "max_output_tokens": 16, "concurrency": 2, "rounds": 2})
                summary = bench["metrics"]["native_benchmark"]["summary"]
                assert summary["completed"] == 4 and summary["output_tokens_per_s"] > 0, summary
                assert summary["usage_complete"] and summary["quality_exact_match_rate"] == 1, summary
                print(engine, "prefix", tuning, json.dumps(summary), flush=True)
                outputs.append(bench["id"])
                if args.official_crosscheck:
                    receipt = loaded["result"]["native_engine_receipt"]
                    argv = receipt["launch_argv"]
                    port = argv[argv.index("--port") + 1]
                    config = next(d for d in json.loads(Path(".console/native-deployments.json").read_text())["items"] if d["id"] == identity)
                    out = root / ("official-" + engine)
                    out.mkdir(exist_ok=True)
                    command = [str(Path(".venv-vllm/bin/vllm").absolute()), "bench", "serve", "--backend", "openai",
                        "--base-url", "http://127.0.0.1:" + port, "--model", config["model"],
                        "--tokenizer", str(Path(config["model_path"]).expanduser()),
                        "--dataset-name", "random", "--random-input-len", "64", "--random-output-len", "16",
                        "--num-prompts", "8", "--max-concurrency", "2", "--ignore-eos", "--seed", "0",
                        "--save-result", "--save-detailed", "--result-dir", str(out), "--result-filename", "result.json"]
                    with (out / "client.log").open("w") as stream:
                        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=120, check=True,
                                       env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
                    measured = json.loads((out / "result.json").read_text())
                    assert measured["completed"] == 8 and measured["total_output_tokens"] == 128, measured
                    print("official crosscheck", engine, measured["completed"], measured["total_output_tokens"], flush=True)
                task(f"/deployments/{identity}/unload", {})
        finally:
            state = next(row for row in call("/deployments")["items"] if row["id"] == identity)
            if state["state"] == "ready":
                task(f"/deployments/{identity}/unload", {})
    if args.engine == "both" and len(outputs) in {2, 4}:
        report = call("/native-engines/comparisons", {"baseline_run_id": outputs[0], "candidate_run_id": outputs[len(outputs)//2]})
        assert report["comparable"], report
        (root / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (root / "acceptance.json").write_text(json.dumps({"benchmark_runs": outputs, "scope": "real_model_smoke_and_reload"}, indent=2))


if __name__ == "__main__":
    main()
