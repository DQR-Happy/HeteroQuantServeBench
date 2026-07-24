"""Explicit real-device acceptance, invoked through remote_run.sh only.

Loads the configured reference model, submits true inference, validates cancellation,
then leaves it ready for browser acceptance. Artifacts contain test prompts only.
"""

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--data-dir", type=Path, default=Path(".console"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/console/live-acceptance.json")
    )
    args = parser.parse_args()
    token = (args.data_dir / "access-token").read_text().strip()
    client = httpx.Client(
        base_url=args.url + "/api/console/v1",
        headers={"Authorization": "Bearer " + token},
        timeout=30,
    )
    evidence = {
        "started_at": time.time(),
        "mode": "real_device",
        "checks": [],
        "runs": [],
    }

    def check(name, condition, details=None):
        evidence["checks"].append(
            {"name": name, "passed": bool(condition), "details": details}
        )
        assert condition, name

    def post(path, payload=None, key=None):
        response = client.post(
            path,
            json=payload or {},
            headers={"Idempotency-Key": key or uuid.uuid4().hex},
        )
        response.raise_for_status()
        return response.json()

    def wait(run, timeout=300):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            row = client.get("/runs/" + run["id"]).json()
            if row["state"] in {
                "completed",
                "failed",
                "cancelled",
                "timed_out",
                "interrupted",
            }:
                evidence["runs"].append(row)
                return row
            time.sleep(0.3)
        raise TimeoutError(run["id"])

    try:
        deployment = client.get("/deployments").json()["items"][0]
        if deployment["state"] != "ready":
            loaded = wait(post("/deployments/" + deployment["id"] + "/load"))
            check(
                "real_model_load", loaded["state"] == "completed", loaded.get("error")
            )
        deployment = client.get("/deployments").json()["items"][0]
        evidence["deployment"] = deployment
        body = {
            "deployment_id": deployment["id"],
            "expected_epoch": deployment["epoch"],
            "messages": [
                {
                    "role": "user",
                    "content": "请简要解释 GPU 内存带宽为什么影响大模型逐 token 解码。",
                }
            ],
            "max_output_tokens": 48,
            "deadline_ms": 120000,
            "save_input": True,
        }
        key = uuid.uuid4().hex
        run = post("/requests", body, key)
        check(
            "real_idempotent_submission",
            post("/requests", body, key)["id"] == run["id"],
        )
        first = wait(run)
        check(
            "real_chinese_inference",
            first["state"] == "completed" and bool(first["output"]),
            first.get("error"),
        )
        metrics = first["metrics"]
        check(
            "actual_token_accounting",
            metrics["output_tokens"] == len(metrics["generated_token_ids"])
            and len(metrics["token_itl_ms"]) == metrics["output_tokens"] - 1,
        )
        check(
            "layered_ttft_measured",
            metrics["console_first_content_ms"]
            >= metrics["runtime_first_token_ms"]
            > 0,
        )
        check("worker_memory_measured", metrics["memory"]["peak_allocated_bytes"] > 0)
        response = client.get("/requests/" + run["id"] + "/events?after=2")
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        check(
            "sse_replay_order_and_terminal",
            events[0]["seq"] == 3
            and events[-1]["kind"] == "completed"
            and all(b["seq"] == a["seq"] + 1 for a, b in zip(events, events[1:])),
        )
        second = wait(post("/requests", body))
        check(
            "greedy_repeat_token_identity",
            second["metrics"].get("generated_token_ids")
            == metrics["generated_token_ids"],
        )
        comparison = post("/comparisons", {"run_ids": [first["id"], second["id"]]})
        check("same_configuration_comparison", comparison["comparable"])
        long = post("/requests", {**body, "max_output_tokens": 512})
        for _ in range(100):
            row = client.get("/runs/" + long["id"]).json()
            if row["output"]:
                break
            time.sleep(0.1)
        post("/requests/" + long["id"] + "/cancel")
        cancelled = wait(long)
        check(
            "real_cancellation_and_cleanup",
            cancelled["state"] == "cancelled" and cancelled["cleanup"] == "succeeded",
        )
        followup = wait(post("/requests", {**body, "max_output_tokens": 8}))
        check("reuse_after_cancel", followup["state"] == "completed")
        deadline_run = wait(
            post("/requests", {**body, "max_output_tokens": 512, "deadline_ms": 1000})
        )
        check(
            "real_deadline_and_cleanup",
            deadline_run["state"] == "timed_out"
            and deadline_run["cleanup"] == "succeeded",
        )
        rejected = post("/comparisons", {"run_ids": [first["id"], followup["id"]]})
        check("different_output_length_blocked", not rejected["comparable"])
        catalog = client.get("/evidence").json()["items"]
        check("real_history_indexed", len(catalog) > 0, len(catalog))
        raw = client.get("/evidence/" + catalog[0]["id"] + "/download")
        check(
            "evidence_download_hash",
            hashlib.sha256(raw.content).hexdigest() == raw.headers["x-content-sha256"],
        )
        check(
            "telemetry_has_real_samples",
            len(client.get("/telemetry").json()["samples"]) > 0,
        )
        unloaded = wait(post("/deployments/" + deployment["id"] + "/unload"))
        check(
            "real_model_unload",
            unloaded["state"] == "completed"
            and client.get("/deployments").json()["items"][0]["state"] == "unloaded",
        )
        reloaded = wait(post("/deployments/" + deployment["id"] + "/load"))
        check(
            "real_model_reload_new_epoch",
            reloaded["state"] == "completed"
            and client.get("/deployments").json()["items"][0]["epoch"]
            != deployment["epoch"],
        )
        stale = client.post(
            "/requests", json=body, headers={"Idempotency-Key": uuid.uuid4().hex}
        )
        check(
            "real_stale_epoch_rejected",
            stale.status_code == 409 and stale.json()["error"]["code"] == "STALE_EPOCH",
        )
        evidence["verdict"] = "PASS"
    except Exception as exc:
        evidence["verdict"] = "FAIL"
        evidence["error"] = repr(exc)
        raise
    finally:
        evidence["finished_at"] = time.time()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {
                    "verdict": evidence["verdict"],
                    "checks": len(evidence["checks"]),
                    "output": str(args.output),
                }
            )
        )
        client.close()


if __name__ == "__main__":
    main()
