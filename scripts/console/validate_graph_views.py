"""Explicit target acceptance; idle console only; restores the resident recipe."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.console import validate_l3 as api

OUT = Path("reports/console/graph-views")


def restart():
    # Stop only this checkout's console listener, after all model workers unloaded.
    import psutil
    processes = []
    for proc in psutil.process_iter(["cmdline", "cwd"]):
        args = proc.info["cmdline"] or []
        if "hqsb.console" in args and "--port" in args and "8768" in args and proc.info["cwd"] == str(Path.cwd()):
            processes.append(proc)
    assert len(processes) == 1, [p.pid for p in processes]
    processes[0].send_signal(signal.SIGTERM)
    processes[0].wait(timeout=45)
    log = (OUT / "service.log").open("ab")
    subprocess.Popen([sys.executable, "-m", "hqsb.console", "--config", "configs/console/3090-inf.yaml", "--port", "8768"],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        env={**os.environ, "TORCHINDUCTOR_COMPILE_THREADS": "2", "OMP_NUM_THREADS": "2"})
    log.close()
    for _ in range(60):
        try:
            api.call("/runtime/catalog")
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("console restart failed; see service.log")


def main(snapshot, do_restart):
    OUT.mkdir(parents=True, exist_ok=True)
    api.OUT = OUT
    original = json.loads(Path(snapshot).read_text())
    assert api.call("/overview")["pending"] == 0
    assert api.call("/scheduling-benchmarks")["active_id"] is None
    assert api.call("/benchmark-catalog?deployment_id=" + original["id"])["active_id"] is None
    recipe = {k:v for k,v in original["detail"]["precision_receipt"]["resolved_recipe"].items() if k != "calibration"}
    if original.get("quantization", {}).get("artifact_id"):
        recipe["artifact_id"] = original["quantization"]["artifact_id"]
    summary = {"models": {}, "performance_claim": False}
    try:
        api.unload_all("initial")
        if do_restart:
            restart()
        deadline = api.call("/runtime/catalog")["limits"]["max_deadline_ms"]
        for identity in ["ms-586060631c81c17023c0", "ms-6d9b6a4a3f944e041fd6"]:
            print("MODEL", identity, flush=True)
            api.unload_all("next")
            api.wait(api.call(f"/deployments/{identity}/load", {"precision":"float16", "kernel_profile":"framework-native", "quantization":{"method":"none"}}), identity+"-load")
            d = next(d for d in api.call("/deployments")["items"] if d["id"] == identity)
            body = {"deployment_id": identity, "expected_epoch": d["epoch"],
                    "prompt":"Once upon a time, there was a little", "max_output_tokens":2,
                    "observation_mode":"off", "deadline_ms":deadline}
            before = api.wait(api.call("/requests", body), identity+"-before")
            rows = summary["models"][identity] = {}
            for kind in ["onnx", "inductor"]:
                plan = {"mode":"eager" if kind == "onnx" else "inductor", "shape_policy":"static", "fullgraph":False}
                run = api.wait(api.call("/requests", {**body, "runtime_plan":plan, "graph_capture":{"kind":kind}}), identity+"-"+kind)
                views = api.call(f'/runs/{run["id"]}/graph-views')
                api.save(identity+"-"+kind+"-views", views)
                rows[kind] = {"run_id":run["id"], "statuses":[{"kind":v["kind"], "status":v["status"], "nodes":len(v.get("nodes",[])), "edges":len(v.get("edges",[])), "error":v.get("error")} for v in views["views"]]}
                api.save("summary", summary)
                print(kind, json.dumps(rows[kind]), flush=True)
                assert views["views"] and all(v["status"] == "available" for v in views["views"]), rows[kind]
                assert all(v.get("nodes") for v in views["views"] if v["kind"] != "generated_code")
                if kind == "onnx":
                    assert run["metrics"]["generated_token_ids"] == before["metrics"]["generated_token_ids"]
                # All public files must be accessible only through sealed indices.
                for v in views["views"]:
                    assert type(v["artifact_index"]) is int
                if kind == "inductor":
                    assert any(v.get("edges") for v in views["views"] if v["kind"] == "inductor_post")
                    again = api.wait(api.call("/requests", {**body, "runtime_plan":plan, "graph_capture":{"kind":kind}}), identity+"-inductor-repeat")
                    reused = api.call(f'/runs/{again["id"]}/graph-views')
                    assert all(v["status"] == "available" for v in reused["views"])
                    assert any(v.get("first_run_id") == run["id"] for v in reused["views"])
                    rows["inductor_repeat"] = again["id"]
            after = api.wait(api.call("/requests", body), identity+"-legacy-after")
            assert after["metrics"]["generated_token_ids"] == before["metrics"]["generated_token_ids"]
            rows["legacy_preserved"] = True
    finally:
        print("RESTORE", original["id"], flush=True)
        api.unload_all("restore")
        api.wait(api.call(f'/deployments/{original["id"]}/load', {"precision":original["precision"], "kernel_profile":original.get("kernel_profile","framework-native"), "quantization":recipe}), "restored-load")
        current = next(d for d in api.call("/deployments")["items"] if d["id"] == original["id"])
        assert current["quantization"]["method"] == original["quantization"]["method"]
        assert current["quantization"]["kv"] == original["quantization"]["kv"]
        assert current["quantization"].get("artifact_id") == original["quantization"].get("artifact_id")
        summary["restored"] = True
        api.save("restored", current)
        api.save("summary", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--restart", action="store_true")
    args = p.parse_args()
    main(args.snapshot, args.restart)
