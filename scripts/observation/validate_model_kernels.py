#!/usr/bin/env python3
"""Bounded real-model L2 integration/quality matrix, one profile per process.

This validates executability, not performance adoption. The existing zero-error
quality gate is retained; nonzero error stays a failed gate with raw evidence.
All hardware execution must go through remote_run.sh.
"""
import argparse
import array
import json
from pathlib import Path
import subprocess
import sys
import threading
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from hqsb.core.kernel_profiles import PROFILES
from hqsb.observation.storage import atomic_json


def child(args):
    from hqsb.backends.interactive import InteractivePyTorch
    from hqsb.experiments.artifacts import seal

    provider = InteractivePyTorch({"model_path": str(Path(args.model_path).expanduser()), "attention": "eager",
        "context_limit": 2048, "manifest": args.manifest, "kernel_profile": args.profile,
        "kernel_manifest": args.kernel_manifest, "device_lock_path": str(Path.home()/".cache/hqsb/device0.lock")})
    rows = []
    try:
        provider.load()
        for mode in ("full_prefill", "incremental_decode"):
            run_id = "run_" + uuid.uuid4().hex
            path = Path(args.output_root)/args.profile/run_id
            path.mkdir(parents=True)
            request = {"messages": [{"role":"user","content":"Explain cache."}], "max_output_tokens":2,
                "remaining_ms":120000, "observation_mode":"off", "capture_id":run_id,"capture_dir":str(path),
                "experiment_role":"quality", "optimization":{"changes":{"kernel_profile":args.profile}},
                "quality_plan":{"samples":[{"id":"en","text":"The GPU reads model weights from memory."},
                                            {"id":"zh","text":"推理需要读取权重和缓存。"}],
                                "max_tokens":32,"generation_tokens":2,"forward_mode":mode}}
            result = None
            for event in provider.generate(request, threading.Event()):
                if event["kind"] == "result":
                    result = event
            if result is None:
                raise RuntimeError("VALIDATION_NO_RESULT")
            seal(path,run_id=run_id)
            atomic_json(path/"result.json",result)
            rows.append({"mode":mode,"path":str(path),"kernel_status":result["metrics"]["optimization_receipt"]["kernel_receipt"]["status"]})
        atomic_json(Path(args.output_root)/args.profile/"validation.json",{"profile":args.profile,"runs":rows,"status":"executed"})
    finally:
        provider.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--kernel-manifest",required=True)
    parser.add_argument("--model-path",default="~/models/hqsb/Qwen3-1.7B")
    parser.add_argument("--output-root",default="experiment_results/L2/validation")
    parser.add_argument("--profile",choices=list(PROFILES))
    parser.add_argument("--analyze-only",action="store_true")
    parser.add_argument("--full-comparison",action="store_true",help="Compute all G01 metrics; slower on the target CPU")
    parser.add_argument("--profiles",nargs="+",choices=list(PROFILES),default=list(PROFILES))
    args = parser.parse_args()
    args.output_root = str(Path(args.output_root).resolve())
    if args.profile:
        child(args)
        return
    directory = Path(args.output_root)
    if not args.analyze_only:
        directory.mkdir(parents=True,exist_ok=False)
        atomic_json(directory/"protocol.json",{"quality_limits":{"max_logit_abs_error":0.,"max_kl":0.,
            "max_nll_increase":0.,"max_accuracy_drop":0.},"purpose":"integration_not_performance_adoption",
            "modes":["full_prefill","incremental_decode"],"profiles":args.profiles})
    records = []
    for name in ([] if args.analyze_only else args.profiles):
        command = [sys.executable,__file__,"--profile",name,"--manifest",args.manifest,
                   "--kernel-manifest",args.kernel_manifest,"--model-path",args.model_path,"--output-root",args.output_root]
        with (directory/(name+".log")).open("w") as stream:
            process = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=480)
        record = {"profile":name,"exit_code":process.returncode}
        records.append(record)
        atomic_json(directory/"progress.json",records)
        print(json.dumps(record),flush=True)
    if args.analyze_only:
        records = json.loads((directory/"progress.json").read_text())
    from hqsb.experiments.artifacts import load_bundle
    from hqsb.experiments.comparison import _quality

    baseline_file = directory/"framework-native/validation.json"
    baseline = json.loads(baseline_file.read_text()) if baseline_file.exists() else None
    limits = json.loads((directory/"protocol.json").read_text())["quality_limits"]
    for record in records:
        path = directory/record["profile"]/"validation.json"
        if not path.exists() or not baseline:
            continue
        runs = json.loads(path.read_text())["runs"]
        record["quality"] = []
        for a,b in zip(baseline["runs"],runs):
            left,right = load_bundle(a["path"]),load_bundle(b["path"])
            if args.full_comparison:
                quality = _quality(Path(a["path"]),Path(b["path"]),left["quality"],right["quality"],limits)
            else:
                x,y = left["quality"],right["quality"]
                if x["protocol"] != y["protocol"] or x["dataset_sha256"] != y["dataset_sha256"] or len(x["positions"]) != len(y["positions"]):
                    raise ValueError("MATRIX_QUALITY_PROTOCOL_MISMATCH")
                rows = []
                for u,v in zip(x["positions"],y["positions"]):
                    if any(u[key] != v[key] for key in ("sample_id","position","vocab_size")):
                        raise ValueError("MATRIX_POSITION_MISMATCH")
                    vectors = []
                    for run,item in ((a,u),(b,v)):
                        vector = array.array("f")
                        vector.frombytes((Path(run["path"])/item["path"]).read_bytes())
                        if sys.byteorder != "little":
                            vector.byteswap()
                        vectors.append(vector)
                    error = max(abs(p-q) for p,q in zip(*vectors))
                    rows.append({"position":u["position"],"sample_id":u["sample_id"],
                                 "logit_max_abs_error":error,"nll_increase":v["nll"]-u["nll"]})
                exact = all(row["logit_max_abs_error"]==0 and row["nll_increase"]<=0 for row in rows)
                quality = {"status":"passed" if exact else "failed","positions":rows,
                    "scope":"strict_zero_error_screen_not_full_g01_metrics",
                    "kl_status":"zero_for_identical_logits_otherwise_not_computed_gate_already_failed",
                    "baseline_perplexity":x["perplexity"],"candidate_perplexity":y["perplexity"],
                    "free_generation_tokens_equal":[s["output_token_sha256"] for s in x["samples"]]==[s["output_token_sha256"] for s in y["samples"]]}
            record["quality"].append({"mode":b["mode"],"kernel_status":b["kernel_status"],
                "status":quality["status"],"max_logit_abs_error":max(x["logit_max_abs_error"] for x in quality["positions"]),
                "free_generation_tokens_equal":quality["free_generation_tokens_equal"],
                "baseline_perplexity":quality["baseline_perplexity"],"candidate_perplexity":quality["candidate_perplexity"]})
            atomic_json(Path(b["path"])/("quality-comparison.json" if args.full_comparison else "quality-zero-screen.json"),quality)
    atomic_json(directory/"summary.json",{"records":records,"adoption":"not_evaluated_no_performance_pairs"})


if __name__ == "__main__":
    main()
