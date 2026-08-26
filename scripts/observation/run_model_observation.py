#!/usr/bin/env python3
"""Real fixed-workload capture on the target. Invoke through remote_run.sh."""

import argparse
import json
from pathlib import Path
import sys
import threading
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hqsb.backends.interactive import InteractivePyTorch
from hqsb.observation.storage import atomic_json, publish
from hqsb.observation.analysis import analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["off", "basic", "memory", "pages", "operators"],
        default="basic",
    )
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument(
        "--cleanup-policy",
        choices=["empty_cache", "preserve_allocator"],
        default="empty_cache",
    )
    parser.add_argument("--isl", type=int, default=128)
    parser.add_argument("--osl", type=int, default=16)
    parser.add_argument("--model-path", default="~/models/hqsb/Qwen3-1.7B")
    parser.add_argument("--manifest")
    parser.add_argument(
        "--device-lock-path", default=str(Path.home() / ".cache/hqsb/device0.lock")
    )
    parser.add_argument(
        "--nvtx", action="store_true", help="Explicit tool-only phase annotations"
    )
    parser.add_argument("--output-root", default="experiment_results/OBS/OBS-01")
    parser.add_argument("--run-id")
    parser.add_argument("--optimization-json")
    parser.add_argument("--observation-plan-json")
    parser.add_argument("--experiment-role", choices=["performance", "diagnostic", "quality"])
    parser.add_argument("--experiment-links-json")
    parser.add_argument("--quality-plan")
    args = parser.parse_args()
    optimization = json.loads(args.optimization_json) if args.optimization_json else None
    from hqsb.core.optimization import resolve_plan

    resolved = resolve_plan(optimization, {"attention_backend": args.attention,
                           "cleanup_policy": args.cleanup_policy, "output_budget": args.osl})
    args.attention = resolved["resolved"]["attention_backend"]
    if (
        not 1 <= args.isl <= 2048
        or not 1 <= args.osl <= 512
        or args.isl + args.osl > 4096
    ):
        parser.error("Invalid workload bounds")
    from hqsb.observation.contracts import RUN_ID

    run_id = args.run_id or "run_" + uuid.uuid4().hex
    if not RUN_ID.fullmatch(run_id):
        parser.error("Invalid run ID")
    path = Path(args.output_root).expanduser().resolve() / run_id
    path.mkdir(parents=True, exist_ok=False)
    atomic_json(path / "execution-receipt.json", {**resolved, "run_id": run_id,
                "status": "resolved_not_applied", "measurement_role": args.experiment_role or ("performance" if args.mode == "off" else "diagnostic"),
                "proof_scope": "load_attempt_only", "execution_status": "not_started"})
    provider = InteractivePyTorch(
        {
            "model_path": str(Path(args.model_path).expanduser()),
            "attention": args.attention,
            "context_limit": 4096,
            "manifest": args.manifest,
            "device_lock_path": str(Path(args.device_lock_path).expanduser().resolve()),
        }
    )
    result = None
    try:
        loaded = provider.load()
        pattern = provider.tokenizer.encode(
            "请解释模型推理和内存缓存。", add_special_tokens=False
        )
        tokens = (pattern * (args.isl // len(pattern) + 1))[: args.isl]
        request = {
            "messages": [{"role": "user", "content": "fixed-token replay"}],
            "input_token_ids": tokens,
            "fixed_output_length": True,
            "max_output_tokens": args.osl,
            "remaining_ms": 600_000,
            "observation_mode": args.mode,
            "capture_dir": str(path),
            "capture_id": run_id,
        }
        request["tool_ranges"] = args.nvtx
        request["cleanup_policy"] = args.cleanup_policy
        request.update(optimization=optimization,
                       observation_plan=json.loads(args.observation_plan_json) if args.observation_plan_json else None,
                       experiment_role=args.experiment_role,
                       experiment_links=json.loads(args.experiment_links_json) if args.experiment_links_json else {})
        if args.experiment_role == "quality":
            if not args.quality_plan:
                raise ValueError("QUALITY_PLAN_REQUIRED")
            request["quality_plan"] = json.loads(Path(args.quality_plan).read_text())
        atomic_json(
            path / "recipe.json",
            {
                "argv": sys.argv,
                "run_id": run_id,
                "input_token_ids": tokens,
                "load": loaded,
                "source": provider.source_environment,
            },
        )
        for event in provider.generate(request, threading.Event()):
            if event["kind"] == "result":
                result = event
        if result is None:
            raise RuntimeError("NO_GENERATION_RESULT")
        if args.mode != "off":
            manifest = publish(path, expected_run_id=run_id)
            result["metrics"]["observation_bundle"].update(
                publication=manifest["publication"],
                capture_status=manifest["capture_status"],
            )
            atomic_json(path / "analysis.json", analyze(path))
        from hqsb.experiments.artifacts import seal

        if (path / "execution-receipt.json").exists():
            result["metrics"]["experiment_bundle"] = seal(path, run_id=run_id,
                status="complete" if result["finish_reason"] not in {"cancelled", "timed_out"} else "partial")
        atomic_json(path / "result.json", result)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "path": str(path),
                    "finish_reason": result["finish_reason"],
                    "output_tokens": result["metrics"].get("output_tokens"),
                    "runtime_e2e_ms": result["metrics"].get("runtime_e2e_ms"),
                    "capture": result["metrics"].get("observation_bundle"),
                },
                ensure_ascii=False,
            )
        )
    except Exception as exc:
        failure = {"error": f"{type(exc).__name__}: {exc}", "run_id": run_id}
        from hqsb.experiments.artifacts import seal

        try:
            seal(path, run_id=run_id, status="partial", error=failure["error"])
        except (OSError, ValueError) as seal_error:
            failure["experiment_publication_error"] = str(seal_error)[:400]
        if (path / "observations/capture-manifest.json").exists():
            try:
                sealed = publish(
                    path, expected_run_id=run_id, supervisor_failure=failure["error"]
                )
                failure["capture_status"] = sealed["capture_status"]
            except (OSError, ValueError) as capture_error:
                failure["publication_error"] = str(capture_error)[:400]
        atomic_json(path / "failure.json", failure)
        raise
    finally:
        provider.close()


if __name__ == "__main__":
    main()
