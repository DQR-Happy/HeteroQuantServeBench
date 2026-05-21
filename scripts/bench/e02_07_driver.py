#!/usr/bin/env python3
"""E02-07 frozen-workload driver shared by all three profiling tools.

One driver, three behaviours
----------------------------
The experiment is only valid if PyTorch Profiler, Nsight Systems and Nsight
Compute observe *the same* workload. Instead of three scripts that could
drift apart, this driver always executes the identical model-core sequence
and only changes the instrumentation:

``--profiler off``
    Un-instrumented reference: the E02-01 clock convention
    (``host_monotonic_ns`` + ``torch.cuda.synchronize`` at every phase
    boundary) and ``hqsb.benchmark.model_core.benchmark_model_core`` for the
    derived metrics. NVTX ranges are still emitted because they are free, so
    Nsight Systems and Nsight Compute can select sub-regions.
``--profiler light|detailed``
    Same sequence, plus a ``torch.profiler.profile`` that is started and
    stopped around the pre-registered windows (prefill, decode-early,
    decode-late) and exported as one Chrome trace. ``light`` collects
    op/call/device time plus shapes; ``detailed`` additionally collects
    memory and stacks. They are separate *runs*, never one run with every
    heaviest option, because the protocol forbids that.
``--layer0-roles``
    Additionally annotate the submodules of one audited layer with
    ``e02_07_role.<role>`` ranges. That is the anchor for the
    module -> op -> kernel mapping: a kernel found inside
    ``e02_07_role.q_proj`` is evidence, a kernel matched by name would only
    be a guess.

Phase ranges (frozen, see :mod:`hqsb.benchmark.multilevel_profiling`)::

    e02_07_run
      e02_07_prefill
      e02_07_first_token_selection
      e02_07_decode_early      (first E decode steps, profiled)
      e02_07_decode_middle     (remaining decode steps, not profiled)
      e02_07_decode_late       (last L decode steps, profiled)
      e02_07_result_handling   (host work, never a model phase)

Only ``early`` and ``late`` are profiled. The middle steps still execute and
are still annotated, so the decode window is partitioned exactly once and the
unattributed bucket stays empty for model work.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from hqsb.benchmark.correctness import hash_token_sequence
from hqsb.benchmark.model_core import benchmark_model_core
from hqsb.benchmark.multilevel_profiling import (
    DECODE_EARLY_RANGE,
    DECODE_LATE_RANGE,
    DECODE_MIDDLE_RANGE,
    FIRST_TOKEN_RANGE,
    MODULE_ROLE_PREFIX,
    PREFILL_RANGE,
    RESULT_RANGE,
    RUN_RANGE,
    phase_ledger,
)
from hqsb.benchmark.workload import make_fixed_token_input
from hqsb.models.loader import load_qwen3

logger = logging.getLogger("e02_07.driver")

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ── environment / identity ───────────────────────────────────────────────


def _shell(command: List[str], timeout: int = 30) -> Optional[str]:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _file_hash(path: str) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _thermal_c() -> Optional[List[float]]:
    temps: List[float] = []
    for zone in sorted(Path("/sys/devices/virtual/thermal").glob("thermal_zone*")):
        try:
            temps.append(int((zone / "temp").read_text().strip()) / 1000.0)
        except Exception:
            continue
    return temps or None


def _gpu_cur_freq_hz() -> Optional[int]:
    for path in Path("/sys").glob("devices/gpu.0/devfreq/*/cur_freq"):
        try:
            return int(path.read_text().strip())
        except Exception:
            continue
    return None


def _environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "cudnn_version": torch.backends.cudnn.version(),
        "nvpmodel": _shell(["nvpmodel", "-q"]),
        "jetson_clocks": _shell(["sudo", "-n", "jetson_clocks", "--show"]),
        "gpu_cur_freq_hz": _gpu_cur_freq_hz(),
        "thermal_c": _thermal_c(),
        "caching_allocator_disabled": os.environ.get(
            "PYTORCH_NO_CUDA_MEMORY_CACHING"
        )
        == "1",
    }
    try:
        import transformers

        env["transformers_version"] = transformers.__version__
    except Exception:
        env["transformers_version"] = None
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        env["device"] = torch.cuda.get_device_name(0)
        env["compute_capability"] = [int(major), int(minor)]
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        env["device_free_bytes_at_start"] = int(free_bytes)
        env["device_total_bytes"] = int(total_bytes)
    return env


# ── Annotation helpers ───────────────────────────────────────────────────
#
# Two independent mechanisms carry the same frozen range names, because the
# three tools read different channels:
#
# * NVTX (``nvtxRangePush/Pop``) is what Nsight Systems and Nsight Compute
#   can select on (``--capture-range=nvtx``, ``--nvtx-include``).
# * ``torch.profiler.record_function`` is what Kineto writes into the Chrome
#   trace as ``cat="user_annotation"``. Kineto does **not** forward external
#   NVTX ranges into its own trace, so relying on NVTX alone would leave the
#   PyTorch-profiler trace with no phase boundaries at all.
#
# Both are disabled for the un-instrumented reference pass, which must stay
# free of any annotation so its wall-clock numbers mean what they say.


class Annotator:
    """Balanced, failure-tolerant range annotations for one run."""

    def __init__(self, *, nvtx: bool, record_function: bool) -> None:
        self.nvtx = nvtx
        self.record_function = record_function
        self._stack: List[Any] = []

    def push(self, name: str) -> None:
        if self.nvtx:
            try:
                torch.cuda.nvtx.range_push(name)
            except Exception:
                pass
        handle = None
        if self.record_function:
            try:
                handle = torch.profiler.record_function(name)
                handle.__enter__()
            except Exception:
                handle = None
        self._stack.append(handle)

    def pop(self) -> None:
        handle = self._stack.pop() if self._stack else None
        if handle is not None:
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
        if self.nvtx:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass

    def close_all(self) -> None:
        while self._stack:
            self.pop()

    def range(self, name: str) -> "_AnnotatedRange":
        return _AnnotatedRange(self, name)


class _AnnotatedRange:
    def __init__(self, annotator: Annotator, name: str) -> None:
        self._annotator = annotator
        self.name = name

    def __enter__(self) -> "_AnnotatedRange":
        self._annotator.push(self.name)
        return self

    def __exit__(self, *exc: Any) -> bool:
        self._annotator.pop()
        return False


def _attach_layer_role_hooks(
    model: torch.nn.Module, prefix: str, annotator: Annotator
) -> List[Any]:
    """Annotate one layer's submodules with ``e02_07_role.<role>`` ranges."""
    handles: List[Any] = []
    for path, module in model.named_modules():
        if path == prefix or not path.startswith(prefix + "."):
            continue
        role = path[len(prefix) + 1 :]
        label = f"{MODULE_ROLE_PREFIX}{role}"

        def _pre(_module, _args, _kwargs, _label=label):
            annotator.push(_label)

        def _post(_module, _args, _output):
            annotator.pop()

        handles.append(module.register_forward_pre_hook(_pre, with_kwargs=True))
        handles.append(module.register_forward_hook(_post))
    return handles


def _detach(handles: List[Any]) -> None:
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass
    handles.clear()


# ── the frozen model-core sequence ───────────────────────────────────────


def _decode_step(
    model: torch.nn.Module,
    token: torch.Tensor,
    past: Any,
    context_len: int,
    device: torch.device,
) -> Any:
    """One decode forward, byte-for-byte the E02-01 reference shape."""
    mask = torch.ones((1, context_len), dtype=torch.long, device=device)
    outputs = model(
        input_ids=token,
        attention_mask=mask,
        past_key_values=past,
        use_cache=True,
    )
    return outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True), outputs.past_key_values


@torch.inference_mode()
def run_instrumented_pass(
    *,
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    output_tokens: int,
    early_steps: int,
    late_steps: int,
    profiler: Optional[Any],
    annotator: Annotator,
    profile_window: str,
    layer0_prefix: Optional[str],
    cuda_profiler_api: bool,
    trace_path: Optional[str],
) -> Dict[str, Any]:
    """Execute the annotated model-core sequence and collect phase evidence.

    ``profile_window`` selects which part of the (always complete) generation
    is collected by the PyTorch profiler:

    ``full``
        One session from the prefill to the last decode step.
    ``early``
        One session from the prefill through ``decode_early``; the middle and
        late windows still execute, they are just not collected.
    ``late``
        One session over ``decode_late`` only. Reaching the late window still
        requires the whole generation, which is why this is a separate run.

    Only *one* ``start``/``stop`` session is ever used per run. On this
    PyTorch/Kineto build a second session silently discards the annotations
    collected by the first, so "profile early then late in one process" would
    have produced a trace whose late window is the only correctly labelled
    one — a failure mode that is easy to miss and impossible to audit. This
    was verified explicitly with a minimal two-session probe.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    device = inputs["input_ids"].device
    input_len = int(input_ids.shape[1])
    total_decode = output_tokens - 1

    ledger = phase_ledger(
        input_len=input_len,
        output_tokens=output_tokens,
        early_steps=early_steps,
        late_steps=late_steps,
    )
    middle_steps = len(ledger["decode_middle"]["steps"])

    order = (
        PREFILL_RANGE,
        FIRST_TOKEN_RANGE,
        DECODE_EARLY_RANGE,
        DECODE_MIDDLE_RANGE,
        DECODE_LATE_RANGE,
    )
    if profile_window == "full":
        profiled = set(order)
    elif profile_window == "early":
        profiled = {PREFILL_RANGE, FIRST_TOKEN_RANGE, DECODE_EARLY_RANGE}
    elif profile_window == "late":
        profiled = {DECODE_LATE_RANGE}
    else:
        profiled = set()

    active = [window for window in order if window in profiled]
    session_first = active[0] if active else None
    session_last = active[-1] if active else None

    roles = (
        _attach_layer_role_hooks(model, layer0_prefix, annotator)
        if layer0_prefix
        else []
    )

    phase_wall: Dict[str, float] = {}
    step_itl: List[Dict[str, Any]] = []
    profiled_windows: List[str] = []

    def _begin(name: str) -> None:
        if profiler is not None and name == session_first:
            profiler.start()

    def _end(name: str) -> None:
        if profiler is not None and name == session_last:
            profiler.stop()
            profiled_windows.extend(active)

    if device.type == "cuda":
        torch.cuda.synchronize()

    try:
        if cuda_profiler_api:
            torch.cuda.profiler.start()

        annotator.push(RUN_RANGE)

        # ── prefill ───────────────────────────────────────────────────
        _begin(PREFILL_RANGE)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        annotator.push(PREFILL_RANGE)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        annotator.pop()
        if device.type == "cuda":
            torch.cuda.synchronize()
        phase_wall["prefill"] = (time.perf_counter() - start) * 1000.0
        _end(PREFILL_RANGE)

        # ── first-token selection (kept separate from the prefill) ────
        _begin(FIRST_TOKEN_RANGE)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        annotator.push(FIRST_TOKEN_RANGE)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        annotator.pop()
        if device.type == "cuda":
            torch.cuda.synchronize()
        phase_wall["first_token_selection"] = (time.perf_counter() - start) * 1000.0
        _end(FIRST_TOKEN_RANGE)
        past = outputs.past_key_values
        del outputs

        # ── decode: early / middle / late ─────────────────────────────
        generated: List[int] = [int(next_token.item())]
        context_len = input_len
        windows = [
            (DECODE_EARLY_RANGE, list(range(1, early_steps + 1))),
            (
                DECODE_MIDDLE_RANGE,
                list(range(early_steps + 1, early_steps + middle_steps + 1)),
            ),
            (
                DECODE_LATE_RANGE,
                list(range(early_steps + middle_steps + 1, total_decode + 1)),
            ),
        ]

        for range_name, steps in windows:
            if not steps:
                continue
            _begin(range_name)
            window_ms = 0.0
            annotator.push(range_name)
            for step in steps:
                context_len += 1
                if device.type == "cuda":
                    torch.cuda.synchronize()
                step_start = time.perf_counter()
                next_token, past = _decode_step(
                    model, next_token, past, context_len, device
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - step_start) * 1000.0
                window_ms += elapsed
                generated.append(int(next_token.item()))
                step_itl.append(
                    {
                        "step": step,
                        "context_len": context_len,
                        "itl_ms": elapsed,
                        "window": range_name,
                        "profiled": range_name in profiled,
                    }
                )
            annotator.pop()
            _end(range_name)
            phase_wall[range_name] = window_ms

        annotator.pop()

        if cuda_profiler_api:
            if device.type == "cuda":
                torch.cuda.synchronize()
            torch.cuda.profiler.stop()
    finally:
        _detach(roles)
        annotator.close_all()

    # ── result handling: explicitly outside every model range ─────────
    start = time.perf_counter()
    annotator.push(RESULT_RANGE)
    sequence_sha256 = hash_token_sequence(generated)
    annotator.pop()
    phase_wall["result_handling"] = (time.perf_counter() - start) * 1000.0

    exported = False
    if profiler is not None and trace_path:
        try:
            profiler.export_chrome_trace(trace_path)
            exported = True
        except Exception as exc:  # pragma: no cover - device dependent
            logger.warning("trace export failed: %s", exc)

    return {
        "input_len": input_len,
        "output_tokens": output_tokens,
        "decode_steps": total_decode,
        "decode_probe_steps": sorted(
            ledger["decode_early"]["steps"] + ledger["decode_late"]["steps"]
        ),
        "ledger": ledger,
        "phase_wall_ms": phase_wall,
        "step_itl": step_itl,
        "generated_token_ids": generated,
        "sequence_sha256": sequence_sha256,
        "profiled_windows": profiled_windows,
        "trace": trace_path if exported else None,
        "layer0_prefix": layer0_prefix,
        "cuda_profiler_api_used": cuda_profiler_api,
    }


# ── CLI ──────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E02-07 frozen-workload driver")
    parser.add_argument("--model-path", default="~/models/hqsb/Qwen3-1.7B")
    parser.add_argument(
        "--manifest",
        default=str(_REPO_ROOT / "docs" / "benchmark" / "model_sha256_manifest.txt"),
    )
    parser.add_argument("--isl", type=int, default=128)
    parser.add_argument("--osl", type=int, default=32)
    parser.add_argument("--early-steps", type=int, default=8)
    parser.add_argument("--late-steps", type=int, default=8)
    parser.add_argument(
        "--profiler",
        choices=("off", "light", "detailed"),
        default="off",
        help="off = un-instrumented reference; light = op/call/device+shapes; "
        "detailed = light + memory + stacks",
    )
    parser.add_argument(
        "--profile-window",
        choices=("full", "early", "late"),
        default="full",
        help="which window the single profiler session covers; 'early' and "
        "'late' are separate runs because a second profiler session discards "
        "the first session's annotations on this Kineto build",
    )
    parser.add_argument(
        "--reference-pass",
        action="store_true",
        help="also run the un-instrumented E02-01 reference in this process",
    )
    parser.add_argument(
        "--cuda-profiler-api",
        action="store_true",
        help="bracket the model core with cudaProfilerStart/Stop (Nsight "
        "Systems --capture-range=cudaProfilerApi)",
    )
    parser.add_argument("--layer0-roles", action="store_true")
    parser.add_argument(
        "--layer0-prefix", default="model.layers.0", help="layer used for role ranges"
    )
    parser.add_argument("--warmup-isl", type=int, default=32)
    parser.add_argument("--trace-path", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tag", default="")
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _build_parser().parse_args()

    started_at = datetime.datetime.now(datetime.timezone.utc)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model, load_time_s = load_qwen3(
        args.model_path,
        dtype=torch.float16,
        attention_backend="eager",
        verify_manifest=args.manifest,
        allow_extra=("model_sha256_manifest.txt",),
        cpu_staging=True,
    )
    model.to(device)

    # Warm up *outside* every NVTX range and outside the profiler, with a
    # short sequence so a long prefill does not have to stage a full-length
    # buffer while the device is still cold.
    warm = make_fixed_token_input(tokenizer, args.warmup_isl, device=str(device))
    with torch.inference_mode():
        _ = model(
            input_ids=warm["input_ids"],
            attention_mask=warm["attention_mask"],
            use_cache=True,
        )
    del warm, _
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    inputs = make_fixed_token_input(tokenizer, args.isl, device=str(device))

    reference: Optional[Dict[str, Any]] = None
    if args.reference_pass:
        reference = benchmark_model_core(model, inputs, args.osl)
        logger.info(
            "reference pass: TTFT=%.2f ms E2E=%.2f ms",
            reference["model_core_ttft_ms"],
            reference["model_core_e2e_ms"],
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize()

    if args.profiler == "off":
        profiler = None
        profiler_config: Dict[str, Any] = {"enabled": False}
    else:
        detailed = args.profiler == "detailed"
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=detailed,
            with_stack=detailed,
        )
        profiler_config = {
            "enabled": True,
            "level": args.profiler,
            "activities": ["CPU", "CUDA"],
            "record_shapes": True,
            "profile_memory": detailed,
            "with_stack": detailed,
            "profile_window": args.profile_window,
            "sessions": 1,
            "session_scope": "single contiguous start/stop session",
        }

    trace_path = args.trace_path
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)

    annotator = Annotator(
        nvtx=True,
        # Kineto only records ``record_function`` annotations, never external
        # NVTX ranges, so the phase boundaries need this second channel when
        # (and only when) a Chrome trace is actually being written.
        record_function=args.profiler != "off",
    )

    captured = run_instrumented_pass(
        model=model,
        inputs=inputs,
        output_tokens=args.osl,
        early_steps=args.early_steps,
        late_steps=args.late_steps,
        profiler=profiler,
        annotator=annotator,
        profile_window=args.profile_window if profiler is not None else "none",
        layer0_prefix=args.layer0_prefix if args.layer0_roles else None,
        cuda_profiler_api=args.cuda_profiler_api,
        trace_path=trace_path,
    )
    del profiler

    token_parity = None
    if reference is not None:
        token_parity = hash_token_sequence(
            reference["generated_token_ids"]
        ) == captured["sequence_sha256"]

    payload = {
        "tag": args.tag,
        "run_id": f"run_{started_at.strftime('%Y%m%d_%H%M%S_%f')}",
        "started_at": started_at.isoformat(),
        "ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _shell(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool((_shell(["git", "status", "--porcelain"]) or "").strip()),
        "identity": {
            "model_path": str(Path(args.model_path).expanduser().resolve()),
            "model_manifest_sha256": _file_hash(args.manifest),
            "environment": _environment(),
            "load_time_s": load_time_s,
            "declared": {
                "dtype": "float16",
                "attention_backend": "eager",
                "cache_impl": "transformers DynamicCache (use_cache=True)",
                "batch_size": 1,
                "sampling": "greedy",
            },
        },
        "workload": {"isl": args.isl, "osl": args.osl},
        "instrumentation": {
            "profiler": profiler_config,
            "cuda_profiler_api": bool(args.cuda_profiler_api),
            "layer0_roles": bool(args.layer0_roles),
            "layer0_prefix": args.layer0_prefix if args.layer0_roles else None,
            "threads": torch.get_num_threads(),
        },
        "reference": (
            {
                "generated_token_ids": reference["generated_token_ids"],
                "sequence_sha256": hash_token_sequence(
                    reference["generated_token_ids"]
                ),
                "prefill_forward_ms": reference["prefill_forward_ms"],
                "first_token_selection_ms": reference["first_token_selection_ms"],
                "model_core_ttft_ms": reference["model_core_ttft_ms"],
                "decode_total_ms": reference["decode_total_ms"],
                "model_core_e2e_ms": reference["model_core_e2e_ms"],
                "raw_itl_ms": reference["raw_itl_ms"],
                "device_used_bytes": int(
                    torch.cuda.memory_allocated() if device.type == "cuda" else 0
                ),
            }
            if reference is not None
            else None
        ),
        "captured": captured,
        "token_parity_vs_reference": token_parity,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "sequence_sha256": captured["sequence_sha256"],
                "token_parity_vs_reference": token_parity,
                "profiled_windows": captured["profiled_windows"],
                "trace": captured["trace"],
                "phase_wall_ms": captured["phase_wall_ms"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
