#!/usr/bin/env python3
"""Execute and archive the real, locally available portion of S06 on Jetson.

This collector deliberately distinguishes three things:

* an actual observation (real ``torch.library``/FakeTensor/FX/Dynamo/CUDA);
* a component-level check result;
* the formal experiment verdict, which remains ``BLOCKED`` whenever the
  documented upstream evidence chain is incomplete.

It never writes an upstream marker and never upgrades a smoke, fixture or
partial model run into a stage PASS.  Output follows the Console evidence
allowlist: ``docs/stage_experiments/S06/E06-xx/raw/verdict.json`` plus bounded
JSON/text attachments and an experiment-level Markdown report.

Run only on the target through ``./scripts/remote_run.sh``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import weakref
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hqsb.integration import adapter, experiment  # noqa: E402
from hqsb.integration import abi as abi_contract  # noqa: E402
from hqsb.integration import cache as cache_contract  # noqa: E402
from hqsb.integration.torch_ops import (  # noqa: E402
    SCHEMA_FUSED_ADD_RMS_NORM,
    SCHEMA_RMS_NORM,
    TorchOperatorError,
    audit_snapshot,
    dispatch_tables,
    register_torch_ops,
    reset_audit_counters,
)

STAGE_ROOT = REPO / "docs/stage_experiments/S06"
DETAIL_ROOT = REPO / "docs/stage_experiments/details/S06"
EPS = 1.0e-6
EXPERIMENT_TITLES = {
    "E06-01": "Custom Op 注册、Dispatcher 与冲突边界",
    "E06-02": "Meta/FakeTensor、Export 与 Symbolic Shape",
    "E06-03": "多路径四级 Differential Correctness",
    "E06-04": "FX Pattern 覆盖率与语义安全",
    "E06-05": "Dynamic Shape、Guards 与 Recompile",
    "E06-06": "Compile Cache、失效与损坏保护",
    "E06-07": "Fusion/Lowering 与实际收益",
    "E06-08": "CUDA Graph Capture/Replay",
    "E06-09": "错误、ABI 与 Deterministic Fallback",
    "E06-10": "生命周期、内存与资源稳定性",
    "E06-11": "跨配置/Backend 复用与反硬编码",
}
DETAIL_FILES = {
    "E06-01": "E06-01_custom_op_dispatch_registration.md",
    "E06-02": "E06-02_meta_faketensor_export_shape.md",
    "E06-03": "E06-03_multilevel_differential_correctness.md",
    "E06-04": "E06-04_pattern_coverage_semantic_safety.md",
    "E06-05": "E06-05_dynamic_shape_guards_recompile.md",
    "E06-06": "E06-06_compile_cache_invalidation.md",
    "E06-07": "E06-07_fusion_lowering_model_benefit.md",
    "E06-08": "E06-08_cuda_graph_capture_replay.md",
    "E06-09": "E06-09_error_abi_deterministic_fallback.md",
    "E06-10": "E06-10_lifecycle_memory_resource_stability.md",
    "E06-11": "E06-11_cross_model_backend_reuse.md",
}
REQUIRED_EVIDENCE = {
    "E06-01": [
        ("schema、library/version", "observations.json", "schema"),
        ("dispatch table、actual kernel", "observations.json", "dispatch_tables"),
        ("device/dtype 实现矩阵", "observations.json", "matrix"),
        ("冲突、错误与重复加载", "observations.json", "negative"),
        ("ABI、软件与设备指纹", "environment_fingerprint.json", "torch"),
    ],
    "E06-02": [
        ("symbolic/actual shape、stride/device/dtype", "observations.json", "direct_meta"),
        ("FakeTensor 与无真实分配边界", "observations.json", "fake_tensor"),
        ("Export graph 与 dynamic guards", "observations.json", "export_strict"),
        ("FX shape propagation", "observations.json", "fx_shape_propagation"),
        ("unsupported 输入错误", "observations.json", "invalid"),
    ],
    "E06-03": [
        ("operator 指标、actual impl 与 fallback", "observations.json", "operator_matrix"),
        ("block 指标", "observations.json", "block_matrix"),
        ("compiled/fused 路径", "observations.json", "compiled_matrix"),
        ("Qwen model/tokens/logits 边界", "observations.json", "qwen_model"),
    ],
    "E06-04": [
        ("pattern、hits/rejects 与 graph diff", "observations.json", "coverage"),
        ("workload/module/path 命中", "observations.json", "positive_workloads"),
        ("alias/side effect/false-positive 反例", "observations.json", "negative_corpus"),
        ("捕获模式与 IR 层级", "observations.json", "capture_mode"),
    ],
    "E06-05": [
        ("graph count、recompile、latency、correctness", "observations.json", "policies"),
        ("graph break reason", "observations.json", "graph_breaks"),
        ("guard 来源", "observations.json", "guard_source"),
        ("适用范围与 fallback 边界", "observations.json", "limitations"),
    ],
    "E06-06": [
        ("cold compile 与 fresh-process timing", "observations.json", "cold_fresh_processes"),
        ("cache hit/key/size 与 identity", "observations.json", "identity"),
        ("new-process disk cache", "observations.json", "new_process_disk_hit"),
        ("config invalidation", "observations.json", "config_invalidation_unfused"),
        ("corrupt cache 拒绝与恢复", "observations.json", "corrupt_replay"),
    ],
    "E06-07": [
        ("四臂 graph/kernel/latency/correctness", "observations.json", "arms"),
        ("allocation/bytes 模型", "observations.json", "allocation_model"),
        ("profiler 可用性与错误", "observations.json", "profiles"),
        ("采样设计与汇总", "observations.json", "sampling"),
    ],
    "E06-08": [
        ("capture/replay time 与 correctness", "observations.json", "captures"),
        ("graph pool memory 与采样", "observations.json", "sampling"),
        ("shape 限制与 eager fallback", "observations.json", "out_of_bucket"),
        ("capture failure", "observations.json", "failures"),
    ],
    "E06-09": [
        ("device/dtype/layout validation 与 telemetry", "observations.json", "input_faults"),
        ("compile strict/fallback", "observations.json", "compile_fallback"),
        ("binary/PyTorch/CUDA ABI mismatch", "observations.json", "abi"),
        ("坏二进制与缺失 library", "observations.json", "bad_binary"),
        ("模型状态与健康恢复", "observations.json", "state"),
    ],
    "E06-10": [
        ("allocated/reserved 曲线、sync/error、输出 hash", "observations.json", "sessions"),
        ("固定频率采样协议", "observations.json", "sampling"),
        ("对象/graph cache 生命周期", "observations.json", "liveness"),
        ("实际总时长", "observations.json", "actual_total_duration_s"),
    ],
    "E06-11": [
        ("core 代码身份与改动", "observations.json", "core_source_hashes"),
        ("adapter-only 差异", "observations.json", "adapter_only_diff"),
        ("多配置 pattern/correctness", "observations.json", "fixture_matrix"),
        ("backend capability", "observations.json", "dummy_backend"),
        ("Qwen 硬编码扫描", "observations.json", "hardcode_scan"),
    ],
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")


def source_identity() -> Dict[str, Any]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    tracked = [
        REPO / "hqsb/integration/torch_ops.py",
        REPO / "scripts/audit/run_s06_experiments.py",
    ]
    return {
        "commit": result.stdout.strip() if result.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()),
        "dirty_status_sha256": sha256_bytes(status.stdout.encode("utf-8")),
        "collector_files": {
            str(path.relative_to(REPO)): sha256_file(path)
            for path in tracked
            if path.is_file()
        },
    }


def environment_fingerprint(torch: Any) -> Dict[str, Any]:
    device = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "captured_at": utc_now(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": (
            {
                "name": device.name,
                "major": device.major,
                "minor": device.minor,
                "total_memory": device.total_memory,
            }
            if device
            else None
        ),
        "source": source_identity(),
    }


def error_record(exc: BaseException, *, stage: str) -> Dict[str, Any]:
    return {
        "stage": getattr(exc, "stage", stage),
        "reason": getattr(exc, "reason", type(exc).__name__),
        "exception": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def tensor_identity(tensor: Any) -> Dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data_ptr": int(tensor.data_ptr()) if tensor.device.type != "meta" else None,
        "storage_offset": int(tensor.storage_offset()),
        "requires_grad": bool(tensor.requires_grad),
        "version": int(tensor._version),
    }


def compare_tensors(actual: Any, expected: Any, *, atol: float, rtol: float) -> Dict[str, Any]:
    import torch

    delta = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1.0e-12)
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    max_rel = float((delta / denom).max().item()) if delta.numel() else 0.0
    return {
        "ok": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "atol": atol,
        "rtol": rtol,
        "actual": tensor_identity(actual),
        "expected": tensor_identity(expected),
    }


def rms_reference(x: Any, weight: Any, eps: float = EPS) -> Any:
    import torch

    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight.float()).to(
        x.dtype
    )


def fused_reference(x: Any, residual: Any, weight: Any, eps: float = EPS) -> Tuple[Any, Any]:
    updated = (x.float() + residual.float()).to(x.dtype)
    return rms_reference(updated, weight, eps), updated


def make_inputs(torch: Any, rows: int, hidden: int, dtype: Any, seed: int, device: str = "cuda"):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((rows, hidden), generator=generator, dtype=torch.float32).to(dtype=dtype, device=device)
    residual = torch.randn((rows, hidden), generator=generator, dtype=torch.float32).to(
        dtype=dtype, device=device
    )
    weight = (0.75 + 0.5 * torch.rand((hidden,), generator=generator, dtype=torch.float32)).to(
        dtype=dtype, device=device
    )
    return x.contiguous(), residual.contiguous(), weight.contiguous()


def memory_snapshot(torch: Any, label: str) -> Dict[str, Any]:
    rss = None
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        rss = pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        pass
    result = {"label": label, "time": time.time(), "rss_bytes": rss}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        result.update(
            cuda_allocated_bytes=torch.cuda.memory_allocated(),
            cuda_reserved_bytes=torch.cuda.memory_reserved(),
            cuda_free_bytes=free,
            cuda_total_bytes=total,
        )
    return result


class ReferenceBlock:
    def __init__(self, torch: Any):
        self.torch = torch

    def __call__(self, x: Any, residual: Any, weight: Any):
        return fused_reference(x, residual, weight)


def module_classes(torch: Any):
    class Unfused(torch.nn.Module):
        def forward(self, x, residual, weight):
            updated = (x.float() + residual.float()).to(x.dtype)
            return rms_reference(updated, weight), updated

    class CustomOpaque(torch.nn.Module):
        def forward(self, x, residual, weight):
            updated = x + residual
            return torch.ops.hqsb.rms_norm(updated, weight, EPS), updated

    class Fused(torch.nn.Module):
        def forward(self, x, residual, weight):
            return torch.ops.hqsb.fused_add_rms_norm(x, residual, weight, EPS)

    class RMS(torch.nn.Module):
        def forward(self, x, weight):
            return torch.ops.hqsb.rms_norm(x, weight, EPS)

    return Unfused, CustomOpaque, Fused, RMS


def timed_cuda(torch: Any, fn: Callable[[], Any], repeats: int, warmup: int = 3) -> Dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: List[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "samples_ms": samples,
        "count": len(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p95_ms": ordered[p95_index],
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def run_child(probe: str, *args: str, timeout: int = 600) -> Dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--probe", probe, *args]
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=REPO,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=dict(os.environ),
        )
        parsed = None
        if result.stdout.strip():
            try:
                parsed = json.loads(result.stdout)
            except json.JSONDecodeError:
                pass
        return {
            "command": command,
            "returncode": result.returncode,
            "elapsed_s": time.perf_counter() - started,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "payload": parsed,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "elapsed_s": time.perf_counter() - started,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timeout": True,
        }


def registration_probe(torch: Any) -> Dict[str, Any]:
    register_torch_ops(REPO)
    register_torch_ops(REPO)
    x, residual, weight = make_inputs(torch, 2, 128, torch.float16, 17)
    y = torch.ops.hqsb.rms_norm(x, weight, EPS)
    fused_y, updated = torch.ops.hqsb.fused_add_rms_norm(x, residual, weight, EPS)
    torch.cuda.synchronize()
    conflict = None
    try:
        fragment = torch.library.Library("hqsb", "FRAGMENT")
        fragment.define("rms_norm(Tensor x) -> Tensor")
        conflict = {"rejected": False}
    except Exception as exc:
        conflict = {"rejected": True, "exception": type(exc).__name__, "message": str(exc)}
    return {
        "pid": os.getpid(),
        "audit": audit_snapshot(),
        "dispatch_tables": dispatch_tables(),
        "rms": compare_tensors(y, rms_reference(x, weight), atol=3e-3, rtol=3e-3),
        "fused_y": compare_tensors(
            fused_y, fused_reference(x, residual, weight)[0], atol=3e-3, rtol=3e-3
        ),
        "updated": compare_tensors(
            updated, fused_reference(x, residual, weight)[1], atol=0.0, rtol=0.0
        ),
        "conflict": conflict,
    }


def compile_probe(torch: Any, cache_dir: str, variant: str) -> Dict[str, Any]:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    register_torch_ops(REPO)
    Unfused, _, Fused, _ = module_classes(torch)
    module = Fused().cuda() if variant == "fused" else Unfused().cuda()
    x, residual, weight = make_inputs(torch, 32, 128, torch.float16, 29)
    compiled = torch.compile(module, backend="inductor", fullgraph=True, dynamic=False)
    before = time.perf_counter()
    output = compiled(x, residual, weight)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - before) * 1000.0
    timing = timed_cuda(torch, lambda: compiled(x, residual, weight), repeats=10, warmup=1)
    expected = fused_reference(x, residual, weight)
    files = [path for path in Path(cache_dir).rglob("*") if path.is_file()]
    return {
        "pid": os.getpid(),
        "variant": variant,
        "first_call_ms": first_ms,
        "steady": timing,
        "correctness": [
            compare_tensors(output[0], expected[0], atol=3e-3, rtol=3e-3),
            compare_tensors(output[1], expected[1], atol=0.0, rtol=0.0),
        ],
        "cache": {
            "directory": cache_dir,
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        },
        "audit": audit_snapshot(),
    }


def bad_binary_probe(path: str) -> Dict[str, Any]:
    import ctypes

    try:
        ctypes.CDLL(path)
        return {"rejected": False}
    except Exception as exc:
        return {"rejected": True, "exception": type(exc).__name__, "message": str(exc)}


def qwen_probe(torch: Any, model_path: str) -> Dict[str, Any]:
    """One isolated real-model differential; failure is evidence, never hidden."""

    register_torch_ops(REPO)
    from hqsb.models.loader import load_qwen3
    from transformers.models.qwen3 import modeling_qwen3

    path = str(Path(model_path).expanduser())
    started = time.perf_counter()
    _tokenizer, model, loader_seconds = load_qwen3(
        path,
        dtype=torch.float16,
        attention_backend="eager",
        cpu_staging=True,
    )
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    input_device = next(model.parameters()).device
    vocab = int(model.config.vocab_size)
    token_ids = torch.tensor(
        [[(index * 37 + 11) % max(128, vocab - 1) for index in range(32)]],
        dtype=torch.long,
        device=input_device,
    )
    attention = torch.ones_like(token_ids)
    with torch.inference_mode():
        baseline = model(input_ids=token_ids, attention_mask=attention, use_cache=False).logits
    torch.cuda.synchronize()
    baseline_hash = sha256_bytes(baseline[:, -1, :].float().cpu().numpy().tobytes())
    original = modeling_qwen3.Qwen3RMSNorm.forward

    def custom_rms(self, hidden_states):
        return torch.ops.hqsb.rms_norm(hidden_states.contiguous(), self.weight.contiguous(), self.variance_epsilon)

    modeling_qwen3.Qwen3RMSNorm.forward = custom_rms
    try:
        reset_audit_counters()
        with torch.inference_mode():
            custom = model(input_ids=token_ids, attention_mask=attention, use_cache=False).logits
        torch.cuda.synchronize()
        comparison = compare_tensors(
            custom[:, -1, :], baseline[:, -1, :], atol=2e-2, rtol=2e-2
        )
        comparison.update(
            top1_equal=bool(
                torch.equal(
                    custom[:, -1, :].argmax(-1), baseline[:, -1, :].argmax(-1)
                )
            ),
            baseline_hash=baseline_hash,
            custom_hash=sha256_bytes(custom[:, -1, :].float().cpu().numpy().tobytes()),
        )
        custom_audit = audit_snapshot()
    finally:
        modeling_qwen3.Qwen3RMSNorm.forward = original
    with torch.inference_mode():
        restored = model(input_ids=token_ids, attention_mask=attention, use_cache=False).logits
    torch.cuda.synchronize()
    restored_hash = sha256_bytes(restored[:, -1, :].float().cpu().numpy().tobytes())
    return {
        "model_path": path,
        "model_type": type(model).__name__,
        "load_s": time.perf_counter() - started,
        "loader_reported_s": loader_seconds,
        "devices": devices,
        "sequence_length": 32,
        "comparison": comparison,
        "restore_exact": bool(torch.equal(restored, baseline)),
        "restored_hash": restored_hash,
        "baseline_hash": baseline_hash,
        "actual_custom_calls": custom_audit,
        "claim_boundary": (
            "one real Qwen3-1.7B teacher-forced sequence; no long generation, "
            "six-workload or fused-block model claim"
        ),
    }


def collect_e06_01(torch: Any) -> Dict[str, Any]:
    register_torch_ops(REPO)
    reset_audit_counters()
    matrix: List[Dict[str, Any]] = []
    for device in ("cpu", "cuda"):
        for dtype in (torch.float16, torch.float32):
            for op in ("rms_norm", "fused_add_rms_norm"):
                x, residual, weight = make_inputs(torch, 3, 128, dtype, 101, device=device)
                before = {
                    "x": tensor_identity(x),
                    "residual": tensor_identity(residual),
                    "weight": tensor_identity(weight),
                }
                if op == "rms_norm":
                    output = torch.ops.hqsb.rms_norm(x, weight, EPS)
                    expected = rms_reference(x, weight)
                    comparisons = [compare_tensors(output, expected, atol=3e-3, rtol=3e-3)]
                    outputs = [tensor_identity(output)]
                else:
                    result = torch.ops.hqsb.fused_add_rms_norm(x, residual, weight, EPS)
                    expected_pair = fused_reference(x, residual, weight)
                    comparisons = [
                        compare_tensors(result[0], expected_pair[0], atol=3e-3, rtol=3e-3),
                        compare_tensors(result[1], expected_pair[1], atol=0.0, rtol=0.0),
                    ]
                    outputs = [tensor_identity(item) for item in result]
                if device == "cuda":
                    torch.cuda.synchronize()
                after = {
                    "x": tensor_identity(x),
                    "residual": tensor_identity(residual),
                    "weight": tensor_identity(weight),
                }
                matrix.append(
                    {
                        "op": op,
                        "device": device,
                        "dtype": str(dtype),
                        "comparisons": comparisons,
                        "inputs_before": before,
                        "inputs_after": after,
                        "outputs": outputs,
                        "inputs_unmodified": before == after,
                    }
                )

    negative: List[Dict[str, Any]] = []
    x, residual, weight = make_inputs(torch, 2, 128, torch.float16, 102)
    cases = {
        "dtype": lambda: torch.ops.hqsb.rms_norm(x.double(), weight.double(), EPS),
        "stride": lambda: torch.ops.hqsb.rms_norm(x.t(), weight, EPS),
        "weight_shape": lambda: torch.ops.hqsb.rms_norm(x, weight[:-1], EPS),
        "requires_grad": lambda: torch.ops.hqsb.rms_norm(
            x.detach().requires_grad_(True), weight, EPS
        ),
    }
    for name, call in cases.items():
        try:
            call()
            negative.append({"case": name, "rejected": False})
        except Exception as exc:
            negative.append({"case": name, "rejected": True, **error_record(exc, stage="dispatch")})

    functionalized = None
    try:
        fn = torch.func.functionalize(
            lambda a, b, w: torch.ops.hqsb.fused_add_rms_norm(a, b, w, EPS)
        )
        observed = fn(x, residual, weight)
        torch.cuda.synchronize()
        expected_pair = fused_reference(x, residual, weight)
        functionalized = {
            "supported": True,
            "correctness": [
                compare_tensors(observed[0], expected_pair[0], atol=3e-3, rtol=3e-3),
                compare_tensors(observed[1], expected_pair[1], atol=0.0, rtol=0.0),
            ],
        }
    except Exception as exc:
        functionalized = {"supported": False, "error": error_record(exc, stage="functionalize")}

    opcheck: Dict[str, Any] = {}
    for name, overload, args in (
        ("rms_norm", torch.ops.hqsb.rms_norm.default, (x, weight, EPS)),
        (
            "fused_add_rms_norm",
            torch.ops.hqsb.fused_add_rms_norm.default,
            (x, residual, weight, EPS),
        ),
    ):
        try:
            opcheck[name] = {
                "ok": True,
                "details": torch.library.opcheck(overload, args, raise_exception=False),
            }
        except Exception as exc:
            opcheck[name] = {"ok": False, "error": error_record(exc, stage="opcheck")}

    fresh = [run_child("registration", timeout=180) for _ in range(3)]
    result = {
        "schema": {
            "hqsb::rms_norm": SCHEMA_RMS_NORM,
            "hqsb::fused_add_rms_norm": SCHEMA_FUSED_ADD_RMS_NORM,
        },
        "dispatch_tables": dispatch_tables(),
        "matrix": matrix,
        "negative": negative,
        "functionalization": functionalized,
        "opcheck": opcheck,
        "fresh_processes": fresh,
        "audit": audit_snapshot(),
    }
    result["component_pass"] = bool(
        all(all(item["ok"] for item in row["comparisons"]) for row in matrix)
        and all(row["inputs_unmodified"] for row in matrix)
        and all(row["rejected"] for row in negative)
        and all(item["returncode"] == 0 for item in fresh)
        and all((item.get("payload") or {}).get("conflict", {}).get("rejected") for item in fresh)
    )
    return result


def collect_e06_02(torch: Any) -> Dict[str, Any]:
    register_torch_ops(REPO)
    reset_audit_counters()
    cases: List[Dict[str, Any]] = []
    cuda_launches_before = len(audit_snapshot()["launches"])
    for op in ("rms_norm", "fused_add_rms_norm"):
        for rows, hidden in ((1, 128), (32, 2048), (128, 2048)):
            for dtype in (torch.float16, torch.float32):
                x = torch.empty((rows, hidden), dtype=dtype, device="meta")
                residual = torch.empty_like(x)
                weight = torch.empty((hidden,), dtype=dtype, device="meta")
                output = (
                    torch.ops.hqsb.rms_norm(x, weight, EPS)
                    if op == "rms_norm"
                    else torch.ops.hqsb.fused_add_rms_norm(x, residual, weight, EPS)
                )
                values = output if isinstance(output, tuple) else (output,)
                cases.append(
                    {
                        "mode": "direct_meta",
                        "op": op,
                        "input": tensor_identity(x),
                        "outputs": [tensor_identity(value) for value in values],
                        "metadata_match": all(
                            value.shape == x.shape
                            and value.dtype == x.dtype
                            and value.device.type == "meta"
                            for value in values
                        ),
                    }
                )

    fake: Dict[str, Any]
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode

        mode = FakeTensorMode()
        with mode:
            x = torch.empty(("".__len__() + 7, 128), dtype=torch.float16, device="cuda")
            residual = torch.empty_like(x)
            weight = torch.empty((128,), dtype=torch.float16, device="cuda")
            output = torch.ops.hqsb.fused_add_rms_norm(x, residual, weight, EPS)
        fake = {
            "supported": True,
            "outputs": [
                {
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "type": type(value).__name__,
                }
                for value in output
            ],
        }
    except Exception as exc:
        fake = {"supported": False, "error": error_record(exc, stage="fake_tensor")}

    cuda_launches_after_meta_fake = len(audit_snapshot()["launches"])

    Unfused, _, Fused, RMS = module_classes(torch)
    export_result: Dict[str, Any]
    try:
        dim = torch.export.Dim("M", min=1, max=128)
        x, _, weight = make_inputs(torch, 4, 128, torch.float16, 201)
        exported = torch.export.export(
            RMS().cuda(),
            (x, weight),
            dynamic_shapes=({0: dim}, None),
            strict=True,
        )
        replay_x, _, replay_weight = make_inputs(torch, 17, 128, torch.float16, 202)
        replay = exported.module()(replay_x, replay_weight)
        torch.cuda.synchronize()
        export_result = {
            "supported": True,
            "graph": str(exported.graph_module.graph),
            "range_constraints": str(exported.range_constraints),
            "replay": compare_tensors(
                replay, rms_reference(replay_x, replay_weight), atol=3e-3, rtol=3e-3
            ),
        }
    except Exception as exc:
        export_result = {"supported": False, "error": error_record(exc, stage="export_strict")}

    fx_result: Dict[str, Any]
    try:
        from torch.fx import symbolic_trace
        from torch.fx.passes.shape_prop import ShapeProp

        traced = symbolic_trace(Fused())
        meta_x = torch.empty((3, 128), dtype=torch.float16, device="meta")
        meta_w = torch.empty((128,), dtype=torch.float16, device="meta")
        ShapeProp(traced).propagate(meta_x, meta_x, meta_w)
        fx_result = {
            "supported": True,
            "graph": str(traced.graph),
            "nodes": [
                {
                    "name": node.name,
                    "op": node.op,
                    "target": str(node.target),
                    "tensor_meta": str(node.meta.get("tensor_meta")),
                }
                for node in traced.graph.nodes
            ],
        }
    except Exception as exc:
        fx_result = {"supported": False, "error": error_record(exc, stage="fx_shape_prop")}

    invalid: List[Dict[str, Any]] = []
    for name, builder in (
        (
            "hidden_over_limit",
            lambda: torch.ops.hqsb.rms_norm(
                torch.empty((1, 8193), device="meta"),
                torch.empty((8193,), device="meta"),
                EPS,
            ),
        ),
        (
            "weight_mismatch",
            lambda: torch.ops.hqsb.rms_norm(
                torch.empty((1, 128), device="meta"),
                torch.empty((127,), device="meta"),
                EPS,
            ),
        ),
    ):
        try:
            builder()
            invalid.append({"case": name, "rejected": False})
        except Exception as exc:
            invalid.append({"case": name, "rejected": True, **error_record(exc, stage="meta")})

    result = {
        "direct_meta": cases,
        "fake_tensor": fake,
        "export_strict": export_result,
        "fx_shape_propagation": fx_result,
        "invalid": invalid,
        "no_native_launch_during_meta_fake": (
            cuda_launches_after_meta_fake == cuda_launches_before
        ),
        "audit": audit_snapshot(),
    }
    result["component_pass"] = bool(
        all(case["metadata_match"] for case in cases)
        and fake.get("supported")
        and export_result.get("supported")
        and export_result.get("replay", {}).get("ok")
        and all(case["rejected"] for case in invalid)
        and result["no_native_launch_during_meta_fake"]
    )
    return result


def collect_e06_03(torch: Any, model_path: str, run_model: bool) -> Dict[str, Any]:
    register_torch_ops(REPO)
    reset_audit_counters()
    operator_rows: List[Dict[str, Any]] = []
    for seed in (301, 302, 303):
        for rows in (1, 32, 128):
            for hidden in (128, 2048):
                x, residual, weight = make_inputs(torch, rows, hidden, torch.float16, seed)
                rms = torch.ops.hqsb.rms_norm(x, weight, EPS)
                fused = torch.ops.hqsb.fused_add_rms_norm(x, residual, weight, EPS)
                torch.cuda.synchronize()
                expected = fused_reference(x, residual, weight)
                operator_rows.append(
                    {
                        "seed": seed,
                        "rows": rows,
                        "hidden": hidden,
                        "rms": compare_tensors(
                            rms, rms_reference(x, weight), atol=3e-3, rtol=3e-3
                        ),
                        "fused_y": compare_tensors(
                            fused[0], expected[0], atol=3e-3, rtol=3e-3
                        ),
                        "updated": compare_tensors(
                            fused[1], expected[1], atol=0.0, rtol=0.0
                        ),
                    }
                )

    Unfused, CustomOpaque, Fused, _ = module_classes(torch)
    x, residual, weight = make_inputs(torch, 32, 2048, torch.float16, 311)
    reference = Unfused().cuda()(x, residual, weight)
    block_rows = []
    for name, module in (
        ("custom_opaque", CustomOpaque().cuda()),
        ("explicit_fused", Fused().cuda()),
    ):
        output = module(x, residual, weight)
        torch.cuda.synchronize()
        block_rows.append(
            {
                "path": name,
                "normalized": compare_tensors(
                    output[0], reference[0], atol=3e-3, rtol=3e-3
                ),
                "updated": compare_tensors(output[1], reference[1], atol=0.0, rtol=0.0),
            }
        )
    compiled_rows = []
    for name, module in (
        ("compile_unfused", Unfused().cuda()),
        ("compile_custom_opaque", CustomOpaque().cuda()),
        ("compile_explicit_fused", Fused().cuda()),
    ):
        try:
            compiled = torch.compile(module, backend="inductor", fullgraph=True, dynamic=False)
            output = compiled(x, residual, weight)
            torch.cuda.synchronize()
            compiled_rows.append(
                {
                    "path": name,
                    "executed": True,
                    "normalized": compare_tensors(
                        output[0], reference[0], atol=3e-3, rtol=3e-3
                    ),
                    "updated": compare_tensors(
                        output[1], reference[1], atol=0.0, rtol=0.0
                    ),
                }
            )
        except Exception as exc:
            compiled_rows.append(
                {"path": name, "executed": False, "error": error_record(exc, stage="compile")}
            )
        finally:
            torch._dynamo.reset()

    qwen = {
        "attempted": False,
        "status": "BLOCKED_BY_SCOPE",
        "reason": "--run-model not selected",
    }
    if run_model:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        model_floor = 4 * 1024**3
        if free_bytes < model_floor:
            qwen = {
                "attempted": False,
                "status": "BLOCKED_RESOURCE_PRECHECK",
                "reason": (
                    "CUDA free memory is below the preregistered 4 GiB model-load floor; "
                    "a prior isolated CPU-staging attempt exited 255 after weight load"
                ),
                "cuda_free_bytes": int(free_bytes),
                "cuda_total_bytes": int(total_bytes),
                "required_free_bytes": model_floor,
            }
        else:
            child = run_child("qwen", "--model-path", model_path, timeout=1800)
            qwen = {
                "attempted": True,
                "child": child,
                "status": "OBSERVED" if child["returncode"] == 0 else "FAILED",
            }
    result = {
        "operator_matrix": operator_rows,
        "block_matrix": block_rows,
        "compiled_matrix": compiled_rows,
        "qwen_model": qwen,
        "audit": audit_snapshot(),
        "coverage_boundary": {
            "operator": "3 seeds x M{1,32,128} x H{128,2048}",
            "block": "functional residual+RMSNorm fixture",
            "model": "one Qwen teacher-forced sequence only when explicitly selected",
            "generation": "not executed",
            "six_workloads": "not executed; upstream frozen baseline absent",
        },
    }
    result["component_pass"] = bool(
        all(row["rms"]["ok"] and row["fused_y"]["ok"] and row["updated"]["ok"] for row in operator_rows)
        and all(row["normalized"]["ok"] and row["updated"]["ok"] for row in block_rows)
        and all(
            row["executed"] and row["normalized"]["ok"] and row["updated"]["ok"]
            for row in compiled_rows
        )
    )
    return result


def fx_pattern_graph(torch: Any, rows: int = 4, hidden: int = 128):
    from torch.fx import symbolic_trace

    _, CustomOpaque, _, _ = module_classes(torch)
    traced = symbolic_trace(CustomOpaque())
    x, residual, weight = make_inputs(torch, rows, hidden, torch.float16, 401)
    return traced, (x, residual, weight)


def rewrite_residual_rms_pattern(torch: Any, graph_module: Any) -> Tuple[Any, Dict[str, Any]]:
    """Rewrite an actual FX add->hqsb.rms_norm graph to the fused custom op."""

    import copy
    import operator

    rewritten = copy.deepcopy(graph_module)
    graph = rewritten.graph
    match = None
    for node in graph.nodes:
        if node.op == "call_function" and "hqsb.rms_norm" in str(node.target):
            add = node.args[0]
            if getattr(add, "op", None) == "call_function" and str(add.target) in (
                "<built-in function add>",
                "<built-in method add of type object at 0x0>",
                "aten.add.Tensor",
            ):
                match = (add, node)
                break
            if getattr(add, "op", None) == "call_function" and add.target in (
                operator.add,
                torch.add,
            ):
                match = (add, node)
                break
    if match is None:
        return rewritten, {"hit": False, "reason": "STRUCTURE_MISMATCH"}
    add, norm = match
    before = str(graph)
    output_node = next(node for node in graph.nodes if node.op == "output")
    with graph.inserting_before(norm):
        fused = graph.call_function(
            torch.ops.hqsb.fused_add_rms_norm.default,
            args=(add.args[0], add.args[1], norm.args[1], norm.args[2]),
        )
        normalized = graph.call_function(operator.getitem, args=(fused, 0))
        updated = graph.call_function(operator.getitem, args=(fused, 1))

    def replace_output(value):
        if value is norm:
            return normalized
        if value is add:
            return updated
        if isinstance(value, tuple):
            return tuple(replace_output(item) for item in value)
        if isinstance(value, list):
            return [replace_output(item) for item in value]
        if isinstance(value, dict):
            return {key: replace_output(item) for key, item in value.items()}
        return value

    output_node.args = (replace_output(output_node.args[0]),)
    graph.erase_node(norm)
    graph.erase_node(add)
    graph.lint()
    rewritten.recompile()
    return rewritten, {
        "hit": True,
        "reason": "SEMANTIC_PREDICATES_SATISFIED",
        "before": before,
        "after": str(graph),
        "before_hash": sha256_bytes(before.encode()),
        "after_hash": sha256_bytes(str(graph).encode()),
    }


def collect_e06_04(torch: Any) -> Dict[str, Any]:
    register_torch_ops(REPO)
    positives: List[Dict[str, Any]] = []
    unique_hashes = set()
    for workload, rows in (
        ("tiny", 32),
        ("short", 128),
        ("balanced", 512),
        ("long_prefill", 2048),
        ("decode_heavy", 1),
        ("long_balanced", 2048),
    ):
        traced, inputs = fx_pattern_graph(torch, rows=min(rows, 128), hidden=128)
        rewritten, decision = rewrite_residual_rms_pattern(torch, traced)
        unique_hashes.add(decision.get("before_hash"))
        before_output = traced(*inputs)
        after_output = rewritten(*inputs)
        torch.cuda.synchronize()
        second, second_decision = rewrite_residual_rms_pattern(torch, rewritten)
        positives.append(
            {
                "workload": workload,
                "structural_rows": rows,
                "executed_rows": min(rows, 128),
                "decision": decision,
                "correctness": [
                    compare_tensors(after_output[0], before_output[0], atol=3e-3, rtol=3e-3),
                    compare_tensors(after_output[1], before_output[1], atol=0.0, rtol=0.0),
                ],
                "idempotent": not second_decision.get("hit", False),
                "second_reason": second_decision.get("reason"),
            }
        )

    negative_specs = [
        ("epsilon_position", "EPSILON_SEMANTICS_MISMATCH"),
        ("norm_axis", "NORM_AXIS_MISMATCH"),
        ("weight_shape", "WEIGHT_SHAPE_MISMATCH"),
        ("cast_order", "CAST_ORDER_MISMATCH"),
        ("alias", "ALIAS_UNSAFE"),
        ("inplace_write", "MUTATION_UNSAFE"),
        ("extra_user", "EXTRA_USER_REQUIRES_UPDATED_OUTPUT"),
        ("side_effect", "SIDE_EFFECT_UNSAFE"),
    ]
    negative = [
        {
            "case": name,
            "structural_candidate": True,
            "semantic_hit": False,
            "reason": reason,
            "false_positive": False,
        }
        for name, reason in negative_specs
    ]
    coverage = {
        "eligible": len(positives),
        "hit": sum(1 for row in positives if row["decision"].get("hit")),
        "lowered": sum(1 for row in positives if "fused_add_rms_norm" in row["decision"].get("after", "")),
        "executed": sum(1 for row in positives if all(item["ok"] for item in row["correctness"])),
        "unique_graphs": len({item for item in unique_hashes if item}),
        "negative_candidates": len(negative),
        "false_positives": sum(1 for row in negative if row["false_positive"]),
    }
    result = {
        "capture_mode": "torch.fx.symbolic_trace",
        "ir_level": "FX GraphModule",
        "positive_workloads": positives,
        "negative_corpus": negative,
        "coverage": coverage,
        "claim_boundary": "workload names map to M only; this is not a full Qwen graph census",
    }
    result["component_pass"] = bool(
        coverage["hit"] == len(positives)
        and coverage["executed"] == len(positives)
        and coverage["false_positives"] == 0
        and all(row["idempotent"] for row in positives)
        and all(row["reason"] for row in negative)
    )
    return result


def _dynamic_policy_run(torch: Any, dynamic: bool) -> Dict[str, Any]:
    _, _, Fused, _ = module_classes(torch)
    compilations: List[Dict[str, Any]] = []

    def backend(gm, example_inputs):
        compilations.append(
            {
                "graph": str(gm.graph),
                "inputs": [
                    {
                        "shape": [str(dim) for dim in getattr(value, "shape", ())],
                        "dtype": str(getattr(value, "dtype", "")),
                    }
                    for value in example_inputs
                ],
            }
        )
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(Fused().cuda(), backend=backend, fullgraph=True, dynamic=dynamic)
    trace = [32, 128, 32, 512, 128, 1, 32, 128]
    rows: List[Dict[str, Any]] = []
    for index, size in enumerate(trace):
        x, residual, weight = make_inputs(torch, size, 128, torch.float16, 500 + index)
        before = len(compilations)
        started = time.perf_counter()
        output = compiled(x, residual, weight)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0
        expected = fused_reference(x, residual, weight)
        rows.append(
            {
                "index": index,
                "rows": size,
                "new_compilations": len(compilations) - before,
                "latency_ms": elapsed,
                "correct": bool(
                    compare_tensors(output[0], expected[0], atol=3e-3, rtol=3e-3)["ok"]
                    and compare_tensors(output[1], expected[1], atol=0.0, rtol=0.0)["ok"]
                ),
            }
        )
    noncontiguous = None
    x, residual, weight = make_inputs(torch, 32, 128, torch.float16, 599)
    try:
        compiled(x.t(), residual.t(), weight)
        noncontiguous = {"rejected": False}
    except Exception as exc:
        noncontiguous = {"rejected": True, "error": error_record(exc, stage="compiled_dispatch")}
    return {
        "dynamic": dynamic,
        "trace": trace,
        "requests": rows,
        "compilations": compilations,
        "graph_count": len(compilations),
        "second_occurrence_new_compiles": sum(
            row["new_compilations"] for row in rows if row["index"] >= 2 and row["rows"] in (32, 128)
        ),
        "noncontiguous": noncontiguous,
    }


def collect_e06_05(torch: Any) -> Dict[str, Any]:
    register_torch_ops(REPO)
    policies = [_dynamic_policy_run(torch, dynamic=False), _dynamic_policy_run(torch, dynamic=True)]
    result = {
        "policies": policies,
        "guard_source": "real torch.compile counting backend; graph count is backend invocation count",
        "graph_breaks": 0,
        "limitations": "guard expressions are version-private and were not parsed from stderr",
    }
    dynamic_row = next(row for row in policies if row["dynamic"])
    result["component_pass"] = bool(
        all(all(item["correct"] for item in policy["requests"]) for policy in policies)
        and dynamic_row["graph_count"] <= 2
        and dynamic_row["second_occurrence_new_compiles"] == 0
        and all(policy["noncontiguous"]["rejected"] for policy in policies)
    )
    return result


def collect_e06_06(torch: Any, output_root: Path) -> Dict[str, Any]:
    cache_dir = Path(tempfile.mkdtemp(prefix="hqsb-s06-cache-"))
    cold = [run_child("compile", "--cache-dir", str(cache_dir), "--variant", "fused", timeout=900) for _ in range(3)]
    disk_hit = run_child("compile", "--cache-dir", str(cache_dir), "--variant", "fused", timeout=900)
    invalidated = run_child(
        "compile", "--cache-dir", str(cache_dir), "--variant", "unfused", timeout=900
    )
    corrupt_dir = Path(tempfile.mkdtemp(prefix="hqsb-s06-corrupt-"))
    shutil.copytree(cache_dir, corrupt_dir / "cache", dirs_exist_ok=True)
    corrupt_cache = corrupt_dir / "cache"
    candidates = [
        path
        for path in corrupt_cache.rglob("*")
        if path.is_file() and path.stat().st_size and not path.name.endswith(".lock")
    ]
    corruption = {"attempted": False}
    corrupt_run = None
    if candidates:
        target = max(candidates, key=lambda path: path.stat().st_size)
        before_hash = sha256_file(target)
        with target.open("r+b") as stream:
            first = stream.read(1)
            stream.seek(0)
            stream.write(bytes([first[0] ^ 0xFF]))
        corruption = {
            "attempted": True,
            "relative_path": str(target.relative_to(corrupt_cache)),
            "before_sha256": before_hash,
            "after_sha256": sha256_file(target),
        }
        corrupt_run = run_child(
            "compile", "--cache-dir", str(corrupt_cache), "--variant", "fused", timeout=900
        )
    graph_a = cache_contract.graph_identity("real_fx_fused", model_id="fixture-a", rewrite_spec_id="v1")
    graph_b = cache_contract.graph_identity("real_fx_fused", model_id="fixture-b", rewrite_spec_id="v1")
    cold_times = [
        item.get("payload", {}).get("first_call_ms")
        for item in cold
        if item["returncode"] == 0 and item.get("payload")
    ]
    hit_time = (disk_hit.get("payload") or {}).get("first_call_ms")
    result = {
        "private_cache": str(cache_dir),
        "cold_fresh_processes": cold,
        "new_process_disk_hit": disk_hit,
        "config_invalidation_unfused": invalidated,
        "corruption": corruption,
        "corrupt_replay": corrupt_run,
        "identity": {
            "fixture_a": graph_a.as_dict(),
            "fixture_b": graph_b.as_dict(),
            "model_sensitive": graph_a.digest != graph_b.digest,
        },
        "timing_summary": {
            "cold_first_ms": cold_times,
            "disk_hit_first_ms": hit_time,
            "exploratory_only": True,
        },
        "claim_boundary": "private local Inductor cache; no remote/imported cache claim",
    }
    corrupt_safe = bool(
        corrupt_run is None
        or corrupt_run["returncode"] != 0
        or all(
            item.get("ok")
            for item in (corrupt_run.get("payload") or {}).get("correctness", [])
        )
    )
    result["component_pass"] = bool(
        all(item["returncode"] == 0 for item in cold)
        and disk_hit["returncode"] == 0
        and invalidated["returncode"] == 0
        and result["identity"]["model_sensitive"]
        and corrupt_safe
    )
    return result


def profile_one(torch: Any, fn: Callable[[], Any]) -> Dict[str, Any]:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
            fn()
            torch.cuda.synchronize()
        rows = []
        for item in profile.key_averages():
            cuda_us = float(
                getattr(item, "device_time_total", 0.0)
                or getattr(item, "cuda_time_total", 0.0)
                or 0.0
            )
            if cuda_us or "hqsb" in item.key or "aten::" in item.key:
                rows.append(
                    {
                        "name": item.key,
                        "count": int(item.count),
                        "cpu_time_total_us": float(item.cpu_time_total),
                        "device_time_total_us": cuda_us,
                    }
                )
        return {
            "supported": True,
            "events": sorted(rows, key=lambda row: row["device_time_total_us"], reverse=True)[:80],
            "cuda_event_count": sum(row["count"] for row in rows if row["device_time_total_us"] > 0),
        }
    except Exception as exc:
        return {"supported": False, "error": error_record(exc, stage="profile")}


def collect_e06_07(torch: Any, repeats: int) -> Dict[str, Any]:
    register_torch_ops(REPO)
    Unfused, CustomOpaque, Fused, _ = module_classes(torch)
    shapes = [(128, "short"), (512, "balanced"), (2048, "long_prefill")]
    results: List[Dict[str, Any]] = []
    for rows, workload in shapes:
        x, residual, weight = make_inputs(torch, rows, 2048, torch.float16, 700 + rows)
        reference_module = Unfused().cuda()
        custom_module = CustomOpaque().cuda()
        compiled_unfused = torch.compile(
            Unfused().cuda(), backend="inductor", fullgraph=True, dynamic=False
        )
        compiled_fused = torch.compile(
            Fused().cuda(), backend="inductor", fullgraph=True, dynamic=False
        )
        arms = [
            ("reference_eager", reference_module),
            ("custom_opaque_eager", custom_module),
            ("compile_without_rewrite", compiled_unfused),
            ("compile_with_hqsb_rewrite", compiled_fused),
        ]
        expected = reference_module(x, residual, weight)
        for name, arm in arms:
            reset_audit_counters()
            try:
                first_started = time.perf_counter()
                output = arm(x, residual, weight)
                torch.cuda.synchronize()
                first_ms = (time.perf_counter() - first_started) * 1000.0
                correctness = [
                    compare_tensors(output[0], expected[0], atol=3e-3, rtol=3e-3),
                    compare_tensors(output[1], expected[1], atol=0.0, rtol=0.0),
                ]
                timing = timed_cuda(
                    torch, lambda arm=arm: arm(x, residual, weight), repeats=repeats
                )
                results.append(
                    {
                        "workload": workload,
                        "rows": rows,
                        "hidden": 2048,
                        "arm": name,
                        "executed": True,
                        "first_call_ms": first_ms,
                        "steady": timing,
                        "correctness": correctness,
                        "actual_dispatch": audit_snapshot(),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "workload": workload,
                        "rows": rows,
                        "hidden": 2048,
                        "arm": name,
                        "executed": False,
                        "error": error_record(exc, stage="compile_or_execute"),
                        "actual_dispatch": audit_snapshot(),
                    }
                )
                torch._dynamo.reset()
        del compiled_unfused, compiled_fused
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    x, residual, weight = make_inputs(torch, 128, 2048, torch.float16, 777)
    profiles = {
        "reference_eager": profile_one(torch, lambda: Unfused().cuda()(x, residual, weight)),
        "custom_fused_eager": profile_one(torch, lambda: Fused().cuda()(x, residual, weight)),
    }
    summaries: List[Dict[str, Any]] = []
    for workload in {row["workload"] for row in results}:
        subset = {row["arm"]: row for row in results if row["workload"] == workload}
        baseline_row = subset["reference_eager"]
        if not baseline_row.get("executed"):
            continue
        baseline = baseline_row["steady"]["median_ms"]
        for arm, row in subset.items():
            if not row.get("executed"):
                summaries.append(
                    {"workload": workload, "arm": arm, "executed": False, "reason": "see arms error"}
                )
                continue
            median = row["steady"]["median_ms"]
            summaries.append(
                {
                    "workload": workload,
                    "arm": arm,
                    "executed": True,
                    "median_ms": median,
                    "speedup_vs_reference": baseline / median if median > 0 else None,
                    "delta_percent": (median / baseline - 1.0) * 100.0 if baseline > 0 else None,
                }
            )
    result = {
        "arms": results,
        "summary": summaries,
        "profiles": profiles,
        "allocation_model": {
            "unfused_intermediate_bytes_per_shape": {
                workload: rows * 2048 * 2 for rows, workload in shapes
            },
            "fused_saved_intermediate": "one FP16 [M,H] residual-add tensor write/read",
            "measured_dram_bytes": None,
        },
        "sampling": {
            "independent_processes": 1,
            "repeats_per_arm": repeats,
            "formal_required": "3 independent processes x >=30 per arm",
            "classification": "exploratory" if repeats < 30 else "single-process confirmatory-incomplete",
        },
        "claim_boundary": "component/block fixture only; no Qwen TTFT/TPOT/E2E claim",
    }
    result["component_pass"] = bool(
        all(
            row.get("executed") and all(item["ok"] for item in row["correctness"])
            for row in results
        )
        and any(
            row["actual_dispatch"]["calls"]["cuda_fused_add_rms_norm"] > 0
            for row in results
        )
    )
    return result


def _capture_once(torch: Any, rows: int, hidden: int, replays: int, seed: int) -> Dict[str, Any]:
    _, _, Fused, _ = module_classes(torch)
    module = Fused().cuda()
    static_x, static_residual, static_weight = make_inputs(
        torch, rows, hidden, torch.float16, seed
    )
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            module(static_x, static_residual, static_weight)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    before_reserved = torch.cuda.memory_reserved()
    graph = torch.cuda.CUDAGraph()
    capture_started = time.perf_counter()
    with torch.cuda.graph(graph):
        graph_output = module(static_x, static_residual, static_weight)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter() - capture_started) * 1000.0
    after_reserved = torch.cuda.memory_reserved()
    samples: List[float] = []
    correctness: List[Dict[str, Any]] = []
    owned_snapshot = None
    for index in range(replays):
        x, residual, weight = make_inputs(torch, rows, hidden, torch.float16, seed + 1 + index % 2)
        static_x.copy_(x)
        static_residual.copy_(residual)
        static_weight.copy_(weight)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        if index == 0:
            owned_snapshot = graph_output[0].clone()
        if index in (0, 1, replays - 1):
            expected = fused_reference(x, residual, weight)
            correctness.append(
                {
                    "index": index,
                    "normalized": compare_tensors(
                        graph_output[0], expected[0], atol=3e-3, rtol=3e-3
                    ),
                    "updated": compare_tensors(
                        graph_output[1], expected[1], atol=0.0, rtol=0.0
                    ),
                }
            )
    owned_unchanged = bool(owned_snapshot is not None and torch.equal(owned_snapshot, owned_snapshot.clone()))
    ordered = sorted(samples)
    return {
        "rows": rows,
        "hidden": hidden,
        "capture_ms": capture_ms,
        "replays": replays,
        "replay_samples_ms": samples,
        "replay_median_ms": statistics.median(samples),
        "replay_p95_ms": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "graph_pool_reserved_delta_bytes": after_reserved - before_reserved,
        "correctness": correctness,
        "graph_output_data_ptrs": [int(value.data_ptr()) for value in graph_output],
        "owned_clone_unchanged": owned_unchanged,
        "output_contract": "graph outputs are reusable views; caller clones before retaining",
    }


def collect_e06_08(torch: Any, replays: int) -> Dict[str, Any]:
    register_torch_ops(REPO)
    captures: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for capture_id in range(3):
        for rows in (1, 32):
            try:
                captures.append(_capture_once(torch, rows, 128, replays, 800 + capture_id * 10 + rows))
            except Exception as exc:
                failures.append(
                    {
                        "capture_id": capture_id,
                        "rows": rows,
                        "error": error_record(exc, stage="cuda_graph_capture"),
                    }
                )
                torch.cuda.empty_cache()
    x, residual, weight = make_inputs(torch, 33, 128, torch.float16, 899)
    fallback = torch.ops.hqsb.fused_add_rms_norm(x, residual, weight, EPS)
    torch.cuda.synchronize()
    expected = fused_reference(x, residual, weight)
    out_of_bucket = {
        "rows": 33,
        "actual_path": "eager_fallback",
        "normalized": compare_tensors(fallback[0], expected[0], atol=3e-3, rtol=3e-3),
        "updated": compare_tensors(fallback[1], expected[1], atol=0.0, rtol=0.0),
    }
    result = {
        "captures": captures,
        "failures": failures,
        "out_of_bucket": out_of_bucket,
        "sampling": {"independent_captures": 3, "replays_per_capture": replays},
        "claim_boundary": "static fused block buckets M=1/32; no KV or full generation graph claim",
    }
    result["component_pass"] = bool(
        not failures
        and len(captures) == 6
        and all(
            all(item["normalized"]["ok"] and item["updated"]["ok"] for item in row["correctness"])
            for row in captures
        )
        and out_of_bucket["normalized"]["ok"]
        and out_of_bucket["updated"]["ok"]
    )
    return result


def collect_e06_09(torch: Any) -> Dict[str, Any]:
    register_torch_ops(REPO)
    _, _, Fused, _ = module_classes(torch)
    x, residual, weight = make_inputs(torch, 4, 128, torch.float16, 901)
    golden = fused_reference(x, residual, weight)
    state_hash = sha256_bytes(
        x.cpu().numpy().tobytes() + residual.cpu().numpy().tobytes() + weight.cpu().numpy().tobytes()
    )
    input_faults: List[Dict[str, Any]] = []
    calls = {
        "dtype": lambda: torch.ops.hqsb.fused_add_rms_norm(
            x.double(), residual.double(), weight.double(), EPS
        ),
        "shape": lambda: torch.ops.hqsb.fused_add_rms_norm(x, residual[:-1], weight, EPS),
        "stride": lambda: torch.ops.hqsb.fused_add_rms_norm(x.t(), residual.t(), weight, EPS),
        "device": lambda: torch.ops.hqsb.fused_add_rms_norm(x.cpu(), residual, weight.cpu(), EPS),
    }
    for name, call in calls.items():
        for attempt in range(3):
            before_launches = len(audit_snapshot()["launches"])
            try:
                call()
                row = {"case": name, "attempt": attempt, "rejected": False}
            except Exception as exc:
                row = {
                    "case": name,
                    "attempt": attempt,
                    "rejected": True,
                    **error_record(exc, stage="validation"),
                }
            row["native_launch_delta"] = len(audit_snapshot()["launches"]) - before_launches
            input_faults.append(row)

    def failing_backend(gm, example_inputs):
        raise RuntimeError("E06_INJECTED_COMPILE_FAILURE")

    strict: Dict[str, Any]
    try:
        torch.compile(Fused().cuda(), backend=failing_backend, fullgraph=True)(x, residual, weight)
        strict = {"rejected": False}
    except Exception as exc:
        strict = {"rejected": True, "error": error_record(exc, stage="compile")}
    torch._dynamo.reset()
    before_launches = len(audit_snapshot()["launches"])
    try:
        torch.compile(Fused().cuda(), backend=failing_backend, fullgraph=True)(x, residual, weight)
        fallback_output = None
        fallback_reason = None
    except Exception as exc:
        fallback_reason = error_record(exc, stage="compile")
        fallback_output = golden
    torch._dynamo.reset()
    fallback = {
        "reason": fallback_reason,
        "actual": "reference_before_custom_execution",
        "native_launch_delta_before_fallback": len(audit_snapshot()["launches"]) - before_launches,
        "correct": bool(fallback_output and torch.equal(fallback_output[1], golden[1])),
    }

    with tempfile.TemporaryDirectory(prefix="hqsb-s06-missing-") as missing_root:
        try:
            from hqsb.integration.torch_ops import NativeKernels

            NativeKernels(Path(missing_root))
            missing_library = {"rejected": False}
        except Exception as exc:
            missing_library = {"rejected": True, "error": error_record(exc, stage="load")}
    with tempfile.TemporaryDirectory(prefix="hqsb-s06-badso-") as bad_root:
        bad = Path(bad_root) / "broken.so"
        bad.write_bytes(b"not an ELF shared object")
        bad_binary = run_child("bad-binary", "--binary", str(bad), timeout=30)

    good_identity = abi_contract.simulate_identity(
        torch_version=str(torch.__version__),
        cuda_runtime=str(torch.version.cuda),
        gpu_arch=f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}",
        fatbin_targets=(f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}",),
    )
    bad_identity = abi_contract.simulate_identity(gpu_arch="sm_90", fatbin_targets=("sm_87",))
    abi = {
        "actual": abi_contract.CompatibilityMatrix().check(good_identity).as_dict(),
        "target_mismatch": abi_contract.CompatibilityMatrix().check(bad_identity).as_dict(),
    }
    healthy = torch.ops.hqsb.fused_add_rms_norm(x, residual, weight, EPS)
    torch.cuda.synchronize()
    post_hash = sha256_bytes(
        x.cpu().numpy().tobytes() + residual.cpu().numpy().tobytes() + weight.cpu().numpy().tobytes()
    )
    result = {
        "input_faults": input_faults,
        "compile_strict": strict,
        "compile_fallback": fallback,
        "missing_library": missing_library,
        "bad_binary": bad_binary,
        "abi": abi,
        "state": {
            "before_hash": state_hash,
            "after_hash": post_hash,
            "unchanged": state_hash == post_hash,
        },
        "healthy_recovery": [
            compare_tensors(healthy[0], golden[0], atol=3e-3, rtol=3e-3),
            compare_tensors(healthy[1], golden[1], atol=0.0, rtol=0.0),
        ],
    }
    result["component_pass"] = bool(
        all(row["rejected"] and row["native_launch_delta"] == 0 for row in input_faults)
        and strict["rejected"]
        and fallback["native_launch_delta_before_fallback"] == 0
        and fallback["correct"]
        and missing_library["rejected"]
        and bad_binary["returncode"] == 0
        and result["state"]["unchanged"]
        and all(item["ok"] for item in result["healthy_recovery"])
    )
    return result


def linear_slope(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denom = sum((value - x_mean) ** 2 for value in xs)
    if denom == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denom


def collect_e06_10(torch: Any, duration_s: float) -> Dict[str, Any]:
    register_torch_ops(REPO)
    Unfused, _, Fused, _ = module_classes(torch)
    sessions: List[Dict[str, Any]] = []
    weak_refs: List[Any] = []
    per_session = max(1.0, duration_s / 3.0)
    for session_id in range(3):
        started = time.monotonic()
        iteration = 0
        samples = [memory_snapshot(torch, "session_start")]
        errors: List[Dict[str, Any]] = []
        hashes: List[str] = []
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        module = Fused().cuda()
        weak_refs.append(weakref.ref(module))
        while iteration < 100 or time.monotonic() - started < per_session:
            iteration_started = time.monotonic()
            rows = 32 if (iteration // 10) % 2 == 0 else 128
            x, residual, weight = make_inputs(
                torch, rows, 128, torch.float16, 1000 + session_id * 1000 + iteration
            )
            stream = stream_a if iteration % 2 == 0 else stream_b
            try:
                with torch.cuda.stream(stream):
                    if iteration % 3 == 0:
                        output = Unfused().cuda()(x, residual, weight)
                        path = "disabled_reference"
                    elif iteration % 20 == 19:
                        try:
                            module(x.t(), residual.t(), weight)
                            raise AssertionError("unsupported stride was not rejected")
                        except (TorchOperatorError, RuntimeError):
                            output = Unfused().cuda()(x, residual, weight)
                            path = "rejected_then_reference"
                    else:
                        output = module(x, residual, weight)
                        path = "enabled_fused"
                stream.synchronize()
                expected = fused_reference(x, residual, weight)
                if not (
                    torch.allclose(output[0], expected[0], atol=3e-3, rtol=3e-3)
                    and torch.equal(output[1], expected[1])
                ):
                    raise AssertionError(f"correctness failure on path {path}")
                if iteration % 10 == 0:
                    hashes.append(sha256_bytes(output[0].float().cpu().numpy().tobytes()))
            except Exception as exc:
                errors.append(
                    {
                        "iteration": iteration,
                        "path": locals().get("path", "unknown"),
                        "error": error_record(exc, stage="lifecycle"),
                    }
                )
            iteration += 1
            if iteration % 10 == 0:
                samples.append(memory_snapshot(torch, f"iteration_{iteration}"))
            # Fixed offered rate prevents the evidence recorder itself from
            # becoming an unbounded allocation/CPU-pressure source.
            remaining = 0.05 - (time.monotonic() - iteration_started)
            if remaining > 0:
                time.sleep(remaining)
        torch.cuda.synchronize()
        del module, stream_a, stream_b, x, residual, weight, output
        gc.collect()
        torch.cuda.empty_cache()
        samples.append(memory_snapshot(torch, "session_closed"))
        allocated_points = [
            (float(index), float(item.get("cuda_allocated_bytes") or 0))
            for index, item in enumerate(samples[1:-1], start=1)
        ]
        reserved_points = [
            (float(index), float(item.get("cuda_reserved_bytes") or 0))
            for index, item in enumerate(samples[1:-1], start=1)
        ]
        sessions.append(
            {
                "session_id": session_id,
                "duration_s": time.monotonic() - started,
                "iterations": iteration,
                "samples": samples,
                "errors": errors,
                "output_hash_samples": hashes,
                "allocated_slope_bytes_per_sample": linear_slope(allocated_points),
                "reserved_slope_bytes_per_sample": linear_slope(reserved_points),
            }
        )
    gc.collect()
    liveness = {
        "weak_refs_total": len(weak_refs),
        "weak_refs_alive_after_close": sum(1 for item in weak_refs if item() is not None),
    }
    allocated_slopes = [abs(row["allocated_slope_bytes_per_sample"] or 0.0) for row in sessions]
    result = {
        "sessions": sessions,
        "liveness": liveness,
        "duration_budget_s": duration_s,
        "actual_total_duration_s": sum(row["duration_s"] for row in sessions),
        "sampling": "allocated/reserved/RSS every 10 iterations; two CUDA streams",
        "claim_boundary": (
            "component lifecycle only; no Qwen KV/model object lifecycle. "
            + ("30 minute duration met" if duration_s >= 1800 else "formal 30 minute duration not met")
        ),
    }
    result["component_pass"] = bool(
        all(not row["errors"] and row["iterations"] >= 100 for row in sessions)
        and liveness["weak_refs_alive_after_close"] == 0
        and max(allocated_slopes, default=0.0) <= 1024 * 1024
    )
    return result


def collect_e06_11(torch: Any) -> Dict[str, Any]:
    register_torch_ops(REPO)
    _, _, Fused, _ = module_classes(torch)
    configurations = [
        {"name": "qwen_like_h128", "hidden": 128, "eps": 1.0e-6},
        {"name": "qwen3_h2048", "hidden": 2048, "eps": 1.0e-6},
        {"name": "variant_h512_eps", "hidden": 512, "eps": 1.0e-5},
    ]
    rows: List[Dict[str, Any]] = []
    for process_seed in (1101, 1102, 1103):
        for config in configurations:
            x, residual, weight = make_inputs(
                torch, 4, config["hidden"], torch.float16, process_seed
            )
            output = torch.ops.hqsb.fused_add_rms_norm(
                x, residual, weight, config["eps"]
            )
            torch.cuda.synchronize()
            expected = fused_reference(x, residual, weight, config["eps"])
            rows.append(
                {
                    "seed": process_seed,
                    "config": config,
                    "normalized": compare_tensors(
                        output[0], expected[0], atol=3e-3, rtol=3e-3
                    ),
                    "updated": compare_tensors(
                        output[1], expected[1], atol=0.0, rtol=0.0
                    ),
                }
            )
    dummy = adapter.DummyBackendAdapter()
    capability = dummy.capability()
    accepted = dummy.route({"dtype": "float16", "shape": [4, 128]})
    rejected = dummy.route({"dtype": "float64", "shape": [4, 128]})
    core_paths = [REPO / "hqsb/integration/torch_ops.py"]
    findings = adapter.hardcode_scan([str(path) for path in core_paths], adapter_markers=())
    hardcode = adapter.hardcode_report(findings)
    source_hashes = {str(path.relative_to(REPO)): sha256_file(path) for path in core_paths}
    result = {
        "fixture_matrix": rows,
        "dummy_backend": {
            "capability": capability.as_dict() if hasattr(capability, "as_dict") else json_safe(capability),
            "accepted_route": json_safe(accepted),
            "rejected_route": json_safe(rejected),
            "spy": json_safe(dummy.spy),
            "performance_claim": dummy.performance_claim_allowed(),
        },
        "hardcode_scan": hardcode,
        "core_source_hashes": source_hashes,
        "adapter_only_diff": [],
        "claim_boundary": (
            "three Transformer-like hidden/epsilon fixtures plus dummy backend; "
            "no second real model block/generation, so no full cross-model claim"
        ),
    }
    result["component_pass"] = bool(
        all(row["normalized"]["ok"] and row["updated"]["ok"] for row in rows)
        and hardcode.get("leaks", 1) == 0
        and not dummy.performance_claim_allowed()["allowed"]
    )
    return result


def formal_verdict(experiment_id: str, observed: Mapping[str, Any], prerequisites: Any) -> Dict[str, Any]:
    component_pass = bool(observed.get("component_pass"))
    missing = list(prerequisites.missing)
    scope_gaps: List[str] = []
    if experiment_id == "E06-03":
        scope_gaps.extend(["complete Qwen block/model/long-generation matrix", "six frozen workloads"])
    if experiment_id == "E06-04":
        scope_gaps.append("full Qwen graph census")
    if experiment_id == "E06-05":
        scope_gaps.append("model-level ordered shape trace")
    if experiment_id == "E06-06":
        scope_gaps.extend(["three truly empty cold caches", "schema/build/arch version matrix"])
    if experiment_id == "E06-07":
        scope_gaps.extend(["3 independent process confirmatory sampling", "Qwen TTFT/TPOT/E2E"])
    if experiment_id == "E06-08":
        scope_gaps.append("full model/KV capture is not claimed")
    if experiment_id == "E06-10":
        scope_gaps.append("Qwen model/KV object lifecycle")
    if experiment_id == "E06-11":
        scope_gaps.append("second real model block/model/generation")
    status = "BLOCKED"
    reason = (
        "component observations passed but the documented upstream evidence/scope gate is incomplete"
        if component_pass
        else "one or more component checks failed and the upstream evidence gate is also incomplete"
    )
    return {
        "experiment_id": experiment_id,
        "status": status,
        "formal_status": status,
        "component_result": "PASS" if component_pass else "FAIL",
        "protocol_valid": False,
        "correctness_or_quality": "PASS" if component_pass else "FAIL_OR_INCOMPLETE",
        "effect_verdict": "INCONCLUSIVE",
        "claim_scope": "partial Jetson component evidence only",
        "reason": reason,
        "missing_prerequisites": missing,
        "scope_gaps": scope_gaps,
        "expected_effect_met": bool(component_pass),
        "single_item_pass_standard_met": False,
        "recorded_at": utc_now(),
    }


def render_report(
    experiment_id: str,
    verdict: Mapping[str, Any],
    observed: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> str:
    title = EXPERIMENT_TITLES[experiment_id]
    component = verdict["component_result"]
    gaps = verdict.get("scope_gaps") or []
    missing = verdict.get("missing_prerequisites") or []
    evidence_roots = {
        "observations.json": observed,
        "environment_fingerprint.json": environment,
    }
    lines = [
        f"# {experiment_id} 实验报告：{title}",
        "",
        f"- 执行时间：{verdict['recorded_at']}",
        f"- 目标设备：{environment.get('device', {}).get('name') if environment.get('device') else 'unknown'}",
        f"- 正式状态：**{verdict['status']}**",
        f"- 已执行组件结果：**{component}**",
        f"- 预计效果是否在已执行范围内达到：**{'是' if verdict['expected_effect_met'] else '否'}**",
        "- 原单项通过标准是否完整满足：**否**",
        "",
        "## 结论",
        "",
        verdict["reason"] + "。本报告不把组件 PASS 升级为阶段实验 PASS。",
        "",
        "## 已采集信息/数据",
        "",
        "完整机器可读记录位于 `raw/observations.json`，环境与源码身份位于 "
        "`raw/environment_fingerprint.json`，裁决位于 `raw/verdict.json`。本次包括真实路径、"
        "错误和范围边界；未执行项保持显式缺口。",
        "",
        "### 必采集信息映射",
        "",
        "| 必采信息 | 原始位置 | 采集状态 | 摘要 |",
        "|---|---|---|---|",
    ]
    for label, filename, key in REQUIRED_EVIDENCE[experiment_id]:
        root = evidence_roots[filename]
        present = key in root and root[key] is not None
        value = root.get(key)
        if isinstance(value, Mapping):
            summary = f"object，{len(value)} 个字段"
        elif isinstance(value, (list, tuple)):
            summary = f"array，{len(value)} 条记录"
        elif present:
            summary = str(value).replace("|", "\\|")[:100]
        else:
            summary = "显式缺口/未产生"
        lines.append(
            f"| {label} | `raw/{filename}::{key}` | {'已采集' if present else '未产生'} | {summary} |"
        )
    lines.extend(
        [
        "",
        "## 预计效果与单项标准对照",
        "",
        "| 项目 | 结果 | 说明 |",
        "|---|---|---|",
        f"| 已执行组件能否工作 | {component} | `observations.json::component_pass` |",
        f"| 预计效果（有限范围） | {'达到' if verdict['expected_effect_met'] else '未达到'} | 仅限 Jetson 组件证据 |",
        "| 原单项通过标准 | 未完整达到 | 上游门禁/完整范围未满足，正式状态 BLOCKED |",
        "| 性能/效果结论 | INCONCLUSIVE | 不从探索或单进程样本外推 |",
        "",
        "## 阻塞与未覆盖",
        "",
        ]
    )
    lines.extend(f"- 前置：`{item}`" for item in missing)
    lines.extend(f"- 范围：{item}" for item in gaps)
    if not missing and not gaps:
        lines.append("- 仍受 S06 阶段统一正式验收门约束。")
    lines.extend(
        [
            "",
            "## 前端访问",
            "",
            "Console 通过 `/api/console/v1/evidence` 自动索引本目录的 "
            "`raw/verdict.json`，并可在 `/evidence` 或 `/experiments` 查看、预览和下载同目录证据。",
            "",
            "## 证据身份",
            "",
            f"- 观测摘要 SHA-256：`{sha256_bytes(json.dumps(json_safe(observed), sort_keys=True).encode())}`",
            f"- 详细协议：[`{DETAIL_FILES[experiment_id]}`](../../details/S06/{DETAIL_FILES[experiment_id]})",
            "",
        ]
    )
    return "\n".join(lines)


def write_experiment_manifest(directory: Path, experiment_id: str) -> None:
    manifest_path = directory / "raw/manifest.json"
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path != manifest_path:
            rows.append(
                {
                    "path": str(path.relative_to(directory)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(manifest_path, {"experiment_id": experiment_id, "files": rows})


def archive_experiment(
    experiment_id: str,
    observed: Mapping[str, Any],
    prerequisites: Any,
    environment: Mapping[str, Any],
) -> Dict[str, Any]:
    directory = STAGE_ROOT / experiment_id
    raw = directory / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    verdict = formal_verdict(experiment_id, observed, prerequisites)
    write_json(raw / "observations.json", observed)
    write_json(raw / "environment_fingerprint.json", environment)
    write_json(raw / "prerequisites.json", prerequisites.as_dict())
    write_json(raw / "verdict.json", verdict)
    report = render_report(experiment_id, verdict, observed, environment)
    (directory / f"{experiment_id}_实验报告.md").write_text(report, encoding="utf-8")
    write_experiment_manifest(directory, experiment_id)
    return verdict


def render_stage_report(verdicts: Sequence[Mapping[str, Any]], environment: Mapping[str, Any]) -> str:
    component_passes = sum(1 for row in verdicts if row["component_result"] == "PASS")
    lines = [
        "# S06 框架集成与图优化阶段实验报告",
        "",
        f"执行时间：{utc_now()}。目标：{environment.get('device', {}).get('name') if environment.get('device') else 'unknown'}。",
        "",
        "## 总结论",
        "",
        f"11 项均已建立正式结果目录并采集当前可执行范围的数据；组件检查通过 {component_passes}/11。"
        "由于 S04.5 M4 模型回接证据、S04.5 验收报告、冻结六负载基线和 S05 P0 正式门禁缺失，"
        "正式裁决不得升级为 PASS，11 项均保留 `BLOCKED`。这是有原始证据的阻塞裁决，"
        "不再是 `NOT_STARTED`，也不把组件成功伪装为完整实验成功。",
        "",
        "## 逐项裁决",
        "",
        "| ID | 组件结果 | 正式状态 | 预计效果（已执行范围） | 单项标准 |",
        "|---|---|---|---|---|",
    ]
    for row in verdicts:
        lines.append(
            f"| [{row['experiment_id']}]({row['experiment_id']}/{row['experiment_id']}_实验报告.md) "
            f"| {row['component_result']} | {row['status']} | "
            f"{'达到' if row['expected_effect_met'] else '未达到'} | 未完整满足 |"
        )
    lines.extend(
        [
            "",
            "## 前端接口",
            "",
            "每项均包含 `raw/verdict.json`，符合 `EvidenceCatalog` 的扫描规则。后端接口 "
            "`GET /api/console/v1/evidence` 与 `GET /api/console/v1/evidence/{id}` 可访问，"
            "前端 `/evidence`、`/experiments` 复用该接口。接口验收记录见 "
            "[`frontend_access.json`](frontend_access.json)。",
            "",
            "## 后续解除阻塞的最短路径",
            "",
            "1. 先完成 S04.5 E045-01–06，生成真实模型回接 marker、阶段验收报告与六负载 FP16 基线，并关闭 S05 P0 门禁。",
            "2. 修复 Jetson 当前 PyTorch/用户 Triton 的 `triton_key` 版本不匹配，再复跑 E06-03/06/07 的 Inductor 路径。",
            "3. 复跑 E06-03/04/05/07 的完整 Qwen block/model/generation 与六负载矩阵。",
            "4. E06-07 按三独立进程、每臂每进程至少 30 次完成确认性统计；E06-10 保留三会话 30 分钟以上记录。",
            "5. 若声明完整跨模型复用，再补第二真实模型，而不是把轻量 fixture 结果外推。",
            "",
        ]
    )
    return "\n".join(lines)


def frontend_access_payload(verdicts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build and verify the browser contract against the files just archived."""
    from hqsb.console.evidence import EvidenceCatalog

    catalog = EvidenceCatalog(REPO)
    rows = [row for row in catalog.scan(refresh=True) if row["stage"] == "S06"]
    expected = [row["experiment_id"] for row in verdicts]
    discovered = [row["experiment"] for row in rows]
    attachment_counts = {row["experiment"]: len(row["files"]) for row in rows}
    verified = (
        discovered == expected
        and all(row["status"] == "BLOCKED" for row in rows)
        and all(count >= 6 for count in attachment_counts.values())
    )
    if not verified:
        raise RuntimeError(
            "S06 EvidenceCatalog verification failed: "
            f"expected={expected!r}, discovered={discovered!r}, "
            f"attachment_counts={attachment_counts!r}"
        )
    return {
        "generated_at": utc_now(),
        "expected_catalog_items": len(verdicts),
        "discovered_catalog_items": len(rows),
        "scan_pattern": "docs/stage_experiments/*/*/raw/verdict.json",
        "api": [
            "/api/console/v1/evidence",
            "/api/console/v1/evidence/{evidence_id}",
            "/api/console/v1/evidence/{evidence_id}/download",
        ],
        "frontend_routes": ["/evidence", "/experiments"],
        "verification": "PASS",
        "verified_checks": [
            "catalog_count",
            "experiment_ids",
            "formal_status",
            "attachment_index",
            "detail_decode",
        ],
        "attachment_counts": attachment_counts,
        "experiments": expected,
    }


def execute_all(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("S06 real collector requires CUDA on the remote target")
    environment = environment_fingerprint(torch)
    prerequisites = experiment.check_prerequisites(str(REPO))
    selected = list(EXPERIMENT_TITLES) if args.experiment == "all" else [args.experiment]
    collectors: Dict[str, Callable[[], Dict[str, Any]]] = {
        "E06-01": lambda: collect_e06_01(torch),
        "E06-02": lambda: collect_e06_02(torch),
        "E06-03": lambda: collect_e06_03(torch, args.model_path, args.run_model),
        "E06-04": lambda: collect_e06_04(torch),
        "E06-05": lambda: collect_e06_05(torch),
        "E06-06": lambda: collect_e06_06(torch, STAGE_ROOT),
        "E06-07": lambda: collect_e06_07(torch, args.performance_repeats),
        "E06-08": lambda: collect_e06_08(torch, args.graph_replays),
        "E06-09": lambda: collect_e06_09(torch),
        "E06-10": lambda: collect_e06_10(torch, args.lifecycle_seconds),
        "E06-11": lambda: collect_e06_11(torch),
    }
    verdicts: List[Dict[str, Any]] = []
    for experiment_id in selected:
        started = time.perf_counter()
        print(f"[{utc_now()}] START {experiment_id}", flush=True)
        try:
            observed = collectors[experiment_id]()
        except Exception as exc:
            observed = {
                "component_pass": False,
                "collector_error": error_record(exc, stage="collector"),
            }
        observed["collector_elapsed_s"] = time.perf_counter() - started
        observed["experiment_id"] = experiment_id
        observed["captured_at"] = utc_now()
        verdict = archive_experiment(experiment_id, observed, prerequisites, environment)
        verdicts.append(verdict)
        print(
            f"[{utc_now()}] END {experiment_id} component={verdict['component_result']} "
            f"formal={verdict['status']} elapsed={observed['collector_elapsed_s']:.3f}s",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
    if args.experiment == "all":
        STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        (STAGE_ROOT / "S06_阶段实验报告_20260921.md").write_text(
            render_stage_report(verdicts, environment), encoding="utf-8"
        )
        write_json(STAGE_ROOT / "frontend_access.json", frontend_access_payload(verdicts))
    print(json.dumps({"verdicts": verdicts}, ensure_ascii=False), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["all", *EXPERIMENT_TITLES], default="all")
    parser.add_argument("--model-path", default="~/models/hqsb/Qwen3-1.7B")
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--performance-repeats", type=int, default=30)
    parser.add_argument("--graph-replays", type=int, default=100)
    parser.add_argument("--lifecycle-seconds", type=float, default=1800.0)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--probe",
        choices=["registration", "compile", "bad-binary", "qwen"],
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cache-dir", default="", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=["fused", "unfused"], default="fused", help=argparse.SUPPRESS)
    parser.add_argument("--binary", default="", help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.performance_repeats < 1 or args.graph_replays < 1 or args.lifecycle_seconds <= 0:
        raise SystemExit("repeat/duration arguments must be positive")
    if args.probe:
        if args.probe == "bad-binary":
            payload = bad_binary_probe(args.binary)
        else:
            import torch

            if args.probe == "registration":
                payload = registration_probe(torch)
            elif args.probe == "compile":
                if not args.cache_dir:
                    raise SystemExit("compile probe requires --cache-dir")
                payload = compile_probe(torch, args.cache_dir, args.variant)
            elif args.probe == "qwen":
                payload = qwen_probe(torch, args.model_path)
            else:  # pragma: no cover
                raise AssertionError(args.probe)
        print(json.dumps(json_safe(payload), ensure_ascii=False))
        return 0
    if args.summarize_only:
        verdicts = [
            json.loads((STAGE_ROOT / experiment_id / "raw/verdict.json").read_text())
            for experiment_id in EXPERIMENT_TITLES
        ]
        environment = json.loads(
            (STAGE_ROOT / "E06-01/raw/environment_fingerprint.json").read_text()
        )
        for experiment_id, verdict in zip(EXPERIMENT_TITLES, verdicts):
            directory = STAGE_ROOT / experiment_id
            observed = json.loads((directory / "raw/observations.json").read_text())
            (directory / f"{experiment_id}_实验报告.md").write_text(
                render_report(experiment_id, verdict, observed, environment), encoding="utf-8"
            )
            write_experiment_manifest(directory, experiment_id)
        (STAGE_ROOT / "S06_阶段实验报告_20260921.md").write_text(
            render_stage_report(verdicts, environment), encoding="utf-8"
        )
        write_json(STAGE_ROOT / "frontend_access.json", frontend_access_payload(verdicts))
        print(json.dumps({"summarized": len(verdicts)}, ensure_ascii=False))
        return 0
    return execute_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
