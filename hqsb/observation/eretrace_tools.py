"""ERETrace runtime identity, admission scope and child-only launch environment."""
import json
import importlib.util
import os
from pathlib import Path
import sys

from hqsb.observation.storage import sha256


def discover(runtime):
    result = {"tool": "eretrace", "installed": False, "collection_permission": "checked_when_collecting",
              "scope": "selected_Triton_ordinary_launches_one_producer_one_context"}
    if runtime is None:
        return {**result, "reason": "在执行节点运行 scripts/setup/install_eretrace.py，并配置 eretrace_runtime。"}
    root = Path(runtime).resolve()
    try:
        manifest = json.loads((root / "hqsb-runtime.json").read_text())
        if manifest["schema_version"] != "hqsb-eretrace-1" or not manifest["runtime_files"]:
            raise ValueError("ERE_RUNTIME_SCHEMA")
        for relative, expected in manifest["runtime_files"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha256(path) != expected:
                raise ValueError("ERE_RUNTIME_FILE_CHANGED:" + relative)
        result.update(installed=True, executable=str(root), version="ERETrace / HQSB adapter 1",
                      build_manifest_sha256=sha256(root / "hqsb-runtime.json"), arch=manifest["arch"],
                      adapter_sha256=manifest["adapter_sha256"])
        result["capabilities"] = manifest.get("capabilities", ["warp"])
    except (OSError, ValueError, KeyError) as exc:
        result["reason"] = str(exc)[:500]
    return result


def validate_scope(options, workload):
    if not options.get("enforce_eager"):
        raise ValueError("ERE_EAGER_SOURCE_REQUIRED: 请在原生引擎配置中启用 Eager，重新评测后选择该记录。")
    if options.get("tensor_parallel_size", 1) != 1 or options.get("pipeline_parallel_size", 1) != 1:
        raise ValueError("ERE_SINGLE_RANK_SOURCE_REQUIRED: 请使用 TP=1、PP=1 的原生评测记录。")
    if workload.get("concurrency", 1) != 1:
        raise ValueError("ERE_SINGLE_REQUEST_SOURCE_REQUIRED: 请使用请求并发为 1 的原生评测记录。")


def launch_environment(runtime, directory, plan, triton_cache):
    root, directory = Path(runtime).resolve(), Path(directory).resolve()
    build = json.loads((root / "hqsb-runtime.json").read_text())
    driver = Path(build["driver_library"])
    if sha256(driver) != build["driver_sha256"]:
        raise ValueError("ERE_DRIVER_CHANGED: 请用当前驱动重新构建 ERETrace runtime。")
    trace = directory / "ere-traces"
    trace.mkdir(parents=True, exist_ok=True)
    probes = (root / "build/block_sched.toml").read_text()
    # The collector's historical controls are explicit here, independent of any
    # paper benchmark's environment, request generator or model-loading adapter.
    reference = True
    memory = plan.get("ere_mode") == "memory"
    if memory and "dmat-dual-clock" not in build.get("capabilities", []):
        raise ValueError("DMAT_RUNTIME_ABI_REQUIRED: 使用当前安装脚本构建双时钟 runtime 后再采集。")
    # Triton retains PTX for its bundled assembler. Use that same tool for
    # warp and memory probes; changing the .version would change evidence.
    assembler_path = []
    triton = importlib.util.find_spec("triton")
    if triton and triton.origin:
        bundled = Path(triton.origin).parent / "backends/nvidia/bin/ptxas"
        if bundled.is_file():
            assembler_path.append(str(bundled.parent))
    return {"NEUTRINO_REAL_DRIVER": str(driver), "NEUTRINO_DRIVER_NAME": "libcuda.so.1",
            "NEUTRINO_HOOK_DRIVER": str(root / "neutrino/build/libcuda.so.1"),
            "NEUTRINO_PROBING_PY": str(root / "neutrino/probe/cuda.py"),
            "NEUTRINO_PYTHON": sys.executable, "NEUTRINO_PROBES": probes,
            "NEUTRINO_TRACEDIR": str(trace), "NEUTRINO_CALLBACK": "/dev/null",
            "NEUTRINO_ENABLE": "1", "NEUTRINO_BENCHMARK": "0",
            "NEUTRINO_MEMUSAGE": "0", "NEUTRINO_ABLATION_MODE": "7", "NEUTRINO_ABLATION_STATS": "1",
            "NEUTRINO_ARENA": "1", "NEUTRINO_CAPTURE_CLEAR": "1", "NEUTRINO_ARENA_DEVICE_MIB": "32",
            "NEUTRINO_ARENA_HOST_MIB": "64", "NEUTRINO_ARENA_CHUNK_MIB": "16",
            "NEUTRINO_ARENA_INCREMENTAL": "1", "NEUTRINO_ARENA_BATCH_ENCODER": "1",
            "NEUTRINO_ARENA_DECOUPLE": "0", "NEUTRINO_ARENA_WRITER": "1", "NEUTRINO_ARENA_HOTPATH": "3",
            "NEUTRINO_ARENA_DIAG": "1", "NEUTRINO_ARENA_VERIFY": str(int(plan["ere_verify"])),
            "NEUTRINO_ARENA_CODEC_MIN_KIB": "16",
            "NEUTRINO_ARENA_DELTA": str(int(reference)), "NEUTRINO_ARENA_DELTA_WIDTH": "3",
            "NEUTRINO_ARENA_ELAPSED_WIDTH": "2", "NEUTRINO_ARENA_HIER": str(int(reference)),
            "NEUTRINO_ARENA_REFERENCE": str(int(reference)), "NEUTRINO_ARENA_LOCAL_BITS": "12",
            "NEUTRINO_ARENA_OMIT_ANCHOR": str(int(reference)), "NEUTRINO_CORE_MODE": "shift",
            "NEUTRINO_CAPTURE_ENCODER": str(root / "build/capture_pack.cubin"),
            "HQSB_ERE_KERNEL": plan["ere_kernel"], "HQSB_ERE_SKIP": str(plan["launch_skip"]),
            "HQSB_ERE_COUNT": str(plan["launch_count"]), "HQSB_ERE_TRITON_CACHE": str(triton_cache),
            **({"HQSB_ERE_CODEC_RAW": "1"} if plan["ere_codec"] == "raw" else {}),
            "HQSB_ERE_RUNTIME": str(root), "TRITON_LIBCUDA_PATH": str(root / "neutrino/build"),
            "PYTHONPATH": os.pathsep.join([str(Path(__file__).resolve().parents[2]), str(root), str(root / "python"), os.environ.get("PYTHONPATH", "")]),
            "PATH": os.pathsep.join([*assembler_path, str(Path(build["cuda_home"]) / "bin"), os.environ["PATH"]]),
            "LD_PRELOAD": str(root / "neutrino/build/preload.so"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            **({"HQSB_DMAT_CAPACITY": str(plan["memory_capacity"]), "HQSB_ERE_ARCH": build["arch"]} if memory else {})}
