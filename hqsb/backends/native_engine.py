"""Owned upstream servers: lifecycle, streaming transport and execution evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

from hqsb.core.native_engine import VERSIONS, launch_arguments, resolve_options
from hqsb.core.optimization import fingerprint


class NativeEngineProvider:
    def __init__(self, config):
        self.config = config
        self.process = self.log = self.lease = None
        self.base_url = None
        self.receipt = None

    def snapshot(self):
        """Raw engine counters keep their native names and instance scope."""
        import httpx
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("NATIVE_ENGINE_PROCESS_EXITED")
        response = httpx.get(self.base_url + "/metrics", timeout=5, trust_env=False)
        response.raise_for_status()
        import psutil
        from hqsb.observation.procfs import process_key, sample_process
        processes = []
        for proc in psutil.Process(self.process.pid).children(recursive=True):
            try:
                processes.append(sample_process(process_key(proc.pid)))
            except (OSError, ValueError, psutil.NoSuchProcess) as exc:
                processes.append({"pid": proc.pid, "availability": "not_collected", "reason": str(exc)})
        return {"sampled_at": time.time(), "scope": "engine_instance_all_requests",
                "prometheus_text": response.text, "instance_id": self.receipt["instance_id"],
                "processes": processes, "process_scope": "upstream_server_and_descendants"}

    def load(self, *, engine_options=None, **selection):
        import httpx
        from hqsb.models.manifest import verify_or_raise
        from hqsb.observation.lease import DeviceLease
        if self.process:
            raise ValueError("NATIVE_ENGINE_REQUIRES_UNLOAD")
        if any(v is not None for v in selection.values()):
            raise ValueError("NATIVE_ENGINE_USE_ENGINE_OPTIONS")
        engine = self.config["provider"]
        options = resolve_options(engine, engine_options if engine_options is not None else self.config.get("engine_options", {}))
        model_path = Path(self.config["model_path"]).expanduser().resolve(strict=True)
        manifest = self.config.get("manifest")
        if not manifest:
            raise ValueError("NATIVE_BASELINE_REQUIRES_MODEL_MANIFEST")
        verification = verify_or_raise(str(model_path), str(Path(manifest).expanduser()),
                                       allow_extra=self.config.get("manifest_allow_extra", []))
        metadata_path = model_path / "config.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.stat().st_size <= 1024**2 else {}
        artifact_metadata = {key: metadata[key] for key in ("model_type", "architectures", "dtype", "torch_dtype", "quantization_config") if key in metadata}
        python = str(Path(self.config["engine_python"]).expanduser().absolute())
        probe = subprocess.run([python, "-c", "import json,importlib.metadata as m; print(json.dumps({k:m.version(k) for k in ['" + engine + "','torch','transformers']}))"],
                               capture_output=True, text=True, timeout=30, check=True)
        versions = json.loads(probe.stdout)
        if versions[engine] != VERSIONS[engine]:
            raise ValueError(f"NATIVE_ENGINE_VERSION_MISMATCH:{versions[engine]} expected {VERSIONS[engine]}")
        self.lease = DeviceLease(self.config["device_lock_path"])
        self.lease.acquire()
        try:
            try:
                hardware = {"availability": "measured", "scope": "host_visible_gpus",
                            "csv": subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,power.limit", "--format=csv,noheader"],
                                                   capture_output=True, text=True, timeout=10, check=True).stdout.strip()}
            except (OSError, subprocess.SubprocessError) as exc:
                hardware = {"availability": "not_collected", "reason": str(exc)[:300]}
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            identity = uuid.uuid4().hex
            directory = Path(self.config["engine_log_root"]) / identity
            directory.mkdir(parents=True, exist_ok=False)
            cache = Path(self.config["engine_cache_root"])
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "tmp").mkdir(exist_ok=True)
            self.log = (directory / "server.log").open("wb")
            argv = [python, *launch_arguments(engine, {**self.config, "model_path": str(model_path)}, options, port)]
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
                   "DO_NOT_TRACK": "1", "TOKENIZERS_PARALLELISM": "false"}
            env.update(TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"),
                       FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"), VLLM_CACHE_ROOT=str(cache / "vllm"),
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TORCHINDUCTOR_COMPILE_THREADS="2", TMPDIR=str(cache / "tmp"))
            if engine == "vllm" and options["attention_backend"] != "auto":
                env["VLLM_ATTENTION_BACKEND"] = options["attention_backend"]
            else:
                env.pop("VLLM_ATTENTION_BACKEND", None)
            # Diagnostic-only launch configuration never enters saved engine options.
            env.pop("VLLM_TORCH_PROFILER_DIR", None)
            if self.config.get("_native_profile_control"):
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
                if engine == "vllm":
                    env["VLLM_TORCH_PROFILER_DIR"] = str(directory / "cuda-profile-control")
                    argv.extend(["--worker-cls", "hqsb.observation.vllm_profile_worker.ProfileWorker"])
            if self.config.get("_eretrace"):
                from hqsb.observation.eretrace_tools import launch_environment
                ere = self.config["_eretrace"]
                env.update(launch_environment(ere["runtime"], ere["directory"], ere["plan"], cache / "triton"))
                if engine == "sglang":
                    argv[2] = "hqsb.observation.sglang_profile_server"
            self.process = subprocess.Popen([sys.executable, "-m", "hqsb.backends.native_supervisor", *argv],
                                            stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT,
                                            env=env, start_new_session=True, pass_fds=(self.lease.fd,))
            self.base_url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 250
            with httpx.Client(timeout=2, trust_env=False) as client:
                while True:
                    if self.process.poll() is not None:
                        raise RuntimeError("NATIVE_ENGINE_START_FAILED:" + str(directory / "server.log") + ":" + (directory / "server.log").read_text(errors="replace")[-500:])
                    try:
                        response = client.get(self.base_url + "/v1/models")
                        if response.status_code == 200 and self.config["model"] in {x["id"] for x in response.json()["data"]}:
                            break
                    except (httpx.TransportError, KeyError, ValueError):
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError("NATIVE_ENGINE_START_TIMEOUT:" + str(directory / "server.log"))
                    time.sleep(0.25)
                effective = None
                if engine == "sglang":
                    response = client.get(self.base_url + "/get_server_info")
                    response.raise_for_status()
                    effective = response.json()
                response = client.get(self.base_url + "/metrics")
                response.raise_for_status()
                load_metrics = response.text
            with (directory / "server.log").open("rb") as stream:
                stream.seek(max(0, (directory / "server.log").stat().st_size - 512 * 1024))
                lines = stream.read().decode(errors="replace").splitlines()
            dispatch_logs = [line for line in lines if any(key in line.lower() for key in
                             ("attention backend", "attention_backend", "kv cache", "kv_cache", "cuda graph", "cudagraph"))][-48:]
            self.receipt = {"schema_version": "1.0.0", "engine": engine, "versions": versions,
                            "instance_id": identity, "supervisor_pid": self.process.pid,
                            "hardware_inventory": hardware,
                            "requested": options, "plan_sha256": fingerprint(options),
                            "launch_argv": argv, "launch_environment": {k: env.get(k) for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES", "VLLM_ATTENTION_BACKEND", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS"]},
                            "model_manifest_sha256": hashlib.sha256(Path(manifest).expanduser().read_bytes()).hexdigest(),
                            "verified_files": verification.as_dict(),
                            "artifact_metadata": artifact_metadata,
                            "effective_engine_config": effective, "evidence_level": "server_reported_configuration" if effective else "owned_process_arguments_and_ready_probe",
                            "kernel_dispatch_evidence": "upstream_configuration_and_logs",
                            "selected_backend_log_lines": dispatch_logs, "load_prometheus_text": load_metrics,
                            "server_log": str(directory / "server.log")}
            if self.config.get("_native_profile_control"):
                self.receipt["diagnostic_instrumentation"] = {
                    "control": "cudaProfilerStart_Stop_in_upstream_gpu_worker",
                    "worker_adapter": "hqsb.observation.vllm_profile_worker.ProfileWorker" if engine == "vllm" else "upstream_CUDA_PROFILER_activity",
                    "scope": "steady_state_requests_after_load_and_explicit_warmup"}
                if self.config.get("_eretrace"):
                    self.receipt["diagnostic_instrumentation"].update(tool="eretrace", scope="selected_ordinary_launches_one_producer_one_context",
                        plan=self.config["_eretrace"]["plan"])
                    if engine == "sglang":
                        self.receipt["diagnostic_instrumentation"]["worker_adapter"] = "hqsb.observation.sglang_profile_server"
            source_root = Path(__file__).resolve().parents[2]
            self.receipt["hqsb_source_sha256"] = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                for name in ("hqsb/backends/native_engine.py", "hqsb/backends/native_supervisor.py",
                             "hqsb/core/native_engine.py", "hqsb/serving/native_benchmark.py")}
            if self.config.get("_native_profile_control"):
                for name in ("hqsb/observation/vllm_profile_worker.py", "hqsb/observation/diagnostic_target.py"):
                    self.receipt["hqsb_source_sha256"][name] = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                if self.config.get("_eretrace"):
                    for name in ("hqsb/observation/eretrace_tools.py", "hqsb/observation/sglang_profile_server.py"):
                        self.receipt["hqsb_source_sha256"][name] = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            freeze = Path(python).parent.parent / "environment.freeze.txt"
            self.receipt["environment_freeze_sha256"] = hashlib.sha256(freeze.read_bytes()).hexdigest() if freeze.is_file() else None
            (directory / "receipt.json").write_text(json.dumps(self.receipt, ensure_ascii=False, indent=2))
            return {"weight_verification": "manifest_verified", "native_engine_receipt": self.receipt,
                    "device_name": self.config["platform"], "memory": {}, "engine_options": options}
        except BaseException:
            self.close()
            raise

    def stream(self, request, cancel, *, client=None):
        """Measure HTTP first-content/chunk times; usage supplies actual tokens."""
        import httpx
        from contextlib import nullcontext
        start = time.monotonic()
        completion = request.get("prompt") is not None
        body = {"model": self.config["model"], "max_tokens": request["max_output_tokens"],
                "temperature": 0, "top_p": 1, "seed": self.receipt["requested"]["seed"],
                "stream": True, "stream_options": {"include_usage": True}}
        body.update({"prompt": request["prompt"]} if completion else {"messages": request["messages"]})
        if not completion:
            body["chat_template_kwargs"] = {"enable_thinking": self.receipt["requested"].get("enable_thinking", False)}
        if request.get("fixed_output_length"):
            body["ignore_eos"] = True
        text, usage, finish, first, chunks = "", {}, None, None, []
        deadline = request.get("remaining_ms", 120000) / 1000
        request_id = "hqsb-" + uuid.uuid4().hex
        with (nullcontext(client) if client is not None else httpx.Client(timeout=httpx.Timeout(connect=5, read=deadline, write=5, pool=5), trust_env=False)) as client:
            with client.stream("POST", self.base_url + ("/v1/completions" if completion else "/v1/chat/completions"), json=body, headers={"X-Request-Id": request_id}, timeout=httpx.Timeout(connect=5, read=deadline, write=5, pool=5)) as response:
                if response.is_error:
                    response.read()
                response.raise_for_status()
                terminal = False
                for line in response.iter_lines():
                    if cancel.is_set() or time.monotonic() - start > deadline:
                        finish = "cancelled" if cancel.is_set() else "timed_out"
                        break
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        terminal = True
                        break
                    data = json.loads(data)
                    if data.get("error"):
                        raise RuntimeError(str(data["error"]))
                    usage = data.get("usage") or usage
                    for choice in data.get("choices", []):
                        part = choice.get("text") if completion else choice.get("delta", {}).get("content")
                        # Keep reasoning text visible and count it in first-content timing.
                        part = part or choice.get("delta", {}).get("reasoning_content") or ""
                        if part:
                            now = (time.monotonic() - start) * 1000
                            first = now if first is None else first
                            chunks.append(now)
                            text += part
                            yield {"kind": "output", "text": text}
                        finish = choice.get("finish_reason") or finish
                if not terminal and finish not in {"cancelled", "timed_out"}:
                    raise RuntimeError("NATIVE_STREAM_TERMINAL_MISSING")
        elapsed = (time.monotonic() - start) * 1000
        n = usage.get("completion_tokens")
        yield {"kind": "result", "text": text, "finish_reason": finish,
               "metrics": {"proxy_first_content_ms": first, "runtime_first_token_ms": None,
                           "runtime_e2e_ms": elapsed, "input_tokens": usage.get("prompt_tokens"), "output_tokens": n,
                           "output_tokens_per_s": n * 1000 / elapsed if n is not None and elapsed else None,
                           "http_tpot_ms": (chunks[-1] - first) / (n - 1) if n and n > 1 and len(chunks) > 1 else None,
                           "content_chunk_timestamps_ms": chunks, "measurement_profile": "native-http-content-v1",
                           "client_request_id": request_id, "token_timing_scope": "client_content_chunks", "quality": "not_evaluated"}}

    def generate(self, request, cancel):
        import httpx
        observed = request.get("observation_mode", "basic") != "off"
        before = self.snapshot() if observed else None
        try:
            for event in self.stream(request, cancel):
                if event["kind"] == "result":
                    event["metrics"].update(native_engine_receipt=self.receipt,
                                             native_counter_collection="basic" if observed else "off")
                    if observed:
                        event["metrics"]["native_counters"] = {"before": before, "after": self.snapshot()}
                yield event
        except httpx.HTTPStatusError as exc:
            if not 400 <= exc.response.status_code < 500:
                raise
            yield {"kind": "result", "finish_reason": "rejected",
                   "request_error": {"code": "NATIVE_REQUEST_REJECTED", "message": exc.response.text[:1200]},
                   "metrics": {"native_engine_receipt": self.receipt}}

    def benchmark(self, request, cancel):
        from hqsb.serving.native_benchmark import execute
        return execute(self, request, cancel)

    def close(self):
        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait(timeout=10)
            self.process = None
        if self.log:
            self.log.close()
            self.log = None
        if self.lease:
            self.lease.release()
            self.lease = None
