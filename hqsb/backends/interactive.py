"""Real interactive providers. Heavy imports are confined to a worker process.

This diagnostic greedy runtime does not change the frozen C4 benchmark semantics.
It produces observed tokens, not simulated streaming of a finished answer.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

from hqsb.backends.tensor_memory import cache_inventory, parameter_inventory

from hqsb.backends.observation import ObservationRecorder, OperatorCapture


class InteractivePyTorch:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.source_identity = None
        self.load_inventory = None
        self.source_environment = None
        self.device_lease = None
        self.tokenizer_identity = None
        self.kernel_session = None
        self.chat_template_policy = None

    def load(self, *, kernel_profile=None) -> dict:
        if self.model is not None:
            raise RuntimeError("KERNEL_PROFILE_REQUIRES_UNLOAD")
        if kernel_profile is not None:
            from hqsb.core.kernel_profiles import profile

            profile(kernel_profile)
            self.config = {**self.config, "kernel_profile": kernel_profile}
        try:
            return self._load()
        except Exception as error:
            # A failed load still owns any partial model and device lease.
            try:
                self.close()
            except Exception as cleanup_error:
                # Preserve the primary failure and explicitly expose unknown cleanup.
                error.args = (*error.args, f"load_cleanup_unknown:{cleanup_error}")
            raise

    def _load(self) -> dict:
        import torch
        from hqsb.models.loader import load_causal_lm

        if self.config.get("device_lock_path"):
            from hqsb.observation.lease import DeviceLease

            self.device_lease = DeviceLease(self.config["device_lock_path"])
            self.device_lease.acquire()

        precision = self.config.get("precision", "float16")
        if precision not in {"float16", "bfloat16"}:
            raise ValueError("MODEL_PRECISION_UNSUPPORTED")
        if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("MODEL_BFLOAT16_DEVICE_UNSUPPORTED")
        if precision == "bfloat16" and self.config.get("kernel_profile", "framework-native") != "framework-native":
            raise ValueError("KERNEL_PRECISION_UNSUPPORTED")
        observation = ObservationRecorder()
        observation.snapshot("before_model_load", self.memory)
        with observation.phase("model_load"):
            self.tokenizer, self.model, seconds = load_causal_lm(
                self.config["model_path"],
                attention_backend=self.config["attention"],
                dtype={"float16": torch.float16, "bfloat16": torch.bfloat16}[precision],
                cpu_staging=True,
                verify_manifest=self.config.get("manifest"),
                allow_extra=tuple(self.config.get("manifest_allow_extra", [])),
            )
        observation.snapshot("after_model_load", self.memory)
        native_context = getattr(self.model.config, "max_position_embeddings", None)
        if type(native_context) is int and native_context > 0 and self.config["context_limit"] > native_context:
            raise ValueError(
                f"Deployment context {self.config['context_limit']} exceeds model config window {native_context}"
            )
        devices = sorted({str(p.device) for p in self.model.parameters()})
        if devices != ["cuda:0"]:
            self.close()
            raise RuntimeError(
                f"Reference requires full cuda:0 placement; observed {devices}"
            )
        path = Path(os.path.expandvars(self.config["model_path"])).expanduser()
        identity = {}
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            file = path / name
            if file.exists():
                identity[name] = hashlib.sha256(file.read_bytes()).hexdigest()
        if self.config.get("manifest"):
            manifest_file = Path(
                os.path.expandvars(self.config["manifest"])
            ).expanduser()
            self.source_identity = {
                "hash": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
                "scope": "verified_manifest_at_load",
                "revision": "local_manifest_verified",
            }
        else:
            self.source_identity = {
                "hash": "unverified-metadata:"
                + hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode()
                ).hexdigest(),
                "scope": "metadata_only_unverified_weights",
                "revision": "local_weights_not_manifest_verified",
            }
        from hqsb.backends.kernel_session import KernelSession

        self.kernel_session = KernelSession(self.model, self.config)
        self.chat_template_policy = {
            "default_input_mode": self.config.get("input_mode", "chat"),
            "template_sha256": hashlib.sha256(self.tokenizer.get_chat_template().encode()).hexdigest()
            if self.tokenizer.chat_template else None,
            "owner": "local_tokenizer",
            "model_type": self.model.config.model_type,
            "enable_thinking": False if self.model.config.model_type == "qwen3" else "model_template_default",
        }
        with torch.inference_mode(), observation.phase("warmup"):
            x = self.tokenizer("Warmup", return_tensors="pt").to("cuda")
            out = self.model(**x, use_cache=False)
            if not bool(torch.isfinite(out.logits[:, -1, :]).all().item()):
                raise ValueError("MODEL_WARMUP_NONFINITE_LOGITS")
            del out, x
            torch.cuda.synchronize()
        observation.snapshot("after_warmup", self.memory)
        from hqsb.observation.identity import environment

        self.source_environment = environment(Path(__file__).resolve().parents[2])
        from hqsb.core.optimization import fingerprint

        self.tokenizer_identity = fingerprint({key: value for key, value in identity.items() if key.startswith("tokenizer") or key == "chat_template.jinja"})
        self.load_inventory = parameter_inventory(self.model)
        from hqsb.backends.architecture_capture import module_inventory
        return {
            "load_seconds": seconds,
            "parameter_devices": devices,
            "parameter_dtypes": sorted({str(p.dtype).removeprefix("torch.") for p in self.model.parameters()}),
            "metadata_hashes": identity,
            "weight_verification": "manifest_verified"
            if self.config.get("manifest")
            else "not_verified",
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "model_type": self.model.config.model_type,
            "model_class": type(self.model).__name__,
            "native_context_limit": native_context if type(native_context) is int and native_context > 0 else None,
            "chat_template_policy": self.chat_template_policy,
            "memory": self.memory(),
            "parameter_bytes": sum(
                parameter.numel() * parameter.element_size()
                for parameter in self.model.parameters()
            ),
            "observation": observation.data,
            "parameter_inventory": self.load_inventory,
            "module_inventory": module_inventory(self.model),
            "source_environment": self.source_environment,
            "kernel_plan": self.kernel_session.plan,
            "kernel_bindings": self.kernel_session.binding,
        }

    def memory(self) -> dict:
        import torch

        if not torch.cuda.is_initialized():
            return {}
        return {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "scope": "worker_allocator",
        }

    def token_budget(self, request: dict, cancel):
        """CPU-only formatting using the resident tokenizer, serialized by owner."""
        from hqsb.backends.prompting import prepare_prompt
        from jinja2.exceptions import TemplateError
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        try:
            prompt, special = prepare_prompt(self.tokenizer, request, self.model.config.model_type)
            count = len(self.tokenizer.encode(prompt, add_special_tokens=special))
        except (ValueError, TemplateError) as exc:
            return {"request_error": str(exc)[:800]}
        return {"input_tokens": count, "context_limit": self.config["context_limit"],
                "max_output_tokens": max(0, min(self.config["max_output_tokens"], 4096, self.config["context_limit"] - count))}

    def generate(self, request: dict, cancel):
        if request.get("experiment_role") == "quality":
            from hqsb.backends.quality_evaluation import evaluate

            yield from evaluate(self, request, cancel)
            return
        import torch

        from hqsb.backends.optimization_receipt import resolve_request, receipt
        from hqsb.backends.prompting import ContextLimitExceeded
        from hqsb.observation.contracts import resolve_observation, digest

        request, optimization = resolve_request(self.config, request)
        if self.kernel_session:
            self.kernel_session.reset()
        plan = resolve_observation(request.get("observation_plan"), request.get("observation_mode", "basic"))

        if self.model is None:
            raise RuntimeError("Model is not loaded")
        start_ns = time.monotonic_ns()
        start = start_ns / 1_000_000_000
        cpu_start = time.process_time()
        deadline = start + request["remaining_ms"] / 1000
        limit = request["max_output_tokens"]
        session = None
        capture_errors = []
        mode = request.get("observation_mode", "basic")
        if mode != "off" and request.get("capture_dir"):
            from hqsb.backends.observation_session import CaptureSession

            try:
                session = CaptureSession(torch, request["capture_dir"], {
                    "run_id": request.get("capture_id"),
                    "host_time_origin_ns": str(start_ns),
                    "host_clock_domain": "host_monotonic",
                    "source": self.source_environment,
                    "model_identity": self.source_identity,
                    "input_mode": request.get("input_mode", "chat"),
                    "attention_requested": self.config["attention"],
                    "dtype": self.config.get("precision", "float16"), "cleanup_policy": request.get("cleanup_policy", "empty_cache"),
                    "optimization_plan": optimization,
                    "experiment_links": request.get("experiment_links", {}),
                }, mode, request.get("observation_plan"))
                session.start(self.model, self.load_inventory)
            except (OSError, ValueError, RuntimeError) as exc:
                capture_errors.append(f"capture_start:{type(exc).__name__}:{exc}"[:400])
                if session:
                    session.problems.extend(capture_errors)
                    session.finish("not_started", "succeeded")
                    session = None
        observation = ObservationRecorder(
            mode,
            start=start,
            sink=session.emit if session else None,
            execution={
                "input_mode": request.get("input_mode", "chat"),
                "provider": "pytorch",
                "device": "cuda:0",
                "dtype": self.config.get("precision", "float16"),
                "quant_execution_path": "bf16_reference" if self.config.get("precision") == "bfloat16" else "fp16_reference",
                "attention_backend": self.kernel_session.profile["operators"].get("attention", self.config["attention"])
                if self.kernel_session else self.config["attention"],
                "attention_config": self.config["attention"],
                "kernel_profile": self.kernel_session.profile["profile_id"] if self.kernel_session else "framework-native",
                "decoding": "greedy",
                "use_cache": True,
                "gpu_duration_source": "profiler_only",
                "sampling_policy": "all_output_tokens_host_timing",
            },
        )
        capture = OperatorCapture(
            torch,
            enabled=plan.collects("operators"),
            capture_dir=request.get("capture_dir"),
            capture_id=request.get("capture_id"),
            max_decode_steps=plan.max_decode_steps,
        )
        observation.data["profile"] = capture.summary
        ids, times, text = [], [], ""
        cache = current = attention = inputs = out = None
        torch.cuda.reset_peak_memory_stats()
        observation.snapshot("before_request", self.memory)
        finish = "length"
        execution_status, cleanup_status = "failed", "unknown"
        result = None
        input_token_sha256 = None
        architecture = None
        if request.get("capture_architecture"):
            from hqsb.backends.architecture_capture import ArchitectureCapture
            architecture = ArchitectureCapture(self.model)
        try:
            with observation.phase("chat_template"):
                from hqsb.backends.prompting import prepare_prompt

                prompt, add_special_tokens = prepare_prompt(self.tokenizer, request, self.model.config.model_type)
            with observation.phase("tokenize"):
                if "experiment_input_tokens" in request:
                    count = request["experiment_input_tokens"]
                    if type(count) is not int or not 2 <= count <= 512:
                        raise ValueError("EXPERIMENT_INPUT_TOKEN_BOUND")
                    pattern = self.tokenizer.encode("请解释模型推理和内存缓存。", add_special_tokens=False)
                    request = {**request, "input_token_ids": (pattern * (count // len(pattern) + 1))[:count]}
                if "input_token_ids" in request:
                    supplied = request["input_token_ids"]
                    if not supplied or not all(type(x) is int and 0 <= x < self.model.config.vocab_size for x in supplied):
                        raise ValueError("EXPLICIT_TOKEN_IDS_INVALID")
                    inputs = {"input_ids": torch.tensor([supplied], dtype=torch.long),
                              "attention_mask": torch.ones((1, len(supplied)), dtype=torch.long)}
                else:
                    inputs = self.tokenizer(
                        prompt, return_tensors="pt", add_special_tokens=add_special_tokens
                    )
            input_count = int(inputs["input_ids"].shape[1])
            if session:
                from hqsb.observation.contracts import digest

                session.writer.context["workload_hash"] = digest({
                    "input_token_ids": inputs["input_ids"].tolist(), "output_tokens": limit,
                    "fixed_output_length": bool(request.get("fixed_output_length", False)),
                    "decoding": "greedy"})
                session.emit("workload", {"input_token_ids": inputs["input_ids"].tolist()
                             if request.get("save_input") or "input_token_ids" in request else None,
                             "input_storage_policy": "saved" if request.get("save_input") or "input_token_ids" in request else "hash_only",
                             "requested_output_tokens": limit, "decoding": "greedy",
                             "input_mode": request.get("input_mode", "chat"),
                             "fixed_output_length": bool(request.get("fixed_output_length", False))})
            input_token_sha256 = digest(inputs["input_ids"].tolist()[0])
            if input_count + limit > self.config["context_limit"]:
                raise ContextLimitExceeded(input_count, limit, self.config["context_limit"])
            with observation.phase("input_transfer"):
                inputs = {key: value.to("cuda") for key, value in inputs.items()}
            current = inputs["input_ids"]
            attention = inputs["attention_mask"]
            eos = self.model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            runtime_start = time.monotonic()
            with torch.inference_mode():
                for index in range(limit):
                    if cancel.is_set():
                        finish = "cancelled"
                        break
                    if time.monotonic() >= deadline:
                        finish = "timed_out"
                        break
                    if index == 0:
                        with observation.phase("profiler_start"):
                            capture.start()
                            capture.attach_modules(self.model, scope="all")
                    phase = "prefill" if index == 0 else "decode"
                    if self.kernel_session:
                        self.kernel_session.set_phase(phase)
                    token_start = observation.now_ms()
                    with (
                        observation.phase(phase),
                        capture.region(f"hqsb.{phase}.token_{index + 1}"),
                        torch.cuda.nvtx.range(f"hqsb.{phase}") if request.get("tool_ranges") else nullcontext(),
                    ):
                        model_start = time.monotonic()
                        with architecture.forward(phase) if architecture and index < 2 else nullcontext():
                            out = self.model(
                                input_ids=current,
                                attention_mask=attention,
                                past_key_values=cache,
                                use_cache=True,
                            )
                        model_end = time.monotonic()
                        # This existing .item() waits for selection; host timings
                        # intentionally expose the wait instead of calling it GPU time.
                        token = int(out.logits[:, -1, :].argmax(-1).item())
                        selection_end = time.monotonic()
                        cache = out.past_key_values
                        out = None
                        ids.append(token)
                        times.append((selection_end - runtime_start) * 1000)
                        text = self.tokenizer.decode(
                            ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ).rstrip("\ufffd")
                        decode_end = time.monotonic()
                    observation.token(
                        index=index + 1,
                        phase=phase,
                        start_ms=token_start,
                        duration_ms=observation.now_ms() - token_start,
                        model_host_ms=(model_end - model_start) * 1000,
                        selection_ms=(selection_end - model_end) * 1000,
                        detokenize_ms=(decode_end - selection_end) * 1000,
                    )
                    with observation.phase("profiler_step_finalize"):
                        capture.step_completed()
                    if index == 0:
                        observation.snapshot("after_prefill", self.memory)
                    if session and (index == 0 or ((plan.collects("allocator") or plan.collects("pages")) and index <= plan.max_decode_steps)):
                        session.boundary("after_prefill" if index == 0 else "decode_step",
                                         cache=cache, step=index, detailed=index == 0)
                    stream_start = observation.now_ms()
                    yield {
                        "kind": "output",
                        "text": text,
                        "token_id": token,
                        "output_tokens": len(ids),
                        "input_tokens": input_count,
                        "runtime_first_token_ms": times[0],
                    }
                    if observation.mode != "off":
                        observation.data["phases"].append(
                            {
                                "name": "stream_consumer_wait",
                                "start_ms": stream_start,
                                "duration_ms": observation.now_ms() - stream_start,
                                "source": "host_monotonic",
                            }
                        )
                    if token in eos_ids and not request.get("fixed_output_length", False):
                        finish = "stop"
                        break
                    with observation.phase("prepare_decode"):
                        current = torch.tensor(
                            [[token]], device="cuda", dtype=torch.long
                        )
                        attention = torch.cat(
                            (attention, attention.new_ones((1, 1))), dim=1
                        )
            elapsed = (time.monotonic() - runtime_start) * 1000
            execution_status = finish if finish in {"cancelled", "timed_out"} else "completed"
            memory = self.memory()
            observation.snapshot("after_generation", self.memory)
            if session:
                session.writer.context["output_token_hash"] = digest(ids)
                session.boundary("after_generation", cache=cache, step=len(ids) - 1, detailed=True)
                session.allocator_snapshot("before_cleanup")
            result = {
                "kind": "result",
                "text": text,
                "finish_reason": finish,
                "metrics": {
                    "input_mode": request.get("input_mode", "chat"),
                    "input_policy": {"mode": request.get("input_mode", "chat"),
                        "add_special_tokens": add_special_tokens,
                        "chat_template_sha256": self.chat_template_policy.get("template_sha256")
                            if request.get("input_mode", "chat") == "chat" else None},
                    "runtime_first_token_ms": times[0] if times else None,
                    "runtime_e2e_ms": elapsed,
                    "input_tokens": input_count,
                    "output_tokens": len(ids),
                    "output_tokens_per_s": len(ids) * 1000 / elapsed
                    if elapsed
                    else None,
                    "decode_tail_tokens_per_s": (len(ids) - 1)
                    * 1000
                    / (times[-1] - times[0])
                    if len(ids) > 1 and times[-1] > times[0]
                    else None,
                    "token_itl_ms": [b - a for a, b in zip(times, times[1:])],
                    "generated_token_ids": ids,
                    "input_token_sha256": input_token_sha256,
                    "chat_template_policy": self.chat_template_policy,
                    "memory": memory,
                    "measurement_profile": "interactive-greedy-host-monotonic-v1",
                    "quality": "not_evaluated",
                    "observation": observation.data,
                    "kv_inventory": cache_inventory(cache),
                },
            }
        except ContextLimitExceeded as exc:
            result = {"kind": "result", "finish_reason": "rejected", "text": "",
                      "request_error": {"code": "CONTEXT_LIMIT", "message": str(exc)},
                      "metrics": {**exc.budget, "output_tokens": 0, "input_mode": request.get("input_mode", "chat"),
                                  "input_token_sha256": input_token_sha256, "observation": observation.data}}
        finally:
            with observation.phase("profiler_finalize"):
                capture.finish(total_tokens=len(ids))
                capture.save_summary()
                if session and plan.collects("operators"):
                    session.emit("framework_profile", capture.summary)
                    if capture.summary["status"] != "complete":
                        session.problems.append("FRAMEWORK_PROFILE_PARTIAL_OR_UNAVAILABLE")
            try:
                with observation.phase("request_cleanup"):
                    del out, cache, current, attention, inputs
                    gc.collect()
                    if request.get("cleanup_policy", "empty_cache") == "empty_cache":
                        torch.cuda.empty_cache()
                    cleanup_status = "succeeded"
            finally:
                observation.snapshot("after_cleanup", self.memory)
                execution_receipt = receipt(self, request, optimization,
                    role="diagnostic" if mode != "off" or architecture else "performance",
                    output_hash=digest(ids), status=execution_status, cleanup=cleanup_status,
                    proof={"generated_tokens": len(ids), "requested_token_limit": limit,
                           "output_tokens": len(ids),
                           "input_token_sha256": input_token_sha256 if result is not None else None,
                           "runtime_e2e_ms": result["metrics"].get("runtime_e2e_ms") if result is not None else None,
                           "cleanup_call": "torch.cuda.empty_cache" if request["cleanup_policy"] == "empty_cache" else "empty_cache_not_called",
                           "after_cleanup_allocator": self.memory()})
                if result is not None:
                    result["metrics"]["optimization_receipt"] = execution_receipt
                if session:
                    try:
                        session.boundary("after_cleanup", detailed=True)
                        session.emit("optimization_receipt", execution_receipt)
                        report = session.finish(execution_status, cleanup_status)
                        if result is not None:
                            result["metrics"]["observation_bundle"] = {
                                "schema_version": "1.0.0", "run_id": request.get("capture_id"),
                                "capture_status": report["capture_status"],
                                "counts": report["counts"], "publication": report["publication"],
                                "problems": session.problems,
                            }
                    except (OSError, ValueError, RuntimeError) as exc:
                        capture_errors.append(f"capture_finish:{type(exc).__name__}:{exc}"[:400])
                observation.data["limitations"].extend(capture_errors)
                if architecture and request.get("capture_dir"):
                    import json
                    from hqsb.backends.architecture_capture import module_inventory
                    from hqsb.observation.storage import atomic_json, sha256
                    document = {**architecture.document(), "run_id": request.get("capture_id"),
                                "epoch": request.get("expected_epoch"), "input_token_sha256": input_token_sha256,
                                "model_identity": self.source_identity, "tokenizer_identity": self.tokenizer_identity,
                                "source_digest": self.source_environment["source_digest"],
                                "modules": module_inventory(self.model), "execution_status": execution_status}
                    while len(json.dumps(document).encode()) > 3 * 1024**2:
                        document.update(status="partial", truncated=True)
                        document["warnings"] = sorted(set(document["warnings"]) | {"ARTIFACT_BYTE_BUDGET"})
                        if len(document["modules"]["modules"]) > 1:
                            document["modules"]["modules"] = document["modules"]["modules"][:len(document["modules"]["modules"])//2]
                            document["modules"]["status"] = "partial"
                        elif len(document["nodes"]) > 1:
                            document["nodes"] = document["nodes"][:len(document["nodes"])//2]
                            kept = {node["id"] for node in document["nodes"]}
                            document["edges"] = [edge for edge in document["edges"] if edge["source"] in kept and edge["target"] in kept]
                        else:
                            raise ValueError("ARCHITECTURE_MINIMUM_ARTIFACT_EXCEEDS_BUDGET")
                    target = Path(request["capture_dir"]) / "architecture-capture.json"
                    atomic_json(target, document)
                    if result is not None:
                        result["metrics"]["architecture_capture"] = {"status": document["status"], "nodes": len(document["nodes"]),
                            "edges": len(document["edges"]), "phases": document["phases"], "sha256": sha256(target),
                            "scope": "diagnostic_executed_forward_paths_not_performance_sample"}
        if result is not None:
            result["metrics"]["worker_request_total_ms"] = (time.monotonic() - start) * 1000
            result["metrics"]["worker_request_cpu_ms"] = (time.process_time() - cpu_start) * 1000
        yield result

    def quantize(self, request, cancel):
        """Create a storage artifact; leave resident FP16 model weights intact."""
        from hqsb.backends.quantization import quantize_loaded_model

        if self.model is None or self.source_identity is None:
            raise RuntimeError("Model is not loaded")
        try:
            yield from quantize_loaded_model(
                self.model, request, cancel, self.source_identity
            )
        finally:
            gc.collect()
            import torch

            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()

    def close(self):
        import torch

        if self.kernel_session:
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            self.kernel_session.close()
            self.kernel_session = None
        self.model = None
        self.tokenizer = None
        self.source_identity = None
        self.load_inventory = None
        self.source_environment = None
        gc.collect()
        import torch

        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if self.device_lease:
            self.device_lease.release()
            self.device_lease = None


class OpenAIProvider:
    """Connect only to an administrator-configured OpenAI-compatible server."""

    def __init__(self, config: dict):
        self.config = config

    def headers(self):
        env = self.config.get("api_key_env")
        return (
            {"Authorization": "Bearer " + os.environ[env]}
            if env and os.environ.get(env)
            else {}
        )

    def load(self):
        import httpx

        response = httpx.get(
            self.config["base_url"].rstrip("/") + "/models",
            headers=self.headers(),
            timeout=15,
        )
        response.raise_for_status()
        names = {m["id"] for m in response.json().get("data", [])}
        if self.config["model"] not in names:
            raise ValueError("Configured model not advertised by remote server")
        return {
            "weight_verification": "remote_vendor_opaque",
            "model_advertised": True,
            "device_name": self.config["platform"],
            "memory": {},
        }

    def generate(self, request, cancel):
        import httpx

        start = time.monotonic()
        text, usage, finish, first = "", {}, "unknown", None
        payload = {
            "model": self.config["model"],
            "messages": request["messages"],
            "max_tokens": request["max_output_tokens"],
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        timeout = httpx.Timeout(
            connect=10, read=min(15, request["remaining_ms"] / 1000), write=10, pool=10
        )
        with httpx.stream(
            "POST",
            self.config["base_url"].rstrip("/") + "/chat/completions",
            json=payload,
            headers=self.headers(),
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            done = False
            for line in response.iter_lines():
                if cancel.is_set():
                    finish = "cancelled"
                    break
                if (time.monotonic() - start) * 1000 > request["remaining_ms"]:
                    finish = "timed_out"
                    break
                if not line.startswith("data:"):
                    continue
                content = line[5:].strip()
                if content == "[DONE]":
                    done = True
                    break
                data = json.loads(content)
                if data.get("error"):
                    raise RuntimeError("Upstream stream reported an error")
                if data.get("usage"):
                    usage = data["usage"]
                choices = data.get("choices", [])
                if choices:
                    part = choices[0].get("delta", {}).get("content") or ""
                    if part:
                        text += part
                        first = (
                            first
                            if first is not None
                            else (time.monotonic() - start) * 1000
                        )
                        yield {"kind": "output", "text": text, "output_tokens": None}
                    finish = choices[0].get("finish_reason") or finish
            if not done and finish not in {"cancelled", "timed_out", "stop", "length"}:
                raise RuntimeError("Upstream stream ended without a terminal record")
        yield {
            "kind": "result",
            "text": text,
            "finish_reason": finish,
            "metrics": {
                "proxy_first_content_ms": first,
                "runtime_first_token_ms": None,
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "memory": {},
                "measurement_profile": "openai-proxy-content-v1",
                "quality": "not_evaluated",
                "remote_cleanup": "not_observable",
            },
        }

    def close(self):
        pass
