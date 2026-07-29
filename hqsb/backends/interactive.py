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
from pathlib import Path

from hqsb.backends.tensor_memory import cache_inventory, parameter_inventory

from hqsb.backends.observation import ObservationRecorder, OperatorCapture


class InteractivePyTorch:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.source_identity = None

    def load(self) -> dict:
        import torch
        from hqsb.models.loader import load_qwen3

        observation = ObservationRecorder()
        observation.snapshot("before_model_load", self.memory)
        with observation.phase("model_load"):
            self.tokenizer, self.model, seconds = load_qwen3(
                self.config["model_path"],
                attention_backend=self.config["attention"],
                dtype=torch.float16,
                cpu_staging=True,
                verify_manifest=self.config.get("manifest"),
                allow_extra=tuple(self.config.get("manifest_allow_extra", [])),
            )
        observation.snapshot("after_model_load", self.memory)
        devices = sorted({str(p.device) for p in self.model.parameters()})
        if devices != ["cuda:0"]:
            self.close()
            raise RuntimeError(
                f"Reference requires full cuda:0 placement; observed {devices}"
            )
        path = Path(self.config["model_path"]).expanduser()
        identity = {}
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
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
        with torch.inference_mode(), observation.phase("warmup"):
            x = self.tokenizer("Warmup", return_tensors="pt").to("cuda")
            out = self.model(**x, use_cache=False)
            del out, x
            torch.cuda.synchronize()
        observation.snapshot("after_warmup", self.memory)
        return {
            "load_seconds": seconds,
            "parameter_devices": devices,
            "metadata_hashes": identity,
            "weight_verification": "manifest_verified"
            if self.config.get("manifest")
            else "not_verified",
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "memory": self.memory(),
            "parameter_bytes": sum(
                parameter.numel() * parameter.element_size()
                for parameter in self.model.parameters()
            ),
            "observation": observation.data,
            "parameter_inventory": parameter_inventory(self.model),
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

    def generate(self, request: dict, cancel):
        import torch

        if self.model is None:
            raise RuntimeError("Model is not loaded")
        start = time.monotonic()
        deadline = start + request["remaining_ms"] / 1000
        limit = request["max_output_tokens"]
        observation = ObservationRecorder(
            request.get("observation_mode", "basic"),
            start=start,
            execution={
                "provider": "pytorch",
                "device": "cuda:0",
                "dtype": "float16",
                "quant_execution_path": "fp16_reference",
                "attention_backend": self.config["attention"],
                "decoding": "greedy",
                "use_cache": True,
                "gpu_duration_source": "profiler_only",
                "sampling_policy": "all_output_tokens_host_timing",
            },
        )
        capture = OperatorCapture(
            torch,
            enabled=observation.mode == "operators",
            capture_dir=request.get("capture_dir"),
            capture_id=request.get("capture_id"),
        )
        observation.data["profile"] = capture.summary
        ids, times, text = [], [], ""
        cache = current = attention = inputs = out = None
        torch.cuda.reset_peak_memory_stats()
        observation.snapshot("before_request", self.memory)
        finish = "length"
        try:
            with observation.phase("chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    request["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            with observation.phase("tokenize"):
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=False
                )
            input_count = int(inputs["input_ids"].shape[1])
            if input_count + limit > self.config["context_limit"]:
                raise ValueError(
                    f"Context limit exceeded: {input_count}+{limit}>{self.config['context_limit']}"
                )
            with observation.phase("input_transfer"):
                inputs = inputs.to("cuda")
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
                            capture.attach_modules(self.model)
                    phase = "prefill" if index == 0 else "decode"
                    token_start = observation.now_ms()
                    with (
                        observation.phase(phase),
                        capture.region(f"hqsb.{phase}.token_{index + 1}"),
                    ):
                        model_start = time.monotonic()
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
                    if token in eos_ids:
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
            memory = self.memory()
            observation.snapshot("after_generation", self.memory)
            result = {
                "kind": "result",
                "text": text,
                "finish_reason": finish,
                "metrics": {
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
                    "memory": memory,
                    "measurement_profile": "interactive-greedy-host-monotonic-v1",
                    "quality": "not_evaluated",
                    "observation": observation.data,
                    "kv_inventory": cache_inventory(cache),
                },
            }
        finally:
            with observation.phase("profiler_finalize"):
                capture.finish(total_tokens=len(ids))
                capture.save_summary()
            with observation.phase("request_cleanup"):
                del out, cache, current, attention, inputs
                gc.collect()
                torch.cuda.empty_cache()
            observation.snapshot("after_cleanup", self.memory)
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
        self.model = None
        self.tokenizer = None
        self.source_identity = None
        gc.collect()
        import torch

        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


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
