"""Isolated official LLM Compressor PTQ -> HQSB native packed model bundle.

Invoked as a child process, not imported by the serving environment. The device
lease belongs to the parent loading worker until this process has terminated.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path


def main(request_file):
    import ctypes

    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() == 1:
        raise RuntimeError("L4_CALIBRATION_PARENT_EXITED")
    import torch
    import importlib.metadata as metadata
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    from compressed_tensors.quantization.lifecycle.forward import quantize
    from hqsb.backends.precision_artifacts import export_native
    from hqsb.backends.precision_linear import encode_codes
    from hqsb.core.precision import resolve_exclusions

    request = json.loads(Path(request_file).read_text())
    plan, method = request["plan"], request["plan"]["method"]
    tokenizer = AutoTokenizer.from_pretrained(
        request["model_path"], local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        request["model_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="eager",
    ).eval()
    original_keys = set(model.state_dict())
    excluded = resolve_exclusions(
        [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)],
        plan["exclude_modules"],
    )
    scheme = {
        "targets": ["Linear"],
        "weights": {
            "num_bits": 8 if method == "smoothquant" else plan["bits"],
            "type": "int",
            "symmetric": method != "awq",
            "strategy": "channel" if method == "smoothquant" else "group",
            **({} if method == "smoothquant" else {"group_size": plan["group_size"]}),
        },
    }
    if method == "gptq":
        recipe = [
            GPTQModifier(
                config_groups={"main": scheme},
                ignore=excluded,
                dampening_frac=plan["damping_fraction"],
                actorder=None,
            )
        ]
    elif method == "awq":

        class MixedPrecisionAWQModifier(AWQModifier):
            def _set_resolved_mappings(self, model):
                # In 0.9 ignore affects both quantization and sibling matching.
                # A float Q projection still needs inverse scaling when its
                # shared input norm and quantized K/V projections are smoothed.
                # Keep the official grid search/quantizer, resolving full groups
                # only for smoothing; quantization exclusions remain unchanged.
                ignored = self.ignore
                try:
                    self.ignore = []
                    super()._set_resolved_mappings(model)
                finally:
                    self.ignore = ignored
                self._resolved_mappings = [
                    mapping
                    for mapping in self._resolved_mappings
                    if any(
                        getattr(
                            getattr(layer, "quantization_scheme", None), "weights", None
                        )
                        is not None
                        for layer in mapping.balance_layers
                    )
                ]

        recipe = [
            MixedPrecisionAWQModifier(
                config_groups={"main": scheme},
                ignore=excluded,
                n_grid=plan["awq_grid"],
            )
        ]
    else:
        # The deployed INT8 GEMM performs dynamic per-token activation scaling.
        # Calibration smooths weights/activations; do not quantize them twice in
        # the bridge or call this W8A16 at runtime.
        recipe = [
            SmoothQuantModifier(smoothing_strength=plan["smooth_alpha"]),
            QuantizationModifier(config_groups={"main": scheme}, ignore=excluded),
        ]
    dataset = Dataset.from_list(
        [
            {
                "input_ids": tokenizer(
                    text,
                    truncation=True,
                    max_length=min(
                        plan["calibration_tokens"], model.config.max_position_embeddings
                    ),
                )["input_ids"]
            }
            for text in plan["calibration"]
        ]
    )
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=plan["calibration_tokens"],
        tie_word_embeddings=model.config.tie_word_embeddings,
        num_calibration_samples=len(dataset),
        shuffle_calibration_samples=False,
    )
    records, payload = [], {}
    for name, module in model.named_modules():
        quant_scheme = getattr(module, "quantization_scheme", None)
        if (
            not isinstance(module, torch.nn.Linear)
            or quant_scheme is None
            or quant_scheme.weights is None
        ):
            continue
        args = quant_scheme.weights
        if getattr(module, "weight_g_idx", None) is not None:
            expected = (
                torch.arange(module.in_features, device=module.weight_g_idx.device)
                // plan["group_size"]
            )
            if not torch.equal(module.weight_g_idx.long(), expected):
                raise ValueError("L4_NONCANONICAL_GPTQ_GROUP_PERMUTATION")
        scale, zero = (
            module.weight_scale.detach().float(),
            getattr(module, "weight_zero_point", None),
        )
        zero = (
            torch.zeros_like(scale, dtype=torch.int8) if zero is None else zero.detach()
        )
        codes = quantize(
            module.weight.detach(), scale, zero, args=args, dtype=torch.int8
        )
        index = len(records)
        payload[f"q{index}"] = encode_codes(codes, args.num_bits)
        payload[f"s{index}"] = scale
        payload[f"z{index}"] = zero.int()
        records.append(
            {
                "name": name,
                "k": module.in_features,
                "n": module.out_features,
                "bits": args.num_bits,
                "group_size": module.in_features
                if method == "smoothquant"
                else plan["group_size"],
                "codes": f"q{index}",
                "scales": f"s{index}",
                "zeros": f"z{index}",
                "execution": "cuda_int8_mm"
                if method == "smoothquant"
                else "hqsb_fused_dequant",
            }
        )
    if not records:
        raise ValueError("L4_CALIBRATION_DID_NOT_QUANTIZE_ANY_LINEAR")
    export_native(
        model,
        tokenizer,
        request["destination"],
        records,
        payload,
        {
            **request["metadata"],
            "calibration_seconds": time.monotonic() - started,
            "calibration_versions": {
                p: metadata.version(p)
                for p in (
                    "torch",
                    "transformers",
                    "llmcompressor",
                    "compressed-tensors",
                )
            },
        },
        original_keys,
    )


if __name__ == "__main__":
    main(sys.argv[1])
