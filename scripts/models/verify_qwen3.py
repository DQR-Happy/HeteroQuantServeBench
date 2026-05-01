#!/usr/bin/env python3
"""Verify Qwen3-1.7B model architecture and configuration.

Loads the model configuration from the local ModelScope directory
and prints key architecture parameters for validation.

Usage:
    python scripts/models/verify_qwen3.py [--model-path ~/models/hqsb/Qwen3-1.7B]
"""

from __future__ import annotations

import argparse
import os

from modelscope import AutoConfig


def main() -> None:
    """Load and display model configuration."""
    parser = argparse.ArgumentParser(
        description="Verify Qwen3 model architecture",
    )
    parser.add_argument(
        "--model-path",
        default="~/models/hqsb/Qwen3-1.7B",
        help="Local model directory path",
    )
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)

    if not os.path.isdir(model_path):
        print(f"Error: Model directory not found: {model_path}")
        print("Download the model first:")
        print("  python scripts/models/download_qwen3_modelscope.py")
        return

    print(f"Loading config from: {model_path}")
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)

    # Qwen3 stores rope_theta inside rope_parameters dict
    rope_theta = "N/A"
    if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
        rope_theta = config.rope_parameters.get("rope_theta", "N/A")
    elif hasattr(config, "rope_theta"):
        rope_theta = config.rope_theta

    params = [
        ("model_type", config.model_type),
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("num_hidden_layers", config.num_hidden_layers),
        ("num_attention_heads", config.num_attention_heads),
        ("num_key_value_heads", config.num_key_value_heads),
        ("max_position_embeddings", config.max_position_embeddings),
        ("rms_norm_eps", config.rms_norm_eps),
        ("rope_theta", rope_theta),
        ("vocab_size", getattr(config, "vocab_size", "N/A")),
        ("hidden_act", getattr(config, "hidden_act", "N/A")),
        ("tie_word_embeddings", getattr(config, "tie_word_embeddings", "N/A")),
        ("head_dim", getattr(config, "head_dim", "N/A")),
    ]

    print()
    print(f"{'Parameter':<28s} {'Value':<20s}")
    print("-" * 48)
    for name, value in params:
        print(f"  {name:<26s} {str(value):<20s}")

    # Compute model size estimate
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    layers = config.num_hidden_layers
    vocab = getattr(config, "vocab_size", 151936)

    # Rough FP16 size estimate (in GB)
    emb_size = vocab * hidden * 2  # embedding + lm_head
    attn_size = layers * 4 * hidden * hidden * 2  # Q, K, V, O
    mlp_size = layers * 3 * hidden * intermediate * 2  # gate, up, down
    norm_size = layers * 2 * hidden * 2  # input_layernorm + post_attention_layernorm

    total_bytes = emb_size + attn_size + mlp_size + norm_size
    total_gb = total_bytes / (1024**3)

    print()
    print(f"Estimated FP16 model size: {total_gb:.2f} GB")
    print("Note: Actual memory usage includes KV cache and overhead.")


if __name__ == "__main__":
    main()
