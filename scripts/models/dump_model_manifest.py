#!/usr/bin/env python3
"""Generate a JSON manifest for the Qwen3-1.7B model.

Outputs key architecture parameters and metadata as a structured
JSON document, used for reproducibility tracking.

Usage:
    python scripts/models/dump_model_manifest.py \\
        > docs/benchmark/qwen3_model_manifest.json
"""

from __future__ import annotations

import json
import os

from modelscope import AutoConfig


def main() -> None:
    """Generate and print the model manifest JSON."""
    model_path = os.path.expanduser("~/models/hqsb/Qwen3-1.7B")

    if not os.path.isdir(model_path):
        print(json.dumps({
            "error": f"Model directory not found: {model_path}",
            "hint": "Run scripts/models/download_qwen3_modelscope.py first",
        }, indent=2))
        return

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)

    # Qwen3 stores rope_theta inside rope_parameters dict
    rope_theta = None
    if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
        rope_theta = config.rope_parameters.get("rope_theta")
    elif hasattr(config, "rope_theta"):
        rope_theta = config.rope_theta

    manifest = {
        "schema_version": "1.0.0",
        "source": "modelscope",
        "model_id": "Qwen/Qwen3-1.7B",
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": rope_theta,
        "vocab_size": getattr(config, "vocab_size", None),
        "hidden_act": getattr(config, "hidden_act", None),
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", None),
        "head_dim": getattr(config, "head_dim", None),
        "local_path": model_path,
    }

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
