"""Fixed-token workload generation for deterministic benchmarks.

Generates synthetic input sequences of exact token lengths for
model-core benchmarking, ensuring reproducible measurements
independent of prompt content.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

import torch

logger = logging.getLogger(__name__)

# Seed text used to generate fixed-length token sequences.
# The text is designed to contain technical vocabulary that
# produces well-defined token boundaries.
_SEED_TEXT = (
    "CUDA GPU inference optimization "
    "memory bandwidth kernel latency "
    "transformer attention cache "
    "performance benchmark. "
)


def make_fixed_token_input(
    tokenizer: Any,
    input_tokens: int,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Generate a synthetic input of exactly ``input_tokens`` tokens.

    The input is constructed by repeating a fixed seed text to reach
    the desired token count, then truncating to the exact length.
    This guarantees deterministic tokenization and reproducible
    benchmark conditions across runs.

    Args:
        tokenizer: A HuggingFace-compatible tokenizer instance.
        input_tokens: Exact number of input tokens to generate.
            Must be >= 1.
        device: PyTorch device string for output tensors.
            Default: ``"cuda"``.

    Returns:
        A dictionary with keys:
            - ``input_ids``: Tensor of shape ``(1, input_tokens)``.
            - ``attention_mask``: Tensor of shape ``(1, input_tokens)``
              (all ones).

    Raises:
        RuntimeError: If the tokenizer produces zero tokens from the
            seed text.
        ValueError: If ``input_tokens`` < 1.
    """
    if input_tokens < 1:
        raise ValueError(f"input_tokens must be >= 1, got {input_tokens}")

    token_ids = tokenizer(_SEED_TEXT, add_special_tokens=False)["input_ids"]

    if not token_ids:
        raise RuntimeError(
            "Tokenizer produced no tokens from seed text. "
            "Check tokenizer configuration."
        )

    # Repeat seed tokens to reach the desired length
    expanded: list[int] = []
    while len(expanded) < input_tokens:
        expanded.extend(token_ids)

    expanded = expanded[:input_tokens]

    input_ids = torch.tensor([expanded], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    assert input_ids.shape[1] == input_tokens, (
        f"Token count mismatch: expected {input_tokens}, "
        f"got {input_ids.shape[1]}"
    )

    logger.debug(
        "Generated workload: input_tokens=%d, device=%s",
        input_tokens,
        device,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
