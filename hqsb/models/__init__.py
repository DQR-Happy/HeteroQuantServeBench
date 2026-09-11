"""HQSB Models module.

Provides model loading utilities with ModelScope integration,
supporting local-only loading, SHA256 verification, and model
manifest generation.
"""

__all__ = [
    "load_qwen3",
    "load_causal_lm",
]


def __getattr__(name):
    if name in __all__:
        from hqsb.models.loader import load_causal_lm, load_qwen3

        return {"load_qwen3": load_qwen3, "load_causal_lm": load_causal_lm}[name]
    raise AttributeError(name)
