"""HQSB backend adapters.

A backend implements the abstract :class:`hqsb.core.contracts.Backend`
contract. The benchmark engine discovers backends through the registry or
receives them via dependency injection; it never imports a concrete backend
directly.

This package currently ships the reference :class:`DummyBackend`. Real
runtime adapters (PyTorch, vLLM, SGLang, TensorRT-LLM, llama.cpp, Ascend)
arrive in later stages (S02/S07/S09).
"""

from hqsb.backends.dummy import DummyBackend, make_dummy_backend
from hqsb.backends.pytorch import PyTorchBackend, make_pytorch_backend

__all__ = [
    "DummyBackend",
    "make_dummy_backend",
    "PyTorchBackend",
    "make_pytorch_backend",
]
