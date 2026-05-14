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

__all__ = [
    "DummyBackend",
    "make_dummy_backend",
    "PyTorchBackend",
    "make_pytorch_backend",
]


def __getattr__(name: str):
    """Lazily expose the PyTorch backend.

    Importing the Dummy path must not pull in ``torch``/``transformers``
    (optional-dependency isolation, E01-03/E01-04). ``PyTorchBackend`` and
    ``make_pytorch_backend`` are therefore imported only when explicitly
    requested, so ``from hqsb.backends import DummyBackend`` stays clean.
    """
    if name in {"PyTorchBackend", "make_pytorch_backend"}:
        from hqsb.backends.pytorch import PyTorchBackend, make_pytorch_backend

        globals()[name] = {
            "PyTorchBackend": PyTorchBackend,
            "make_pytorch_backend": make_pytorch_backend,
        }[name]
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
