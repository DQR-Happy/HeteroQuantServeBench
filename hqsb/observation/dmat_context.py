"""Producer-local tensor lifetime context, snapshotted by the native collector."""
from contextlib import contextmanager
import ctypes
import json
import os

_hook = None
_stack = []


def descriptor(name, tensor):
    if tensor.device.type != 'cuda' or any(s < 0 for s in tensor.stride()):
        raise ValueError('DMAT_TENSOR_REQUIRES_CUDA_NONNEGATIVE_STRIDES')
    extent = (1 + sum((d - 1) * s for d, s in zip(tensor.shape, tensor.stride()))) * tensor.element_size() if tensor.numel() else 0
    return {'name': name, 'role': name, 'base': str(tensor.data_ptr()), 'bytes': extent,
            'logical_bytes': tensor.numel() * tensor.element_size(), 'shape': list(tensor.shape),
            'stride': list(tensor.stride()), 'dtype': str(tensor.dtype), 'mapping': 'live_tensor_view_byte_envelope'}


@contextmanager
def memory_context(tensors, **context):
    global _hook
    library = os.environ.get('NEUTRINO_HOOK_DRIVER') if os.environ.get('HQSB_DMAT_CAPACITY') else None
    if not library:
        yield
        return
    if _hook is None:
        # Load with the same binding contract as the selective loader bridge.
        # Loading first through ctypes without DEEPBIND would interpose the
        # hook's internal cuLaunchKernel call back onto that bridge.
        _hook = ctypes.CDLL(library, mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND).hqsb_eretrace_context
        _hook.argtypes, _hook.restype = [ctypes.c_char_p], ctypes.c_int
    value = {**context, 'tensors': [descriptor(name, tensor) for name, tensor in tensors.items()],
             'lifetime': 'synchronous_selected_launch_inside_producer_context'}
    payload = json.dumps(value, separators=(',', ':')).encode()
    if len(payload) > 65536 or _hook(payload):
        raise RuntimeError('DMAT_TENSOR_CONTEXT_SET')
    _stack.append(payload)
    try:
        yield
    finally:
        _stack.pop()
        if _hook(_stack[-1] if _stack else None):
            raise RuntimeError('DMAT_TENSOR_CONTEXT_RESTORE')
