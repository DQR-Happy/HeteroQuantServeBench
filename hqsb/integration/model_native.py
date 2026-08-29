"""Tensor guards and a versioned C ABI for model-facing L2 kernels.

Imported lazily by the worker. No allocation, compilation, or probing on import.
Outputs belong to PyTorch; all launches borrow its current device/stream.
"""
import ctypes as C
import hashlib
import json
import math
from pathlib import Path


class ModelNative:
    def __init__(self, manifest_path):
        import torch

        path = Path(manifest_path).expanduser().resolve()
        raw = path.read_bytes()
        self.identity = json.loads(raw)
        self.identity["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        identity = self.identity
        if identity.get("schema_version") != "l2-native-v1" or identity.get("abi") != 1:
            raise ValueError("KERNEL_ABI_MANIFEST_MISMATCH")
        if identity.get("torch_version") != str(torch.__version__) or identity.get("cuda_version") != torch.version.cuda:
            raise ValueError("KERNEL_RUNTIME_VERSION_MISMATCH")
        sm = torch.cuda.get_device_capability(0)
        if sm[0] * 10 + sm[1] not in identity["sm"]:
            raise ValueError("KERNEL_SM_UNSUPPORTED")
        root = Path(__file__).resolve().parents[2]
        for name, expected in identity["sources"].items():
            source = (root / name).resolve()
            if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError("KERNEL_SOURCE_MISMATCH:" + name)
        library = Path(identity["library"])
        if not library.is_absolute() or hashlib.sha256(library.read_bytes()).hexdigest() != identity["library_sha256"]:
            raise ValueError("KERNEL_ARTIFACT_HASH_MISMATCH")
        self.lib = C.CDLL(str(library))
        signatures = {
            "hqsb_l2_abi": ([], C.c_int),
            "hqsb_l2_norm": ([C.c_void_p]*5 + [C.c_int64, C.c_int, C.c_float, C.c_int, C.c_void_p], C.c_int),
            "hqsb_l2_silu": ([C.c_void_p]*3 + [C.c_int64, C.c_void_p], C.c_int),
            "hqsb_l2_rope": ([C.c_void_p]*4 + [C.c_int]*4 + [C.c_void_p], C.c_int),
            "hqsb_l2_lt_create": ([C.POINTER(C.c_void_p)], C.c_int),
            "hqsb_l2_lt_destroy": ([C.c_void_p], None),
            "hqsb_l2_lt_version": ([], C.c_size_t),
            "hqsb_l2_lt_plan_count": ([C.c_void_p], C.c_int),
            "hqsb_l2_lt_plan_info": ([C.c_void_p,C.c_int]+[C.POINTER(C.c_int)]*4, C.c_int),
            "hqsb_l2_lt": ([C.c_void_p]*4 + [C.c_int]*3 + [C.c_void_p], C.c_int),
            "hqsb_l2_gemv": ([C.c_void_p]*3 + [C.c_int]*3 + [C.c_void_p], C.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        if self.lib.hqsb_l2_abi() != 1:
            raise ValueError("KERNEL_ABI_BINARY_MISMATCH")
        self.identity["cublaslt_version"] = self.lib.hqsb_l2_lt_version()
        self.handle = C.c_void_p()

    @staticmethod
    def check(*tensors):
        import torch

        if torch.is_grad_enabled():
            raise ValueError("KERNEL_INFERENCE_ONLY")
        for x in tensors:
            if x.device != torch.device("cuda:0") or x.dtype != torch.float16 or not x.is_contiguous() or x.numel() == 0:
                raise ValueError("KERNEL_SIGNATURE_REQUIRES_CONTIGUOUS_CUDA0_FP16_NONEMPTY")
            if x.numel() > 2**31 - 1:
                raise ValueError("KERNEL_SIZE_BOUND")

    @staticmethod
    def launch(status):
        if status:
            raise RuntimeError(f"KERNEL_NATIVE_STATUS:{status}")

    @staticmethod
    def stream():
        import torch

        return torch.cuda.current_stream(0).cuda_stream

    def norm(self, x, weight, eps, variant=1, residual=None):
        import torch

        self.check(x, weight, *([residual] if residual is not None else []))
        if x.ndim < 1:
            raise ValueError("KERNEL_NORM_SIGNATURE")
        h = x.shape[-1]
        if weight.shape != (h,) or not math.isfinite(eps) or eps <= 0 or variant not in (1, 2, 3):
            raise ValueError("KERNEL_NORM_SIGNATURE")
        if variant == 2 and (h % 2 or x.data_ptr() % 4):
            raise ValueError("KERNEL_VECTOR_ALIGNMENT")
        if variant == 3 and (residual is None or residual.shape != x.shape):
            raise ValueError("KERNEL_RESIDUAL_SIGNATURE")
        y = torch.empty_like(x)
        updated = torch.empty_like(x) if variant == 3 else None
        with torch.cuda.device(x.device):
            self.launch(self.lib.hqsb_l2_norm(x.data_ptr(), residual.data_ptr() if residual is not None else None,
                weight.data_ptr(), y.data_ptr(), updated.data_ptr() if updated is not None else None,
                x.numel() // h, h, eps, variant, self.stream()))
        return (y, updated) if updated is not None else y

    def silu(self, gate, up):
        import torch

        self.check(gate, up)
        if gate.shape != up.shape:
            raise ValueError("KERNEL_ACTIVATION_SHAPE")
        y = torch.empty_like(gate)
        with torch.cuda.device(gate.device):
            self.launch(self.lib.hqsb_l2_silu(gate.data_ptr(), up.data_ptr(), y.data_ptr(), gate.numel(), self.stream()))
        return y

    def rope(self, x, cos, sin):
        import torch

        self.check(x, cos, sin)
        if x.ndim != 4 or cos.shape != sin.shape or cos.shape != (x.shape[0], x.shape[1], x.shape[3]) or x.shape[3] % 2:
            raise ValueError("KERNEL_ROPE_SIGNATURE")
        b, t, h, d = x.shape
        y = torch.empty((b, h, t, d), dtype=x.dtype, device=x.device)
        with torch.cuda.device(x.device):
            self.launch(self.lib.hqsb_l2_rope(x.data_ptr(), cos.data_ptr(), sin.data_ptr(), y.data_ptr(), b,t,h,d,self.stream()))
        return y

    def linear(self, x, weight, small_m=False):
        import torch

        self.check(x, weight)
        if x.ndim < 2 or weight.ndim != 2 or weight.shape[1] != x.shape[-1]:
            raise ValueError("KERNEL_LINEAR_SIGNATURE")
        n, k = weight.shape
        m = x.numel() // k
        if m * n > 2**31 - 1:
            raise ValueError("KERNEL_LINEAR_OUTPUT_SIZE_BOUND")
        y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
        with torch.cuda.device(x.device):
            if small_m:
                if m > 4:
                    raise ValueError("KERNEL_GEMV_SMALL_M_ONLY")
                status = self.lib.hqsb_l2_gemv(x.data_ptr(), weight.data_ptr(), y.data_ptr(), m,n,k,self.stream())
            else:
                if not self.handle.value:
                    self.launch(self.lib.hqsb_l2_lt_create(C.byref(self.handle)))
                status = self.lib.hqsb_l2_lt(self.handle, x.data_ptr(), weight.data_ptr(), y.data_ptr(), m,n,k,self.stream())
            self.launch(status)
        return y

    def close(self):
        # Owner must first synchronize its streams. No borrowed tensor is retained.
        if self.handle.value:
            self.lib.hqsb_l2_lt_destroy(self.handle)
            self.handle = C.c_void_p()

    def plans(self):
        result = []
        for index in range(self.lib.hqsb_l2_lt_plan_count(self.handle)):
            m,n,k,algo = (C.c_int() for _ in range(4))
            self.launch(self.lib.hqsb_l2_lt_plan_info(self.handle,index,C.byref(m),C.byref(n),C.byref(k),C.byref(algo)))
            result.append({"m":m.value,"n":n.value,"k":k.value,"algorithm_id":algo.value,
                           "caller_workspace_bytes":0})
        return result
