"""Bounded, observational ATen dataflow capture on the existing model owner.

The collector records Tensor metadata around the existing forward calls.
The result describes the paths executed for the captured input.
"""
from contextlib import contextmanager
from collections import Counter
import weakref
import json
from itertools import islice

from hqsb.models.architecture_structure import module_role


def module_inventory(model, limit=4096):
    rows, used, partial = [], 0, False
    for path, module in model.named_modules():
        if len(rows) >= limit:
            partial = True
            break
        parameters = list(islice(module.named_parameters(recurse=False), 65))
        buffers = list(islice(module.named_buffers(recurse=False), 65))
        clipped = len(parameters) > 64 or len(buffers) > 64
        partial = partial or clipped
        row = {"path": path, "parent": path.rsplit(".", 1)[0] if "." in path else "" if path else None,
                     "class": type(module).__module__ + "." + type(module).__name__,
                     "role": module_role(path, type(module).__name__),
                     "parameters": [{"name": name, "shape": list(p.shape), "dtype": str(p.dtype),
                                     "device": str(p.device)} for name, p in parameters[:64]],
                     "buffers": [{"name": name, "shape": list(p.shape), "dtype": str(p.dtype)}
                                 for name, p in buffers[:64]], "tensor_metadata_truncated": clipped}
        used += len(json.dumps(row).encode())
        if used > 2 * 1024**2:
            partial = True
            break
        rows.append(row)
    return {"modules": rows, "status": "partial" if partial else "complete",
            "source": "resident_nn_module_tree", "limit": limit, "byte_limit": 2 * 1024**2,
            "scope": "actual_module_containment_not_forward_edges"}


class ArchitectureCapture:
    def __init__(self, model, *, max_nodes=8192, max_edges=40000):
        import torch
        self.torch, self.model = torch, model
        self.max_nodes, self.max_edges = max_nodes, max_edges
        self.nodes, self.edges, self.tensors, self.stack = [], [], {}, []
        self.writers, self.edge_keys, self.names, self.kinds = {}, set(), {}, {}
        self.storage_refs = {}
        self.phases, self.warnings, self.histogram = [], set(), Counter()
        self.truncated = False
        self.phase = "unknown"
        for name, tensor in model.named_parameters(remove_duplicate=False):
            self.names.setdefault(id(tensor), []).append(name)
            self.kinds[id(tensor)] = "parameter"
        for name, tensor in model.named_buffers(remove_duplicate=False):
            self.names.setdefault(id(tensor), []).append(name)
            self.kinds.setdefault(id(tensor), "buffer")

    def walk(self, value):
        if isinstance(value, self.torch.Tensor):
            yield value
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from self.walk(child)
        elif isinstance(value, dict):
            for child in value.values():
                yield from self.walk(child)

    def storage(self, tensor):
        try:
            # Anonymous internal identity only; never serialize an address.
            storage = tensor.untyped_storage()
            if not storage.nbytes():
                return None
            key = storage._cdata
            refs = self.storage_refs.setdefault(key, {})
            if not any(ref() is not None for ref in refs.values()):
                self.writers.pop(key, None)  # An allocator may reuse a dead storage identity.
                refs.clear()
            refs[id(tensor)] = weakref.ref(tensor)
            if len(refs) > 128:
                self.storage_refs[key] = {identity: ref for identity, ref in refs.items() if ref() is not None}
            return key
        except (RuntimeError, NotImplementedError):
            self.warnings.add("STORAGE_ALIAS_IDENTITY_UNAVAILABLE")
            return None

    def metadata(self, tensor):
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device)}

    def node(self, **fields):
        if len(self.nodes) >= self.max_nodes:
            self.truncated = True
            return None
        identity = "n" + str(len(self.nodes))
        self.nodes.append({"id": identity, "phase": self.phase,
                           "module": self.stack[-1][0] if self.stack else "", **fields})
        return identity

    def producer(self, tensor):
        row = self.tensors.get(id(tensor))
        if row and row[0]() is tensor:
            return row[1]
        names = self.names.get(id(tensor), [])
        identity = self.node(kind=self.kinds.get(id(tensor), "external_tensor"), op=None,
                             names=names, outputs=[self.metadata(tensor)])
        self.tensors[id(tensor)] = (weakref.ref(tensor), identity)
        return identity

    def dependencies(self, values):
        dependencies = set()
        for tensor in self.walk(values):
            producer = self.producer(tensor)
            if producer:
                dependencies.add((producer, "data"))
            writer = self.writers.get(self.storage(tensor))
            if writer:
                dependencies.add((writer, "storage_write"))
        return dependencies

    def record(self, op, arguments, outputs, dependencies, *, kind="operator", writes=()):
        tensors = list(self.walk(outputs))
        if len(tensors) > 32:
            self.warnings.add("OUTPUT_METADATA_TRUNCATED")
        identity = self.node(kind=kind, op=op, outputs=[self.metadata(t) for t in tensors[:32]])
        if identity is None:
            return
        for source, reason in sorted(dependencies):
            key = (source, identity, reason)
            if source != identity and key not in self.edge_keys:
                if len(self.edges) >= self.max_edges:
                    self.truncated = True
                    break
                self.edge_keys.add(key)
                self.edges.append({"source": source, "target": identity, "kind": reason})
        for tensor in tensors:
            self.tensors[id(tensor)] = (weakref.ref(tensor), identity)
            self.storage(tensor)
        for tensor in self.walk(writes):
            storage = self.storage(tensor)
            if storage is not None:
                self.writers[storage] = identity
        self.histogram[op] += 1

    @contextmanager
    def forward(self, phase):
        from torch.utils._python_dispatch import TorchDispatchMode
        owner, handles = self, []
        owner.phase = phase
        start = len(self.nodes)

        class Dispatch(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if owner.truncated:
                    return func(*args, **kwargs)
                dependencies = owner.dependencies((args, kwargs))
                writes = []
                for i, argument in enumerate(func._schema.arguments):
                    if argument.alias_info and argument.alias_info.is_write:
                        writes.append(args[i] if i < len(args) else kwargs.get(argument.name))
                result = func(*args, **kwargs)
                owner.record(str(func), (args, kwargs), result, dependencies, writes=writes)
                return result

        def enter(path):
            def hook(module, args, kwargs):
                dependencies = self.dependencies((args, kwargs))
                self.stack.append((path, len(self.nodes), dependencies))
            return hook

        def leave(module, args, kwargs, output):
            if not self.stack:
                return
            path, before, dependencies = self.stack[-1]
            if output is not None and before == len(self.nodes) and list(self.walk(output)) and not self.truncated:
                self.record(type(module).__name__, (args, kwargs), output, dependencies, kind="opaque_module")
                self.warnings.add("MODULE_WITHOUT_DISPATCH_EVENTS")
            self.stack.pop()

        completed = False
        try:
            for path, module in self.model.named_modules():
                handles.append(module.register_forward_pre_hook(enter(path), with_kwargs=True))
                handles.append(module.register_forward_hook(leave, with_kwargs=True, always_call=True))
            with Dispatch():
                yield
            completed = True
        finally:
            for handle in reversed(handles):
                handle.remove()
            self.stack.clear()
            self.phases.append({"name": phase, "node_start": start, "node_end": len(self.nodes), "completed": completed})

    def document(self):
        return {"schema_version": "runtime-architecture-v1", "status": "partial" if self.truncated or self.warnings or any(not phase["completed"] for phase in self.phases) else "captured",
                "source": "torch_dispatch_and_module_hooks", "nodes": self.nodes, "edges": self.edges,
                "phases": self.phases, "operator_counts": dict(self.histogram), "warnings": sorted(self.warnings),
                "limits": {"max_nodes": self.max_nodes, "max_edges": self.max_edges}, "truncated": self.truncated,
                "limitations": ["覆盖本次输入实际执行的 forward、分支与专家路径。",
                    "边表示采集到的 Tensor 生产消费与 storage 写依赖；GPU 活动见设备时间线。",
                    "共享 storage 的视图按存储写依赖关联，字节区间重叠需结合视图布局分析。",
                    "融合 ATen 与自定义算子按其公开调用边界展示。",
                    "Tensor 记录 shape、dtype 和 device；外部 Tensor 标明本次采集的输入边界。",
                    "采集包含 Python 与元数据开销，性能评测请使用独立测量运行。"]}
