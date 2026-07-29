"""Read tensor metadata without copying data or importing device libraries.

Storage identity is process-local and opaque. Inventory sums are distinct from
allocator and host accounting: a view retains its complete underlying storage.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from typing import Any


MAX_ENTRIES = 4096
_PROCESS_SALT = secrets.token_bytes(16)


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a tensor dimension or byte count")
    result = int(value)
    if result < 0:
        raise ValueError("Negative tensor metadata")
    return result


def _value(owner: Any, name: str) -> Any:
    value = getattr(owner, name)
    return value() if callable(value) else value


class _Inventory:
    def __init__(self, scope: str):
        self.scope = scope
        self.entries: list[dict] = []
        self.storages: dict[str, int] = {}
        self.unobserved: list[dict] = []
        self.limitations: list[str] = []
        self.truncated = False
        self.attempted_entries = 0

    def unknown(self, name: str, reason: str):
        if len(self.unobserved) < MAX_ENTRIES:
            self.unobserved.append({"name": name, "reason": reason[:400]})
        else:
            self.truncated = True

    def add(
        self, name: str, tensor: Any, kind: str, *, layer: int | None = None
    ) -> bool:
        if self.attempted_entries >= MAX_ENTRIES:
            self.truncated = True
            return False
        self.attempted_entries += 1
        try:
            shape = [_nonnegative_int(dimension) for dimension in tensor.shape]
            element_bytes = _nonnegative_int(_value(tensor, "element_size"))
            logical_bytes = _nonnegative_int(_value(tensor, "numel")) * element_bytes
            device = str(tensor.device)
            entry = {
                "name": str(name)[:1000],
                "kind": kind,
                "shape": shape,
                "dtype": str(tensor.dtype),
                "device": device,
                "logical_bytes": logical_bytes,
                "storage_key": None,
                "storage_offset": None,
                "storage_offset_bytes": None,
                "storage_bytes": None,
            }
            if layer is not None:
                entry["layer"] = layer
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            self.unknown(name, "Tensor metadata unavailable: " + str(exc))
            return True
        try:
            if device == "meta":
                raise ValueError("Meta tensor has no materialized backing allocation")
            entry["storage_offset"] = _nonnegative_int(_value(tensor, "storage_offset"))
            entry["storage_offset_bytes"] = entry["storage_offset"] * element_bytes
            if hasattr(tensor, "untyped_storage"):
                storage = tensor.untyped_storage()
                storage_bytes = _nonnegative_int(_value(storage, "nbytes"))
            else:
                storage = tensor.storage()
                storage_bytes = (
                    _nonnegative_int(_value(storage, "size")) * element_bytes
                )
            pointer = _nonnegative_int(_value(storage, "data_ptr"))
            if pointer == 0 and storage_bytes:
                raise ValueError("Nonempty storage has no observable data pointer")
            identity = hashlib.sha256(
                _PROCESS_SALT + f"{device}:{pointer}".encode()
            ).hexdigest()[:32]
            entry.update(storage_key=identity, storage_bytes=storage_bytes)
            previous = self.storages.get(identity)
            if previous is not None and previous != storage_bytes:
                self.unknown(
                    name,
                    "Storage size changed during inventory; snapshot may be inconsistent",
                )
            self.storages[identity] = max(previous or 0, storage_bytes)
        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            NotImplementedError,
        ) as exc:
            self.unknown(name, "Backing storage unavailable: " + str(exc))
        self.entries.append(entry)
        return True

    def result(self, *, layout: str) -> dict:
        unavailable = not self.entries and bool(self.unobserved)
        incomplete = self.truncated or bool(self.unobserved)
        return {
            "schema_version": 1,
            "scope": self.scope,
            "layout": layout,
            "sampled_at": time.time(),
            "availability": "unavailable"
            if unavailable
            else "partial"
            if incomplete
            else "complete",
            "entries": self.entries,
            "entry_count": len(self.entries),
            "limit": MAX_ENTRIES,
            "truncated": self.truncated,
            "logical_total_bytes": None
            if unavailable
            else sum(row["logical_bytes"] for row in self.entries),
            "unique_storage_bytes": None
            if unavailable or (self.entries and not self.storages and self.unobserved)
            else sum(self.storages.values()),
            "unique_storages": len(self.storages),
            "storage_observed_entries": sum(
                row["storage_key"] is not None for row in self.entries
            ),
            "unobserved": self.unobserved,
            "limitations": self.limitations
            + [
                "logical_total_bytes 按已列出的名称求和，别名可重复；unique_storage_bytes 按已观察底层 storage 去重。",
                "view 保留完整 storage；storage 字节可能包含视图未使用区域，不能与 logical 字节相加。",
                "storage_key 仅在此进程内可关联；不暴露裸指针，不是物理地址或跨进程分配身份。",
                "未采集 tensor 值；不覆盖激活、临时 workspace、allocator 余量、驱动或其他进程内存。",
                "partial 时总量只覆盖已观察条目，不能当作完整模型或设备内存。",
            ],
        }


def parameter_inventory(model: Any) -> dict:
    """Inventory named parameters and buffers, retaining tied-name aliases."""
    inventory = _Inventory("model_parameters_and_buffers")
    for accessor, kind in (
        ("named_parameters", "parameter"),
        ("named_buffers", "buffer"),
    ):
        try:
            method = getattr(model, accessor)
            try:
                iterator = method(remove_duplicate=False)
            except TypeError:
                iterator = method()
                inventory.limitations.append(
                    f"{accessor} 不支持 remove_duplicate=False；使用兼容接口，名称别名可能已被去重。"
                )
                inventory.unknown(
                    accessor, "Legacy enumeration can omit tied tensor names"
                )
            # Consume lazily and stop at the shared budget. No list(model.*()).
            for name, tensor in iterator:
                if not inventory.add(name, tensor, kind):
                    break
            if inventory.truncated:
                break
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            inventory.unknown(
                accessor, "Named tensor enumeration unavailable: " + str(exc)
            )
    return inventory.result(layout="named_parameters_and_buffers")


def cache_inventory(past_key_values: Any) -> dict:
    """Inspect supported HF/legacy cache containers without mutation or copies."""
    inventory = _Inventory("retained_kv_cache")
    inventory.limitations.append(
        "这是请求结束时保留的 KV storage 快照，不推断分页、物理地址、历史分配峰值或缓存复用命中。"
    )
    if past_key_values is None:
        inventory.unknown("cache", "Model returned no observable past_key_values")
        return inventory.result(layout="unavailable")
    try:
        keys = getattr(past_key_values, "key_cache", None)
        values = getattr(past_key_values, "value_cache", None)
        layers = getattr(past_key_values, "layers", None)
        if isinstance(keys, (list, tuple)) and isinstance(values, (list, tuple)):
            layout = "hf_key_value_cache"
            for layer in range(min(max(len(keys), len(values)), MAX_ENTRIES // 2 + 1)):
                for items, kind in ((keys, "key"), (values, "value")):
                    if layer >= len(items):
                        inventory.unknown(
                            f"layer.{layer}.{kind}", "Cache counterpart is missing"
                        )
                        continue
                    if not inventory.add(
                        f"layer.{layer}.{kind}", items[layer], kind, layer=layer
                    ):
                        break
                if inventory.truncated:
                    break
        elif isinstance(layers, (list, tuple)):
            layout = "hf_cache_layers"
            for layer, item in enumerate(layers[: MAX_ENTRIES // 2 + 1]):
                for attribute, kind in (("keys", "key"), ("values", "value")):
                    if not inventory.add(
                        f"layer.{layer}.{kind}",
                        getattr(item, attribute, None),
                        kind,
                        layer=layer,
                    ):
                        break
                if inventory.truncated:
                    break
        elif isinstance(past_key_values, (tuple, list)):
            layout = "legacy_layer_tuples"
            for layer, item in enumerate(past_key_values[: MAX_ENTRIES + 1]):
                if not isinstance(item, (tuple, list)):
                    inventory.unknown(f"layer.{layer}", "Expected a key/value tuple")
                    continue
                for index, tensor in enumerate(item[:4]):
                    kind = ("key", "value", "cross_key", "cross_value")[index]
                    if not inventory.add(
                        f"layer.{layer}.{kind}", tensor, kind, layer=layer
                    ):
                        break
                if len(item) not in (2, 4):
                    inventory.unknown(
                        f"layer.{layer}",
                        "Expected two self-attention or four self/cross-attention tensors",
                    )
                if inventory.truncated:
                    break
        else:
            layout = "unsupported"
            inventory.unknown(
                "cache",
                "Unsupported cache container; no conversion or tensor copy was attempted",
            )
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        layout = "unsupported"
        inventory.unknown("cache", "Cache enumeration unavailable: " + str(exc))
    return inventory.result(layout=layout)
