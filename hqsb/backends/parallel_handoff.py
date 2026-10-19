"""Explicit same-host Gloo KV handoff. CPU staging is deliberate and measured."""

import time


def transfer(cache, first, model, device, rank):
    import torch
    import torch.distributed as dist
    from transformers import DynamicCache

    started = time.monotonic()
    count = 0
    if rank == 0:
        payload = []
        for layer in cache.layers:
            if layer.keys is None or layer.values is None:
                raise ValueError("L9_INCOMPLETE_KV_HANDOFF")
            payload.extend(
                [
                    layer.keys.detach().cpu().contiguous(),
                    layer.values.detach().cpu().contiguous(),
                ]
            )
        metadata = [
            {
                "first": first,
                "shapes": [list(t.shape) for t in payload],
                "dtypes": [str(t.dtype).split(".")[-1] for t in payload],
            }
        ]
        count = sum(t.numel() * t.element_size() for t in payload)
        if count > 256 * 1024**2:
            raise ValueError("L9_KV_HANDOFF_BUDGET_EXCEEDED")
        dist.send_object_list(metadata, dst=1)
        for tensor in payload:
            dist.send(tensor, dst=1)
    else:
        metadata = [None]
        dist.recv_object_list(metadata, src=0)
        metadata = metadata[0]
        tensors = []
        for shape, dtype in zip(metadata["shapes"], metadata["dtypes"]):
            tensor = torch.empty(shape, dtype=getattr(torch, dtype), device="cpu")
            count += tensor.numel() * tensor.element_size()
            if count > 256 * 1024**2:
                raise ValueError("L9_KV_HANDOFF_BUDGET_EXCEEDED")
            dist.recv(tensor, src=0)
            tensors.append(tensor.to(device))
        cache = DynamicCache(config=model.config)
        for i in range(0, len(tensors), 2):
            cache.update(tensors[i], tensors[i + 1], i // 2)
        first = metadata["first"]
    if device != "cpu":
        torch.cuda.synchronize()
    return (
        cache,
        first,
        {
            "kv_payload_bytes": count,
            "handoff_ms": (time.monotonic() - started) * 1000,
            "transport": "gloo_cpu_tensor_send_recv",
            "includes": "CPU staging + blocking send/receive + receiver GPU placement; no cross-host clock subtraction",
        },
    )
