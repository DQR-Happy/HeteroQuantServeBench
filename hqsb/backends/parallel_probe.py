"""Isolated, read-only L9 hardware probe; never imports a model."""


def probe():
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "distributed": torch.distributed.is_available(),
        "nccl": getattr(torch.distributed, "is_nccl_available", lambda: False)(),
        "gloo": getattr(torch.distributed, "is_gloo_available", lambda: False)(),
        "devices": [
            {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "total_bytes": torch.cuda.get_device_properties(i).total_memory,
                "free_bytes": torch.cuda.mem_get_info(i)[0],
                "uuid": str(
                    getattr(torch.cuda.get_device_properties(i), "uuid", "unavailable")
                ),
            }
            for i in range(torch.cuda.device_count())
        ],
        "peer_access": [
            [
                torch.cuda.can_device_access_peer(i, j) if i != j else None
                for j in range(torch.cuda.device_count())
            ]
            for i in range(torch.cuda.device_count())
        ],
    }


if __name__ == "__main__":
    import json

    print(json.dumps(probe()))
