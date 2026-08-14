"""Reconstruct allocator lifetimes from native sequence; never invent a clock.

An address is reusable. Generations below are local to this retained trace,
not globally unique allocation IDs, and unknown prefixes remain unknown.
"""


def lifetimes(snapshot, *, prefix_may_be_overwritten=False):
    rows, active = [], {}
    for device, trace in enumerate(snapshot.get("device_traces", [])):
        generations = {}
        for index, event in enumerate(trace):
            action, address = event["action"], event.get("addr")
            if address is None or action not in {
                "alloc",
                "free_requested",
                "free_completed",
            }:
                continue
            key = (device, address)
            if action == "alloc":
                generation = generations.get(address, 0) + 1
                generations[address] = generation
                item = {
                    "device": device,
                    "address": hex(address),
                    "bytes": event.get("size"),
                    "generation": generation,
                    "alloc_seq": index,
                    "free_requested_seq": None,
                    "free_completed_seq": None,
                    "start_status": "observed",
                    "end_status": "not_observed",
                }
                active[key] = item
                rows.append(item)
            else:
                item = active.get(key)
                if item is None:
                    item = {
                        "device": device,
                        "address": hex(address),
                        "bytes": event.get("size"),
                        "generation": None,
                        "alloc_seq": None,
                        "free_requested_seq": None,
                        "free_completed_seq": None,
                        "start_status": "prefix_missing_or_preexisting",
                        "end_status": "not_observed",
                    }
                    active[key] = item
                    rows.append(item)
                item[action + "_seq"] = index
                if action == "free_completed":
                    item["end_status"] = "observed"
                    active.pop(key, None)
    return {
        "items": rows,
        "sequence_domain": "per_device_native_allocator_trace",
        "time_source": "sequence_only",
        "prefix_may_be_overwritten": prefix_may_be_overwritten,
        "generation_scope": "retained_capture_local",
        "scope": "torch_allocator_not_driver",
    }


def snapshot_attribution(tensors, snapshot, lifetime_report):
    blocks = [
        (segment.get("device", 0), block)
        for segment in snapshot.get("segments", [])
        for block in segment.get("blocks", [])
        if block["state"] == "active_allocated"
    ]
    active = {
        (row["device"], int(row["address"], 16)): row
        for row in lifetime_report["items"]
        if row["free_completed_seq"] is None
    }
    result = []
    for entry in tensors:
        device = entry.get("device", "")
        if "storage_address" not in entry or not device.startswith("cuda:"):
            continue
        device_id = int(device.split(":")[1])
        address, size = int(entry["storage_address"], 16), entry.get("storage_bytes")
        if size is None:
            continue
        matches = [
            block
            for dev, block in blocks
            if dev == device_id
            and block["address"] <= address
            and address + size <= block["address"] + block["size"]
        ]
        block = matches[0] if len(matches) == 1 else None
        event = active.get((device_id, block["address"])) if block else None
        result.append(
            {
                "tensor_name": entry["name"],
                "kind": entry["kind"],
                "storage_key": entry["storage_key"],
                "storage_bytes": size,
                "allocation_address": hex(block["address"]) if block else None,
                "confidence": "snapshot_correlated" if block else "unavailable",
                "anchor": "before_cleanup",
                "device": device,
                "generation": event["generation"] if event else None,
                "generation_scope": "retained_capture_local_or_preexisting_unknown",
                "scope": "device_virtual_address_not_physical",
            }
        )
    return result
