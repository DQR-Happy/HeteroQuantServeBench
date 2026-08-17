from hqsb.observation.allocator import lifetimes, snapshot_attribution


def test_address_reuse_is_two_lifetimes_and_free_without_start_stays_unknown():
    snapshot = {
        "device_traces": [
            [
                {"action": "free_requested", "addr": 8, "size": 4},
                {"action": "free_completed", "addr": 8, "size": 4},
                {"action": "alloc", "addr": 16, "size": 8},
                {"action": "free_requested", "addr": 16, "size": 8},
                {"action": "free_completed", "addr": 16, "size": 8},
                {"action": "alloc", "addr": 16, "size": 8},
            ]
        ]
    }
    result = lifetimes(snapshot, prefix_may_be_overwritten=True)
    missing, first, second = result["items"]
    assert missing["generation"] is None and missing["alloc_seq"] is None
    assert first["generation"] == 1 and first["free_completed_seq"] == 4
    assert second["generation"] == 2 and second["free_completed_seq"] is None
    assert result["prefix_may_be_overwritten"]
    assert result["time_source"] == "sequence_only"


def test_snapshot_attribution_checks_device_bounds_alias_and_preexisting():
    snapshot = {
        "device_traces": [[{"action": "alloc", "addr": 16, "size": 16}]],
        "segments": [
            {
                "device": 0,
                "blocks": [
                    {"state": "active_allocated", "address": 16, "size": 16},
                    {"state": "active_allocated", "address": 64, "size": 16},
                ],
            }
        ],
    }
    entry = {
        "device": "cuda:0",
        "storage_address": "0x10",
        "storage_bytes": 8,
        "name": "weight",
        "kind": "parameter",
        "storage_key": "alias",
    }
    rows = [
        entry,
        {**entry, "name": "tied"},
        {**entry, "device": "cpu"},
        {**entry, "device": "cuda:1"},
        {**entry, "storage_bytes": 17},
        {**entry, "storage_address": "0x40"},
    ]
    result = snapshot_attribution(rows, snapshot, lifetimes(snapshot))
    assert len(result) == 5
    assert result[0]["allocation_address"] == result[1]["allocation_address"] == "0x10"
    assert result[0]["generation"] == 1
    assert result[2]["confidence"] == result[3]["confidence"] == "unavailable"
    assert (
        result[-1]["confidence"] == "snapshot_correlated"
        and result[-1]["generation"] is None
    )
