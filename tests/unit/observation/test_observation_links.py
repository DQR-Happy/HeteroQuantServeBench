from hqsb.observation.correlation import trace_links


def event(index, category, identity=None):
    return {
        "id": index,
        "category": category,
        "args": {"External id": identity} if identity is not None else {},
    }


def test_only_unambiguous_tool_identity_creates_edge():
    rows = [
        event(1, "cpu_op", 7),
        event(2, "kernel", 7),
        event(3, "cpu_op", 8),
        event(4, "cpu_op", 8),
        event(5, "kernel", 8),
        event(6, "kernel"),
    ]
    result = trace_links(rows)
    assert result["edges"] == [
        {
            "from_event": 1,
            "to_event": 2,
            "method": "tool_external_id",
            "confidence": "tool_correlated",
        }
    ]
    assert result["ambiguous"] == result["unmatched"] == 1


def test_runtime_rows_repeating_unique_operator_id_do_not_make_it_ambiguous():
    rows = [
        event(1, "cpu_op", 7),
        event(2, "cuda_runtime", 7),
        event(3, "cuda_runtime", 7),
        event(4, "kernel", 7),
        event(5, "kernel", 7),
    ]
    result = trace_links(rows, limit=1)
    assert result["edges"][0]["from_event"] == 1
    assert result["total_correlated"] == 2 and result["truncated"]
    assert result["ambiguous"] == 0
