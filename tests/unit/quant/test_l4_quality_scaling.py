import math
import pytest
from hqsb.quant.quality import topk_indices, margin, jensen_shannon


def test_partial_topk_keeps_index_ties_and_short_vocabulary_semantics():
    values = [3.0, 9.0, 9.0, -4.0, 3.0]
    for k in (1, 2, 3, 10):
        assert (
            topk_indices(values, k)
            == sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]
        )
    assert margin(values) == 0
    assert margin([3.0, 1.0, 2.0]) == 1


def test_js_is_finite_when_both_distributions_underflow_at_one_token():
    assert jensen_shannon([0, -2000], [0, -3000]) == pytest.approx(0, abs=1e-12)
    assert jensen_shannon([2000, 0], [0, 2000]) == pytest.approx(math.log(2), abs=1e-12)
