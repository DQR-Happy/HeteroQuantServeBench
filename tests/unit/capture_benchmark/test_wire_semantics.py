import pytest

from benchmarks.capture.fair.decode import sql_uint64
from benchmarks.capture.fair.wire_semantics import quill_json_text


def test_real_sanitizer_utf8_bytes_and_literal_hex_have_distinct_meanings():
    wire = r'{"path":"C:\\xE6","name":"\xE6\x8E\xA8\xE7\x90\x86","data":"\xC2\xB5"}'
    assert quill_json_text(wire) == r'{"path":"C:\\xE6","name":"推理","data":"µ"}'


def test_original_backslash_followed_by_utf8_remains_byte_exact():
    assert quill_json_text(r'{"data":"\\\xE6\x8E\xA8"}') == r'{"data":"\\推"}'


def test_invalid_utf8_is_not_silently_replaced():
    with pytest.raises(UnicodeDecodeError):
        quill_json_text(r'{"name":"\xFF"}')


def test_sql_unsigned_bit_pattern_round_trip():
    assert sql_uint64(str(-(2**63) + 1)) == 2**63 + 1
    assert sql_uint64('-1') == 2**64 - 1
    assert sql_uint64('0') == 0
    assert sql_uint64(str(2**53 + 1)) == 2**53 + 1
    with pytest.raises(ValueError):
        sql_uint64(str(-(2**63) - 1))
