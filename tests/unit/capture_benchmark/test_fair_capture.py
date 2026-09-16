"""Contract regressions for independent stock-format adapters and prepared sources."""
import gzip
import json
import struct
from types import SimpleNamespace

import pytest

from benchmarks.capture.fair.decode import events
from benchmarks.capture.fair.run import input_files
from benchmarks.capture.oracle import verify
from benchmarks.capture.schedule import payload
from test_capture_benchmark import fixture_capture


def test_prepared_body_period_preserves_giant_and_cardinality_cycles(tmp_path):
    case = dict(duration_s=.01, rate=1000, seed=100, body_bytes=128,
                giant_every=100, giant_bytes=65536, cardinality=64)
    prepared = input_files(tmp_path, case)
    with (prepared/'bodies.bin').open('rb') as stream:
        period = struct.unpack('<I', stream.read(4))[0]
        assert period == 1600  # max(100,64) would silently corrupt later events.
        bodies = []
        for _ in range(period):
            size = struct.unpack('<I', stream.read(4))[0]
            bodies.append(stream.read(size).decode())
    for index in (99, 100, 1599, 1600, 1699, 10000):
        assert bodies[index % period] == payload(index, case)


@pytest.mark.parametrize('compressed', [False, True])
def test_dftracer_decoder_preserves_json_text_and_uint64_timestamp(tmp_path, compressed):
    body = json.dumps({'name': 'module_汉字', 'data': '"\\\n|☃'})
    expected = dict(run_id='r', producer_id=2, source_seq=7,
                    source_timestamp=39_719_559_540_269_779, body=body)
    text = json.dumps(dict(name='capture', args={k: str(v) for k,v in expected.items()}))+'\n'
    if compressed:
        with gzip.open(tmp_path/'dftrace.pfw.gz', 'wt') as stream:
            stream.write(text)
    else:
        (tmp_path/'dftrace.pfw').write_text(text)
    assert list(events(tmp_path, 'dftracer', tmp_path)) == [expected]


def test_quill_decoder_does_not_split_pipe_inside_body(tmp_path):
    body = json.dumps({'data':'汉字|"\\'})
    (tmp_path/'quill.log').write_text('HQSB|r|2|7|39719559540269779|'+body+'\n')
    assert list(events(tmp_path, 'quill', tmp_path))[0]['body'] == body


def test_external_decoder_cannot_hide_timestamp_rounding(tmp_path):
    rows = fixture_capture(tmp_path)
    rows[0]['source_timestamp'] += 1
    raw = json.loads((tmp_path/'worker.json').read_text())
    raw['encoded_output_files'] = ['original.bin']
    (tmp_path/'original.bin').write_bytes(b'original')
    (tmp_path/'worker.json').write_text(json.dumps(raw))
    result = verify(tmp_path,tmp_path,decode_events=lambda *_: iter(rows))
    assert result['semantic_corruption'] == 1
    assert result['status'] == 'correctness_fail'
    assert result['encoded_artifacts'][0]['bytes'] == 8


def test_external_decoder_keeps_per_producer_order_contract(tmp_path):
    rows = fixture_capture(tmp_path)
    raw = json.loads((tmp_path/'worker.json').read_text())
    raw['encoded_output_files'] = []
    (tmp_path/'worker.json').write_text(json.dumps(raw))
    result = verify(tmp_path,tmp_path,decode_events=lambda *_: reversed(rows))
    assert result['ordering_violations'] == 1
    assert result['status'] == 'correctness_fail'


@pytest.mark.parametrize('configured', [False, True])
def test_nanolog_receiver_configuration_and_exact_fields(tmp_path, monkeypatch, configured):
    body = json.dumps({'text':'推理|\\xE6'})
    text = 'prefix HQSB|r|3|18446744073709551615|9223372036854775809|'+body+'\n'
    chosen = '/official/configured16m/decompressor' if configured else str(tmp_path/'NanoLog/runtime/decompressor')
    if configured:
        monkeypatch.setenv('HQSB_NANOLOG_DECOMPRESSOR', chosen)
    else:
        monkeypatch.delenv('HQSB_NANOLOG_DECOMPRESSOR', raising=False)
    def invoke(argv, **kwargs):
        assert argv[0] == chosen
        kwargs['stdout'].write(text.encode())
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('benchmarks.capture.fair.decode.subprocess.run', invoke)
    assert list(events(tmp_path, 'nanolog', tmp_path)) == [dict(
        run_id='r', producer_id=3, source_seq=2**64-1, source_timestamp=2**63+1, body=body)]
