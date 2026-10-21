"""Compare HQSB normalized events with the original Neutrino C++ sparsifier.

Exports the same events into Neutrino's thread-map ABI, executes its unchanged
C++ tool, then compares every exact (cycle, 64-KiB page, density) entry.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import numpy as np
    from hqsb.observation.dmat_reports import read_frame
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = read_frame(args.capture)
    records, counts, cap = frame['records'], frame['counts'], frame['capacity']
    assert frame['event_bytes'] == 48 and np.all(counts <= cap)
    enabled = np.arange(cap)[None, :] < counts[:, None]
    flat = records[enabled]
    reference = Counter((int(r['clock']) - int(r['entry']), int(r['address']) & 0xffffffffffff0000) for r in flat)
    converted = np.zeros((*records.shape, 2), dtype='<u8')
    converted[:, :, 0] = np.uint64(2**64 - 1)
    converted[:, :, 0][enabled] = records['clock'][enabled] - records['entry'][enabled]
    converted[:, :, 1][enabled] = records['address'][enabled]
    raw = args.output / 'neutrino-input.bin'
    raw.write_bytes(struct.pack('<8I', *frame['grid'], *frame['block'], frame['shared_bytes'], 1)
                    + struct.pack('<IIQ', cap * 16, 1, 48) + converted.tobytes())
    executable = args.output / 'original-dmat'
    subprocess.run(['g++', str(args.source), '-std=gnu++17', '-O3', '-o', str(executable)], check=True)
    result = args.output / 'neutrino-output.dmat'
    subprocess.run([str(executable), str(raw), str(result)], check=True)
    observed = {}
    with result.open('rb') as stream:
        pages, clocks = struct.unpack('<QQ', stream.read(16))
        stream.read(pages * 8)
        for _ in range(clocks):
            clock, size = struct.unpack('<QQ', stream.read(16))
            for _ in range(size):
                page, count = struct.unpack('<QI', stream.read(12))
                observed[clock, page] = count
        assert not stream.read(1)
    assert observed == reference, 'Original Neutrino time/page/density matrix differs'
    receipt = {'passed': True, 'events': len(flat), 'exact_density_entries': len(reference),
               'original_sparsifier_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
               'capture_sha256': hashlib.sha256(args.capture.read_bytes()).hexdigest(),
               'clock': 'clock64_minus_actual_thread_entry', 'page_bytes': 65536,
               'scope': 'same_record_semantics_and_exact_sparsified_matrix; hardware_timing_and_workload_shapes_vary'}
    (args.output / 'acceptance.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
