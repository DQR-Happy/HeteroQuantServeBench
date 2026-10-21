"""Unprobed, interleaved GPU timing for the fixed DMAT Attention workload."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert 'HQSB_DMAT_CAPACITY' not in os.environ and 'LD_PRELOAD' not in os.environ
    import torch
    import triton
    from hqsb.backends.model_kernels.triton_attention import run
    from hqsb.observation.lease import DeviceLease
    lease = DeviceLease(Path.home() / '.cache/hqsb/device0.lock')
    lease.acquire()
    try:
        torch.manual_seed(20261021)
        with torch.inference_mode():
            q = torch.randn(1, 512, 2, 64, device='cuda', dtype=torch.float16).transpose(1, 2)
            k, v = torch.randn_like(q), torch.randn_like(q)
            graphs, keepalive = {}, []
            for tile in (32, 64):
                for _ in range(10):
                    run(q, k, v, 64**-.5, block_n=tile)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    keepalive.extend(run(q, k, v, 64**-.5, block_n=tile) for _ in range(16))
                graphs[tile] = graph
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(keepalive[0], keepalive[-1], rtol=.02, atol=.003)
            for _ in range(10):
                for tile in (32, 64, 64, 32):
                    graphs[tile].replay()
            torch.cuda.synchronize()
            samples = {32: [], 64: []}
            for _ in range(20):
                for tile in (32, 64, 64, 32):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[tile].replay()
                    end.record()
                    end.synchronize()
                    samples[tile].append(start.elapsed_time(end) / 16)
            result = {'passed': True, 'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'triton': triton.__version__,
                      'workload': 'B1 H2 S512 D64 FP16 causal direct-GQA online softmax',
                      'measurement_role': 'uninstrumented_GPU_kernel_time_16_launches_per_graph',
                      'order': 'ABBA_interleaved_20_rounds', 'clock_control': 'default', 'cache_policy': 'warm_default',
                      'source_sha256': hashlib.sha256(Path('hqsb/backends/model_kernels/triton_attention.py').read_bytes()).hexdigest(),
                      'variants': {str(t): {'samples_ms': s, 'median_ms': statistics.median(s),
                                      'min_ms': min(s), 'max_ms': max(s)} for t, s in samples.items()}}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
            print(json.dumps({**result, 'variants': {t: {k: v for k, v in r.items() if k != 'samples_ms'} for t, r in result['variants'].items()}}, indent=2))
    finally:
        lease.release()


if __name__ == '__main__':
    main()
