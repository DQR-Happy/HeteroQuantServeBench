"""Jetson Attention acceptance: numerical oracle, unprobed timing, real DMAT pair.

The fixed K/V tile experiment tests a known change in loop/access structure.
It does not claim to reproduce FlashAttention-v1/v2 implementations or speedups.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def child(args):
    import torch
    from hqsb.backends.model_kernels.triton_attention import run
    torch.manual_seed(20261021)
    torch.set_grad_enabled(False)
    cases = [(247, 247, 16, 8, 128, True), (1, 248, 16, 8, 128, True),
             (512, 512, 2, 2, 64, True), (37, 37, 4, 2, 64, True),
             (1, 67, 4, 2, 128, True), (3, 67, 4, 2, 64, True), (33, 41, 2, 2, 64, False)]
    results = []
    for m, n, h, hk, d, causal in cases:
        # Transposed storage exercises noncontiguous head/sequence strides.
        q = torch.randn(1, m, h, d, device='cuda', dtype=torch.float16).transpose(1, 2)
        k = torch.randn(1, n, hk, d, device='cuda', dtype=torch.float16).transpose(1, 2)
        v = torch.randn_like(k)
        logits = q.float() @ k.float().repeat_interleave(h // hk, dim=1).transpose(-1, -2) * d**-.5
        if causal:
            mask = torch.arange(n, device='cuda')[None, :] <= torch.arange(m, device='cuda')[:, None] + n - m
            logits = logits.masked_fill(~mask, float('-inf'))
        expected = (logits.softmax(-1) @ v.float().repeat_interleave(h // hk, dim=1)).transpose(1, 2)
        actual = run(q, k, v, d**-.5, block_n=args.tile, causal=causal)
        torch.testing.assert_close(actual.float(), expected, rtol=.02, atol=.003)
        results.append({'shape': [m, n, h, hk, d], 'causal': causal,
                        'max_abs_error': (actual.float() - expected).abs().max().item()})
        if m == 512:
            saved = q, k, v, expected
    q, k, v, expected = saved
    torch.cuda.synchronize()
    if args.child == 'capture':
        hook = ctypes.CDLL(os.environ['NEUTRINO_HOOK_DRIVER'])
        assert hook.hqsb_eretrace_start() == 0
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            actual = run(q, k, v, 64**-.5, block_n=args.tile, layer='attention-oracle', phase='prefill')
        stream.synchronize()
        assert hook.hqsb_eretrace_stop() == 0
        torch.testing.assert_close(actual.float(), expected, rtol=.02, atol=.003)
        timing = None
    else:
        for _ in range(10):
            run(q, k, v, 64**-.5, block_n=args.tile)
        samples = []
        for _ in range(20):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                run(q, k, v, 64**-.5, block_n=args.tile)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 10)
        timing = {'samples_ms': samples, 'median_ms': statistics.median(samples),
                  'min_ms': min(samples), 'max_ms': max(samples), 'measurement_role': 'uninstrumented_kernel_wrapper'}
    args.output.write_text(json.dumps({'tile': args.tile, 'cases': results, 'timing': timing}, indent=2))


def main(args):
    from hqsb.core.diagnostics import DiagnosticPlan
    from hqsb.observation.eretrace_tools import launch_environment
    from hqsb.observation.dmat_reports import summarize_memory
    from hqsb.observation.dmat_compare import compare_memory
    from hqsb.observation.lease import DeviceLease
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    lease = DeviceLease(Path.home() / '.cache/hqsb/device0.lock')
    lease.acquire()
    panels, numerical = [], []
    try:
        for tile in (32, 64):
            output = root / f'n{tile}'
            output.mkdir()
            command = [sys.executable, __file__, '--tile', str(tile)]
            subprocess.run([*command, '--child', 'timing', '--output', str(output / 'timing.json')], check=True, timeout=180)
            plan = DiagnosticPlan(source_run_id='run_' + '0' * 32, tool='eretrace', ere_mode='memory',
                                  ere_codec='raw', ere_kernel='hqsb_dmat_attention', memory_capacity=args.capacity, launch_count=1)
            env = {**os.environ, **launch_environment(args.runtime, output, plan.model_dump(), output / 'triton')}
            env['TRITON_CACHE_DIR'] = str(output / 'triton')
            subprocess.run([*command, '--child', 'capture', '--output', str(output / 'numerical.json')], env=env, check=True, timeout=300)
            panel = summarize_memory(output, page_bytes=4096, time_bins=160)
            assert panel['availability'] == 'observed', panel['counts']
            launch = panel['memory_launches'][0]
            assert launch['plot']['unknown_events'] == 0, launch['plot']['objects']
            assert {o['role'] for o in launch['plot']['objects']} == {'Q', 'K', 'V', 'O'}
            assert len(launch['plot']['addresses']) > 16
            # Independent causal tile accounting: every query block reloads
            # ceil(its causal endpoint / BN) complete K/V tiles. Q and O once.
            expected_kv = 2 * sum((i * 32 + tile - 1) // tile for i in range(1, 17)) * tile * 64 * 2
            for obj in launch['plot']['objects']:
                expected_bytes = expected_kv if obj['role'] in ('K', 'V') else 2 * 512 * 64 * 2
                assert obj['instruction_bytes'] == expected_bytes, (obj, expected_bytes)
            (output / 'panel.json').write_text(json.dumps(panel))
            panels.append(panel)
            numerical.append(json.loads((output / 'numerical.json').read_text()))
    finally:
        lease.release()
    comparison = compare_memory(panels[0], panels[1], roots=[root / 'n32', root / 'n64'])
    assert comparison['comparable'], comparison
    assert comparison['structure_changed'], comparison
    for role in ('K', 'V'):
        row = next(r for r in comparison['objects'] if r['role'] == role)
        assert row['baseline_events'] != row['candidate_events'], row
    result = {'passed': True, 'workload': 'causal Attention B1 H2 S512 D64 FP16',
              'baseline': 'online softmax K/V tile32', 'candidate': 'online softmax K/V tile64',
              'clock_domains': ['thread_entry_clock64', 'GPU_globaltimer'], 'numerical': numerical,
              'comparison': comparison, 'timings': [json.loads((root / f'n{t}/timing.json').read_text())['timing'] for t in (32, 64)],
              'code_identity': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                  'hqsb/observation/dmat_instrument.py', 'hqsb/observation/dmat_grid.py', 'hqsb/observation/dmat_compare.py',
                  'hqsb/backends/model_kernels/triton_attention.py', __file__)},
              'runtime_manifest_sha256': hashlib.sha256((Path(args.runtime) / 'hqsb-runtime.json').read_bytes()).hexdigest()}
    (root / 'acceptance.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('comparison', 'numerical')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capacity', type=int, default=256)
    p.add_argument('--tile', type=int, default=32)
    p.add_argument('--child', choices=['timing', 'capture'])
    a = p.parse_args()
    child(a) if a.child else main(a)
