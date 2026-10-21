"""Real cp.async oracle including zero-filled lanes and predicated global stores."""
import argparse
from collections import Counter
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PTX = r'''.version 8.0
.target sm_87
.address_size 64
.visible .entry hqsb_dmat_async(.param .u64 X, .param .u64 Y) {
.reg .u32 %r<8>;
.reg .u64 %rd<8>;
.reg .pred %p;
.shared .align 16 .b8 buf[128];
ld.param.u64 %rd0, [X];
ld.param.u64 %rd1, [Y];
mov.u32 %r0, %tid.x;
mul.wide.u32 %rd2, %r0, 4;
add.u64 %rd3, %rd0, %rd2;
mov.u32 %r1, buf;
shl.b32 %r2, %r0, 2;
add.u32 %r3, %r1, %r2;
setp.lt.u32 %p, %r0, 17;
selp.u32 %r4, 4, 0, %p;
cp.async.ca.shared.global.L2::64B [%r3], [%rd3], 4, %r4;
cp.async.commit_group;
cp.async.wait_group 0;
bar.sync 0;
ld.shared.u32 %r5, [%r3];
add.u64 %rd4, %rd1, %rd2;
st.global.u32 [%rd4], %r5;
ret;
}
'''


def child(args):
    import torch
    from hqsb.observation.dmat_context import memory_context
    x = torch.arange(32, device='cuda', dtype=torch.int32) + 1
    y = torch.empty_like(x)
    torch.cuda.synchronize()
    hook = ct.CDLL(os.environ['NEUTRINO_HOOK_DRIVER'], mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND)
    module, function = ct.c_void_p(), ct.c_void_p()
    cached = args.output.parent / 'triton/oracle'
    cached.mkdir(parents=True)
    (cached / 'async.ptx').write_text(PTX)
    subprocess.run(['ptxas', '-arch=sm_87', str(cached / 'async.ptx'), '-o', str(cached / 'async.cubin')], check=True)
    image = ct.create_string_buffer((cached / 'async.cubin').read_bytes())
    hook.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p]
    hook.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
    assert hook.cuModuleLoadData(ct.byref(module), image) == 0
    assert hook.cuModuleGetFunction(ct.byref(function), module, b'hqsb_dmat_async') == 0
    hook.cuLaunchKernel.argtypes = [ct.c_void_p, *([ct.c_uint] * 7), ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p]
    values = [ct.c_uint64(x.data_ptr()), ct.c_uint64(y.data_ptr())]
    parameters = (ct.c_void_p * 2)(*(ct.cast(ct.pointer(v), ct.c_void_p) for v in values))
    assert hook.hqsb_eretrace_start() == 0
    stream = torch.cuda.Stream()
    with memory_context({'Q': x, 'O': y}, operation='async_copy_oracle'):
        assert hook.cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, stream.cuda_stream, parameters, None) == 0
    stream.synchronize()
    assert hook.hqsb_eretrace_stop() == 0
    expected = torch.cat((x[:17], torch.zeros(15, device='cuda', dtype=torch.int32)))
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    args.output.write_text(json.dumps({'x': str(x.data_ptr()), 'y': str(y.data_ptr()), 'output_equal': True}))


def main(args):
    from hqsb.core.diagnostics import DiagnosticPlan
    from hqsb.observation.eretrace_tools import launch_environment
    from hqsb.observation.dmat_reports import read_frame, summarize_memory
    from hqsb.observation.lease import DeviceLease
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = DiagnosticPlan(source_run_id='run_' + '0' * 32, tool='eretrace', ere_mode='memory', ere_codec='raw',
                          ere_kernel='hqsb_dmat_async', memory_capacity=4, launch_count=1)
    env = {**os.environ, **launch_environment(args.runtime, root, plan.model_dump(), root / 'triton'), 'PYTHONFAULTHANDLER': '1'}
    lease = DeviceLease(Path.home() / '.cache/hqsb/device0.lock')
    lease.acquire()
    try:
        subprocess.run([sys.executable, __file__, '--child', '--output', str(root / 'oracle.json')], env=env, check=True, timeout=120)
    finally:
        lease.release()
    panel = summarize_memory(root)
    assert panel['counts']['events'] == 49 and panel['availability'] == 'observed', panel['counts']
    launch = panel['memory_launches'][0]
    rows = read_frame(root / launch['trace'])
    observed = rows['records'][__import__('numpy').arange(rows['capacity'])[None, :] < rows['counts'][:, None]]
    oracle = json.loads((root / 'oracle.json').read_text())
    site = next(s['site'] for s in launch['probe']['sites'] if s['opcode'].startswith('cp.async'))
    assert Counter(int(r['address']) for r in observed if r['site'] == site) == Counter(int(oracle['x']) + i * 4 for i in range(17))
    assert launch['plot']['unknown_events'] == 0
    (root / 'panel.json').write_text(json.dumps(panel))
    (root / 'acceptance.json').write_text(json.dumps({'passed': True, 'read_events': 17, 'write_events': 32,
                        'zero_fill_lanes_excluded_from_reads': 15, 'output_equal': True}))
    print((root / 'acceptance.json').read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--child', action='store_true')
    a = p.parse_args()
    child(a) if a.child else main(a)
