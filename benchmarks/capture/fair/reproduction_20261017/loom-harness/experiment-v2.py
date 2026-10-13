"""Actual Loom Q-sync track; query-export evidence is NOT durable capture throughput."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import time
import uuid
from benchmarks.capture.fair.run import input_files
from benchmarks.capture.fair.campaign import cases
from benchmarks.capture.oracle import verify
from benchmarks.capture.run import sample
from run_sustained import archive

WORK = Path('/tmp/hqsb-capture-fair-20261017')
ROOT = Path('/root/autodl-tmp/hqsb-capture-fair-20261017/results/loom-q-v2')
HEADER = struct.Struct('<64sIQQI')

def decoded(directory, adapter, upstream):
    for path in sorted(directory.glob('loom-source-*.bin')):
        records = []
        with path.open('rb') as stream:
            while prefix := stream.read(4):
                length = struct.unpack('<I', prefix)[0]
                block = stream.read(length)
                if length != len(block) or length < HEADER.size:
                    raise ValueError('truncated Loom query result')
                run, p, i, ts, n = HEADER.unpack_from(block)
                if n != length-HEADER.size:
                    raise ValueError('Loom query record size')
                records.append(dict(run_id=run.rstrip(b'\0').decode(), producer_id=p,
                                    source_seq=i, source_timestamp=ts,body=block[HEADER.size:].decode()))
        # Public raw_scan traverses each source in reverse insertion order.
        # The Rust wrapper separately checks that this returned order is descending.
        yield from reversed(records)

def run(case, profile, label, sampling=0):
    directory = ROOT / label
    directory.mkdir()
    case = dict(case, run_id='run_'+uuid.uuid4().hex, adapter='loom', warmup_events=20,
                measurement_pass='latency' if sampling else 'throughput')
    (directory/'config.json').write_text(json.dumps(case,indent=2))
    source=input_files(ROOT, {k:v for k,v in case.items() if k not in {'run_id','adapter','measurement_pass'}})
    env=dict(os.environ,MALLOC_ARENA_MAX='2',HQSB_FAIR_PROFILE=str(profile))
    binary=WORK/'loom-harness/target/release/libhqsb_loom_capture.so'
    cmd=['taskset','-c','192-195','prlimit','--as=1073741824','--stack=2097152','--',str(WORK/'build/capture-fair'),str(binary),str(directory/'loom-native'),str(directory),case['run_id'],str(source/'offsets.bin'),str(source/'bodies.bin'),str(case.get('producers',1)),str(int(case['duration_s']*1e9)),str(sampling),str(case['seed']),str(int(case.get('skew',False))),'20']
    timeline=[]
    with (directory/'native.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+case['duration_s']+60
        while process.poll() is None:
            timeline.append(dict(ns=time.monotonic_ns(),sample=sample(process.pid)))
            if time.monotonic()>deadline:
                process.kill();process.wait();raise TimeoutError('Loom bounded close/query deadline')
            time.sleep(.02)
        if process.wait():raise RuntimeError('Loom worker failed')
    raw=json.loads((directory/'native.json').read_text())
    boundaries=json.loads((directory/'loom-boundaries.json').read_text())
    raw.update(adapter='loom',case=case,track='LOOM_QUERYABLE',acknowledgement='unavailable_void_api',
        errors=['adapter_error'] if raw.pop('adapter_errors') or boundaries['reverse_order_errors'] else [],
        endpoint_end_ns=boundaries['q_end_ns'],encoded_output_files=[p.name for p in directory.glob('loom-source-*.bin')],
        terminal=dict(endpoint='Q-sync',close_start_ns=raw['close_start_ns'],close_end_ns=boundaries['q_end_ns']))
    (directory/'worker.json').write_text(json.dumps(raw,indent=2))
    (directory/'process-timeline.json').write_text(json.dumps(timeline))
    row=verify(directory,Path('/tmp/hqsb-capture-bench-20261017/upstream'),decode_events=decoded)
    row.update(profile=profile,query_export_ms=boundaries['query_export_ns']/1e6,
        native_storage_apparent_bytes=sum(p.stat().st_size for p in (directory/'loom-native').rglob('*') if p.is_file()),
        native_storage_allocated_bytes=sum(p.stat().st_blocks*512 for p in (directory/'loom-native').rglob('*') if p.is_file()),
        sampled_peak_process_PSS_KiB=max((t['sample'].get('smaps_rollup:Pss',0) for t in timeline),default=0),
        CPU_scope='native cpu_seconds includes final query/export; not Q-only CPU cost',
        encoded_bytes_scope='query export verification sidecars; not Loom persisted representation')
    row['encoding_space_ratio']=None
    (directory/'metrics.json').write_text(json.dumps(row,indent=2))
    archive(directory,extra_paths=[p for p in (directory/'loom-native').rglob('*') if p.is_file()])
    return row

def main():
    ROOT.mkdir(exist_ok=False)
    os.sched_setaffinity(0,{196})
    protocol=dict(endpoint='Q-sync, not D2',stock_commit='fb0f7e222b9665c11d944b499eed954037e9a7f8',
        replaced_interrupted='loom-q-v1-interrupted: tuning may overlap storage relocation, retained but entirely excluded; v2 repeats all tuning and held-out trials',
        wrapper='bounded std::sync::mpsc ingress to one stock Partitions owner; 16 sources; one 100-bin u64 index per source',
        queue_profiles=[256,4096,65536],aggregate_AS_MiB=1024,cpus=[192,193,194,195],stack_MiB=2,MALLOC_ARENA_MAX=2,
        max_record_bytes=65500,excluded_cases='64KiB/2MiB snapshot and mixed giant records do not fit stock 64KiB chunks',
        source_gate_campaign=str(ROOT.parent/'campaign-v1/source-gates.json'),
        tuning_seeds=[0,1],test_seeds=[100,101,102],
        selection='fewest failed/corrupt; minimum scheduled loss; minimum sum H+Q drain; lowest profile ID',
        sha256={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [WORK/'build/capture-fair',WORK/'loom-harness/target/release/libhqsb_loom_capture.so',WORK/'loom-harness/src/lib.rs',WORK/'loom-harness/Cargo.lock']})
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    all_cases=cases();rows=[]
    with Path('/tmp/hqsb-capture-bench-20261017/measurement.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        training=[(dict(all_cases[i],duration_s=.5,seed=s),p) for i in (0,2,11) for s in (0,1) for p in range(3)]
        random.Random(20261017).shuffle(training)
        for index,(case,p) in enumerate(training):
            print('tune',index,p,case['id'],flush=True)
            label=f'tune-{index:03d}-p{p}'
            try:row=run(case,p,label)
            except Exception as exc:row=dict(status='failed',profile=p,error=f'{type(exc).__name__}: {exc}')
            rows.append(row);(ROOT/'tuning.json').write_text(json.dumps(rows,indent=2))
        scores={p:[sum(r['status'] not in {'complete','lossy'} for r in rows if r['profile']==p),
            sum(1-r.get('scheduled_delivery_ratio',0) for r in rows if r['profile']==p),
            sum(r.get('H_plus_drain_s',60) for r in rows if r['profile']==p),p] for p in range(3)}
        chosen=min(scores,key=scores.get)
        (ROOT/'selected-profile.json').write_text(json.dumps(dict(profile=chosen,scores=scores),indent=2))
        tests=[(dict(all_cases[i],seed=s),s) for i in (1,2,3,5,11) for s in (100,101,102)]
        random.Random(20261017).shuffle(tests)
        rows=[]
        for index,(case,s) in enumerate(tests):
            print('test',index,chosen,case['id'],s,flush=True)
            try:row=run(case,chosen,f'test-{index:03d}-{case["id"]}-s{s}',sampling=16)
            except Exception as exc:row=dict(status='failed',profile=chosen,case_id=case['id'],seed=s,error=f'{type(exc).__name__}: {exc}')
            rows.append(row);(ROOT/'all-runs.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'completed.json').write_text(json.dumps(dict(runs=len(rows))))
if __name__=='__main__':main()
