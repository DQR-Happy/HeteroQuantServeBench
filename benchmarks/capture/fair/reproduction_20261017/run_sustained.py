"""Longer confirmations using frozen profiles; archive only outside timed windows."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import shutil

from benchmarks.capture.fair import run as runner
from benchmarks.capture.fair.campaign import ADAPTERS

WORK=Path('/tmp/hqsb-capture-fair-20261017')
ROOT=Path('/root/autodl-tmp/hqsb-capture-fair-20261017/results/sustained-v1')


def digest(stream):
    h=hashlib.sha256()
    for b in iter(lambda:stream.read(1024**2),b''):
        h.update(b)
    return h.hexdigest()


def archive(directory, extra_paths=()):
    """Lossless evidence preservation, not part of the measured encoder."""
    metric=directory/'metrics.json'
    if not metric.exists():return
    records=[]
    row=json.loads(metric.read_text())
    paths={directory/a['path']:a['sha256'] for a in row.get('encoded_artifacts',[])}
    paths.update({p:None for p in directory.glob('receipts-*.bin')})
    paths.update({p:None for p in extra_paths})
    manifest=directory/'raw-compression-manifest.json'
    for p,expected in paths.items():
        if p.stat().st_size<65536:continue
        with p.open('rb') as stream:original=digest(stream)
        if expected and original!=expected:raise ValueError('changed raw artifact')
        z=p.with_name(p.name+'.zst')
        subprocess.run(['zstd','-q','-T2','-1','--check','-o',str(z),'--',str(p)],check=True)
        decoder=subprocess.Popen(['zstd','-q','-d','-c','--',str(z)],stdout=subprocess.PIPE)
        restored=digest(decoder.stdout)
        if decoder.wait() or restored!=original:raise ValueError('lossless archive mismatch')
        with z.open('rb') as stream:encoded=digest(stream)
        records.append(dict(original=str(p.relative_to(directory)),archive=str(z.relative_to(directory)),
                            original_bytes=p.stat().st_size,original_sha256=original,
                            archive_bytes=z.stat().st_size,archive_sha256=encoded,decoded_sha256=restored))
        manifest.write_text(json.dumps(dict(format='post-measurement zstd; not SUT encoding',files=records),indent=2))
        p.unlink()


def main():
    os.sched_setaffinity(0,{196})
    with Path('/tmp/hqsb-capture-bench-20261017/measurement.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not shutil.which('zstd'):raise RuntimeError('zstd must be on PATH before any measurement')
        ROOT.mkdir(exist_ok=False)
        selected=json.loads((ROOT.parent/'campaign-v1/selected-profiles.json').read_text())
        cases=[dict(id='sustained-30s-10k',workload='S30',rate=10000,duration_s=30,producers=1,body_bytes=128),
               dict(id='sustained-10s-100k-p4',workload='S10',rate=100000,duration_s=10,producers=4,body_bytes=128)]
        protocol=dict(endpoint='FULL5/D2',cases=cases,selected_profiles=selected,
                      tuning='reuse frozen campaign-v1 profiles; no extra tuning',
                      test_seeds=[100,101,102],source_gate='each case 2x rate, three controls; no unattempted and <=1% >1ms dispatch lag',
                      purpose='10/30s confirmation; not saturation search or long-term soak',
                      resource_budget='same CPU4 / aggregate AS1GiB / stack2MiB / arena2',
                      postprocessing='zstd archive only after endpoint + oracle; decode SHA equality required')
        (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
        controls=[]
        for case in cases:
            for repeat in range(3):
                print('control',case['id'],repeat,flush=True)
                label=f'control-{case["id"]}-r{repeat}'
                row=runner.execute(WORK,ROOT,dict(case,rate=case['rate']*2),'null',1,repeat,'throughput',label)
                controls.append(row)
                (ROOT/'controls.json').write_text(json.dumps(controls,indent=2))
                archive(ROOT/label)
        gates={c['id']:all(r['source_gate_pass'] for r in controls if r['case_id']==c['id']) for c in cases}
        (ROOT/'source-gates.json').write_text(json.dumps(gates,indent=2))
        jobs=[(c,a,r) for c in cases for a in ADAPTERS for r in range(3)]
        random.Random(20261017).shuffle(jobs)
        rows=[]
        for i,(case,adapter,repeat) in enumerate(jobs):
            p=selected[adapter][0];label=f'{i:03d}-{case["id"]}-{adapter}-p{p}-r{repeat}'
            print(i+1,len(jobs),label,flush=True)
            try:
                row=runner.execute(WORK,ROOT,case,adapter,p,repeat,'throughput',label)
            except Exception as exc:
                row=dict(case_id=case['id'],adapter=adapter,profile=p,status='failed',repeat=repeat,
                         error=f'{type(exc).__name__}: {exc}')
                (ROOT/label/'failure.json').write_text(json.dumps(row,indent=2))
            rows.append(row)
            (ROOT/'all-runs.json').write_text(json.dumps(rows,indent=2))
            archive(ROOT/label)
        (ROOT/'completed.json').write_text(json.dumps(dict(runs=len(rows))))


if __name__=='__main__':main()
