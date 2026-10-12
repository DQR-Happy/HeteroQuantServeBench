"""Real upstream codec admission; no network capacity or D2 claims."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

WORK=Path('/tmp/hqsb-capture-fair-20261017')
ROOT=Path('/root/autodl-tmp/hqsb-capture-fair-20261017/results/tracezip-admission-v1')

def main():
    with Path('/tmp/hqsb-capture-bench-20261017/measurement.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        binary=WORK/'tracezip-admission'
        if not binary.exists():
            raise RuntimeError('upstream build unavailable; inspect tracezip-build.log')
        ROOT.mkdir(exist_ok=False)
        protocol=dict(stock_commit='c0f23cf663d83d3c0a44ba22df33a016ec0c0318',
                      inputs='512 complete 5-field spans; real upstream ptrace codec and receiver handlers',
                      endpoint='finite batch decode; not D2; no network sender',
                      resource_budget='4 logical CPUs; GOMAXPROCS4; GOMEMLIMIT512MiB soft GC target (not RSS limit)',
                      modes=['otlp-json','tracezip','tracezip-gzip'],shapes=['regular','ns64','wide'],
                      profiles=[[256,16,1000],[1024,100,10000],[4096,256,100000]],
                      binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      gate='all unique canonical fields and span timestamps preserved exactly; failed admissions excluded from speed ranking')
        (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
        rows=[]
        for shape in protocol['shapes']:
            for mode in protocol['modes']:
                for p in ([0] if mode=='otlp-json' else range(3)):
                    directory=ROOT/f'{shape}-{mode}-p{p}'
                    directory.mkdir()
                    env=dict(os.environ,HQSB_SHAPE=shape,HQSB_MODE=mode,HQSB_PROFILE=str(p),
                             HQSB_OUTPUT=str(directory),GOMAXPROCS='4',GOMEMLIMIT='512MiB')
                    with (directory/'native.log').open('w') as log:
                        try:
                            done=subprocess.run(['taskset','-c','192-195',str(binary),'-test.run','^TestHQSBAdmission$','-test.v'],
                                                cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                            rc=done.returncode
                        except subprocess.TimeoutExpired:
                            rc=124
                    row=json.loads((directory/'metrics.json').read_text()) if (directory/'metrics.json').exists() else dict(mode=mode,shape=shape,profile=p)
                    row.update(exit_code=rc,status='pass' if rc==0 else 'semantic_or_runtime_fail')
                    rows.append(row)
                    print(shape,mode,p,row['status'],row.get('valid_unique_5fields'),flush=True)
                    (ROOT/'all-runs.json').write_text(json.dumps(rows,indent=2))
        (ROOT/'completed.json').write_text(json.dumps(dict(runs=len(rows))))

if __name__=='__main__':main()
