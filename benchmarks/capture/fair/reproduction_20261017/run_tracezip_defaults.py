"""Additional correctness check of author factory defaults, no speed ranking."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

WORK=Path('/tmp/hqsb-capture-fair-20261017')
ROOT=Path('/root/autodl-tmp/hqsb-capture-fair-20261017/results/tracezip-defaults-v1')

with Path('/tmp/hqsb-capture-bench-20261017/measurement.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    ROOT.mkdir(exist_ok=False)
    binary=WORK/'tracezip-admission-defaults'
    env=dict(os.environ,GOPROXY='https://goproxy.cn,direct',GOMAXPROCS='4',GOPATH=str(WORK/'go'),GOCACHE=str(WORK/'go-cache'))
    import shutil
    shutil.copyfile(WORK/'research/tracezip_admission_test.go',WORK/'upstream/tracezip/otel-compressor/prefix-compressed-receiver/hqsb_admission_test.go')
    with (ROOT/'build.log').open('w') as log:
        subprocess.run(['go','test','-p','4','-c','-o',str(binary)],env=env,
                       cwd=WORK/'upstream/tracezip/otel-compressor/prefix-compressed-receiver',stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    protocol=dict(component='unmodified upstream TraceZip pdata + actual receiver handlers',factory_defaults=[30000,100,50000],
                  factory_source='prefix-compressed-exporter/factory.go:47-53; DeleteResource=true also tested',
                  baseline='tracezip-admission-v1: stock OTLP JSON passed all three shapes',
                  scope='semantic robustness only; not additional throughput tuning',
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    for shape in ['regular','ns64','wide']:
        for mode in ['tracezip','tracezip-gzip']:
            for delete in ['0','1']:
                directory=ROOT/f'{shape}-{mode}-delete{delete}';directory.mkdir()
                trial_env=dict(env,HQSB_MODE=mode,HQSB_SHAPE=shape,HQSB_PROFILE='3',HQSB_DELETE_RESOURCE=delete,
                               HQSB_OUTPUT=str(directory),GOMEMLIMIT='512MiB')
                with (directory/'native.log').open('w') as log:
                    done=subprocess.run(['taskset','-c','192-195',str(binary),'-test.run','^TestHQSBAdmission$','-test.v'],
                                        env=trial_env,cwd=directory,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                row=json.loads((directory/'metrics.json').read_text()) if (directory/'metrics.json').exists() else dict(shape=shape,mode=mode)
                row.update(delete_resource=delete=='1',exit_code=done.returncode,status='pass' if done.returncode==0 else 'semantic_or_runtime_fail')
                rows.append(row);(ROOT/'all-runs.json').write_text(json.dumps(rows,indent=2))
                print(shape,mode,delete,row['status'],row.get('valid_unique_5fields'),flush=True)
    (ROOT/'completed.json').write_text(json.dumps(dict(runs=len(rows))))
