"""Read-only provenance; collect outside SUT windows. Never records environment secrets."""
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

WORK=Path('/tmp/hqsb-capture-fair-20261017')
OLD=Path('/tmp/hqsb-capture-bench-20261017')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def command(args):
    result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
    return dict(argv=args,exit_code=result.returncode,output=result.stdout)

with (OLD/'measurement.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    paths=list((WORK/'build').glob('*'))+[OLD/'build/libcapture_nanolog.so',OLD/'upstream/NanoLog/runtime/decompressor']
    paths+=list((WORK/'dftracer-venv/lib').glob('python*/site-packages/dftracer.libs/*'))
    paths+=list((WORK/'lttng216').glob('lib/*.so*'))
    paths+=list((WORK/'lttng216/bin').glob('lttng*'))
    paths+=list((WORK/'lttng216').rglob('lttng-consumerd'))
    paths += [WORK/'tracezip-admission', WORK/'tracezip-admission-defaults',
              WORK/'probes/ownership-probe', WORK/'probes/ownership-probe-low-pressure',
              WORK/'loom-harness/target/release/libhqsb_loom_capture.so',
              Path('/usr/bin/lttng'),Path('/usr/bin/lttng-sessiond'),
              Path('/usr/lib/x86_64-linux-gnu/lttng/libexec/lttng-consumerd')]
    paths += list(Path('/usr/lib/x86_64-linux-gnu').glob('liblttng-ust*.so.*'))
    paths += list(Path('/usr/lib/x86_64-linux-gnu').glob('liburcu*.so.*'))
    paths += list((WORK/'upstream/perfetto-sdk').glob('perfetto.*'))
    artifacts={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths if p.is_file() and not p.is_symlink()}
    versions={}
    for name in ['quill','loom','tracezip','lttng-ust-216','lttng-tools-216']:
        path=WORK/'upstream'/name
        if path.exists():
            versions[name]=dict(head=command(['git','-c',f'safe.directory={path}','-C',str(path),'rev-parse','HEAD']),
                                status=command(['git','-c',f'safe.directory={path}','-C',str(path),'status','--porcelain']))
    commands=[['g++','--version'],['go','version'],['lttng','--version'],
              ['dpkg-query','-W','lttng-tools','liblttng-ust1','babeltrace2','python3-bt2']]
    data=dict(collected_at_ns=time.time_ns(),scope='post-build read-only provenance; campaign protocol separately freezes pre-run source/build hashes',
              artifacts=artifacts,upstream=versions,toolchain=[command(args) for args in commands],
              python=dict(executable=sys.executable,version=sys.version,
                          distributions=sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name'])))
    (WORK/'research/runtime-manifest.json').write_text(json.dumps(data,indent=2))
