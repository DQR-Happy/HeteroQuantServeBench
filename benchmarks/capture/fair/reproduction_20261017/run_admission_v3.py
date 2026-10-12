"""Stock-library ownership, UTF-8, zero and uint64 ABI probes; not throughput."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess

from benchmarks.capture.fair.decode import events
from benchmarks.capture.fair.run import EXTENSIONS, bounded_command, cli
from wire_semantics import quill_json_text

WORK=Path('/tmp/hqsb-capture-fair-20261017')
OLD=Path('/tmp/hqsb-capture-bench-20261017')
ROOT=Path('/root/autodl-tmp/hqsb-capture-fair-20261017/results/stock-admission-v3')


def main():
    with (OLD/'measurement.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ROOT.mkdir(exist_ok=False)
        (WORK/'probes').mkdir(exist_ok=True)
        subprocess.run(['g++','-std=c++17','-O3','-DNDEBUG',str(WORK/'research/ownership_probe.cc'),
                        '-ldl','-o',str(WORK/'probes/ownership-probe')],check=True)
        expected=[]
        for i in range(256):
            base=[0,2**32+1,2**53+1,2**63+1][i%4]
            seq=base+i*512
            body=json.dumps(dict(name='推理层µ',address=2**64-1,data='☃|literal:'+r'\xE6\\xFF'+'x'*1024),ensure_ascii=False,separators=(',',':'))
            expected.append(dict(run_id='run_probe',producer_id=i%4,source_seq=seq,source_timestamp=0 if i==0 else seq+123,body=body))
        (ROOT/'expected.json').write_text(json.dumps(expected,ensure_ascii=False,indent=2))
        with (ROOT/'input.bin').open('wb') as stream:
            stream.write(struct.pack('<I',len(expected)))
            for row in expected:
                data=row['body'].encode()
                stream.write(struct.pack('<IQQI',row['producer_id'],row['source_seq'],row['source_timestamp'],len(data)))
                stream.write(data)
        libs=dict(nanolog=OLD/'build/libcapture_nanolog.so',hindsight=WORK/'build/hindsight.so',
                  quill=WORK/'build/quill0.so',perfetto=WORK/'build/perfetto.so',
                  dftracer=WORK/'build/dftracer.so',hqsb=WORK/'build/python_bridge.so',otel=WORK/'build/python_bridge.so',
                  lttng214=WORK/'build/lttng.so',lttng216=WORK/'lttng216/lttng216.so')
        sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
        (ROOT/'protocol.json').write_text(json.dumps(dict(scope='finite semantic admission, not performance',
            expected_records=len(expected),input_sha256=sha(ROOT/'input.bin'),
            probe_source_sha256=sha(WORK/'research/ownership_probe.cc'),probe_binary_sha256=sha(WORK/'probes/ownership-probe'),
            library_sha256={name:sha(path) for name,path in libs.items()},
            canonicalization='Perfetto SQL signed int64 reinterpreted as uint64; Quill sanitizer hex escapes inverted within compact JSON domain; raw stock output preserved',
            wire_decoder_sha256=sha(WORK/'research/wire_semantics.py')),indent=2))
        rows=[]
        for name,lib in libs.items():
            directory=ROOT/name;directory.mkdir()
            adapter='lttng' if name.startswith('lttng') else name
            case=dict(id='ownership-uint64',workload='ADMISSION',adapter=adapter,run_id='run_probe',seed=100,body_bytes=1024)
            (directory/'config.json').write_text(json.dumps(case,indent=2))
            env=dict(os.environ,MALLOC_ARENA_MAX='2',HQSB_FAIR_PROFILE='1',HQSB_FAIR_ADAPTER=adapter,
                     PYTHONPATH=str(WORK/'source')+':'+str(OLD/'venv/lib/python3.12/site-packages'))
            daemon=None;row=dict(adapter=name,endpoint='D2' if adapter!='lttng' else 'D2-external-stop',scheduled=len(expected))
            try:
                if adapter=='lttng':
                    os.environ['HQSB_LTTNG_TARGET']=name[-3:]
                    spec=importlib.util.spec_from_file_location('admission_'+name,WORK/'research/run_lttng_v3.py')
                    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                    daemon=module.start(directory,env,193,1)
                with (directory/'native.log').open('w') as log:
                    subprocess.run(bounded_command([str(WORK/'probes/ownership-probe'),str(lib),
                                    str(directory/EXTENSIONS[adapter]),str(ROOT/'input.bin'),str(directory)],
                                    384 if adapter=='lttng' else 1024,'192-195'),cwd=directory,env=env,
                                   stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
                if daemon:
                    cli(['stop'],env,directory,'stop',193);cli(['destroy','hqsb'],env,directory,'destroy',193)
                    for path in (directory/'ctf').rglob('*'):
                        if path.is_file():
                            with path.open('rb') as stream:os.fsync(stream.fileno())
                os.environ['HQSB_TRACE_PROCESSOR']=str(WORK/'build/trace_processor_shell')
                actual=list(events(directory,adapter,OLD/'upstream'))
                (directory/'decoded-stock.json').write_text(json.dumps(actual,ensure_ascii=False,indent=2))
                # The SQL public integer type is signed int64; canonical fields
                # are explicitly uint64. Reinterpret their original bit pattern.
                mapped=0
                if adapter=='quill':
                    for record in actual:record['body']=quill_json_text(record['body'])
                if adapter=='perfetto':
                    for record in actual:
                        for field in ('source_seq','source_timestamp'):
                            if record[field]<0:record[field]+=2**64;mapped+=1
                previous={};order_errors=0
                for record in actual:
                    p=record['producer_id'];i=record['source_seq']
                    order_errors+=i<=previous.get(p,-1);previous[p]=i
                key=lambda record:(record['producer_id'],record['source_seq'])
                passed=json.dumps(sorted(actual,key=key),sort_keys=True)==json.dumps(sorted(expected,key=key),sort_keys=True) and not order_errors
                row.update(status='pass' if passed else 'semantic_fail',decoded=len(actual),ordering_errors=order_errors,
                           perfetto_signed_SQL_fields_reinterpreted_as_uint64=mapped,
                           observed='caller run/body storage overwritten immediately after every return; zero, >2^32, >2^53, >2^63 integer fields; UTF-8 and pipe in JSON text')
            except Exception as exc:
                row.update(status='runtime_or_decoder_fail',error=f'{type(exc).__name__}: {exc}')
            finally:
                if daemon and daemon.poll() is None:
                    os.killpg(daemon.pid,signal.SIGTERM)
                    try:daemon.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(daemon.pid,signal.SIGKILL);daemon.wait()
            (directory/'result.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
            rows.append(row);(ROOT/'all-probes.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
            print(name,row['status'],flush=True)
        (ROOT/'completed.json').write_text(json.dumps(dict(probes=len(rows))))


if __name__=='__main__':main()
