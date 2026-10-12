"""Corrected matched-budget LTTng2.14/2.16; preserve earlier diagnostic results."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import signal
import shutil
import subprocess
import time
import uuid

from benchmarks.capture.fair import run as runner
from benchmarks.capture.fair.campaign import cases
from benchmarks.capture.fair.campaign import fingerprint
from run_sustained import archive

WORK = Path('/tmp/hqsb-capture-fair-20261017')
TARGET = os.environ.get('HQSB_LTTNG_TARGET','216')
if TARGET not in {'214','216'}:raise ValueError('HQSB_LTTNG_TARGET must be214 or216')
ROOT = Path('/root/autodl-tmp/hqsb-capture-fair-20261017/results') / f'lttng{TARGET}-v3'
if os.environ.get('HQSB_LTTNG_SMOKE')=='1':ROOT=ROOT.with_name(ROOT.name+'-smoke')
PREFIX = WORK / 'lttng216' if TARGET=='216' else Path('/usr')
LIBRARY = WORK/'lttng216/lttng216.so' if TARGET=='216' else WORK/'build/lttng.so'
PROFILES = [('per-cpu', 8192), ('per-channel', 1048576), ('per-channel', 4194304)] if TARGET=='216' else [('per-cpu',8192),('per-cpu',32768),('per-cpu',131072)]
STATE = {}


def descendants(roots):
    ids=set(roots);frontier=list(ids)
    while frontier:
        pid=frontier.pop();children=Path(f'/proc/{pid}/task/{pid}/children')
        try:values=children.read_text().split()
        except OSError:continue
        for child in map(int,values):
            if child not in ids:ids.add(child);frontier.append(child)
    return ids


def bounded_cli(args,env,directory,label,cpu):
    with (directory/(label+'.log')).open('w') as log:
        return subprocess.run(runner.bounded_command(['lttng',*args],384,cpu),env=env,stdout=log,
                              stderr=subprocess.STDOUT,timeout=30,check=True)


def start(directory, env, cpu, profile):
    home = Path('/tmp') / ('h216-' + uuid.uuid4().hex[:12])
    home.mkdir()
    env.update(PATH=str(PREFIX/'bin')+':'+os.environ['PATH'],
               LTTNG_HOME=str(home),
               LTTNG_RUNDIR=str(home), LTTNG_UST_CTL_PATH=str(home),
               LTTNG_UST_APP_PATH=str(home), LTTNG_UST_REGISTER_TIMEOUT='10000')
    (directory/'lttng-rundir.txt').write_text(str(home))
    env.pop('LTTNG_DEBUG_NOCLONE',None)
    if TARGET=='216':env['LD_LIBRARY_PATH']=str(PREFIX/'lib')
    else:env.pop('LD_LIBRARY_PATH',None)
    consumers = list((PREFIX/'lib/lttng/libexec').glob('lttng-consumerd')) if TARGET=='216' else [Path('/usr/lib/x86_64-linux-gnu/lttng/libexec/lttng-consumerd')]
    if len(consumers) != 1:
        raise RuntimeError(f'expected one isolated consumer, found {consumers}')
    wrapper = directory/'consumer-wrapper.sh'
    wrapper.write_text('#!/bin/sh\nset -e\nulimit -v 262144\nulimit -s 2048\nexec '+str(consumers[0])+' "$@"\n')
    wrapper.chmod(0o700)
    with (directory/'sessiond.log').open('w') as log:
        process = subprocess.Popen(runner.bounded_command(
            [str(PREFIX/'bin/lttng-sessiond'), '--no-kernel', '--consumerd64-path='+str(wrapper)],
            256, cpu), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(250):
            if list(home.rglob('client-lttng-sessiond')):
                break
            if process.poll() is not None:
                raise RuntimeError('isolated sessiond exited')
            time.sleep(.02)
        env['LTTNG_NO_SESSIOND'] = '1'
        bounded_cli(['create','hqsb',*(['--trace-format=ctf-1.8'] if TARGET=='216' else []),'--output='+str(directory/'ctf')],env,directory,'create',cpu)
        allocation, size = PROFILES[profile]
        bounded_cli(['enable-channel','--userspace',*(['--buffer-ownership=process',
                    '--buffer-allocation='+allocation,'--buffer-preallocation=on-demand'] if TARGET=='216' else ['--buffers-pid']),
                    '--subbuf-size='+str(size),'--num-subbuf=4','--discard','capture'],
                   env,directory,'channel',cpu)
        bounded_cli(['enable-event','--userspace','--channel=capture','hqsb_capture:event'],env,directory,'enable',cpu)
        bounded_cli(['start'],env,directory,'start',cpu)
        # consumerd daemonizes and may be reparented to PID1: PPID traversal
        # alone misses it. Its private control-socket directory proves ownership.
        roots=set()
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:
                args=(path/'cmdline').read_bytes().split(b'\0')
                comm=(path/'comm').read_text().strip()
            except OSError:continue
            if args and Path(args[0].decode(errors='replace')).name=='lttng-consumerd' and any(str(home).encode() in a for a in args) and 'runas' not in comm:
                roots.add(int(path.name))
        if len(roots)!=1:raise RuntimeError(f'expected one owned detached consumer, found {roots}')
        family=descendants({process.pid});ids=descendants(family|roots);limits={};helpers=[]
        for pid in ids:
            comm=Path(f'/proc/{pid}/comm').read_text().strip()
            helper='runas' in comm
            ceiling=(64 if helper else 256)*1024**2
            if helper:
                status=Path(f'/proc/{pid}/status').read_text()
                size=int(next(l for l in status.splitlines() if l.startswith('VmSize:')).split()[1])*1024
                if size>ceiling:raise RuntimeError(f'helper {pid} VmSize {size} exceeds declared64MiB')
                resource.prlimit(pid,resource.RLIMIT_AS,(ceiling,ceiling));helpers.append(pid)
            if resource.prlimit(pid,resource.RLIMIT_AS)!=(ceiling,ceiling):raise RuntimeError('unexpected actual AS limit')
            command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            limits[str(pid)]=dict(AS_bytes=ceiling,comm=comm,command=command)
        if len(helpers)>2 or len(ids)-len(helpers)!=2:raise RuntimeError(f'unexpected owned background process set {limits}')
        if 384*1024**2+sum(v['AS_bytes'] for v in limits.values())>1024**3:raise RuntimeError('aggregate AS budget exceeded')
        process.hqsb_detached_pids=sorted(ids-family)
        STATE.update(daemon_pid=process.pid,extra_pids=process.hqsb_detached_pids,extra_samples=[])
        (directory/'verified-daemon-limits.json').write_text(json.dumps(limits,indent=2))
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            process.wait(timeout=5)
        raise
    return process


def execute_matched(work,output,case,adapter,profile,repeat,measurement_pass,label):
    """Account detached consumers using the existing observer's sampling ticks."""
    STATE.clear()
    original_start,original_sample,original_cli=runner.start_lttng,runner.sample,runner.cli
    def sample(pid):
        result=original_sample(pid)
        if pid==STATE.get('daemon_pid'):
            STATE['extra_samples'].append({str(p):original_sample(p) for p in STATE.get('extra_pids',[])})
        return result
    runner.start_lttng,runner.sample,runner.cli=start,sample,bounded_cli
    try:row=runner.execute(work,output,case,adapter,profile,repeat,measurement_pass,label)
    finally:
        runner.start_lttng,runner.sample,runner.cli=original_start,original_sample,original_cli
        (output/label/'detached-process-samples.json').write_text(json.dumps(STATE.get('extra_samples',[])))
    timeline=json.loads((output/label/'process-tree-timeline.json').read_text())
    extras=STATE.get('extra_samples',[])
    if len(extras)!=len(timeline):raise RuntimeError('observer tick alignment differs')
    for tick,extra in zip(timeline,extras):tick['processes'].update(extra)
    (output/label/'complete-process-tree-timeline.json').write_text(json.dumps(timeline))
    row['original_parent_tree_PSS_KiB']=row['sampled_peak_tree_PSS_KiB']
    row['sampled_peak_tree_PSS_KiB']=max((sum(v.get('smaps_rollup:Pss',0) for v in t['processes'].values()) for t in timeline),default=None)
    row['process_accounting']='registered detached consumer plus its helpers; actual process snapshots merged by observer tick; sampled during native-worker lifetime'
    (output/label/'metrics.json').write_text(json.dumps(row,indent=2))
    return row


def main():
    os.sched_setaffinity(0,{196})
    with Path('/tmp/hqsb-capture-bench-20261017/measurement.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not shutil.which('zstd'):raise RuntimeError('zstd must be on PATH before any measurement; archive is postprocessing')
        if not LIBRARY.exists():raise RuntimeError('required real provider build unavailable')
        ROOT.mkdir(exist_ok=False)
        shim=WORK/f'work{TARGET}v2'
        (shim/'build').mkdir(parents=True,exist_ok=True)
        if not (shim/'source').exists():(shim/'source').symlink_to(WORK/'source',target_is_directory=True)
        for binary in (WORK/'build').iterdir():
            if binary.is_file():
                link=shim/'build'/binary.name
                if not link.exists():link.symlink_to(LIBRARY if binary.name=='lttng.so' else binary)
        runner.start_lttng=start
        protocol=dict(version='LTTng tools/UST stable2.16.0' if TARGET=='216' else 'LTTng tools2.14.1/UST2.14.0', endpoint='FULL5/D2',
                      trace_format='CTF1.8, matched to main campaign',
                      profiles=PROFILES, allocation='on-demand; process ownership; discard; 4 subbuffers',
                      budget='same 4 logical CPUs; AS app384+sessiond256+consumer256+up to2 runas helpers64MiB each; stack2MiB; arena2; sum<=1024MiB',
                      run_as='normal upstream forked helpers; lower only helper AS ceilings after checking actual VmSize; private-socket consumer registry handles reparenting',
                      tuning_seeds=[0,1],test_seeds=[100,101,102],
                      selection='failed/corrupt; mean scheduled loss; H+drain; lifecycle CPU; PSS; profile ID',
                      source_gates=str(ROOT.parent/'campaign-v1/source-gates.json'),
                      frozen_main_harness_sha256=fingerprint(WORK),
                      sha256={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [LIBRARY,WORK/'build/capture-fair',Path(__file__)]})
        (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
        all_cases=cases()
        if os.environ.get('HQSB_LTTNG_SMOKE')=='1':
            row=execute_matched(shim,ROOT,dict(all_cases[0],duration_s=.2),'lttng',1,0,'throughput','smoke')
            if row['status']!='complete':raise RuntimeError('matched-budget smoke did not preserve full event cohort')
            (ROOT/'completed.json').write_text(json.dumps(row,indent=2))
            return
        def trial(case,p,repeat,pass_name,label):
            try:
                return execute_matched(shim,ROOT,case,'lttng',p,repeat,pass_name,label)
            except Exception as exc:
                row=dict(adapter='lttng',case_id=case['id'],profile=p,status='failed',repeat=repeat,error=f'{type(exc).__name__}: {exc}')
                (ROOT/label/'failure.json').write_text(json.dumps(row,indent=2))
                return row
        tune=[(dict(all_cases[i],duration_s=.5,seed=s),p) for i in (0,2,11) for s in (0,1) for p in range(3)]
        random.Random(20261017).shuffle(tune)
        rows=[]
        for i,(case,p) in enumerate(tune):
            print('tune',i,case['id'],p,flush=True)
            rows.append(trial(case,p,0,'throughput',f'tune-{i:03d}-p{p}'))
            (ROOT/'tuning.json').write_text(json.dumps(rows,indent=2))
        scores={}
        for p in range(3):
            rs=[r for r in rows if r['profile']==p];ok=[r for r in rs if r['status'] in {'complete','lossy'}]
            scores[p]=[len(rs)-len(ok),sum(1-r['scheduled_delivery_ratio'] for r in ok)/max(1,len(ok)),
                       sum(r['H_plus_drain_s'] for r in ok),sum(r['lifecycle_children_cpu_s'] for r in ok),
                       max((r['sampled_peak_tree_PSS_KiB'] or 0 for r in ok),default=0),p]
        chosen=min(scores,key=scores.get)
        (ROOT/'selected-profile.json').write_text(json.dumps(dict(profile=chosen,scores=scores),indent=2))
        tests=[(c,r,'throughput') for c in all_cases for r in range(3)]
        tests += [(all_cases[i],r,'latency') for i in (1,3,7) for r in range(3)]
        random.Random(20261017).shuffle(tests)
        rows=[]
        for i,(case,r,pass_name) in enumerate(tests):
            print('test',i,len(tests),case['id'],r,pass_name,flush=True)
            rows.append(trial(case,chosen,r,pass_name,f'test-{i:03d}-{case["id"]}-{pass_name}-r{r}'))
            (ROOT/'all-runs.json').write_text(json.dumps(rows,indent=2))
        sustained=ROOT/'sustained';sustained.mkdir()
        long_cases=[dict(id='sustained-30s-10k',workload='S30',rate=10000,duration_s=30,producers=1,body_bytes=128),
                    dict(id='sustained-10s-100k-p4',workload='S10',rate=100000,duration_s=10,producers=4,body_bytes=128)]
        long_rows=[]
        for case in long_cases:
            for repeat in range(3):
                label=f'{case["id"]}-r{repeat}'
                print('sustained',label,flush=True)
                try:row=execute_matched(shim,sustained,case,'lttng',chosen,repeat,'throughput',label)
                except Exception as exc:
                    row=dict(case_id=case['id'],adapter='lttng',profile=chosen,status='failed',error=f'{type(exc).__name__}: {exc}')
                    (sustained/label/'failure.json').write_text(json.dumps(row,indent=2))
                long_rows.append(row)
                (sustained/'all-runs.json').write_text(json.dumps(long_rows,indent=2))
                archive(sustained/label)
        (ROOT/'completed.json').write_text(json.dumps(dict(runs=len(rows))))


if __name__=='__main__':
    main()
