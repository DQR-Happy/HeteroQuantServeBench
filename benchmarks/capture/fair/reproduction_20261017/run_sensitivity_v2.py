"""Equal-profile high-load response sweep; never retunes the frozen main result."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import random

from benchmarks.capture.fair import run as runner
from benchmarks.capture.fair.campaign import fingerprint, cases
from run_sustained import archive

WORK=Path('/tmp/hqsb-capture-fair-20261017')
ROOT=Path('/root/autodl-tmp/hqsb-capture-fair-20261017/results/profile-response-v2')


def load_lttng(target):
    os.environ['HQSB_LTTNG_TARGET']=target
    spec=importlib.util.spec_from_file_location('response_lttng'+target,WORK/'research/run_lttng_v3.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    shim=WORK/f'work{target}v2'
    (shim/'build').mkdir(parents=True,exist_ok=True)
    if not (shim/'source').exists():(shim/'source').symlink_to(WORK/'source',target_is_directory=True)
    for path in (WORK/'build').iterdir():
        link=shim/'build'/path.name
        if path.is_file() and not link.exists():link.symlink_to(module.LIBRARY if path.name=='lttng.so' else path)
    return module,shim


def main():
    os.sched_setaffinity(0,{196})
    with Path('/tmp/hqsb-capture-bench-20261017/measurement.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ROOT.mkdir(exist_ok=False)
        variants=['nanolog','hindsight','quill','perfetto','dftracer','hqsb','otel','lttng214','lttng216']
        profiles={v:[1] if v=='nanolog' else [0,1,2] for v in variants}
        protocol=dict(purpose='high-load parameter response; all profiles displayed; main profiles/ranking unchanged',
                      source_case='steady-1m-p8; native source gates from campaign-v1 apply',seeds=[2000,2001,2002],profiles=profiles,
                      budget='same one collector CPU + three producer CPUs; AS1GiB; LTTng v3 app384+daemon256+consumer256+up to2 helpers64 each',
                      interpretation='not a new best-of-test leaderboard; knobs may change together; main training did not cover every overload/large-record condition',
                      frozen_source_and_binary_sha256=fingerprint(WORK))
        (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
        lttng={target:load_lttng(target) for target in ('214','216')}
        default_start=runner.start_lttng
        jobs=[(v,p,r) for v in variants for p in profiles[v] for r in range(3)]
        random.Random(20261017).shuffle(jobs)
        rows=[]
        for i,(variant,p,r) in enumerate(jobs):
            label=f'{i:03d}-{variant}-p{p}-r{r}'
            case=dict(cases()[3],seed=2000+r)
            adapter='lttng' if variant.startswith('lttng') else variant
            work=WORK
            if variant.startswith('lttng'):
                module,work=lttng[variant[-3:]];runner.start_lttng=module.start
            else:runner.start_lttng=default_start
            print(i+1,len(jobs),label,flush=True)
            try:
                if variant.startswith('lttng'):row=module.execute_matched(work,ROOT,case,adapter,p,r,'throughput',label)
                else:row=runner.execute(work,ROOT,case,adapter,p,r,'throughput',label)
            except Exception as exc:
                row=dict(adapter=adapter,case_id=case['id'],profile=p,status='failed',error=f'{type(exc).__name__}: {exc}')
                path=ROOT/label;path.mkdir(exist_ok=True)
                (path/'failure.json').write_text(json.dumps(row,indent=2))
            row['variant']=variant
            rows.append(row);(ROOT/'all-runs.json').write_text(json.dumps(rows,indent=2))
            archive(ROOT/label)
        runner.start_lttng=default_start
        # Loom remains a distinct Q endpoint. Same three queues, three seeds.
        spec=importlib.util.spec_from_file_location('response_loom',WORK/'loom-harness/experiment-v2.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.ROOT=ROOT/'loom-q';module.ROOT.mkdir()
        qrows=[]
        for p in range(3):
            for r in range(3):
                case=dict(cases()[3],seed=2000+r);label=f'p{p}-r{r}'
                print('Loom-Q',label,flush=True)
                try:row=module.run(case,p,label,sampling=0)
                except Exception as exc:row=dict(adapter='loom',case_id=case['id'],profile=p,status='failed',error=f'{type(exc).__name__}: {exc}')
                qrows.append(row);(module.ROOT/'all-runs.json').write_text(json.dumps(qrows,indent=2))
        (ROOT/'completed.json').write_text(json.dumps(dict(core_trials=len(rows),query_trials=len(qrows))))


if __name__=='__main__':main()
