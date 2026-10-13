"""Author-supported Config.h staging sweep; original default is not a tool limit."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

WORK = Path('/tmp/hqsb-capture-fair-20261017')
OLD = Path('/tmp/hqsb-capture-bench-20261017')
ROOT = Path('/root/autodl-tmp/hqsb-capture-fair-20261017/results/nanolog-config-response-v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    os.sched_setaffinity(0, {196})
    with (OLD/'measurement.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sys.path.insert(0, str(WORK/'validation/source'))
        from benchmarks.capture.fair import run as runner
        from benchmarks.capture.fair.campaign import cases
        from run_sustained import archive
        assert shutil.which('zstd')
        # CTF cohorts are closed before this lock can be acquired. Preserve
        # byte-identical archives before building/running further SUTs.
        for cohort in ('lttng214-v3','lttng216-v3'):
            phase=ROOT.parent/cohort
            assert (phase/'completed.json').exists()
            for metric in sorted([*phase.glob('test-*/metrics.json'), *phase.glob('tune-*/metrics.json')]):
                directory=metric.parent
                if not (directory/'raw-compression-manifest.json').exists():
                    archive(directory)
            print('completed CTF cohort preserved',cohort,flush=True)
        if shutil.disk_usage(ROOT.parent).free<2*1024**3:
            raise RuntimeError('archive preflight did not restore2GiB storage headroom')
        ROOT.mkdir(exist_ok=False)
        builds = {1: OLD/'build/libcapture_nanolog.so'}
        metadata = {}
        bridge = WORK/'source/benchmarks/capture/native/nanolog_bridge.cc'
        for mib, shift in ((4, 22), (16, 24)):
            repo = WORK/'upstream'/f'nanolog-staging-{mib}m'
            with (ROOT/f'build-{mib}m.log').open('w') as log:
                subprocess.run(['git', 'clone', '--no-hardlinks', str(OLD/'upstream/NanoLog'), str(repo)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
                head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
                assert head == '2a94d70f9d1db4da416053b1b926387fa068a59b'
                config = repo/'runtime/Config.h'
                before = config.read_text()
                needle = 'static const uint32_t STAGING_BUFFER_SIZE = 1<<20;'
                assert before.count(needle) == 1
                config.write_text(before.replace(needle, f'static const uint32_t STAGING_BUFFER_SIZE = 1<<{shift};'))
                # OUTPUT_BUFFER_SIZE stays64MiB; release threshold remains the
                # author's derived half-staging rule. No algorithm edits.
                subprocess.run(['make', 'clean'], cwd=repo/'runtime', stdout=log, stderr=subprocess.STDOUT, check=True)
                subprocess.run(['make', '-j4', 'CXX_ARGS=-std=c++17 -g -O3 -fPIC'], cwd=repo/'runtime',
                               stdout=log, stderr=subprocess.STDOUT, check=True)
                binary = repo/f'libcapture_nanolog_{mib}m.so'
                command = ['g++', '-std=c++17', '-O3', '-fPIC', '-shared', '-Werror=format',
                           '-I'+str(repo/'runtime'), str(bridge), str(repo/'runtime/libNanoLog.a'),
                           '-lrt', '-pthread', '-o', str(binary)]
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                diff = subprocess.check_output(['git', '-C', str(repo), 'diff', '--', 'runtime/Config.h'], text=True)
                (ROOT/f'config-{mib}m.patch').write_text(diff)
                builds[mib] = binary
                metadata[mib] = dict(commit=head, library_sha256=sha(binary), config_sha256=sha(config),
                                     configuration_only_diff=diff, bridge_sha256=sha(bridge), compile_command=command)
        receiver=WORK/'upstream/nanolog-staging-16m/runtime/decompressor'
        os.environ['HQSB_NANOLOG_DECOMPRESSOR']=str(receiver)
        (ROOT/'protocol.json').write_text(json.dumps(dict(
            purpose='all three staging configurations displayed; no retuning of main selected profile',
            profiles_MiB=[1,4,16], output_buffer_MiB=64, release_threshold='stock staging/2',
            budget='same CPU192-195, one collector193, AS1GiB, stack2MiB, arena2',
            seeds=[3000,3001,3002], cases=['steady-1m-p8','mixed-records','giant-2m'],
            source_gates='same case shapes and2x controls from campaign-v1',
            default_exclusions='1MiB build cannot encode a single1/2MiB string plus header; keep unsupported-build entries, not whole-tool exclusions',
            builds=metadata, default_library_sha256=sha(builds[1]),
            receiver_decoder=dict(path=str(receiver),sha256=sha(receiver),staging_MiB=16,
                                  scope='official same-version receiver outside SUT timer; larger records require matching decoder capacity'),
            maintained_runner_sha256=sha(Path(runner.__file__)), source_script_sha256=sha(Path(__file__))), indent=2))
        jobs = [(c, mib, r) for c in [cases()[i] for i in (3,10,14)]
                for mib in (1,4,16) for r in range(3) if mib!=1 or c['id']=='steady-1m-p8']
        random.Random(20261017).shuffle(jobs)
        rows = []
        for index, (case, mib, repeat) in enumerate(jobs):
            label=f'{index:03d}-{case["id"]}-{mib}m-r{repeat}'
            print(index+1, len(jobs), label, flush=True)
            os.environ['HQSB_NANOLOG_LIBRARY'] = str(builds[mib])
            try:
                row = runner.execute(WORK/'validation', ROOT, dict(case,seed=3000+repeat),
                                     'nanolog',1,repeat,'throughput',label)
            except Exception as exc:
                row = dict(adapter='nanolog',case_id=case['id'],status='failed',error=f'{type(exc).__name__}: {exc}')
                (ROOT/label/'failure.json').write_text(json.dumps(row,indent=2))
            row['staging_MiB']=mib
            rows.append(row)
            (ROOT/'all-runs.json').write_text(json.dumps(rows,indent=2))
            archive(ROOT/label)
        os.environ.pop('HQSB_NANOLOG_LIBRARY', None)
        os.environ.pop('HQSB_NANOLOG_DECOMPRESSOR', None)
        subprocess.run([sys.executable,'-m','pytest','-q',str(WORK/'validation/source/tests/unit/capture_benchmark')],
                       env=dict(os.environ,PYTHONPATH=str(WORK/'validation/source')),check=True)
        subprocess.run(['/tmp/hqsb-capture-bench-20261017/analysis-venv/bin/ruff','check','--select','F',
                        str(WORK/'validation/source/benchmarks/capture'),
                        str(WORK/'validation/source/tests/unit/capture_benchmark')],check=True)
        (ROOT/'completed.json').write_text(json.dumps(dict(trials=len(rows),configurations=3)))


if __name__=='__main__':
    main()
