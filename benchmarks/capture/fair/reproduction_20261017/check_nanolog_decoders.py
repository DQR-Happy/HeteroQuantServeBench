"""Distinguish decoder buffer limits from data loss; preserve old derivations."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

WORK=Path('/tmp/hqsb-capture-fair-20261017')
OLD=Path('/tmp/hqsb-capture-bench-20261017')
ROOT=Path('/root/autodl-tmp/hqsb-capture-fair-20261017/results/nanolog-config-response-v1')
OUT=ROOT.parent/'nanolog-decoder-validation-v1'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def restore(directory):
    manifest=directory/'raw-compression-manifest.json'
    if manifest.exists():
        subprocess.run([sys.executable,str(WORK/'research/restore_raw.py'),str(directory),str(manifest)],check=True)


def main():
    os.sched_setaffinity(0,{196})
    with (OLD/'measurement.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert (ROOT/'completed.json').exists()
        OUT.mkdir(exist_ok=False)
        rows=json.loads((ROOT/'all-runs.json').read_text())
        target=ROOT/'001-giant-2m-16m-r1'
        restore(target)
        decoders={1:OLD/'upstream/NanoLog/runtime/decompressor',
                  4:WORK/'upstream/nanolog-staging-4m/runtime/decompressor',
                  16:WORK/'upstream/nanolog-staging-16m/runtime/decompressor'}
        diagnostics=[]
        for mib,binary in decoders.items():
            output=OUT/f'decoder-{mib}m.txt'
            with output.open('w') as stdout,(OUT/f'decoder-{mib}m.stderr').open('w') as stderr:
                result=subprocess.run([str(binary),'decompress',str(target/'nanolog.bin')],
                                      stdout=stdout,stderr=stderr,timeout=120)
            with output.open() as stream:
                count=sum('HQSB|' in line for line in stream)
            diagnostics.append(dict(decoder_staging_MiB=mib,returncode=result.returncode,
                                    HQSB_lines=count,stdout_bytes=output.stat().st_size,
                                    decoder_sha256=sha(binary)))
        (OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
        # Only rederive if the larger official decoder actually recovers the
        # configured writer's records. Otherwise leave the original verdict.
        if diagnostics[-1]['returncode'] or not diagnostics[-1]['HQSB_lines']:
            raise RuntimeError('larger official decoder did not recover records; investigate, do not relabel')
        sys.path.insert(0,str(WORK/'source'))
        from benchmarks.capture.oracle import verify
        binary=decoders[16]

        def events(directory,adapter,upstream):
            output=directory/'decoded-configured.txt'
            with output.open('w') as stdout,(directory/'decode-configured.stderr').open('w') as stderr:
                subprocess.run([str(binary),'decompress',str(directory/'nanolog.bin')],
                               stdout=stdout,stderr=stderr,timeout=120,check=True)
            with output.open() as stream:
                for line in stream:
                    if 'HQSB|' not in line:continue
                    parts=line.split('HQSB|',1)[1].rstrip('\n').split('|',4)
                    if len(parts)!=5:raise ValueError('truncated official NanoLog output')
                    run,p,i,ts,body=parts
                    yield dict(run_id=run,producer_id=int(p),source_seq=int(i),source_timestamp=int(ts),body=body)

        updated=[]
        changes=[]
        for index,previous in enumerate(rows):
            paths=list(ROOT.glob(f'{index:03d}-*'))
            assert len(paths)==1
            directory=paths[0]
            print('redecode',index+1,len(rows),directory.name,flush=True)
            restore(directory)
            original=json.loads((directory/'metrics.json').read_text())
            shutil.copyfile(directory/'metrics.json',directory/'metrics-original-default-decoder.json')
            row=verify(directory,OLD/'upstream',decode_events=events)
            row.update({k:v for k,v in original.items() if k not in row})
            row['staging_MiB']=previous['staging_MiB']
            row['decoder_scope']='official same-version16MiB Config build; receiver buffer accepts2MiB records; decoder is outside SUT timer/resource budget'
            (directory/'metrics.json').write_text(json.dumps(row,indent=2))
            updated.append(row)
            changes.append(dict(trial=directory.name,before=original['status'],after=row['status'],
                                valid_before=original['valid_unique'],valid_after=row['valid_unique']))
            (directory/'decoded-configured.txt').unlink()
            # Only remove a restored duplicate whose archive and original SHA
            # both still match the already committed lossless manifest.
            manifest=directory/'raw-compression-manifest.json'
            if manifest.exists():
                for entry in json.loads(manifest.read_text())['files']:
                    original_path=directory/entry['original']
                    assert sha(directory/entry['archive'])==entry['archive_sha256']
                    if original_path.exists():
                        assert sha(original_path)==entry['original_sha256']
                        original_path.unlink()
        shutil.copyfile(ROOT/'all-runs.json',ROOT/'all-runs-original-default-decoder.json')
        (ROOT/'all-runs.json').write_text(json.dumps(updated,indent=2))
        (OUT/'adjudication.json').write_text(json.dumps(dict(
            policy='rederive existing raw with official larger configured receiver; no SUT rerun, timings unchanged; original metrics retained',
            decoder_sha256=sha(binary),changes=changes),indent=2))
        (OUT/'completed.json').write_text(json.dumps(dict(trials=len(updated),source_captures_unchanged=True)))


if __name__=='__main__':main()
