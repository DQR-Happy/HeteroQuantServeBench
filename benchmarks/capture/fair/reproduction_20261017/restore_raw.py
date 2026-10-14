"""Restore selected archived raw bytes with exact digest checks, on remote only."""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import subprocess


def digest(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path,help='directory relative to which manifest paths are stored')
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--match',action='append',default=[],help='optional glob over original relative paths')
    args=parser.parse_args();root=args.root.resolve()
    count=0
    for row in json.loads(args.manifest.read_text())['files']:
        if args.match and not any(fnmatch.fnmatch(row['original'],p) for p in args.match):continue
        original=(root/row['original']).resolve();archive=(root/row['archive']).resolve()
        if not original.is_relative_to(root) or not archive.is_relative_to(root):raise ValueError('unsafe manifest path')
        if original.exists():
            with original.open('rb') as stream:observed=digest(stream)
            if observed!=row['original_sha256']:raise ValueError('existing original differs; refusing overwrite')
            continue
        with archive.open('rb') as stream:observed=digest(stream)
        if observed!=row['archive_sha256']:raise ValueError('archive digest differs')
        temporary=original.with_name(original.name+'.restore-partial')
        with temporary.open('xb') as output:
            subprocess.run(['zstd','-q','-d','-c','--',str(archive)],stdout=output,check=True)
        with temporary.open('rb') as stream:observed=digest(stream)
        if observed!=row['original_sha256'] or temporary.stat().st_size!=row['original_bytes']:
            raise ValueError('restored bytes differ; partial file preserved')
        temporary.replace(original);count+=1
    print('Verified restored raw files:',count)


if __name__=='__main__':main()
