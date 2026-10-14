"""Archive only a completed phase while holding the shared measurement lock."""
import argparse
import fcntl
from pathlib import Path
from run_sustained import archive

parser=argparse.ArgumentParser()
parser.add_argument('folder',type=Path)
parser.add_argument('--completion',required=True,type=Path)
args=parser.parse_args()
base=Path('/root/autodl-tmp/hqsb-capture-fair-20261017/results').resolve()
if not args.folder.resolve().is_relative_to(base):raise ValueError('outside new evidence root')
with Path('/tmp/hqsb-capture-bench-20261017/measurement.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if not args.completion.exists():raise RuntimeError('phase completion not confirmed')
    for index,metric in enumerate(sorted(args.folder.glob('*/metrics.json'))):
        directory=metric.parent
        if not (directory/'raw-compression-manifest.json').exists():archive(directory)
        if index%10==0:print('archived completed phase trials',index,flush=True)
    print('phase archive complete',args.folder,flush=True)
