#!/usr/bin/env bash
# Invoke ONLY through scripts/remote_run.sh on the selected CUDA target.
set -euo pipefail
runtime_python="$(pwd)/.venv-console/bin/python"
calibration_env="$(pwd)/.venv-l4-calibration"
mkdir -p reports/console/l4
"$runtime_python" - <<'PY'
import importlib.metadata as m
from pathlib import Path
assert m.version('torch').startswith('2.8.0'), 'This lock was verified with torch 2.8; use a separate target lock otherwise'
assert m.version('transformers') == '5.8.0'
Path('reports/console/l4/runtime-constraints.txt').write_text('torch=='+m.version('torch')+'\ntransformers==5.8.0\n')
PY
"$runtime_python" -m pip install --index-url https://pypi.org/simple --no-cache-dir \
  -c reports/console/l4/runtime-constraints.txt \
  bitsandbytes==0.50.2 compressed-tensors==0.13.0 optimum-quanto==0.2.7
# Keep the pinned serving Transformers/CUDA stack intact. The calibration tool
# uses the same torch installation but an isolated, officially compatible HF stack.
if [ ! -d "$calibration_env" ]; then
  "$runtime_python" -m venv --system-site-packages "$calibration_env"
fi
"$calibration_env/bin/python" -m pip install --index-url https://pypi.org/simple --no-cache-dir \
  -c reports/console/l4/runtime-constraints.txt --dry-run bitsandbytes==0.50.2 >/dev/null
"$calibration_env/bin/python" -m pip install --index-url https://pypi.org/simple --no-cache-dir \
  'torch==2.8.0' 'llmcompressor==0.9.0' 'transformers==4.57.3'
"$runtime_python" -m pip freeze > reports/console/l4/runtime-freeze.txt
"$calibration_env/bin/python" -m pip freeze > reports/console/l4/calibration-freeze.txt
