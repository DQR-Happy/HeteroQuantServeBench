#!/usr/bin/env bash
# Target-only setup. Invoke via remote_run.sh, with a CUDA torch already installed.
# Reuse the target's torch inside an isolated overlay; never install Jetson wheels
# or upgrade the target's base Python. Node/npm are explicit prerequisites.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Run this setup on the Linux target via scripts/remote_run.sh.' >&2
  exit 2
fi
command -v node >/dev/null || { echo 'Node.js >=20.19 is required for the web build.' >&2; exit 2; }
command -v npm >/dev/null || { echo 'npm is required for the web build.' >&2; exit 2; }
bootstrap_python="${HQSB_PYTHON:-python3}"
console_venv="${HQSB_CONSOLE_VENV:-.venv-console}"
if [[ ! -e "${console_venv}" ]]; then
  "${bootstrap_python}" -m venv --system-site-packages "${console_venv}"
fi
console_python="$(cd "${console_venv}" && pwd)/bin/python"
"${console_python}" -m pip install --upgrade pip 'setuptools>=64' wheel
"${console_python}" -c 'import torch; assert torch.cuda.is_available(), "Install a target-compatible CUDA PyTorch first"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
"${console_python}" -m pip install -e '.[console,console-model,dev]'
"${console_python}" -c 'import hashlib, inspect; from transformers.models.qwen3 import modeling_qwen3; from hqsb.backends.model_kernels.qwen3 import SOURCE_SHA256; assert hashlib.sha256(open(inspect.getfile(modeling_qwen3), "rb").read()).hexdigest() == SOURCE_SHA256, "Unsupported Qwen3 adapter source"'
"${console_python}" scripts/models/download_qwen3_modelscope.py --model-path "${HOME}/models/hqsb/Qwen3-1.7B"
"${console_python}" scripts/observation/build_model_kernels.py
(cd web/console && npm ci && npm run build)
echo "Ready. From the checkout root: ${console_python} -m hqsb.console --config configs/console/cuda.example.yaml --port 8768"
