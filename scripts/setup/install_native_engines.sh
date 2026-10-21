#!/usr/bin/env bash
# Run through scripts/remote_run.sh; all environments live on the execution host.
set -euo pipefail
cd "$(dirname "$0")/../.."
root="${HQSB_NATIVE_ENV_ROOT:-$PWD/.native-envs}"
mkdir -p "$root/tmp"
for engine in vllm sglang; do
  version=0.11.0
  [[ "$engine" == sglang ]] && version=0.5.5
  python -m venv --system-site-packages "$root/$engine"
  TMPDIR="$root/tmp" "$root/$engine/bin/pip" install --index-url https://pypi.org/simple --no-cache-dir \
    -c configs/environment/native-engines.constraints.txt \
    "$engine==$version" 'transformers==4.57.1' 'torchvision==0.23.0' \
    'compressed-tensors==0.11.0' 'fastapi==0.115.12' 'pydantic==2.11.7' \
    'openai==2.6.1' 'openai-harmony==0.0.4'
  if [[ "$engine" == vllm ]]; then
    TMPDIR="$root/tmp" "$root/$engine/bin/pip" install --index-url https://pypi.org/simple --no-cache-dir 'prometheus-fastapi-instrumentator==7.1.0'
  fi
  if [[ "$engine" == sglang ]]; then
    TMPDIR="$root/tmp" "$root/$engine/bin/pip" install --index-url https://pypi.org/simple --no-cache-dir 'cuda-python==12.8.0'
  fi
  if [[ ! -e ".venv-$engine" ]]; then ln -s "$root/$engine" ".venv-$engine"; fi
  "$root/$engine/bin/python" -m pip freeze > "$root/$engine/environment.freeze.txt"
done
