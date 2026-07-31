#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python3 scripts/audit/run_e05_07_08_activation_kv.py "${1:-cpu}"
