#!/usr/bin/env bash
# Invoke through scripts/remote_run.sh on Jetson. Never run models on the Mac.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ ! -f web/console/dist/index.html ]]; then
  echo 'Frontend missing: build web/console on the remote host first.' >&2
  exit 1
fi
exec .venv-console/bin/python -m hqsb.console "$@"
