#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${HQSB_REMOTE_HOST:-jetson}"
if [[ -n "${HQSB_REMOTE_DIR:-}" ]]; then
    REMOTE_DIR="${HQSB_REMOTE_DIR}"
else
    case "${REMOTE_HOST}" in
        jetson) REMOTE_DIR="/home/jetson/work/HQSB-main" ;;
        3090-inf) REMOTE_DIR="/root/HeteroQuantServeBench" ;;
        4090D-hqsb) REMOTE_DIR="/root/HeteroQuantServeBench-4090D" ;;
        *) echo 'Set HQSB_REMOTE_DIR for this execution host.' >&2; exit 2 ;;
    esac
fi

# 检查是否有输入命令
if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/remote_run.sh <command>"
    exit 1
fi

# Quote each argument before handing it to the remote shell.  This keeps a
# caller's argv intact and prevents spaces or shell metacharacters in an
# argument from changing the command.  HQSB_REMOTE_DIR lets recovery and
# feature branches run in an independent remote clone instead of overwriting
# the experiment worktree.
printf -v remote_dir_q '%q' "${REMOTE_DIR}"
printf -v remote_argv_q ' %q' "$@"
# Use the selected checkout's isolated runtime when present. No activation or
# execution happens on the editing client; argv quoting remains unchanged.
ssh -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -- "${REMOTE_HOST}" "cd ${remote_dir_q} && if [ -d .venv-console/bin ]; then export PATH=\"\$PWD/.venv-console/bin:\$PATH\"; fi && exec${remote_argv_q}"
