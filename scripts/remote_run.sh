#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${HQSB_REMOTE_HOST:-jetson@192.168.10.7}"
REMOTE_DIR="${HQSB_REMOTE_DIR:-/home/jetson/work/HeteroQuantServeBench}"

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
ssh -- "${REMOTE_HOST}" "cd ${remote_dir_q} && exec${remote_argv_q}"
