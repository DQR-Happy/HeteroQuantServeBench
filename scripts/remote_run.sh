#!/usr/bin/env bash
set -e

REMOTE_HOST="jetson@192.168.10.7"
REMOTE_DIR="/home/jetson/work/HeteroQuantServeBench"

# 检查是否有输入命令
if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/remote_run.sh <command>"
    exit 1
fi

# 执行远端命令
ssh "${REMOTE_HOST}" "cd ${REMOTE_DIR} && $@"