#!/usr/bin/env bash
# 从开发机把代码同步到 Jetson（rsync push）。
#
# 用法（在【开发机】的项目根目录执行）：
#   ./scripts/sync_to_jetson.sh
#
# 可用环境变量覆盖默认值：
#   JETSON_USER=jetson  JETSON_HOST=192.168.10.7  REMOTE_DIR=/home/jetson/work/HeteroQuantServeBench
#
# 说明：只同步源码/配置，排除 git、编译缓存、以及 Jetson 本地生成的
# 实验数据（docs/stage_experiments、reports）。不加 --delete，避免误删
# Jetson 上的本地证据区。
set -euo pipefail

JETSON_USER="${JETSON_USER:-jetson}"
JETSON_HOST="${JETSON_HOST:-192.168.10.7}"
REMOTE_DIR="${REMOTE_DIR:-/home/jetson/work/HeteroQuantServeBench}"

DEST="${JETSON_USER}@${JETSON_HOST}:${REMOTE_DIR}"

echo "==> 同步 $(pwd)  ->  ${DEST}"
rsync -avz \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.codebuddy/' \
  --exclude 'docs/stage_experiments/' \
  --exclude 'reports/' \
  --exclude 'third_party/' \
  ./ "${DEST}/"

echo "==> 同步完成"
