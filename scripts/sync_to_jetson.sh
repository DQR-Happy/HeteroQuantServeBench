#!/usr/bin/env bash
# 从开发机把代码同步到 Jetson（rsync push）。
#
# 用法（在【开发机】的项目根目录执行）：
#   ./scripts/sync_to_jetson.sh
#
# 可用环境变量覆盖默认值：
#   JETSON_USER=jetson  JETSON_HOST=192.168.10.7  REMOTE_DIR=/home/jetson/work/HeteroQuantServeBench
#
# 说明：只同步源码/配置，排除 git、编译缓存、构建产物、以及 Jetson 本地生成的
# 实验数据（docs/stage_experiments、reports）。不加 --delete，避免误删
# Jetson 上的本地证据区。
#
# `build/` 必须排除：开发机上可能残留一份从 Jetson 拉回的构建树快照，
# 一旦被 rsync 推回去，会用旧的 .o/.so 与旧的 build.ninja 覆盖 Jetson 上
# 刚编译好的库（E03-01 实测过：远端 `nm` 已出现的新 C ABI 符号在下次同步后
# 消失，且 ninja 认为目标"已是最新"而拒绝重编）。
set -euo pipefail

cd "$(dirname "$0")/.."

JETSON_USER="${JETSON_USER:-jetson}"
JETSON_HOST="${JETSON_HOST:-192.168.10.7}"
REMOTE_DIR="${HQSB_REMOTE_DIR:-${REMOTE_DIR:-/home/jetson/work/HeteroQuantServeBench}}"

DEST="${HQSB_REMOTE_HOST:-${JETSON_USER}@${JETSON_HOST}}:${REMOTE_DIR}"

RSYNC_OPTIONS=(-avz)
if [[ "${1:-}" == "--dry-run" && $# -eq 1 ]]; then
  RSYNC_OPTIONS+=(--dry-run)
elif [[ $# -ne 0 ]]; then
  echo "Usage: ./scripts/sync_to_jetson.sh [--dry-run]" >&2
  exit 2
fi

echo "==> 同步 $(pwd)  ->  ${DEST}"
rsync "${RSYNC_OPTIONS[@]}" \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.codebuddy/' \
  --exclude '.vscode/' \
  --exclude '.agents/' \
  --exclude '.codex/' \
  --exclude '.venv*/' \
  --exclude 'venv/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '*.egg-info/' \
  --exclude 'buddy_history/' \
  --exclude 'build/' \
  --exclude 'node_modules/' \
  --exclude 'dist/' \
  --exclude '.console/' \
  --exclude 'test-results/' \
  --exclude 'playwright-report/' \
  --exclude 'docs/stage_experiments/' \
  --exclude '/reports/' \
  --exclude '/models/' \
  --exclude '/artifacts/' \
  --exclude '/datasets/' \
  --exclude '/checkpoints/' \
  --exclude '/experiment_results/' \
  --exclude '/third_party/cutlass/' \
  --exclude '.env*' \
  --exclude '*.pem' \
  --exclude '*.key' \
  ./ "${DEST}/"

echo "==> 同步完成"
