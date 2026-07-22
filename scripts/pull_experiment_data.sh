#!/usr/bin/env bash
# 从 Jetson 拉回实验报告/数据到开发机（rsync pull）。
#
# 用法（在【开发机】的项目根目录执行）：
#   ./scripts/pull_experiment_data.sh
#
# 可用环境变量覆盖默认值：
#   JETSON_USER=jetson  JETSON_HOST=192.168.10.7  REMOTE_DIR=/home/jetson/work/HeteroQuantServeBench
set -euo pipefail

cd "$(dirname "$0")/.."

JETSON_USER="${JETSON_USER:-jetson}"
JETSON_HOST="${JETSON_HOST:-192.168.10.7}"
REMOTE_DIR="${HQSB_REMOTE_DIR:-${REMOTE_DIR:-/home/jetson/work/HeteroQuantServeBench}}"

SRC="${HQSB_REMOTE_HOST:-${JETSON_USER}@${JETSON_HOST}}:${REMOTE_DIR}/docs/stage_experiments/"

echo "==> 拉取 ${SRC}  ->  ./docs/stage_experiments/"
rsync -avz "${SRC}" ./docs/stage_experiments/

echo "==> 拉取完成"
