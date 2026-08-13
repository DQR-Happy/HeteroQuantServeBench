#!/usr/bin/env bash
# 从开发机把代码同步到边缘设备（rsync push）。
#
# 用法（在【开发机】的项目根目录执行）：
#   ./scripts/sync_to_jetson.sh                      # 默认目标：Jetson
#   ./scripts/sync_to_jetson.sh --target orangepi    # 目标：OrangePi AI Pro
#   ./scripts/sync_to_jetson.sh --dry-run            # 仅预览，不写入
#   ./scripts/sync_to_jetson.sh --verify             # 只校验两端一致性，不传输
#
# 可用环境变量覆盖默认值：
#   HQSB_TARGET=jetson|orangepi
#   HQSB_REMOTE_HOST=user@host        直连地址覆盖（对两个目标都生效）
#   HQSB_REMOTE_DIR=/remote/path      远端项目根目录覆盖
#   JETSON_USER / JETSON_HOST         仅 jetson 目标的兼容变量
#
# 设计约定（.vscode/sftp.json 的 ignore 必须是本脚本排除项的超集）：
# * 只同步源码/配置；排除 .git、构建产物、编译缓存与边缘端本地生成的实验数据。
# * 不加 --delete：避免误删边缘端本地的证据区。
# * `build/` 必须排除：开发机上可能残留一份从边缘端拉回的构建树快照，
#   一旦被 rsync 推回去，会用旧的 .o/.so 与旧的 build.ninja 覆盖边缘端上
#   刚编译好的库（E03-01 实测过：远端 `nm` 已出现的新 C ABI 符号在下次同步后
#   消失，且 ninja 认为目标"已是最新"而拒绝重编）。
# * `docs/stage_experiments/` 整体排除（约 4.4GB 私有实验档案），但**单独补同步
#   details/**（约 3MB）：它是 hqsb/ascend/experiment.protocol_steps 的运行期依赖，
#   缺它会让 `run_e09.py --interface-map` 在干净目标上直接 ConfigError。
#   拆成两次 rsync 而不是写 include/exclude 递归规则，是因为 macOS 自带
#   openrsync（2.6.9 compatible）的 `**` 通配支持不可靠。
set -euo pipefail

cd "$(dirname "$0")/.."

TARGET="${HQSB_TARGET:-jetson}"
MODE="sync"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  MODE="dry-run"; shift ;;
    --verify)   MODE="verify";  shift ;;
    --target)   TARGET="${2:-}"; shift 2 ;;
    --target=*) TARGET="${1#*=}"; shift ;;
    -h|--help)  sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "${TARGET}" in
  jetson)
    JETSON_USER="${JETSON_USER:-jetson}"
    JETSON_HOST="${JETSON_HOST:-192.168.10.7}"
    DEFAULT_HOST="${JETSON_USER}@${JETSON_HOST}"
    DEFAULT_DIR="/home/jetson/work/HeteroQuantServeBench"
    ;;
  orangepi)
    DEFAULT_HOST="OrangePi-AI-Pro"
    DEFAULT_DIR="/home/HwHiAiUser/work/HeteroQuantServeBench"
    ;;
  *)
    echo "unknown target '${TARGET}'; expected 'jetson' or 'orangepi'" >&2
    exit 2
    ;;
esac

REMOTE_HOST="${HQSB_REMOTE_HOST:-${DEFAULT_HOST}}"
REMOTE_DIR="${HQSB_REMOTE_DIR:-${DEFAULT_DIR}}"
DEST="${REMOTE_HOST}:${REMOTE_DIR}"

RSYNC_OPTIONS=(-avz)
if [[ "${MODE}" == "dry-run" ]]; then
  RSYNC_OPTIONS+=(--dry-run)
fi

# 排除规则单一真源；.vscode/sftp.json 的 ignore 必须是它的超集。
EXCLUDES=(
  --exclude '._*'
  --exclude '.git/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.codebuddy/'
  --exclude '.vscode/'
  --exclude '.agents/'
  --exclude '.codex/'
  --exclude '.venv*/'
  --exclude 'venv/'
  --exclude '.pytest_cache/'
  --exclude '.mypy_cache/'
  --exclude '.ruff_cache/'
  --exclude '*.egg-info/'
  --exclude 'buddy_history/'
  --exclude 'build/'
  --exclude 'node_modules/'
  --exclude 'dist/'
  --exclude '.console/'
  --exclude 'test-results/'
  --exclude 'playwright-report/'
  --exclude 'docs/stage_experiments/'
  --exclude '/reports/'
  --exclude '/models/'
  --exclude '/artifacts/'
  --exclude '/datasets/'
  --exclude '/checkpoints/'
  --exclude '/experiment_results/'
  --exclude '/third_party/cutlass/'
  --exclude '.env*'
  --exclude '*.pem'
  --exclude '*.key'
)

# 需要跨端保持一致的源码路径（--verify 使用）。
VERIFY_PATHS=(
  hqsb
  ops
  scripts
  tests
  configs
  benchmarks
  web
  infra
  contracts
  pyproject.toml
  README.md
  CMakeLists.txt
)

sync_primary() {
  echo "==> [${TARGET}] 主同步 $(pwd)  ->  ${DEST}"
  rsync "${RSYNC_OPTIONS[@]}" "${EXCLUDES[@]}" ./ "${DEST}/"
}

sync_details() {
  echo "==> [${TARGET}] 补同步运行期依赖 docs/stage_experiments/details/"
  rsync "${RSYNC_OPTIONS[@]}" \
    docs/stage_experiments/details/ \
    "${DEST}/docs/stage_experiments/details/"
}

verify() {
  local path out diffs=0
  echo "==> [${TARGET}] 校验一致性  ->  ${DEST}"
  for path in "${VERIFY_PATHS[@]}"; do
    [[ -e "${path}" ]] || continue
    if [[ -d "${path}" ]]; then
      out=$(rsync -n -r --checksum --itemize-changes \
        --exclude='__pycache__/' --exclude='*.pyc' \
        "${path}/" "${DEST}/${path}/" 2>/dev/null | grep -E '^[<>ch]' || true)
    else
      out=$(rsync -n --checksum --itemize-changes \
        "${path}" "${DEST}/${path}" 2>/dev/null | grep -E '^[<>ch]' || true)
    fi
    if [[ -n "${out}" ]]; then
      diffs=$((diffs + 1))
      echo "  [DIFF] ${path}"
      echo "${out}" | head -10 | sed 's/^/    /'
    fi
  done
  if [[ ${diffs} -eq 0 ]]; then
    echo "==> 一致：${TARGET} 的源码与开发机相同"
  else
    echo "==> 不一致：${diffs} 个路径存在差异（见上）" >&2
    return 1
  fi
}

case "${MODE}" in
  verify)
    verify
    ;;
  dry-run)
    sync_primary
    sync_details
    echo "==> 预览完成（未写入远端）"
    ;;
  *)
    sync_primary
    sync_details
    echo "==> 同步完成"
    ;;
esac
