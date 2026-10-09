#!/usr/bin/env bash
set -euo pipefail
work=/tmp/hqsb-capture-fair-20261017
if ! command -v go >/dev/null; then
  DEBIAN_FRONTEND=noninteractive apt-get install -y golang-go > "$work/logs/go-install.log" 2>&1
fi
export GOPROXY=https://goproxy.cn,direct GOMAXPROCS=4 GOPATH="$work/go" GOCACHE="$work/go-cache"
cd "$work/upstream/tracezip/otel-compressor/prefix-compressed-receiver"
go test -p 4 -c -o "$work/tracezip-admission" > "$work/logs/tracezip-build.log" 2>&1
