#!/usr/bin/env bash
# Run on an authorized Linux target through scripts/remote_run.sh, never on Mac.
set -euo pipefail
if [[ $# != 1 || "$1" != /* ]]; then
  echo 'usage: bootstrap.sh /absolute/isolated/work-root' >&2
  exit 2
fi
CAPTURE_WORK_ROOT="$1"
CAPTURE_SOURCE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$CAPTURE_WORK_ROOT"/{upstream,build,logs}
exec 9>"$CAPTURE_WORK_ROOT/measurement.lock"
flock -n 9 || { echo 'measurement/build active; refusing to replace loaded libraries' >&2; exit 1; }
python3 - "$CAPTURE_SOURCE_ROOT/benchmarks/capture/baselines.lock.json" "$CAPTURE_WORK_ROOT" <<'PY'
import json,pathlib,subprocess,sys
lock=json.loads(pathlib.Path(sys.argv[1]).read_text())
root=pathlib.Path(sys.argv[2])
for name,meta in lock['upstream'].items():
    directory=root/'upstream'/name
    if not directory.exists():
        directory.mkdir()
        subprocess.run(['git','init',str(directory)],check=True)
        subprocess.run(['git','-C',str(directory),'remote','add','origin',meta['url']],check=True)
        subprocess.run(['git','-C',str(directory),'fetch','--depth','1','origin',meta['commit']],check=True,timeout=240)
        subprocess.run(['git','-C',str(directory),'checkout','--detach','FETCH_HEAD'],check=True)
    actual=subprocess.check_output(['git','-c',f'safe.directory={directory}','-C',str(directory),'rev-parse','HEAD'],text=True).strip()
    if actual != meta['commit']: raise SystemExit(f'{name}: unexpected existing checkout; refusing reset')
PY
if [[ ! -x "$CAPTURE_WORK_ROOT/venv/bin/python" ]]; then
  python3 -m venv "$CAPTURE_WORK_ROOT/venv"
fi
"$CAPTURE_WORK_ROOT/venv/bin/pip" install \
  opentelemetry-sdk==1.45.1 opentelemetry-exporter-zipkin==1.45.1 \
  opentelemetry-proto==1.15.0 protobuf==3.20.3 pytest==9.1.1 \
  >"$CAPTURE_WORK_ROOT/logs/install-python.log" 2>&1
"$CAPTURE_WORK_ROOT/venv/bin/pip" freeze >"$CAPTURE_WORK_ROOT/logs/python-freeze.txt"
export PATH="$CAPTURE_WORK_ROOT/venv/bin:$PATH"
make -C "$CAPTURE_WORK_ROOT/upstream/NanoLog/runtime" clean >"$CAPTURE_WORK_ROOT/logs/build-nanolog.log" 2>&1
make -C "$CAPTURE_WORK_ROOT/upstream/NanoLog/runtime" -j4 CXX_ARGS='-std=c++17 -g -O3 -fPIC' \
  >>"$CAPTURE_WORK_ROOT/logs/build-nanolog.log" 2>&1
make -C "$CAPTURE_WORK_ROOT/upstream/hindsight/client" -j4 dirs lib/libtracer.so lib/libtracer.a \
  CFLAGS='-std=gnu11 -Wall -rdynamic -O3 -pedantic -include string.h -include stdlib.h -include sys/stat.h' \
  >"$CAPTURE_WORK_ROOT/logs/build-hindsight.log" 2>&1
g++ -std=c++17 -O3 -fPIC -shared -Werror=format \
  -I"$CAPTURE_WORK_ROOT/upstream/NanoLog/runtime" \
  "$CAPTURE_SOURCE_ROOT/benchmarks/capture/native/nanolog_bridge.cc" \
  "$CAPTURE_WORK_ROOT/upstream/NanoLog/runtime/libNanoLog.a" -lrt -pthread \
  -o "$CAPTURE_WORK_ROOT/build/libcapture_nanolog.so"
g++ -std=c++17 -O3 -fPIC -shared \
  -I"$CAPTURE_WORK_ROOT/upstream/hindsight/client/include" -I"$CAPTURE_WORK_ROOT/upstream/hindsight/client/src" \
  "$CAPTURE_SOURCE_ROOT/benchmarks/capture/native/hindsight_bridge.cc" \
  "$CAPTURE_WORK_ROOT/upstream/hindsight/client/lib/libtracer.a" -lm -pthread \
  -o "$CAPTURE_WORK_ROOT/build/libcapture_hindsight.so"
g++ -std=c++17 -O3 -pthread "$CAPTURE_SOURCE_ROOT/benchmarks/capture/native/runner.cc" \
  -L"$CAPTURE_WORK_ROOT/build" -lcapture_nanolog -Wl,-rpath,"$CAPTURE_WORK_ROOT/build" \
  -o "$CAPTURE_WORK_ROOT/build/capture-native"
g++ -std=c++17 -O3 -pthread "$CAPTURE_SOURCE_ROOT/benchmarks/capture/native/competitor.cc" \
  -o "$CAPTURE_WORK_ROOT/build/capture-competitor"
printf '%s\n' 'CPU libraries built. OTAP needs its separately pinned Rust toolchain/build; no GPU dependencies installed.'
