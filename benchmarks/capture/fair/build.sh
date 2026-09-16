#!/usr/bin/env bash
# Prepared, pinned Linux dependencies only. Invoke through remote_run.sh.
set -euo pipefail
if [[ $# != 1 || "$1" != /* ]]; then
  echo 'usage: build.sh /absolute/prepared-work-root' >&2
  exit 2
fi
capture_work="$1"
capture_old=/tmp/hqsb-capture-bench-20261017
capture_source="$(cd "$(dirname "$0")" && pwd)"
exec 9>"$capture_old/measurement.lock"
flock -n 9 || { echo 'measurement/build active; refusing library replacement' >&2; exit 1; }
mkdir -p "$capture_work/build" "$capture_work/logs"
capture_cxx="${CXX:-g++}"
capture_python_config="${PYTHON_CONFIG:-python3-config}"
capture_flags=(-std=c++17 -O3 -DNDEBUG -pthread)
"$capture_cxx" "${capture_flags[@]}" "$capture_source/native_source.cc" -ldl -o "$capture_work/build/capture-fair"
for capture_profile in 0 1 2; do
  capture_sizes=(131072 1048576 4194304)
  "$capture_cxx" "${capture_flags[@]}" -fPIC -shared \
    -DHQSB_QUILL_QUEUE_BYTES="${capture_sizes[$capture_profile]}" \
    -I"$capture_work/upstream/quill/include" "$capture_source/quill_adapter.cc" \
    -o "$capture_work/build/quill$capture_profile.so"
done
"$capture_cxx" "${capture_flags[@]}" -fPIC -shared \
  -I"$capture_work/upstream/perfetto-sdk" "$capture_source/perfetto_adapter.cc" \
  "$capture_work/upstream/perfetto-sdk/perfetto.cc" -ldl -o "$capture_work/build/perfetto.so"
"$capture_cxx" "${capture_flags[@]}" -fPIC -shared \
  -I"$capture_old/upstream/hindsight/client/include" -I"$capture_old/upstream/hindsight/client/src" \
  "$capture_source/hindsight_adapter.cc" "$capture_old/upstream/hindsight/client/lib/libtracer.a" \
  -lm -o "$capture_work/build/hindsight.so"
"$capture_cxx" "${capture_flags[@]}" -fPIC -shared -I"$capture_source" \
  "$capture_source/lttng_adapter.cc" -llttng-ust -ldl -o "$capture_work/build/lttng.so"
# Word splitting here is intentional for compiler flag output from python-config.
# Select the same CPython development installation as the measured runtime.
"$capture_cxx" "${capture_flags[@]}" -fPIC -shared \
  $("$capture_python_config" --includes) "$capture_source/python_bridge.cc" \
  $("$capture_python_config" --embed --ldflags) -o "$capture_work/build/python_bridge.so"
capture_cores=("$capture_work"/dftracer-venv/lib/python*/site-packages/dftracer.libs/libdftracer_core-*.so.*)
if [[ ${#capture_cores[@]} != 1 || ! -f "${capture_cores[0]}" ]]; then
  echo 'expected exactly one pinned DFTracer wheel core; refusing ambiguous linking' >&2
  exit 1
fi
"$capture_cxx" "${capture_flags[@]}" -fPIC -shared "$capture_source/dftracer_adapter.cc" \
  "${capture_cores[0]}" -Wl,-rpath,"$(dirname "${capture_cores[0]}")" -o "$capture_work/build/dftracer.so"
sha256sum "$capture_work"/build/* > "$capture_work/logs/fair-build-sha256.txt"
