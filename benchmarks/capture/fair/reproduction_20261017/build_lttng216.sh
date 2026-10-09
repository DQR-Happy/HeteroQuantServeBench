#!/usr/bin/env bash
set -euo pipefail
work=/tmp/hqsb-capture-fair-20261017
prefix="$work/lttng216"
export DEBIAN_FRONTEND=noninteractive
apt-get install -y autoconf automake libtool flex bison liburcu-dev libpopt-dev libxml2-dev > "$work/logs/lttng216-deps.log" 2>&1
export CFLAGS='-O3 -DNDEBUG' CXXFLAGS='-O3 -DNDEBUG'
cd "$work/upstream/lttng-ust-216"
./bootstrap > "$work/logs/lttng216-ust-bootstrap.log" 2>&1
./configure --prefix="$prefix" --disable-numa --disable-man-pages --disable-examples --disable-java-agent-all --disable-python-agent > "$work/logs/lttng216-ust-configure.log" 2>&1
make -j4 > "$work/logs/lttng216-ust-build.log" 2>&1
make install > "$work/logs/lttng216-ust-install.log" 2>&1
export PKG_CONFIG_PATH="$prefix/lib/pkgconfig" LD_LIBRARY_PATH="$prefix/lib" CPPFLAGS="-I$prefix/include" LDFLAGS="-L$prefix/lib -Wl,-rpath,$prefix/lib"
cd "$work/upstream/lttng-tools-216"
./bootstrap > "$work/logs/lttng216-tools-bootstrap.log" 2>&1
./configure --prefix="$prefix" --disable-tests --disable-man-pages --without-kmod > "$work/logs/lttng216-tools-configure.log" 2>&1
make -j4 > "$work/logs/lttng216-tools-build.log" 2>&1
make install > "$work/logs/lttng216-tools-install.log" 2>&1
# Keep the measured 2.14 binary untouched.
g++ -std=c++17 -O3 -DNDEBUG -fPIC -shared "$work/source/benchmarks/capture/fair/lttng_adapter.cc" -I"$work/source/benchmarks/capture/fair" -I"$prefix/include" -L"$prefix/lib" -Wl,-rpath,"$prefix/lib" -llttng-ust -ldl -pthread -o "$work/lttng216/lttng216.so"
"$prefix/bin/lttng" --version > "$work/logs/lttng216-version.log"
find "$prefix" -name lttng-consumerd -type f >> "$work/logs/lttng216-version.log"
