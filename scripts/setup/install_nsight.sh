#!/usr/bin/env bash
# Remote-only, standalone NVIDIA CLI; no driver or inference environment changes.
set -euo pipefail
if [[ "$(uname -s)-$(uname -m)" != Linux-x86_64 ]]; then
  echo 'Install the NVIDIA Nsight CLI package for this target architecture.' >&2
  exit 2
fi
root="${HQSB_NSIGHT_ROOT:?Set HQSB_NSIGHT_ROOT to an execution-node data directory}"
mkdir -p "$root"
deb="$root/nsys-2025.3.1.deb"
curl -fL --retry 2 -o "$deb" https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb
echo "d2484ad0faf6831b11fa0bf73c54232d9ea8beafb50414019e6ba299c4ed5718  $deb" | sha256sum -c -
dpkg-deb -x "$deb" "$root/nsys-2025.3.1"
"$root/nsys-2025.3.1/opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/nsys" --version
