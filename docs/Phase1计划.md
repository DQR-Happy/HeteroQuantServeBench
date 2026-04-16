今天先把范围锁定为 **Jetson CUDA 工程基线**，不同时引入模型权重、量化框架和推理引擎。完成后仓库应具备：

1.  Jetson软硬件环境清单；
    
2.  可重复构建的CMake/CUDA工程；
    
3.  GPU设备探测程序；
    
4.  CPU参考实现＋CUDA RMSNorm基线；
    
5.  Nsight/Tegrastats可采集的Benchmark流程；
    
6.  三个结构清晰的Git Commit并推送到GitHub。
    

这些交付物直接对应岗位中频繁要求的C/C++、Linux、CUDA Kernel、体系结构、Profiler、工程规范和性能分析能力。

\*\*今天不要升级JetPack、不要刷机、不要改L4T软件源、不要执行`apt dist-upgrade`。\*\*先锁定现有环境。NVIDIA支持在Jetson宿主机安装JetPack组件，也支持容器化CUDA；当前阶段优先使用宿主机原生CUDA，后面再补容器复现环境。([NVIDIA Docs](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/setup_jetpack.html?utm_source=chatgpt.com "JetPack SDK Setup — Jetson Orin Nano Developer Kit - User Guide"))

---

# 0\. 开两个SSH终端

建议同时保留两个SSH窗口：

-   **终端A**：代码、编译、Git；
    
-   **终端B**：`tegrastats`和资源监控。
    

终端A先执行：

```bash
sudo -v

sudo apt update
sudo apt install -y \
  git \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  jq \
  tree \
  tmux
```

不要运行：

```bash
sudo apt upgrade
sudo apt dist-upgrade
sudo apt install nvidia-cuda-toolkit
```

启动一个持久会话，防止SSH断开导致编译或测试中断：

```bash
tmux new -s hqsb-jetson
```

后续SSH断开后可恢复：

```bash
tmux attach -t hqsb-jetson
```

定义项目路径：

```bash
export HQSB_HOME="$HOME/work/HeteroQuantServeBench"
mkdir -p "$HOME/work"
```

---

# 1\. 完整调查Jetson环境

## 1.1 执行第一轮环境检查

复制执行：

```bash
mkdir -p "$HOME/jetson_audit"

AUDIT_FILE="$HOME/jetson_audit/jetson_precheck_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "===== TIMESTAMP ====="
  date --iso-8601=seconds

  echo
  echo "===== DEVICE MODEL ====="
  tr -d '\0' < /proc/device-tree/model 2>/dev/null || true
  echo

  echo
  echo "===== KERNEL / ARCH ====="
  uname -a
  uname -m

  echo
  echo "===== OPERATING SYSTEM ====="
  cat /etc/os-release

  echo
  echo "===== L4T RELEASE ====="
  cat /etc/nv_tegra_release 2>/dev/null || true

  echo
  echo "===== NVIDIA CORE PACKAGES ====="
  dpkg-query -W \
    nvidia-l4t-core \
    nvidia-l4t-kernel \
    nvidia-jetpack \
    nvidia-jetpack-runtime \
    nvidia-jetpack-dev \
    2>/dev/null || true

  echo
  echo "===== CUDA PATH ====="
  command -v nvcc || true
  readlink -f /usr/local/cuda 2>/dev/null || true
  ls -ld /usr/local/cuda* 2>/dev/null || true

  echo
  echo "===== CUDA VERSION ====="
  nvcc --version 2>/dev/null || true

  echo
  echo "===== CUDA RUNTIME LIBRARY ====="
  ldconfig -p 2>/dev/null | grep -E 'libcudart|libcublas|libcudnn|libnvinfer' || true

  echo
  echo "===== COMPILERS ====="
  gcc --version | head -n 1
  g++ --version | head -n 1
  cmake --version | head -n 1
  ninja --version
  python3 --version
  git --version

  echo
  echo "===== CPU ====="
  lscpu

  echo
  echo "===== MEMORY ====="
  free -h

  echo
  echo "===== STORAGE ====="
  df -h /
  df -h "$HOME"
  lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS

  echo
  echo "===== POWER MODE ====="
  sudo nvpmodel -q 2>&1 || true

  echo
  echo "===== CLOCK STATE ====="
  sudo jetson_clocks --show 2>&1 || true

} | tee "$AUDIT_FILE"

echo
echo "Audit saved to: $AUDIT_FILE"
```

NVIDIA官方使用`cat /etc/nv_tegra_release`确认Jetson Linux版本，使用`apt list --installed | grep nvidia-jetpack`检查JetPack安装状态。([NVIDIA Docs](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/setup_jetpack.html?utm_source=chatgpt.com "JetPack SDK Setup — Jetson Orin Nano Developer Kit - User Guide"))

## 1.2 重点查看结果

依次确认：

```bash
grep -E \
"R[0-9]+|aarch64|CUDA compilation tools|nvidia-jetpack|Power Mode|NV Power Mode" \
"$AUDIT_FILE"
```

应确认：

-   架构为`aarch64`；
    
-   `/etc/nv_tegra_release`有有效输出；
    
-   `/usr/local/cuda`指向某个CUDA目录；
    
-   `nvcc --version`能够运行；
    
-   `gcc`、`g++`、`cmake`、`ninja`可用；
    
-   根目录和Home目录有足够空间；
    
-   `nvpmodel`能读取当前功耗模式。
    

Jetson Orin Nano的CUDA Compute Capability是**8.7**，后面的CMake配置会使用`87`。([NVIDIA Developer](https://developer.nvidia.com/cuda/gpus?source=post_page-----20244437e036---------------------------------------&utm_source=chatgpt.com "CUDA GPU Compute Capability | NVIDIA Developer"))

---

# 2\. 仅在`nvcc`不可用时修复CUDA

先检查：

```bash
command -v nvcc
ls -l /usr/local/cuda/bin/nvcc 2>/dev/null
ls -l /usr/local/cuda-*/bin/nvcc 2>/dev/null
```

## 情况A：文件存在，只是PATH没有配置

执行：

```bash
grep -q "HQSB CUDA ENV" "$HOME/.bashrc" || cat >> "$HOME/.bashrc" <<'EOF'

# ===== HQSB CUDA ENV =====
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# ===== END HQSB CUDA ENV =====
EOF

source "$HOME/.bashrc"

nvcc --version
```

## 情况B：`/usr/local/cuda/bin/nvcc`确实不存在

先模拟安装，查看会安装什么：

```bash
apt-cache policy nvidia-jetpack
sudo apt-get -s install nvidia-jetpack | tee "$HOME/jetson_audit/nvidia_jetpack_install_simulation.txt"
```

确认磁盘空间正常后：

```bash
sudo apt install nvidia-jetpack
```

安装完成后：

```bash
source "$HOME/.bashrc"
nvcc --version
```

NVIDIA为Jetson提供的标准宿主机安装方式是安装`nvidia-jetpack`元包；验证CUDA时可以运行CUDA Sample或自己的CUDA工作负载。([NVIDIA Docs](https://docs.nvidia.com/jetson/jetpack/6.2/install-setup/index.html?utm_source=chatgpt.com "How to Install and Configure JetPack SDK — JetPack 6.2 documentation"))

如果原本已经有`nvcc`，不要重复安装。

---

# 3\. 检查功耗模式，但暂不锁频

执行：

```bash
sudo nvpmodel -q
sudo jetson_clocks --show
```

当前先使用默认功耗模式完成编译和正确性验证。

在最后正式跑Benchmark时，再切换为性能模式。Jetson官方建议用`nvpmodel -q`查看可用模式，再用显示出的Mode ID执行`nvpmodel -m <ID>`；资源监控使用`tegrastats`，而不是把`nvidia-smi`作为主要工具。([NVIDIA Docs](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/howto.html?utm_source=chatgpt.com "How-to Guides — Jetson Orin Nano Developer Kit - User Guide"))

**注意执行顺序：**

```text
先设置 nvpmodel
再执行 jetson_clocks
```

一旦执行`jetson_clocks`，再修改`nvpmodel`可能需要重启。([NVIDIA Docs](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html?utm_source=chatgpt.com "Jetson Orin Nano Series, Jetson Orin NX Series, and Jetson AGX Orin Series — NVIDIA Jetson Linux Developer Guide"))

---

# 4\. 配置Jetson到GitHub的SSH连接

所有Git命令都以普通用户执行，**不要使用`sudo git`**。

## 4.1 配置Git身份

将下列内容替换成你的真实信息：

```bash
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub邮箱"
git config --global init.defaultBranch main

git config --global --list
```

邮箱可以使用GitHub提供的`noreply`邮箱。

## 4.2 检查已有SSH Key

```bash
ls -la "$HOME/.ssh"
```

不要覆盖已有密钥。

创建一把Jetson专用密钥：

```bash
ssh-keygen \
  -t ed25519 \
  -C "你的GitHub邮箱" \
  -f "$HOME/.ssh/id_ed25519_github_jetson"
```

建议设置Passphrase。

启动SSH Agent：

```bash
eval "$(ssh-agent -s)"
ssh-add "$HOME/.ssh/id_ed25519_github_jetson"
ssh-add -l
```

GitHub官方建议使用Ed25519密钥，并将私钥加入`ssh-agent`管理。([GitHub Docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux&utm_source=chatgpt.com "Generating a new SSH key and adding it to the ssh-agent - GitHub Docs"))

## 4.3 配置SSH

先检查是否已有GitHub配置：

```bash
grep -n -A 5 -B 1 "Host github.com" "$HOME/.ssh/config" 2>/dev/null || true
```

如果没有，执行：

```bash
touch "$HOME/.ssh/config"
chmod 600 "$HOME/.ssh/config"

cat >> "$HOME/.ssh/config" <<'EOF'

Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github_jetson
    IdentitiesOnly yes
EOF
```

设置权限：

```bash
chmod 700 "$HOME/.ssh"
chmod 600 "$HOME/.ssh/id_ed25519_github_jetson"
chmod 644 "$HOME/.ssh/id_ed25519_github_jetson.pub"
```

## 4.4 将公钥添加到GitHub

只复制`.pub`文件：

```bash
cat "$HOME/.ssh/id_ed25519_github_jetson.pub"
```

在本地浏览器进入GitHub：

```text
Settings
→ SSH and GPG keys
→ New SSH key
```

标题建议：

```text
Jetson Orin Nano Super - HQSB
```

**绝对不要复制或上传：**

```text
~/.ssh/id_ed25519_github_jetson
```

那个是私钥。

## 4.5 测试连接

```bash
ssh -T git@github.com
```

成功时会看到：

```text
Hi YOUR_USERNAME! You've successfully authenticated,
but GitHub does not provide shell access.
```

这个测试命令成功认证后仍可能返回退出码1，这是GitHub的正常行为。([GitHub Docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection?platform=windows&utm_source=chatgpt.com "Testing your SSH connection - GitHub Docs"))

如果22端口超时：

```bash
ssh -T -p 443 git@ssh.github.com
```

若443可用，把`~/.ssh/config`中的GitHub配置改为：

```text
Host github.com
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/id_ed25519_github_jetson
    IdentitiesOnly yes
```

GitHub官方支持通过`ssh.github.com:443`建立SSH连接。([GitHub Docs](https://docs.github.com/en/authentication/troubleshooting-ssh/using-ssh-over-the-https-port?apiVersion=2022-11-28&utm_source=chatgpt.com "Using SSH over the HTTPS port - GitHub Docs"))

---

# 5\. 在GitHub建立空仓库

在GitHub网页创建：

```text
Repository name: HeteroQuantServeBench
Visibility: Private
```

初期建议设为Private，清理完环境信息和代码后再公开。

创建仓库时不要勾选：

-   Add a README；
    
-   Add `.gitignore`；
    
-   Choose a license。
    

本地会自己生成这些文件。

---

# 6\. 建立项目目录架构

进入项目目录：

```bash
mkdir -p "$HQSB_HOME"
cd "$HQSB_HOME"
```

建立顶层结构：

```bash
mkdir -p \
  configs/models \
  configs/quantization \
  configs/backends \
  configs/benchmarks \
  docs/architecture \
  docs/hardware \
  docs/adr \
  hqsb/core \
  hqsb/quant \
  hqsb/backends \
  hqsb/serving \
  hqsb/benchmark \
  ops/cuda/common \
  ops/cuda/device_query \
  ops/cuda/rmsnorm \
  ops/triton \
  ops/ascend \
  cpp/common \
  cpp/edge_worker \
  tests/unit \
  tests/correctness \
  tests/integration \
  tests/test_vectors \
  benchmarks/raw \
  benchmarks/normalized \
  reports/jetson \
  scripts/env \
  scripts/bench \
  scripts/setup
```

查看：

```bash
tree -d -L 3
```

预期顶层结构：

```text
HeteroQuantServeBench
├── benchmarks
├── configs
├── cpp
├── docs
├── hqsb
├── ops
│   ├── ascend
│   ├── cuda
│   └── triton
├── reports
├── scripts
└── tests
```

---

# 7\. 创建README和.gitignore

## 7.1 README

```bash
cat > README.md <<'EOF'
# HeteroQuantServeBench

Heterogeneous quantization, kernel optimization, serving, and benchmarking
platform for NVIDIA CUDA and Huawei Ascend C/CANN backends.

## Current milestone

Milestone 0 focuses on the Jetson Orin Nano Super CUDA baseline:

- reproducible hardware/software manifest;
- CUDA device discovery;
- CMake-based CUDA build;
- CPU reference and CUDA RMSNorm baseline;
- correctness and latency measurements;
- tegrastats/Nsight-ready profiling workflow.

## Planned components

- QuantLab: RTN, GPTQ, AWQ, SmoothQuant and paper-method reproduction
- KernelLab: CUDA, Triton and Ascend C operators
- Runtime adapters: TensorRT, llama.cpp, vLLM and Ascend runtimes
- ServeFabric: OpenAI-compatible heterogeneous inference gateway
- BenchLab: latency, throughput, memory, accuracy and energy reports

## Hardware

- NVIDIA Jetson Orin Nano Super 8GB
- Orange Pi AI Pro 20T
- On-demand NVIDIA datacenter/desktop GPU instances

## Status

Work in progress.
EOF
```

## 7.2 `.gitignore`

```bash
cat > .gitignore <<'EOF'
# Build
build/
cmake-build-*/
out/
dist/
*.o
*.a
*.so
*.d
*.tmp

# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Secrets
.env
.env.*
*.pem
*.key
credentials*
secrets/

# SSH keys must never enter the repository
id_rsa*
id_ed25519*

# Models and large artifacts
models/
artifacts/
datasets/
checkpoints/
*.safetensors
*.gguf
*.onnx
*.engine
*.plan
*.pt
*.pth
*.bin

# Profiling files
*.ncu-rep
*.nsys-rep
*.qdrep

# Generated logs
*.log

# Raw transient benchmark data
benchmarks/raw/*
!benchmarks/raw/.gitkeep

# OS
.DS_Store
Thumbs.db
EOF

touch benchmarks/raw/.gitkeep
```

---

# 8\. 创建第一版环境采集脚本

```bash
cat > scripts/env/collect_jetson_env.sh <<'EOF'
#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${1:-$ROOT_DIR/docs/hardware/jetson_environment.md}"

mkdir -p "$(dirname "$OUTPUT")"

{
  echo "# Jetson Environment Manifest"
  echo
  echo "Generated: $(date --iso-8601=seconds)"
  echo

  echo "## Device"
  echo '```text'
  tr -d '\0' < /proc/device-tree/model 2>/dev/null || true
  echo
  uname -m
  echo '```'
  echo

  echo "## Jetson Linux / L4T"
  echo '```text'
  cat /etc/nv_tegra_release 2>/dev/null || true
  echo '```'
  echo

  echo "## Operating system"
  echo '```text'
  cat /etc/os-release
  echo '```'
  echo

  echo "## CUDA"
  echo '```text'
  echo "CUDA_HOME=${CUDA_HOME:-not-set}"
  echo "nvcc=$(command -v nvcc 2>/dev/null || echo not-found)"
  readlink -f /usr/local/cuda 2>/dev/null || true
  nvcc --version 2>/dev/null || true
  echo '```'
  echo

  echo "## Toolchain"
  echo '```text'
  gcc --version | head -n 1
  g++ --version | head -n 1
  cmake --version | head -n 1
  ninja --version 2>/dev/null || true
  python3 --version
  git --version
  echo '```'
  echo

  echo "## NVIDIA packages"
  echo '```text'
  dpkg-query -W \
    nvidia-l4t-core \
    nvidia-jetpack \
    nvidia-jetpack-runtime \
    nvidia-jetpack-dev \
    2>/dev/null || true
  echo '```'
  echo

  echo "## Memory"
  echo '```text'
  free -h
  echo '```'
  echo

  echo "## Storage"
  echo '```text'
  df -h /
  df -h "$HOME"
  lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS
  echo '```'
  echo

  echo "## Power mode"
  echo '```text'
  sudo nvpmodel -q 2>&1 || true
  echo '```'
  echo

  echo "## Clock state"
  echo '```text'
  sudo jetson_clocks --show 2>&1 || true
  echo '```'

} > "$OUTPUT"

echo "Environment manifest written to: $OUTPUT"
EOF

chmod +x scripts/env/collect_jetson_env.sh
```

先执行一次：

```bash
./scripts/env/collect_jetson_env.sh
```

查看：

```bash
less docs/hardware/jetson_environment.md
```

确认里面没有：

-   IP地址；
    
-   MAC地址；
    
-   SSH Key；
    
-   Token；
    
-   用户密码；
    
-   私有仓库凭证。
    

---

# 9\. 初始化Git仓库并做第一个Commit

```bash
cd "$HQSB_HOME"

git init -b main
git status
```

空目录不会被Git记录，给主要目录创建`.gitkeep`：

```bash
find \
  configs \
  hqsb \
  ops/triton \
  ops/ascend \
  cpp \
  tests \
  benchmarks/normalized \
  reports \
  -type d -empty \
  -exec touch "{}/.gitkeep" \;
```

提交：

```bash
git add \
  README.md \
  .gitignore \
  configs \
  docs \
  hqsb \
  ops \
  cpp \
  tests \
  benchmarks \
  reports \
  scripts

git status

git commit -m "chore: initialize HeteroQuantServeBench structure"
```

检查：

```bash
git log --oneline --decorate -n 3
```

暂时还不Push，先完成CUDA基线。

---

# 10\. 创建CUDA错误检查工具

```bash
cat > ops/cuda/common/cuda_check.cuh <<'EOF'
#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "      \
                << cudaGetErrorString(error) << " (" << static_cast<int>(error) \
                << ")" << std::endl;                                            \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)
EOF
```

---

# 11\. 创建设备探测程序

```bash
cat > ops/cuda/device_query/device_query.cu <<'EOF'
#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  if (device_count <= 0) {
    std::cerr << "No CUDA device detected." << std::endl;
    return 1;
  }

  int driver_version = 0;
  int runtime_version = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

  std::cout << "CUDA device count: " << device_count << "\n";
  std::cout << "CUDA driver version: "
            << driver_version / 1000 << "."
            << (driver_version % 1000) / 10 << "\n";
  std::cout << "CUDA runtime version: "
            << runtime_version / 1000 << "."
            << (runtime_version % 1000) / 10 << "\n\n";

  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUDA_CHECK(cudaSetDevice(device));

    std::size_t free_memory = 0;
    std::size_t total_memory = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));

    std::cout << "Device " << device << "\n";
    std::cout << "  Name: " << prop.name << "\n";
    std::cout << "  Compute capability: "
              << prop.major << "." << prop.minor << "\n";
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cout << "  Global memory: "
              << std::fixed << std::setprecision(2)
              << static_cast<double>(total_memory) / (1024.0 * 1024.0 * 1024.0)
              << " GiB\n";
    std::cout << "  Currently free memory: "
              << static_cast<double>(free_memory) / (1024.0 * 1024.0 * 1024.0)
              << " GiB\n";
    std::cout << "  Shared memory per block: "
              << prop.sharedMemPerBlock / 1024.0 << " KiB\n";
    std::cout << "  Registers per block: " << prop.regsPerBlock << "\n";
    std::cout << "  Warp size: " << prop.warpSize << "\n";
    std::cout << "  Max threads per block: "
              << prop.maxThreadsPerBlock << "\n";
    std::cout << "  Max block dimensions: "
              << prop.maxThreadsDim[0] << " x "
              << prop.maxThreadsDim[1] << " x "
              << prop.maxThreadsDim[2] << "\n";
    std::cout << "  Memory bus width: "
              << prop.memoryBusWidth << " bits\n";
    std::cout << "  Integrated GPU: "
              << (prop.integrated ? "yes" : "no") << "\n";
    std::cout << "  Unified addressing: "
              << (prop.unifiedAddressing ? "yes" : "no") << "\n";
    std::cout << "  Managed memory: "
              << (prop.managedMemory ? "yes" : "no") << "\n";
    std::cout << "  Concurrent kernels: "
              << (prop.concurrentKernels ? "yes" : "no") << "\n";
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
EOF
```

创建设备探测CMake：

```bash
cat > ops/cuda/device_query/CMakeLists.txt <<'EOF'
add_executable(hqsb_device_query device_query.cu)

target_include_directories(
  hqsb_device_query
  PRIVATE
  ${PROJECT_SOURCE_DIR}/ops/cuda/common
)

target_compile_options(
  hqsb_device_query
  PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)
EOF
```

---

# 12\. 创建CPU参考＋CUDA RMSNorm基线

```bash
cat > ops/cuda/rmsnorm/rmsnorm_baseline.cu <<'EOF'
#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

struct Options {
  int rows = 512;
  int hidden = 1024;
  int warmup = 20;
  int iterations = 200;
  int threads = 256;
  float epsilon = 1.0e-5F;
};

int parse_positive_int(const char* value, const char* name) {
  const int parsed = std::stoi(value);
  if (parsed <= 0) {
    std::cerr << name << " must be positive." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return parsed;
}

Options parse_options(int argc, char** argv) {
  Options options;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];

    auto require_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << std::endl;
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };

    if (argument == "--rows") {
      options.rows = parse_positive_int(require_value("--rows"), "--rows");
    } else if (argument == "--hidden") {
      options.hidden =
          parse_positive_int(require_value("--hidden"), "--hidden");
    } else if (argument == "--warmup") {
      options.warmup =
          parse_positive_int(require_value("--warmup"), "--warmup");
    } else if (argument == "--iterations") {
      options.iterations =
          parse_positive_int(require_value("--iterations"), "--iterations");
    } else if (argument == "--threads") {
      options.threads =
          parse_positive_int(require_value("--threads"), "--threads");
    } else if (argument == "--help") {
      std::cout
          << "Usage: hqsb_rmsnorm_baseline [options]\n"
          << "  --rows N\n"
          << "  --hidden N\n"
          << "  --warmup N\n"
          << "  --iterations N\n"
          << "  --threads N\n";
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << argument << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  if ((options.threads & (options.threads - 1)) != 0 ||
      options.threads > 1024) {
    std::cerr << "--threads must be a power of two and <= 1024."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return options;
}

void rmsnorm_cpu(const std::vector<float>& input,
                 const std::vector<float>& weight,
                 std::vector<float>& output,
                 int rows,
                 int hidden,
                 float epsilon) {
  for (int row = 0; row < rows; ++row) {
    const std::size_t offset =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(hidden);

    double square_sum = 0.0;
    for (int column = 0; column < hidden; ++column) {
      const double value = input[offset + column];
      square_sum += value * value;
    }

    const float inverse_rms =
        1.0F / std::sqrt(static_cast<float>(square_sum / hidden) + epsilon);

    for (int column = 0; column < hidden; ++column) {
      output[offset + column] =
          input[offset + column] * inverse_rms * weight[column];
    }
  }
}

__global__ void rmsnorm_cuda_kernel(const float* input,
                                    const float* weight,
                                    float* output,
                                    int hidden,
                                    float epsilon) {
  extern __shared__ float shared_square_sum[];

  const int row = static_cast<int>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  const std::size_t row_offset =
      static_cast<std::size_t>(row) * static_cast<std::size_t>(hidden);

  float local_square_sum = 0.0F;

  for (int column = thread; column < hidden; column += blockDim.x) {
    const float value = input[row_offset + column];
    local_square_sum += value * value;
  }

  shared_square_sum[thread] = local_square_sum;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread < static_cast<int>(stride)) {
      shared_square_sum[thread] +=
          shared_square_sum[thread + static_cast<int>(stride)];
    }
    __syncthreads();
  }

  const float inverse_rms =
      rsqrtf(shared_square_sum[0] / static_cast<float>(hidden) + epsilon);

  for (int column = thread; column < hidden; column += blockDim.x) {
    output[row_offset + column] =
        input[row_offset + column] * inverse_rms * weight[column];
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);

  int current_device = 0;
  CUDA_CHECK(cudaGetDevice(&current_device));

  cudaDeviceProp device_properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&device_properties, current_device));

  const std::size_t element_count =
      static_cast<std::size_t>(options.rows) *
      static_cast<std::size_t>(options.hidden);

  std::vector<float> host_input(element_count);
  std::vector<float> host_weight(options.hidden);
  std::vector<float> host_reference(element_count);
  std::vector<float> host_output(element_count);

  std::mt19937 random_generator(42);
  std::uniform_real_distribution<float> input_distribution(-1.0F, 1.0F);
  std::uniform_real_distribution<float> weight_distribution(0.5F, 1.5F);

  for (float& value : host_input) {
    value = input_distribution(random_generator);
  }

  for (float& value : host_weight) {
    value = weight_distribution(random_generator);
  }

  rmsnorm_cpu(host_input,
              host_weight,
              host_reference,
              options.rows,
              options.hidden,
              options.epsilon);

  float* device_input = nullptr;
  float* device_weight = nullptr;
  float* device_output = nullptr;

  CUDA_CHECK(cudaMalloc(&device_input, element_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_weight, options.hidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, element_count * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(device_input,
                        host_input.data(),
                        element_count * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(device_weight,
                        host_weight.data(),
                        options.hidden * sizeof(float),
                        cudaMemcpyHostToDevice));

  const dim3 grid(static_cast<unsigned int>(options.rows));
  const dim3 block(static_cast<unsigned int>(options.threads));
  const std::size_t shared_memory_bytes =
      static_cast<std::size_t>(options.threads) * sizeof(float);

  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    rmsnorm_cuda_kernel<<<grid, block, shared_memory_bytes>>>(
        device_input,
        device_weight,
        device_output,
        options.hidden,
        options.epsilon);
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start_event{};
  cudaEvent_t stop_event{};

  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  CUDA_CHECK(cudaEventRecord(start_event));

  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    rmsnorm_cuda_kernel<<<grid, block, shared_memory_bytes>>>(
        device_input,
        device_weight,
        device_output,
        options.hidden,
        options.epsilon);
  }

  CUDA_CHECK(cudaEventRecord(stop_event));
  CUDA_CHECK(cudaEventSynchronize(stop_event));
  CUDA_CHECK(cudaGetLastError());

  float total_milliseconds = 0.0F;
  CUDA_CHECK(
      cudaEventElapsedTime(&total_milliseconds, start_event, stop_event));

  CUDA_CHECK(cudaMemcpy(host_output.data(),
                        device_output,
                        element_count * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float maximum_absolute_error = 0.0F;
  double mean_absolute_error = 0.0;

  for (std::size_t index = 0; index < element_count; ++index) {
    const float error =
        std::abs(host_output[index] - host_reference[index]);

    maximum_absolute_error =
        std::max(maximum_absolute_error, error);
    mean_absolute_error += static_cast<double>(error);
  }

  mean_absolute_error /= static_cast<double>(element_count);

  const double average_milliseconds =
      static_cast<double>(total_milliseconds) /
      static_cast<double>(options.iterations);

  const double bytes_per_iteration =
      static_cast<double>(element_count) *
      static_cast<double>(sizeof(float)) *
      3.0;

  const double effective_bandwidth_gb_per_second =
      bytes_per_iteration /
      (average_milliseconds / 1000.0) /
      1.0e9;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "device=" << device_properties.name << "\n";
  std::cout << "compute_capability="
            << device_properties.major << "."
            << device_properties.minor << "\n";
  std::cout << "rows=" << options.rows << "\n";
  std::cout << "hidden=" << options.hidden << "\n";
  std::cout << "threads=" << options.threads << "\n";
  std::cout << "warmup=" << options.warmup << "\n";
  std::cout << "iterations=" << options.iterations << "\n";
  std::cout << "max_abs_error=" << maximum_absolute_error << "\n";
  std::cout << "mean_abs_error=" << mean_absolute_error << "\n";
  std::cout << "average_kernel_ms=" << average_milliseconds << "\n";
  std::cout << "effective_bandwidth_gbps="
            << effective_bandwidth_gb_per_second << "\n";

  const bool correctness_passed = maximum_absolute_error <= 5.0e-4F;
  std::cout << "correctness="
            << (correctness_passed ? "PASS" : "FAIL") << "\n";

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_weight));
  CUDA_CHECK(cudaFree(device_output));

  return correctness_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
EOF
```

创建CMake文件：

```bash
cat > ops/cuda/rmsnorm/CMakeLists.txt <<'EOF'
add_executable(hqsb_rmsnorm_baseline rmsnorm_baseline.cu)

target_include_directories(
  hqsb_rmsnorm_baseline
  PRIVATE
  ${PROJECT_SOURCE_DIR}/ops/cuda/common
)

target_compile_options(
  hqsb_rmsnorm_baseline
  PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)
EOF
```

---

# 13\. 创建顶层CMakeLists

```bash
cat > CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.18)

project(
  HeteroQuantServeBench
  VERSION 0.1.0
  LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(
  CMAKE_RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

add_subdirectory(ops/cuda/device_query)
add_subdirectory(ops/cuda/rmsnorm)
EOF
```

---

# 14\. 创建一键构建和Baseline脚本

```bash
cat > scripts/bench/run_jetson_baseline.sh <<'EOF'
#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/jetson-release"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$ROOT_DIR/reports/jetson/$RUN_ID"

mkdir -p "$OUTPUT_DIR"

echo "Run ID: $RUN_ID"
echo "Output: $OUTPUT_DIR"

"$ROOT_DIR/scripts/env/collect_jetson_env.sh" \
  "$OUTPUT_DIR/environment.md"

cmake \
  -S "$ROOT_DIR" \
  -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  2>&1 | tee "$OUTPUT_DIR/configure.txt"

cmake \
  --build "$BUILD_DIR" \
  --parallel "$(nproc)" \
  2>&1 | tee "$OUTPUT_DIR/build.txt"

"$BUILD_DIR/bin/hqsb_device_query" \
  2>&1 | tee "$OUTPUT_DIR/device_query.txt"

: > "$OUTPUT_DIR/rmsnorm_runs.txt"

for run in 1 2 3 4 5; do
  echo "===== RUN $run =====" | tee -a "$OUTPUT_DIR/rmsnorm_runs.txt"

  "$BUILD_DIR/bin/hqsb_rmsnorm_baseline" \
    --rows 512 \
    --hidden 1024 \
    --warmup 50 \
    --iterations 500 \
    --threads 256 \
    2>&1 | tee -a "$OUTPUT_DIR/rmsnorm_runs.txt"
done

cat > "$OUTPUT_DIR/README.md" <<REPORT
# Jetson CUDA Baseline

- Run ID: $RUN_ID
- Build type: Release
- CUDA architecture: 87
- RMSNorm shape: rows=512, hidden=1024
- Warmup iterations: 50
- Measured iterations: 500
- Repetitions: 5

Files:

- environment.md
- configure.txt
- build.txt
- device_query.txt
- rmsnorm_runs.txt
- tegrastats.log, if captured separately
REPORT

echo
echo "Baseline completed."
echo "Results: $OUTPUT_DIR"
EOF

chmod +x scripts/bench/run_jetson_baseline.sh
```

---

# 15\. 做第二个Commit：CUDA代码

先检查：

```bash
cd "$HQSB_HOME"

git diff --check
git status
```

提交：

```bash
git add \
  CMakeLists.txt \
  ops/cuda \
  scripts/env \
  scripts/bench \
  .gitignore \
  README.md

git commit -m "feat(jetson): add CUDA environment probe and RMSNorm baseline"
```

查看历史：

```bash
git log --oneline --decorate -n 5
```

应看到类似：

```text
xxxxxxxx feat(jetson): add CUDA environment probe and RMSNorm baseline
xxxxxxxx chore: initialize HeteroQuantServeBench structure
```

---

# 16\. 第一轮编译：只验证正确性

执行：

```bash
cd "$HQSB_HOME"

cmake \
  -S . \
  -B build/jetson-release \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87

cmake \
  --build build/jetson-release \
  --parallel "$(nproc)"
```

确认二进制文件：

```bash
ls -lh build/jetson-release/bin
```

应包含：

```text
hqsb_device_query
hqsb_rmsnorm_baseline
```

## 16.1 运行设备探测

```bash
./build/jetson-release/bin/hqsb_device_query
```

关键结果应包括：

```text
CUDA device count: 1
Compute capability: 8.7
Integrated GPU: yes
Unified addressing: yes
```

可用内存不会正好等于8GB，因为Jetson CPU和GPU共享系统内存，系统本身也会占用一部分。

## 16.2 运行小规模正确性测试

```bash
./build/jetson-release/bin/hqsb_rmsnorm_baseline \
  --rows 16 \
  --hidden 256 \
  --warmup 5 \
  --iterations 20 \
  --threads 256
```

必须看到：

```text
correctness=PASS
```

## 16.3 运行正式Shape

```bash
./build/jetson-release/bin/hqsb_rmsnorm_baseline \
  --rows 512 \
  --hidden 1024 \
  --warmup 50 \
  --iterations 500 \
  --threads 256
```

仍必须是：

```text
correctness=PASS
```

当前阶段不要关注速度是否“够快”。这只是V0基线，后面要做：

-   Warp Shuffle；
    
-   向量化访存；
    
-   Half/half2；
    
-   Fused Residual＋RMSNorm；
    
-   Nsight瓶颈分析。
    

---

# 17\. 设置正式Benchmark功耗模式

先确认有稳定电源和主动散热。

查看模式：

```bash
sudo nvpmodel -q
```

输出中找到：

```text
MAXN_SUPER
```

及其Mode ID。

不要盲目照抄数字，以你设备实际显示的ID为准：

```bash
sudo nvpmodel -m <MAXN_SUPER对应的ID>
```

如果命令要求重启：

```bash
sudo reboot
```

重连SSH后：

```bash
cd "$HQSB_HOME"

sudo jetson_clocks --store "$HOME/.jetsonclocks_before_hqsb.txt"
sudo jetson_clocks
sudo jetson_clocks --show
```

Jetson Orin Nano Super的公开功耗模式配置中，MAXN\_SUPER能够提高CPU、GPU和内存频率；具体Mode ID和可用模式仍应以本机`nvpmodel -q`输出为准。([NVIDIA Docs](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html?utm_source=chatgpt.com "Jetson Orin Nano Series, Jetson Orin NX Series, and Jetson AGX Orin Series — NVIDIA Jetson Linux Developer Guide"))

---

# 18\. 使用Tegrastats监控正式Baseline

在**终端B**执行：

```bash
cd "$HQSB_HOME"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
mkdir -p "reports/jetson/$RUN_ID"

echo "$RUN_ID" > "$HOME/hqsb_current_run_id"

sudo tegrastats \
  --interval 500 \
  --logfile "$HQSB_HOME/reports/jetson/$RUN_ID/tegrastats.log"
```

保持运行。

`tegrastats`会记录CPU、GPU、内存、频率、温度和功耗相关数据；官方也支持通过`--interval`和`--logfile`写入日志。([NVIDIA Docs](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html?utm_source=chatgpt.com "Tegrastats Utility — NVIDIA Jetson Linux Developer Guide 1 documentation"))

在**终端A**执行完整Baseline：

```bash
cd "$HQSB_HOME"

./scripts/bench/run_jetson_baseline.sh
```

注意：这个脚本会自己生成另一个Run ID。因此更简单的做法是，在脚本打印出：

```text
Output: /home/.../reports/jetson/YYYYMMDD_HHMMSS
```

后，在终端B停止`tegrastats`：

```text
Ctrl+C
```

再把日志移动到脚本生成的结果目录：

```bash
LATEST_REPORT="$(find "$HQSB_HOME/reports/jetson" \
  -mindepth 1 -maxdepth 1 -type d \
  -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"

MONITOR_RUN_ID="$(cat "$HOME/hqsb_current_run_id")"

mv \
  "$HQSB_HOME/reports/jetson/$MONITOR_RUN_ID/tegrastats.log" \
  "$LATEST_REPORT/tegrastats.log"

rmdir \
  "$HQSB_HOME/reports/jetson/$MONITOR_RUN_ID" \
  2>/dev/null || true

echo "$LATEST_REPORT"
```

查看结果：

```bash
cat "$LATEST_REPORT/device_query.txt"
cat "$LATEST_REPORT/rmsnorm_runs.txt"
tail -n 20 "$LATEST_REPORT/tegrastats.log"
```

---

# 19\. 检查五次运行是否稳定

执行：

```bash
grep \
  -E "average_kernel_ms|effective_bandwidth_gbps|correctness" \
  "$LATEST_REPORT/rmsnorm_runs.txt"
```

应有5组数据，且全部：

```text
correctness=PASS
```

现在不设强制性能阈值，只检查：

-   五次结果没有数量级差异；
    
-   没有OOM；
    
-   没有CUDA Error；
    
-   温度没有持续失控；
    
-   GPU频率没有明显异常下降；
    
-   Tegrastats中没有持续功耗或热降频异常。
    

检查热状态：

```bash
grep -Eo \
'GPU@[0-9.]+C|CPU@[0-9.]+C|tj@[0-9.]+C|GR3D_FREQ [^ ]+' \
"$LATEST_REPORT/tegrastats.log" | tail -n 30
```

---

# 20\. 生成首次Baseline摘要

执行：

```bash
cat >> "$LATEST_REPORT/README.md" <<'EOF'

## Validation checklist

- [ ] CUDA device detected
- [ ] Compute capability is 8.7
- [ ] Release build completed
- [ ] Five RMSNorm runs completed
- [ ] All correctness checks passed
- [ ] tegrastats log captured
- [ ] No CUDA runtime error
- [ ] No out-of-memory failure

## Interpretation

This is the unoptimized FP32 RMSNorm V0 baseline.

The next optimization stages will evaluate:

1. warp-level reduction;
2. vectorized memory access;
3. FP16 and half2;
4. register/shared-memory trade-offs;
5. fused residual and RMSNorm;
6. Nsight Compute bottleneck analysis.
EOF
```

然后编辑勾选结果：

```bash
nano "$LATEST_REPORT/README.md"
```

将已经通过的：

```text
- [ ]
```

改成：

```text
- [x]
```

---

# 21\. 做第三个Commit：记录真实设备Baseline

先检查是否包含敏感信息：

```bash
grep -RniE \
'password|token|secret|private.key|BEGIN.*PRIVATE|ssh-rsa|OPENSSH PRIVATE' \
docs reports scripts ops \
|| true
```

查看待提交文件：

```bash
git status --short
```

提交环境清单和Baseline结果：

```bash
git add \
  docs/hardware/jetson_environment.md \
  reports/jetson

git commit -m "docs(jetson): record initial CUDA baseline"
```

检查三次Commit：

```bash
git log --oneline --decorate -n 5
```

期望：

```text
xxxxxxxx docs(jetson): record initial CUDA baseline
xxxxxxxx feat(jetson): add CUDA environment probe and RMSNorm baseline
xxxxxxxx chore: initialize HeteroQuantServeBench structure
```

---

# 22\. 连接远程仓库并Push

替换`YOUR_GITHUB_USERNAME`：

```bash
cd "$HQSB_HOME"

git remote add origin \
  git@github.com:YOUR_GITHUB_USERNAME/HeteroQuantServeBench.git

git remote -v
```

Push：

```bash
git push -u origin main
```

检查状态：

```bash
git status
git branch -vv
```

应显示：

```text
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean
```

---

# 23\. Benchmark结束后恢复时钟

如果不再做性能测试：

```bash
sudo jetson_clocks \
  --restore "$HOME/.jetsonclocks_before_hqsb.txt"
```

查看：

```bash
sudo jetson_clocks --show
sudo nvpmodel -q
```

若需要切换回其他`nvpmodel`模式，而系统拒绝或提示需要重启：

```bash
sudo reboot
```

---

# 24\. 今天的最终验收命令

依次执行：

```bash
cd "$HQSB_HOME"

nvcc --version

./build/jetson-release/bin/hqsb_device_query

./build/jetson-release/bin/hqsb_rmsnorm_baseline \
  --rows 512 \
  --hidden 1024 \
  --warmup 50 \
  --iterations 500 \
  --threads 256

git status

git log --oneline --decorate -n 5

git remote -v

git ls-remote origin HEAD

find reports/jetson -maxdepth 2 -type f | sort
```

全部满足以下条件即完成今天目标：

```text
[ ] nvcc正常
[ ] CUDA设备可见
[ ] Compute Capability为8.7
[ ] CMake/Ninja编译成功
[ ] RMSNorm correctness=PASS
[ ] 五次Baseline完成
[ ] Tegrastats日志存在
[ ] 环境清单存在
[ ] 三个Git Commit存在
[ ] GitHub Push成功
[ ] git status干净
```

今天形成的是可信的 **V0 CUDA基线**。下一阶段应在这个基线上依次实现Warp Shuffle Reduction、FP16/half2、向量化访存和Nsight Compute分析，而不是立即开始堆叠更多算子。执行中某一步失败时，保留该命令及其完整输出，后续直接从对应检查点排障。