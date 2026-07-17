# 每 2 分钟崩一次：一次 VS Code 扩展宿主 OOM 的完整根因分析

> 一次真实故障的复盘。从"服务莫名其妙重启"到"定位到 4 GB 这个常量"，
> 中间排除了 7 个候选原因、3 次修复尝试全部被证伪、最终在压缩后的 `server-main.js` 里
> 找到两行代码。附可复用手册与 4 个反模式。
>
> 证据等级：`development`（本机实测）。所有数字均为现场命令输出，标注了采集时间。
> 项目内的原始记录见 `AGENTS.md` §8/§9。

---

## TL;DR

| | |
|---|---|
| **症状** | AI 编码会话每 2–4 分钟被打断一次，对话与工具调用丢失，恢复后提示 `command 'xxx' not found` |
| **表象** | "VS Code 服务崩了" / "网络抖动" / "机器内存不够" |
| **真因** | **扩展宿主（Extension Host）进程的 V8 堆撞上 4 GB 默认上限 → `FATAL ERROR: Reached heap limit` → `abort()` → SIGABRT** |
| **为什么难查** | ① 死的是**子进程**，主进程健康；② V8 默认堆上限是**编译期常量**，与物理内存无关（机器有 503 GB 也一样）；③ 官方文档给的三条修复路径在当前 CLI 布局下**全部失效**，且失效方式各不相同 |
| **为什么看似"服务重启"** | VS Code 检测到宿主进程非正常退出后**自动重建**，客户端体验就是"重连了" |
| **最终修复** | 在 `server/bin/code-server` 的 exec 行前注入 **node CLI 实参** `--max-old-space-size=16384`（不是环境变量——它会被人为剥离） |
| **根因之外的放大器** | 会话历史体积。把 220 KB 文档一次性读进上下文 ≈ 自己往堆里灌 220 KB 常驻字符串 |
| **一句话教训** | 物理内存再大也救不了堆：**堆上限是常量，不是比例**；而"环境变量注入"这条最直觉的路，被框架主动堵死了 |

---

## 1. 症状：不是"崩了"，是"反复重生"

最初的现象完全不像是 OOM：

- 会话与远程服务反复"重启"，正在进行的**对话、工具调用、流式输出被打断**；
- 恢复后客户端日志出现：
  ```
  Error: command 'codebuddy.show.sidebar' not found
  Error: command 'codebuddy.session.commitNewSession' not found
  ```
- **频次**：同一远程会话内，扩展宿主被拉起 **7 次**，其中 SIGABRT 终止 **6 次**：

  | # | 时间 |
  |---|---|
  | 1 | 2026-09-18 23:55:55 |
  | 2 | 2026-09-19 00:12:05 |
  | 3 | 2026-09-19 01:49:02 |
  | 4 | 2026-09-19 01:51:35 |
  | 5 | 2026-09-19 01:54:24 |
  | 6 | 2026-09-19 01:58:32 |

  间隔 **2–4 分钟**，前期还更长 → 典型的"越用越快崩"。

那两条 `command not found` 是**关键线索**，很多人会忽略它：
它说明**扩展的命令尚未注册**，也就是扩展宿主进程是**全新的、还没初始化完**的。
所以这不是"命令坏了"，而是**承载命令的那个进程刚被重建**。

> **推论**：问题在"进程生命周期"，不在"功能实现"。看到 `command not found` 就该先查宿主进程，而不是去翻扩展代码。

---

## 2. 第一原则：先建立可观测性，再谈猜测

猜测之前，先确定**证据在哪**。VS Code 的远程日志布局：

```bash
# 主会话日志目录（每次 server 启动新建一个）
ls -1 ~/.vscode-server/data/logs/
#   20260919T163742/          # 会话目录 = server 启动时间戳
#     remoteagent.log         # ← 主进程 / 宿主的生死记录都在这里
#     exthost1/               # ← 每个扩展宿主一个子目录
#       exthost.log           #   宿主自身的日志
#       <publisher>.<ext>/    #   各扩展的输出通道

# 扩展宿主的进程身份
ps -eo pid,rss,cmd | grep "[t]ype=extensionHost"
```

三条**必查**命令（顺序执行，不要跳步）：

```bash
# ① 是谁杀的？——找终止信号
grep -h "signal: SIGABRT\|signal: SIGKILL\|exit code" remoteagent.log | tail -5

# ② 为什么杀？——找 V8 的死亡遗言
grep -h "Reached heap limit" remoteagent.log | tail -1
grep -h "OOMErrorHandler"   remoteagent.log | tail -1

# ③ 什么时刻？——建立时间线
grep -h "signal: SIGABRT" remoteagent.log | tail -10
```

> ⚠️ **取证命令本身会被日志记录**。一次 `grep -rh "SIGABRT" <logs>/` 让我得到 48 条"命中"，
> 打开一看——全是**我自己那条 grep 命令**被扩展的终端执行器记进了它的输出日志（48 条里 46 条是自指）。
> **教训**：取证时用 `grep -h` 明确指定文件，不要 `-r` 全目录递归；
> 并且**永远打开命中内容看一眼**，不要只看计数。

---

## 3. 排除法：7 个候选原因，逐一证伪

OOM 类问题的第一反应通常是"内存不够"。**这次恰好全部相反**——机器极其空闲。

| # | 候选原因 | 实测证据 | 判定 |
|---|---|---|---|
| 1 | 容器内存不足 / 被 OOM killer 杀 | cgroup `memory.max` 见 §3.1；`memory.current` ≈ 5.7 GB；`oom_kill 0` | ❌ 排除 |
| 2 | 宿主物理内存不够 | `free` 显示可用 682 GB | ❌ 排除 |
| 3 | 磁盘写满导致日志/缓存异常 | 根分区 30 GB，已用 48%，可用 16 GB | ❌ 排除 |
| 4 | `inotify` 句柄耗尽（大仓库常见） | `max_user_instances=128`，实际 inotify 实例仅 **8** | ❌ 排除 |
| 5 | 内核层面异常 / OOM-killer | `dmesg` 无 OOM、无 SIGABRT 相关记录 | ❌ 排除 |
| 6 | 网络抖动导致"重连" | 无网络错误日志；宿主**确实是被 SIGABRT 终止的**（有信号记录） | ❌ 排除 |
| 7 | 扩展代码 bug（命令未注册） | `command not found` 是**宿主重建的后果**，不是原因（§1） | ❌ 排除 |

### 3.1 一个没复现的数字（记录会漂移，数字要现测）

`AGENTS.md` §8.2 当时记录的是 cgroup `memory.max` = **90 GB**。
本次复测（同一台机器）得到的是 **62.0 GB**：

```bash
$ cat /proc/self/cgroup
0::/                                  # 已在 cgroup 根，无嵌套父限额
$ cat /sys/fs/cgroup/memory.max
66571993088                           # = 62.0 GB
$ awk '/^MemTotal|^MemAvailable/{printf "%-14s %.1f GB\n",$1,$2/1048576}' /proc/meminfo
MemTotal:      503.5 GB
MemAvailable:  366.0 GB
```

**处理方式**：如实登记为"**未能复现**"，而不是挑一个数写进结论。
（可能来源：实例规格在这两次采样之间被调整过；或当时读到的是另一层 cgroup。）
**这不影响根因判定**——因为无论配额是 62 还是 90 GB，都远大于 V8 的 4 GB 堆上限，
瓶颈**都不是系统内存**。这正是记录漂移时应当做的：**确认它是否影响结论，然后继续**。

---

## 4. 锁定根因：读 V8 的死亡遗言

在 `remoteagent.log` 里，宿主的死亡现场是这样一串（**原样摘录**）：

```text
<8985><stderr> [8985:0x866d000] 230083 ms: Mark-Compact 3729.4 (4215.8) -> 3665.5 (4184.5) MB,
    pooled: 70 MB, 210.81 / 22.68 ms (average mu = 0.797, current mu = 0.576)
    allocation failure; scavenge might not succeed
<8985><stderr> [8985:0x866d000] 230576 ms: Mark-Compact 3766.5 (4228.6) -> 3709.7 (4228.8) MB ...
<8985><stderr> FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory
<8985><stderr>  1: 0x74eae8 node::OOMErrorHandler(char const*, v8::OOMDetails const&) [.../server/node]
[ExtensionHostConnection] <8985> Extension Host Process exited with code: null, signal: SIGABRT.
```

这段日志信息量极大，逐行翻译：

| 片段 | 含义 |
|---|---|
| `Mark-Compact` | V8 正在做**全量标记压缩 GC**——已经是最重的回收手段 |
| `3729.4 (4215.8) -> 3665.5 (4184.5) MB` | 存活对象 3729 MB，**堆已提交 4215 MB**；GC 后只降到 3665 MB |
| `210.81 / 22.68 ms` | 标记 210 ms / 压缩 22 ms——**单次 GC 停顿已到 200 ms 量级** |
| `allocation failure` | 这次 GC 是**被分配失败逼出来的**，不是定时触发 |
| `scavenge might not succeed` | V8 已预告：**新生代回收也救不了** |
| `average mu = 0.797, current mu = 0.576` | **mutator utilization 从 0.797 掉到 0.576**——见 §5 |
| `Reached heap limit` | **堆上限到了**。这是死刑判决本身 |
| `OOMErrorHandler` | Node 的 `node::OOMErrorHandler` → 调用 `abort()` |
| `code: null, signal: SIGABRT` | `abort()` 发的就是 SIGABRT。**不是 OOM killer，是进程自己自杀** |

再看两个时间数字：

- 宿主从启动到 OOM 只用了 **~230 秒**；
- 堆从 0 涨到 3.7 GB → **约 16 MB/s** 的无界增长，且 GC 已 thrash。

**这不是"缓慢累积撑爆"，而是"某个动作在快速灌入"。** 再往前翻 4 秒：

```text
The client has reconnected.
```

**崩溃前 4 秒客户端刚重连。** 于是链条闭合了：

> 重连 → 扩展宿主重新激活全部扩展 → 未释放的旧状态 + 重新索引/重新加载 → 16 MB/s 灌入 → 4 GB 上限 → abort。

到这里根因已经确定，而**下一步的错误直觉是把注意力放到"降系统内存占用"上**——这正是 §8 要讲的第一个反模式。

---

## 5. 底层原理：V8 的堆为什么会"自杀"

这一节解释三个"为什么"，是整篇里最值得带走的部分。

### 5.1 上限从哪来，为什么它和物理内存无关

V8 的堆是**分代**的：

```text
┌──────────────── V8 Heap (heap_size_limit = 硬上限) ─────────────────┐
│  New Space (semi-space)          Old Space                          │
│  ── 新生代，小、回收频繁          ── 老生代，存活对象搬到这里          │
│     --max-semi-space-size            --max-old-space-size ← 我们调的 │
│                                                                     │
│  超过 heap_size_limit 且 GC 回收不掉 → node::OOMErrorHandler → abort │
└─────────────────────────────────────────────────────────────────────┘
```

**关键事实：`heap_size_limit` 不按物理内存缩放。** 同一条命令的对照实验：

```bash
$ NODE=~/.vscode-server/cli/servers/Stable-*/server/node   # node v24.18.1

$ "$NODE" -e 'const v8=require("v8"),os=require("os");
  const h=v8.getHeapStatistics().heap_size_limit;
  console.log("totalmem="+(os.totalmem()/1073741824).toFixed(1)+"GB",
              "heap_limit="+(h/1073741824).toFixed(2)+"GB",
              "ratio="+(h/os.totalmem()).toFixed(4));'
totalmem=503.5GB heap_limit=4.19GB ratio=0.0083     # ← 只有 0.83%
```

**503.5 GB 的机器，默认堆上限 4.19 GB。** 上限来自 V8 的 `ResourceConstraints` **默认值**
（本机 node v24.18.1 实测），不是"物理内存的百分比"。

> **为什么这个误解如此普遍？因为 JVM 是反过来的。**
> JVM 的 `-XX:MaxRAMPercentage` 默认 **25%**——内存越大默认堆越大。
> 对 Java 工程师来说，"机器有 503 GB，堆当然很大"是**正确直觉**；
> 在 Node 上这条直觉**直接失效**。这是本次故障里最容易踩的认知陷阱。
>
> 实用对照（同一 node 二进制，实测）：

| node 参数 | `heap_size_limit` |
|---|---|
| 不传（默认） | **4.19 GB** ← 崩溃日志里的 `(4215.8) MB` 与之吻合 |
| `--max-old-space-size=8192` | 8.19 GB |
| `--max-old-space-size=16384` | 16.19 GB |
| `--max-old-space-size=32768` | 32.19 GB |

### 5.2 为什么是 `abort()` 而不是抛一个异常

堆耗尽时 V8 走的是 **fail-stop**，不是 **fail-soft**。原因很实际：
**抛一个 JS 异常本身需要分配对象**——而此时分配已经失败了。
在 C++ 层，堆状态已无法安全回滚，`node::OOMErrorHandler` 只能 `abort()`。

**所以 `signal: SIGABRT` 是一个"进程自杀"信号**——这一点很重要：

| 信号 | 来源 | 含义 |
|---|---|---|
| `SIGKILL` | 内核 OOM-killer | 系统内存不够，别人杀了你 |
| **`SIGABRT`** | **进程自己** | **自己撞到了内部上限，主动 abort** |
| `SIGSEGV` | 硬件/代码 | 段错误 |

看到 `SIGABRT` + `code: null`，就该**立刻怀疑内部上限**，而不是系统内存。
本次一开始往"机器内存不够"方向查，方向就是错的——**信号类型第 0 秒就能告诉你答案**。

### 5.3 `mu = 0.576` 是什么意思：GC 已经在空转

```
Mark-Compact ... (average mu = 0.797, current mu = 0.576) allocation failure
```

`mu` = **mutator utilization**：你的业务代码（mutator）相对 GC 所占的时间比例。

| mu | 含义 |
|---|---|
| 0.99 | 99% 时间跑业务，1% 做 GC —— 健康 |
| 0.797 | 平均有 20% 时间在 GC —— 开始吃力 |
| **0.576** | **瞬时 42% 时间在做 GC** —— 典型 **GC thrashing** |

再配合 `allocation failure` 这个词：这次 GC 不是定时触发的，而是**分配失败逼出来的**。
V8 已经到了"每次想分配新对象都得先做一次全量压缩"的地步——
**回收速度追不上分配速度**。此时即使把上限翻倍，也只是把同一过程延长而已（见 §8 反模式 4）。

### 5.4 顺带解释：为什么 RSS 会比堆上限大

一个常见困惑——崩溃时 **RSS 5.99 GB > 堆上限 4.19 GB**。因为 RSS 统计的是
**进程占住的全部物理页**，而堆只是其中一部分：

| 组成 | 本机实测（当前宿主） |
|---|---|
| `VmRSS`（总量） | 4730 MB |
| ├ `Anonymous`（私有匿名页：堆 + 原生分配） | 4630 MB |
| └ `Shared_Clean`（共享库等） | 63 MB |
| `VmSwap` | 0 MB |
| `VmSize`（虚拟地址空间，**不是真内存**） | 48141 MB |

非堆部分包括：JIT 代码段、external/ArrayBuffer、glibc malloc arena、
C++ 原生模块分配（在扩展宿主里这项很大）、被 touch 的共享库页。

> **所以 RSS 是"水位"，堆上限是"警戒线"。崩溃是水位撞破警戒线；
> RSS 只是我用来判断风险的尺子，调不了，也不需要调。**

---

## 6. 为什么三条"标准做法"全部失效

知道要传 `--max-old-space-size` 只是开始。**真正难的是：怎么把它送进宿主进程。**
三条看起来都对的路径，全部失灵，而且**失败方式各不相同**——这也是本次故障最耗时的部分。

### 6.1 尝试 1：官方 hook（`server-env-setup`）——不是"没生效"，是"从未被执行"

VS Code Remote 有个官方机制：`~/.vscode-server/server-env-setup`，
文档说它会在 server 启动前被 `source`。先装一个**探针**（而不是直接下结论）：

```bash
cat >> ~/.vscode-server/server-env-setup <<'EOF'
echo "$(date -u +%FT%TZ) sourced by pid=$$" >> /tmp/hqsb-env-setup-ran.log
EOF
```

**结果**：探针文件 `/tmp/hqsb-env-setup-ran.log` **从未出现**。
但"没出现"也可能是权限/路径问题，所以做**带对照组的机制核查**（这一步是关键）：

```bash
$ grep -a -c serve-web  <CLI 二进制>              # 4   ← 对照组
$ grep -a -c env-setup  <CLI 二进制>              # 0
$ grep -c    server-env-setup  server/out/server-main.js   # 0
```

对照组 `serve-web` 能检出 4 次，证明**这个二进制是可被字符串检索的**；
而 `env-setup` 在一次都检不到 → **当前 "CLI + servers" 布局下，
服务端根本不读这个 hook**。它不是"配错了"，是**这条路径在这个版本里已经不存在**。

> **方法论**：判定"某机制失效"时，必须带一个**已知能命中的对照组**。
> 否则你分不清"机制失效"和"我的检索方法不对"。

### 6.2 尝试 2：`Developer: Reload Window` —— 它不重启 server

```bash
$ ps -eo pid,lstart,cmd | grep "[s]erver-main.js"
1312  2026-09-18 23:44:52   ...      # ← 重载窗口后，server 的 pid 和启动时间纹丝不动
```

**`Reload Window` 只重载工作区，不重启 VS Code Server。**
而 `server-env-setup` 是"server 启动前"的 hook → 不重启 server ⇒ 永远不触发。

### 6.3 尝试 3：关闭窗口再重连 —— server 会继续存活

直觉上"关掉 VS Code 窗口 = 服务停了"。实测：

```bash
$ ps -eo pid,lstart,cmd | grep "[s]erver-main.js"
1312  2026-09-18 23:44:52   ...      # ← 关了窗口，server 还是它，还是那个启动时间
```

**VS Code Server 在窗口关闭后继续存活**（这是它的设计：下次重连直接接回，省去冷启动）。
所以"关窗重连" ≠ "重启 server"。

**正确的重启方式**（二选一）：

```bash
# 客户端 GUI：Ctrl+Shift+P → "Remote-SSH: Kill VS Code Server on Host"
# 等价命令行：
ssh <host> "pkill -f .vscode-server"
```

执行后 server 确实变成新进程（`pid 17134`，起于 `02:54:32`，新日志目录 `logs/20260919T025434/`）
——**但 `/proc/17134/environ` 里依然没有 `NODE_OPTIONS`**。回到 §6.1 的结论：hook 根本没被执行。

### 6.4 一个差点误判的信号：`affinity` 的"生效又失效"

我们同时试了用 `extensions.experimental.affinity` 把每个扩展分到独立宿主。过程极具迷惑性：

| 时刻 | 实测 | 当时的判定 |
|---|---|---|
| 02:43 | 宿主数 **1 → 2**，各 ≈970 MB | ✅ 判定"隔离已生效" |
| 02:57（server 重启后） | 宿主数回到 **1**，且 codebuddy / pylance / python / anyscale **全在同一宿主** | ❌ **推翻** |

**教训**：一次观测到"数字变了"不等于机制生效。
这里 02:43 的"2 个宿主"可能是别的窗口/别的宿主，**没有逐宿主核对扩展归属**就下了结论。
**正确的判据是**："不同宿主的日志里**不再同时出现** codebuddy 与 pylance"，
而不是"宿主总数 ≥ 2"。

> 这条直接导致判定**降级**：从"已生效"改成"未稳定生效 / 待复测"。
> **看到符合预期的数字时，要问一句"这个数字还有没有别的解释"。**

### 6.5 真正的注入点：去读压缩后的 `server-main.js`

环境变量这条路被 §6.1 判了死刑，但宿主 cmdline 里留了一条线索：

```text
server/node --dns-result-order=ipv4first .../bootstrap-fork --type=extensionHost
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^ 这个参数不是我们传的 → VS Code 自己构造了宿主的 node 参数
```

于是直接在 `server/out/server-main.js`（压缩后的单文件）里搜关键字，找到两段关键代码：

```js
// ① 环境变量被「定向剥离」——扩展宿主子进程的环境里删掉这些
function Hd(r){
  let t = new Set(["DEBUG","NODE_OPTIONS","VSCODE_NODE_OPTIONS","LD_PRELOAD","DYLD_INSERT_LIBRARIES"]);
  for (let e of Object.keys(r)) t.has(e.toUpperCase()) && delete r[e];
}

// ② 但 execArgv 会被转发，且 VS Code 自己往里塞参数
s.execArgv.unshift("--dns-result-order=ipv4first");
```

**这两行解释了全部现象**：

| 通道 | 是否到达宿主 | 原因 |
|---|---|---|
| 环境变量 `NODE_OPTIONS` | ❌ **被主动删除** | `Hd()` 显式剥掉，防止通过 `--require` 注入扩展宿主（**这是安全设计，不是 bug**） |
| **node CLI 实参** | ✅ **被继承** | `execArgv` 默认继承 `process.execArgv`，仅过滤 `--inspect*` |

> **这是整次排查的转折点**：环境变量是"应用层可写的通道"，框架出于安全考虑关掉了；
> 而 CLI 实参是"**进程启动者**才能控制的通道"——而 server 的启动者正是我们。

于是方案变得极简：**给 server 的 node 传一个 CLI 参数，宿主会自己继承过去。**

---

## 7. 解决方案

### 7.1 找到真正 exec node 的那个脚本

```bash
$ f=~/.vscode-server/cli/servers/Stable-<commit>/server/bin/code-server
$ file "$f"; tail -3 "$f"
POSIX shell script, ASCII text executable
...
"$ROOT/node" ${INSPECT:-} "$ROOT/out/server-main.js" "$@"
```

这就是答案：**当前 "CLI + servers" 布局下，真正 exec node 的是一个普通的 `sh` 脚本**，
它一个堆参数都不传，所以宿主吃默认的 4 GB。

### 7.2 补丁内容

在 exec 行**之前**插入两行（一条就够，但两条互为兜底）：

```sh
export NODE_OPTIONS="${NODE_OPTIONS:-} --max-old-space-size=16384"
"$ROOT/node" ${INSPECT:-} --max-old-space-size=16384 "$ROOT/out/server-main.js" "$@"
```

| 通道 | 作用 | 依据 |
|---|---|---|
| **CLI 实参**（真正起作用的那条） | 宿主 fork 时 `execArgv` 继承 `process.execArgv` → 参数被转发进宿主 | §6.5 |
| `export NODE_OPTIONS` | 兜底：server 进程自身、以及**非宿主**的子 node 进程能收到 | 对宿主无效（`Hd()` 会剥），但保留以防未来改机制 |

> **注意 `--max-old-space-size` 是上限，不是预分配。** 设 16 GB 不等于吃掉 16 GB；
> RSS 用多少涨多少（实测：上限 16 GB 时宿主 RSS 4.7 GB = 上限的 29%）。

### 7.3 脚本化：`scripts/env/patch_vscode_server_heap.sh`

一次性手工改易丢（版本升级会覆盖），所以做成脚本。它的四条性质都很关键：

| 性质 | 实现 | 为什么需要 |
|---|---|---|
| **幂等** | 判据落在 **exec 行是否已含目标标记**，不是"export 是否存在" | 只有 export 而 exec 行没有，正是"看起来修好了但没到宿主"的半修复 |
| **可回退** | 同目录保留 `code-server.orig`；`--revert` 一键还原 | 机器状态改动必须可逆 |
| **版本升级安全** | 遍历**全部** `Stable-*`，不是只改当前那个 | 升级后自动覆盖，天花板会再次丢失 |
| **拒绝猜测** | 找不到 `"$ROOT/node"` 那行时**报错退出**，不改文件 | 布局变了就该人来判断，而不是脚本瞎猜 |

```bash
sh scripts/env/patch_vscode_server_heap.sh --check                  # 看磁盘上实际值
VSCODE_HEAP_MB=16384 sh scripts/env/patch_vscode_server_heap.sh     # 显式改
sh scripts/env/patch_vscode_server_heap.sh                          # 不加参数＝沿用已有值
sh scripts/env/patch_vscode_server_heap.sh --revert                 # 回退
```

### 7.4 复验：查 `cmdline`，**不要**查 `environ`

这是最容易踩的一个坑。正确的复验：

```bash
ps -eo cmd | grep "[t]ype=extensionHost" | grep -o -- "--max-old-space-size=[0-9]*"
# → --max-old-space-size=16384
```

**为什么不能像直觉那样查环境变量？**

```bash
# ❌ 错误判据：永远查不到，即使已经生效
tr '\0' '\n' < /proc/<exthost-pid>/environ | grep NODE_OPTIONS
```

因为 `Hd()` **恰恰就是把 `NODE_OPTIONS` 删掉的函数**（§6.5）。
用它做判据会稳定产出**假阴性**——明明修好了，却报告"没生效"。
正确判据是 **`cmdline`**：`execArgv` 转发的是实参，实参不会被剥。

> 实测：`/proc/<pid>/cmdline` = `server/node --dns-result-order=ipv4first
> --max-old-space-size=16384 .../bootstrap-fork --type=extensionHost` ✅

**生效条件**：必须**重启 server**（`Remote-SSH: Kill VS Code Server on Host`）。
窗口重载不够，关闭窗口也不够（§6.2 / §6.3）。

---

## 8. 一个我自己引入的静默降级（诚实记录）

修好之后，我验证"幂等性"，**结果自己制造了一次降级**：

```bash
$ sh scripts/env/patch_vscode_server_heap.sh          # 不加参数，想验证幂等
[patch] patched: .../Stable-7debcd0e.../code-server   # ← 它又"打"了一遍！
$ grep -o -- '--max-old-space-size=[0-9]*' <launcher>
--max-old-space-size=8192                             # ← 天花板从 16384 掉回 8192
```

**根因**：脚本的默认值是 `8192`，而幂等判据是
`exec_line 是否包含 "max-old-space-size=${HEAP_MB}"`。
磁盘上是 `16384` → **不匹配** → 判定为"未修补" → 用默认值重打了一遍。
脚本的默认值，把之前显式设的自定义值**覆盖**了。

**三个值得记住的点**：

1. **"幂等"的判据必须落在"目标状态"，而不是"这次想写什么"。**
   我要表达的是"这台机器应该有什么天花板"，而不是"我这次命令里写的是多少"。
2. **它只在下次 server 重启后才可见。** 当时运行中的宿主仍是 16384（进程已启动，
   不读启动脚本），磁盘却已是 8192 → **改动的效果被延迟到重启，静默期里没有任何提示。**
   这正是本项目硬规则里"不得静默降级"要防的那种 bug。
3. **修复方式：显式优先，否则从磁盘继承。**

```sh
if [ -n "${VSCODE_HEAP_MB:-}" ]; then
    HEAP_MB="$VSCODE_HEAP_MB";  HEAP_SOURCE="VSCODE_HEAP_MB (explicit)"
else
    # 从磁盘上已有的标记继承，避免"无参重跑"把自定义值降回默认
    HEAP_MB="<detected from exec line>"; HEAP_SOURCE="already installed on disk"
fi
```

**修复后复验**：显式设 16384 → 磁盘 16384；无参重跑 → `already patched ×3, patched 0`，
天花板保持 16384；`--check` 如实报告 `source: already installed on disk`。

> **教训**：**"验证幂等"这个动作本身会改变状态**——所以验证必须在**真正幂等**之后再做，
> 或者先用 `--check`（只读）而不是直接跑 patch。
> 而 `--check` 的汇总行也曾经打印"请求值"而不是"磁盘实际值"（在 16384 的机器上显示 8192），
> 同样修掉了：**报告当前状态时，永远报"实测值"，不报"参数值"。**

---

## 9. 为什么不能把堆上限设成 60 GB

既然有 62 GB 容器配额、503 GB 物理内存，为什么不直接给 32 GB？

### 9.1 三条硬理由

| # | 理由 | 说明 |
|---|---|---|
| 1 | **cgroup OOM-kill 比 SIGABRT 更糟** | 堆上限若接近容器配额，一个失控宿主会吃光配额，内核直接**杀掉整个容器**（server + 全部进程树）。SIGABRT 只死一个宿主、能自动重建；OOM-kill 连重连的机会都没有 |
| 2 | **堆越大 GC 停顿越长** | 全量 Mark-Compact 的代价随存活对象集增长。崩溃日志里单次 GC 已经 210 ms；堆再大一倍，停顿会更长，宿主"活着但卡死" |
| 3 | **它不修泄漏，只延长引信** | 见下 |

### 9.2 "买时间"到底买了多少（模型外推，非实测）

崩溃日志给了我们一个可靠的斜率：**堆增长 ≈ 16 MB/s**（230 秒涨到 3.7 GB）。
拿这个斜率外推（**这是模型计算，不是测量结果**）：

| 堆上限 | 按 16 MB/s 推算的存活时间 |
|---|---|
| 4.19 GB（默认） | ≈ 260 s（**实测 230 s，与模型吻合** ✅） |
| 8.19 GB | ≈ 510 s（≈ 8.5 min） |
| 16.19 GB | ≈ 1010 s（≈ 17 min） |
| 32.19 GB | ≈ 2010 s（≈ 34 min） |

**上限 ×4 ⇒ 崩溃间隔也 ×4。** 这条等式本身就是结论：

> **提高堆上限是纯粹的"时间银行"，它不改变斜率。**
> 如果泄漏仍在以 16 MB/s 增长，32 GB 也只是把 2 分钟一次的崩溃变成 34 分钟一次。

所以我把上限设在 **16 GB**（4× 默认）：足够覆盖正常工作负载
（实测稳态 RSS 4.7 GB，仅占上限 29%），又远低于 62 GB 配额，留足安全边际。
**真正的斜率控制见 §10。**

### 9.3 一个容易被忽略的副作用

`--max-old-space-size` 是作为 **server 的 node 实参**传入的，
而 `execArgv` 会被**所有继承它的子 node 进程**带走 —— 不只是扩展宿主。
好处是"一次注入，处处受益"；代价是**任何一个子进程失控，也能长到 16 GB**。
这正是 §9.1 第 1 条要求"上限必须远小于配额"的原因。

---

## 10. 根因之外的放大器：会话历史体积

修完堆上限之后，崩溃确实停了。但复盘时有一条我必须诚实记下的**我自己的贡献因素**：

### 10.1 现象与机制

两次崩溃与我的两个动作**时间上严格对应**：

| 崩溃时刻 | 我的动作 | 载荷 |
|---|---|---|
| 15:52:22 | 把 12 份 S14 协议文档**整份读进会话** | ≈ 220 KB |
| 15:57:47 | 单会话内**连续写入 8 个源码文件** | ≈ 210 KB |

**机制**：扩展宿主是一个**长生命周期**进程，而工具调用的内容（读到的文档、写入的文件全文）
都会成为宿主堆里**存活的 JS 字符串**。更要命的是同一条内容会被**多次物化**：
构造请求体 → `JSON.stringify` → HTTP body → 日志/遥测 → 会话状态。
**所以 1 KB 的载荷在堆里不是 1 KB。**

> **诚实边界**：这是**相关性 + 机制解释**，不是对照实验——
> 我没有能力在宿主里做 A/B（那本身就要加压）。
> 但机制上无争议：**长寿命进程 + 大字符串进堆 + 不主动释放 = 堆增长**，
> 而这两次崩溃的斜率（16 MB/s）也支持"有东西在快速灌入"。

### 10.2 对策：把"载荷纪律"变成可审计的硬约束

改的不是"我要小心点"，而是一组**量化阈值**写进文档（`docs/reports/S14_开发报告.md` §0.1）：

| # | 规则 | 阈值 |
|---|---|---|
| 1 | 不整份读文档 | 用 `sed -n 'a,bp'`；单次读 ≤ 200 行 |
| 2 | 单次写文件载荷 | **≤ 12 KB**（超出拆成多段） |
| 3 | 单会话累计写入 | **≤ 120 KB**，达到即收尾 |
| 4 | 命令输出 | 一律 `head -c` / `wc -l` / `grep -c` |
| 5 | 会话长度 | 工具调用 ≥ 40 次即主动收尾 |
| 6 | 恢复成本 | 新会话只读"进度表 + 摘要"（≈6 KB），**而非 220 KB** |

**第 6 条是杠杆最大的一条**：把恢复成本从 220 KB 压到 6 KB，
等于每次会话重启都省下一次崩溃级别的载荷。

### 10.3 这三层防护的关系（很重要，别搞混）

```text
① 消除泄漏源   —— 禁用无关 AI 扩展、工作区排除 28,077 → 4,019 文件
                   ↓  目标：降低增长斜率
② 抬高天花板   —— --max-old-space-size=16384
                   ↓  目标：即使斜率不变，也把存活时间 ×4
③ 压低载荷     —— 会话/命令/写入的量化阈值
                   ↓  目标：减少"我自己的"堆增长贡献
```

**三者不可互相替代**：
① 治本但可能漏（未定位到具体泄漏扩展）；
② 只买时间（§9.2）；③ 只管我自己那一份。**必须三条一起上。**

---

## 11. 可复用手册

下次遇到"编辑器 / AI 助手服务反复重启"，按这六步走，**不要跳步**。

### 第 0 步：快速分诊（30 秒）

| 观察 | 指向 |
|---|---|
| 恢复后提示 `command 'xxx' not found` | **宿主被重建**（扩展尚未注册）→ 进程生命周期问题 |
| `ps` 里 `extensionHost` 的 pid 变了、启动时间很新 | 同上 |
| 宿主存活时间很短（分钟级）且反复 | 有东西在快速灌堆 |

### 第 1 步：取证——分清"自杀"还是"他杀"

```bash
L=~/.vscode-server/data/logs/$(ls -1t ~/.vscode-server/data/logs/ | head -1)/remoteagent.log

grep -h "signal:"            "$L" | tail -5    # 谁终止的
grep -h "Reached heap limit" "$L" | tail -1    # V8 是否喊了 OOM
grep -h "mu = "              "$L" | tail -2    # GC 是否 thrash
```

**判据（这一步就已经能定性了）**：

| 看到 | 结论 | 方向 |
|---|---|---|
| `signal: SIGABRT` + `Reached heap limit` | **进程自杀：撞到 V8 堆上限** | §第 2 步 |
| `signal: SIGKILL` | 他杀：内核 OOM-killer | 查系统内存/cgroup |
| 无信号记录，正常退出 | 逻辑退出/崩溃循环 | 查扩展代码 |

### 第 2 步：确认"上限"这个变量（不要先怀疑系统内存）

```bash
# 默认堆上限是多少？（与物理内存无关）
node -e 'console.log((require("v8").getHeapStatistics().heap_size_limit/2**30).toFixed(2)+" GB")'
# → 4.19        ← 无论在多大的机器上

# 宿主当前有没有堆参数？
ps -eo cmd | grep "[t]ype=extensionHost" | grep -o -- "--max-old-space-size=[0-9]*"
# → 空 = 没有，正在吃 4 GB 默认值
```

### 第 3 步：注入（**node CLI 实参**，不是环境变量）

```bash
sh scripts/env/patch_vscode_server_heap.sh --check     # 先只读地看现状
VSCODE_HEAP_MB=16384 sh scripts/env/patch_vscode_server_heap.sh
```

### 第 4 步：重启 **server**（不是重载窗口）

```bash
# 客户端：Ctrl+Shift+P → "Remote-SSH: Kill VS Code Server on Host"
# 或：   ssh <host> 'pkill -f .vscode-server'
```

### 第 5 步：复验（**查 cmdline**）

```bash
ps -eo cmd | grep "[t]ype=extensionHost" | grep -o -- "--max-old-space-size=[0-9]*"
# → --max-old-space-size=16384
```

### 第 6 步：留痕（**最容易被跳过、但最不该跳过**）

把"时间 + 现象 + 动作 + 结果 + **原始日志摘录**"写进版本控制里的文档。
理由见附录 B——**不这么做的证据已经丢了**。

---

## 12. 五个反模式（这次全部踩过）

### 反模式 1：看到"服务重启"就往网络 / 系统内存方向查

**为什么错**：`SIGABRT` 是**进程自杀**，与系统内存无关。
本次实测系统极空闲（`oom_kill 0`、可用 682 GB、磁盘 52% 空闲、inotify 只用 8/128），
**7 个候选原因全部证伪**，浪费的每一分钟都源于没有先看**信号类型**。

### 反模式 2：以为"重载窗口 / 关闭窗口"能重启服务

**实测**：`Reload Window` 后 server 的 **pid 与启动时间纹丝不动**；
关掉窗口后 server **继续存活**（设计如此）。二者都**不会**让启动脚本被重新读取。

### 反模式 3：用 `environ` 验证 `NODE_OPTIONS`

**为什么错**：`Hd()` 正是**删掉 `NODE_OPTIONS`** 的那个函数。
用被删掉的变量做判据 → **稳定的假阴性**：明明修好了却报告"没生效"。
**正确判据是 `cmdline`。**

### 反模式 4：只调大堆上限就宣布修复

**为什么错**：上限 ×4 只是存活时间 ×4（§9.2）。
**斜率没变，问题没修。** 必须同时做降载（① 禁用无关扩展 ③ 压载荷）。

### 反模式 5：不带对照组就宣布"某机制失效"

**为什么错**：`grep env-setup` 返回 0，可能只是"我搜错了文件"。
必须带一个**已知能命中**的对照（本次用 `serve-web`：命中 4 次），
才能把"机制失效"和"检索方法错误"区分开。

> **加一条元反模式**：**"验证幂等"这个动作本身会改变状态**（§8）。
> 验证前先用只读的 `--check`；报告状态时永远报**实测值**，不报**参数值**。

---

## 13. 附录

### A. 速查命令表（一页纸）

```bash
# ── 分诊：宿主在哪、活了多久、多重 ────────────────────────────────
ps -eo pid,etime,rss,cmd | grep "[t]ype=extensionHost"

# ── 是自杀还是他杀 ───────────────────────────────────────────────
L=~/.vscode-server/data/logs/$(ls -1t ~/.vscode-server/data/logs/ | head -1)/remoteagent.log
grep -h "signal:"            "$L" | tail -5      # SIGABRT=自杀  SIGKILL=他杀
grep -h "Reached heap limit" "$L" | tail -1

# ── 堆上限状态（唯一正确的判据：cmdline）────────────────────────
ps -eo cmd | grep "[t]ype=extensionHost" | grep -o -- "--max-old-space-size=[0-9]*"

# ── 默认上限是多少（与物理内存无关）─────────────────────────────
node -e 'console.log((require("v8").getHeapStatistics().heap_size_limit/2**30).toFixed(2)+" GB")'

# ── 进程内存分解（区分堆 / 非堆 / 虚拟地址空间）─────────────────
p=$(ps -eo pid,cmd | grep "[t]ype=extensionHost" | awk '{print $1}' | head -1)
grep -E "^VmRSS|^VmSize|^VmSwap" /proc/$p/status
grep -E "^Rss|^Anonymous|^Shared_Clean" /proc/$p/smaps_rollup

# ── 容器真实配额（不是 free 看到的值）───────────────────────────
cat /sys/fs/cgroup/memory.max

# ── 修复 / 复验 ─────────────────────────────────────────────────
sh scripts/env/patch_vscode_server_heap.sh --check
VSCODE_HEAP_MB=16384 sh scripts/env/patch_vscode_server_heap.sh
# 然后重启 server，再回到上面第一条命令复验 cmdline
```

### B. 最反直觉的一条教训：**证据会被产生它的工具清理掉**

写这篇复盘时，我想回去引原始崩溃日志——**发现它们已经不在了**：

```bash
$ ls -1 ~/.vscode-server/data/logs/ | head -1
20260919T162322                    # ← 现存最早的日志目录

$ ls -1 ~/.vscode-server/data/logs/ | wc -l
10
```

**01:5x 与 15:5x 那三次崩溃的日志目录，在 server 重启时被清理了。**
（`2026-09-19 16:27` 那次重启之后，只剩 `1623xx` 以后的目录。）

于是 6 次 SIGABRT 的**唯一幸存原始证据**，是我当时**手工抄进 `AGENTS.md` §8.2 的那 16 行**。

**还有一个更隐蔽的陷阱**：我用递归 grep 去找它们时，得到了 48 条"命中"：

```bash
$ grep -rh "SIGABRT\|Reached heap limit" ~/.vscode-server/data/logs/ | wc -l
48
```

打开一看——**46 条是我自己那条 `grep` 命令**，被扩展的终端执行器记进了它自己的输出日志。
**取证命令会被日志记录，从而命中自己。**

两条可操作结论：

| 结论 | 做法 |
|---|---|
| **原始证据必须立刻转写进受版本控制的文档** | 崩溃后第一件事是把 `grep` 到的日志摘录**粘贴进受跟踪的文件**。不要指望它还在那儿 |
| **取证用 `grep -h` 指定文件，不要 `-r` 递归**；并且**永远打开命中内容看一眼** | 只看计数 = 把假阳性当证据 |

### B.1 一个二次发现：存放摘录的那个文件，本身也不受版本控制

写这篇复盘时顺手查了一下那 16 行摘录的"存活条件"，结果值得单独说：

```bash
$ git check-ignore -v AGENTS.md
.gitignore:94:AGENTS.md     AGENTS.md      # ← 被显式忽略
$ git ls-files | grep -i agents
                                          # ← 受跟踪文件里没有它
```

`AGENTS.md` 是**刻意 gitignore 的本机文件**（合理设计：它记录的是"这台机器"的角色与约束，
本就不该跟着仓库走）。但这也意味着：

| 载体 | 跨会话存活 | 跨机器存活 |
|---|---|---|
| `~/.vscode-server/data/logs/**` | ❌（已被 server 重启清理） | ❌ |
| `AGENTS.md`（gitignored） | ✅ | ❌ |
| **`docs/reports/*.md`（本文）** | ✅ | ✅ |

**所以本文的存在本身就是那次修复的最后一步**：
唯一幸存的原始证据原本只在一个**换台机器就消失**的文件里，
现在它随 `docs/` 进入版本控制。**"记下来了"和"留在仓库里"是两件不同的事。**

此外，如果你需要长期保留，可以在重启前**主动复制**日志目录：

```bash
cp -r ~/.vscode-server/data/logs/<session> /tmp/vscode-logs-preserved/
```

### C. 完整时间线

| 时刻（UTC+8） | 事件 | 结果 |
|---|---|---|
| 09-18 23:44 | server 启动（`pid 1312`） | 此后长期存活，**关窗/重载都不重启它** |
| 09-18 23:55 ~ 09-19 01:58 | **6 次 SIGABRT**，间隔 2–4 分钟 | 每次宿主被自动重建 |
| 01:58 后 | 加工作区排除（28,077 → 4,019 文件） | 崩溃间隔拉长 |
| 02:15 | 记录 §8.4 复验：隔离 ❌、堆上限 ❌、排除 ✅ | 三项里只有一项生效 |
| 02:43 | 客户端写 `affinity` + `allowed` 并重载 | 宿主 1→2（**后被推翻**）；两泄漏扩展进程消失 ✅ |
| 02:53 | 验证"关窗是否等于重启 server" | **不等于**，server 仍是 `pid 1312` |
| 02:54 | `Kill VS Code Server on Host` | server 真重启（`pid 17134`），**但 `NODE_OPTIONS` 仍无** |
| 02:55 | 装探针 + 带对照组的机制核查 | 判定 hook **从未被 source**（机制性失效） |
| 02:57 | 逐宿主核对扩展归属 | **推翻"隔离已生效"**，降级为"待复测" |
| 15:52 / 15:57 | **第 7、8 次 SIGABRT** | 与我"整份读 220 KB 文档""单会话写 210 KB"严格对应 |
| 16:0x | 读 `server-main.js`，找到 `Hd()` 与 `execArgv` | **定位真正注入点** |
| 16:10 | 补丁 + server 重启 | server 进程有 `NODE_OPTIONS`，**宿主仍无** |
| 16:1x | 确认 `Hd()` 剥离 + 改用 **CLI 实参** | ✅ 宿主 cmdline 出现 `--max-old-space-size` |
| 16:2x | 复验通过（4.19 → 8.19 GB），用户决定提到 16 GB | ✅ 16.19 GB |
| 本次复盘 | 修脚本 3 个 bug + 写本文 | 见 §8、附录 D |

### D. 本次复盘额外修掉的问题（都属"报告与状态不一致"）

| # | 问题 | 性质 |
|---|---|---|
| 1 | 脚本复验提示让用户查 `environ` 里的 `NODE_OPTIONS` | **假阴性**：修好了也会报告"没生效" |
| 2 | `--check` 汇总行打印"请求值 8192"而非"磁盘实际值 16384" | 报告与状态不符 |
| 3 | 无参重跑会把已装的自定义上限**静默降级**回默认值 | **静默降级**，且延迟到下次重启才可见 |

### E. 未复现项（如实登记）

| 项 | 记录值 | 本次实测 | 处理 |
|---|---|---|---|
| cgroup `memory.max` | 90 GB（`AGENTS.md` §8.2） | **62.0 GB** | 登记为"未能复现"；**不影响根因判定**（两者都远大于 4 GB 堆上限） |
| affinity 宿主隔离 | 一度观测到 2 个宿主 | 现为 1 个宿主 | 仍登记为**未稳定生效**，未声称"已隔离" |
| 具体是哪个扩展在涨堆 | — | 未定位 | **未归因**。需要 `Take Extension Host Heap Snapshot`，该操作本身显著加压，本轮不执行 |

### F. 交叉引用

- 项目内原始记录（含 16 行日志摘录）：`AGENTS.md` §8 / §9
- 加固脚本：`scripts/env/patch_vscode_server_heap.sh`
- 载荷纪律：`docs/reports/S14_开发报告.md` §0.1
- 证据分级口径：`docs/evidence_ledger.md`

---

> **一句话总结**：
> 崩溃不是"机器不够强"，而是**一个 4 GB 的编译期常量**；
> 修复不是"加内存"，而是**把参数送进正确进程的正确通道**；
> 而最持久的收益来自**承认自己也是负载的一部分**。
