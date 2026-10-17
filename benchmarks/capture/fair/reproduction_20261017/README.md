# 2026-10-17 已执行补充实验的复现入口

这些是实际 CPU 实验使用的具体适配/实验代码，放在 benchmark 内维护。不是生产后端组件，也不新建依赖 repo。外部项目保持其上游 repo 和固定提交；版本见上级 `versions.lock.json`。

Mac 只编辑。执行必须通过 `HQSB_REMOTE_HOST=cpu-hqsb HQSB_REMOTE_DIR=/tmp/hqsb-capture-fair-20261017 ./scripts/remote_run.sh ...`。这些日期化脚本使用本次服务器的明确目录与 CPU192–196；迁机需修改路径/允许 CPU 并重做源端校准。不存在目录即报错，不偷偷改用本地运行。

| 代码 | 实际工作、输入和输出 |
|---|---|
| `run_lttng_v3.py` | 两版 UST 的合法3配置训练、54次留出、6次长窗口；验证独立consumer/helper限额，按观察tick补齐PSS；输入固定cases，输出配置/metrics/进程样本。 |
| `run_sustained.py` | 主实验已冻结profile，10/30秒确认和两倍源控制；`archive()`在测量及oracle后保留SHA等价Zstd副本。 |
| `run_sensitivity_v2.py` | 每个实现至多3配置×3种子，高负载全配置响应；独立输出，不能反向更改主实验选择。 |
| `run_admission_v3.py` / `ownership_probe.cc` | 256次有限ABI准入；调用返回立即覆写run/body缓冲；零值、uint64、UTF8和分隔符；输出stock解码与逐实现结果。 |
| `run_admission_lttng_low_pressure.py` | 16条相同类型的LTTng字段探针、最大已定义缓冲配置；区分有限突发丢失与字段本身不保真。 |
| `run_tracezip*.py` / `tracezip_admission_test.go` | 在作者receiver包内调用真实Go编解码与HTTP handler；匹配OTLP JSON、字典数据、gzip和factory defaults；任何字段不保真则不作速度排名。 |
| `loom-harness/` | Rust cdylib调真实Partitions、bounded ingress和sync；native源端同主轨；输出Q端点时间、反向扫描顺序校验和查询副本；不是D2。 |
| `restore_raw.py` | 校验archive SHA后恢复选中raw，核验原字节数与SHA；拒绝覆盖不同数据。 |
| `build_*.sh` / `runtime_manifest.py` | 独立前缀依赖构建与实际二进制/依赖来源记录；不能把后者当作测量成功标志。 |

运行布局为 `WORK/research/*.py`、`WORK/loom-harness/**`；依赖 `WORK/source`、`WORK/build` 和之前准备的 `/tmp/hqsb-capture-bench-20261017`。库构建的具体命令见上级 `build.sh`；它要求已经取得并核对的源码/SDK/wheel，不会自动覆盖未知版本。Go helper复制到 `upstream/tracezip/otel-compressor/prefix-compressed-receiver/` 后再构建。

所有测量、构建、压缩和重型分析共享 `/tmp/hqsb-capture-bench-20261017/measurement.lock`。多数脚本自行持锁，不要外套同一把锁造成死锁。源文件、二进制与输入在一轮运行中冻结；待运行的Python进程可能已经完成import，因此启动后不能编辑它读取过的脚本。

脚本以 `ROOT.mkdir(exist_ok=False)` 防止覆盖证据。重复整轮前须在**新脚本副本**选择新的结果目录，不可删除旧目录后重用其标识。修改后的代码/二进制必须重新生成protocol。主实验旧harness快照仍保存在服务器及本地私有证据目录；维护版解码器/会话控制的后续修复有独立smoke与单元验证，不能冒充旧实验原始源码。

完整文献筛选、逐试验表、失败与存储维护判定见 `docs/research/hf_capture_2024_2026_20261017/`。文献PDF和大raw按项目约定不混进生产源码。

`run_nanolog_response_v2.py` 另外使用作者 `runtime/Config.h` 公开容量配置，隔离构建4/16MiB staging，仅改单个常量，保留64MiB输出buffer与派生release threshold规则。与默认1MiB在相同CPU4/AS1GiB下比较百万事件、混合和2MiB记录；源码diff、构建命令/哈希、21次配置响应另存。运行用 `HQSB_NANOLOG_LIBRARY` 选择具体真实库，不替换任何已加载的旧库。调参预算允许配置常量，不能把默认构建的记录上限写成整个工具不支持大记录。

容量配置还必须包含**解码器配置**。本次默认1MiB解码器对一个2MiB记录试验返回0却输出0条；同源码4/16MiB官方解码器恢复120条（20预热+100输入）。`check_nanolog_decoders.py` 保留旧派生结果，以容量足够的官方接收端对既有raw重新校验，不重跑采集、不改计时。维护版 `run_nanolog_response_v2.py` 直接设置 `HQSB_NANOLOG_DECOMPRESSOR`，记录其真实hash；修正后的新入口还有21项单元测试及真实大记录接收端校验。
