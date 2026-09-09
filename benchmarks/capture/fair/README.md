# 原生发压的 CPU 高频采集通路基准

此目录是实际运行的 benchmark，不是生产后端替代实现。2026-10-17 的原始数据、文献筛选和运行记录位于 [调查目录](../../../docs/research/hf_capture_2024_2026_20261017/README.md)。原始旧 Python 发压结果与这里的原生发压结果分开解释。

## 具体边界

共同输入是五字段事件：run_id、producer_id、source_seq、source_timestamp:uint64 和 UTF-8 JSON 文本 body。按源保持顺序、保留每条事件及字段；完成端点为关闭、排空及 fsync（D2）。不要求跨源总序；没有逐事件持久化 ACK 的库不填这项指标。

桥接接口是 C ABI：

```cpp
int hf_open(const char* encoded_output_path);
int hf_offer(const char* run, uint32_t producer, uint64_t sequence,
             uint64_t source_timestamp, const char* utf8_body);
int hf_close();
```

run/body 指针只在调用期间有效，异步实现需要复制。调用返回成功与持久化成功分开；仅 HQSB 的原 writer 暴露明确 admission 结果。其他 stock void API 记为 submitted/unknown。LTTng 的 D2 由外部 stop/destroy/fsync 完成。

| 文件 | 具体职责 |
|---|---|
| `native_source.cc` | 加载单个真实库；绝对时间开放到达；原生生产线程；保留原计划及未尝试债务；输出33字节 receipts。无 Python 发压循环。 |
| `*_adapter.cc` / `lttng_provider.h` | 调用作者库的真实入口，负责必须的复制/格式转换、关闭和 fsync；不替换其队列或算法。 |
| `python_bridge.cc` / `python_endpoint.py` | 嵌入 CPython 调用原 HQSB/OTel 实现，GIL、Python 对象与序列化计入 SUT。 |
| `run.py` | 隔离进程和 LTTng session；资源限制、PSS采样、计时；调用独立解码/Oracle；保留失败。 |
| `decode.py` / `read_ctf.py` | 在计时窗口外用官方 trace_processor、Babeltrace2 等读取真实产物。 |
| `campaign.py` | 冻结输入/二进制；两倍率源端预检；至多3组配置训练；冻结选择；独立留出测试。 |
| `versions.lock.json` | 精确版本和复现范围；不是“版本写在文件中即复现成功”的声明。 |

`../oracle.py` 验证唯一身份、字段字节、时间戳和源内顺序；重复/损坏不能计入 goodput。`../schedule.py` 生成确定性输入。body 模板的周期取大小混合、快照、名字基数与熵周期的 LCM，避免缓存模板改变后续真实输入。

## 此次服务器配置

固定用于 cpu-hqsb：逻辑 CPU192/194/195 放生产线程，193 放采集后台，196 放控制/观测；这是一采集核的目标部署配置，不是每个工具任意使用4核时的最高性能。总 AS 上限1GiB；它不是物理内存 cgroup 配额。统一栈2MiB和 `MALLOC_ARENA_MAX=2`，另采样整个进程树 PSS。迁移到其他机器必须重选允许的 CPU、核对拓扑，并重做源端预检。

吞吐 pass 不对每次 offer 计时；延迟 pass 确定性抽样1/16。每种 case 的两倍输入用空通路做3次预检，要求全部尝试且 >1ms 的派发迟到 ≤1%。实际 SUT 造成的背压/GIL竞争仍计入扰动，不能通过重排计划隐藏。

工具的配置与格式各不相同。评价同时看 V/N、V/A、未尝试数、V/(H+drain)、调用延迟、内存和原始编码字节；不把单个 offer 延迟当作整体性能。PMU/能耗/精确队列字节及逐事件落盘时间不可用，保持缺失。

## 在本次执行目录复测

Mac 仅编辑、传文件和版本管理。执行全部经过项目代理；下面命令中的输出目录需要使用新名字，不能覆盖既有证据。

```bash
HQSB_REMOTE_HOST=cpu-hqsb \
HQSB_REMOTE_DIR=/tmp/hqsb-capture-fair-20261017 \
./scripts/remote_run.sh bash -lc '
  cd source
  /tmp/hqsb-capture-bench-20261017/venv/bin/python \
    -m benchmarks.capture.fair.campaign \
    --work .. \
    --output /root/autodl-tmp/hqsb-capture-fair-20261017/results/campaign-new
'
```

现有构建位于工作目录 `build/`，包含 `capture-fair`、`quill0/1/2.so`、`perfetto.so`、`hindsight.so`、`lttng.so`、`dftracer.so`、`python_bridge.so` 和官方 `trace_processor_shell`。NanoLog 实际库及官方解码器在此前独立工作目录；版本见 lock。LTTng2.16 另装在 `lttng216/`，不覆盖系统2.14。

实验依赖具体源码/SDK/wheel，不能用仓库 main 的 SHA 冒充实际 SDK 或 wheel 的版本。主轨 protocol 冻结源码和二进制 hash；运行时依赖另有 manifest。构建和实验通过同一个 measurement.lock 隔离，计时中不能替换库或加入重负载。

此次还运行 Loom 的真实 Partitions 查询可见轨道以及 TraceZip 的真实 Go 编解码准入。这两项端点不属于主轨 D2，不放到同一速度排行榜。LMTracer 官方 GPU graph/doorbell 通路要求 CUDA 设备，CPU-only 环境不能给出其核心性能结论。

## 证据目录

每条 trial 保存 config、native/worker、receipts、实际编码文件、进程树样本、解码错误和 metrics；全局保存 protocol、resolved、训练分数、选定配置、source gates。数据以 run_id / producer_id / source_seq 关联。

部分已完成 raw 用 Zstd 无损归档：核对原 SHA、压缩后解压流核对同 SHA、写 manifest 后移除未压缩副本。这属于事后存储，不计入 SUT 的时间或编码比。恢复前核对 manifest，避免覆盖不同原件。损坏/失败/中断及被替代的旧试验仍保留。

1–2秒短窗口是压力筛选；另有10/30秒确认。当前不是最大持续容量搜索、长期 soak、GPU 设备回传或真实模型扰动验收。旧 SERVICE/OTAP 的顺序 Python 发送结果不属于本轮排除了源端瓶颈的主轨数据。
