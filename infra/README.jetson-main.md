# Jetson 主服务与开发执行端（2026-10-21）

源码仍在 Mac 编辑，通过 `scripts/sync_to_jetson.sh` 上传。默认远端目录为 `/home/jetson/work/HQSB-main`，与历史 checkout 分开；`scripts/remote_run.sh` 默认进入此目录。

## 环境与持久状态

- 本机 NVIDIA PyTorch / CUDA 保留；`.venv-console` 使用 Jetson 自身的 system-site-packages，在隔离层安装控制服务依赖。
- 前端在 Jetson 执行 `npm ci`、`npm run build`，静态文件由同一后端提供。
- `build/l2/model-kernels.json` 在 Jetson 根据实际 CUDA 和 SM 重新构建。
- Qwen3-1.7B 使用 `/home/jetson/models/hqsb/Qwen3-1.7B` 的已有权重及原 SHA256 清单；新下载进入同一模型根目录的 `downloaded-models`。
- 主数据目录为新 checkout 的 `.console`。历史 Jetson 控制目录 `/home/jetson/work/HQSB-g01-g08-20261014/.console/connected` 保留原件，使用 SQLite backup API 生成一致数据库副本，复制配套采集文件。旧历史证据根目录保持只读索引，旧运行中的绝对路径继续指向保留原件。
- `3090-inf` 当前连接关闭；该端数据库、权重及独有原始报告尚待恢复连接后补迁。当前 Jetson 历史与3090历史分别标识来源。

## 服务管理

配置为 `configs/console/jetson-main.yaml`；普通用户 worker 保留量化、结构和解码等现有路径。Tegra CUPTI 的提权 worker 是单独的管理员选项，启用后部分普通 worker 功能有各自的能力要求；依靠当前能力接口显示实际可用范围。

以下命令从 Mac 根目录通过代理执行：

```bash
./scripts/remote_run.sh systemctl --user status hqsb-console
./scripts/remote_run.sh journalctl --user -u hqsb-console -n 80 --no-pager
./scripts/remote_run.sh systemctl --user restart hqsb-console
```

远端 `~/.config/systemd/user/hqsb-console.service` 来自 `infra/systemd/hqsb-console.service`。用户开启 linger，SSH 退出后服务继续运行，重启后由 systemd 拉起。监听 `127.0.0.1:8768`，保持测试免登录。编辑端使用 SSH 转发访问 `http://127.0.0.1:8878`，关闭转发只影响该客户端访问。

## 能力与迁移边界

Jetson 共享内存为 8GB 级别，当前默认 Qwen3-1.7B、FP16、上下文 2048、输出上限 512。较大的模型与高并发负载留给后续 GPU 节点。实际推理性能由当前 Jetson 测量，3090 的数值继续保留原硬件身份。

vLLM/SGLang 的3090虚拟环境、x86_64 Nsight 工具及 ERETrace 编译产物保留在原节点；Jetson 的平台和安装状态通过能力接口显示。新版代码、页面及接口保留；原生引擎与诊断链的迁入需要分别完成 ARM64/JetPack 安装及真实验证。

## 本次验收

远端环境为 Python 3.10.12、NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08`、CUDA 12.6、SM87、Transformers 5.8.0。隔离环境使用 NumPy 1.26.4；NVIDIA torch 和系统 Python 保持原安装。

| 检查 | 结果与范围 |
|---|---|
| 原有真实模型 HTTP 验收 | 17项通过：加载、生成、逐token计数、幂等、SSE回放、重复输出、取消、超时、清理、重载与epoch校验等 |
| 机制与L2补充验收 | 8项通过：正token/s、memory包封存、allocator事件、下载SHA256、OS页样本、不同观测模式token一致、原生RMSNorm加载、候选与基线greedy token一致；单个短输入，不据此作性能或广泛质量结论 |
| 前端 | 生产构建通过，50项单测通过；临时服务与部署服务各3项免登录浏览器检查通过 |
| 后端Console回归 | 505项中483通过、22失败；完整失败保存在原始pytest日志和XML，保留实际失败判定 |
| 历史迁移 | 207条运行、5333条事件、207条幂等记录；SQLite backup副本完整性检查通过，源目录原样保留 |
| 服务生命周期 | systemd用户服务enabled/active、linger开启；以配置保证退出SSH后常驻，未执行整机重启演练 |

22项回归失败来自目标环境兼容差异：15项遇到厂商SDPA缺少`enable_gqa`参数；5项涉及Dynamo配置键`fail_on_recompile_limit_hit`/`recompile_limit`；1项ONNX导出使用该torch版本缺少的`optimize`参数；1项CPU多进程并行测试遇到厂商torch缺少`torch.distributed.Store`。这些高级路径需要后续逐项适配Jetson，当前默认eager模型链路已完成真实验证。

证据位于远端与已回传本地的 `reports/console/jetson-migration/`，包含环境freeze、pytest XML/日志、构建日志、真实运行原件、迁移凭据和验收JSON。主服务启动后当前模型为unloaded，由用户在界面显式加载。

## 回退

历史服务目录及配置保持原样。若需要回退，先停止 `hqsb-console`，用历史启动命令和历史数据目录恢复 8768；新旧进程各自使用独立数据目录。源码默认目标可用 `HQSB_REMOTE_HOST`、`HQSB_REMOTE_DIR` 显式覆盖。SFTP 修改仅在本项目 `.vscode/sftp.json` 中生效。
