# 开发执行端与后端机器连接

默认开发执行端为 SSH config 的 `3090-inf`，目录 `/root/HeteroQuantServeBench`。Mac 继续只编辑、传输文件与管理版本：

```bash
./scripts/sync_to_jetson.sh
./scripts/remote_run.sh .venv-console/bin/python -m pytest tests/unit/console -q
./scripts/remote_run.sh .venv-console/bin/python scripts/observation/build_model_kernels.py
./scripts/remote_run.sh scripts/console/serve.sh --config configs/console/3090-inf.yaml --port 8768
```

旧执行端继续通过 `HQSB_REMOTE_HOST` / `HQSB_REMOTE_DIR`，或同步脚本 `--target jetson|4090d` 选择。默认 SFTP 只更新本项目配置。不同执行端独立安装依赖与编译，不复制旧设备的虚拟环境和 CUDA 编译结果。

## 工作台操作

在「环境 → 机器连接」保存连接、测试连接，点击「切换到此机器」。页面顶部也可以选择已保存节点。切换先验证目标 HQSB API，再重新加载工作台；原节点的运行、模型和证据不迁移、不重提、不自动卸载。需要查看原记录时切回该节点。

- **SSH 隧道**：填写主机或控制服务机器上 SSH config 中的别名，可选用户名、SSH 端口和私钥路径；目标 HQSB 端口默认 8768。目标端必须已启动服务。控制端使用 OpenSSH、已有 known_hosts、私钥或 agent；不接收私钥内容或密码。Mac 的 SSH config 和密钥不会自动复制到控制服务器。别名解析保留主机、用户、端口、密钥、ProxyJump、host key alias、identity agent 与 known_hosts；不继承无关端口转发或 ProxyCommand。复杂跳板可在控制端预先建立转发后使用 HTTP 方式。
- **HTTP / HTTPS**：填写服务 origin，如 `https://node.example:8768`，不包含 `/api`、用户密码或查询参数。连接协议为 HQSB Console API v1。已有 OpenAI provider 配置仍在各节点自己的模型部署配置中使用。
- **认证**：测试服务继续免登录；若目标开启认证，连接配置只填写控制服务进程中令牌环境变量的名称。令牌内容不保存到连接 JSON，也不返回前端。HTTPS 默认验证证书，HTTP 客户端不继承代理环境、不自动跟随重定向。
- **失败与恢复**：连接失败不更改本标签页目标，也不自动回落到本机、不重试 POST。目标离线时管理页仍由控制服务提供，可切回本机。代理保留目标状态码、错误、SSE 游标、Idempotency-Key、二进制制品和下载响应头。

## 模块与接口

| 模块 | 职责 / 输入输出 | 语言 |
| --- | --- | --- |
| `machine_registry.py` | 校验配置；原子保存 `<data_dir>/machines.json` 和稳定节点 ID；不管理 GPU / SSH | Python |
| `machine_transport.py` | 配置 → 探测 / SSH 隧道 / HTTP 流；拥有连接资源，关闭时清理 | Python + 系统 OpenSSH |
| `machine_routes.py` | 列表、新增、探测、选择确认与移除；不控制模型或安装环境 | Python / FastAPI |
| 前端 Machines / MachineSwitch | 编辑连接、显示目标、验证后切换并丢弃旧文档缓存 | TypeScript / React |

管理接口始终位于控制节点的 `/api/console/v1/machines`。`GET` 返回列表；`POST` 新增，`POST /{id}/probe` 探测，`POST /{id}/activate` 验证选择，`POST /{id}/remove` 移除。启动 YAML 的 `machines` 仅用于第一次创建 registry，已有配置以持久化文件为准。

连接 ID 不可复用，包括已移除的 ID。更换目标地址需使用新 ID，避免旧标签页或已保存的 API 地址意外指向另一台机器。

选择属于**浏览器标签页**，存于 sessionStorage；所有远端数据接口、SSE 和下载都固定使用 `/api/console/v1/nodes/{connection_id}/...`，后端转换为目标同版本 API。没有全局可变的「当前机器」，其他标签页的选择不会改变旧界面的提交目的地。管理与本地 API 保持兼容，不依赖所选远端是否存活。代理不转发浏览器 Cookie 或控制端访问令牌，并拒绝嵌套节点转发。切换后不沿用旧运行 ID、epoch 或模型草稿。

模型状态、任务队列、数据库与证据继续由实际执行节点所有。当前不提供跨机器分布式调度、自动代码部署、依赖安装、模型传输、记录合并或跨节点性能判优。HTTP 需要网络可达；SSH 隧道不能让控制服务器访问 Mac 所在的局域网节点。

3090 配置为 `configs/console/3090-inf.yaml`，4090D 的八模型配置与原始运行保持在原机器。3090 现有非 HQSB 数据不属于本次清理范围；新权重继续从 ModelScope 下载并验证。

本次保留 Jetson 的本地入口 `http://127.0.0.1:8890`，并通过 Mac 到 3090 的反向转发提供控制端 `http://127.0.0.1:18878`，连接名称明确标记「经 Mac 隧道」。该地址通过上述两条隧道访问 Jetson；Mac 断线后会显示不可达。4090D 已由用户确认关机，保留 SSH 配置，待开机后再在目标机器授权专用公钥并验证。

## 编辑端隧道重建

主入口为 `http://127.0.0.1:8878/machines`。已有隧道时直接使用，无需重复建立。需要重建时，在 Mac 上执行以下纯 SSH 转发命令；不运行本地项目代码。3090 地址对应本次 SSH config，后续端口变更时同步更新参数。隔离 SSH 配置是为了不继承与工作台无关的 RemoteForward。

```bash
# 3090 主入口；目标端服务须已在 8768 启动。
# <ssh-host> / <ssh-port> / <ssh-user> / <identity-file> 由本机 SSH config 提供，不写入仓库。
ssh -F /dev/null -i <identity-file> -p <ssh-port> -f -N \
  -o BatchMode=yes -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes \
  -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8878:127.0.0.1:8768 <ssh-user>@<ssh-host>

# 可选：保留 Jetson 原服务，并让 3090 控制端通过 HTTP 访问它。
ssh -f -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8890:127.0.0.1:8768 <jetson-user>@<jetson-lan-ip>
ssh -F /dev/null -i <identity-file> -p <ssh-port> -f -N \
  -o BatchMode=yes -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes \
  -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:18878:127.0.0.1:8890 <ssh-user>@<ssh-host>
```

真实端点只存在于本机 SSH config 与运行期 `.console/machines.json`；请勿把实际地址、端口或私钥路径写回仓库文档。
