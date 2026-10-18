import { useEffect, type ReactNode } from 'react';
import { Alert, Button, Space } from 'antd';
import { Link, useLocation } from 'react-router-dom';
import { PageHead } from '../components';
import '../features/help/styles.css';

const chapters = [
  ['start', '先完成一次对照'],
  ['machines', '选择执行机器'],
  ['models', '下载与接入模型'],
  ['architecture', '查看模型架构与计算依赖'],
  ['benchmarks', '固定输入与多方案评测'],
  ['playground', '交互推理与参数'],
  ['kernels', '替换算子实现与版本'],
  ['experiments', '高级优化与质量实验'],
  ['observations', '内存、KV、系统页与算子'],
  ['evidence', '查找记录与导出证据'],
  ['boundaries', '十层能力与替换边界'],
  ['troubleshooting', '常见问题与恢复'],
] as const;

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} className="manual-section" aria-labelledby={`${id}-title`}>
      <h2 id={`${id}-title`}>{title}</h2>
      {children}
      <a className="manual-top" href="#start">
        回到开始 ↑
      </a>
    </section>
  );
}

function ReferenceTable({ headings, rows }: { headings: string[]; rows: ReactNode[][] }) {
  return (
    <div
      className="manual-table-scroll"
      tabIndex={0}
      role="region"
      aria-label={headings.join(' / ')}
    >
      <table className="manual-table">
        <thead>
          <tr>
            {headings.map((heading) => (
              <th key={heading} scope="col">
                {heading}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {row.map((cell, column) => (
                <td key={column}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function UserGuide() {
  const { hash } = useLocation();
  useEffect(() => {
    // Also handles a contextual guide link after this lazy page has mounted.
    if (hash) document.getElementById(hash.slice(1))?.scrollIntoView({ block: 'start' });
  }, [hash]);
  return (
    <>
      <PageHead
        kicker="USER GUIDE"
        title="使用指南"
        description="按任务查入口：选择机器与模型，替换执行方案，比较输出，再追踪实际生效与内存机制。这里只说明现有功能，不会自动提交任务。"
        extra={
          <Link to="/benchmarks">
            <Button type="primary">开始固定输入对照</Button>
          </Link>
        }
      />
      <div className="manual-layout">
        <aside className="manual-sidebar">
          <nav aria-label="使用指南目录">
            <strong>操作目录</strong>
            {chapters.map(([id, title], index) => (
              <a href={`#${id}`} key={id}>
                {String(index + 1).padStart(2, '0')}
                <span>{title}</span>
              </a>
            ))}
          </nav>
          <p>操作前先看顶部机器名称。部署、任务和证据都属于选中的节点。</p>
          <Link to="/settings">查看当前会话与采集能力 →</Link>
        </aside>
        <article className="manual-article" aria-label="HQSB 操作手册">
          <Section id="start" title="01 / 先完成一次对照">
            <p>
              以“同一个模型，框架原生与候选算子方案有什么变化”为起点。用固定输入筛查，再用独立诊断请求解释差异。
            </p>
            <ol>
              <li>
                检查顶部机器选择器；到 <Link to="/machines">环境 → 机器连接</Link>{' '}
                测试连接，确认进入目标节点。
              </li>
              <li>
                打开 <Link to="/benchmarks">评测 → 性能与质量对照</Link>
                ，在“模型部署”选模型，基线保留框架原生，候选使用预选的可用方案。
              </li>
              <li>
                保留典型输入、64 token 预算、1
                次测量，点击“开始固定输入对照”。这是快速预览；需要更稳定的样本时选 3 或 5 次。
              </li>
              <li>
                等待测量完成与原部署恢复。逐个切换用例，看输出、任务正确率、token
                一致率、生成耗时和比较条件。
              </li>
              <li>
                要解释内存变化，下一轮勾选“另做内存 / KV /
                系统页诊断”；在结果的“独立机制诊断”里进入具体运行。
              </li>
              <li>
                从“生效凭据与原件”核对 effective 方案、算子调用、源码与制品身份，再保存原始对照记录
                JSON。
              </li>
            </ol>
            <Alert
              type="info"
              showIcon
              message="先核对可比性，再解释性能差异"
              description="输出预算相同不代表实际输出数量相同。输出 token 数变化、模型或分词器身份不同、执行未确认时，系统会保留结果并拒绝部分速度比例。1 次预览与驻留 worker 对照不能替代正式性能验收。"
            />
          </Section>

          <Section id="machines" title="02 / 选择执行机器">
            <p>
              入口：顶部机器下拉框，或 <Link to="/machines">环境 → 机器连接</Link>
              。目标端必须已经运行 HQSB Console；保存 SSH 连接不会自动安装项目或下载模型。
            </p>
            <ReferenceTable
              headings={['要做什么', '在哪里填写 / 操作', '实际作用与边界']}
              rows={[
                [
                  '切换已保存节点',
                  '先“测试连接”，再“切换到此机器”',
                  '核验服务后刷新工作台并进入设备页。只影响本标签页；原机器任务继续运行，记录和权重不搬运。切换前记下原任务 ID。',
                ],
                [
                  '新增 SSH 节点',
                  '新增连接 → SSH；填写唯一 ID、名称、主机或别名、用户、SSH 端口、HQSB 服务端口',
                  'SSH 端口是登录端口；服务端口是目标机器上的 HQSB 监听端口。通过隧道访问服务，支持服务端密钥或 Agent，不支持网页输入密码。',
                ],
                [
                  '指定 SSH 密钥',
                  'SSH 表单中的密钥文件路径（可选）',
                  '路径与 SSH 别名都属于控制服务所在机器；不是浏览器所在 Mac。不要粘贴私钥内容。',
                ],
                [
                  '新增网络节点',
                  '新增连接 → HTTP / HTTPS；填写服务源地址，如 http://host:8768',
                  '填 HQSB 服务根地址，不含 /api/console/v1。不是 /v1/chat/completions，也不是任意 OpenAI 兼容服务。TLS 证书必须有效。',
                ],
                [
                  '连接需要令牌的目标',
                  '令牌环境变量名（可选）',
                  '只填写变量名，实际值由控制服务进程的环境提供。当前本项目测试环境保持免登录。',
                ],
                [
                  '删除连接',
                  '连接列表 → 删除',
                  '移除连接配置，不删除目标服务器数据；当前节点与控制节点不能直接删除，已删除 ID 不再复用。',
                ],
              ]}
            />
            <p>
              当前部署保留 3090、Jetson 和 4090D 连接配置。4090D 关机时不能切换；Jetson 的 HTTP
              桥依赖已有 Mac
              隧道，隧道断开会影响可达性。机器列表的“最近测试”是本页测试结果，不是持续在线监控。
            </p>
          </Section>

          <Section id="models" title="03 / 下载与接入模型">
            <p>
              入口：<Link to="/models">环境 → 模型下载与接入</Link>
              ，设备页“下载与接入模型”或评测页“添加模型”。先确认顶部机器，所有下载、进度和部署登记都属于该节点。
            </p>
            <ol>
              <li>
                选择发布方，点击“加载发布方目录”。优先整批读取公开仓库，必要时分页补齐，并核对去重后的仓库总数；刷新可更新目录，不固定为几个预置模型。
              </li>
              <li>
                在已索引目录中搜索名称或参数规模。默认显示资源候选及缺少体积摘要的待检查纯文本模型；待检查不代表资源已通过。
                缺少架构时可取消筛选或输入完整仓库 ID
                检查。刷新期间保留已有目录；初次读取会逐页更新，目录不完整时不能当作已覆盖该发布方全部模型。
              </li>
              <li>
                点击“选择版本与检查”，从该仓库公开的分支与标签中选版本，再点击“检查所选版本与机器资源”。权重规模、KV
                与运行余量是预估，配置不兼容和资源不足会显示具体原因。
              </li>
              <li>
                阅读模型卡与许可证，检查通过后点击“下载到当前机器”。按实际传输字节显示进度、速率、当前文件与已校验文件数；100%
                字节完成后仍需 SHA-256 校验封存。
              </li>
              <li>
                取消、失败或服务重启后保留任务与部分文件；点击“继续下载”校验已有文件并尝试 Range
                续传。上游拒绝范围或哈希不同会明确失败，不将错误内容登记为模型。
              </li>
              <li>
                完成后点击“登记为推理部署”，再到推理页显式加载验证。默认框架原生；不会自动替换当前
                GPU 模型。登记后可在推理与固定输入评测选择该部署，服务重启后登记仍保留。
              </li>
            </ol>
            <p>
              当前来源策略登记 Qwen、Google、DeepSeek、Microsoft、智谱、InternLM、01ai；LLM-Research
              作为平台镜像单列，不冒充原厂组织。覆盖这些发布方的公开仓库与公开分支 / 标签，不代表
              ModelScope
              全站所有上传者。受限仓库仍需节点已有权限；不在网页接受授权协议或执行远程仓库 Python
              代码。
            </p>
            <p>
              当前仅接纯文本、非量化、原生 Transformers 架构的 FP16 单卡候选；按 batch 1、最多 2048
              上下文保守估算。GPU
              总量用于卸载当前模型后的容量估计，主机暂存与下载盘余量另行检查。加载预热和真实推理才是运行验证，下载成功不等于质量通过。每节点最多登记
              16 个部署，下载任务最多 8 个排队、单路传输，登记上限不删除已下载权重。
            </p>
            <p>
              下载目录由服务端 <code>model_download_dir</code> 配置；私有数据目录的{' '}
              <code>model-hub/</code> 保存任务、来源计划、manifest
              和动态部署登记。备份模型时需同时保存权重目录与这些身份文件。分支可能变化，下载按检查时发布方逐文件
              commit 与 SHA-256 锁定；不自动更新已封存快照。
            </p>
          </Section>

          <Section id="benchmarks" title="04 / 固定输入与多方案评测">
            <p>
              入口：<Link to="/benchmarks">评测 → 性能与质量对照</Link>
              。这是快速比较多种推理方案的主界面；每轮冻结模型、方案、输入版本、预算和重复次数，之后修改配置会创建新对照。
            </p>
            <ReferenceTable
              headings={['可以替换 / 调整什么', '具体位置', '如何使用']}
              rows={[
                [
                  '模型',
                  '新对照配置 → 模型部署',
                  '选择当前节点已登记的 PyTorch 模型。列表不等于机器上所有模型；不兼容的候选方案会被禁用。',
                ],
                [
                  '基线与候选',
                  '基线方案 / 候选方案（最多 3 个）',
                  '可选框架原生、适配的算子组合及保留 allocator 缓存方案。一次只改变一个因素更容易解释原因。目录可选表示可加载，不等于已证明有效。',
                ],
                [
                  '典型输入',
                  '固定输入表格左侧复选框；展开行查看提示词与输入 SHA-256',
                  '可选择算术、结构化提取、检索和解释生成等内置用例。此页不编辑用例正文；自定义提示词到推理工作台，质量样本到高级优化实验。',
                ],
                [
                  '输出长度与重复数',
                  '每次输出预算 / 每个用例测量次数',
                  '预算 8–128 token，受部署上限约束；测量 1 / 3 / 5 次。greedy 固定，每用例预热 1 次，预热不进入统计。',
                ],
                [
                  '机制诊断',
                  '另做内存 / KV / 系统页诊断',
                  '额外提交诊断请求；单独保存证据，不混入性能统计。需要更细的 kernel 事件时另做 operators 观测。',
                ],
                [
                  '下一轮配置',
                  '打开历史对照后 → 配置下一轮对照',
                  '修改草稿并新建一轮，不修改历史冻结计划。运行中不修改本轮配置。',
                ],
                [
                  '终止对照',
                  '执行状态 → 取消对照并恢复部署',
                  '请求停止后续任务并恢复原部署；以终态与恢复结果为准。恢复失败按页面提示到设备与部署检查。',
                ],
              ]}
            />
            <h3>结果按这个顺序看</h3>
            <ol>
              <li>
                <strong>完整度与生效：</strong>
                确认测量数完成，实际方案匹配，打开运行凭据看调用与缺失项。
              </li>
              <li>
                <strong>输出质量：</strong>选择用例和测量轮次，逐方案看正文与 finish reason。算术 /
                检索按参考答案、结构化提取按 JSON 语义评分；解释生成不自动判优。Token
                一致率只表示输出变化。
              </li>
              <li>
                <strong>性能：</strong>各指标取本用例测量中位数。E2E 相对值 = 候选 / 基线 −
                1；负值表示生成循环更短。若“比较条件”不满足，先看原因，不从短输出推导加速。
              </li>
              <li>
                <strong>机制：</strong>在独立诊断中比较相同阶段、相同单位的 allocator、KV
                和页指标，查看原始文件支持什么解释。
              </li>
            </ol>
            <p>
              模型加载、预热、诊断和恢复均不计入生成性能统计；运行记录比测量数多是正常的。页面刷新可重新打开已接受的对照，服务重启会将未完成对照保留为中断，不自动重提。
            </p>
          </Section>

          <Section id="architecture" title="模型架构与计算依赖">
            <p>
              入口：<Link to="/model-architecture">环境 → 模型架构</Link>
              ，或设备与部署中每个模型的“查看模型架构”。下载完成后先登记为部署即可查看，不需要加载到
              GPU。
            </p>
            <p>
              先看“通用模块与权重结构”：未加载时从实际权重名称建立层级；加载后展示真实 nn.Module
              类、参数与
              buffer。未知模型族也能查看这部分，不依赖模型族白名单。层级包含关系不等于计算依赖。
            </p>
            <p>
              加载模型后点击“采集运行架构”，系统用固定 Hello 输入和最多 2 个输出 Token
              提交诊断运行，记录前两次 forward 的实际 ATen 算子、所属模块、输出 shape / dtype 与
              Tensor 依赖。可筛选阶段、点选节点查看前驱后继、导出完整 JSON，并追溯运行凭据。
              页面读取或刷新不会自动推理。采集有额外开销，其 Token/s 不能作为正式性能比较。
            </p>
            <p>
              下方经过核对的模型族示意图可帮助理解：先看整体串行路径，再选具体层号。Decoder Block
              展示 Attention、MLP 与残差关系；切换 Attention 内部、MLP
              内部或点击节点查看模块路径、权重 shape、存储 dtype 和字节。Gemma 3 的局部 / 全局
              Attention 按层显示。
            </p>
            <p>
              实线表示数据依赖，虚线表示残差恒等路径，点线表示 KV 状态依赖。Q/K/V 和 Gate/Up
              分支在逻辑上可独立计算；当前 forward
              可能按顺序调用，融合后端也可能合并数学节点。没有实际 trace 不能断言 GPU 同时执行、SM
              分配或 Tensor Core 使用。
            </p>
            <p>
              已核对的静态示意图覆盖当前实现的 Llama、Qwen2、Qwen3、Gemma 3 纯文本模型；
              Llama、Qwen、Gemma、DeepSeek、GLM、Mistral / Mixtral、Phi、GPT-OSS、MiniMax、InternLM
              等模型族另有配置语义标注。缺少专用示意图不会阻止通用结构解析和执行采集。
              实际采集仅代表这次输入走过的路径，融合算子内部、未访问的专家和分支不能凭空展开；
              达到节点、边或字节上限会明确标为部分采集。导出快照不替代完整权重校验或运行验收。
            </p>
          </Section>

          <Section id="playground" title="05 / 交互推理与参数">
            <p>
              入口：<Link to="/playground">运行 → 推理工作台</Link>
              。适合自定义提示词、验证单次行为和采集详细证据。侧栏的修改只影响后续提交，不能改变旧运行。
            </p>
            <ReferenceTable
              headings={['内容', '位置与操作', '注意事项']}
              rows={[
                [
                  '模型与算子',
                  '左侧执行配置 → 模型部署 → 算子实现与版本 → 应用 / 加载按钮',
                  '先选部署和方案，加载成功、状态 ready 后才提交。选择下拉项本身不会替换驻留模型。',
                ],
                [
                  '清理策略',
                  '请求后清理策略 → 释放 / 保留 allocator 空闲缓存',
                  '影响请求结束后的 allocator 缓存，不等于卸载权重、不等于清空 GPU L1/L2。',
                ],
                [
                  '输入与输出预算',
                  '中间“输入”文本框；左侧“最大输出 token”',
                  '总窗口包含 System prompt、聊天模板、输入与输出预算；最大输出仅限制生成部分。窗口由部署配置决定，模型声明窗口另列；超限输入保留原值并提示，不自动截断。字符数不是 token 数。',
                ],
                [
                  'System prompt 与复现输入',
                  '左侧“高级设置”',
                  '可修改 System prompt，勾选保存输入供复现。默认仅保存输入摘要；输出和指标仍保存在运行记录。',
                ],
                [
                  '采样与量化',
                  '侧栏显示 Greedy / 量化方式',
                  '当前交互路径固定 greedy；W4/W8 尚无已验证在线部署，不是可切换的推理精度。外部引擎以其实际能力为准。',
                ],
                [
                  '观测模式',
                  '左侧“观测模式”',
                  '提交前选择，详见观测章节。不能给已经结束的请求补采内部证据。',
                ],
                [
                  '执行与取消',
                  '开始推理 / 停止执行',
                  '取消只清理本请求 KV 与临时资源，模型通常继续驻留。彻底释放模型到设备与部署卸载。',
                ],
              ]}
            />
            <p>
              运行结束后，在“本次运行凭据与观测”查看 applied 与
              effective；通过“查看完整运行”和“检查机制证据”进入同一 run
              ID。浏览器首内容、服务端首内容与 Runtime 首 token 使用不同计时起点，不能互相替代。
            </p>
          </Section>

          <Section id="kernels" title="06 / 替换算子实现与版本">
            <p>
              入口：<Link to="/playground">推理工作台</Link>{' '}
              左栏“算子实现与版本”。当前自定义配置主要适配 <strong>Qwen3 1.7B</strong>
              ；其他模型通常保留框架原生。下表说明现有实现类别，实际可选项取决于当前节点目录、模型形状与制品校验。
            </p>
            <ReferenceTable
              headings={['替换对象', '目录中的具体实现', '作用范围']}
              rows={[
                ['框架基线', <code>framework-native</code>, '全部保留当前模型的框架默认实现。'],
                [
                  'RMSNorm',
                  'CUDA scalar · v1 / CUDA half2 向量化 · v2',
                  '替换受支持的 RMSNorm 调用点；版本号不是性能排名。',
                ],
                [
                  '残差与激活',
                  'CUDA residual + RMSNorm · v1 / CUDA SiLU × up · v1',
                  '分别替换残差归一化融合或 MLP 激活乘法。',
                ],
                ['RoPE', 'CUDA split-half RoPE · v1', '替换受支持的位置编码计算。'],
                [
                  'Linear',
                  'cuBLASLt FP32 累加 · v1 / CUDA small-M GEMV + cuBLASLt · v1',
                  '改变受支持线性层的执行实现，不会把权重自动量化。',
                ],
                [
                  'Attention',
                  'PyTorch SDPA math · v1 / PyTorch SDPA flash · v1',
                  '选择已接入的 SDPA 后端路径；不等于接入独立 flash-attn 软件包。',
                ],
                [
                  '组合方案',
                  'Norm + residual + SiLU + RoPE · v1 / 局部融合 + cuBLASLt + SDPA math · v1',
                  '替换目录列明的算子，其他调用继续使用原实现。组合变化需要单项实验帮助归因。',
                ],
              ]}
            />
            <ol>
              <li>选择具体实现，看下方“算子 → variant”与原生制品身份；禁用项不能强行提交。</li>
              <li>
                点击“应用并重载模型”或“切换模型并加载方案”。系统排空并卸载当前 GPU
                模型，再加载和预热。
              </li>
              <li>确认“当前已加载方案”与 epoch 已更新，随后单独提交推理或固定输入评测。</li>
              <li>
                在运行的“生效凭据与原件”核对
                requested、resolved、effective，以及算子调用点、调用次数、not_exercised、源码 / 库
                SHA-256。只看到配置名不代表对应分支已执行。
              </li>
            </ol>
            <p>
              页面使用服务端登记的完整 profile，暂不提供任意拼接算子或上传 CUDA 代码。
              <Link to="/kernels">证据 → 算子与配置</Link>{' '}
              用于查看目录与源码，不是部署编辑器。新实现、新版本、模型适配与 native manifest
              需要开发端集成并在目标机器编译验证。
            </p>
          </Section>

          <Section id="experiments" title="07 / 高级优化与质量实验">
            <p>
              入口：<Link to="/optimization">优化 → 优化实验</Link>。用于预先声明基线 /
              候选、质量样本和机制预期。JSON 注册后冻结；要改输入或阈值，注册新计划。
            </p>
            <ReferenceTable
              headings={['计划字段', '可以调整什么', '如何理解']}
              rows={[
                [
                  <code>baseline / candidate.changes</code>,
                  '后端目录允许的措施，例如 cleanup_policy、kernel_profile',
                  'declared_changes 必须与实际变化相符；非集成措施会被拒绝。不要把十层目录当成已实现开关。',
                ],
                [
                  <code>performance_pairs / input_tokens / output_tokens</code>,
                  '性能配对数与负载长度',
                  '任务分别记录基线和候选；当前驻留 worker 串行执行，不满足独立进程的正式性能验收条件。',
                ],
                [
                  <code>quality_plan</code>,
                  'samples 的文本、期望答案与 slice，以及计分 / 生成 token 预算',
                  '目前为有界小样本：最多 4 个样本，计分长度 2–64 token。报告含 PPL、任务评分等，只说明注册样本的结果。',
                ],
                [
                  <code>observation_plan</code>,
                  '观测模式、sources、max_pages、max_decode_steps',
                  '约束采集窗口与预算；实际字段与限额由后端校验，不能用更大数字推导完整覆盖。',
                ],
                [
                  <code>mechanism_expectations / max_relative_e2e</code>,
                  '预先声明指标、阶段、方向、最小变化与性能门',
                  '先定义再运行，避免看完结果后改阈值。只用真实可用且口径一致的指标。',
                ],
              ]}
            />
            <ol>
              <li>从最小示例开始修改，点击“冻结并注册计划”；注册本身不执行推理。</li>
              <li>
                选正确的已就绪部署，保证 Attention 等加载配置匹配。要换算子
                profile，先在推理工作台重载。
              </li>
              <li>
                点击“依次执行未提交任务”，或单独运行槽位。默认示例是一对性能、一对机制诊断和一对质量，共六个任务。
              </li>
              <li>
                终态和清理完成后查看“质量与机制对照报告”，核对可比性、质量门、机制变化和封存原件。
              </li>
            </ol>
            <p>
              此页不会按每个实验臂自动重载不同 kernel / Attention
              配置；与当前加载配置不符时会拒绝执行。多算子方案的自动切换优先用固定输入评测；高级实验先从同一驻留配置下的清理策略对照开始。
            </p>
            <p>
              “停止后续提交并取消当前任务”会请求取消；离开页面只停止浏览器继续提交，已接受任务仍在后端运行。失败
              / 中断槽位保留记录，不自动覆盖重试；需要重新实验时注册新计划。默认算术 fixture
              只检查闭环，不代表通用模型质量。
            </p>
          </Section>

          <Section id="observations" title="08 / 内存、KV、系统页与算子">
            <p>
              先到推理工作台选观测模式并提交，再到 <Link to="/observations">观测 → 机制观测</Link>{' '}
              选该 run ID。详细诊断有采集开销，应与无插桩性能样本分开。
            </p>
            <ReferenceTable
              headings={['提交前选择', '主要得到什么', '怎样查看 / 限制']}
              rows={[
                [
                  'off · 关闭详录',
                  '请求状态、输出和基础请求指标',
                  '不生成详细内部轨迹；外部兼容 Provider 主要提供此范围。',
                ],
                [
                  'basic · 基础观测',
                  '阶段、逐 token 与内存快照',
                  'worker 主机时间不是 GPU kernel 时间。适合先了解请求结构。',
                ],
                [
                  'operators · 算子诊断',
                  '有限窗口 PyTorch Profiler、框架算子、可用时的 CUDA kernel / copy 与原始 trace',
                  '运行详情 → 全链路观测；检查 activities、status、采集窗口。缺少 CUDA 活动可能只有 CPU 数据。',
                ],
                [
                  'memory · 内存诊断',
                  '分配生命周期、allocator、KV 与相关 OS 证据',
                  '机制观测按阶段核对 allocated / reserved、快照、完整度与限制。',
                ],
                [
                  'pages · 页诊断',
                  'CPU 映射与有界页样本',
                  '看采样范围、权限与来源。CPU /proc 页证据不能代表 GPU 物理页表。',
                ],
              ]}
            />
            <h3>不同问题到不同页面</h3>
            <ul>
              <li>
                <Link to="/observations">机制观测：</Link>看请求内存、KV、allocator、CPU
                映射和页样本，分清直接测量、派生关联与不可用项。
              </li>
              <li>
                <Link to="/memory-flow">内存与数据流：</Link>选择历史请求，点选权重容量图、逐层
                KV、主机阶段与复制方向。权重按加载记录关联，KV
                是请求释放前快照；矩形按容量排布，不是物理地址图。
              </li>
              <li>
                <Link to="/runs">运行详情 → 全链路观测：</Link>看阶段、token、算子聚合、trace
                事件过滤与下载。框架算子和 GPU kernel 并非一一对应；GPU copy 字节不包含 kernel
                内全部权重 / KV 访问。
              </li>
              <li>
                <Link to="/research">深度分析：</Link>主要展示既有阶段实验与历史 NCU
                等原件。它们不自动成为新请求的硬件计数器结果。
              </li>
              <li>
                <Link to="/devices">设备与部署：</Link>
                看当前资源、模型加载状态与张量账本；当前遥测和历史请求快照的时刻不同，不能混作一条时间线。
              </li>
            </ul>
            <Alert
              type="info"
              showIcon
              message="三个常见误读"
              description="allocated 与 reserved 是包含关系，不能相加；CPU RSS、主机可用内存与设备显存口径不同；reserved 保留不等于 KV 仍存活。Cache 命中率、GPU 页与物理布局只有相应事件源和真实采集证据时才能讨论。"
            />
          </Section>

          <Section id="evidence" title="09 / 查找记录与导出证据">
            <ol>
              <li>
                从 <Link to="/runs">运行 → 运行记录</Link> 按任务筛选并打开详情，先核对节点、部署
                epoch、模型身份、终态和 cleanup。加载、卸载、推理与量化是不同类型任务。
              </li>
              <li>
                在“优化与质量凭据”等对应页签查看 requested / resolved /
                effective、质量口径和封存状态；在全链路观测中检查采集完整度。
              </li>
              <li>
                从原件表下载文件并保留 SHA-256；固定输入评测底部可打开“原始对照记录 JSON”，把 bench
                ID 与各 run ID 一并保存。
              </li>
              <li>
                到 <Link to="/compare">优化 → 运行对比</Link> 选基线 A 与候选
                B，先检查可比性门。跨模型或跨机器的原始耗时不直接形成同条件优化结论。
              </li>
              <li>
                <Link to="/evidence">证据中心</Link> 与 <Link to="/experiments">实验地图</Link>{' '}
                用于查历史报告和阶段原件；它们不是新任务的实时采集窗口。
              </li>
            </ol>
            <p>
              记录与原件保存在执行节点的私有数据目录，默认 <code>.console/</code>，包括
              SQLite、评测计划和采集 /
              实验制品。切换机器不会聚合数据库；查旧记录要切回原节点。备份由维护者处理整个数据目录及依赖原件，网页下载单个报告不等于备份全部模型和历史数据。
            </p>
          </Section>

          <Section id="boundaries" title="10 / 十层能力与替换边界">
            <p>
              目录按十层整理不代表十层都能在网页任意替换。实际可执行措施，以当前节点{' '}
              <Link to="/optimization">优化实验 → 十层优化能力目录</Link> 及加载校验为准。
            </p>
            <ReferenceTable
              headings={['层面', '现在可以在哪里操作', '还不能据此声称什么']}
              rows={[
                [
                  '1 · 硬件与系统',
                  '机器连接切换执行节点；设备页查看资源',
                  '没有网页 NUMA、频率、功耗上限或硬件分配调优开关。',
                ],
                [
                  '2 · 指令与 Kernel',
                  '推理工作台选择已接入算子 profile 与版本，重载并核对凭据',
                  '不支持任意模型、任意 ABI 或任意上传实现；具体调用必须实测。',
                ],
                [
                  '3 · 编译与运行时',
                  '查看已加载执行配置与运行证据',
                  '当前没有通用编译器、CUDA Graph 或自动调优的一键切换界面。',
                ],
                [
                  '4 · 数值精度与量化',
                  <>
                    <Link to="/precision">表示精度与量化</Link> 选择
                    FP32/FP16/BF16、RTN、GPTQ、AWQ、SmoothQuant、INT8/NF4/FP4，加载紧凑制品并独立评测。
                  </>,
                  '先应用方案并重载，再核对实际凭据。GPTQ/AWQ/SmoothQuant 需要独立校准数据；质量任务的 run ID 可生成对照。3090 不支持 FP8；历史 RTN 存储制品入口继续保留。',
                ],
                [
                  '5 · 模型结构',
                  '选择服务端已登记的不同模型部署',
                  '切模型同时改变结构与训练；没有在线剪枝、蒸馏、GQA / MoE 结构改写开关。',
                ],
                [
                  '6 · 内存与 KV',
                  '切换请求后 allocator 清理策略；在 L4 选择单请求 Quanto INT2/INT4 KV；查看 KV 与内存生命周期',
                  '量化 KV 仅支持完整 attention，暂不能与 L8 组合。没有通用 PagedAttention 或 KV 卸载 / 迁移 / 淘汰策略切换。allocator 缓存不是 prefix cache。',
                ],
                [
                  '7 · 解码算法',
                  '交互路径固定 greedy，可调整输出预算',
                  '没有推测解码或多 token 候选的已验证界面控制。',
                ],
                [
                  '8 · 单实例调度',
                  '查看排队、取消、生命周期和串行任务',
                  '不是 continuous batching 或 chunked prefill 的可切换实验台。',
                ],
                [
                  '9 · 分布式执行',
                  '可切换不同单节点 HQSB 服务',
                  '切机器不等于多机并行，不会自动接通 TP / PP / EP 或 PD 分离。',
                ],
                [
                  '10 · 服务与应用',
                  '切节点、管理部署和请求、保存和比较证据',
                  '没有自动扩缩容、缓存感知路由与完整 Agent 调用优化开关。',
                ],
              ]}
            />
            <h3>需要开发或服务配置的改动</h3>
            <p>
              登记发布方的兼容模型可以从“模型下载与接入”完成下载与动态登记；其他新模型或定制部署需要在目标端取得权重、校验
              manifest，并登记到该节点的服务 YAML；当前 3090 配置为{' '}
              <code>configs/console/3090-inf.yaml</code>
              。Provider、模型路径、上下文上限、native manifest
              与服务权限由服务端维护，修改后按部署流程重启验证；网页不直接编辑这些文件。新固定用例需要维护后端
              suite 及其版本与输入哈希。
            </p>
            <p>
              直接可读的证据包括框架内存统计、张量 shape / dtype / 去重 storage、CPU /proc
              映射，以及实际捕获到的 profiler 事件。逻辑 KV 字节、容量图和派生比值要看计算口径。完整
              GPU 物理页映射、逐地址缓存命中轨迹、每个 tensor 的 L1/L2
              命中率尚不能由当前通路直接恢复；历史 NCU 只覆盖其原始实验。
            </p>
          </Section>

          <Section id="troubleshooting" title="11 / 常见问题与恢复">
            <ReferenceTable
              headings={['看到的现象', '先检查什么', '恢复动作']}
              rows={[
                [
                  '没有想要的模型 / 记录为空',
                  '顶部机器是否正确；该节点是否登记模型或保存过此 run',
                  '切回原节点查记录；新增模型先走服务端接入。不要删数据库重建来掩盖缺失。',
                ],
                [
                  '选择算子后不能开始推理',
                  '是否提示“选择尚未应用”，模型是否 ready',
                  '点击应用并重载，等加载任务终态；看当前方案与 epoch，再单独提交。',
                ],
                [
                  '算子选项灰色 / 制品身份检查失败',
                  '是否支持该模型，native manifest、库和源码身份是否匹配',
                  '选择框架原生或由开发者在目标节点重新编译接入。不要关闭哈希校验强行加载。',
                ],
                [
                  'DEVICE_BUSY / DEPLOYMENT_NOT_READY / STALE_EPOCH',
                  '是否有其他驻留模型、加载中、量化中或刚刚重载',
                  '等待任务完成；必要时设备页排空卸载，刷新并重新选择部署。',
                ],
                [
                  '评测没有速度比例',
                  '比较条件、测量完整度、实际 token 数和执行凭据',
                  '先检查输入输出和原因；补齐独立同条件实验，不把空值填成 0。',
                ],
                [
                  '没有 trace / CUDA 耗时 / 页数据',
                  '提交前模式、采集窗口、activities、权限、status 与 limitations',
                  '旧请求不能补采；选对应模式提交新诊断，接受 partial / unavailable 的事实。',
                ],
                [
                  '停止后显存仍高',
                  '模型是否仍 ready，allocator 是否保留空闲缓存',
                  '要完全释放模型，设备页卸载；用请求快照与资源口径区分权重、KV 和缓存。',
                ],
                [
                  '输出在半句停止',
                  'finish_reason 是否为 length；是否发生超时',
                  '提高允许的预算后新建请求，或缩短提示词；固定输入对照要同步修改各方案预算。',
                ],
                [
                  '连接失败 / 页面中断',
                  '机器电源、SSH / API 端口、密钥、已知主机与隧道',
                  '机器连接页仍可返回控制节点；确认后端任务状态再操作，不重复点击提交。',
                ],
                [
                  '改了源码但网页没变',
                  '实际服务节点、项目目录与远端前端 dist 是否更新',
                  '显式同步并在目标端构建；API 改动需按流程重启。刷新页面加载新版本。Mac 不运行模型或构建。',
                ],
              ]}
            />
            <Space wrap className="section-gap">
              <Link to="/benchmarks">
                <Button>回到性能与质量对照</Button>
              </Link>
              <Link to="/playground">
                <Button>打开推理工作台</Button>
              </Link>
              <Link to="/settings">指标契约与采集能力</Link>
            </Space>
          </Section>
        </article>
      </div>
    </>
  );
}
