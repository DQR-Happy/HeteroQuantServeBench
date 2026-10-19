import Button from '../components/ActionButton';
import { useEffect, type ReactNode } from 'react';
import { Alert, Space } from 'antd';
import { Link, useLocation } from 'react-router-dom';
import { PageHead } from '../components';
import '../features/help/styles.css';

const chapters = [
  ['start', '先完成一次对照'],
  ['machines', '选择执行机器'],
  ['models', '下载与接入模型'],
  ['architecture', '查看模型架构与计算依赖'],
  ['system', 'L1 硬件与系统配置'],
  ['model-structure', 'L5 模型结构与压缩'],
  ['memory-cache', 'L6 内存与 KV 管理'],
  ['decoding', 'L7 解码与生成算法'],
  ['parallel', 'L9 多卡与分布式执行'],
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
        description="按任务查找操作入口：选择机器与模型，替换执行方案，比较输出，再查看生效凭据与内存变化。"
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
              description="实际输出数量以生成结果为准。输出 token 数变化、模型或分词器身份不同、执行未确认时，系统会保留结果并拒绝部分速度比例。正式性能验收采用独立进程和多轮交替测量。"
            />
          </Section>

          <Section id="machines" title="02 / 选择执行机器">
            <p>
              入口：顶部机器下拉框，或 <Link to="/machines">环境 → 机器连接</Link>
              。先在目标端启动 HQSB
              Console，再保存连接并测试；模型权重在“模型下载”取得，在“本地模型管理”接入与清理。
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
                  '路径与 SSH 别名由控制服务所在机器解析，私钥请通过该机器的文件路径配置。',
                ],
                [
                  '新增网络节点',
                  '新增连接 → HTTP / HTTPS；填写服务源地址，如 http://host:8768',
                  '填写 HQSB Console API v1 的服务根地址，例如 https://node.example:8768，并使用有效的 TLS 证书。',
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
              隧道，隧道断开会影响可达性。机器列表的“最近测试”显示最近一次手动连接检查的结果和时间。
            </p>
          </Section>

          <Section id="models" title="03 / 下载与接入模型">
            <p>
              下载入口：<Link to="/models">环境 → 模型下载</Link>
              ，设备页“下载与接入模型”或评测页“添加模型”。先确认顶部机器，所有下载、进度和部署登记都属于该节点。
            </p>
            <ol>
              <li>
                选择发布方，点击“加载发布方目录”。优先整批读取公开仓库，必要时分页补齐，并核对去重后的仓库总数；刷新可更新目录，不固定为几个预置模型。
              </li>
              <li>
                在已索引目录中搜索名称或参数规模。默认显示资源候选及缺少体积摘要的待检查纯文本模型；待检查模型将在版本检查时核对资源需求。
                缺少架构时可取消筛选或输入完整仓库 ID
                检查。刷新期间保留已有目录；初次读取会逐页更新，目录完整度以刷新进度和仓库总数核对结果为准。
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
                完成后打开<Link to="/models/local">本地模型管理</Link>
                ，点击“登记为推理部署”，再到推理页显式加载验证。默认框架原生；不会自动替换当前 GPU
                模型。登记后可在推理与固定输入评测选择该部署，服务重启后登记仍保留。
              </li>
              <li>
                在<Link to="/models/local">本地模型管理 → 本地模型清单</Link>
                中点击“删除本地制品”可释放权重空间；取消或失败的下载可点击“清理下载文件”。
                先在设备与部署页卸载关联模型，等待节点任务结束，再核对确认框中的模型、版本和目录。
                删除会移除关联的下载部署，历史推理、评测与观测记录继续保留；再次使用时重新选择版本下载。
                删除失败可查看具体原因并重试。
              </li>
            </ol>
            <p>
              本地模型页集中列出下载制品、节点配置的模型及 L4/L5 制品。文件大小和磁盘占用分别统计，
              汇总按设备与 inode 去重。可按占用排序、搜索目录、查看部署引用，并手动刷新空间统计。
              其他目录可在“扫描范围与其他模型目录”中填写当前节点的绝对路径后添加。 节点 YAML
              引用的目录需先整理配置；页面会提供具体原因和对应部署链接。
            </p>
            <p>
              当前来源策略登记 Qwen、Google、DeepSeek、Microsoft、智谱、InternLM、01ai；LLM-Research
              作为平台镜像单列。目录收录这些发布方的公开仓库与公开分支 / 标签。
              受限仓库使用节点已有的访问权限，模型按原生 Transformers 实现加载。
            </p>
            <p>
              当前支持纯文本、非量化、原生 Transformers 架构的 FP16 单卡候选；按 batch 1、最多 2048
              上下文保守估算。GPU
              总量用于卸载当前模型后的容量估计，主机暂存与下载盘余量另行检查。下载完成后可依次进行加载预热、真实推理和质量评测。每节点最多登记
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
                  '列表展示当前节点已登记的 PyTorch 模型；候选方案按部署兼容性开放。',
                ],
                [
                  '基线与候选',
                  '基线方案 / 候选方案（最多 3 个）',
                  '可选框架原生、适配的算子组合及保留 allocator 缓存方案。一次只改变一个因素更容易解释原因。选择后加载模型，再通过生效凭据和评测结果比较效果。',
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

          <Section id="model-structure" title="L5 / 模型结构与压缩">
            <p>
              进入<Link to="/model-structure/build">模型结构与压缩</Link>
              ，先加载原始浮点模型，检查真实模块，再选择变换目标。幅值或 Wanda
              评分剪枝把权重置零，仍用稠密 Linear；MLP 通道裁剪成组缩小 Gate/Up/Down；SVD
              将一个权重换成两次紧凑 Linear，不在推理中重建完整矩阵。
            </p>
            <p>
              点击创建后，在独立 CPU 子进程上生成新权重，不修改源权重或驻留模型。完成后去
              <Link to="/model-structure/artifacts">制品与加载</Link>
              显式应用，或恢复原始浮点模型。制品在所选后端的数据目录 l5-artifacts 下，格式为 HQSB
              配方加 safetensors；不是 Mac 上另存的文件，也不是任意 Transformers
              加载器都支持的检查点。
            </p>
            <p>
              在<Link to="/model-structure/evaluation">评测与对照</Link>对原模型和制品分别运行
              PPL、固定 16 Token 生成，选择相同类型的基线与候选，查看 PPL/NLL
              的增加及性能、参数变化。压缩通常有损，参数少不保证更快；Wanda
              的一次校准变体也不保证优于幅值剪枝。L5 制品不会被当作原始浮点 baseline。架构页会区分源
              config 与实际被覆写的模块形状。
            </p>
          </Section>
          <Section id="memory-cache" title="L6 / 内存与 KV 管理">
            <p>
              进入<Link to="/memory-cache/configure">优化 → 内存与 KV 管理</Link>
              ，选择已加载模型，点击“检查缓存能力”。动态 KV 按需增长，静态 KV 预分配到输入＋输出−1
              的容量；卸载模式把逐层 KV 搬到 CPU 并预取回 GPU；重计算关闭
              KV，每步重新执行完整历史。已有 L4 KV 量化请用“模型默认 KV”，暂不和其他缓存布局组合。
            </p>
            <p>
              选择典型原始文本、固定输出 Token 后执行同一个输入。在
              <Link to="/memory-cache/records">运行对照</Link>选择两条完成记录，检查生成
              Token、logprob、Token/s 和 Worker 总时间差。首 Token
              与生成循环时间不含前缀还原/快照及清理；Worker
              总时间包含这些成本。前缀未命中与命中样本要分开比较。
            </p>
            <p>
              动态且全部 full attention 的模型可启用有界前缀复用。第一次请求成功后保存 GPU
              前缀，第二次相同前缀命中并复制到私有
              KV；切换模型会清空。想测冷缓存，先点“清空驻留前缀”，再重新检查能力。Gemma3
              混合滑窗目前不支持该前缀适配器。
            </p>
            <p>
              <Link to="/scheduling">多请求调度实验</Link>可选原有稠密独立行或 GPU
              块池，调整块大小、块池预算，观察新增 KV
              写回、批次拼装、尾部浪费和回收。块池仍还原为稠密 Attention；它不是 PagedAttention
              Kernel，也不是操作系统物理页。CPU 缺页是 worker 进程统计，GPU L2
              命中率和逐页迁移暂未接入。
            </p>
          </Section>
          <Section id="decoding" title="L7 / 解码与生成算法">
            <p>
              进入<Link to="/decoding/run">解码策略与运行</Link>
              ，选择已加载的原始浮点目标模型，使用固定重复数字、英文续写或有限候选分类样例。可切换逐
              Token
              贪心/随机采样、提示词查找推测和独立草稿模型推测。普通推理工作台继续保留原有行为。
            </p>
            <p>
              随机采样可调整温度、Top-k、Top-p、重复惩罚、最少长度与请求独立
              Seed。有限候选每行填写一个完整回答，取消固定输出长度，为最长候选与 EOS
              预留预算；可比较全词表 Mask 与候选 Gather。它不是通用 JSON Schema 约束。
            </p>
            <p>
              独立草稿必须先在模型页下载并登记，且 Tokenizer 编码与 ID
              映射完全一致。第一次明确运行会加载草稿，后续请求复用驻留权重；到
              <Link to="/decoding/draft">草稿模型与资源</Link>
              查看身份、参数字节和卸载。目标卸载也会释放草稿，两者请求 KV
              每次独立清理。相同检查点副本可验证算法，但不代表获得小草稿模型的成本优势。
            </p>
            <p>
              在<Link to="/decoding/compare">结果对照</Link>
              选择相同输入、预算和观测配置的基线/候选。查看目标与草稿 Forward
              数、验证批次、接受率、KV 回滚、Token/s、首 Token 和含加载/清理的请求总时间。逐 Token
              一致仅支持当前贪心样例的结论；采样需具体任务的多 Seed 评测，Teacher-forcing PPL
              不测解码策略。批内多个 Token 同时提交可出现零到达间隔，不能解释为零计算时间。
            </p>
            <p>
              本阶段不开放随机推测拒绝采样、自推测早退、训练多 Token
              头、树形验证或通用语法编译；与量化、结构压缩、编译、特殊 KV
              和连续合批的组合需另行验收。按钮禁用时悬停可查看前置操作与链接。
            </p>
          </Section>
          <Section id="system" title="L1 / 硬件与系统配置">
            <p>
              进入<Link to="/system/run">配置实验</Link>
              ，选择当前节点已下载并校验的原始浮点模型。先在<Link to="/devices">设备与部署</Link>
              卸载驻留模型。设置两组 CPU 亲和性、PyTorch 算子内/间线程数、NUMA 后续分配策略和 GPU
              功耗上限，每行一个固定原始文本输入。
            </p>
            <p>
              CPU 留空表示继承允许的全部 CPU；线程数不改变容器的 CPU 配额。NUMA 绑定仅在操作系统允许
              mempolicy 时提供，在模型加载前应用，不能据此声称迁移了 GPU 显存页。GPU
              功耗留空表示继承实验前的原值；驱动写权限在应用时检查，不能仅由 root 身份推断。
              <Link to="/system/topology">硬件与能力</Link>列出真实允许范围、UUID、NUMA 和容器限制。
            </p>
            <p>
              默认每配置两轮，按 A/B、B/A 顺序使用独立进程加载和预热；生成计时排除加载预热。在
              <Link to="/system/results">结果与观测</Link>查看 Token/s、首
              Token、分配器峰值、RSS、缺页、上下文切换、板卡能量和逐 Token
              输出，并查看候选相对基线的变化。250 ms
              遥测是设备级样本，短任务的样本可能较少。板卡能量计数器与功率积分估算分开保存。
            </p>
            <p>
              结束或取消时先退出模型进程，再恢复原功耗并回读。恢复未确认会保留
              .console/l1-power-journal.json 并阻止后续 L1 实验；请在执行节点检查原
              UUID、原功耗和守护进程，确认恢复后归档日志并重启服务。CPU governor、GPU/EMC
              锁频、Jetson nvpmodel、NPU
              执行和异构图切分尚未开放，不能把硬件探测当作这些措施已实施。
            </p>
          </Section>
          <Section id="parallel" title="L9 / 多卡与分布式执行">
            <p>
              进入<Link to="/parallel/run">执行方案</Link>，选择已下载并校验的原始浮点源模型。先在
              <Link to="/devices">设备与部署</Link>卸载驻留模型；L9
              的私有进程负责加载、执行和释放权重，结束后不自动改变单请求部署。
            </p>
            <p>
              单设备基线用于对照；DP 将不同请求分给独立完整副本；TP
              使用框架原生权重分片和通信；按层放置顺序经过多张 GPU，不等于多微批流水；Prefill/Decode
              分离将真实 KV 经 CPU 暂存和 Gloo 传给 Decode
              rank。原始文本每行一条，固定输出预算便于比较；CPU
              多进程诊断只验证逻辑，不能代替多卡性能。
            </p>
            <p>
              <Link to="/parallel/topology">设备与通信</Link>读取当前节点可见
              GPU、库版本与后端能力。只有一张 GPU
              时无法运行多卡策略，按钮会解释原因并链接到节点选择。
              <Link to="/parallel/peers">跨节点 DP</Link>支持已登记 HTTP / SSH 节点，每节点提供新版
              L9 API、相同权重及 Tokenizer，并卸载各自驻留模型。双节点 DP 只传请求和结果，不传模型或
              KV。
            </p>
            <p>
              运行结果保存每 rank 参数字节、全局/局部分片形状、设备、通信 CPU 事件、allocator 与请求
              KV；P/D 另存传输字节与阻塞阶段时间。CPU 事件不是完成的 GPU 通信耗时，P/D
              的接收等待也不是入口排队时间。在<Link to="/parallel/compare">结果对照</Link>
              选基线与候选，核对实际输入和逐 Token 输出；单次结果不作为正式加速结论。PP
              流水重叠、CP/SP、MoE EP、混合并行及跨主机 KV/TP 尚未接通。
            </p>
          </Section>
          <Section id="architecture" title="模型架构与计算依赖">
            <p>
              模型架构按任务分为“模块与权重、静态计算结构、运行算子观测、编译与导出、来源与凭据”五个子页。
              顶部导航保留所选模型和历史运行；编译与导出页选择 FX、ONNX 或 Inductor IR/代码。
              每个子页都可直接分享链接。灰色业务按钮可悬停、点击或用 Tab 聚焦查看当前不可用原因，
              提示中提供加载模型、配置精度或查看任务的入口；提示不会自行执行前置操作。
            </p>
            <p>
              各子页分别展示静态语义结构、实际 ATen；“编译与导出”的“计算图类型”可切换 Dynamo
              FX、ONNX、Inductor 融合前 IR、融合后 IR
              和生成代码。切换只读取记录；静态结构不需要加载模型，其他图需要对应的采集记录。 ONNX
              页点击“采集 ONNX 导出诊断样例”，导出固定输入、无 KV 的 prefill 结构，权重作为图输入，
              不另存一份权重，不是可独立部署的 ONNX 模型。Inductor
              任一视图点击采集按钮，一次保存融合前后 IR
              与代码，之后可选择同一运行切换查看和下载。历史记录未采集的图会明确提示缺失。
              新增诊断先支持原生浮点权重、默认 KV，含采集开销；它们的耗时不能作为正式性能对照。
            </p>
            <p>
              “计算图与 L3 执行”可按请求选择 Eager、Dynamo 图捕获或 Inductor 编译，调整自动 / 静态 /
              动态形状与 Fullgraph
              策略。先运行固定样例作为基线，再切换方案运行；同一方案重复运行可查看新图数和图调用。
              独立 PPL
              按钮会实际执行所选策略。编译图节点通过模块路径关联到原架构与权重，历史图随运行封存。
              首批编译只支持原生浮点权重和默认 KV；首次成本包含编译，FX
              节点数量不能当作物理融合或加速证明。
            </p>
            <p>
              入口：<Link to="/model-architecture">环境 → 模型架构</Link>
              ，或设备与部署中每个模型的“查看模型架构”。下载完成后先登记为部署即可查看，不需要加载到
              GPU。
            </p>
            <p>
              先看“通用模块与权重结构”：未加载时从实际权重名称建立层级；加载后展示真实 nn.Module
              类、参数与
              buffer。各模型族均使用同一条通用解析通路。树线表示模块包含关系，计算依赖见下方执行图。
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
              达到节点、边或字节上限时标记为部分采集；完整权重校验与运行结果见对应凭据。
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
                  '控制请求结束后的 allocator 空闲缓存回收，模型权重保持驻留。',
                ],
                [
                  '输入与输出预算',
                  '中间“输入”文本框；左侧“最大输出 token”',
                  '总窗口包含 System prompt、聊天模板、输入与输出预算；最大输出仅限制生成部分。窗口由部署配置决定，模型声明窗口另列；超限输入保留原值并提示，不自动截断。Token 数由当前分词器计算。',
                ],
                [
                  'System prompt 与复现输入',
                  '左侧“高级设置”',
                  '可修改 System prompt，勾选保存输入供复现。默认仅保存输入摘要；输出和指标仍保存在运行记录。',
                ],
                [
                  '采样与量化',
                  '侧栏显示 Greedy / 量化方式',
                  '当前交互路径采用 greedy 解码；权重量化和 KV 精度请在表示精度页配置，外部引擎选项由其能力决定。',
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
              ID。浏览器首内容、服务端首内容与 Runtime 首 Token 分别标注计时起点。
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
                  '替换受支持的 RMSNorm 调用点；版本号用于区分实现，性能请通过相同负载的评测比较。',
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
                  '选择 PyTorch SDPA 提供的 Attention 后端路径。',
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
                SHA-256。通过实际调用记录核对所选配置的执行情况。
              </li>
            </ol>
            <p>
              页面使用服务端登记的完整 profile，暂不提供任意拼接算子或上传 CUDA 代码。
              <Link to="/kernels">证据 → 算子与配置</Link>{' '}
              用于查看目录与源码，部署操作在推理工作台完成。新实现、新版本、模型适配与 native
              manifest 需要开发端集成并在目标机器编译验证。
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
              用于检查评测闭环；应用质量请使用对应任务的数据集评测。
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
                  '展示 worker 的主机阶段耗时，适合了解请求结构；GPU kernel 耗时见设备时间线。',
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
                  '看采样范围、权限与来源。CPU 页状态取自 /proc，GPU 页指标由对应设备采集器提供。',
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
                是请求释放前快照；矩形按各分量容量排布。
              </li>
              <li>
                <Link to="/runs">运行详情 → 全链路观测：</Link>看阶段、token、算子聚合、trace
                事件过滤与下载。框架算子和 GPU kernel 按工具关联 ID 建立多对多映射；GPU copy
                字节不包含 kernel 内全部权重 / KV 访问。
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
              message="指标口径"
              description="reserved 包含 allocated；CPU RSS、主机可用内存与设备显存分别统计。KV 生命周期见张量账本，Cache 命中率、GPU 页与物理布局见对应采集器的数据。"
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
                B，先检查可比性门。跨模型或跨机器对照时，先核对硬件、负载和模型身份。
              </li>
              <li>
                <Link to="/evidence">证据中心</Link> 与 <Link to="/experiments">实验地图</Link>{' '}
                用于查询历史报告与阶段原件；本次任务的采集数据位于运行详情。
              </li>
            </ol>
            <p>
              记录与原件保存在执行节点的私有数据目录，默认 <code>.console/</code>，包括
              SQLite、评测计划和采集 /
              实验制品。切换机器不会聚合数据库；查旧记录要切回原节点。备份由维护者处理整个数据目录及依赖原件，完整备份需包含模型、数据目录和历史运行引用的原件。
            </p>
          </Section>

          <Section id="boundaries" title="10 / 十层能力与替换边界">
            <p>
              目录按十层整理，各层可执行措施见当前节点{' '}
              <Link to="/optimization">优化实验 → 十层优化能力目录</Link>，加载时会核对兼容性。
            </p>
            <ReferenceTable
              headings={['层面', '操作位置', '使用说明']}
              rows={[
                [
                  '1 · 硬件与系统',
                  <>
                    <Link to="/system">硬件与系统配置</Link>执行固定输入的 CPU 亲和性、线程、NUMA 和
                    GPU 功耗对照；机器页切换节点。
                  </>,
                  '权限与硬件能力逐节点核验；NUMA 只影响随后分配；功耗必须恢复回读。频率仅观测，尚无锁频、NPU 执行或异构图切分。',
                ],
                [
                  '2 · 指令与 Kernel',
                  '推理工作台选择已接入算子 profile 与版本，重载并核对凭据',
                  '不支持任意模型、任意 ABI 或任意上传实现；具体调用必须实测。',
                ],
                [
                  '3 · 编译与运行时',
                  '在模型架构的编译与导出子页选择 Eager、Dynamo 或 Inductor，查看多种计算图与实际编译凭据',
                  '首轮包含编译成本；CUDA Graph 尚未开放，和 L6 自定义 KV、量化组合需要独立验收。',
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
                  <>
                    <Link to="/model-structure">模型结构与压缩</Link>创建幅值/Wanda 评分剪枝、MLP
                    通道缩维和 SVD 低秩制品，加载后对照原模型质量和性能。
                  </>,
                  '没有蒸馏或恢复训练、GQA/MLA/MoE 结构转换、稀疏 Attention 与真实稀疏 GPU kernel；原生模型选择不等于已经完成结构转换。',
                ],
                [
                  '6 · 内存与 KV',
                  <>
                    <Link to="/memory-cache">内存与 KV 管理</Link>选择动态、静态、CPU
                    卸载预取、重计算及有界前缀复用；L8 可选真实 GPU 块池。L4 保留 Quanto INT2/INT4
                    KV。
                  </>,
                  '块池暂需还原成稠密 Attention；不是 PagedAttention Kernel。量化 KV、编译和自定义缓存布局暂不组合；前缀复用仅验收 full attention。GPU L2 与逐页迁移未采集。',
                ],
                [
                  '7 · 解码算法',
                  <>
                    <Link to="/decoding">解码与生成算法</Link>
                    切换贪心/采样、提示词查找或独立草稿推测、有限候选
                    Mask/Gather，核对真实批量验证和 KV 回滚。
                  </>,
                  '只支持已验收的单请求原生浮点/动态 KV 路径；不把 PPL 当解码质量分数，没有随机推测分布校正、训练多 Token 头与通用 JSON Schema。',
                ],
                [
                  '8 · 单实例调度',
                  '在多请求调度实验选择串行、静态和 Continuous Batching，配置 Chunked Prefill、调度策略和重计算抢占',
                  '真实多请求执行；混合滑窗、量化 KV 和自定义算子组合仍受能力检查约束。',
                ],
                [
                  '9 · 分布式执行',
                  <>
                    <Link to="/parallel">多卡与分布式执行</Link>支持独立副本 DP、原生
                    TP、顺序按层跨卡、真实 KV 传输的 P/D 与已登记双节点请求 DP。
                  </>,
                  '单 GPU 不能冒充多卡；CPU/Gloo 只验证逻辑。没有多微批 PP 流水重叠、CP/SP、MoE EP、混合并行及跨主机 KV/TP。',
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
              登记发布方的兼容模型可从“模型下载”取得权重，在“本地模型管理”完成动态登记；其他新模型或定制部署需要在目标端取得权重、校验
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
                  '先检查输入输出和原因，按相同条件补齐独立实验；缺失项保留为空。',
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
