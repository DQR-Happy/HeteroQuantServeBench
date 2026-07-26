import { useQuery } from '@tanstack/react-query';
import { Alert, Button, Descriptions, Steps, Table, Tag } from 'antd';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import type { Session } from '../api/types';
import { PageHead, Panel, QueryState } from '../components';

export function Showcase() {
  return (
    <>
      <PageHead
        kicker="A TECHNICAL STORY, WITH PROOF"
        title="技术展示路线"
        description="沿着问题、测量、优化、质量与限制，讲清楚一次推理工程决策。每一步都落到真实页面和证据。"
      />
      <div className="story-banner">
        <Tag>建议 8–12 分钟</Tag>
        <h2>
          性能数字背后，
          <br />
          是可解释的工程选择。
        </h2>
        <p>
          从真实请求开始，以证据和边界收尾。展示你如何发现问题、验证方案，并识别尚未成立的结论。
        </p>
        <Link to="/playground">
          <Button type="primary" size="large">
            开始真实演示
          </Button>
        </Link>
      </div>
      <Panel title="五步演示" subtitle="无需伪造 GPU 曲线或预填性能结果">
        <Steps
          direction="vertical"
          items={[
            {
              title: '1. 提出约束：8 GB 端侧设备能稳定服务什么？',
              description: (
                <p>
                  打开 <Link to="/devices">设备与部署</Link>
                  ，展示实际模型、内存、执行平台和上下文上限。解释为何一次只驻留一个模型。
                </p>
              ),
            },
            {
              title: '2. 跑通一条请求，分清时间花在哪里',
              description: (
                <p>
                  在 <Link to="/playground">推理工作台</Link> 提交提示词，区分浏览器首内容、API
                  首内容、Runtime 首 token，演示取消与资源清理。
                </p>
              ),
            },
            {
              title: '3. 从整体指标追到具体实现',
              description: (
                <p>
                  打开 <Link to="/runs">运行详情</Link> 看逐 token 延迟和实际 token ID，再看{' '}
                  <Link to="/kernels">算子源码</Link>。清楚说明单算子优化与模型回接是两层验证。
                </p>
              ),
            },
            {
              title: '4. 展示一个没有通过的优化，以及原因',
              description: (
                <p>
                  进入 <Link to="/quantization">量化与质量</Link> 打开 E05-02 的 verdict
                  和报告。解释原生低比特与 FP16 反量化控制路径的区别。
                </p>
              ),
            },
            {
              title: '5. 说明多平台的实用价值与尚待完成的部分',
              description: (
                <p>
                  用 <Link to="/compare">运行对比</Link>{' '}
                  展示可比性门，说明多平台用于识别硬件约束、验证移植和部署决策。没有设备证据时，不宣布该平台获得加速。
                </p>
              ),
            },
          ]}
        />
      </Panel>
      <div className="grid-two">
        <Panel title="与推理引擎、编译器的关系">
          <p>
            HQSB Console 是面向异构推理工程的实验与观测入口。vLLM / SGLang
            可以作为提供推理服务的引擎；编译器与自定义算子可以提供候选执行路径。
          </p>
          <p>
            项目要证明的是：在明确约束下，为什么选择某个方案、实际运行了什么、质量和性能证据是否支持这个选择。
          </p>
        </Panel>
        <Panel title="秋招里值得展示的工程能力">
          <p>AI Infra：API 契约、进程隔离、背压、幂等、取消和重启恢复。</p>
          <p>推理优化：真实 token 计时、测量范围、内存语义与可比性门。</p>
          <p>算子与异构：源码、数值验证、模型回接以及能力缺口的明确表达。</p>
        </Panel>
      </div>
    </>
  );
}

export function SettingsPage() {
  const query = useQuery({ queryKey: ['session'], queryFn: () => api<Session>('/session') });
  const session = query.data;
  return (
    <>
      <PageHead
        kicker="CONSOLE CONTRACT"
        title="设置与使用指引"
        description="当前工作台采用单操作者模式。部署配置与访问令牌由服务端管理，页面不读取 SSH 密钥和供应商 API key。"
      />
      <QueryState loading={query.isPending} error={query.error} />
      <Panel title="当前会话">
        <Descriptions
          column={2}
          items={[
            { key: 'version', label: 'Console 版本', children: session?.version },
            { key: 'api', label: 'API 契约', children: `v${session?.api_version ?? '—'}` },
            { key: 'mode', label: '数据模式', children: '真实执行 + 历史证据' },
            { key: 'queue', label: '队列上限', children: session?.limits.max_pending },
            {
              key: 'deadline',
              label: '请求最大截止时间',
              children: `${(session?.limits.max_deadline_ms ?? 0) / 1000} s`,
            },
            { key: 'auth', label: '会话有效期', children: '8 小时 · HttpOnly cookie' },
          ]}
        />
      </Panel>
      <Panel title="指标说明">
        <Table
          size="small"
          pagination={false}
          rowKey="name"
          dataSource={[
            {
              name: '浏览器流首内容',
              meaning: '点击提交到 SSE 首个非空输出；包含网络、API 和事件轮询。刷新后不重造数值。',
            },
            {
              name: '服务端首内容',
              meaning: 'API 接受请求到收到首个非空解码文本，包含排队和工作进程传输。',
            },
            {
              name: 'Runtime 首 token',
              meaning:
                '本地 Provider 开始生成到首个 token ID 就绪；可能先于可显示文本。外部服务未知则为空。',
            },
            {
              name: 'Decode token/s',
              meaning: '首 token 之后的 token 数 / 首末 token 时间间隔。输出不足 2 token 时为空。',
            },
            {
              name: '共享内存 / CUDA allocator',
              meaning:
                '主机 MemAvailable 与 PyTorch allocator 是不同口径，不能相加或替代设备物理显存。',
            },
            {
              name: '质量未评估',
              meaning: '成功生成文本只说明功能执行完成，不等于正确性、精度或应用质量通过。',
            },
          ]}
          columns={[
            { title: '指标', dataIndex: 'name', width: 200 },
            { title: '口径', dataIndex: 'meaning' },
          ]}
        />
      </Panel>
      <Alert
        showIcon
        type="info"
        message="完整安装、逐页操作、故障处理与接口说明"
        description="见仓库 docs/manual/前端使用说明书.md；可版本管理的运行入口说明位于 web/console/README.md。依赖安装、构建、测试与模型执行均在远端进行。"
      />
    </>
  );
}
