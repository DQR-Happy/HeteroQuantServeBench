import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Alert,
  App,
  Button,
  Checkbox,
  Collapse,
  Descriptions,
  Input,
  InputNumber,
  Select,
  Space,
  Tag,
  Tooltip,
} from 'antd';
import {
  ArrowUpOutlined,
  StopOutlined,
  CopyOutlined,
  ArrowRightOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import { Link, useSearchParams } from 'react-router-dom';
import { api, fmt, gib, terminal } from '../api/client';
import type { Deployment, InferenceRequest, Run, Session } from '../api/types';
import { PageHead, Panel, Stat, Status, QueryState } from '../components';
import { useRun } from '../hooks';
import ObservationControl from '../features/observability/ObservationControl';
import type { ObservationMode } from '../features/observability/types';
import KernelControl from '../features/kernels/KernelControl';
import RunEvidence from '../features/experiments/RunEvidence';
import ModelOutput from '../features/inference/ModelOutput';

export default function Playground() {
  const [params] = useSearchParams();
  const { message } = App.useApp();
  const client = useQueryClient();
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 2000,
  });
  const [selected, setSelected] = useState(params.get('deployment') ?? '');
  const [kernelDrafts, setKernelDrafts] = useState<Record<string, string>>({});
  const [lifecycleBusy, setLifecycleBusy] = useState(false);
  const deployment =
    deployments.data?.items.find((x) => x.id === selected) ?? deployments.data?.items[0];
  const [prompt, setPrompt] = useState(
    '请用一个简单的例子解释，大模型推理为什么可能受到内存带宽的限制？',
  );
  const [system, setSystem] = useState('你是一位严谨的技术助手。请用中文简明回答。');
  const [tokens, setTokens] = useState<number | null>(128);
  const [save, setSave] = useState(false);
  const [cleanup, setCleanup] = useState('empty_cache');
  const [observationMode, setObservationMode] = useState<ObservationMode>('basic');
  const [submitted, setSubmitted] = useState<{ id: string; time: number }>();
  const [sending, setSending] = useState(false);
  const run = useRun(submitted?.id, submitted?.time);
  const observed = run.data?.metrics.observation;
  const localProvider = deployment?.provider === 'pytorch';
  const kernelProfile =
    (deployment && kernelDrafts[deployment.id]) ?? deployment?.kernel_profile ?? 'framework-native';
  const kernelPending =
    localProvider && kernelProfile !== (deployment?.kernel_profile ?? 'framework-native');
  const deadlineLimit = client.getQueryData<Session>(['session'])?.limits.max_deadline_ms ?? 120000;
  const active = sending || (!!run.data && !terminal(run.data.state));
  const configLocked = active || lifecycleBusy;
  const outputLimit = deployment?.max_output_tokens;
  const outputError =
    tokens == null || !Number.isInteger(tokens) || tokens < 1
      ? '请输入大于等于 1 的整数。'
      : outputLimit != null && tokens > outputLimit
        ? `当前部署最多输出 ${outputLimit} tokens；请修改输出预算，输入值不会自动截断。`
        : deployment && tokens >= deployment.context_limit
          ? '输出预算必须小于部署总窗口，为输入和聊天模板留出空间。'
          : '';
  const start = async () => {
    if (
      !deployment?.epoch ||
      !prompt.trim() ||
      outputError ||
      tokens == null ||
      configLocked ||
      kernelPending ||
      deployment.state !== 'ready'
    )
      return;
    setSending(true);
    const time = performance.now();
    try {
      const body: InferenceRequest = {
        deployment_id: deployment.id,
        expected_epoch: deployment.epoch,
        messages: [
          ...(system.trim() ? [{ role: 'system' as const, content: system }] : []),
          { role: 'user', content: prompt },
        ],
        max_output_tokens: tokens,
        deadline_ms: Math.min(120000, deadlineLimit),
        save_input: save,
        observation_mode: localProvider ? observationMode : 'off',
        ...(localProvider
          ? {
              optimization: { changes: { cleanup_policy: cleanup, kernel_profile: kernelProfile } },
            }
          : {}),
      };
      const row = await api<Run>('/requests', body);
      client.setQueryData(['run', row.id], row);
      setSubmitted({ id: row.id, time });
    } catch (e) {
      void message.error((e as Error).message);
    } finally {
      setSending(false);
    }
  };
  const stop = async () => {
    if (submitted)
      try {
        await api(`/requests/${submitted.id}/cancel`, {});
        void client.invalidateQueries({ queryKey: ['run', submitted.id] });
      } catch (e) {
        void message.error((e as Error).message);
      }
  };
  const load = async () => {
    if (deployment)
      try {
        await api(`/deployments/${deployment.id}/load`, {});
        void client.invalidateQueries({ queryKey: ['deployments'] });
        void message.info('模型正在远端加载与预热，可在运行列表查看进度。');
      } catch (e) {
        void message.error((e as Error).message);
      }
  };
  return (
    <>
      <PageHead
        kicker="INFERENCE WORKSPACE"
        title="推理工作台"
        description="选择真实部署，输入文本，观察生成过程与分层指标。"
        extra={deployment && <Status state={deployment.state} />}
      />
      <QueryState error={deployments.error} />
      <div className="workspace">
        <Panel className="config-panel" title="执行配置" subtitle="参数以服务端能力为准">
          <label className="field-label" htmlFor="deployment">
            模型部署
          </label>
          <Select
            id="deployment"
            value={deployment?.id}
            onChange={setSelected}
            options={deployments.data?.items.map((d) => ({ label: d.name, value: d.id }))}
            style={{ width: '100%' }}
            disabled={configLocked}
          />
          <div className="config-meta">
            <span>执行平台</span>
            <strong>{deployment?.platform ?? '—'}</strong>
            <span>推理引擎</span>
            <strong>
              {deployment?.provider === 'pytorch' ? 'HQSB PyTorch Reference' : 'OpenAI-compatible'}
            </strong>
          </div>
          {localProvider && (
            <>
              <label className="field-label" htmlFor="cleanup-policy">
                请求后清理策略
              </label>
              <Select
                id="cleanup-policy"
                value={cleanup}
                onChange={setCleanup}
                disabled={configLocked}
                style={{ width: '100%' }}
                options={[
                  { value: 'empty_cache', label: '释放 allocator 空闲缓存' },
                  { value: 'preserve_allocator', label: '保留 allocator 空闲缓存' },
                ]}
              />
              <div className="help-text">
                实际生效值记录在运行凭据中。<Link to="/optimization">建立质量与机制对照</Link>
              </div>
            </>
          )}
          <label className="field-label">量化方式</label>
          <Select
            aria-label="量化方式"
            value={deployment?.precision ?? 'float16'}
            style={{ width: '100%' }}
            options={
              localProvider
                ? [
                    { label: 'FP16 · 框架原生', value: 'float16' },
                    { label: 'W8 · 尚无已验证交互部署', value: 'w8', disabled: true },
                    { label: 'W4 · 尚无已验证交互部署', value: 'w4', disabled: true },
                  ]
                : [
                    {
                      label: `${deployment?.precision ?? '—'} · 管理员声明`,
                      value: deployment?.precision ?? 'float16',
                    },
                  ]
            }
          />
          <div className="help-text">
            量化选择绑定实际制品与执行路径。<Link to="/quantization">查看量化证据</Link>
          </div>
          {localProvider && deployment ? (
            <KernelControl
              key={deployment.id}
              deployment={deployment}
              selected={kernelProfile}
              onSelect={(value) => setKernelDrafts((old) => ({ ...old, [deployment.id]: value }))}
              disabled={active}
              onBusyChange={setLifecycleBusy}
            />
          ) : (
            <div className="help-text">外部引擎的算子由上游管理，不能在此替换。</div>
          )}
          <label className="field-label" htmlFor="max-tokens">
            最大输出 token（仅生成部分）
          </label>
          <InputNumber
            id="max-tokens"
            value={tokens}
            onChange={setTokens}
            status={outputError ? 'error' : undefined}
            aria-describedby="token-budget-help"
            aria-invalid={!!outputError}
            style={{ width: '100%' }}
            disabled={configLocked}
          />
          <div
            id="token-budget-help"
            className="help-text"
            role={outputError ? 'alert' : undefined}
          >
            {outputError || `本部署输出范围：1–${outputLimit ?? '—'} tokens。`}
          </div>
          <div className="config-meta">
            <span>采样策略</span>
            <strong>Greedy · temperature 0</strong>
            <span>部署总窗口（输入＋输出）</span>
            <strong>{deployment?.context_limit ?? '—'} tokens</strong>
            {deployment?.detail.native_context_limit && (
              <>
                <span>模型配置声明窗口</span>
                <strong>{deployment.detail.native_context_limit} tokens</strong>
              </>
            )}
          </div>
          <div className="help-text">
            总窗口包含 System prompt、聊天模板、输入和输出预算；以分词后的 token 数校验，
            不按字符计数。部署上限是当前服务配置，不等于模型理论极限；增大窗口仍受显存与截止时间约束。
          </div>
          <ObservationControl
            value={observationMode}
            onChange={setObservationMode}
            disabled={configLocked}
            local={localProvider}
          />
          <Collapse
            ghost
            items={[
              {
                key: 'advanced',
                label: (
                  <span>
                    <SettingOutlined /> 高级设置
                  </span>
                ),
                children: (
                  <>
                    <label className="field-label">System prompt</label>
                    <Input.TextArea
                      aria-label="System prompt"
                      rows={4}
                      value={system}
                      onChange={(e) => setSystem(e.target.value)}
                    />
                    <Checkbox checked={save} onChange={(e) => setSave(e.target.checked)}>
                      保存输入，供后续精确复现
                    </Checkbox>
                    <div className="help-text">
                      默认仅保存输入摘要；输出与指标会保存在私有运行记录中。
                    </div>
                  </>
                ),
              },
            ]}
          />
          {!localProvider &&
          (deployment?.state === 'unloaded' || deployment?.state === 'failed') ? (
            <Button block onClick={() => void load()}>
              加载并预热模型
            </Button>
          ) : null}
        </Panel>
        <div className="generation-column">
          <Panel
            title="输入"
            subtitle={
              localProvider
                ? '使用各模型自带聊天模板；Qwen3 关闭 thinking，其他模型保留模板默认行为'
                : '聊天模板与 thinking 策略由上游引擎管理'
            }
          >
            <Input.TextArea
              aria-label="推理输入"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={5}
              maxLength={32768}
              showCount
            />
            <div className="input-actions">
              <Space>
                <Tag>真实执行</Tag>
                <span className="muted small">长请求受上下文与截止时间约束</span>
              </Space>
              <Space>
                {active && (
                  <Button danger icon={<StopOutlined />} onClick={() => void stop()}>
                    停止执行
                  </Button>
                )}
                <Button
                  type="primary"
                  icon={<ArrowUpOutlined />}
                  disabled={
                    deployment?.state !== 'ready' ||
                    configLocked ||
                    kernelPending ||
                    !prompt.trim() ||
                    !!outputError
                  }
                  loading={sending}
                  onClick={() => void start()}
                >
                  开始推理
                </Button>
              </Space>
            </div>
          </Panel>
          <Panel
            className="output-panel"
            title="模型输出"
            subtitle={
              submitted
                ? `${submitted.id.slice(0, 20)} · ${run.connection === 'reconnecting' ? '正在恢复观察连接' : run.connection === 'completed' ? '运行已结束' : '实时事件流'}`
                : '输出会在真正生成后逐步呈现'
            }
            extra={
              <Space>
                {run.data && <Status state={run.data.state} />}
                <Tooltip title="复制输出">
                  <Button
                    aria-label="复制输出"
                    type="text"
                    icon={<CopyOutlined />}
                    disabled={!run.data?.output}
                    onClick={() => {
                      void navigator.clipboard
                        .writeText(run.data?.output ?? '')
                        .then(() => message.success('已复制'))
                        .catch(() => message.error('浏览器不允许访问剪贴板'));
                    }}
                  />
                </Tooltip>
              </Space>
            }
          >
            {run.data?.output ? (
              <ModelOutput text={run.data.output} active={active} />
            ) : (
              <div className="output-empty">
                <h3>{active ? '等待首个生成内容…' : '尚未提交请求'}</h3>
                <p>
                  {deployment?.state === 'ready'
                    ? '模型已就绪。点击「开始推理」查看真实输出。'
                    : '先加载部署，再开始推理。模型加载也会生成独立运行记录。'}
                </p>
              </div>
            )}
            {run.data?.error && (
              <Alert type="error" showIcon message="运行失败" description={run.data.error} />
            )}
            {run.data?.finish_reason === 'length' && (
              <div className="help-text">
                已到达输出 token 上限；如需更长回答，请提高预算后重新提交。
              </div>
            )}
            {submitted && (
              <div className="output-footer">
                <span>清理状态：{run.data?.cleanup ?? 'pending'} · 质量未评估</span>
                {run.data && terminal(run.data.state) && (
                  <Link to={`/observations?run=${submitted.id}`}>检查机制证据 →</Link>
                )}
                <Link to={`/runs/${submitted.id}`}>
                  查看完整运行 <ArrowRightOutlined />
                </Link>
              </div>
            )}
          </Panel>
          {run.data && (
            <Panel
              title="本次运行凭据与观测"
              subtitle="绑定本次运行与部署版本；切换模型后历史凭据保持原样"
            >
              <RunEvidence run={run.data} />
              <h3>本次观测摘要</h3>
              <Descriptions
                size="small"
                column={{ xs: 1, md: 2 }}
                items={[
                  { key: 'mode', label: '观测模式', children: observed?.mode ?? '尚无详细观测' },
                  { key: 'phases', label: '阶段记录', children: observed?.phases.length ?? '—' },
                  {
                    key: 'tokens',
                    label: '逐 token 样本',
                    children: observed?.tokens.length ?? '—',
                  },
                  {
                    key: 'memory',
                    label: '峰值已分配 / GiB',
                    children: fmt(gib(run.data.metrics.memory?.peak_allocated_bytes), 3),
                  },
                  {
                    key: 'profile',
                    label: 'Profiler 状态',
                    children: observed?.profile?.status ?? '未采集',
                  },
                  {
                    key: 'gpu',
                    label: 'GPU 活动',
                    children: observed?.profile?.activities?.includes('CUDA')
                      ? '已采集 CUDA 活动'
                      : '未采集 CUDA 活动',
                  },
                ]}
              />
              <Space wrap className="section-gap">
                <Link to={`/observations?run=${run.data.id}`}>查看内存、KV 与系统页观测</Link>
                <Link to={`/runs/${run.data.id}?tab=observation`}>
                  查看阶段、token 与 GPU 算子观测
                </Link>
              </Space>
            </Panel>
          )}
        </div>
        <aside className="observation-column">
          <div className="observation-title">
            <span className="live-dot" /> 请求观测
          </div>
          <Stat
            label="浏览器流首内容"
            value={run.clientFirst}
            unit="ms"
            hint="本次提交 → SSE 首个非空内容"
            accent
          />
          <Stat
            label="服务端首内容"
            value={run.data?.metrics.console_first_content_ms}
            unit="ms"
            hint="API 入队 → 首内容接收（含排队）"
          />
          <Stat
            label="Runtime 首 token"
            value={run.data?.metrics.runtime_first_token_ms}
            unit="ms"
            hint="引擎执行开始 → 首 token 就绪"
          />
          <Stat
            label="已生成 token"
            value={run.data?.metrics.output_tokens}
            hint="以引擎记账为准，非网络分片数"
          />
          <Stat
            label="生成速度"
            value={run.data?.metrics.output_tokens_per_s}
            unit="Token/s"
            hint="输出 Token 数 ÷ 生成循环时间，含首 Token 等待；完成后给出"
          />
          <Stat
            label="Decode 速度"
            value={run.data?.metrics.decode_tail_tokens_per_s}
            unit="Token/s"
            hint="首 Token 之后的生成速度；少于 2 个 Token 时无定义"
          />
          <div className="info-note">
            <strong>指标有边界</strong>
            <p>浏览器、服务和模型的计时不同。此处是交互诊断，不能替代正式性能基准。</p>
          </div>
        </aside>
      </div>
    </>
  );
}
