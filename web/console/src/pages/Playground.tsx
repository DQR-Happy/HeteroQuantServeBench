import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Alert,
  App,
  Button,
  Checkbox,
  Collapse,
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
import ReactMarkdown from 'react-markdown';
import { Link } from 'react-router-dom';
import { api, terminal } from '../api/client';
import type { Deployment, InferenceRequest, Run, Session } from '../api/types';
import { PageHead, Panel, Stat, Status, QueryState } from '../components';
import { useRun } from '../hooks';
import ObservationControl from '../features/observability/ObservationControl';
import type { ObservationMode } from '../features/observability/types';

export default function Playground() {
  const { message } = App.useApp();
  const client = useQueryClient();
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 2000,
  });
  const [selected, setSelected] = useState('');
  const deployment =
    deployments.data?.items.find((x) => x.id === selected) ?? deployments.data?.items[0];
  const [prompt, setPrompt] = useState(
    '请用一个简单的例子解释，大模型推理为什么可能受到内存带宽的限制？',
  );
  const [system, setSystem] = useState('你是一位严谨的技术助手。请用中文简明回答。');
  const [tokens, setTokens] = useState(128);
  const [save, setSave] = useState(false);
  const [observationMode, setObservationMode] = useState<ObservationMode>('basic');
  const [submitted, setSubmitted] = useState<{ id: string; time: number }>();
  const [sending, setSending] = useState(false);
  const run = useRun(submitted?.id, submitted?.time);
  const localProvider = deployment?.provider === 'pytorch';
  const deadlineLimit = client.getQueryData<Session>(['session'])?.limits.max_deadline_ms ?? 120000;
  const active = sending || (!!run.data && !terminal(run.data.state));
  const start = async () => {
    if (!deployment?.epoch || !prompt.trim()) return;
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
        max_output_tokens: Math.min(tokens, deployment.max_output_tokens),
        deadline_ms: Math.min(120000, deadlineLimit),
        save_input: save,
        observation_mode: localProvider ? observationMode : 'off',
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
            disabled={active}
          />
          <div className="config-meta">
            <span>执行平台</span>
            <strong>{deployment?.platform ?? '—'}</strong>
            <span>推理引擎</span>
            <strong>
              {deployment?.provider === 'pytorch' ? 'HQSB PyTorch Reference' : 'OpenAI-compatible'}
            </strong>
          </div>
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
          <label className="field-label">算子配置</label>
          <Select
            aria-label="算子配置"
            value="native"
            style={{ width: '100%' }}
            options={[
              {
                label: localProvider
                  ? `Framework native · ${deployment?.attention ?? 'eager'}`
                  : `${deployment?.kernel ?? '—'} · 上游管理`,
                value: 'native',
              },
              { label: '自定义 kernel · 模型回接待验证', value: 'custom', disabled: true },
            ]}
          />
          <div className="help-text">
            {localProvider
              ? '当前实际执行框架原生算子，无静默替换。'
              : '外部引擎的实际内核与权重精度尚未独立核验。'}
          </div>
          <label className="field-label" htmlFor="max-tokens">
            最大输出 token
          </label>
          <InputNumber
            id="max-tokens"
            min={1}
            max={deployment?.max_output_tokens ?? 512}
            value={Math.min(tokens, deployment?.max_output_tokens ?? 512)}
            onChange={(n) => setTokens(n ?? 128)}
            style={{ width: '100%' }}
            disabled={active}
          />
          <div className="config-meta">
            <span>采样策略</span>
            <strong>Greedy · temperature 0</strong>
            <span>上下文上限</span>
            <strong>{deployment?.context_limit ?? '—'} tokens</strong>
          </div>
          <ObservationControl
            value={observationMode}
            onChange={setObservationMode}
            disabled={active}
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
          {deployment?.state === 'unloaded' || deployment?.state === 'failed' ? (
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
                ? '默认关闭 Qwen thinking；服务端唯一应用聊天模板'
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
                  disabled={deployment?.state !== 'ready' || active || !prompt.trim()}
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
              <div className="markdown output-text">
                <ReactMarkdown skipHtml>{run.data.output}</ReactMarkdown>
                {active && <span className="cursor" />}
              </div>
            ) : (
              <div className="output-empty">
                <div className="output-mark">H</div>
                <h3>{active ? '等待首个生成内容…' : '让一次推理成为可观察的实验'}</h3>
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
                <Link to={`/runs/${submitted.id}`}>
                  查看完整运行 <ArrowRightOutlined />
                </Link>
              </div>
            )}
          </Panel>
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
          <div className="info-note">
            <strong>指标有边界</strong>
            <p>浏览器、服务和模型的计时不同。此处是交互诊断，不能替代正式性能基准。</p>
          </div>
        </aside>
      </div>
    </>
  );
}
