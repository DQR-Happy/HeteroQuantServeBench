import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Checkbox, Input, Select, Space, Table } from 'antd';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import Button, { type ActionBlock } from '../components/ActionButton';
import { api, fmt, terminal } from '../api/client';
import type { Deployment, Run } from '../api/types';
import { JsonView, PageHead, Panel, QueryState, Status } from '../components';
import { useRun } from '../hooks';
import OutputTokenBudget from '../features/inference/OutputTokenBudget';
import PlanControls from '../features/decoding/PlanControls';
import DecodingEvidence, { ComparisonView } from '../features/decoding/DecodingEvidence';
import {
  defaults,
  type DecodePlan,
  type DecodeRun,
  type DecodeComparison,
} from '../features/decoding/types';
import '../features/decoding/styles.css';

const examples = [
  { id: 'repeat', label: '重复数字 / 查找与验证', prompt: '1 2 3 4 5 1 2 3 4 5 1 2 3 4 5' },
  {
    id: 'story',
    label: '英文续写 / 采样',
    prompt: 'Once upon a time, there was a little fox. The fox',
  },
  {
    id: 'classify',
    label: '情感分类 / 有限候选',
    prompt:
      'Classify the sentiment. Reply with positive, negative, or neutral.\nText: I love this movie!\nAnswer:',
  },
];
export default function Decoding() {
  const { section = 'run' } = useParams();
  const [params, setParams] = useSearchParams();
  const [plan, setPlan] = useState<DecodePlan>(defaults);
  const [choices, setChoices] = useState('');
  const [prompt, setPrompt] = useState(examples[0].prompt);
  const [tokens, setTokens] = useState<number | null>(32);
  const [fixed, setFixed] = useState(true);
  const [mode, setMode] = useState('off');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [base, setBase] = useState('');
  const [candidate, setCandidate] = useState('');
  const [comparison, setComparison] = useState<DecodeComparison>();
  const ds = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 2500,
  });
  const id =
    params.get('deployment') ??
    ds.data?.items.find((d) => d.state === 'ready')?.id ??
    ds.data?.items[0]?.id ??
    '';
  const d = ds.data?.items.find((d) => d.id === id);
  const device = `/devices#deployment-${encodeURIComponent(id)}`;
  const overview = useQuery({
    queryKey: ['decoding-overview'],
    queryFn: () => api<{ pending: number; benchmark_owner?: unknown }>('/overview'),
    refetchInterval: 1500,
  });
  const catalog = useQuery({
    queryKey: ['decoding-catalog'],
    queryFn: () =>
      api<{ draft_candidates: { id: string; name: string }[]; not_implemented: string[] }>(
        '/decoding/catalog',
      ),
  });
  const records = useQuery({
    queryKey: ['decoding-runs', id],
    queryFn: () =>
      api<{ items: DecodeRun[] }>(`/decoding/runs?deployment_id=${encodeURIComponent(id)}`),
    enabled: !!id,
    refetchInterval: 2000,
  });
  const run = useRun(params.get('run') ?? undefined);
  const record = run.data as DecodeRun | undefined;
  const active =
    !!overview.data?.pending ||
    !!overview.data?.benchmark_owner ||
    (!!record && !terminal(record.state));
  const update = (key: string, value: string) =>
    setParams((previous) => {
      const next = new URLSearchParams(previous);
      next.set(key, value);
      return next;
    });
  useEffect(() => {
    if (id && !params.has('deployment'))
      setParams(
        (p) => {
          const next = new URLSearchParams(p);
          next.set('deployment', id);
          return next;
        },
        { replace: true },
      );
  }, [id, params, setParams]);
  useEffect(() => {
    setBase('');
    setCandidate('');
    setComparison(undefined);
    setError('');
  }, [id]);
  const control: ActionBlock | undefined =
    busy || active
      ? {
          reason: '设备或当前任务仍在执行，请等待清理完成。',
          next: { to: record ? `/runs/${record.id}` : '/runs', label: '查看当前运行' },
        }
      : d?.state !== 'ready'
        ? { reason: '请先加载所选目标模型。', next: { to: device, label: '加载模型' } }
        : d.provider !== 'pytorch'
          ? {
              reason: 'L7 需要本机非特权 PyTorch 模型。',
              next: { to: device, label: '选择本地模型' },
            }
          : undefined;
  const modified =
    !!d?.structure_artifact_id ||
    (d?.quantization?.method ?? 'none') !== 'none' ||
    !!d?.quantization?.kv?.bits ||
    (d?.kernel_profile ?? 'framework-native') !== 'framework-native';
  const setup =
    control ??
    (modified
      ? {
          reason: '当前 L7 验证路径需要原始浮点权重、默认 KV 与框架原生算子，请先恢复。',
          next: {
            to: d?.structure_artifact_id
              ? `/model-structure/artifacts?deployment=${id}`
              : `/precision?deployment=${id}`,
            label: '恢复原始模型与默认配置',
          },
        }
      : undefined);
  const [budgetPrompt, setBudgetPrompt] = useState(prompt);
  useEffect(() => {
    const timer = setTimeout(() => setBudgetPrompt(prompt), 350);
    return () => clearTimeout(timer);
  }, [prompt]);
  const budget = useQuery({
    queryKey: ['decoding-budget', id, d?.epoch, budgetPrompt],
    queryFn: () =>
      api<{ max_output_tokens: number; input_tokens: number }>('/requests/token-budget', {
        deployment_id: id,
        expected_epoch: d?.epoch,
        prompt: budgetPrompt,
        max_output_tokens: 1,
      }),
    enabled: section === 'run' && !!d?.epoch && !control && !!budgetPrompt.trim(),
    retry: false,
  });
  const maximum = budget.data ? Math.min(512, budget.data.max_output_tokens) : undefined;
  const status = useQuery({
    queryKey: ['decoding-status', id, d?.epoch],
    queryFn: () =>
      api<Record<string, unknown>>(`/decoding/status?deployment_id=${encodeURIComponent(id)}`),
    enabled: section === 'draft' && !!d?.epoch && !control,
    retry: false,
  });
  const block =
    setup ??
    (!prompt.trim()
      ? { reason: '请先在本页填写输入文本。' }
      : plan.method === 'assistant' && !plan.draft_deployment_id
        ? {
            reason: '请先在本页选择已下载并登记的草稿模型。',
            next: { to: '/models', label: '下载与接入模型' },
          }
        : choices.trim() && fixed
          ? { reason: '有限候选需要 EOS 结束，请取消本页“固定输出长度”。' }
          : tokens == null || maximum == null || tokens > maximum
            ? { reason: '请等待输入预算检查完成，再选择不超过当前窗口的输出长度。' }
            : undefined);
  const perform = async (action: () => Promise<void>) => {
    setBusy(true);
    setError('');
    try {
      await action();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const options =
    records.data?.items
      .filter((r) => r.state === 'completed')
      .map((r) => ({
        value: r.id,
        label: `${String(r.metrics.decoding_receipt?.plan.method)} / ${String(r.metrics.decoding_receipt?.plan.sampling)} · ${r.id}`,
      })) ?? [];
  return (
    <>
      <PageHead
        kicker="L7"
        title="解码与生成算法"
        description="选择真实生成策略，核对候选验证、KV 回滚、采样和资源成本，再对照同一负载。"
      />
      <nav className="decoding-nav" aria-label="L7 子页">
        {[
          ['run', '策略与运行'],
          ['draft', '草稿模型与资源'],
          ['compare', '结果对照'],
        ].map(([key, label]) => (
          <Link
            key={key}
            aria-current={section === key ? 'page' : undefined}
            to={`/decoding/${key}?${params}`}
          >
            {label}
          </Link>
        ))}
      </nav>
      <Panel title="当前目标模型">
        <Space wrap>
          <Select
            aria-label="L7 目标模型"
            showSearch
            optionFilterProp="label"
            style={{ width: 330 }}
            value={id || undefined}
            options={ds.data?.items.map((d) => ({ value: d.id, label: `${d.name} · ${d.state}` }))}
            onChange={(v) => setParams({ deployment: v })}
          />
          <Status state={d?.state ?? 'unloaded'} />
          <Link to={device}>加载、卸载或切换目标模型 →</Link>
          <Link to={`/playground?deployment=${encodeURIComponent(id)}`}>原有交互推理 →</Link>
        </Space>
      </Panel>
      <QueryState error={ds.error ?? catalog.error ?? records.error} />
      {error && (
        <Alert type="error" showIcon message={error} closable onClose={() => setError('')} />
      )}
      {section === 'run' ? (
        <>
          <Panel title="生成策略">
            <PlanControls
              plan={plan}
              onChange={setPlan}
              choices={choices}
              onChoices={setChoices}
              drafts={catalog.data?.draft_candidates ?? []}
            />
          </Panel>
          <Panel title="固定样例与输入">
            <Space wrap>
              <Select
                aria-label="L7 典型输入"
                style={{ width: 260 }}
                placeholder="选择典型输入"
                options={examples.map((x) => ({ value: x.id, label: x.label }))}
                onChange={(v) => {
                  setPrompt(examples.find((x) => x.id === v)!.prompt);
                  if (v === 'classify') {
                    setPlan(defaults);
                    setChoices('positive\nnegative\nneutral');
                    setFixed(false);
                  }
                }}
              />
              <Checkbox checked={fixed} onChange={(e) => setFixed(e.target.checked)}>
                固定输出长度
              </Checkbox>
              <Select
                aria-label="L7 观测模式"
                value={mode}
                onChange={setMode}
                options={[
                  { value: 'off', label: '关闭详录 / 保留执行凭据' },
                  { value: 'basic', label: '基础阶段、OS 与内存观测' },
                ]}
                style={{ width: 290 }}
              />
            </Space>
            <Input.TextArea
              aria-label="L7 输入文本"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={5}
            />
            <label className="field-label" htmlFor="l7-output-budget">
              最大输出 Token（当前输入剩余窗口内，上限 512）
            </label>
            <div style={{ maxWidth: 430 }}>
              <OutputTokenBudget
                id="l7-output-budget"
                value={tokens}
                onChange={setTokens}
                maximum={maximum}
              />
            </div>
            <p>
              原始文本续写，不自动套聊天模板。每次显式提交才执行推理；草稿加载属于首次请求成本。
            </p>
            <Button
              type="primary"
              loading={busy}
              disabled={!!block}
              disabledReason={block}
              onClick={() =>
                void perform(async () => {
                  const result = await api<Run>('/requests', {
                    deployment_id: id,
                    expected_epoch: d?.epoch,
                    prompt,
                    max_output_tokens: tokens,
                    save_input: true,
                    fixed_output_length: fixed,
                    observation_mode: mode,
                    decoding_plan: {
                      ...plan,
                      choices:
                        plan.method === 'autoregressive'
                          ? choices.split('\n').filter((c) => c.trim())
                          : [],
                    },
                  });
                  update('run', result.id);
                })
              }
            >
              运行 L7 解码
            </Button>
          </Panel>
          {record && (
            <Panel title="本次真实输出">
              <p>
                <Link to={`/runs/${record.id}`}>{record.id}</Link> · {record.state}
              </p>
              {record.error && <Alert type="error" message={record.error} />}
              <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
                {record.output || '等待实际输出'}
              </pre>
              <p>
                生成速度：{fmt(record.metrics.output_tokens_per_s, 2)} Token/s · 首 Token：
                {fmt(record.metrics.runtime_first_token_ms, 2)} ms
              </p>
              <Button
                disabled={terminal(record.state) || busy}
                disabledReason={{ reason: '任务已经结束或当前操作正在提交。' }}
                onClick={() =>
                  void perform(async () => {
                    await api(`/requests/${record.id}/cancel`, {});
                  })
                }
              >
                取消当前 L7 运行
              </Button>
            </Panel>
          )}
          {record?.metrics.decoding_receipt && (
            <DecodingEvidence receipt={record.metrics.decoding_receipt} />
          )}
        </>
      ) : section === 'draft' ? (
        <Panel title="独立草稿模型的驻留与释放">
          <p>
            草稿与目标共享当前 worker 的 GPU 所有权；权重可跨请求驻留，请求 KV
            每次独立创建并释放。更换草稿前请先卸载。模型来源只允许已登记的本地制品。
          </p>
          <Space wrap>
            <Button
              disabled={!!control}
              disabledReason={control}
              onClick={() =>
                void perform(async () => {
                  await status.refetch();
                })
              }
            >
              检查草稿驻留状态
            </Button>
            <Button
              disabled={!!control || status.data?.status !== 'loaded'}
              disabledReason={
                control ?? {
                  reason:
                    '当前尚未确认有驻留草稿，请先检查本页状态，或在策略页运行一次独立草稿模型推测。',
                  next: { to: `/decoding/run?${params}`, label: '选择并运行草稿推测' },
                }
              }
              onClick={() =>
                void perform(async () => {
                  const r = await api<Run>('/decoding/draft/unload', {
                    deployment_id: id,
                    expected_epoch: d?.epoch,
                  });
                  update('run', r.id);
                  await status.refetch();
                })
              }
            >
              卸载草稿模型
            </Button>
            <Link to={`/decoding/run?${params}`}>配置并运行草稿推测 →</Link>
            <Link to="/models">下载较小且同 Tokenizer 的草稿模型 →</Link>
          </Space>
          <QueryState error={status.error} />
          <JsonView value={status.data} />
          {record?.kind === 'draft_unload' && (
            <p>
              <Link to={`/runs/${record.id}`}>{record.id}</Link> · {record.state}
            </p>
          )}
          <p>
            同一个检查点的独立副本可用来验证算法，但不代表获得小草稿模型的成本优势。更换目标模型会一并释放草稿。
          </p>
        </Panel>
      ) : (
        <>
          <Panel title="同负载的解码方案对照">
            <p>
              先用逐 Token
              贪心运行基线，再保持输入、输出预算、精度与观测口径运行推测候选。有限候选可比较 Mask 与
              Gather；随机采样的输出差异不等于量化质量下降。
            </p>
            <Space wrap>
              <Select
                aria-label="L7 基线"
                showSearch
                optionFilterProp="label"
                style={{ width: 350 }}
                placeholder="选择基线运行"
                value={base || undefined}
                options={options}
                onChange={(v) => {
                  setBase(v);
                  setComparison(undefined);
                }}
              />
              <Select
                aria-label="L7 候选"
                showSearch
                optionFilterProp="label"
                style={{ width: 350 }}
                placeholder="选择候选运行"
                value={candidate || undefined}
                options={options}
                onChange={(v) => {
                  setCandidate(v);
                  setComparison(undefined);
                }}
              />
              <Button
                disabled={busy || !base || !candidate || base === candidate}
                disabledReason={{
                  reason: busy
                    ? '正在读取与核对封存证据。'
                    : '请在本页选择两条已完成的不同运行；先到策略页执行基线和候选。',
                  next: { to: `/decoding/run?${params}`, label: '运行基线与候选' },
                }}
                onClick={() =>
                  void perform(async () =>
                    setComparison(
                      await api<DecodeComparison>('/decoding/comparisons', {
                        baseline_run_id: base,
                        candidate_run_id: candidate,
                      }),
                    ),
                  )
                }
              >
                比较 L7 结果
              </Button>
            </Space>
            {comparison && <ComparisonView data={comparison} />}
          </Panel>
          <Panel title="当前模型的解码记录">
            <Table
              rowKey="id"
              size="small"
              pagination={{ pageSize: 8 }}
              scroll={{ x: 800 }}
              dataSource={records.data?.items}
              columns={[
                {
                  title: '运行',
                  dataIndex: 'id',
                  render: (v) => <Link to={`/decoding/run?deployment=${id}&run=${v}`}>{v}</Link>,
                },
                { title: '状态', dataIndex: 'state' },
                {
                  title: '方法',
                  render: (_, r) => String(r.metrics.decoding_receipt?.plan.method ?? '—'),
                },
                { title: 'Token/s', render: (_, r) => fmt(r.metrics.output_tokens_per_s, 2) },
                {
                  title: '接受率',
                  render: (_, r) => {
                    const v = r.metrics.decoding_receipt?.statistics.acceptance_rate;
                    return typeof v === 'number' ? `${fmt(v * 100, 2)}%` : '—';
                  },
                },
              ]}
            />
          </Panel>
        </>
      )}
      <details>
        <summary>当前边界与未接入方向</summary>
        {catalog.data?.not_implemented.map((x) => <p key={x}>{x}</p>)}
        <p>
          首批真实推测仅支持单请求、原生浮点/默认动态 KV；与 L3 编译、L4/L5、L6 特殊缓存及 L8
          调度组合尚未验收。浮点批量与单步计算在接近并列最大值时可能选出不同 Token，必须查看真实逐
          Token 对照。
        </p>
      </details>
    </>
  );
}
