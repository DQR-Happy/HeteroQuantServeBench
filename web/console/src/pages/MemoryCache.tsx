import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Checkbox, Input, InputNumber, Select, Space, Table } from 'antd';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import Button from '../components/ActionButton';
import type { ActionBlock } from '../components/ActionButton';
import { api, fmt, terminal } from '../api/client';
import type { Deployment } from '../api/types';
import { PageHead, Panel, QueryState, Status } from '../components';
import MemoryEvidence from '../features/memory-cache/MemoryEvidence';
import MemoryComparison, { type Comparison } from '../features/memory-cache/MemoryComparison';
import type { Capability, MemoryPlan, MemoryRun } from '../features/memory-cache/types';
import '../features/memory-cache/styles.css';

const samples = [
  {
    label: '英文续写 / 短输入',
    text: 'Once upon a time, a little fox found a bright red ball. The fox',
  },
  {
    label: '机制解释 / 中文',
    text: '请用简洁的语言说明大语言模型推理时 KV Cache 的作用，以及它为什么需要占用显存。',
  },
  {
    label: '重复长前缀',
    text:
      'An inference engine stores keys and values to avoid repeated computation. Memory capacity and transfer overhead affect serving efficiency. '.repeat(
        12,
      ) + 'Explain the tradeoff:',
  },
];
export default function MemoryCache() {
  const { section = 'configure' } = useParams();
  const [params, setParams] = useSearchParams();
  const [mode, setMode] = useState('dynamic');
  const [prefix, setPrefix] = useState(false);
  const [budget, setBudget] = useState(64);
  const [capacity, setCapacity] = useState<number | null>(null);
  const [tokens, setTokens] = useState(32);
  const [prompt, setPrompt] = useState(samples[0].text);
  const [baseline, setBaseline] = useState('');
  const [candidate, setCandidate] = useState('');
  const [comparison, setComparison] = useState<Comparison>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [capability, setCapability] = useState<{ epoch: string; id: string; value: Capability }>();
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 3000,
  });
  const id =
    params.get('deployment') ??
    deployments.data?.items.find((d) => d.state === 'ready')?.id ??
    deployments.data?.items[0]?.id ??
    '';
  const deployment = deployments.data?.items.find((d) => d.id === id);
  const epoch = deployment?.epoch;
  const cap = capability?.id === id && capability.epoch === epoch ? capability.value : undefined;
  const overview = useQuery({
    queryKey: ['memory-overview'],
    queryFn: () => api<{ pending: number; benchmark_owner?: string | null }>('/overview'),
    refetchInterval: 3000,
  });
  const catalog = useQuery({
    queryKey: ['memory-catalog'],
    queryFn: () =>
      api<{ max_deadline_ms: number; modes: { id: string; label: string }[] }>('/memory/catalog'),
  });
  const history = useQuery({
    queryKey: ['memory-runs', id],
    queryFn: () =>
      api<{ items: MemoryRun[] }>(`/memory/runs?deployment_id=${encodeURIComponent(id)}`),
    enabled: !!id,
    refetchInterval: 3000,
  });
  const runId = params.get('run') ?? '';
  const selected = history.data?.items.find((r) => r.id === runId);
  const pending = !!selected && !terminal(selected.state);
  const deviceBusy = !!overview.data?.pending || !!overview.data?.benchmark_owner || pending;
  useEffect(() => {
    setBaseline('');
    setCandidate('');
    setComparison(undefined);
    setError('');
  }, [id]);
  const update = (key: string, value: string) =>
    setParams((previous) => {
      const next = new URLSearchParams(previous);
      next.set(key, value);
      if (key === 'deployment') next.delete('run');
      return next;
    });
  const deviceLink = `/devices#deployment-${encodeURIComponent(id)}`;
  const block: ActionBlock | undefined = busy
    ? { reason: '当前操作正在提交，请等待响应。' }
    : !deployment || deployment.state !== 'ready'
      ? {
          reason: '所选模型尚未就绪，请先在设备与部署页加载模型。',
          next: { to: deviceLink, label: '加载模型' },
        }
      : deviceBusy
        ? {
            reason: '设备有运行中的任务，请先等待完成或在运行记录中取消。',
            next: { to: runId ? `/runs/${runId}` : '/runs', label: '查看当前运行' },
          }
        : !cap
          ? { reason: '还没有检查当前驻留模型的 KV 能力，请点击本页“检查缓存能力”。' }
          : !cap.modes.find((m) => m.id === mode)?.supported
            ? {
                reason:
                  cap.modes.find((m) => m.id === mode)?.reason ??
                  '所选方案不支持当前模型，请重新检查缓存能力。',
                next: { to: deviceLink, label: '检查部署配置' },
              }
            : prefix && (mode !== 'dynamic' || !cap.prefix_supported)
              ? {
                  reason:
                    mode !== 'dynamic'
                      ? '前缀复用需要先在本页选择动态 KV。'
                      : (cap.prefix_reason ?? '当前模型不支持前缀复用。'),
                  next: { to: deviceLink, label: '查看模型' },
                }
              : !prompt.trim()
                ? { reason: '请先在本页填写输入文本或选择一个典型输入。' }
                : undefined;
  const controlBlock: ActionBlock | undefined =
    busy || deviceBusy
      ? {
          reason: '设备正在执行任务，请等待清理完成后再操作。',
          next: { to: '/runs', label: '查看运行记录' },
        }
      : deployment?.state !== 'ready'
        ? { reason: '请先在设备与部署页加载模型。', next: { to: deviceLink, label: '加载模型' } }
        : undefined;
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
  const submit = () =>
    perform(async () => {
      if (!deployment || !epoch) return;
      const plan: MemoryPlan = {
        mode,
        prefix_reuse: prefix,
        prefix_budget_bytes: budget * 1024 ** 2,
      };
      if (mode.includes('static') && capacity != null) plan.max_cache_tokens = capacity;
      const run = await api<MemoryRun>('/requests', {
        deployment_id: id,
        expected_epoch: epoch,
        prompt,
        max_output_tokens: tokens,
        fixed_output_length: true,
        deadline_ms: catalog.data?.max_deadline_ms ?? 120000,
        observation_mode: 'off',
        save_input: true,
        memory_plan: plan,
      });
      setParams({ deployment: id, run: run.id });
      await history.refetch();
    });
  const options =
    history.data?.items
      .filter((r) => r.state === 'completed' && r.metrics.memory_receipt)
      .map((r) => ({
        value: r.id,
        label: `${r.config.memory_plan?.mode} · ${r.metrics.memory_receipt?.prefix.lookup} · ${r.id}`,
      })) ?? [];
  return (
    <>
      <PageHead
        kicker="内存与缓存"
        title="L6 内存与 KV 管理"
        description="冻结缓存策略，执行真实模型，核对存储、搬运、复用及生成结果。"
      />
      <nav className="memory-cache-nav" aria-label="L6 子页">
        <Link
          aria-current={section === 'configure' ? 'page' : undefined}
          to={`/memory-cache/configure?${params}`}
        >
          配置与试运行
        </Link>
        <Link
          aria-current={section === 'records' ? 'page' : undefined}
          to={`/memory-cache/records?${params}`}
        >
          运行对照
        </Link>
        <Link to="/scheduling">L8 多请求块池与抢占实验 →</Link>
      </nav>
      <Panel title="当前模型">
        <Space wrap>
          <Select
            aria-label="L6 模型"
            style={{ minWidth: 280 }}
            value={id || undefined}
            options={deployments.data?.items.map((d) => ({
              value: d.id,
              label: `${d.name} · ${d.state}`,
            }))}
            onChange={(v) => update('deployment', v)}
          />
          <Link to={deviceLink}>加载、卸载或切换模型 →</Link>
          <Status state={deployment?.state ?? 'unloaded'} />
        </Space>
      </Panel>
      {error && (
        <Alert type="error" showIcon message={error} closable onClose={() => setError('')} />
      )}
      <QueryState error={deployments.error ?? history.error ?? catalog.error} />
      {section === 'configure' ? (
        <>
          <Panel title="缓存策略">
            <Space wrap>
              <Button
                disabled={!!controlBlock}
                disabledReason={controlBlock}
                loading={busy}
                onClick={() =>
                  void perform(async () => {
                    const value = await api<Capability>(
                      `/memory/capabilities?deployment_id=${encodeURIComponent(id)}`,
                    );
                    setCapability({ id, epoch: epoch!, value });
                  })
                }
              >
                检查缓存能力
              </Button>
              <label>
                KV 实现{' '}
                <Select
                  aria-label="KV 实现"
                  style={{ minWidth: 300 }}
                  value={mode}
                  options={(cap?.modes ?? catalog.data?.modes ?? []).map((m) => ({
                    value: m.id,
                    label: m.label,
                  }))}
                  onChange={(v) => {
                    setMode(v);
                    if (v !== 'dynamic') setPrefix(false);
                  }}
                />
              </label>
              {mode.includes('static') && (
                <label>
                  静态容量 Token（空为输入+输出−1）{' '}
                  <InputNumber
                    aria-label="静态容量"
                    min={1}
                    max={deployment?.context_limit ?? 131072}
                    value={capacity}
                    onChange={setCapacity}
                  />
                </label>
              )}
            </Space>
            <p>
              <Checkbox checked={prefix} onChange={(e) => setPrefix(e.target.checked)}>
                复用已完成请求的完整前缀（动态 / full attention）
              </Checkbox>
            </p>
            {prefix && (
              <label>
                驻留前缀预算 MiB{' '}
                <InputNumber
                  aria-label="前缀预算 MiB"
                  min={1}
                  max={512}
                  value={budget}
                  onChange={(v) => setBudget(v ?? 64)}
                />
              </label>
            )}
            <p className="muted">
              前缀快照在请求成功且设备完成后发布，命中时复制到请求私有 KV；不会跨模型 worker
              共用。关闭复用不会自动清空旧快照。
            </p>
            {cap?.prefix_reason && <p>{cap.prefix_reason}</p>}
            <Space wrap>
              <Button
                disabled={!!controlBlock}
                disabledReason={controlBlock}
                onClick={() =>
                  void perform(async () => {
                    await api('/memory/prefix/clear', { deployment_id: id, expected_epoch: epoch });
                    setCapability(undefined);
                  })
                }
              >
                清空驻留前缀
              </Button>
              <span>
                上次检查的保留量：
                {cap ? `${fmt(cap.prefix_retained_bytes / 1024 ** 2, 3)} MiB` : '未检查'}
              </span>
            </Space>
          </Panel>
          <Panel title="固定典型输入与执行">
            <Space wrap>
              <Select
                aria-label="典型输入"
                placeholder="选择典型输入"
                style={{ width: 240 }}
                options={samples.map((s, i) => ({ value: i, label: s.label }))}
                onChange={(i) => setPrompt(samples[i].text)}
              />
              <label>
                最大输出 Token{' '}
                <Select
                  aria-label="L6 输出 Token"
                  value={tokens}
                  options={[8, 16, 32, 64, 128]
                    .filter((n) => n <= (deployment?.max_output_tokens ?? 128))
                    .map((n) => ({ value: n, label: String(n) }))}
                  onChange={setTokens}
                />
              </label>
            </Space>
            <Input.TextArea
              aria-label="L6 原始输入"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={6}
            />
            <p>
              使用原始文本续写、贪心解码和固定输出长度。切换 KV
              方案后再次运行同一输入，可在“运行对照”选择基线与候选。
            </p>
            <Space>
              <Button
                type="primary"
                disabled={!!block}
                disabledReason={block}
                loading={busy}
                onClick={() => void submit()}
              >
                运行 L6 方案
              </Button>
              <Button
                disabled={!pending || busy}
                disabledReason={{
                  reason: busy
                    ? '当前操作正在提交，请等待响应后再取消。'
                    : '仅能取消本页当前正在运行的请求；先提交一个 L6 方案。',
                }}
                onClick={() =>
                  void perform(async () => {
                    await api(`/requests/${runId}/cancel`, {});
                    await history.refetch();
                  })
                }
              >
                取消当前运行
              </Button>
            </Space>
            {selected && (
              <p>
                <Link to={`/runs/${selected.id}`}>{selected.id}</Link> ·{' '}
                <Status state={selected.state} /> {selected.error}
              </p>
            )}
            {selected?.output && <pre className="memory-cache-output">{selected.output}</pre>}
          </Panel>
          <MemoryEvidence run={selected} compact />
        </>
      ) : (
        <>
          <Panel title="同输入运行对照">
            <p>
              对照会检查模型、源码、精度、输入 Token、输出预算和观测口径。生成 Token
              一致性只用于此输入的正确性检查，不等于模型质量得分。
            </p>
            <Space wrap>
              <Select
                aria-label="L6 基线"
                placeholder="选择基线"
                style={{ width: 350 }}
                value={baseline || undefined}
                options={options}
                onChange={(v) => {
                  setBaseline(v);
                  setComparison(undefined);
                }}
              />
              <Select
                aria-label="L6 候选"
                placeholder="选择候选"
                style={{ width: 350 }}
                value={candidate || undefined}
                options={options}
                onChange={(v) => {
                  setCandidate(v);
                  setComparison(undefined);
                }}
              />
              <Button
                loading={busy}
                disabled={!baseline || !candidate || baseline === candidate || busy}
                disabledReason={{
                  reason: busy
                    ? '比较请求正在提交，请等待响应。'
                    : '请先在本页选择两条不同的已完成 L6 记录；在“配置与试运行”生成记录。',
                  next: { to: `/memory-cache/configure?${params}`, label: '运行缓存方案' },
                }}
                onClick={() =>
                  void perform(async () =>
                    setComparison(
                      await api<Comparison>('/memory/comparisons', {
                        baseline_run_id: baseline,
                        candidate_run_id: candidate,
                      }),
                    ),
                  )
                }
              >
                核对可比性与差异
              </Button>
            </Space>
            {comparison != null && <MemoryComparison data={comparison} />}
          </Panel>
          <Panel title="L6 运行记录">
            <Table<MemoryRun>
              size="small"
              rowKey="id"
              dataSource={history.data?.items}
              pagination={{ pageSize: 12 }}
              scroll={{ x: 1000 }}
              columns={[
                {
                  title: '运行',
                  dataIndex: 'id',
                  render: (v: string) => (
                    <Link
                      to={`/memory-cache/records?deployment=${encodeURIComponent(id)}&run=${v}`}
                    >
                      {v.slice(0, 16)}…
                    </Link>
                  ),
                },
                { title: '策略', render: (_, r) => r.config.memory_plan?.mode },
                { title: '状态', dataIndex: 'state', render: (v: string) => <Status state={v} /> },
                { title: '生成 Token/s', render: (_, r) => fmt(r.metrics.output_tokens_per_s) },
                {
                  title: '请求总时间 ms（含清理）',
                  render: (_, r) => fmt(r.metrics.worker_request_total_ms),
                },
                {
                  title: '复用 Token',
                  render: (_, r) => fmt(r.metrics.memory_receipt?.prefix.reused_tokens, 0),
                },
                {
                  title: '卸载 MiB',
                  render: (_, r) =>
                    r.metrics.memory_receipt
                      ? fmt(r.metrics.memory_receipt.transfers.d2h_bytes / 1024 ** 2, 3)
                      : '—',
                },
              ]}
            />
          </Panel>
          <MemoryEvidence run={selected} />
        </>
      )}
      <Alert
        type="info"
        showIcon
        message="观测边界"
        description="这里测量实际 KV 张量、API 传输、分配器和进程缺页。块池在 L8 选择；其块是张量切片。GPU L2 命中率、操作系统逐页迁移和真正 PagedAttention Kernel 暂未接入。"
      />
    </>
  );
}
