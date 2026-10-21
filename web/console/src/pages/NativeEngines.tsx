import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Checkbox, Input, InputNumber, Select, Space, Table, Tag } from 'antd';
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import Button from '../components/ActionButton';
import { api, fmt, terminal } from '../api/client';
import type { Deployment, Run } from '../api/types';
import { JsonView, PageHead, Panel, QueryState } from '../components';
import { useRun } from '../hooks';

type Value = string | number | boolean;
type Control = {
  id: string;
  layers: number[];
  label: string;
  default: Value;
  values?: (string | number)[];
  minimum?: number;
  maximum?: number;
};
type Engine = { engine: string; version: string; installed: boolean; items: Control[] };
type NativeComparison = {
  comparable: boolean;
  quality_gate: { status: string };
  eligible_for_adoption: boolean;
};
type NativeDeployment = Deployment & { engine_options?: Record<string, Value> };
type Receipt = {
  engine: { requested: Record<string, Value> };
  summary: Record<string, unknown>;
  requests: {
    index: number;
    text?: string;
    status: string;
    error?: string;
    metrics?: { output_tokens?: number; runtime_e2e_ms?: number; proxy_first_content_ms?: number };
  }[];
};
const defaultPrompts =
  'Compute 17 + 25. Give only the number.\nExplain why memory bandwidth matters for LLM decoding in two sentences.\nContinue the story: A small fox found a key beside the old tree.\nList three practical ways to reduce inference latency.';

export default function NativeEngines() {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const { section: routeSection } = useParams();
  const section = routeSection ?? (params.has('run') ? 'records' : 'configure');
  const client = useQueryClient();
  const [engine, setEngine] = useState('vllm');
  const [source, setSource] = useState('');
  const [drafts, setDrafts] = useState<Record<string, Record<string, Value>>>({});
  const [prompts, setPrompts] = useState(defaultPrompts);
  const [concurrency, setConcurrency] = useState(4);
  const [rounds, setRounds] = useState(2);
  const [tokens, setTokens] = useState(32);
  const [rate, setRate] = useState(2);
  const [mode, setMode] = useState('closed_loop');
  const [fixed, setFixed] = useState(false);
  const [expected, setExpected] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [baseline, setBaseline] = useState('');
  const [candidate, setCandidate] = useState('');
  const [comparison, setComparison] = useState<NativeComparison>();
  const [declaredChanges, setDeclaredChanges] = useState<string[]>([]);
  const ds = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: NativeDeployment[] }>('/deployments'),
    refetchInterval: 2000,
  });
  const catalog = useQuery({
    queryKey: ['native-engines'],
    queryFn: () => api<{ items: Engine[] }>('/native-engines/catalog'),
  });
  const jobs = useQuery({
    queryKey: ['native-benchmarks'],
    queryFn: () => api<{ items: Run[] }>('/native-engines/benchmarks'),
    refetchInterval: 3000,
  });
  const overview = useQuery({
    queryKey: ['native-overview'],
    queryFn: () => api<{ pending: number }>('/overview'),
    refetchInterval: 2000,
  });
  const natives = ds.data?.items.filter((d) => ['vllm', 'sglang'].includes(d.provider)) ?? [];
  const selectedRun = useRun(params.get('run') ?? undefined);
  const id =
    params.get('deployment') ?? selectedRun.data?.config.deployment.id ?? natives[0]?.id ?? '';
  const selected = natives.find((d) => d.id === id);
  const definition = catalog.data?.items.find((e) => e.engine === selected?.provider);
  const defaults = Object.fromEntries(definition?.items.map((c) => [c.id, c.default]) ?? []);
  const options = { ...defaults, ...selected?.engine_options, ...drafts[id] };
  const receipt = selectedRun.data?.metrics.native_benchmark as Receipt | undefined;
  const pending = busy || !!overview.data?.pending;
  const resident = ds.data?.items.find(
    (d) =>
      ['pytorch', 'vllm', 'sglang'].includes(d.provider) &&
      !['unloaded', 'failed'].includes(d.state),
  );
  const choose = (key: string, value: string) =>
    setParams((p) => {
      const next = new URLSearchParams(p);
      next.set(key, value);
      return next;
    });
  const sectionHref = (next: string, overrides: Record<string, string> = {}) => {
    const query = new URLSearchParams(params);
    if (id) query.set('deployment', id);
    for (const [key, value] of Object.entries(overrides)) query.set(key, value);
    return `/engines/${next}?${query}`;
  };
  async function action(path: string, body: unknown) {
    setBusy(true);
    setError('');
    try {
      const result = await api<Run & { id: string }>(path, body);
      if (path.endsWith('/deployments')) choose('deployment', result.id);
      else if (path.endsWith('/retire')) setParams({});
      else if (path === '/native-engines/benchmarks')
        navigate(sectionHref('records', { run: result.id }));
      else choose('run', result.id);
      await client.invalidateQueries({ queryKey: ['deployments'] });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHead
        kicker="原生执行与公共基线"
        title="原生推理引擎"
        description="以同一份本地制品运行 vLLM / SGLang，按 HQSB 分层调优并保留实际执行证据。"
      />
      <QueryState loading={ds.isPending || catalog.isPending} error={ds.error ?? catalog.error} />
      {error && <Alert type="error" message={error} showIcon />}
      {section !== 'records' && selectedRun.data?.error && (
        <Alert type="error" message={selectedRun.data.error} showIcon />
      )}
      <nav aria-label="原生引擎工作区">
        <Space wrap>
          {[
            ['configure', '部署与参数'],
            ['benchmark', '公共压测'],
            ['records', '运行与对照'],
          ].map(([key, label]) => (
            <Link
              key={key}
              to={sectionHref(key)}
              aria-current={section === key ? 'page' : undefined}
            >
              {label}
            </Link>
          ))}
        </Space>
      </nav>
      <Space wrap style={{ margin: '16px 0' }}>
        <Select
          aria-label="原生部署"
          style={{ minWidth: 320 }}
          value={id || undefined}
          onChange={(v) => choose('deployment', v)}
          options={natives.map((d) => ({ value: d.id, label: `${d.name} · ${d.state}` }))}
        />
        {selected && <Tag>{selected.state}</Tag>}
      </Space>
      {section === 'configure' && (
        <>
          <Panel
            title="建立原生部署"
            subtitle="共享已下载的权重和 SHA256 清单；两套引擎使用独立运行环境。"
          >
            <Space wrap>
              <Select
                aria-label="源模型"
                style={{ minWidth: 280 }}
                value={source || undefined}
                placeholder="选择本地模型"
                onChange={setSource}
                options={ds.data?.items
                  .filter((d) => d.provider === 'pytorch')
                  .map((d) => ({ value: d.id, label: d.name }))}
              />
              <Select
                aria-label="引擎"
                value={engine}
                onChange={setEngine}
                options={catalog.data?.items.map((e) => ({
                  value: e.engine,
                  label: `${e.engine} ${e.version}${e.installed ? '' : ' · 待安装'}`,
                }))}
              />
              <Button
                disabled={pending || !source}
                disabledReason={{
                  reason: pending ? '请等待当前任务完成。' : '请先选择带有制品清单的本地模型。',
                  next: { to: '/models/local', label: '本地模型管理' },
                }}
                onClick={() =>
                  void action('/native-engines/deployments', {
                    source_deployment_id: source,
                    engine,
                  })
                }
              >
                建立部署
              </Button>
            </Space>
          </Panel>
          <Panel
            title="分层参数与生命周期"
            subtitle="参数在加载时应用。切换实现前先卸载当前模型；引擎原始计数器按实例关联保存。"
          >
            <Space wrap>
              <Button
                disabled={
                  pending ||
                  !selected ||
                  !['unloaded', 'failed'].includes(selected.state) ||
                  !!resident ||
                  !definition?.installed
                }
                disabledReason={{
                  reason: pending
                    ? '请等待当前任务完成。'
                    : !definition?.installed
                      ? '请先在执行机器安装当前引擎的固定版本环境。'
                      : resident
                        ? `请先卸载 ${resident.name}。`
                        : '请先建立并选择一个原生部署。',
                  next: { to: '/devices', label: '设备与部署' },
                }}
                onClick={() => void action(`/deployments/${id}/load`, { engine_options: options })}
              >
                按当前参数加载
              </Button>
              <Button
                disabled={pending || selected?.state !== 'ready'}
                disabledReason={{
                  reason: '请等待当前任务完成，并选择处于就绪状态的部署。',
                  next: { to: '/devices', label: '设备与部署' },
                }}
                onClick={() => void action(`/deployments/${id}/unload`, {})}
              >
                卸载
              </Button>
              <Button
                disabled={pending || !selected || !['unloaded', 'failed'].includes(selected.state)}
                disabledReason={{
                  reason: '请先卸载所选原生部署，并等待当前任务完成。',
                  next: { to: '/devices', label: '设备与部署' },
                }}
                onClick={() => void action(`/native-engines/deployments/${id}/retire`, {})}
              >
                移除部署引用
              </Button>
            </Space>
            <Table<Control>
              rowKey="id"
              size="small"
              pagination={false}
              dataSource={(definition?.items ?? []).filter(
                (c) => !params.get('layer') || c.layers.includes(Number(params.get('layer'))),
              )}
              columns={[
                {
                  title: '层级',
                  dataIndex: 'layers',
                  render: (layers: number[]) => layers.map((l) => `L${l}`).join(' / '),
                },
                { title: '功能', dataIndex: 'label' },
                { title: '参数', dataIndex: 'id' },
                {
                  title: '配置值',
                  render: (_, c) => {
                    const value = options[c.id];
                    const update = (v: Value) =>
                      setDrafts((old) => ({ ...old, [id]: { ...old[id], [c.id]: v } }));
                    return c.values ? (
                      <Select
                        aria-label={c.id}
                        value={value as string | number}
                        onChange={update}
                        options={c.values.map((v) => ({ value: v, label: String(v) }))}
                        style={{ minWidth: 160 }}
                      />
                    ) : typeof c.default === 'boolean' ? (
                      <Checkbox
                        aria-label={c.id}
                        checked={!!value}
                        onChange={(e) => update(e.target.checked)}
                      />
                    ) : typeof c.default === 'number' ? (
                      <InputNumber
                        aria-label={c.id}
                        min={c.minimum}
                        max={c.maximum}
                        value={value as number}
                        onChange={(v) => v != null && update(v)}
                      />
                    ) : (
                      <Input
                        aria-label={c.id}
                        value={String(value)}
                        onChange={(e) => update(e.target.value)}
                        style={{ maxWidth: 200 }}
                      />
                    );
                  },
                },
              ]}
            />
            <p>
              Attention：vLLM 可选 auto / FLASH_ATTN / FLASHINFER / TRITON_ATTN；SGLang 可选 auto /
              triton / flashinfer /
              torch_native。具体模型与硬件能力在加载时验证。量化加载需要匹配的制品格式。
            </p>
            {selected?.detail.native_engine_receipt != null && (
              <details>
                <summary>实际生效凭据</summary>
                <JsonView value={selected.detail.native_engine_receipt} />
              </details>
            )}
          </Panel>
        </>
      )}
      {section === 'benchmark' && (
        <Panel
          title="公共负载压测"
          subtitle="每行一个固定输入，直接并发访问原生服务；输出和逐请求指标进入运行记录。"
        >
          <Input.TextArea
            aria-label="固定输入"
            value={prompts}
            onChange={(e) => setPrompts(e.target.value)}
            rows={4}
          />
          <Space wrap>
            <label>
              并发{' '}
              <InputNumber
                aria-label="并发"
                min={1}
                max={64}
                value={concurrency}
                onChange={(v) => v && setConcurrency(v)}
              />
            </label>
            <label>
              轮数{' '}
              <InputNumber min={1} max={128} value={rounds} onChange={(v) => v && setRounds(v)} />
            </label>
            <label>
              最大输出 Token{' '}
              <InputNumber min={1} max={4096} value={tokens} onChange={(v) => v && setTokens(v)} />
            </label>
            <Select
              value={mode}
              onChange={setMode}
              options={[
                { value: 'closed_loop', label: '固定并发' },
                { value: 'open_loop', label: '固定到达率' },
              ]}
            />
            {mode === 'open_loop' && (
              <label>
                请求/s <InputNumber min={0.01} value={rate} onChange={(v) => v && setRate(v)} />
              </label>
            )}
            <Checkbox checked={fixed} onChange={(e) => setFixed(e.target.checked)}>
              固定输出长度（性能负载）
            </Checkbox>
          </Space>
          <Input.TextArea
            aria-label="参考输出"
            placeholder="可选：逐行填写精确参考输出，用于 exact-match 质量检查；留空时保存输出，质量状态为待评测。"
            value={expected}
            onChange={(e) => setExpected(e.target.value)}
            rows={2}
          />
          <Button
            type="primary"
            disabled={pending || selected?.state !== 'ready' || !prompts.trim()}
            disabledReason={{
              reason: pending ? '请等待当前任务完成。' : '请先加载所选原生部署，并填写固定输入。',
              next: { to: sectionHref('configure'), label: '原生引擎配置' },
            }}
            onClick={() =>
              void action('/native-engines/benchmarks', {
                deployment_id: id,
                expected_epoch: selected?.epoch,
                prompts: prompts.split('\n').filter((p) => p.trim()),
                expected_outputs: expected.trim() ? expected.split('\n') : null,
                concurrency,
                rounds,
                max_output_tokens: tokens,
                arrival_mode: mode,
                request_rate: rate,
                fixed_output_length: fixed,
              })
            }
          >
            运行公共负载
          </Button>
        </Panel>
      )}
      {section === 'records' && (
        <Panel
          title="运行与对照"
          subtitle="相同工作负载哈希、模型制品、生成语义及采集模式用于受控比较；不同缓存状态单独解释。"
        >
          <Space wrap>
            <Select
              aria-label="基线运行"
              showSearch
              optionFilterProp="label"
              placeholder="基线运行"
              style={{ minWidth: 260 }}
              value={baseline || undefined}
              onChange={setBaseline}
              options={jobs.data?.items.map((r) => ({ value: r.id, label: r.id }))}
            />
            <Select
              aria-label="候选运行"
              showSearch
              optionFilterProp="label"
              placeholder="候选运行"
              style={{ minWidth: 260 }}
              value={candidate || undefined}
              onChange={setCandidate}
              options={jobs.data?.items.map((r) => ({ value: r.id, label: r.id }))}
            />
            <Button
              disabled={busy || !baseline || !candidate || baseline === candidate}
              disabledReason={{ reason: '请选择两个不同的公共负载运行，作为基线与候选。' }}
              onClick={() => {
                setBusy(true);
                setError('');
                void api<NativeComparison>('/native-engines/comparisons', {
                  baseline_run_id: baseline,
                  candidate_run_id: candidate,
                  declared_changes: declaredChanges,
                })
                  .then(setComparison)
                  .catch((e) => setError((e as Error).message))
                  .finally(() => setBusy(false));
              }}
            >
              检查可比性与差异
            </Button>
          </Space>
          <Select
            mode="multiple"
            showSearch
            optionFilterProp="label"
            aria-label="声明的配置变化"
            placeholder="同引擎调优时，声明要比较的配置变化"
            value={declaredChanges}
            onChange={setDeclaredChanges}
            style={{ minWidth: 360, marginTop: 12 }}
            options={[
              ...new Map(
                catalog.data?.items.flatMap((e) =>
                  e.items.map((c) => [c.id, { value: c.id, label: c.label }] as const),
                ) ?? [],
              ).values(),
            ]}
          />
          {comparison != null && (
            <>
              <Alert
                showIcon
                type={comparison.quality_gate.status === 'failed' ? 'error' : 'info'}
                message={
                  !comparison.comparable
                    ? '当前配置存在待核对差异，请查看可比性详情。'
                    : comparison.quality_gate.status === 'failed'
                      ? '该候选的固定样例质量发生回退，保留性能和机制差异供分析。'
                      : comparison.quality_gate.status === 'passed'
                        ? '固定样例质量门通过。'
                        : '已记录性能差异，输出等待任务评分。'
                }
              />
              <JsonView value={comparison} />
            </>
          )}
          <Table<Run>
            rowKey="id"
            size="small"
            dataSource={jobs.data?.items ?? []}
            pagination={{ pageSize: 8 }}
            columns={[
              {
                title: '运行',
                render: (_, r) => <a onClick={() => choose('run', r.id)}>{r.id}</a>,
              },
              { title: '状态', dataIndex: 'state' },
              {
                title: '部署',
                render: (_, r) => String((r.config.deployment as { name?: string })?.name ?? ''),
              },
              {
                title: '输出 token/s',
                render: (_, r) => fmt(r.metrics.output_tokens_per_s as number | undefined),
              },
            ]}
          />
          {selectedRun.data && (
            <>
              <Button
                disabled={
                  !receipt || !natives.some((d) => d.id === selectedRun.data?.config.deployment.id)
                }
                disabledReason={{ reason: '请选择一条对应部署仍在当前节点登记的原生压测记录。' }}
                onClick={() => {
                  const deploymentId = selectedRun.data!.config.deployment.id;
                  setDrafts((old) => ({ ...old, [deploymentId]: receipt!.engine.requested }));
                  navigate(sectionHref('configure', { deployment: deploymentId }));
                }}
              >
                复制为待加载参数
              </Button>
              <p>
                {selectedRun.data.state} {selectedRun.data.error}{' '}
                <Link to={`/runs/${selectedRun.data.id}`}>完整运行记录</Link>
                {' · '}<Link to={`/diagnostics?source=${selectedRun.data.id}`}>诊断此方案（nsys / ncu）</Link>
              </p>
              {!terminal(selectedRun.data.state) && (
                <Button onClick={() => void action(`/requests/${selectedRun.data!.id}/cancel`, {})}>
                  取消当前运行
                </Button>
              )}
              {receipt && (
                <>
                  <JsonView value={receipt.summary} />
                  <Table
                    rowKey="index"
                    dataSource={receipt.requests}
                    pagination={{ pageSize: 8 }}
                    columns={[
                      { title: '请求', dataIndex: 'index' },
                      { title: '状态', dataIndex: 'status' },
                      { title: '输出', render: (_, r) => r.text ?? r.error },
                      { title: 'Token', render: (_, r) => r.metrics?.output_tokens },
                      {
                        title: '首内容 ms',
                        render: (_, r) => fmt(r.metrics?.proxy_first_content_ms),
                      },
                      { title: '端到端 ms', render: (_, r) => fmt(r.metrics?.runtime_e2e_ms) },
                    ]}
                  />
                  <details>
                    <summary>配置、工作负载哈希与原始观测</summary>
                    <JsonView value={receipt} />
                  </details>
                </>
              )}
            </>
          )}
        </Panel>
      )}
    </>
  );
}
