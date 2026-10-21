import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Input, InputNumber, Select, Space, Table, Tag } from 'antd';
import { Link, useSearchParams } from 'react-router-dom';
import Button from '../components/ActionButton';
import { api, base, fmt, terminal } from '../api/client';
import type { Deployment, Run } from '../api/types';
import { JsonView, PageHead, Panel, QueryState } from '../components';
import { useRun } from '../hooks';
import ERETracePanel, { type ERETraceData } from './ERETracePanel';
import DMATPanel, { type DMATData, type DMATLaunch } from './DMATPanel';
import DMATChart from './DMATChart';

type Event = {
  name: string;
  kind: string;
  start_ns: number;
  end_ns: number;
  device: string;
  stream: number;
};
type Metric = { name: string; value: number | null; unit: string; raw: string };
type Kernel = {
  name: string;
  grid: string;
  block: string;
  launch_id: string;
  process_id: string;
  metrics: Metric[];
};
type PanelData = ERETraceData &
  DMATData & {
    counts?: Record<string, number>;
    top_kernels?: { name: string; launches: number; summed_duration_ns: number }[];
    device_activity?: Record<string, unknown>[];
    copies?: Record<string, unknown>[];
    cuda_apis?: Record<string, unknown>[];
    timeline?: Event[];
    timeline_truncated?: boolean;
    kernels?: Kernel[];
    metric_rows?: number;
  };
type Receipt = {
  tool: string;
  panel?: PanelData;
  replay_outputs_match_source?: boolean | null;
  replay_outputs_complete?: boolean;
  replay_workloads_match_source?: boolean;
  source_run_id: string;
  error?: string;
};
type Tool = {
  tool: string;
  installed: boolean;
  version?: string;
  reason?: string;
  collection_permission?: string;
  capabilities?: string[];
};
type Comparison = {
  comparable: boolean;
  mismatches: string[];
  configuration_changes: unknown;
  candidate_run_id: string;
  dmat?: {
    comparable: boolean;
    objects: {
      launch: number;
      role: string;
      baseline_events: number;
      candidate_events: number;
      delta_events: number;
      baseline_instruction_bytes: number;
      candidate_instruction_bytes: number;
    }[];
    pairs: {
      launch: number;
      baseline: DMATLaunch['plot'];
      candidate: DMATLaunch['plot'];
      aligned_density_l1: number;
    }[];
  };
  kernel_changes?: {
    name: string;
    baseline_launches: number | null;
    candidate_launches: number | null;
    baseline_summed_ms: number | null;
    candidate_summed_ms: number | null;
  }[];
  metric_changes?: {
    name: string;
    metric: string;
    baseline: number | null;
    candidate: number | null;
    unit: string;
    delta: number | null;
    occurrence: number;
  }[];
};

function Timeline({ events }: { events: Event[] }) {
  if (!events.length) return <p>当前窗口等待 GPU 事件。</p>;
  const lanes = [...new Set(events.map((e) => `${e.device}:${e.stream}`))].slice(0, 12);
  const start = Math.min(...events.map((e) => e.start_ns));
  const end = Math.max(...events.map((e) => e.end_ns));
  const span = Math.max(1, end - start);
  return (
    <div style={{ overflowX: 'auto' }}>
      <svg
        role="img"
        aria-label="GPU 执行时间线"
        viewBox={`0 0 1000 ${lanes.length * 28 + 30}`}
        style={{ minWidth: 720, width: '100%' }}
      >
        {lanes.map((lane, index) => (
          <g key={lane}>
            <text x={0} y={index * 28 + 18} fontSize={11} fill="currentColor">
              GPU:stream {lane}
            </text>
            <line
              x1={135}
              x2={995}
              y1={index * 28 + 25}
              y2={index * 28 + 25}
              stroke="currentColor"
              opacity={0.1}
            />
            {events
              .filter((e) => `${e.device}:${e.stream}` === lane)
              .map((e, i) => (
                <rect
                  key={i}
                  x={135 + ((e.start_ns - start) / span) * 860}
                  y={index * 28 + 5}
                  width={Math.max(0.6, ((e.end_ns - e.start_ns) / span) * 860)}
                  height={17}
                  fill={e.kind === 'kernel' ? '#467fba' : '#b58043'}
                  opacity={0.85}
                >
                  <title>
                    {e.name} · {fmt((e.end_ns - e.start_ns) / 1e6, 4)} ms
                  </title>
                </rect>
              ))}
          </g>
        ))}
        <text x={135} y={lanes.length * 28 + 18} fontSize={11} fill="currentColor">
          首个 GPU 事件 → {fmt(span / 1e6, 3)} ms；蓝：Kernel，棕：拷贝/填充
        </text>
      </svg>
    </div>
  );
}

export default function Diagnostics() {
  const [params, setParams] = useSearchParams();
  const client = useQueryClient();
  const [tool, setTool] = useState('nsys');
  const [filter, setFilter] = useState('.*');
  const [skip, setSkip] = useState(0);
  const [count, setCount] = useState(3);
  const [sections, setSections] = useState('balanced');
  const [ereKernel, setEreKernel] = useState('_fwd_kernel');
  const [ereCodec, setEreCodec] = useState('reference');
  const [ereVerify, setEreVerify] = useState(false);
  const [ereMode, setEreMode] = useState('warp');
  const [memoryCapacity, setMemoryCapacity] = useState(256);
  const [memoryClock, setMemoryClock] = useState('thread_entry_clock64');
  const [memoryPageBytes, setMemoryPageBytes] = useState(65536);
  const [memoryTimeBins, setMemoryTimeBins] = useState(128);
  const [kernel, setKernel] = useState(0);
  const [search, setSearch] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [baseline, setBaseline] = useState('');
  const [comparison, setComparison] = useState<Comparison>();
  const sources = useQuery({
    queryKey: ['diagnostic-sources'],
    queryFn: () => api<{ items: Run[] }>('/diagnostics/sources'),
  });
  const jobs = useQuery({
    queryKey: ['diagnostics'],
    queryFn: () => api<{ items: Run[] }>('/diagnostics/jobs'),
    refetchInterval: 2500,
  });
  const catalog = useQuery({
    queryKey: ['diagnostic-tools'],
    queryFn: () => api<{ items: Tool[] }>('/diagnostics/catalog'),
  });
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 2500,
  });
  const overview = useQuery({
    queryKey: ['diagnostic-overview'],
    queryFn: () => api<{ pending: number }>('/overview'),
    refetchInterval: 2500,
  });
  const source = params.get('source') ?? '';
  const selected = useRun(params.get('run') ?? undefined);
  const receipt = selected.data?.metrics.diagnostic as Receipt | undefined;
  const panel = receipt?.panel;
  const resident = deployments.data?.items.find(
    (d) =>
      ['pytorch', 'vllm', 'sglang'].includes(d.provider) &&
      !['unloaded', 'failed'].includes(d.state),
  );
  const installed = catalog.data?.items.find((t) => t.tool === tool)?.installed;
  const sourceRecord = sources.data?.items.find((r) => r.id === source);
  const pytorchSource = sourceRecord?.kind === 'generate';
  const tritonAttentionSource = String(
    (sourceRecord?.config.actual_runtime as { kernel_plan?: { profile_id?: string } } | undefined)
      ?.kernel_plan?.profile_id ?? '',
  ).startsWith('qwen3-attention-triton-');
  const sourceReceipt = sourceRecord?.metrics.native_benchmark as
    | {
        engine: {
          requested: {
            enforce_eager: boolean;
            tensor_parallel_size: number;
            pipeline_parallel_size: number;
          };
        };
        plan: { concurrency: number };
      }
    | undefined;
  const scopeReason = pytorchSource
    ? tool !== 'eretrace' || ereMode !== 'memory'
      ? '此 PyTorch 推理记录可用于 ERETrace 的“真实访存 / DMAT”模式。'
      : !sourceRecord?.config.save_input &&
          !sourceRecord?.config.messages &&
          !sourceRecord?.config.prompt
        ? '请在推理页启用保存输入，完成一次推理后选择新记录。'
        : ''
    : tool === 'eretrace' && sourceReceipt
      ? !sourceReceipt.engine.requested.enforce_eager
        ? '请在原生引擎配置中启用 Eager，重新评测后选择新记录。'
        : sourceReceipt.engine.requested.tensor_parallel_size !== 1 ||
            sourceReceipt.engine.requested.pipeline_parallel_size !== 1
          ? '请使用 TP=1、PP=1 的原生评测记录。'
          : sourceReceipt.plan.concurrency !== 1
            ? '请在原生引擎公共评测中将请求并发设为 1，完成评测后选择新记录。'
            : ''
      : '';
  const scopePage =
    sourceReceipt?.engine.requested.enforce_eager &&
    sourceReceipt.engine.requested.tensor_parallel_size === 1 &&
    sourceReceipt.engine.requested.pipeline_parallel_size === 1
      ? 'benchmark'
      : 'configure';
  const memoryRuntimeMissing =
    tool === 'eretrace' &&
    ereMode === 'memory' &&
    !catalog.data?.items
      .find((t) => t.tool === 'eretrace')
      ?.capabilities?.includes('dmat-dual-clock');
  const disabled =
    busy ||
    !!overview.data?.pending ||
    !!resident ||
    !source ||
    !installed ||
    !!scopeReason ||
    memoryRuntimeMissing;
  const reason = resident
    ? `请先卸载 ${resident.name}，诊断任务将独立加载源运行的模型与参数。`
    : overview.data?.pending
      ? '请等待当前任务完成，或在运行记录中取消。'
      : !source
        ? '请选择一条已完成的原生公共压测记录。'
        : !installed
          ? '请在执行节点安装所选工具，并在服务配置中设置路径。'
          : scopeReason ||
            (memoryRuntimeMissing
              ? '请在执行节点构建包含 memory 能力的 ERETrace runtime，并更新服务配置。'
              : '诊断任务正在提交。');
  function choose(key: string, value: string) {
    setParams((p) => {
      const q = new URLSearchParams(p);
      q.set(key, value);
      return q;
    });
  }
  async function submit() {
    setBusy(true);
    setError('');
    try {
      const run = await api<Run>('/diagnostics/jobs', {
        source_run_id: source,
        tool,
        kernel_filter: filter,
        launch_skip: skip,
        launch_count: count,
        sections,
        ere_kernel: ereKernel,
        ere_codec: ereCodec,
        ere_verify: ereVerify,
        ere_mode: ereMode,
        memory_capacity: memoryCapacity,
        memory_page_bytes: memoryPageBytes,
        memory_time_bins: memoryTimeBins,
        memory_clock: memoryClock,
      });
      choose('run', run.id);
      await client.invalidateQueries({ queryKey: ['diagnostics'] });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const kernels = panel?.kernels ?? [];
  const activeKernel = kernels[Math.min(kernel, kernels.length - 1)];
  return (
    <>
      <PageHead
        kicker="Nsight / ERETrace"
        title="执行诊断"
        description="复现已记录的推理负载，关联 GPU 时间线、硬件计数器、真实访存图和执行凭据。"
      />
      <QueryState
        loading={catalog.isPending}
        error={catalog.error ?? jobs.error ?? sources.error}
      />
      {error && <Alert type="error" message={error} showIcon />}
      <Panel
        title="从评测创建诊断"
        subtitle="模型加载和预热后开启采集窗口。诊断实例独占 GPU，完成后释放模型；正式性能比较使用原公共压测结果。"
      >
        <Space wrap>
          <Select
            aria-label="诊断源运行"
            placeholder="已完成的公共压测 / PyTorch 推理"
            showSearch
            optionFilterProp="label"
            style={{ minWidth: 370 }}
            value={source || undefined}
            onChange={(v) => {
              choose('source', v);
              if (sources.data?.items.find((r) => r.id === v)?.kind === 'generate') {
                setTool('eretrace');
                setEreMode('memory');
                setEreCodec('raw');
                setEreVerify(false);
                setCount(1);
                const chosen = sources.data?.items.find((r) => r.id === v);
                const profile = (
                  chosen?.config.actual_runtime as
                    | { kernel_plan?: { profile_id?: string } }
                    | undefined
                )?.kernel_plan?.profile_id;
                if (profile?.startsWith('qwen3-attention-triton-')) {
                  setEreKernel('hqsb_dmat_attention');
                  setSkip(0);
                } else if (ereKernel === '_fwd_kernel') setEreKernel('vectorized_elementwise');
              }
            }}
            options={sources.data?.items
              .filter((r) => r.state === 'completed')
              .map((r) => ({ value: r.id, label: `${r.config.deployment.name} · ${r.id}` }))}
          />
          <Select
            aria-label="诊断工具"
            value={tool}
            onChange={setTool}
            options={[
              { value: 'nsys', label: 'nsys · 系统时间线' },
              { value: 'ncu', label: 'ncu · Kernel 计数器' },
              { value: 'eretrace', label: 'ERETrace · warp / SM 观测' },
            ]}
          />
          <Button
            type="primary"
            disabled={disabled}
            disabledReason={{
              reason,
              next: {
                to: resident
                  ? resident.provider === 'pytorch'
                    ? `/devices?deployment=${resident.id}`
                    : `/engines/configure?deployment=${resident.id}`
                  : overview.data?.pending
                    ? '/runs'
                    : scopeReason
                      ? pytorchSource
                        ? `/models?deployment=${sourceRecord?.config.deployment.id ?? ''}`
                        : `/engines/${scopePage}?deployment=${sourceRecord?.config.deployment.id ?? ''}`
                      : '/engines/records',
                label: resident
                  ? '卸载驻留部署'
                  : overview.data?.pending
                    ? '查看当前任务'
                    : scopeReason
                      ? '配置并重新评测'
                      : '查看原生评测',
              },
            }}
            onClick={() => void submit()}
          >
            开始独立诊断
          </Button>
        </Space>
        {tool === 'ncu' && (
          <Space wrap style={{ marginTop: 12 }}>
            <label>
              Kernel 正则{' '}
              <Input
                aria-label="Kernel 过滤"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                style={{ width: 260 }}
              />
            </label>
            <label>
              跳过匹配启动{' '}
              <InputNumber min={0} max={100000} value={skip} onChange={(v) => setSkip(v ?? 0)} />
            </label>
            <label>
              采集启动数{' '}
              <InputNumber min={1} max={32} value={count} onChange={(v) => setCount(v ?? 3)} />
            </label>
            <Select
              aria-label="计数器组"
              value={sections}
              onChange={setSections}
              options={[
                { value: 'balanced', label: '计算、访存与 occupancy' },
                { value: 'memory', label: '访存' },
                { value: 'compute', label: '计算与 occupancy' },
              ]}
            />
          </Space>
        )}
        {tool === 'eretrace' && (
          <>
            <Space wrap style={{ marginTop: 12 }}>
              <Select
                aria-label="ERETrace 观测模式"
                value={ereMode}
                onChange={(value) => {
                  setEreMode(value);
                  if (value === 'memory') {
                    setEreCodec('raw');
                    setEreVerify(false);
                    setCount(1);
                    if (tritonAttentionSource) setEreKernel('hqsb_dmat_attention');
                  }
                }}
                options={[
                  { value: 'warp', label: 'warp / SM 执行观测' },
                  { value: 'memory', label: '真实访存 / DMAT' },
                ]}
              />
              <label>
                Kernel 名称包含{' '}
                <Input
                  aria-label="ERETrace Kernel 名称"
                  value={ereKernel}
                  onChange={(e) => setEreKernel(e.target.value)}
                  style={{ width: 260 }}
                />
              </label>
              <label>
                跳过匹配启动{' '}
                <InputNumber
                  aria-label="ERETrace 跳过启动"
                  min={0}
                  max={100000}
                  value={skip}
                  onChange={(v) => setSkip(v ?? 0)}
                />
              </label>
              <label>
                最多尝试启动{' '}
                <InputNumber
                  aria-label="ERETrace 启动数"
                  min={1}
                  max={32}
                  value={count}
                  onChange={(v) => setCount(v ?? 3)}
                />
              </label>
              {ereMode === 'warp' && (
                <Select
                  aria-label="ERETrace 编码"
                  value={ereCodec}
                  onChange={setEreCodec}
                  options={[
                    { value: 'reference', label: 'SM / CTA 参考编码' },
                    { value: 'raw', label: '原始记录对照' },
                  ]}
                />
              )}
              {ereMode === 'warp' && (
                <Select
                  aria-label="ERETrace 精确校验"
                  value={ereVerify ? 1 : 0}
                  onChange={(v) => setEreVerify(v === 1)}
                  options={[
                    { value: 0, label: '常规采集' },
                    { value: 1, label: '同次执行原始记录校验（增加采集开销）' },
                  ]}
                />
              )}
              {ereMode === 'memory' && (
                <>
                  {tritonAttentionSource && (
                    <Select
                      aria-label="Attention 观测位置"
                      placeholder="选择 Attention 阶段"
                      onChange={(v) => {
                        setSkip(v);
                        setCount(1);
                        setEreKernel('hqsb_dmat_attention');
                      }}
                      options={[
                        { value: 0, label: 'Qwen3-1.7B · 第 1 层 Prefill' },
                        { value: 28, label: 'Qwen3-1.7B · 第 1 层 Decode' },
                      ]}
                    />
                  )}
                  <Select
                    aria-label="DMAT 采集时间口径"
                    value={memoryClock}
                    onChange={setMemoryClock}
                    options={[
                      { value: 'thread_entry_clock64', label: '线程入口归一化 · 论文图 1 / 11' },
                      { value: 'GPU_globaltimer', label: '全局时间 · 论文图 15' },
                    ]}
                  />
                  <label>
                    每线程事件容量{' '}
                    <InputNumber
                      aria-label="DMAT 每线程容量"
                      min={1}
                      max={2048}
                      value={memoryCapacity}
                      onChange={(v) => setMemoryCapacity(v ?? 256)}
                    />
                  </label>
                  <Select
                    aria-label="DMAT 地址粒度"
                    value={memoryPageBytes}
                    onChange={setMemoryPageBytes}
                    options={[4096, 16384, 65536].map((value) => ({
                      value,
                      label: `${value / 1024} KiB 地址区间`,
                    }))}
                  />
                  <Select
                    aria-label="DMAT 时间分桶"
                    value={memoryTimeBins}
                    onChange={setMemoryTimeBins}
                    options={[32, 64, 128, 256, 512].map((value) => ({
                      value,
                      label: `${value} 个时间区间`,
                    }))}
                  />
                </>
              )}
            </Space>
            <p>
              {pytorchSource
                ? 'PyTorch 记录按保存的输入、模型和算子配置重放。Kernel 名称按子串匹配，例如 vectorized_elementwise；采集具备受支持 PTX 的普通启动。'
                : '使用 Eager、TP=1、PP=1、请求并发=1 的源评测。名称按子串匹配，例如 _fwd_kernel；具备 PTX 的 Triton 普通 CUDA 启动进入采集。'}
              {ereMode === 'warp'
                ? ' warp 记录使用 SM 本地时钟周期。'
                : ' 访存事件同时保存线程入口时钟和 GPU 全局时间，可按分析目的切换。'}
            </p>
            {ereMode === 'memory' && (
              <p>
                采集 PTX 全局读写和经典 cp.async，保留真实地址、双时钟、指令位置及访问宽度。
                每次启动的原始缓冲区上限为 256
                MiB；线程事件容量溢出会保留已取得的数据并报告丢失数量。 使用 Qwen3 Triton Attention
                方案可关联 Q/K/V/输出、层号和阶段。建议先采集一个启动，再对照相同位置的其他方案。
              </p>
            )}
          </>
        )}
        <p>
          nsys：CPU/CUDA API、Kernel 与拷贝时间线。ncu：选定 Kernel 的硬件指标；application
          replay、cache-control=none、clock-control=none。ERETrace：选定 Kernel 内的 warp 时间、SM
          分布和真实访存图。
        </p>
        <Space wrap>
          {catalog.data?.items.map((t) => (
            <Tag key={t.tool} title={t.version ?? t.reason}>
              {t.tool} ·{' '}
              {t.collection_permission === 'permission_required'
                ? '需要宿主机开放硬件计数器权限'
                : t.installed
                  ? '工具已发现，采集时检查权限'
                  : t.reason}
            </Tag>
          ))}
        </Space>
      </Panel>
      <Panel title="诊断记录">
        <Table<Run>
          size="small"
          rowKey="id"
          dataSource={jobs.data?.items ?? []}
          pagination={{ pageSize: 8 }}
          columns={[
            {
              title: '诊断运行',
              render: (_, r) => (
                <a
                  onClick={() => {
                    choose('run', r.id);
                    setKernel(0);
                  }}
                >
                  {r.id}
                </a>
              ),
            },
            {
              title: '工具',
              render: (_, r) => String((r.config.diagnostic_plan as { tool: string })?.tool),
            },
            { title: '模型', render: (_, r) => r.config.deployment.name },
            { title: '状态', dataIndex: 'state' },
          ]}
        />
        {selected.data && (
          <>
            <Space wrap>
              <Tag>{selected.data.state}</Tag>
              <Link to={`/runs/${selected.data.id}`}>完整运行记录</Link>
              <Link to={`/runs/${receipt?.source_run_id ?? selected.data.config.source_run_id}`}>
                源性能与质量记录
              </Link>
              {!terminal(selected.data.state) && (
                <Button
                  onClick={() =>
                    void api(`/requests/${selected.data!.id}/cancel`, {}).catch((e) =>
                      setError(e.message),
                    )
                  }
                >
                  取消诊断
                </Button>
              )}
            </Space>
            {selected.data.error && <Alert type="error" message={selected.data.error} showIcon />}
            {selected.data.progress != null && <JsonView value={selected.data.progress} />}
            {receipt?.panel && (
              <>
                <Space wrap style={{ marginTop: 12 }}>
                  <Select
                    aria-label="诊断对照基线"
                    showSearch
                    optionFilterProp="label"
                    placeholder="选择另一条同工具诊断记录"
                    style={{ minWidth: 350 }}
                    value={baseline || undefined}
                    onChange={setBaseline}
                    options={jobs.data?.items
                      .filter(
                        (r) =>
                          r.id !== selected.data?.id &&
                          r.state === 'completed' &&
                          r.config.diagnostic_plan?.tool === receipt.tool,
                      )
                      .map((r) => ({
                        value: r.id,
                        label: `${r.config.deployment.name} · ${r.id}`,
                      }))}
                  />
                  <Button
                    disabled={busy || !baseline}
                    loading={busy}
                    disabledReason={{
                      reason: busy
                        ? '正在对齐两份原始记录的时间、张量地址和密度尺度。'
                        : '请选择另一条已完成、使用同一工具的诊断记录。',
                    }}
                    onClick={() => {
                      setBusy(true);
                      setError('');
                      void api<Comparison>('/diagnostics/comparisons', {
                        baseline_run_id: baseline,
                        candidate_run_id: selected.data!.id,
                      })
                        .then(setComparison)
                        .catch((e) => setError(e.message))
                        .finally(() => setBusy(false));
                    }}
                  >
                    比较机制变化
                  </Button>
                </Space>
                {comparison?.candidate_run_id === selected.data.id && (
                  <>
                    <Alert
                      type="info"
                      message={
                        comparison.comparable
                          ? '采集口径一致，以下展示诊断窗口中的机制变化。'
                          : `请核对采集差异：${comparison.mismatches.join('、')}`
                      }
                    />
                    <details>
                      <summary>引擎参数变化</summary>
                      <JsonView value={comparison.configuration_changes} />
                    </details>
                    {comparison.kernel_changes && (
                      <Table
                        size="small"
                        rowKey="name"
                        dataSource={comparison.kernel_changes}
                        pagination={{ pageSize: 8 }}
                        columns={[
                          { title: 'Kernel', dataIndex: 'name', ellipsis: true },
                          { title: '基线启动', dataIndex: 'baseline_launches' },
                          { title: '候选启动', dataIndex: 'candidate_launches' },
                          { title: '基线累计 ms', render: (_, r) => fmt(r.baseline_summed_ms, 4) },
                          { title: '候选累计 ms', render: (_, r) => fmt(r.candidate_summed_ms, 4) },
                        ]}
                      />
                    )}
                    {comparison.metric_changes && (
                      <Table
                        size="small"
                        rowKey={(r) => `${r.name}-${r.metric}-${r.occurrence}`}
                        dataSource={comparison.metric_changes}
                        pagination={{ pageSize: 10 }}
                        columns={[
                          { title: '指标', dataIndex: 'metric' },
                          { title: '基线', render: (_, r) => fmt(r.baseline, 4) },
                          { title: '候选', render: (_, r) => fmt(r.candidate, 4) },
                          { title: '变化量', render: (_, r) => fmt(r.delta, 4) },
                          { title: '单位', dataIndex: 'unit' },
                        ]}
                      />
                    )}
                    {comparison.dmat?.comparable && panel?.memory_launches?.[0] && (
                      <section aria-label="DMAT 机制对照">
                        <h3>访存结构对照 · 统一时间和密度尺度</h3>
                        <p>
                          按张量角色、相对页偏移对齐。两图使用共同时间分箱及合并密度分位数，显示采集扰动下的结构变化。
                          时间分布也会受到调度和探针影响，请结合访问计数及重复采集解释。
                        </p>
                        <Table
                          size="small"
                          rowKey={(r) => `${r.launch}-${r.role}`}
                          dataSource={comparison.dmat.objects}
                          pagination={{ pageSize: 8 }}
                          columns={[
                            { title: '启动', dataIndex: 'launch' },
                            { title: '张量', dataIndex: 'role' },
                            { title: '基线事件', dataIndex: 'baseline_events' },
                            { title: '候选事件', dataIndex: 'candidate_events' },
                            { title: '事件变化', dataIndex: 'delta_events' },
                            { title: '基线指令字节', dataIndex: 'baseline_instruction_bytes' },
                            { title: '候选指令字节', dataIndex: 'candidate_instruction_bytes' },
                          ]}
                        />
                        {comparison.dmat.pairs.map((pair) => (
                          <div
                            key={pair.launch}
                            style={{
                              display: 'grid',
                              gridTemplateColumns:
                                'repeat(auto-fit, minmax(min(100%, 520px), 1fr))',
                              gap: 16,
                            }}
                          >
                            <div>
                              <h4>启动 {pair.launch + 1} · 基线</h4>
                              <DMATChart
                                title={`启动 ${pair.launch + 1} · 基线`}
                                launch={{ ...panel.memory_launches![0], plot: pair.baseline }}
                              />
                            </div>
                            <div>
                              <h4>启动 {pair.launch + 1} · 候选</h4>
                              <DMATChart
                                title={`启动 ${pair.launch + 1} · 候选`}
                                launch={{ ...panel.memory_launches![0], plot: pair.candidate }}
                              />
                            </div>
                          </div>
                        ))}
                      </section>
                    )}
                  </>
                )}
              </>
            )}
            {receipt && (
              <p>
                复现负载：{receipt.replay_workloads_match_source ? '哈希一致' : '查看采集凭据'}
                ；输出与源运行：
                {selected.data.state !== 'completed' && !receipt.replay_outputs_complete
                  ? '本次输出证据待补充'
                  : receipt.replay_outputs_match_source === true
                    ? '一致'
                    : receipt.replay_outputs_match_source === false
                      ? '存在变化，查看每次 replay 的输出'
                      : '本次输出证据待补充'}
                。
              </p>
            )}
            {panel?.timeline && (
              <>
                <Timeline events={panel.timeline} />
                {panel.timeline_truncated && (
                  <p>页面展示前 5000 条 GPU 活动；完整事件保存在 SQLite 和原生报告中。</p>
                )}
                <Table
                  size="small"
                  rowKey="name"
                  dataSource={panel.top_kernels}
                  pagination={{ pageSize: 10 }}
                  columns={[
                    { title: 'Kernel', dataIndex: 'name', ellipsis: true },
                    { title: '启动数', dataIndex: 'launches' },
                    {
                      title: '累计执行 ms（含重叠）',
                      render: (_, r) => fmt(r.summed_duration_ns / 1e6, 4),
                    },
                    {
                      title: '进一步诊断',
                      render: (_, r) => (
                        <Button
                          onClick={() => {
                            setFilter(r.name.slice(0, 120).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
                            setTool('ncu');
                            if (receipt?.source_run_id) choose('source', receipt.source_run_id);
                            window.scrollTo({ top: 0, behavior: 'smooth' });
                          }}
                        >
                          选择此 Kernel
                        </Button>
                      ),
                    },
                  ]}
                />
                <details>
                  <summary>GPU 活动、拷贝与 CPU CUDA API 统计</summary>
                  <JsonView
                    value={{
                      activity: panel.device_activity,
                      copies: panel.copies,
                      cuda_apis: panel.cuda_apis,
                    }}
                  />
                </details>
              </>
            )}
            {panel?.tool === 'eretrace' && panel.mode !== 'memory' && selected.data && (
              <ERETracePanel data={panel} run={selected.data.id} />
            )}
            {panel?.mode === 'memory' && selected.data && (
              <DMATPanel key={selected.data.id} data={panel} run={selected.data.id} />
            )}
            {!!kernels.length && (
              <>
                <Select
                  aria-label="计数器 Kernel"
                  style={{ width: '100%', marginTop: 12 }}
                  value={Math.min(kernel, kernels.length - 1)}
                  onChange={setKernel}
                  options={kernels.map((k, i) => ({
                    value: i,
                    label: `PID ${k.process_id} · launch ${k.launch_id} · ${k.name} · grid ${k.grid} / block ${k.block}`,
                  }))}
                />
                <Input
                  aria-label="搜索硬件指标"
                  placeholder="搜索指标：lts / dram / sm / tensor / occupancy"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
                <Table<Metric>
                  size="small"
                  rowKey="name"
                  dataSource={activeKernel?.metrics.filter((m) =>
                    m.name.toLowerCase().includes(search.toLowerCase()),
                  )}
                  pagination={{ pageSize: 15 }}
                  columns={[
                    { title: '原生指标', dataIndex: 'name' },
                    {
                      title: '数值',
                      render: (_, r) => (r.value == null ? r.raw : fmt(r.value, 4)),
                    },
                    { title: '单位', dataIndex: 'unit' },
                  ]}
                />
              </>
            )}
            {terminal(selected.data.state) && (
              <Space wrap style={{ marginTop: 12 }}>
                {[
                  'receipt.json',
                  'manifest.json',
                  'profiler.log',
                  ...(receipt?.panel
                    ? [
                        'panel.json',
                        ...(receipt.tool === 'eretrace'
                          ? []
                          : [
                              receipt.tool === 'nsys' ? 'capture.nsys-rep' : 'capture.ncu-rep',
                              receipt.tool === 'nsys' ? 'capture.sqlite' : 'metrics.csv',
                            ]),
                      ]
                    : []),
                ].map((name) => (
                  <a
                    key={name}
                    href={`${base}/diagnostics/jobs/${selected.data!.id}/artifacts/${name}`}
                    download
                  >
                    {name}
                  </a>
                ))}
              </Space>
            )}
          </>
        )}
      </Panel>
    </>
  );
}
