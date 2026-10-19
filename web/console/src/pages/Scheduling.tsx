import Button from '../components/ActionButton';
import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Checkbox, Input, InputNumber, Select, Space, Table, Tag } from 'antd';
import { Link, useSearchParams } from 'react-router-dom';
import { api, fmt, terminal } from '../api/client';
import type { Deployment, Run } from '../api/types';
import { PageHead, Panel, QueryState, Status } from '../components';
import { applyKernelProfile, unloadDeployment } from '../features/kernels/lifecycle';

const capabilityReasons: Record<string, string> = {
  SLIDING_OR_HYBRID_CACHE_NOT_ADAPTED: '该模型使用滑动窗口或混合 KV，目前尚未适配多请求合批。',
  FORWARD_MASK_POSITION_ADAPTER_NOT_VERIFIED: '该模型的合批 mask 与位置编码适配尚未验证。',
  CUSTOM_KERNEL_PROFILE_NOT_BATCH_VERIFIED: '当前自定义算子尚未验证合批，请重载为框架原生算子。',
  LOAD_MODEL_TO_INSPECT_CACHE_LAYOUT: '请先加载所选模型，再检查实际 KV 布局。',
  EXPLICIT_POSITION_IDS_REQUIRED: '该模型暂不支持合批所需的显式位置编码。',
  LOCAL_PYTORCH_REQUIRED: '调度实验目前只支持本机 PyTorch 部署。',
  PRIVILEGED_SESSION_PROTOCOL_NOT_IMPLEMENTED: '当前特权 worker 尚未接入多请求调度协议。',
};

type RequestResult = {
  id: string;
  reason: string;
  input_tokens: number;
  output_tokens: number;
  ttft_ms: number | null;
  tpot_ms: number | null;
  quality: string;
  text: string;
  selected_logprob_max_abs_delta: number | null;
  first_difference?: {
    output_index: number;
    baseline_token: number;
    candidate_token: number;
  } | null;
};
type Variant = {
  session_id: string;
  mode: string;
  repeat: number;
  output_tokens_per_s: number;
  ttft_p95_ms: number | null;
  max_actual_batch: number;
  membership_changes_with_overlap: number;
  dynamic_join_requests: string[];
  kv_copy_bytes: number;
  kv_layout?: string;
  memory_management?: {
    peak_pool_storage_bytes: number;
    peak_tail_waste_tokens: number;
    pack_copy_bytes: number;
    scatter_copy_bytes: number;
  };
  peak_allocator_bytes: number;
  padding_positions: number;
  slo_goodput_requests_per_s: number;
  requests: RequestResult[];
  loadgen: { validity: string };
  cleanup: { device_kv_rows: number };
  outcome: string;
};
type Experiment = {
  id: string;
  state: string;
  error: string | null;
  quality: string;
  config: { requests: { id: string }[] };
  result?: {
    variants: Variant[];
    capability: unknown;
    evidence: unknown;
    quality: string;
    quality_coverage?: unknown;
  };
  progress?: unknown;
};
const defaults = Array.from({ length: 6 }, (_, i) => ({
  id: `request-${i + 1}`,
  prompt:
    i % 2
      ? 'Once upon a time, a little fox found a bright red ball. The fox'
      : 'Explain why inference uses a key value cache. The cache',
  arrival_ms: i < 2 ? 0 : (i - 1) * 100,
  output_tokens: i % 2 ? 16 : 64,
  priority: i === 4 ? 2 : 0,
  tenant: i % 2 ? 'short' : 'long',
}));

export default function Scheduling() {
  const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const identity = params.get('experiment') ?? '';
  const [deploymentId, setDeploymentId] = useState('');
  const [workload, setWorkload] = useState(JSON.stringify(defaults, null, 2));
  const [policy, setPolicy] = useState('fcfs');
  const [phasePolicy, setPhasePolicy] = useState('balanced');
  const [prefillQuota, setPrefillQuota] = useState(16);
  const [weights, setWeights] = useState('{}');
  const [arrival, setArrival] = useState('open_loop');
  const [modes, setModes] = useState(['serial', 'static', 'continuous']);
  const [active, setActive] = useState(4);
  const [batch, setBatch] = useState(4);
  const [tokens, setTokens] = useState(256);
  const [chunk, setChunk] = useState(64);
  const [kv, setKv] = useState(8192);
  const [kvLayout, setKvLayout] = useState('dense');
  const [kvBlock, setKvBlock] = useState(16);
  const [kvPoolMiB, setKvPoolMiB] = useState(256);
  const [padded, setPadded] = useState(16384);
  const [repeats, setRepeats] = useState(1);
  const [chunked, setChunked] = useState(true);
  const [preempt, setPreempt] = useState(false);
  const [saveInput, setSaveInput] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [lifecycleAction, setLifecycleAction] = useState<'load' | 'unload' | null>(null);
  const lifecycleBusy = lifecycleAction !== null;
  const [lifecycleRun, setLifecycleRun] = useState<Run>();
  const lifecycleController = useRef<AbortController | null>(null);
  useEffect(() => () => lifecycleController.current?.abort(), []);
  const [recordKind, setRecordKind] = useState('forward');
  const [after, setAfter] = useState(0);
  const [showRecords, setShowRecords] = useState(false);
  useEffect(() => setAfter(0), [identity, recordKind]);
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 3000,
  });
  const deployment =
    deployments.data?.items.find((d) => d.id === deploymentId) ??
    deployments.data?.items.find((d) => d.provider === 'pytorch' && d.state === 'ready') ??
    deployments.data?.items.find((d) => d.provider === 'pytorch');
  useEffect(() => {
    if (!deploymentId && deployment) setDeploymentId(deployment.id);
  }, [deploymentId, deployment]);
  const residents =
    deployments.data?.items.filter((d) => d.provider === 'pytorch' && d.state === 'ready') ?? [];
  const transitioning = deployments.data?.items.some(
    (d) => d.provider === 'pytorch' && !['ready', 'unloaded', 'failed'].includes(d.state),
  );
  const listing = useQuery({
    queryKey: ['scheduling-list'],
    queryFn: () => api<{ items: Experiment[]; active_id: string | null }>('/scheduling-benchmarks'),
    refetchInterval: 2000,
  });
  const result = useQuery({
    queryKey: ['scheduling', identity],
    queryFn: () => api<Experiment>(`/scheduling-benchmarks/${identity}`),
    enabled: !!identity,
    refetchInterval: (q) => (q.state.data && terminal(q.state.data.state) ? false : 1500),
  });
  const capability = useQuery({
    queryKey: ['scheduling-capability', deployment?.id, deployment?.epoch, deployment?.state],
    queryFn: () =>
      api<{ supported: boolean; reasons: string[]; kv_layout?: string }>(
        `/deployments/${deployment!.id}/scheduling-capabilities`,
      ),
    enabled:
      deployment?.state === 'ready' && !listing.data?.active_id && !lifecycleBusy && !transitioning,
    retry: false,
    refetchInterval: (q) => (q.state.error ? 3000 : false),
  });
  const modelActionBlocked =
    busy ||
    lifecycleBusy ||
    transitioning ||
    !!listing.data?.active_id ||
    !listing.data ||
    !!listing.error ||
    !!deployments.error;
  const launchReason = deployments.isPending
    ? '正在读取模型列表。'
    : deployments.error
      ? '模型列表读取失败，请重试。'
      : !deployment
        ? '尚无可用的本机模型，请先前往模型目录下载并接入。'
        : lifecycleBusy || transitioning
          ? '模型正在加载或卸载，请等待完成。'
          : listing.error
            ? '无法确认设备是否被调度实验占用，请重试。'
            : !listing.data
              ? '正在检查调度实验状态。'
              : listing.data.active_id
                ? `调度实验 ${listing.data.active_id} 正在运行，请等待完成或停止该实验。`
                : deployment.state !== 'ready'
                  ? `所选模型「${deployment.name}」尚未加载，请点击上方的加载或切换按钮。`
                  : capability.error
                    ? `模型能力检查失败：${capability.error.message}`
                    : !capability.data
                      ? '正在检查所选模型的多请求合批能力。'
                      : !capability.data.supported
                        ? `所选模型暂不能运行调度对照：${capability.data.reasons.map((r) => capabilityReasons[r] ?? r).join(' ')}`
                        : !modes.length
                          ? '请至少选择一种执行方式。'
                          : busy
                            ? '正在提交调度实验。'
                            : '';
  const changeModel = async (action: 'load' | 'unload') => {
    if (!deployment || modelActionBlocked || lifecycleController.current) return;
    const controller = new AbortController();
    lifecycleController.current = controller;
    setLifecycleAction(action);
    setLifecycleRun(undefined);
    setError('');
    try {
      const actual =
        action === 'load'
          ? await applyKernelProfile(
              deployment,
              'framework-native',
              controller.signal,
              setLifecycleRun,
            )
          : await unloadDeployment(deployment, controller.signal, setLifecycleRun);
      client.setQueryData<{ items: Deployment[] }>(
        ['deployments'],
        (old) =>
          old && {
            items: old.items.map((item) => (item.id === actual.id ? actual : item)),
          },
      );
    } catch (failure) {
      if (!controller.signal.aborted) setError((failure as Error).message);
    } finally {
      if (!controller.signal.aborted) {
        lifecycleController.current = null;
        setLifecycleAction(null);
        void client.invalidateQueries({ queryKey: ['deployments'] });
        void client.invalidateQueries({ queryKey: ['scheduling-capability'] });
      }
    }
  };
  const launch = async () => {
    if (!deployment || launchReason) return;
    setBusy(true);
    setError('');
    try {
      const run = await api<Experiment>('/scheduling-benchmarks', {
        deployment_id: deployment.id,
        expected_epoch: deployment.epoch,
        requests: JSON.parse(workload),
        modes: ['serial', 'static', 'continuous'].filter((m) => modes.includes(m)),
        repeats,
        arrival_mode: arrival,
        concurrency: active,
        save_input: saveInput,
        limits: {
          policy,
          phase_policy: phasePolicy,
          prefill_quota_tokens: prefillQuota,
          tenant_weights: JSON.parse(weights),
          max_active_sequences: active,
          max_batch_sequences: batch,
          max_batched_tokens: tokens,
          prefill_chunk_tokens: chunk,
          chunked_prefill: chunked,
          kv_capacity_tokens: kv,
          kv_layout: kvLayout,
          kv_block_tokens: kvBlock,
          kv_pool_max_bytes: kvPoolMiB * 1024 ** 2,
          max_padded_positions: padded,
          preemption: preempt ? 'recompute' : 'off',
        },
      });
      setParams({ experiment: run.id });
      void listing.refetch();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const control = async (path: string) => {
    try {
      await api(path, {});
      void result.refetch();
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const running = !!result.data && !terminal(result.data.state);
  const records = useQuery({
    queryKey: ['scheduling-records', identity, recordKind, after],
    queryFn: () =>
      api<{ items: unknown[]; next_after: number }>(
        `/scheduling-benchmarks/${identity}/records?kind=${recordKind}&after=${after}&limit=50`,
      ),
    enabled: showRecords && !!result.data && terminal(result.data.state),
    retry: false,
  });
  const replay = async () => {
    if (modelActionBlocked) return;
    try {
      const run = await api<Experiment>(`/scheduling-benchmarks/${identity}/replay`, {});
      setParams({ experiment: run.id });
      void listing.refetch();
    } catch (e) {
      setError((e as Error).message);
    }
  };
  return (
    <>
      <PageHead
        kicker="L8"
        title="多请求调度实验"
        description="同一驻留模型，对照串行、静态批次和真实 Continuous Batching。记录逐次 forward、KV 生命周期、质量和 Token/s。"
        extra={<Link to="/benchmarks">算子性能与质量对照</Link>}
      />
      <Panel
        title="实验模型"
        subtitle="在此加载、卸载或切换模型。切换会等待当前模型排空并卸载成功，再加载所选模型；L8 使用框架原生算子。"
      >
        <Space wrap>
          <Select
            aria-label="实验模型"
            style={{ minWidth: 280 }}
            value={deployment?.id}
            disabled={lifecycleBusy || busy}
            showSearch
            optionFilterProp="label"
            onChange={(value) => {
              setDeploymentId(value);
              setLifecycleRun(undefined);
              setError('');
            }}
            options={deployments.data?.items
              .filter((d) => d.provider === 'pytorch')
              .map((d) => ({ value: d.id, label: `${d.name} · ${d.state}` }))}
          />
          {deployment && <Status state={deployment.state} />}
          <Button
            loading={lifecycleAction === 'load'}
            disabledReason={{
              reason: modelActionBlocked
                ? '当前调度实验或模型切换仍在执行，请先等待完成。'
                : !deployment
                  ? '请先在本页选择实验模型。'
                  : '所选模型已使用原生算子加载，无需重复加载。',
              next: { to: '/devices', label: '查看模型状态' },
            }}
            disabled={
              !deployment ||
              modelActionBlocked ||
              (deployment.state === 'ready' &&
                (deployment.kernel_profile ?? 'framework-native') === 'framework-native')
            }
            onClick={() => void changeModel('load')}
          >
            {deployment?.state === 'ready'
              ? (deployment.kernel_profile ?? 'framework-native') === 'framework-native'
                ? '模型已加载'
                : '重载为原生算子'
              : residents.some((d) => d.id !== deployment?.id)
                ? '切换并加载模型'
                : '加载所选模型'}
          </Button>
          <Button
            loading={lifecycleAction === 'unload'}
            disabledReason={{
              reason: modelActionBlocked
                ? '当前调度实验或模型切换仍在执行，请先等待完成。'
                : '所选模型没有就绪的驻留实例，当前无需卸载。',
              next: { to: '/devices', label: '查看模型状态' },
            }}
            disabled={!deployment || deployment.state !== 'ready' || modelActionBlocked}
            onClick={() => void changeModel('unload')}
          >
            卸载所选模型
          </Button>
          <Link to="/models">下载与接入模型</Link>
        </Space>
        <p className="help-text">
          当前 GPU 驻留：{residents.map((d) => d.name).join('、') || '无'}。
          {deployment &&
            residents.some((d) => d.id !== deployment.id) &&
            `切换时将卸载当前模型并加载「${deployment.name}」。`}
          {deployment?.state === 'ready' &&
            ` 当前算子：${deployment.kernel_profile ?? deployment.kernel}。`}
        </p>
        {deployment?.last_error && <Alert type="error" showIcon message={deployment.last_error} />}
        {deployment?.state === 'ready' &&
          capability.data &&
          !capability.data.supported &&
          !lifecycleBusy && (
            <Alert
              type="warning"
              showIcon
              message="模型已加载，但尚不支持当前 L8 调度实验"
              description={capability.data.reasons.map((r) => capabilityReasons[r] ?? r).join(' ')}
            />
          )}
        {lifecycleRun && (
          <p role="status">
            {lifecycleRun.kind === 'load' ? '加载与预热' : '排空与卸载'}：
            <Status state={lifecycleRun.state} />{' '}
            {lifecycleBusy ? (
              <span className="mono">{lifecycleRun.id}</span>
            ) : (
              <Link to={`/runs/${lifecycleRun.id}`}>查看模型任务记录</Link>
            )}
          </p>
        )}
        {lifecycleBusy && (
          <p className="help-text">
            请保持本页打开以完成切换。离开后已提交的任务继续执行，后续加载不会再提交。
          </p>
        )}
      </Panel>
      <Panel
        title="运行方式"
        subtitle="此处使用原始文本续写，所有方案共用冻结输入；原单请求推理入口继续保留。"
      >
        <Space wrap>
          <Select
            aria-label="排序策略"
            value={policy}
            onChange={setPolicy}
            options={['fcfs', 'priority_aging', 'tenant_fair'].map((value) => ({
              value,
              label: value,
            }))}
          />
          <Select
            aria-label="到达模式"
            value={arrival}
            onChange={setArrival}
            options={[
              { value: 'open_loop', label: '开放环 · 固定到达' },
              { value: 'closed_loop', label: '闭合环 · 完成反馈' },
            ]}
          />
          <Select
            aria-label="阶段分配"
            value={phasePolicy}
            onChange={setPhasePolicy}
            options={[
              { value: 'balanced', label: '按请求顺序分配' },
              { value: 'decode_first', label: 'Decode 优先，保留 Prefill 配额' },
            ]}
          />
          <Checkbox.Group
            value={modes}
            onChange={setModes}
            options={[
              { label: '串行基线', value: 'serial' },
              { label: '静态批次', value: 'static' },
              { label: '连续批次', value: 'continuous' },
            ]}
          />
        </Space>
        <p>
          当前能力：
          {capability.data?.supported &&
          !capability.error &&
          deployment?.state === 'ready' &&
          !lifecycleBusy
            ? '支持稠密 KV 动态合批'
            : '加载模型并通过能力检查后可运行'}
          。分页 KV、滑动窗口及混合 Cache 暂未接入。
        </p>
        <QueryState error={capability.error ?? deployments.error} />
        <Space wrap>
          <label>
            最小 Prefill 配额{' '}
            <InputNumber
              min={1}
              max={4096}
              value={prefillQuota}
              onChange={(v) => setPrefillQuota(v ?? 16)}
            />
          </label>
          {policy === 'tenant_fair' && (
            <label>
              租户权重 JSON{' '}
              <Input
                aria-label="租户权重 JSON"
                value={weights}
                onChange={(e) => setWeights(e.target.value)}
                style={{ width: 220 }}
              />
            </label>
          )}
          <label>
            活动上限{' '}
            <InputNumber min={1} max={16} value={active} onChange={(v) => setActive(v ?? 4)} />
          </label>
          <label>
            Forward 行数{' '}
            <InputNumber min={1} max={16} value={batch} onChange={(v) => setBatch(v ?? 4)} />
          </label>
          <label>
            每轮有效 Token{' '}
            <InputNumber min={1} max={4096} value={tokens} onChange={(v) => setTokens(v ?? 256)} />
          </label>
          <label>
            Prefill 分块{' '}
            <InputNumber min={1} max={4096} value={chunk} onChange={(v) => setChunk(v ?? 64)} />
          </label>
          <label>
            KV 预留 Token{' '}
            <InputNumber min={1} max={131072} value={kv} onChange={(v) => setKv(v ?? 8192)} />
          </label>
          <label>
            KV 存储布局{' '}
            <Select
              aria-label="KV 存储布局"
              value={kvLayout}
              style={{ width: 250 }}
              onChange={setKvLayout}
              options={[
                { value: 'dense', label: '稠密独立行（原有）' },
                { value: 'block_pool', label: 'GPU 块池 / 稠密 Attention' },
              ]}
            />
          </label>
          {kvLayout === 'block_pool' && (
            <>
              <label>
                块大小 Token{' '}
                <InputNumber
                  aria-label="KV 块大小"
                  min={1}
                  max={256}
                  value={kvBlock}
                  onChange={(v) => setKvBlock(v ?? 16)}
                />
              </label>
              <label>
                块池预算 MiB{' '}
                <InputNumber
                  aria-label="KV 块池预算"
                  min={1}
                  max={2048}
                  value={kvPoolMiB}
                  onChange={(v) => setKvPoolMiB(v ?? 256)}
                />
              </label>
            </>
          )}
          <label>
            Padding 后位置上限{' '}
            <InputNumber
              min={1}
              max={131072}
              value={padded}
              onChange={(v) => setPadded(v ?? 16384)}
            />
          </label>
          <label>
            重复次数{' '}
            <InputNumber min={1} max={5} value={repeats} onChange={(v) => setRepeats(v ?? 1)} />
          </label>
        </Space>
        <p>
          <Space wrap>
            <Checkbox checked={chunked} onChange={(e) => setChunked(e.target.checked)}>
              启用分块 Prefill
            </Checkbox>
            <Checkbox checked={preempt} onChange={(e) => setPreempt(e.target.checked)}>
              高优先级抢占与历史重计算
            </Checkbox>
            <Checkbox checked={saveInput} onChange={(e) => setSaveInput(e.target.checked)}>
              保存输入供回放
            </Checkbox>
          </Space>
        </p>
        <Input.TextArea
          aria-label="冻结请求负载 JSON"
          rows={12}
          value={workload}
          onChange={(e) => setWorkload(e.target.value)}
        />
        <p>
          每项可改
          prompt、arrival_ms、output_tokens、priority、tenant、deadline_ms、cancel_after_ms、fixed_output_length。最多
          128 请求；上下文不足会明确拒绝。不同输入长度由实际 tokenizer 决定。
        </p>
        <Button
          type="primary"
          loading={busy}
          disabledReason={{
            reason: launchReason || '调度实验尚未准备完成，请检查本页模型、能力与负载配置。',
            next: {
              to: deployment?.state !== 'ready' ? '/devices' : '/scheduling',
              label: deployment?.state !== 'ready' ? '加载实验模型' : '检查调度模型、能力与负载',
            },
          }}
          disabled={!!launchReason}
          onClick={() => void launch()}
        >
          运行调度对照
        </Button>
        {launchReason && (
          <Alert
            className="section-gap"
            type="warning"
            showIcon
            message={launchReason}
            action={
              (capability.error || deployments.error || listing.error) && (
                <Button
                  size="small"
                  onClick={() => {
                    void deployments.refetch();
                    void listing.refetch();
                    if (deployment?.state === 'ready' && !modelActionBlocked)
                      void capability.refetch();
                  }}
                >
                  重试检查
                </Button>
              )
            }
          />
        )}
        {error && <Alert type="error" showIcon message={error} />}
      </Panel>
      <Panel title="实验记录">
        <Select
          aria-label="选择调度实验"
          style={{ minWidth: 360 }}
          value={identity || undefined}
          onChange={(value) => setParams({ experiment: value })}
          options={listing.data?.items.map((r) => ({ value: r.id, label: `${r.id} · ${r.state}` }))}
        />
        <QueryState
          loading={result.isFetching && !result.data}
          error={result.error ?? listing.error}
        />
        {result.data && (
          <>
            <p>
              <Status state={result.data.state} /> · 输出质量：
              {result.data.result?.quality ?? '等待对照'}
            </p>
            {!running && (
              <Button
                disabledReason={{
                  reason: '当前实验或模型切换尚未结束，请等待完成后再回放冻结负载。',
                  next: { to: '/scheduling', label: '查看调度实验进度' },
                }}
                disabled={!!modelActionBlocked}
                onClick={() => void replay()}
              >
                回放已保存的冻结负载
              </Button>
            )}
            {running && (
              <Space wrap>
                <Button
                  danger
                  onClick={() => void control(`/scheduling-benchmarks/${identity}/cancel`)}
                >
                  停止整个实验
                </Button>
                {result.data.config.requests.map((r) => (
                  <Button
                    key={r.id}
                    onClick={() =>
                      void control(`/scheduling-benchmarks/${identity}/requests/${r.id}/cancel`)
                    }
                  >
                    取消 {r.id}
                  </Button>
                ))}
              </Space>
            )}
            {result.data.error && <Alert type="error" message={result.data.error} />}
            <Table<Variant>
              rowKey="session_id"
              dataSource={result.data.result?.variants}
              pagination={false}
              scroll={{ x: 1300 }}
              columns={[
                { title: '方案 / 重复', render: (_, v) => `${v.mode} / ${v.repeat + 1}` },
                { title: 'Token/s', dataIndex: 'output_tokens_per_s', render: (v) => fmt(v) },
                { title: 'TTFT p95 ms', dataIndex: 'ttft_p95_ms', render: (v) => fmt(v) },
                {
                  title: 'SLO goodput req/s',
                  dataIndex: 'slo_goodput_requests_per_s',
                  render: (v) => fmt(v, 2),
                },
                { title: '真实 B 最大值', dataIndex: 'max_actual_batch' },
                {
                  title: 'KV 布局',
                  dataIndex: 'kv_layout',
                  render: (v: string | undefined) => v ?? '历史稠密布局',
                },
                {
                  title: '块池峰值 MiB',
                  render: (_, v) =>
                    v.memory_management
                      ? fmt(v.memory_management.peak_pool_storage_bytes / 1024 ** 2, 3)
                      : '—',
                },
                {
                  title: '尾部浪费 Token 峰值',
                  render: (_, v) => fmt(v.memory_management?.peak_tail_waste_tokens, 0),
                },
                {
                  title: '拼装 / 写回 MiB',
                  render: (_, v) =>
                    v.memory_management
                      ? `${fmt(v.memory_management.pack_copy_bytes / 1024 ** 2, 3)} / ${fmt(v.memory_management.scatter_copy_bytes / 1024 ** 2, 3)}`
                      : '—',
                },
                { title: '成员变更', dataIndex: 'membership_changes_with_overlap' },
                {
                  title: '实际动态加入',
                  dataIndex: 'dynamic_join_requests',
                  render: (v: string[] | undefined) => v?.join(', ') || '—',
                },
                {
                  title: 'KV 复制 MiB',
                  dataIndex: 'kv_copy_bytes',
                  render: (v) => fmt(v / 1024 ** 2),
                },
                {
                  title: '显存峰值 MiB',
                  dataIndex: 'peak_allocator_bytes',
                  render: (v) => fmt(v / 1024 ** 2),
                },
                {
                  title: '负载有效性',
                  render: (_, v) => (
                    <Tag color={v.loadgen.validity === 'VALID' ? 'green' : 'red'}>
                      {v.loadgen.validity}
                    </Tag>
                  ),
                },
                { title: '回收后 KV 行数', render: (_, v) => v.cleanup.device_kv_rows },
              ]}
              expandable={{
                expandedRowRender: (v) => (
                  <Table<RequestResult>
                    rowKey="id"
                    pagination={false}
                    dataSource={v.requests}
                    columns={[
                      { title: '请求', dataIndex: 'id' },
                      { title: '结束', dataIndex: 'reason' },
                      {
                        title: '输入/输出',
                        render: (_, r) => `${r.input_tokens}/${r.output_tokens}`,
                      },
                      { title: 'TTFT ms', dataIndex: 'ttft_ms', render: (x) => fmt(x) },
                      { title: 'TPOT ms', dataIndex: 'tpot_ms', render: (x) => fmt(x) },
                      {
                        title: '逐 Token 一致性',
                        dataIndex: 'quality',
                        render: (v: string) => (
                          <Tag
                            color={
                              v === 'exact_tokens_fail'
                                ? 'red'
                                : v === 'exact_tokens_pass'
                                  ? 'green'
                                  : undefined
                            }
                          >
                            {v}
                          </Tag>
                        ),
                      },
                      {
                        title: '首次分歧',
                        dataIndex: 'first_difference',
                        render: (v: RequestResult['first_difference']) =>
                          v
                            ? `第 ${v.output_index} Token：${v.baseline_token} → ${v.candidate_token}`
                            : '—',
                      },
                      {
                        title: '所选 Token logprob 误差',
                        dataIndex: 'selected_logprob_max_abs_delta',
                        render: (x) => fmt(x, 5),
                      },
                      { title: '实际输出', dataIndex: 'text', width: 320 },
                    ]}
                  />
                ),
              }}
            />
            <details>
              <summary>实际能力与证据完整性</summary>
              <pre>
                {JSON.stringify(
                  {
                    capability: result.data.result?.capability,
                    evidence: result.data.result?.evidence,
                    quality_coverage: result.data.result?.quality_coverage,
                    progress: result.data.progress,
                  },
                  null,
                  2,
                )}
              </pre>
            </details>
            <p>
              原始记录接口：
              <code>/api/console/v1/scheduling-benchmarks/{identity}/records?kind=forward</code>
              。支持 kind=kv.commit、kv.release、scheduler.plan、token、request.result，并用 after /
              limit 分页。GPU 时间按共享批次记录；进程缺页与 RSS 按会话记录。
            </p>
            <Space wrap>
              <Button onClick={() => setShowRecords((v) => !v)}>
                {showRecords ? '收起原始记录' : '查看原始记录'}
              </Button>
              {showRecords && (
                <>
                  <Select
                    aria-label="原始记录类型"
                    value={recordKind}
                    onChange={setRecordKind}
                    options={[
                      'forward',
                      'scheduler.plan',
                      'kv.reserve',
                      'kv.commit',
                      'kv.release',
                      'token',
                      'request.result',
                      'session.result',
                    ].map((value) => ({ value, label: value }))}
                  />
                  <Button
                    disabledReason={{ reason: '已经位于第一页，无需返回第一页。' }}
                    disabled={!after}
                    onClick={() => setAfter(0)}
                  >
                    第一页
                  </Button>
                  <Button
                    disabledReason={{
                      reason: '当前没有更多请求记录，请等待采集完成或回到第一页查看。',
                    }}
                    disabled={!records.data?.items.length}
                    onClick={() => setAfter(records.data?.next_after ?? after)}
                  >
                    下一页
                  </Button>
                </>
              )}
            </Space>
            {showRecords && (
              <>
                <QueryState loading={records.isFetching} error={records.error} />
                <pre style={{ maxHeight: 500, overflow: 'auto' }}>
                  {JSON.stringify(records.data?.items ?? [], null, 2)}
                </pre>
              </>
            )}
          </>
        )}
      </Panel>
    </>
  );
}
