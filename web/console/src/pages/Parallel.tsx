import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Checkbox, Input, InputNumber, Select, Space, Table } from 'antd';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import Button, { type ActionBlock } from '../components/ActionButton';
import { api, fmt, terminal } from '../api/client';
import type { Deployment, Run } from '../api/types';
import { JsonView, PageHead, Panel, QueryState, Status } from '../components';
import { useRun } from '../hooks';
import ParallelEvidence from '../features/parallel/ParallelEvidence';
import ParallelComparison, { type ParallelPair } from '../features/parallel/ParallelComparison';
import type { ParallelPlan, ParallelRun } from '../features/parallel/types';

const initial: ParallelPlan = {
  strategy: 'single',
  device_type: 'cuda',
  devices: [0],
  precision: 'float16',
  prompts: [
    'Once upon a time, there was a little fox.',
    'Explain why the sky is blue in one sentence.',
  ],
  max_output_tokens: 32,
  fixed_output_length: true,
  trace_collectives: false,
  timeout_s: 120,
};
export default function Parallel() {
  const { section = 'run' } = useParams();
  const [params, setParams] = useSearchParams();
  const [plan, setPlan] = useState(initial);
  const [peer, setPeer] = useState('');
  const [peerDeployment, setPeerDeployment] = useState('qwen-reference');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [baseline, setBaseline] = useState('');
  const [candidate, setCandidate] = useState('');
  const [comparison, setComparison] = useState<ParallelPair>();
  const ds = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 2500,
  });
  const id = params.get('deployment') ?? ds.data?.items[0]?.id ?? '';
  const update = (key: string, value: string) =>
    setParams((p) => {
      const n = new URLSearchParams(p);
      n.set(key, value);
      return n;
    });
  const overview = useQuery({
    queryKey: ['parallel-overview'],
    queryFn: () => api<{ pending: number; benchmark_owner?: unknown }>('/overview'),
    refetchInterval: 1500,
  });
  const hardware = useQuery({
    queryKey: ['parallel-hardware'],
    queryFn: () =>
      api<{
        devices: { id: number; name: string; total_bytes: number; free_bytes: number }[];
        torch: string;
        transformers: string;
        nccl: boolean;
        gloo: boolean;
      }>('/parallel/hardware'),
    staleTime: 30000,
  });
  const catalog = useQuery({
    queryKey: ['parallel-catalog'],
    queryFn: () =>
      api<{ strategies: { id: string; label: string }[]; not_implemented: string[] }>(
        '/parallel/catalog',
      ),
  });
  const peers = useQuery({
    queryKey: ['parallel-peers'],
    queryFn: () => api<{ items: { id: string; name: string }[] }>('/parallel/peers'),
  });
  const records = useQuery({
    queryKey: ['parallel-jobs'],
    queryFn: () => api<{ items: ParallelRun[] }>('/parallel/jobs'),
    refetchInterval: 2000,
  });
  const run = useRun(params.get('run') ?? undefined);
  const r = run.data as ParallelRun | undefined;
  const device = `/devices#deployment-${encodeURIComponent(id)}`;
  useEffect(() => {
    if (!params.has('deployment') && id) update('deployment', id);
  }, [id]);
  useEffect(() => setComparison(undefined), [baseline, candidate]);
  const active =
    !!overview.data?.pending || !!overview.data?.benchmark_owner || (!!r && !terminal(r.state));
  const resident = ds.data?.items.some(
    (d) => d.provider === 'pytorch' && !['unloaded', 'failed'].includes(d.state),
  );
  const needed = plan.strategy === 'single' ? 1 : 2;
  const hardwareMissing =
    plan.device_type === 'cuda' &&
    (!hardware.data ||
      plan.devices.some((i) => i >= hardware.data.devices.length) ||
      plan.devices.length < needed);
  const block: ActionBlock | undefined = !id
    ? { reason: '请先在模型页下载并登记模型制品。', next: { to: '/models', label: '下载模型' } }
    : busy || active
      ? {
          reason: '当前任务或设备清理尚未结束，请在运行记录中等待或取消。',
          next: { to: '/runs', label: '查看运行记录' },
        }
      : resident
        ? {
            reason: 'L9 独立进程需要拥有设备。请先在设备与部署页卸载当前驻留模型。',
            next: { to: device, label: '卸载驻留模型' },
          }
        : hardwareMissing
          ? {
              reason: `当前节点可见 ${hardware.data?.devices.length ?? 0} 张 GPU；所选策略需要所选设备全部可见且至少 ${needed} 张。请选择多卡节点或使用明确标识的 CPU 逻辑诊断。`,
              next: { to: '/machines', label: '选择执行节点' },
            }
          : !['single', 'layer_shard'].includes(plan.strategy) &&
              hardware.data &&
              (!hardware.data.gloo ||
                (plan.device_type === 'cuda' &&
                  ['tp', 'dp'].includes(plan.strategy) &&
                  !hardware.data.nccl))
            ? {
                reason:
                  '当前节点没有所需的 Gloo / NCCL 后端。请选择支持相应通信库的执行节点；跨节点 HTTP DP 不需要这些进程组后端。',
                next: { to: '/machines', label: '选择执行节点' },
              }
            : undefined;
  async function submit() {
    setBusy(true);
    setError('');
    try {
      const next = await api<Run>('/parallel/jobs', { deployment_id: id, plan });
      update('run', next.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function compare() {
    setBusy(true);
    setError('');
    try {
      setComparison(
        await api<ParallelPair>('/parallel/comparisons', {
          baseline_run_id: baseline,
          candidate_run_id: candidate,
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function submitPeer() {
    setBusy(true);
    setError('');
    try {
      const next = await api<Run>('/parallel/federated-jobs', {
        deployment_id: id,
        peer_machine_id: peer,
        peer_deployment_id: peerDeployment,
        plan,
      });
      update('run', next.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const changes = (patch: Partial<ParallelPlan>) => setPlan((p) => ({ ...p, ...patch }));
  return (
    <>
      <PageHead
        kicker="L9"
        title="多卡与分布式执行"
        description="按真实权重分片、请求归属、KV 传输和通信凭据，核对并行策略的成本与输出。"
      />
      <Space wrap style={{ marginBottom: 20 }}>
        {[
          ['run', '执行方案'],
          ['topology', '设备与通信'],
          ['peers', '跨节点 DP'],
          ['compare', '结果对照'],
        ].map(([key, label]) => (
          <Link
            key={key}
            aria-current={section === key ? 'page' : undefined}
            to={`/parallel/${key}?${params}`}
          >
            {label}
          </Link>
        ))}
      </Space>
      <Panel title="当前源模型">
        <Select
          aria-label="L9 源模型"
          style={{ width: 340, maxWidth: '100%' }}
          value={id || undefined}
          options={ds.data?.items
            .filter((d) => d.provider === 'pytorch')
            .map((d) => ({ value: d.id, label: `${d.name} · ${d.state}` }))}
          onChange={(v) => update('deployment', v)}
        />
        <Space wrap>
          <Link to={device}>加载、卸载与切换模型 →</Link>
          <Link to="/machines">执行节点 →</Link>
        </Space>
        <p>
          L9 从已校验的原始浮点制品独立加载。任务结束会退出全部私有进程，不修改现有单请求部署配置。
        </p>
      </Panel>
      <QueryState error={ds.error ?? catalog.error ?? hardware.error ?? records.error} />
      {error && <Alert type="error" showIcon message={error} />}
      {section === 'run' && (
        <>
          <Panel title="并行方案与固定负载">
            <Space wrap>
              <Select
                aria-label="L9 并行策略"
                value={plan.strategy}
                style={{ width: 330, maxWidth: '100%' }}
                options={catalog.data?.strategies.map((s) => ({ value: s.id, label: s.label }))}
                onChange={(strategy) =>
                  changes({ strategy, devices: strategy === 'single' ? [0] : [0, 1] })
                }
              />
              <Select
                aria-label="L9 执行设备类型"
                value={plan.device_type}
                options={[
                  { value: 'cuda', label: '真实 CUDA 设备' },
                  { value: 'cpu', label: 'CPU 多进程诊断 / 不代表 GPU 性能' },
                ]}
                onChange={(device_type) =>
                  changes({ device_type, precision: device_type === 'cpu' ? 'float32' : 'float16' })
                }
              />
              <Select
                aria-label="L9 精度"
                value={plan.precision}
                options={['float16', 'bfloat16', 'float32'].map((value) => ({
                  value,
                  label: value,
                }))}
                onChange={(precision) => changes({ precision })}
              />
              <Select
                aria-label="L9 设备索引"
                mode="multiple"
                value={plan.devices}
                options={Array.from(
                  {
                    length: plan.device_type === 'cuda' ? (hardware.data?.devices.length ?? 0) : 4,
                  },
                  (_, i) => ({
                    value: i,
                    label: `${plan.device_type === 'cuda' ? 'GPU' : 'CPU rank'} ${i}`,
                  }),
                )}
                onChange={(devices) => changes({ devices })}
                style={{ minWidth: 180 }}
              />
            </Space>
            <p>
              每行一条固定原始文本请求；所有策略使用同一输入及输出预算。DP 按请求索引分片；TP 所有
              rank 共同执行每条请求。按层放置是顺序跨卡执行，没有多微批流水重叠。
            </p>
            <Input.TextArea
              aria-label="L9 固定请求负载"
              rows={4}
              value={plan.prompts.join('\n')}
              onChange={(e) => changes({ prompts: e.target.value.split('\n') })}
            />
            <Space wrap style={{ marginTop: 12 }}>
              <span>输出 Token 上限</span>
              <Select
                aria-label="L9 输出预算预设"
                value={plan.max_output_tokens}
                options={[8, 16, 32, 64, 128].map((value) => ({
                  value,
                  label: value === 128 ? '128 / 本阶段最大值' : String(value),
                }))}
                onChange={(max_output_tokens) => changes({ max_output_tokens })}
              />
              <Checkbox
                checked={plan.fixed_output_length}
                onChange={(e) => changes({ fixed_output_length: e.target.checked })}
              >
                固定输出长度
              </Checkbox>
              <Checkbox
                checked={plan.trace_collectives}
                onChange={(e) => changes({ trace_collectives: e.target.checked })}
              >
                记录通信 CPU 事件（有额外开销）
              </Checkbox>
              <span>任务超时 s</span>
              <InputNumber
                aria-label="L9 超时"
                min={15}
                max={300}
                value={plan.timeout_s}
                onChange={(v) => changes({ timeout_s: v ?? 120 })}
              />
            </Space>
            <p>
              CPU 诊断显式使用 FP32/Gloo。Prefill/Decode 当前用 CPU 暂存与 Gloo 传
              KV，并不声称零拷贝或 GPU 直连。
            </p>
            <Button
              type="primary"
              disabled={!!block}
              loading={busy}
              disabledReason={block}
              onClick={() => void submit()}
            >
              运行 L9 方案
            </Button>
          </Panel>
          {r && (
            <Panel title="当前任务">
              <Status state={r.state} />
              <Link to={`/runs/${r.id}`}>{r.id}</Link>
              {r.error && <Alert type="error" message={r.error} />}
              <Button
                disabled={terminal(r.state) || busy}
                disabledReason={{
                  reason: terminal(r.state)
                    ? '此任务已结束，无需取消。'
                    : '取消正在提交，请等待确认。',
                }}
                onClick={() =>
                  void api(`/requests/${r.id}/cancel`, {}).catch((e) => setError(e.message))
                }
              >
                取消 L9 任务
              </Button>
              {r.metrics.parallel_receipt && (
                <ParallelEvidence receipt={r.metrics.parallel_receipt} />
              )}
            </Panel>
          )}
        </>
      )}
      {section === 'topology' && (
        <>
          <Panel title="当前节点真实设备">
            <p>
              PyTorch {hardware.data?.torch} · Transformers {hardware.data?.transformers} · NCCL{' '}
              {String(hardware.data?.nccl)} · Gloo {String(hardware.data?.gloo)}
            </p>
            <Table
              size="small"
              rowKey="id"
              pagination={false}
              scroll={{ x: 600 }}
              dataSource={hardware.data?.devices}
              columns={[
                { title: '索引', dataIndex: 'id' },
                { title: '设备', dataIndex: 'name' },
                { title: '容量 GiB', render: (_, d) => fmt(d.total_bytes / 2 ** 30, 2) },
                { title: '采样空闲 GiB', render: (_, d) => fmt(d.free_bytes / 2 ** 30, 2) },
              ]}
            />
            <p>
              这里只显示当前进程可见设备。跨主机连接配置不等于 NCCL 互连；未探测的
              NVLink、带宽和拓扑路径不填成已支持。
            </p>
            <Button onClick={() => void hardware.refetch()}>重新读取设备</Button>
          </Panel>
          <Panel title="通信与证据口径">
            <p>
              TP 保存原生分片计划与全局/局部参数形状，拒绝没有实际分片的运行。P/D 记录传输的 KV
              字节与各自进程的阻塞传输时间。通信 CPU 事件统计不等于 GPU Kernel 完成时间。
            </p>
            <JsonView value={catalog.data?.not_implemented} />
          </Panel>
        </>
      )}
      {section === 'peers' && (
        <Panel title="已登记节点之间的请求副本 DP">
          <p>
            使用当前节点及一个已登记的 HTTP / SSH
            节点。两端各自加载同一份已校验源权重，只传请求及结果；不传权重或 KV，不要求两端 CUDA /
            CPU 架构相同。两端均需卸载驻留模型、提供本版本 L9 API。
          </p>
          <Space wrap>
            <Select
              aria-label="L9 远端节点"
              placeholder="选择已登记节点"
              value={peer || undefined}
              options={peers.data?.items.map((p) => ({ value: p.id, label: p.name }))}
              onChange={setPeer}
              style={{ width: 300, maxWidth: '100%' }}
            />
            <Input
              aria-label="L9 远端部署 ID"
              value={peerDeployment}
              onChange={(e) => setPeerDeployment(e.target.value)}
              placeholder="远端已登记源模型部署 ID"
            />
            <Link to="/machines">登记 / 检查节点连接 →</Link>
          </Space>
          <p>
            使用“执行方案”页配置的负载、精度和预算；策略需为单设备基线，至少两条请求。偶数请求归当前节点，奇数请求归远端节点；来源或
            Tokenizer 不一致则拒绝合格结果。父任务取消会取消本次拥有的两个子任务并等待各自清理确认。
          </p>
          <Button
            disabled={
              !!block ||
              !peer ||
              !peerDeployment ||
              plan.strategy !== 'single' ||
              plan.prompts.length < 2
            }
            loading={busy}
            disabledReason={
              block ?? {
                reason: !peer
                  ? '请先在本页选择一个已登记远端节点。'
                  : '请在执行方案页选择单设备基线并至少填写两条固定请求，填写远端部署 ID。',
                next: {
                  to: !peer ? '/machines' : `/parallel/run?deployment=${encodeURIComponent(id)}`,
                  label: !peer ? '登记节点' : '配置负载',
                },
              }
            }
            onClick={() => void submitPeer()}
          >
            运行双节点 DP
          </Button>
          {r && (
            <>
              <p>
                <Status state={r.state} />
                <Link to={`/parallel/run?deployment=${encodeURIComponent(id)}&run=${r.id}`}>
                  查看当前任务、取消与完整凭据 →
                </Link>
              </p>
              {r.error && <Alert type="error" message={r.error} />}
            </>
          )}
        </Panel>
      )}
      {section === 'compare' && (
        <Panel title="同负载 L9 对照">
          <Space wrap>
            {[
              [baseline, setBaseline, 'L9 基线'],
              [candidate, setCandidate, 'L9 候选'],
            ].map(([value, setter, label]) => (
              <Select
                key={label as string}
                aria-label={label as string}
                value={(value as string) || undefined}
                style={{ width: 340, maxWidth: '100%' }}
                showSearch
                options={records.data?.items
                  .filter((x) => x.state === 'completed')
                  .map((x) => ({
                    value: x.id,
                    label: `${x.id} · ${x.metrics.parallel_receipt?.plan.strategy}`,
                  }))}
                onChange={setter as (s: string) => void}
              />
            ))}
            <Button
              disabled={!baseline || !candidate || baseline === candidate || busy}
              disabledReason={{
                reason: busy
                  ? '正在读取并核对封存证据。'
                  : '请在本页选择两条不同的已完成 L9 运行，先运行单设备基线，再运行候选。',
                next: {
                  to: `/parallel/run?deployment=${encodeURIComponent(id)}`,
                  label: '运行基线与候选',
                },
              }}
              onClick={() => void compare()}
            >
              比较 L9 结果
            </Button>
          </Space>
          <p>
            核对源权重、实际输入 Token、精度、源码、运行版本和逐 Token
            输出；一次对照只提供描述性观测。
          </p>
          {comparison !== undefined && <ParallelComparison value={comparison} />}
        </Panel>
      )}
      <Panel title="L9 运行记录">
        <Table
          size="small"
          rowKey="id"
          scroll={{ x: 650 }}
          pagination={{ pageSize: 8 }}
          dataSource={records.data?.items}
          columns={[
            {
              title: '运行',
              render: (_, x) => (
                <Link to={`/parallel/run?deployment=${x.config.deployment.id}&run=${x.id}`}>
                  {x.id}
                </Link>
              ),
            },
            { title: '状态', dataIndex: 'state', render: (v) => <Status state={v} /> },
            {
              title: '策略',
              render: (_, x) => x.metrics.parallel_receipt?.plan.strategy ?? '执行中',
            },
            { title: 'Token/s', render: (_, x) => fmt(x.metrics.output_tokens_per_s, 2) },
          ]}
        />
      </Panel>
    </>
  );
}
