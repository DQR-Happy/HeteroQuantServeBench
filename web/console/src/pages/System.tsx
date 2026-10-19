import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Input, InputNumber, Select, Space, Table } from 'antd';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import Button, { type ActionBlock } from '../components/ActionButton';
import { api, terminal } from '../api/client';
import type { Deployment, Run } from '../api/types';
import { JsonView, PageHead, Panel, QueryState, Status } from '../components';
import { useRun } from '../hooks';
import ProfileEditor from '../features/system/ProfileEditor';
import SystemEvidence from '../features/system/SystemEvidence';
import type {
  Hardware,
  SystemPlan,
  SystemProfile,
  SystemReceipt,
  SystemRun,
} from '../features/system/types';

const profile: SystemProfile = {
  cpu_ids: null,
  intra_threads: 2,
  interop_threads: 1,
  numa_node: null,
  power_limit_w: null,
};
const initial: SystemPlan = {
  baseline: profile,
  candidate: { ...profile, intra_threads: 1 },
  device: 0,
  precision: 'float16',
  prompts: [
    'Once upon a time, there was a little fox.',
    'Explain why the sky is blue in one sentence.',
  ],
  max_output_tokens: 32,
  rounds: 2,
  timeout_s: 300,
};
export default function System() {
  const { section = 'run' } = useParams();
  const [params, setParams] = useSearchParams();
  const [plan, setPlan] = useState<SystemPlan>(initial);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const ds = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 2500,
  });
  const hardware = useQuery({
    queryKey: ['system-hardware'],
    queryFn: () => api<Hardware>('/system/hardware'),
    refetchInterval: 30000,
  });
  const overview = useQuery({
    queryKey: ['system-overview'],
    queryFn: () => api<{ pending: number; benchmark_owner?: unknown }>('/overview'),
    refetchInterval: 1500,
  });
  const jobs = useQuery({
    queryKey: ['system-jobs'],
    queryFn: () => api<{ items: SystemRun[] }>('/system/jobs'),
    refetchInterval: 2500,
  });
  const id =
    params.get('deployment') ?? ds.data?.items.find((d) => d.provider === 'pytorch')?.id ?? '';
  const update = (key: string, value: string) =>
    setParams((p) => {
      const n = new URLSearchParams(p);
      if (key === 'deployment' && value !== id) n.delete('run');
      n.set(key, value);
      return n;
    });
  const run = useRun(params.get('run') ?? undefined);
  const r = run.data as SystemRun | undefined;
  const sealed = useQuery({
    queryKey: ['system-receipt', r?.id, r?.state],
    queryFn: () => api<SystemReceipt>(`/runs/${r!.id}/system`),
    enabled: !!r && terminal(r.state),
  });
  const active =
    !!overview.data?.pending || !!overview.data?.benchmark_owner || (!!r && !terminal(r.state));
  const resident = ds.data?.items.some(
    (d) => d.provider === 'pytorch' && !['unloaded', 'failed'].includes(d.state),
  );
  const device = `/devices#deployment-${encodeURIComponent(id)}`;
  const block: ActionBlock | undefined = hardware.data?.quarantined
    ? {
        reason: hardware.data.recovery_reason,
        next: { to: '/system/topology', label: '查看硬件状态与恢复要求' },
      }
    : busy || active
      ? {
          reason: '当前任务尚未完成或仍在清理。请在运行记录等待或取消。',
          next: { to: '/runs', label: '查看运行记录' },
        }
      : resident
        ? {
            reason: 'L1 需要独立加载源模型。请先在设备与部署页卸载驻留模型。',
            next: { to: device, label: '卸载模型' },
          }
        : !hardware.data?.devices.length
          ? {
              reason: '尚未确认可见 CUDA 设备。请检查执行节点连接与驱动。',
              next: { to: '/machines', label: '选择执行节点' },
            }
          : !id
            ? { reason: '请先下载并登记原始浮点模型。', next: { to: '/models', label: '下载模型' } }
            : undefined;
  const changes = (patch: Partial<SystemPlan>) => setPlan((p) => ({ ...p, ...patch }));
  async function action(f: () => Promise<void>) {
    setBusy(true);
    setError('');
    try {
      await f();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHead
        kicker="L1"
        title="硬件与系统配置"
        description="固定模型和输入，对照 CPU 亲和性、线程、NUMA 与 GPU 功耗配置，核对实际生效值及观测变化。"
      />
      <Space wrap style={{ marginBottom: 20 }}>
        {[
          ['run', '配置实验'],
          ['topology', '硬件与能力'],
          ['results', '结果与观测'],
        ].map(([key, name]) => (
          <Link
            key={key}
            aria-current={section === key ? 'page' : undefined}
            to={`/system/${key}?${params}`}
          >
            {name}
          </Link>
        ))}
      </Space>
      <Panel title="源模型与执行节点">
        <Select
          aria-label="L1 源模型"
          style={{ width: 360, maxWidth: '100%' }}
          value={id || undefined}
          options={ds.data?.items
            .filter((d) => d.provider === 'pytorch')
            .map((d) => ({ value: d.id, label: `${d.name} · ${d.state}` }))}
          onChange={(v) => update('deployment', v)}
        />
        <Space wrap>
          <Link to={device}>加载、卸载与切换模型 →</Link>
          <Link to="/machines">选择 SSH / HTTP 执行节点 →</Link>
        </Space>
        <p>
          本页从已校验原始浮点制品独立加载模型，使用原生算子、固定精度与贪心解码。退出实验进程后释放权重；原有交互推理继续在推理工作台使用。
        </p>
      </Panel>
      <QueryState error={ds.error ?? hardware.error ?? jobs.error ?? sealed.error} />
      {error && <Alert type="error" showIcon message={error} />}
      {section === 'run' && (
        <>
          <Panel title="固定负载与对照边界">
            <Space wrap>
              <Select
                aria-label="L1 GPU"
                value={plan.device}
                options={hardware.data?.devices.map((d) => ({
                  value: d.id,
                  label: `GPU ${d.id} · ${d.name}`,
                }))}
                onChange={(device) => changes({ device })}
                style={{ width: 300 }}
              />
              <Select
                aria-label="L1 精度"
                value={plan.precision}
                options={['float16', 'bfloat16', 'float32'].map((value) => ({
                  value,
                  label: value,
                }))}
                onChange={(precision) => changes({ precision })}
              />
              <label>
                固定输出 Token{' '}
                <InputNumber
                  aria-label="L1 输出 Token"
                  min={2}
                  max={128}
                  value={plan.max_output_tokens}
                  onChange={(n) => changes({ max_output_tokens: n ?? 32 })}
                />
              </label>
              <label>
                每配置轮数{' '}
                <InputNumber
                  aria-label="L1 轮数"
                  min={1}
                  max={4}
                  value={plan.rounds}
                  onChange={(n) => changes({ rounds: n ?? 2 })}
                />
              </label>
              <label>
                任务超时秒{' '}
                <InputNumber
                  aria-label="L1 超时"
                  min={30}
                  max={1200}
                  value={plan.timeout_s}
                  onChange={(n) => changes({ timeout_s: n ?? 300 })}
                />
              </label>
            </Space>
            <p>
              每行一个原始文本输入，最多 4 条；不套用聊天模板。默认按 A/B、B/A
              顺序运行，每轮包含独立加载与预热；生成计时排除加载和预热。
            </p>
            <Input.TextArea
              aria-label="L1 固定输入"
              rows={3}
              value={plan.prompts?.join('\n')}
              onChange={(e) => changes({ prompts: e.target.value.split('\n') })}
            />
          </Panel>
          <ProfileEditor
            label="基线 A"
            value={plan.baseline ?? profile}
            hardware={hardware.data}
            device={plan.device ?? 0}
            onChange={(baseline) => changes({ baseline })}
          />
          <ProfileEditor
            label="候选 B"
            value={plan.candidate ?? profile}
            hardware={hardware.data}
            device={plan.device ?? 0}
            onChange={(candidate) => changes({ candidate })}
          />
          <Button
            type="primary"
            loading={busy}
            disabled={!!block}
            disabledReason={block}
            onClick={() =>
              void action(async () => {
                const next = await api<Run>('/system/jobs', { deployment_id: id, plan });
                update('run', next.id);
              })
            }
          >
            运行 L1 配置对照
          </Button>
        </>
      )}
      {section === 'topology' && (
        <Panel title="硬件事实与支持边界">
          <p>
            允许 CPU：{hardware.data?.allowed_cpus.join(', ')}；cgroup CPU 配额：
            {hardware.data?.cpu_quota ?? '未暴露'}。配额格式为 quota / period（微秒），不等于亲和性
            CPU 数量。
          </p>
          {hardware.data?.quarantined && (
            <Alert type="error" message={hardware.data.recovery_reason} />
          )}
          <Table
            size="small"
            pagination={false}
            rowKey="id"
            dataSource={hardware.data?.devices}
            scroll={{ x: 700 }}
            columns={[
              { title: '设备', dataIndex: 'name' },
              { title: 'UUID', dataIndex: 'uuid' },
              { title: '显存 GiB', render: (_, x) => (x.total_bytes / 2 ** 30).toFixed(1) },
              { title: '功耗上限 W', dataIndex: 'power_limit_w' },
              {
                title: '可设范围 W',
                render: (_, x) => x.power_constraints_w?.join('–') ?? x.power_reason,
              },
            ]}
          />
          <h3>NUMA 拓扑</h3>
          <JsonView value={hardware.data?.numa_nodes} />
          <ul>{hardware.data?.limitations.map((x) => <li key={x}>{x}</li>)}</ul>
          <details>
            <summary>完整硬件与容器边界</summary>
            <JsonView value={hardware.data} />
          </details>
        </Panel>
      )}
      {r && (
        <Panel title="当前实验">
          <Space wrap>
            <Status state={r.state} />
            <code>{r.id}</code>
            <Link to={`/system/results?deployment=${encodeURIComponent(id)}&run=${r.id}`}>
              结果与观测 →
            </Link>
            <Button
              loading={busy}
              disabled={terminal(r.state) || busy}
              disabledReason={{
                reason: terminal(r.state)
                  ? '任务已结束，无需取消。可在配置实验页提交新的对照。'
                  : '取消请求正在提交。',
              }}
              onClick={() =>
                void action(async () => {
                  await api(`/requests/${r.id}/cancel`, {});
                  await run.refetch();
                })
              }
            >
              取消实验
            </Button>
          </Space>
          {r.error && <Alert type="error" message={r.error} />}
          {!terminal(r.state) && (
            <p>
              当前第 {(r.progress?.trial ?? 0) + 1} 轮 · {r.progress?.label ?? '准备中'}
              ，独立加载、预热和重复测量中。取消后会先清理进程，再核对功耗恢复。
            </p>
          )}
          <details>
            <summary>本次实验冻结配置</summary>
            <JsonView value={r.config} />
          </details>
        </Panel>
      )}
      {section === 'results' && sealed.data && <SystemEvidence receipt={sealed.data} />}
      <Panel title="L1 历史实验">
        <Table
          size="small"
          rowKey="id"
          dataSource={jobs.data?.items}
          pagination={{ pageSize: 8 }}
          scroll={{ x: 650 }}
          columns={[
            {
              title: '运行',
              render: (_, x) => (
                <Link
                  to={`/system/results?deployment=${encodeURIComponent(x.config.deployment.id)}&run=${x.id}`}
                >
                  {x.id}
                </Link>
              ),
            },
            { title: '源模型', render: (_, x) => x.config.deployment.name },
            { title: '状态', render: (_, x) => <Status state={x.state} /> },
            { title: '清理 / 恢复', dataIndex: 'cleanup' },
          ]}
        />
      </Panel>
    </>
  );
}
