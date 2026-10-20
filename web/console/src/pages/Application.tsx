import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Checkbox, Input, InputNumber, Select, Space, Table } from 'antd';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import Button, { type ActionBlock } from '../components/ActionButton';
import { api, terminal } from '../api/client';
import type { Deployment, Run } from '../api/types';
import { JsonView, PageHead, Panel, QueryState, Status } from '../components';
import { useRun } from '../hooks';
import PolicyEditor from '../features/application/PolicyEditor';
import ApplicationEvidence from '../features/application/ApplicationEvidence';
import type { ApplicationRun, Plan, Policy, Receipt } from '../features/application/types';
const policy: Policy = {
  routing: 'round_robin',
  short_input_chars: 128,
  result_cache: false,
  cache_ttl_s: 60,
  cache_max_entries: 32,
  cache_max_bytes: 4194304,
  prefix_reuse: false,
  input_transform: 'none',
  input_token_cap: 256,
  recent_messages: 4,
  output_cap: null,
  autoscale: false,
  min_replicas: 0,
  max_replicas: 1,
  idle_unload_ms: 1000,
};
const initial: Plan = {
  targets: [
    {
      id: 'primary',
      machine_id: 'local',
      deployment_id: 'qwen-reference',
      lifecycle: 'existing',
      role: 'default',
    },
  ],
  requests: [
    {
      id: 'r1',
      prompt: 'Once upon a time, there was a little fox.',
      tenant: 'default',
      arrival_ms: 0,
      max_output_tokens: 32,
    },
    {
      id: 'r2',
      prompt: 'Once upon a time, there was a little fox.',
      tenant: 'default',
      arrival_ms: 150,
      max_output_tokens: 32,
    },
    {
      id: 'r3',
      prompt: 'Explain why the sky is blue in one sentence.',
      tenant: 'default',
      arrival_ms: 300,
      max_output_tokens: 32,
    },
    {
      id: 'r4',
      prompt: 'Explain why the sky is blue in one sentence.',
      tenant: 'default',
      arrival_ms: 450,
      max_output_tokens: 32,
    },
  ],
  baseline: policy,
  candidate: { ...policy, routing: 'cache_affinity', result_cache: true },
  concurrency: 2,
  request_deadline_s: 90,
  timeout_s: 300,
  fixed_output_length: false,
};
export default function Application() {
  const { section = 'run' } = useParams();
  const [params, setParams] = useSearchParams();
  const [plan, setPlan] = useState<Plan>(() => ({
    ...initial,
    targets: [
      { ...initial.targets[0], deployment_id: params.get('deployment') ?? 'qwen-reference' },
    ],
  }));
  const [targetJson, setTargetJson] = useState('');
  const [requestJson, setRequestJson] = useState('');
  const [busy, setBusy] = useState(false),
    [error, setError] = useState('');
  const ds = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 2500,
  });
  const overview = useQuery({
    queryKey: ['application-overview'],
    queryFn: () => api<{ pending: number; benchmark_owner?: unknown }>('/overview'),
    refetchInterval: 1500,
  });
  const catalog = useQuery({
    queryKey: ['application-catalog'],
    queryFn: () =>
      api<{ limits: string[]; quarantined: boolean; lease_owner?: string }>('/application/catalog'),
    refetchInterval: 3000,
  });
  const targets = useQuery({
    queryKey: ['application-targets'],
    queryFn: () =>
      api<{ nodes: { id: string; name: string; transport: string }[] }>('/application/targets'),
  });
  const jobs = useQuery({
    queryKey: ['application-jobs'],
    queryFn: () => api<{ items: ApplicationRun[] }>('/application/jobs'),
    refetchInterval: 2500,
  });
  const run = useRun(params.get('run') ?? undefined);
  const r = run.data as ApplicationRun | undefined;
  const receipt = useQuery({
    queryKey: ['application-receipt', r?.id, r?.state],
    queryFn: () => api<Receipt>(`/runs/${r!.id}/application`),
    enabled: !!r && terminal(r.state),
  });
  const local = plan.targets.find((t) => (t.machine_id ?? 'local') === 'local');
  const selected = ds.data?.items.find((d) => d.id === local?.deployment_id);
  const device = `/devices#deployment-${encodeURIComponent(local?.deployment_id ?? '')}`;
  const active =
    !!overview.data?.pending || !!overview.data?.benchmark_owner || (!!r && !terminal(r.state));
  const block: ActionBlock | undefined =
    busy || active
      ? {
          reason: '当前任务、节点租约或设备清理尚未完成。请在运行记录等待或取消。',
          next: { to: '/runs', label: '查看运行记录' },
        }
      : catalog.data?.quarantined
        ? {
            reason:
              '上次 L10 节点清理未确认。请检查该实验的子任务、租约和托管模型状态，再恢复服务。',
            next: { to: '/application/targets', label: '查看目标与恢复说明' },
          }
        : local && !selected
          ? {
              reason: '所选本地源模型尚未登记。请先下载并登记模型。',
              next: { to: '/models', label: '登记模型' },
            }
          : local && local.lifecycle === 'existing' && selected?.state !== 'ready'
            ? {
                reason: '已有副本模式需要先加载所选模型。请在设备与部署页完成。',
                next: { to: device, label: '加载模型' },
              }
            : local &&
                local.lifecycle === 'managed' &&
                ds.data?.items.some(
                  (d) => d.provider === 'pytorch' && !['unloaded', 'failed'].includes(d.state),
                )
              ? {
                  reason:
                    '托管副本要求设备上没有其他驻留模型，实验只接管自己加载的模型。请先在设备页卸载。',
                  next: { to: device, label: '卸载驻留模型' },
                }
              : undefined;
  const update = (key: string, value: string) =>
    setParams((p) => {
      const n = new URLSearchParams(p);
      n.set(key, value);
      return n;
    });
  const changes = (patch: Partial<Plan>) => setPlan((p) => ({ ...p, ...patch }));
  async function action(f: () => Promise<void> | void) {
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
        kicker="L10"
        title="服务与应用优化"
        description="固定多请求负载，对照路由、结果缓存、输入与输出预算，以及已登记节点上的模型副本加载和回收。"
      />
      <Space wrap style={{ marginBottom: 20 }}>
        {[
          ['run', '策略与负载'],
          ['targets', '目标与副本'],
          ['results', '结果与机制'],
        ].map(([key, name]) => (
          <Link
            key={key}
            aria-current={section === key ? 'page' : undefined}
            to={`/application/${key}?${params}`}
          >
            {name}
          </Link>
        ))}
      </Space>
      <QueryState
        error={ds.error ?? catalog.error ?? jobs.error ?? targets.error ?? receipt.error}
      />
      {error && <Alert type="error" message={error} showIcon />}
      {section === 'run' && (
        <>
          <Panel title="执行目标">
            <Select
              aria-label="L10 本地模型"
              style={{ width: 360, maxWidth: '100%' }}
              value={local?.deployment_id}
              options={ds.data?.items
                .filter((d) => d.provider === 'pytorch')
                .map((d) => ({ value: d.id, label: `${d.name} · ${d.state}` }))}
              onChange={(deployment_id) => {
                changes({
                  targets: [
                    {
                      id: 'primary',
                      machine_id: 'local',
                      deployment_id,
                      lifecycle: 'existing',
                      role: 'default',
                    },
                  ],
                });
                update('deployment', deployment_id);
              }}
            />
            {local && (
              <Select
                aria-label="L10 副本生命周期"
                value={local.lifecycle ?? 'existing'}
                style={{ width: 240 }}
                options={[
                  { value: 'existing', label: '使用已有驻留模型' },
                  { value: 'managed', label: '实验托管 / 加载和回收' },
                ]}
                onChange={(lifecycle) =>
                  changes({
                    targets: plan.targets.map((t) => (t.id === local.id ? { ...t, lifecycle } : t)),
                  })
                }
              />
            )}
            <Space wrap>
              <Link to={device}>加载、卸载与切换模型 →</Link>
              <Link to={`/application/targets?${params}`}>配置多节点和托管副本 →</Link>
            </Space>
            <p>
              默认使用已有驻留模型，不自动加载或卸载。实验中一个节点一次处理一个请求；节点内连续合批请使用
              L8 多请求调度实验。
            </p>
          </Panel>
          <Panel title="固定多请求负载">
            <Space wrap>
              <label>
                网关并发{' '}
                <InputNumber
                  aria-label="L10 网关并发"
                  min={1}
                  max={8}
                  value={plan.concurrency}
                  onChange={(n) => changes({ concurrency: n ?? 2 })}
                />
              </label>
              <label>
                每请求时限秒{' '}
                <InputNumber
                  min={5}
                  max={300}
                  value={plan.request_deadline_s}
                  onChange={(n) => changes({ request_deadline_s: n ?? 90 })}
                />
              </label>
              <label>
                总时限秒{' '}
                <InputNumber
                  min={30}
                  max={900}
                  value={plan.timeout_s}
                  onChange={(n) => changes({ timeout_s: n ?? 300 })}
                />
              </label>
              <Checkbox
                checked={plan.fixed_output_length}
                onChange={(e) => changes({ fixed_output_length: e.target.checked })}
              >
                固定输出长度
              </Checkbox>
              <Button
                onClick={() => {
                  changes({ requests: initial.requests });
                  setRequestJson('');
                }}
              >
                恢复典型负载
              </Button>
              <Button
                onClick={() => {
                  changes({
                    requests: [0, 1].map((i) => ({
                      id: `chat-${i}`,
                      tenant: 'default',
                      arrival_ms: i * 150,
                      max_output_tokens: 32,
                      messages: [
                        { role: 'system', content: 'Answer clearly in one sentence.' },
                        { role: 'user', content: 'My favorite animal is a fox.' },
                        { role: 'assistant', content: 'A fox is a small mammal.' },
                        { role: 'user', content: 'Explain why the sky is blue.' },
                      ],
                    })),
                  });
                  setRequestJson('');
                }}
              >
                载入典型多轮聊天
              </Button>
            </Space>
            <p>
              默认 4
              个请求包含重复输入，可直接比较缓存命中与实际计算量。支持不同租户和到达时间；同租户命中不允许跨租户复用。输入作为冻结实验计划保存。
            </p>
            <Table
              size="small"
              rowKey="id"
              pagination={false}
              scroll={{ x: 760 }}
              dataSource={plan.requests}
              columns={[
                { title: '请求', dataIndex: 'id' },
                { title: '租户', dataIndex: 'tenant' },
                { title: '到达 ms', dataIndex: 'arrival_ms' },
                { title: '输出预算', dataIndex: 'max_output_tokens' },
                { title: '输入', render: (_, x) => x.prompt ?? `${x.messages?.length} 条聊天消息` },
              ]}
            />
            <details>
              <summary>
                编辑请求 JSON（prompt 或 messages、tenant、arrival_ms、max_output_tokens）
              </summary>
              <Input.TextArea
                aria-label="L10 请求 JSON"
                rows={10}
                value={requestJson || JSON.stringify(plan.requests, null, 2)}
                onChange={(e) => setRequestJson(e.target.value)}
              />
              <Button
                onClick={() =>
                  void action(async () => {
                    const value = JSON.parse(requestJson || JSON.stringify(plan.requests));
                    if (!Array.isArray(value)) throw new Error('请求必须是 JSON 数组');
                    const checked = await api<Plan>('/application/validate', {
                      plan: { ...plan, requests: value },
                    });
                    changes({ requests: checked.requests });
                  })
                }
              >
                应用负载 JSON
              </Button>
            </details>
          </Panel>
          <PolicyEditor
            title="基线 A"
            value={plan.baseline ?? policy}
            onChange={(baseline) => changes({ baseline })}
          />
          <PolicyEditor
            title="候选 B"
            value={plan.candidate ?? policy}
            onChange={(candidate) => changes({ candidate })}
          />
          <Button
            type="primary"
            loading={busy}
            disabled={!!block}
            disabledReason={block}
            onClick={() =>
              void action(async () => {
                const next = await api<Run>('/application/jobs', { plan });
                update('run', next.id);
              })
            }
          >
            运行 L10 服务对照
          </Button>
        </>
      )}
      {section === 'targets' && (
        <>
          <Panel title="节点与模型副本">
            <Space wrap>
              <Link to="/machines">管理 SSH / HTTP 节点连接 →</Link>
              <Link to={device}>本机模型生命周期 →</Link>
            </Space>
            <p>
              远程节点来自当前执行端的注册表，SSH
              凭据也属于该执行端。每个物理节点只配置一个目标；别名指向同一节点会被拒绝。
            </p>
            <Table
              size="small"
              rowKey="id"
              pagination={false}
              dataSource={targets.data?.nodes}
              columns={[
                { title: '节点 ID', dataIndex: 'id' },
                { title: '名称', dataIndex: 'name' },
                { title: '连接', dataIndex: 'transport' },
              ]}
            />
            <Input.TextArea
              aria-label="L10 目标 JSON"
              rows={10}
              value={targetJson || JSON.stringify(plan.targets, null, 2)}
              onChange={(e) => setTargetJson(e.target.value)}
            />
            <Button
              onClick={() =>
                void action(async () => {
                  const value = JSON.parse(targetJson || JSON.stringify(plan.targets));
                  if (!Array.isArray(value) || !value.length) throw new Error('至少需要一个目标');
                  const checked = await api<Plan>('/application/validate', {
                    plan: { ...plan, targets: value },
                  });
                  changes({ targets: checked.targets });
                })
              }
            >
              应用目标 JSON
            </Button>
            <p>
              字段：id、machine_id、deployment_id、lifecycle（existing / managed）、role（default /
              small / large）。existing 要求模型已加载；managed
              要求模型未加载，实验会使用节点默认精度加载，结束时核对 epoch 和加载 run ID 后回收。
            </p>
            <p>
              原生轮询、空闲优先与结果缓存感知要求副本的模型、Tokenizer
              和精度一致。不同模型请用长度模型路由，并为不同节点标记 small / large。
            </p>
          </Panel>
          <Panel title="支持边界与故障恢复">
            <ul>{catalog.data?.limits.map((x) => <li key={x}>{x}</li>)}</ul>
            <p>
              控制端异常退出后远端模型或租约可能仍在。请从根实验的 backend.submitting /
              backend.accepted 事件查找稳定提交键和子 run ID，先确认取消和清理，再释放该 owner 的
              /application/leases/release。恢复前不能删除记录或把未知状态当作成功。
            </p>
            <JsonView
              value={{
                targets: plan.targets,
                lease_owner: catalog.data?.lease_owner,
                quarantined: catalog.data?.quarantined,
              }}
            />
          </Panel>
        </>
      )}
      {r && (
        <Panel title="当前 L10 实验">
          <Space wrap>
            <Status state={r.state} />
            <code>{r.id}</code>
            <Link
              to={`/application/results?${new URLSearchParams({ ...Object.fromEntries(params), run: r.id })}`}
            >
              查看结果与机制 →
            </Link>
            <Button
              loading={busy}
              disabled={terminal(r.state) || busy}
              disabledReason={{
                reason: terminal(r.state)
                  ? '实验已结束，无需取消。可返回策略与负载页提交新实验。'
                  : '取消操作正在提交。',
              }}
              onClick={() =>
                void action(async () => {
                  await api(`/requests/${r.id}/cancel`, {});
                  await run.refetch();
                })
              }
            >
              取消 L10 实验
            </Button>
          </Space>
          {r.error && <Alert type="error" message={r.error} />}
          {!terminal(r.state) && (
            <p>
              当前阶段：{r.progress?.stage ?? '准备节点'}
              。关闭页面不会终止实验；取消后会等待子推理清理，并回收本实验实际加载的副本。
            </p>
          )}
          <details>
            <summary>冻结配置</summary>
            <JsonView value={r.config} />
          </details>
        </Panel>
      )}
      {section === 'results' && receipt.data && <ApplicationEvidence receipt={receipt.data} />}
      <Panel title="L10 历史实验">
        <Table
          size="small"
          rowKey="id"
          dataSource={jobs.data?.items}
          pagination={{ pageSize: 8 }}
          scroll={{ x: 650 }}
          columns={[
            {
              title: '运行',
              render: (_, x) => <Link to={`/application/results?run=${x.id}`}>{x.id}</Link>,
            },
            { title: '状态', render: (_, x) => <Status state={x.state} /> },
            { title: '清理', dataIndex: 'cleanup' },
          ]}
        />
      </Panel>
    </>
  );
}
