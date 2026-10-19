import Button from '../components/ActionButton';
import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Collapse, Input, Select, Space, Table, Tabs, Tag } from 'antd';
import { Link, useSearchParams } from 'react-router-dom';
import { api, date, terminal } from '../api/client';
import type { Deployment, Run } from '../api/types';
import { JsonView, PageHead, Panel, QueryState, Status } from '../components';
import ComparisonReport from '../features/experiments/ComparisonReport';
import {
  armLabel,
  initialSpec,
  layerLabels,
  pendingSlots,
  roleLabel,
  slotKey,
  slots,
} from '../features/experiments/model';
import type {
  Catalogue,
  ExperimentJobRequest,
  Job,
  Plan,
  Report,
  Slot,
} from '../features/experiments/types';

export default function Optimization() {
  const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const id = params.get('experiment') ?? '';
  const [draft, setDraft] = useState(JSON.stringify(initialSpec, null, 2));
  const [selected, setSelected] = useState('');
  const [error, setError] = useState('');
  const [registering, setRegistering] = useState(false);
  const [executing, setExecuting] = useState(false);
  const controller = useRef<AbortController | null>(null);
  useEffect(
    () => () => {
      controller.current?.abort();
    },
    [id],
  );
  const listing = useQuery({
    queryKey: ['optimization-plans'],
    queryFn: () =>
      api<{ items: Plan[]; total: number; next_offset: number | null }>('/experiments?limit=50'),
  });
  const catalogue = useQuery({
    queryKey: ['optimization-catalog'],
    queryFn: () => api<Catalogue>('/optimization-catalog'),
    staleTime: 60000,
  });
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 2000,
  });
  const deployment =
    deployments.data?.items.find((item) => item.id === selected) ?? deployments.data?.items[0];
  const plan = useQuery({
    queryKey: ['optimization-plan', id],
    queryFn: () => api<Plan>(`/experiments/${encodeURIComponent(id)}/plan`),
    enabled: !!id,
    retry: false,
  });
  const jobs = useQuery({
    queryKey: ['optimization-jobs', id],
    queryFn: () => api<{ items: Job[] }>(`/experiments/${encodeURIComponent(id)}/jobs`),
    enabled: !!plan.data && !plan.isError,
    refetchInterval: 2000,
    retry: false,
  });
  const evidenceVersion = (jobs.data?.items ?? [])
    .filter((item) => terminal(item.state))
    .map((item) => `${item.run_id}:${item.state}`)
    .join('|');
  const tasksActive = jobs.data?.items.some((item) => !terminal(item.state));
  const report = useQuery({
    queryKey: ['optimization-report', id, evidenceVersion],
    queryFn: () => api<Report>(`/experiments/${encodeURIComponent(id)}/report`),
    enabled: !!plan.data && !!jobs.data && !plan.isError && !executing && !tasksActive,
    retry: false,
  });
  const locked = executing || tasksActive;
  const refresh = async () => {
    await Promise.allSettled([
      client.invalidateQueries({ queryKey: ['optimization-jobs', id] }),
      client.invalidateQueries({ queryKey: ['optimization-report', id] }),
      client.invalidateQueries({ queryKey: ['runs'] }),
    ]);
  };
  const register = async () => {
    setRegistering(true);
    setError('');
    try {
      const created = await api<Plan>('/experiments/plans', { spec: JSON.parse(draft) });
      client.setQueryData(['optimization-plan', created.experiment_id], created);
      await client.invalidateQueries({ queryKey: ['optimization-plans'] });
      setParams({ experiment: created.experiment_id });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRegistering(false);
    }
  };
  const execute = async (requested: Slot[]) => {
    if (!deployment?.epoch || !plan.data || !requested.length || locked) return;
    const abort = new AbortController();
    controller.current = abort;
    setExecuting(true);
    setError('');
    try {
      // One submission at a time. Navigation/disconnection never retries a write.
      for (const slot of requested) {
        if (abort.signal.aborted) break;
        const payload: ExperimentJobRequest = {
          ...slot,
          deployment_id: deployment.id,
          expected_epoch: deployment.epoch,
        };
        let current = await api<Run>(`/experiments/${id}/jobs`, payload);
        await refresh();
        while (!terminal(current.state) && !abort.signal.aborted) {
          await new Promise<void>((resolve) => {
            const finish = () => {
              clearTimeout(timer);
              abort.signal.removeEventListener('abort', finish);
              resolve();
            };
            const timer = setTimeout(finish, 1500);
            abort.signal.addEventListener('abort', finish, { once: true });
          });
          if (!abort.signal.aborted) current = await api<Run>(`/runs/${current.id}`);
        }
        // Cancellation can arrive before the submission response carries its ID.
        // A navigation abort only stops future submissions; explicit cancellation
        // waits for the accepted ID and then requests worker cleanup.
        if (abort.signal.reason === 'user_cancel' && !terminal(current.state)) {
          await api(`/requests/${current.id}/cancel`, {});
        }
        await refresh();
        if (!abort.signal.aborted && current.state !== 'completed')
          throw new Error(
            `运行 ${current.id} ${current.state}：${current.error ?? '未完成；已停止后续提交'}`,
          );
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setExecuting(false);
      await refresh();
    }
  };
  const cancel = () => controller.current?.abort('user_cancel');
  const canExecute =
    !locked && deployment?.provider === 'pytorch' && deployment.state === 'ready' && !!jobs.data;
  const rows = plan.data
    ? slots(plan.data).map((slot) => ({
        ...slot,
        job: jobs.data?.items.find((item) => slotKey(item) === slotKey(slot)),
      }))
    : [];
  return (
    <>
      <PageHead
        kicker="OPTIMIZATION EXPERIMENTS"
        title="优化实验"
        description="冻结配置，核对实际生效，分别比较质量、性能和机制变化。"
        extra={
          <Button
            disabledReason={{
              reason: locked
                ? '当前计划或任务正在执行，请等待完成后再操作。'
                : '当前任务执行中，请等待完成后再新建计划。',
              next: { to: '/runs', label: '查看当前任务' },
            }}
            disabled={!!locked}
            onClick={() => setParams({})}
          >
            新建计划
          </Button>
        }
      />
      <QueryState error={listing.error ?? catalogue.error ?? deployments.error} />
      {error && (
        <Alert
          className="section-gap"
          type="error"
          showIcon
          message="操作未完成"
          description={error}
        />
      )}
      <div className="experiment-layout">
        <aside className="experiment-library">
          <h2>实验计划</h2>
          <Select
            aria-label="选择优化实验"
            style={{ width: '100%' }}
            value={id || undefined}
            placeholder="选择已冻结的实验"
            disabled={!!locked}
            onChange={(experiment) => setParams({ experiment })}
            options={listing.data?.items.map((item) => ({
              value: item.experiment_id,
              label: `${date(item.registered_at)} · ${item.experiment_id} · ${item.spec.declared_changes.join(', ')}`,
            }))}
          />
          {listing.data && (
            <div className="help-text">
              显示最近 {listing.data.items.length} / {listing.data.total}{' '}
              个计划；更早的实验可通过其详情链接打开。
            </div>
          )}
          <div className="experiment-plan-list">
            {listing.data?.items.slice(0, 8).map((item) => (
              <Link
                key={item.experiment_id}
                to={`/optimization?experiment=${item.experiment_id}`}
                aria-current={id === item.experiment_id ? 'page' : undefined}
                onClick={(event) => {
                  if (locked) event.preventDefault();
                }}
                aria-disabled={!!locked}
              >
                <strong>{item.spec.declared_changes.join(', ')}</strong>
                <code>{item.experiment_id.slice(0, 20)}</code>
                <small>{date(item.registered_at)}</small>
              </Link>
            ))}
          </div>
          <p className="help-text">注册后冻结配置。打开已有实验只读取状态，不重复提交任务。</p>
        </aside>
        <div className="experiment-notebook">
          {!id ? (
            <Panel
              title="对照配置草稿"
              subtitle="默认是最小闭环 fixture：一对性能、机制诊断与质量，共六个任务。"
            >
              <Alert
                type="info"
                showIcon
                message="注册后配置与阈值冻结"
                description="评测文本会保存在服务器的私有实验目录。最多 4 个样本，2–64 个 token；超限拒绝，不静默截断。默认算术样本用于检查功能闭环，任务质量可使用自定义评测集。"
              />
              <label className="field-label" htmlFor="experiment-spec">
                实验计划 JSON
              </label>
              <Input.TextArea
                id="experiment-spec"
                className="mono experiment-editor"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                autoSize={{ minRows: 16, maxRows: 24 }}
                spellCheck={false}
              />
              <Space className="section-gap">
                <Button type="primary" loading={registering} onClick={() => void register()}>
                  冻结并注册计划
                </Button>
                <Button onClick={() => setDraft(JSON.stringify(initialSpec, null, 2))}>
                  恢复最小示例
                </Button>
              </Space>
            </Panel>
          ) : (
            <>
              <QueryState loading={plan.isPending} error={plan.error} />
              {plan.data && (
                <>
                  <Panel title="冻结配置与执行" subtitle={plan.data.experiment_id}>
                    <div className="help-text mono">配置 SHA-256：{plan.data.spec_sha256}</div>
                    <Collapse
                      items={[
                        {
                          key: 'spec',
                          label: '查看注册配置（只读）',
                          children: <JsonView value={plan.data.spec} />,
                        },
                      ]}
                    />
                    <div className="experiment-actions">
                      <Select
                        aria-label="实验执行部署"
                        value={deployment?.id}
                        disabled={!!locked}
                        onChange={setSelected}
                        options={deployments.data?.items.map((item) => ({
                          value: item.id,
                          label: `${item.name} · ${item.state}`,
                        }))}
                        style={{ minWidth: 240 }}
                      />
                      {deployment && <Status state={deployment.state} />}
                      <Button
                        disabledReason={{
                          reason: locked
                            ? '当前计划或任务正在执行，请等待完成后再操作。'
                            : deployment?.provider !== 'pytorch'
                              ? '此计划需要本地 PyTorch 部署，请先选择本地模型。'
                              : '部署已加载或正在切换。已就绪的部署可直接执行；其他状态请先等待完成。',
                          next: { to: '/devices', label: '检查执行部署' },
                        }}
                        disabled={
                          !!locked ||
                          deployment?.provider !== 'pytorch' ||
                          deployment.state !== 'unloaded'
                        }
                        onClick={() => {
                          setError('');
                          void api(`/deployments/${deployment!.id}/load`, {})
                            .then(() => client.invalidateQueries({ queryKey: ['deployments'] }))
                            .catch((e) => setError(e.message));
                        }}
                      >
                        加载执行部署
                      </Button>
                      <Button
                        type="primary"
                        loading={executing}
                        disabledReason={{
                          reason: locked
                            ? '当前计划或任务正在执行，请等待完成后再操作。'
                            : !plan.data
                              ? '请先在本页登记或选择一个工程计划。'
                              : !canExecute
                                ? '执行部署尚未就绪，请先加载执行部署。'
                                : '此计划没有未提交槽位，请到任务表查看已提交结果。',
                          next: { to: '/devices', label: '检查执行部署' },
                        }}
                        disabled={
                          !canExecute ||
                          !plan.data ||
                          !pendingSlots(plan.data, jobs.data?.items ?? []).length
                        }
                        onClick={() => void execute(pendingSlots(plan.data!, jobs.data!.items))}
                      >
                        依次执行未提交任务
                      </Button>
                      {executing && (
                        <Button danger onClick={() => void cancel()}>
                          停止后续提交并取消当前任务
                        </Button>
                      )}
                      <Button onClick={() => void refresh()}>刷新报告</Button>
                    </div>
                    <p className="help-text">
                      同一驻留 worker 串行执行，性能样本不满足独立进程验收。Attention
                      必须匹配已加载部署，否则后端拒绝并要求重新加载。中断或失败的槽位保持原记录，需要重新实验时注册新计划。
                    </p>
                    <QueryState error={jobs.error} />
                    <Table
                      size="small"
                      rowKey={slotKey}
                      loading={jobs.isPending}
                      scroll={{ x: 780 }}
                      pagination={{ pageSize: 10 }}
                      dataSource={rows}
                      columns={[
                        {
                          title: '角色',
                          dataIndex: 'role',
                          render: (role: Slot['role']) => roleLabel[role],
                        },
                        { title: '配对', dataIndex: 'pair', render: (value: number) => value + 1 },
                        {
                          title: '方案',
                          dataIndex: 'arm',
                          render: (arm: Slot['arm']) => armLabel[arm],
                        },
                        {
                          title: '状态',
                          render: (_, row) =>
                            row.job ? <Status state={row.job.state} /> : '未提交',
                        },
                        {
                          title: '运行与证据',
                          render: (_, row) =>
                            row.job ? (
                              <Link to={`/runs/${row.job.run_id}?tab=experiment`}>
                                {row.job.run_id.slice(0, 18)}
                              </Link>
                            ) : (
                              <Button
                                size="small"
                                disabledReason={{
                                  reason: locked
                                    ? '当前计划或任务正在执行，请等待完成后再操作。'
                                    : '执行部署未就绪，请先在本页加载执行部署后再运行槽位。',
                                  next: { to: '/devices', label: '查看部署与加载状态' },
                                }}
                                disabled={!canExecute}
                                onClick={() => void execute([row])}
                              >
                                运行此槽位
                              </Button>
                            ),
                        },
                        {
                          title: '错误 / 清理',
                          render: (_, row) => row.job?.error ?? row.job?.cleanup ?? '—',
                        },
                        {
                          title: '控制',
                          render: (_, row) =>
                            row.job && !terminal(row.job.state) ? (
                              <Button
                                size="small"
                                danger
                                onClick={() => {
                                  void api(`/requests/${row.job!.run_id}/cancel`, {})
                                    .then(refresh)
                                    .catch((e) => setError(e.message));
                                }}
                              >
                                取消此任务
                              </Button>
                            ) : (
                              '—'
                            ),
                        },
                      ]}
                    />
                  </Panel>
                  <Panel title="质量与机制对照报告" subtitle="判定、完整度和来源由后端返回。">
                    <QueryState loading={report.isPending} error={report.error} />
                    {locked ? (
                      <Alert type="info" message="任务执行中；终态与清理确认后生成新版报告。" />
                    ) : (
                      report.data && <ComparisonReport report={report.data} />
                    )}
                  </Panel>
                </>
              )}
            </>
          )}
          {catalogue.data && (
            <details>
              <summary>浏览十层优化能力目录</summary>
              <Panel
                title="十层优化能力目录"
                subtitle="已接入表示存在控制路径，是否生效仍需本次执行凭据。"
              >
                <Tabs
                  items={layerLabels.map((label, index) => ({
                    key: String(index + 1),
                    label: `${index + 1}. ${label}`,
                    children: (
                      <Table
                        size="small"
                        rowKey="id"
                        pagination={false}
                        dataSource={catalogue.data!.items.filter((item) =>
                          item.layers.includes(index + 1),
                        )}
                        columns={[
                          { title: '措施', dataIndex: 'id' },
                          {
                            title: '实现状态',
                            render: (_, item) => (
                              <Tag>
                                {item.implementation_status === 'not_integrated'
                                  ? '未接入'
                                  : '已接入 · 待执行验证'}
                              </Tag>
                            ),
                          },
                          {
                            title: '作用域',
                            dataIndex: 'scope',
                            render: (scope?: string) => scope ?? '—',
                          },
                          { title: '运行能力', dataIndex: 'runtime_capability' },
                        ]}
                      />
                    ),
                  }))}
                />
              </Panel>
            </details>
          )}
        </div>
      </div>
    </>
  );
}
