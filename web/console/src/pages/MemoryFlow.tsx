import { useQuery } from '@tanstack/react-query';
import { Alert, Button, Empty, Select, Space, Tag } from 'antd';
import { Link, useSearchParams } from 'react-router-dom';
import { api, base, date } from '../api/client';
import type { Deployment, RunPage } from '../api/types';
import { PageHead, Panel, QueryState } from '../components';
import type { Resources } from '../features/memory/types';
import type { Dataflow, DeploymentInventory, ObservedRun } from '../features/memory-flow/types';
import MemoryTopology from '../features/memory-flow/MemoryTopology';
import WeightMap from '../features/memory-flow/WeightMap';
import KVMap from '../features/memory-flow/KVMap';
import RequestFlow from '../features/memory-flow/RequestFlow';
import '../features/memory-flow/styles.css';

export default function MemoryFlow() {
  const [params, setParams] = useSearchParams();
  const resources = useQuery({
    queryKey: ['resources'],
    queryFn: () => api<Resources>('/resources'),
    refetchInterval: 10000,
  });
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 10000,
  });
  const runs = useQuery({
    queryKey: ['memory-flow-runs'],
    queryFn: () => api<RunPage>('/runs?limit=100'),
    staleTime: 15000,
  });
  const candidates = ((runs.data?.items ?? []) as ObservedRun[]).filter(
    (run) =>
      run.kind === 'generate' &&
      run.state === 'completed' &&
      (run.config.observation_mode === 'operators' ||
        run.config.observation_mode === 'basic' ||
        run.metrics.observation?.mode === 'operators' ||
        run.metrics.observation?.mode === 'basic'),
  );
  const fallback =
    candidates.find(
      (run) =>
        run.config.observation_mode === 'operators' ||
        run.metrics.observation?.profile?.trace_available,
    ) ?? candidates[0];
  const runId = params.get('run') || fallback?.id;
  const runQuery = useQuery({
    queryKey: ['run', runId],
    queryFn: () => api<ObservedRun>(`/runs/${runId}`),
    enabled: !!runId,
    staleTime: 30000,
  });
  const run = runQuery.data;
  const deployment = deployments.data?.items.find(
    (entry) => entry.id === run?.config.deployment.id,
  );
  const historical = !deployment?.epoch || deployment.epoch !== run?.config.epoch;
  const loadId = run?.config.load_run_id;
  const useCurrentInventory = !!deployment && !loadId && !historical;
  const inventory = useQuery({
    queryKey: ['memory-inventory', deployment?.id, deployment?.epoch],
    queryFn: () => api<DeploymentInventory>(`/deployments/${deployment!.id}/memory-inventory`),
    enabled: useCurrentInventory,
    staleTime: 30000,
  });
  const load = useQuery({
    queryKey: ['run', loadId],
    queryFn: () => api<ObservedRun>(`/runs/${loadId}`),
    enabled: !!loadId,
    staleTime: Infinity,
  });
  const dataflow = useQuery({
    queryKey: ['dataflow', runId],
    queryFn: () => api<Dataflow>(`/runs/${runId}/dataflow`),
    enabled: !!runId,
    staleTime: Infinity,
  });
  // The immutable load record belongs to this request's exact model version.
  const weights = loadId
    ? load.data?.result?.parameter_inventory
    : useCurrentInventory
      ? inventory.data?.parameters
      : undefined;
  const weightTime =
    weights?.sampled_at ?? (loadId ? load.data?.updated_at : inventory.data?.sampled_at);
  const options = candidates.map((entry) => ({
    value: entry.id,
    label: `${date(entry.created_at)} · ${entry.config.observation_mode === 'operators' ? '算子详录' : '基础观测'} · ${entry.id.slice(0, 14)}`,
  }));
  if (run && !options.some((entry) => entry.value === run.id))
    options.unshift({ value: run.id, label: `${date(run.created_at)} · ${run.id.slice(0, 14)}` });
  const selectedLocal = !run || run.config.deployment.provider === 'pytorch';
  return (
    <div className="memory-flow-page">
      <PageHead
        kicker="MEMORY & DATA FLOW"
        title="内存与数据流"
        description="从真实请求出发，看清权重与 KV 的结构、CPU/GPU 的逻辑分工，以及 trace 实际捕获到的数据传输。"
        extra={
          <Button
            onClick={() => {
              void runs.refetch();
              void resources.refetch();
            }}
          >
            刷新证据
          </Button>
        }
      />
      <Panel className="mf-run-picker">
        <div className="mf-run-picker-inner">
          <div>
            <label className="field-label" htmlFor="mf-run-select">
              选择真实推理记录
            </label>
            <Select
              id="mf-run-select"
              aria-label="数据流推理记录"
              loading={runs.isPending}
              value={runId}
              options={options}
              onChange={(value) => setParams({ run: value })}
              style={{ width: '100%' }}
              placeholder="寻找最近有观测的请求"
            />
          </div>
          <Space wrap>
            {run && (
              <>
                <Tag color={historical ? 'orange' : 'green'}>
                  {historical ? '历史部署版本' : '当前部署版本的请求'}
                </Tag>
                <Link to={`/runs/${run.id}`}>查看完整运行 →</Link>
                <Button
                  href={`${base}/runs/${run.id}/trace?format=chrome`}
                  disabled={!run.metrics.observation?.profile?.trace_available}
                >
                  下载 Trace
                </Button>
              </>
            )}
          </Space>
        </div>
        <p className="mf-caption">
          默认打开最近的算子详录，无需重新推理。系统内存为当前采样，权重和 KV
          带各自时间；历史快照不会被标成当前驻留。
        </p>
        <QueryState
          loading={runs.isPending || (runQuery.isPending && !!runId)}
          error={runs.error ?? runQuery.error}
        />
        {historical && run && (
          <Alert
            type="info"
            showIcon
            message="正在回放历史请求的真实证据"
            description={
              loadId
                ? '当前模型可能已卸载或重载。权重从这次请求绑定的加载记录读取；KV 来自该请求结束前快照。不会为回放自动加载模型。'
                : '当前模型可能已卸载或重载。该旧请求没有绑定加载记录，缺失的权重证据不会用当前模型替代。KV 仅展示该请求保存的快照。'
            }
          />
        )}
        {run?.kind === 'generate' &&
          (!run.metrics.observation || run.metrics.observation.mode === 'off') && (
            <Alert
              className="section-gap"
              type="info"
              showIcon
              message="本次请求未采集阶段与 allocator 观测"
              description="已保存的权重或 KV 元数据仍可展示；未采集的主机阶段、GPU 活动不能事后补录。请从上方选择带观测的推理记录。"
            />
          )}
        {run && run.kind !== 'generate' && (
          <Alert
            className="section-gap"
            type="warning"
            message="此 ID 对应的不是推理请求，请从上方选择推理记录。"
          />
        )}
      </Panel>
      <nav className="mf-section-nav" aria-label="内存与数据流图形导航">
        <a href="#memory-topology-section">01 统一内存</a>
        <a href="#weight-map-section">02 权重结构</a>
        <a href="#kv-map-section">03 KV 层结构</a>
        <a href="#request-dataflow-section">04 请求数据流</a>
      </nav>
      {!runId && !runs.isPending && (
        <Panel>
          <Empty description="最近 100 条记录还没有带观测的完整推理。到推理工作台选择“算子详录”，执行一次请求后返回。" />
          <Link to="/playground">前往推理工作台 →</Link>
        </Panel>
      )}
      {selectedLocal ? (
        <section id="memory-topology-section">
          <MemoryTopology
            resources={resources.data}
            observation={run?.metrics.observation}
            historical={historical}
          />
        </section>
      ) : (
        <Alert
          className="section-gap"
          type="info"
          message="此记录来自外部推理服务；本机 Jetson 的内存不能当作外部服务的内存结构。"
        />
      )}
      <QueryState error={resources.error ?? inventory.error ?? load.error} />
      {run?.kind === 'generate' && (
        <>
          <section id="weight-map-section">
            <WeightMap
              inventory={weights}
              source={
                loadId
                  ? '绑定加载记录 · 权重快照'
                  : useCurrentInventory
                    ? '当前版本 · 加载时权重快照'
                    : '缺少绑定版本的权重证据'
              }
              sampledAt={weightTime}
            />
          </section>
          <section id="kv-map-section">
            <KVMap inventory={run.metrics.kv_inventory} />
          </section>
          <section id="request-dataflow-section">
            <RequestFlow
              run={run}
              dataflow={dataflow.data}
              error={dataflow.error}
              loading={dataflow.isPending}
            />
          </section>
        </>
      )}
    </div>
  );
}
