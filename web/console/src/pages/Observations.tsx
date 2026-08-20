import { useQuery } from '@tanstack/react-query';
import { Alert, Button, Empty, Select, Space, Tag } from 'antd';
import { Link, useSearchParams } from 'react-router-dom';
import { api, date, terminal } from '../api/client';
import type { RunPage } from '../api/types';
import { PageHead, Panel, QueryState, Status } from '../components';
import { useRun } from '../hooks';
import MechanismPanel from '../features/observability/MechanismPanel';

export default function Observations() {
  const [params, setParams] = useSearchParams();
  const selected = params.get('run') ?? undefined;
  const candidates = useQuery({
    queryKey: ['observation-candidates'],
    queryFn: () => api<RunPage>('/runs?limit=100'),
  });
  const run = useRun(selected);
  const rows =
    candidates.data?.items.filter(
      (r) =>
        r.kind === 'generate' &&
        (r.config.observation_mode ?? r.metrics.observation?.mode) &&
        (r.config.observation_mode ?? r.metrics.observation?.mode) !== 'off',
    ) ?? [];
  return (
    <>
      <PageHead
        kicker="Observation workspace"
        title="机制观测"
        description="沿一次真实运行检查内存、布局、分配与分页；所有视图绑定同一运行身份。"
        extra={
          <Link to="/playground">
            <Button type="primary">配置新观测</Button>
          </Link>
        }
      />
      <Panel
        title="当前观测对象"
        subtitle="候选来自最近 100 条任务；更早记录可通过运行详情进入。失败与部分采集仍保留。"
      >
        <QueryState error={candidates.error} />
        <Select
          aria-label="选择观测运行"
          showSearch
          optionFilterProp="label"
          value={selected}
          placeholder="选择一次已启用观测的推理"
          className="wide-select"
          loading={candidates.isPending}
          options={rows.map((r) => ({
            value: r.id,
            label: `${r.id} · ${r.config.observation_mode ?? r.metrics.observation?.mode} · ${r.config.deployment.name}`,
          }))}
          onChange={(value) => setParams({ run: value })}
        />
        <QueryState loading={!!selected && run.isPending} error={run.error} />
        {run.data && (
          <div className="object-context">
            <Space wrap>
              <Status state={run.data.state} />
              <Tag>{run.data.metrics.observation?.mode ?? '未采集'}</Tag>
              <span>
                {run.data.config.deployment.model} · {run.data.config.deployment.precision}
              </span>
              <span className="muted">{date(run.data.created_at)}</span>
            </Space>
            <Space wrap>
              <Link to={`/runs/${selected}?tab=observation`}>阶段、算子与输出</Link>
              <Link to={`/memory-flow?run=${selected}`}>空间结构</Link>
              <Link to={`/compare?left=${selected}`}>设为基线 A</Link>
            </Space>
          </div>
        )}
      </Panel>
      {!selected ? (
        <Panel>
          <Empty description="选择观测运行后检查机制证据；模型已卸载的封存包也可读取。" />
        </Panel>
      ) : run.data ? (
        <>
          {run.data.error && (
            <Alert className="section-gap" type="error" showIcon message={run.data.error} />
          )}
          <Panel>
            {terminal(run.data.state) ? (
              <MechanismPanel key={selected} runId={selected} />
            ) : (
              <Alert
                type="info"
                showIcon
                message="运行尚未封存"
                description="等待请求终态与清理完成后读取机制证据。当前请求仍由原运行 ID 跟踪。"
              />
            )}
          </Panel>
        </>
      ) : null}
    </>
  );
}
