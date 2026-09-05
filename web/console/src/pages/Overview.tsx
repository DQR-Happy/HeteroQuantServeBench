import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  ArrowRightOutlined,
  PlusOutlined,
  ReloadOutlined,
  SearchOutlined,
  BranchesOutlined,
} from '@ant-design/icons';
import { Button, Input, Segmented } from 'antd';
import { Link } from 'react-router-dom';
import { api, date, fmt, gib } from '../api/client';
import type { Overview as OverviewData } from '../api/types';
import { PageHead, Status, QueryState, runKindLabel } from '../components';

export default function Overview() {
  const query = useQuery({
    queryKey: ['overview'],
    queryFn: () => api<OverviewData>('/overview'),
    refetchInterval: 5000,
  });
  const [search, setSearch] = useState('');
  const [scope, setScope] = useState('全部活动');
  const data = query.data;
  const last = data?.telemetry.samples.at(-1);
  const rows = (data?.runs.items ?? []).filter(
    (run) =>
      `${run.id} ${run.config.deployment.name} ${run.kind}`
        .toLowerCase()
        .includes(search.toLowerCase()) &&
      (scope !== '观测记录' ||
        (run.config.observation_mode && run.config.observation_mode !== 'off')),
  );
  return (
    <>
      <PageHead
        kicker=""
        title="工作空间"
        description="HeteroQuantServeBench · 从真实运行追踪优化的实际影响。"
        extra={
          <>
            <Link to="/benchmarks">
              <Button type="primary" icon={<PlusOutlined />}>
                开始固定输入对照
              </Button>
            </Link>
            <Link to="/playground">
              <Button>新建推理</Button>
            </Link>
          </>
        }
      />
      <div className="home-layout">
        <section className="activity-workspace" aria-label="近期活动">
          <div className="activity-toolbar">
            <Segmented
              aria-label="近期活动范围"
              options={['全部活动', '观测记录']}
              value={scope}
              onChange={setScope}
            />
            <div className="grow" />
            <Button
              type="text"
              aria-label="刷新活动"
              icon={<ReloadOutlined />}
              loading={query.isFetching}
              onClick={() => void query.refetch()}
            />
          </div>
          <Input
            className="activity-search"
            prefix={<SearchOutlined />}
            aria-label="筛选近期活动"
            placeholder="筛选运行 ID、部署或任务类型…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            allowClear
          />
          <QueryState loading={query.isPending} error={query.error} />
          <div className="activity-list">
            {rows.map((run) => (
              <article className="activity-row" key={run.id}>
                <div className={`activity-node ${run.state === 'failed' ? 'node-failed' : ''}`}>
                  <BranchesOutlined />
                </div>
                <div className="activity-main">
                  <div className="activity-title">
                    <Link to={`/runs/${run.id}`}>{run.config.deployment.name}</Link>
                    <span className="activity-kind">{runKindLabel(run.kind)}</span>
                  </div>
                  <div className="activity-meta">
                    <code>{run.id.slice(0, 20)}</code>
                    <span>{date(run.created_at)}</span>
                    <span>
                      {run.config.deployment.precision} · {run.config.deployment.attention}
                    </span>
                  </div>
                </div>
                <div className="activity-end">
                  <Status state={run.state} />
                  {run.config.observation_mode && run.config.observation_mode !== 'off' ? (
                    <Link to={`/observations?run=${run.id}`}>
                      检查 {run.config.observation_mode} <ArrowRightOutlined />
                    </Link>
                  ) : (
                    <Link to={`/runs/${run.id}`}>
                      查看记录 <ArrowRightOutlined />
                    </Link>
                  )}
                </div>
              </article>
            ))}
            {data && !rows.length && (
              <div className="list-empty">
                <h3>{search ? '没有匹配的近期记录' : '这里将显示实际运行'}</h3>
                <p>
                  {search
                    ? '清除筛选或打开全部运行继续查找。'
                    : '选择部署并开始一次推理，生成的证据会保存在运行记录中。'}
                </p>
                <Link to={search ? '/runs' : '/playground'}>
                  {search ? '打开全部运行' : '配置推理'} →
                </Link>
              </div>
            )}
          </div>
          <div className="list-footer">
            <span>
              当前 {rows.length} 条 / 全部 {data?.runs.total ?? '—'} 条
            </span>
            <Link to="/runs">
              全部运行 <ArrowRightOutlined />
            </Link>
          </div>
        </section>
        <aside className="project-aside">
          <section>
            <h2>执行环境</h2>
            <QueryState error={query.error} />
            {data?.deployments.map((deployment) => (
              <div className="environment-entry" key={deployment.id}>
                <div>
                  <strong>{deployment.name}</strong>
                  <Status state={deployment.state} />
                </div>
                <p>
                  {deployment.platform} · {deployment.precision}
                </p>
                <code>{deployment.model.split('/').at(-1)}</code>
              </div>
            ))}
            <dl className="aside-facts">
              <div>
                <dt>主机可用内存</dt>
                <dd>
                  {fmt(gib(last?.host_available_bytes), 2)} <span>GiB</span>
                </dd>
              </div>
              <div>
                <dt>队列待处理</dt>
                <dd>{data?.pending ?? '—'}</dd>
              </div>
              <div>
                <dt>归档实验</dt>
                <dd>{data?.evidence_count ?? '—'}</dd>
              </div>
            </dl>
            <Link className="aside-link" to="/devices">
              管理设备与部署 <ArrowRightOutlined />
            </Link>
            <p className="help-text">
              整机采样，不代表模型独占内存。
              <br />
              {data ? date(data.observed_at) : '等待采样'}
            </p>
          </section>
          <section>
            <h2>继续分析</h2>
            <div className="task-links">
              <Link to="/observations">
                <span>
                  <strong>机制观测</strong>
                  <small>KV、布局、分配与分页</small>
                </span>
                <ArrowRightOutlined />
              </Link>
              <Link to="/optimization">
                <span>
                  <strong>优化实验</strong>
                  <small>冻结配置，建立受控对照</small>
                </span>
                <ArrowRightOutlined />
              </Link>
              <Link to="/evidence">
                <span>
                  <strong>证据档案</strong>
                  <small>原始文件、来源与结论</small>
                </span>
                <ArrowRightOutlined />
              </Link>
            </div>
          </section>
          <section className="aside-note">
            <h2>证据原则</h2>
            <p>运行完成只代表执行终态。质量、性能和机制变化分别核验，缺失的测量保留为空。</p>
            <Link to="/settings">查看指标契约 →</Link>
          </section>
        </aside>
      </div>
    </>
  );
}
