import Button from '../components/ActionButton';
import { useQuery } from '@tanstack/react-query';
import { Input, Table, Tag } from 'antd';
import { PlusOutlined, ReloadOutlined, SearchOutlined } from '@ant-design/icons';
import { Link, useSearchParams } from 'react-router-dom';
import { api, date, fmt } from '../api/client';
import type { Run, RunPage } from '../api/types';
import { PageHead, QueryState, Status, runKindLabel } from '../components';
import { isQualityRun } from '../features/experiments/model';
import { QualityBaselineCell } from '../features/experiments/QualityBaseline';

const states = [
  ['all', '全部状态'],
  ['completed', '已完成'],
  ['running', '执行中'],
  ['queued', '排队中'],
  ['failed', '失败'],
  ['cancelled', '已取消'],
  ['timed_out', '超时'],
];
export default function RunDirectory() {
  const [params, setParams] = useSearchParams();
  const requestedPage = Number(params.get('page') || 1);
  const offset =
    (Number.isSafeInteger(requestedPage) && requestedPage > 0 ? requestedPage - 1 : 0) * 30;
  const filter = params.get('state') || 'all';
  const kind = params.get('kind') || 'all';
  const search = params.get('q') || '';
  const update = (key: string, value: string) =>
    setParams(
      (previous) => {
        const next = new URLSearchParams(previous);
        if (!value || value === 'all') next.delete(key);
        else next.set(key, value);
        return next;
      },
      { replace: key === 'q' },
    );
  const query = useQuery({
    queryKey: ['runs', offset],
    queryFn: () => api<RunPage>(`/runs?limit=30&offset=${offset}`),
    refetchInterval: 5000,
  });
  const rows = (query.data?.items ?? []).filter(
    (run) =>
      (filter === 'all' || run.state === filter) &&
      (kind === 'all' ||
        (kind === 'quality'
          ? isQualityRun(run)
          : run.kind === kind && (kind !== 'generate' || !isQualityRun(run)))) &&
      `${run.id} ${run.config.deployment.name}`.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <>
      <PageHead
        kicker=""
        title="运行记录"
        description="每次请求、加载和失败都有独立记录，配置与证据绑定到实际执行身份。"
        extra={
          <Link to="/playground">
            <Button type="primary" icon={<PlusOutlined />}>
              新建推理
            </Button>
          </Link>
        }
      />
      <div className="directory-layout">
        <aside className="filter-rail" aria-label="运行筛选">
          <h2>运行状态</h2>
          <nav>
            {states.map(([value, label]) => (
              <button
                key={value}
                aria-pressed={filter === value}
                onClick={() => update('state', value)}
              >
                {label}
                <span>
                  {value === 'all'
                    ? (query.data?.items.length ?? '—')
                    : (query.data?.items.filter((run) => run.state === value).length ?? '—')}
                </span>
              </button>
            ))}
          </nav>
          <h2>任务类型</h2>
          <nav>
            {[
              'all',
              'generate',
              'quality',
              'load',
              'unload',
              'quantize',
              'scheduling',
              'system',
              'application',
              'distributed',
            ].map((value) => (
              <button
                key={value}
                aria-pressed={kind === value}
                onClick={() => update('kind', value)}
              >
                {value === 'all'
                  ? '全部任务'
                  : value === 'quality'
                    ? '独立质量评测'
                    : runKindLabel(value)}
              </button>
            ))}
          </nav>
          <p>筛选与计数作用于当前页的 30 条记录。</p>
        </aside>
        <section className="directory-content">
          <div className="directory-toolbar">
            <Input
              prefix={<SearchOutlined />}
              aria-label="搜索本页运行"
              placeholder="搜索本页运行 ID / 部署"
              value={search}
              onChange={(event) => update('q', event.target.value)}
              allowClear
            />
            <Button
              icon={<ReloadOutlined />}
              aria-label="刷新运行记录"
              loading={query.isFetching}
              onClick={() => void query.refetch()}
            >
              刷新
            </Button>
          </div>
          <QueryState error={query.error} />
          <div className="table-caption">
            <strong>{rows.length} 条匹配</strong>
            <span>
              共 {query.data?.total ?? '—'} 条任务 · 第 {offset / 30 + 1} 页
            </span>
            {(filter !== 'all' || kind !== 'all' || search) && (
              <Button type="link" size="small" onClick={() => setParams({})}>
                清除筛选
              </Button>
            )}
          </div>
          <Table<Run>
            loading={query.isPending}
            rowKey="id"
            dataSource={rows}
            size="middle"
            scroll={{ x: 960 }}
            pagination={{
              current: offset / 30 + 1,
              pageSize: 30,
              total: query.data?.total ?? 0,
              showSizeChanger: false,
              onChange: (page) => update('page', String(page)),
            }}
            columns={[
              {
                title: '任务 / 运行身份',
                width: 300,
                render: (_, run) => (
                  <div className="table-object">
                    <Link
                      to={
                        run.kind === 'scheduling'
                          ? `/scheduling?experiment=${run.id}`
                          : `/runs/${run.id}`
                      }
                    >
                      {run.config.deployment.name}
                    </Link>
                    <span>
                      <Tag>{isQualityRun(run) ? '独立质量评测' : runKindLabel(run.kind)}</Tag>
                      <code>{run.id.slice(0, 20)}</code>
                    </span>
                  </div>
                ),
              },
              {
                title: '状态',
                dataIndex: 'state',
                width: 110,
                render: (state) => <Status state={state} />,
              },
              {
                title: '观测证据',
                width: 120,
                render: (_, run) => {
                  if (isQualityRun(run))
                    return <Link to={`/runs/${run.id}?tab=quality`}>质量评分 →</Link>;
                  const mode = run.config.observation_mode ?? run.metrics.observation?.mode;
                  return mode && mode !== 'off' ? (
                    <Link to={`/observations?run=${run.id}`}>{mode} →</Link>
                  ) : (
                    <span className="muted">未采集</span>
                  );
                },
              },
              {
                title: '首内容 / PPL',
                align: 'right',
                render: (_, run) => (
                  <span className="numeric">
                    {isQualityRun(run)
                      ? `${fmt(run.metrics.quality_evaluation?.perplexity, 3)} PPL`
                      : `${fmt(run.metrics.console_first_content_ms)} ms`}
                  </span>
                ),
              },
              {
                title: 'PPL 相对基线',
                render: (_, run) =>
                  isQualityRun(run) ? (
                    <QualityBaselineCell run={run} />
                  ) : (
                    <span className="muted">不适用</span>
                  ),
              },
              {
                title: '输出 / 计分 Token',
                align: 'right',
                render: (_, run) =>
                  fmt(
                    isQualityRun(run)
                      ? run.metrics.quality_evaluation?.scored_tokens
                      : run.metrics.output_tokens,
                    0,
                  ),
              },
              { title: '创建时间', dataIndex: 'created_at', render: date },
            ]}
          />
        </section>
      </div>
    </>
  );
}
