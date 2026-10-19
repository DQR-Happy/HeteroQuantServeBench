import Button from '../components/ActionButton';
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Input, Select, Space, Table, Tag } from 'antd';
import { ArrowLeftOutlined, PlusOutlined, SearchOutlined } from '@ant-design/icons';
import { Link, useSearchParams } from 'react-router-dom';
import { api, date, terminal } from '../api/client';
import type { Run, RunPage } from '../api/types';
import { PageHead, QueryState, Status } from '../components';
import { useRun } from '../hooks';
import MechanismPanel from '../features/observability/MechanismPanel';

export default function Observations() {
  const [params, setParams] = useSearchParams();
  const selected = params.get('run') ?? undefined;
  const [search, setSearch] = useState('');
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
    <div className="observation-workspace">
      <PageHead
        kicker=""
        title="机制观测"
        description={
          selected
            ? '同一运行身份下，逐层检查测量结果与来源。'
            : '选择已保存的运行，从内存和布局追踪到分配器与操作系统。'
        }
        extra={
          <Link to="/playground">
            <Button icon={<PlusOutlined />}>配置新观测</Button>
          </Link>
        }
      />
      <QueryState error={candidates.error} />
      {!selected ? (
        <>
          <div className="directory-heading">
            <div>
              <h2>
                已采集的运行 <span className="count-badge">{rows.length}</span>
              </h2>
              <p>最近 100 条任务中的观测记录；失败和部分采集同样保留。</p>
            </div>
            <Input
              prefix={<SearchOutlined />}
              aria-label="筛选观测记录"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="运行 ID / 模型 / 采集模式"
              allowClear
            />
          </div>
          <Table<Run>
            rowKey="id"
            loading={candidates.isPending}
            size="middle"
            pagination={{ pageSize: 12, showSizeChanger: false }}
            scroll={{ x: 800 }}
            dataSource={rows.filter((r) =>
              `${r.id} ${r.config.deployment.name} ${r.config.observation_mode}`
                .toLowerCase()
                .includes(search.toLowerCase()),
            )}
            columns={[
              {
                title: '观测对象',
                render: (_, row) => (
                  <div className="table-object">
                    <Link to={`/observations?run=${row.id}`}>{row.config.deployment.name}</Link>
                    <code>{row.id}</code>
                  </div>
                ),
              },
              {
                title: '采集模式',
                render: (_, row) => (
                  <Tag>{row.config.observation_mode ?? row.metrics.observation?.mode}</Tag>
                ),
              },
              { title: '终态', dataIndex: 'state', render: (value) => <Status state={value} /> },
              { title: '创建时间', dataIndex: 'created_at', render: date },
              {
                title: '',
                render: (_, row) => <Link to={`/observations?run=${row.id}`}>检查证据 →</Link>,
              },
            ]}
          />
          <div className="evidence-legend">
            <span>
              <i />
              直接测量
            </span>
            <span>
              <i />
              派生 / 关联
            </span>
            <span>
              <i />
              不可用 / 未采集
            </span>
            <p>每条指标标明自己的作用域与来源。CPU 页样本展示所选进程虚拟页的状态。</p>
          </div>
        </>
      ) : (
        <>
          <div className="observation-objectbar">
            <Button
              type="text"
              icon={<ArrowLeftOutlined />}
              aria-label="返回观测目录"
              onClick={() => setParams({})}
            />
            <Select
              aria-label="选择观测运行"
              showSearch
              optionFilterProp="label"
              value={selected}
              className="object-select"
              loading={candidates.isPending}
              options={rows.map((r) => ({
                value: r.id,
                label: `${r.id} · ${r.config.observation_mode ?? r.metrics.observation?.mode} · ${r.config.deployment.name}`,
              }))}
              onChange={(value) => setParams({ run: value })}
            />
            {run.data && (
              <>
                <Status state={run.data.state} />
                <Tag>
                  {run.data.config.observation_mode ??
                    run.data.metrics.observation?.mode ??
                    '未采集'}
                </Tag>
              </>
            )}
          </div>
          <QueryState loading={run.isPending} error={run.error} />
          {run.data && (
            <>
              <div className="observation-context">
                <span>
                  <strong>{run.data.config.deployment.model.split('/').at(-1)}</strong>
                  <span className="context-separator">/</span>
                  {run.data.config.deployment.precision}
                  <span className="context-separator">/</span>
                  {date(run.data.created_at)}
                </span>
                <Space wrap>
                  <Link to={`/runs/${selected}?tab=observation`}>阶段与输出</Link>
                  <Link to={`/memory-flow?run=${selected}`}>空间结构</Link>
                  <Link to={`/compare?left=${selected}`}>设为基线 A</Link>
                </Space>
              </div>
              {run.data.error && (
                <Alert className="section-gap" type="error" showIcon message={run.data.error} />
              )}
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
            </>
          )}
        </>
      )}
    </div>
  );
}
