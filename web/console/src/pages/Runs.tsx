import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { App, Button, Input, Select, Space, Table, Tabs, Tag, Alert } from 'antd';
import { DownloadOutlined, ReloadOutlined, StopOutlined } from '@ant-design/icons';
import { Link, useParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import { api, base, date, fmt, gib, terminal } from '../api/client';
import type { Run, RunPage } from '../api/types';
import { Chart, JsonView, PageHead, Panel, QueryState, Stat, Status } from '../components';
import { useRun } from '../hooks';

export function Runs() {
  const [offset, setOffset] = useState(0),
    [filter, setFilter] = useState('all'),
    [search, setSearch] = useState('');
  const query = useQuery({
    queryKey: ['runs', offset],
    queryFn: () => api<RunPage>(`/runs?limit=30&offset=${offset}`),
    refetchInterval: 3000,
  });
  const rows =
    query.data?.items.filter(
      (r) =>
        (filter === 'all' || r.state === filter) &&
        (r.id.includes(search) || r.config.deployment.name.includes(search)),
    ) ?? [];
  return (
    <>
      <PageHead
        kicker="RUN HISTORY"
        title="运行记录"
        description="加载、生成、取消与失败都保留记录。每次运行绑定当时实际部署。"
        extra={
          <Button icon={<ReloadOutlined />} onClick={() => void query.refetch()}>
            刷新
          </Button>
        }
      />
      <Panel>
        <div className="toolbar">
          <Input.Search
            placeholder="搜索本页运行 ID / 部署"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ maxWidth: 330 }}
          />
          <Select
            aria-label="运行状态筛选"
            value={filter}
            onChange={setFilter}
            style={{ width: 150 }}
            options={[
              'all',
              'completed',
              'running',
              'queued',
              'failed',
              'cancelled',
              'timed_out',
            ].map((x) => ({ label: x === 'all' ? '全部状态' : x, value: x }))}
          />
        </div>
        <QueryState error={query.error} />
        <Table<Run>
          loading={query.isPending}
          rowKey="id"
          dataSource={rows}
          scroll={{ x: 850 }}
          pagination={{
            current: offset / 30 + 1,
            pageSize: 30,
            total: query.data?.total ?? 0,
            showSizeChanger: false,
            onChange: (n) => setOffset((n - 1) * 30),
          }}
          columns={[
            {
              title: '运行 ID',
              dataIndex: 'id',
              render: (id: string) => (
                <Link className="mono" to={`/runs/${id}`}>
                  {id.slice(0, 18)}
                </Link>
              ),
            },
            { title: '类型', dataIndex: 'kind' },
            { title: '部署', render: (_, r) => r.config.deployment.name },
            { title: '状态', dataIndex: 'state', render: (s: string) => <Status state={s} /> },
            { title: '首内容 / ms', render: (_, r) => fmt(r.metrics.console_first_content_ms) },
            { title: '输出 token', render: (_, r) => fmt(r.metrics.output_tokens, 0) },
            { title: '创建时间', dataIndex: 'created_at', render: date },
          ]}
        />
      </Panel>
    </>
  );
}

export function RunDetail() {
  const { id } = useParams();
  const query = useRun(id);
  const r = query.data;
  const { message } = App.useApp();
  const itl = r?.metrics.token_itl_ms ?? [];
  return (
    <>
      <PageHead
        kicker="RUN INSPECTOR"
        title="运行详情"
        description={id ?? ''}
        extra={
          <Space>
            {r && <Status state={r.state} />}
            <Button icon={<DownloadOutlined />} href={`${base}/runs/${id}/export`}>
              JSON
            </Button>
            <Button href={`${base}/runs/${id}/export?format=markdown`}>报告</Button>
          </Space>
        }
      />
      <QueryState loading={query.isPending} error={query.error} />
      {r && (
        <>
          <Alert
            className="section-gap"
            type={r.error ? 'error' : 'info'}
            showIcon
            message={r.error ?? '这是实际运行记录；质量未评估，单次交互不能形成正式加速结论。'}
            action={
              !terminal(r.state) && (
                <Button
                  danger
                  icon={<StopOutlined />}
                  onClick={() => {
                    void api(`/requests/${id}/cancel`, {}).catch((e) => message.error(e.message));
                  }}
                >
                  取消
                </Button>
              )
            }
          />
          <div className="stats-grid">
            <Stat label="服务端首内容" value={r.metrics.console_first_content_ms} unit="ms" />
            <Stat label="Runtime 首 token" value={r.metrics.runtime_first_token_ms} unit="ms" />
            <Stat label="请求总时间" value={r.metrics.console_e2e_ms} unit="ms" />
            <Stat
              label="峰值分配内存"
              value={gib(r.metrics.memory?.peak_allocated_bytes)}
              unit="GiB"
              hint="worker allocator · 非整机内存"
            />
          </div>
          <Panel>
            <Tabs
              items={[
                {
                  key: 'output',
                  label: '输出与计时',
                  children: (
                    <>
                      <div className="grid-two">
                        <div className="markdown detail-output">
                          <ReactMarkdown skipHtml>
                            {r.output || '此任务没有文本输出。'}
                          </ReactMarkdown>
                        </div>
                        <div>
                          <h3>计时边界</h3>
                          <div className="timeline">
                            <div>
                              <i />
                              请求创建 <span>{date(r.created_at)}</span>
                            </div>
                            <div>
                              <i />
                              排队 <span>{fmt(r.metrics.queue_ms)} ms</span>
                            </div>
                            <div>
                              <i />
                              Runtime 首 token{' '}
                              <span>{fmt(r.metrics.runtime_first_token_ms)} ms</span>
                            </div>
                            <div>
                              <i />
                              服务端首内容 <span>{fmt(r.metrics.console_first_content_ms)} ms</span>
                            </div>
                            <div>
                              <i />
                              完成与清理 <span>{r.cleanup}</span>
                            </div>
                          </div>
                          <Tag>{r.metrics.measurement_profile ?? 'deployment lifecycle'}</Tag>
                        </div>
                      </div>
                      {itl.length > 0 && (
                        <Chart
                          label="真实生成 token 间隔"
                          option={{
                            tooltip: { trigger: 'axis' },
                            grid: { left: 50, right: 20, bottom: 35, top: 30 },
                            xAxis: {
                              type: 'category',
                              name: 'step',
                              data: itl.map((_, i) => i + 2),
                            },
                            yAxis: { type: 'value', name: 'ms' },
                            series: [
                              {
                                type: 'line',
                                data: itl,
                                lineStyle: { color: '#238f80' },
                                showSymbol: itl.length < 30,
                              },
                            ],
                          }}
                        />
                      )}
                    </>
                  ),
                },
                { key: 'config', label: '实际配置', children: <JsonView value={r.config} /> },
                {
                  key: 'metrics',
                  label: '指标与原始结果',
                  children: <JsonView value={r.metrics} />,
                },
                {
                  key: 'identity',
                  label: '身份与边界',
                  children: (
                    <>
                      <div className="key-values">
                        <span>部署</span>
                        <strong>{r.config.deployment.name}</strong>
                        <span>Epoch</span>
                        <code>{r.config.epoch ?? '—'}</code>
                        <span>输入摘要</span>
                        <code>{r.config.input_sha256 ?? '—'}</code>
                        <span>精度 / 算子</span>
                        <strong>
                          {r.config.deployment.precision} / {r.config.deployment.kernel}
                        </strong>
                        <span>质量裁决</span>
                        <strong>未执行质量基准</strong>
                        <span>输入保留</span>
                        <strong>
                          {r.config.messages ? '已按选择保存' : '仅摘要；精确复现需再次提供原输入'}
                        </strong>
                      </div>
                    </>
                  ),
                },
              ]}
            />
          </Panel>
        </>
      )}
    </>
  );
}
