import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, App, Button, Select, Space, Table, Tag } from 'antd';
import { Link } from 'react-router-dom';
import { api, fmt } from '../api/client';
import type { Run, RunPage } from '../api/types';
import { Chart, PageHead, Panel, QueryState } from '../components';

type Comparison = { items: Run[]; comparable: boolean; reasons: string[]; notice: string };
export default function Compare() {
  const { message } = App.useApp();
  const [ids, setIds] = useState<string[]>([]);
  const [result, setResult] = useState<Comparison>();
  const [busy, setBusy] = useState(false);
  const query = useQuery({
    queryKey: ['comparison-candidates'],
    queryFn: () => api<RunPage>('/runs?limit=100'),
    refetchInterval: 10000,
  });
  const compare = async () => {
    setBusy(true);
    try {
      setResult(await api<Comparison>('/comparisons', { run_ids: ids }));
    } catch (e) {
      void message.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <PageHead
        kicker="COMPARE WITH CONTEXT"
        title="运行对比"
        description="先检查输入与测量口径，再阅读数字。单次交互结果用于定位问题，不能自动形成加速结论。"
      />
      <Panel
        title="选择 2–6 次已完成的推理"
        subtitle="候选来自最近 100 条任务。部署加载与失败请求不参与对比。"
      >
        <QueryState error={query.error} />
        <Space.Compact block>
          <Select
            aria-label="选择对比运行"
            mode="multiple"
            maxCount={6}
            value={ids}
            onChange={(value) => {
              setIds(value);
              setResult(undefined);
            }}
            placeholder="选择运行 ID"
            style={{ width: '100%' }}
            options={query.data?.items
              .filter((r) => r.kind === 'generate' && r.state === 'completed')
              .map((r) => ({
                value: r.id,
                label: `${r.id.slice(0, 16)} · ${r.config.deployment.name} · ${r.metrics.output_tokens ?? '?'} tokens`,
              }))}
          />
          <Button
            type="primary"
            loading={busy}
            disabled={ids.length < 2}
            onClick={() => void compare()}
          >
            验证并对比
          </Button>
        </Space.Compact>
      </Panel>
      {result && (
        <>
          <Alert
            className="section-gap"
            showIcon
            type={result.comparable ? 'info' : 'warning'}
            message={
              result.comparable
                ? '基础诊断口径一致，仍不代表质量门通过'
                : '不可直接比较：已阻止统一加速比结论'
            }
            description={result.reasons.length ? result.reasons.join('；') : result.notice}
          />
          <Panel title="请求指标" extra={<Tag>Diagnostic only</Tag>}>
            <Table<Run>
              rowKey="id"
              dataSource={result.items}
              pagination={false}
              scroll={{ x: 800 }}
              columns={[
                {
                  title: '运行',
                  dataIndex: 'id',
                  render: (v: string) => <Link to={`/runs/${v}`}>{v.slice(0, 16)}</Link>,
                },
                { title: '部署', render: (_, r) => r.config.deployment.name },
                { title: '首内容 (ms)', render: (_, r) => fmt(r.metrics.console_first_content_ms) },
                {
                  title: 'Runtime TTFT (ms)',
                  render: (_, r) => fmt(r.metrics.runtime_first_token_ms),
                },
                { title: '输出 token', render: (_, r) => r.metrics.output_tokens ?? '—' },
                {
                  title: 'Decode (token/s)',
                  render: (_, r) => fmt(r.metrics.decode_tail_tokens_per_s),
                },
                { title: '质量', render: () => <Tag>未评估</Tag> },
              ]}
            />
          </Panel>
          {result.comparable && (
            <Panel
              title="同口径延迟分布"
              subtitle="API 首内容包含排队与传输；Runtime 首 token 仅在本地 Provider 可测"
            >
              <Chart
                label="选定运行首内容与首 token 延迟对比"
                option={{
                  tooltip: { trigger: 'axis' },
                  legend: { bottom: 0 },
                  grid: { left: 60, right: 24, top: 20, bottom: 60 },
                  xAxis: { type: 'category', data: result.items.map((r) => r.id.slice(-8)) },
                  yAxis: { type: 'value', name: 'ms' },
                  series: [
                    {
                      type: 'bar',
                      name: '服务端首内容',
                      data: result.items.map((r) => r.metrics.console_first_content_ms ?? null),
                      itemStyle: { color: '#f17b53' },
                    },
                    {
                      type: 'bar',
                      name: 'Runtime 首 token',
                      data: result.items.map((r) => r.metrics.runtime_first_token_ms ?? null),
                      itemStyle: { color: '#258c7e' },
                    },
                  ],
                }}
              />
            </Panel>
          )}
        </>
      )}
    </>
  );
}

type Historical = {
  method: string;
  workload: string;
  metrics: Record<string, number | null>;
  verdict: string;
  evidence_id: string;
  source_sha256: string;
};
export function Efficiency() {
  const query = useQuery({
    queryKey: ['historical-performance'],
    queryFn: () => api<{ items: Historical[]; notice: string }>('/historical/performance'),
  });
  const [workload, setWorkload] = useState('');
  const all = query.data?.items ?? [];
  const chosen = workload || all[0]?.workload;
  const rows = all.filter((x) => x.workload === chosen);
  const keys = [...new Set(rows.flatMap((x) => Object.keys(x.metrics)))];
  return (
    <>
      <PageHead
        kicker="EFFICIENCY & TRADEOFFS"
        title="性能与能效"
        description="展示已归档 E05-02 汇总中的真实中位数，保留实验的质量和执行路径限制。"
      />
      <Alert
        className="section-gap"
        type="warning"
        showIcon
        message={query.data?.notice ?? '正在读取历史实验'}
        description="没有统一请求边界的功耗积分、设备租用价格与质量通过证据时，不生成每 token 能耗、成本排名或最优平台推荐。"
      />
      <Panel
        title="历史工作负载"
        extra={
          <Select
            aria-label="选择工作负载"
            style={{ minWidth: 180 }}
            value={chosen}
            onChange={setWorkload}
            options={[...new Set(all.map((x) => x.workload))].map((value) => ({
              value,
              label: value,
            }))}
          />
        }
      >
        <QueryState
          loading={query.isPending}
          error={query.error}
          empty={!query.isPending && !rows.length}
        />
        <Table<Historical>
          rowKey="method"
          dataSource={rows}
          pagination={false}
          scroll={{ x: 900 }}
          columns={[
            { title: '方法', dataIndex: 'method', fixed: 'left' },
            ...keys.map((key) => ({
              title: key,
              dataIndex: ['metrics', key],
              render: (v: number | null) => fmt(v, 3),
            })),
            { title: '原始结论', dataIndex: 'verdict' },
          ]}
        />
        <div className="help-text">
          数据源：docs/stage_experiments/S05/E05-02/raw/summary.json。单位以列名和原始报告为准。
        </div>
        {rows[0] && <div className="mono small muted">SHA-256 {rows[0].source_sha256}</div>}
        <Link to="/quantization">打开质量门与原始报告 →</Link>
      </Panel>
    </>
  );
}
