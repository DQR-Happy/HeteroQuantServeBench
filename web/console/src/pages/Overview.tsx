import { useQuery } from '@tanstack/react-query';
import { ArrowRightOutlined, DeploymentUnitOutlined } from '@ant-design/icons';
import { Button, Space, Table, Tag } from 'antd';
import { Link } from 'react-router-dom';
import { api, date, fmt, gib } from '../api/client';
import type { Overview as OverviewData, Run } from '../api/types';
import { PageHead, Panel, Status, Chart, QueryState, runKindLabel } from '../components';

export default function Overview() {
  const query = useQuery({
    queryKey: ['overview'],
    queryFn: () => api<OverviewData>('/overview'),
    refetchInterval: 3000,
  });
  const data = query.data;
  const samples = data?.telemetry.samples ?? [];
  const last = samples.at(-1);
  return (
    <>
      <PageHead
        kicker="HeteroQuantServeBench"
        title="工作空间"
        description="从真实运行进入机制证据，核对一次优化改变了什么。"
        extra={
          <Space wrap>
            <Link to="/observations">
              <Button>打开机制观测</Button>
            </Link>
            <Link to="/playground">
              <Button type="primary">开始观测</Button>
            </Link>
          </Space>
        }
      />
      <QueryState loading={query.isPending} error={query.error} />
      <div className="workspace-strip">
        <span>
          部署{' '}
          <strong>
            {data
              ? `${data.deployments.filter((d) => d.state === 'ready').length} / ${data.deployments.length} 就绪`
              : '—'}
          </strong>
        </span>
        <span>
          运行 <strong>{data?.runs.total ?? '—'}</strong>
        </span>
        <span>
          待处理 <strong>{data?.pending ?? '—'}</strong>
        </span>
        <span>
          归档实验 <strong>{data?.evidence_count ?? '—'}</strong>
        </span>
        <span className="muted">采样于 {data ? date(data.observed_at) : '—'}</span>
      </div>
      <Panel
        title="最近运行"
        subtitle="观察范围与运行终态独立；已完成不代表质量通过。"
        extra={
          <Link to="/runs">
            全部运行 <ArrowRightOutlined />
          </Link>
        }
      >
        <Table<Run>
          rowKey="id"
          size="small"
          pagination={false}
          dataSource={data?.runs.items ?? []}
          scroll={{ x: 900 }}
          columns={[
            {
              title: '运行',
              dataIndex: 'id',
              render: (id: string) => (
                <Link className="mono" to={`/runs/${id}`}>
                  {id.slice(0, 20)}
                </Link>
              ),
            },
            { title: '任务', dataIndex: 'kind', render: runKindLabel },
            { title: '部署', render: (_, r) => r.config.deployment.name },
            { title: '状态', dataIndex: 'state', render: (s: string) => <Status state={s} /> },
            {
              title: '采集模式',
              render: (_, r) =>
                (r.config.observation_mode ?? r.metrics.observation?.mode) &&
                (r.config.observation_mode ?? r.metrics.observation?.mode) !== 'off' ? (
                  <Link to={`/observations?run=${r.id}`}>
                    {r.config.observation_mode ?? r.metrics.observation?.mode} →
                  </Link>
                ) : (
                  <span className="muted">未采集</span>
                ),
            },
            {
              title: '首内容 / ms',
              align: 'right',
              render: (_, r) => fmt(r.metrics.console_first_content_ms),
            },
            { title: '创建时间', dataIndex: 'created_at', render: date },
          ]}
        />
      </Panel>
      <div className="grid-two">
        <Panel
          title="执行环境"
          subtitle="部署当前状态 · 与归档记录分开"
          extra={<Link to="/devices">设备与部署 →</Link>}
        >
          {data?.deployments.map((d) => (
            <div className="deployment-row" key={d.id}>
              <DeploymentUnitOutlined />
              <div className="grow">
                <strong>{d.name}</strong>
                <div className="muted">
                  {d.platform} · {d.precision} · {d.attention}
                </div>
              </div>
              <Status state={d.state} />
            </div>
          ))}
          <div className="resource-line">
            <span>整机可用内存</span>
            <strong>{fmt(gib(last?.host_available_bytes), 2)} GiB</strong>
          </div>
          <p className="help-text">
            {last
              ? `${data?.telemetry.host} · 主机共享内存口径，非模型独占显存`
              : '等待远端采集；缺失值不会补零'}
          </p>
        </Panel>
        <Panel title="控制节点采样" subtitle="最近 120 点 · 实时整机范围">
          <Chart
            label="控制节点 GPU 利用率时间序列"
            height={190}
            option={{
              grid: { left: 45, right: 16, top: 25, bottom: 30 },
              tooltip: { trigger: 'axis' },
              xAxis: {
                type: 'category',
                data: samples.map((x) => new Date(x.time * 1000).toLocaleTimeString()),
              },
              yAxis: { type: 'value', min: 0, max: 100, name: '% GPU busy' },
              series: [
                {
                  type: 'line',
                  data: samples.map((x) => x.gpu_util_pct),
                  showSymbol: false,
                  connectNulls: false,
                  lineStyle: { color: '#435f82', width: 1.5 },
                },
              ],
            }}
          />
          <p className="help-text">
            {last?.gpu_availability === 'measured'
              ? '来源：tegrastats；GPU busy 不等于 SM occupancy，不归因某次请求。'
              : 'GPU 利用率未采集，曲线保留空缺。'}
          </p>
        </Panel>
      </div>
      <div className="workflow-links">
        <Link to="/playground">
          <span>01</span>
          <strong>配置与采集</strong>
          <small>选择真实部署与观测模式</small>
        </Link>
        <Link to="/observations">
          <span>02</span>
          <strong>检查机制</strong>
          <small>布局 / 分配 / CPU 页</small>
        </Link>
        <Link to="/compare">
          <span>03</span>
          <strong>受控对照</strong>
          <small>检查变量与证据差异</small>
        </Link>
        <Link to="/evidence">
          <span>04</span>
          <strong>追溯原件</strong>
          <small>保留来源与实验裁决</small>
        </Link>
      </div>
      <p className="foot-note">
        <Tag>诊断范围</Tag> 单次运行不构成优化结论；详细采集有开销，质量门和重复实验需分别验证。
      </p>
    </>
  );
}
