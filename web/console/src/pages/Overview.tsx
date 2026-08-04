import { useQuery } from '@tanstack/react-query';
import {
  ArrowRightOutlined,
  ExperimentOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  DeploymentUnitOutlined,
} from '@ant-design/icons';
import { Button, Progress, Space, Table, Tag } from 'antd';
import { Link } from 'react-router-dom';
import { api, date, fmt, gib } from '../api/client';
import type { Overview as OverviewData, Run } from '../api/types';
import { Panel, Stat, Status, Chart, QueryState, runKindLabel } from '../components';

export default function Overview() {
  const query = useQuery({
    queryKey: ['overview'],
    queryFn: () => api<OverviewData>('/overview'),
    refetchInterval: 3000,
  });
  const data = query.data;
  const samples = data?.telemetry.samples ?? [];
  const last = samples.at(-1);
  const ready = data?.deployments.filter((x) => x.state === 'ready').length ?? 0;
  return (
    <>
      <div className="hero">
        <div>
          <div className="eyebrow">HETEROGENEOUS INFERENCE · OBSERVABLE BY DESIGN</div>
          <h1>
            从一个 token，
            <br />
            看到整条推理链路<span>。</span>
          </h1>
          <p>
            运行真实模型，观察资源变化，比较优化方案。
            <br />
            让每一个性能结论，都有可以打开的证据。
          </p>
          <Space>
            <Link to="/playground">
              <Button type="primary" size="large" icon={<ThunderboltOutlined />}>
                进入推理工作台
              </Button>
            </Link>
            <Link to="/evidence">
              <Button size="large">
                浏览实验证据 <ArrowRightOutlined />
              </Button>
            </Link>
          </Space>
        </div>
        <div className="hero-flow" aria-label="模型到证据的工作流">
          <div className="flow-orbit" />
          <div className="flow-node node-model">
            <DatabaseOutlined />
            <span>MODEL</span>
          </div>
          <div className="flow-node node-kernel">
            <ThunderboltOutlined />
            <span>KERNEL</span>
          </div>
          <div className="flow-node node-device">
            <DeploymentUnitOutlined />
            <span>DEVICE</span>
          </div>
          <div className="flow-node node-evidence">
            <ExperimentOutlined />
            <span>EVIDENCE</span>
          </div>
          <div className="flow-center">
            HQ<span>SB</span>
            <small>INFERENCE LAB</small>
          </div>
        </div>
      </div>
      <div className="release-strip">
        <Tag color="green">v0.2.1 可视化工作流</Tag>
        <Link to="/memory-flow">
          <strong>查看内存结构与数据流 →</strong>
        </Link>
        <Link to="/playground">采集阶段与算子</Link>
        <Link to="/research">钻取 kernel / 硬件证据</Link>
        <Link to="/quantization">生成 RTN 量化候选</Link>
      </div>
      <QueryState loading={query.isPending} error={query.error} />
      <div className="stats-grid">
        <Stat
          label="可用部署"
          value={ready}
          unit={`/ ${data?.deployments.length ?? 0}`}
          hint="模型加载并预热完成"
          accent
        />
        <Stat label="运行记录" value={data?.runs.total} hint="交互推理与部署任务" />
        <Stat label="历史实验" value={data?.evidence_count} hint="来自已有 raw/verdict.json" />
        <Stat label="待处理任务" value={data?.pending} hint="独立执行进程 · 有界队列" />
      </div>
      <div className="grid-two">
        <Panel
          title="执行环境"
          subtitle="当前配置与真实生命周期"
          extra={
            <Link to="/devices">
              管理设备 <ArrowRightOutlined />
            </Link>
          }
        >
          {data?.deployments.map((d) => (
            <div className="deployment-row" key={d.id}>
              <div className="device-icon">
                <DeploymentUnitOutlined />
              </div>
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
            <span>主机可用内存</span>
            <strong>
              {fmt(gib(last?.host_available_bytes), 2)} <small>GiB</small>
            </strong>
          </div>
          {last?.host_total_bytes && last.host_available_bytes != null ? (
            <Progress
              percent={Math.round((1 - last.host_available_bytes / last.host_total_bytes) * 100)}
              showInfo={false}
              strokeColor="#f17b53"
            />
          ) : null}
          <div className="muted small">
            {last
              ? `${data?.telemetry.host} · 主机共享内存口径，非模型独占显存`
              : '等待远端采集；缺失值不会补零'}
          </div>
        </Panel>
        <Panel title="设备观测" subtitle="最近 120 个真实采样点 · 控制节点">
          <Chart
            label="控制节点 GPU 利用率时间序列"
            height={220}
            option={{
              grid: { left: 38, right: 12, top: 20, bottom: 28 },
              tooltip: { trigger: 'axis' },
              xAxis: {
                type: 'category',
                data: samples.map((x) => new Date(x.time * 1000).toLocaleTimeString()),
              },
              yAxis: { type: 'value', max: 100, axisLabel: { formatter: '{value}%' } },
              series: [
                {
                  name: 'GPU busy',
                  type: 'line',
                  data: samples.map((x) => x.gpu_util_pct),
                  showSymbol: false,
                  connectNulls: false,
                  lineStyle: { color: '#258c7e', width: 2 },
                  areaStyle: { color: 'rgba(37,140,126,.08)' },
                },
              ],
            }}
          />
          <div className="muted small">
            {last?.gpu_availability === 'measured'
              ? '来源：tegrastats；GPU busy 不等于 SM occupancy。'
              : '当前未采集 GPU 利用率。曲线保持空缺。'}
          </div>
        </Panel>
      </div>
      <Panel
        title="最近运行"
        subtitle="执行完成与质量通过是不同结论"
        extra={
          <Link to="/runs">
            全部运行 <ArrowRightOutlined />
          </Link>
        }
      >
        <Table<Run>
          rowKey="id"
          size="middle"
          pagination={false}
          dataSource={data?.runs.items ?? []}
          scroll={{ x: 700 }}
          columns={[
            {
              title: '运行',
              dataIndex: 'id',
              render: (id: string) => (
                <Link className="mono" to={`/runs/${id}`}>
                  {id.slice(0, 16)}
                </Link>
              ),
            },
            {
              title: '类型',
              dataIndex: 'kind',
              render: runKindLabel,
            },
            { title: '状态', dataIndex: 'state', render: (s: string) => <Status state={s} /> },
            {
              title: '首内容',
              render: (_, r) => <span>{fmt(r.metrics.console_first_content_ms)} ms</span>,
            },
            { title: '创建时间', dataIndex: 'created_at', render: date },
          ]}
        />
      </Panel>
      <div className="foot-note">
        <Tag color="blue">证据优先</Tag>{' '}
        交互数据用于功能诊断。正式加速结论还需要质量门、重复实验和可比性验证。
      </div>
    </>
  );
}
