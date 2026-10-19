import Button from '../../components/ActionButton';
import { Alert, Empty, Space, Table, Tabs, Tag } from 'antd';
import { DownloadOutlined } from '@ant-design/icons';
import { base, date, fmt, gib } from '../../api/client';
import { JsonView } from '../../components';
import PhaseTimeline from './PhaseTimeline';
import OperatorTable from './OperatorTable';
import TraceExplorer from './TraceExplorer';
import OptimizationFindings from './OptimizationFindings';
import MechanismPanel from './MechanismPanel';
import type { Observation } from './types';

export default function ObservationPanel({
  runId,
  observation,
}: {
  runId: string;
  observation?: Observation;
}) {
  if (!observation || observation.mode === 'off')
    return (
      <Empty description="本次请求未启用详细观测。历史请求无法事后补录；请在推理工作台选择基础观测或算子诊断后重新提交。" />
    );
  return (
    <>
      <div className="toolbar observation-toolbar">
        <Space wrap>
          <Tag color="green">
            {
              { operators: '算子诊断', memory: '内存诊断', pages: '页诊断', basic: '基础观测' }[
                observation.mode
              ]
            }
          </Tag>
          <span className="small muted">{observation.clock_domain}</span>
        </Space>
        <Space wrap>
          <Button
            icon={<DownloadOutlined />}
            disabledReason={{
              reason: '此运行没有采集到可下载的设备时间线。请在交互推理中选择算子观测后重新运行。',
              next: { to: '/playground', label: '配置算子观测并运行' },
            }}
            disabled={!observation.profile?.trace_available}
            href={
              observation.profile?.trace_available
                ? `${base}/runs/${runId}/trace?format=chrome`
                : undefined
            }
          >
            下载 Chrome trace
          </Button>
          <Button href={`${base}/runs/${runId}/optimization?format=markdown`}>导出优化方案</Button>
          <Button href={`${base}/runs/${runId}/optimization?format=json`}>方案 JSON</Button>
          <Button href={`${base}/runs/${runId}/bundle`}>证据摘要包 ZIP</Button>
        </Space>
      </div>
      <Alert
        className="section-gap"
        type="info"
        showIcon
        message="可追溯范围"
        description={
          observation.limitations.join('；') ||
          '观测由工作进程采集；基础模式展示主机阶段，详细模式提供算子与设备活动。'
        }
      />
      <Tabs
        defaultActiveKey="phases"
        items={[
          {
            key: 'mechanisms',
            label: '布局与系统机制',
            children: <MechanismPanel runId={runId} />,
          },
          {
            key: 'phases',
            label: '阶段与逐 token',
            children: <PhaseTimeline phases={observation.phases} tokens={observation.tokens} />,
          },
          {
            key: 'operators',
            label: '算子耗时',
            children: <OperatorTable profile={observation.profile} />,
          },
          {
            key: 'trace',
            label: '原始事件查询',
            children: (
              <TraceExplorer runId={runId} available={!!observation.profile?.trace_available} />
            ),
          },
          {
            key: 'optimization',
            label: '优化假设',
            children: <OptimizationFindings runId={runId} />,
          },
          {
            key: 'memory',
            label: '请求内存快照',
            children: (
              <>
                <Table<Observation['memory_snapshots'][number]>
                  rowKey={(row, index) => `${row.label}:${index}`}
                  size="small"
                  dataSource={observation.memory_snapshots}
                  pagination={false}
                  scroll={{ x: 900 }}
                  columns={[
                    { title: '生命周期', dataIndex: 'label' },
                    { title: '采样时间', dataIndex: 'time', render: date },
                    {
                      title: '整机可用 / GiB',
                      render: (_, row) => fmt(gib(row.host_available_bytes), 3),
                    },
                    {
                      title: 'Allocated / GiB',
                      render: (_, row) => fmt(gib(row.allocated_bytes), 3),
                    },
                    {
                      title: 'Reserved / GiB',
                      render: (_, row) => fmt(gib(row.reserved_bytes), 3),
                    },
                    {
                      title: 'Peak / GiB',
                      render: (_, row) => fmt(gib(row.peak_allocated_bytes), 3),
                    },
                  ]}
                />
                <p className="help-text">
                  整机可用内存与 worker allocator
                  分别按各自口径统计，快照记录指定采样时刻的分配状态。
                </p>
              </>
            ),
          },
          {
            key: 'execution',
            label: '执行真实性',
            children: <JsonView value={observation.execution} />,
          },
          { key: 'raw', label: '原始观测', children: <JsonView value={observation} /> },
        ]}
      />
    </>
  );
}
