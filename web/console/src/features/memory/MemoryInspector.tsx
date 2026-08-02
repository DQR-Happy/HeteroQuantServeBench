import { useQuery } from '@tanstack/react-query';
import { Alert, Button, Empty, Table, Tag } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import { api, date, fmt, gib } from '../../api/client';
import { Panel, QueryState, Stat } from '../../components';
import type { MemoryObservation, Resources } from './types';

const roles: Record<string, string> = {
  api: 'Console API',
  control: 'Console API',
  console_api: 'Console API',
  worker: '模型工作进程',
  model_worker: '模型工作进程',
};

function Snapshot({ value, label }: { value?: MemoryObservation | null; label: string }) {
  return (
    <div className="memory-snapshot">
      <strong>{label}</strong>
      <span className="small muted">
        {value?.sampled_at ? date(value.sampled_at) : '采样时刻未提供'}
        {value?.run_id && (
          <>
            {' '}
            · <Link to={`/runs/${value.run_id}`}>对应运行</Link>
          </>
        )}
      </span>
      {value?.memory ? (
        <dl className="metric-pairs">
          <dt>Allocated</dt>
          <dd>{fmt(gib(value.memory.allocated_bytes), 3)} GiB</dd>
          <dt>Reserved</dt>
          <dd>{fmt(gib(value.memory.reserved_bytes), 3)} GiB</dd>
          <dt>Peak allocated</dt>
          <dd>{fmt(gib(value.memory.peak_allocated_bytes), 3)} GiB</dd>
          <dt>统计范围</dt>
          <dd>{value.memory.scope ?? 'worker allocator'}</dd>
        </dl>
      ) : (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="该生命周期尚无 allocator 快照" />
      )}
    </div>
  );
}

export default function MemoryInspector() {
  const query = useQuery({
    queryKey: ['resources'],
    queryFn: () => api<Resources>('/resources'),
    refetchInterval: 5000,
  });
  const resource = query.data;
  return (
    <Panel
      title="内存归因与生命周期"
      subtitle={
        resource
          ? `控制节点 · 采样于 ${date(resource.sampled_at)}`
          : '读取整机、进程与工作进程的不同内存口径'
      }
      extra={
        <Button
          icon={<ReloadOutlined />}
          loading={query.isFetching}
          onClick={() => void query.refetch()}
        >
          刷新内存
        </Button>
      }
    >
      <QueryState loading={query.isPending} error={query.error} />
      {resource && (
        <>
          <div className="stats-grid memory-stats">
            <Stat
              label="系统可见总内存"
              value={gib(resource.host.total_bytes)}
              unit="GiB"
              hint="Linux MemTotal"
            />
            <Stat
              label="系统现在可用"
              value={gib(resource.host.available_bytes)}
              unit="GiB"
              hint="MemAvailable · 已包含模型占用的影响"
              accent
            />
            <Stat
              label="完全空闲页"
              value={gib(resource.host.free_bytes)}
              unit="GiB"
              hint="MemFree · 不等于可用内存"
            />
            <Stat
              label="已用 Swap"
              value={gib(resource.host.swap_used_bytes)}
              unit="GiB"
              hint="占用不代表此刻正在换页"
            />
          </div>
          <Alert
            type="info"
            showIcon
            className="section-gap"
            message="模型已加载时，可用内存已经扣除了它的占用；不要再减一次权重大小。"
            description="RSS、PSS 和 CUDA allocator 的统计范围不同，不做相加饼图。Reserved 包含 Allocated。工作进程快照有独立采样时间，不冒充实时值；外部推理服务的内存不在此控制节点统计中。"
          />
          <Table<Resources['processes'][number]>
            rowKey="pid"
            size="small"
            pagination={false}
            dataSource={resource.processes}
            scroll={{ x: 540 }}
            columns={[
              {
                title: '进程',
                render: (_, row) =>
                  roles[row.role] ??
                  (row.role.startsWith('worker:')
                    ? `模型工作进程 · ${row.role.slice(7)}`
                    : row.role),
              },
              { title: 'PID', dataIndex: 'pid' },
              {
                title: 'RSS / MiB',
                render: (_, row) =>
                  fmt(row.rss_bytes == null ? null : row.rss_bytes / 1024 ** 2, 2),
              },
              {
                title: 'PSS / MiB',
                render: (_, row) =>
                  fmt(row.pss_bytes == null ? null : row.pss_bytes / 1024 ** 2, 2),
              },
              {
                title: '可用性',
                dataIndex: 'availability',
                render: (value: string) => <Tag>{value}</Tag>,
              },
            ]}
          />
          {resource.deployments.map((deployment) => (
            <div className="memory-deployment" key={deployment.id}>
              <h3>
                {deployment.id} <Tag>{deployment.state}</Tag>
              </h3>
              <div className="grid-two">
                <Snapshot label="加载完成快照" value={deployment.load_observation} />
                <Snapshot label="最近请求快照" value={deployment.latest_observation} />
              </div>
              {!!deployment.lifecycle?.length && (
                <Table
                  size="small"
                  rowKey={(row, index) => `${row.label}:${index}`}
                  pagination={false}
                  dataSource={deployment.lifecycle}
                  columns={[
                    { title: '生命周期采样', dataIndex: 'label' },
                    { title: '时间', dataIndex: 'sampled_at', render: date },
                    {
                      title: '整机可用 / GiB',
                      render: (_, row) => fmt(gib(row.available_bytes), 3),
                    },
                  ]}
                />
              )}
            </div>
          ))}
          <div className="help-text">
            {resource.limitations.map((limitation, index) => (
              <p key={index}>{limitation}</p>
            ))}
            不再需要推理时，可在上方卸载模型；关闭网页不会释放远端模型权重。
          </div>
        </>
      )}
    </Panel>
  );
}
