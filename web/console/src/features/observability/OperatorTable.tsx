import { useMemo, useState } from 'react';
import { Alert, Empty, Input, Table, Tag } from 'antd';
import { fmt } from '../../api/client';
import { JsonView } from '../../components';
import type { Observation, Operator } from './types';

export default function OperatorTable({ profile }: { profile?: Observation['profile'] }) {
  const [search, setSearch] = useState('');
  const rows = useMemo(
    () =>
      (profile?.operators ?? [])
        .map((operator, index) => ({ ...operator, key: `${operator.name}:${index}` }))
        .filter((operator) => operator.name.toLowerCase().includes(search.toLowerCase())),
    [profile?.operators, search],
  );
  if (!profile || !profile.operators?.length)
    return (
      <Empty
        description={`没有已采集算子事件${profile?.status ? ` · ${profile.status}` : ''}。请在推理工作台提交前启用算子诊断。`}
      />
    );
  return (
    <>
      <Alert
        type="warning"
        showIcon
        className="section-gap"
        message="插桩诊断 · CPU self、CUDA self 和阶段墙钟时间不可混为一谈"
        description={(profile.limitations ?? []).join('；')}
      />
      <div className="toolbar">
        <Input.Search
          aria-label="搜索采集算子"
          placeholder="搜索算子名称"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          style={{ maxWidth: 340 }}
        />
        <Tag>
          {profile.source} · {profile.status}
        </Tag>
        <Tag>活动：{profile.activities?.join(' + ') || '未声明'}</Tag>
      </div>
      {profile.coverage && (
        <p className="small muted">
          采集窗口：prefill {profile.coverage.prefill ? '已覆盖' : '未覆盖'}；decode{' '}
          {profile.coverage.decode_steps} step；输出共 {profile.coverage.total_output_tokens}{' '}
          token。{profile.coverage.scope}
        </p>
      )}
      {profile.coverage?.module_scope === 'first_decoder_layer_only' && (
        <p className="help-text">
          模块归属仅标注首个 decoder 层；算子记录覆盖整个采集窗口。其他层没有逐模块
          range，不能据此声称已完成全部层归因。
        </p>
      )}
      {profile.operator_groups_truncated && (
        <Alert
          type="info"
          className="section-gap"
          message={`界面摘要已截断；原始分组共 ${profile.operator_groups_total ?? '未知'} 个。完整信息请下载 trace。`}
        />
      )}
      <Table<Operator & { key: string }>
        size="small"
        rowKey="key"
        dataSource={rows}
        pagination={{ pageSize: 15 }}
        scroll={{ x: 1000 }}
        expandable={{
          expandedRowRender: (row) => (
            <JsonView
              value={{
                input_shapes: row.input_shapes,
                source: row.source,
                cpu_memory_bytes: row.cpu_memory_bytes,
                device_memory_bytes: row.device_memory_bytes,
              }}
            />
          ),
        }}
        columns={[
          {
            title: '算子 / 分组',
            dataIndex: 'name',
            width: 300,
            render: (value: string) => <code className="kernel-name">{value}</code>,
          },
          { title: '调用', dataIndex: 'calls', sorter: (a, b) => a.calls - b.calls },
          {
            title: 'CPU total / ms',
            dataIndex: 'cpu_ms',
            render: (value: number) => fmt(value, 3),
          },
          {
            title: 'CPU self / ms',
            dataIndex: 'self_cpu_ms',
            render: (value: number) => fmt(value, 3),
            sorter: (a, b) => a.self_cpu_ms - b.self_cpu_ms,
            defaultSortOrder: 'descend',
          },
          {
            title: 'CUDA total / ms',
            dataIndex: 'cuda_ms',
            render: (value: number | null) => fmt(value, 3),
          },
          {
            title: 'CUDA self / ms',
            dataIndex: 'self_cuda_ms',
            render: (value: number | null) => fmt(value, 3),
            sorter: (a, b) => (a.self_cuda_ms ?? -1) - (b.self_cuda_ms ?? -1),
          },
        ]}
      />
      <p className="help-text">
        同名算子可能按输入 shape 分组。CUDA 指标为 — 表示没有可用设备计时，不解释成
        0；展开行查看形状和内存事件。
      </p>
    </>
  );
}
