import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Empty, Input, Select, Table, Tabs, Tag } from 'antd';
import { api, date, fmt, gib } from '../../api/client';
import type { Deployment } from '../../api/types';
import { Panel, QueryState, Stat } from '../../components';

type TensorEntry = {
  name: string;
  kind: string;
  shape: number[];
  dtype: string;
  device: string;
  logical_bytes: number;
  storage_key: string | null;
  storage_offset: number | null;
  storage_bytes: number | null;
  layer?: number | string | null;
};
type Inventory = {
  schema_version: number;
  scope: string;
  availability: 'complete' | 'partial' | 'unavailable';
  entries: TensorEntry[];
  logical_total_bytes: number | null;
  unique_storage_bytes: number | null;
  unique_storages: number | null;
  entry_count: number;
  truncated: boolean;
  limit: number;
  limitations: string[];
  unobserved: { name: string; reason: string }[];
};
type DeploymentInventory = {
  schema_version: number;
  deployment_id: string;
  source: string;
  sampled_at: number | null;
  parameters: Inventory | null;
  latest_kv: { run_id: string; sampled_at: number; inventory: Inventory } | null;
  limitations: string[];
};

function InventoryTable({
  inventory,
  sampledAt,
  description,
}: {
  inventory: Inventory | null;
  sampledAt: number | null;
  description: string;
}) {
  const [search, setSearch] = useState('');
  const rows = useMemo(
    () =>
      (inventory?.entries ?? [])
        .map((entry, index) => ({ ...entry, key: `${entry.name}:${index}` }))
        .filter((entry) =>
          `${entry.name} ${entry.layer ?? ''} ${entry.kind}`
            .toLowerCase()
            .includes(search.toLowerCase()),
        ),
    [inventory?.entries, search],
  );
  if (!inventory)
    return (
      <Empty description="该部署尚无此类快照。模型卸载后，不将旧工作进程账本当作当前驻留数据。" />
    );
  return (
    <>
      <div className="toolbar">
        <Tag
          color={
            inventory.availability === 'complete'
              ? 'green'
              : inventory.availability === 'partial'
                ? 'orange'
                : undefined
          }
        >
          {inventory.availability === 'complete'
            ? '本次账本范围完整'
            : inventory.availability === 'partial'
              ? '部分可见'
              : '当前不可用'}
        </Tag>
        <span className="small muted">
          {sampledAt ? `采样于 ${date(sampledAt)}` : '采样时间未提供'} · {description}
        </span>
      </div>
      <div className="stats-grid">
        <Stat
          label="张量逻辑字节合计"
          value={gib(inventory.logical_total_bytes)}
          unit="GiB"
          hint="按张量元素计数；共享 storage 可能重复"
        />
        <Stat
          label="去重 storage 容量"
          value={gib(inventory.unique_storage_bytes)}
          unit="GiB"
          hint="仅此账本中的 storage；不等于全部显存"
        />
        <Stat
          label="独立 storage 数"
          value={inventory.unique_storages}
          hint="按本次工作进程内匿名标识去重"
        />
        <Stat
          label="账本张量数"
          value={inventory.entry_count}
          hint={`当前返回 ${inventory.entries.length} 条；上限 ${inventory.limit}`}
        />
      </div>
      {inventory.truncated && (
        <Alert
          className="section-gap"
          type="warning"
          showIcon
          message="账本达到条目上限，当前表格只包含已记录部分。请结合统计范围与限制解释总量。"
        />
      )}
      <Input.Search
        className="section-gap"
        aria-label="搜索张量名称或层"
        placeholder="搜索权重名称 / layer / kind"
        allowClear
        value={search}
        onChange={(event) => setSearch(event.target.value)}
        style={{ maxWidth: 400 }}
      />
      <Table<TensorEntry & { key: string }>
        rowKey="key"
        size="small"
        dataSource={rows}
        scroll={{ x: 1160 }}
        pagination={{
          pageSize: 20,
          showSizeChanger: false,
          showTotal: (total) => `${total} 条匹配记录`,
        }}
        columns={[
          {
            title: '张量 / buffer',
            dataIndex: 'name',
            width: 300,
            render: (value: string) => <code className="kernel-name">{value}</code>,
          },
          {
            title: '种类 / 层',
            render: (_, entry) => (
              <>
                {entry.kind}
                {entry.layer != null && <div className="small muted">layer {entry.layer}</div>}
              </>
            ),
          },
          {
            title: 'Shape',
            dataIndex: 'shape',
            render: (shape: number[]) => (
              <code className="kernel-name">{shape.length ? shape.join(' × ') : 'scalar'}</code>
            ),
            width: 150,
          },
          {
            title: 'Dtype / 设备',
            render: (_, entry) => (
              <>
                {entry.dtype}
                <div className="small muted">{entry.device}</div>
              </>
            ),
          },
          {
            title: '逻辑 / MiB',
            dataIndex: 'logical_bytes',
            render: (value: number) => fmt(value / 1024 ** 2, 4),
            sorter: (left, right) => left.logical_bytes - right.logical_bytes,
          },
          {
            title: 'Storage / MiB',
            dataIndex: 'storage_bytes',
            render: (value: number | null) => fmt(value == null ? null : value / 1024 ** 2, 4),
          },
          {
            title: '共享 storage 匿名标识',
            dataIndex: 'storage_key',
            width: 170,
            render: (value: string | null) =>
              value ? (
                <code title={value} className="kernel-name">
                  {value}
                </code>
              ) : (
                '—'
              ),
          },
          {
            title: 'Storage offset',
            dataIndex: 'storage_offset',
            render: (value: number | null) => fmt(value, 0),
          },
        ]}
      />
      <div className="help-text">
        <p>
          范围：{inventory.scope}。Storage
          标识仅用于进程内关联共享存储，不能解释成物理地址，也不能跨工作进程关联；offset
          的单位以张量元素计。
        </p>
        {inventory.limitations.map((limitation, index) => (
          <p key={index}>{limitation}</p>
        ))}
        {inventory.unobserved.length > 0 && (
          <details>
            <summary>{inventory.unobserved.length} 个对象未完整观测</summary>
            {inventory.unobserved.map((entry, index) => (
              <p key={index}>
                <code className="kernel-name">{entry.name}</code>：{entry.reason}
              </p>
            ))}
          </details>
        )}
      </div>
    </>
  );
}

export default function TensorLedger() {
  const [selected, setSelected] = useState('');
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 5000,
  });
  const deployment =
    deployments.data?.items.find((entry) => entry.id === selected) ?? deployments.data?.items[0];
  const query = useQuery({
    queryKey: ['memory-inventory', deployment?.id],
    queryFn: () => api<DeploymentInventory>(`/deployments/${deployment!.id}/memory-inventory`),
    enabled: !!deployment,
    refetchInterval: 10000,
  });
  return (
    <Panel
      title="张量内存账本"
      subtitle="权重、buffer 与最近 KV 的真实形状、精度、存储容量及共享关系"
    >
      <Alert
        className="section-gap"
        type="info"
        showIcon
        message="这是指定时刻的软件张量账本，不是逐字节的物理显存、cache 或寄存器轨迹。"
        description="加载快照记录权重与 buffer；KV 来自最近一次请求释放前的快照。逻辑字节、去重 storage、allocator 和系统内存代表不同范围，不能互相替代。"
      />
      <label className="field-label" htmlFor="inventory-deployment">
        账本部署
      </label>
      <Select
        id="inventory-deployment"
        className="wide-select"
        value={deployment?.id}
        onChange={setSelected}
        options={deployments.data?.items.map((entry) => ({
          value: entry.id,
          label: `${entry.name} · ${entry.state}`,
        }))}
      />
      <QueryState
        loading={query.isPending && !!deployment}
        error={query.error ?? deployments.error}
      />
      {!deployment && !deployments.isPending && <Empty description="尚无可查询部署" />}
      {query.data && (
        <>
          <Tabs
            items={[
              {
                key: 'parameters',
                label: '权重与 buffer',
                children: (
                  <InventoryTable
                    inventory={query.data.parameters}
                    sampledAt={query.data.sampled_at}
                    description="加载时快照"
                  />
                ),
              },
              {
                key: 'kv',
                label: '最近请求 KV',
                children: (
                  <InventoryTable
                    inventory={query.data.latest_kv?.inventory ?? null}
                    sampledAt={query.data.latest_kv?.sampled_at ?? null}
                    description={
                      query.data.latest_kv ? `请求 ${query.data.latest_kv.run_id}` : '没有 KV 快照'
                    }
                  />
                ),
              },
            ]}
          />
          <div className="help-text">
            {query.data.limitations.map((limitation, index) => (
              <p key={index}>{limitation}</p>
            ))}
          </div>
        </>
      )}
    </Panel>
  );
}
