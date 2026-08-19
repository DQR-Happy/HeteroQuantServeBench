import { useState } from 'react';
import { Alert, Empty, Input, Select, Table, Tabs, Tag } from 'antd';
import { Chart, JsonView } from '../../components';
import { fmt } from '../../api/client';
import { byteValue, scopeName, type Evidence, type Metric, type Tensor } from './mechanisms';

export function MetricView({ evidence }: { evidence: Evidence }) {
  const [scope, setScope] = useState('all');
  const [search, setSearch] = useState('');
  const rows = evidence.metrics.filter(
    (r) =>
      (scope === 'all' || r.scope === scope) &&
      r.metric_id.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <>
      <div className="toolbar">
        <Input.Search
          aria-label="搜索观测指标"
          placeholder="搜索指标，如 RSS / fault / reserved"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ maxWidth: 350 }}
        />
        <Select
          aria-label="指标测量范围"
          value={scope}
          onChange={setScope}
          style={{ minWidth: 220 }}
          options={[
            { value: 'all', label: '全部测量范围' },
            ...[...new Set(evidence.metrics.map((r) => r.scope))].map((value) => ({
              value,
              label: scopeName(value),
            })),
          ]}
        />
        <span className="muted small">
          {rows.length} / {evidence.metrics.length} 项 · 当前观测包
        </span>
      </div>
      <Table<Metric>
        size="small"
        rowKey={(r) => `${r.metric_id}:${r.label ?? ''}`}
        dataSource={rows}
        pagination={{ pageSize: 12, showSizeChanger: false }}
        scroll={{ x: 920 }}
        locale={{ emptyText: '当前条件没有匹配指标' }}
        columns={[
          { title: '指标', dataIndex: 'metric_id', render: (v) => <code>{v}</code> },
          { title: '采样边界', dataIndex: 'label', render: (v) => v ?? '请求区间差值' },
          {
            title: '数值',
            align: 'right',
            render: (_, r) => (
              <span
                className="numeric"
                title={r.value == null ? r.availability : `${r.value} ${r.unit}`}
              >
                {r.value == null
                  ? '未采集'
                  : r.unit === 'byte'
                    ? byteValue(r.value)
                    : `${fmt(r.value, 0)} ${r.unit}`}
              </span>
            ),
          },
          { title: '测量范围', dataIndex: 'scope', render: scopeName },
          {
            title: '证据类型',
            render: (_, r) => (
              <Tag>
                {r.evidence_kind === 'derived'
                  ? '派生'
                  : r.evidence_kind === 'measured'
                    ? '直接测量'
                    : r.evidence_kind}
              </Tag>
            ),
          },
          { title: '来源序号', render: (_, r) => r.source_record_seqs?.join(', ') ?? '—' },
        ]}
      />
      <p className="help-text">
        Allocated / Reserved 属于 Torch 分配器；RSS / PSS 属于进程；system_*
        属于整机。这些口径互相重叠，不能相加。悬停数值可查看原始单位。
      </p>
    </>
  );
}

function TensorTable({ entries }: { entries: Tensor[] }) {
  const [search, setSearch] = useState('');
  const rows = entries.filter((r) => r.name.toLowerCase().includes(search.toLowerCase()));
  return (
    <>
      <div className="toolbar">
        <Input.Search
          aria-label="搜索 Tensor 名称"
          placeholder="按层或 Tensor 名称筛选"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ maxWidth: 350 }}
        />
        <span className="muted small">{rows.length} 个已列出 Tensor</span>
      </div>
      <Table<Tensor>
        size="small"
        rowKey="name"
        dataSource={rows}
        pagination={{ pageSize: 12, showSizeChanger: false }}
        scroll={{ x: 1000 }}
        columns={[
          { title: 'Tensor', dataIndex: 'name', render: (v) => <code>{v}</code> },
          {
            title: 'Shape / Stride',
            render: (_, r) => (
              <span className="mono">
                [{r.shape.join(', ')}]<br />
                <span className="muted">[{r.stride.join(', ')}]</span>
              </span>
            ),
          },
          {
            title: '类型 / 设备',
            render: (_, r) => (
              <>
                {r.dtype}
                <br />
                <span className="muted">{r.device}</span>
              </>
            ),
          },
          {
            title: '逻辑占用',
            align: 'right',
            dataIndex: 'logical_bytes',
            render: byteValue,
            sorter: (a, b) => a.logical_bytes - b.logical_bytes,
          },
          { title: 'Storage', align: 'right', dataIndex: 'storage_bytes', render: byteValue },
          { title: '连续', render: (_, r) => (r.contiguous ? '是' : '否') },
          {
            title: '虚拟地址',
            dataIndex: 'storage_address',
            render: (v) => <code>{v ?? '未采集'}</code>,
          },
        ]}
        expandable={{ expandedRowRender: (row) => <JsonView value={row} /> }}
      />
    </>
  );
}

export function LayoutView({ evidence, kind }: { evidence: Evidence; kind: 'kv' | 'weights' }) {
  const [selected, setSelected] = useState<string>();
  const snapshots = (
    kind === 'kv'
      ? (evidence.kv ?? [])
      : (evidence.tensor_inventories ?? []).map((r) => ({
          ...r.inventory,
          label: r.label,
          logical_bytes: r.inventory.logical_total_bytes,
        }))
  ).map((r, index) => ({
    ...r,
    selectionId: String(index),
    boundary: 'step' in r && r.step != null ? `${r.label} · step ${r.step}` : r.label,
  }));
  const current = snapshots.find((r) => r.selectionId === selected) ?? snapshots.at(-1);
  if (!current)
    return (
      <Empty
        description={`本次观测没有${kind === 'kv' ? ' KV' : '权重'}布局记录；请核对采集模式与范围。`}
      />
    );
  return (
    <>
      {kind === 'kv' && (
        <Chart
          height={190}
          label="已采样 KV 的逻辑字节与去重 Storage 字节"
          option={{
            tooltip: { trigger: 'axis' },
            legend: { top: 0 },
            grid: { left: 65, right: 20, top: 35, bottom: 40 },
            xAxis: {
              type: 'category',
              data: snapshots.map((r) => r.boundary),
              axisLabel: { hideOverlap: true },
            },
            yAxis: { type: 'value', name: 'MiB' },
            series: [
              {
                name: '逻辑大小',
                type: 'line',
                data: snapshots.map((r) => r.logical_bytes / 1024 ** 2),
                itemStyle: { color: '#435f82' },
              },
              {
                name: '去重 Storage',
                type: 'line',
                data: snapshots.map((r) => r.unique_storage_bytes / 1024 ** 2),
                itemStyle: { color: '#91703e' },
              },
            ],
          }}
        />
      )}
      <div className="capture-summary">
        <div>
          <span>采样边界</span>
          <Select
            aria-label="选择布局采样边界"
            value={current.selectionId}
            onChange={setSelected}
            options={snapshots.map((r) => ({ value: r.selectionId, label: r.boundary }))}
          />
        </div>
        <div>
          <span>逻辑字节 · 别名可重复</span>
          <strong>{byteValue(current.logical_bytes)}</strong>
        </div>
        <div>
          <span>Storage 字节 · 已观察去重</span>
          <strong>{byteValue(current.unique_storage_bytes)}</strong>
        </div>
        <div>
          <span>覆盖状态</span>
          <strong>{current.availability}</strong>
        </div>
      </div>
      <p className="help-text">
        虚拟地址仅定位当前进程与设备中的 Storage；不能推断物理 DRAM 布局。KV
        折线连接采样点，不代表连续逐 token 采集。
      </p>
      <TensorTable key={`${kind}:${current.selectionId}`} entries={current.entries} />
    </>
  );
}

export function AllocationView({ evidence }: { evidence: Evidence }) {
  const lifetime = evidence.allocation_lifetimes;
  return (
    <>
      <p className="help-text">
        按原生分配器序号观察申请与回收；地址复用使用 generation
        区分。序号不能作为纳秒时间，也不能推出驱动或物理页的释放时刻。
      </p>
      <Tabs
        items={[
          {
            key: 'blocks',
            label: '块与状态',
            children: (
              <Table
                size="small"
                rowKey="label"
                dataSource={evidence.allocator ?? []}
                pagination={false}
                scroll={{ x: 800 }}
                columns={[
                  { title: '快照边界', dataIndex: 'label' },
                  { title: 'Segments', dataIndex: 'segments', align: 'right' },
                  { title: 'Blocks', dataIndex: 'blocks', align: 'right' },
                  { title: 'Active', dataIndex: 'active_bytes', render: byteValue, align: 'right' },
                  {
                    title: 'Awaiting free',
                    dataIndex: 'awaiting_free_bytes',
                    render: byteValue,
                    align: 'right',
                  },
                  {
                    title: 'Inactive',
                    dataIndex: 'inactive_bytes',
                    render: byteValue,
                    align: 'right',
                  },
                  {
                    title: '历史前缀',
                    render: (_, r) => (r.history_prefix_may_be_overwritten ? '可能覆盖' : '已保留'),
                  },
                ]}
              />
            ),
          },
          {
            key: 'lifetimes',
            label: '分配代次',
            children: (
              <>
                <div className="help-text">
                  锚点 {lifetime?.anchor ?? '未采集'} · 预览 {lifetime?.items.length ?? 0} /{' '}
                  {lifetime?.total_items ?? lifetime?.items.length ?? 0} 条 · 完整序列见原始证据
                  {lifetime?.prefix_may_be_overwritten ? ' · 历史前缀可能已覆盖' : ''}
                </div>
                <Table
                  size="small"
                  rowKey={(r) => `${r.device}:${r.address}:${r.generation}`}
                  dataSource={lifetime?.items ?? []}
                  pagination={{ pageSize: 12, showSizeChanger: false }}
                  scroll={{ x: 900 }}
                  columns={[
                    {
                      title: '设备 / 虚拟地址',
                      render: (_, r) => (
                        <code>
                          {r.device} / {r.address}
                        </code>
                      ),
                    },
                    { title: 'Generation', dataIndex: 'generation' },
                    { title: '字节', dataIndex: 'bytes', render: byteValue },
                    { title: '申请序号', dataIndex: 'alloc_seq', render: (v) => v ?? '未知' },
                    { title: '请求释放', dataIndex: 'free_requested_seq', render: (v) => v ?? '—' },
                    { title: '完成释放', dataIndex: 'free_completed_seq', render: (v) => v ?? '—' },
                    { title: '起止状态', render: (_, r) => `${r.start_status} / ${r.end_status}` },
                  ]}
                />
              </>
            ),
          },
          {
            key: 'attribution',
            label: 'Tensor 关联',
            children: (
              <Table
                size="small"
                rowKey="tensor_name"
                dataSource={evidence.allocation_attribution ?? []}
                pagination={{ pageSize: 12, showSizeChanger: false }}
                scroll={{ x: 920 }}
                columns={[
                  { title: 'Tensor', dataIndex: 'tensor_name', render: (v) => <code>{v}</code> },
                  { title: 'Storage', dataIndex: 'storage_bytes', render: byteValue },
                  {
                    title: '分配虚拟地址',
                    dataIndex: 'allocation_address',
                    render: (v) => <code>{v ?? '未关联'}</code>,
                  },
                  { title: '代次', dataIndex: 'generation', render: (v) => v ?? '预存 / 未知' },
                  { title: '关联依据', dataIndex: 'confidence', render: (v) => <Tag>{v}</Tag> },
                  { title: '锚点', dataIndex: 'anchor' },
                ]}
              />
            ),
          },
        ]}
      />
    </>
  );
}

export function PageView({ evidence }: { evidence: Evidence }) {
  return (
    <>
      <Alert
        type="info"
        showIcon
        message="CPU 页样本 · 有界抽样"
        description="present / swapped 来自选定 CPU 虚拟页的 pagemap。抽样不提供 GPU 物理页表，不等于整个模型的页驻留率；PFN 不导出。"
      />
      <Table
        className="section-gap"
        size="small"
        rowKey="label"
        dataSource={evidence.page_samples ?? []}
        pagination={false}
        scroll={{ x: 780 }}
        locale={{ emptyText: '本次没有 CPU 页样本。请使用页诊断模式采集。' }}
        columns={[
          { title: '采样边界', dataIndex: 'label' },
          { title: '抽样页数', dataIndex: 'sampled_pages' },
          { title: 'Present 页', dataIndex: 'present_pages' },
          { title: 'Swapped 页', dataIndex: 'swapped_pages' },
          { title: '样本范围', dataIndex: 'scope' },
        ]}
        expandable={{ expandedRowRender: (row) => <JsonView value={row.selection} /> }}
      />
      <h3>OS 数据源可用性</h3>
      <div className="source-status">
        {Object.entries(evidence.os_source_availability ?? {}).map(([key, value]) => (
          <div key={key}>
            <code>{key}</code>
            <Tag>{value}</Tag>
          </div>
        ))}
      </div>
    </>
  );
}
