import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Empty, Input, InputNumber, Space, Table } from 'antd';
import { api, fmt } from '../../api/client';
import { JsonView, QueryState } from '../../components';

type TraceEvent = {
  id: string;
  name: string;
  category: string;
  pid: number | string;
  tid: number | string;
  start_ms: number;
  duration_ms: number | null;
  args: unknown;
  source: string;
};
type TracePage = {
  items: TraceEvent[];
  total: number;
  next_offset: number | null;
  summary: unknown;
};

export default function TraceExplorer({ runId, available }: { runId: string; available: boolean }) {
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('');
  const [start, setStart] = useState<number | null>(null);
  const [end, setEnd] = useState<number | null>(null);
  const [offset, setOffset] = useState(0);
  const rangeValid = start == null || end == null || start <= end;
  const query = useQuery({
    queryKey: ['trace-events', runId, search, category, start, end, offset],
    queryFn: () => {
      const params = new URLSearchParams({ offset: String(offset), limit: '50' });
      if (search) params.set('search', search);
      if (category) params.set('category', category);
      if (start != null) params.set('start_ms', String(start));
      if (end != null) params.set('end_ms', String(end));
      return api<TracePage>(`/runs/${runId}/trace/events?${params}`);
    },
    enabled: available && rangeValid,
    staleTime: 60000,
  });
  if (!available)
    return (
      <Empty description="当前请求没有可查询 trace。选择算子诊断，等待采集与导出完成后查看。" />
    );
  return (
    <>
      <Alert
        className="section-gap"
        showIcon
        type="info"
        message="查询原始采集事件"
        description="服务端过滤与分页，每页最多读取 50 条；start / end 是 trace 内的相对毫秒，不与主机阶段时钟强行对齐。展开事件查看 correlation、stream、shape 或原始参数。"
      />
      <div className="toolbar">
        <Input.Search
          aria-label="搜索 trace 事件"
          placeholder="搜索 kernel / 算子 / memcpy"
          allowClear
          onSearch={(value) => {
            setSearch(value);
            setOffset(0);
          }}
          style={{ maxWidth: 300 }}
        />
        <Input.Search
          aria-label="筛选 trace 类别"
          placeholder="类别，如 kernel / cpu_op"
          allowClear
          onSearch={(value) => {
            setCategory(value);
            setOffset(0);
          }}
          style={{ maxWidth: 250 }}
        />
        <Space wrap>
          <InputNumber
            aria-label="Trace 起始毫秒"
            placeholder="起始 ms"
            min={0}
            value={start}
            onChange={(value) => {
              setStart(value);
              setOffset(0);
            }}
          />
          <span>—</span>
          <InputNumber
            aria-label="Trace 结束毫秒"
            placeholder="结束 ms"
            min={0}
            value={end}
            onChange={(value) => {
              setEnd(value);
              setOffset(0);
            }}
          />
        </Space>
      </div>
      {!rangeValid && <Alert type="warning" message="结束时间必须大于等于开始时间。" />}
      <QueryState loading={query.isPending && rangeValid} error={query.error} />
      <Table<TraceEvent>
        rowKey="id"
        size="small"
        dataSource={query.data?.items ?? []}
        scroll={{ x: 950 }}
        pagination={{
          current: offset / 50 + 1,
          pageSize: 50,
          total: query.data?.total ?? 0,
          showSizeChanger: false,
          onChange: (page) => setOffset((page - 1) * 50),
          showTotal: (total) => `${total} 个匹配事件`,
        }}
        expandable={{
          expandedRowRender: (row) => <JsonView value={{ args: row.args, source: row.source }} />,
        }}
        columns={[
          {
            title: '采集事件',
            dataIndex: 'name',
            width: 380,
            render: (value: string) => <code className="kernel-name">{value}</code>,
          },
          { title: '类别', dataIndex: 'category', width: 130 },
          { title: '进程 / 线程', render: (_, row) => `${row.pid} / ${row.tid}`, width: 140 },
          { title: '开始 / ms', dataIndex: 'start_ms', render: (value: number) => fmt(value, 4) },
          {
            title: '持续 / ms',
            dataIndex: 'duration_ms',
            render: (value: number | null) => fmt(value, 4),
          },
        ]}
      />
      {query.data?.summary != null && (
        <details>
          <summary>Trace 覆盖与完整度</summary>
          <JsonView value={query.data.summary} />
        </details>
      )}
    </>
  );
}
