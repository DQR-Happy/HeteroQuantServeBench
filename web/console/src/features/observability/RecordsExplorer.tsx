import Button from '../../components/ActionButton';
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Select, Table } from 'antd';
import { api } from '../../api/client';
import { JsonView, QueryState } from '../../components';

type RecordRow = {
  seq: number;
  kind: string;
  host_offset_ns?: string;
  payload: unknown;
  projection?: string;
};
type RecordPage = { items: RecordRow[]; next_after: number | null; response_budget_bytes: number };
export default function RecordsExplorer({ runId }: { runId: string }) {
  const [kind, setKind] = useState('all');
  const [cursors, setCursors] = useState([0]);
  const after = cursors.at(-1)!;
  const query = useQuery({
    queryKey: ['observation-records', runId, kind, after],
    queryFn: () =>
      api<RecordPage>(
        `/runs/${runId}/observations/records?limit=30&after=${after}${kind === 'all' ? '' : `&kind=${encodeURIComponent(kind)}`}`,
      ),
    staleTime: 60000,
  });
  return (
    <>
      <Alert
        type="info"
        showIcon
        message="有界原始记录查询"
        description="每次最多读取 30 条；超过单条响应预算的 payload 以占位说明显示，完整内容保留在原件。序号只属于本次观测包。"
      />
      <div className="toolbar section-gap">
        <Select
          aria-label="原始记录类型"
          value={kind}
          style={{ minWidth: 230 }}
          onChange={(v) => {
            setKind(v);
            setCursors([0]);
          }}
          options={[
            'all',
            'boundary',
            'memory_sample',
            'os_sample',
            'kv_inventory',
            'tensor_inventory',
            'allocator_snapshot',
            'page_sample',
            'source_status',
          ].map((value) => ({ value, label: value === 'all' ? '全部记录类型' : value }))}
        />
        <span className="muted small">
          第 {cursors.length} 页 · 当前游标 {after}
        </span>
      </div>
      <QueryState loading={query.isPending} error={query.error} />
      <Table<RecordRow>
        size="small"
        rowKey="seq"
        dataSource={query.data?.items ?? []}
        pagination={false}
        scroll={{ x: 580 }}
        columns={[
          { title: '来源序号', dataIndex: 'seq' },
          { title: '记录类型', dataIndex: 'kind' },
          { title: '预览状态', render: (_, r) => r.projection ?? 'payload 可展开' },
        ]}
        expandable={{
          expandedRowRender: (row) =>
            row.projection ? (
              <Alert
                type="warning"
                message={row.projection}
                description="预览暂无数据，请下载原始记录查看。"
              />
            ) : (
              <JsonView value={row} />
            ),
        }}
      />
      <div className="toolbar section-gap">
        <Button
          disabledReason={{
            reason: query.isPending
              ? '观测记录正在读取，请稍候。'
              : '已经位于第一页，没有更早的分页。',
          }}
          disabled={cursors.length === 1 || query.isPending}
          onClick={() => setCursors((prev) => prev.slice(0, -1))}
        >
          上一页
        </Button>
        <Button
          disabledReason={{
            reason: query.isPending
              ? '观测记录正在读取，请稍候。'
              : '当前记录没有下一页。可调整本页筛选条件后重新查询。',
          }}
          disabled={query.isPending || query.data?.next_after == null}
          onClick={() => {
            const next = query.data?.next_after;
            if (next != null) setCursors((prev) => [...prev, next]);
          }}
        >
          下一页
        </Button>
      </div>
    </>
  );
}
