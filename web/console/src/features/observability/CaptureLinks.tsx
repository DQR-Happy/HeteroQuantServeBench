import { useQuery } from '@tanstack/react-query';
import { Alert, Empty, Table, Tag } from 'antd';
import { api } from '../../api/client';
import { JsonView, QueryState } from '../../components';

type Edge = { from_event: string; to_event: string; method: string; confidence: string };
type Links = {
  edges: Edge[];
  ambiguous: number;
  unmatched: number;
  total_correlated: number;
  truncated: boolean;
  trace: unknown;
};
export default function CaptureLinks({ runId }: { runId: string }) {
  const query = useQuery({
    queryKey: ['capture-links', runId],
    queryFn: () => api<Links>(`/runs/${runId}/observations/links`),
    staleTime: 60000,
  });
  return (
    <>
      <Alert
        type="info"
        showIcon
        message="同一采集内的工具 ID 关联"
        description="只使用 External ID 建立 CPU op / launch → GPU kernel 关系。时间接近不构成关联，跨运行不建立因果链。硬件 Cache 计数器请在深度分析查看对应采集范围。"
      />
      <QueryState loading={query.isPending} error={query.error} />
      {query.data && (
        <>
          <div className="workspace-strip">
            <span>
              可关联 <strong>{query.data.total_correlated}</strong>
            </span>
            <span>
              歧义 <strong>{query.data.ambiguous}</strong>
            </span>
            <span>
              未匹配 <strong>{query.data.unmatched}</strong>
            </span>
            <Tag>
              {query.data.truncated ? `有界预览 ${query.data.edges.length} 条` : '关联投影'}
            </Tag>
          </div>
          <Table<Edge>
            size="small"
            dataSource={query.data.edges}
            rowKey="to_event"
            pagination={{ pageSize: 12, showSizeChanger: false }}
            scroll={{ x: 760 }}
            locale={{
              emptyText: (
                <Empty description="没有工具 ID 关联。请核对算子诊断采集与 trace 覆盖。" />
              ),
            }}
            columns={[
              {
                title: 'CPU op / launch 事件',
                dataIndex: 'from_event',
                render: (v) => <code>{v}</code>,
              },
              { title: 'GPU kernel 事件', dataIndex: 'to_event', render: (v) => <code>{v}</code> },
              { title: '关联方式', dataIndex: 'method' },
              { title: '证据依据', dataIndex: 'confidence' },
            ]}
          />
          <details>
            <summary>原始 trace 覆盖与预算</summary>
            <JsonView value={query.data.trace} />
          </details>
        </>
      )}
    </>
  );
}
