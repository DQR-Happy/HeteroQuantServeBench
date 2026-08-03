import { useQuery } from '@tanstack/react-query';
import { Table, Tag } from 'antd';
import { api } from '../../api/client';
import { Panel, QueryState } from '../../components';

export type Capability = {
  id: string;
  name: string;
  status: 'available' | 'unavailable' | 'historical';
  reason: string;
  scope: string;
};
export type Capabilities = { schema_version: number; items: Capability[] };

export function useCapabilities() {
  return useQuery({
    queryKey: ['capabilities'],
    queryFn: () => api<Capabilities>('/capabilities'),
    staleTime: 30000,
  });
}

export default function CapabilityMatrix() {
  const query = useCapabilities();
  return (
    <Panel title="采集能力与边界" subtitle="能力由服务端报告；历史证据不等同当前部署在线采集能力">
      <QueryState loading={query.isPending} error={query.error} />
      <Table<Capability>
        rowKey="id"
        size="small"
        pagination={false}
        dataSource={query.data?.items ?? []}
        scroll={{ x: 600 }}
        columns={[
          { title: '能力', dataIndex: 'name', width: 180 },
          {
            title: '状态',
            dataIndex: 'status',
            width: 110,
            render: (status: Capability['status']) => (
              <Tag
                color={
                  status === 'available' ? 'green' : status === 'historical' ? 'blue' : undefined
                }
              >
                {status === 'available' ? '可用' : status === 'historical' ? '历史证据' : '未接入'}
              </Tag>
            ),
          },
          { title: '范围', dataIndex: 'scope', width: 150 },
          { title: '说明', dataIndex: 'reason' },
        ]}
      />
    </Panel>
  );
}
