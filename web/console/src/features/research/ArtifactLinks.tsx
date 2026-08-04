import { Button, Table } from 'antd';
import { DownloadOutlined } from '@ant-design/icons';
import { base, fmt } from '../../api/client';
import type { ResearchArtifact } from './types';

export default function ArtifactLinks({ artifacts }: { artifacts: ResearchArtifact[] }) {
  return (
    <Table<ResearchArtifact>
      rowKey="id"
      size="small"
      dataSource={artifacts}
      pagination={{ pageSize: 8, hideOnSinglePage: true }}
      scroll={{ x: 680 }}
      columns={[
        {
          title: '原始证据',
          dataIndex: 'relative_path',
          render: (path: string) => <span className="evidence-path">{path}</span>,
        },
        { title: '大小', width: 100, render: (_, row) => `${fmt(row.bytes / 1024)} KiB` },
        {
          title: 'SHA-256',
          width: 140,
          render: (_, row) => <code title={row.sha256}>{row.sha256.slice(0, 12)}…</code>,
        },
        {
          title: '原件',
          width: 100,
          render: (_, row) => (
            <Button
              size="small"
              icon={<DownloadOutlined />}
              href={`${base}/research/artifacts/${encodeURIComponent(row.id)}/download`}
            >
              下载
            </Button>
          ),
        },
      ]}
    />
  );
}
