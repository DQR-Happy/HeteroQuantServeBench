import { useQuery } from '@tanstack/react-query';
import { Alert, Button, Collapse, Table, Tag } from 'antd';
import { Link } from 'react-router-dom';
import { api, base, fmt, terminal } from '../../api/client';
import type { Run } from '../../api/types';
import { JsonView, QueryState } from '../../components';
import type { Bundle, Receipt } from './types';
import KernelReceiptView from '../kernels/KernelReceiptView';

export function ReceiptView({ receipt }: { receipt: Receipt }) {
  const keys = Array.from(
    new Set([
      ...Object.keys(receipt.requested),
      ...Object.keys(receipt.resolved),
      ...Object.keys(receipt.effective),
    ]),
  );
  return (
    <>
      <Alert
        type={receipt.status === 'applied' ? 'info' : 'warning'}
        showIcon
        message={`执行凭据：${receipt.status}`}
        description="凭据核对执行控制流与实际模型配置；不等于指令级追踪，也不代表优化已通过质量或性能验收。"
      />
      <Table
        rowKey="key"
        size="small"
        pagination={false}
        scroll={{ x: 580 }}
        dataSource={keys.map((key) => ({ key }))}
        columns={[
          { title: '措施', dataIndex: 'key' },
          ...(['requested', 'resolved', 'effective'] as const).map((field) => ({
            title: field,
            render: (_: unknown, row: { key: string }) =>
              Object.hasOwn(receipt[field], row.key)
                ? JSON.stringify(receipt[field][row.key])
                : '未显式指定',
          })),
        ]}
      />
      <div className="help-text mono">计划 SHA-256：{receipt.plan_sha256}</div>
      {receipt.kernel_receipt && <KernelReceiptView receipt={receipt} />}
      <Collapse
        items={[
          {
            key: 'proof',
            label: '生效证据与测量角色',
            children: (
              <JsonView
                value={{
                  measurement_role: receipt.measurement_role,
                  proof_scope: receipt.proof_scope,
                  proof: receipt.proof,
                }}
              />
            ),
          },
        ]}
      />
    </>
  );
}
export default function RunEvidence({ run }: { run: Run }) {
  const present = !!run.metrics.optimization_receipt;
  const experimental = !!run.config.experiment_role;
  const hasBundle = experimental || !!run.metrics.experiment_bundle;
  const query = useQuery({
    queryKey: ['experiment-bundle', run.id],
    queryFn: () => api<Bundle>(`/runs/${run.id}/experiment`),
    enabled: hasBundle && terminal(run.state),
    retry: false,
  });
  const receipt = query.data?.receipt ?? run.metrics.optimization_receipt;
  return (
    <>
      {run.config.experiment_links?.campaign_id && (
        <Link
          to={`/optimization?experiment=${encodeURIComponent(run.config.experiment_links.campaign_id)}`}
        >
          返回所属优化实验
        </Link>
      )}
      {!present && !hasBundle && terminal(run.state) && (
        <Alert type="info" message="该历史运行没有优化生效凭据；不会用当前配置补写历史事实。" />
      )}
      {!terminal(run.state) && (
        <Alert type="info" message="等待任务终态与 worker 清理，原件尚未封存。" />
      )}
      <QueryState loading={query.isFetching} error={query.error} />
      {receipt && <ReceiptView receipt={receipt} />}
      {query.data && (
        <>
          <h3>封存原件</h3>
          <Tag>
            {query.data.manifest.status} · {query.data.manifest.publication}
          </Tag>
          <div className="help-text mono">Manifest SHA-256：{query.data.manifest_sha256}</div>
          {query.data.quality && (
            <>
              <Alert
                type="info"
                message={`限定质量评测：${query.data.quality.status}`}
                description={`PPL ${fmt(query.data.quality.perplexity, 6)} · exact-match ${fmt(query.data.quality.task_accuracy, 6)} · 计分 token ${fmt(query.data.quality.scored_tokens, 0)}。口径只覆盖注册样本，不是通用任务质量结论；质量任务不参与推理性能比较。`}
              />
              <Collapse
                items={[
                  {
                    key: 'quality',
                    label: '样本与评分协议',
                    children: <JsonView value={query.data.quality} />,
                  },
                ]}
              />
            </>
          )}
          <Table
            size="small"
            rowKey="path"
            scroll={{ x: 730 }}
            dataSource={query.data.manifest.artifacts}
            pagination={{ pageSize: 8 }}
            columns={[
              { title: '原件', dataIndex: 'path' },
              { title: '字节', dataIndex: 'bytes', render: (v: number) => fmt(v, 0) },
              {
                title: 'SHA-256',
                dataIndex: 'sha256',
                render: (v: string) => <code className="small">{v}</code>,
              },
              {
                title: '操作',
                render: (_, row) => (
                  <Button
                    size="small"
                    href={`${base}/runs/${run.id}/experiment/artifacts/${query.data!.manifest.artifacts.indexOf(row)}`}
                  >
                    下载原件
                  </Button>
                ),
              },
            ]}
          />
        </>
      )}
    </>
  );
}
