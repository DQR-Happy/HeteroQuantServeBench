import Button from '../../components/ActionButton';
import { useQuery } from '@tanstack/react-query';
import { Alert, Collapse, Table, Tag } from 'antd';
import { Link } from 'react-router-dom';
import { api, base, fmt, terminal } from '../../api/client';
import type { Run } from '../../api/types';
import { JsonView, QueryState } from '../../components';
import type { Bundle, Receipt } from './types';
import KernelReceiptView from '../kernels/KernelReceiptView';
import QualityBaseline from './QualityBaseline';

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
        description="凭据展示执行控制流与实际模型配置；质量与性能结果见对应评测记录。"
      />
      {receipt.runtime_receipt && (
        <details>
          <summary>L3 实际执行与编译凭据</summary>
          <JsonView value={receipt.runtime_receipt} />
        </details>
      )}
      {receipt.memory_receipt != null && (
        <details>
          <summary>L6 实际缓存、前缀与传输凭据</summary>
          <JsonView value={receipt.memory_receipt} />
        </details>
      )}
      {receipt.structure_receipt?.artifact_id != null && (
        <details>
          <summary>L5 实际结构、制品来源与调用凭据</summary>
          <JsonView value={receipt.structure_receipt} />
        </details>
      )}
      {receipt.decoding_receipt && (
        <details>
          <summary>L7 实际生成、验证与回滚凭据</summary>
          <JsonView value={receipt.decoding_receipt} />
        </details>
      )}
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
              <QualityBaseline run={run} />
              <Alert
                type="info"
                message={`限定质量评测：${query.data.quality.status}`}
                description={`PPL ${fmt(query.data.quality.perplexity, 6)} · 平均 NLL ${fmt(query.data.quality.mean_nll, 6)} · 计分 token ${fmt(query.data.quality.scored_tokens, 0)}${query.data.quality.task_accuracy != null ? ` · exact-match ${fmt(query.data.quality.task_accuracy, 6)}` : ' · 不含标准答案匹配率'}。结果覆盖本次注册样本；生成性能在推理评测中单独统计。`}
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
