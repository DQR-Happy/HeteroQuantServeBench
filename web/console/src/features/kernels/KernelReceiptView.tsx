import { Alert, Descriptions, Table } from 'antd';
import type { Receipt } from '../experiments/types';
import { JsonView } from '../../components';

export default function KernelReceiptView({ receipt }: { receipt: Receipt }) {
  const summary = receipt.kernel_receipt;
  if (!summary)
    return <Alert type="info" message="该运行没有算子调用凭据，无法确认具体替换实现。" />;
  const rows = Array.from(new Set(summary.sites.map((site) => `${site.op}:${site.variant}`))).map(
    (key) => {
      const sites = summary.sites.filter((site) => `${site.op}:${site.variant}` === key);
      return {
        key,
        op: sites[0].op,
        variant: sites[0].variant,
        sites: sites.length,
        prefill: sites.reduce((total, site) => total + site.calls.prefill, 0),
        decode: sites.reduce((total, site) => total + site.calls.decode, 0),
      };
    },
  );
  return (
    <section aria-label="实际算子调用凭据" className="kernel-receipt">
      <h3>实际算子调用</h3>
      <Descriptions
        size="small"
        column={{ xs: 1, md: 2 }}
        items={[
          { key: 'profile', label: '实际方案', children: <code>{summary.profile_id}</code> },
          { key: 'status', label: '调用状态', children: summary.status },
          { key: 'fallback', label: '记录的 fallback', children: summary.fallbacks.length },
          { key: 'missing', label: '未覆盖站点', children: summary.not_exercised.length },
        ]}
      />
      {rows.length > 0 ? (
        <Table
          size="small"
          rowKey="key"
          pagination={false}
          scroll={{ x: 580 }}
          dataSource={rows}
          columns={[
            { title: '算子', dataIndex: 'op' },
            { title: '实际实现与版本', dataIndex: 'variant' },
            { title: '绑定站点', dataIndex: 'sites' },
            { title: 'Prefill 调用', dataIndex: 'prefill' },
            { title: 'Decode 调用', dataIndex: 'decode' },
          ]}
        />
      ) : (
        <p className="help-text">框架原生参考路径，不记录替换站点计数。</p>
      )}
      <p className="help-text">
        以上计数来自模型逻辑站点的成功分派；GPU kernel 活动见独立 profiler
        观测；缓存命中与物理访存见对应硬件计数器。
      </p>
      {summary.not_exercised.length > 0 && (
        <Alert type="warning" message="部分绑定站点未执行，不能视为完整生效" />
      )}
      <details>
        <summary>算子计划、绑定哈希与逐站点原件</summary>
        <JsonView value={summary} />
      </details>
    </section>
  );
}
