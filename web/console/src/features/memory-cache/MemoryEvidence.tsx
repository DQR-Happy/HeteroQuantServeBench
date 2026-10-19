import { Descriptions, Table } from 'antd';
import { Link } from 'react-router-dom';
import { fmt } from '../../api/client';
import { JsonView, Panel } from '../../components';
import type { MemoryRun } from './types';
const mib = (n: number | undefined) => (n == null ? '—' : `${fmt(n / 1024 ** 2, 3)} MiB`);

export default function MemoryEvidence({
  run,
  compact = false,
}: {
  run?: MemoryRun;
  compact?: boolean;
}) {
  const receipt = run?.metrics.memory_receipt;
  if (!receipt)
    return <p>选择已完成的 L6 记录后显示实际 KV 存储、复用和传输；未采集的指标不会补成零。</p>;
  return (
    <>
      <Panel title="实际执行与存储">
        <p>
          <Link to={`/runs/${run.id}`}>{run.id} · 完整运行记录</Link> · {receipt.status}
        </p>
        {compact && (
          <p>
            <Link
              to={`/memory-cache/records?deployment=${encodeURIComponent(run.config.deployment.id)}&run=${run.id}`}
            >
              查看逐层存储、冻结计划与运行对照 →
            </Link>
          </p>
        )}
        <Descriptions
          size="small"
          column={{ xs: 1, md: 3 }}
          items={[
            { key: 'class', label: '实际 Cache 类', children: receipt.cache_class ?? '未使用 KV' },
            {
              key: 'tokens',
              label: '实际处理输入位置',
              children: fmt(receipt.processed_input_positions, 0),
            },
            {
              key: 'capacity',
              label: '静态容量 / Token',
              children: fmt(receipt.static_capacity_tokens, 0),
            },
            {
              key: 'gpu',
              label: '释放前 GPU KV storage',
              children: mib(receipt.storage_before_release.bytes_by_device.cuda),
            },
            {
              key: 'cpu',
              label: '释放前 CPU KV storage',
              children: mib(receipt.storage_before_release.bytes_by_device.cpu),
            },
            {
              key: 'allocated',
              label: '快照发布后 allocator allocated',
              children: mib(receipt.allocator_after_prefix_publish?.allocated_bytes),
            },
            {
              key: 'reserved',
              label: '快照发布后 allocator reserved',
              children: mib(receipt.allocator_after_prefix_publish?.reserved_bytes),
            },
            {
              key: 'peak',
              label: '本请求 allocator 峰值（含前缀）',
              children: mib(receipt.allocator_after_prefix_publish?.peak_allocated_bytes),
            },
            {
              key: 'lookup',
              label: '前缀查找 / 复用 Token',
              children: `${receipt.prefix.lookup} / ${receipt.prefix.reused_tokens}`,
            },
            {
              key: 'prefix',
              label: '保留前缀 / 条目',
              children: `${mib(receipt.prefix.retained_bytes)} / ${receipt.prefix.retained_entries}`,
            },
            {
              key: 'restore',
              label: '前缀还原 / 快照复制',
              children: `${mib(receipt.prefix.restore_copy_bytes)} / ${mib(receipt.prefix.snapshot_copy_bytes)}`,
            },
            { key: 'evict', label: '本次前缀淘汰', children: fmt(receipt.prefix.evictions, 0) },
            {
              key: 'd2h',
              label: '卸载 D2H / 次数',
              children: `${mib(receipt.transfers.d2h_bytes)} / ${receipt.transfers.offload_calls}`,
            },
            {
              key: 'h2d',
              label: '预取 H2D / 次数',
              children: `${mib(receipt.transfers.h2d_bytes)} / ${receipt.transfers.prefetch_calls}`,
            },
            {
              key: 'fault',
              label: '进程 minor / major faults',
              children: `${receipt.process_faults.minor} / ${receipt.process_faults.major}`,
            },
          ]}
        />
        <p className="muted">
          传输为 Cache API 实际搬运的张量字节，并非 PCIe 硬件计数。进程缺页不能归因成 GPU
          分页或单个张量缺页。保留前缀单独计费。
        </p>
        {!compact && (
          <Table
            size="small"
            pagination={{ pageSize: 12 }}
            scroll={{ x: 700 }}
            rowKey={(r) => `${r.layer}-${r.role}`}
            dataSource={receipt.storage_before_release.tensors}
            columns={[
              { title: '层', dataIndex: 'layer' },
              { title: '角色', dataIndex: 'role' },
              { title: '形状', dataIndex: 'shape', render: (v: number[]) => v.join(' × ') },
              { title: '类型', dataIndex: 'dtype' },
              { title: '位置', dataIndex: 'device' },
              { title: '去重 storage', dataIndex: 'unique_storage_bytes', render: mib },
            ]}
          />
        )}
      </Panel>
      {!compact && (
        <details>
          <summary>冻结计划与原始凭据</summary>
          <JsonView value={receipt} />
        </details>
      )}
    </>
  );
}
