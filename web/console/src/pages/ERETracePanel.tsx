import { Space, Table, Tag } from 'antd';
import { base, fmt } from '../api/client';
import { JsonView } from '../components';

export type ERETraceData = {
  tool?: string;
  counts?: Record<string, number>;
  trace_bytes?: number;
  logical_bytes?: number;
  logical_to_stored_ratio?: number | null;
  encoding_counts?: Record<string, number>;
  coverage?: Record<string, unknown>[];
  collector?: Record<string, unknown>[];
  ptx_sources?: Record<string, unknown>[];
  sm_distribution?: { sm: number; records: number; summed_elapsed_cycles: number }[];
  elapsed_histogram?: { log2_bucket: number; records: number }[];
  decoded_artifacts?: { path: string; frames: number; decoded_sha256: string }[];
  launches?: {
    trace: string;
    index: number;
    kernel: string;
    grid: number[];
    block: number[];
    records: number;
    observed_records: number;
    encoding: string;
    elapsed_cycles_mean: number | null;
    elapsed_cycles_p50: number | null;
    elapsed_cycles_p95: number | null;
    elapsed_cycles_max: number | null;
  }[];
};

function byteSize(value?: number) {
  if (value == null) return '—';
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${fmt(value / 1024, 2)} KiB`;
  return `${fmt(value / 1024 ** 2, 3)} MiB`;
}

export default function ERETracePanel({ data, run }: { data: ERETraceData; run: string }) {
  const sms = data.sm_distribution ?? [];
  const maximum = Math.max(1, ...sms.map((x) => x.records));
  return (
    <section aria-label="ERETrace 观测结果">
      <p>
        已采集 {data.counts?.launches ?? 0} 次启动、{data.counts?.observed_records ?? 0} 条 warp
        记录。时间单位为 SM 本地时钟周期，记录区间为 lane 0 进入到自身退出。
      </p>
      <Space wrap>
        <Tag>记录文件 {byteSize(data.trace_bytes)}</Tag>
        <Tag>还原大小 {byteSize(data.logical_bytes)}</Tag>
        <Tag>逻辑 / 保存体积 {fmt(data.logical_to_stored_ratio, 2)} 倍</Tag>
        <Tag>空记录槽位 {data.counts?.zero_slots ?? 0}</Tag>
        {Object.entries(data.encoding_counts ?? {}).map(([name, count]) => (
          <Tag key={name}>
            {name} · {count} 帧
          </Tag>
        ))}
      </Space>
      <h3>SM 上的记录分布</h3>
      <p>柱高表示被观测 warp 记录数，可悬停查看 SM 和累计周期。</p>
      <div style={{ overflowX: 'auto' }}>
        <svg
          role="img"
          aria-label="SM warp 记录分布"
          viewBox="0 0 1000 180"
          style={{ width: '100%', minWidth: 650 }}
        >
          <line x1={35} x2={975} y1={140} y2={140} stroke="currentColor" opacity={0.2} />
          <text x={0} y={18} fontSize={10} fill="currentColor">
            {maximum}
          </text>
          <text x={0} y={140} fontSize={10} fill="currentColor">
            0
          </text>
          {sms.map((item, index) => {
            const width = 940 / Math.max(1, sms.length);
            const barWidth = Math.min(40, width * 0.75);
            const x = 35 + index * width + (width - barWidth) / 2;
            const height = (item.records / maximum) * 125;
            return (
              <g key={item.sm}>
                <rect
                  x={x}
                  y={140 - height}
                  width={Math.max(1, barWidth)}
                  height={height}
                  fill="#467fba"
                >
                  <title>
                    SM {item.sm} · {item.records} 条 · 累计 {item.summed_elapsed_cycles} cycles
                  </title>
                </rect>
                {width > 25 && (
                  <text
                    x={x + barWidth / 2}
                    y={135 - height}
                    textAnchor="middle"
                    fontSize={10}
                    fill="currentColor"
                  >
                    {item.records}
                  </text>
                )}
                <text
                  x={x + barWidth / 2}
                  y={156}
                  textAnchor="middle"
                  fontSize={9}
                  fill="currentColor"
                >
                  {item.sm}
                </text>
              </g>
            );
          })}
          <text x={35} y={176} fontSize={11} fill="currentColor">
            SM ID（每个柱形可悬停查看记录数）
          </text>
        </svg>
      </div>
      <Table
        size="small"
        rowKey={(r) => `${r.trace}-${r.index}`}
        dataSource={data.launches ?? []}
        pagination={{ pageSize: 8 }}
        scroll={{ x: 1000 }}
        columns={[
          { title: 'Kernel', dataIndex: 'kernel', ellipsis: true },
          { title: 'Grid / Block', render: (_, r) => `${r.grid.join('×')} / ${r.block.join('×')}` },
          { title: 'Warp 记录', dataIndex: 'observed_records' },
          { title: 'P50 cycles', render: (_, r) => fmt(r.elapsed_cycles_p50, 0) },
          { title: 'P95 cycles', render: (_, r) => fmt(r.elapsed_cycles_p95, 0) },
          { title: '最大 cycles', render: (_, r) => fmt(r.elapsed_cycles_max, 0) },
          { title: '保存格式', dataIndex: 'encoding' },
        ]}
      />
      <details>
        <summary>Warp 时间分布、采集覆盖与缓冲池记录</summary>
        <JsonView
          value={{
            elapsed_histogram: data.elapsed_histogram,
            coverage: data.coverage,
            collector: data.collector,
            ptx_sources: data.ptx_sources,
          }}
        />
      </details>
      <p>
        缓存命中率可结合 ncu
        硬件计数器分析；操作系统页由进程内存观测提供。地址追踪属于独立的访存事件扩展。
      </p>
      <Space wrap>
        {data.decoded_artifacts?.map((artifact) => (
          <a
            key={artifact.path}
            href={`${base}/diagnostics/jobs/${run}/artifacts/${artifact.path}`}
            download
          >
            下载 warp 记录 · {artifact.frames} 帧
          </a>
        ))}
      </Space>
    </section>
  );
}
