import { useState } from 'react';
import { Alert, Empty, Table, Tag } from 'antd';
import { fmt } from '../../api/client';
import { Panel } from '../../components';
import type { PhaseSpan } from '../observability/types';
import { memorySize } from './geometry';
import type { CopyEvent, Dataflow, ObservedRun } from './types';

const nodes = [
  {
    id: 'template',
    title: '文本准备',
    location: 'CPU / Host',
    names: ['chat_template', 'tokenize'],
    description:
      '应用 chat template，再由 tokenizer 生成输入 ID 与 attention mask。来源是工作进程 host_monotonic 阶段计时。',
  },
  {
    id: 'transfer',
    title: 'CUDA 输入',
    location: 'CPU → GPU 逻辑位置',
    names: ['input_transfer'],
    description:
      '输入张量执行 .to("cuda")。这是逻辑设备迁移：Jetson 仍共享物理 DRAM。当前这类采集在输入迁移之后启动 profiler，此节点的主机跨度不能冒充捕获到的 GPU memcpy。',
  },
  {
    id: 'prefill',
    title: 'Prefill',
    location: 'GPU 算子执行',
    names: ['prefill'],
    description:
      '处理完整提示词并创建 KV。展示的是模型调用与 token 选择所在的主机阶段包络；真正的 GPU kernel 数来自下方 trace，不能用主机耗时替代 GPU 执行时间。',
  },
  {
    id: 'decode',
    title: 'Decode / KV',
    location: 'GPU 算子执行',
    names: ['decode'],
    description:
      '逐 token 复用并扩展 KV。阶段计时覆盖已记录 decode 步，GPU trace 通常只覆盖限定窗口。KV 图来自这次请求释放缓存前的快照。',
  },
  {
    id: 'return',
    title: 'Token 返回',
    location: 'Host → 浏览器',
    names: ['stream_consumer_wait'],
    description:
      '已选择的 token 在 CPU 反分词，经 IPC 与 SSE 流式返回。这里的等待阶段不能当作完整网络耗时；token 选择/反分词的独立计时可在运行详情查看。',
  },
  {
    id: 'cleanup',
    title: '请求清理',
    location: '工作进程',
    names: ['request_cleanup'],
    description:
      '清理请求持有的输入与 KV 引用，保留驻留模型。allocator 保留的池内存可能仍在 Reserved 中，释放 Python 引用不等于立即归还整机所有页面。',
  },
];
const directionNames: Record<string, string> = {
  host_to_device: 'Host → Device',
  device_to_host: 'Device → Host',
  device_to_device: 'Device → Device',
  peer_to_peer: 'Device → Peer',
  host_to_host: 'Host → Host',
  unknown: '方向未识别',
};
function elapsed(phases: PhaseSpan[], names: string[]) {
  const matching = phases.filter((phase) => names.includes(phase.name));
  return matching.length ? matching.reduce((sum, phase) => sum + phase.duration_ms, 0) : null;
}
export default function RequestFlow({
  run,
  dataflow,
  error,
  loading,
}: {
  run?: ObservedRun;
  dataflow?: Dataflow;
  error?: Error | null;
  loading: boolean;
}) {
  const [selected, setSelected] = useState('prefill');
  const [direction, setDirection] = useState('all');
  const node = nodes.find((entry) => entry.id === selected)!;
  const phases = run?.metrics.observation?.phases ?? [];
  const profile = run?.metrics.observation?.profile;
  const spans = phases.filter((phase) => node.names.includes(phase.name));
  const events = (dataflow?.copy_events ?? []).filter(
    (event) => direction === 'all' || event.direction === direction,
  );
  const activities = dataflow?.coverage.activities;
  const gpuObserved =
    Array.isArray(activities) &&
    activities.includes('CUDA') &&
    !['not_collected', 'invalid', 'download_only'].includes(
      String(dataflow?.coverage.trace_status),
    );
  return (
    <Panel
      title="04 · 请求数据流"
      subtitle="上方是可点击的逻辑执行流程；下方是限定窗口中实际捕获的 CUDA copy 事件。"
    >
      <div
        className="mf-logical-flow"
        data-testid="request-dataflow"
        role="group"
        aria-label="推理逻辑数据流"
      >
        {nodes.map((entry, index) => (
          <button
            type="button"
            key={entry.id}
            className={`mf-flow-node ${entry.id === selected ? 'is-selected' : ''}`}
            onClick={() => setSelected(entry.id)}
            aria-pressed={entry.id === selected}
          >
            <span className="mf-step-number">0{index + 1}</span>
            <strong>{entry.title}</strong>
            <small>{entry.location}</small>
            <span className="mf-flow-time">
              {elapsed(phases, entry.names) == null
                ? '未采集 host span'
                : `${fmt(elapsed(phases, entry.names), 2)} ms · host`}
            </span>
          </button>
        ))}
      </div>
      <p className="mf-caption">
        箭头表达程序处理顺序，不是逐字节物理追踪；不推断 HBM、cache、寄存器位置，也不把 HtoD 等同
        PCIe 流量。节点点击只展开主机阶段证据，不筛选或归属下方 GPU 事件：两者时钟原点不同。
      </p>
      <div className="mf-flow-detail">
        <div className="mf-section-title">
          <h3>{node.title}</h3>
          <Tag color="cyan">{spans.length} 个 host span</Tag>
        </div>
        <p>{node.description}</p>
        <div className="mf-span-list">
          {spans.slice(0, 12).map((span, index) => (
            <span key={index}>
              <code>{span.name}</code> {fmt(span.start_ms, 2)} →{' '}
              {fmt(span.start_ms + span.duration_ms, 2)} ms
            </span>
          ))}
          {spans.length > 12 && <span>还有 {spans.length - 12} 个，见运行详情</span>}
        </div>
      </div>
      <div className="mf-trace-heading">
        <div>
          <h3>实际 CUDA copy 方向</h3>
          <p className="mf-caption">
            采集窗口：
            {profile?.coverage
              ? `Prefill ${profile.coverage.prefill ? '包含' : '不包含'} · Decode ${profile.coverage.decode_steps} 步 / 输出 ${profile.coverage.total_output_tokens} token`
              : '未记录详细范围'}
            。事件持续时间求和可能重叠，不能当请求墙钟时间。
          </p>
        </div>
        <Tag color={dataflow?.status === 'available' ? 'green' : 'orange'}>
          {loading ? '读取真实 trace…' : (dataflow?.status ?? '尚无结果')}
        </Tag>
      </div>
      <Alert
        className="section-gap"
        type="info"
        showIcon
        message="这里的字节数仅属于显式 memcpy，不是整次推理的内存流量"
        description="此采集窗口不含模型加载和首次输入迁移，也不统计 GPU kernel 对权重、KV、激活的普通读写。即使这里只显示少量字节，也不能据此判断带宽开销很小。"
      />
      {error && (
        <Alert type="error" showIcon message="读取数据流失败" description={error.message} />
      )}
      {dataflow && (
        <>
          <div className="mf-summary-row" data-testid="transfer-summary">
            <div>
              <strong>{gpuObserved ? fmt(dataflow.totals.kernel_events, 0) : '未采集'}</strong>
              <span>GPU kernel 事件</span>
            </div>
            <div>
              <strong>{gpuObserved ? fmt(dataflow.totals.copy_events, 0) : '未采集'}</strong>
              <span>GPU copy 事件</span>
            </div>
            <div>
              <strong>
                {dataflow.totals.bytes == null
                  ? '总字节未完整采集'
                  : memorySize(dataflow.totals.bytes)}
              </strong>
              <span>{dataflow.totals.bytes_known_events} 条 copy 有明确字节字段</span>
            </div>
          </div>
          {dataflow.edges.length ? (
            <div className="mf-copy-paths" role="group" aria-label="实际CUDA拷贝方向与事件数">
              {dataflow.edges.map((edge) => (
                <button
                  type="button"
                  key={edge.direction}
                  className={`mf-copy-path${direction === edge.direction ? ' is-selected' : ''}`}
                  aria-pressed={direction === edge.direction}
                  onClick={() =>
                    setDirection(direction === edge.direction ? 'all' : edge.direction)
                  }
                >
                  <span className="mf-copy-endpoints">
                    {directionNames[edge.direction] ?? edge.label}
                  </span>
                  <span className="mf-copy-line" aria-hidden="true">
                    ━━━━━━━━━━ →
                  </span>
                  <strong>
                    {edge.count} 次 · {fmt(edge.duration_ms_sum, 3)} ms
                  </strong>
                  <span>{edge.bytes == null ? '总字节未知' : memorySize(edge.bytes)}</span>
                  {edge.bytes == null && edge.known_bytes != null && (
                    <small>仅已知部分 {memorySize(edge.known_bytes)}</small>
                  )}
                </button>
              ))}
            </div>
          ) : (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description="这个采集窗口没有可展示的 copy 方向证据。未捕获不代表没有发生传输。"
            />
          )}
          {dataflow.copy_events.length > 0 && (
            <details className="mf-event-details">
              <summary>
                查看真实拷贝事件 · {directionNames[direction] ?? '所有方向'} · {events.length}{' '}
                条已返回记录
              </summary>
              <Table<CopyEvent>
                size="small"
                rowKey={(_, index) => String(index)}
                dataSource={events}
                scroll={{ x: 750 }}
                pagination={{ pageSize: 8, showSizeChanger: false }}
                columns={[
                  {
                    title: '事件 / 方向',
                    render: (_, event) => (
                      <>
                        <code>{event.name}</code>
                        <div className="small muted">
                          {directionNames[event.direction] ?? event.direction}
                        </div>
                      </>
                    ),
                  },
                  {
                    title: '开始 / ms',
                    dataIndex: 'start_ms',
                    render: (value: number) => fmt(value, 4),
                  },
                  {
                    title: '持续 / ms',
                    dataIndex: 'duration_ms',
                    render: (value: number) => fmt(value, 4),
                  },
                  { title: '字节', dataIndex: 'bytes', render: memorySize },
                  {
                    title: 'Stream / Correlation',
                    render: (_, event) => (
                      <>
                        {event.stream ?? '未采集'} / {event.correlation ?? '未采集'}
                      </>
                    ),
                  },
                ]}
              />
              {dataflow.copy_events_truncated > 0 && (
                <p className="mf-caption">
                  事件表只返回有界样本；汇总遵循接口 coverage，原始 Chrome trace 可完整下载。
                </p>
              )}
            </details>
          )}
          <details className="mf-event-details">
            <summary>数据源与不可观测范围</summary>
            <p className="mf-caption">
              {dataflow.source} · {dataflow.clock_domain}
            </p>
            {dataflow.limitations.map((limitation, index) => (
              <p className="mf-caption" key={index}>
                {limitation}
              </p>
            ))}
          </details>
        </>
      )}
    </Panel>
  );
}
