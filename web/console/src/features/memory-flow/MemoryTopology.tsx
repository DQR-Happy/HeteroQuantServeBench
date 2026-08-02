import { useState } from 'react';
import { Empty, Select, Tag } from 'antd';
import { date } from '../../api/client';
import { Panel } from '../../components';
import type { Resources } from '../memory/types';
import type { Observation } from '../observability/types';
import { memorySize } from './geometry';

const snapshotNames: Record<string, string> = {
  before_request: '请求前',
  after_prefill: 'Prefill 后',
  after_generation: '生成结束、清理前',
  after_cleanup: '请求清理后',
};
export default function MemoryTopology({
  resources,
  observation,
  historical,
}: {
  resources?: Resources;
  observation?: Observation;
  historical: boolean;
}) {
  const [selected, setSelected] = useState('after_prefill');
  const snapshots = observation?.memory_snapshots ?? [];
  const snapshot = snapshots.find((value) => value.label === selected) ?? snapshots[0];
  const host = resources?.host;
  const availablePercent =
    host?.total_bytes && host.available_bytes != null
      ? Math.max(0, Math.min(100, (host.available_bytes / host.total_bytes) * 100))
      : null;
  const validAllocator =
    snapshot?.reserved_bytes != null &&
    snapshot.allocated_bytes != null &&
    snapshot.allocated_bytes >= 0 &&
    snapshot.reserved_bytes >= snapshot.allocated_bytes &&
    snapshot.reserved_bytes >= 0;
  const allocatedPercent =
    validAllocator && snapshot!.reserved_bytes! > 0
      ? (snapshot!.allocated_bytes! / snapshot!.reserved_bytes!) * 100
      : 0;
  return (
    <Panel
      title="01 · Jetson 统一内存结构"
      subtitle="CPU 与 GPU 共享物理 DRAM；下图区分硬件关系和软件统计口径。"
    >
      <div className="mf-topology" data-testid="memory-topology">
        <div className="mf-compute-pair">
          <div className="mf-compute mf-cpu">
            <span className="mf-icon-chip">CPU</span>
            <div>
              <strong>CPU / Host</strong>
              <small>模板 · 分词 · 调度 · 反分词</small>
            </div>
          </div>
          <div className="mf-compute mf-gpu">
            <span className="mf-icon-chip">GPU</span>
            <div>
              <strong>GPU / CUDA</strong>
              <small>模型算子 · 权重 · KV cache</small>
            </div>
          </div>
        </div>
        <div className="mf-memory-connectors" aria-hidden="true">
          <span>↕ 共享物理内存</span>
          <span>↕ 共享物理内存</span>
        </div>
        <div className="mf-dram">
          <div className="mf-section-title">
            <strong>共享 DRAM</strong>
            <Tag color="cyan">Jetson SoC</Tag>
            <span>没有一块单独的 HBM 可与系统内存相加</span>
          </div>
          <div className="mf-system-meter">
            <div className="mf-meter-label">
              <strong>Linux 可见总量 {memorySize(host?.total_bytes)}</strong>
              <span>现在可用 {memorySize(host?.available_bytes)}</span>
            </div>
            <div
              className="mf-host-track"
              role="img"
              aria-label={`系统总内存 ${memorySize(host?.total_bytes)}，当前可用 ${memorySize(host?.available_bytes)}`}
            >
              {availablePercent != null && (
                <>
                  <span
                    className="mf-host-unavailable"
                    style={{ width: `${100 - availablePercent}%` }}
                  />
                  <span className="mf-host-available" style={{ width: `${availablePercent}%` }} />
                </>
              )}
            </div>
            <div className="mf-legend">
              <span>
                <i className="mf-dot mf-dot-muted" />
                总量减可用量（不是模型独占）
              </span>
              <span>
                <i className="mf-dot mf-dot-green" />
                MemAvailable · 含可回收页估计
              </span>
            </div>
            <p className="mf-caption">
              来源 /proc/meminfo · {resources ? date(resources.sampled_at) : '等待采样'}
              。当前系统值包含其他进程，不反推历史请求内存。
            </p>
          </div>
          <div className="mf-allocator">
            <div className="mf-section-title">
              <strong>工作进程 CUDA allocator</strong>
              <Tag>{historical ? '历史请求快照' : '所选请求快照'}</Tag>
            </div>
            {!!snapshots.length && (
              <Select
                aria-label="内存快照时刻"
                value={snapshot?.label}
                onChange={setSelected}
                options={snapshots.map((value) => ({
                  label: snapshotNames[value.label] ?? value.label,
                  value: value.label,
                }))}
                style={{ width: '100%', maxWidth: 290 }}
              />
            )}
            {validAllocator ? (
              <>
                <div
                  className="mf-reserved"
                  role="img"
                  aria-label={`Reserved ${memorySize(snapshot!.reserved_bytes)} 包含 Allocated ${memorySize(snapshot!.allocated_bytes)}`}
                >
                  <div className="mf-reserved-label">
                    Reserved {memorySize(snapshot!.reserved_bytes)}
                  </div>
                  <div className="mf-allocated-track">
                    <div className="mf-allocated" style={{ width: `${allocatedPercent}%` }} />
                  </div>
                  <div className="mf-meter-label">
                    <strong>Allocated {memorySize(snapshot!.allocated_bytes)}</strong>
                    <span>
                      池内未分配{' '}
                      {memorySize(snapshot!.reserved_bytes! - snapshot!.allocated_bytes!)}
                    </span>
                  </div>
                </div>
                <p className="mf-caption">
                  {date(snapshot!.time)} · {snapshot!.source}。两个数来自同一时刻，Allocated 包含在
                  Reserved 内；此框单独归一化，面积不与上方系统条形对应。
                </p>
              </>
            ) : (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description="这次请求没有可配对的 allocator 快照"
              />
            )}
          </div>
          <div className="mf-unknown-bands">
            <span>激活 / workspace：未独立核算</span>
            <span>L2 / L1 cache：未采集流向</span>
            <span>寄存器：未采集分配</span>
          </div>
        </div>
      </div>
    </Panel>
  );
}
