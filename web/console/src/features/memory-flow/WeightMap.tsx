import { useMemo, useState } from 'react';
import { Alert, Empty, Table, Tag } from 'antd';
import { date } from '../../api/client';
import { Panel } from '../../components';
import { groupWeights, memorySize, weightTiles } from './geometry';
import type { Inventory, TensorEntry } from './types';

export default function WeightMap({
  inventory,
  source,
  sampledAt,
}: {
  inventory?: Inventory | null;
  source: string;
  sampledAt?: number | null;
}) {
  const [selected, setSelected] = useState('');
  const { groups, unknownEntries } = useMemo(() => groupWeights(inventory), [inventory]);
  const tiles = useMemo(() => weightTiles(groups), [groups]);
  const group = groups.find((entry) => entry.id === selected);
  const total = groups.reduce((sum, entry) => sum + entry.bytes, 0);
  return (
    <Panel
      title="02 · 权重驻留结构"
      subtitle="矩形面积 = 已观测、全局去重的 storage 字节。点击层或 Embedding 查看真实张量。"
    >
      <div className="mf-summary-row">
        <div>
          <strong>{memorySize(inventory ? total : undefined)}</strong>
          <span>已绘制的去重 storage</span>
        </div>
        <div>
          <strong>{inventory ? groups.length : '未采集'}</strong>
          <span>逻辑分组</span>
        </div>
        <div>
          <strong>{inventory?.entry_count ?? '—'}</strong>
          <span>张量名称（可能共享）</span>
        </div>
      </div>
      <div className="mf-evidence-line">
        <Tag color="blue">{source}</Tag>
        <span>{sampledAt ? date(sampledAt) : '时间未提供'}</span>
        <span>按模型命名映射层；不是物理地址布局</span>
      </div>
      {tiles.length ? (
        <div
          className="mf-treemap"
          data-testid="weight-treemap"
          role="group"
          aria-label="权重内存结构图，按层展示去重storage容量"
        >
          {tiles.map((tile) => (
            <button
              type="button"
              key={tile.id}
              className={`mf-tile mf-tile-${tile.kind}${group?.id === tile.id ? ' mf-tile-selected' : ''}`}
              style={{
                left: `${(tile.x / 840) * 100}%`,
                top: `${(tile.y / 340) * 100}%`,
                width: `${(tile.width / 840) * 100}%`,
                height: `${(tile.height / 340) * 100}%`,
              }}
              title={`${tile.label} · ${memorySize(tile.bytes)} · ${tile.storages} 个独立 storage`}
              aria-label={`${tile.label} ${memorySize(tile.bytes)}，${tile.entries.length} 个张量名称`}
              aria-pressed={group?.id === tile.id}
              onClick={() => setSelected(tile.id)}
            >
              {tile.width > 62 && tile.height > 30 && (
                <>
                  <span className="mf-tile-label">{tile.label}</span>
                  <span className="mf-tile-compact" aria-hidden="true">
                    {tile.kind === 'layer'
                      ? tile.label.replace('Decoder ', 'L')
                      : tile.kind === 'embedding'
                        ? 'Embedding'
                        : tile.label}
                  </span>
                </>
              )}
              {tile.width > 65 && tile.height > 57 && (
                <span className="mf-tile-bytes">{memorySize(tile.bytes)}</span>
              )}
            </button>
          ))}
        </div>
      ) : (
        <Empty description="未找到该模型版本的权重账本。加载时未记录的旧数据不会被估算填充。" />
      )}
      {(unknownEntries > 0 || inventory?.truncated) && (
        <Alert
          className="section-gap"
          type="warning"
          message={`${unknownEntries} 条记录缺少 storage 大小，未绘制面积。${inventory?.truncated ? '原账本达到条目上限，仅展示已采集范围。' : ''}`}
        />
      )}
      {group && (
        <div className="mf-selection-detail">
          <div className="mf-section-title">
            <h3>{group.label}</h3>
            <Tag>{memorySize(group.bytes)}</Tag>
            <span>
              {group.storages} 个独立 storage · {group.entries.length} 个名称
            </span>
          </div>
          <p className="mf-caption">
            共享 storage 只计一次；跨组别名归入“跨组共享 Storage”。表中 logical bytes
            是张量元素量，与 storage 不可相加。
          </p>
          <Table<TensorEntry>
            rowKey={(entry, index) => `${entry.name}:${index}`}
            size="small"
            dataSource={group.entries}
            scroll={{ x: 650 }}
            pagination={{ pageSize: 5, showSizeChanger: false }}
            columns={[
              {
                title: '张量',
                dataIndex: 'name',
                render: (name: string) => <code className="kernel-name">{name}</code>,
              },
              {
                title: 'Shape / Dtype',
                render: (_, entry) => (
                  <>
                    <code>{entry.shape.join(' × ') || 'scalar'}</code>
                    <div className="small muted">
                      {entry.dtype} · {entry.device}
                    </div>
                  </>
                ),
              },
              {
                title: 'Logical / Storage',
                render: (_, entry) => (
                  <>
                    {memorySize(entry.logical_bytes)}
                    <div className="small muted">{memorySize(entry.storage_bytes)}</div>
                  </>
                ),
              },
              {
                title: 'Storage / Offset',
                render: (_, entry) => (
                  <>
                    <code title={entry.storage_key ?? ''}>
                      {entry.storage_key?.slice(0, 10) ?? '未知'}
                    </code>
                    <div className="small muted">
                      offset{' '}
                      {entry.storage_offset_bytes == null
                        ? '未知'
                        : `${entry.storage_offset_bytes} B`}
                    </div>
                  </>
                ),
              },
            ]}
          />
        </div>
      )}
    </Panel>
  );
}
