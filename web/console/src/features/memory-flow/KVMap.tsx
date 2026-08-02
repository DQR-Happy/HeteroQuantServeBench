import { useMemo, useState } from 'react';
import { Alert, Empty, Tag } from 'antd';
import { date } from '../../api/client';
import { Panel } from '../../components';
import { layerKV, memorySize } from './geometry';
import type { Inventory } from './types';

export default function KVMap({ inventory }: { inventory?: Inventory | null }) {
  const layers = useMemo(() => layerKV(inventory), [inventory]);
  const [selected, setSelected] = useState('');
  const layer = layers.find((entry) => entry.layer === selected) ?? layers[0];
  const max = Math.max(1, ...layers.map((entry) => entry.key + entry.value + entry.other));
  return (
    <Panel
      title="03 · KV cache 层结构"
      subtitle="所选请求生成结束、释放前的快照。关闭请求后不表示这些 KV 仍然驻留。"
    >
      <div className="mf-evidence-line">
        <Tag color="purple">请求历史快照</Tag>
        {inventory && (
          <Tag
            color={
              inventory.availability === 'complete' && !inventory.truncated ? 'green' : 'orange'
            }
          >
            {inventory.availability === 'complete' && !inventory.truncated
              ? '账本范围完整'
              : '仅部分可见'}
          </Tag>
        )}
        <span>{inventory?.sampled_at ? date(inventory.sampled_at) : '未采集'}</span>
        <strong>去重 storage {memorySize(inventory?.unique_storage_bytes)}</strong>
      </div>
      {inventory && (inventory.availability !== 'complete' || inventory.truncated) && (
        <Alert
          className="section-gap"
          type="warning"
          showIcon
          message="仅绘制已观测的 KV 分量，缺失部分不是零容量"
          description={
            inventory.truncated
              ? '原始账本达到条目上限；层数与容量只覆盖已返回的条目。'
              : '部分张量元数据不可见；总量和图形只代表已采集范围。'
          }
        />
      )}
      <div className="mf-legend">
        <span>
          <i className="mf-dot mf-dot-purple" />K 逻辑字节
        </span>
        <span>
          <i className="mf-dot mf-dot-blue" />V 逻辑字节
        </span>
        {layers.some((entry) => entry.other > 0) && <span>灰色：其他已观测 KV 分量</span>}
        <span>条形比例按最大层归一化 · 点击查看 Shape</span>
      </div>
      {layers.length ? (
        <div className="mf-kv-layout" data-testid="kv-layer-chart">
          <div className="mf-kv-bars" role="group" aria-label="KV缓存每层K和V逻辑容量">
            {layers.map((entry) => {
              const hasKey = entry.entries.some((tensor) => tensor.kind === 'key');
              const hasValue = entry.entries.some((tensor) => tensor.kind === 'value');
              const completePair = hasKey && hasValue;
              return (
                <button
                  type="button"
                  className={`mf-kv-row${layer?.layer === entry.layer ? ' is-selected' : ''}`}
                  key={entry.layer}
                  aria-pressed={layer?.layer === entry.layer}
                  aria-label={`Layer ${entry.layer}，K ${hasKey ? memorySize(entry.key) : '未采集'}，V ${hasValue ? memorySize(entry.value) : '未采集'}`}
                  title={
                    !completePair
                      ? `${!hasKey ? 'K 未采集 ' : ''}${!hasValue ? 'V 未采集' : ''}，条形仅含已知分量`
                      : undefined
                  }
                  onClick={() => setSelected(entry.layer)}
                >
                  <span>L{entry.layer}</span>
                  <span className="mf-kv-track">
                    <span className="mf-kv-key" style={{ width: `${(entry.key / max) * 100}%` }} />
                    <span
                      className="mf-kv-value"
                      style={{ width: `${(entry.value / max) * 100}%` }}
                    />
                    <span
                      className="mf-kv-other"
                      style={{ width: `${(entry.other / max) * 100}%` }}
                    />
                  </span>
                  <span>
                    {!completePair && '已知 '}
                    {memorySize(entry.key + entry.value + entry.other)}
                  </span>
                </button>
              );
            })}
          </div>
          <div className="mf-kv-detail">
            <h3>Layer {layer?.layer}</h3>
            {layer && !layer.entries.some((entry) => entry.kind === 'key') && (
              <Tag color="orange">K 未采集</Tag>
            )}
            {layer && !layer.entries.some((entry) => entry.kind === 'value') && (
              <Tag color="orange">V 未采集</Tag>
            )}
            {layer?.entries.map((entry, index) => (
              <div key={`${entry.name}:${index}`}>
                <Tag color={entry.kind === 'key' ? 'purple' : 'blue'}>{entry.kind}</Tag>
                <strong>{memorySize(entry.logical_bytes)}</strong>
                <p className="mono">{entry.shape.join(' × ')}</p>
                <p>
                  {entry.dtype} · {entry.device}
                </p>
                <small>
                  Storage {memorySize(entry.storage_bytes)} ·{' '}
                  {entry.storage_key?.slice(0, 10) ?? '未采集'}
                </small>
              </div>
            ))}
            <p className="mf-caption">
              图按张量逻辑容量绘制，可能保留更大的底层 storage；别名关系以 storage 标识判断。
            </p>
          </div>
        </div>
      ) : (
        <Empty description="该请求没有 KV 元数据。可选择有观测记录的其他推理请求。" />
      )}
    </Panel>
  );
}
