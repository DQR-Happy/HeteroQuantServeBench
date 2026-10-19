import Button from '../../components/ActionButton';
import { useState } from 'react';
import { Alert, Select, Table, Tabs, Tag } from 'antd';
import { fmt, gib } from '../../api/client';
import { Panel } from '../../components';
import DependencyGraph from './DependencyGraph';
import type { Architecture, ArchitectureNode } from './types';
export default function StaticArchitecture({ data }: { data: Architecture }) {
  const [layer, setLayer] = useState(0);
  const [view, setView] = useState('block');
  const [selected, setSelected] = useState<ArchitectureNode>();
  const current = data.layers[layer];
  const prefix = current?.path ?? '';
  const modulePath = selected?.module ? `${prefix}.${selected.module}` : null;
  const weights = data.tensors.filter((row) =>
    modulePath
      ? row.name.startsWith(modulePath + '.')
      : !selected && row.name.startsWith(prefix + '.'),
  );
  const sizes = data.config;
  return (
    <>
      {' '}
      {data.graphs && (
        <div className="architecture-sequence" aria-label="模型总体串行数据流">
          <div className="architecture-stage">
            <strong>Token Embedding</strong>
            <small>
              token ID → hidden states
              {sizes.model_type === 'gemma3_text' ? ' · Gemma 输入缩放' : ''}
            </small>
          </div>
          <span>→</span>
          <div className="architecture-stage active">
            <strong>Decoder Block × {sizes.num_hidden_layers ?? '?'}</strong>
            <small>层间有依赖，下面展开单层</small>
          </div>
          <span>→</span>
          <div className="architecture-stage">
            <strong>Final RMSNorm</strong>
            <small>model.norm</small>
          </div>
          <span>→</span>
          <div className="architecture-stage">
            <strong>LM Head → logits</strong>
            <small>再由生成算法选取下一 token</small>
          </div>
        </div>
      )}
      {!data.graphs ? (
        <Alert
          showIcon
          type="info"
          message="通用结构已读取；暂无经过核对的模型族语义示意图"
          description={[
            '“模块与权重”和“运行算子观测”子页适用于各模型族。',
            data.implementation.reason,
            ...data.issues,
          ]
            .filter(Boolean)
            .join('；')}
        />
      ) : (
        <>
          <div className="architecture-picker">
            <div>
              <label className="field-label" htmlFor="architecture-layer">
                选择具体 Decoder 层
              </label>
              <Select
                id="architecture-layer"
                className="wide-select"
                value={layer}
                onChange={(value) => {
                  setLayer(value);
                  setSelected(undefined);
                }}
                options={data.layers.map((item) => ({
                  value: item.index,
                  label: `Layer ${item.index} · ${item.attention_type === 'sliding_attention' ? '局部 Attention' : item.attention_type === 'full_attention' ? '全局 Attention' : '类型未知'}`,
                }))}
              />
            </div>
            <Tag>{current?.path}</Tag>
            <Tag>
              {current?.tensor_count} 个权重 tensor · {fmt(gib(current?.stored_bytes), 3)} GiB 存储
            </Tag>
          </div>
          <Tabs
            activeKey={view}
            onChange={(value) => {
              setView(value);
              setSelected(undefined);
            }}
            items={[
              { key: 'block', label: 'Decoder Block' },
              { key: 'attention', label: 'Attention 内部' },
              { key: 'mlp', label: 'MLP 内部' },
            ]}
          />
          <div className="architecture-legend">
            <span>实线 → 数据依赖</span>
            <span>虚线 → 残差恒等路径</span>
            <span>点线 → KV 状态依赖</span>
            <span>横向滚动查看完整路径；节点可点击或用 Tab / Enter 选择</span>
          </div>
          <DependencyGraph
            graph={data.graphs[view]}
            identity={view}
            selected={selected?.id}
            onSelect={setSelected}
          />
          <div className="architecture-detail" aria-live="polite">
            {selected ? (
              <>
                <strong>{selected.label}</strong>
                <p>
                  <code>{modulePath ?? '函数级操作：模块树中可能没有独立 nn.Module'}</code>
                </p>
                <p>{selected.note || '箭头表示逻辑先后依赖，实际 kernel 调用见设备时间线。'}</p>
                {(selected.module === 'self_attn' || selected.module === 'mlp') && (
                  <Button
                    size="small"
                    onClick={() => {
                      setView(selected.module === 'self_attn' ? 'attention' : 'mlp');
                      setSelected(undefined);
                    }}
                  >
                    展开内部计算单元
                  </Button>
                )}
              </>
            ) : (
              <span className="muted">
                选择一个节点，查看模块路径与对应权重；不选节点时列出当前层的全部权重。
              </span>
            )}
          </div>
          <Alert
            showIcon
            type="info"
            message={
              view === 'attention'
                ? 'Q、K、V 有共同输入且彼此无数据依赖；图示为逻辑可并行，当前 Python forward 仍可能依次调用。'
                : view === 'mlp'
                  ? 'Gate 与 Up 分支逻辑独立，逐元素乘之前必须汇合；Gate 分支内部激活依赖 Linear 输出。'
                  : 'Attention → 残差相加 → MLP → 残差相加有先后依赖；标准逐 token 解码还存在跨 token 依赖。'
            }
            description={
              view === 'attention'
                ? `本层 ${current?.attention_type}；${current?.attention_type === 'sliding_attention' ? `局部窗口配置 ${sizes.sliding_window ?? '未知'} tokens。` : '因果 mask 控制可见位置。'}数学子步骤由所选后端独立或融合执行。`
                : '图中展示逻辑并行关系；执行重叠情况请到机制观测或深度分析的时间线查看。'
            }
          />
        </>
      )}
      <Panel
        title={selected ? '所选模块权重' : '当前层权重'}
        subtitle="名称、shape、存储 dtype 与字节来自实际 Safetensors header；无需加载 GPU"
      >
        <Table
          rowKey="name"
          size="small"
          scroll={{ x: 900 }}
          dataSource={data.graphs ? weights : data.tensors}
          locale={{
            emptyText: selected
              ? '所选节点为函数操作、残差或 KV 状态，其参数信息可沿依赖关系查看。'
              : '未读取到此层的权重',
          }}
          pagination={{ pageSize: 10, showSizeChanger: false }}
          columns={[
            { title: 'Tensor 路径', dataIndex: 'name', render: (value) => <code>{value}</code> },
            { title: 'Shape', dataIndex: 'shape', render: (value) => value.join(' × ') },
            { title: '存储 dtype', dataIndex: 'dtype' },
            { title: '存储字节', dataIndex: 'stored_bytes', render: (value) => fmt(value) },
            { title: '分片', dataIndex: 'file' },
          ]}
        />
      </Panel>
    </>
  );
}
