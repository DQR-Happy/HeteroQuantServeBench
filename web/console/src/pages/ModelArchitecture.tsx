import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Button, Descriptions, Empty, Select, Table, Tabs, Tag } from 'antd';
import { Link, useSearchParams } from 'react-router-dom';
import { api, fmt, gib } from '../api/client';
import type { Deployment } from '../api/types';
import { JsonView, PageHead, Panel, QueryState } from '../components';
import DependencyGraph from '../features/architecture/DependencyGraph';
import StructureExplorer from '../features/architecture/StructureExplorer';
import ExecutionExplorer from '../features/architecture/ExecutionExplorer';
import type { Architecture, ArchitectureNode } from '../features/architecture/types';
import '../features/architecture/styles.css';

function ArchitectureView({ data, deployment }: { data: Architecture; deployment?: Deployment }) {
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
      <Descriptions
        size="small"
        column={{ xs: 1, md: 3 }}
        items={[
          {
            key: 'type',
            label: '模型族 / 类',
            children: `${sizes.model_type} / ${(sizes.architectures as string[] | null)?.join(', ') ?? '未知'}`,
          },
          {
            key: 'layers',
            label: 'Decoder 层 / 隐藏维度',
            children: `${sizes.num_hidden_layers ?? '未知'} / ${sizes.hidden_size ?? '未知'}`,
          },
          {
            key: 'heads',
            label: 'Q heads / KV heads',
            children: `${sizes.num_attention_heads ?? '未知'} / ${sizes.num_key_value_heads ?? '未知'}`,
          },
          {
            key: 'weights',
            label: '存储 tensor / 元素数',
            children: `${data.stored_tensor_count} / ${fmt(data.stored_parameter_elements / 1e9, 3)} B`,
          },
          {
            key: 'bytes',
            label: '权重文件 tensor 区域',
            children: `${fmt(gib(data.stored_bytes), 3)} GiB · ${Object.keys(data.dtypes).join(' / ')}`,
          },
          {
            key: 'runtime',
            label: '当前加载类 / 实现',
            children: `${data.runtime.model_class ?? '未加载'} / ${data.runtime.kernel_profile ?? '未知'}`,
          },
        ]}
      />
      <StructureExplorer data={data} />
      <ExecutionExplorer data={data} deployment={deployment} />
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
            '可使用上方通用结构与实际算子采集；模型族识别不限制这些功能。',
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
                <p>{selected.note || '箭头表示逻辑先后依赖，节点不与单个 GPU kernel 一一对应。'}</p>
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
                ? `本层 ${current?.attention_type}；${current?.attention_type === 'sliding_attention' ? `局部窗口配置 ${sizes.sliding_window ?? '未知'} tokens。` : '因果 mask 控制可见位置。'}数学分解不表示融合后端实际发起这些独立 kernel。`
                : '并行机会不等于已观测到实际重叠；实际时间线请到机制观测或深度分析查看。'
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
              ? '所选计算节点没有独立的模型权重；函数操作、残差与 KV 状态不等于可训练参数。'
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
      <Panel title="证据与能力边界" subtitle="制品静态结构、安装实现与当前加载状态分别记录">
        <p>
          静态语义示意图状态：
          <Tag>
            {data.status === 'verified_static'
              ? '制品关键结构与已核对实现匹配'
              : '暂无专用示意图；通用结构可用'}
          </Tag>
          实现：Transformers {data.implementation.version ?? '未识别'}
        </p>
        <p>
          模型族整理：<Tag>{data.semantic?.family ?? sizes.model_type}</Tag>
          依据配置声明与模块名称辅助标注；不代表模型执行或 kernel 已通过认证。
        </p>
        <details>
          <summary>已覆盖的模型族与配置声明</summary>
          <JsonView value={data.semantic} />
        </details>
        <p>
          当前驻留模型的 config 与本次制品匹配：
          {data.runtime.loaded_config_matches == null
            ? '未加载，无法确认'
            : data.runtime.loaded_config_matches
              ? '是'
              : '否，请重新核对制品与加载记录'}
        </p>
        {data.runtime.load_run_id && (
          <Link to={`/runs/${data.runtime.load_run_id}`}>打开当前加载凭据 →</Link>
        )}
        <ul>
          {data.limitations.map((text) => (
            <li key={text}>{text}</li>
          ))}
        </ul>
        <details>
          <summary>配置、实现与分片 header 身份</summary>
          <JsonView
            value={{
              config: data.config,
              config_sha256: data.config_sha256,
              implementation: data.implementation,
              shards: data.shards,
              runtime: data.runtime,
              evidence_scope: data.evidence_scope,
              issues: data.issues,
            }}
          />
        </details>
      </Panel>
    </>
  );
}

export default function ModelArchitecture() {
  const [params, setParams] = useSearchParams();
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 3000,
  });
  const id =
    params.get('deployment') ??
    deployments.data?.items.find((item) => item.provider === 'pytorch')?.id;
  const query = useQuery({
    queryKey: [
      'model-architecture',
      id,
      deployments.data?.items.find((item) => item.id === id)?.epoch,
    ],
    queryFn: () => api<Architecture>(`/deployments/${encodeURIComponent(id!)}/architecture`),
    enabled: !!id,
    staleTime: 60000,
    retry: false,
  });
  return (
    <>
      <PageHead
        kicker="ARTIFACT / LOGICAL ARCHITECTURE"
        title="模型架构"
        description="从当前节点已下载制品查看模块与计算依赖，区分串行关系、逻辑并行机会和实际执行证据。"
        extra={
          <Link to="/devices">
            <Button>设备与部署</Button>
          </Link>
        }
      />
      <div className="architecture-picker">
        <div>
          <label className="field-label" htmlFor="architecture-model">
            已登记的本地模型制品
          </label>
          <Select
            id="architecture-model"
            className="wide-select"
            value={id}
            options={deployments.data?.items.map((item) => ({
              value: item.id,
              label: item.name,
              disabled: item.provider !== 'pytorch',
            }))}
            onChange={(value) => setParams({ deployment: value })}
          />
        </div>
        <Button loading={query.isFetching} onClick={() => void query.refetch()}>
          重新读取制品
        </Button>
        <Button
          disabled={!query.data}
          onClick={() => {
            if (!query.data) return;
            const url = URL.createObjectURL(
              new Blob([JSON.stringify(query.data, null, 2)], { type: 'application/json' }),
            );
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = `architecture-${query.data.deployment_id}-${query.data.snapshot_sha256.slice(0, 12)}.json`;
            anchor.click();
            setTimeout(() => URL.revokeObjectURL(url), 0);
          }}
        >
          导出结构快照
        </Button>
        <Link to="/guide#architecture">架构阅读指南 →</Link>
      </div>
      <QueryState loading={query.isPending && !!id} error={query.error ?? deployments.error} />
      {query.data ? (
        <ArchitectureView
          key={query.data.deployment_id + query.data.snapshot_sha256 + query.data.runtime.epoch}
          data={query.data}
          deployment={deployments.data?.items.find((item) => item.id === id)}
        />
      ) : !query.isPending && !query.error ? (
        <Empty description="当前节点尚无可读取的本地模型制品" />
      ) : null}
    </>
  );
}
