import Button from '../../components/ActionButton';
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Input, Select, Space, Table, Tag } from 'antd';
import { api, base } from '../../api/client';
import { JsonView, QueryState } from '../../components';
import DependencyGraph from './DependencyGraph';

type Node = {
  id: string;
  name: string;
  target: string;
  kind: string;
  outputs?: { shape: (string | number)[]; dtype: string }[];
};
type View = {
  id: string;
  kind: string;
  status: string;
  phase?: string;
  first_run_id?: string;
  graph_id?: string;
  scope?: string;
  error?: string;
  reason?: string;
  artifact?: string;
  artifact_index?: number;
  text?: string;
  nodes?: Node[];
  edges?: { source: string; target: string; kind: string }[];
  truncated?: boolean;
  text_truncated?: boolean;
  parser_status?: string;
  checker?: string;
};

export default function CapturedGraph({
  runId,
  kind,
  enabled,
}: {
  runId?: string;
  kind: string;
  enabled: boolean;
}) {
  const [viewId, setViewId] = useState('');
  const [nodeId, setNodeId] = useState('');
  const [search, setSearch] = useState('');
  const result = useQuery({
    queryKey: ['graph-views', runId],
    queryFn: () =>
      api<{ views: View[]; epoch: string; limitations?: string[] }>(`/runs/${runId}/graph-views`),
    enabled: !!runId && enabled,
    retry: false,
    staleTime: Infinity,
  });
  const views = result.data?.views.filter((v) => v.kind === kind) ?? [];
  const view = views.find((v) => v.id === viewId) ?? views[0];
  const chosen =
    view?.nodes?.find((n) => n.id === nodeId) ??
    view?.nodes?.find((n) => n.kind !== 'placeholder') ??
    view?.nodes?.[0];
  const incoming =
    view?.edges
      ?.filter((e) => e.target === chosen?.id)
      .map((e) => e.source)
      .slice(0, 8) ?? [];
  const outgoing =
    view?.edges
      ?.filter((e) => e.source === chosen?.id)
      .map((e) => e.target)
      .slice(0, 8) ?? [];
  const ids = chosen ? [...incoming, chosen.id, ...outgoing] : [];
  const neighborhood = {
    nodes: (view?.nodes ?? [])
      .filter((n) => ids.includes(n.id))
      .map((n) => ({
        ...n,
        module: null,
        note: n.name,
        label: view?.kind.startsWith('inductor_') ? n.id.slice(0, 24) : n.target.slice(0, 24),
        column: n.id === chosen?.id ? 1 : incoming.includes(n.id) ? 0 : 2,
        row:
          n.id === chosen?.id
            ? 0
            : incoming.includes(n.id)
              ? incoming.indexOf(n.id)
              : outgoing.indexOf(n.id),
      })),
    edges: view?.edges?.filter((e) => ids.includes(e.source) && ids.includes(e.target)) ?? [],
  };
  return (
    <div className="captured-graph">
      <QueryState loading={result.isLoading} error={result.error} />
      {enabled && result.data && !views.length && (
        <Alert
          showIcon
          type="info"
          message="此记录未采集所选图类型"
          description="历史 FX 记录不能自动还原 ONNX 或融合后 IR。请显式采集诊断样例。"
        />
      )}
      {!!views.length && (
        <Select
          aria-label="导出或编译产物"
          style={{ width: 420, maxWidth: '100%' }}
          value={view?.id}
          onChange={(v) => {
            setViewId(v);
            setNodeId('');
          }}
          options={views.map((v) => ({
            value: v.id,
            label: `${v.id} · ${v.phase ?? ''} · ${v.status}`,
          }))}
        />
      )}
      {view && (
        <>
          <p>
            <Tag>
              {view.kind} · {view.status}
            </Tag>
            <Tag>来源运行 {view.first_run_id}</Tag>
            {view.checker && <Tag>ONNX checker: {view.checker}</Tag>}
            {(view.truncated || view.text_truncated) && (
              <Tag color="warning">部分显示，已标记截断</Tag>
            )}
          </p>
          {view.status !== 'available' && (
            <Alert
              type="warning"
              showIcon
              message={view.status}
              description={view.error ?? view.reason}
            />
          )}
          <p className="help-text">{view.scope}</p>
          {!!view.nodes?.length && (
            <>
              <Input.Search
                aria-label="筛选图节点"
                placeholder="节点名称 / 算子类型"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                allowClear
              />
              <Table
                size="small"
                rowKey="id"
                scroll={{ x: 700 }}
                pagination={{ pageSize: 8, showSizeChanger: false }}
                dataSource={view.nodes.filter((n) =>
                  `${n.name} ${n.target}`.toLowerCase().includes(search.toLowerCase()),
                )}
                onRow={(n) => ({ onClick: () => setNodeId(n.id) })}
                rowSelection={{
                  type: 'radio',
                  selectedRowKeys: chosen ? [chosen.id] : [],
                  onChange: (keys) => setNodeId(String(keys[0])),
                }}
                columns={[
                  { title: '节点', dataIndex: 'id' },
                  {
                    title: '运算 / 调度单元',
                    dataIndex: 'target',
                    render: (v) => <code>{v}</code>,
                  },
                  {
                    title: '输出 Shape',
                    dataIndex: 'outputs',
                    render: (v: Node['outputs']) =>
                      v?.length
                        ? v.map((x) => `[${x.shape.join(', ')}]`).join(' / ')
                        : '见原始表示',
                  },
                ]}
              />
              {chosen && (
                <>
                  <DependencyGraph
                    graph={neighborhood}
                    identity={kind}
                    selected={chosen.id}
                    onSelect={(n) => setNodeId(n.id)}
                  />
                  <details>
                    <summary>所选节点与原始依赖</summary>
                    <JsonView value={chosen} />
                  </details>
                </>
              )}
            </>
          )}
          {view.text && (
            <details>
              <summary>
                {kind === 'generated_code'
                  ? '查看生成的 Triton / C++ / Python 代码'
                  : '查看原始 IR 文本'}
              </summary>
              <pre className="graph-source">{view.text}</pre>
            </details>
          )}
          <Space wrap>
            {view.artifact_index !== undefined && (
              <Button
                href={`${base}/runs/${runId}/experiment/artifacts/${view.artifact_index}`}
                target="_blank"
              >
                下载{kind === 'onnx' ? ' ONNX 结构文件' : '原始编译文件'}
              </Button>
            )}
            <Button href={`${base}/runs/${runId}/graph-views`} target="_blank">
              查看图证据 JSON
            </Button>
          </Space>
          <details>
            <summary>导出范围、解析状态与文件</summary>
            <JsonView value={{ ...view, nodes: undefined, edges: undefined, text: undefined }} />
          </details>
          <Alert
            type="info"
            showIcon
            message="图、IR 和代码的粒度不同；不能把节点数量差当成实测 kernel 数或加速比。"
            description={
              kind === 'onnx'
                ? '固定输入、无 KV 的 prefill 结构。权重作为图输入，没有复制权重；不是可独立部署的 ONNX 模型。checker 只验证结构合法，不证明数值等价。'
                : '依赖箭头只读取 IR 中明确的缓冲区读写关系。融合分组与生成代码属于编译证据，实际调用和耗时请结合设备时间线。'
            }
          />
        </>
      )}
    </div>
  );
}
