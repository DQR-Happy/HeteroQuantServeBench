import { captureBlock } from './availability';
import Button from '../../components/ActionButton';
import { useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Checkbox, Input, Select, Space, Table, Tag } from 'antd';
import { Link, useSearchParams } from 'react-router-dom';
import { api, base, fmt, terminal } from '../../api/client';
import type { Deployment, Run } from '../../api/types';
import { useRun } from '../../hooks';
import { JsonView, Panel, QueryState } from '../../components';
import DependencyGraph from './DependencyGraph';
import CapturedGraph from './CapturedGraph';
import type { Architecture } from './types';

type Plan = { mode: string; shape_policy: string; fullgraph: boolean };
type Node = {
  id: string;
  name: string;
  kind: string;
  target: string;
  module: string | null;
  module_paths: string[];
  outputs: { shape: (number | string)[]; dtype: string }[];
  source_stack: string;
};
type Graph = {
  id: string;
  phase: string;
  nodes: Node[];
  edges: { source: string; target: string; kind: string }[];
  status: string;
  graph_sha256: string;
  first_run_id: string;
  invocations: number;
};
type Document = {
  run_id: string;
  epoch: string;
  plan: Plan;
  status: string;
  lowering: string;
  forward_calls: number;
  graph_invocations: number;
  new_graphs: number;
  session_graphs: number;
  backend_compile_ms: number;
  forward_host_ms: number;
  sample_class: string;
  used_graph_ids: string[];
  graph_break_events: { reason: string; count: number }[];
  graphs: Graph[];
  analysis: Record<string, unknown>;
  truncated: boolean;
  limitations: string[];
};
type History = { run_id: string; state: string; epoch: string; role: string; plan: Plan };

export default function RuntimeExplorer({
  data,
  deployment,
  initialRun,
  representation = 'fx',
}: {
  data: Architecture;
  deployment?: Deployment;
  initialRun?: string | null;
  representation?: string;
}) {
  const client = useQueryClient();
  const [, setParams] = useSearchParams();
  const [mode, setMode] = useState('eager');
  const [shape, setShape] = useState('auto');
  const [fullgraph, setFullgraph] = useState(false);
  const [tokens, setTokens] = useState(16);
  const [selectedRun, setSelectedRun] = useState(initialRun ?? '');
  const [submitted, setSubmitted] = useState('');
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const [graphId, setGraphId] = useState('');
  const [nodeId, setNodeId] = useState('');
  const [search, setSearch] = useState('');
  const [baseline, setBaseline] = useState('');
  const [comparison, setComparison] = useState<unknown>();
  const [comparing, setComparing] = useState(false);
  const history = useQuery({
    queryKey: ['runtime-runs', data.deployment_id],
    queryFn: () =>
      api<{ items: History[] }>(
        `/runtime/runs?deployment_id=${encodeURIComponent(data.deployment_id)}`,
      ),
    refetchInterval: 10000,
  });
  const identity = selectedRun || history.data?.items[0]?.run_id;
  const run = useRun(identity);
  const pending = useRun(submitted || undefined);
  const active =
    sending ||
    (!!submitted && (!pending.data || !terminal(pending.data.state))) ||
    (!!identity && run.isPending) ||
    (!!run.data && !terminal(run.data.state));
  const document = useQuery({
    queryKey: ['runtime-graph', identity],
    queryFn: () => api<Document>(`/runs/${identity}/runtime-graph`),
    enabled: !!identity && !!run.data && terminal(run.data.state),
    retry: false,
    staleTime: Infinity,
  });
  useEffect(() => {
    if (submitted && run.data?.id === submitted && terminal(run.data.state)) {
      void client.invalidateQueries({ queryKey: ['runtime-runs', data.deployment_id] });
    }
  }, [submitted, run.data?.id, run.data?.state, client, data.deployment_id]);
  const native =
    (deployment?.kernel_profile ?? 'framework-native') === 'framework-native' &&
    (deployment?.quantization?.method ?? 'none') === 'none' &&
    !(deployment?.quantization as { kv?: { bits?: number | null } })?.kv?.bits;
  const diagnostic =
    representation === 'onnx'
      ? 'onnx'
      : representation.startsWith('inductor_') || representation === 'generated_code'
        ? 'inductor'
        : undefined;
  const start = async (quality: boolean, capture?: string) => {
    if (!deployment?.epoch || active) return;
    setSending(true);
    setError('');
    try {
      const capabilities = await api<{ limits: { max_deadline_ms: number } }>('/runtime/catalog');
      const runtime_plan = {
        mode: capture ? (capture === 'onnx' ? 'eager' : 'inductor') : mode,
        shape_policy: capture ? 'static' : shape,
        fullgraph: capture ? false : fullgraph,
      };
      const row = await api<Run>(
        quality ? '/precision/quality-jobs' : '/requests',
        quality
          ? {
              deployment_id: deployment.id,
              expected_epoch: deployment.epoch,
              runtime_plan,
              quality_plan: { benchmark_id: 'wikitext2-mini-v1' },
            }
          : {
              deployment_id: deployment.id,
              expected_epoch: deployment.epoch,
              runtime_plan,
              ...(deployment.input_mode === 'completion'
                ? { prompt: 'Once upon a time, there was a little' }
                : {
                    messages: [
                      {
                        role: 'user',
                        content: 'Explain why the sky is blue in three short sentences.',
                      },
                    ],
                  }),
              max_output_tokens: capture ? 2 : tokens,
              ...(capture ? { graph_capture: { kind: capture } } : {}),
              deadline_ms: capabilities.limits.max_deadline_ms,
              observation_mode: 'off',
              save_input: false,
            },
      );
      client.setQueryData(['run', row.id], row);
      setSelectedRun(row.id);
      setParams(
        (previous) => {
          const next = new URLSearchParams(previous);
          next.set('runtime_run', row.id);
          return next;
        },
        { replace: true },
      );
      setSubmitted(row.id);
      setGraphId('');
      setNodeId('');
      setComparison(undefined);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  };
  const doc = document.data;
  const graph =
    doc?.graphs.find((g) => g.id === graphId) ??
    doc?.graphs.find((g) => doc.used_graph_ids.includes(g.id)) ??
    doc?.graphs[0];
  const operations = graph?.nodes.filter((n) => n.kind.startsWith('call_')) ?? [];
  const chosen = graph?.nodes.find((n) => n.id === nodeId) ?? operations[0];
  const incoming =
    graph?.edges
      .filter((e) => e.target === chosen?.id)
      .map((e) => e.source)
      .slice(0, 10) ?? [];
  const outgoing =
    graph?.edges
      .filter((e) => e.source === chosen?.id)
      .map((e) => e.target)
      .slice(0, 10) ?? [];
  const ids = chosen ? [...incoming, chosen.id, ...outgoing] : [];
  const neighborhood = {
    nodes: (graph?.nodes ?? [])
      .filter((n) => ids.includes(n.id))
      .map((n) => ({
        id: n.id,
        label: n.kind === 'placeholder' ? `输入 · ${n.name.slice(0, 17)}` : n.target.slice(0, 26),
        module: n.module,
        kind: n.kind,
        note: n.name,
        column: n.id === chosen?.id ? 1 : incoming.includes(n.id) ? 0 : 2,
        row:
          n.id === chosen?.id
            ? 0
            : incoming.includes(n.id)
              ? incoming.indexOf(n.id)
              : outgoing.indexOf(n.id),
      })),
    edges: graph?.edges.filter((e) => ids.includes(e.source) && ids.includes(e.target)) ?? [],
  };
  return (
    <Panel
      className="runtime-explorer"
      title="计算图与 L3 执行"
      subtitle="FX 是编译输入；ONNX 是导出表示；Inductor IR 与代码来自实际编译。"
    >
      {diagnostic ? (
        <>
          <Button
            type="primary"
            loading={sending}
            disabledReason={captureBlock(deployment, active, submitted || identity, true)}
            disabled={!!captureBlock(deployment, active, submitted || identity, true)}
            onClick={() => void start(false, diagnostic)}
          >
            采集{diagnostic === 'onnx' ? ' ONNX 导出' : ' Inductor IR 与代码'}诊断样例
          </Button>
          <p className="help-text">
            独立诊断使用 2 tokens 输出预算、关闭其他采集。ONNX 固定 prefill、无 KV；Inductor
            在新图编译时收集融合前后 IR，并关闭 FX
            编译缓存以取得真实诊断文件。采集增加开销，此次耗时不能用于性能结论。切换融合前、融合后、代码时可复用同一条记录，无需重复执行。
          </p>
        </>
      ) : (
        <>
          <Space wrap>
            <Select
              aria-label="L3执行策略"
              value={mode}
              onChange={setMode}
              disabled={active}
              style={{ width: 230, maxWidth: '100%' }}
              options={[
                { value: 'eager', label: 'Eager 原生执行' },
                { value: 'dynamo_eager', label: 'Dynamo 捕获 / FX 原样执行', disabled: !native },
                { value: 'inductor', label: 'Inductor 编译优化', disabled: !native },
              ]}
            />
            <Select
              aria-label="形状策略"
              value={shape}
              onChange={setShape}
              disabled={active || mode === 'eager'}
              options={[
                { value: 'auto', label: '自动形状泛化' },
                { value: 'static', label: '静态形状特化' },
                { value: 'dynamic', label: '动态形状' },
              ]}
            />
            <Checkbox
              checked={fullgraph}
              onChange={(e) => setFullgraph(e.target.checked)}
              disabled={active || mode === 'eager'}
            >
              Fullgraph：遇图断裂即失败
            </Checkbox>
            <Select
              aria-label="L3输出预算"
              value={tokens}
              onChange={setTokens}
              disabled={active}
              options={[8, 16, 32, 64]
                .filter((n) => n <= (deployment?.max_output_tokens ?? 64))
                .map((n) => ({ value: n, label: `${n} 输出 tokens` }))}
            />
            <Button
              type="primary"
              disabledReason={captureBlock(
                deployment,
                active,
                submitted || identity,
                mode !== 'eager',
              )}
              disabled={!!captureBlock(deployment, active, submitted || identity, mode !== 'eager')}
              loading={sending}
              onClick={() => void start(false)}
            >
              运行固定样例
            </Button>
            <Button
              disabledReason={captureBlock(
                deployment,
                active,
                submitted || identity,
                mode !== 'eager',
              )}
              disabled={!!captureBlock(deployment, active, submitted || identity, mode !== 'eager')}
              onClick={() => void start(true)}
            >
              独立 PPL 评测所选策略
            </Button>
          </Space>
          {!native && (
            <Alert
              type="info"
              showIcon
              message="编译组合首批支持原生浮点权重与默认 KV；当前量化或自定义算子组合尚未验证。"
              description={
                <Link to="/precision">在表示精度页选择 Floating point 与默认 KV，再加载模型 →</Link>
              }
            />
          )}
          <p className="help-text">
            固定输入因聊天/续写模式分别设定，实际输入 token
            摘要随凭据保存。首次运行可能耗时数分钟；再次运行同一方案可观察编译复用。切换方案会清空本
            worker 的编译会话缓存，磁盘编译缓存仍可能被上游复用。CUDA Graph 暂未启用。
          </p>
        </>
      )}
      <QueryState error={history.error ?? document.error} />
      {error && <Alert type="error" message={error} />}
      {run.data?.error && <Alert type="error" message={run.data.error} />}
      {active && (
        <p role="status">
          任务执行中：{run.data?.state ?? '已提交'}。离开页面不会重提任务，可在运行记录中取消。
        </p>
      )}
      <Space wrap>
        <Select
          aria-label="L3运行记录"
          placeholder="选择已保存的执行图"
          style={{ width: 380, maxWidth: '100%' }}
          value={identity}
          onChange={(id) => {
            setSelectedRun(id);
            setParams(
              (previous) => {
                const next = new URLSearchParams(previous);
                next.set('runtime_run', id);
                return next;
              },
              { replace: true },
            );
            setGraphId('');
            setNodeId('');
            setComparison(undefined);
          }}
          options={(history.data?.items ?? []).map((r) => ({
            value: r.run_id,
            label: `${r.plan.mode} · ${r.role} · ${r.state} · ${r.run_id.slice(-10)}`,
          }))}
        />
        {identity && <Link to={`/runs/${identity}`}>运行凭据、输出与质量 →</Link>}
        <Button size="small" onClick={() => void history.refetch()}>
          刷新记录
        </Button>
      </Space>
      {diagnostic && (
        <CapturedGraph
          runId={identity}
          kind={representation}
          enabled={!!run.data && terminal(run.data.state)}
        />
      )}
      {doc && (
        <>
          {doc.epoch !== deployment?.epoch && (
            <Alert type="info" message="这是历史模型实例的图证据；不代表当前已加载方案。" />
          )}
          <p>
            <Tag>
              {doc.plan.mode} · {doc.status}
            </Tag>
            <Tag>本次 forward {doc.forward_calls}</Tag>
            <Tag>本次图调用 {doc.graph_invocations}</Tag>
            <Tag>
              本次新图 {doc.new_graphs} / 会话累计 {doc.session_graphs}
            </Tag>
            <Tag>后端编译 {fmt(doc.backend_compile_ms, 1)} ms</Tag>
            <Tag>
              {doc.sample_class === 'diagnostic_capture'
                ? '诊断采集，含观测开销'
                : doc.sample_class === 'includes_compilation'
                  ? '包含编译成本'
                  : '本次未观察到新后端图'}
            </Tag>
            {doc.truncated && <Tag color="warning">证据达到预算，部分保存</Tag>}
          </p>
          {!diagnostic && (
            <>
              <Alert
                type="info"
                message="FX 节点是编译器输入；不能用节点减少直接宣称 kernel 融合、显存下降或 GPU Cache 命中改善。"
                description="Dynamo 捕获仍原样执行 FX；Inductor 才进行编译优化。图断裂可能留下 eager 区域，图调用次数不能换算成全模型覆盖率。性能为本次样本，正式结论需要同条件重复对照；PPL 另测。"
              />
              {doc.graph_break_events.length > 0 && (
                <details>
                  <summary>本次图断裂事件（{doc.graph_break_events.length} 类）</summary>
                  <JsonView value={doc.graph_break_events} />
                </details>
              )}
              {doc.graphs.length > 0 ? (
                <>
                  <Space wrap>
                    <Select
                      aria-label="编译子图"
                      style={{ width: 320, maxWidth: '100%' }}
                      value={graph?.id}
                      onChange={(id) => {
                        setGraphId(id);
                        setNodeId('');
                      }}
                      options={doc.graphs.map((g) => ({
                        value: g.id,
                        label: `${g.id} · ${g.phase} · ${g.nodes.length} 节点 · ${g.status}${doc.used_graph_ids.includes(g.id) ? ' · 本次调用' : ' · 历史特化'}`,
                      }))}
                    />
                    <Input.Search
                      aria-label="筛选编译节点"
                      placeholder="算子 / 模块路径"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      allowClear
                    />
                  </Space>
                  <Table
                    size="small"
                    rowKey="id"
                    scroll={{ x: 800 }}
                    pagination={{ pageSize: 8, showSizeChanger: false }}
                    dataSource={operations.filter((n) =>
                      `${n.target} ${n.module_paths.join(' ')}`
                        .toLowerCase()
                        .includes(search.toLowerCase()),
                    )}
                    onRow={(n) => ({ onClick: () => setNodeId(n.id) })}
                    rowSelection={{
                      type: 'radio',
                      selectedRowKeys: chosen ? [chosen.id] : [],
                      onChange: (keys) => setNodeId(String(keys[0])),
                    }}
                    columns={[
                      { title: 'FX 算子', dataIndex: 'target', render: (v) => <code>{v}</code> },
                      {
                        title: '关联模块路径',
                        dataIndex: 'module',
                        render: (v) => <code>{v ?? '未映射 / 函数级节点'}</code>,
                      },
                      {
                        title: '输出 Shape',
                        dataIndex: 'outputs',
                        render: (v: Node['outputs']) =>
                          v.map((m) => `[${m.shape.join(', ')}]`).join(' / '),
                      },
                    ]}
                  />
                  {chosen && (
                    <>
                      <DependencyGraph
                        graph={neighborhood}
                        identity="compiled"
                        selected={chosen.id}
                        onSelect={(n) => setNodeId(n.id)}
                      />
                      <details>
                        <summary>所选编译节点、源码和关联权重</summary>
                        <JsonView
                          value={{
                            node: chosen,
                            runtime_module: data.runtime.modules?.modules.find(
                              (m) => m.path === chosen.module,
                            ),
                            weights: chosen.module
                              ? data.tensors
                                  .filter((t) => t.name.startsWith(chosen.module + '.'))
                                  .slice(0, 64)
                              : [],
                          }}
                        />
                      </details>
                    </>
                  )}
                </>
              ) : (
                <p className="help-text">
                  此运行未保存 FX 子图。Eager 原生执行可以在“运行算子观测”子页采集 ATen 依赖；选择
                  Dynamo 或 Inductor 后重新提交，可查看编译图。
                </p>
              )}
              <details>
                <summary>L3 分析、形状与测量边界</summary>
                <JsonView value={doc} />
              </details>
              <Space wrap>
                <Select
                  aria-label="L3对照基线"
                  placeholder="选择同模型、同输入的基线记录"
                  style={{ width: 380, maxWidth: '100%' }}
                  value={baseline || undefined}
                  onChange={setBaseline}
                  options={(history.data?.items ?? [])
                    .filter((r) => r.run_id !== identity && r.state === 'completed')
                    .map((r) => ({
                      value: r.run_id,
                      label: `${r.plan.mode} · ${r.role} · ${r.run_id.slice(-10)}`,
                    }))}
                />
                <Button
                  loading={comparing}
                  disabledReason={{
                    reason: comparing
                      ? '正在核对这两条记录，请等待完成。'
                      : !identity
                        ? '请先选择一条已完成的候选运行。'
                        : '请在此按钮左侧选择另一条已完成运行作为基线。',
                    next: {
                      to: '/model-architecture/compiler?deployment=' + data.deployment_id,
                      label: '选择编译运行与基线',
                    },
                  }}
                  disabled={!baseline || !identity || comparing}
                  onClick={() => {
                    setComparing(true);
                    setError('');
                    void api('/runtime/comparisons', {
                      baseline_run_id: baseline,
                      candidate_run_id: identity,
                    })
                      .then(setComparison)
                      .catch((e) => setError(e.message))
                      .finally(() => setComparing(false));
                  }}
                >
                  比较图、质量与运行样本
                </Button>
                <Button href={`${base}/runs/${identity}/runtime-graph`} target="_blank">
                  查看完整图 JSON
                </Button>
              </Space>
            </>
          )}
          {comparison !== undefined && <JsonView value={comparison} />}
        </>
      )}
    </Panel>
  );
}
