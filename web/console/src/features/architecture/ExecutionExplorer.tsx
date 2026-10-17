import { useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Button, Input, Select, Space, Table, Tag } from 'antd';
import { Link } from 'react-router-dom';
import { api, terminal } from '../../api/client';
import type { Deployment, Run } from '../../api/types';
import { useRun } from '../../hooks';
import { Panel, JsonView, QueryState } from '../../components';
import DependencyGraph from './DependencyGraph';
import type { Architecture, ArchitectureNode, ExecutionArchitecture } from './types';

export default function ExecutionExplorer({
  data,
  deployment,
}: {
  data: Architecture;
  deployment?: Deployment;
}) {
  const client = useQueryClient();
  const [submitted, setSubmitted] = useState<string>();
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const run = useRun(submitted);
  const active = sending || (!!submitted && (!run.data || !terminal(run.data.state)));
  const identity = submitted ?? data.execution?.run_id;
  const ready = !submitted || (!!run.data && terminal(run.data.state));
  const query = useQuery({
    queryKey: ['architecture-execution', data.deployment_id, identity],
    queryFn: () =>
      api<ExecutionArchitecture>(
        `/deployments/${data.deployment_id}/architecture/execution/${identity}`,
      ),
    enabled: !!identity && ready,
    retry: false,
    staleTime: Infinity,
  });
  const [selected, setSelected] = useState('');
  const [phase, setPhase] = useState('all');
  const [search, setSearch] = useState('');
  useEffect(() => {
    if (submitted && run.data && terminal(run.data.state)) {
      void client.invalidateQueries({ queryKey: ['model-architecture', data.deployment_id] });
    }
  }, [submitted, run.data?.state, client, data.deployment_id]);
  const start = async () => {
    if (!deployment?.epoch || deployment.state !== 'ready' || active) return;
    setSending(true);
    setError('');
    try {
      const row = await api<Run>('/requests', {
        deployment_id: deployment.id,
        expected_epoch: deployment.epoch,
        ...(deployment.input_mode === 'completion'
          ? { prompt: 'Hello' }
          : { messages: [{ role: 'user', content: 'Hello' }] }),
        max_output_tokens: 2,
        observation_mode: 'basic',
        capture_architecture: true,
        save_input: false,
      });
      client.setQueryData(['run', row.id], row);
      setSubmitted(row.id);
      setSelected('');
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  };
  const document = query.data;
  const operations =
    document?.nodes.filter((row) => row.kind === 'operator' || row.kind === 'opaque_module') ?? [];
  const chosen = document?.nodes.find((row) => row.id === selected) ?? operations[0];
  const sources = chosen
    ? [
        ...new Set(
          document?.edges.filter((edge) => edge.target === chosen.id).map((edge) => edge.source),
        ),
      ].slice(0, 12)
    : [];
  const targets = chosen
    ? [
        ...new Set(
          document?.edges.filter((edge) => edge.source === chosen.id).map((edge) => edge.target),
        ),
      ].slice(0, 12)
    : [];
  const ids = chosen ? [...sources, chosen.id, ...targets] : [];
  const neighborhood = ids
    .map((id) => document!.nodes.find((row) => row.id === id)!)
    .filter(Boolean)
    .map(
      (row) =>
        ({
          id: row.id,
          label: (row.op ?? row.names?.[0] ?? row.kind).slice(0, 27),
          module: row.module,
          kind: row.kind,
          note: row.op ?? row.names?.join(', ') ?? '',
          column: row.id === chosen?.id ? 1 : sources.includes(row.id) ? 0 : 2,
          row:
            row.id === chosen?.id
              ? 0
              : sources.includes(row.id)
                ? sources.indexOf(row.id)
                : targets.indexOf(row.id),
        }) satisfies ArchitectureNode,
    );
  const graph = {
    nodes: neighborhood,
    edges: (document?.edges ?? []).filter(
      (edge) => ids.includes(edge.source) && ids.includes(edge.target),
    ),
  };
  return (
    <Panel
      title="实际执行的算子依赖"
      subtitle="通用采集，无模型族白名单；记录前两个实际生成步骤的 forward 路径。"
    >
      <Space wrap>
        <Button
          type="primary"
          loading={active}
          disabled={active || deployment?.state !== 'ready'}
          onClick={() => void start()}
        >
          采集运行架构
        </Button>
        {identity && <Link to={`/runs/${identity}`}>查看采集运行与原始制品 →</Link>}
        <Tag>诊断样本 · 不用于正式性能比较</Tag>
      </Space>
      {deployment?.state !== 'ready' && (
        <p className="help-text">
          先在设备或推理工作台加载此模型；读取本页不会自动加载或提交推理。
        </p>
      )}
      <QueryState error={query.error} />
      {error && <Alert type="error" message={error} />}
      {run.data?.error && (
        <Alert type="error" message="采集运行未完成" description={run.data.error} />
      )}
      {document ? (
        <>
          <p>
            <Tag>{document.status === 'captured' ? '本次路径已采集' : '部分采集'}</Tag>
            <Tag>
              {document.nodes.length} 节点 / {document.edges.length} 依赖
            </Tag>
            {document.phases.map((item, i) => (
              <Tag key={i}>
                {item.name} · {item.completed ? '完成' : '未完成'}
              </Tag>
            ))}
          </p>
          {data.execution && !data.execution.epoch_matches && !submitted && (
            <Alert type="info" message="这是历史加载实例的采集证据；可重新采集当前实例。" />
          )}
          <Space wrap>
            <Select
              aria-label="执行阶段"
              value={phase}
              onChange={setPhase}
              options={[
                { value: 'all', label: '全部阶段' },
                { value: 'prefill', label: 'Prefill' },
                { value: 'decode', label: 'Decode' },
              ]}
            />
            <Input.Search
              aria-label="筛选实际算子"
              placeholder="算子名或模块路径"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              allowClear
            />
          </Space>
          <Table
            size="small"
            rowKey="id"
            scroll={{ x: 900 }}
            pagination={{ pageSize: 12, showSizeChanger: false }}
            dataSource={operations.filter(
              (row) =>
                (phase === 'all' || row.phase === phase) &&
                (String(row.op) + ' ' + row.module).toLowerCase().includes(search.toLowerCase()),
            )}
            onRow={(row) => ({ onClick: () => setSelected(row.id) })}
            rowSelection={{
              type: 'radio',
              selectedRowKeys: chosen ? [chosen.id] : [],
              onChange: (keys) => setSelected(String(keys[0])),
            }}
            columns={[
              { title: '实际算子', dataIndex: 'op', render: (value) => <code>{value}</code> },
              {
                title: '所属模块',
                dataIndex: 'module',
                render: (value) => <code>{value || '(root)'}</code>,
              },
              { title: '阶段', dataIndex: 'phase' },
              {
                title: '输出 Shape',
                dataIndex: 'outputs',
                render: (value) =>
                  value
                    .map((item: { shape: number[] }) => `[${item.shape.join(', ')}]`)
                    .join(' / '),
              },
            ]}
          />
          {chosen && (
            <>
              <p className="help-text">
                所选节点的直接前驱与后继（每侧最多 12 个）；完整依赖随结构快照保存。
              </p>
              <DependencyGraph
                graph={graph}
                identity="observed"
                selected={chosen.id}
                onSelect={(node) => setSelected(node.id)}
              />
              <details>
                <summary>所选算子与 Tensor 元数据</summary>
                <JsonView value={chosen} />
              </details>
            </>
          )}
          <details>
            <summary>采集边界、身份与完整结构快照</summary>
            <ul>
              {document.limitations.map((text) => (
                <li key={text}>{text}</li>
              ))}
            </ul>
            <JsonView
              value={{
                warnings: document.warnings,
                truncated: document.truncated,
                source: document.source,
                source_digest: document.source_digest,
                input_token_sha256: document.input_token_sha256,
              }}
            />
            <Button
              onClick={() => {
                const url = URL.createObjectURL(
                  new Blob([JSON.stringify(document, null, 2)], { type: 'application/json' }),
                );
                const a = window.document.createElement('a');
                a.href = url;
                a.download = `architecture-${identity}.json`;
                a.click();
                setTimeout(() => URL.revokeObjectURL(url), 0);
              }}
            >
              导出完整运行架构
            </Button>
          </details>
        </>
      ) : (
        !active &&
        !query.isFetching && (
          <p className="help-text">
            尚无此模型的运行架构。采集使用固定 Hello 输入与最多 2 个输出
            Token，不修改权重或算子配置。
          </p>
        )
      )}
    </Panel>
  );
}
