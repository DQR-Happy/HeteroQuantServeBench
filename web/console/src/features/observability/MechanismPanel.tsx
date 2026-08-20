import { useQuery } from '@tanstack/react-query';
import { Alert, Button, Empty, Table, Tabs, Tag } from 'antd';
import { useSearchParams } from 'react-router-dom';
import { api, base } from '../../api/client';
import { JsonView, QueryState } from '../../components';
import { AllocationView, LayoutView, MetricView, PageView } from './MechanismViews';
import { byteValue, type Evidence } from './mechanisms';
import CaptureLinks from './CaptureLinks';
import RecordsExplorer from './RecordsExplorer';

export default function MechanismPanel({ runId }: { runId: string }) {
  const [params, setParams] = useSearchParams();
  const query = useQuery({
    queryKey: ['mechanisms', runId],
    queryFn: () => api<Evidence>(`/runs/${runId}/observations`),
  });
  const evidence = query.data;
  const source = evidence?.context?.source;
  const manifest = useQuery({
    queryKey: ['observation-manifest', runId],
    queryFn: () =>
      api<{ artifacts: { path: string; bytes: number; sha256: string }[] }>(
        `/runs/${runId}/observations/manifest`,
      ),
    enabled: evidence?.publication === 'supervisor_acknowledged',
  });
  if (query.isPending || query.isError)
    return <QueryState loading={query.isPending} error={query.error} />;
  if (!evidence || evidence.availability === 'not_collected')
    return <Empty description="没有已封存的观测包；关闭观测或旧记录不能事后补录" />;
  const tabs = [
    { key: 'metrics', label: '内存与缺页', children: <MetricView evidence={evidence} /> },
    { key: 'kv', label: 'KV 增长与布局', children: <LayoutView evidence={evidence} kind="kv" /> },
    {
      key: 'tensors',
      label: '权重布局',
      children: <LayoutView evidence={evidence} kind="weights" />,
    },
    { key: 'allocator', label: '申请与回收', children: <AllocationView evidence={evidence} /> },
    { key: 'pages', label: 'CPU 页样本', children: <PageView evidence={evidence} /> },
    { key: 'links', label: '算子关联', children: <CaptureLinks runId={runId} /> },
    {
      key: 'identity',
      label: '执行身份',
      children: (
        <>
          <div className="identity-grid">
            {Object.entries(evidence.context ?? {})
              .filter(([k]) => !['source', 'plan', 'process'].includes(k))
              .map(([key, value]) => (
                <div key={key}>
                  <span>{key}</span>
                  <code>
                    {typeof value === 'object' ? JSON.stringify(value) : String(value ?? '—')}
                  </code>
                </div>
              ))}
          </div>
          <details>
            <summary>软件、进程与采集计划</summary>
            <JsonView
              value={{
                ...evidence.context,
                source: source
                  ? { ...source, source_files: '完整文件哈希见 manifest 下载' }
                  : undefined,
              }}
            />
          </details>
        </>
      ),
    },
    {
      key: 'artifacts',
      label: '原始证据',
      children: (
        <>
          <QueryState
            loading={manifest.isPending && manifest.fetchStatus === 'fetching'}
            error={manifest.error}
          />
          <div className="toolbar">
            <p className="help-text">
              封存原件带 SHA-256 校验。视图只是有界投影，完整记录以原件为准。
            </p>
            <Button
              href={`${base}/runs/${runId}/observations/manifest`}
              download={`${runId}-manifest.json`}
            >
              下载 manifest（含源码哈希）
            </Button>
          </div>
          <Table
            size="small"
            dataSource={manifest.data?.artifacts ?? []}
            rowKey="path"
            pagination={false}
            scroll={{ x: 760 }}
            columns={[
              {
                title: '文件',
                render: (_, row, index) => (
                  <a
                    className="mono"
                    href={`${base}/runs/${runId}/observations/artifacts/${index}`}
                    download
                  >
                    {row.path}
                  </a>
                ),
              },
              { title: '大小', dataIndex: 'bytes', render: byteValue, align: 'right' },
              {
                title: 'SHA-256',
                dataIndex: 'sha256',
                render: (v) => <code className="hash-value">{v}</code>,
              },
            ]}
          />
          <details>
            <summary>按来源序号查询原始记录</summary>
            <RecordsExplorer key={runId} runId={runId} />
          </details>
        </>
      ),
    },
    {
      key: 'coverage',
      label: '缺失与边界',
      children: (
        <>
          <div className="coverage-grid">
            <div>
              <Tag>直接测量</Tag>
              <p>采样时刻的 Tensor 元数据、分配器快照和 OS 计数器。是否可用以本次数据源为准。</p>
            </div>
            <div>
              <Tag>派生 / 关联</Tag>
              <p>计数器区间差、Storage 去重、Tensor 与块的快照关联；保留来源和作用域。</p>
            </div>
            <div>
              <Tag color="warning">暂不可测</Tag>
              <p>
                完整 GPU 物理页表、每字节访存轨迹、模型整体 Cache 命中率。不能从 CPU 页样本或少量
                kernel 推断。
              </p>
            </div>
          </div>
          <ul className="boundary-list">
            {evidence.limitations?.map((text) => <li key={text}>{text}</li>)}
          </ul>
          <details>
            <summary>采集问题、OS 可用性与投影来源</summary>
            <JsonView
              value={{
                source_problems: evidence.source_problems,
                os_source_availability: evidence.os_source_availability,
                provenance: evidence.provenance,
              }}
            />
          </details>
        </>
      ),
    },
  ];
  const active = tabs.some((t) => t.key === params.get('view')) ? params.get('view')! : 'metrics';
  return (
    <>
      <Alert
        type={evidence.capture_status === 'complete' ? 'info' : 'warning'}
        showIcon
        message={`采集 ${evidence.capture_status} · ${evidence.publication}`}
        description="完整仅指声明的采集范围。分配器、进程、整机和页样本分别计量，不能相加或推断物理内存布局。"
      />
      <Tabs
        className="mechanism-tabs"
        activeKey={active}
        onChange={(view) =>
          setParams(
            (prev) => {
              const next = new URLSearchParams(prev);
              next.set('view', view);
              return next;
            },
            { replace: true },
          )
        }
        items={tabs}
      />
    </>
  );
}
