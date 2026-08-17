import { useQuery } from '@tanstack/react-query';
import { Alert, Empty, Table, Tabs, Tag } from 'antd';
import { api, base, fmt } from '../../api/client';
import { JsonView, QueryState } from '../../components';

type Metric = {
  metric_id: string;
  label?: string;
  value: number | null;
  unit: string;
  scope: string;
  evidence_kind: string;
  availability: string;
};
type Evidence = {
  availability?: string;
  capture_status?: string;
  publication?: string;
  metrics: Metric[];
  kv?: Record<string, unknown>[];
  allocator?: Record<string, unknown>[];
  allocation_lifetimes?: unknown;
  allocation_attribution?: unknown;
  tensor_inventories?: unknown;
  page_samples?: Record<string, unknown>[];
  source_problems?: unknown[];
  limitations?: string[];
  context?: { source?: Record<string, unknown>; [key: string]: unknown };
  os_source_availability?: unknown;
  provenance?: unknown;
};

export default function MechanismPanel({ runId }: { runId: string }) {
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
  return (
    <>
      <QueryState error={query.error} />
      {!evidence ? (
        <Empty description="正在读取证据" />
      ) : evidence.availability === 'not_collected' ? (
        <Empty description="没有已封存的观测包；关闭观测或旧记录不能事后补录" />
      ) : (
        <>
          <Alert
            type={evidence.capture_status === 'complete' ? 'info' : 'warning'}
            showIcon
            message={`采集 ${evidence.capture_status} · ${evidence.publication}`}
            description="内存分配器、CPU 进程、系统内存和页样本分别显示。采集完整仅指声明范围，不代表所有物理内存或硬件访问可见。"
          />
          <Tabs
            items={[
              {
                key: 'metrics',
                label: '内存与缺页',
                children: (
                  <Table<Metric>
                    size="small"
                    rowKey={(r) => `${r.metric_id}:${r.label ?? ''}`}
                    dataSource={evidence.metrics}
                    pagination={{ pageSize: 15 }}
                    scroll={{ x: 900 }}
                    columns={[
                      { title: '指标', dataIndex: 'metric_id' },
                      { title: '时刻', dataIndex: 'label' },
                      {
                        title: '数值',
                        render: (_, r) => (r.value === null ? '未采集' : fmt(r.value, 0)),
                      },
                      { title: '单位', dataIndex: 'unit' },
                      { title: '范围', dataIndex: 'scope' },
                      {
                        title: '证据',
                        render: (_, r) => (
                          <Tag>
                            {r.evidence_kind} / {r.availability}
                          </Tag>
                        ),
                      },
                    ]}
                  />
                ),
              },
              { key: 'kv', label: 'KV 增长与布局', children: <JsonView value={evidence.kv} /> },
              {
                key: 'tensors',
                label: '权重布局',
                children: <JsonView value={evidence.tensor_inventories} />,
              },
              {
                key: 'allocator',
                label: '申请与回收',
                children: (
                  <Tabs
                    items={[
                      {
                        key: 'blocks',
                        label: '块与状态',
                        children: <JsonView value={evidence.allocator} />,
                      },
                      {
                        key: 'lifetimes',
                        label: '分配代次',
                        children: <JsonView value={evidence.allocation_lifetimes} />,
                      },
                      {
                        key: 'attribution',
                        label: 'Tensor 关联',
                        children: <JsonView value={evidence.allocation_attribution} />,
                      },
                    ]}
                  />
                ),
              },
              {
                key: 'pages',
                label: 'CPU 页样本',
                children: <JsonView value={evidence.page_samples} />,
              },
              {
                key: 'identity',
                label: '执行身份',
                children: (
                  <JsonView
                    value={{
                      ...evidence.context,
                      source: source
                        ? { ...source, source_files: '完整文件哈希见 manifest 下载' }
                        : undefined,
                    }}
                  />
                ),
              },
              {
                key: 'artifacts',
                label: '原始证据',
                children: (
                  <>
                    <QueryState error={manifest.error} />
                    <p>原件带 SHA-256 校验；申请事件使用原生序列，界面最多预览 500 个生命周期。</p>
                    <a
                      href={`${base}/runs/${runId}/observations/manifest`}
                      download={`${runId}-manifest.json`}
                    >
                      下载 manifest（含源码哈希）
                    </a>
                    <Table
                      size="small"
                      dataSource={manifest.data?.artifacts ?? []}
                      rowKey="path"
                      pagination={false}
                      scroll={{ x: 700 }}
                      columns={[
                        {
                          title: '文件',
                          render: (_, row, index) => (
                            <a
                              href={`${base}/runs/${runId}/observations/artifacts/${index}`}
                              download
                            >
                              {row.path}
                            </a>
                          ),
                        },
                        { title: '字节', dataIndex: 'bytes' },
                        { title: 'SHA-256', dataIndex: 'sha256' },
                      ]}
                    />
                  </>
                ),
              },
              {
                key: 'coverage',
                label: '缺失与边界',
                children: (
                  <JsonView
                    value={{
                      limitations: evidence.limitations,
                      source_problems: evidence.source_problems,
                      os_source_availability: evidence.os_source_availability,
                      provenance: evidence.provenance,
                    }}
                  />
                ),
              },
            ]}
          />
        </>
      )}
    </>
  );
}
