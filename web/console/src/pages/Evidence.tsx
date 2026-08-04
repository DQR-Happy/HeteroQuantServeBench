import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Button, Drawer, Input, Select, Space, Table, Tag } from 'antd';
import { DownloadOutlined, FileSearchOutlined } from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import { api, date, fmt } from '../api/client';
import type { Evidence, EvidenceDetail, EvidenceFile } from '../api/types';
import { JsonView, PageHead, Panel, QueryState, Status } from '../components';
import QuantizationBuilder from '../features/research/QuantizationBuilder';

export function EvidenceViewer({ id, onClose }: { id?: string; onClose: () => void }) {
  const query = useQuery({
    queryKey: ['evidence-file', id],
    queryFn: () => api<EvidenceDetail>(`/evidence/${id}`),
    enabled: !!id,
  });
  return (
    <Drawer
      title={query.data?.name ?? '证据文件'}
      width={850}
      open={!!id}
      onClose={onClose}
      extra={
        id && (
          <Button href={`/api/console/v1/evidence/${id}/download`} icon={<DownloadOutlined />}>
            下载原件
          </Button>
        )
      }
    >
      <QueryState loading={query.isPending} error={query.error} />
      {query.data && (
        <>
          <div className="evidence-identity">
            <Tag>只读文件</Tag>
            <span>{fmt(query.data.bytes / 1024, 1)} KiB</span>
            <p className="mono">SHA-256 {query.data.sha256}</p>
          </div>
          {query.data.format === 'md' ? (
            <div className="markdown">
              <ReactMarkdown skipHtml>{String(query.data.content)}</ReactMarkdown>
            </div>
          ) : typeof query.data.content === 'string' ? (
            <pre className="json-view">{query.data.content}</pre>
          ) : (
            <JsonView value={query.data.content} />
          )}
        </>
      )}
    </Drawer>
  );
}

export default function EvidencePage({
  mode = 'all',
}: {
  mode?: 'all' | 'quantization' | 'profiling' | 'experiments';
}) {
  const query = useQuery({
    queryKey: ['evidence'],
    queryFn: () => api<{ items: Evidence[] }>('/evidence'),
  });
  const [search, setSearch] = useState('');
  const [stage, setStage] = useState<string>();
  const [selected, setSelected] = useState<string>();
  const all = query.data?.items ?? [];
  const filtered = all.filter(
    (x) =>
      (!stage || x.stage === stage) &&
      `${x.stage} ${x.experiment}`.toLowerCase().includes(search.toLowerCase()) &&
      (mode !== 'quantization' || ['S05', 'S13'].includes(x.stage)) &&
      (mode !== 'profiling' || ['S02', 'S03', 'S04', 'S06', 'S07', 'S11', 'S12'].includes(x.stage)),
  );
  const titles = {
    all: '证据中心',
    quantization: '量化与质量',
    profiling: '算子与性能证据',
    experiments: '实验地图',
  };
  return (
    <>
      <PageHead
        kicker="REPRODUCIBLE EVIDENCE"
        title={titles[mode]}
        description="直接索引阶段实验的原始判定、汇总与报告。查看结论的同时，打开它所依据的文件。"
      />
      {mode === 'quantization' && (
        <Alert
          className="section-gap"
          type="warning"
          showIcon
          message="W8 / W4 不等于原生低比特推理已接通"
          description="历史模型级路径含全权重 FP16 反量化控制实验；请同时阅读 verdict、质量门和实际 kernel 路径。当前交互工作台只开放经过接线的部署。"
        />
      )}
      {mode === 'quantization' && <QuantizationBuilder />}
      {mode === 'experiments' && (
        <Alert
          className="section-gap"
          type="info"
          showIcon
          message="已有实验只读展示"
          description="本版不从网页启动任意阶段脚本或批量占用租用设备。实验执行仍通过项目远端流程完成；归档后自动进入证据中心。"
        />
      )}
      <div className="stats-grid">
        <div className="stat">
          <div className="stat-label">可索引实验</div>
          <div className="stat-value">{filtered.length}</div>
          <div className="stat-hint">以 verdict.json 实际存在为准</div>
        </div>
        {['PASS', 'FAIL', 'BLOCKED', 'N/A_BY_ADR'].map((s) => (
          <div className="stat" key={s}>
            <div className="stat-label">
              <Status state={s} />
            </div>
            <div className="stat-value">{filtered.filter((x) => x.status === s).length}</div>
            <div className="stat-hint">保留原始结论，不按展示需要修改</div>
          </div>
        ))}
      </div>
      <Panel
        title="阶段档案"
        extra={
          <Space wrap>
            <Input
              aria-label="搜索实验"
              placeholder="搜索 E05-02 / S05"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              allowClear
            />
            <Select
              aria-label="筛选阶段"
              placeholder="全部阶段"
              allowClear
              style={{ width: 140 }}
              value={stage}
              onChange={setStage}
              options={[...new Set(all.map((x) => x.stage))]
                .sort()
                .map((value) => ({ value, label: value }))}
            />
          </Space>
        }
      >
        <QueryState loading={query.isPending} error={query.error} />
        <Table<Evidence>
          rowKey="id"
          dataSource={filtered}
          pagination={{ pageSize: 12, showSizeChanger: false }}
          scroll={{ x: 700 }}
          expandable={{
            expandedRowRender: (row) => (
              <div className="evidence-files">
                {row.files.map((f) => (
                  <Button
                    key={f.id}
                    icon={<FileSearchOutlined />}
                    onClick={() => setSelected(f.id)}
                  >
                    {f.name}
                  </Button>
                ))}
              </div>
            ),
          }}
          columns={[
            {
              title: '实验',
              dataIndex: 'experiment',
              render: (v: string, r) => (
                <Button type="link" onClick={() => setSelected(r.id)}>
                  {v}
                </Button>
              ),
            },
            { title: '阶段', dataIndex: 'stage' },
            { title: '原始结论', dataIndex: 'status', render: (s: string) => <Status state={s} /> },
            { title: '归档文件', render: (_, r) => `${r.files.length} 个` },
            { title: '文件更新时间', dataIndex: 'updated_at', render: date },
            { title: '来源', render: () => <Tag>历史证据</Tag> },
          ]}
        />
      </Panel>
      <EvidenceViewer id={selected} onClose={() => setSelected(undefined)} />
    </>
  );
}

type Source = EvidenceFile & { category: string; verification: string };
export function KernelCatalog() {
  const query = useQuery({
    queryKey: ['catalog'],
    queryFn: () => api<{ items: Source[] }>('/catalog'),
  });
  const [search, setSearch] = useState('');
  const [id, setId] = useState<string>();
  const rows = (query.data?.items ?? []).filter((x) =>
    x.relative_path.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <>
      <PageHead
        kicker="KERNEL & ARTIFACT CATALOG"
        title="算子与配置目录"
        description="浏览当前证据根目录中的算子源码与模型配置。源码存在、单算子通过、模型回接通过分别需要证据。"
      />
      <Alert
        className="section-gap"
        type="info"
        showIcon
        message="这里是源码与配置索引，不是可点击生效的算子切换器"
        description="当前推理执行路径使用 framework-native。Triton / CUDA 自定义实现的模型级激活，需要独立的正确性与性能验证。"
      />
      <Panel
        title="源码档案"
        extra={
          <Input
            aria-label="搜索算子"
            placeholder="rmsnorm / gemm / quant"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            allowClear
          />
        }
      >
        <QueryState loading={query.isPending} error={query.error} />
        <Table<Source>
          rowKey="id"
          dataSource={rows}
          pagination={{ pageSize: 14, showSizeChanger: false }}
          scroll={{ x: 600 }}
          columns={[
            {
              title: '文件路径',
              dataIndex: 'relative_path',
              render: (v: string, r) => (
                <Button className="source-link" type="link" onClick={() => setId(r.id)}>
                  {v}
                </Button>
              ),
            },
            {
              title: '分类',
              dataIndex: 'category',
              render: (v: string) => <Tag>{v === 'kernel' ? '算子' : '配置'}</Tag>,
            },
            {
              title: '身份',
              render: () => <span className="muted">源码存在 · 未声明在线激活</span>,
            },
          ]}
        />
      </Panel>
      <EvidenceViewer id={id} onClose={() => setId(undefined)} />
    </>
  );
}
