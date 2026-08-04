import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, App, Button, Select, Space, Table, Tag } from 'antd';
import { Link } from 'react-router-dom';
import { api, base, date } from '../../api/client';
import type { Deployment, Run } from '../../api/types';
import { JsonView, Panel, QueryState, Status } from '../../components';

type QuantArtifact = {
  id: string;
  run_id: string;
  bits: number;
  group_size: number | null;
  created_at: number;
  status: string;
  manifest?: unknown;
  summary?: unknown;
  execution_label: string;
  quality: string;
  native_deployment_available: boolean;
};

export default function QuantizationBuilder() {
  const client = useQueryClient();
  const { message } = App.useApp();
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api<{ items: Deployment[] }>('/deployments'),
    refetchInterval: 5000,
  });
  const artifacts = useQuery({
    queryKey: ['quantization-artifacts'],
    queryFn: () => api<{ items: QuantArtifact[] }>('/quantization/artifacts'),
    refetchInterval: 5000,
  });
  const [selected, setSelected] = useState('');
  const [bits, setBits] = useState<4 | 8>(4);
  const [busy, setBusy] = useState(false);
  const [submitted, setSubmitted] = useState<string>();
  const candidates = deployments.data?.items.filter((row) => row.provider === 'pytorch') ?? [];
  const deployment = candidates.find((row) => row.id === selected) ?? candidates[0];
  const create = async () => {
    if (!deployment?.epoch) return;
    setBusy(true);
    try {
      const run = await api<Run>('/quantization/jobs', {
        deployment_id: deployment.id,
        expected_epoch: deployment.epoch,
        bits,
        group_size: bits === 4 ? 128 : null,
      });
      client.setQueryData(['run', run.id], run);
      setSubmitted(run.id);
      void message.success('量化转换任务已提交，可在运行记录查看进度与取消。');
    } catch (error) {
      void message.error((error as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Panel
      title="自行量化 FP16 权重"
      subtitle="从已加载模型生成 RTN 候选制品；离线转换与在线推理部署分离"
    >
      <Alert
        type="warning"
        showIcon
        className="section-gap"
        message="生成的是存储制品，质量尚未评估；不会自动切换当前 FP16 推理路径。"
        description="转换进入设备任务队列并占用计算与内存资源。W4 使用 group size 128，W8 使用逐通道缩放；完成后下载 manifest 查看制品身份、覆盖范围与哈希。"
      />
      <QueryState error={deployments.error} />
      <div className="quantization-form">
        <div>
          <label className="field-label" htmlFor="quantization-deployment">
            源模型部署
          </label>
          <Select
            id="quantization-deployment"
            className="wide-select"
            value={deployment?.id}
            onChange={setSelected}
            disabled={busy}
            options={candidates.map((row) => ({
              value: row.id,
              label: `${row.name} · ${row.state}`,
            }))}
          />
        </div>
        <div>
          <label className="field-label" htmlFor="quantization-bits">
            转换方式
          </label>
          <Select
            id="quantization-bits"
            className="wide-select"
            value={bits}
            onChange={setBits}
            disabled={busy}
            options={[
              { label: 'RTN W4 · group 128', value: 4 },
              { label: 'RTN W8 · per-channel', value: 8 },
            ]}
          />
        </div>
        <Button
          type="primary"
          disabled={deployment?.state !== 'ready' || !deployment.epoch}
          loading={busy}
          onClick={() => void create()}
        >
          创建量化候选
        </Button>
      </div>
      {deployment?.state !== 'ready' && (
        <p className="help-text">先在设备与部署中加载 PyTorch 模型，再执行权重转换。</p>
      )}
      {submitted && (
        <Alert
          className="section-gap"
          type="success"
          message={
            <Space wrap>
              任务已提交<Link to={`/runs/${submitted}`}>查看转换进度与结果 →</Link>
            </Space>
          }
        />
      )}
      <h3>已生成候选</h3>
      <QueryState loading={artifacts.isPending} error={artifacts.error} />
      <Table<QuantArtifact>
        rowKey="id"
        size="small"
        dataSource={artifacts.data?.items ?? []}
        pagination={{ pageSize: 6, hideOnSinglePage: true }}
        scroll={{ x: 900 }}
        expandable={{
          expandedRowRender: (row) => (
            <JsonView
              value={{
                summary: row.summary,
                manifest: row.manifest,
                native_deployment_available: row.native_deployment_available,
              }}
            />
          ),
        }}
        columns={[
          {
            title: '量化候选',
            render: (_, row) => (
              <>
                <strong>W{row.bits}</strong>
                <div className="small mono">{row.id.slice(0, 18)}</div>
              </>
            ),
          },
          { title: '状态', render: (_, row) => <Status state={row.status} /> },
          {
            title: '执行 / 质量',
            render: (_, row) => (
              <>
                <Tag>{row.execution_label}</Tag>
                <Tag>{row.quality === 'not_evaluated' ? '质量未评估' : row.quality}</Tag>
              </>
            ),
          },
          { title: '创建时间', dataIndex: 'created_at', render: date },
          {
            title: '追溯',
            render: (_, row) => (
              <Space>
                <Link to={`/runs/${row.run_id}`}>运行记录</Link>
                <Button
                  size="small"
                  disabled={row.status !== 'completed'}
                  href={
                    row.status === 'completed'
                      ? `${base}/quantization/artifacts/${encodeURIComponent(row.run_id)}/manifest`
                      : undefined
                  }
                >
                  Manifest
                </Button>
              </Space>
            ),
          },
        ]}
      />
    </Panel>
  );
}
