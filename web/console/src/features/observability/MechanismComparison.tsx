import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Button, Select, Table, Tag } from 'antd';
import { api, fmt } from '../../api/client';
import { QueryState } from '../../components';
import { byteValue, scopeName, type Evidence } from './mechanisms';

type Difference = {
  metric_id: string;
  label?: string;
  left: number | null;
  right: number | null;
  delta: number | null;
  scope: string;
};
type Result = {
  comparable: boolean;
  reasons: string[];
  differences: Difference[];
  decision: string;
  quality_status: string;
  notice: string;
  intervention: { field: string; left: string; right: string };
};

export default function MechanismComparison({ ids }: { ids: string[] }) {
  const [intervention, setIntervention] = useState<'attention_requested' | 'cleanup_policy'>(
    'cleanup_policy',
  );
  const [stored, setStored] = useState<{ key: string; body: Result }>();
  const selectionKey = intervention + ':' + ids.join(':');
  const result = stored?.key === selectionKey ? stored.body : undefined;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const metrics = useQuery({
    queryKey: ['mechanisms', ids[0]],
    queryFn: () => api<Evidence>(`/runs/${ids[0]}/observations`),
    enabled: ids.length === 2,
  });
  const compare = async () => {
    setStored(undefined);
    setBusy(true);
    setError(undefined);
    try {
      const body = await api<Result>('/observation-comparisons', { run_ids: ids, intervention });
      setStored({ key: selectionKey, body });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const value = (v: number | null, row: Difference) => {
    if (v == null) return '—';
    const unit = metrics.data?.metrics.find(
      (m) => m.metric_id === row.metric_id && m.label === row.label,
    )?.unit;
    return unit === 'byte' ? byteValue(v) : `${fmt(v, 0)} ${unit ?? '(原始单位见观测包)'}`;
  };
  return (
    <div className="mechanism-comparison">
      <div className="toolbar">
        <Select
          aria-label="机制对照变量"
          value={intervention}
          onChange={(v) => {
            setIntervention(v);
            setError(undefined);
          }}
          style={{ minWidth: 245 }}
          options={[
            { value: 'attention_requested', label: '控制变量：Attention 实现' },
            { value: 'cleanup_policy', label: '控制变量：分配器清理策略' },
          ]}
        />
        <Button
          type="primary"
          loading={busy}
          disabled={ids.length !== 2 || ids[0] === ids[1]}
          onClick={() => void compare()}
        >
          检查可比性与机制差异
        </Button>
      </div>
      {ids.length !== 2 && (
        <p className="help-text">
          选择两次已完成运行。服务器检查模型、输入、输出
          token、软件环境与采集计划，并只放开指定变量。
        </p>
      )}
      <QueryState error={error || metrics.error} />
      {result && (
        <>
          <Alert
            className="section-gap"
            showIcon
            type={result.comparable ? 'info' : 'warning'}
            message={result.comparable ? '描述性机制口径一致' : '对照受阻：控制条件不一致'}
            description={
              result.reasons.length
                ? result.reasons.join('；')
                : '仅描述已观测的差异；缺少质量门和成对无插桩实验时，不形成优化采纳结论。'
            }
          />
          <div className="workspace-strip">
            <span>
              A <code>{result.intervention.left ?? '—'}</code>
            </span>
            <span>
              B <code>{result.intervention.right ?? '—'}</code>
            </span>
            <Tag>质量 {result.quality_status}</Tag>
            <Tag color="warning">决策 {result.decision}</Tag>
          </div>
          <Table<Difference>
            size="small"
            dataSource={result.differences}
            rowKey={(r) => `${r.metric_id}:${r.label ?? ''}`}
            pagination={{ pageSize: 12, showSizeChanger: false }}
            scroll={{ x: 1000 }}
            columns={[
              { title: '指标', dataIndex: 'metric_id', render: (v) => <code>{v}</code> },
              { title: '采样边界', dataIndex: 'label', render: (v) => v ?? '请求区间' },
              { title: 'A · 基线', align: 'right', render: (_, r) => value(r.left, r) },
              { title: 'B · 候选', align: 'right', render: (_, r) => value(r.right, r) },
              { title: '差值 · B − A', align: 'right', render: (_, r) => value(r.delta, r) },
              { title: '测量范围', dataIndex: 'scope', render: scopeName },
            ]}
          />
          <p className="help-text">
            差值由服务器提供；—
            表示缺失或不可计算。较小数值不自动代表更好的优化，整机分页变化不能直接归因模型。
          </p>
        </>
      )}
    </div>
  );
}
