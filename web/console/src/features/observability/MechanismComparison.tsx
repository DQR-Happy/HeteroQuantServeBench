import { useState } from 'react';
import { Alert, App, Button, Select, Table } from 'antd';
import { api, fmt } from '../../api/client';

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
  notice: string;
};

export default function MechanismComparison({ ids }: { ids: string[] }) {
  const [intervention, setIntervention] = useState<'attention_requested' | 'cleanup_policy'>(
    'attention_requested',
  );
  const [stored, setStored] = useState<{ key: string; body: Result }>();
  const selectionKey = intervention + ':' + ids.join(':');
  const result = stored?.key === selectionKey ? stored.body : undefined;
  const [busy, setBusy] = useState(false);
  const { message } = App.useApp();
  const compare = async () => {
    setBusy(true);
    try {
      const body = await api<Result>('/observation-comparisons', {
        run_ids: ids.slice(0, 2),
        intervention,
      });
      setStored({ key: selectionKey, body });
    } catch (error) {
      void message.error((error as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <div className="toolbar">
        <Select
          aria-label="机制对照变量"
          value={intervention}
          onChange={setIntervention}
          options={[
            { value: 'attention_requested', label: '变量：Attention 实现' },
            { value: 'cleanup_policy', label: '变量：分配器清理策略' },
          ]}
        />
        <Button loading={busy} disabled={ids.length !== 2} onClick={() => void compare()}>
          对比两次运行的内存与 OS 机制
        </Button>
      </div>
      {result && (
        <>
          <Alert
            className="section-gap"
            type={result.comparable ? 'info' : 'warning'}
            message={result.comparable ? '描述性机制口径一致' : result.reasons.join('；')}
            description={`${result.notice} · 决策 ${result.decision}`}
          />
          <Table<Difference>
            size="small"
            dataSource={result.differences}
            rowKey={(r) => `${r.metric_id}:${r.label ?? ''}`}
            pagination={{ pageSize: 15 }}
            scroll={{ x: 800 }}
            columns={[
              { title: '指标', dataIndex: 'metric_id' },
              { title: '时刻', dataIndex: 'label' },
              { title: '左', render: (_, r) => fmt(r.left, 0) },
              { title: '右', render: (_, r) => fmt(r.right, 0) },
              { title: '差值', render: (_, r) => fmt(r.delta, 0) },
              { title: '范围', dataIndex: 'scope' },
            ]}
          />
        </>
      )}
    </>
  );
}
