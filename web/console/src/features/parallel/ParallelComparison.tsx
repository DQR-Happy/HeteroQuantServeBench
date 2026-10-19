import { Alert, Table } from 'antd';
import { fmt } from '../../api/client';
import { JsonView } from '../../components';
export type ParallelPair = {
  comparable: boolean;
  reasons: string[];
  greedy_tokens_exact: boolean;
  performance: Record<string, (number | null)[]>;
  performance_scope?: Record<string, string>;
  performance_comparable?: boolean;
  claim: string;
};
export default function ParallelComparison({ value: v }: { value: ParallelPair }) {
  const labels: Record<string, string> = {
    output_tokens_per_s: '工作负载生成速度 / Token/s',
    workload_ms: '工作负载墙钟 / ms',
    job_total_ms: '完整任务（含加载与退出）/ ms',
  };
  return (
    <>
      <Alert
        showIcon
        type={v.comparable ? 'success' : 'warning'}
        message={v.comparable ? '来源、负载与逐 Token 核对通过' : '当前结果不能作为合格对照'}
        description={
          v.reasons.join('；') || '一次对照仅为描述性观测；不同计时口径不能直接求加速比。'
        }
      />
      <p>逐 Token 一致：{v.greedy_tokens_exact ? '是' : '否'}</p>
      <Table
        size="small"
        rowKey="key"
        pagination={false}
        scroll={{ x: 550 }}
        dataSource={Object.entries(v.performance).map(([key, values]) => ({
          key,
          label: labels[key] ?? key,
          values,
          scope: v.performance_scope?.[key],
        }))}
        columns={[
          { title: '指标与单位', dataIndex: 'label' },
          { title: '基线', render: (_, r) => fmt(r.values[0], 3) },
          { title: '候选', render: (_, r) => fmt(r.values[1], 3) },
          {
            title: '计时口径',
            render: (_, r) =>
              r.scope === 'clock_boundary_mismatch'
                ? '不同：仅展示，不能求加速比'
                : r.scope === 'both_include_cold_load_and_cleanup'
                  ? '均包含冷加载与清理'
                  : '同工作负载边界',
          },
        ]}
      />
      <details>
        <summary>可比性与封存身份</summary>
        <JsonView value={v} />
      </details>
    </>
  );
}
